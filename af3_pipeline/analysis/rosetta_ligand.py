#!/usr/bin/env python3
"""
rosetta_ligand.py — Local ligand-focused refinement/scoring after RosettaRelax
============================================================================

This module is intentionally NOT AutoDock/Vina.
It runs a *local* refinement around the ligand starting from the relaxed complex.

Inputs:
  - Reads <job_dir>/latest_rosetta_relax.json produced by rosetta_minimize.py
  - Uses the canonical "relaxed_pdb" (restored for covalent runs)

Outputs:
  - <out_dir>/score.sc
  - <out_dir>/rosetta_ligand_outputs.json
  - <job_dir>/latest_rosetta_ligand.json  (pointer to latest run)

Multi-seed:
  - If latest_rosetta_relax.json indicates multi_seed, iterates per seed/sample subfolders
    and writes per-subfolder outputs + an index file in the run_dir.
"""

from __future__ import annotations

import json
import platform
import subprocess
import re
import shutil
from pathlib import Path, PurePosixPath
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Optional

from af3_pipeline.config import cfg

# =========================================================
# 🔧 Config / paths
# =========================================================
ROSETTA_BASE = (cfg.get("rosetta_relax_bin") or "").strip()
if not ROSETTA_BASE:
    raise RuntimeError(
        "Missing config key: rosetta_relax_bin (set to Rosetta bundle root, e.g. /home/<user>/rosetta/<bundle>)."
    )
ROSETTA_BASE = str(PurePosixPath(ROSETTA_BASE))

ROSETTA_SCRIPTS_BIN = f"{ROSETTA_BASE}/main/source/bin/rosetta_scripts.static.linuxgccrelease"
ROSETTA_DB = f"{ROSETTA_BASE}/main/database"

DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")
LINUX_HOME = cfg.get("linux_home_root", "")

APP_TZ = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))

def now_local() -> datetime:
    return datetime.now(APP_TZ)

WSL_EXE = r"C:\Windows\System32\wsl.exe"

def linuxize_path(p: Path) -> str:
    """Convert UNC/Windows paths → WSL Linux paths."""
    s = str(p)
    s = s.replace(f"\\\\wsl.localhost\\{DISTRO_NAME}", "")
    s = s.replace("\\", "/")
    s = s.replace("C:/Users", "/mnt/c/Users").replace("c:/Users", "/mnt/c/Users")
    if not (s.startswith("/home") or s.startswith("/mnt") or s.startswith("/tmp") or s.startswith("/var")):
        if not s.startswith("/"):
            s = f"{LINUX_HOME}/" + s.lstrip("/")
    return s

def linuxize_extra_res_fa(extra_res_fa: str) -> str:
    """
    Rosetta accepts multiple params files after -extra_res_fa.
    Convert each token to a WSL Linux path and return a safely-quoted string.
    """
    # split on whitespace (good enough for typical paths; params paths should not contain spaces)
    toks = [t.strip().strip("'").strip('"') for t in extra_res_fa.split() if t.strip()]
    linux_toks = [linuxize_path(Path(t)) for t in toks]
    return " ".join(f"'{t}'" for t in linux_toks)

def run_wsl(cmd: str, cwd_wsl: str | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command inside WSL via bash -lc. Capture stdout/stderr."""
    if cwd_wsl:
        cmd = f"pushd '{cwd_wsl}' >/dev/null 2>&1; {cmd}; rc=$?; popd >/dev/null 2>&1; exit $rc"

    if platform.system() == "Windows":
        full = [WSL_EXE, "-d", DISTRO_NAME, "--", "bash", "-lc", cmd]
    else:
        full = ["bash", "-lc", cmd]

    print("▶", " ".join(full))
    proc = subprocess.run(full, text=True, capture_output=True)

    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "COMMAND:\n" + " ".join(full) +
            "\n\n=== STDOUT ===\n" + proc.stdout +
            "\n=== STDERR ===\n" + proc.stderr,
            encoding="utf-8", errors="replace",
        )

    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"❌ WSL command failed (exit {proc.returncode}):\n{cmd}\n\nSTDERR:\n{proc.stderr}")

    return proc

# =========================================================
# 🧠 RosettaScripts XML (local refinement around ligand)
# =========================================================
def _rosetta_ligand_xml(*, neighbor_dist: float, is_covalent: bool, cst_file_wsl: str | None = None) -> str:
    cst_mover = ""
    cst_protocol = ""
    if is_covalent and cst_file_wsl:
        cst_mover = f'<ConstraintSetMover name="load_cst" add_constraints="1" cst_file="{cst_file_wsl}"/>'
        cst_protocol = '<Add mover="load_cst"/>'

    return f"""<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="sfxn" weights="ref2015_cart"/>
    <ScoreFunction name="sfxn_cst" weights="ref2015_cart">
      <Reweight scoretype="atom_pair_constraint" weight="25.0"/>
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <ResidueName name="lig" residue_name3="LIG"/>
    <Neighborhood name="nbr" selector="lig" distance="{neighbor_dist}" include_focus_in_subset="true"/>
    <Not name="not_nbr" selector="nbr"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="prevent_pack_outside" selector="not_nbr">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
  </TASKOPERATIONS>

  <MOVERS>
    {cst_mover}
    <FastRelax name="relax_local"
               scorefxn="sfxn_cst"
               repeats="3"
               cartesian="true"
               task_operations="prevent_pack_outside"/>
  </MOVERS>

  <PROTOCOLS>
    {cst_protocol}
    <Add mover="relax_local"/>
  </PROTOCOLS>

  <OUTPUT scorefxn="sfxn"/>
</ROSETTASCRIPTS>
"""


# =========================================================
# 📦 IO helpers
# =========================================================
def _load_latest_relax(job_dir: Path) -> dict[str, Any]:
    p = job_dir / "latest_rosetta_relax.json"
    if not p.exists():
        raise FileNotFoundError(f"latest_rosetta_relax.json not found in {job_dir}")
    return json.loads(p.read_text(encoding="utf-8"))

def _ensure_rosetta_scripts_available():
    # Check executable inside WSL
    cmd = f"test -x '{ROSETTA_SCRIPTS_BIN}'"
    run_wsl(cmd)

def _pick_relaxed_pdb(relax_record: dict[str, Any], base_dir: Path) -> Path:
    # Prefer canonical relaxed_pdb (restored if covalent)
    p = relax_record.get("relaxed_pdb") or relax_record.get("relaxed_pdb_raw")
    if not p:
        raise FileNotFoundError("Relax record missing relaxed_pdb/relaxed_pdb_raw")
    path = Path(p)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Relaxed PDB not found: {path}")
    return path

def _read_meta_flags(relax_record: dict[str, Any]) -> tuple[bool, str | None]:
    is_covalent = bool(relax_record.get("covalent"))
    extra_res_fa = relax_record.get("extra_res_fa")
    return is_covalent, extra_res_fa

def _write_ligand_outputs(
    *,
    job_dir: Path,
    out_dir: Path,
    input_pdb: Path,
    scorefile: Path,
    best_pdb: Path | None,
    is_covalent: bool,
    extra_res_fa: str | None,
    relax_record: dict[str, Any],
):
    out = {
        "out_dir": str(out_dir),
        "input_pdb": str(input_pdb),
        "scorefile": str(scorefile) if scorefile.exists() else None,
        "best_pdb": str(best_pdb) if (best_pdb and best_pdb.exists()) else None,
        "covalent": bool(is_covalent),
        "extra_res_fa": extra_res_fa,
        "from_relax": {
            "prep_dir": relax_record.get("prep_dir"),
            "relaxed_pdb": relax_record.get("relaxed_pdb"),
            "relaxed_pdb_raw": relax_record.get("relaxed_pdb_raw"),
            "scorefile": relax_record.get("scorefile"),
        },
    }
    (out_dir / "rosetta_ligand_outputs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (job_dir / "latest_rosetta_ligand.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

def _user_jobs_root() -> Path:
    root = cfg.get("jobs_root", None)
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".af3_pipeline" / "jobs").resolve()

def _bundle_ligand_outputs(job_dir: Path, out_dir: Path, out_record: dict[str, Any]) -> None:
    """
    Copy rosetta_ligand_<timestamp>/ into ~/.af3_pipeline/jobs/<job_name>/ and
    write latest_rosetta_ligand.json in the bundled job dir with bundled paths.
    """
    try:
        jobs_root = _user_jobs_root()  # you already have this in rosetta_minimize.py
        bundled_job_dir = jobs_root / job_dir.name
        bundled_job_dir.mkdir(parents=True, exist_ok=True)

        bundled_out_dir = bundled_job_dir / out_dir.name

        # Copy folder (merge/overwrite safely)
        if bundled_out_dir.exists():
            shutil.rmtree(bundled_out_dir)
        shutil.copytree(out_dir, bundled_out_dir)

        # Rewrite pointer so metrics.py (running on Windows) can read it
        bundled_record = dict(out_record)
        bundled_record["out_dir"] = str(bundled_out_dir)
        if bundled_record.get("scorefile"):
            bundled_record["scorefile"] = str(bundled_out_dir / "score.sc")
        if bundled_record.get("best_pdb"):
            # best_pdb may not exist / may be None
            best_name = Path(bundled_record["best_pdb"]).name
            bundled_record["best_pdb"] = str(bundled_out_dir / best_name)

        (bundled_job_dir / "latest_rosetta_ligand.json").write_text(
            json.dumps(bundled_record, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"⚠️ Bundling ligand outputs failed: {e}")

def _pdb_pose_index_map(pdb_path: Path) -> dict[tuple[str, int, str], int]:
    """
    Build a mapping (chain, resseq, resname3) -> pose_index (1-based)
    by scanning ATOM/HETATM lines in file order and counting unique residues.
    This matches Rosetta's typical pose construction for a PDB input.
    """
    pose_map: dict[tuple[str, int, str], int] = {}
    seen: set[tuple[str, int, str]] = set()
    pose_i = 0

    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 26:
            continue

        chain = (line[21] or " ").strip() or "A"
        resname = line[17:20].strip()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue

        key = (chain, resseq, resname)
        if key in seen:
            continue

        seen.add(key)
        pose_i += 1
        pose_map[key] = pose_i

    return pose_map


def _find_first_lig_pose_index(pose_map: dict[tuple[str, int, str], int], *, lig_resname: str = "LIG") -> int | None:
    """
    Return pose index of the first residue with resname==lig_resname, by lowest pose index.
    """
    hits = [(pose, key) for key, pose in pose_map.items() if key[2] == lig_resname]
    if not hits:
        return None
    hits.sort(key=lambda x: x[0])
    return hits[0][0]


def _write_covalent_atom_pair_constraint(
    *,
    out_cst: Path,
    input_pdb: Path,
    covalent_meta: dict[str, Any],
    lig_atom: str,
    lig_resname: str = "LIG",
    dist: float = 1.80,
    sd: float = 0.10,
) -> tuple[str, int, str, int]:
    """
    Writes AtomPair constraint using *pose indices* derived from input_pdb.
    Returns (prot_atom, prot_pose, lig_atom, lig_pose).
    """
    prot_atom = (covalent_meta.get("prot_atom") or "").strip().upper()
    if not prot_atom:
        raise ValueError("covalent_meta missing 'prot_atom'")

    # This is the PDB residue number (e.g., 79) and we assume chain A unless provided.
    # If you later store chain in covalent_meta, we'll respect it.
    target_resseq = int(covalent_meta.get("residue"))
    prot_chain = (covalent_meta.get("chain") or "A").strip()[:1] or "A"

    pose_map = _pdb_pose_index_map(input_pdb)

    # Protein pose index: try exact resname match first, then fall back to "any resname on that chain/resseq"
    prot_pose = None
    # Most of your runs are CYS or LYS etc; but we don't want to hardcode resname.
    candidates = [(key, pose) for key, pose in pose_map.items() if key[0] == prot_chain and key[1] == target_resseq]
    if candidates:
        # If multiple (rare), choose lowest pose index
        candidates.sort(key=lambda x: x[1])
        prot_pose = candidates[0][1]

    if prot_pose is None:
        raise RuntimeError(f"Could not map protein residue chain {prot_chain} resseq {target_resseq} to a pose index in {input_pdb}")

    lig_pose = _find_first_lig_pose_index(pose_map, lig_resname=lig_resname)
    if lig_pose is None:
        raise RuntimeError(f"Could not find ligand residue '{lig_resname}' in {input_pdb}. Is your ligand resname actually LIG?")

    lig_atom = lig_atom.strip().upper()
    if not lig_atom:
        raise ValueError("lig_atom is empty")

    out_cst.parent.mkdir(parents=True, exist_ok=True)
    out_cst.write_text(
        f"AtomPair {prot_atom} {prot_pose} {lig_atom} {lig_pose} HARMONIC {dist:.2f} {sd:.2f}\n",
        encoding="utf-8",
    )

    return prot_atom, prot_pose, lig_atom, lig_pose

# =========================================================
# 🚀 Core runner (single directory)
# =========================================================
def _run_one(job_dir: Path, relax_record: dict[str, Any], *, out_parent: Path | None = None) -> dict[str, Any]:
    _ensure_rosetta_scripts_available()

    is_covalent, extra_res_fa = _read_meta_flags(relax_record)
    
    # Parameters (configurable)
    neighbor_dist = float(cfg.get("rosetta_ligand_neighbor_dist", 6.0))
    nstruct = int(cfg.get("rosetta_ligand_nstruct", 5 if not is_covalent else 1))

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    out_dir = (out_parent or job_dir) / f"rosetta_ligand_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick input PDB first (we need it to compute pose indices)
    input_pdb = _pick_relaxed_pdb(relax_record, job_dir)

    # Covalent constraints (AtomPair) — write using pose indices derived from input_pdb
    cst_path: Path | None = None
    cst_file_wsl: str | None = None
    covalent_meta = relax_record.get("covalent_meta") or {}

    if is_covalent:
        # Try to discover ligand reactive atom name from metadata; default to C7 (your case)
        lig_atom = (
            covalent_meta.get("lig_atom")
            or covalent_meta.get("ligand_atom")
            or relax_record.get("lig_atom")
            or relax_record.get("ligand_atom")
            or "C7"
        )

        # Allow overriding distance/sd from config
        cst_dist = float(cfg.get("covalent_constraint_dist", 1.80))
        cst_sd = float(cfg.get("covalent_constraint_sd", 0.10))

        cst_path = out_dir / "constraints.cst"
        prot_atom, prot_pose, lig_atom_u, lig_pose = _write_covalent_atom_pair_constraint(
            out_cst=cst_path,
            input_pdb=input_pdb,
            covalent_meta=covalent_meta,
            lig_atom=str(lig_atom),
            lig_resname="LIG",
            dist=cst_dist,
            sd=cst_sd,
        )
        cst_file_wsl = linuxize_path(cst_path)

        print(f"🔗 Wrote covalent constraint: AtomPair {prot_atom} {prot_pose} {lig_atom_u} {lig_pose} (pose indices)")


    # Write RosettaScripts XML
    xml_path = out_dir / "rosetta_ligand.xml"
    xml_path.write_text(
        _rosetta_ligand_xml(neighbor_dist=neighbor_dist, is_covalent=is_covalent, cst_file_wsl=cst_file_wsl),
        encoding="utf-8",
    )

    # Run rosetta_scripts
    out_dir_linux = linuxize_path(out_dir)
    xml_linux = linuxize_path(xml_path)
    pdb_linux = linuxize_path(input_pdb)
    scorefile = out_dir / "score.sc"

    extra_res_flags = ""
    if extra_res_fa:
        extra_res_flags += f" -extra_res_fa {linuxize_extra_res_fa(extra_res_fa)}"

    packing_flags = " -ex1 -ex2 -ex2aro -use_input_sc -flip_HNQ"

    cst_flags = ""
    if is_covalent and cst_file_wsl:
        cst_flags = f" -constraints:cst_file '{cst_file_wsl}'"

    cmd = (
        f"'{ROSETTA_SCRIPTS_BIN}' -database '{ROSETTA_DB}' "
        f"-s '{pdb_linux}' "
        f"-parser:protocol '{xml_linux}' "
        f"-out:path:all '{out_dir_linux}' "
        f"-out:file:scorefile '{out_dir_linux}/score.sc' "
        f"-nstruct {nstruct} -overwrite "
        f"-relax:cartesian "
        f"{extra_res_flags}"
        f"{packing_flags}"
        f"{cst_flags}"
    )

    run_wsl(cmd, log=out_dir / "rosetta_scripts.log")

    # Pick a "best" PDB if present: look for output PDBs in out_dir
    # RosettaScripts usually writes *_0001.pdb / *_0002.pdb, etc.
    best_pdb = None
    pdbs = sorted(out_dir.glob("*.pdb"), key=lambda p: p.stat().st_mtime)
    if pdbs:
        best_pdb = pdbs[-1]

    out_record = _write_ligand_outputs(
        job_dir=job_dir,
        out_dir=out_dir,
        input_pdb=input_pdb,
        scorefile=scorefile,
        best_pdb=best_pdb,
        is_covalent=is_covalent,
        extra_res_fa=extra_res_fa,
        relax_record=relax_record,
    )
    _bundle_ligand_outputs(job_dir, out_dir, out_record)
    return out_record

# =========================================================
# 🧩 Public entrypoint
# =========================================================
def run(job_dir: str | Path, multi_seed: bool = False):
    job_dir = Path(job_dir)

    relax_root = _load_latest_relax(job_dir)

    # If rosetta_minimize wrote the multi_seed pointer object, follow it.
    if bool(relax_root.get("multi_seed")) or bool(multi_seed):
        idx_path = relax_root.get("index")
        if not idx_path:
            raise FileNotFoundError("Multi-seed mode requested but latest_rosetta_relax.json has no 'index' field.")
        idx_path = Path(idx_path)
        if not idx_path.exists():
            raise FileNotFoundError(f"Multi-seed relax index not found: {idx_path}")

        entries = json.loads(idx_path.read_text(encoding="utf-8"))
        run_dir = Path(relax_root.get("run_dir")) if relax_root.get("run_dir") else idx_path.parent

        out_index: list[dict[str, Any]] = []
        for entry in entries:
            # Each entry's prep_dir is the sub_dir (seed/sample folder)
            prep_dir = Path(entry.get("prep_dir", ""))
            if not prep_dir.exists():
                print(f"⚠️ Missing prep_dir, skipping: {prep_dir}")
                continue

            print(f"🧲 RosettaLigand for {prep_dir.name} ...")
            # Write outputs under that same prep_dir (keeps seed/sample self-contained)
            out = _run_one(prep_dir, entry, out_parent=prep_dir)
            out_index.append(out)

        # Write a multi-seed ligand index under run_dir
        ligand_index = run_dir / "rosetta_ligand_multi_index.json"
        ligand_index.write_text(json.dumps(out_index, indent=2), encoding="utf-8")

        # Point job_dir at the latest ligand index
        (job_dir / "latest_rosetta_ligand.json").write_text(
            json.dumps({"multi_seed": True, "run_dir": str(run_dir), "index": str(ligand_index)}, indent=2),
            encoding="utf-8",
        )

        print(f"✅ RosettaLigand multi-seed complete → {ligand_index}")
        return out_index

    # Single job
    return _run_one(job_dir, relax_root, out_parent=job_dir)

# =========================================================
# CLI (optional)
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run local RosettaLigand-style refinement around the ligand.")
    ap.add_argument("--job", required=True, help="Job directory (bundled job root)")
    ap.add_argument("--multi_seed", action="store_true", help="Run for each seed/sample")
    args = ap.parse_args()
    run(Path(args.job), multi_seed=bool(args.multi_seed))
