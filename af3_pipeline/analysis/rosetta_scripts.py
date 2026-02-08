#!/usr/bin/env python3
"""
rosetta_scripts.py — Local ligand-focused refinement/scoring (RosettaScripts)
============================================================================

Refactor goals:
- Can run in two modes:
    1) From latest RosettaRelax outputs (default)  -> uses latest_rosetta_relax.json
    2) AF3-direct mode (skip_rosetta=True)        -> prepares PDB + ligand params itself

- Pulls protocol defaults from rosetta_dicts.yaml via rosetta_config.py
- Uses prep_lig_for_rosetta.py for ligand params generation
- Implements "covalent proxy patch" when needed:
    * If relax ran: relax output already has restored atom + (optional) resname alias (e.g. LYD)
    * If relax did NOT run: patch the prepared input PDB BEFORE RosettaScripts
      to swap residue 3-letter name to alias (e.g. LYS -> LYD), and ensure the alias
      params are provided via -in:file:extra_res_fa.

Key enhancement in this version:
- ALWAYS ensures ligand params (e.g., LIG.params) are available when RosettaScripts
  is the only stage you run (AF3-direct), and also robustly handles "from_relax"
  when extra_res_fa is missing by regenerating ligand params from metadata SMILES.

Outputs:
  - <out_dir>/score.sc
  - <out_dir>/rosetta_scripts_outputs.json
  - <job_dir>/latest_rosetta_scripts.json  (pointer to latest run)
"""

from __future__ import annotations

import json
import platform
import re
import shutil
from pathlib import Path, PurePosixPath
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Optional, Tuple, List

from af3_pipeline.config import cfg
from .rosetta_config import load_rosetta_dicts
from .rosetta_runtime import run_wsl, linuxize_path, wsl_unc_from_linux
from .prep_lig_for_rosetta import prep_ligand_for_rosetta
from .rosetta_runtime import (
    run_wsl,
    linuxize_path,
    safe_write_text,
    extra_res_fa_from_list,
)


# =========================================================
# Config / paths
# =========================================================
ROSETTA_BASE = (cfg.get("rosetta_relax_bin") or "").strip()
if not ROSETTA_BASE:
    raise RuntimeError(
        "Missing config key: rosetta_relax_bin (set to Rosetta bundle root, e.g. /home/<user>/rosetta/<bundle>)."
    )
ROSETTA_BASE = str(PurePosixPath(ROSETTA_BASE))

ROSETTA_SCRIPTS_BIN = f"{ROSETTA_BASE}/main/source/bin/rosetta_scripts.static.linuxgccrelease"
ROSETTA_DB = f"{ROSETTA_BASE}/main/database"
CLEAN_PDB_PY = f"{ROSETTA_BASE}/main/source/src/apps/public/relax_w_allatom_cst/clean_pdb_keep_ligand.py"

DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")
APP_TZ = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))


def now_local() -> datetime:
    return datetime.now(APP_TZ)


# =========================================================
# RosettaScripts XML (local refinement around ligand)
# =========================================================
def _rosetta_ligand_xml(
    *,
    neighbor_dist: float,
    lig_chain: str,
    is_covalent: bool,
    # protocol knobs
    local_rb_enabled: bool,
    local_rb_translate: dict[str, Any],
    local_rb_rotate: dict[str, Any],
    local_rb_slide_together: bool,
    highres_cycles: int,
    highres_repack_every_nth: int,
    ligand_area: dict[str, Any],
    # scorefunctions
    sfxn_soft_weights: str,
    sfxn_hires_weights: str,
    sfxn_hires_cst_base: str,
    sfxn_hires_cst_reweights: dict[str, float],
    # constraints
    cst_file_wsl: str | None = None,
) -> str:
    """
    Rosetta 2021.16-compatible RosettaLigand-style protocol.

    Uses YAML for:
      - Local RB wiggle knobs
      - HighResDocker knobs
      - LigandArea knobs (covalent vs noncovalent)
      - Scorefunction names + reweights
      - Optional constraint load via ConstraintSetMover
    """
    # -------------------------
    # Constraints wiring
    # -------------------------
    cst_mover = ""
    cst_protocol = ""
    scorefxn_hires = "sfxn_hires"

    if cst_file_wsl:
        cst_mover = (
            f'<ConstraintSetMover name="load_cst" add_constraints="1" cst_file="{cst_file_wsl}"/>'
        )
        cst_protocol = '<Add mover="load_cst"/>'
        scorefxn_hires = "sfxn_hires_cst"

    # -------------------------
    # LigandArea knobs
    # -------------------------
    mode_key = "covalent" if is_covalent else "noncovalent"
    la = (ligand_area or {}).get(mode_key, {}) if isinstance(ligand_area, dict) else {}
    high_res_angstroms = str(la.get("high_res_angstroms", 0.20 if is_covalent else 0.50))
    high_res_degrees   = str(la.get("high_res_degrees",   5 if is_covalent else 10))
    tether_ligand      = str(la.get("tether_ligand",      1.0 if is_covalent else 0.0))
    minimize_ligand    = str(la.get("minimize_ligand",    2.0 if is_covalent else 5.0))

    # -------------------------
    # Local RB block (noncovalent only)
    # -------------------------
    rb_block = ""
    rb_protocol = ""
    do_local_rb = (not is_covalent) and bool(local_rb_enabled)

    if do_local_rb:
        t = local_rb_translate or {}
        r = local_rb_rotate or {}
        t_ang = float(t.get("angstroms", 1.5))
        t_cyc = int(t.get("cycles", 25))
        t_dist = str(t.get("distribution", "gaussian"))

        r_deg = float(r.get("degrees", 15))
        r_cyc = int(r.get("cycles", 50))
        r_dist = str(r.get("distribution", "gaussian"))

        slide = bool(local_rb_slide_together)

        rb_block = f"""
    <Translate name="rb_translate" chain="{lig_chain}" distribution="{t_dist}" angstroms="{t_ang}" cycles="{t_cyc}"/>
    <Rotate    name="rb_rotate"    chain="{lig_chain}" distribution="{r_dist}" degrees="{r_deg}" cycles="{r_cyc}"/>
"""
        rb_protocol = """
    <Add mover="rb_translate"/>
    <Add mover="rb_rotate"/>
"""
        if slide:
            rb_block += f'    <SlideTogether name="rb_slide" chains="{lig_chain}"/>\n'
            rb_protocol += '    <Add mover="rb_slide"/>\n'

    # -------------------------
    # Scorefunction reweights
    # -------------------------
    reweight_lines = []
    for k, v in (sfxn_hires_cst_reweights or {}).items():
        try:
            reweight_lines.append(f'      <Reweight scoretype="{k}" weight="{float(v)}"/>')
        except Exception:
            continue
    reweight_block = "\n".join(reweight_lines)

    return f"""<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="sfxn_soft"  weights="{sfxn_soft_weights}"/>
    <ScoreFunction name="sfxn_hires" weights="{sfxn_hires_weights}"/>
    <ScoreFunction name="sfxn_hires_cst" weights="{sfxn_hires_cst_base}">
{reweight_block}
    </ScoreFunction>
  </SCOREFXNS>

  <LIGAND_AREAS>
    <LigandArea name="lig1"
                chain="{lig_chain}"
                cutoff="{neighbor_dist}"
                add_nbr_radius="true"
                all_atom_mode="false"
                minimize_ligand="{minimize_ligand}"
                Calpha_restraints="0.0"
                high_res_angstroms="{high_res_angstroms}"
                high_res_degrees="{high_res_degrees}"
                tether_ligand="{tether_ligand}"/>
  </LIGAND_AREAS>

  <INTERFACE_BUILDERS>
    <InterfaceBuilder name="iface"
                      ligand_areas="lig1"
                      extension_window="3"/>
  </INTERFACE_BUILDERS>

  <MOVEMAP_BUILDERS>
    <MoveMapBuilder name="mm"
                    sc_interface="iface"
                    minimize_water="false"/>
  </MOVEMAP_BUILDERS>

  <MOVERS>
    {cst_mover}
{rb_block}
    <HighResDocker name="hires"
                   cycles="{int(highres_cycles)}"
                   repack_every_Nth="{int(highres_repack_every_nth)}"
                   scorefxn="{scorefxn_hires}"
                   movemap_builder="mm"/>

    <FinalMinimizer name="final"
                    scorefxn="{scorefxn_hires}"
                    movemap_builder="mm"/>

    <InterfaceScoreCalculator name="iface_scores"
                              chains="{lig_chain}"
                              scorefxn="{scorefxn_hires}"/>
  </MOVERS>

  <PROTOCOLS>
    {cst_protocol}
{rb_protocol}
    <Add mover="hires"/>
    <Add mover="final"/>
    <Add mover="iface_scores"/>
  </PROTOCOLS>

  <OUTPUT scorefxn="{scorefxn_hires}"/>
</ROSETTASCRIPTS>
"""



# =========================================================
# IO helpers
# =========================================================
def _safe_name(s: str) -> str:
    s = (s or "").strip().replace("\r", "").replace("\n", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _user_jobs_root() -> Path:
    root = cfg.get("jobs_root", None)
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".af3_pipeline" / "jobs").resolve()

def _resolve_job_dir(job_dir: Path) -> Path:
    if job_dir.is_absolute() and job_dir.exists():
        return job_dir.resolve()
    if job_dir.exists():
        return job_dir.resolve()
    cand = _user_jobs_root() / job_dir.name
    if cand.exists():
        return cand.resolve()
    return job_dir.resolve()

def _load_latest_relax(job_dir: Path) -> dict[str, Any]:
    p = job_dir / "latest_rosetta_relax.json"
    if not p.exists():
        raise FileNotFoundError(f"latest_rosetta_relax.json not found in {job_dir}")
    return json.loads(p.read_text(encoding="utf-8"))

def _load_job_metadata(job_dir: Path) -> dict[str, Any]:
    # prefer prepared_meta.json, then job_metadata.json
    for name in ("prepared_meta.json", "job_metadata.json"):
        p = job_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    # also allow timestamped metadata used in your old code
    cand = job_dir / f"{job_dir.name}_job_metadata.json"
    if cand.exists():
        return json.loads(cand.read_text(encoding="utf-8"))
    return {}

def _write_outputs_pointer(job_dir: Path, out_dir: Path, out_record: dict[str, Any]) -> None:
    (out_dir / "rosetta_scripts_outputs.json").write_text(json.dumps(out_record, indent=2), encoding="utf-8")
    (job_dir / "latest_rosetta_scripts.json").write_text(json.dumps(out_record, indent=2), encoding="utf-8")

def _bundle_outputs(job_dir: Path, out_dir: Path, out_record: dict[str, Any]) -> None:
    """
    Copy rosetta_scripts_<timestamp>/ into ~/.af3_pipeline/jobs/<job_name>/ and
    write latest_rosetta_scripts.json in the bundled job dir with bundled paths.
    """
    try:
        jobs_root = _user_jobs_root().resolve()
        out_dir_res = out_dir.resolve()

        if jobs_root in out_dir_res.parents:
            bundled_job_dir = jobs_root / _safe_name(job_dir.name)
            bundled_job_dir.mkdir(parents=True, exist_ok=True)
            (bundled_job_dir / "latest_rosetta_scripts.json").write_text(json.dumps(out_record, indent=2), encoding="utf-8")
            return

        bundled_job_dir = jobs_root / _safe_name(job_dir.name)
        bundled_job_dir.mkdir(parents=True, exist_ok=True)
        bundled_out_dir = bundled_job_dir / _safe_name(out_dir.name)

        if bundled_out_dir.exists():
            shutil.rmtree(bundled_out_dir)
        shutil.copytree(out_dir, bundled_out_dir)

        bundled_record = dict(out_record)
        bundled_record["out_dir"] = str(bundled_out_dir)
        if bundled_record.get("scorefile"):
            bundled_record["scorefile"] = str(bundled_out_dir / "score.sc")
        if bundled_record.get("best_pdb"):
            best_name = Path(bundled_record["best_pdb"]).name
            bundled_record["best_pdb"] = str(bundled_out_dir / best_name)

        (bundled_job_dir / "latest_rosetta_scripts.json").write_text(json.dumps(bundled_record, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Bundling scripts outputs failed: {e}")

_CST_LINE_RE = re.compile(
    r"^(AtomPair|Angle|Dihedral|CoordinateConstraint)\s+(.+)$"
)

def _find_ligand_pose_index_in_pdb(
    pdb_path: Path,
    *,
    lig_resname: str = "LIG",
    lig_atom: str | None = None,
) -> int:
    """
    Return the 1-based *pose index* of the ligand residue in the PDB Rosetta will read.

    Strategy:
      1) If lig_atom provided: find first residue of resname==lig_resname that contains that atom.
      2) Else: first residue with resname==lig_resname.
    Pose index is computed by first occurrence order of unique (chain, resseq, resname) across ATOM/HETATM lines.
    """
    lig_resname = (lig_resname or "LIG").strip().upper()[:3]
    lig_atom_u = (lig_atom or "").strip().upper() if lig_atom else None

    # Build pose_map in the same way as your structured constraints code
    pose_map = _pdb_pose_index_map(pdb_path)  # keys (chain, resseq, resname) -> pose_idx

    # We also need an ordered list of residues (pose_idx -> key) to scan atoms
    # We'll reconstruct by sorting pose_map items by pose idx:
    ordered = sorted(pose_map.items(), key=lambda kv: kv[1])  # [((chain,resseq,resname), pose_idx), ...]

    if lig_atom_u:
        # Scan PDB lines and record which residue keys contain lig_atom
        atom_hits: set[tuple[str, int, str]] = set()
        for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
                continue
            resname = line[17:20].strip().upper()
            if resname != lig_resname:
                continue
            atom = line[12:16].strip().upper()
            if atom != lig_atom_u:
                continue
            chain = (line[21] or " ").strip() or "A"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            atom_hits.add((chain, resseq, resname))

        for key, pose_idx in ordered:
            if key in atom_hits:
                return int(pose_idx)

    # Fallback: first residue of that resname
    for (chain, resseq, resname), pose_idx in ordered:
        if resname.upper() == lig_resname:
            return int(pose_idx)

    raise RuntimeError(f"Could not find ligand resname '{lig_resname}' in PDB: {pdb_path}")


def _rewrite_custom_cst_ligand_residue_numbers(
    cst_in: Path,
    *,
    input_pdb: Path,
    prot_pose_res: int,
    lig_resname: str = "LIG",
    lig_atom: str | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Rewrite custom constraint file so that all *residue number tokens* that are NOT prot_pose_res
    are replaced with the ligand pose residue index for input_pdb.

    IMPORTANT:
    - Does not touch atom names like ND1 / S1 / CE1 / NE2.
    - Preserves line breaks.
    """
    lig_pose_res = _find_ligand_pose_index_in_pdb(
        input_pdb, lig_resname=lig_resname, lig_atom=lig_atom
    )
    prot_pose_res = int(prot_pose_res)
    lig_pose_res = int(lig_pose_res)

    lines = cst_in.read_text(encoding="utf-8", errors="replace").splitlines(True)  # keep \n

    def _replace_resnums_for_line(raw_line: str) -> str:
        line = raw_line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            return raw_line  # keep as-is (including newline)

        parts = line.split()
        if not parts:
            return raw_line

        kind = parts[0]

        # AtomPair:  AtomPair A1 R1 A2 R2 FUNC x0 sd
        if kind == "AtomPair" and len(parts) >= 7:
            # residue tokens are parts[2] and parts[4]
            for idx in (2, 4):
                if parts[idx].isdigit():
                    n = int(parts[idx])
                    if n != prot_pose_res:
                        parts[idx] = str(lig_pose_res)
            return " ".join(parts) + ("\n" if raw_line.endswith("\n") else "")

        # Angle: Angle A1 R1 A2 R2 A3 R3 FUNC x0 sd
        if kind == "Angle" and len(parts) >= 9:
            # residue tokens are parts[2], parts[4], parts[6]
            for idx in (2, 4, 6):
                if parts[idx].isdigit():
                    n = int(parts[idx])
                    if n != prot_pose_res:
                        parts[idx] = str(lig_pose_res)
            return " ".join(parts) + ("\n" if raw_line.endswith("\n") else "")

        # Dihedral: Dihedral A1 R1 A2 R2 A3 R3 A4 R4 FUNC x0 sd
        if kind == "Dihedral" and len(parts) >= 11:
            # residue tokens are parts[2], parts[4], parts[6], parts[8]
            for idx in (2, 4, 6, 8):
                if parts[idx].isdigit():
                    n = int(parts[idx])
                    if n != prot_pose_res:
                        parts[idx] = str(lig_pose_res)
            return " ".join(parts) + ("\n" if raw_line.endswith("\n") else "")

        # CoordinateConstraint: CoordinateConstraint ATOM RES X Y Z FUNC x0 sd
        if kind == "CoordinateConstraint" and len(parts) >= 9:
            # residue token is parts[2]
            if parts[2].isdigit():
                n = int(parts[2])
                if n != prot_pose_res:
                    parts[2] = str(lig_pose_res)
            return " ".join(parts) + ("\n" if raw_line.endswith("\n") else "")

        # Unknown / unhandled: keep line unchanged
        return raw_line

    out_lines = [_replace_resnums_for_line(ln) for ln in lines]

    if out_path is None:
        out_path = cst_in.parent / f"{cst_in.stem}_posefix{cst_in.suffix}"

    out_path.write_text("".join(out_lines), encoding="utf-8")

    print(
        f"🧷 Rewrote custom constraints residue numbers: protein_pose={prot_pose_res}, ligand_pose={lig_pose_res}\n"
        f"    in:  {cst_in}\n"
        f"    out: {out_path}",
        flush=True,
    )
    return out_path

def _compute_protein_pose_index(
    input_pdb: Path,
    *,
    prot_chain: str,
    prot_resseq: int,
    prot_resname_hint: str | None = None,
) -> int:
    """
    Map the protein residue (chain + PDB resseq [+ optional resname]) to Rosetta's
    1-based pose residue index for the PDB Rosetta will actually run on.
    """
    pose_map = _pdb_pose_index_map(input_pdb)

    chain = (prot_chain or "A").strip()[:1] or "A"
    resseq = int(prot_resseq)
    rh = prot_resname_hint.strip().upper()[:3] if prot_resname_hint else None

    # Uses your existing logic: exact resname match if provided, else fallback to first match
    return _pose_index_for_atomref(pose_map, chain=chain, resseq=resseq, resname_hint=rh)

def _ensure_rosetta_scripts_available():
    cmd = f"test -x '{ROSETTA_SCRIPTS_BIN}'"
    run_wsl(cmd)

def _detect_ligand_chain_and_resname(
    pdb_path: Path,
    *,
    prefer_resnames: tuple[str, ...] = ("LIG",),
) -> tuple[str, str]:
    lines = pdb_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for resname_pref in prefer_resnames:
        for line in lines:
            if not (line.startswith("HETATM") or line.startswith("ATOM")):
                continue
            if len(line) < 26:
                continue
            resname = line[17:20].strip()
            if resname != resname_pref:
                continue
            chain = (line[21] or " ").strip() or "A"
            return chain, resname

    skip = {"HOH", "WAT"}
    for line in lines:
        if not line.startswith("HETATM"):
            continue
        if len(line) < 26:
            continue
        resname = line[17:20].strip()
        if resname in skip:
            continue
        chain = (line[21] or " ").strip() or "A"
        return chain, resname

    return "X", "LIG"

def _looks_like_linux_abs_path(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("/") and not s.startswith("//")

def _host_path_for_maybe_linux_path(p: str | Path) -> Path:
    """
    Return a host-visible Path that exists on the current OS.
    - On Windows: convert /home/... or /mnt/... to UNC via wsl_unc_from_linux
    - Else: return Path(p)
    """
    s = str(p).strip()
    if not s:
        return Path()
    if platform.system() == "Windows" and _looks_like_linux_abs_path(s):
        return wsl_unc_from_linux(s)
    return Path(s)

def _resolve_and_stage_constraints_file(
    constraints_file: str | Path,
    *,
    out_dir: Path,
    job_dir: Path,
) -> Path:
    """
    Resolve a user-provided constraints file and copy it into out_dir for reproducibility.
    Accepts either:
      - WSL POSIX path: /home/olive/.../constraints/foo.cst
      - Windows UNC path: \\wsl.localhost\\Ubuntu-22.04\\home\\olive\\...\\foo.cst
      - Relative path: interpreted relative to job_dir
    Returns the staged file path (host-visible Path in out_dir).
    """
    s = str(constraints_file).strip()
    if not s:
        raise ValueError("constraints_file was provided but empty")

    # Relative -> assume relative to job_dir (host path)
    if not Path(s).is_absolute() and not _looks_like_linux_abs_path(s):
        s = str((job_dir / s).resolve())

    src_host = _host_path_for_maybe_linux_path(s)

    if not src_host.exists():
        raise FileNotFoundError(f"Constraints file not found (host path): {src_host}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # stage into run dir
    staged = out_dir / f"custom_{src_host.name}"
    shutil.copy2(src_host, staged)

    # record provenance
    try:
        (out_dir / "custom_constraints_source.txt").write_text(f"{s}\n", encoding="utf-8")
    except Exception:
        pass

    return staged



# =========================================================
# Pose index mapping for constraints
# =========================================================
def _pdb_pose_index_map(pdb_path: Path) -> dict[tuple[str, int, str], int]:
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
    hits = [(pose, key) for key, pose in pose_map.items() if key[2] == lig_resname]
    if not hits:
        return None
    hits.sort(key=lambda x: x[0])
    return hits[0][0]

def _get_stage_cfg(dicts) -> dict[str, Any]:
    """
    Supports either:
      1) dict-like: dicts["rosetta_scripts_stage"] or dicts.get("rosetta_scripts_stage")
      2) config-wrapper: dicts.get(section, key, default=...)
         where stage is under section="rosetta_scripts_stage" and key=None or "".
    """
    # plain dict
    if isinstance(dicts, dict):
        stage = dicts.get("rosetta_scripts_stage", {}) or {}
        return stage if isinstance(stage, dict) else {}

    # wrapper with get(section, key, default)
    try:
        stage = dicts.get("rosetta_scripts_stage", None, default={}) or {}
        return stage if isinstance(stage, dict) else {}
    except TypeError:
        # wrapper that also supports dict-style get(key, default)
        stage = dicts.get("rosetta_scripts_stage", {}) or {}
        return stage if isinstance(stage, dict) else {}


def _get_protocol_cfg(stage: dict[str, Any]) -> dict[str, Any]:
    p = stage.get("protocol", {}) if isinstance(stage, dict) else {}
    return p if isinstance(p, dict) else {}

def _get_scorefxn_cfg(stage: dict[str, Any]) -> dict[str, Any]:
    s = stage.get("scorefunctions", {}) if isinstance(stage, dict) else {}
    return s if isinstance(s, dict) else {}

def _get_constraints_cfg(stage: dict[str, Any]) -> dict[str, Any]:
    c = stage.get("constraints", {}) if isinstance(stage, dict) else {}
    return c if isinstance(c, dict) else {}

def _pose_index_for_atomref(
    pose_map: dict[tuple[str, int, str], int],
    *,
    chain: str,
    resseq: int,
    resname_hint: str | None = None,
) -> int:
    """
    Map (chain, pdb resseq[, resname_hint]) -> pose residue index.
    Your pose_map keys are (chain, resseq, resname) with pose_i values.
    Prefer exact resname match if provided, else take lowest pose match for chain+resseq.
    """
    chain = (chain or "A").strip()[:1] or "A"
    if resname_hint:
        rh = resname_hint.strip().upper()[:3]
        k = (chain, int(resseq), rh)
        if k in pose_map:
            return pose_map[k]
    # fallback: any resname at that chain+resseq
    hits = [(key, pose) for key, pose in pose_map.items() if key[0] == chain and key[1] == int(resseq)]
    if not hits:
        raise RuntimeError(f"Could not map chain {chain} resseq {resseq} to a pose index")
    hits.sort(key=lambda x: x[1])
    return hits[0][1]

def _iter_active_constraint_items(
    *,
    constraints_cfg: dict[str, Any],
    mode: str,              # "covalent"|"noncovalent"
    stage_name: str,        # e.g. "stage2"
) -> list[dict[str, Any]]:
    """
    Returns a flat list of item dicts from enabled sets selected by active_sets[mode],
    plus any stages[stage_name].extra_sets[mode], skipping disabled sets.
    """
    mode = mode.strip().lower()
    sets_root = constraints_cfg.get("sets", {})
    if not isinstance(sets_root, dict):
        sets_root = {}

    # stage enable
    stages = constraints_cfg.get("stages", {})
    if isinstance(stages, dict):
        st = stages.get(stage_name, {})
        if isinstance(st, dict) and ("enabled" in st) and (not bool(st.get("enabled"))):
            return []

    active_sets = []
    active_sets_cfg = constraints_cfg.get("active_sets", {})
    if isinstance(active_sets_cfg, dict):
        v = active_sets_cfg.get(mode, [])
        if isinstance(v, list):
            active_sets.extend([str(x).strip() for x in v if str(x).strip()])

    # stage extras
    if isinstance(stages, dict):
        st = stages.get(stage_name, {})
        if isinstance(st, dict):
            extra_sets_cfg = st.get("extra_sets", {})
            if isinstance(extra_sets_cfg, dict):
                vv = extra_sets_cfg.get(mode, [])
                if isinstance(vv, list):
                    active_sets.extend([str(x).strip() for x in vv if str(x).strip()])

    # collect items
    out: list[dict[str, Any]] = []
    for set_name in active_sets:
        s = sets_root.get(set_name)
        if not isinstance(s, dict):
            continue
        if ("enabled" in s) and (not bool(s.get("enabled"))):
            continue
        items = s.get("items", [])
        if not isinstance(items, list):
            continue
        for it in items:
            if isinstance(it, dict):
                out.append(it)
    return out

def _weights_for_mode(constraints_cfg: dict[str, Any], mode: str) -> dict[str, float]:
    mode = mode.strip().lower()
    w = constraints_cfg.get("weights", {})
    if not isinstance(w, dict):
        return {}
    mw = w.get(mode, {})
    if not isinstance(mw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in mw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out

def _pdb_find_atom_xyz(
    pdb_path: Path,
    *,
    chain: str,
    resseq: int,
    atom_name: str,
    resname_hint: str | None = None,
) -> tuple[float, float, float]:
    """
    Find the first matching ATOM/HETATM line and return its (x,y,z).
    Matches on chain + resseq + atom_name; optionally also resname.
    """
    chain = (chain or "A").strip()[:1] or "A"
    resseq = int(resseq)
    atom_name = (atom_name or "").strip().upper()
    resname_hint_u = (resname_hint or "").strip().upper()[:3] if resname_hint else None

    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            continue

        # PDB fixed columns
        a_name = line[12:16].strip().upper()
        r_name = line[17:20].strip().upper()
        ch = (line[21] or " ").strip() or "A"
        try:
            rs = int(line[22:26])
        except ValueError:
            continue

        if ch != chain or rs != resseq or a_name != atom_name:
            continue
        if resname_hint_u and r_name != resname_hint_u:
            continue

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        return (x, y, z)

    raise RuntimeError(
        f"Could not find atom XYZ in {pdb_path}: chain={chain} res={resseq} atom={atom_name}"
        + (f" resname={resname_hint_u}" if resname_hint_u else "")
    )

def _write_covalent_atom_pair_constraint_to_line(
    *,
    input_pdb: Path,
    covalent_meta: dict[str, Any],
    lig_atom: str,
    lig_resname: str = "LIG",
    dist: float = 1.80,
    sd: float = 0.10,
) -> tuple[str, int, str, int]:
    # same logic as _write_covalent_atom_pair_constraint, but returns mapped atoms/pose indices
    prot_atom = (covalent_meta.get("prot_atom") or "").strip().upper()
    if not prot_atom:
        raise ValueError("covalent_meta missing 'prot_atom'")

    target_resseq = int(covalent_meta.get("residue"))
    prot_chain = (covalent_meta.get("chain") or covalent_meta.get("prot_chain") or "A").strip()[:1] or "A"

    pose_map = _pdb_pose_index_map(input_pdb)

    candidates = [(key, pose) for key, pose in pose_map.items() if key[0] == prot_chain and key[1] == target_resseq]
    if not candidates:
        raise RuntimeError(f"Could not map protein residue chain {prot_chain} resseq {target_resseq} to a pose index in {input_pdb}")
    candidates.sort(key=lambda x: x[1])
    prot_pose = candidates[0][1]

    lig_pose = _find_first_lig_pose_index(pose_map, lig_resname=lig_resname)
    if lig_pose is None:
        raise RuntimeError(f"Could not find ligand residue '{lig_resname}' in {input_pdb}. Is your ligand resname actually {lig_resname}?")

    lig_atom_u = lig_atom.strip().upper()
    if not lig_atom_u:
        raise ValueError("lig_atom is empty")

    return prot_atom, prot_pose, lig_atom_u, lig_pose

def _pdb_find_ligand_residue_by_atom(
    pdb_path: Path,
    *,
    lig_resname: str,
    lig_atom: str,
) -> tuple[str, int]:
    lig_resname = (lig_resname or "LIG").strip().upper()[:3]
    lig_atom = (lig_atom or "").strip().upper()
    if not lig_atom:
        raise ValueError("lig_atom is empty")

    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("HETATM", "ATOM")) or len(line) < 26:
            continue
        resname = line[17:20].strip().upper()
        if resname != lig_resname:
            continue
        atom = line[12:16].strip().upper()
        if atom != lig_atom:
            continue
        chain = (line[21] or " ").strip() or "A"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        return chain, resseq

    # fallback: first residue with that lig_resname
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("HETATM") or len(line) < 26:
            continue
        resname = line[17:20].strip().upper()
        if resname != lig_resname:
            continue
        chain = (line[21] or " ").strip() or "A"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        return chain, resseq

    raise RuntimeError(f"Could not locate ligand {lig_resname} (atom {lig_atom}) in {pdb_path}")

def _rewrite_ligand_atomrefs_in_items(
    items: list[dict[str, Any]],
    *,
    lig_chain: str,
    lig_resseq: int,
    lig_resname: str,
    placeholder_chains: tuple[str, ...] = ("X",),
) -> list[dict[str, Any]]:
    lig_chain = (lig_chain or "A").strip()[:1] or "A"
    lig_resname = (lig_resname or "LIG").strip().upper()[:3]

    def _fix_atomref(ar: Any) -> Any:
        if not isinstance(ar, dict):
            return ar
        ch = str(ar.get("chain", "")).strip()[:1].upper()
        rn = str(ar.get("resname", "")).strip().upper()[:3] if ar.get("resname") is not None else None

        # Treat chain X as "ligand placeholder" OR explicit resname==LIG as ligand
        if (ch in placeholder_chains) or (rn == lig_resname):
            ar = dict(ar)
            ar["chain"] = lig_chain
            ar["res"] = int(lig_resseq)
            # optionally keep resname hint to help mapping
            ar["resname"] = lig_resname
        return ar

    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        it2 = dict(it)
        for key in ("a", "b", "c", "d", "atom", "ref"):
            if key in it2:
                it2[key] = _fix_atomref(it2[key])
        out.append(it2)
    return out


def _write_constraints_from_yaml(
    *,
    out_dir: Path,
    input_pdb: Path,
    dicts,
    mode: str,          # "covalent"|"noncovalent"
    stage_name: str,    # "stage2" for your use-case
    covalent_meta: dict[str, Any] | None = None,
    lig_resname: str = "LIG",
) -> Path | None:
    """
    Build constraints.cst from YAML + (optionally) auto covalent-bond AtomPair constraint.

    Auto behavior:
      - If mode=="covalent" and covalent_meta is provided (or discoverable),
        a covalent AtomPair is prepended unless YAML already defines it.
    """
    stage = _get_stage_cfg(dicts)
    constraints_cfg = _get_constraints_cfg(stage)
    if not bool(constraints_cfg.get("enabled", True)):
        return None

    # stage-specific enable handled inside _iter_active_constraint_items()
    items = _iter_active_constraint_items(constraints_cfg=constraints_cfg, mode=mode, stage_name=stage_name)

    raw_cfg = constraints_cfg.get("raw", {})
    raw_enabled = bool(raw_cfg.get("enabled", False)) if isinstance(raw_cfg, dict) else False
    raw_lines = []
    if raw_enabled and isinstance(raw_cfg, dict):
        rl = raw_cfg.get("lines", [])
        if isinstance(rl, list):
            raw_lines = [str(x).rstrip("\n") for x in rl]

    # nothing to write? (we may still auto-add covalent below)
    pose_map = _pdb_pose_index_map(input_pdb)

    def _ar(d: dict[str, Any]) -> tuple[str, int, str, str | None]:
        atom = str(d.get("atom", "")).strip().upper()
        res = int(d.get("res"))
        chain = str(d.get("chain", "A")).strip()[:1] or "A"
        resname_hint = d.get("resname")
        if resname_hint is not None:
            resname_hint = str(resname_hint).strip().upper()[:3]
        return atom, res, chain, resname_hint

    def _default_for(kind: str, key: str, fallback: Any) -> Any:
        dflt = constraints_cfg.get("defaults", {})
        if not isinstance(dflt, dict):
            return fallback
        kk = dflt.get(kind, {})
        if not isinstance(kk, dict):
            return fallback
        return kk.get(key, fallback)

    out_lines: list[str] = []

    # -------------------------------
    # ✅ AUTO: add covalent AtomPair + rewrite YAML ligand refs
    # -------------------------------
    auto_cov = bool(constraints_cfg.get("auto_from_covalent", True))
    if mode == "covalent" and auto_cov:
        cov = covalent_meta or {}
        if isinstance(cov, dict) and cov:
            # ligand atom can be stored under several names in your pipeline
            lig_atom = (
                (cov.get("ligand_atom") or cov.get("lig_atom") or cov.get("atom_lig") or "")
            ).strip().upper()

            # Find actual ligand res in THIS pdb (chain + resseq)
            if lig_atom:
                lig_chain_real, lig_resseq_real = _pdb_find_ligand_residue_by_atom(
                    input_pdb, lig_resname=lig_resname, lig_atom=lig_atom
                )
            else:
                # if we can't find by atom, at least try first ligand residue
                lig_chain_real, lig_resname_detected = _detect_ligand_chain_and_resname(
                    input_pdb, prefer_resnames=(lig_resname,)
                )
                # pick the first residue number for that resname+chain
                # (simple scan)
                lig_resseq_real = None
                for line in input_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.startswith("HETATM") or len(line) < 26:
                        continue
                    if line[17:20].strip().upper() != lig_resname:
                        continue
                    ch = (line[21] or " ").strip() or "A"
                    if ch != lig_chain_real:
                        continue
                    try:
                        lig_resseq_real = int(line[22:26])
                        break
                    except ValueError:
                        pass
                if lig_resseq_real is None:
                    raise RuntimeError(f"Could not determine ligand residue for resname {lig_resname} in {input_pdb}")

            # ✅ rewrite YAML items that used placeholder chain X / resname LIG
            items = _rewrite_ligand_atomrefs_in_items(
                items,
                lig_chain=lig_chain_real,
                lig_resseq=int(lig_resseq_real),
                lig_resname=lig_resname,
                placeholder_chains=("X",),
            )

            # Distance/sd defaults from YAML (ensure your YAML defaults are merged; see note above)
            cov_dist = float(_default_for("covalent_atom_pair", "dist", 1.80))
            cov_sd   = float(_default_for("covalent_atom_pair", "sd", 0.10))

            # Avoid duplicating if YAML already defines the same covalent AtomPair
            def _has_matching_covalent_atom_pair(items_list: list[dict[str, Any]]) -> bool:
                prot_atom = (cov.get("prot_atom") or "").strip().upper()
                prot_chain = (cov.get("chain") or cov.get("prot_chain") or "A").strip()[:1] or "A"
                prot_res = int(cov.get("residue")) if cov.get("residue") is not None else None
                if not prot_atom or not lig_atom or prot_res is None:
                    return False

                for it in items_list:
                    if str(it.get("type", "")).strip() != "AtomPair":
                        continue
                    a = it.get("a"); b = it.get("b")
                    if not isinstance(a, dict) or not isinstance(b, dict):
                        continue
                    a_atom, a_res, a_chain, _ = _ar(a)
                    b_atom, b_res, b_chain, _ = _ar(b)

                    if (
                        a_atom == prot_atom and a_chain == prot_chain and a_res == prot_res and
                        b_atom == lig_atom
                    ) or (
                        b_atom == prot_atom and b_chain == prot_chain and b_res == prot_res and
                        a_atom == lig_atom
                    ):
                        return True
                return False

            if lig_atom and not _has_matching_covalent_atom_pair(items):
                prot_atom, prot_pose, lig_atom_u, lig_pose = _write_covalent_atom_pair_constraint_to_line(
                    input_pdb=input_pdb,
                    covalent_meta=cov,
                    lig_atom=lig_atom,
                    lig_resname=lig_resname,
                    dist=cov_dist,
                    sd=cov_sd,
                )
                out_lines.append(
                    f"AtomPair {prot_atom} {prot_pose} {lig_atom_u} {lig_pose} HARMONIC {cov_dist:.2f} {cov_sd:.2f}"
                )

    # If still nothing at all and no raw lines, bail
    if not items and not raw_lines and not out_lines:
        return None

    # -------------------------------
    # Existing structured items logic
    # -------------------------------
    for it in items:
        if "raw" in it and str(it.get("raw", "")).strip():
            out_lines.append(str(it["raw"]).rstrip("\n"))
            continue

        typ = str(it.get("type", "")).strip()
        func = str(it.get("func", "")).strip().upper()
        x0 = it.get("x0", None)
        sd = it.get("sd", None)

        if typ == "AtomPair":
            a = it.get("a"); b = it.get("b")
            if not isinstance(a, dict) or not isinstance(b, dict):
                raise ValueError("AtomPair item requires dict fields a and b")
            a_atom, a_res, a_chain, a_rh = _ar(a)
            b_atom, b_res, b_chain, b_rh = _ar(b)

            a_pose = _pose_index_for_atomref(pose_map, chain=a_chain, resseq=a_res, resname_hint=a_rh)
            b_pose = _pose_index_for_atomref(pose_map, chain=b_chain, resseq=b_res, resname_hint=b_rh)

            if not func:
                func = str(_default_for("atom_pair", "func", "HARMONIC")).strip().upper()
            if sd is None:
                sd = float(_default_for("atom_pair", "sd", 0.10))
            if x0 is None:
                raise ValueError("AtomPair requires x0")

            out_lines.append(f"AtomPair {a_atom} {a_pose} {b_atom} {b_pose} {func} {float(x0)} {float(sd)}")
            continue

        if typ == "Angle":
            a = it.get("a"); b = it.get("b"); c = it.get("c")
            if not isinstance(a, dict) or not isinstance(b, dict) or not isinstance(c, dict):
                raise ValueError("Angle item requires dict fields a,b,c")
            a_atom, a_res, a_chain, a_rh = _ar(a)
            b_atom, b_res, b_chain, b_rh = _ar(b)
            c_atom, c_res, c_chain, c_rh = _ar(c)

            a_pose = _pose_index_for_atomref(pose_map, chain=a_chain, resseq=a_res, resname_hint=a_rh)
            b_pose = _pose_index_for_atomref(pose_map, chain=b_chain, resseq=b_res, resname_hint=b_rh)
            c_pose = _pose_index_for_atomref(pose_map, chain=c_chain, resseq=c_res, resname_hint=c_rh)

            if not func:
                func = str(_default_for("angle", "func", "HARMONIC")).strip().upper()
            if sd is None:
                sd = float(_default_for("angle", "sd", 10.0))
            if x0 is None:
                raise ValueError("Angle requires x0")

            out_lines.append(f"Angle {a_atom} {a_pose} {b_atom} {b_pose} {c_atom} {c_pose} {func} {float(x0)} {float(sd)}")
            continue

        if typ == "Dihedral":
            a = it.get("a"); b = it.get("b"); c = it.get("c"); d = it.get("d")
            if not isinstance(a, dict) or not isinstance(b, dict) or not isinstance(c, dict) or not isinstance(d, dict):
                raise ValueError("Dihedral item requires dict fields a,b,c,d")
            a_atom, a_res, a_chain, a_rh = _ar(a)
            b_atom, b_res, b_chain, b_rh = _ar(b)
            c_atom, c_res, c_chain, c_rh = _ar(c)
            d_atom, d_res, d_chain, d_rh = _ar(d)

            a_pose = _pose_index_for_atomref(pose_map, chain=a_chain, resseq=a_res, resname_hint=a_rh)
            b_pose = _pose_index_for_atomref(pose_map, chain=b_chain, resseq=b_res, resname_hint=b_rh)
            c_pose = _pose_index_for_atomref(pose_map, chain=c_chain, resseq=c_res, resname_hint=c_rh)
            d_pose = _pose_index_for_atomref(pose_map, chain=d_chain, resseq=d_res, resname_hint=d_rh)

            if not func:
                func = str(_default_for("dihedral", "func", "CIRCULARHARMONIC")).strip().upper()
            if sd is None:
                sd = float(_default_for("dihedral", "sd", 10.0))
            if x0 is None:
                raise ValueError("Dihedral requires x0")

            out_lines.append(
                f"Dihedral {a_atom} {a_pose} {b_atom} {b_pose} {c_atom} {c_pose} {d_atom} {d_pose} {func} {float(x0)} {float(sd)}"
            )
            continue

        if typ == "Coordinate":
            atomref = it.get("atom")
            if not isinstance(atomref, dict):
                raise ValueError("Coordinate item requires dict field 'atom'")

            a_atom, a_res, a_chain, a_rh = _ar(atomref)
            a_pose = _pose_index_for_atomref(pose_map, chain=a_chain, resseq=a_res, resname_hint=a_rh)

            ref_xyz = it.get("ref_xyz", None)
            if isinstance(ref_xyz, list) and len(ref_xyz) == 3:
                x, y, z = float(ref_xyz[0]), float(ref_xyz[1]), float(ref_xyz[2])
            else:
                x, y, z = _pdb_find_atom_xyz(
                    input_pdb,
                    chain=a_chain,
                    resseq=a_res,
                    atom_name=a_atom,
                    resname_hint=a_rh,
                )

            if not func:
                func = str(_default_for("coordinate", "func", "HARMONIC")).strip().upper()
            if sd is None:
                sd = float(_default_for("coordinate", "sd", 1.0))
            if x0 is None:
                x0 = float(it.get("x0", 0.0) or 0.0)

            out_lines.append(
                f"CoordinateConstraint {a_atom} {a_pose} {x:.3f} {y:.3f} {z:.3f} {func} {float(x0)} {float(sd)}"
            )
            continue

        raise ValueError(
            f"Unsupported structured constraint type '{typ}'. "
            f"Use raw.lines or per-item raw for this type."
        )

    for ln in raw_lines:
        out_lines.append(ln)

    filename = str(constraints_cfg.get("filename", "constraints.cst")).strip() or "constraints.cst"
    out_cst = out_dir / filename
    out_cst.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    return out_cst


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
    prot_atom = (covalent_meta.get("prot_atom") or "").strip().upper()
    if not prot_atom:
        raise ValueError("covalent_meta missing 'prot_atom'")

    target_resseq = int(covalent_meta.get("residue"))
    prot_chain = (covalent_meta.get("chain") or covalent_meta.get("prot_chain") or "A").strip()[:1] or "A"

    pose_map = _pdb_pose_index_map(input_pdb)

    candidates = [(key, pose) for key, pose in pose_map.items() if key[0] == prot_chain and key[1] == target_resseq]
    if not candidates:
        raise RuntimeError(f"Could not map protein residue chain {prot_chain} resseq {target_resseq} to a pose index in {input_pdb}")
    candidates.sort(key=lambda x: x[1])
    prot_pose = candidates[0][1]

    lig_pose = _find_first_lig_pose_index(pose_map, lig_resname=lig_resname)
    if lig_pose is None:
        raise RuntimeError(f"Could not find ligand residue '{lig_resname}' in {input_pdb}. Is your ligand resname actually {lig_resname}?")

    lig_atom_u = lig_atom.strip().upper()
    if not lig_atom_u:
        raise ValueError("lig_atom is empty")

    out_cst.parent.mkdir(parents=True, exist_ok=True)
    out_cst.write_text(
        f"AtomPair {prot_atom} {prot_pose} {lig_atom_u} {lig_pose} HARMONIC {dist:.2f} {sd:.2f}\n",
        encoding="utf-8",
    )

    return prot_atom, prot_pose, lig_atom_u, lig_pose


# =========================================================
# Extra-res params collection (LIG.params + covalent patch params + alias params)
# =========================================================
def _rosetta_db_host_path(rel_under_db: str) -> Path:
    rel = (rel_under_db or "").strip().lstrip("/").replace("\\", "/")
    db_posix = str(PurePosixPath(str(ROSETTA_DB)) / rel)
    return wsl_unc_from_linux(db_posix) if platform.system() == "Windows" else Path(db_posix)

def _extra_res_fa_string_from_list(paths: list[Path]) -> str | None:
    toks = [str(p) for p in paths if p and Path(p).exists()]
    return " ".join(toks) if toks else None

def _dedup_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        out.append(p)
    return out

def _collect_covalent_patch_params_from_dicts(
    *,
    dicts,
    covalent_meta: dict[str, Any] | None,
) -> list[Path]:
    """
    Mirrors rosetta_relax.py behavior:
    If covalent and covalent_patches[prot_atom].extra_res_fa exists, include it.
    Supports:
      - extra_res_fa: {db_rel: "..."}  (under ROSETTA_DB)
      - extra_res_fa: "/abs/linux/path/to/file.params"
    """
    if not covalent_meta:
        return []

    prot_atom = (covalent_meta.get("prot_atom") or "").strip().upper()
    if not prot_atom:
        return []

    patch = dicts.get("covalent_patches", prot_atom, default=None)
    if not isinstance(patch, dict):
        return []

    ex = patch.get("extra_res_fa")
    if isinstance(ex, dict) and ex.get("db_rel"):
        return [_rosetta_db_host_path(str(ex["db_rel"]))]

    if isinstance(ex, str) and ex.strip():
        s = ex.strip()
        return [wsl_unc_from_linux(s) if platform.system() == "Windows" else Path(s)]

    return []

def _parse_extra_res_fa_tokens(extra_res_fa: str | None) -> list[Path]:
    if not extra_res_fa:
        return []
    toks = [t.strip().strip("'").strip('"') for t in str(extra_res_fa).split() if t.strip()]
    return [Path(t) for t in toks]

def _resolve_extra_res_list_from_relax_record(relax_record: dict[str, Any]) -> list[Path]:
    """
    Prefer canonical extra_res_fa_list if present; else fall back to extra_res_fa string.
    """
    lst = relax_record.get("extra_res_fa_list")
    if isinstance(lst, list) and lst:
        return [Path(str(x)) for x in lst if x]
    return _parse_extra_res_fa_tokens(relax_record.get("extra_res_fa"))

def _maybe_generate_ligand_params_into_outdir(
    *,
    out_dir: Path,
    dicts,
    meta: dict[str, Any],
    yaml_path: str | Path | None,
) -> Path | None:
    """
    Generate ligand params into out_dir if metadata has SMILES.
    Returns params path if created and exists.
    """
    print(f"Using meta:{meta}", flush=True)

    # --- Robust SMILES lookup ---
    smiles = (meta.get("smiles") or meta.get("ligand_smiles") or "").strip()

    # Common nested shape: meta["ligand"]["smiles"]
    if not smiles:
        lig = meta.get("ligand")
        if isinstance(lig, dict):
            smiles = (lig.get("smiles") or lig.get("SMILES") or "").strip()

    # (Optional) other fallbacks you might have elsewhere
    if not smiles:
        paths = meta.get("paths")
        if isinstance(paths, dict):
            # nothing here for SMILES in your example, but keeping pattern
            pass

    if not smiles:
        print("ℹ️ No SMILES found in meta (checked meta.smiles / meta.ligand.smiles / meta.ligand_smiles).", flush=True)
        return None

    print(f"🧬 Found SMILES (len={len(smiles)}): {smiles}", flush=True)

    lig_resname3 = str(dicts.get("ligand", "resname", default="LIG")).strip().upper()[:3] or "LIG"

    prep = prep_ligand_for_rosetta(
        out_dir=out_dir,
        smiles=smiles,
        resname=lig_resname3,
        yaml_path=yaml_path,
    )

    if getattr(prep, "params_path", None):
        p = Path(prep.params_path)
        if p.exists():
            print(f"✅ Ligand params written: {p}", flush=True)
            return p
        else:
            print(f"⚠️ prep_ligand_for_rosetta returned params_path but file missing: {p}", flush=True)

    return None


# =========================================================
# Covalent proxy patch (pre-scripts when relax is skipped)
# =========================================================
def _make_3letter_alias_params(*, src_params: Path, alias3: str, out_params: Path) -> Path:
    alias3 = (alias3 or "").strip().upper()[:3]
    if len(alias3) != 3:
        raise ValueError(f"alias3 must be 3 letters, got: {alias3!r}")
    if not src_params.exists():
        raise FileNotFoundError(f"Source params not found: {src_params}")

    txt = src_params.read_text(encoding="utf-8", errors="replace").splitlines(True)

    out_lines: list[str] = []
    saw_name = False
    saw_io = False

    for line in txt:
        if line.startswith("NAME "):
            out_lines.append(f"NAME {alias3}\n")
            saw_name = True
            continue
        if line.startswith("IO_STRING "):
            parts = line.split()
            if len(parts) >= 2:
                parts[1] = alias3
                out_lines.append(" ".join(parts) + ("\n" if not line.endswith("\n") else ""))
            else:
                out_lines.append(f"IO_STRING {alias3}\n")
            saw_io = True
            continue
        out_lines.append(line)

    if not saw_name:
        out_lines.insert(0, f"NAME {alias3}\n")
    if not saw_io:
        insert_at = 1 if out_lines and out_lines[0].startswith("NAME ") else 0
        out_lines.insert(insert_at, f"IO_STRING {alias3}\n")

    out_params.parent.mkdir(parents=True, exist_ok=True)
    out_params.write_text("".join(out_lines), encoding="utf-8")
    return out_params

def _apply_resname_alias_in_pdb(
    *,
    pdb_in: Path,
    pdb_out: Path,
    target_chain: str,
    target_resseq: int,
    orig3: str,
    alias3: str,
) -> Path:
    """
    Replace residue name field (cols 18-20) for matching (chain, resseq, resname==orig3)
    on ATOM/HETATM lines.
    """
    target_chain = (target_chain or "A").strip()[:1] or "A"
    orig3 = (orig3 or "").strip().upper()[:3]
    alias3 = (alias3 or "").strip().upper()[:3]

    lines = pdb_in.read_text(encoding="utf-8", errors="replace").splitlines(True)
    out_lines: list[str] = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
            chain = (line[21] or " ").strip() or "A"
            try:
                resseq = int(line[22:26])
            except ValueError:
                resseq = None
            resname = line[17:20].strip().upper()

            if chain == target_chain and resseq == target_resseq and resname == orig3:
                line = f"{line[:17]}{alias3:>3s}{line[20:]}"
        out_lines.append(line)

    pdb_out.write_text("".join(out_lines), encoding="utf-8")
    return pdb_out

def _maybe_apply_covalent_proxy_patch(
    *,
    input_pdb: Path,
    out_dir: Path,
    covalent_meta: dict[str, Any],
    dicts,
) -> tuple[Path, list[Path]]:
    """
    If relax was skipped, apply any residue-name alias (e.g., LYS->LYD) required for Rosetta typing.
    Returns (pdb_to_use, extra_params_paths_to_append).
    """
    prot_atom = (covalent_meta.get("prot_atom") or "").strip().upper()
    if not prot_atom:
        return input_pdb, []

    rules = dicts.get("resname_alias_rules", default={}) or {}

    target_resseq = int(covalent_meta.get("residue"))
    target_chain = (covalent_meta.get("chain") or covalent_meta.get("prot_chain") or "A").strip()[:1] or "A"

    # Find original residue name at that position
    orig_res3 = None
    for line in input_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
            continue
        chain = (line[21] or " ").strip() or "A"
        if chain != target_chain:
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        if resseq != target_resseq:
            continue
        orig_res3 = line[17:20].strip().upper()
        break

    if not orig_res3:
        return input_pdb, []

    rule = rules.get(orig_res3)
    if not isinstance(rule, dict) or not rule:
        return input_pdb, []

    alias3 = (rule.get("alias3") or "").strip().upper()[:3]
    src_rel = (rule.get("src_rel") or "").strip()
    if not alias3 or not src_rel:
        return input_pdb, []

    # Create alias params under out_dir/params_alias
    alias_dir = out_dir / "params_alias"
    alias_dir.mkdir(parents=True, exist_ok=True)

    src_params = _rosetta_db_host_path(src_rel)
    out_params = alias_dir / f"{alias3}.params"
    _make_3letter_alias_params(src_params=src_params, alias3=alias3, out_params=out_params)

    # Patch PDB residue name
    patched_pdb = out_dir / "model_patched_for_scripts.pdb"
    _apply_resname_alias_in_pdb(
        pdb_in=input_pdb,
        pdb_out=patched_pdb,
        target_chain=target_chain,
        target_resseq=target_resseq,
        orig3=orig_res3,
        alias3=alias3,
    )

    print(f"🧩 Covalent proxy patch: {orig_res3}→{alias3} at {target_chain}{target_resseq} (added {out_params.name})")
    return patched_pdb, [out_params]

def _ensure_job_metadata(latest_dir: Path, base_job: str) -> Path:
    """
    Ensure job_metadata.json exists in the AF3 output folder.

    Preserves your original behavior:
      - If missing, try to copy legacy sibling: <parent>/<base_job>/<base_job>_job_metadata.json
      - Else write minimal stub.
    """
    latest_dir = Path(latest_dir)
    meta = latest_dir / "job_metadata.json"
    if meta.exists():
        print("Meta already exists", flush=True)
        return meta

    legacy_meta = latest_dir.parent / base_job / f"{base_job}_job_metadata.json"
    print(f"Searching for metadata: {legacy_meta}", flush=True)

    if legacy_meta.exists():
        try:
            shutil.copy2(legacy_meta, meta)
            print(f"🧾 Copied metadata → {meta} (from {legacy_meta})", flush=True)
            return meta
        except Exception as e:
            print(f"⚠️ Failed to copy legacy metadata: {e}", flush=True)

    payload = {"job_name": base_job, "created_by": "alphafold_gui"}
    safe_write_text(meta, json.dumps(payload, indent=2))
    print(f"🧩 Wrote missing job_metadata.json → {meta}", flush=True)
    return meta


# =========================================================
# AF3-direct preparation (when skip_rosetta=True)
# =========================================================
_TS_RE = re.compile(r"_[0-9]{8}[-_][0-9]{6}$")
def _trim_timestamp(name: str) -> str:
    return re.sub(_TS_RE, "", name)

def _resolve_model_cif(job_dir: Path, job_name: str, base_job: str, model_path: str | Path | None) -> Path:
    if model_path is not None:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Provided model_path does not exist: {p}")

    candidates = [
        job_dir / f"{base_job}_model.cif",
        job_dir / f"{job_name}_model.cif",
    ]
    for p in candidates:
        if p.exists():
            return p

    numbered = sorted(job_dir.glob("*_model.cif"))
    if numbered:
        return numbered[0]

    any_cif = sorted(job_dir.glob("*.cif"))
    if any_cif:
        return any_cif[0]

    raise FileNotFoundError(f"Could not find model CIF in {job_dir}")

def _cif_to_pdb_with_gemmi(*, cif_path: Path, out_pdb: Path, log_dir: Path) -> Path:
    cif_linux = linuxize_path(cif_path)
    pdb_linux = linuxize_path(out_pdb)
    gemmi_py = (
        "import gemmi; "
        f"c='{cif_linux}'; p='{pdb_linux}'; "
        "st=gemmi.read_structure(c); "
        "st.remove_alternative_conformations(); "
        "st.write_pdb(p); "
        "print('✅ Wrote', p)"
    )
    run_wsl(f"python3 -c \"{gemmi_py}\"", log=log_dir / "gemmi_convert.log")
    if not out_pdb.exists():
        raise FileNotFoundError(f"Gemmi conversion failed to produce: {out_pdb}")
    return out_pdb

def _clean_pdb_keep_ligand(*, pdb_in: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"cd '{linuxize_path(out_dir)}' && "
        f"python3 '{CLEAN_PDB_PY}' '{linuxize_path(pdb_in)}' -ignorechain"
    )
    run_wsl(cmd, log=out_dir / "clean_pdb.log")
    cleaned = out_dir / (pdb_in.name + "_00.pdb")
    if not cleaned.exists():
        cand = sorted(out_dir.glob("*_00.pdb"))
        if cand:
            cleaned = cand[0]
    if not cleaned.exists():
        raise FileNotFoundError(f"clean_pdb_keep_ligand did not produce *_00.pdb in {out_dir}")
    return cleaned


# =========================================================
# Main runner
# =========================================================
def _read_meta_flags(relax_record: dict[str, Any]) -> tuple[bool, str | None, list[Path]]:
    is_covalent = bool(relax_record.get("covalent"))
    extra_list = _resolve_extra_res_list_from_relax_record(relax_record)
    extra_str = relax_record.get("extra_res_fa") if relax_record.get("extra_res_fa") else _extra_res_fa_string_from_list(extra_list)
    return is_covalent, extra_str, extra_list

def _write_run_record(
    *,
    job_dir: Path,
    out_dir: Path,
    input_pdb: Path,
    scorefile: Path,
    best_pdb: Path | None,
    is_covalent: bool,
    extra_res_fa: str | None,
    extra_res_fa_list: list[str],
    mode: str,
    from_relax: dict[str, Any] | None = None,
    constraints_file: str | None = None,
) -> dict[str, Any]:
    out = {
        "mode": mode,
        "out_dir": str(out_dir),
        "input_pdb": str(input_pdb),
        "scorefile": str(scorefile) if scorefile.exists() else None,
        "best_pdb": str(best_pdb) if (best_pdb and best_pdb.exists()) else None,
        "covalent": bool(is_covalent),
        "extra_res_fa_list": extra_res_fa_list or [],
        "extra_res_fa": extra_res_fa,
        "from_relax": from_relax,
        "constraints_file": constraints_file,
    }
    _write_outputs_pointer(job_dir, out_dir, out)
    return out

def _run_one(
    job_dir: Path,
    *,
    skip_rosetta: bool,
    multi_seed: bool,
    model_path: str | Path | None,
    yaml_path: str | Path | None,
    constraints_file: str | Path | None = None, 
    out_parent: Path | None = None,
    relax_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_rosetta_scripts_available()
    dicts = load_rosetta_dicts(yaml_path)
    job_name = job_dir.name
    base_job = _trim_timestamp(job_name)

    stage_cfg = _get_stage_cfg(dicts)
    protocol_cfg = _get_protocol_cfg(stage_cfg)
    scorefxn_cfg = _get_scorefxn_cfg(stage_cfg)
    constraints_cfg = _get_constraints_cfg(stage_cfg)

    neighbor_dist = float(protocol_cfg.get("neighbor_dist", 6.0))

    # nstruct rules
    nstruct_cfg = protocol_cfg.get("nstruct", {}) if isinstance(protocol_cfg.get("nstruct", {}), dict) else {}
    nstruct_default_noncov = int(nstruct_cfg.get("noncovalent", 5))
    nstruct_default_cov    = int(nstruct_cfg.get("covalent", 1))

    # local rb
    local_rb = protocol_cfg.get("local_rb", {}) if isinstance(protocol_cfg.get("local_rb", {}), dict) else {}
    local_rb_enabled = bool(local_rb.get("enabled", True))
    local_rb_translate = local_rb.get("translate", {}) if isinstance(local_rb.get("translate", {}), dict) else {}
    local_rb_rotate = local_rb.get("rotate", {}) if isinstance(local_rb.get("rotate", {}), dict) else {}
    local_rb_slide = bool(local_rb.get("slide_together", True))

    # highres knobs
    highres = protocol_cfg.get("highres", {}) if isinstance(protocol_cfg.get("highres", {}), dict) else {}
    highres_cycles = int(highres.get("cycles", 6))
    highres_repack_every_nth = int(highres.get("repack_every_nth", 3))

    # ligand_area knobs
    ligand_area = protocol_cfg.get("ligand_area", {}) if isinstance(protocol_cfg.get("ligand_area", {}), dict) else {}

    # scorefunction names
    sfxn_soft_weights = str(scorefxn_cfg.get("soft", "ligand_soft_rep"))
    sfxn_hires_weights = str(scorefxn_cfg.get("hires", "ligand"))

    hires_cst_cfg = scorefxn_cfg.get("hires_cst", {}) if isinstance(scorefxn_cfg.get("hires_cst", {}), dict) else {}
    sfxn_hires_cst_base = str(hires_cst_cfg.get("base", sfxn_hires_weights))
    # NOTE: actual reweights come from constraints.weights[mode] (below)

    # packing flags (exactly your desired YAML location)
    packing_flags = str(stage_cfg.get("packing_flags", " -ex1 -ex2 -ex2aro -use_input_sc -flip_HNQ"))

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    out_dir = (out_parent or job_dir) / f"rosetta_scripts_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # We'll store params as a list, and also keep a legacy string for CLI flags.
    extra_params_paths: list[Path] = []

    # Always define these before any use
    job_name = job_dir.name
    base_job = _trim_timestamp(job_name)

    job_meta_path = _ensure_job_metadata(job_dir, base_job)
    meta = json.loads(job_meta_path.read_text(encoding="utf-8"))

    # -------------------------
    # Choose input + metadata
    # -------------------------
    if not skip_rosetta:
        if relax_record is None:
            relax_record = _load_latest_relax(job_dir)

        is_covalent, extra_res_fa, extra_list_from_relax = _read_meta_flags(relax_record)
        covalent_meta = relax_record.get("covalent_meta") or {}

        # Prefer canonical relaxed_pdb
        p = relax_record.get("relaxed_pdb") or relax_record.get("relaxed_pdb_raw")
        if not p:
            raise FileNotFoundError("Relax record missing relaxed_pdb/relaxed_pdb_raw")
        input_pdb = Path(p)
        if not input_pdb.is_absolute():
            input_pdb = (job_dir / input_pdb).resolve()
        if not input_pdb.exists():
            raise FileNotFoundError(f"Relaxed PDB not found: {input_pdb}")

        # Start from relax-provided list
        extra_params_paths.extend(extra_list_from_relax)

        # If relax record didn't carry ligand params, regenerate from job metadata (robust mode).
        # This helps if you run scripts standalone but still point at a relaxed PDB.
        if not extra_params_paths or not any(str(pp).lower().endswith(".params") and Path(pp).exists() for pp in extra_params_paths):
            meta = _load_job_metadata(job_dir)
            regen = _maybe_generate_ligand_params_into_outdir(out_dir=out_dir, dicts=dicts, meta=meta, yaml_path=yaml_path)
            if regen:
                extra_params_paths.append(regen)
                print(f"🧬 Regenerated ligand params for scripts: {regen}")

        # Also (optionally) include covalent patch params from dicts, if not already present.
        extra_params_paths.extend(_collect_covalent_patch_params_from_dicts(dicts=dicts, covalent_meta=covalent_meta))

        mode = "from_relax"

    else:
        # AF3-direct: build a cleaned PDB and ligand params in out_dir
        job_name = job_dir.name
        base_job = _trim_timestamp(job_name)
        meta = _load_job_metadata(job_dir)

        lig_meta = meta.get("ligand") if isinstance(meta.get("ligand"), dict) else {}

        is_covalent = bool(meta.get("covalent") or lig_meta.get("covalent"))

        # prefer nested ligand fields, fall back to legacy top-level keys if present
        def _get(k: str, *alts: str):
            for key in (k, *alts):
                if key in lig_meta and lig_meta.get(key) not in (None, "", 0):
                    return lig_meta.get(key)
                if key in meta and meta.get(key) not in (None, "", 0):
                    return meta.get(key)
            return None

        covalent_meta = {
            "residue": int(str(_get("residue") or "").strip()) if str(_get("residue") or "").strip() else None,
            "prot_atom": str(_get("prot_atom") or "").strip().upper(),
            "chain": str(_get("chain", "prot_chain") or "A").strip()[:1] or "A",
            "ligand_atom": str(_get("ligand_atom", "lig_atom", "ligandAtom") or "").strip(),
            "lig_resname": str(dicts.get("ligand", "resname", default="LIG")).strip().upper()[:3],
        }
        covalent_meta = {k: v for k, v in covalent_meta.items() if v not in (None, "", 0)}

        covalent_meta = {k: v for k, v in covalent_meta.items() if v not in (None, "", 0)}

        # Convert CIF -> PDB -> clean
        cif = _resolve_model_cif(job_dir, job_name, base_job, model_path)
        raw_pdb = out_dir / "model.pdb"
        _cif_to_pdb_with_gemmi(cif_path=cif, out_pdb=raw_pdb, log_dir=out_dir)
        cleaned = _clean_pdb_keep_ligand(pdb_in=raw_pdb, out_dir=out_dir)
        input_pdb = cleaned

        # 1) Covalent patch params from dicts (e.g., LYX/LYD protonation states)
        extra_params_paths.extend(_collect_covalent_patch_params_from_dicts(dicts=dicts, covalent_meta=covalent_meta if is_covalent else None))

        # 2) Ligand params from SMILES (this is what writes LIG.params into out_dir)
        lig_params = _maybe_generate_ligand_params_into_outdir(out_dir=out_dir, dicts=dicts, meta=meta, yaml_path=yaml_path)
        if lig_params:
            extra_params_paths.append(lig_params)
        else:
            print("ℹ️ AF3-direct mode: no SMILES in metadata; skipping ligand params generation.")

        # 3) Covalent proxy patch BEFORE scripts if needed (LYS->LYD etc.)
        if is_covalent and ("residue" in covalent_meta) and ("prot_atom" in covalent_meta):
            patched_pdb, alias_params = _maybe_apply_covalent_proxy_patch(
                input_pdb=input_pdb,
                out_dir=out_dir,
                covalent_meta=covalent_meta,
                dicts=dicts,
            )
            input_pdb = patched_pdb
            extra_params_paths.extend(alias_params)

        mode = "af3_direct"

    # Finalize extra params list/string
    extra_params_paths = _dedup_paths([Path(p) for p in extra_params_paths if p])
    extra_res_fa_list_str = [str(p) for p in extra_params_paths if p and Path(p).exists()]
    extra_res_fa = " ".join(extra_res_fa_list_str) if extra_res_fa_list_str else None

    # Detect ligand chain/resname in the chosen input PDB
    lig_chain, lig_resname = _detect_ligand_chain_and_resname(
        input_pdb,
        prefer_resnames=(str(dicts.get("ligand", "resname", default="LIG")),),
    )
    print(f"🧪 Detected ligand: chain={lig_chain} resname={lig_resname}")

    # -------------------------
    # Constraints selection (CUSTOM > RELAX > AUTO)
    # -------------------------
    mode_key = "covalent" if is_covalent else "noncovalent"
    stage_name = "stage2"

    cst_path: Path | None = None

    # 1) ✅ CUSTOM constraints always win
    if constraints_file:
        staged = _resolve_and_stage_constraints_file(
            constraints_file,
            out_dir=out_dir,
            job_dir=job_dir,
        )

        lig_atom = (covalent_meta.get("ligand_atom") or covalent_meta.get("lig_atom") or "").strip() if isinstance(covalent_meta, dict) else ""

        if isinstance(covalent_meta, dict) and covalent_meta.get("residue") and (covalent_meta.get("chain") or covalent_meta.get("prot_chain")):
            prot_chain = str(covalent_meta.get("chain") or covalent_meta.get("prot_chain") or "A")
            prot_resseq = int(covalent_meta["residue"])
            prot_resname_hint = (
                str(covalent_meta.get("restype") or covalent_meta.get("prot_restype") or "").strip() or None
            )

            prot_pose_res = _compute_protein_pose_index(
                input_pdb,
                prot_chain=prot_chain,
                prot_resseq=prot_resseq,
                prot_resname_hint=prot_resname_hint,
            )

            cst_path = _rewrite_custom_cst_ligand_residue_numbers(
                staged,
                input_pdb=input_pdb,
                prot_pose_res=prot_pose_res,
                lig_resname=lig_resname,
                lig_atom=lig_atom or None,
                out_path=out_dir / f"custom_{staged.stem}_posefix{staged.suffix}",
            )
        else:
            cst_path = staged

        print(f"📌 Using CUSTOM constraints: {cst_path}", flush=True)

    else:
        relax_cst = None
        if (not skip_rosetta) and isinstance(relax_record, dict):
            for k in ("constraints_file", "cst_file", "cst_path", "constraints"):
                v = relax_record.get(k)
                if not (isinstance(v, str) and v.strip()):
                    continue

                cand_raw = v.strip()

                # Relative path -> relative to job_dir (host path)
                if not Path(cand_raw).is_absolute() and not _looks_like_linux_abs_path(cand_raw):
                    cand_raw = str((job_dir / cand_raw).resolve())

                cand_host = _host_path_for_maybe_linux_path(cand_raw)
                if cand_host.exists():
                    relax_cst = cand_host
                    break

        if relax_cst:
            # stage it into out_dir for reproducibility
            staged = out_dir / f"relax_{relax_cst.name}"
            shutil.copy2(relax_cst, staged)
            cst_path = staged
            print(f"📎 Using RELAX constraints: {cst_path}", flush=True)

        else:
            # 3) Auto-generate from YAML/AF3 (existing behavior)
            cst_path = _write_constraints_from_yaml(
                out_dir=out_dir,
                input_pdb=input_pdb,
                dicts=dicts,
                mode=mode_key,
                stage_name=stage_name,
                covalent_meta=covalent_meta if is_covalent else None,
                lig_resname=lig_resname,
            )
            if cst_path:
                print(f"🧷 Using AUTO constraints: {cst_path}", flush=True)

    cst_file_wsl = linuxize_path(cst_path) if cst_path else None

    # weights for constraints scorefxn (YAML-driven)
    sfxn_hires_cst_reweights = _weights_for_mode(_get_constraints_cfg(_get_stage_cfg(dicts)), mode_key)

    # -------------------------
    # Write XML + run rosetta_scripts
    # -------------------------
    xml_path = out_dir / "rosetta_scripts.xml"
    xml_path.write_text(
        _rosetta_ligand_xml(
            neighbor_dist=neighbor_dist,
            lig_chain=lig_chain,
            is_covalent=is_covalent,
            # protocol
            local_rb_enabled=local_rb_enabled,
            local_rb_translate=local_rb_translate,
            local_rb_rotate=local_rb_rotate,
            local_rb_slide_together=local_rb_slide,
            highres_cycles=highres_cycles,
            highres_repack_every_nth=highres_repack_every_nth,
            ligand_area=ligand_area,
            # scorefxn
            sfxn_soft_weights=sfxn_soft_weights,
            sfxn_hires_weights=sfxn_hires_weights,
            sfxn_hires_cst_base=sfxn_hires_cst_base,
            sfxn_hires_cst_reweights=sfxn_hires_cst_reweights,
            # constraints
            cst_file_wsl=cst_file_wsl,
        ),
        encoding="utf-8",
    )


    nstruct = nstruct_default_cov if is_covalent else nstruct_default_noncov

    out_dir_linux = linuxize_path(out_dir)
    xml_linux = linuxize_path(xml_path)
    pdb_linux = linuxize_path(input_pdb)
    scorefile = out_dir / "score.sc"

    extra_res_flags = ""
    if extra_res_fa:
        toks = [t.strip().strip("'").strip('"') for t in str(extra_res_fa).split() if t.strip()]
        linux_toks = [linuxize_path(Path(t)) for t in toks]
        extra_res_flags = " -in:file:extra_res_fa " + " ".join(f"'{t}'" for t in linux_toks) + " "

    cmd = (
        f"'{ROSETTA_SCRIPTS_BIN}' -database '{ROSETTA_DB}' "
        f"-s '{pdb_linux}' "
        f"-parser:protocol '{xml_linux}' "
        f"-out:path:all '{out_dir_linux}' "
        f"-out:file:scorefile '{out_dir_linux}/score.sc' "
        f"-nstruct {nstruct} -overwrite "
        f"{extra_res_flags}"
        f"{packing_flags}"
    )

    run_wsl(cmd, log=out_dir / "rosetta_scripts.log")

    # Pick a "best" PDB: newest PDB in out_dir
    best_pdb = None
    pdbs = sorted(out_dir.glob("*.pdb"), key=lambda p: p.stat().st_mtime)
    if pdbs:
        best_pdb = pdbs[-1]

    from_relax = None
    if mode == "from_relax" and relax_record is not None:
        from_relax = {
            "prep_dir": relax_record.get("prep_dir"),
            "relaxed_pdb": relax_record.get("relaxed_pdb"),
            "relaxed_pdb_raw": relax_record.get("relaxed_pdb_raw"),
            "scorefile": relax_record.get("scorefile"),
        }

    out_record = _write_run_record(
        job_dir=job_dir,
        out_dir=out_dir,
        input_pdb=input_pdb,
        scorefile=scorefile,
        best_pdb=best_pdb,
        is_covalent=is_covalent,
        extra_res_fa=extra_res_fa,
        extra_res_fa_list=extra_res_fa_list_str,
        mode=mode,
        from_relax=from_relax,
        constraints_file=str(cst_path) if cst_path else None,
    )

    _bundle_outputs(job_dir, out_dir, out_record)
    return out_record


# =========================================================
# Public entrypoint
# =========================================================
def run(
    job_dir: str | Path,
    *,
    skip_rosetta: bool = False,
    multi_seed: bool = False,
    model_path: str | Path | None = None,
    yaml_path: str | Path | None = None,
    constraints_file: str | Path | None = None,  
):
    """
    Run RosettaScripts refinement.

    If skip_rosetta=False:
        - reads latest_rosetta_relax.json (single or multi_seed index)
    If skip_rosetta=True:
        - runs AF3-direct preparation and then RosettaScripts

    multi_seed:
        - if using relax outputs and latest_rosetta_relax.json points to an index, iterates per entry
        - AF3-direct multi_seed is not implemented here
    """
    job_dir = _resolve_job_dir(Path(job_dir))

    # If we're not skipping relax, follow latest_rosetta_relax.json (possibly multi-seed)
    if not skip_rosetta:
        relax_root = _load_latest_relax(job_dir)

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
                prep_dir = Path(entry.get("prep_dir", ""))
                if not prep_dir.exists():
                    print(f"⚠️ Missing prep_dir, skipping: {prep_dir}")
                    continue

                print(f"🧲 RosettaScripts for {prep_dir.name} ...")
                out = _run_one(
                    prep_dir,
                    skip_rosetta=False,
                    multi_seed=True,
                    model_path=None,
                    yaml_path=yaml_path,
                    constraints_file=constraints_file,
                    out_parent=prep_dir,
                    relax_record=entry,
                )
                out_index.append(out)

            scripts_index = run_dir / "rosetta_scripts_multi_index.json"
            scripts_index.write_text(json.dumps(out_index, indent=2), encoding="utf-8")

            (job_dir / "latest_rosetta_scripts.json").write_text(
                json.dumps({"multi_seed": True, "run_dir": str(run_dir), "index": str(scripts_index)}, indent=2),
                encoding="utf-8",
            )

            print(f"✅ RosettaScripts multi-seed complete → {scripts_index}")
            return out_index

        # Single job from relax root record
        return _run_one(
            job_dir,
            skip_rosetta=False,
            multi_seed=False,
            model_path=None,
            yaml_path=yaml_path,
            constraints_file=constraints_file,
            out_parent=job_dir,
            relax_record=relax_root,
        )

    # AF3-direct single job
    return _run_one(
        job_dir,
        skip_rosetta=True,
        multi_seed=False,
        model_path=model_path,
        yaml_path=yaml_path,
        constraints_file=constraints_file,
        out_parent=job_dir,
        relax_record=None,
    )


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run RosettaScripts ligand-focused refinement.")
    ap.add_argument("--job", required=True, help="Job directory (bundled job root or af_output job dir)")
    ap.add_argument("--skip_rosetta", action="store_true", help="Run AF3-direct (do not use latest_rosetta_relax.json)")
    ap.add_argument("--multi_seed", action="store_true", help="Run for each seed/sample (requires relax index)")
    ap.add_argument("--model", default=None, help="Optional model CIF path (AF3-direct mode)")
    ap.add_argument("--yaml", default=None, help="Path to rosetta_dicts.yaml (optional)")
    ap.add_argument("--constraints", default=None, help="Optional constraints file (.cst). Overrides relax/auto.")
    args = ap.parse_args()

    run(
        Path(args.job),
        skip_rosetta=bool(args.skip_rosetta),
        multi_seed=bool(args.multi_seed),
        model_path=args.model,
        yaml_path=args.yaml,
        constraints_file=args.constraints,   
    )
