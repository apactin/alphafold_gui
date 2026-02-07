#!/usr/bin/env python3
"""
rosetta_minimize.py  — Windows launcher to run Rosetta relax on AF3 CIF outputs
===============================================================================

Runs the full flow:
  1. Convert CIF → PDB (Gemmi)
  2. (Optional) Generate ligand params from SMILES (RDKit → molfile_to_params.py)
  3. Clean PDB via clean_pdb_keep_ligand.py
  4. Rosetta relax (cartesian)
     - For covalent: temporarily removes reactive atom for relax, then restores after relax
  5. Writes outputs JSON for downstream steps (RosettaLigand, metrics, etc.)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
import re
import platform
from pathlib import Path, PurePosixPath
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Any
import shutil

import pandas as pd

from af3_pipeline.config import cfg

# =========================================================
# 🧩 Rosetta paths (inside WSL, config-driven)
# =========================================================
ROSETTA_BASE = (cfg.get("rosetta_relax_bin") or "").strip()
if not ROSETTA_BASE:
    raise RuntimeError(
        "Missing config key: rosetta_relax_bin (set to Rosetta bundle root, e.g. /home/<user>/rosetta/<bundle>)."
    )

ROSETTA_BASE = str(PurePosixPath(ROSETTA_BASE))

RELAX_BIN = f"{ROSETTA_BASE}/main/source/bin/relax.static.linuxgccrelease"
ROSETTA_DB = f"{ROSETTA_BASE}/main/database"
CLEAN_PDB_PY = f"{ROSETTA_BASE}/main/source/src/apps/public/relax_w_allatom_cst/clean_pdb_keep_ligand.py"
M2P_PY = f"{ROSETTA_BASE}/main/source/scripts/python/public/molfile_to_params.py"

DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")
LINUX_HOME = cfg.get("linux_home_root", "")
ALPHAFOLD_BASE = cfg.get("af3_dir", f"{LINUX_HOME}/Repositories/alphafold")

APP_TZ = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))


def now_local() -> datetime:
    return datetime.now(APP_TZ)


# =========================================================
# 🧰 Windows → WSL launcher setup
# =========================================================
WSL_EXE = r"C:\Windows\System32\wsl.exe"


def run_wsl(cmd: str, cwd_wsl: str | None = None, log: Path | None = None):
    """Run a command inside WSL via bash -lc. Capture stdout/stderr."""
    if cwd_wsl:
        cmd = f"pushd '{cwd_wsl}' >/dev/null 2>&1; {cmd}; rc=$?; popd >/dev/null 2>&1; exit $rc"
    # Keep your existing behavior (distro-less). If you prefer explicit distro:
    # full = [WSL_EXE, "-d", DISTRO_NAME, "--", "bash", "-lc", cmd]
    full = [WSL_EXE, "bash", "-lc", cmd]

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
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"❌ WSL command failed (exit {proc.returncode})")
    return proc


def to_wsl_path(subpath: str) -> Path:
    """Return correct path depending on OS (UNC on Windows, /home/... on WSL)."""
    sub = subpath.replace("\\", "/").strip("/")
    if platform.system() == "Windows":
        base_path = ALPHAFOLD_BASE.replace("/", "\\")
        base = f"\\\\wsl.localhost\\{DISTRO_NAME}{base_path}"
        return Path(base + (("\\" + sub.replace("/", "\\")) if sub else ""))
    else:
        return Path(ALPHAFOLD_BASE + (("/" + sub) if sub else ""))


AF_OUTPUT_DIR = to_wsl_path("af_output")


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


_TS_RE = re.compile(r"_[0-9]{8}[-_][0-9]{6}$")


def _trim_timestamp(name: str) -> str:
    """Remove trailing timestamp like _YYYYMMDD_HHMMSS or _YYYYMMDD-HHMMSS."""
    return re.sub(_TS_RE, "", name)

def _ensure_job_metadata(latest_dir: Path, job_name: str) -> Path:
    latest_dir = Path(latest_dir)
    meta = latest_dir / "job_metadata.json"
    if meta.exists():
        print("Meta already exists", flush=True)
        return meta

    # legacy = sibling folder under af_output/
    legacy_meta = latest_dir.parent / job_name / f"{job_name}_job_metadata.json"
    print(f"Searching for metadata: {legacy_meta}", flush=True)

    if legacy_meta.exists():
        try:
            shutil.copy2(legacy_meta, meta)
            print(f"🧾 Copied metadata → {meta} (from {legacy_meta})", flush=True)
            return meta
        except Exception as e:
            print(f"⚠️ Failed to copy legacy metadata: {e}", flush=True)

    payload = {"job_name": job_name, "created_by": "alphafold_gui"}
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"🧩 Wrote missing job_metadata.json → {meta}", flush=True)
    return meta


# =========================================================
# 🔗 Covalent “relax patch” registry (easy to extend later)
# =========================================================
# Notes:
# - anchor_atom should ideally survive the temp mutation (e.g., CB/CA always exist).
# - element should match the deleted atom element.
COVALENT_PATCHES: dict[str, dict[str, Any]] = {
    # Lysine reactive nitrogen — custom residue LYX (LYS minus NZ)
    "NZ": {
        "deleted_atom": "NZ",
        "anchor_atom": "CE",
        "temp_resname": "LYX",
        "temp_by_resname": {},  # not needed
        "extra_res_fa": f"{ROSETTA_DB}/chemical/residue_type_sets/fa_standard/residue_types/l-caa/LYX.params",
        "element": "N",
    },

    # Cysteine thiol — convenient simplification to alanine (delete SG, keep CB anchor)
    "SG": {
        "deleted_atom": "SG",
        "anchor_atom": "CB",
        "temp_resname": "ALA",   # explicitly mutate
        "temp_by_resname": {},
        "extra_res_fa": None,
        "element": "S",
    },

    # Tyrosine phenolic oxygen — convenient alternative phenylalanine (delete OH, keep CZ)
    "OH": {
        "deleted_atom": "OH",
        "anchor_atom": "CZ",
        "temp_resname": "PHE",   # explicitly mutate
        "temp_by_resname": {},
        "extra_res_fa": None,
        "element": "O",
    },

    # Histidine ND1 — same atom name could appear in HIS only (practically),
    # but your rule says: don't mutate complex side chains unless you explicitly code it.
    # So KEEP residue type.
    "ND1": {
        "deleted_atom": "ND1",
        "anchor_atom": "CG",     # stable within HIS; and since we KEEP, it's safe
        "temp_resname": "KEEP",
        "temp_by_resname": {
            "HIS": "KEEP",
            "HID": "KEEP",
            "HIE": "KEEP",
            "HIP": "KEEP",
        },
        "extra_res_fa": None,
        "element": "N",
    },

    # NE1 — can be TRP NE1; can also exist in HIS depending on naming in some contexts.
    # KEEP residue type; choose an anchor that exists in both TRP and HIS forms:
    # - TRP has CD1/CE2, etc; HIS has CD2/CE1.
    # CD2 exists in both TRP and HIS, so it's a decent shared anchor if present.
    "NE1": {
        "deleted_atom": "NE1",
        "anchor_atom": "CD2",
        "temp_resname": "KEEP",
        "temp_by_resname": {
            "TRP": "KEEP",
            "HIS": "KEEP",
            "HID": "KEEP",
            "HIE": "KEEP",
            "HIP": "KEEP",
        },
        "extra_res_fa": None,
        "element": "N",
    },

    # -----------------------------
    # New residues you requested
    # -----------------------------

    # Serine hydroxyl — easiest is KEEP (simple side chain; no need to mutate unless you want ALA)
    "OG": {
        "deleted_atom": "OG",
        "anchor_atom": "CB",     # survives in SER (and most reasonable)
        "temp_resname": "KEEP",
        "temp_by_resname": {"SER": "KEEP"},
        "extra_res_fa": None,
        "element": "O",
    },

    # Threonine hydroxyl — OG1
    "OG1": {
        "deleted_atom": "OG1",
        "anchor_atom": "CB",
        "temp_resname": "KEEP",
        "temp_by_resname": {"THR": "KEEP"},
        "extra_res_fa": None,
        "element": "O",
    },

    # Aspartate carboxylate — OD1/OD2
    "OD1": {
        "deleted_atom": "OD1",
        "anchor_atom": "CG",
        "temp_resname": "KEEP",
        "temp_by_resname": {"ASP": "KEEP"},
        "extra_res_fa": None,
        "element": "O",
    },
    "OD2": {
        "deleted_atom": "OD2",
        "anchor_atom": "CG",
        "temp_resname": "KEEP",
        "temp_by_resname": {"ASP": "KEEP"},
        "extra_res_fa": None,
        "element": "O",
    },

    # Glutamate carboxylate — OE1/OE2
    "OE1": {
        "deleted_atom": "OE1",
        "anchor_atom": "CD",
        "temp_resname": "KEEP",
        "temp_by_resname": {"GLU": "KEEP"},
        "extra_res_fa": None,
        "element": "O",
    },
    "OE2": {
        "deleted_atom": "OE2",
        "anchor_atom": "CD",
        "temp_resname": "KEEP",
        "temp_by_resname": {"GLU": "KEEP"},
        "extra_res_fa": None,
        "element": "O",
    },

    # Methionine thioether — SD
    "SD": {
        "deleted_atom": "SD",
        "anchor_atom": "CG",
        "temp_resname": "KEEP",
        "temp_by_resname": {"MET": "KEEP"},
        "extra_res_fa": None,
        "element": "S",
    },
}


# =========================================================
# 🧪 Ligand generation (RDKit)
# =========================================================
def generate_sdf_from_smiles(smiles: str, out_sdf: Path) -> Path:
    """Generate multi-conformer 3D SDF from SMILES with proper aromatic flags and MMFF minimization."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("❌ Invalid SMILES — cannot parse molecule.")
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=20, params=params)
    for cid in conf_ids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    for cid in conf_ids:
        w.write(mol, confId=cid)
    w.close()

    print(f"🧬 Generated aromatic RDKit SDF with {len(conf_ids)} conformers → {out_sdf}")
    return out_sdf


def generate_rosetta_params_from_sdf(sdf_path: Path, out_dir: Path, residue_name: str = "LIG") -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / f"{residue_name}.params"
    if params_path.exists():
        print(f"⏭️  Found existing params, skipping generation: {params_path}")
        return params_path

    sdf_linux = linuxize_path(sdf_path)
    out_linux = linuxize_path(out_dir)

    cmd = (
        f"cd '{out_linux}' && "
        f"python3 '{M2P_PY}' -n {residue_name} -p {residue_name} "
        f"--conformers-in-one-file '{sdf_linux}'"
    )
    run_wsl(cmd, log=out_dir / "molfile_to_params.log")

    if params_path.exists():
        print(f"✅ Generated params: {params_path}")
        return params_path

    print("⚠️ Rosetta params not found — check molfile_to_params.log")
    return None


# =========================================================
# 📄 Model resolution
# =========================================================
def _resolve_model_cif(job_dir: Path, job_name: str, base_job: str, model_path: str | Path | None) -> Path:
    """Resolve the AF3 model CIF for single-model runs."""
    if model_path is not None:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"❌ Provided model_path does not exist: {p}")

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

    raise FileNotFoundError(f"❌ Could not find model CIF in {job_dir}")


def _user_jobs_root() -> Path:
    root = cfg.get("jobs_root", None)
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".af3_pipeline" / "jobs").resolve()


# =========================================================
# 🧾 Outputs JSON (for downstream steps)
# =========================================================
def _write_relax_outputs(
    *,
    job_dir: Path,
    prep_dir: Path,
    params_path: Path | None,
    is_covalent: bool,
    covalent_meta: dict[str, Any] | None,
    relaxed_pdb_raw: Path | None,
    relaxed_pdb_final: Path | None,
):
    scorefile = prep_dir / "score.sc"
    extra_paths: list[str] = []

    # covalent patch params (already a linux path string)
    if is_covalent and covalent_meta:
        prot_atom = (covalent_meta.get("prot_atom") or "").strip().upper()
        patch = COVALENT_PATCHES.get(prot_atom)
        if patch and patch.get("extra_res_fa"):
            extra_paths.append(str(patch["extra_res_fa"]))

    # ligand params (UNC/host path string)
    if params_path and params_path.exists():
        extra_paths.append(str(params_path))

    out = {
        "prep_dir": str(prep_dir),
        "scorefile": str(scorefile) if scorefile.exists() else None,
        "relaxed_pdb_raw": str(relaxed_pdb_raw) if relaxed_pdb_raw else None,
        "relaxed_pdb": str(relaxed_pdb_final) if relaxed_pdb_final else (str(relaxed_pdb_raw) if relaxed_pdb_raw else None),
        "extra_res_fa": " ".join(extra_paths) if extra_paths else None,
        "covalent": bool(is_covalent),
        "covalent_meta": covalent_meta or None,
    }

    (prep_dir / "rosetta_relax_outputs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (job_dir / "latest_rosetta_relax.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# =========================================================
# 🔁 Restore deleted reactive atom after relax
# =========================================================
def _pdb_iter_atoms(lines: list[str]):
    """Yield (idx, line, resnum, chain, resname, atom_name, x,y,z) for ATOM/HETATM lines."""
    for i, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        chain = (line[21] or " ").strip() or "A"
        resname = line[17:20].strip()
        atom_name = line[12:16].strip()
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        yield i, line, resnum, chain, resname, atom_name, x, y, z


def _restore_deleted_atom_after_relax(
    *,
    original_cleaned_pdb: Path,
    relaxed_pdb: Path,
    target_resnum: int,
    deleted_atom: str,
    anchor_atoms: list[str],
    element: str,
    out_pdb: Path,
) -> Path:
    """
    Reinsert deleted reactive atom into relaxed PDB and restore original residue name.

    Uses original vector (anchor->deleted) from original_cleaned_pdb and applies it
    to the relaxed anchor position. Chooses the first anchor that exists in BOTH
    original and relaxed (fallback anchors supported).
    """
    orig_lines = original_cleaned_pdb.read_text(encoding="utf-8", errors="replace").splitlines(True)
    rel_lines = relaxed_pdb.read_text(encoding="utf-8", errors="replace").splitlines(True)

    deleted_atom = deleted_atom.strip().upper()
    anchor_atoms = [a.strip().upper() for a in anchor_atoms if a and a.strip()]

    # Find original residue name, deleted coords, and any anchor coords we might use
    orig_resname: str | None = None
    orig_chain: str | None = None
    orig_deleted: tuple[float, float, float] | None = None
    orig_anchor_by_name: dict[str, tuple[float, float, float]] = {}

    for _, _, resnum, chain, resname, atom, x, y, z in _pdb_iter_atoms(orig_lines):
        if resnum != target_resnum:
            continue
        if orig_resname is None:
            orig_resname = resname
            orig_chain = chain
        if atom == deleted_atom:
            orig_deleted = (x, y, z)
        if atom in anchor_atoms:
            orig_anchor_by_name[atom] = (x, y, z)

    if not orig_deleted:
        print(f"⚠️ Cannot restore {deleted_atom}: missing {deleted_atom} in original: {original_cleaned_pdb}")
        return relaxed_pdb

    # Choose an anchor that exists in both original and relaxed
    chosen_anchor: str | None = None
    rel_anchor: tuple[float, float, float] | None = None
    insert_after_idx: int | None = None

    for a in anchor_atoms:
        if a not in orig_anchor_by_name:
            continue
        for idx, _, resnum, chain, resname, atom, x, y, z in _pdb_iter_atoms(rel_lines):
            if resnum == target_resnum and atom == a:
                chosen_anchor = a
                rel_anchor = (x, y, z)
                insert_after_idx = idx
                break
        if chosen_anchor:
            break

    if not chosen_anchor or not rel_anchor or insert_after_idx is None:
        print(f"⚠️ Cannot restore {deleted_atom}: no usable anchor in relaxed for residue {target_resnum}. Tried {anchor_atoms}")
        return relaxed_pdb

    orig_anchor = orig_anchor_by_name[chosen_anchor]

    # Compute new position
    dx = orig_deleted[0] - orig_anchor[0]
    dy = orig_deleted[1] - orig_anchor[1]
    dz = orig_deleted[2] - orig_anchor[2]
    new_x = rel_anchor[0] + dx
    new_y = rel_anchor[1] + dy
    new_z = rel_anchor[2] + dz

    # Use original chain + residue name (no SG/NZ hardcoding)
    chain_out = (orig_chain or "A")[:1]
    resname_out = (orig_resname or "UNK")[:3]

    # Serial number: safe large value (won't break most tools; OK for reinsertion)
    serial = 99999
    element = (element or "").strip().upper()[:2] or "X"
    new_line = (
        f"ATOM  {serial:5d} {deleted_atom:<4s} {resname_out:>3s} {chain_out:1s}"
        f"{target_resnum:4d}    "
        f"{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )

    # Rewrite relaxed file:
    # - restore residue name for all ATOM/HETATM lines at target residue
    # - insert missing atom after chosen anchor line (only once)
    out_lines: list[str] = []
    inserted = False

    for i, line in enumerate(rel_lines):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
            try:
                resnum = int(line[22:26])
            except ValueError:
                resnum = None
            if resnum == target_resnum and orig_resname:
                line = f"{line[:17]}{orig_resname:>3s}{line[20:]}"
        out_lines.append(line)

        if (not inserted) and (i == insert_after_idx):
            out_lines.append(new_line)
            inserted = True

    if not inserted:
        out_lines.append(new_line)

    out_pdb.write_text("".join(out_lines), encoding="utf-8")
    print(f"✅ Restored {deleted_atom} (anchor={chosen_anchor}) into relaxed PDB → {out_pdb}")
    return out_pdb


# =========================================================
# 🔧 Covalent relax-input writer
# =========================================================
def _make_relax_input_pdb_for_covalent(
    *,
    cleaned_pdb_original: Path,
    out_pdb: Path,
    target_resnum: int,
    patch: dict[str, Any],
    target_chain: str | None = None,
) -> Path:
    """
    Create a modified copy used ONLY as relax input:
      - delete reactive atom (e.g., NZ/SG/ND1/NE1/OH/OG/OD1/...)
      - optionally rename residue to a temp residue type

    New behavior:
      - Automatically detects the *original* residue name at (chain, resnum)
      - Supports patch["temp_by_resname"] overrides
      - Supports temp_resname="KEEP" (or empty/None) to keep original residue type
      - Respects chain if provided; otherwise applies to all chains with matching resnum
        (recommended: pass chain from metadata for safety)
    """
    deleted_atom = (patch.get("deleted_atom") or "").strip().upper()
    base_temp = (patch.get("temp_resname") or "").strip().upper() or None
    temp_by_resname = patch.get("temp_by_resname") or {}
    if not isinstance(temp_by_resname, dict):
        temp_by_resname = {}

    if target_chain:
        target_chain = target_chain.strip()[:1] or None

    # --------
    # Detect original residue name at target residue
    # --------
    orig_resname: str | None = None
    with open(cleaned_pdb_original, "r", encoding="utf-8", errors="replace") as f_in:
        for line in f_in:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            chain = (line[21] or " ").strip() or "A"
            if resnum != target_resnum:
                continue
            if target_chain and chain != target_chain:
                continue
            orig_resname = line[17:20].strip().upper()
            break

    # Compute effective temp residue name
    temp_resname = None
    if orig_resname and orig_resname in temp_by_resname:
        choice = (temp_by_resname.get(orig_resname) or "").strip().upper()
        if choice and choice != "KEEP":
            temp_resname = choice
    else:
        if base_temp and base_temp != "KEEP":
            temp_resname = base_temp

    # --------
    # Write relax input
    # --------
    with open(cleaned_pdb_original, "r", encoding="utf-8", errors="replace") as f_in, \
         open(out_pdb, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    resnum = None
                chain = (line[21] or " ").strip() or "A"
                atom_name = line[12:16].strip().upper()

                is_target = (resnum == target_resnum) and (not target_chain or chain == target_chain)

                # delete reactive atom line
                if is_target and atom_name == deleted_atom:
                    continue

                # rename residue if requested
                if is_target and temp_resname:
                    line = f"{line[:17]}{temp_resname:>3s}{line[20:]}"

            f_out.write(line)

    msg = f"✅ Wrote covalent relax input → {out_pdb}"
    if orig_resname:
        msg += f" (orig_resname={orig_resname}, temp_resname={temp_resname or 'KEEP'})"
    print(msg)
    return out_pdb


# =========================================================
# 🚀 Main pipeline
# =========================================================
def run(job_dir: str | Path, multi_seed: bool = False, model_path: str | Path | None = None):
    job_dir = Path(job_dir)
    job_name = job_dir.name
    base_job = _trim_timestamp(job_name)

    job_meta_path = _ensure_job_metadata(job_dir, base_job)
    if not job_meta_path.exists():
        job_meta_path = job_dir / "prepared_meta.json"
    job_meta = json.loads(job_meta_path.read_text(encoding="utf-8"))

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")

    print(f"DEBUG: multi_seed={multi_seed!r} (type={type(multi_seed)})")

    # =====================================================
    # Multi-seed mode
    # =====================================================
    if multi_seed:
        runs_root = job_dir / "rosetta_runs"
        runs_root.mkdir(exist_ok=True)
        run_dir = runs_root / f"rosetta_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        scores_csv = job_dir / f"{base_job}_ranking_scores.csv"
        if not scores_csv.exists():
            raise FileNotFoundError(f"❌ Could not find ranking scores file: {scores_csv}")
        print(f"📊 Reading ranking scores: {scores_csv}")

        df = pd.read_csv(scores_csv, header=None, names=["seed", "sample", "score"])
        top_samples = (
            df.sort_values("score", ascending=False)
            .groupby("seed", as_index=False)
            .first()
        )

        ligand_smiles = job_meta.get("smiles")
        RESNAME = "LIG"
        params_path = run_dir / f"{RESNAME}.params"
        if params_path.exists():
            print(f"⏭️  Found existing params at {params_path}; skipping SDF/params regeneration.")
        else:
            sdf_rdkit = run_dir / "lig.sdf"
            sdf_rdkit = generate_sdf_from_smiles(ligand_smiles, sdf_rdkit)
            params_path = generate_rosetta_params_from_sdf(sdf_rdkit, run_dir, residue_name=RESNAME)

        outputs_index: list[dict[str, Any]] = []

        print("🧬 Found top-scoring samples per seed:")
        for _, row in top_samples.iterrows():
            seed = int(row["seed"])
            sample = int(row["sample"])
            score = float(row["score"])
            print(f"   Seed {seed} → sample {sample} (score={score:.4f})")

            sub_model = job_dir / f"seed-{seed}_sample-{sample}" / f"{base_job}_seed-{seed}_sample-{sample}_model.cif"
            if not sub_model.exists():
                print(f"⚠️ Missing CIF for seed {seed}, sample {sample}: {sub_model}")
                continue

            sub_dir = run_dir / f"rosetta_relax_seed{seed}_sample{sample}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            # CIF → PDB
            model_cif_linux = linuxize_path(sub_model)
            model_pdb_linux = linuxize_path(sub_dir / "model.pdb")
            gemmi_py = (
                "import gemmi; "
                f"c='{model_cif_linux}'; p='{model_pdb_linux}'; "
                "st=gemmi.read_structure(c); "
                "st.remove_alternative_conformations(); "
                "st.write_pdb(p); "
                "print('✅ Wrote', p)"
            )
            run_wsl(f"python3 -c \"{gemmi_py}\"", log=sub_dir / "gemmi_convert.log")

            # Clean PDB
            print("🧼 Running Rosetta clean_pdb_keep_ligand.py ...")
            clean_cmd = (
                f"cd '{linuxize_path(sub_dir)}' && "
                f"python3 '{CLEAN_PDB_PY}' model.pdb -ignorechain"
            )
            run_wsl(clean_cmd, log=sub_dir / "clean_pdb.log")

            cleaned_pdb_original = sub_dir / "model.pdb_00.pdb"
            if not cleaned_pdb_original.exists():
                raise FileNotFoundError(f"❌ cleaned PDB missing: {cleaned_pdb_original}")
            print(f"✅ Cleaned PDB ready: {cleaned_pdb_original}")

            # Build relax command (covalent-aware)
            prep_dir_linux = linuxize_path(sub_dir)
            is_covalent = bool(job_meta.get("covalent"))
            cov_meta: dict[str, Any] | None = None

            relax_input = cleaned_pdb_original
            extra_res_flags = ""

            if is_covalent:
                target_resnum = int(job_meta.get("residue"))
                prot_atom = (job_meta.get("prot_atom") or "").strip().upper()
                patch = COVALENT_PATCHES.get(prot_atom)

                cov_meta = {
                    "residue": target_resnum,
                    "prot_atom": prot_atom,
                    "chain": (job_meta.get("chain") or job_meta.get("prot_chain") or "A").strip()[:1] or "A",
                    "ligand_atom": (job_meta.get("ligand_atom") or job_meta.get("lig_atom") or job_meta.get("ligandAtom") or "").strip(),
                    "lig_resname": "LIG",
                }

                if patch:
                    relax_input = _make_relax_input_pdb_for_covalent(
                        cleaned_pdb_original=cleaned_pdb_original,
                        out_pdb=sub_dir / "model.pdb_00_relax_input.pdb",
                        target_resnum=target_resnum,
                        patch=patch,
                    )
                    if patch.get("extra_res_fa"):
                        extra_res_flags += f" -extra_res_fa '{patch['extra_res_fa']}'"
                else:
                    print(f"⚠️ covalent=True but prot_atom={prot_atom!r} has no patch rule. Proceeding without deletion/rename.")

            if params_path and params_path.exists():
                extra_res_flags += f" -extra_res_fa '{linuxize_path(params_path)}'"

            relax_cmd = (
                f"'{RELAX_BIN}' -database '{ROSETTA_DB}' "
                f"-s '{linuxize_path(relax_input)}' "
                f"-relax:ramp_constraints false "
                f"-score:weights ref2015_cart "
                f"-relax:cartesian "
                f"-out:path:all '{prep_dir_linux}' "
                f"-out:file:scorefile '{prep_dir_linux}/score.sc' "
                f"-nstruct 1 -overwrite "
                f"{extra_res_flags}"
            )

            run_wsl(relax_cmd, log=sub_dir / "relax.log")

            # Identify relaxed output and restore atom (if covalent)
            candidates = sorted(sub_dir.glob("*_0001.pdb")) + sorted(sub_dir.glob("*_0002.pdb"))
            if not candidates:
                candidates = sorted(sub_dir.glob("*_000*.pdb"))
            if not candidates:
                candidates = sorted(sub_dir.glob("*.pdb"))

            relaxed_pdb_raw = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
            relaxed_pdb_final = relaxed_pdb_raw

            if is_covalent and relaxed_pdb_raw and cov_meta:
                prot_atom = str(cov_meta["prot_atom"]).strip().upper()
                patch = COVALENT_PATCHES.get(prot_atom)
                if patch:
                    anchor0 = str(patch.get("anchor_atom") or "CB").strip().upper()
                    anchor_candidates = [anchor0, "CB", "CA", "C", "N"]
                    # de-dup preserving order
                    seen = set()
                    anchor_candidates = [a for a in anchor_candidates if not (a in seen or seen.add(a))]

                    relaxed_pdb_final = _restore_deleted_atom_after_relax(
                        original_cleaned_pdb=cleaned_pdb_original,
                        relaxed_pdb=relaxed_pdb_raw,
                        target_resnum=int(cov_meta["residue"]),
                        deleted_atom=str(patch["deleted_atom"]),
                        anchor_atoms=anchor_candidates,
                        element=str(patch.get("element") or "X"),
                        out_pdb=sub_dir / "model_relaxed_restored.pdb",
                    )

            out = _write_relax_outputs(
                job_dir=job_dir,
                prep_dir=sub_dir,
                params_path=params_path,
                is_covalent=is_covalent,
                covalent_meta=cov_meta,
                relaxed_pdb_raw=relaxed_pdb_raw,
                relaxed_pdb_final=relaxed_pdb_final,
            )
            outputs_index.append(out)

            print(f"✅ Finished relax for seed {seed}, sample {sample}\n")

        (run_dir / "rosetta_relax_multi_index.json").write_text(json.dumps(outputs_index, indent=2), encoding="utf-8")
        (job_dir / "latest_rosetta_relax.json").write_text(
            json.dumps({"multi_seed": True, "run_dir": str(run_dir), "index": str(run_dir / "rosetta_relax_multi_index.json")}, indent=2),
            encoding="utf-8",
        )

        print(f"📁 All per-seed relaxes written to {run_dir}")
        return

    # =====================================================
    # Single-model mode
    # =====================================================
    prep_dir = job_dir / f"rosetta_relax_{timestamp}"
    prep_dir.mkdir(parents=True, exist_ok=True)

    model_path = _resolve_model_cif(job_dir, job_name, base_job, model_path)
    print(f"📦 Processing AF3 model: {model_path}")

    # CIF → PDB
    model_cif_linux = linuxize_path(model_path)
    model_pdb_linux = linuxize_path(prep_dir / "model.pdb")
    gemmi_py = (
        "import gemmi; "
        f"c='{model_cif_linux}'; p='{model_pdb_linux}'; "
        "st=gemmi.read_structure(c); "
        "st.remove_alternative_conformations(); "
        "st.write_pdb(p); "
        "print('✅ Wrote', p)"
    )
    run_wsl(f"python3 -c \"{gemmi_py}\"", log=prep_dir / "gemmi_convert.log")

    # Ligand params (optional)
    ligand_smiles = job_meta.get("smiles")
    RESNAME = "LIG"
    params_path: Path | None = None
    if not ligand_smiles:
        print("ℹ️ No 'smiles' field found — skipping ligand generation (apo model).")
    else:
        params_path = prep_dir / f"{RESNAME}.params"
        if params_path.exists():
            print(f"⏭️  Found existing params at {params_path}; skipping SDF/params regeneration.")
        else:
            sdf_rdkit = prep_dir / "lig.sdf"
            sdf_rdkit = generate_sdf_from_smiles(ligand_smiles, sdf_rdkit)
            params_path = generate_rosetta_params_from_sdf(sdf_rdkit, prep_dir, residue_name=RESNAME)

    # Clean PDB
    print("🧼 Running Rosetta clean_pdb_keep_ligand.py ...")
    clean_cmd = (
        f"cd '{linuxize_path(prep_dir)}' && "
        f"python3 '{CLEAN_PDB_PY}' model.pdb -ignorechain"
    )
    run_wsl(clean_cmd, log=prep_dir / "clean_pdb.log")

    cleaned_pdb_original = prep_dir / "model.pdb_00.pdb"
    if not cleaned_pdb_original.exists():
        raise FileNotFoundError(f"❌ cleaned PDB missing: {cleaned_pdb_original}")

    print(f"✅ Cleaned PDB ready: {cleaned_pdb_original}")

    # Build relax command (covalent-aware)
    prep_dir_linux = linuxize_path(prep_dir)
    is_covalent = bool(job_meta.get("covalent"))
    cov_meta: dict[str, Any] | None = None

    relax_input = cleaned_pdb_original
    extra_res_flags = ""

    if is_covalent:
        target_resnum = int(job_meta.get("residue"))
        prot_atom = (job_meta.get("prot_atom") or "").strip().upper()
        patch = COVALENT_PATCHES.get(prot_atom)

        cov_meta = {
            "residue": target_resnum,
            "prot_atom": prot_atom,
            "chain": (job_meta.get("chain") or job_meta.get("prot_chain") or "A").strip()[:1] or "A",
            "ligand_atom": (job_meta.get("ligand_atom") or job_meta.get("lig_atom") or job_meta.get("ligandAtom") or "").strip(),
            "lig_resname": "LIG",
        }

        if patch:
            relax_input = _make_relax_input_pdb_for_covalent(
                cleaned_pdb_original=cleaned_pdb_original,
                out_pdb=prep_dir / "model.pdb_00_relax_input.pdb",
                target_resnum=target_resnum,
                patch=patch,
            )
            if patch.get("extra_res_fa"):
                extra_res_flags += f" -extra_res_fa '{patch['extra_res_fa']}'"
        else:
            print(f"⚠️ covalent=True but prot_atom={prot_atom!r} has no patch rule. Proceeding without deletion/rename.")

    if params_path and params_path.exists():
        extra_res_flags += f" -extra_res_fa '{linuxize_path(params_path)}'"

    relax_cmd = (
        f"'{RELAX_BIN}' "
        f"-database '{ROSETTA_DB}' "
        f"-s '{linuxize_path(relax_input)}' "
        f"-relax:ramp_constraints false "
        f"-score:weights ref2015_cart "
        f"-relax:cartesian "
        f"-out:path:all '{prep_dir_linux}' "
        f"-out:file:scorefile '{prep_dir_linux}/score.sc' "
        f"-nstruct 1 -overwrite "
        f"{extra_res_flags}"
    )

    run_wsl(relax_cmd, log=prep_dir / "relax.log")

    # Identify relaxed output and restore atom (if covalent)
    candidates = sorted(prep_dir.glob("*_0001.pdb")) + sorted(prep_dir.glob("*_0002.pdb"))
    if not candidates:
        candidates = sorted(prep_dir.glob("*_000*.pdb"))
    if not candidates:
        candidates = sorted(prep_dir.glob("*.pdb"))

    relaxed_pdb_raw = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    relaxed_pdb_final = relaxed_pdb_raw

    if is_covalent and relaxed_pdb_raw and cov_meta:
        prot_atom = str(cov_meta["prot_atom"]).strip().upper()
        patch = COVALENT_PATCHES.get(prot_atom)
        if patch:
            anchor0 = str(patch.get("anchor_atom") or "CB").strip().upper()
            anchor_candidates = [anchor0, "CB", "CA", "C", "N"]
            seen = set()
            anchor_candidates = [a for a in anchor_candidates if not (a in seen or seen.add(a))]

            relaxed_pdb_final = _restore_deleted_atom_after_relax(
                original_cleaned_pdb=cleaned_pdb_original,
                relaxed_pdb=relaxed_pdb_raw,
                target_resnum=int(cov_meta["residue"]),
                deleted_atom=str(patch["deleted_atom"]),
                anchor_atoms=anchor_candidates,
                element=str(patch.get("element") or "X"),
                out_pdb=prep_dir / "model_relaxed_restored.pdb",
            )

    # Write outputs JSON for downstream steps
    _write_relax_outputs(
        job_dir=job_dir,
        prep_dir=prep_dir,
        params_path=params_path,
        is_covalent=is_covalent,
        covalent_meta=cov_meta,
        relaxed_pdb_raw=relaxed_pdb_raw,
        relaxed_pdb_final=relaxed_pdb_final,
    )

    # -------------------------------------------------
    # (Optional) Copy to ~/.af3_pipeline/jobs (unchanged)
    # -------------------------------------------------
    import shutil
    print("\n✅ Done.")
    print(f"  CIF → PDB:           {model_pdb_linux}")
    print(f"  Cleaned PDB:         {cleaned_pdb_original}")
    if params_path:
        print(f"  Ligand params:       {params_path}")
    print(f"  Output directory:    {prep_dir}")
    print(f"  Scorefile:           {prep_dir}/score.sc")

    try:
        jobs_root = _user_jobs_root()
        dest_dir = jobs_root / job_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Copying key results to {dest_dir} ...")

        patterns = [
            "*_model.cif",
            "*_model.pdb",
            "*_ranking_scores.csv",
            "*_ranking_scores.xlsx",
            "*_confidences.json",
            "*_summary_confidences.json",
            "prepared_meta.json",
            "job_metadata.json",
            "metrics_summary.csv",
            "latest_rosetta_relax.json",
        ]

        copied_any = False
        for pat in patterns:
            for src in sorted(job_dir.glob(pat)):
                if src.is_file():
                    shutil.copy2(src, dest_dir / src.name)
                    print(f"   ✅ Copied {src.name}")
                    copied_any = True

        if not copied_any:
            print("   ⚠️ No AF3 artifacts matched expected patterns to copy.")

        relax_folders = sorted(job_dir.glob("rosetta_relax_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if relax_folders:
            latest_relax = relax_folders[0]
            print(f"   Using {latest_relax.name}")

            rosetta_out = dest_dir / latest_relax.name
            rosetta_out.mkdir(parents=True, exist_ok=True)

            for fname, dest_name in [
                ("LIG.pdb", "LIG.pdb"),
                ("lig.sdf", "lig.sdf"),
                ("LIG.params", "LIG.params"),
                ("model.pdb", "model.pdb"),
                ("model.pdb_00.pdb", "cleaned_af3_model.pdb"),
                ("model.pdb_00_relax_input.pdb", "model.pdb_00_relax_input.pdb"),
                ("model_relaxed_restored.pdb", "model_relaxed_restored.pdb"),
                ("rosetta_relax_outputs.json", "rosetta_relax_outputs.json"),
                ("score.sc", "score.sc"),
                ("relax.log", "relax.log"),
                ("clean_pdb.log", "clean_pdb.log"),
                ("gemmi_convert.log", "gemmi_convert.log"),
            ]:
                src = latest_relax / fname
                if src.exists():
                    shutil.copy2(src, rosetta_out / dest_name)
                    print(f"   ✅ Copied {latest_relax.name}/{dest_name}")
                else:
                    print(f"   ⚠️ Missing {latest_relax.name}/{fname}")

            relaxed_model = next(latest_relax.glob("model.pdb_00_*.pdb"), None)
            if relaxed_model:
                shutil.copy2(relaxed_model, rosetta_out / "model_relaxed_raw.pdb")
                print(f"   ✅ Copied {latest_relax.name}/{relaxed_model.name} → {latest_relax.name}/model_relaxed_raw.pdb")

        print("📦 File transfer complete.\n")

    except Exception as e:
        print(f"⚠️ Copying outputs failed: {e}")


# =========================================================
# 🧩 Entry point
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AF3 model & Rosetta relax (covalent-aware).")
    parser.add_argument("--job", required=True, help="Job folder under af_output (full folder name incl. timestamp)")
    parser.add_argument("--multi_seed", action="store_true", help="Analyze top model per seed")
    parser.add_argument("--model", required=False, help="Path to model CIF/PDB (optional)")
    args = parser.parse_args()

    run(AF_OUTPUT_DIR / args.job, multi_seed=bool(args.multi_seed), model_path=args.model)
