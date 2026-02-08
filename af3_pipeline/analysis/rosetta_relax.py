#!/usr/bin/env python3
"""
rosetta_relax.py — Prepare AF3 model & Rosetta relax (covalent-aware)
===============================================================

Refactor of your original rosetta_minimize.py with these goals:
- Reads dictionaries/defaults from rosetta_dicts.yaml (via rosetta_config.py)
- Uses shared runtime helpers (rosetta_runtime.py) for WSL + path handling
- Ligand params generation moved out to prep_lig_for_rosetta.py
- Writes a clearer handoff JSON:
    - extra_res_fa_list (canonical)
    - extra_res_fa      (legacy whitespace string, for compatibility)

Behavior preserved (nearly identical):
  1) Resolve model CIF
  2) CIF → PDB (Gemmi in WSL)
  3) Optional ligand prep (SMILES → SDF/MOL2/params) via prep_lig_for_rosetta.py
  4) Clean PDB via clean_pdb_keep_ligand.py
  5) If covalent: delete reactive atom (+ optional temp residue rename) for relax input
  6) Run relax.static (cartesian, ref2015_cart)
  7) If covalent: restore deleted atom into relaxed PDB (+ optional alias residue mapping like LYS->LYD)
  8) Write outputs JSON + latest pointer JSON
  9) Multi-seed mode supported (top per seed from ranking_scores.csv)

Option A hook:
- This file includes a configuration hook for stage-1 covalent constraints,
  but DOES NOT yet switch to RosettaScripts-relax. We'll implement that next.

"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from af3_pipeline.config import cfg

from .rosetta_config import load_rosetta_dicts
from .rosetta_runtime import (
    run_wsl,
    linuxize_path,
    safe_write_text,
    extra_res_fa_from_list,
)

from .prep_lig_for_rosetta import prep_ligand_for_rosetta


# =========================================================
# Config / paths
# =========================================================
ROSETTA_BASE = (cfg.get("rosetta_relax_bin") or "").strip()
if not ROSETTA_BASE:
    raise RuntimeError("Missing config key: rosetta_relax_bin (Rosetta bundle root inside WSL).")
ROSETTA_BASE = str(PurePosixPath(ROSETTA_BASE))

RELAX_BIN = f"{ROSETTA_BASE}/main/source/bin/relax.static.linuxgccrelease"
ROSETTA_DB = f"{ROSETTA_BASE}/main/database"
CLEAN_PDB_PY = f"{ROSETTA_BASE}/main/source/src/apps/public/relax_w_allatom_cst/clean_pdb_keep_ligand.py"

DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")
LINUX_HOME = cfg.get("linux_home_root", "")
ALPHAFOLD_BASE = cfg.get("af3_dir", f"{LINUX_HOME}/Repositories/alphafold")

APP_TZ = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))


def now_local() -> datetime:
    return datetime.now(APP_TZ)


def _trim_timestamp(name: str) -> str:
    # _YYYYMMDD_HHMMSS or _YYYYMMDD-HHMMSS at end
    return re.sub(r"_[0-9]{8}[-_][0-9]{6}$", "", name or "")


def to_wsl_path(subpath: str) -> Path:
    """
    Return correct AF3 path depending on OS:
      - Windows: UNC \\wsl.localhost\\<distro>\\home\\... style
      - Non-Windows: /home/... style
    """
    sub = (subpath or "").replace("\\", "/").strip("/")
    if platform.system() == "Windows":
        base_path = ALPHAFOLD_BASE.replace("/", "\\")
        base = f"\\\\wsl.localhost\\{DISTRO_NAME}{base_path}"
        return Path(base + (("\\" + sub.replace("/", "\\")) if sub else ""))
    return Path(ALPHAFOLD_BASE + (("/" + sub) if sub else ""))


AF_OUTPUT_DIR = to_wsl_path("af_output")


def wsl_unc_from_linux(linux_abs: str) -> Path:
    """
    Map an absolute Linux path (/home/... or /mnt/...) to a Windows UNC path:
      \\wsl.localhost\\<distro>\\home\\...
    On non-Windows, returns Path(linux_abs).
    """
    s = (linux_abs or "").replace("\\", "/").strip()
    if platform.system() != "Windows":
        return Path(s)
    if not s.startswith("/"):
        raise ValueError(f"Expected absolute Linux path, got: {linux_abs!r}")
    return Path(f"\\\\wsl.localhost\\{DISTRO_NAME}" + s.replace("/", "\\"))


# =========================================================
# Metadata handling
# =========================================================
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
# Model resolution
# =========================================================
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


# =========================================================
# Covalent patching + restore
# =========================================================
def _normalize_res3(s: str | None) -> str:
    return (s or "").strip().upper()[:3]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rosetta_db_host_path(db_rel: str) -> Path:
    """
    Convert a DB-relative path to a host-readable path:
      - Windows: UNC path under \\wsl.localhost\\<distro>\\home\\...\\main\\database\\...
      - Non-Windows: /home/.../main/database/...
    """
    rel = (db_rel or "").strip().lstrip("/").replace("\\", "/")
    db_posix = str(PurePosixPath(ROSETTA_DB) / rel)
    return wsl_unc_from_linux(db_posix) if platform.system() == "Windows" else Path(db_posix)


def _make_3letter_alias_params(*, src_params: Path, alias3: str, out_params: Path) -> Path:
    """
    Create a new params file that is a 3-letter alias of src_params.
    Rewrites NAME and IO_STRING tokens (and keeps chemistry unchanged).
    """
    alias3 = _normalize_res3(alias3)
    if len(alias3) != 3:
        raise ValueError(f"alias3 must be 3 letters, got: {alias3!r}")
    if not src_params.exists():
        raise FileNotFoundError(f"Source params not found: {src_params}")

    txt = src_params.read_text(encoding="utf-8", errors="replace").splitlines(True)

    out_lines = []
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

    _safe_mkdir(out_params.parent)
    out_params.write_text("".join(out_lines), encoding="utf-8")
    return out_params


def _patch_params_resname_tokens(*, params_path: Path, alias3: str) -> None:
    """
    Ensure NAME and IO_STRING are alias3, and also patch AA/ROTAMER_AA if present.
    """
    alias3 = _normalize_res3(alias3)
    lines = params_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []

    did_name = False
    did_io = False
    for ln in lines:
        s = ln.strip()

        if s.startswith("NAME ") and not did_name:
            out.append(f"NAME {alias3}")
            did_name = True
            continue

        if s.startswith("IO_STRING ") and not did_io:
            parts = s.split()
            one = parts[2] if len(parts) >= 3 else "X"
            out.append(f"IO_STRING {alias3} {one}")
            did_io = True
            continue

        if s.startswith("AA "):
            out.append(f"AA {alias3}")
            continue

        if s.startswith("ROTAMER_AA "):
            out.append(f"ROTAMER_AA {alias3}")
            continue

        out.append(ln)

    params_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _ensure_alias_params_for_resname(
    *,
    orig_resname3: str,
    alias_rules: dict[str, Any],
    alias_dir: Path,
) -> tuple[str, Path] | None:
    """
    If orig_resname3 is in alias_rules, create/update the alias params and return:
      (alias3, alias_params_path)
    else None.
    """
    orig_resname3 = _normalize_res3(orig_resname3)
    rule = alias_rules.get(orig_resname3)
    if not isinstance(rule, dict):
        return None

    alias3 = _normalize_res3(rule.get("alias3"))
    src_rel = (rule.get("src_rel") or "").strip()
    if not alias3 or not src_rel:
        raise ValueError(f"Bad alias rule for {orig_resname3}: {rule}")

    src_params = _rosetta_db_host_path(src_rel)
    out_params = alias_dir / f"{alias3}.params"

    _make_3letter_alias_params(src_params=src_params, alias3=alias3, out_params=out_params)
    _patch_params_resname_tokens(params_path=out_params, alias3=alias3)

    return alias3, out_params


def _pdb_iter_atoms(lines: list[str]):
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


def _make_relax_input_pdb_for_covalent(
    *,
    cleaned_pdb_original: Path,
    out_pdb: Path,
    target_resnum: int,
    patch: dict[str, Any],
    target_chain: str | None,
) -> Path:
    """
    Relax input edit:
      - delete reactive atom
      - optionally rename residue type to temp_resname (or KEEP)
    IMPORTANT: respects chain to avoid editing multiple chains with same resseq.
    """
    deleted_atom = (patch.get("deleted_atom") or "").strip().upper()
    base_temp = (patch.get("temp_resname") or "").strip().upper() or None
    temp_by_resname = patch.get("temp_by_resname") or {}
    if not isinstance(temp_by_resname, dict):
        temp_by_resname = {}

    if target_chain:
        target_chain = target_chain.strip()[:1] or None

    # detect original resname at target
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

    # choose effective temp
    temp_resname = None
    if orig_resname and orig_resname in temp_by_resname:
        choice = (temp_by_resname.get(orig_resname) or "").strip().upper()
        if choice and choice != "KEEP":
            temp_resname = choice
    else:
        if base_temp and base_temp != "KEEP":
            temp_resname = base_temp

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

                if is_target and atom_name == deleted_atom:
                    continue
                if is_target and temp_resname:
                    line = f"{line[:17]}{temp_resname:>3s}{line[20:]}"
            f_out.write(line)

    msg = f"✅ Wrote covalent relax input → {out_pdb}"
    if orig_resname:
        msg += f" (orig_resname={orig_resname}, temp_resname={temp_resname or 'KEEP'}, chain={target_chain or 'ANY'})"
    print(msg, flush=True)
    return out_pdb


def _restore_deleted_atom_after_relax(
    *,
    original_cleaned_pdb: Path,
    relaxed_pdb: Path,
    target_resnum: int,
    target_chain: str | None,
    deleted_atom: str,
    anchor_atoms: list[str],
    element: str,
    out_pdb: Path,
    alias_rules: dict[str, Any],
    alias_dir: Path,
    extra_params_out: list[Path],
) -> Path:
    """
    Reinsert deleted atom into relaxed PDB and optionally map residue name to an alias (e.g., LYS->LYD).
    Respects target_chain when identifying and editing the residue.
    """
    orig_lines = original_cleaned_pdb.read_text(encoding="utf-8", errors="replace").splitlines(True)
    rel_lines = relaxed_pdb.read_text(encoding="utf-8", errors="replace").splitlines(True)

    deleted_atom = deleted_atom.strip().upper()
    anchor_atoms = [a.strip().upper() for a in anchor_atoms if a and a.strip()]

    if target_chain:
        target_chain = target_chain.strip()[:1] or None

    # find original residue info (by chain+resnum)
    orig_resname: str | None = None
    orig_chain: str | None = None
    orig_deleted: tuple[float, float, float] | None = None
    orig_anchor_by_name: dict[str, tuple[float, float, float]] = {}

    for _, _, resnum, chain, resname, atom, x, y, z in _pdb_iter_atoms(orig_lines):
        if resnum != target_resnum:
            continue
        if target_chain and chain != target_chain:
            continue
        if orig_resname is None:
            orig_resname = (resname or "").strip().upper()
            orig_chain = chain
        if atom == deleted_atom:
            orig_deleted = (x, y, z)
        if atom in anchor_atoms:
            orig_anchor_by_name[atom] = (x, y, z)

    if not orig_deleted:
        print(f"⚠️ Cannot restore {deleted_atom}: missing in original {original_cleaned_pdb}", flush=True)
        return relaxed_pdb

    # choose an anchor present in both original and relaxed
    chosen_anchor: str | None = None
    rel_anchor: tuple[float, float, float] | None = None
    insert_after_idx: int | None = None

    for a in anchor_atoms:
        if a not in orig_anchor_by_name:
            continue
        for idx, _, resnum, chain, resname, atom, x, y, z in _pdb_iter_atoms(rel_lines):
            if resnum != target_resnum:
                continue
            if target_chain and chain != target_chain:
                continue
            if atom == a:
                chosen_anchor = a
                rel_anchor = (x, y, z)
                insert_after_idx = idx
                break
        if chosen_anchor:
            break

    if not chosen_anchor or not rel_anchor or insert_after_idx is None:
        print(f"⚠️ Cannot restore {deleted_atom}: no usable anchor in relaxed for {target_chain or ''}{target_resnum}", flush=True)
        return relaxed_pdb

    orig_anchor = orig_anchor_by_name[chosen_anchor]

    # translate deleted atom vector relative to anchor
    dx = orig_deleted[0] - orig_anchor[0]
    dy = orig_deleted[1] - orig_anchor[1]
    dz = orig_deleted[2] - orig_anchor[2]
    new_x = rel_anchor[0] + dx
    new_y = rel_anchor[1] + dy
    new_z = rel_anchor[2] + dz

    chain_out = (orig_chain or target_chain or "A")[:1]
    orig_res3 = _normalize_res3(orig_resname or "UNK")

    # apply alias if rule exists
    resname_out = orig_res3
    alias = _ensure_alias_params_for_resname(
        orig_resname3=orig_res3,
        alias_rules=alias_rules,
        alias_dir=alias_dir,
    )
    if alias:
        alias3, alias_params = alias
        resname_out = alias3
        extra_params_out.append(alias_params)

    serial = 99999
    element = (element or "").strip().upper()[:2] or "X"
    new_line = (
        f"ATOM  {serial:5d} {deleted_atom:<4s} {resname_out:>3s} {chain_out:1s}"
        f"{target_resnum:4d}    "
        f"{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )

    out_lines: list[str] = []
    inserted = False

    for i, line in enumerate(rel_lines):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
            try:
                resnum = int(line[22:26])
            except ValueError:
                resnum = None
            chain = (line[21] or " ").strip() or "A"
            if resnum == target_resnum and (not target_chain or chain == target_chain):
                line = f"{line[:17]}{resname_out:>3s}{line[20:]}"
        out_lines.append(line)
        if (not inserted) and (i == insert_after_idx):
            out_lines.append(new_line)
            inserted = True

    if not inserted:
        out_lines.append(new_line)

    out_pdb.write_text("".join(out_lines), encoding="utf-8")

    if resname_out != orig_res3:
        print(f"✅ Restored {deleted_atom} and remapped {orig_res3}→{resname_out} → {out_pdb}", flush=True)
    else:
        print(f"✅ Restored {deleted_atom} → {out_pdb}", flush=True)

    return out_pdb


# =========================================================
# Outputs JSON
# =========================================================
def _write_relax_outputs(
    *,
    job_dir: Path,
    prep_dir: Path,
    relaxed_pdb_raw: Path | None,
    relaxed_pdb_final: Path | None,
    covalent_meta: dict[str, Any] | None,
    extra_res_fa_list: list[Path],
):
    """
    Writes:
      - prep_dir/rosetta_relax_outputs.json
      - job_dir/latest_rosetta_relax.json

    Canonical field:
      - extra_res_fa_list: [ ... ]
    Compatibility field:
      - extra_res_fa: "path1 path2 ..."
    """
    scorefile = prep_dir / "score.sc"

    extra_list_str = [str(p) for p in extra_res_fa_list if p and Path(p).exists()]
    extra_str = extra_res_fa_from_list(extra_list_str) if extra_list_str else None

    out = {
        "prep_dir": str(prep_dir),
        "scorefile": str(scorefile) if scorefile.exists() else None,
        "relaxed_pdb_raw": str(relaxed_pdb_raw) if relaxed_pdb_raw else None,
        "relaxed_pdb": str(relaxed_pdb_final) if relaxed_pdb_final else (str(relaxed_pdb_raw) if relaxed_pdb_raw else None),
        "extra_res_fa_list": extra_list_str or [],
        "extra_res_fa": extra_str,
        "covalent": bool(covalent_meta is not None),
        "covalent_meta": covalent_meta or None,
    }

    (prep_dir / "rosetta_relax_outputs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (job_dir / "latest_rosetta_relax.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# =========================================================
# Core runner
# =========================================================
def _cif_to_pdb_with_gemmi(*, model_cif: Path, out_pdb: Path, log: Path) -> None:
    model_cif_linux = linuxize_path(model_cif)
    out_pdb_linux = linuxize_path(out_pdb)

    gemmi_py = (
        "import gemmi; "
        f"c='{model_cif_linux}'; p='{out_pdb_linux}'; "
        "st=gemmi.read_structure(c); "
        "st.remove_alternative_conformations(); "
        "st.write_pdb(p); "
        "print('✅ Wrote', p)"
    )
    run_wsl(f"python3 -c \"{gemmi_py}\"", log=log)


def _clean_pdb_keep_ligand(*, work_dir: Path, input_pdb: Path, log: Path) -> Path:
    print("🧼 Running Rosetta clean_pdb_keep_ligand.py ...", flush=True)
    cmd = (
        f"cd '{linuxize_path(work_dir)}' && "
        f"python3 '{CLEAN_PDB_PY}' '{linuxize_path(input_pdb)}' -ignorechain"
    )
    run_wsl(cmd, log=log)

    cleaned = work_dir / (input_pdb.name + "_00.pdb")
    if not cleaned.exists():
        # Rosetta script usually outputs model.pdb_00.pdb if input was model.pdb
        # Keep the common name too
        fallback = work_dir / "model.pdb_00.pdb"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Cleaned PDB missing: {cleaned}")
    return cleaned


def _pick_relax_output_pdb(prep_dir: Path) -> Path | None:
    candidates = sorted(prep_dir.glob("*_0001.pdb")) + sorted(prep_dir.glob("*_0002.pdb"))
    if not candidates:
        candidates = sorted(prep_dir.glob("*_000*.pdb"))
    if not candidates:
        candidates = sorted(prep_dir.glob("*.pdb"))
    return candidates[0] if candidates else None


def _build_extra_res_fa_list_for_run(
    *,
    dicts,
    covalent_meta: dict[str, Any] | None,
    ligand_params_path: Path | None,
    alias_params_created: list[Path],
) -> list[Path]:
    """
    Collect all params paths that must be passed to Rosetta for THIS run directory.
    This returns host-readable paths; command assembly must linuxize them.
    """
    extra: list[Path] = []

    # covalent patch extra params (e.g., LYX.params under DB) must be a linux path when passed to Rosetta,
    # but we can store it as a Path via UNC mapping for existence checks if desired.
    if covalent_meta:
        prot_atom = (covalent_meta.get("prot_atom") or "").strip().upper()
        patch = dicts.get("covalent_patches", prot_atom, default=None)
        if isinstance(patch, dict):
            ex = patch.get("extra_res_fa")
            if isinstance(ex, dict) and ex.get("db_rel"):
                # DB path exists in WSL; store as UNC on Windows so downstream can read if needed
                db_rel = str(ex["db_rel"])
                extra.append(_rosetta_db_host_path(db_rel))
            elif isinstance(ex, str) and ex.strip():
                # treat as absolute posix; map to UNC if windows
                extra.append(wsl_unc_from_linux(ex.strip()) if platform.system() == "Windows" else Path(ex.strip()))

    if ligand_params_path and ligand_params_path.exists():
        extra.append(ligand_params_path)

    for p in (alias_params_created or []):
        if p and Path(p).exists():
            extra.append(Path(p))

    # de-dup
    seen = set()
    out: list[Path] = []
    for p in extra:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        out.append(p)
    return out


def run(job_dir: str | Path, *, multi_seed: bool = False, model_path: str | Path | None = None, yaml_path: str | Path | None = None):
    dicts = load_rosetta_dicts(yaml_path)

    job_dir = Path(job_dir)
    job_name = job_dir.name
    base_job = _trim_timestamp(job_name)

    job_meta_path = _ensure_job_metadata(job_dir, base_job)
    job_meta = json.loads(job_meta_path.read_text(encoding="utf-8"))

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")

    # -----------------------------------------
    # Multi-seed mode
    # -----------------------------------------
    if multi_seed:
        runs_root = job_dir / "rosetta_runs"
        runs_root.mkdir(exist_ok=True)
        run_dir = runs_root / f"rosetta_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        scores_csv = job_dir / f"{base_job}_ranking_scores.csv"
        if not scores_csv.exists():
            raise FileNotFoundError(f"Could not find ranking scores file: {scores_csv}")
        print(f"📊 Reading ranking scores: {scores_csv}", flush=True)

        df = pd.read_csv(scores_csv, header=None, names=["seed", "sample", "score"])
        top_samples = df.sort_values("score", ascending=False).groupby("seed", as_index=False).first()

        # Prepare ligand once per run_dir (same as your original behavior)
        ligand_smiles = (job_meta.get("smiles") or "").strip() or None
        ligand_params: Path | None = None
        if ligand_smiles:
            lig_resname = str(dicts.get("ligand", "resname", default="LIG")).strip().upper()[:3]
            lig_result = prep_ligand_for_rosetta(
                out_dir=run_dir,
                smiles=ligand_smiles,
                resname=lig_resname,
                yaml_path=yaml_path,
            )
            ligand_params = Path(lig_result.params_path) if lig_result.params_path else None
        else:
            print("ℹ️ No 'smiles' field found — assuming protein-only model (apo).", flush=True)

        outputs_index: list[dict[str, Any]] = []

        print("🧬 Found top-scoring samples per seed:", flush=True)
        for _, row in top_samples.iterrows():
            seed = int(row["seed"])
            sample = int(row["sample"])
            score = float(row["score"])
            print(f"   Seed {seed} → sample {sample} (score={score:.4f})", flush=True)

            sub_model = job_dir / f"seed-{seed}_sample-{sample}" / f"{base_job}_seed-{seed}_sample-{sample}_model.cif"
            if not sub_model.exists():
                print(f"⚠️ Missing CIF for seed {seed}, sample {sample}: {sub_model}", flush=True)
                continue

            sub_dir = run_dir / f"rosetta_relax_seed{seed}_sample{sample}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            # CIF → PDB
            model_pdb = sub_dir / "model.pdb"
            _cif_to_pdb_with_gemmi(model_cif=sub_model, out_pdb=model_pdb, log=sub_dir / "gemmi_convert.log")

            # Clean PDB
            cleaned_pdb_original = _clean_pdb_keep_ligand(work_dir=sub_dir, input_pdb=model_pdb, log=sub_dir / "clean_pdb.log")
            print(f"✅ Cleaned PDB ready: {cleaned_pdb_original}", flush=True)

            # Covalent-aware relax input
            is_covalent = bool(job_meta.get("covalent"))
            cov_meta: dict[str, Any] | None = None
            relax_input = cleaned_pdb_original

            alias_params_created: list[Path] = []

            if is_covalent:
                target_resnum = int(job_meta.get("residue"))
                prot_atom = (job_meta.get("prot_atom") or "").strip().upper()

                cov_meta = {
                    "residue": target_resnum,
                    "prot_atom": prot_atom,
                    "chain": (job_meta.get("chain") or job_meta.get("prot_chain") or "A").strip()[:1] or "A",
                    "ligand_atom": (job_meta.get("ligand_atom") or job_meta.get("lig_atom") or job_meta.get("ligandAtom") or "").strip(),
                    "lig_resname": str(dicts.get("ligand", "resname", default="LIG")).strip().upper()[:3],
                }

                patch = dicts.get("covalent_patches", prot_atom, default=None)
                if isinstance(patch, dict):
                    relax_input = _make_relax_input_pdb_for_covalent(
                        cleaned_pdb_original=cleaned_pdb_original,
                        out_pdb=sub_dir / str(dicts.get("relax_stage", "filenames", "relax_input_pdb", default="model.pdb_00_relax_input.pdb")),
                        target_resnum=target_resnum,
                        patch=patch,
                        target_chain=cov_meta["chain"],  # IMPORTANT: chain-aware now
                    )
                else:
                    print(f"⚠️ covalent=True but prot_atom={prot_atom!r} has no patch rule. Proceeding without deletion/rename.", flush=True)

            # Build extra_res_fa list and flags
            extra_res_fa_list = _build_extra_res_fa_list_for_run(
                dicts=dicts,
                covalent_meta=cov_meta,
                ligand_params_path=ligand_params,
                alias_params_created=alias_params_created,
            )
            extra_flags = ""
            for p in extra_res_fa_list:
                # For DB entries mapped as UNC, linuxize_path will strip the UNC prefix fine
                extra_flags += f" -extra_res_fa '{linuxize_path(Path(p))}'"

            # Run relax
            relax_cmd = (
                f"'{RELAX_BIN}' -database '{ROSETTA_DB}' "
                f"-s '{linuxize_path(relax_input)}' "
                f"-relax:ramp_constraints false "
                f"-score:weights ref2015_cart "
                f"-relax:cartesian "
                f"-out:path:all '{linuxize_path(sub_dir)}' "
                f"-out:file:scorefile '{linuxize_path(sub_dir)}/score.sc' "
                f"-nstruct 1 -overwrite "
                f"{extra_flags}"
            )
            run_wsl(relax_cmd, log=sub_dir / "relax.log")

            # Restore deleted atom (if covalent)
            relaxed_pdb_raw = _pick_relax_output_pdb(sub_dir)
            relaxed_pdb_final = relaxed_pdb_raw

            if is_covalent and relaxed_pdb_raw and cov_meta:
                prot_atom = str(cov_meta["prot_atom"]).strip().upper()
                patch = dicts.get("covalent_patches", prot_atom, default=None)
                if isinstance(patch, dict):
                    anchor0 = str(patch.get("anchor_atom") or "CB").strip().upper()
                    fallback = list(dicts.get("relax_stage", "restore_anchor_fallback", default=["CB", "CA", "C", "N"]))
                    anchors = [anchor0] + [a for a in fallback if a]
                    # de-dup
                    seen = set()
                    anchors = [a for a in anchors if not (a in seen or seen.add(a))]

                    alias_rules = dicts.get("resname_alias_rules", default={}) or {}
                    alias_dir = sub_dir / "params_alias"
                    restored_name = str(dicts.get("relax_stage", "filenames", "restored_pdb", default="model_relaxed_restored.pdb"))

                    relaxed_pdb_final = _restore_deleted_atom_after_relax(
                        original_cleaned_pdb=cleaned_pdb_original,
                        relaxed_pdb=relaxed_pdb_raw,
                        target_resnum=int(cov_meta["residue"]),
                        target_chain=str(cov_meta["chain"]),
                        deleted_atom=str(patch.get("deleted_atom") or prot_atom),
                        anchor_atoms=anchors,
                        element=str(patch.get("element") or "X"),
                        out_pdb=sub_dir / restored_name,
                        alias_rules=alias_rules if isinstance(alias_rules, dict) else {},
                        alias_dir=alias_dir,
                        extra_params_out=alias_params_created,
                    )

                    # Rebuild list including alias params created
                    extra_res_fa_list = _build_extra_res_fa_list_for_run(
                        dicts=dicts,
                        covalent_meta=cov_meta,
                        ligand_params_path=ligand_params,
                        alias_params_created=alias_params_created,
                    )

            out = _write_relax_outputs(
                job_dir=job_dir,
                prep_dir=sub_dir,
                relaxed_pdb_raw=relaxed_pdb_raw,
                relaxed_pdb_final=relaxed_pdb_final,
                covalent_meta=cov_meta,
                extra_res_fa_list=extra_res_fa_list,
            )
            outputs_index.append(out)
            print(f"✅ Finished relax for seed {seed}, sample {sample}\n", flush=True)

        (run_dir / "rosetta_relax_multi_index.json").write_text(json.dumps(outputs_index, indent=2), encoding="utf-8")
        (job_dir / "latest_rosetta_relax.json").write_text(
            json.dumps({"multi_seed": True, "run_dir": str(run_dir), "index": str(run_dir / "rosetta_relax_multi_index.json")}, indent=2),
            encoding="utf-8",
        )
        print(f"📁 All per-seed relaxes written to {run_dir}", flush=True)
        return

    # -----------------------------------------
    # Single-model mode
    # -----------------------------------------
    prep_dir = job_dir / f"rosetta_relax_{timestamp}"
    prep_dir.mkdir(parents=True, exist_ok=True)

    model_cif = _resolve_model_cif(job_dir, job_name, base_job, model_path)
    print(f"📦 Processing AF3 model: {model_cif}", flush=True)

    # CIF → PDB
    model_pdb = prep_dir / "model.pdb"
    _cif_to_pdb_with_gemmi(model_cif=model_cif, out_pdb=model_pdb, log=prep_dir / "gemmi_convert.log")

    # Ligand prep (optional)
    ligand_smiles = (job_meta.get("smiles") or "").strip() or None
    ligand_params: Path | None = None
    if ligand_smiles:
        lig_resname = str(dicts.get("ligand", "resname", default="LIG")).strip().upper()[:3]
        lig_result = prep_ligand_for_rosetta(
            out_dir=prep_dir,
            smiles=ligand_smiles,
            resname=lig_resname,
            yaml_path=yaml_path,
        )
        ligand_params = Path(lig_result.params_path) if lig_result.params_path else None
    else:
        print("ℹ️ No 'smiles' field found — skipping ligand generation (apo model).", flush=True)

    # Clean PDB
    cleaned_pdb_original = _clean_pdb_keep_ligand(work_dir=prep_dir, input_pdb=model_pdb, log=prep_dir / "clean_pdb.log")
    print(f"✅ Cleaned PDB ready: {cleaned_pdb_original}", flush=True)

    # Covalent-aware relax input
    is_covalent = bool(job_meta.get("covalent"))
    cov_meta: dict[str, Any] | None = None
    relax_input = cleaned_pdb_original

    alias_params_created: list[Path] = []

    if is_covalent:
        target_resnum = int(job_meta.get("residue"))
        prot_atom = (job_meta.get("prot_atom") or "").strip().upper()

        cov_meta = {
            "residue": target_resnum,
            "prot_atom": prot_atom,
            "chain": (job_meta.get("chain") or job_meta.get("prot_chain") or "A").strip()[:1] or "A",
            "ligand_atom": (job_meta.get("ligand_atom") or job_meta.get("lig_atom") or job_meta.get("ligandAtom") or "").strip(),
            "lig_resname": str(dicts.get("ligand", "resname", default="LIG")).strip().upper()[:3],
        }

        patch = dicts.get("covalent_patches", prot_atom, default=None)
        if isinstance(patch, dict):
            relax_input = _make_relax_input_pdb_for_covalent(
                cleaned_pdb_original=cleaned_pdb_original,
                out_pdb=prep_dir / str(dicts.get("relax_stage", "filenames", "relax_input_pdb", default="model.pdb_00_relax_input.pdb")),
                target_resnum=target_resnum,
                patch=patch,
                target_chain=cov_meta["chain"],  # IMPORTANT: chain-aware now
            )
        else:
            print(f"⚠️ covalent=True but prot_atom={prot_atom!r} has no patch rule. Proceeding without deletion/rename.", flush=True)

    # Build extra_res_fa list and flags
    extra_res_fa_list = _build_extra_res_fa_list_for_run(
        dicts=dicts,
        covalent_meta=cov_meta,
        ligand_params_path=ligand_params,
        alias_params_created=alias_params_created,
    )
    extra_flags = ""
    for p in extra_res_fa_list:
        extra_flags += f" -extra_res_fa '{linuxize_path(Path(p))}'"

    # Run relax
    relax_cmd = (
        f"'{RELAX_BIN}' -database '{ROSETTA_DB}' "
        f"-s '{linuxize_path(relax_input)}' "
        f"-relax:ramp_constraints false "
        f"-score:weights ref2015_cart "
        f"-relax:cartesian "
        f"-out:path:all '{linuxize_path(prep_dir)}' "
        f"-out:file:scorefile '{linuxize_path(prep_dir)}/score.sc' "
        f"-nstruct 1 -overwrite "
        f"{extra_flags}"
    )
    run_wsl(relax_cmd, log=prep_dir / "relax.log")

    # Restore deleted atom (if covalent)
    relaxed_pdb_raw = _pick_relax_output_pdb(prep_dir)
    relaxed_pdb_final = relaxed_pdb_raw

    if is_covalent and relaxed_pdb_raw and cov_meta:
        prot_atom = str(cov_meta["prot_atom"]).strip().upper()
        patch = dicts.get("covalent_patches", prot_atom, default=None)
        if isinstance(patch, dict):
            anchor0 = str(patch.get("anchor_atom") or "CB").strip().upper()
            fallback = list(dicts.get("relax_stage", "restore_anchor_fallback", default=["CB", "CA", "C", "N"]))
            anchors = [anchor0] + [a for a in fallback if a]
            seen = set()
            anchors = [a for a in anchors if not (a in seen or seen.add(a))]

            alias_rules = dicts.get("resname_alias_rules", default={}) or {}
            alias_dir = prep_dir / "params_alias"
            restored_name = str(dicts.get("relax_stage", "filenames", "restored_pdb", default="model_relaxed_restored.pdb"))

            relaxed_pdb_final = _restore_deleted_atom_after_relax(
                original_cleaned_pdb=cleaned_pdb_original,
                relaxed_pdb=relaxed_pdb_raw,
                target_resnum=int(cov_meta["residue"]),
                target_chain=str(cov_meta["chain"]),
                deleted_atom=str(patch.get("deleted_atom") or prot_atom),
                anchor_atoms=anchors,
                element=str(patch.get("element") or "X"),
                out_pdb=prep_dir / restored_name,
                alias_rules=alias_rules if isinstance(alias_rules, dict) else {},
                alias_dir=alias_dir,
                extra_params_out=alias_params_created,
            )

            extra_res_fa_list = _build_extra_res_fa_list_for_run(
                dicts=dicts,
                covalent_meta=cov_meta,
                ligand_params_path=ligand_params,
                alias_params_created=alias_params_created,
            )

    _write_relax_outputs(
        job_dir=job_dir,
        prep_dir=prep_dir,
        relaxed_pdb_raw=relaxed_pdb_raw,
        relaxed_pdb_final=relaxed_pdb_final,
        covalent_meta=cov_meta,
        extra_res_fa_list=extra_res_fa_list,
    )

    print("\n✅ Done.", flush=True)
    print(f"  Cleaned PDB:         {cleaned_pdb_original}", flush=True)
    if ligand_params:
        print(f"  Ligand params:       {ligand_params}", flush=True)
    print(f"  Output directory:    {prep_dir}", flush=True)
    print(f"  Scorefile:           {prep_dir}/score.sc", flush=True)


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare AF3 model & Rosetta relax (covalent-aware).")
    ap.add_argument("--job", required=True, help="Job folder under af_output (full folder name incl. timestamp)")
    ap.add_argument("--multi_seed", action="store_true", help="Analyze top model per seed")
    ap.add_argument("--model", required=False, help="Path to model CIF/PDB (optional)")
    ap.add_argument("--yaml", required=False, help="Path to rosetta_dicts.yaml (optional)")
    args = ap.parse_args()

    run(AF_OUTPUT_DIR / args.job, multi_seed=bool(args.multi_seed), model_path=args.model, yaml_path=args.yaml)
