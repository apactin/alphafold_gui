#!/usr/bin/env python3
"""
json_builder.py
================
AF3 fold_input.json generator.
- Resolves MSA, templates, ligands.
- Portable mode: writes JSON to workspace/jobs/<job>/input/fold_input.json
- Legacy mode: writes JSON to WSL path (/home/.../af_input) via UNC (your current behavior)
- Caches metadata for post-AF3 analysis (link atoms, etc.) in the job folder.

This is a drop-in replacement that preserves current JSON structure and behavior,
but makes output location portable.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Optional

# --- Configuration ---
from .config import cfg

DEFAULT_MODEL_SEEDS = [10, 42]

# =========================
# 🧩 Helper imports
# =========================
from .cache_utils import (
    get_cached_msa,
    get_cached_ligand_cif,
    save_ligand_cif_cache,
    save_msa_cache,
)
from .msa_utils import build_msa
from .template_utils import get_template_mapping
from .ligand_utils import prepare_ligand_from_smiles


# =========================
# 🧭 Progress hook
# =========================
def _emit(msg: str):
    hook = globals().get("_progress_hook")
    if callable(hook):
        hook(msg)
    else:
        print(msg)


# =========================
# 🧠 Helpers
# =========================
def _ensure_clean_msa(msa_input):
    """Force inline cleaned MSA text for AF3."""
    if isinstance(msa_input, (str, Path)):
        p = Path(msa_input)
        text = ""
        if p.exists():
            if p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    _emit(f"⚠️ Failed to read MSA file: {e}")
            else:
                _emit(f"MSA path is a directory, ignoring: {p}")
        else:
            text = str(msa_input)
    else:
        text = ""
    text = text.replace("\x00", "").replace("\r", "").replace(".", "-")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text


def _split_ccd(s: str):
    if not s:
        return []
    return [t for t in (x.strip().upper() for x in s.split(",")) if t and t not in {"NONE", "NULL"}]


def _sanitize_jobname(name: str) -> str:
    return (
        (name or "").strip()
        .replace("+", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

def _as_mol_list(mol) -> list[dict]:
    """
    Normalize RNA/DNA payload to a list of dict entries.
    Accepts:
      - None -> []
      - dict -> [dict]
      - list[dict] -> list
    """
    if mol is None:
        return []
    if isinstance(mol, dict):
        return [mol]
    if isinstance(mol, list):
        return [x for x in mol if isinstance(x, dict)]
    return []



def _build_modifications(mod_type_ccd: str, pos: str, mode: str = "protein"):
    if not pos or not mod_type_ccd or mod_type_ccd.lower() == "none":
        return []
    try:
        pos_int = int(pos)
    except ValueError:
        return []

    if mode == "protein":
        return [{"ptmType": mod_type_ccd, "ptmPosition": pos_int}]
    else:
        return [{"modificationType": mod_type_ccd, "basePosition": pos_int}]


def _legacy_wsl_paths():
    """
    Reconstruct your existing AF_INPUT/AF_OUTPUT POSIX and UNC paths.
    Used only when backend_mode is wsl_legacy.
    """
    # Prefer new nested key if present, otherwise fall back to old flat keys
    wsl_cfg = cfg.get("wsl_legacy", {}) or {}
    distro = wsl_cfg.get("wsl_distro") or cfg.get("wsl_distro", "Ubuntu-22.04")
    base_linux = wsl_cfg.get("af3_dir") or cfg.get("af3_dir", "")
    base_linux = str(base_linux).replace("\\", "/").rstrip("/")

    af_in_posix = PurePosixPath(f"{base_linux}/af_input")
    af_out_posix = PurePosixPath(f"{base_linux}/af_output")

    af_in_unc = Path(rf"\\wsl.localhost\{distro}" + str(af_in_posix).replace("/", "\\"))
    af_out_unc = Path(rf"\\wsl.localhost\{distro}" + str(af_out_posix).replace("/", "\\"))

    return distro, base_linux, af_in_posix, af_out_posix, af_in_unc, af_out_unc


# =========================
# 🧾 Metadata writer for post-AF3
# =========================
def _write_job_metadata(job_root: Path, ligand: dict):
    """
    Cache ligand + linkage info for downstream analysis.

    Portable: written to workspace/jobs/<job>/job_metadata.json
    Legacy: written to WSL output folder like before (caller passes that dir)
    """
    meta = {
        "smiles": ligand.get("smiles", ""),
        "covalent": bool(ligand.get("covalent", False)),
        "chain": ligand.get("chain", ""),
        "residue": ligand.get("residue", ""),
        "prot_atom": ligand.get("prot_atom", ""),
        "ligand_atom": ligand.get("ligand_atom", ""),
        "ions": ligand.get("ions", ""),
        "cofactors": ligand.get("cofactors", ""),
        "modelSeeds": ligand.get("modelSeeds", None),
    }
    out = job_root / "job_metadata.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _emit(f"🧾 Wrote metadata → {out}")
    return out


# =========================
# 🧬 Main builder
# =========================
def build_input(jobname, proteins, rna, dna, ligand):
    """
    Returns a path to the written JSON.

    - Portable (backend_mode=docker): returns Windows path to workspace/jobs/<job>/input/fold_input.json
    - Legacy (backend_mode=wsl_legacy): returns Linux POSIX path under AF_INPUT (as before)
    """
    jobname = _sanitize_jobname(jobname)
    if not jobname:
        raise ValueError("Job name is empty after sanitization")

    distro, base_linux, AF_INPUT_POSIX, AF_OUTPUT_POSIX, AF_INPUT_UNC, AF_OUTPUT_UNC = _legacy_wsl_paths()

    # Create directories in Linux filesystem (via wsl mkdir) + ensure visible via UNC
    os.system(f"wsl mkdir -p {AF_INPUT_POSIX}")
    os.system(f"wsl mkdir -p {AF_OUTPUT_POSIX}")
    AF_INPUT_UNC.mkdir(parents=True, exist_ok=True)
    AF_OUTPUT_UNC.mkdir(parents=True, exist_ok=True)

    json_path_unc = AF_INPUT_UNC / f"{jobname}_fold_input.json"
    json_return = str(AF_INPUT_POSIX / f"{jobname}_fold_input.json")

    # Legacy: output folder under WSL
    job_out_dir = AF_OUTPUT_UNC / jobname
    job_out_dir.mkdir(parents=True, exist_ok=True)

    sequences = []
    chain_ids = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # --------------------------------
    # 🧬 Proteins
    # --------------------------------
    for p in proteins:
        seq = (p.get("sequence", "") or "").strip()
        if not seq:
            continue
        pid = next(chain_ids)

        msa_str = get_cached_msa(seq)
        if msa_str is None:
            _emit(f"Building MSA for chain {pid}…")
            msa_str = build_msa(seq, pid)
        msa_str = _ensure_clean_msa(msa_str)
        save_msa_cache(seq, msa_str)

        entry = {
            "protein": {
                "id": pid,
                "sequence": seq,
                "description": p.get("name") or f"Protein_{pid}",
                "modifications": _build_modifications(
                    p.get("modification", ""), p.get("mod_position", ""), "protein"
                ),
                "unpairedMsa": msa_str,
                "pairedMsa": "",
            }
        }

        template_id = (p.get("template", "") or "").strip()
        if template_id:
            _emit(f"Fetching template {template_id} for chain {pid}…")
            mapping = get_template_mapping(seq, template_id)
            cif_str = mapping["mmcif"]
            qidx, tidx = mapping["queryIndices"], mapping["templateIndices"]
            entry["protein"]["templates"] = [
                {"mmcif": cif_str, "queryIndices": qidx, "templateIndices": tidx}
            ]
        else:
            entry["protein"]["templates"] = []

        sequences.append(entry)

    # --------------------------------
    # 🧬 RNA / DNA (supports single dict OR list of dicts)
    # --------------------------------
    for mol, tag in [(rna, "rna"), (dna, "dna")]:
        for entry_in in _as_mol_list(mol):
            seq = (entry_in.get("sequence", "") or "").strip()
            if not seq:
                continue

            cid = next(chain_ids)

            # RNA gets MSA; DNA doesn't
            msa_str = ""
            if tag == "rna":
                msa_str = get_cached_msa(seq)
                if msa_str is None:
                    _emit(f"Building MSA for {tag} chain {cid}…")
                    msa_str = build_msa(seq, cid)
                    if msa_str:
                        save_msa_cache(seq, msa_str)
                msa_str = _ensure_clean_msa(msa_str)

            sequences.append({
                tag: {
                    "id": cid,
                    "sequence": seq,
                    "modifications": _build_modifications(
                        entry_in.get("modification", ""),
                        entry_in.get("pos", ""),
                        tag
                    ),
                    **({"unpairedMsa": msa_str} if tag == "rna" else {}),
                }
            })


    # --------------------------------
    # 💊 Ligands / ions / cofactors
    # --------------------------------
    ligand_smiles = (ligand.get("smiles", "") or "").strip()
    ions = _split_ccd(ligand.get("ions", ""))
    cofactors = _split_ccd(ligand.get("cofactors", ""))
    ccd_list = ions + cofactors

    smiles_lid = None
    ccd_lid = None
    user_ccd_data = None

    if ccd_list:
        ccd_lid = next(chain_ids)
        sequences.append({"ligand": {"id": ccd_lid, "ccdCodes": ccd_list}})

    if ligand_smiles:
        smiles_lid = next(chain_ids)
        cif_data = get_cached_ligand_cif(ligand_smiles)

        if cif_data is None:
            _emit(f"Generating ligand CIF for SMILES (chain {smiles_lid})…")
            cif_path = prepare_ligand_from_smiles(
                ligand_smiles,
                "LIG",
                skip_if_cached=True,
            )
            cif_data = Path(cif_path).read_text(encoding="utf-8")
            save_ligand_cif_cache(ligand_smiles, cif_data)
        else:
            _emit("✅ Using cached ligand CIF")

        first_line = cif_data.splitlines()[0].strip()
        match = re.match(r"^data[_\-\s]*(\S+)", first_line)
        ligand_ccd = match.group(1).strip() if match else "LIG"

        sequences.append({"ligand": {"id": smiles_lid, "ccdCodes": [ligand_ccd]}})
        user_ccd_data = cif_data

    # --------------------------------
    # 🔗 Covalent bonds
    # --------------------------------
    bondedAtomPairs = []
    if ligand.get("covalent"):
        chain = (ligand.get("chain", "") or "").strip()
        res = (ligand.get("residue", "") or "").strip()
        prot_atom = (ligand.get("prot_atom", "") or "").strip()
        lig_atom = (ligand.get("ligand_atom", "") or "").strip()
        lig_id_to_use = smiles_lid or ccd_lid

        if chain and res and prot_atom and lig_atom and lig_id_to_use:
            try:
                res_int = int(res)
                bondedAtomPairs.append([[chain, res_int, prot_atom], [lig_id_to_use, 1, lig_atom]])
            except Exception:
                _emit(f"⚠️ Invalid residue number for covalent bond: {res}")
        else:
            _emit("⚠️ Missing covalent bond parameters — skipping link.")

    # --------------------------------
    # 🧾 Assemble JSON
    # --------------------------------
    model_seeds = ligand.get("modelSeeds") or DEFAULT_MODEL_SEEDS
    af3_input = {
        "name": jobname,
        "modelSeeds": model_seeds,
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }
    if bondedAtomPairs:
        af3_input["bondedAtomPairs"] = bondedAtomPairs
    if user_ccd_data:
        af3_input["userCCD"] = user_ccd_data

    json_text = json.dumps(af3_input, indent=2)

    # --------------------------------
    # 💾 Write JSON
    # --------------------------------
    json_path_unc.parent.mkdir(parents=True, exist_ok=True)
    json_path_unc.write_text(json_text, encoding="utf-8")

    _emit(f"✅ Wrote AF3 input → {json_return}")

    # --------------------------------
    # 🧾 Write metadata for post-AF3
    # --------------------------------
    _write_job_metadata(job_out_dir, ligand)

    # In portable mode we return the Windows path to the JSON file.
    # In legacy mode we return the Linux posix path (as before) so your legacy runner can use it.
    return json_return
