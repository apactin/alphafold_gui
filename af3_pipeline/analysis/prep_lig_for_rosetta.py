#!/usr/bin/env python3
"""
prep_lig_for_rosetta.py
=======================

Drop-in ligand preparation for Rosetta, refactored to:
- Pull defaults from rosetta_dicts.yaml via rosetta_config.py
- Be callable from rosetta_relax.py and rosetta_scripts.py interchangeably
- Use shared runtime helpers (rosetta_runtime.py) for WSL + path conversions
- Still supports standalone CLI usage

Produces in out_dir:
  - lig.sdf (if SMILES provided, or if you copy/normalize an input SDF)
  - lig.mol2 (optional, via OpenBabel in WSL; best-effort)
  - <RESNAME>.params (via Rosetta molfile_to_params.py in WSL)
  - ligand_prep.json (optional)

Returns a small record describing outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
from typing import Optional

from af3_pipeline.config import cfg
from .rosetta_config import load_rosetta_dicts
from .rosetta_runtime import linuxize_path, run_wsl


# -------------------------
# Rosetta paths (inside WSL)
# -------------------------
ROSETTA_BASE = (cfg.get("rosetta_relax_bin") or "").strip()
if not ROSETTA_BASE:
    raise RuntimeError("Missing config key: rosetta_relax_bin (Rosetta bundle root inside WSL).")
ROSETTA_BASE = str(PurePosixPath(ROSETTA_BASE))

M2P_PY = f"{ROSETTA_BASE}/main/source/scripts/python/public/molfile_to_params.py"


# -------------------------
# Ligand prep outputs
# -------------------------
@dataclass
class LigandPrepResult:
    resname: str
    sdf_path: Optional[str]
    mol2_path: Optional[str]
    params_path: Optional[str]
    extra_res_fa_token: Optional[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# -------------------------
# RDKit SDF generation
# -------------------------
def generate_sdf_from_smiles(
    smiles: str,
    out_sdf: Path,
    *,
    num_confs: int,
    random_seed: int,
) -> Path:
    """
    Generate multi-conformer 3D SDF from SMILES with aromatic handling + MMFF minimization.
    Matches behavior from your original pipeline.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES — cannot parse molecule.")

    # Preserve aromatic handling
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol)

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(num_confs), params=params)
    for cid in conf_ids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    for cid in conf_ids:
        w.write(mol, confId=cid)
    w.close()

    return out_sdf


# -------------------------
# OpenBabel conversion (best-effort)
# -------------------------
def convert_sdf_to_mol2_wsl(*, sdf: Path, mol2: Path, out_dir: Path) -> None:
    """
    Convert SDF -> MOL2 using OpenBabel inside WSL.
    Writes sdf_to_mol2.log via run_wsl log parameter.
    """
    sdf_linux = linuxize_path(sdf)
    mol2_linux = linuxize_path(mol2)
    out_linux = linuxize_path(out_dir)

    cmd = (
        "set -euo pipefail; "
        "command -v obabel >/dev/null 2>&1 || "
        "(echo 'OpenBabel not installed. Try: sudo apt-get update && sudo apt-get install -y openbabel' && exit 2); "
        f"cd '{out_linux}' && "
        f"obabel -isdf '{sdf_linux}' -omol2 -O '{mol2_linux}'"
    )
    run_wsl(cmd, log=out_dir / "sdf_to_mol2.log")


# -------------------------
# Rosetta params generation
# -------------------------
def generate_rosetta_params_from_sdf(*, sdf: Path, out_dir: Path, residue_name: str) -> Path:
    """
    Run Rosetta molfile_to_params.py inside WSL to produce <residue_name>.params in out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / f"{residue_name}.params"
    if params_path.exists():
        return params_path

    sdf_linux = linuxize_path(sdf)
    out_linux = linuxize_path(out_dir)

    cmd = (
        f"cd '{out_linux}' && "
        f"python3 '{M2P_PY}' "
        f"-n {residue_name} -p {residue_name} "
        f"--conformers-in-one-file '{sdf_linux}'"
    )
    run_wsl(cmd, log=out_dir / "molfile_to_params.log")

    if not params_path.exists():
        raise FileNotFoundError(f"Rosetta params not found after molfile_to_params.py: {params_path}")

    return params_path


# -------------------------
# Public API
# -------------------------
def prep_ligand_for_rosetta(
    *,
    out_dir: str | Path,
    smiles: str | None = None,
    sdf_in: str | Path | None = None,
    # Optional overrides; if None, pulled from YAML defaults
    resname: str | None = None,
    make_mol2: bool | None = None,
    num_confs: int | None = None,
    random_seed: int | None = None,
    yaml_path: str | Path | None = None,
) -> LigandPrepResult:
    """
    Prepare ligand artifacts in out_dir.

    Exactly one of (smiles, sdf_in) must be provided.

    Defaults come from rosetta_dicts.yaml unless explicitly overridden.
    """
    dicts = load_rosetta_dicts(yaml_path)

    # Defaults from YAML
    y_resname = str(dicts.get("ligand", "resname", default="LIG"))
    y_num_confs = int(dicts.get("ligand", "rdkit_sdf", "num_confs", default=20))
    y_seed = int(dicts.get("ligand", "rdkit_sdf", "random_seed", default=42))
    y_make_mol2 = bool(dicts.get("ligand", "openbabel", "enabled", default=True)) and bool(
        dicts.get("ligand", "openbabel", "sdf_to_mol2", default=True)
    )

    resname3 = (resname or y_resname).strip().upper()[:3]
    make_mol2 = y_make_mol2 if make_mol2 is None else bool(make_mol2)
    num_confs = y_num_confs if num_confs is None else int(num_confs)
    random_seed = y_seed if random_seed is None else int(random_seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exactly one input mode
    if (smiles is None) == (sdf_in is None):
        raise ValueError("Provide exactly one of smiles or sdf_in.")

    sdf_path = out_dir / "lig.sdf"
    mol2_path = out_dir / "lig.mol2"

    # Create/copy SDF
    if sdf_in is not None:
        sdf_in_p = Path(sdf_in)
        if not sdf_in_p.exists():
            raise FileNotFoundError(f"SDF input not found: {sdf_in_p}")
        sdf_path.write_bytes(sdf_in_p.read_bytes())
    else:
        generate_sdf_from_smiles(smiles or "", sdf_path, num_confs=num_confs, random_seed=random_seed)

    # Optional MOL2 (best-effort; do NOT overwrite real log on failure)
    if make_mol2:
        try:
            convert_sdf_to_mol2_wsl(sdf=sdf_path, mol2=mol2_path, out_dir=out_dir)
        except Exception as e:
            (out_dir / "sdf_to_mol2.failed.txt").write_text(
                f"FAILED to generate MOL2 via OpenBabel:\n{e}\n",
                encoding="utf-8",
                errors="replace",
            )

    # Params (required if ligand is to be used in Rosetta)
    params_path = generate_rosetta_params_from_sdf(sdf=sdf_path, out_dir=out_dir, residue_name=resname3)

    return LigandPrepResult(
        resname=resname3,
        sdf_path=str(sdf_path) if sdf_path.exists() else None,
        mol2_path=str(mol2_path) if mol2_path.exists() else None,
        params_path=str(params_path) if params_path.exists() else None,
        extra_res_fa_token=str(params_path) if params_path.exists() else None,
    )


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare ligand SDF/MOL2/params for Rosetta.")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--smiles", default=None, help="Ligand SMILES (mutually exclusive with --sdf)")
    ap.add_argument("--sdf", default=None, help="Input SDF path (mutually exclusive with --smiles)")
    ap.add_argument("--yaml", default=None, help="Path to rosetta_dicts.yaml (optional)")

    ap.add_argument("--resname", default=None, help="Override ligand 3-letter resname (default from YAML)")
    ap.add_argument("--no-mol2", action="store_true", help="Skip MOL2 generation via OpenBabel")
    ap.add_argument("--num-confs", type=int, default=None, help="Override RDKit conformer count (default from YAML)")
    ap.add_argument("--seed", type=int, default=None, help="Override RDKit random seed (default from YAML)")
    ap.add_argument("--write-json", action="store_true", help="Write ligand_prep.json in output directory")

    args = ap.parse_args()

    result = prep_ligand_for_rosetta(
        out_dir=args.out,
        smiles=args.smiles,
        sdf_in=args.sdf,
        resname=args.resname,
        make_mol2=(False if args.no_mol2 else None),
        num_confs=args.num_confs,
        random_seed=args.seed,
        yaml_path=args.yaml,
    )

    if args.write_json:
        (Path(args.out) / "ligand_prep.json").write_text(result.to_json(), encoding="utf-8")

    print(result.to_json())