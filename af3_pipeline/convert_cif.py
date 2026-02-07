#!/usr/bin/env python3
"""
convert_cif.py
==============
Converts ligand PDB files to AlphaFold 3–compatible CIF files.

Preserves:
- Atom naming (PDB ↔ CIF): CIF atom_id is exactly the PDB atom name
- Coordinates: taken from PDB
- Bond order + aromatic flags: prefers SDF sidecar (LIG.sdf) if available
- Stereo config: prefers SDF-based CIP assignment if available
- Falls back to RDKit-from-PDB and then CONECT-count heuristic if SDF is missing

Key change for portability:
- Uses the SAME cache root as cache_utils (CACHE_ROOT / ligands / <hash>)
"""

from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

from rdkit import Chem

from af3_pipeline.cache_utils import get_cache_dir, get_cache_file

def bond_order_from_count(count: int) -> Tuple[str, str]:
    # Legacy fallback heuristic based on duplicated CONECT entries
    if count >= 3:
        return "TRIP", "N"
    if count == 2:
        return "DOUB", "N"
    return "SING", "N"


def _infer_element_from_atom_name(atom_name: str) -> str:
    """Best-effort element inference when PDB element columns are blank."""
    s = (atom_name or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z]", "", s)
    if not s:
        return ""
    two = s[:2].upper()
    if two in {
        "CL","BR","NA","LI","MG","AL","SI","CA","ZN","FE","CU","MN","CO","NI",
        "SE","AS","HG","PB","SN","AG","AU","CD","CR"
    }:
        return two.title()
    return s[0].title()


def _order_aromatic_from_bond(bond: Chem.Bond) -> Tuple[str, str]:
    if bond.GetIsAromatic():
        return "SING", "Y"  # keep your convention
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return "SING", "N"
    if bt == Chem.BondType.DOUBLE:
        return "DOUB", "N"
    if bt == Chem.BondType.TRIPLE:
        return "TRIP", "N"
    return "SING", "N"


def _load_sdf_mol(sdf_path: Path) -> Optional[Chem.Mol]:
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
        if mol is None:
            return None
        Chem.SanitizeMol(mol, catchErrors=True)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return mol
    except Exception:
        return None


def convert_pdb_to_cif(key_hash: str, compound_id: str) -> Path:
    """
    Convert a cached ligand PDB to an AF3-compatible CIF.
    Returns the written CIF Path.
    """
    compound_id_clean = re.sub(r"[^A-Za-z0-9._]", "", compound_id).strip("_").upper()

    ligand_dir = get_cache_dir("ligands", key_hash)
    ligand_pdb = get_cache_file("ligands", key_hash, "LIG.pdb")
    ligand_sdf = get_cache_file("ligands", key_hash, "LIG.sdf")

    if not ligand_pdb.exists():
        raise FileNotFoundError(f"Missing ligand PDB at: {ligand_pdb}")

    atoms = []
    bond_counts = defaultdict(int)

    # ----------------------------
    # Read atoms + CONECT from PDB
    # ----------------------------
    with ligand_pdb.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("HETATM"):
                atom_serial = int(line[6:11])
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) >= 78 else ""
                if not element:
                    element = _infer_element_from_atom_name(atom_name)
                atoms.append((atom_serial, atom_name, element, x, y, z))

            elif line.startswith("CONECT"):
                parts = line.split()
                if len(parts) < 3:
                    continue
                a1 = int(parts[1])
                for p in parts[2:]:
                    try:
                        a2 = int(p)
                    except Exception:
                        continue
                    bond_counts[tuple(sorted((a1, a2)))] += 1

    atoms_sorted = sorted(atoms, key=lambda t: t[0])
    serial_to_name = {serial: name for serial, name, *_ in atoms_sorted}

    # PDB serial order → RDKit atom index mapping assumption (MolToPDBFile writes in atom index order)
    idx_to_serial = [serial for serial, *_ in atoms_sorted]

    # ----------------------------
    # Preferred: SDF chemistry
    # ----------------------------
    bond_info: Dict[Tuple[int, int], Tuple[str, str]] = {}
    atom_stereo_cfg: Dict[int, str] = {}
    atom_charge: Dict[int, int] = {}

    mol_sdf = _load_sdf_mol(ligand_sdf) if ligand_sdf.exists() else None
    if mol_sdf is not None:
        print(f"✅ Using SDF chemistry source for bond orders/aromaticity/stereo: {ligand_sdf}")

        for bond in mol_sdf.GetBonds():
            i1 = bond.GetBeginAtomIdx()
            i2 = bond.GetEndAtomIdx()
            if i1 >= len(idx_to_serial) or i2 >= len(idx_to_serial):
                continue
            s1 = idx_to_serial[i1]
            s2 = idx_to_serial[i2]
            bond_info[tuple(sorted((s1, s2)))] = _order_aromatic_from_bond(bond)

        for a in mol_sdf.GetAtoms():
            idx = a.GetIdx()
            if idx >= len(idx_to_serial):
                continue
            serial = idx_to_serial[idx]

            # Stereo (CIP) if available
            atom_stereo_cfg[serial] = a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "N"

            # Formal charge from SDF (reliable)
            atom_charge[serial] = int(a.GetFormalCharge())


    elif ligand_sdf.exists():
        print(f"⚠️ Found SDF but failed to parse; falling back to PDB inference: {ligand_sdf}")

    # ----------------------------
    # Fallback: RDKit from PDB
    # ----------------------------
    if not bond_info:
        try:
            mol_pdb = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False, sanitize=True)
            if mol_pdb is not None:
                Chem.SanitizeMol(mol_pdb, catchErrors=True)
                Chem.AssignStereochemistry(mol_pdb, force=True, cleanIt=True)

                for bond in mol_pdb.GetBonds():
                    a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                    info1, info2 = a1.GetPDBResidueInfo(), a2.GetPDBResidueInfo()
                    if info1 is None or info2 is None:
                        continue
                    s1, s2 = info1.GetSerialNumber(), info2.GetSerialNumber()
                    bond_info[tuple(sorted((s1, s2)))] = _order_aromatic_from_bond(bond)

                for atom in mol_pdb.GetAtoms():
                    info = atom.GetPDBResidueInfo()
                    if info is None:
                        continue
                    serial = info.GetSerialNumber()
                    atom_stereo_cfg[serial] = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "N"
                    atom_charge[serial] = int(atom.GetFormalCharge())
        except Exception as e:
            print(f"⚠️ Warning: RDKit PDB parsing failed; using CONECT heuristic: {e}")

    # ----------------------------
    # Final fallback: fill missing from CONECT
    # ----------------------------
    for key, count in bond_counts.items():
        if key not in bond_info:
            bond_info[key] = bond_order_from_count(count)

    # ----------------------------
    # Write CIF
    # ----------------------------
    output_path = ligand_pdb.with_suffix(".cif")

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"data_{compound_id_clean}\n#\n")
        out.write(f"_chem_comp.id {compound_id_clean}\n")
        out.write("_chem_comp.name ?\n")
        out.write("_chem_comp.type non-polymer\n")
        out.write("_chem_comp.formula ?\n")
        out.write("_chem_comp.mon_nstd_parent_comp_id ?\n")
        out.write("_chem_comp.pdbx_synonyms ?\n")
        out.write("_chem_comp.formula_weight ?\n#\n")

        out.write("loop_\n")
        out.write("_chem_comp_atom.comp_id\n")
        out.write("_chem_comp_atom.atom_id\n")
        out.write("_chem_comp_atom.type_symbol\n")
        out.write("_chem_comp_atom.charge\n")
        out.write("_chem_comp_atom.pdbx_leaving_atom_flag\n")
        out.write("_chem_comp_atom.pdbx_model_Cartn_x_ideal\n")
        out.write("_chem_comp_atom.pdbx_model_Cartn_y_ideal\n")
        out.write("_chem_comp_atom.pdbx_model_Cartn_z_ideal\n")
        out.write("_chem_comp_atom.pdbx_stereo_config\n")

        for serial, name, element, x, y, z in atoms_sorted:
            stereo = atom_stereo_cfg.get(serial, "N")
            chg = atom_charge.get(serial, 0)
            out.write(f"{compound_id_clean} {name} {element} {chg} N {x:.3f} {y:.3f} {z:.3f} {stereo}\n")


        out.write("loop_\n")
        out.write("_chem_comp_bond.atom_id_1\n")
        out.write("_chem_comp_bond.atom_id_2\n")
        out.write("_chem_comp_bond.value_order\n")
        out.write("_chem_comp_bond.pdbx_aromatic_flag\n")

        for (s1, s2), (order, aromatic_flag) in sorted(bond_info.items()):
            if s1 not in serial_to_name or s2 not in serial_to_name:
                continue
            out.write(f"{serial_to_name[s1]} {serial_to_name[s2]} {order} {aromatic_flag}\n")
        out.write("#\n")

    print(f"✅ Converted {ligand_pdb} → {output_path} (AF3-ready CIF with bond order + aromaticity)")
    return output_path
