#!/usr/bin/env python3
"""
auto_covalent_constraints.py

Generate Rosetta constraints for a covalent linkage by:
- reading a reference PDB (protein + ligand as present in the refined structure)
- reading a "raw ligand" file (CIF/PDB/SDF) from a ligand cache to recover "true" atom names
- mapping reference-PDB ligand atom names -> true ligand atom names
- computing distance / angles / dihedrals around the covalent bond
- emitting Rosetta constraints (AtomPair, Angle, Dihedral)

Dependencies:
  pip install gemmi numpy rdkit-pypi

Notes:
- Best mapping happens when raw ligand is CIF or PDB that includes the "true" atom names (e.g. S1, C30).
- SDF often lacks stable atom names; we try to use available properties but may fall back to element+index labels.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import gemmi
import re
from af3_pipeline.config import cfg
import sys

# RDKit is used only for robust connectivity / matching if possible
from rdkit import Chem
from rdkit.Chem import rdFMCS


def linuxize_path(p: Path | str) -> Path:
    """
    Convert a possibly-Windows path to a POSIX/WSL path if needed.
    If the path is already POSIX (/home/...), return as-is.
    """
    p = Path(p)

    # Already looks like a linux path
    if str(p).startswith("/"):
        return p

    # If Windows path like C:\Users\olive\..., try to map via config
    linux_home = cfg.get("linux_home_root")
    if linux_home and p.drive:
        # strip "C:\" and replace with /home/olive
        rel = p.relative_to(p.anchor)
        return Path(linux_home) / rel

    # fallback: return original
    return p

def wsl_unc_path(posix_path: str, distro: str) -> Path:
    """
    Convert /home/olive/... -> \\wsl.localhost\\<distro>\\home\\olive\\...
    so Windows Python can write into the WSL filesystem.
    """
    p = str(PurePosixPath(posix_path))
    p = p.lstrip("/")  # remove leading slash for UNC join
    return Path(rf"\\wsl.localhost\{distro}\{p}")

def write_text_smart(path_str: str, text: str) -> None:
    """
    Write to a POSIX path even when running on Windows Python:
      - If path starts with '/', write via WSL UNC mapping
      - Else, write normally
    """
    if str(path_str).startswith("/") and sys.platform.startswith("win"):
        from af3_pipeline.config import cfg
        distro = str(cfg.get("wsl_distro", "Ubuntu-22.04"))
        outp = wsl_unc_path(path_str, distro)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        return

    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

# ------------------------- geometry helpers -------------------------

def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # angle ABC (vertex at B)
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-12 or nbc < 1e-12:
        return float("nan")
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

def dihedral_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    # standard torsion angle
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # normals
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    if n1n < 1e-12 or n2n < 1e-12:
        return float("nan")

    n1 /= n1n
    n2 /= n2n
    b2n = np.linalg.norm(b2)
    if b2n < 1e-12:
        return float("nan")
    b2u = b2 / b2n

    m1 = np.cross(n1, b2u)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    ang = math.degrees(math.atan2(y, x))
    return float(ang)


# ------------------------- file picking -------------------------

def pick_raw_ligand_file(cache_path: Path, ligand_name: str) -> Optional[Path]:
    """
    cache_path can be a directory or a single file.
    Preference: CIF > PDB > SDF
    We try to pick something that contains ligand_name in filename; otherwise pick any.
    """
    if cache_path.is_file():
        return cache_path

    if not cache_path.is_dir():
        return None

    exts = [".cif", ".mmcif", ".pdb", ".ent", ".sdf", ".mol"]
    files = [p for p in cache_path.rglob("*") if p.suffix.lower() in exts]

    if not files:
        return None

    # Prefer name matches
    name_hits = [p for p in files if ligand_name.lower() in p.name.lower()]
    cand = name_hits if name_hits else files

    # Prefer CIF then PDB then SDF/MOL
    def _rank(p: Path) -> int:
        s = p.suffix.lower()
        if s in [".cif", ".mmcif"]:
            return 0
        if s in [".pdb", ".ent"]:
            return 1
        if s in [".sdf", ".mol"]:
            return 2
        return 9

    cand.sort(key=_rank)
    return cand[0]

def _sanitize_folder_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        raise SystemExit("--out must be a non-empty folder name")
    # keep safe chars
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._-") or "constraints"


def get_constraints_root() -> Path:
    root = cfg.get("paths.constraints_root")
    if not root:
        raise SystemExit("Config missing: paths.constraints_root")
    # Force POSIX even if running on Windows
    return Path(PurePosixPath(str(root)))

def resolve_ligand_cache_dir(ligand_key: str) -> Path:
    """
    Resolve a ligand cache key to the on-disk cache directory.

    Assumes your cache layout is:
      <cache_root>/ligands/<ligand_key>/

    cache_root comes from config (already in your YAML: cache_root: ...).
    """
    ligand_key = (ligand_key or "").strip()
    if not ligand_key:
        raise SystemExit("--ligand-key must be non-empty.")

    cache_root = cfg.get("cache_root")
    if not cache_root:
        raise SystemExit("Missing config key 'cache_root' needed to resolve ligand cache.")

    base = Path(str(cache_root)).expanduser()
    lig_dir = base / "ligands" / ligand_key

    if not lig_dir.exists():
        raise SystemExit(f"Ligand cache dir not found for key '{ligand_key}': {lig_dir}")

    return lig_dir


# ------------------------- read structures -------------------------

def load_reference_pdb(pdb_path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(pdb_path))
    st.setup_entities()
    return st

def iter_residues(st: gemmi.Structure):
    for model in st:
        for chain in model:
            for res in chain:
                yield model, chain, res

def residue_atoms_with_coords(res: gemmi.Residue) -> Dict[str, np.ndarray]:
    out = {}
    for a in res:
        pos = a.pos
        out[a.name.strip()] = np.array([pos.x, pos.y, pos.z], dtype=float)
    return out

def heavy_atoms(coords: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # heuristic: exclude hydrogens by atom name starting with H or digit-H patterns
    out = {}
    for name, xyz in coords.items():
        n = name.strip()
        if not n:
            continue
        if n.startswith("H"):
            continue
        if len(n) >= 2 and n[0].isdigit() and n[1] == "H":
            continue
        out[n] = xyz
    return out


def find_protein_residue(
    st: gemmi.Structure,
    chain_id: str,
    resnum: int,
    restype: str,
) -> gemmi.Residue:
    restype = restype.upper().strip()
    for model in st:
        for chain in model:
            if chain.name.strip() != chain_id.strip():
                continue
            for res in chain:
                if res.seqid.num == resnum and res.name.upper().strip() == restype:
                    return res
    raise SystemExit(f"Could not find protein residue {chain_id} {restype} {resnum} in reference PDB.")


def find_ligand_residue_in_reference(
    st: gemmi.Structure,
    ligand_resname: Optional[str] = None,
) -> gemmi.Residue:
    """
    If ligand_resname is provided: return the first residue whose res.name matches it
    (no heuristics).

    Otherwise: heuristic selection.
    """
    print(ligand_resname)
    if ligand_resname is not None and str(ligand_resname).strip() != "":
        target = str(ligand_resname).upper().strip()
        for model in st:
            for chain in model:
                for res in chain:
                    if res.name.upper().strip() == target:
                        return res
        raise SystemExit(f"Could not find residue with resname '{target}' in reference PDB.")

    # Heuristic mode only when resname not provided
    best = None
    best_score = -1

    for model in st:
        for chain in model:
            for res in chain:
                rname = res.name.upper().strip()
                if rname in {"HOH", "WAT", "SOL", "NA", "CL", "K", "ZN", "MG", "CA"}:
                    continue
                coords = residue_atoms_with_coords(res)
                hvy = heavy_atoms(coords)
                if len(hvy) < 6:
                    continue
                score = len(hvy)
                if score > best_score:
                    best = res
                    best_score = score

    if best is None:
        raise SystemExit("Could not find a ligand-like residue in reference PDB (try --ref-ligand-resname).")
    return best



# ------------------------- raw ligand reading + atom names -------------------------

import gemmi
import numpy as np

def _read_ligand_from_cif_flex(raw_path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    doc = gemmi.cif.read(str(raw_path))

    # Try structure-like CIF first
    try:
        st = gemmi.read_structure(str(raw_path))
        best = None
        best_n = 0
        for model in st:
            for chain in model:
                for res in chain:
                    coords_map = residue_atoms_with_coords(res)
                    hvy = heavy_atoms(coords_map)
                    if len(hvy) > best_n:
                        best = res
                        best_n = len(hvy)

        if best is not None and best_n > 0:
            names = []
            elems = []
            coords = []
            for a in best:
                nm = a.name.strip()
                if nm.startswith("H") or (len(nm) >= 2 and nm[0].isdigit() and nm[1] == "H"):
                    continue
                names.append(nm)
                e = a.element.name.strip() if a.element.name else nm[0].upper()
                elems.append(e.upper())
                pos = a.pos
                coords.append([pos.x, pos.y, pos.z])
            return names, elems, np.array(coords, dtype=float)
    except Exception:
        pass

    # chem_comp_atom table
    block = doc.sole_block()
    atom_id = block.find_values("_chem_comp_atom.atom_id")
    type_symbol = block.find_values("_chem_comp_atom.type_symbol")
    if atom_id:
        x = (block.find_values("_chem_comp_atom.model_Cartn_x") or
             block.find_values("_chem_comp_atom.pdbx_model_Cartn_x_ideal") or
             block.find_values("_chem_comp_atom.x"))
        y = (block.find_values("_chem_comp_atom.model_Cartn_y") or
             block.find_values("_chem_comp_atom.pdbx_model_Cartn_y_ideal") or
             block.find_values("_chem_comp_atom.y"))
        z = (block.find_values("_chem_comp_atom.model_Cartn_z") or
             block.find_values("_chem_comp_atom.pdbx_model_Cartn_z_ideal") or
             block.find_values("_chem_comp_atom.z"))

        names = [a.strip() for a in atom_id]
        elems = [t.strip().upper() for t in type_symbol] if type_symbol else [n[0].upper() for n in names]

        if x and y and z and len(x) == len(names):
            coords = np.array([[float(xi), float(yi), float(zi)] for xi, yi, zi in zip(x, y, z)], dtype=float)
            return names, elems, coords

        coords = np.full((len(names), 3), np.nan, dtype=float)
        return names, elems, coords

    raise SystemExit(f"CIF parsed but no atoms found: {raw_path}")


def read_raw_ligand_atomnames_and_coords(
    raw_path: Path
) -> Tuple[List[str], List[str], np.ndarray, Optional[Chem.Mol]]:
    """
    Read the *cached/raw ligand file* and return:
      - atom_names: list[str]  (best-effort "true" names like S1, C30, etc.)
      - coords: Nx3 array
      - rdkit mol (if parsed, else None)

    NOTE: This function MUST NOT use find_ligand_residue_in_reference().
    It only reads the ligand file itself.
    """

    suf = raw_path.suffix.lower()

    # ---------- CIF / mmCIF ----------
    if suf in [".cif", ".mmcif"]:
        names, elems, coords = _read_ligand_from_cif_flex(raw_path)
        return names, elems, coords, None

    # ---------- PDB ----------
    if suf in [".pdb", ".ent"]:
        st = gemmi.read_structure(str(raw_path))

        # Same heuristic as above
        best = None
        best_natoms = 0

        for model in st:
            for chain in model:
                for res in chain:
                    rname = res.name.strip().upper()
                    if rname in {"HOH","WAT","SOL"}:
                        continue

                    coords_map = residue_atoms_with_coords(res)
                    hvy = heavy_atoms(coords_map)

                    if len(hvy) > best_natoms:
                        best = res
                        best_natoms = len(hvy)

        if best is None:
            raise SystemExit(f"Could not find ligand in cache PDB: {raw_path}")

        coords_map = residue_atoms_with_coords(best)
        hvy = heavy_atoms(coords_map)
        names = list(hvy.keys())
        coords = np.vstack([hvy[n] for n in names])
        elems = [_infer_elem_from_name(n) for n in names]

        pdb_block = gemmi_to_pdb_block_single_residue(best)
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=True)

        return names, elems, coords, mol

    # ---------- SDF / MOL ----------
    if suf in [".sdf", ".mol"]:
        mol = Chem.MolFromMolFile(str(raw_path), sanitize=True, removeHs=False)
        if mol is None:
            raise SystemExit(f"RDKit could not read ligand file: {raw_path}")

        conf = mol.GetConformer()
        coords = []
        names = []
        elems,

        for i, a in enumerate(mol.GetAtoms()):
            p = conf.GetAtomPosition(i)
            coords.append([p.x, p.y, p.z])

            nm = None
            for key in ("atomLabel", "atomName", "_TriposAtomName", "name"):
                if a.HasProp(key):
                    nm = a.GetProp(key).strip()
                    if nm:
                        break

            if not nm:
                nm = f"{a.GetSymbol()}{i+1}"

            names.append(nm)
            elems.append(a.GetSymbol().upper())

        return names, np.array(coords, dtype=float), mol

    raise SystemExit(f"Unsupported ligand file type: {raw_path}")



def gemmi_to_pdb_block_single_residue(res: gemmi.Residue) -> str:
    """
    Make a minimal PDB block RDKit can ingest for a single residue.
    """
    lines = []
    serial = 1
    chain_id = "L"
    resname = res.name.upper().strip()
    resseq = 1

    for a in res:
        pos = a.pos
        name = a.name.strip()
        # Attempt element inference
        element = (a.element.name if a.element.name else name[0]).strip().upper()
        lines.append(
            f"HETATM{serial:5d} {name:<4s} {resname:>3s} {chain_id}{resseq:4d}    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00 20.00          {element:>2s}"
        )
        serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


# ------------------------- mapping raw ligand atoms to reference ligand atoms -------------------------

def map_ligand_atoms_by_mcs_or_coords(
    ref_lig_res: gemmi.Residue,
    raw_names: List[str],
    raw_elems: List[str],
    raw_coords: np.ndarray,
    raw_mol: Optional[Chem.Mol],
) -> Dict[str, str]:
    """
    Produce mapping: raw_atom_name -> ref_pdb_atom_name

    Improved fallback:
      - element-aware
      - stable anchor selection (rarest elements first)
      - global assignment (Hungarian) within each element group
    """

    # ---------------------------
    # Build ref heavy atom lists
    # ---------------------------
    ref_names: List[str] = []
    ref_elems: List[str] = []
    ref_coords_list: List[List[float]] = []

    for a in ref_lig_res:
        nm = a.name.strip()
        if _is_hydrogen_name(nm):
            continue
        ref_names.append(nm)

        # gemmi element is best; fallback to first character
        el = a.element.name.strip().upper() if a.element.name else (nm[0].upper() if nm else "X")
        ref_elems.append(el)

        pos = a.pos
        ref_coords_list.append([pos.x, pos.y, pos.z])

    refX = np.asarray(ref_coords_list, dtype=float)

    # ---------------------------
    # Try RDKit MCS (only safe when raw_names == rdkit atom order)
    # ---------------------------
    try:
        if raw_mol is not None:
            ref_pdb_block = gemmi_to_pdb_block_single_residue(ref_lig_res)
            ref_mol = Chem.MolFromPDBBlock(ref_pdb_block, removeHs=False, sanitize=True)
            if ref_mol is not None:
                mcs = rdFMCS.FindMCS([raw_mol, ref_mol], timeout=10, ringMatchesRingOnly=True)
                if mcs and mcs.numAtoms >= min(raw_mol.GetNumAtoms(), ref_mol.GetNumAtoms()) - 2:
                    patt = Chem.MolFromSmarts(mcs.smartsString)
                    raw_match = raw_mol.GetSubstructMatch(patt)
                    ref_match = ref_mol.GetSubstructMatch(patt)
                    if raw_match and ref_match and len(raw_match) == len(ref_match):
                        # Ref names in RDKit order
                        ref_rdkit_names = []
                        for i in range(ref_mol.GetNumAtoms()):
                            info = ref_mol.GetAtomWithIdx(i).GetPDBResidueInfo()
                            ref_rdkit_names.append(info.GetName().strip() if info else f"X{i+1}")

                        # Raw names are assumed RDKit-order ONLY for SDF/MOL (you already do this upstream)
                        mapping = {}
                        for raw_i, ref_i in zip(raw_match, ref_match):
                            if raw_i >= len(raw_names):
                                continue
                            raw_nm = raw_names[raw_i]
                            ref_nm = ref_rdkit_names[ref_i]
                            if ref_nm in set(ref_names):
                                mapping[raw_nm] = ref_nm
                        if mapping:
                            return mapping
    except Exception:
        pass

    # ---------------------------
    # Fallback: element-aware alignment + global assignment
    # ---------------------------
    rawX = np.asarray(raw_coords, dtype=float)
    if rawX.ndim != 2 or rawX.shape[1] != 3:
        raise SystemExit("raw_coords must be Nx3")

    N, M = rawX.shape[0], refX.shape[0]
    if N == 0 or M == 0:
        raise SystemExit("No ligand atoms found for mapping.")

    # Infer raw elements from names (good enough for S1/C45/N3/etc)

    # 1) Choose anchors: rare elements first (S, P, Halogens, then O/N, then C)
    anchors_raw, anchors_ref = _pick_anchor_pairs_by_element(
        raw_names, raw_elems, rawX,
        ref_names, ref_elems, refX,
        max_anchors=12
    )

    # If we couldn't find any anchors (should be rare), fall back to using all atoms as "anchors"
    if len(anchors_raw) < 3 or len(anchors_ref) < 3:
        anchors_raw = np.arange(N)
        anchors_ref = np.arange(M)

    # 2) Compute rigid alignment raw->ref using anchors (Kabsch)
    R, t = _kabsch_fit(rawX[anchors_raw], refX[anchors_ref])
    raw_aligned = (rawX @ R) + t

    # 3) Global assignment within each element group (Hungarian)
    mapping: Dict[str, str] = {}
    used_ref = set()

    # process element groups in "hardest" order first (rarest in ref)
    elem_order = _sorted_elements_by_rarity(ref_elems)

    for el in elem_order:
        raw_idx = [i for i in range(N) if raw_elems[i] == el]
        ref_idx = [j for j in range(M) if ref_elems[j] == el and j not in used_ref]
        if not raw_idx or not ref_idx:
            continue

        # cost matrix: distances after alignment
        cost = np.zeros((len(raw_idx), len(ref_idx)), dtype=float)
        for ii, i in enumerate(raw_idx):
            di = raw_aligned[i]
            for jj, j in enumerate(ref_idx):
                cost[ii, jj] = float(np.linalg.norm(refX[j] - di))

        assign = _hungarian_min_cost(cost)  # list of (row, col)
        for r, c in assign:
            i = raw_idx[r]
            j = ref_idx[c]
            mapping[raw_names[i]] = ref_names[j]
            used_ref.add(j)

    # 4) If any raw atoms remain unmapped, map by nearest unused ref (same element if possible)
    for i in range(N):
        if raw_names[i] in mapping:
            continue
        el = raw_elems[i]
        candidates = [j for j in range(M) if j not in used_ref and ref_elems[j] == el]
        if not candidates:
            candidates = [j for j in range(M) if j not in used_ref]
        if not candidates:
            break
        dists = [(float(np.linalg.norm(refX[j] - raw_aligned[i])), j) for j in candidates]
        dists.sort()
        j = dists[0][1]
        mapping[raw_names[i]] = ref_names[j]
        used_ref.add(j)

    return mapping


# ------------------------- helpers -------------------------

def _is_hydrogen_name(nm: str) -> bool:
    nm = (nm or "").strip()
    if not nm:
        return False
    if nm.startswith("H"):
        return True
    if len(nm) >= 2 and nm[0].isdigit() and nm[1] == "H":
        return True
    return False

def _infer_elem_from_name(nm: str) -> str:
    nm = (nm or "").strip()
    if not nm:
        return "X"
    # common: "CL1", "BR2", "NA", etc
    up = nm.upper()
    # handle two-letter elements
    for two in ("CL", "BR", "NA", "MG", "ZN", "FE", "SI", "SE", "LI", "AL", "CA", "CU", "MN", "CO", "NI", "CD", "HG", "PB", "SN", "AG", "AU", "PT"):
        if up.startswith(two):
            return two
    return up[0]

def _sorted_elements_by_rarity(ref_elems: List[str]) -> List[str]:
    # rarest first; ties by custom chemistry preference (S/P/halogens before O/N before C)
    from collections import Counter
    c = Counter(ref_elems)
    priority = {"S": 0, "P": 1, "F": 2, "CL": 3, "BR": 4, "I": 5, "O": 6, "N": 7, "C": 9}
    def key(el: str):
        return (c.get(el, 10**9), priority.get(el, 50), el)
    return sorted(set(ref_elems), key=key)

def _pick_anchor_pairs_by_element(
    raw_names: List[str], raw_elems: List[str], rawX: np.ndarray,
    ref_names: List[str], ref_elems: List[str], refX: np.ndarray,
    max_anchors: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick anchors by matching element types; for each element, pick up to a few atoms closest
    to their respective centroids (stable-ish) but *within element*.
    Returns (raw_idx, ref_idx) arrays of equal length.
    """
    raw_idx_all = []
    ref_idx_all = []

    elem_order = _sorted_elements_by_rarity(ref_elems)

    raw_cent = rawX.mean(axis=0)
    ref_cent = refX.mean(axis=0)

    for el in elem_order:
        raw_idx = [i for i in range(len(raw_names)) if raw_elems[i] == el]
        ref_idx = [j for j in range(len(ref_names)) if ref_elems[j] == el]
        if not raw_idx or not ref_idx:
            continue

        # choose up to n anchors from this element group
        n = min(len(raw_idx), len(ref_idx), 3)  # at most 3 per element group
        # pick closest-to-centroid *within the element* (reduces symmetry chaos a bit)
        raw_sorted = sorted(raw_idx, key=lambda i: float(np.linalg.norm(rawX[i] - raw_cent)))
        ref_sorted = sorted(ref_idx, key=lambda j: float(np.linalg.norm(refX[j] - ref_cent)))

        raw_idx_all.extend(raw_sorted[:n])
        ref_idx_all.extend(ref_sorted[:n])

        if len(raw_idx_all) >= max_anchors:
            break

    k = min(len(raw_idx_all), len(ref_idx_all))
    return np.asarray(raw_idx_all[:k], dtype=int), np.asarray(ref_idx_all[:k], dtype=int)

def _kabsch_fit(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find R, t that minimizes ||(P R + t) - Q|| (least squares).
    P, Q are (k,3) with k>=3.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Q.mean(axis=0) - (P.mean(axis=0) @ R)
    return R, t

def _hungarian_min_cost(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Hungarian algorithm (minimum cost assignment) for a rectangular matrix.
    Returns assignments for min(n_rows, n_cols) pairs.

    NOTE: pure python implementation; OK for ligand sizes (<~200).
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    transposed = False
    if n > m:
        cost = cost.T
        n, m = cost.shape
        transposed = True

    # potentials
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] = matched row for column j
    assignment = []
    for j in range(1, m + 1):
        i = p[j]
        if i != 0 and i <= n:
            assignment.append((i - 1, j - 1))

    if transposed:
        # we solved on cost.T => swap indices
        assignment = [(c, r) for (r, c) in assignment]

    return assignment

def find_closest_atom(
    coords_map: Dict[str, np.ndarray],
    target_xyz: np.ndarray,
    cutoff: float = 3.0,
) -> Tuple[Optional[str], float]:
    """
    Return (atom_name, distance) for the closest atom in coords_map to target_xyz,
    restricted to atoms within cutoff. If none within cutoff, returns (None, inf).
    """
    best_name = None
    best_d = float("inf")
    for name, xyz in coords_map.items():
        d = float(np.linalg.norm(xyz - target_xyz))
        if d < best_d:
            best_d = d
            best_name = name
    if best_d <= cutoff:
        return best_name, best_d
    return None, best_d


# ------------------------- neighbor picking -------------------------

def pick_preferred_ligand_parent(
    coords_map: Dict[str, np.ndarray],
    anchor: str,
) -> Optional[str]:
    """
    Prefer a CARBON neighbor of the anchor atom if one exists.
    If none exists, fall back to nearest heavy atom.
    """

    if anchor not in coords_map:
        return None

    hvy = heavy_atoms(coords_map)
    base = hvy[anchor]

    # --- Step 1: collect neighbors sorted by distance ---
    candidates = []
    for nm, xyz in hvy.items():
        if nm == anchor:
            continue
        d = distance(base, xyz)
        candidates.append((d, nm))

    candidates.sort(key=lambda x: x[0])  # nearest first

    # --- Step 2: try to find a carbon among close neighbors ---
    for d, nm in candidates:
        if nm.startswith("C"):   # <-- key preference
            return nm

    # --- Step 3: fallback to nearest heavy atom
    return candidates[0][1] if candidates else None

def pick_nearest_heavy_neighbor_within_residue(
    coords_map: Dict[str, np.ndarray],
    atom_name: str,
    exclude: Optional[set] = None,
) -> Optional[str]:
    atom_name = atom_name.strip()
    if atom_name not in coords_map:
        return None
    hvy = heavy_atoms(coords_map)
    if atom_name not in hvy:
        return None
    exclude = exclude or set()
    base = hvy[atom_name]
    best = None
    best_d = 1e9
    for nm, xyz in hvy.items():
        if nm == atom_name or nm in exclude:
            continue
        d = distance(base, xyz)
        if d < best_d:
            best_d = d
            best = nm
    return best

def pick_second_neighbor(
    coords_map: Dict[str, np.ndarray],
    anchor: str,
    first: str,
) -> Optional[str]:
    exclude = {anchor, first}
    return pick_nearest_heavy_neighbor_within_residue(coords_map, first, exclude=exclude)


# ------------------------- Rosetta constraint formatting -------------------------

def fmt_atompair(a1: str, r1: str, a2: str, r2: str, mean: float, sd: float) -> str:
    return f"AtomPair {a1} {r1} {a2} {r2} HARMONIC {mean:.3f} {sd:.3f}"

def fmt_angle(a1: str, r1: str, a2: str, r2: str, a3: str, r3: str, mean: float, sd: float) -> str:
    return f"Angle {a1} {r1} {a2} {r2} {a3} {r3} HARMONIC {mean:.3f} {sd:.3f}"

def fmt_dihedral(a1: str, r1: str, a2: str, r2: str, a3: str, r3: str, a4: str, r4: str, mean: float, sd: float) -> str:
    return f"Dihedral {a1} {r1} {a2} {r2} {a3} {r3} {a4} {r4} CIRCULARHARMONIC {mean:.3f} {sd:.3f}"

def rosetta_resid(chain: str, resnum: int) -> str:
    return str(resnum)


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    print("DEBUG: script path =", __file__)
    ap.add_argument("--ref-pdb", required=True, type=Path, help="Reference PDB containing protein+ligand")
    ap.add_argument(
        "--ligand-key",
        required=True,
        help="Ligand cache key/id (the dropdown value used in the GUI).",
    )
    ap.add_argument(
        "--ligand-name",
        default="LIG",
        help="Ligand name hint (only used to prefer matching filenames inside the cache).",
    )

    ap.add_argument(
        "--ref-ligand-resname",
        default=None,
        help="Ligand resname in the *reference PDB* (variable; e.g. 2P5, UNK, A5L, etc.). "
            "If omitted, script will guess a ligand-like residue.",
    )
    ap.add_argument("--chain", required=True, help="Protein chain ID for covalent residue (e.g. H)")
    ap.add_argument("--resnum", required=True, type=int, help="Protein residue number (e.g. 94)")
    ap.add_argument("--restype", required=True, help="Protein residue 3-letter (e.g. HIS)")
    ap.add_argument("--protein-atom", required=True, help="Protein atom name that bonds (e.g. ND1)")
    ap.add_argument("--ligand-atom-true", required=True, help="Ligand TRUE atom name (e.g. S1) from raw ligand file/AF view")
    ap.add_argument("--sd-dist", type=float, default=0.10, help="SD for bond distance constraint (A)")
    ap.add_argument("--sd-ang", type=float, default=5.0, help="SD for angle constraints (deg)")
    ap.add_argument("--sd-dih", type=float, default=10.0, help="SD for dihedral constraints (deg)")
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Base name for constraints file (e.g. '2-31-5'). "
            "Will be written as <name>_constraints.cst in constraints_root."
        ),
    )
    args = ap.parse_args()

    st = load_reference_pdb(args.ref_pdb)

    prot_res = find_protein_residue(st, args.chain, args.resnum, args.restype)
    prot_coords = residue_atoms_with_coords(prot_res)
    if args.protein_atom.strip() not in prot_coords:
        raise SystemExit(f"Protein atom '{args.protein_atom}' not found in {args.chain} {args.restype} {args.resnum}.")

    print(f"DEBUG: args.ref_ligand_resname={args.ref_ligand_resname!r}")
    lig_res = find_ligand_residue_in_reference(st, ligand_resname=args.ref_ligand_resname)
    lig_coords = residue_atoms_with_coords(lig_res)

    lig_cache_dir = resolve_ligand_cache_dir(args.ligand_key)

    raw_path = pick_raw_ligand_file(lig_cache_dir, args.ligand_name)
    if raw_path is None:
        raise SystemExit(f"Could not find ligand file under: {lig_cache_dir}")
    raw_names, raw_elems, raw_xyz, raw_mol = read_raw_ligand_atomnames_and_coords(raw_path)




    # Build mapping raw(true)->refPDBname
    mapping_true_to_ref = map_ligand_atoms_by_mcs_or_coords(lig_res, raw_names, raw_elems, raw_xyz, raw_mol)

    # We also want inverse map for reporting
    mapping_ref_to_true = {}
    for true_nm, ref_nm in mapping_true_to_ref.items():
        mapping_ref_to_true.setdefault(ref_nm, true_nm)

    lig_atom_true = args.ligand_atom_true.strip()
    if lig_atom_true not in mapping_true_to_ref:
        # show helpful candidates
        sample = ", ".join(list(mapping_true_to_ref.keys())[:30])
        raise SystemExit(
            f"Could not map true ligand atom '{lig_atom_true}' to reference PDB atom.\n"
            f"Raw ligand atom names seen (sample): {sample}\n"
            f"Try providing a CIF/PDB raw ligand file with stable atom names, or verify --ligand-atom-true."
        )

    # --- Determine covalent ligand atom in ref PDB from the TRUE atom name ---
    prot_atom = args.protein_atom.strip()
    P = prot_coords[prot_atom]

    lig_atom_true = args.ligand_atom_true.strip()

    # Map TRUE -> refPDB atom name
    lig_atom_ref = mapping_true_to_ref.get(lig_atom_true)
    if not lig_atom_ref:
        sample = ", ".join(list(mapping_true_to_ref.keys())[:30])
        raise SystemExit(
            f"Could not map true ligand atom '{lig_atom_true}' to reference PDB atom.\n"
            f"Raw ligand atom names seen (sample): {sample}"
        )

    # Sanity-check distance (but do NOT override)
    lig_xyz = lig_coords.get(lig_atom_ref)
    if lig_xyz is None:
        raise SystemExit(f"Mapped ref ligand atom '{lig_atom_ref}' not found in ref ligand residue atoms.")

    d_PL = distance(P, lig_xyz)
    if d_PL > 3.0:
        # print nearest few candidates to help debug
        # (nearest ref ligand atoms to the protein atom coordinate)
        hvy = heavy_atoms(lig_coords)
        ranked = sorted([(distance(P, xyz), nm) for nm, xyz in hvy.items()])
        top = ", ".join([f"{nm}:{d:.2f}" for d, nm in ranked[:10]])
        raise SystemExit(
            f"Mapped covalent atom looks wrong: {lig_atom_true} -> {lig_atom_ref} at {d_PL:.2f} Å\n"
            f"Nearest ref-ligand atoms to protein {prot_atom}: {top}\n"
            f"Check mapping / ref-ligand selection / raw ligand file."
        )
    print(f"DEBUG: covalent ligand atom by TRUE mapping: {lig_atom_true} -> {lig_atom_ref} at {d_PL:.3f} Å")
    if lig_atom_ref not in lig_coords:
        raise SystemExit(f"Mapped reference ligand atom '{lig_atom_ref}' not found in reference PDB ligand residue.")

    # Pick neighbor atoms automatically
    prot_ref1 = pick_nearest_heavy_neighbor_within_residue(prot_coords, prot_atom)
    if prot_ref1 is None:
        raise SystemExit("Could not pick a neighbor atom on the protein side (check residue atoms).")
    prot_ref2 = pick_second_neighbor(prot_coords, prot_atom, prot_ref1)

    lig_ref1 = pick_preferred_ligand_parent(lig_coords, lig_atom_ref)
    if lig_ref1 is None:
        raise SystemExit("Could not pick a neighbor atom on the ligand side (check ligand atoms).")
    lig_ref2 = pick_second_neighbor(lig_coords, lig_atom_ref, lig_ref1)

    # Coordinates
    P = prot_coords[prot_atom]
    Pr1 = prot_coords[prot_ref1]
    L = lig_coords[lig_atom_ref]
    Lr1 = lig_coords[lig_ref1]

    # Compute values
    d_PL = distance(P, L)
    ang_Pr1_P_L = angle_deg(Pr1, P, L)
    ang_P_L_Lr1 = angle_deg(P, L, Lr1)

    dih_prot = float("nan")
    if prot_ref2 is not None:
        Pr2 = prot_coords[prot_ref2]
        dih_prot = dihedral_deg(Pr2, Pr1, P, L)

    dih_lig = float("nan")
    if lig_ref2 is not None:
        Lr2 = lig_coords[lig_ref2]
        dih_lig = dihedral_deg(P, L, Lr1, Lr2)

    # Rosetta residue IDs
    prot_r = rosetta_resid(args.chain, args.resnum)

    # Ligand resid/chain from the reference PDB residue
    # (Gemmi residue doesn't directly expose chain here; we infer by re-searching)
    lig_chain = None
    lig_resnum = None
    for model in st:
        for chain in model:
            for res in chain:
                if res is lig_res:
                    lig_chain = chain.name.strip()
                    lig_resnum = res.seqid.num
                    break
            if lig_chain:
                break
        if lig_chain:
            break
    if lig_chain is None or lig_resnum is None:
        raise SystemExit("Could not determine ligand chain/resnum in reference PDB.")

    lig_r = rosetta_resid(lig_chain, lig_resnum)

    # Map reference PDB ligand neighbor names back to TRUE names for reporting
    lig_atom_ref_true = mapping_ref_to_true.get(lig_atom_ref, lig_atom_ref)
    lig_ref1_true = mapping_ref_to_true.get(lig_ref1, lig_ref1)
    lig_ref2_true = mapping_ref_to_true.get(lig_ref2, lig_ref2) if lig_ref2 else None

    # Emit constraints
    lines = []
    lines.append(f"# Reference PDB: {args.ref_pdb}")
    lines.append(f"# Raw ligand file: {raw_path}")
    lines.append(f"# Protein covalent: {args.chain} {args.restype.upper()} {args.resnum} {prot_atom}")
    lines.append(f"# Ligand (refPDB): chain {lig_chain} resnum {lig_resnum} resname {lig_res.name}")
    lines.append(f"# Ligand atom TRUE '{lig_atom_true}' -> refPDB '{lig_atom_ref}'")
    lines.append(f"# Auto-picked protein neighbor: {prot_ref1} (and {prot_ref2 if prot_ref2 else 'none'})")
    lines.append(f"# Auto-picked ligand neighbor:  {lig_ref1} (and {lig_ref2 if lig_ref2 else 'none'})")
    lines.append("# ---- Numeric measurements (from reference PDB geometry) ----")
    lines.append(f"# distance({prot_atom}-{lig_atom_ref}) = {d_PL:.3f} Å")
    lines.append(f"# angle({prot_ref1}-{prot_atom}-{lig_atom_ref}) = {ang_Pr1_P_L:.3f} deg")
    lines.append(f"# angle({prot_atom}-{lig_atom_ref}-{lig_ref1}) = {ang_P_L_Lr1:.3f} deg")
    if not math.isnan(dih_prot):
        lines.append(f"# dihedral({prot_ref2}-{prot_ref1}-{prot_atom}-{lig_atom_ref}) = {dih_prot:.3f} deg")
    if not math.isnan(dih_lig):
        lines.append(f"# dihedral({prot_atom}-{lig_atom_ref}-{lig_ref1}-{lig_ref2}) = {dih_lig:.3f} deg")

    # --- Use TRUE ligand atom names in the constraint file ---
    lig_atom_cst = lig_atom_ref_true
    lig_ref1_cst = lig_ref1_true
    lig_ref2_cst = lig_ref2_true if lig_ref2 is not None else None

    lines.append("# ---- Rosetta constraints (ligand atoms use TRUE names) ----")
    lines.append(fmt_atompair(prot_atom, prot_r, lig_atom_cst, lig_r, d_PL, args.sd_dist))
    lines.append(fmt_angle(prot_ref1, prot_r, prot_atom, prot_r, lig_atom_cst, lig_r, ang_Pr1_P_L, args.sd_ang))
    lines.append(fmt_angle(prot_atom, prot_r, lig_atom_cst, lig_r, lig_ref1_cst, lig_r, ang_P_L_Lr1, args.sd_ang))

    if prot_ref2 is not None and not math.isnan(dih_prot):
        lines.append(fmt_dihedral(prot_ref2, prot_r, prot_ref1, prot_r, prot_atom, prot_r, lig_atom_cst, lig_r, dih_prot, args.sd_dih))
    if lig_ref2_cst is not None and not math.isnan(dih_lig):
        lines.append(fmt_dihedral(prot_atom, prot_r, lig_atom_cst, lig_r, lig_ref1_cst, lig_r, lig_ref2_cst, lig_r, dih_lig, args.sd_dih))


    constraints_root = str(cfg.get("paths.constraints_root") or "")
    if not constraints_root:
        raise SystemExit("Config missing: paths.constraints_root")

    # Force POSIX formatting
    constraints_root = str(PurePosixPath(constraints_root))

    base = _sanitize_folder_name(args.out)
    out_file_posix = str(PurePosixPath(constraints_root) / f"{base}_constraints.cst")

    write_text_smart(out_file_posix, "\n".join(lines) + "\n")

    print(f"Wrote constraints file: {out_file_posix}")

    # Console report with TRUE atom names too
    print("=== Mapping + geometry report ===")
    print(f"Reference PDB: {args.ref_pdb}")
    print(f"Raw ligand file: {raw_path}")
    print(f"Protein: {args.chain} {args.restype.upper()} {args.resnum}  atom {prot_atom}")
    print(f"Ligand in ref PDB: chain {lig_chain} resnum {lig_resnum} resname {lig_res.name}")
    print(f"Ligand atom TRUE '{lig_atom_true}' -> refPDB '{lig_atom_ref}'")
    print(f"Auto ligand neighbor refPDB '{lig_ref1}' (TRUE '{lig_ref1_true}')")
    if lig_ref2:
        print(f"Auto ligand 2nd neighbor refPDB '{lig_ref2}' (TRUE '{lig_ref2_true}')")
    print(f"Auto protein neighbor: {prot_ref1} (2nd: {prot_ref2 if prot_ref2 else 'none'})")
    print("")
    print(f"Bond distance: {d_PL:.3f} Å")
    print(f"Angle (prot_ref - prot - lig): {ang_Pr1_P_L:.3f} deg")
    print(f"Angle (prot - lig - lig_ref): {ang_P_L_Lr1:.3f} deg")
    if not math.isnan(dih_prot):
        print(f"Dihedral (prot_ref2-prot_ref-prot-lig): {dih_prot:.3f} deg")
    if not math.isnan(dih_lig):
        print(f"Dihedral (prot-lig-lig_ref-lig_ref2): {dih_lig:.3f} deg")
    print("")
    print("=== Done ===")


if __name__ == "__main__":
    main()
