#!/usr/bin/env python3
"""
compare_structs.py (GUI compatible)

Supports:
  1) Normal Python execution (GUI): uses pymol2 embedded interpreter if available.
     python -m af3_pipeline.analysis.compare_structs --pdb1 ... --out results.json

  2) Running inside PyMOL (old style): `from pymol import cmd` already exists.

Notes:
- If pymol2 isn't available, you can still run via external PyMOL:
    pymol -cq compare_structs.py -- --pdb1 ... --out results.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def _default_out_path(pdb2: str) -> str:
    """
    Default output JSON path: same directory as query pdb (pdb2),
    filename: <query_stem>_compare.json
    """
    p2 = Path(pdb2)
    stem = p2.stem
    return str(p2.with_name(f"{stem}_compare.json"))

# ----------------------------
# PyMOL access (embedded or in-PyMOL)
# ----------------------------

def _with_pymol_cmd():
    """
    Context manager yielding a `cmd` object.
    - If running in PyMOL, imports `from pymol import cmd` and yields it (no context).
    - Otherwise tries pymol2 (embedded), yields cmd, then stops cleanly.
    """
    try:
        from pymol import cmd  # type: ignore
        # Running inside PyMOL already
        class _NoCtx:
            def __enter__(self): return cmd
            def __exit__(self, *exc): return False
        return _NoCtx()
    except Exception:
        pass

    try:
        import pymol2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyMOL Python API not available. Install pymol2 (comes with many PyMOL builds), "
            "or run using external PyMOL:\n"
            "  pymol -cq compare_structs.py -- --pdb1 ...\n"
            f"Original import error: {e}"
        )

    class _PyMOL2Ctx:
        def __init__(self):
            self.p = None
        def __enter__(self):
            self.p = pymol2.PyMOL()
            self.p.start()
            return self.p.cmd
        def __exit__(self, *exc):
            try:
                if self.p is not None:
                    self.p.stop()
            except Exception:
                pass
            return False

    return _PyMOL2Ctx()


# ----------------------------
# Helpers: sequences & residues
# ----------------------------

@dataclass
class ResidueID:
    chain: str
    resi: str
    resn: str


def _list_chains(cmd, obj: str) -> List[str]:
    return cmd.get_chains(obj)

def _select_name(cmd, name: str) -> str:
    cmd.delete(name)
    return name

def _get_ca_residue_list(cmd, obj: str, chain: str) -> List[ResidueID]:
    sel = f"({obj} and chain {chain} and polymer.protein and name CA)"
    model = cmd.get_model(sel)
    residues = []
    seen = set()
    for a in model.atom:
        key = (a.chain, a.resi, a.resn)
        if key in seen:
            continue
        seen.add(key)
        residues.append(ResidueID(chain=a.chain, resi=a.resi, resn=a.resn))
    return residues

def _get_sequence(cmd, obj: str, chain: str) -> str:
    fasta = cmd.get_fastastr(f"({obj} and chain {chain} and polymer.protein)")
    lines = [ln.strip() for ln in fasta.splitlines() if ln.strip()]
    seq = "".join([ln for ln in lines if not ln.startswith(">")])
    return seq


# ----------------------------
# Simple global alignment (Needleman-Wunsch)
# ----------------------------

def _global_align(a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    tb = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap
        tb[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        tb[0][j] = 2

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            s_diag = dp[i - 1][j - 1] + (match if ai == bj else mismatch)
            s_up   = dp[i - 1][j] + gap
            s_left = dp[i][j - 1] + gap

            best = s_diag
            move = 0
            if s_up > best:
                best = s_up
                move = 1
            if s_left > best:
                best = s_left
                move = 2

            dp[i][j] = best
            tb[i][j] = move

    i, j = n, m
    a_aln = []
    b_aln = []
    while i > 0 or j > 0:
        move = tb[i][j] if i >= 0 and j >= 0 else 0
        if i > 0 and j > 0 and move == 0:
            a_aln.append(a[i - 1])
            b_aln.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            a_aln.append(a[i - 1])
            b_aln.append("-")
            i -= 1
        else:
            a_aln.append("-")
            b_aln.append(b[j - 1])
            j -= 1

    return "".join(reversed(a_aln)), "".join(reversed(b_aln))

def _build_residue_mapping(
    residues1: List[ResidueID],
    seq1: str,
    residues2: List[ResidueID],
    seq2: str
) -> Dict[Tuple[str, str], Tuple[str, str]]:
    a_aln, b_aln = _global_align(seq1, seq2)
    i1 = 0
    i2 = 0
    mapping = {}
    for aa, bb in zip(a_aln, b_aln):
        if aa != "-" and bb != "-":
            if i1 < len(residues1) and i2 < len(residues2):
                r1 = residues1[i1]
                r2 = residues2[i2]
                mapping[(r1.chain, r1.resi)] = (r2.chain, r2.resi)
            i1 += 1
            i2 += 1
        elif aa != "-" and bb == "-":
            i1 += 1
        elif aa == "-" and bb != "-":
            i2 += 1
    return mapping


# ----------------------------
# Geometry helpers (RMSD)
# ----------------------------

def _rmsd(coordsA: List[Tuple[float,float,float]], coordsB: List[Tuple[float,float,float]]) -> float:
    if len(coordsA) != len(coordsB) or len(coordsA) == 0:
        return float("nan")
    s = 0.0
    n = len(coordsA)
    for (ax,ay,az), (bx,by,bz) in zip(coordsA, coordsB):
        dx, dy, dz = ax-bx, ay-by, az-bz
        s += dx*dx + dy*dy + dz*dz
    return math.sqrt(s / n)

def _centroid(coords: List[Tuple[float,float,float]]) -> Tuple[float,float,float]:
    if not coords:
        return (float("nan"), float("nan"), float("nan"))
    x = sum(c[0] for c in coords) / len(coords)
    y = sum(c[1] for c in coords) / len(coords)
    z = sum(c[2] for c in coords) / len(coords)
    return (x,y,z)

def _dist(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


# ----------------------------
# Ligand handling + RDKit mapping
# ----------------------------

_COV_RAD = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "B": 0.85, "SI": 1.11,
}
def _guess_bond(a_el: str, b_el: str, d: float) -> bool:
    ra = _COV_RAD.get(a_el.upper(), 0.77)
    rb = _COV_RAD.get(b_el.upper(), 0.77)
    return d <= (ra + rb + 0.45)

def _get_ligand_atoms(cmd, obj: str, lig_resn: str) -> List[Tuple[str, Tuple[float,float,float]]]:
    sel = f"({obj} and resn {lig_resn} and not hydrogens)"
    model = cmd.get_model(sel)
    out = []
    for a in model.atom:
        el = (a.symbol or a.name[0]).upper()
        out.append((el, (a.coord[0], a.coord[1], a.coord[2])))
    return out

def _ligand_centroid_metrics(cmd, obj1: str, lig1: str, obj2: str, lig2: str) -> Dict[str, float]:
    a = [c for _, c in _get_ligand_atoms(cmd, obj1, lig1)]
    b = [c for _, c in _get_ligand_atoms(cmd, obj2, lig2)]
    ca = _centroid(a)
    cb = _centroid(b)
    return {"ligand_centroid_distance": _dist(ca, cb)}

def _ligand_rmsd_rdkit(cmd, obj1: str, lig1: str, obj2: str, lig2: str) -> Tuple[float, Dict[str, Any]]:
    details: Dict[str, Any] = {"method": None, "mapped_atoms": 0, "note": ""}

    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdFMCS  # type: ignore
    except Exception as e:
        details["note"] = f"RDKit not available: {e}"
        return float("nan"), details

    atomsA = _get_ligand_atoms(cmd, obj1, lig1)
    atomsB = _get_ligand_atoms(cmd, obj2, lig2)
    if not atomsA or not atomsB:
        details["note"] = "Missing ligand atoms in one or both structures."
        return float("nan"), details

    def build_mol(atoms):
        rw = Chem.RWMol()
        for el, _ in atoms:
            a = Chem.Atom(el.capitalize() if el != "CL" else "Cl")
            rw.AddAtom(a)

        coords = [c for _, c in atoms]
        els = [el for el, _ in atoms]
        n = len(coords)
        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i+1, n):
                xj, yj, zj = coords[j]
                d = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if _guess_bond(els[i], els[j], d):
                    rw.AddBond(i, j, Chem.BondType.SINGLE)

        m = rw.GetMol()
        conf = Chem.Conformer(m.GetNumAtoms())
        for i, (_, (x,y,z)) in enumerate(atoms):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        m.AddConformer(conf, assignId=True)
        try:
            Chem.SanitizeMol(m)
        except Exception:
            pass
        return m

    mA = build_mol(atomsA)
    mB = build_mol(atomsB)

    mcs = rdFMCS.FindMCS(
        [mA, mB],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=10,
    )
    if not mcs.smartsString:
        details["note"] = "MCS failed to find a common substructure."
        return float("nan"), details

    patt = Chem.MolFromSmarts(mcs.smartsString)
    matchA = mA.GetSubstructMatch(patt)
    matchB = mB.GetSubstructMatch(patt)
    if not matchA or not matchB or len(matchA) != len(matchB):
        details["note"] = "Substructure match failed after MCS."
        return float("nan"), details

    coordsA = [mA.GetConformer().GetAtomPosition(i) for i in matchA]
    coordsB = [mB.GetConformer().GetAtomPosition(i) for i in matchB]
    rms = _rmsd([(p.x,p.y,p.z) for p in coordsA], [(p.x,p.y,p.z) for p in coordsB])

    details["method"] = "RDKit_MCS_on_guessed_bonds"
    details["mapped_atoms"] = len(matchA)
    return rms, details


# ----------------------------
# Best chain pairing + alignment
# ----------------------------

def _try_align_chain_pair(cmd, obj1: str, ch1: str, obj2: str, ch2: str) -> float:
    mobile = f"({obj2} and chain {ch2} and polymer.protein and backbone)"
    target = f"({obj1} and chain {ch1} and polymer.protein and backbone)"
    out = cmd.align(mobile, target, cycles=2, transform=0)
    return float(out[0]) if isinstance(out, (list, tuple)) else float(out)

def _pick_best_chain_pair(cmd, obj1: str, obj2: str) -> Tuple[str, str, float]:
    chains1 = _list_chains(cmd, obj1)
    chains2 = _list_chains(cmd, obj2)
    best = (None, None, float("inf"))
    for c1 in chains1:
        if cmd.count_atoms(f"({obj1} and chain {c1} and polymer.protein and name CA)") < 20:
            continue
        for c2 in chains2:
            if cmd.count_atoms(f"({obj2} and chain {c2} and polymer.protein and name CA)") < 20:
                continue
            try:
                rms = _try_align_chain_pair(cmd, obj1, c1, obj2, c2)
                if rms < best[2]:
                    best = (c1, c2, rms)
            except Exception:
                continue
    if best[0] is None:
        raise RuntimeError("Could not find a viable protein chain pair to align.")
    return best[0], best[1], best[2]

def _apply_alignment(cmd, obj1: str, ch1: str, obj2: str, ch2: str) -> float:
    mobile = f"({obj2} and chain {ch2} and polymer.protein and backbone)"
    target = f"({obj1} and chain {ch1} and polymer.protein and backbone)"
    out = cmd.align(mobile, target, cycles=2, transform=1)
    return float(out[0]) if isinstance(out, (list, tuple)) else float(out)


# ----------------------------
# Pocket selection & pocket RMSD
# ----------------------------

def _pocket_residues(cmd, obj: str, chain: str, lig_resn: str, cutoff: float) -> List[ResidueID]:
    lig_sel = _select_name(cmd, f"lig_{obj}")
    cmd.select(lig_sel, f"({obj} and resn {lig_resn})")

    pocket_sel = _select_name(cmd, f"pocket_{obj}")
    expr = (
        f"byres ("
        f"({obj} and polymer.protein and chain {chain}) "
        f"within {float(cutoff):.3f} of ({lig_sel})"
        f")"
    )
    cmd.select(pocket_sel, expr)

    model = cmd.get_model(f"({pocket_sel} and name CA)")
    seen = set()
    out = []
    for a in model.atom:
        key = (a.chain, a.resi, a.resn)
        if key in seen:
            continue
        seen.add(key)
        out.append(ResidueID(chain=a.chain, resi=a.resi, resn=a.resn))
    return out

def _pocket_ca_rmsd(
    cmd,
    obj1: str, chain1: str, pocket1: List[ResidueID],
    obj2: str, chain2: str, map12: Dict[Tuple[str,str], Tuple[str,str]]
) -> Tuple[float, List[Dict[str, Any]]]:
    coordsA = []
    coordsB = []
    used: List[Dict[str, Any]] = []

    for r1 in pocket1:
        k = (r1.chain, r1.resi)
        if k not in map12:
            continue
        ch2, resi2 = map12[k]
        if ch2 != chain2:
            continue

        selA = f"({obj1} and chain {r1.chain} and resi {r1.resi} and name CA)"
        selB = f"({obj2} and chain {ch2} and resi {resi2} and name CA)"
        if cmd.count_atoms(selA) != 1 or cmd.count_atoms(selB) != 1:
            continue

        a = cmd.get_atom_coords(selA)
        b = cmd.get_atom_coords(selB)
        coordsA.append(tuple(a))
        coordsB.append(tuple(b))

        used.append({
            "pdb1": {"chain": r1.chain, "resi": r1.resi, "resn": r1.resn},
            "pdb2": {"chain": ch2, "resi": resi2},
        })

    return _rmsd(coordsA, coordsB), used


# ----------------------------
# Public API for GUI
# ----------------------------

def compare_structures(
    pdb1: str,
    lig1: str,
    pdb2: str,
    lig2: str,
    cutoff: float = 5.0,
) -> Dict[str, Any]:
    pdb1 = str(Path(pdb1))
    pdb2 = str(Path(pdb2))

    with _with_pymol_cmd() as cmd:
        cmd.reinitialize()

        cmd.load(pdb1, "pdb1")
        cmd.load(pdb2, "pdb2")

        ch1, ch2, rms_pre = _pick_best_chain_pair(cmd, "pdb1", "pdb2")
        rms_align = _apply_alignment(cmd, "pdb1", ch1, "pdb2", ch2)

        lig_rmsd, lig_details = _ligand_rmsd_rdkit(cmd, "pdb1", lig1, "pdb2", lig2)
        centroid_metrics = _ligand_centroid_metrics(cmd, "pdb1", lig1, "pdb2", lig2)

        seq1 = _get_sequence(cmd, "pdb1", ch1)
        seq2 = _get_sequence(cmd, "pdb2", ch2)
        residues1 = _get_ca_residue_list(cmd, "pdb1", ch1)
        residues2 = _get_ca_residue_list(cmd, "pdb2", ch2)
        map12 = _build_residue_mapping(residues1, seq1, residues2, seq2)

        pocket1 = _pocket_residues(cmd, "pdb1", ch1, lig1, cutoff)
        pocket2 = _pocket_residues(cmd, "pdb2", ch2, lig2, cutoff)
        pocket_rmsd, used = _pocket_ca_rmsd(cmd, "pdb1", ch1, pocket1, "pdb2", ch2, map12)

        cmd.select("prot1_sel", "pdb1 and polymer.protein")
        cmd.select("prot2_sel", "pdb2 and polymer.protein")
        cmd.select("lig1_sel", f"pdb1 and resn {lig1}")
        cmd.select("lig2_sel", f"pdb2 and resn {lig2}")
        contacts1 = cmd.count_atoms(f"prot1_sel within {cutoff} of lig1_sel")
        contacts2 = cmd.count_atoms(f"prot2_sel within {cutoff} of lig2_sel")

    return {
        "inputs": {"pdb1": pdb1, "lig1": lig1, "pdb2": pdb2, "lig2": lig2, "cutoff_A": float(cutoff)},
        "best_chain_pair": {
            "pdb1_chain": ch1,
            "pdb2_chain": ch2,
            "pre_alignment_rmsd_backbone": float(rms_pre),
            "final_alignment_rmsd_backbone": float(rms_align),
        },
        "ligand": {"rmsd_A": float(lig_rmsd), "details": lig_details, **centroid_metrics},
        "pocket": {
            "definition": f"protein residues within {cutoff} Å of ligand",
            "pdb1_pocket_residue_count": int(len(pocket1)),
            "pdb2_pocket_residue_count": int(len(pocket2)),
            "mapped_residue_count_used_for_rmsd": int(len(used)),
            "ca_rmsd_A": float(pocket_rmsd),
            "residues_used": used,
        },
        "contacts": {
            "protein_atoms_within_cutoff_pdb1": int(contacts1),
            "protein_atoms_within_cutoff_pdb2": int(contacts2),
        },
    }

def compare_and_write(
    pdb1: str,
    lig1: str,
    pdb2: str,
    lig2: str,
    cutoff: float,
    out: str | None = None,
):
    if not out:
        out = _default_out_path(pdb2)

    res = compare_structures(pdb1, lig1, pdb2, lig2, cutoff=cutoff)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


# ----------------------------
# CLI
# ----------------------------

def cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb1", required=True)
    ap.add_argument("--lig1", required=True)
    ap.add_argument("--pdb2", required=True)
    ap.add_argument("--lig2", required=True)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = compare_and_write(args.pdb1, args.lig1, args.pdb2, args.lig2, args.cutoff, args.out)
    print(f"Wrote: {Path(args.out) if args.out else _default_out_path(args.pdb2)}")

    # Short console summary (helpful for logs)
    print("\n=== Compare Summary ===")
    bc = res["best_chain_pair"]
    print(f"Best chain pair: pdb1:{bc['pdb1_chain']} <-> pdb2:{bc['pdb2_chain']}")
    print(f"Backbone RMSD after alignment: {bc['final_alignment_rmsd_backbone']:.3f} Å")
    lig = res["ligand"]
    if math.isfinite(lig["rmsd_A"]):
        print(f"Ligand RMSD: {lig['rmsd_A']:.3f} Å  [{lig['details'].get('method')}, n={lig['details'].get('mapped_atoms')}]")
    else:
        print(f"Ligand RMSD: NaN ({lig['details'].get('note','unknown')})")
    print(f"Ligand centroid distance: {lig['ligand_centroid_distance']:.3f} Å")
    print(f"Wrote: {args.out}\n")


if __name__ == "__main__":
    cli_main()
