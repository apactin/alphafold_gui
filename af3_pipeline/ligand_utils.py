#!/usr/bin/env python3
"""
ligand_utils.py 
===================================================

This version assumes the user supplies a *pre-reduced* ligand — i.e. a SMILES
string already adjusted so that, when a covalent bond is added by AlphaFold3,
the ligand valence and geometry remain correct.

All functionalities from previous versions remain:
- SMILES → RDKit Mol → PDB → CIF (AF3-compatible)
- Hash-based caching
- Optional normalization with rdMolStandardize
- PDB validation via RDKit re-read
- CIF atom relabeling for Rosetta compatibility
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
import re
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

try:
    from rdkit.Chem import rdMolStandardize
except ImportError:
    rdMolStandardize = None

# --- Configuration ---
from .config import cfg

# Internal converter: convert_pdb_to_cif(hash_id: str, basename: str)
from .convert_cif import convert_pdb_to_cif

# =========================================
# 📁 Cache (config-driven)
# =========================================
from .cache_utils import compute_hash, get_cache_dir

# =========================================
# 🧾 Defaults (config-driven)
# =========================================
LIGAND_BASENAME = str(cfg.get("ligand_basename", "LIG"))
LIGAND_NAME_DEFAULT = str(cfg.get("ligand_name_default", "LIG"))
LIGAND_PNG_SIZE = tuple(cfg.get("ligand_png_size", (420, 320)))

# =========================================
# 🔄 Canonical SMILES + soft sanitize
# =========================================
def _canonical_smiles(smiles: str) -> str:
    try:
        m = Chem.MolFromSmiles(smiles, sanitize=True)
        if m is None:
            raise ValueError("MolFromSmiles returned None")
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception as e:
        print(f"⚠️ SMILES canonicalization failed ({smiles}): {e}")
        return smiles.strip()

def _report_chiral_centers(mol: Chem.Mol, stage: str):
    """Log chiral centers for debugging stereo retention."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    if centers:
        print(f"🧬 Chiral centers {stage}: {centers}")
    else:
        print(f"🧬 No chiral centers detected {stage}.")
    return centers

def _require_assigned_stereo_or_warn(mol: Chem.Mol) -> None:
    """
    If cfg says require stereo, hard fail. Otherwise warn.
    Useful for fused ring systems where missing stereo makes the 3D ambiguous.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    ch = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    unassigned = [a for a, cfg_ in ch if cfg_ == "?"]
    if not unassigned:
        return

    require = bool(cfg.get("ligand_require_assigned_stereo", False))
    msg = (
        f"Unassigned stereocenters detected at atom indices {unassigned}. "
        "If this ligand has fused/bridged rings, missing junction stereo can yield the wrong geometry. "
        "Provide explicit stereochemistry in the SMILES (e.g., @ / @@, /, \\)."
    )
    if require:
        raise ValueError(msg)
    print(f"⚠️ {msg}")

def _normalize_if_available(mol: Chem.Mol) -> Chem.Mol:
    """
    Optional normalization. To avoid breaking existing behavior, we keep your default:
    Cleanup + Uncharger + LargestFragment.
    You can disable uncharging via config:
      cfg['ligand_keep_charge'] = True
    """
    if rdMolStandardize is None:
        print("ℹ️ rdMolStandardize not available.")
        return mol

    try:
        print("⚙️ Normalizing ligand (charges/tautomers/fragments)…")
        cleaner = rdMolStandardize.Cleanup(mol)

        keep_charge = bool(cfg.get("ligand_keep_charge", False))
        if keep_charge:
            mol2 = cleaner
        else:
            uncharger = rdMolStandardize.Uncharger()
            mol2 = uncharger.uncharge(cleaner)

        lfc = rdMolStandardize.LargestFragmentChooser()
        mol2 = lfc.choose(mol2)

        Chem.AssignStereochemistry(mol2, cleanIt=True, force=True)
        _report_chiral_centers(mol2, "post-standardize")
        return mol2
    except Exception as e:
        print(f"⚠️ Normalization skipped: {e}")
        return mol
    
def _embed_optimize_pick_lowest(
    mol: Chem.Mol,
    *,
    n_confs: int,
    seed: int,
    prune_rms: float,
    max_embed_attempts: int,
    max_mmff_iters: int,
    max_uff_iters: int,
) -> Tuple[Chem.Mol, int, float, List[Tuple[int, float]]]:
    """
    Embed multiple conformers using ETKDGv3, optimize, pick lowest-energy.
    Returns: (mol_with_confs, best_conf_id, best_energy, energies_sorted)
    """
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()

    def _set_if_exists(obj, attr, value):
        if hasattr(obj, attr):
            setattr(obj, attr, value)
        else:
            print(f"ℹ️ ETKDG param not supported in this RDKit: {attr}")

    _set_if_exists(params, "randomSeed", int(seed))
    _set_if_exists(params, "numThreads", int(cfg.get("ligand_rdkit_threads", 0)))
    _set_if_exists(params, "pruneRmsThresh", float(prune_rms))
    _set_if_exists(params, "maxAttempts", int(max_embed_attempts))
    _set_if_exists(params, "useSmallRingTorsions", True)
    _set_if_exists(params, "useMacrocycleTorsions", True)

    print(f"⚙️ Embedding {n_confs} conformers (ETKDGv3)…")
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params))
    if not conf_ids:
        raise ValueError("3D embedding failed: no conformers generated.")

    energies: List[Tuple[int, float]] = []

    # Prefer MMFF94s; fall back to UFF if needed.
    try:
        print("⚙️ Optimizing conformers with MMFF94s…")
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=0,
            maxIters=int(max_mmff_iters),
            mmffVariant="MMFF94s",
        )
        for cid, (status, e) in zip(conf_ids, results):
            if status == 0:
                energies.append((int(cid), float(e)))
    except Exception as e:
        print(f"⚠️ MMFF optimization failed ({e}); falling back to UFF…")
        results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=int(max_uff_iters))
        for cid, (status, e2) in zip(conf_ids, results):
            if status == 0:
                energies.append((int(cid), float(e2)))

    if not energies:
        raise ValueError("Optimization failed: no conformers converged.")

    energies_sorted = sorted(energies, key=lambda x: x[1])
    best_conf, best_E = energies_sorted[0]
    print(f"✅ Lowest-energy conformer: confId={best_conf}, E={best_E:.4f}")
    return mol, best_conf, best_E, energies_sorted

def _write_single_conformer_sdf(mol_with_confs: Chem.Mol, conf_id: int, out_sdf: Path) -> None:
    """
    Write an SDF containing exactly one conformer (conf_id), preserving bond orders/aromaticity.

    Important: Don't reuse the same Conformer object after RemoveAllConformers();
    RDKit 2025+ can throw "Number of atom mismatch". Clone coordinates instead.
    """
    m = Chem.Mol(mol_with_confs)  # copy mol + conformers
    conf_src = m.GetConformer(int(conf_id))

    # Build a fresh conformer with identical coordinates
    conf_new = Chem.Conformer(m.GetNumAtoms())
    conf_new.Set3D(True)
    for i in range(m.GetNumAtoms()):
        pos = conf_src.GetAtomPosition(i)
        conf_new.SetAtomPosition(i, pos)

    # Keep only this new conformer
    m.RemoveAllConformers()
    m.AddConformer(conf_new, assignId=True)

    w = Chem.SDWriter(str(out_sdf))
    w.write(m)
    w.close()

def _write_2d_depiction_png(
    mol: Chem.Mol,
    out_png: Path,
    *,
    size=(1400, 1000),
    prefer_coordgen=True,
    padding=0.1,          # ↓ whitespace
    bond_length=42,        # ↑ molecule size (try 35–60)
    base_font_size=7,     # ↑ labels
    stereo_label_scale=0.9,
    bond_line_width=2,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    m2 = Chem.Mol(mol)
    Chem.AssignStereochemistry(m2, cleanIt=True, force=True)

    # Better 2D coords (CoordGen if available)
    try:
        rdDepictor.SetPreferCoordGen(bool(prefer_coordgen))
        rdDepictor.Compute2DCoords(m2, canonOrient=True)
    except Exception:
        pass

    try:
        pm = rdMolDraw2D.PrepareMolForDrawing(
            m2, kekulize=True, addChiralHs=False, wedgeBonds=True, forceCoords=True
        )
    except Exception:
        pm = m2

    w, h = map(int, size)
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()

    # ---- “fill the canvas” knobs ----
    if hasattr(opts, "padding"):
        opts.padding = float(padding)
    if hasattr(opts, "fixedBondLength"):
        opts.fixedBondLength = float(bond_length)

    # ---- readability knobs ----
    if hasattr(opts, "baseFontSize"):
        opts.baseFontSize = float(base_font_size)
    if hasattr(opts, "bondLineWidth"):
        opts.bondLineWidth = int(bond_line_width)

    opts.addStereoAnnotation = True
    if hasattr(opts, "annotationFontScale"):
        opts.annotationFontScale = float(stereo_label_scale)

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, pm)
    drawer.FinishDrawing()
    out_png.write_bytes(drawer.GetDrawingText())

def _rewrite_svg_viewbox(svg: str, *, xmin: float, ymin: float, xmax: float, ymax: float,
                         out_w: int, out_h: int, pad_px: float = 100.0) -> str:
    # Pad in drawing pixels
    xmin -= pad_px
    ymin -= pad_px
    xmax += pad_px
    ymax += pad_px
    vb_w = max(1.0, xmax - xmin)
    vb_h = max(1.0, ymax - ymin)

    # Replace width/height + viewBox on the <svg ...> tag.
    # RDKit usually emits: <svg ... width='900px' height='650px' viewBox='0 0 900 650'>
    svg = re.sub(r"width='[^']*'", f"width='{out_w}px'", svg, count=1)
    svg = re.sub(r"height='[^']*'", f"height='{out_h}px'", svg, count=1)

    if "viewBox=" in svg:
        svg = re.sub(r"viewBox='[^']*'", f"viewBox='{xmin:.2f} {ymin:.2f} {vb_w:.2f} {vb_h:.2f}'", svg, count=1)
    else:
        # If no viewBox present, inject it into the <svg ...> tag
        svg = svg.replace("<svg ", f"<svg viewBox='{xmin:.2f} {ymin:.2f} {vb_w:.2f} {vb_h:.2f}' ", 1)

    return svg


def _write_2d_depiction_svg(
    mol: Chem.Mol,
    out_svg: Path,
    *,
    size=(900, 650),
    prefer_coordgen=True,
    base_font_size=7,
    stereo_label_scale=0.9,
    bond_line_width=2,
    pad_px=100,   # padding around structure AFTER fitting
) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    m2 = Chem.Mol(mol)
    Chem.AssignStereochemistry(m2, cleanIt=True, force=True)

    try:
        rdDepictor.SetPreferCoordGen(bool(prefer_coordgen))
        rdDepictor.Compute2DCoords(m2, canonOrient=True)
    except Exception:
        pass

    try:
        pm = rdMolDraw2D.PrepareMolForDrawing(
            m2, kekulize=True, addChiralHs=False, wedgeBonds=True, forceCoords=True
        )
    except Exception:
        pm = m2

    out_w, out_h = map(int, size)

    # Draw at a big working resolution to get stable coords for cropping
    work_w, work_h = max(out_w, 2400), max(out_h, 1800)
    drawer = rdMolDraw2D.MolDraw2DSVG(work_w, work_h)
    opts = drawer.drawOptions()

    # Keep these sane; the viewBox tightening will "zoom" everything together
    if hasattr(opts, "baseFontSize"):
        opts.baseFontSize = float(base_font_size)
    if hasattr(opts, "bondLineWidth"):
        opts.bondLineWidth = int(bond_line_width)

    opts.addStereoAnnotation = True
    if hasattr(opts, "annotationFontScale"):
        opts.annotationFontScale = float(stereo_label_scale)

    # Draw
    drawer.DrawMolecule(pm)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Compute tight bbox from atom draw coords (in pixels of work canvas)
    xs, ys = [], []
    for i in range(pm.GetNumAtoms()):
        pt = drawer.GetDrawCoords(i)
        xs.append(pt.x)
        ys.append(pt.y)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Rewrite viewBox + set final output size
    svg = _rewrite_svg_viewbox(
        svg,
        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
        out_w=out_w, out_h=out_h,
        pad_px=float(pad_px),
    )

    out_svg.write_text(svg, encoding="utf-8")



# =========================================
# 🧱 Core ligand builder (no reduction)
# =========================================
def prepare_ligand_from_smiles(
    smiles: str,
    name: str = LIGAND_NAME_DEFAULT,     
    skip_if_cached: bool = True,            
) -> Path:
    """Prepare and cache ligand structure from SMILES (no automatic reduction)."""
    smiles = _canonical_smiles(smiles)
    if not smiles:
        raise ValueError("No SMILES provided for ligand preparation")

    ligand_hash = compute_hash(smiles)   # or compute_hash(canonical_smiles)
    cache_dir = get_cache_dir("ligands", ligand_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(LIGAND_BASENAME)
    pdb_path = cache_dir / f"{LIGAND_BASENAME}.pdb"
    cif_path = cache_dir / f"{LIGAND_BASENAME}.cif"
    sdf_path = cache_dir / f"{LIGAND_BASENAME}.sdf"
    png_path = cache_dir / f"{LIGAND_BASENAME}.png"
    svg_path = cache_dir / f"{LIGAND_BASENAME}.svg"


    make_png = bool(cfg.get("ligand_make_png", True))
    make_svg = bool(cfg.get("ligand_make_svg", True))

    if skip_if_cached and cif_path.exists():
        # If CIF exists but depictions don't (old cache), backfill
        if (make_png and not png_path.exists()) or (make_svg and not svg_path.exists()):
            try:
                mol2d = Chem.MolFromSmiles(smiles, sanitize=True)
                if mol2d:
                    if rdMolStandardize is not None:
                        try:
                            mol2d = _normalize_if_available(mol2d)
                        except Exception as e:
                            print(f"⚠️ rdMolStandardize normalization failed: {e}")

                    Chem.AssignStereochemistry(mol2d, cleanIt=True, force=True)

                    # Use config sizes if you want separate control
                    png_size = tuple(cfg.get("ligand_png_size", list(LIGAND_PNG_SIZE)))
                    svg_size = tuple(cfg.get("ligand_svg_size", list(LIGAND_PNG_SIZE)))

                    if make_svg and not svg_path.exists():
                        _write_2d_depiction_svg(mol2d, svg_path, size=svg_size)
                        print(f"🖼️ Backfilled ligand SVG: {svg_path}")

                    if make_png and not png_path.exists():
                        _write_2d_depiction_png(mol2d, png_path, size=png_size)
                        print(f"🖼️ Backfilled ligand PNG: {png_path}")

            except Exception as e:
                print(f"⚠️ Failed to backfill ligand depictions: {e}")

        print(f"✅ Using cached ligand CIF: {cif_path}")
        return cif_path



    print(f"⚙️ Generating ligand → cache key {ligand_hash}")

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"❌ Failed to parse SMILES: {smiles}")

    if rdMolStandardize is not None:
        try:
            mol = _normalize_if_available(mol)
        except Exception as e:
            print(f"⚠️ rdMolStandardize normalization failed: {e}")
    else:
        print("ℹ️ rdMolStandardize not available.")


    # Standardize valence/stereo
    try:
        print("⚙️ Standardizing valence/stereo…")
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        mol.UpdatePropertyCache(strict=False)
    except Exception as e:
        print(f"⚠️ Core normalization warning: {e}")

    _report_chiral_centers(mol, "post-sanitize")
    _require_assigned_stereo_or_warn(mol)
    # 2D depiction PNG (for GUI)
    make_png = bool(cfg.get("ligand_make_png", True))
    make_svg = bool(cfg.get("ligand_make_svg", True))

    png_size = tuple(cfg.get("ligand_png_size", list(LIGAND_PNG_SIZE)))
    svg_size = tuple(cfg.get("ligand_svg_size", list(LIGAND_PNG_SIZE)))

    # optional depiction tuning knobs (only if your _write_* functions accept them)
    png_kwargs = {
        "padding": float(cfg.get("ligand_depict_padding", 0.02)),
        "bond_length": float(cfg.get("ligand_depict_bond_length", 42)),
        "base_font_size": float(cfg.get("ligand_depict_font_size", 7)),
        "stereo_label_scale": float(cfg.get("ligand_depict_stereo_scale", 0.9)),
        "bond_line_width": int(cfg.get("ligand_depict_bond_line_width", 2)),
    }

    svg_kwargs = {
        "base_font_size": float(cfg.get("ligand_depict_font_size", 7)),
        "stereo_label_scale": float(cfg.get("ligand_depict_stereo_scale", 0.9)),
        "bond_line_width": int(cfg.get("ligand_depict_bond_line_width", 2)),
        "pad_px": float(cfg.get("ligand_depict_pad_px", 100.0)),
    }

    try:
        if make_svg:
            _write_2d_depiction_svg(mol, svg_path, size=svg_size, **svg_kwargs)
            print(f"🖼️ Wrote ligand depiction SVG: {svg_path}")

        if make_png:
            _write_2d_depiction_png(mol, png_path, size=png_size, **png_kwargs)
            print(f"🖼️ Wrote ligand depiction PNG: {png_path}")

    except Exception as e:
        print(f"⚠️ Failed to write ligand depictions: {e}")



    # --- Robust 3D: ensemble → optimize → pick best
    n_confs = int(cfg.get("ligand_nconfs", 200))
    seed = int(cfg.get("ligand_seed", 0))
    prune_rms = float(cfg.get("ligand_prune_rms", 0.25))
    max_embed_attempts = int(cfg.get("ligand_max_embed_attempts", 1000))
    max_mmff_iters = int(cfg.get("ligand_max_mmff_iters", 2000))
    max_uff_iters = int(cfg.get("ligand_max_uff_iters", 5000))

    mol3d, best_conf, best_E, energies_sorted = _embed_optimize_pick_lowest(
        mol,
        n_confs=n_confs,
        seed=seed,
        prune_rms=prune_rms,
        max_embed_attempts=max_embed_attempts,
        max_mmff_iters=max_mmff_iters,
        max_uff_iters=max_uff_iters,
    )

    _report_chiral_centers(mol3d, "after 3D embedding")

    # Write PDB for downstream naming/coords expectations
    Chem.MolToPDBFile(mol3d, str(pdb_path), confId=int(best_conf))
    print(f"💾 Wrote PDB: {pdb_path}")

    # Write SDF sidecar (bond order/aromaticity preserved) for CIF conversion
    try:
        _write_single_conformer_sdf(mol3d, int(best_conf), sdf_path)
        print(f"💾 Wrote SDF (chemistry source for CIF): {sdf_path}")
    except Exception as e:
        print(f"⚠️ Failed to write SDF sidecar (converter will fall back to PDB inference): {e}")

    # Convert to CIF (internal converter)
    try:
        convert_pdb_to_cif(ligand_hash, LIGAND_BASENAME)
        print(f"✅ Converted ligand using internal converter: {cif_path}")
    except Exception as e:
        print(f"❌ CIF conversion failed: {e}")
        raise

    # Relabel CIF atom ids for Rosetta
    try:
        cif_lines = cif_path.read_text(encoding="utf-8").splitlines()
        atoms = list(mol.GetAtoms())
        atom_names = [a.GetSymbol() + str(a.GetIdx() + 1) for a in atoms]
        new_lines, in_loop, idx = [], False, 0
        for line in cif_lines:
            if line.strip().startswith("loop_"):
                in_loop = False
            if "_atom_site.label_atom_id" in line:
                in_loop = True
            elif in_loop and not line.startswith("_") and line.strip():
                parts = line.split()
                if len(parts) >= 1 and idx < len(atom_names):
                    parts[0] = atom_names[idx]
                    idx += 1
                    line = " ".join(parts)
            new_lines.append(line)
        cif_path.write_text("\n".join(new_lines), encoding="utf-8")
        print("🔤 Updated CIF atom labels for Rosetta compatibility")
    except Exception as e:
        print(f"⚠️ CIF atom labeling skipped: {e}")

    # Validate via PDB readback
    mol_pdb = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    if mol_pdb is not None:
        print(f"✅ PDB sanity check: {mol_pdb.GetNumAtoms()} atoms, {mol_pdb.GetNumBonds()} bonds")
    else:
        print("⚠️ PDB sanity check failed: could not reparse output")

    print(f"✅ Ligand CIF ready: {cif_path}")
    return cif_path


# =========================================
# 🔁 Cache helpers
# =========================================
def get_cached_ligand_cif(smiles: str) -> Optional[str]:
    canonical = _canonical_smiles(smiles)
    if not canonical:
        return None
    ligand_hash = compute_hash(canonical)  # FIX: hash canonical for stable cache lookup
    cache_dir = get_cache_dir("ligands", ligand_hash)
    cif_path = cache_dir / f"{LIGAND_BASENAME}.cif"
    return cif_path.read_text(encoding="utf-8") if cif_path.exists() else None


def save_ligand_cif_cache(smiles: str, cif_data: str):
    canonical = _canonical_smiles(smiles)
    if not canonical:
        raise ValueError("No valid SMILES to cache")
    ligand_hash = compute_hash(canonical)  # FIX: hash canonical for stable cache
    cache_dir = get_cache_dir("ligands", ligand_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{LIGAND_BASENAME}.cif").write_text(cif_data, encoding="utf-8")
    print(f"💾 Cached CIF: {cache_dir}/{LIGAND_BASENAME}.cif")

def get_cached_ligand_png(smiles: str) -> Optional[Path]:
    canonical = _canonical_smiles(smiles)
    if not canonical:
        return None
    ligand_hash = compute_hash(canonical)
    cache_dir = get_cache_dir("ligands", ligand_hash)
    p = cache_dir / f"{LIGAND_BASENAME}.png"
    return p if p.exists() else None