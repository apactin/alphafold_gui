"""
metrics.py — Structural and energetic analysis for AF3–Rosetta pipeline

Computes:
 - Protein and ligand RMSD (AF3 vs RosettaRelax)
 - RosettaRelax energy summary
 - RosettaLigand (local refinement) energy summary (if present)
 - Covalent bond length, angle, and dihedral (if covalent=True and metadata present)

Key design changes (drop-in):
 - Prefer machine-readable pointers:
     - <job_dir>/latest_rosetta_relax.json
     - <job_dir>/latest_rosetta_ligand.json
 - Fallback to older folder scanning for backwards compatibility.
 - Multi-seed mode reads rosetta_run_* and per-seed relax outputs as before,
   but prefers per-folder rosetta_relax_outputs.json when available.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import json
import re
import subprocess
import platform
from af3_pipeline.config import cfg

DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")
LINUX_HOME = cfg.get("linux_home_root", "")
WSL_EXE = r"C:\Windows\System32\wsl.exe" if platform.system() == "Windows" else "wsl"


# ============================================================
# 🧰 Path + WSL helpers
# ============================================================

def linuxize_path(p: Path) -> str:
    """Convert UNC/Windows paths → WSL Linux paths."""
    s = str(p)

    # UNC -> WSL absolute (\\wsl.localhost\Distro\home\... -> /home/...)
    s = s.replace(f"\\\\wsl.localhost\\{DISTRO_NAME}", "")
    s = s.replace("\\", "/")

    # Windows drive -> /mnt/<drive>/...
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        return f"/mnt/{drive}/{rest}"

    if s.startswith("/home") or s.startswith("/mnt") or s.startswith("/tmp") or s.startswith("/var"):
        return s

    if LINUX_HOME:
        if not s.startswith("/"):
            s = f"{LINUX_HOME}/" + s.lstrip("/")
        else:
            s = f"{LINUX_HOME}" + s
    return s


def run_wsl(cmd: str) -> subprocess.CompletedProcess:
    if platform.system() == "Windows":
        full = [WSL_EXE, "-d", DISTRO_NAME, "--", "bash", "-lc", cmd]
    else:
        full = ["bash", "-lc", cmd]
    proc = subprocess.run(full, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"WSL cmd failed ({proc.returncode}):\n{cmd}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return proc


# ============================================================
# 🧬 Minimal PDB parser
# ============================================================

class _PDBAtom:
    __slots__ = ("name", "x", "y", "z", "element")
    def __init__(self, name, x, y, z, element):
        self.name = name.strip()
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.element = (element or "").strip().upper() or self.name[:1].upper()

    def xyz(self):
        return np.array([self.x, self.y, self.z], dtype=float)

class _PDBResidue:
    __slots__ = ("resname", "resnum", "atoms")
    def __init__(self, resname, resnum):
        self.resname = resname.strip().upper()
        self.resnum = int(resnum)
        self.atoms = []

class _PDBChain:
    __slots__ = ("chain_id", "residues")
    def __init__(self, chain_id):
        self.chain_id = (chain_id or "").strip() or "A"
        self.residues = {}  # resnum -> _PDBResidue

class _PDBStructure:
    __slots__ = ("chains",)
    def __init__(self):
        self.chains = {}  # chain_id -> _PDBChain


def read_pdb_simple(pdb_path: Path) -> _PDBStructure:
    st = _PDBStructure()
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            resname   = line[17:20].strip()
            chain_id  = line[21:22].strip() or "A"
            resnum    = line[22:26].strip()
            x = line[30:38].strip()
            y = line[38:46].strip()
            z = line[46:54].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""

            if not resnum or not x or not y or not z:
                continue

            ch = st.chains.get(chain_id)
            if ch is None:
                ch = _PDBChain(chain_id)
                st.chains[chain_id] = ch

            rn = int(resnum)
            res = ch.residues.get(rn)
            if res is None:
                res = _PDBResidue(resname, rn)
                ch.residues[rn] = res

            res.atoms.append(_PDBAtom(atom_name, x, y, z, element))
    return st


def cif_to_pdb_via_wsl(cif_path: Path, out_pdb: Path) -> Path:
    cif_linux = linuxize_path(cif_path)
    out_linux = linuxize_path(out_pdb)
    py = (
        "import gemmi; "
        f"st=gemmi.read_structure('{cif_linux}'); "
        "st.remove_alternative_conformations(); "
        f"st.write_pdb('{out_linux}'); "
        f"print('{out_linux}')"
    )
    run_wsl(f"python3 -c \"{py}\"")
    if not out_pdb.exists():
        raise FileNotFoundError(f"Expected converted PDB not found: {out_pdb}")
    return out_pdb


# ============================================================
# 📦 Structure loading and mapping
# ============================================================

def load_structure(path: Path) -> _PDBStructure:
    """Load PDB directly; if CIF, convert to PDB via gemmi in WSL then load."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure not found: {path}")

    suf = path.suffix.lower()
    if suf == ".cif":
        out_pdb = path.with_suffix(".pdb")
        if not out_pdb.exists():
            cif_to_pdb_via_wsl(path, out_pdb)
        return read_pdb_simple(out_pdb)

    if suf == ".pdb":
        return read_pdb_simple(path)

    raise ValueError(f"Unsupported structure type: {path}")


def get_atom_coords(struct: _PDBStructure, chain_id: str, resnum: int, atom_name: str):
    chain_id = (chain_id or "").strip() or "A"
    atom_name = atom_name.strip()
    ch = struct.chains.get(chain_id)
    if not ch:
        return None
    res = ch.residues.get(int(resnum))
    if not res:
        return None
    for atom in res.atoms:
        if atom.name == atom_name:
            return atom.xyz()
    return None


# ============================================================
# 🧩 Core geometry helpers
# ============================================================

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC (in degrees)"""
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return float("nan")
    cosang = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def _dihedral(p1, p2, p3, p4) -> float:
    """Dihedral angle between four points, in degrees"""
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    nb1 = np.linalg.norm(b1)
    if nb1 == 0:
        return float("nan")
    b1 /= nb1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


# ============================================================
# 🧬 RMSD calculations
# ============================================================

def compute_ca_rmsd(struct_a: _PDBStructure, struct_b: _PDBStructure, chain="A", verbose=True) -> float:
    chain = (chain or "").strip() or "A"
    ch_a = struct_a.chains.get(chain)
    ch_b = struct_b.chains.get(chain)
    if not ch_a or not ch_b:
        if verbose:
            print(f"⚠️ Missing chain {chain} in one of the structures.")
        return float("nan")

    coords_a, coords_b = [], []
    common = sorted(set(ch_a.residues.keys()) & set(ch_b.residues.keys()))
    for rn in common:
        ra = ch_a.residues[rn]
        rb = ch_b.residues[rn]
        ca_a = next((at for at in ra.atoms if at.name == "CA"), None)
        ca_b = next((at for at in rb.atoms if at.name == "CA"), None)
        if ca_a and ca_b:
            coords_a.append(ca_a.xyz())
            coords_b.append(ca_b.xyz())

    if verbose:
        print(f"🧩 Chain {chain}: matched {len(coords_a)} CA atoms by residue number")

    if len(coords_a) < 3:
        if verbose:
            print(f"⚠️ Too few matched CA atoms for chain {chain}")
        return float("nan")

    A = np.vstack(coords_a)
    B = np.vstack(coords_b)
    rmsd = np.sqrt(((A - B) ** 2).sum(axis=1).mean())
    if verbose:
        print(f"✅ Chain {chain} RMSD: {rmsd:.3f} Å")
    return float(rmsd)


def compute_ligand_rmsd(struct_a: _PDBStructure, struct_b: _PDBStructure, lig_names=("LIG"), verbose=True) -> float:
    lig_names = {n.upper() for n in lig_names}

    def collect(struct: _PDBStructure, label: str):
        atoms = []
        for ch in struct.chains.values():
            for res in ch.residues.values():
                if res.resname in lig_names:
                    heavy = [a for a in res.atoms if a.element != "H"]
                    atoms.extend(heavy)
        if verbose:
            print(f"🔹 {label}: found {len(atoms)} heavy atoms")
        return sorted(atoms, key=lambda a: a.name)

    a = collect(struct_a, "Structure A")
    b = collect(struct_b, "Structure B")
    if not a or not b or len(a) != len(b):
        if verbose:
            print("⚠️ Ligand mismatch or missing.")
        return float("nan")

    diffs = np.vstack([aa.xyz() - bb.xyz() for aa, bb in zip(a, b)])
    rmsd = np.sqrt((diffs ** 2).sum(axis=1).mean())
    if verbose:
        print(f"✅ Ligand RMSD: {rmsd:.3f} Å")
    return float(rmsd)


# ============================================================
# ⚗️ Covalent bond geometry
# ============================================================

def compute_covalent_geometry(
    struct: _PDBStructure,
    prot_chain: str,
    prot_resnum: int,
    prot_atom: str,
    lig_chain: str,
    lig_resnum: int,
    lig_atom: str,
    lig_parent: str | None = None,
    lig_grandparent: str | None = None,
):
    """Compute bond length, angle, and dihedral for a covalent linkage."""
    prot_xyz = get_atom_coords(struct, prot_chain, prot_resnum, prot_atom)
    lig_xyz  = get_atom_coords(struct, lig_chain, lig_resnum, lig_atom)
    if prot_xyz is None or lig_xyz is None:
        return {"bond_length": np.nan, "bond_angle": np.nan, "bond_dihedral": np.nan}

    geom = {"bond_length": _distance(prot_xyz, lig_xyz),
            "bond_angle": np.nan, "bond_dihedral": np.nan}

    if lig_parent:
        parent_xyz = get_atom_coords(struct, lig_chain, lig_resnum, lig_parent)
        if parent_xyz is not None:
            geom["bond_angle"] = _angle(parent_xyz, lig_xyz, prot_xyz)
        if lig_grandparent:
            gp_xyz = get_atom_coords(struct, lig_chain, lig_resnum, lig_grandparent)
            if gp_xyz is not None and parent_xyz is not None:
                geom["bond_dihedral"] = _dihedral(gp_xyz, parent_xyz, lig_xyz, prot_xyz)
    return geom


# ============================================================
# 📉 Rosetta score parsing
# ============================================================

def read_rosetta_scorefile(scorefile: Path, verbose=True) -> pd.DataFrame:
    if not scorefile.exists():
        if verbose:
            print(f"⚠️ Missing scorefile: {scorefile}")
        return pd.DataFrame()

    header = None
    rows = []

    with open(scorefile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("SEQUENCE:"):
                continue
            if not line.startswith("SCORE:"):
                continue

            parts = line.split()

            # Header line
            if len(parts) > 2 and parts[1] == "total_score":
                header = parts[1:]
                continue

            # Data line
            if header is None:
                continue

            vals = parts[1:]
            if len(vals) != len(header):
                # Skip malformed lines instead of producing shifted columns
                continue

            rows.append(vals)

    if not header or not rows:
        if verbose:
            print(f"⚠️ No valid SCORE table found in {scorefile}")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)

    for c in df.columns:
        if c != "description":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _min_score_fields(df: pd.DataFrame) -> dict[str, float]:
    """Return min(total_score) and a few common terms if present."""
    out: dict[str, float] = {}
    if df is None or df.empty:
        return out
    if "total_score" in df.columns:
        out["total_score_min"] = float(df["total_score"].min())
    # Add a few common energy terms (take row of min total_score if possible)
    try:
        if "total_score" in df.columns:
            best = df.loc[df["total_score"].idxmin()]
        else:
            best = df.iloc[-1]
        for key in ["fa_atr", "fa_rep", "fa_sol", "fa_elec", "hbond_sc", "hbond_bb_sc", "dslf_fa13"]:
            if key in df.columns:
                out[key] = float(best[key])
    except Exception:
        pass
    return out


# ============================================================
# 🧭 Pointer-based discovery for Relax/Ligand
# ============================================================

def _load_json_if_exists(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_path_maybe_relative(base: Path, maybe: str | None) -> Path | None:
    if not maybe:
        return None
    p = Path(maybe)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _find_af3_model(job_dir: Path) -> Path | None:
    af3_pdb = next(job_dir.glob("*_model.pdb"), None)
    af3_cif = next(job_dir.glob("*_model.cif"), None)
    return af3_pdb or af3_cif


def _find_relax_from_pointers(job_dir: Path) -> tuple[Path | None, Path | None, dict | None]:
    """
    Returns (relaxed_pdb, scorefile, relax_record)
    - Prefers latest_rosetta_relax.json and its "relaxed_pdb"/"scorefile".
    - Falls back to newest rosetta_relax_* folder scanning.
    """
    relax_record = _load_json_if_exists(job_dir / "latest_rosetta_relax.json")

    # Multi-seed pointer object: not a single relaxed PDB
    if relax_record and relax_record.get("multi_seed"):
        return None, None, relax_record

    if relax_record:
        prep_dir = _resolve_path_maybe_relative(job_dir, relax_record.get("prep_dir"))
        relaxed_pdb = _resolve_path_maybe_relative(job_dir, relax_record.get("relaxed_pdb")) or \
                      _resolve_path_maybe_relative(job_dir, relax_record.get("relaxed_pdb_raw"))
        scorefile = _resolve_path_maybe_relative(job_dir, relax_record.get("scorefile"))
        if relaxed_pdb and relaxed_pdb.exists():
            return relaxed_pdb, (scorefile if (scorefile and scorefile.exists()) else None), relax_record

    # Fallback scan
    relax_folders = sorted(
        [p for p in job_dir.glob("rosetta_relax_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    relax_dir = relax_folders[0] if relax_folders else None
    if relax_dir:
        # Prefer restored, then raw, then legacy patterns
        candidates = [
            relax_dir / "model_relaxed_restored.pdb",
            relax_dir / "model_relaxed_raw.pdb",
            *relax_dir.glob("model.pdb_00_*.pdb"),
            *relax_dir.glob("*relaxed*.pdb"),
        ]
        relaxed_model = next((p for p in candidates if p.exists()), None)
        scorefile = relax_dir / "score.sc"
        return relaxed_model, (scorefile if scorefile.exists() else None), relax_record

    return None, None, relax_record


def _find_ligand_from_pointers(job_dir: Path) -> tuple[Path | None, Path | None, dict | None]:
    """
    Returns (best_pdb, scorefile, ligand_record)
    - Prefers latest_rosetta_ligand.json
    - Falls back to newest rosetta_ligand_* folder scanning.
    """
    lig_record = _load_json_if_exists(job_dir / "latest_rosetta_ligand.json")

    # Multi-seed pointer object: not a single best PDB
    if lig_record and lig_record.get("multi_seed"):
        return None, None, lig_record

    if lig_record:
        out_dir = _resolve_path_maybe_relative(job_dir, lig_record.get("out_dir"))
        best_pdb = _resolve_path_maybe_relative(job_dir, lig_record.get("best_pdb"))
        scorefile = _resolve_path_maybe_relative(job_dir, lig_record.get("scorefile"))
        if best_pdb and best_pdb.exists():
            return best_pdb, (scorefile if (scorefile and scorefile.exists()) else None), lig_record
        # If best_pdb missing but out_dir exists, try scanning
        if out_dir and out_dir.exists():
            scorefile2 = out_dir / "score.sc"
            best2 = next(sorted(out_dir.glob("*.pdb"), key=lambda p: p.stat().st_mtime, reverse=True), None)
            return best2, (scorefile2 if scorefile2.exists() else None), lig_record

    # Fallback scan
    lig_folders = sorted(
        [p for p in job_dir.glob("rosetta_ligand_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if lig_folders:
        out_dir = lig_folders[0]
        scorefile = out_dir / "score.sc"
        best = next(sorted(out_dir.glob("*.pdb"), key=lambda p: p.stat().st_mtime), None)
        return best, (scorefile if scorefile.exists() else None), lig_record

    return None, None, lig_record


# ============================================================
# 📊 Summary aggregation (single job folder)
# ============================================================

def summarize_metrics(job_dir: Path):
    """
    Analyze a single bundled job folder at:
      ~/.af3_pipeline/jobs/<job_name>/

    Prefers pointer files:
      - latest_rosetta_relax.json
      - latest_rosetta_ligand.json
    """
    job_dir = Path(job_dir)
    if not job_dir.exists():
        raise FileNotFoundError(f"Job folder not found: {job_dir}")

    job_name = job_dir.name
    print(f"\n📊 Analyzing {job_name} ...")

    af3_model = _find_af3_model(job_dir)
    if not af3_model:
        print(f"⚠️ No AF3 model found in {job_dir} (expected *_model.cif or *_model.pdb).")
        return None

    # Discover Relax (pointer-aware)
    relaxed_model, relax_scorefile, relax_record = _find_relax_from_pointers(job_dir)

    # Discover RosettaLigand (pointer-aware)
    ligand_best, ligand_scorefile, ligand_record = _find_ligand_from_pointers(job_dir)

    # Metadata (optional)
    meta_file = job_dir / "prepared_meta.json"
    meta = _load_json_if_exists(meta_file) if meta_file.exists() else None

    # Fast-path: if no Rosetta relax model exists, we cannot compute RMSDs or covalent geometry anyway.
    # Avoid CIF->PDB conversion via WSL/gemmi in "skip rosetta" scenarios.
    need_structure = bool(relaxed_model and relaxed_model.exists())
    need_structure = need_structure or bool(meta and meta.get("covalent"))

    if not need_structure:
        metrics: dict[str, object] = {
            "job": job_name,
            "af3_model": str(af3_model),
            "relaxed_model": str(relaxed_model) if relaxed_model else "",
            "relax_scorefile": str(relax_scorefile) if relax_scorefile else "",
            "ligand_best_model": str(ligand_best) if ligand_best else "",
            "ligand_scorefile": str(ligand_scorefile) if ligand_scorefile else "",
            "protein_RMSD": np.nan,
            "ligand_RMSD": np.nan,
        }

        # Ligand energies can still be read without structure parsing
        if ligand_scorefile and ligand_scorefile.exists():
            df = read_rosetta_scorefile(ligand_scorefile)
            lig_fields = _min_score_fields(df)
            for k, v in lig_fields.items():
                metrics[f"ligand_{k}"] = v

        out_csv = job_dir / "metrics_summary.csv"
        pd.DataFrame([metrics]).to_csv(out_csv, index=False)
        print(f"✅ Metrics summary written to {out_csv} (minimal; no Rosetta relax model found)")
        return metrics

    # If we get here, we actually need structures
    st_af3 = load_structure(af3_model)


    metrics: dict[str, object] = {
        "job": job_name,
        "af3_model": str(af3_model),
        "relaxed_model": str(relaxed_model) if relaxed_model else "",
        "relax_scorefile": str(relax_scorefile) if relax_scorefile else "",
        "ligand_best_model": str(ligand_best) if ligand_best else "",
        "ligand_scorefile": str(ligand_scorefile) if ligand_scorefile else "",
        "protein_RMSD": np.nan,
        "ligand_RMSD": np.nan,
    }

    # ----------------------------
    # Relax metrics
    # ----------------------------
    if relaxed_model and relaxed_model.exists():
        print(f"📄 Using relaxed model: {relaxed_model}")
        st_relax = load_structure(relaxed_model)

        metrics["protein_RMSD"] = compute_ca_rmsd(st_af3, st_relax)
        metrics["ligand_RMSD"] = compute_ligand_rmsd(st_af3, st_relax)

        if relax_scorefile and relax_scorefile.exists():
            df = read_rosetta_scorefile(relax_scorefile)
            relax_fields = _min_score_fields(df)
            # prefix relax_
            for k, v in relax_fields.items():
                metrics[f"relax_{k}"] = v
        else:
            print("ℹ️ Relax score.sc not found via pointer/fallback.")

        # Covalent geometry (if possible)
        if meta and meta.get("covalent"):
            try:
                geom = compute_covalent_geometry(
                    st_relax,
                    meta.get("prot_chain", "A"),
                    int(meta["prot_resnum"]),
                    meta["prot_atom"],
                    meta.get("ligand_chain", "L"),
                    int(meta["ligand_resnum"]),
                    meta["ligand_atom"],
                    meta.get("ligand_atom_parent"),
                    meta.get("ligand_atom_grandparent"),
                )
                metrics.update(geom)
            except Exception as e:
                print(f"⚠️ Failed covalent geometry parsing: {e}")
    else:
        print("ℹ️ Rosetta relaxed model not found (pointer + fallback scan).")

    # ----------------------------
    # RosettaLigand metrics (local refinement)
    # ----------------------------
    if ligand_scorefile and ligand_scorefile.exists():
        df = read_rosetta_scorefile(ligand_scorefile)
        lig_fields = _min_score_fields(df)
        for k, v in lig_fields.items():
            metrics[f"ligand_{k}"] = v
    else:
        # Only warn if a ligand run is expected (record exists) but scorefile missing
        if ligand_record and not (ligand_record.get("multi_seed") or False):
            print("ℹ️ RosettaLigand score.sc not found via pointer/fallback.")
        # otherwise silent

    # Write summary
    out_csv = job_dir / "metrics_summary.csv"
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    print(f"✅ Metrics summary written to {out_csv}")

    # Pretty print a few key lines
    if relaxed_model and relaxed_model.exists():
        pr = metrics.get("protein_RMSD", np.nan)
        lr = metrics.get("ligand_RMSD", np.nan)
        rs = metrics.get("relax_total_score_min", np.nan)
        ls = metrics.get("ligand_total_score_min", np.nan)
        print(
            f"   Protein RMSD (AF3 vs Relax): {float(pr):.3f} Å\n"
            f"   Ligand RMSD  (AF3 vs Relax): {float(lr):.3f} Å\n"
            f"   Relax total_score (min):     {float(rs):.2f}\n"
            f"   Ligand total_score (min):    {float(ls):.2f}\n"
        )

    return metrics


# ============================================================
# 🧠 Multi-seed mode
# ============================================================

def _summarize_seed_folder(seed_dir: Path) -> dict | None:
    """
    Compute metrics for a single per-seed directory (e.g., rosetta_relax_seedX_sampleY)
    which likely contains:
      - rosetta_relax_outputs.json
      - relaxed pdb(s)
      - score.sc
      - may also contain rosetta_ligand_* outputs
    """
    # For multi-seed folders, treat that folder as a "job_dir" itself
    return summarize_metrics(seed_dir)


def run(job_dir: str | Path, multi_seed: bool = False):
    job_dir = Path(job_dir)

    # If multi_seed: prefer latest_rosetta_relax.json index pointer
    if multi_seed:
        print(f"🧠 Multi-seed metrics mode enabled for {job_dir.name}")

        relax_ptr = _load_json_if_exists(job_dir / "latest_rosetta_relax.json") or {}
        if relax_ptr.get("multi_seed") and relax_ptr.get("index"):
            idx = Path(relax_ptr["index"])
            if idx.exists():
                entries = json.loads(idx.read_text(encoding="utf-8"))
                # entries contain prep_dir per seed/sample
                all_metrics = []
                for ent in entries:
                    prep = Path(ent.get("prep_dir", ""))
                    if not prep.exists():
                        continue
                    print(f"\n📊 Processing metrics for {prep.name} ...")
                    m = _summarize_seed_folder(prep)
                    if m:
                        mm = re.search(r"seed(\d+)_sample(\d+)", prep.name)
                        if mm:
                            m["seed"] = int(mm.group(1))
                            m["sample"] = int(mm.group(2))
                        all_metrics.append(m)

                if not all_metrics:
                    print("⚠️ No metrics computed from relax index.")
                    return None

                df = pd.DataFrame(all_metrics)
                # Save into run_dir if available, else jobs root
                run_dir = Path(relax_ptr.get("run_dir")) if relax_ptr.get("run_dir") else idx.parent
                out_csv = run_dir / "multi_seed_metrics_summary.csv"
                df.to_csv(out_csv, index=False)
                print(f"\n✅ Multi-seed metrics summary written to {out_csv}")

                # Print best by relax_total_score_min if present; else by relax_total_score_min missing -> skip
                if "relax_total_score_min" in df.columns:
                    best_idx = df["relax_total_score_min"].idxmin()
                    best_row = df.loc[best_idx]
                    print(
                        f"\n🏆 Best seed–sample by relax_total_score_min: "
                        f"seed {best_row.get('seed')} sample {best_row.get('sample')} "
                        f"(score={best_row['relax_total_score_min']:.2f})"
                    )
                return df

        # Fallback: older behavior scanning rosetta_runs
        runs_root = job_dir / "rosetta_runs"
        if not runs_root.exists():
            print("⚠️ No 'rosetta_runs' folder found — cannot perform multi-seed analysis.")
            return None

        run_folders = sorted(runs_root.glob("rosetta_run_*"), key=lambda p: p.stat().st_mtime)
        if not run_folders:
            print("⚠️ No rosetta_run_* folders found in", runs_root)
            return None

        latest_run = run_folders[-1]
        print(f"📂 Using latest run folder: {latest_run.name}")

        relax_dirs = sorted(latest_run.glob("rosetta_relax_seed*"), key=lambda p: p.stat().st_mtime)
        if not relax_dirs:
            print("⚠️ No rosetta_relax_seed* directories found inside", latest_run)
            return None

        all_metrics = []
        for rdir in relax_dirs:
            print(f"\n📊 Processing metrics for {rdir.name} ...")
            try:
                metrics = summarize_metrics(rdir)
                if metrics:
                    m = re.search(r"seed(\d+)_sample(\d+)", rdir.name)
                    if m:
                        metrics["seed"] = int(m.group(1))
                        metrics["sample"] = int(m.group(2))
                    all_metrics.append(metrics)
            except Exception as e:
                print(f"⚠️ Metrics computation failed for {rdir.name}: {e}")

        if not all_metrics:
            print("⚠️ No metrics computed in", latest_run)
            return None

        df = pd.DataFrame(all_metrics)
        out_csv = latest_run / "multi_seed_metrics_summary.csv"
        df.to_csv(out_csv, index=False)

        print(f"\n✅ Multi-seed metrics summary written to {out_csv}")
        return df

    # Single job
    return summarize_metrics(job_dir)


# ============================================================
# 🔍 CLI entry (optional)
# ============================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute metrics for AF3→Rosetta pipeline (pointer-aware).")
    ap.add_argument("job_dir", help="Bundled job directory (or seed folder in multi-seed)")
    ap.add_argument("--multi_seed", action="store_true", help="Multi-seed aggregation mode")
    args = ap.parse_args()

    run(Path(args.job_dir), multi_seed=bool(args.multi_seed))
