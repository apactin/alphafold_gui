"""
metrics.py — Structural and energetic analysis for AF3–Rosetta pipeline

Computes:
 - Protein and ligand RMSD (AF3 vs Rosetta)
 - Rosetta energy summary
 - Covalent bond length, angle, and dihedral (if covalent=True)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import json
import re
from af3_pipeline.config import cfg 
import subprocess
import platform

DISTRO_NAME  = cfg.get("wsl_distro", "Ubuntu-22.04")
LINUX_HOME   = cfg.get("linux_home_root", "")
WSL_EXE = r"C:\Windows\System32\wsl.exe" if platform.system() == "Windows" else "wsl"

def linuxize_path(p: Path) -> str:
    s = str(p)
    s = s.replace(f"\\\\wsl.localhost\\{DISTRO_NAME}", "")
    s = s.replace("\\", "/")
    if not (s.startswith("/home") or s.startswith("/mnt")):
        s = f"{LINUX_HOME}/" + s.lstrip("/")
    return s

def run_wsl(cmd: str) -> subprocess.CompletedProcess:
    if platform.system() == "Windows":
        full = [WSL_EXE, "-d", DISTRO_NAME, "--", "bash", "-lc", cmd]
    else:
        full = ["bash", "-lc", cmd]
    return subprocess.run(full, text=True, capture_output=True)

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
    proc = run_wsl(f"python3 -c \"{py}\"")
    if proc.returncode != 0:
        raise RuntimeError("CIF->PDB conversion failed:\n" + proc.stdout + "\n" + proc.stderr)
    if not out_pdb.exists():
        raise FileNotFoundError(f"Expected converted PDB not found: {out_pdb}")
    return out_pdb


# ============================================================
# 🧩 Core geometry helpers
# ============================================================

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC (in degrees)"""
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def _dihedral(p1, p2, p3, p4) -> float:
    """Dihedral angle between four points, in degrees"""
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


# ============================================================
# 📦 Structure loading and mapping
# ============================================================

def load_structure(path: Path):
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


def get_atom_coords(struct, chain_id: str, resnum: int, atom_name: str):
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
# 🧬 RMSD calculations
# ============================================================

def compute_ca_rmsd(struct_a, struct_b, chain="A", verbose=True):
    chain = (chain or "").strip() or "A"
    ch_a = struct_a.chains.get(chain)
    ch_b = struct_b.chains.get(chain)
    if not ch_a or not ch_b:
        print(f"⚠️ Missing chain {chain} in one of the structures.")
        return np.nan

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
        print(f"⚠️ Too few matched CA atoms for chain {chain}")
        return np.nan

    A = np.vstack(coords_a)
    B = np.vstack(coords_b)
    rmsd = np.sqrt(((A - B) ** 2).sum(axis=1).mean())
    print(f"✅ Chain {chain} RMSD: {rmsd:.3f} Å")
    return rmsd



def compute_ligand_rmsd(struct_a, struct_b, lig_names=("LIG", "LIGAND"), verbose=True):
    lig_names = {n.upper() for n in lig_names}

    def collect(struct, label):
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
        print("⚠️ Ligand mismatch or missing.")
        return np.nan

    diffs = np.vstack([aa.xyz() - bb.xyz() for aa, bb in zip(a, b)])
    rmsd = np.sqrt((diffs ** 2).sum(axis=1).mean())
    if verbose:
        print(f"✅ Ligand RMSD: {rmsd:.3f} Å")
    return rmsd



# ============================================================
# ⚗️ Covalent bond geometry
# ============================================================

def compute_covalent_geometry(struct, prot_chain, prot_resnum, prot_atom,
                              lig_chain, lig_resnum, lig_atom,
                              lig_parent=None, lig_grandparent=None):
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
            if gp_xyz is not None:
                geom["bond_dihedral"] = _dihedral(gp_xyz, parent_xyz, lig_xyz, prot_xyz)
    return geom


# ============================================================
# 📉 Rosetta score parsing
# ============================================================

def read_rosetta_scorefile(scorefile: Path, verbose=True):
    """Parse Rosetta score.sc → DataFrame."""
    if not scorefile.exists():
        print(f"⚠️ Missing scorefile: {scorefile}")
        return pd.DataFrame()

    with open(scorefile) as f:
        raw = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    lines = [l for l in raw if not l.startswith("SEQUENCE:")]
    if not any(l.startswith("SCORE:") for l in lines):
        print(f"⚠️ No SCORE lines in {scorefile}")
        return pd.DataFrame()

    cleaned = [re.sub(r"^SCORE:\s*", "", l) for l in lines]
    header = re.split(r"\s+", cleaned[0])
    data_rows = [re.split(r"\s+", l) for l in cleaned[1:] if len(l.split()) > 1]
    data_rows = [r + [""] if len(r) == len(header) - 1 else r for r in data_rows]
    valid_rows = [r for r in data_rows if len(r) == len(header)]

    if not valid_rows:
        print(f"⚠️ No valid rows in {scorefile}")
        return pd.DataFrame()

    df = pd.DataFrame(valid_rows, columns=header)
    for c in df.columns:
        if c != "description":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "total_score" not in df.columns:
        for alt in ["score", "fa_score", "total"]:
            if alt in df.columns:
                df["total_score"] = df[alt]
                break
    return df


# ============================================================
# 📊 Summary aggregation
# ============================================================
def summarize_metrics(job_dir: Path):
    """
    Analyze a single bundled job folder at:
      ~/.af3_pipeline/jobs/<job_name>/

    Expected AF3 outputs in job_dir:
      *_model.cif (or *_model.pdb)
      prepared_meta.json (optional)

    Expected Rosetta outputs in the newest subfolder:
      rosetta_relax_*/model_relaxed.pdb   (copied by rosetta_minimize)
      rosetta_relax_*/score.sc
    """
    job_dir = Path(job_dir)
    if not job_dir.exists():
        raise FileNotFoundError(f"Job folder not found: {job_dir}")

    job_name = job_dir.name
    print(f"\n📊 Analyzing {job_name} ...")

    # --- AF3 model (prefer PDB if present, else CIF) ---
    af3_pdb = next(job_dir.glob("*_model.pdb"), None)
    af3_cif = next(job_dir.glob("*_model.cif"), None)
    af3_model = af3_pdb or af3_cif
    if not af3_model:
        print(f"⚠️ No AF3 model found in {job_dir} (expected *_model.cif or *_model.pdb).")
        return None

    # --- Locate newest Rosetta relax folder (if any) ---
    relax_folders = sorted(
        [p for p in job_dir.glob("rosetta_relax_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    relax_dir = relax_folders[0] if relax_folders else None

    # --- Find relaxed model + scorefile ---
    relaxed_model = None
    scorefile = None

    if relax_dir:
        # Preferred: file you copy explicitly
        candidates = [
            relax_dir / "model_relaxed.pdb",
            *relax_dir.glob("*_model_relaxed.pdb"),
            *relax_dir.glob("*relaxed*.pdb"),
            *relax_dir.glob("*_0001.pdb"),
        ]
        relaxed_model = next((p for p in candidates if p and p.exists()), None)
        scorefile = relax_dir / "score.sc"
        print(f"📂 Using relax dir: {relax_dir}")

    # Fallback: older layouts where relaxed outputs are in job root
    if relaxed_model is None:
        relaxed_candidates = [
            *job_dir.glob("*_model_relaxed.pdb"),
            *job_dir.glob("*relaxed*.pdb"),
            *job_dir.glob("*_0001.pdb"),
        ]
        relaxed_model = relaxed_candidates[0] if relaxed_candidates else None
        scorefile = job_dir / "score.sc"

    meta_file = job_dir / "prepared_meta.json"

    # Always load AF3 structure
    st_af3 = load_structure(af3_model)

    metrics = {
        "job": job_name,
        "af3_model": str(af3_model),
        "relax_dir": str(relax_dir) if relax_dir else "",
        "relaxed_model": str(relaxed_model) if relaxed_model else "",
        "protein_RMSD": np.nan,
        "ligand_RMSD": np.nan,
        "total_score": np.nan,
    }

    # If Rosetta outputs exist, compute comparisons
    if relaxed_model and relaxed_model.exists():
        print(f"📄 Using relaxed model: {relaxed_model}")
        st_relax = load_structure(relaxed_model)

        metrics["protein_RMSD"] = compute_ca_rmsd(st_af3, st_relax)
        metrics["ligand_RMSD"] = compute_ligand_rmsd(st_af3, st_relax)

        if scorefile and scorefile.exists():
            scores = read_rosetta_scorefile(scorefile)
            if not scores.empty:
                if "total_score" in scores.columns:
                    metrics["total_score"] = float(scores["total_score"].min())
                for key in ["fa_atr", "fa_rep", "fa_sol", "fa_elec"]:
                    if key in scores.columns:
                        metrics[key] = float(scores[key].iloc[-1])
        else:
            print(f"ℹ️ score.sc not found (looked at: {scorefile})")

        # covalent geometry if meta exists
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("covalent"):
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
        print("ℹ️ Rosetta relaxed model not found (looked in rosetta_relax_* and job root).")

    out_csv = job_dir / "metrics_summary.csv"
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)

    print(f"✅ Metrics summary written to {out_csv}")
    if relaxed_model and relaxed_model.exists():
        print(
            f"   Protein RMSD: {metrics['protein_RMSD']:.3f} Å\n"
            f"   Ligand RMSD:  {metrics['ligand_RMSD']:.3f} Å\n"
            f"   Total score:  {metrics['total_score']:.2f}\n"
        )

    return metrics

# ============================================================
# 🧩 Unified pipeline entrypoint
# ============================================================
def run(job_dir: str | Path, multi_seed: bool = False):
    job_dir = Path(job_dir)

    if multi_seed:
        print(f"🧠 Multi-seed metrics mode enabled for {job_dir.name}")

        # locate rosetta_runs folder
        runs_root = job_dir / "rosetta_runs"
        if not runs_root.exists():
            print("⚠️ No 'rosetta_runs' folder found — cannot perform multi-seed analysis.")
            return None

        # find the most recent rosetta_run_* folder
        run_folders = sorted(runs_root.glob("rosetta_run_*"), key=lambda p: p.stat().st_mtime)
        if not run_folders:
            print("⚠️ No rosetta_run_* folders found in", runs_root)
            return None

        latest_run = run_folders[-1]
        print(f"📂 Using latest run folder: {latest_run.name}")

        # find all per-seed relax folders inside that run
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
                    # Extract seed/sample identifiers if available
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

        # Aggregate all metrics into a summary CSV
        df = pd.DataFrame(all_metrics)
        out_csv = latest_run / "multi_seed_metrics_summary.csv"
        df.to_csv(out_csv, index=False)

        print(f"\n✅ Multi-seed metrics summary written to {out_csv}")
        summary_cols = ["seed", "sample", "protein_RMSD", "ligand_RMSD", "total_score"]
        summary_cols = [c for c in summary_cols if c in df.columns]

        print("\n📈 Summary of all minimizations:")
        print(df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        best_idx = df["total_score"].idxmin() if "total_score" in df.columns else None
        if best_idx is not None:
            best_row = df.loc[best_idx]
            print(f"\n🏆 Best seed–sample: seed {best_row.get('seed')} sample {best_row.get('sample')} "
                  f"(score={best_row['total_score']:.2f})")

        return df

    else:
        # normal single-job metrics
        return summarize_metrics(job_dir)



# ============================================================
# 🔍 CLI entry (optional)
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python metrics.py <job_folder>")
        sys.exit(1)
    summarize_metrics(Path(sys.argv[1]))
