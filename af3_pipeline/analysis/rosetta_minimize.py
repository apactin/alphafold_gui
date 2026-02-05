#!/usr/bin/env python3
"""
rosetta_minimize.py  — Windows launcher to run Rosetta relax on AF3 CIF outputs
===============================================================================

Runs the full flow:
  1. Convert CIF → PDB (Gemmi)
  2. (Optional) Generate ligand params from SMILES (RDKit → OpenBabel → molfile_to_params.py)
  3. Clean PDB via clean_pdb_keep_ligand.py
  4. Rosetta relax (generates its own constraints)
  5. Writes unified output folder and log files
"""

import argparse
import subprocess
import sys
import json
import re
import platform
from pathlib import Path, PurePosixPath
import pandas as pd
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from af3_pipeline.config import cfg

# =========================================================
# 🧩 Rosetta paths (inside WSL, now config-driven)
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

DISTRO_NAME  = cfg.get("wsl_distro", "Ubuntu-22.04")
LINUX_HOME   = cfg.get("linux_home_root", "")
ALPHAFOLD_BASE = cfg.get("af3_dir", f"{LINUX_HOME}/Repositories/alphafold")

APP_TZ = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))

def now_local() -> datetime:
    return datetime.now(APP_TZ)

# =========================================================
# 🧰 Windows → WSL launcher setup
# =========================================================
WSL_EXE = r"C:\Windows\System32\wsl.exe"

def to_wsl_path(subpath: str) -> Path:
    """Return correct path depending on OS (UNC on Windows, /home/... on WSL)."""
    sub = subpath.replace("\\", "/").strip("/")
    if platform.system() == "Windows":
        base_path = ALPHAFOLD_BASE.replace("/", "\\")
        base = f"\\\\wsl.localhost\\{DISTRO_NAME}{base_path}"
        return Path(base + (("\\" + sub.replace("/", "\\")) if sub else ""))
    else:
        return Path(ALPHAFOLD_BASE + (("/" + sub) if sub else ""))

def run_wsl(cmd: str, cwd_wsl: str | None = None, log: Path | None = None):
    """Run a command inside WSL via bash -lc. Capture stdout/stderr."""
    if cwd_wsl:
        cmd = f"pushd '{cwd_wsl}' >/dev/null 2>&1; {cmd}; rc=$?; popd >/dev/null 2>&1; exit $rc"
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

def _trim_timestamp(name: str) -> str:
    """Remove trailing timestamp like _YYYYMMDD_HHMMSS."""
    return re.sub(r"_[0-9]{8}[-_][0-9]{6}$", "", name)

# ============================================
# 📁 Directories
# ============================================
AF_OUTPUT_DIR = to_wsl_path("af_output")

# ============================================
# 🧪 Ligand generation (RDKit + optional OpenBabel)
# ============================================
def linuxize_path(p: Path) -> str:
    """Convert UNC/Windows paths → WSL Linux paths."""
    s = str(p)
    s = s.replace(f"\\\\wsl.localhost\\{DISTRO_NAME}", "")
    s = s.replace("\\", "/")
    s = s.replace("C:/Users", "/mnt/c/Users").replace("c:/Users", "/mnt/c/Users")
    if not (s.startswith("/home") or s.startswith("/mnt") or s.startswith("/tmp") or s.startswith("/var")):
        # Only prefix linux home if it looks like a relative path
        if not s.startswith("/"):
            s = f"{LINUX_HOME}/" + s.lstrip("/")
    return s

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
        # write each conformer with aromatic flags intact
        w.write(mol, confId=cid)
    w.close()
    print(f"🧬 Generated aromatic RDKit SDF with {len(conf_ids)} conformers → {out_sdf}")
    return out_sdf


def generate_rosetta_params_from_sdf(sdf_path: Path, out_dir: Path, residue_name: str = "LIG") -> Path | None:
    """Call Rosetta molfile_to_params.py inside WSL. Skips if params already exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / f"{residue_name}.params"
    if params_path.exists():
        print(f"⏭️  Found existing params, skipping generation: {params_path}")
        return params_path
    sdf_linux = linuxize_path(sdf_path)
    cmd = ["wsl", "python3", M2P_PY, "-n", residue_name, "-p", residue_name, "--conformers-in-one-file", sdf_linux]
    try:
        subprocess.run(cmd, check=True, cwd=str(out_dir))
    except subprocess.CalledProcessError as e:
        print(f"❌ molfile_to_params failed: {e}")
        return None
    if params_path.exists():
        print(f"✅ Generated params: {params_path}")
        return params_path
    else:
        print("⚠️ Rosetta params not found — check WSL execution/logs.")
        return None
    
def _resolve_model_cif(job_dir: Path, job_name: str, base_job: str, model_path: str | Path | None) -> Path:
    """
    Resolve the AF3 model CIF to use for single-model (non-multi-seed) runs.

    Priority:
      1) Explicit model_path if provided
      2) Common AF3 output: *_model.cif
      3) Legacy naming: <base_job>_model.cif, <job_name>_model.cif
      4) Any .cif in job_dir (last resort)
    """
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

    # model_*.cif fallback (prefer lowest number)
    numbered = sorted(job_dir.glob("*_model.cif"))
    if numbered:
        return numbered[0]

    # last resort: any cif
    any_cif = sorted(job_dir.glob("*.cif"))
    if any_cif:
        return any_cif[0]

    raise FileNotFoundError(f"❌ Could not find model CIF in {job_dir}")

def _user_jobs_root() -> Path:
    # Prefer explicit config; otherwise default to ~/.af3_pipeline/jobs
    root = cfg.get("jobs_root", None)
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".af3_pipeline" / "jobs").resolve()



# =========================================================
# 🚀 Main pipeline (unchanged logic)
# =========================================================
def run(job_dir: str | Path, multi_seed: bool = False, model_path: str | Path | None = None):
    job_dir = Path(job_dir)
    job_name = job_dir.name
    base_job = _trim_timestamp(job_name)
    
    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    prep_dir = job_dir / f"rosetta_relax_{timestamp}"
    prep_dir.mkdir(parents=True, exist_ok=True)

    job_meta_path = job_dir / "job_metadata.json"
    print(job_meta_path)
    if not job_meta_path.exists():
        job_meta_path = job_dir / "prepared_meta.json"
    job_meta = json.loads(job_meta_path.read_text(encoding="utf-8"))

    print(f"DEBUG: multi_seed={multi_seed!r} (type={type(multi_seed)})")
    
    if multi_seed:
        # Create a per-run directory to hold all seed/sample minimizations
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

            # Each seed–sample minimization goes in its own subfolder under this run
            sub_dir = run_dir / f"rosetta_relax_seed{seed}_sample{sample}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            # replicate the normal minimization pipeline for this sub-model
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

            ligand_smiles = job_meta.get("smiles")
            sdf_rdkit = run_dir / "ligand.sdf"
            sdf_final = sdf_rdkit
            RESNAME = "LIGAND"
            params_path = run_dir / f"{RESNAME}.params"

            if not ligand_smiles:
                print("ℹ️ No 'smiles' field found — assuming protein-only model (apo).")
                ligand_present = False
            else:
                ligand_present = True

            if ligand_present:
                # existing SDF and params generation
                sdf_rdkit = generate_sdf_from_smiles(ligand_smiles, sdf_rdkit)
                params_path = generate_rosetta_params_from_sdf(sdf_final, prep_dir, residue_name=RESNAME)
            else:
                params_path = None

            print("🧼 Running Rosetta clean_pdb_keep_ligand.py ...")
            clean_cmd = (
                f"cd '{linuxize_path(sub_dir)}' && "
                f"python3 '{CLEAN_PDB_PY}' model.pdb -ignorechain"
            )
            run_wsl(clean_cmd, log=sub_dir / "clean_pdb.log")

            cleaned_pdb = sub_dir / "model.pdb_00.pdb"
            #run_wsl(f"test -s '{linuxize_path(cleaned_pdb)}' || (echo '❌ cleaned PDB missing' && false)")
            print(f"✅ Cleaned PDB ready: {cleaned_pdb}")

            prep_dir_linux = linuxize_path(sub_dir)

            if job_meta.get("covalent"):
                output_path = cleaned_pdb.with_name(cleaned_pdb.stem + "_renamed.pdb")
                target_resnum = int(job_meta.get("residue"))
                resname = job_meta.get("prot_atom")
                if resname == "NZ":
                    new_resname = "LYX"
                elif resname == "SG":
                    new_resname = "ALA"
                else:
                    new_resname = None

                with open(cleaned_pdb) as f_in, open(output_path, "w") as f_out:
                    for line in f_in:
                        if line.startswith(("ATOM", "HETATM")):
                            try:
                                resnum = int(line[22:26])
                            except ValueError:
                                resnum = None
                            atom_name = line[12:16].strip()
                            if resnum == target_resnum and atom_name in ("NZ", "SG"):
                                continue
                            if new_resname and resnum == target_resnum:
                                line = f"{line[:17]}{new_resname:>3s}{line[20:]}"
                        f_out.write(line)
                cleaned_pdb_linux = linuxize_path(output_path)
                relax_cmd = (
                    f"'{RELAX_BIN}' -database '{ROSETTA_DB}' "
                    f"-s '{cleaned_pdb_linux}' "
                    f"-relax:constrain_relax_to_start_coords "
                    f"-relax:coord_constrain_sidechains "
                    f"-relax:ramp_constraints false "
                    "-score:weights ref2015_cart "
                    "-relax:cartesian "
                    f"-out:path:all '{prep_dir_linux}' "
                    f"-out:file:scorefile '{prep_dir_linux}/score.sc' "
                    "-nstruct 1 -overwrite "
                    f"-extra_res_fa '{ROSETTA_DB}/chemical/residue_type_sets/fa_standard/residue_types/l-caa/LYX.params' "
                )
            else:
                relax_cmd = (
                    f"'{RELAX_BIN}' -database '{ROSETTA_DB}' "
                    f"-s '{linuxize_path(cleaned_pdb)}' "
                    f"-relax:constrain_relax_to_start_coords "
                    f"-relax:coord_constrain_sidechains "
                    f"-relax:ramp_constraints false "
                    "-score:weights ref2015_cart "
                    "-relax:cartesian "
                    f"-out:path:all '{prep_dir_linux}' "
                    f"-out:file:scorefile '{prep_dir_linux}/score.sc' "
                    "-nstruct 1 -overwrite "
                )

            if params_path and params_path.exists():
                relax_cmd += f" -extra_res_fa '{linuxize_path(params_path)}'"
            run_wsl(relax_cmd, log=sub_dir / "relax.log")
            #run_wsl(f"test -s '{prep_dir_linux}/score.sc' || (echo 'score.sc missing' && false)")

            print(f"✅ Finished minimization for seed {seed}, sample {sample}\n")

        print(f"📁 All per-seed minimizations written to {run_dir}")


    else:
        model_path = _resolve_model_cif(job_dir, job_name, base_job, model_path)
        print(f"📦 Processing AF3 model: {model_path}")

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

        ligand_smiles = job_meta.get("smiles")
        sdf_rdkit = prep_dir / "ligand.sdf"
        sdf_final = sdf_rdkit
        RESNAME = "LIGAND"
        params_path = prep_dir / f"{RESNAME}.params"

        if not ligand_smiles:
            print("ℹ️ No 'smiles' field found — skipping ligand generation (apo model).")
            params_path = None
        else:
            if params_path.exists():
                print(f"⏭️  Found existing params at {params_path}; skipping SDF/params regeneration.")
            else:
                sdf_rdkit = generate_sdf_from_smiles(ligand_smiles, sdf_rdkit)
                params_path = generate_rosetta_params_from_sdf(sdf_final, prep_dir, residue_name=RESNAME)

        print("🧼 Running Rosetta clean_pdb_keep_ligand.py ...")
        cleaned_pdb = prep_dir / "model.pdb_00.pdb"
        clean_cmd = (
            f"cd '{linuxize_path(prep_dir)}' && "
            f"python3 '{CLEAN_PDB_PY}' model.pdb -ignorechain"
        )
        run_wsl(clean_cmd, log=prep_dir / "clean_pdb.log")

        cleaned_pdb_wsl = f"{linuxize_path(prep_dir)}/model.pdb_00.pdb"
        #run_wsl(f"test -s '{cleaned_pdb_wsl}' || (echo '❌ cleaned PDB missing' && false)")
        print(f"✅ Cleaned PDB ready: {cleaned_pdb_wsl}")

        prep_dir_linux = linuxize_path(prep_dir)

        # --- covalent logic unchanged ---
        if job_meta.get("covalent"):
            cleaned_pdb = Path(f"{prep_dir}/model.pdb_00.pdb")
            output_path = cleaned_pdb.with_name(cleaned_pdb.stem + "_renamed.pdb")
            print(output_path)
            target_resnum = int(job_meta.get("residue"))
            resname = job_meta.get("prot_atom")
            if resname == "NZ":
                new_resname = "LYX"
            elif resname == "SG":
                new_resname = "ALA"
            else:
                new_resname = None
            print(f"Target resnum={target_resnum}, prot_atom={resname}, new_resname={new_resname}")
            with open(cleaned_pdb) as f_in, open(output_path, "w") as f_out:
                for line in f_in:
                    if line.startswith(("ATOM", "HETATM")):
                        try:
                            resnum = int(line[22:26])
                        except ValueError:
                            resnum = None
                        atom_name = line[12:16].strip()
                        if resnum == target_resnum and atom_name in ("NZ", "SG"):
                            print(f"Removing atom {atom_name} from residue {resnum}")
                            continue
                        if new_resname and resnum == target_resnum:
                            line = f"{line[:17]}{new_resname:>3s}{line[20:]}"
                    f_out.write(line)
            print(f"✅ Wrote renamed file → {output_path}")
            cleaned_pdb_linux = linuxize_path(output_path)
            relax_cmd = (
                f"'{RELAX_BIN}' "
                f"-database '{ROSETTA_DB}' "
                f"-s '{cleaned_pdb_linux}' "
                f"-relax:constrain_relax_to_start_coords "
                f"-relax:coord_constrain_sidechains "
                f"-relax:ramp_constraints false "
                "-score:weights ref2015_cart "
                f"-relax:cartesian "
                f"-out:path:all '{prep_dir_linux}' "
                f"-out:file:scorefile '{prep_dir_linux}/score.sc' "
                "-nstruct 1 -overwrite "
                f"-extra_res_fa '{ROSETTA_DB}/chemical/residue_type_sets/fa_standard/residue_types/l-caa/LYX.params' "
            )
        else:
            cleaned_pdb_linux = linuxize_path(cleaned_pdb_wsl)
            relax_cmd = (
                f"'{RELAX_BIN}' "
                f"-database '{ROSETTA_DB}' "
                f"-s '{cleaned_pdb_linux}' "
                f"-relax:constrain_relax_to_start_coords "
                f"-relax:coord_constrain_sidechains "
                f"-relax:ramp_constraints false "
                "-score:weights ref2015_cart "
                f"-relax:cartesian "
                f"-out:path:all '{prep_dir_linux}' "
                f"-out:file:scorefile '{prep_dir_linux}/score.sc' "
                "-nstruct 1 -overwrite "
            )

        if params_path and params_path.exists():
            relax_cmd += f" -extra_res_fa '{linuxize_path(params_path)}'"
        run_wsl(relax_cmd, log=prep_dir / "relax.log")
        #run_wsl(f"test -s '{prep_dir_linux}/score.sc' || (echo 'score.sc missing' && false)")

        # ✅ Summary + copy section untouched
        import shutil
        print("\n✅ Done.")
        print(f"  CIF → PDB:           {model_pdb_linux}")
        print(f"  Cleaned PDB:         {cleaned_pdb_wsl}")
        if params_path:
            print(f"  Ligand params:       {params_path}")
        print(f"  Output directory:    {prep_dir}")
        print(f"  Scorefile:           {prep_dir}/score.sc")

        try:
            jobs_root = _user_jobs_root()
            # Write bundle to ~/.af3_pipeline/jobs/<job_name>/ (job_name includes timestamp)
            dest_dir = jobs_root / job_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📁 Copying key results to {dest_dir} ...")

            # --- Copy key AF3 artifacts (prefer whatever exists) ---
            # model + confidences + ranking + metadata
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

            # --- Copy latest rosetta_relax_* folder contents (if present) ---
            relax_folders = sorted(job_dir.glob("rosetta_relax_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if relax_folders:
                latest_relax = relax_folders[0]
                print(f"   Using {latest_relax.name}")

                rosetta_out = dest_dir / latest_relax.name
                rosetta_out.mkdir(parents=True, exist_ok=True)

                # Copy the “canonical” things people care about
                for fname, dest_name in [
                    ("LIGAND.pdb", "LIGAND.pdb"),
                    ("ligand.sdf", "ligand.sdf"),
                    ("LIGAND.params", "LIGAND.params"),
                    ("model.pdb", "model.pdb"),
                    ("model.pdb_00.pdb", "cleaned_af3_model.pdb"),
                    ("model.pdb_00_renamed.pdb", "model.pdb_00_renamed.pdb"),
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
                        # quiet-ish: these are optional
                        print(f"   ⚠️ Missing {latest_relax.name}/{fname}")

                # Copy the relaxed model produced by relax (usually model.pdb_00_*.pdb)
                relaxed_model = next(latest_relax.glob("model.pdb_00_*.pdb"), None)
                if relaxed_model:
                    shutil.copy2(relaxed_model, rosetta_out / "model_relaxed.pdb")
                    print(f"   ✅ Copied {latest_relax.name}/{relaxed_model.name} → {latest_relax.name}/model_relaxed.pdb")

            print("📦 File transfer complete.\n")

        except Exception as e:
            print(f"⚠️ Copying outputs failed: {e}")


# =========================================================
# 🧩 Entry point
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AF3 model & Rosetta ligand params.")
    parser.add_argument("--job", required=True, help="Job folder under af_output (full folder name incl. timestamp)")
    args = parser.parse_args()
    run(AF_OUTPUT_DIR / args.job)
