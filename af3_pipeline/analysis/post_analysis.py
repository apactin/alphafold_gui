#!/usr/bin/env python3
"""
post_analysis.py — Unified Rosetta + AF3 Post-Processing Pipeline
=================================================================
Orchestrates all post-AF3 analysis steps:
  1️⃣ build_meta.py         → writes meta.json summarizing inputs & outputs
  2️⃣ rosetta_minimize.py    → runs Cartesian minimization
  3️⃣ metrics.py         → aggregates Rosetta + AF3 scores into metrics.json
"""

import sys
import traceback
import importlib
from pathlib import Path
import io, os, platform
from af3_pipeline.config import cfg
import re

# Force UTF-8 output for WSL/Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Dynamically imported analysis steps
SUBMODULES = {
    "build_meta":        "af3_pipeline.analysis.build_meta",
    "rosetta_relax": "af3_pipeline.analysis.rosetta_relax",     
    "rosetta_scripts": "af3_pipeline.analysis.rosetta_scripts",
    "metrics":           "af3_pipeline.analysis.metrics",
}

# ============================
# 🔧 Config-based paths
# ============================
DISTRO = cfg.get("wsl_distro", "Ubuntu-22.04")
BASE_LINUX = cfg.get("af3_dir", "")
AF_OUTPUT_DIR = Path(cfg.get("af3_output_dir", f"{BASE_LINUX}/af_output"))

def _user_jobs_root() -> Path:
    root = cfg.get("jobs_root", None)
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".af3_pipeline" / "jobs").resolve()

def _bundle_dir_for_job(resolved_job_name: str) -> Path:
    return _user_jobs_root() / resolved_job_name

def to_wsl_path(subpath: str) -> Path:
    """Return correct path depending on OS (UNC on Windows, /home/... on Linux)."""
    sub = subpath.replace("\\", "/").strip("/")
    if platform.system() == "Windows":
        unc_prefix = f"\\\\wsl.localhost\\{DISTRO}"
        base_win = BASE_LINUX.replace("/", "\\")
        return Path(unc_prefix + base_win + ("\\" + sub.replace("/", "\\")) if sub else "")
    else:
        return Path(BASE_LINUX + ("/" + sub if sub else ""))
    
_TS_RE = re.compile(r"_[0-9]{8}[-_][0-9]{6}$")

def _trim_timestamp(name: str) -> str:
    return re.sub(r"_[0-9]{8}[-_][0-9]{6}$", "", name)

def _resolve_job_dir(job_arg: str) -> tuple[str, str, Path]:
    """
    Returns (base_name, resolved_job_name, resolved_job_dir).
    If job_arg is base name and timestamped dirs exist, prefer newest timestamped.
    """
    base = _trim_timestamp(job_arg)
    root = to_wsl_path("af_output")

    is_timestamped = bool(_TS_RE.search(job_arg))

    # If user already passed timestamped folder name, use it directly
    if is_timestamped:
        direct = root / job_arg
        return base, job_arg, direct

    # Otherwise: prefer newest timestamped folder if it exists
    candidates = [p for p in root.glob(f"{base}_*") if p.is_dir()]
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return base, latest.name, latest

    # Fallback: base folder
    return base, job_arg, (root / job_arg)


# ============================
# 🔩 Utilities
# ============================
def _safe_import(module_path: str):
    """Import a submodule and return it (raise if missing)."""
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(f"❌ Failed to import {module_path}: {e}") from e

def _ensure_callable(mod, func="run"):
    if not hasattr(mod, func):
        raise AttributeError(f"Module {mod.__name__} does not define `{func}()`")
    return getattr(mod, func)

def _check_for_custom_constraints():
    constraints_set  = cfg.get("rosetta.constraints_set", "")
    constraints_file = cfg.get("rosetta.constraints_file", "")
    try:
        if constraints_file:
            print(f"Found custom constraints: {constraints_file}")

            return constraints_file
        else:
            print(f"No custom constraints found.")
            return
    except Exception:
        pass



# ============================
# 🧠 Main analysis pipeline
# ============================
def run_analysis(
    job_name: str,
    model_path: str | Path | None,
    multi_seed: bool = False,
    meta: dict | None = None,
    skip_rosetta: bool = False,
    skip_rosetta_ligand: bool = False,
    constraints_file: str | None = None,
):
    base_name, resolved_job_name, job_dir = _resolve_job_dir(job_name)
    bundle_dir = _bundle_dir_for_job(resolved_job_name)
    cst_path: Path | None = None

    try:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "analysis_flags.txt").write_text(
            f"multi_seed={multi_seed}\nskip_rosetta={skip_rosetta}\nskip_rosetta_scripts={skip_rosetta_ligand}",
            encoding="utf-8",
        )
    except Exception:
        pass
    cst_path = _check_for_custom_constraints()
    print(f"Constraints: {cst_path}")
    print(f"post analysis job dir: {job_dir}")
    job_dir.mkdir(parents=True, exist_ok=True)
    print(f"Skip RosettaRelax: {skip_rosetta}")
    print(f"Skip RosettaLigand: {skip_rosetta_ligand}")

    # Normalize model_path:
    # - None / "" / whitespace => treat as not provided
    # - relative paths => resolve relative to job_dir
    mp = None
    if model_path is not None:
        s = str(model_path).strip()
        if s:
            p = Path(s)
            mp = (job_dir / p) if not p.is_absolute() else p

    # If model_path wasn't provided or doesn't exist, default to <base>_model.cif in resolved job_dir
    if mp is None or not mp.exists():
        candidate = job_dir / f"{base_name}_model.cif"
        mp = candidate if candidate.exists() else mp  # keep mp if user passed something real

    print(f"🧠 Starting unified post-AF3 analysis for '{resolved_job_name}'")
    print(f"📄 Model: {mp if mp else '(auto-resolve in rosetta_minimize)'}")
    if meta:
        print(f"🗂️  Metadata: {len(meta)} keys")
    print(f"Using multi_seed mode: {multi_seed}")

    # Step 1 — build_meta
    try:
        mod = _safe_import(SUBMODULES["build_meta"])
        func = _ensure_callable(mod)
        print("📦 Building meta.json …")
        func(job_dir, mp, meta)
    except Exception:
        traceback.print_exc()
        print("⚠️ build_meta failed, continuing …")


    # Step 2 — rosetta_relax (formerly rosetta_minimize)
    ran_relax = False
    if skip_rosetta:
        print("⏭️  Skipping RosettaRelax (--skip_rosetta).")
    else:
        try:
            mod = _safe_import(SUBMODULES["rosetta_relax"])  # update SUBMODULES key
            func = _ensure_callable(mod)
            print("⚙️ Running RosettaRelax …")
            # rosetta_relax signature should match: run(job_dir, multi_seed=..., model_path=...)
            func(job_dir, multi_seed=multi_seed, model_path=mp)
            ran_relax = True
        except Exception:
            traceback.print_exc()
            print("⚠️ rosetta_relax failed, continuing …")

    # Step 3 — rosetta_scripts (formerly rosetta_ligand)
    if skip_rosetta_ligand:
        print("⏭️  Skipping RosettaScripts (--skip_rosetta_ligand).")
    else:
        try:
            mod = _safe_import(SUBMODULES["rosetta_scripts"])  # update SUBMODULES key
            func = _ensure_callable(mod)
            print("🧲 Running RosettaScripts …")

            # Key logic:
            # - If we ran relax in this run (or skip_rosetta=False), scripts should use latest_rosetta_relax.json
            # - If skip_rosetta=True, scripts should run AF3-direct
            #
            # rosetta_scripts.run(..., skip_rosetta=bool, multi_seed=bool, yaml_path=optional)
            #
            # If you have a yaml path in cfg, pass it; otherwise omit.
            yaml_path = cfg.get("rosetta_dicts_yaml", None)

            func(
                job_dir,
                skip_rosetta=bool(skip_rosetta or (not ran_relax)),
                multi_seed=multi_seed,
                yaml_path=yaml_path,
                constraints_file=str(cst_path) if cst_path else None, 
            )

        except Exception:
            traceback.print_exc()
            print("⚠️ rosetta_scripts failed, continuing …")

    # Step 4 — metrics
    try:
        mod = _safe_import(SUBMODULES["metrics"])
        func = _ensure_callable(mod)
        print("📊 Computing metrics …")

        metrics_target = bundle_dir if bundle_dir.exists() else job_dir
        if metrics_target != job_dir:
            print(f"📦 Using bundled job dir for metrics: {metrics_target}")

        func(metrics_target, multi_seed)
    except Exception:
        traceback.print_exc()
        print("⚠️ metrics failed.")

    print(f"✅ Post-AF3 analysis complete for '{job_name}'")


# ============================
# 🧩 CLI entry point
# ============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run unified post-AF3 Rosetta analysis")
    parser.add_argument("--job", required=True, help="Job name (base or timestamped)")
    parser.add_argument("--model", required=False, help="Path to model CIF/PDB (optional)")
    parser.add_argument("--multi_seed", action="store_true", help="Analyze top model per seed")
    parser.add_argument("--skip_rosetta", action="store_true", help="Skip RosettaRelax")
    parser.add_argument("--skip_rosetta_ligand", action="store_true", help="Skip RosettaLigand")
    parser.add_argument("--constraints", required=False, help="Path to constraints file (optional)")
    args = parser.parse_args()

    # Let run_analysis resolve job_dir and default model correctly.
    run_analysis(args.job, args.model, args.multi_seed, skip_rosetta=args.skip_rosetta, skip_rosetta_ligand=args.skip_rosetta_ligand, constraints_file=args.constraints,)
