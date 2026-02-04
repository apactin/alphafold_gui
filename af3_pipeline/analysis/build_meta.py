#!/usr/bin/env python3
"""
build_meta.py — Merge GUI and AF3 outputs into unified metadata
===============================================================

Reads:
  • job_metadata.json
  • fold_input.json
  • *_model.cif
  • *_confidences.json
  • *_summary_confidences.json

Writes:
  • prepared_meta.json — unified metadata record
  • build_meta.log — diagnostics
"""

import json, platform, subprocess, re
from datetime import datetime
from pathlib import Path
from af3_pipeline.config import cfg

LINUX_HOME = cfg.get("linux_home_root", "")

AF3_DIR = Path(str(cfg.get("af3_dir", f"{LINUX_HOME}/Repositories/alphafold")))
AF_INPUT_DIR = Path(str(cfg.get("af3_input_dir", str(AF3_DIR / "af_input"))))
AF_OUTPUT_DIR = Path(str(cfg.get("af3_output_dir", str(AF3_DIR / "af_output"))))

def safe_read_json(path: Path, default=None):
    if not path.exists():
        return default or {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to parse {path.name}: {e}")
        return default or {}

def safe_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_gpu_name() -> str:
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        return res[0] if res else "Unknown GPU"
    except Exception:
        return "N/A"

def _trim_timestamp(job_name: str) -> str:
    """Return the plain job prefix if it includes a timestamp suffix."""
    return re.sub(r"_[0-9]{8}[-_][0-9]{6}$", "", job_name)

def run(job_dir: str | Path, model_path: str | Path = None, meta: dict | None = None) -> dict:
    job_dir = Path(job_dir)
    print(job_dir)
    job_name = job_dir.name
    base_job = _trim_timestamp(job_name)
    print(base_job)
    
    fold_input_path = AF_INPUT_DIR / f"{base_job}_fold_input.json"
    job_meta_path = job_dir / "job_metadata.json"
    print(job_meta_path)
    # Resolve model path
    if model_path:
        model_path = Path(model_path)
    else:
        # Prefer <base_job>_model.cif in this job_dir
        cand = job_dir / f"{base_job}_model.cif"
        model_path = cand if cand.exists() else next(job_dir.glob("*_model.cif"), job_dir / f"{job_name}_model.cif")

    conf_path = job_dir / f"{base_job}_confidences.json"
    summary_path = job_dir / f"{base_job}_summary_confidences.json"
    out_path = job_dir / "prepared_meta.json"
    print(out_path)
    log_path = job_dir / "build_meta.log"

    log = [f"🧱 build_meta started for {job_name}\n"]

    job_metadata = safe_read_json(job_meta_path)
    fold_input = safe_read_json(fold_input_path)
    conf_data = safe_read_json(conf_path)
    summary_data = safe_read_json(summary_path)

    for name, found in [
        ("job_metadata.json", bool(job_metadata)),
        ("fold_input.json", bool(fold_input)),
        ("_confidences.json", bool(conf_data)),
        ("_summary_confidences.json", bool(summary_data)),
    ]:
        log.append(f"📄 {name}: {'found' if found else 'missing'}\n")

    af3_confidence = {}
    for src in (conf_data, summary_data):
        if not src:
            continue
        keys = ["plddt", "atom_plddts", "ptm", "iptm", "ranking_score", "fraction_disordered"]
        for k in keys:
            if k in src:
                val = src[k]
                if isinstance(val, list) and val:
                    af3_confidence[f"mean_{k}"] = round(sum(val) / len(val), 3)
                else:
                    af3_confidence[k] = val

    sys_info = {
        "gpu": get_gpu_name(),
        "os": platform.platform(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    ligand_block = {}
    ligand_block.update(meta or {})
    ligand_block.update(job_metadata or {})

    meta_out = {
        "job_name": job_name,
        "base_job": base_job,
        "paths": {
            "model": str(model_path),
            "fold_input": str(fold_input_path),
            "job_metadata": str(job_meta_path),
        },
        "ligand": ligand_block,
        "af3_confidence": af3_confidence,
        "system": sys_info,
    }

    safe_write_json(out_path, meta_out)
    log.append(f"✅ prepared_meta.json written to {out_path}\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log)

    print(f"✅ Metadata build complete for '{job_name}'")
    return meta_out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build unified AF3 metadata record")
    parser.add_argument("--job", required=True, help="Job folder name under af_output")
    parser.add_argument("--model", help="Optional path to AF3 model CIF")
    args = parser.parse_args()

    job_dir = AF_OUTPUT_DIR / args.job
    model_path = args.model or (job_dir / f"{args.job}_model.cif")
    run(job_dir, model_path)
