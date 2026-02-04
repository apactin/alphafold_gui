#!/usr/bin/env python3
"""
runner.py — Portable orchestration for AF3 (+ optional analysis)
===============================================================
Portable-first:
- All job I/O goes under workspace/jobs/<job_id>/
- Docker containers mount the workspace at /workspace
- AF3 writes outputs to /workspace/jobs/<job_id>/output

Legacy mode:
- Preserves your current WSL UNC + WSL docker behavior when backend_mode=wsl_legacy
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
import re
import json

from af3_pipeline.config import cfg

# ==============================
# ⚙️ Config-driven paths
# ==============================
DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")

# Canonical Linux base inside WSL (used by Docker & AF3)
BASE_LINUX = cfg.get("af3_dir", "")
BASE_LINUX = BASE_LINUX.replace("\\", "/").rstrip("/")

# Allow explicit overrides; fall back to previous BASE_LINUX-derived defaults
AF_INPUT_LINUX   = cfg.get("af3_input_dir", f"{BASE_LINUX}/af_input")
AF_OUTPUT_LINUX  = cfg.get("af3_output_dir", f"{BASE_LINUX}/af_output")
AF_WEIGHTS_LINUX = cfg.get("af3_weights_dir", f"{BASE_LINUX}/af_weights")
AF_CODE_LINUX    = cfg.get("af3_code_dir", f"{BASE_LINUX}/alphafold3")

AF_INPUT_LINUX   = str(AF_INPUT_LINUX).replace("\\", "/").rstrip("/")
AF_OUTPUT_LINUX  = str(AF_OUTPUT_LINUX).replace("\\", "/").rstrip("/")
AF_WEIGHTS_LINUX = str(AF_WEIGHTS_LINUX).replace("\\", "/").rstrip("/")
AF_CODE_LINUX    = str(AF_CODE_LINUX).replace("\\", "/").rstrip("/")

# ==============================
# 🐳 Docker config
# ==============================
DOCKER_BIN = cfg.get("docker_bin", "docker")
AF3_DOCKER_IMAGE = cfg.get("alphafold_docker_image", "alphafold3")

AF3_DOCKER_ENV = cfg.get("alphafold_docker_env", {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_CLIENT_MEM_FRACTION": ".50",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    "TF_FORCE_UNIFIED_MEMORY": "1",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
})

def _user_jobs_root() -> Path:
    # Prefer explicit config; otherwise default to ~/.af3_pipeline/jobs
    root = cfg.get("jobs_root", None)
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".af3_pipeline" / "jobs").resolve()

def copy_af3_outputs_to_jobs_root(*, job_dir: Path, job_name: str) -> Path:
    jobs_root = _user_jobs_root()
    dest_dir = jobs_root / job_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src in job_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, dest_dir / src.name)

    return dest_dir

# ==============================
# ✅ WSL UNC Path Utility (Windows ↔ WSL)
# ==============================
def wsl_path(subpath: str) -> Path:
    # Normalize the subpath to forward slashes without leading slash
    sub = (subpath or "").replace("\\", "/").lstrip("/")

    base_win = BASE_LINUX.replace("/", "\\")
    full = f"\\\\wsl.localhost\\{DISTRO_NAME}{base_win}"
    if sub:
        full += "\\" + sub.replace("/", "\\")

    # Ensure the UNC path starts with exactly two backslashes
    if not full.startswith("\\\\"):
        full = "\\" + full

    return Path(full)

# Windows-visible paths to AF input/output on WSL
AF_INPUT_DIR  = wsl_path("af_input")
AF_OUTPUT_DIR = wsl_path("af_output")

# Ensure these UNC dirs exist from Windows side
def ensure_dirs():
    for d in [AF_INPUT_DIR, AF_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ==============================
# 🔑 Utility helpers
# ==============================
def seq_hash(name: str) -> str:
    """Deterministic short hash for Docker container name."""
    return hashlib.sha1(name.encode()).hexdigest()[:8]

def get_job_metadata(job_name: str) -> dict:
    meta_path = AF_OUTPUT_DIR / job_name / "job_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Failed to read job metadata: {e}")
    return {}

def _get_most_recent_output_folder(job_name: str) -> Path | None:
    base = AF_OUTPUT_DIR
    candidates = [p for p in base.glob(f"{job_name}_*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def _copy_metadata_to_latest_output(job_name: str) -> Path | None:
    """Copy job_metadata.json from <job> → latest timestamped AF3 folder."""
    src_meta = AF_OUTPUT_DIR / job_name / "job_metadata.json"
    if not src_meta.exists():
        print(f"⚠️ No job_metadata.json found in {src_meta}")
        return None

    dest_dir = _get_most_recent_output_folder(job_name)
    if not dest_dir or not dest_dir.exists():
        print(f"⚠️ No timestamped AF3 output folder found for {job_name}")
        return None

    dst_meta = dest_dir / "job_metadata.json"
    try:
        shutil.copy2(src_meta, dst_meta)
        print(f"🧩 Copied job_metadata.json → {dst_meta}")
        return dst_meta
    except Exception as e:
        print(f"⚠️ Failed to copy metadata: {e}")
        return None

# ==============================
# 🐳 Docker command builder
# ==============================
def docker_cmd(container_name: str, json_linux_path: str,
               num_recycles: int = 1, seeds: int = 2,
               gpu: str = "all", extra_mounts=None):
    import os
    uid = subprocess.check_output(["wsl", "id", "-u"]).decode().strip()
    gid = subprocess.check_output(["wsl", "id", "-g"]).decode().strip()

    # --- Derive container-visible relative path
    json_filename = os.path.basename(json_linux_path)
    json_container_path = f"/work/af_input/{json_filename}"

    cmd = [
        "wsl",
        DOCKER_BIN, "run", "-t", "--rm",
        "--name", container_name,
        f"--user={uid}:{gid}",
        "--gpus", gpu,
        "--volume", f"{AF_INPUT_LINUX}:/work/af_input",
        "--volume", f"{AF_OUTPUT_LINUX}:/work/af_output",
        "--volume", f"{AF_WEIGHTS_LINUX}:/work/models",
        "--volume", f"{AF_CODE_LINUX}:/work/alphafold3",
    ]

    # Config-driven env (defaults match your current hardcoded set)
    if isinstance(AF3_DOCKER_ENV, dict):
        for k, v in AF3_DOCKER_ENV.items():
            cmd.extend(["-e", f"{k}={v}"])

    cmd.extend([
        AF3_DOCKER_IMAGE,
        "python", "run_alphafold.py",
        f"--json_path={json_container_path}",
        "--model_dir=/work/models",
        "--output_dir=/work/af_output",
        "--norun_data_pipeline",
        f"--num_recycles={num_recycles}",
    ])

    if extra_mounts:
        for mnt in extra_mounts:
            cmd.extend(["--volume", mnt])

    return cmd


# ==============================
# 🧠 AlphaFold-3 Runner
# ==============================
def run_af3(json_path=None, job_name="GUI_job", num_recycles=1,
            seeds=2, gpu="all", extra_mounts=None, auto_analyze=False, multi_seed=False):
    """
    Run AlphaFold-3 in Docker and optionally post-process results.

    json_path:
      • If a Linux path (starts with "/"), it's assumed to point inside WSL at AF_INPUT_LINUX.
      • If a UNC/Windows path, it's converted to the matching Linux path under BASE_LINUX.
    """
    # Normalize json_path to Linux path string for Docker
    json_linux = None
    ensure_dirs()

    if json_path is None:
        json_linux = f"{AF_INPUT_LINUX}/{job_name}_fold_input.json"
    else:
        if isinstance(json_path, Path):
            json_path = str(json_path)

        if isinstance(json_path, str) and json_path.startswith("/"):
            # Already a Linux path (e.g. from json_builder log)
            json_linux = json_path
        else:
            # Treat as Windows/UNC path and map into Linux space if it lives under our WSL mount
            p = Path(json_path)
            s = str(p)
            # Strip UNC prefix if present
            s_clean = s.replace(r"\\wsl.localhost\\" + DISTRO_NAME, "")
            s_clean = s_clean.replace("\\", "/")
            if "/home/" in s_clean:
                json_linux = s_clean
            else:
                # Fallback: assume standard AF_INPUT layout
                json_linux = f"{AF_INPUT_LINUX}/{job_name}_fold_input.json"

    print(f"📄 Normalized JSON path for WSL: {json_linux}")

    # For sanity, see if the file exists via UNC path; do not block run if it's only missing from Windows view.
    rel_under_base = json_linux.replace(BASE_LINUX + "/", "")
    json_unc = wsl_path(rel_under_base)
    if not json_unc.exists():
        print(f"⚠️ Warning: JSON not visible at {json_unc} from Windows; "
              "assuming it exists inside WSL if just written there.")

    job_out = AF_OUTPUT_DIR / job_name
    job_out.mkdir(parents=True, exist_ok=True)

    container_name = f"af3_{seq_hash(job_name)}"
    cmd = docker_cmd(
        container_name,
        json_linux_path=json_linux,
        num_recycles=num_recycles,
        seeds=seeds,
        gpu=gpu,
        extra_mounts=extra_mounts,
    )

    print(f"🚀 Launching AF3 job '{job_name}'")
    print(f"🐳 Container: {container_name}")
    print(f"📄 Input JSON: {json_linux}")
    print(f"📤 Output Dir: {job_out}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ AF3 failed:")
        print(result.stdout)
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    print(f"✅ AF3 job '{job_name}' completed successfully.")

    # Copy metadata to timestamped output folder
    _copy_metadata_to_latest_output(job_name)
    print(f"✅ Metadata for '{job_name}' copied to most recent job folder: '{_get_most_recent_output_folder(job_name)}.")

    # Optional post-AF3 analysis
    if auto_analyze:
        try:
            latest = _get_most_recent_output_folder(job_name)
            if not latest:
                print(f"⚠️ No timestamped output found for {job_name}, skipping analysis.")
                return

            # If get_latest_af3_output returned a model file, extract folder & name
            if latest.is_file():
                cif = latest
                af_out = latest.parent
            else:
                af_out = latest
                base_name = re.sub(r"_[0-9]{8}[-_][0-9]{6}$", "", af_out.name)
                cif = af_out / f"{base_name}_model.cif"

            if cif.exists():
                print(f"🧠 Auto-analysis: Running post-analysis for {job_name}")
                cmd = ["python", "-m", "af3_pipeline.analysis.post_analysis",
                    "--job", af_out.name,
                    "--model", str(cif)]
                if multi_seed:
                    cmd.append("--multi_seed", multi_seed)
                subprocess.run(cmd, check=True)
            else:
                print(f"⚠️ Auto-analysis skipped: CIF not found ({cif})")

        except Exception as e:
            print(f"⚠️ Auto-analysis failed: {e}")
    else:
            try:
                latest = _get_most_recent_output_folder(job_name)
                if not latest:
                    print(f"⚠️ No timestamped output found for {job_name}; nothing to copy.")
                    return

                # Copy from the real AF3 output folder (job_name_YYYYMMDD_HHMMSS)
                job_dir = latest.parent if latest.is_file() else latest

                # Use the timestamped folder name so bundles match AF3 outputs
                dest_dir = copy_af3_outputs_to_jobs_root(job_dir=job_dir, job_name=job_dir.name)
                print(f"📦 Copied AF3 outputs → {dest_dir}")
            except Exception as e:
                print(f"⚠️ Copying AF3 outputs failed: {e}")


# ==============================
# 🚀 CLI entry point
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Run AlphaFold-3 and optional post-analysis")
    parser.add_argument("--job", help="Job name (required)")
    parser.add_argument("--json_path", help="Path to fold_input.json")
    parser.add_argument("--num_recycles", type=int, default=int(cfg.get("alphafold_default_recycles", 1)))
    parser.add_argument("--seeds", type=int, default=int(cfg.get("alphafold_default_seeds", 1)))
    parser.add_argument("--gpu", default=cfg.get("alphafold_default_gpu", "all"))
    parser.add_argument("--extra_mount", action="append", help="Additional Docker volume mounts")
    parser.add_argument("--auto_analyze", action="store_true", help="Run analysis after AF3")
    args = parser.parse_args()

    job_name = args.job or input("Enter job name: ").strip()
    if not job_name:
        print("❌ --job is required")
        sys.exit(1)

    run_af3(
        json_path=args.json_path,
        job_name=job_name,
        num_recycles=args.num_recycles,
        seeds=args.seeds,
        gpu=args.gpu,
        extra_mounts=args.extra_mount,
        auto_analyze=args.auto_analyze,
    )

if __name__ == "__main__":
    main()
