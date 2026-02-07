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
import os
import sys
from pathlib import Path
from typing import Optional
import re
import json

from af3_pipeline.config import cfg

DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")

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
        dest = dest_dir / src.name

        if src.is_file():
            shutil.copy2(src, dest)

        else:  # copy any folder
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

    return dest_dir

def _get_linux_paths():
    distro = cfg.get("wsl_distro", "Ubuntu-22.04")

    base = (cfg.get("af3_dir", "") or "").replace("\\", "/").rstrip("/")
    af_input   = str(cfg.get("af3_input_dir",   f"{base}/af_input")).replace("\\","/").rstrip("/")
    af_output  = str(cfg.get("af3_output_dir",  f"{base}/af_output")).replace("\\","/").rstrip("/")
    af_weights = str(cfg.get("af3_weights_dir", f"{base}/af_weights")).replace("\\","/").rstrip("/")
    af_code    = str(cfg.get("af3_code_dir",    f"{base}/alphafold3")).replace("\\","/").rstrip("/")

    return distro, base, af_input, af_output, af_weights, af_code

def linux_to_unc(linux_path: str) -> Path:
    distro, *_ = _get_linux_paths()
    lp = (linux_path or "").replace("\\", "/").strip()
    if not lp.startswith("/"):
        lp = "/" + lp
    win_tail = lp.lstrip("/").replace("/", "\\")
    return Path(f"\\\\wsl.localhost\\{distro}\\{win_tail}")


# ==============================
# ✅ WSL UNC Path Utility (Windows ↔ WSL)
# ==============================
def wsl_path(subpath: str) -> Path:
    distro, base_linux, *_ = _get_linux_paths()
    sub = (subpath or "").replace("\\", "/").lstrip("/")
    base_win = base_linux.replace("/", "\\")
    full = f"\\\\wsl.localhost\\{distro}{base_win}"
    if sub:
        full += "\\" + sub.replace("/", "\\")
    if not full.startswith("\\\\"):
        full = "\\" + full
    return Path(full)

# Ensure these UNC dirs exist from Windows side
def ensure_dirs():
    # Make sure WSL dirs exist (via WSL) rather than relying on UNC mkdir
    distro, base, af_in, af_out, *_ = _get_linux_paths()
    subprocess.run(["wsl.exe", "bash", "-lc", f"mkdir -p '{af_in}' '{af_out}'"], check=False)

def _af_output_unc() -> Path:
    _, _, _, af_out, _, _ = _get_linux_paths()
    return linux_to_unc(af_out)

def _af_input_unc() -> Path:
    _, _, af_in, _, _, _ = _get_linux_paths()
    return linux_to_unc(af_in)

# ==============================
# 🔑 Utility helpers
# ==============================
def seq_hash(name: str) -> str:
    """Deterministic short hash for Docker container name."""
    return hashlib.sha1(name.encode()).hexdigest()[:8]

def get_job_metadata(job_name: str) -> dict:
    latest = _get_most_recent_output_folder(job_name)
    if not latest:
        return {}
    meta_path = latest / "job_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Failed to read job metadata: {e}")
    return {}

def _get_most_recent_output_folder(job_name: str) -> Path | None:
    job = (job_name or "").replace("'", "")
    _, _, _, af_out, _, _ = _get_linux_paths()

    # Find dirs in af_out that start with job name, sort by mtime, return newest.
    bash = (
        f"find '{af_out}' -maxdepth 1 -mindepth 1 -type d -name '{job}*' "
        r"-printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-"
    )

    p = subprocess.run(["wsl.exe", "bash", "-lc", bash], capture_output=True, text=True)

    if p.returncode != 0:
        # optional debug:
        # print("DEBUG find stderr:", p.stderr)
        return None

    latest_linux = (p.stdout or "").strip()
    if not latest_linux:
        # optional debug:
        # print("DEBUG: no match. stderr:", p.stderr)
        return None

    return linux_to_unc(latest_linux)



# ==============================
# 🐳 Docker command builder
# ==============================
def docker_cmd(container_name: str, json_linux_path: str,
               num_recycles: int = 1, seeds: int = 2,
               gpu: str = "all", extra_mounts=None):
    uid = subprocess.check_output(["wsl", "id", "-u"]).decode().strip()
    gid = subprocess.check_output(["wsl", "id", "-g"]).decode().strip()

    # --- Derive container-visible relative path
    json_filename = os.path.basename(json_linux_path)
    json_container_path = f"/work/af_input/{json_filename}"

    distro, base, af_in, af_out, af_w, af_code = _get_linux_paths()

    cmd = [
        "wsl",
        DOCKER_BIN, "run", "-t", "--rm",
        "--name", container_name,
        f"--user={uid}:{gid}",
        "--gpus", gpu,
        "--volume", f"{af_in}:/work/af_input",
        "--volume", f"{af_out}:/work/af_output",
        "--volume", f"{af_w}:/work/models",
        "--volume", f"{af_code}:/work/alphafold3",
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

    distro, base, af_in, af_out, af_w, af_code = _get_linux_paths()
    ensure_dirs()

    if json_path is None:
        json_linux = f"{af_in}/{job_name}_fold_input.json"
    else:
        if isinstance(json_path, Path):
            json_path = str(json_path)

        if isinstance(json_path, str) and json_path.startswith("/"):
            json_linux = json_path
        else:
            s = str(Path(json_path))

            m = re.match(r"^\\\\wsl\.localhost\\([^\\]+)\\(.*)$", s, flags=re.IGNORECASE)
            if m:
                s_clean = "/" + m.group(2).replace("\\", "/")
            else:
                s_clean = s.replace("\\", "/")

            if s_clean.startswith("/"):
                json_linux = s_clean
            else:
                json_linux = f"{af_in}/{job_name}_fold_input.json"

    print(f"📄 Normalized JSON path for WSL: {json_linux}")

    # For sanity, see if the file exists via UNC path; do not block run if it's only missing from Windows view.
    json_unc = linux_to_unc(json_linux)
    if not json_unc.exists():
        print(f"⚠️ Warning: JSON not visible at {json_unc} from Windows; assuming it exists inside WSL if just written there.")

    container_name = f"af3_{seq_hash(job_name)}"
    cmd = docker_cmd(
        container_name,
        json_linux_path=json_linux,
        num_recycles=num_recycles,
        seeds=seeds,
        gpu=gpu,
        extra_mounts=extra_mounts,
    )

    distro, base, af_in, af_out, af_w, af_code = _get_linux_paths()
    print(f"🧪 Mount check: weights host path (WSL) = {af_w}")
    print(f"🧪 Mount check: base = {base}")

    print(f"🚀 Launching AF3 job '{job_name}'")
    print(f"🐳 Container: {container_name}")
    print(f"📄 Input JSON: {json_linux}")
    print(f"📤 Output Dir (WSL): {af_out}")
    print(f"📤 Output Dir (UNC): {_af_output_unc()}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ AF3 failed:")
        print(result.stdout)
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    print(f"✅ AF3 job '{job_name}' completed successfully.")

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
                    cmd.extend(["--multi_seed", str(multi_seed)])
                subprocess.run(cmd, check=True)
            else:
                print(f"⚠️ Auto-analysis skipped: CIF not found ({cif})")

        except Exception as e:
            print(f"⚠️ Auto-analysis failed: {e}")
    else:
        try:
            latest = _get_most_recent_output_folder(job_name)
            if not latest:
                print(f"⚠️ No timestamped output found for {job_name}; nothing to copy.4")
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
