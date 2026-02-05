#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional
import os
from .cache_utils import compute_hash, get_cache_dir, get_cache_file, exists_in_cache
from .config import cfg
    
def require(cmd: str, install_hint: str = ""):
    if cmd.lower() == "wsl":
        # On Windows, wsl.exe might not resolve via shutil.which() reliably.
        try:
            subprocess.run(["wsl", "--status"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            raise RuntimeError(f"Required binary '{cmd}' not found or not working. {install_hint}".strip())

    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required binary '{cmd}' not found. {install_hint}".strip())


def _to_wsl_path(win_path: str) -> str:
    p = Path(win_path).resolve()
    drive = p.drive[0].lower()
    path_str = str(p).replace("\\", "/")
    if path_str[1:3] == ":/":
        path_str = path_str[3:]
    else:
        path_str = path_str.lstrip("/")
    return f"/mnt/{drive}/{path_str}".replace("//", "/")

def _normalize_db_path_legacy(db_path) -> str:
    p_str = str(db_path).strip()

    if p_str.lower().startswith("\\\\wsl.localhost\\"):
        parts = p_str.split("\\")
        if len(parts) >= 5:
            p_str = "/" + "/".join(parts[4:])

    p_str = p_str.replace("\\", "/")

    if ":" in p_str and not p_str.startswith("/"):
        drive = p_str[0].lower()
        rest = p_str[2:].lstrip("/").replace(":", "")
        p_str = f"/mnt/{drive}/{rest}"

    if not p_str.startswith("/"):
        p_str = "/" + p_str.lstrip("/")

    while "//" in p_str:
        p_str = p_str.replace("//", "/")

    return p_str


def _wsl_exists(path: str) -> bool:
    try:
        subprocess.run(["wsl", "test", "-e", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def _build_msa_wsl_legacy(
    *,
    sequence: str,
    tag: str,
    threads: int,
    sensitivity: float,
    max_seqs: int,
    db_path: Path,
    skip_if_exists: bool,
) -> Path:
    require("wsl", "Ensure WSL is installed and available (wsl.exe).")

    mmseqs_bin = cfg.get("mmseqs_bin", "/usr/local/bin/mmseqs")
    db_wsl = _normalize_db_path_legacy(db_path)

    if not _wsl_exists(db_wsl):
        raise FileNotFoundError(f"MMseqs2 DB not found inside WSL at {db_wsl}")
    else:
        print(f"✅ Found MMseqs2 DB at {db_wsl}")

    shash = compute_hash(sequence)
    msa_a3m = get_cache_file("msa", shash, "msa.a3m")
    sdir = get_cache_dir("msa", shash)

    if skip_if_exists and exists_in_cache("msa", shash, "msa.a3m"):
        try:
            if msa_a3m.stat().st_size > 0:
                print(f"🧬 Using cached MSA for {tag} ({shash})")
                return msa_a3m
            else:
                print(f"⚠️ Cached MSA exists but is empty; regenerating ({tag}, {shash})")
        except Exception:
            pass


    query_fasta = sdir / "query.fasta"
    query_fasta.write_text(f">query\n{sequence}\n", encoding="utf-8")

    wsl_work = f"/tmp/af3_msa_{shash}"
    # Clean and recreate
    subprocess.run(["wsl", "rm", "-rf", wsl_work], check=False)
    subprocess.run(["wsl", "mkdir", "-p", wsl_work], check=True)

    # Copy query.fasta from Windows -> WSL workdir
    qf_win_wsl = _to_wsl_path(str(query_fasta.resolve()))
    qf = f"{wsl_work}/query.fasta"
    subprocess.run(["wsl", "cp", qf_win_wsl, qf], check=True)

    # All MMseqs outputs in WSL-local FS
    qd = f"{wsl_work}/queryDB"
    rd = f"{wsl_work}/resultDB"
    td = f"{wsl_work}/tmp"
    ma = f"{wsl_work}/msa.a3m"
    subprocess.run(["wsl", "mkdir", "-p", td], check=True)

    # 1) createdb
    subprocess.run(["wsl", mmseqs_bin, "createdb", qf, qd], check=True)

    # 2) search
    cmd = [
        "wsl", mmseqs_bin, "search",
        qd, db_wsl, rd, td,
        "--threads", str(threads),
        "-s", str(sensitivity),
        "--max-seqs", str(max_seqs),
    ]
    print("🔍 Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 3) result2msa
    subprocess.run(["wsl", mmseqs_bin, "result2msa", qd, db_wsl, rd, ma], check=True)

    msa_dst_wsl = _to_wsl_path(str(msa_a3m.resolve()))
    subprocess.run(["wsl", "cp", ma, msa_dst_wsl], check=True)

    if not msa_a3m.exists():
        raise RuntimeError(f"MSA finished but msa.a3m not found at {msa_a3m}")

    print(f"✅ MSA generated and cached for {tag} ({shash}) → {msa_a3m}")
    return msa_a3m

def build_msa(
    sequence: str,
    tag: str,
    threads: Optional[int] = None,
    sensitivity: Optional[float] = None,
    max_seqs: Optional[int] = None,
    db_path: Optional[Path] = None,
    skip_if_exists: bool = True,
) -> Path:
    # Pull defaults from config.yaml unless caller overrides
    threads = int(cfg.get("msa.threads", 10)) if threads is None else int(threads)
    sensitivity = float(cfg.get("msa.sensitivity", 5.7)) if sensitivity is None else float(sensitivity)
    max_seqs = int(cfg.get("msa.max_seqs", 25)) if max_seqs is None else int(max_seqs)

    print("DEBUG cfg.path =", getattr(cfg, "path", None))
    print("DEBUG env AF3_PIPELINE_CONFIG =", os.environ.get("AF3_PIPELINE_CONFIG"))
    print("DEBUG msa.db =", cfg.get("msa.db", None))
    print("DEBUG msa dict =", cfg.get("msa", None))

    if db_path is None:
        raw = (cfg.get("msa.db", "") or "").strip()
        if not raw:
            raise RuntimeError(
                "Missing config value msa.db. Set it to something like "
                "/home/<user>/Repositories/alphafold/mmseqs_db (or the full DB dir)."
            )
        base = Path(raw)

        # Accept either:
        #  - msa.db = /path/to/mmseqs_db
        #  - msa.db = /path/to/mmseqs_db/uniref30_2302_db
        cand1 = base
        cand2 = base / "uniref30_2302_db"

        # Choose the more specific one if it exists (best-effort; existence checked inside WSL later too)
        db_path = cand2 if str(base).endswith("mmseqs_db") else cand1

    return _build_msa_wsl_legacy(
        sequence=sequence,
        tag=tag,
        threads=threads,
        sensitivity=sensitivity,
        max_seqs=max_seqs,
        db_path=Path(db_path),
        skip_if_exists=skip_if_exists,
    )


def use_existing_msa(msa_path: Path, tag: str) -> Path:
    """Use an existing precomputed MSA (bypassing MMseqs2 entirely)."""
    msa_path = Path(msa_path).expanduser().resolve()
    if not msa_path.exists():
        raise FileNotFoundError(f"❌ Precomputed MSA not found at {msa_path}")
    print(f"🧬 Using precomputed MSA for {tag}: {msa_path}")
    return msa_path
