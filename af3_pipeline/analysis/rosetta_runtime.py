# rosetta_runtime.py
"""
Shared runtime utilities for Rosetta pipeline modules.

Use this from:
  - rosetta_relax.py
  - rosetta_scripts.py
  - prep_lig_for_rosetta.py (optional)

Goals:
- One authoritative linuxize_path() so JSON pointers + extra_res_fa behave consistently
- One authoritative run_wsl() so distro selection + logging behave consistently
- Minimal surface area (tiny, predictable, easy to test)
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from af3_pipeline.config import cfg

# -------------------------
# WSL config
# -------------------------
DISTRO_NAME = cfg.get("wsl_distro", "Ubuntu-22.04")
LINUX_HOME = cfg.get("linux_home_root", "").strip()
WSL_EXE = r"C:\Windows\System32\wsl.exe"


def linuxize_path(p: Path) -> str:
    """
    Convert UNC/Windows paths → WSL Linux paths.

    Handles:
      - \\wsl.localhost\\<DISTRO>\\home\\...  -> /home/...
      - C:\\Users\\...                       -> /mnt/c/Users/...
    Leaves already-linux absolute paths untouched.
    """
    s = str(p)

    # UNC from WSL mount
    s = s.replace(f"\\\\wsl.localhost\\{DISTRO_NAME}", "")
    s = s.replace("\\", "/")

    # Drive mapping
    s = s.replace("C:/Users", "/mnt/c/Users").replace("c:/Users", "/mnt/c/Users")

    # If still not an absolute linux path, try to anchor under linux home root
    if not (s.startswith("/home") or s.startswith("/mnt") or s.startswith("/tmp") or s.startswith("/var")):
        if not s.startswith("/"):
            if LINUX_HOME:
                s = f"{LINUX_HOME}/" + s.lstrip("/")
            else:
                # last resort: treat as relative to /
                s = "/" + s.lstrip("/")

    return s

def wsl_unc_from_linux(linux_abs: str) -> Path:
    """
    Map an absolute Linux path (/home/... or /mnt/...) to a Windows UNC path:
      \\wsl.localhost\\<distro>\\home\\...
    On non-Windows, returns Path(linux_abs).
    """
    s = (linux_abs or "").replace("\\", "/").strip()
    if platform.system() != "Windows":
        return Path(s)

    if not s.startswith("/"):
        raise ValueError(f"Expected absolute Linux path, got: {linux_abs!r}")

    return Path(f"\\\\wsl.localhost\\{DISTRO_NAME}" + s.replace("/", "\\"))


def linuxize_many(tokens: list[str]) -> list[str]:
    """Linuxize a list of path-like tokens."""
    out: list[str] = []
    for t in tokens:
        tt = (t or "").strip().strip("'").strip('"')
        if not tt:
            continue
        out.append(linuxize_path(Path(tt)))
    return out


def linuxize_extra_res_fa(extra_res_fa: str) -> str:
    """
    Rosetta accepts multiple params after -in:file:extra_res_fa.
    We store these as whitespace-separated tokens in JSON for compatibility;
    this converts each token to a WSL Linux path and returns a safely quoted string.
    """
    toks = [t.strip().strip("'").strip('"') for t in (extra_res_fa or "").split() if t.strip()]
    linux_toks = linuxize_many(toks)
    return " ".join(f"'{t}'" for t in linux_toks)


def extra_res_fa_from_list(paths: list[str | Path]) -> str:
    """
    Build a stable whitespace-separated token string from a list of paths (host-readable).
    This is the "compat" representation. Prefer storing lists in JSON; use this for legacy fields.
    """
    toks: list[str] = []
    for p in paths:
        pp = Path(p) if not isinstance(p, Path) else p
        toks.append(str(pp))
    # de-dup while preserving order
    seen = set()
    out = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return " ".join(out)


@dataclass
class WSLResult:
    cmd: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_wsl(
    cmd: str,
    *,
    cwd_wsl: str | None = None,
    log: Path | None = None,
    check: bool = True,
) -> WSLResult:
    """
    Run a command inside WSL via bash -lc.

    - Always uses explicit distro on Windows for reproducibility.
    - Writes a combined command/stdout/stderr log if provided.
    - Raises RuntimeError on failure if check=True.

    Returns a WSLResult (lightweight, easy to serialize/debug).
    """
    if cwd_wsl:
        cmd = f"pushd '{cwd_wsl}' >/dev/null 2>&1; {cmd}; rc=$?; popd >/dev/null 2>&1; exit $rc"

    if platform.system() == "Windows":
        full = [WSL_EXE, "-d", DISTRO_NAME, "--", "bash", "-lc", cmd]
    else:
        full = ["bash", "-lc", cmd]

    proc = subprocess.run(full, text=True, capture_output=True)

    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "COMMAND:\n" + " ".join(full) +
            "\n\n=== STDOUT ===\n" + (proc.stdout or "") +
            "\n=== STDERR ===\n" + (proc.stderr or ""),
            encoding="utf-8", errors="replace",
        )

    res = WSLResult(
        cmd=" ".join(full),
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"❌ WSL command failed (exit {proc.returncode}).\n\nCMD:\n{cmd}\n\nSTDERR:\n{proc.stderr}"
        )

    return res


def ensure_wsl_executable_exists(posix_path: str) -> None:
    """Verify an executable exists inside WSL (raises on failure)."""
    p = (posix_path or "").strip()
    if not p:
        raise ValueError("posix_path is empty")
    run_wsl(f"test -x '{p}'")


def safe_write_text(path: Path, text: str) -> None:
    """Atomic-ish write: write tmp then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", errors="replace")
    tmp.replace(path)
