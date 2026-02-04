"""
af3_pipeline package
====================
Core AlphaFold3 + Rosetta pipeline:
- json_builder, ligand_utils, runner, cache_utils, etc.
- analysis subpackage: post-AF3 processing and scoring
"""

import platform
from pathlib import Path
from importlib import import_module
from af3_pipeline.config import cfg  # ✅ new import

__all__ = [
    "analysis",
    "json_builder",
    "ligand_utils",
    "runner",
    "cache_utils",
]

def __getattr__(name):
    if name in __all__:
        return import_module(f"af3_pipeline.{name}")
    raise AttributeError(f"module 'af3_pipeline' has no attribute '{name}'")


def to_wsl_unc(path: str | Path) -> Path:
    """
    Convert a WSL path (/home/<user>/...) to a Windows UNC path when running on Windows.
    Uses config.yaml for distro name and Linux home root.
    """
    p = str(path)
    distro = cfg.get("wsl_distro_name", "Ubuntu-22.04")
    home_root = cfg.get("linux_home_root", "/home/olive")

    if platform.system() == "Windows" and p.startswith(home_root):
        # Build UNC prefix safely (no backslashes in f-string expressions)
        unc_prefix = f"\\\\wsl.localhost\\{distro}"
        p_win = p.replace("/", "\\")
        return Path(unc_prefix + p_win)
    return Path(p)
