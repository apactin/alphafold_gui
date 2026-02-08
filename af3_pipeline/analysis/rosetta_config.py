# rosetta_config.py
"""
Tiny helper to load rosetta_dicts.yaml and provide a few convenience utilities.

Design goals:
- Small, dependency-light (uses PyYAML if available; falls back to ruamel.yaml if you prefer)
- Caller can override yaml path, otherwise we try:
    1) cfg["rosetta_dicts_yaml"]
    2) alongside this file (rosetta_dicts.yaml)
- Provide DB-relative path resolution for entries like:
    covalent_patches: NZ: extra_res_fa: {db_rel: ".../LYX.params"}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from af3_pipeline.config import cfg


def _load_yaml_text(path: Path) -> Dict[str, Any]:
    # Prefer PyYAML (fast, common)
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        pass

    # Fallback: ruamel.yaml (preserves comments if you later want to round-trip)
    try:
        from ruamel.yaml import YAML  # type: ignore
        y = YAML(typ="safe")
        return y.load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "No YAML library available. Install one of: pyyaml (recommended) or ruamel.yaml"
        ) from e


def default_yaml_path() -> Path:
    # 1) from cfg
    p = (cfg.get("rosetta_dicts_yaml") or "").strip()
    if p:
        return Path(p).expanduser().resolve()

    # 2) alongside this helper
    here = Path(__file__).resolve().parent
    return (here / "rosetta_dicts.yaml").resolve()


@dataclass(frozen=True)
class RosettaDicts:
    path: Path
    data: Dict[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safe nested get:
            d.get("ligand","resname",default="LIG")
        """
        cur: Any = self.data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def rosetta_db_abs(self) -> str:
        """
        Return ROSETTA_DB absolute path (POSIX) as a string.
        Assumes cfg['rosetta_relax_bin'] points at the Rosetta bundle root inside WSL.
        """
        rosetta_base = (cfg.get("rosetta_relax_bin") or "").strip()
        if not rosetta_base:
            raise RuntimeError("Missing config key: rosetta_relax_bin")
        rosetta_base = str(PurePosixPath(rosetta_base))
        return f"{rosetta_base}/main/database"

    def resolve_db_rel(self, db_rel: str) -> str:
        """
        Join a Rosetta DB-relative path to the ROSETTA_DB absolute path.
        Returns a POSIX path string.
        """
        rel = (db_rel or "").strip().lstrip("/").replace("\\", "/")
        return str(PurePosixPath(self.rosetta_db_abs()) / rel)


def load_rosetta_dicts(path: str | Path | None = None) -> RosettaDicts:
    p = Path(path).expanduser().resolve() if path else default_yaml_path()
    if not p.exists():
        raise FileNotFoundError(f"rosetta_dicts.yaml not found: {p}")
    data = _load_yaml_text(p)
    return RosettaDicts(path=p, data=data)


def resolve_patch_extra_res_fa(dicts: RosettaDicts, prot_atom: str) -> Optional[str]:
    """
    Convenience: get covalent_patches[prot_atom].extra_res_fa and resolve db_rel if present.
    Returns a POSIX string suitable for passing to Rosetta inside WSL.
    """
    prot_atom = (prot_atom or "").strip().upper()
    patch = dicts.get("covalent_patches", prot_atom, default=None)
    if not isinstance(patch, dict):
        return None

    extra = patch.get("extra_res_fa")
    if extra is None:
        return None

    # YAML may store as {db_rel: "..."} or a plain string; support both
    if isinstance(extra, dict) and extra.get("db_rel"):
        return dicts.resolve_db_rel(str(extra["db_rel"]))
    if isinstance(extra, str) and extra.strip():
        # assume already a usable path (posix)
        return extra.strip()

    return None
