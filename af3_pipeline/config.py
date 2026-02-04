"""
af3_pipeline.config
====================
Unified configuration loader for AlphaFold3 + Rosetta pipeline.

- Loads config.yaml from:
    1) repo root (preferred)
    2) ~/.af3_pipeline/config.yaml (fallback)
- Supports:
    cfg.get("key", default)                (top-level)
    cfg.get("nested.key.path", default)    (dot-path)
- Backward compatible:
    cfg.get("ligand_nconfs") maps to cfg.get("ligand.n_confs")
    cfg.get("msa_threads")   maps to cfg.get("msa.threads")
    etc.
- Optional:
    environment variable overrides like AF3__ligand__n_confs=300
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple
import os
import yaml
import sys


# --------------------------
# Helpers
# --------------------------
def _is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def _deep_get(data: Any, dot_key: str, default=None):
    if not dot_key:
        return default
    cur = data
    for part in dot_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _deep_set(data: Dict[str, Any], dot_key: str, value: Any) -> None:
    if "." not in dot_key:
        data[dot_key] = value
        return
    cur: Dict[str, Any] = data
    parts = dot_key.split(".")
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _expand_pathlike(v: Any) -> Any:
    if not isinstance(v, str):
        return v

    s = v.strip()
    if not s:
        return v

    # On Windows, leave Linux/WSL absolute POSIX paths alone.
    # Otherwise Path("/home/...") turns into "\home\..." when stringified.
    if sys.platform.startswith("win") and s.startswith("/"):
        return str(PurePosixPath(s))

    # Also: if user already entered a UNC path, don't touch it
    if s.startswith("\\\\"):
        return s

    # Expand "~" and normalize Windows paths normally
    if s.startswith("~") or ":" in s or s.startswith("\\"):
        try:
            return str(Path(s).expanduser())
        except Exception:
            return v

    return v


def _parse_env_value(raw: str) -> Any:
    """
    Parse env var values in a YAML-ish way:
      "true" -> True
      "5.7"  -> 5.7
      "[1,2]" -> list
      "{a: 1}" -> dict
      otherwise string
    """
    s = raw.strip()
    if not s:
        return ""
    try:
        # yaml.safe_load handles bool/num/list/dict nicely
        return yaml.safe_load(s)
    except Exception:
        return raw


# --------------------------
# Backward-compat mapping
# --------------------------
LEGACY_KEY_MAP: Dict[str, str] = {
    # --- MSA ---
    "msa_threads": "msa.threads",
    "msa_sensitivity": "msa.sensitivity",
    "msa_max_seqs": "msa.max_seqs",
    # older naming patterns you used before:
    "threads": "msa.threads",
    "sensitivity": "msa.sensitivity",
    "max_seqs": "msa.max_seqs",
    "max_sequences": "msa.max_seqs",

    # --- Ligand ---
    "ligand_nconfs": "ligand.n_confs",
    "ligand_seed": "ligand.seed",
    "ligand_prune_rms": "ligand.prune_rms",
    "ligand_keep_charge": "ligand.keep_charge",
    "ligand_require_assigned_stereo": "ligand.require_assigned_stereo",
    "ligand_basename": "ligand.basename",
    "ligand_name_default": "ligand.name_default",
    "ligand_png_size": "ligand.png_size",
    "ligand_rdkit_threads": "ligand.rdkit_threads",

    # --- Docker / AF3 ---
    "docker_image": "alphafold_docker_image",
    "af3_docker_image": "alphafold_docker_image",
    "docker_env": "alphafold_docker_env",

    # --- Rosetta ---
    "rosetta_bin": "rosetta_relax_bin",  # if you previously used rosetta_bin, map it
}


@dataclass
class ConfigPaths:
    local_cfg: Path
    user_cfg: Path


class Config:
    def __init__(self):
        repo_path = Path(__file__).resolve().parents[1]
        self.paths = ConfigPaths(
            local_cfg=repo_path / "config.yaml",
            user_cfg=Path.home() / ".af3_pipeline" / "config.yaml",
        )

        self.path: Optional[Path] = None
        self.data: Dict[str, Any] = {}

        self._load()

        # Apply env overrides last
        self._apply_env_overrides(prefix="AF3__")

    def _choose_path(self) -> Optional[Path]:
        if self.paths.user_cfg.exists():
            return self.paths.user_cfg
        if self.paths.local_cfg.exists():
            return self.paths.local_cfg
        return None

    def _load(self) -> None:
        path = self._choose_path()
        if not path:
            print("WARNING: No config.yaml found - using built-in defaults.")
            self.path = None
            self.data = {}
            return

        self.path = path
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("config.yaml must contain a mapping at the top level.")
            # Expand pathlike strings
            self.data = self._expand_paths_in_dict(loaded)
            print(f"Loaded configuration from: {path}")
        except Exception as e:
            print(f"Failed to read config.yaml ({e}); using empty config.")
            self.data = {}

    def reload(self) -> None:
        """Reload configuration from disk (useful for GUI Apply without restart)."""
        self._load()
        self._apply_env_overrides(prefix="AF3__")

    def save(self, *, prefer_local: bool = False) -> Path:
        """
        Save current config back to disk.
        Default: save to repo-local config.yaml if possible, else user config.
        """
        out = self.paths.local_cfg if prefer_local else self.paths.user_cfg
        if not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")
        self.path = out
        return out

    def _expand_paths_in_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        def rec(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: rec(v) for k, v in x.items()}
            if isinstance(x, list):
                return [rec(v) for v in x]
            return _expand_pathlike(x)
        return rec(d)

    def _apply_env_overrides(self, prefix: str = "AF3__") -> None:
        """
        Environment override convention:
          AF3__ligand__n_confs=300      -> ligand.n_confs = 300
          AF3__msa__threads=16          -> msa.threads = 16
          AF3__alphafold_docker_env="{...}" -> dict
        """
        for k, raw in os.environ.items():
            if not k.startswith(prefix):
                continue
            key_body = k[len(prefix):]
            # allow __ to mean nesting
            dot_key = key_body.replace("__", ".").lower()
            val = _parse_env_value(raw)
            _deep_set(self.data, dot_key, val)

    def set(self, key: str, value: Any) -> None:
        """
        Set config value.
        Accepts:
          - dot paths: "ligand.n_confs"
          - legacy keys: "ligand_nconfs"
        """
        if not key:
            return
        dot_key = self._resolve_key_to_dotpath(key)
        _deep_set(self.data, dot_key, value)

    def get(self, key: str, default=None):
        if not key:
            return default

        # 1) dot-path lookup
        if "." in key:
            return _deep_get(self.data, key, default)

        # 2) top-level direct
        if _is_dict(self.data) and key in self.data:
            return self.data.get(key, default)

        # 3) legacy mapping
        mapped = LEGACY_KEY_MAP.get(key)
        if mapped:
            return _deep_get(self.data, mapped, default)

        # 4) heuristic: allow keys like "ligand_nconfs" -> ligand.n_confs
        #    and "msa_threads" -> msa.threads
        guess = self._heuristic_flat_to_dot(key)
        if guess:
            got = _deep_get(self.data, guess, None)
            if got is not None:
                return got

        return default

    def _resolve_key_to_dotpath(self, key: str) -> str:
        if "." in key:
            return key
        if key in LEGACY_KEY_MAP:
            return LEGACY_KEY_MAP[key]
        g = self._heuristic_flat_to_dot(key)
        return g or key

    def _heuristic_flat_to_dot(self, key: str) -> Optional[str]:
        """
        Best-effort conversion for old flat keys:
          ligand_nconfs -> ligand.n_confs
          msa_threads   -> msa.threads
        Only applies when it matches common prefixes.
        """
        if "_" not in key:
            return None

        prefix, rest = key.split("_", 1)
        prefix = prefix.strip()
        rest = rest.strip()

        if prefix in {"ligand", "msa"}:
            # Special case: nconfs -> n_confs
            if prefix == "ligand" and rest == "nconfs":
                rest = "n_confs"
            return f"{prefix}.{rest}"

        return None


cfg = Config()
