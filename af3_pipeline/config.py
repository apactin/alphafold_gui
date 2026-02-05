"""
af3_pipeline.config
====================
Unified configuration loader for AlphaFold3 + Rosetta pipeline.

Load order:
  1) AF3_PIPELINE_CONFIG (explicit override; used by GUI + subprocesses)
  2) repo root config.yaml (preferred for dev)
  3) ~/.af3_pipeline/config.yaml (fallback)

Supports:
  cfg.get("key", default)                (top-level)
  cfg.get("nested.key.path", default)    (dot-path)
Backward compatible:
  cfg.get("ligand_nconfs") maps to cfg.get("ligand.n_confs")
  cfg.get("msa_threads")   maps to cfg.get("msa.threads")

Environment overrides (applied last):
  AF3__ligand__n_confs=300 -> ligand.n_confs = 300
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Union
import os
import sys
import yaml


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
    """
    Expand ~ and normalize Windows paths, but avoid mangling Linux/WSL POSIX paths
    when running on Windows.
    """
    if not isinstance(v, str):
        return v

    s = v.strip()
    if not s:
        return v

    # On Windows, leave absolute POSIX alone (so "/home/..." doesn't become "\home\...")
    if sys.platform.startswith("win") and s.startswith("/"):
        return str(PurePosixPath(s))

    # Don't touch UNC paths
    if s.startswith("\\\\"):
        return s

    # Expand "~" and normalize typical Windows pathlikes
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
    s = (raw or "").strip()
    if not s:
        return ""
    try:
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
    # older naming patterns:
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
    "rosetta_bin": "rosetta_relax_bin",
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

    # ---------- path selection ----------
    def _candidate_paths(self) -> list[Path]:
        """
        Profile-aware: AF3_PIPELINE_CONFIG (if set) wins.
        Otherwise: repo-local preferred, then user config.
        """
        env = (os.environ.get("AF3_PIPELINE_CONFIG") or "").strip()
        if env:
            return [Path(env).expanduser()]

        # ✅ Preferred: repo-local first (dev ergonomics)
        return [self.paths.local_cfg, self.paths.user_cfg]

    def _choose_path(self) -> Optional[Path]:
        for p in self._candidate_paths():
            if p.exists():
                return p
        return None

    # ---------- load/save/reload ----------
    def _expand_paths_in_obj(self, x: Any) -> Any:
        if isinstance(x, dict):
            return {k: self._expand_paths_in_obj(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._expand_paths_in_obj(v) for v in x]
        return _expand_pathlike(x)

    def _load(self) -> None:
        path = self._choose_path()
        if not path:
            if bool(os.environ.get("AF3_DEBUG_CONFIG", "")):
                print("WARNING: No config.yaml found - using built-in defaults.")
            self.path = None
            self.data = {}
            return

        self.path = path
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("config.yaml must contain a mapping at the top level.")
            self.data = self._expand_paths_in_obj(loaded)
            if bool(os.environ.get("AF3_DEBUG_CONFIG", "")):
                print(f"Loaded configuration from: {path}")
        except Exception as e:
            print(f"Failed to read config.yaml ({e}); using empty config.")
            self.path = None
            self.data = {}

    def reload(self, path: Optional[Union[str, os.PathLike]] = None) -> None:
        """
        Reload configuration from disk.
        If `path` is provided, also sets AF3_PIPELINE_CONFIG so subprocesses inherit it.
        """
        if path:
            os.environ["AF3_PIPELINE_CONFIG"] = str(Path(path).expanduser().resolve())
        self._load()
        self._apply_env_overrides(prefix="AF3__")

    def save(self, *, prefer_local: bool = False) -> Path:
        """
        Save current config back to disk.

        - If AF3_PIPELINE_CONFIG is set, save there (strongest guarantee for profiles).
        - Else: save to repo-local if prefer_local else user config.
        """
        env = (os.environ.get("AF3_PIPELINE_CONFIG") or "").strip()
        if env:
            out = Path(env).expanduser()
        else:
            out = self.paths.local_cfg if prefer_local else self.paths.user_cfg

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")
        self.path = out
        return out

    # ---------- env overrides ----------
    def _apply_env_overrides(self, prefix: str = "AF3__") -> None:
        """
        Env override convention:
          AF3__ligand__n_confs=300 -> ligand.n_confs = 300
          AF3__msa__threads=16     -> msa.threads = 16
        """
        for k, raw in os.environ.items():
            if not k.startswith(prefix):
                continue
            key_body = k[len(prefix):]
            dot_key = key_body.replace("__", ".").lower()
            val = _parse_env_value(raw)
            _deep_set(self.data, dot_key, val)

    # ---------- get/set ----------
    def _heuristic_flat_to_dot(self, key: str) -> Optional[str]:
        if "_" not in key:
            return None
        prefix, rest = key.split("_", 1)
        prefix = prefix.strip()
        rest = rest.strip()

        if prefix in {"ligand", "msa"}:
            if prefix == "ligand" and rest == "nconfs":
                rest = "n_confs"
            return f"{prefix}.{rest}"
        return None

    def _resolve_key_to_dotpath(self, key: str) -> str:
        if "." in key:
            return key
        if key in LEGACY_KEY_MAP:
            return LEGACY_KEY_MAP[key]
        g = self._heuristic_flat_to_dot(key)
        return g or key

    def set(self, key: str, value: Any) -> None:
        if not key:
            return
        dot_key = self._resolve_key_to_dotpath(key)
        _deep_set(self.data, dot_key, value)

    def get(self, key: str, default=None):
        if not key:
            return default

        # 1) dot-path
        if "." in key:
            return _deep_get(self.data, key, default)

        # 2) top-level
        if _is_dict(self.data) and key in self.data:
            return self.data.get(key, default)

        # 3) legacy mapping
        mapped = LEGACY_KEY_MAP.get(key)
        if mapped:
            return _deep_get(self.data, mapped, default)

        # 4) heuristic mapping (ligand_nconfs -> ligand.n_confs, etc.)
        guess = self._heuristic_flat_to_dot(key)
        if guess:
            got = _deep_get(self.data, guess, None)
            if got is not None:
                return got

        return default


cfg = Config()
