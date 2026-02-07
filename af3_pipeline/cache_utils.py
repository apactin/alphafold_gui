# --- cache_utils.py (profiles + shared msa/templates safe) ---

from __future__ import annotations

import os
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional, Union
from rdkit import Chem

from af3_pipeline.config import cfg


def compute_hash(value: str) -> str:
    return hashlib.sha1((value or "").encode("utf-8")).hexdigest()[:10]


def _default_user_root() -> Path:
    # C:\Users\<user>\.af3_pipeline on Windows; ~/.af3_pipeline on Linux
    return Path.home() / ".af3_pipeline"


def _resolve_root(raw: Union[str, Path], *, user_root: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (user_root / p).resolve()


def _compute_profile_cache_root() -> Path:
    """
    Profile cache root resolution order:
      1) AF3_PIPELINE_CACHE_ROOT env (absolute OR relative to ~/.af3_pipeline)
      2) cfg.cache_root (absolute OR relative to ~/.af3_pipeline)
      3) ~/.af3_pipeline/<cache_dir> (default: ~/.af3_pipeline/cache)
    """
    user_root = _default_user_root()

    env_root = os.environ.get("AF3_PIPELINE_CACHE_ROOT", "").strip()
    if env_root:
        return _resolve_root(env_root, user_root=user_root)

    cache_root_raw = cfg.get("cache_root", None)
    if cache_root_raw:
        return _resolve_root(cache_root_raw, user_root=user_root)

    cache_dir = cfg.get("cache_dir", "cache")
    return (user_root / cache_dir).resolve()


def _compute_shared_root(kind: str) -> Optional[Path]:
    """
    Shared caches (optional):
      - kind="msa"      -> AF3_PIPELINE_MSA_CACHE
      - kind="templates"-> AF3_PIPELINE_TEMPLATE_CACHE

    If env var not set, returns None (falls back to profile cache root).
    """
    user_root = _default_user_root()

    if kind == "msa":
        raw = os.environ.get("AF3_PIPELINE_MSA_CACHE", "").strip()
    elif kind == "templates":
        raw = os.environ.get("AF3_PIPELINE_TEMPLATE_CACHE", "").strip()
    else:
        raw = ""

    if not raw:
        return None

    return _resolve_root(raw, user_root=user_root)


# --- internal cached roots (light caching to avoid repeated mkdirs) ---
_PROFILE_CACHE_ROOT: Optional[Path] = None
_SHARED_MSA_ROOT: Optional[Path] = None
_SHARED_TEMPLATES_ROOT: Optional[Path] = None


def _profile_cache_root() -> Path:
    global _PROFILE_CACHE_ROOT
    root = _compute_profile_cache_root()

    # refresh cache if changed
    if _PROFILE_CACHE_ROOT is None or _PROFILE_CACHE_ROOT != root:
        _PROFILE_CACHE_ROOT = root
        _PROFILE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        if bool(cfg.get("debug_paths", False)):
            print(f"🗃️ PROFILE CACHE_ROOT = {_PROFILE_CACHE_ROOT}")

    return _PROFILE_CACHE_ROOT


def _shared_msa_root() -> Optional[Path]:
    global _SHARED_MSA_ROOT
    root = _compute_shared_root("msa")
    if root is None:
        _SHARED_MSA_ROOT = None
        return None
    if _SHARED_MSA_ROOT is None or _SHARED_MSA_ROOT != root:
        _SHARED_MSA_ROOT = root
        _SHARED_MSA_ROOT.mkdir(parents=True, exist_ok=True)
        if bool(cfg.get("debug_paths", False)):
            print(f"🧬 SHARED MSA CACHE = {_SHARED_MSA_ROOT}")
    return _SHARED_MSA_ROOT


def _shared_templates_root() -> Optional[Path]:
    global _SHARED_TEMPLATES_ROOT
    root = _compute_shared_root("templates")
    if root is None:
        _SHARED_TEMPLATES_ROOT = None
        return None
    if _SHARED_TEMPLATES_ROOT is None or _SHARED_TEMPLATES_ROOT != root:
        _SHARED_TEMPLATES_ROOT = root
        _SHARED_TEMPLATES_ROOT.mkdir(parents=True, exist_ok=True)
        if bool(cfg.get("debug_paths", False)):
            print(f"📦 SHARED TEMPLATE CACHE = {_SHARED_TEMPLATES_ROOT}")
    return _SHARED_TEMPLATES_ROOT

def _norm_category(category: str) -> str:
    return (category or "").strip().lower()


def cache_root_for_category(category: str) -> Path:
    cat = _norm_category(category)
    if cat == "msa":
        shared = _shared_msa_root()
        if shared is not None:
            return shared
    if cat == "templates":
        shared = _shared_templates_root()
        if shared is not None:
            return shared
    return _profile_cache_root()

def get_cache_dir(category: str, key: str) -> Path:
    cat = _norm_category(category)
    root = cache_root_for_category(cat)
    cdir = root / cat / key
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir

def get_cache_file(category: str, key: str, filename: str) -> Path:
    cat = _norm_category(category)
    root = cache_root_for_category(cat)
    return root / cat / key / filename


def exists_in_cache(category: str, key: str, filename: str) -> bool:
    return get_cache_file(category, key, filename).exists()


def save_to_cache(category: str, key: str, filename: str, content: Union[bytes, str, Path]) -> Path:
    cdir = get_cache_dir(category, key)
    fpath = cdir / filename

    if isinstance(content, Path):
        if content.exists() and content.is_file():
            fpath.write_bytes(content.read_bytes())
            return fpath
        content = str(content)

    if isinstance(content, bytes):
        fpath.write_bytes(content)
    else:
        fpath.write_text(str(content), encoding="utf-8")

    return fpath


def load_from_cache(category: str, key: str, filename: str, as_bytes: bool = False) -> Union[str, bytes]:
    fpath = get_cache_file(category, key, filename)
    return fpath.read_bytes() if as_bytes else fpath.read_text(encoding="utf-8")


def save_json_to_cache(category: str, key: str, filename: str, data: dict) -> Path:
    return save_to_cache(category, key, filename, json.dumps(data, indent=2))


def load_json_from_cache(category: str, key: str, filename: str) -> dict:
    return json.loads(load_from_cache(category, key, filename))


# ---------------------------------------------------------------------
# ✅ Legacy API expected by json_builder (restore these names)
# ---------------------------------------------------------------------

def get_cached_msa(seq: str) -> Optional[str]:
    key = compute_hash(seq)
    if exists_in_cache("msa", key, "msa.txt"):
        return load_from_cache("msa", key, "msa.txt")  # type: ignore[return-value]
    if exists_in_cache("msa", key, "msa.a3m"):
        return load_from_cache("msa", key, "msa.a3m")  # type: ignore[return-value]
    return None


def save_msa_cache(seq: str, msa_str: str) -> Path:
    key = compute_hash(seq)
    return save_to_cache("msa", key, "msa.txt", msa_str)


def _canonical_smiles_for_cache(smiles: str) -> str:
    try:
        m = Chem.MolFromSmiles(smiles, sanitize=True)
        if m is None:
            return (smiles or "").strip()
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return (smiles or "").strip()

def get_cached_ligand_cif(smiles: str) -> Optional[str]:
    key = compute_hash(_canonical_smiles_for_cache(smiles))
    if exists_in_cache("ligands", key, "lig.cif"):
        return load_from_cache("ligands", key, "lig.cif")  # type: ignore[return-value]
    if exists_in_cache("ligands", key, "LIG.cif"):
        return load_from_cache("ligands", key, "LIG.cif")  # type: ignore[return-value]
    return None


def save_ligand_cif_cache(smiles: str, cif_data: str) -> Path:
    key = compute_hash(smiles)
    return save_to_cache("ligands", key, "lig.cif", cif_data)


def get_cached_template(pdb_id: str):
    pdb = (pdb_id or "").strip().upper()
    if not pdb:
        return None

    # 1) New style: raw CIF exists?
    cif_name = f"{pdb}.cif"
    if exists_in_cache("templates", pdb, cif_name):
        cif_str = load_from_cache("templates", pdb, cif_name)

        # Legacy mapping info might not exist; return CIF only if you want,
        # but your old API expects indices too, so fall through unless legacy json exists.
        # (If you WANT to allow returning CIF without indices, change your callers.)
        pass

    # 2) Legacy style: template.json under templates/<PDBID>/template.json
    if exists_in_cache("templates", pdb, "template.json"):
        data = load_json_from_cache("templates", pdb, "template.json")
        return data["mmcif"], data["queryIndices"], data["templateIndices"]

    # 3) Old-hash fallback (in case users already have caches on disk)
    key = compute_hash(pdb)
    if exists_in_cache("templates", key, "template.json"):
        data = load_json_from_cache("templates", key, "template.json")
        return data["mmcif"], data["queryIndices"], data["templateIndices"]

    return None


def save_template_cache(pdb_id: str, cif_str: str, qidx: list[int], tidx: list[int]) -> Path:
    pdb = (pdb_id or "").strip().upper()
    if not pdb:
        raise ValueError("pdb_id is required")

    data = {"mmcif": cif_str, "queryIndices": qidx, "templateIndices": tidx}

    # Write legacy mapping where it makes sense long-term (keyed by PDBID)
    save_json_to_cache("templates", pdb, "template.json", data)

    # Also write raw cif in the new canonical place for template_utils reuse
    save_to_cache("templates", pdb, f"{pdb}.cif", cif_str)

    return get_cache_file("templates", pdb, "template.json")


def clear_cache(category: Optional[str] = None, key: Optional[str] = None):
    """
    Clears *profile cache root* by default.
    If you want to clear shared MSA/templates, pass category="msa"/"templates"
    and ensure AF3_PIPELINE_MSA_CACHE / AF3_PIPELINE_TEMPLATE_CACHE are set.
    """
    if category and key:
        target = cache_root_for_category(category) / category / key
    elif category:
        target = cache_root_for_category(category) / category
    else:
        # default: profile cache root only (safe!)
        target = _profile_cache_root()

    if target.exists():
        shutil.rmtree(target)
        print(f"🧹 Cleared cache: {target}")
    else:
        print(f"⚠️ Nothing to clear at: {target}")
