# --- cache_utils.py (compat + new default root) ---

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional, Union

from af3_pipeline.config import cfg


def compute_hash(value: str) -> str:
    return hashlib.sha1((value or "").encode("utf-8")).hexdigest()[:10]


def _default_user_root() -> Path:
    # C:\Users\olive\.af3_pipeline on Windows; ~/.af3_pipeline on Linux
    return Path.home() / ".af3_pipeline"


def _compute_cache_root() -> Path:
    """
    New plan default:
      ~/.af3_pipeline/cache

    Still supports overrides:
      - cfg.cache_root (absolute or relative; if relative, interpret under ~/.af3_pipeline)
      - cfg.cache_dir  (under ~/.af3_pipeline)
    """
    user_root = _default_user_root()

    cache_root_raw = cfg.get("cache_root", None)
    if cache_root_raw:
        p = Path(str(cache_root_raw)).expanduser()
        if p.is_absolute():
            return p
        # relative -> under ~/.af3_pipeline
        return (user_root / p).resolve()

    cache_dir = cfg.get("cache_dir", "cache")
    return (user_root / cache_dir).resolve()


CACHE_ROOT = _compute_cache_root()
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

if bool(cfg.get("debug_paths", False)):
    print(f"🗃️ CACHE_ROOT = {CACHE_ROOT}")


def get_cache_dir(category: str, key: str) -> Path:
    cdir = CACHE_ROOT / category / key
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir


def get_cache_file(category: str, key: str, filename: str) -> Path:
    return CACHE_ROOT / category / key / filename


def exists_in_cache(category: str, key: str, filename: str) -> bool:
    return get_cache_file(category, key, filename).exists()


def save_to_cache(category: str, key: str, filename: str, content: Union[bytes, str, Path]) -> Path:
    cdir = get_cache_dir(category, key)
    fpath = cdir / filename

    if isinstance(content, Path):
        # copy file bytes if it's a file; else write its string path
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
    """
    Back-compat: prefer msa.txt, else msa.a3m
    """
    key = compute_hash(seq)
    if exists_in_cache("msa", key, "msa.txt"):
        return load_from_cache("msa", key, "msa.txt")  # type: ignore[return-value]
    if exists_in_cache("msa", key, "msa.a3m"):
        return load_from_cache("msa", key, "msa.a3m")  # type: ignore[return-value]
    return None


def save_msa_cache(seq: str, msa_str: str) -> Path:
    key = compute_hash(seq)
    return save_to_cache("msa", key, "msa.txt", msa_str)


def get_cached_ligand_cif(smiles: str) -> Optional[str]:
    key = compute_hash(smiles)
    if exists_in_cache("ligands", key, "ligand.cif"):
        return load_from_cache("ligands", key, "ligand.cif")  # type: ignore[return-value]
    if exists_in_cache("ligands", key, "LIGAND.cif"):
        return load_from_cache("ligands", key, "LIGAND.cif")  # type: ignore[return-value]
    return None


def save_ligand_cif_cache(smiles: str, cif_data: str) -> Path:
    key = compute_hash(smiles)
    return save_to_cache("ligands", key, "ligand.cif", cif_data)


def get_cached_template(pdb_id: str):
    key = compute_hash(pdb_id)
    if exists_in_cache("templates", key, "template.json"):
        data = load_json_from_cache("templates", key, "template.json")
        return data["mmcif"], data["queryIndices"], data["templateIndices"]
    return None


def save_template_cache(pdb_id: str, cif_str: str, qidx: list[int], tidx: list[int]) -> Path:
    key = compute_hash(pdb_id)
    data = {"mmcif": cif_str, "queryIndices": qidx, "templateIndices": tidx}
    return save_json_to_cache("templates", key, "template.json", data)


def clear_cache(category: str = None, key: str = None):
    if category and key:
        target = CACHE_ROOT / category / key
    elif category:
        target = CACHE_ROOT / category
    else:
        target = CACHE_ROOT

    if target.exists():
        shutil.rmtree(target)
        print(f"🧹 Cleared cache: {target}")
    else:
        print(f"⚠️ Nothing to clear at: {target}")
