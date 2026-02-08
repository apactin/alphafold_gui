#!/usr/bin/env python3
"""
script.py
=========
GUI-callable editor + CLI for rosetta_dicts.yaml, including:

- ligand defaults (ligand.resname)
- covalent_patches (deleted_atom/temp_resname/extra_res_fa)
- rosetta_scripts_stage.constraints subtree with FULL constraint control:
    - enabled, filename
    - active_sets (covalent/noncovalent)
    - stages overrides
    - defaults (atom_pair/angle/dihedral/coordinate)
    - weights (covalent/noncovalent)
    - sets: named constraint sets containing items (AtomPair/Angle/Dihedral/Coordinate + generic extra)
    - raw constraints (escape hatch)

Prefers ruamel.yaml for round-trip preservation; falls back to PyYAML.

CLI quick tests:
  python script.py show
  python script.py --yaml /path/to/rosetta_dicts.yaml show

  python script.py set-ligand --resname LIG

  python script.py constraints-add-set --name covalent_bond --desc "Primary covalent bond"
  python script.py constraints-add-item --set covalent_bond --type AtomPair \
      --a "SG,79,A" --b "C7,245,X" --func HARMONIC --x0 1.80 --sd 0.10

  python script.py constraints-set-active --mode covalent --sets covalent_bond,covalent_geometry
  python script.py constraints-set-weight --mode covalent --key atom_pair_constraint --value 25.0

Notes for GUI integration:
- Use RosettaDictsEditor.load()/save()
- Read/write constraints via editor.constraints_* methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# If you have rosetta_config.py as shown previously, import default_yaml_path.
# Otherwise replace with your own default resolver.
from af3_pipeline.rosetta_config import default_yaml_path


# ----------------------------
# YAML backend (ruamel preferred)
# ----------------------------
def _yaml_backend():
    """
    Returns (backend_name, loader, dumper):
      - loader(path)->(data, roundtrip_obj)
      - dumper(path, data, roundtrip_obj)->None
    roundtrip_obj is a ruamel YAML object or None for PyYAML.
    """
    try:
        from ruamel.yaml import YAML  # type: ignore

        y = YAML()
        y.preserve_quotes = True

        def load(path: Path) -> Tuple[Dict[str, Any], Any]:
            obj = y.load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(obj, dict):
                obj = {}
            return obj, y

        def save(path: Path, data: Dict[str, Any], yobj: Any) -> None:
            with path.open("w", encoding="utf-8") as f:
                yobj.dump(data, f)

        return ("ruamel", load, save)

    except ModuleNotFoundError:
        pass

    try:
        import yaml  # type: ignore

        def load(path: Path) -> Tuple[Dict[str, Any], Any]:
            obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(obj, dict):
                obj = {}
            return obj, None

        def save(path: Path, data: Dict[str, Any], _yobj: Any) -> None:
            path.write_text(
                yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )

        return ("pyyaml", load, save)

    except ModuleNotFoundError as e:
        raise RuntimeError("Install pyyaml or ruamel.yaml to edit rosetta_dicts.yaml") from e


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    cur = parent.get(key)
    if not isinstance(cur, dict):
        cur = {}
        parent[key] = cur
    return cur


def _ensure_list(parent: Dict[str, Any], key: str) -> List[Any]:
    cur = parent.get(key)
    if not isinstance(cur, list):
        cur = []
        parent[key] = cur
    return cur


def _backup_file(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        if path.exists():
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass


def _upper(s: str) -> str:
    return (s or "").strip().upper()


def _norm_pathish(s: str) -> str:
    return (s or "").strip().replace("\\", "/")


def _parse_atomref(s: str) -> Dict[str, Any]:
    """
    Parse "ATOM,RES,CHAIN" -> {"atom":"ATOM","res":int,"chain":"A"}
    GUI may also pass "ATOM RES CHAIN".
    """
    raw = (s or "").strip()
    if not raw:
        return {}
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
    else:
        parts = [p.strip() for p in raw.split()]
    if len(parts) < 2:
        raise ValueError(f"Invalid atom ref '{s}'. Expected 'ATOM,RES,CHAIN' or 'ATOM RES CHAIN'.")
    atom = _upper(parts[0])
    res = int(parts[1])
    chain = _upper(parts[2]) if len(parts) >= 3 and parts[2] else ""
    out: Dict[str, Any] = {"atom": atom, "res": res}
    if chain:
        out["chain"] = chain
    return out


def _atomref_to_str(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict):
        return ""
    atom = _upper(str(d.get("atom", "")))
    res = d.get("res", "")
    chain = _upper(str(d.get("chain", ""))) if d.get("chain") else ""
    if atom and res != "":
        if chain:
            return f"{atom},{res},{chain}"
        return f"{atom},{res}"
    return ""


# ----------------------------
# Data models
# ----------------------------
@dataclass
class PatchEntry:
    prot_atom: str
    deleted_atom: str = ""
    temp_resname: str = ""
    extra_res_fa_db_rel: str = ""
    extra_res_fa_abs: str = ""

    def normalized(self) -> "PatchEntry":
        return PatchEntry(
            prot_atom=_upper(self.prot_atom),
            deleted_atom=_upper(self.deleted_atom),
            temp_resname=_upper(self.temp_resname),
            extra_res_fa_db_rel=_norm_pathish(self.extra_res_fa_db_rel),
            extra_res_fa_abs=_norm_pathish(self.extra_res_fa_abs),
        )


@dataclass
class ConstraintItem:
    """
    Generic representation of a constraint item suitable for YAML structured sets.

    type: AtomPair | Angle | Dihedral | Coordinate | ...
    For AtomPair: a,b
    For Angle: a,b,c
    For Dihedral: a,b,c,d
    For Coordinate: atom + ref (either atomref dict or explicit xyz in 'ref_xyz')
    func: e.g., HARMONIC, CIRCULARHARMONIC, BOUNDED, ...
    x0, sd: common parameters
    extra: dict for additional parameters (bounded ranges, periodicity, etc.)
    """
    type: str
    a: Dict[str, Any] | None = None
    b: Dict[str, Any] | None = None
    c: Dict[str, Any] | None = None
    d: Dict[str, Any] | None = None
    atom: Dict[str, Any] | None = None
    ref: Dict[str, Any] | None = None
    ref_xyz: List[float] | None = None
    func: str = ""
    x0: float | None = None
    sd: float | None = None
    extra: Dict[str, Any] | None = None

    def normalized(self) -> "ConstraintItem":
        t = _upper(self.type)
        func = _upper(self.func)

        def _norm_ar(ar: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not ar:
                return None
            out = dict(ar)
            if "atom" in out:
                out["atom"] = _upper(str(out["atom"]))
            if "chain" in out and out.get("chain") is not None:
                out["chain"] = _upper(str(out["chain"]))
            return out

        return ConstraintItem(
            type=t,
            a=_norm_ar(self.a),
            b=_norm_ar(self.b),
            c=_norm_ar(self.c),
            d=_norm_ar(self.d),
            atom=_norm_ar(self.atom),
            ref=_norm_ar(self.ref),
            ref_xyz=self.ref_xyz[:] if isinstance(self.ref_xyz, list) else None,
            func=func,
            x0=self.x0,
            sd=self.sd,
            extra=dict(self.extra) if isinstance(self.extra, dict) else None,
        )

    def to_yaml_dict(self) -> Dict[str, Any]:
        it = self.normalized()
        d: Dict[str, Any] = {"type": it.type}

        # common
        if it.func:
            d["func"] = it.func
        if it.x0 is not None:
            d["x0"] = float(it.x0)
        if it.sd is not None:
            d["sd"] = float(it.sd)

        # atom refs
        if it.a: d["a"] = it.a
        if it.b: d["b"] = it.b
        if it.c: d["c"] = it.c
        if it.d: d["d"] = it.d

        if it.atom: d["atom"] = it.atom
        if it.ref: d["ref"] = it.ref
        if it.ref_xyz: d["ref_xyz"] = it.ref_xyz

        # extra freeform
        if it.extra:
            d["extra"] = it.extra

        return d

    @staticmethod
    def from_yaml_dict(d: Dict[str, Any]) -> "ConstraintItem":
        if not isinstance(d, dict):
            return ConstraintItem(type="")
        return ConstraintItem(
            type=str(d.get("type", "")),
            a=d.get("a"),
            b=d.get("b"),
            c=d.get("c"),
            d=d.get("d"),
            atom=d.get("atom"),
            ref=d.get("ref"),
            ref_xyz=d.get("ref_xyz"),
            func=str(d.get("func", "")),
            x0=d.get("x0"),
            sd=d.get("sd"),
            extra=d.get("extra"),
        )


# ----------------------------
# Editor
# ----------------------------
class RosettaDictsEditor:
    """
    Thin YAML editor around rosetta_dicts.yaml.
    Intended for GUI integration (MainWindow methods).
    """

    def __init__(self, yaml_path: str | Path | None = None):
        self.path = Path(yaml_path).expanduser().resolve() if yaml_path else default_yaml_path()
        self.backend_name, self._load, self._save = _yaml_backend()
        self.data: Dict[str, Any] = {}
        self._rt_obj: Any = None  # ruamel YAML object or None

    def load(self) -> "RosettaDictsEditor":
        if not self.path.exists():
            self.data = {}
            self._rt_obj = None
            return self
        self.data, self._rt_obj = self._load(self.path)
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _backup_file(self.path)
        self._save(self.path, self.data, self._rt_obj)

    # ---- ligand defaults ----
    def get_ligand_resname(self, default: str = "LIG") -> str:
        lig = self.data.get("ligand", {})
        if isinstance(lig, dict):
            v = (lig.get("resname") or "").strip()
            if v:
                return v
        return default

    def set_ligand_resname(self, resname: str) -> None:
        lig = _ensure_dict(self.data, "ligand")
        lig["resname"] = _upper(resname) or "LIG"

    # ---- covalent patches ----
    def list_patches(self) -> Dict[str, Any]:
        patches = self.data.get("covalent_patches", {})
        return patches if isinstance(patches, dict) else {}

    def upsert_patch(self, entry: PatchEntry) -> None:
        e = entry.normalized()
        if not e.prot_atom:
            raise ValueError("prot_atom is required")
        patches = _ensure_dict(self.data, "covalent_patches")
        p: Dict[str, Any] = patches.get(e.prot_atom)
        if not isinstance(p, dict):
            p = {}
            patches[e.prot_atom] = p

        if e.deleted_atom:
            p["deleted_atom"] = e.deleted_atom
        else:
            p.pop("deleted_atom", None)

        if e.temp_resname:
            p["temp_resname"] = e.temp_resname
        else:
            p.pop("temp_resname", None)

        if e.extra_res_fa_db_rel:
            p["extra_res_fa"] = {"db_rel": e.extra_res_fa_db_rel}
        elif e.extra_res_fa_abs:
            p["extra_res_fa"] = e.extra_res_fa_abs
        else:
            p.pop("extra_res_fa", None)

    def remove_patch(self, prot_atom: str) -> bool:
        prot_atom = _upper(prot_atom)
        patches = self.data.get("covalent_patches")
        if not isinstance(patches, dict):
            return False
        if prot_atom in patches:
            del patches[prot_atom]
            return True
        return False

    @staticmethod
    def patchentry_from_yaml(prot_atom: str, d: Dict[str, Any]) -> PatchEntry:
        deleted_atom = (d.get("deleted_atom") or "").strip()
        temp_resname = (d.get("temp_resname") or "").strip()
        extra = d.get("extra_res_fa")

        db_rel = ""
        abs_path = ""
        if isinstance(extra, dict) and extra.get("db_rel"):
            db_rel = str(extra["db_rel"]).strip()
        elif isinstance(extra, str):
            abs_path = extra.strip()

        return PatchEntry(
            prot_atom=prot_atom,
            deleted_atom=deleted_atom,
            temp_resname=temp_resname,
            extra_res_fa_db_rel=db_rel,
            extra_res_fa_abs=abs_path,
        )

    # =========================================================
    # Constraints subtree: rosetta_scripts_stage.constraints
    # =========================================================
    def _stage_root(self) -> Dict[str, Any]:
        return _ensure_dict(self.data, "rosetta_scripts_stage")

    def _constraints_root(self) -> Dict[str, Any]:
        stage = self._stage_root()
        return _ensure_dict(stage, "constraints")

    # ---- high-level toggles ----
    def constraints_get_enabled(self, default: bool = True) -> bool:
        c = self._constraints_root()
        v = c.get("enabled", default)
        return bool(v)

    def constraints_set_enabled(self, enabled: bool) -> None:
        c = self._constraints_root()
        c["enabled"] = bool(enabled)

    def constraints_get_filename(self, default: str = "constraints.cst") -> str:
        c = self._constraints_root()
        v = (c.get("filename") or "").strip()
        return v or default

    def constraints_set_filename(self, filename: str) -> None:
        c = self._constraints_root()
        c["filename"] = (filename or "").strip() or "constraints.cst"

    # ---- active sets ----
    def constraints_get_active_sets(self, mode: str) -> List[str]:
        """
        mode: "covalent" | "noncovalent"
        """
        mode = mode.strip().lower()
        c = self._constraints_root()
        active = c.get("active_sets", {})
        if not isinstance(active, dict):
            return []
        sets = active.get(mode, [])
        if not isinstance(sets, list):
            return []
        out: List[str] = []
        for s in sets:
            name = (str(s) or "").strip()
            if name:
                out.append(name)
        return out

    def constraints_set_active_sets(self, mode: str, sets: List[str]) -> None:
        mode = mode.strip().lower()
        if mode not in ("covalent", "noncovalent"):
            raise ValueError("mode must be 'covalent' or 'noncovalent'")
        c = self._constraints_root()
        active = _ensure_dict(c, "active_sets")
        cleaned = [str(s).strip() for s in (sets or []) if str(s).strip()]
        active[mode] = cleaned

    # ---- stage overrides ----
    def constraints_get_stage_enabled(self, stage_name: str, default: bool = True) -> bool:
        c = self._constraints_root()
        stages = c.get("stages", {})
        if not isinstance(stages, dict):
            return default
        st = stages.get(stage_name, {})
        if not isinstance(st, dict):
            return default
        return bool(st.get("enabled", default))

    def constraints_set_stage_enabled(self, stage_name: str, enabled: bool) -> None:
        c = self._constraints_root()
        stages = _ensure_dict(c, "stages")
        st = _ensure_dict(stages, stage_name)
        st["enabled"] = bool(enabled)

    def constraints_get_stage_extra_sets(self, stage_name: str, mode: str) -> List[str]:
        mode = mode.strip().lower()
        c = self._constraints_root()
        stages = c.get("stages", {})
        if not isinstance(stages, dict):
            return []
        st = stages.get(stage_name, {})
        if not isinstance(st, dict):
            return []
        extra = st.get("extra_sets", {})
        if not isinstance(extra, dict):
            return []
        sets = extra.get(mode, [])
        if not isinstance(sets, list):
            return []
        return [str(s).strip() for s in sets if str(s).strip()]

    def constraints_set_stage_extra_sets(self, stage_name: str, mode: str, sets: List[str]) -> None:
        mode = mode.strip().lower()
        if mode not in ("covalent", "noncovalent"):
            raise ValueError("mode must be 'covalent' or 'noncovalent'")
        c = self._constraints_root()
        stages = _ensure_dict(c, "stages")
        st = _ensure_dict(stages, stage_name)
        extra = _ensure_dict(st, "extra_sets")
        extra[mode] = [str(s).strip() for s in (sets or []) if str(s).strip()]

    # ---- defaults ----
    def constraints_get_defaults(self) -> Dict[str, Any]:
        c = self._constraints_root()
        d = c.get("defaults", {})
        return d if isinstance(d, dict) else {}

    def constraints_set_defaults(self, defaults_dict: Dict[str, Any]) -> None:
        c = self._constraints_root()
        c["defaults"] = defaults_dict if isinstance(defaults_dict, dict) else {}

    # ---- weights ----
    def constraints_get_weights(self, mode: str) -> Dict[str, float]:
        mode = mode.strip().lower()
        c = self._constraints_root()
        w = c.get("weights", {})
        if not isinstance(w, dict):
            return {}
        mw = w.get(mode, {})
        if not isinstance(mw, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in mw.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    def constraints_set_weight(self, mode: str, key: str, value: float) -> None:
        mode = mode.strip().lower()
        if mode not in ("covalent", "noncovalent"):
            raise ValueError("mode must be 'covalent' or 'noncovalent'")
        c = self._constraints_root()
        weights = _ensure_dict(c, "weights")
        mw = _ensure_dict(weights, mode)
        mw[str(key)] = float(value)

    def constraints_delete_weight(self, mode: str, key: str) -> bool:
        mode = mode.strip().lower()
        c = self._constraints_root()
        weights = c.get("weights", {})
        if not isinstance(weights, dict):
            return False
        mw = weights.get(mode, {})
        if not isinstance(mw, dict):
            return False
        if key in mw:
            del mw[key]
            return True
        return False

    # ---- raw constraints ----
    def constraints_get_raw_enabled(self, default: bool = False) -> bool:
        c = self._constraints_root()
        raw = c.get("raw", {})
        if not isinstance(raw, dict):
            return default
        return bool(raw.get("enabled", default))

    def constraints_set_raw_enabled(self, enabled: bool) -> None:
        c = self._constraints_root()
        raw = _ensure_dict(c, "raw")
        raw["enabled"] = bool(enabled)

    def constraints_get_raw_lines(self) -> List[str]:
        c = self._constraints_root()
        raw = c.get("raw", {})
        if not isinstance(raw, dict):
            return []
        lines = raw.get("lines", [])
        if not isinstance(lines, list):
            return []
        out: List[str] = []
        for ln in lines:
            s = str(ln).rstrip("\n")
            if s.strip() == "":
                # keep blank lines? optional — here we keep them
                out.append("")
            else:
                out.append(s)
        return out

    def constraints_set_raw_lines(self, lines: List[str]) -> None:
        c = self._constraints_root()
        raw = _ensure_dict(c, "raw")
        raw["lines"] = [str(x).rstrip("\n") for x in (lines or [])]

    # ---- sets ----
    def constraints_list_set_names(self) -> List[str]:
        c = self._constraints_root()
        sets = c.get("sets", {})
        if not isinstance(sets, dict):
            return []
        return [str(k) for k in sets.keys()]

    def constraints_get_set(self, name: str) -> Optional[Dict[str, Any]]:
        name = (name or "").strip()
        if not name:
            return None
        c = self._constraints_root()
        sets = c.get("sets", {})
        if not isinstance(sets, dict):
            return None
        v = sets.get(name)
        return v if isinstance(v, dict) else None

    def constraints_add_or_update_set(
        self,
        name: str,
        description: str = "",
        enabled: bool = True,
    ) -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("set name is required")
        c = self._constraints_root()
        sets = _ensure_dict(c, "sets")
        s = sets.get(name)
        if not isinstance(s, dict):
            s = {}
            sets[name] = s
        if description:
            s["description"] = str(description)
        else:
            s.pop("description", None)
        s["enabled"] = bool(enabled)
        _ensure_list(s, "items")

    def constraints_remove_set(self, name: str) -> bool:
        name = (name or "").strip()
        c = self._constraints_root()
        sets = c.get("sets", {})
        if not isinstance(sets, dict):
            return False
        if name in sets:
            del sets[name]
            return True
        return False

    # ---- items within sets ----
    def constraints_get_items(self, set_name: str) -> List[ConstraintItem]:
        s = self.constraints_get_set(set_name)
        if not isinstance(s, dict):
            return []
        items = s.get("items", [])
        if not isinstance(items, list):
            return []
        out: List[ConstraintItem] = []
        for it in items:
            if isinstance(it, dict):
                out.append(ConstraintItem.from_yaml_dict(it))
        return out

    def constraints_set_items(self, set_name: str, items: List[ConstraintItem]) -> None:
        s = self.constraints_get_set(set_name)
        if not isinstance(s, dict):
            # create if missing
            self.constraints_add_or_update_set(set_name, enabled=True)
            s = self.constraints_get_set(set_name) or {}
        s["items"] = [it.to_yaml_dict() for it in (items or [])]

    def constraints_add_item(self, set_name: str, item: ConstraintItem) -> None:
        s = self.constraints_get_set(set_name)
        if not isinstance(s, dict):
            self.constraints_add_or_update_set(set_name, enabled=True)
            s = self.constraints_get_set(set_name) or {}
        items = _ensure_list(s, "items")
        items.append(item.to_yaml_dict())

    def constraints_delete_item(self, set_name: str, index: int) -> bool:
        s = self.constraints_get_set(set_name)
        if not isinstance(s, dict):
            return False
        items = s.get("items", [])
        if not isinstance(items, list):
            return False
        if 0 <= index < len(items):
            del items[index]
            return True
        return False

    def constraints_set_enabled_for_set(self, set_name: str, enabled: bool) -> None:
        s = self.constraints_get_set(set_name)
        if not isinstance(s, dict):
            self.constraints_add_or_update_set(set_name, enabled=enabled)
            return
        s["enabled"] = bool(enabled)

    def constraints_get_enabled_for_set(self, set_name: str, default: bool = True) -> bool:
        s = self.constraints_get_set(set_name)
        if not isinstance(s, dict):
            return default
        return bool(s.get("enabled", default))


# ----------------------------
# CLI
# ----------------------------
def _print_constraints(ed: RosettaDictsEditor) -> None:
    c = ed._constraints_root()
    print("constraints.enabled:", ed.constraints_get_enabled())
    print("constraints.filename:", ed.constraints_get_filename())
    print("constraints.active_sets.covalent:", ed.constraints_get_active_sets("covalent"))
    print("constraints.active_sets.noncovalent:", ed.constraints_get_active_sets("noncovalent"))
    print("constraints.raw.enabled:", ed.constraints_get_raw_enabled())
    if ed.constraints_get_raw_enabled():
        print("constraints.raw.lines:", len(ed.constraints_get_raw_lines()))

    print("constraint sets:")
    for name in ed.constraints_list_set_names():
        en = ed.constraints_get_enabled_for_set(name, True)
        desc = ""
        s = ed.constraints_get_set(name)
        if isinstance(s, dict):
            desc = str(s.get("description", "")).strip()
        items = ed.constraints_get_items(name)
        print(f"  - {name} (enabled={en}) items={len(items)} desc={desc}")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="", help="Path to rosetta_dicts.yaml (optional)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show")

    p_lig = sub.add_parser("set-ligand")
    p_lig.add_argument("--resname", required=True)

    # constraints toggles
    p_cen = sub.add_parser("constraints-enable")
    p_cen.add_argument("--on", action="store_true")
    p_cen.add_argument("--off", action="store_true")

    p_cfile = sub.add_parser("constraints-set-filename")
    p_cfile.add_argument("--filename", required=True)

    p_cact = sub.add_parser("constraints-set-active")
    p_cact.add_argument("--mode", required=True, choices=["covalent", "noncovalent"])
    p_cact.add_argument("--sets", required=True, help="comma-separated set names")

    p_cw = sub.add_parser("constraints-set-weight")
    p_cw.add_argument("--mode", required=True, choices=["covalent", "noncovalent"])
    p_cw.add_argument("--key", required=True)
    p_cw.add_argument("--value", required=True, type=float)

    p_cr = sub.add_parser("constraints-raw")
    p_cr.add_argument("--enable", action="store_true")
    p_cr.add_argument("--disable", action="store_true")
    p_cr.add_argument("--file", default="", help="Load raw lines from file")

    # sets
    p_sadd = sub.add_parser("constraints-add-set")
    p_sadd.add_argument("--name", required=True)
    p_sadd.add_argument("--desc", default="")
    p_sadd.add_argument("--enabled", action="store_true")
    p_sadd.add_argument("--disabled", action="store_true")

    p_srm = sub.add_parser("constraints-remove-set")
    p_srm.add_argument("--name", required=True)

    # items
    p_iadd = sub.add_parser("constraints-add-item")
    p_iadd.add_argument("--set", dest="set_name", required=True)
    p_iadd.add_argument("--type", required=True, choices=["AtomPair", "Angle", "Dihedral", "Coordinate"])
    p_iadd.add_argument("--a", default="", help='Atom ref "ATOM,RES,CHAIN"')
    p_iadd.add_argument("--b", default="")
    p_iadd.add_argument("--c", default="")
    p_iadd.add_argument("--d", default="")
    p_iadd.add_argument("--atom", default="", help='For Coordinate: atom ref "ATOM,RES,CHAIN"')
    p_iadd.add_argument("--ref", default="", help='For Coordinate: ref atom ref "ATOM,RES,CHAIN"')
    p_iadd.add_argument("--ref-xyz", default="", help='For Coordinate: "x,y,z" (overrides --ref)')
    p_iadd.add_argument("--func", default="HARMONIC")
    p_iadd.add_argument("--x0", type=float, default=None)
    p_iadd.add_argument("--sd", type=float, default=None)

    p_idel = sub.add_parser("constraints-del-item")
    p_idel.add_argument("--set", dest="set_name", required=True)
    p_idel.add_argument("--index", type=int, required=True)

    args = ap.parse_args(argv)

    ed = RosettaDictsEditor(args.yaml or None).load()

    if args.cmd == "show":
        print(f"YAML: {ed.path} (backend={ed.backend_name})")
        print(f"ligand.resname = {ed.get_ligand_resname()}")
        _print_constraints(ed)
        return 0

    if args.cmd == "set-ligand":
        ed.set_ligand_resname(args.resname)
        ed.save()
        return 0

    if args.cmd == "constraints-enable":
        if args.on and args.off:
            raise SystemExit("--on and --off are mutually exclusive")
        if args.off:
            ed.constraints_set_enabled(False)
        else:
            ed.constraints_set_enabled(True)
        ed.save()
        return 0

    if args.cmd == "constraints-set-filename":
        ed.constraints_set_filename(args.filename)
        ed.save()
        return 0

    if args.cmd == "constraints-set-active":
        sets = [s.strip() for s in (args.sets or "").split(",") if s.strip()]
        ed.constraints_set_active_sets(args.mode, sets)
        ed.save()
        return 0

    if args.cmd == "constraints-set-weight":
        ed.constraints_set_weight(args.mode, args.key, args.value)
        ed.save()
        return 0

    if args.cmd == "constraints-raw":
        if args.enable and args.disable:
            raise SystemExit("--enable and --disable are mutually exclusive")
        if args.enable:
            ed.constraints_set_raw_enabled(True)
        if args.disable:
            ed.constraints_set_raw_enabled(False)

        if args.file:
            p = Path(args.file).expanduser().resolve()
            lines = p.read_text(encoding="utf-8").splitlines()
            ed.constraints_set_raw_lines(lines)

        ed.save()
        return 0

    if args.cmd == "constraints-add-set":
        enabled = True
        if args.disabled:
            enabled = False
        if args.enabled:
            enabled = True
        ed.constraints_add_or_update_set(args.name, description=args.desc, enabled=enabled)
        ed.save()
        return 0

    if args.cmd == "constraints-remove-set":
        ed.constraints_remove_set(args.name)
        ed.save()
        return 0

    if args.cmd == "constraints-add-item":
        t = args.type

        item = ConstraintItem(type=t, func=args.func, x0=args.x0, sd=args.sd)

        if t == "AtomPair":
            if not args.a or not args.b:
                raise SystemExit("AtomPair requires --a and --b")
            item.a = _parse_atomref(args.a)
            item.b = _parse_atomref(args.b)

        elif t == "Angle":
            if not args.a or not args.b or not args.c:
                raise SystemExit("Angle requires --a --b --c")
            item.a = _parse_atomref(args.a)
            item.b = _parse_atomref(args.b)
            item.c = _parse_atomref(args.c)

        elif t == "Dihedral":
            if not args.a or not args.b or not args.c or not args.d:
                raise SystemExit("Dihedral requires --a --b --c --d")
            item.a = _parse_atomref(args.a)
            item.b = _parse_atomref(args.b)
            item.c = _parse_atomref(args.c)
            item.d = _parse_atomref(args.d)

        elif t == "Coordinate":
            if not args.atom:
                raise SystemExit("Coordinate requires --atom")
            item.atom = _parse_atomref(args.atom)

            if args.ref_xyz:
                parts = [p.strip() for p in args.ref_xyz.split(",")]
                if len(parts) != 3:
                    raise SystemExit("--ref-xyz must be x,y,z")
                item.ref_xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            elif args.ref:
                item.ref = _parse_atomref(args.ref)
            else:
                # allow missing ref; caller might fill later (GUI)
                pass

        ed.constraints_add_item(args.set_name, item)
        ed.save()
        return 0

    if args.cmd == "constraints-del-item":
        ok = ed.constraints_delete_item(args.set_name, args.index)
        if not ok:
            raise SystemExit("Invalid set or index")
        ed.save()
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
