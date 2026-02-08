# ============================
# PATCH: server.py cleanup
# - remove duplicate routes for /ligands, /sequences, /runs
# - fix TASKS error handling
# - pass multi_seed through to runner.run_af3
# - keep ligand preview + ligand pdb download endpoints
# - keep runs metrics_html + downloads + run_rosetta
# ============================

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile
import subprocess, sys
import io

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DEFAULT_TZ = "America/Los_Angeles"

# ----------------------------
# Filesystem helpers (match your GUI)
# ----------------------------
def _default_user_cfg_dir() -> Path:
    return Path.home() / ".af3_pipeline"

USERS_ROOT = (_default_user_cfg_dir() / "users").resolve()
CURRENT_PROFILE_FILE = (_default_user_cfg_dir() / "current_profile.json").resolve()
SHARED_ROOT = (_default_user_cfg_dir() / "shared").resolve()

def _safe_profile_name(name: str) -> str:
    import re
    name = (name or "").strip()
    name = re.sub(r"[^\w\- ]+", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:64].strip("_")

def _profile_root(name: str) -> Path:
    return (USERS_ROOT / _safe_profile_name(name)).resolve()

def _shared_dir(*parts: str) -> Path:
    p = SHARED_ROOT
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()

def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _now_iso(tz_name: str = DEFAULT_TZ) -> str:
    return datetime.now(ZoneInfo(tz_name)).isoformat(timespec="seconds")

def _sanitize_jobname(s: str) -> str:
    return (s or "").strip().replace("+", "_")

def _ensure_profile_layout(profile: str) -> Dict[str, Path]:
    root = _profile_root(profile)
    root.mkdir(parents=True, exist_ok=True)

    gui_cache = (root / "gui_cache").resolve()
    gui_cache.mkdir(parents=True, exist_ok=True)

    cache_root = (root / "cache").resolve()
    (cache_root / "ligands").mkdir(parents=True, exist_ok=True)

    cfg_yaml = (root / "config.yaml").resolve()

    return {
        "root": root,
        "gui_cache": gui_cache,
        "sequence_cache": (gui_cache / "sequence_cache.json").resolve(),
        "ligand_cache": (gui_cache / "ligand_cache.json").resolve(),
        "queue": (gui_cache / "job_queue.json").resolve(),
        "runs_history": (gui_cache / "runs_history.json").resolve(),
        "config_yaml": cfg_yaml,
        "cache_root": cache_root,
        "ligand_cache_dir": (cache_root / "ligands").resolve(),
    }

def _load_current_profile_name() -> str:
    d = _read_json(CURRENT_PROFILE_FILE, {})
    return str(d.get("current", "") or "").strip()

def _save_current_profile_name(name: str) -> None:
    CURRENT_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _write_json(CURRENT_PROFILE_FILE, {"current": _safe_profile_name(name)})

def _user_root() -> Path:
    return (Path.home() / ".af3_pipeline").resolve()

def _current_profile_name() -> str:
    p = _user_root() / "current_profile.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return (d.get("current") or "").strip()
    return ""

# ----------------------------
# Ligand cache helpers
# ----------------------------
def _ligand_cache_file() -> Path:
    prof = _current_profile_name()
    if not prof:
        raise HTTPException(400, "No active profile selected")
    return (_user_root() / "users" / prof / "gui_cache" / "ligand_cache.json").resolve()

def _ligand_cache_root() -> Path:
    prof = _current_profile_name()
    if not prof:
        raise HTTPException(400, "No active profile selected")
    return (_user_root() / "users" / prof / "cache" / "ligands").resolve()

def _load_ligand_entry(name: str) -> dict:
    cache_file = _ligand_cache_file()
    if not cache_file.exists():
        raise HTTPException(404, "ligand_cache.json not found")
    d = json.loads(cache_file.read_text(encoding="utf-8"))
    entry = d.get(name)
    if not isinstance(entry, dict):
        raise HTTPException(404, f"Unknown ligand: {name}")
    return entry

def _lig_dir_for(name: str) -> Path:
    entry = _load_ligand_entry(name)
    lig_hash = (entry.get("hash") or "").strip()
    if not lig_hash:
        raise HTTPException(500, f"Ligand entry missing hash: {name}")
    return (_ligand_cache_root() / lig_hash).resolve()

# ----------------------------
# Runs helpers
# ----------------------------
def _runs_history_file() -> Path:
    prof = _current_profile_name()
    if not prof:
        raise HTTPException(400, "No active profile selected")
    return (_user_root() / "users" / prof / "gui_cache" / "runs_history.json").resolve()

def _jobs_root() -> Path:
    return (_user_root() / "jobs").resolve()

def _find_job_folder_by_history(rec: dict) -> Path | None:
    """
    Best-effort:
      1) use explicit job_dir in record if present
      2) else search ~/.af3_pipeline/jobs for newest folder starting with jobname
    """
    p = (rec.get("job_dir") or "").strip()
    if p:
        pp = Path(p).expanduser()
        if pp.exists() and pp.is_dir():
            return pp.resolve()

    jobname = (rec.get("jobname") or "").strip()
    if not jobname:
        return None
    root = _jobs_root()
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"{jobname}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None

def _find_metrics_csv(job_dir: Path) -> Path | None:
    for pat in ("metrics_summary.csv", "*metrics*.csv"):
        hits = sorted(job_dir.rglob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    return None

def _zip_dir_bytes(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in dir_path.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(dir_path)))
    buf.seek(0)
    return buf.read()

def _find_af_model(job_dir: Path, jobname: str) -> Path | None:
    p = job_dir / f"{jobname}_model.cif"
    if p.exists():
        return p
    hits = sorted(job_dir.rglob("*_model.cif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None

def _find_latest_rosetta_model(job_dir: Path) -> Path | None:
    rosetta_dirs = sorted(
        [p for p in job_dir.glob("rosetta_ligand_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    for d in rosetta_dirs:
        cand = d / "model_relaxed_restored_0001.pdb"
        if cand.exists():
            return cand
        hits = sorted(d.rglob("model_relaxed_restored_*.pdb"), key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    return None

# ----------------------------
# Backend lazy import
# ----------------------------
cfg = None
json_builder = None
runner = None
cache_utils = None
prepare_ligand_from_smiles = None
_canonical_smiles = None

def _ensure_backend_loaded():
    global cfg, json_builder, runner, cache_utils, prepare_ligand_from_smiles, _canonical_smiles
    if cfg is not None:
        return
    if not (os.environ.get("AF3_PIPELINE_CONFIG") or "").strip():
        raise RuntimeError("AF3_PIPELINE_CONFIG not set (profile not activated).")
    from af3_pipeline.config import cfg as _cfg
    from af3_pipeline import json_builder as _json_builder, runner as _runner, cache_utils as _cache_utils
    from af3_pipeline.ligand_utils import prepare_ligand_from_smiles as _pls, _canonical_smiles as _canon
    cfg = _cfg
    json_builder = _json_builder
    runner = _runner
    cache_utils = _cache_utils
    prepare_ligand_from_smiles = _pls
    _canonical_smiles = _canon

def _activate_profile_env(profile: str) -> Dict[str, Path]:
    paths = _ensure_profile_layout(profile)

    os.environ["AF3_PIPELINE_CONFIG"] = str(paths["config_yaml"])
    os.environ["AF3_PIPELINE_CACHE_ROOT"] = str(paths["cache_root"])
    os.environ["AF3_PIPELINE_MSA_CACHE"] = str(_shared_dir("msa"))
    os.environ["AF3_PIPELINE_TEMPLATE_CACHE"] = str(_shared_dir("templates"))

    _ensure_backend_loaded()
    try:
        cfg.reload(paths["config_yaml"])
    except Exception:
        pass

    _save_current_profile_name(profile)
    return paths

# ----------------------------
# API models
# ----------------------------
class ProfileCreate(BaseModel):
    name: str

class SequenceEntry(BaseModel):
    name: str
    sequence: str
    type: str = Field(..., description="protein/dna/rna")
    template: str = ""

class LigandEntry(BaseModel):
    name: str
    smiles: str

class JobSpec(BaseModel):
    jobname: str
    proteins: List[Dict[str, Any]] = []
    rna: List[Dict[str, Any]] = []
    dna: List[Dict[str, Any]] = []
    ligand: Dict[str, Any] = {}
    created_at: Optional[str] = None
    skip_rosetta: bool = False
    multi_seed: bool = False

# ----------------------------
# In-memory task state
# ----------------------------
TASKS: Dict[str, Dict[str, Any]] = {}   # task_id -> {status, logs, error, result...}

def _task_log(task_id: str, line: str):
    TASKS.setdefault(task_id, {"status": "running", "logs": [], "error": None, "result": None})
    TASKS[task_id]["logs"].append(str(line))

def _new_task_id(prefix: str = "task") -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="AF3 Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# ---- Profiles ----
@app.get("/profiles")
def list_profiles():
    USERS_ROOT.mkdir(parents=True, exist_ok=True)
    profiles = sorted([p.name for p in USERS_ROOT.iterdir() if p.is_dir()])
    return {"profiles": profiles, "current": _load_current_profile_name()}

@app.post("/profiles")
def create_profile(body: ProfileCreate):
    name = _safe_profile_name(body.name)
    if not name:
        raise HTTPException(400, "Invalid profile name")
    _ensure_profile_layout(name)
    return {"created": name}

@app.post("/profiles/{name}/activate")
def activate_profile(name: str):
    name = _safe_profile_name(name)
    if not name:
        raise HTTPException(400, "Invalid profile name")
    paths = _activate_profile_env(name)
    return {"active": name, "paths": {k: str(v) for k, v in paths.items()}}

@app.delete("/profiles/{name}")
def delete_profile(name: str):
    name = _safe_profile_name(name)
    root = _profile_root(name)
    if root.exists():
        shutil.rmtree(root)
    cur = _load_current_profile_name()
    if cur == name:
        _save_current_profile_name("")
    return {"deleted": name}

# ============================
# SEQUENCES (single /sequences endpoint)
# ============================
@app.get("/sequences")
def list_sequences():
    prof = _current_profile_name()
    if not prof:
        return {"protein": [], "dna": [], "rna": []}

    p = (_user_root() / "users" / prof / "gui_cache" / "sequence_cache.json").resolve()
    if not p.exists():
        return {"protein": [], "dna": [], "rna": []}

    d = json.loads(p.read_text(encoding="utf-8"))
    groups = {"protein": [], "dna": [], "rna": []}
    for name, val in d.items():
        if isinstance(val, dict):
            t = str(val.get("type", "")).strip().lower()
            if t in groups:
                groups[t].append(name)
        else:
            groups["protein"].append(name)

    for k in groups:
        groups[k] = sorted(set(groups[k]), key=str.lower)
    return groups

@app.get("/sequences/{kind}/{name}")
def get_sequence(kind: str, name: str):
    kind = (kind or "").strip().lower()
    if kind not in {"protein", "dna", "rna"}:
        raise HTTPException(400, "kind must be protein|dna|rna")

    prof = _current_profile_name()
    if not prof:
        raise HTTPException(400, "No active profile selected")

    p = (_user_root() / "users" / prof / "gui_cache" / "sequence_cache.json").resolve()
    if not p.exists():
        raise HTTPException(404, "sequence_cache.json not found")

    d = json.loads(p.read_text(encoding="utf-8"))
    val = d.get(name)
    if val is None:
        raise HTTPException(404, f"Unknown sequence: {name}")

    if isinstance(val, dict):
        return {
            "name": name,
            "sequence": val.get("sequence", ""),
            "template": val.get("template", ""),
            "type": str(val.get("type", "")),
        }
    return {"name": name, "sequence": str(val), "template": "", "type": "protein"}

@app.post("/sequences")
def upsert_sequence(entry: SequenceEntry, profile: Optional[str] = None):
    profile = profile or _load_current_profile_name()
    if not profile:
        raise HTTPException(400, "No active profile")

    paths = _ensure_profile_layout(profile)
    data = _read_json(paths["sequence_cache"], {})
    data[entry.name] = {
        "sequence": entry.sequence,
        "type": (entry.type or "").strip().lower(),
        "template": entry.template,
    }
    _write_json(paths["sequence_cache"], data)
    return {"ok": True}

@app.delete("/sequences/{name}")
def delete_sequence(name: str, profile: Optional[str] = None):
    profile = profile or _load_current_profile_name()
    if not profile:
        raise HTTPException(400, "No active profile")

    paths = _ensure_profile_layout(profile)
    data = _read_json(paths["sequence_cache"], {})
    data.pop(name, None)
    _write_json(paths["sequence_cache"], data)
    return {"ok": True}

# ============================
# LIGANDS (single /ligands endpoint)
# ============================
@app.get("/ligands")
def list_ligands():
    cache_file = _ligand_cache_file()
    if not cache_file.exists():
        return []
    d = json.loads(cache_file.read_text(encoding="utf-8"))
    out = []
    for name, entry in d.items():
        if isinstance(entry, dict):
            out.append({"name": name, "smiles": entry.get("smiles", ""), "hash": entry.get("hash", "")})
    return sorted(out, key=lambda x: x["name"].lower())

@app.post("/ligands")
def create_ligand(entry: LigandEntry, profile: Optional[str] = None):
    profile = profile or _load_current_profile_name()
    if not profile:
        raise HTTPException(400, "No active profile")

    _activate_profile_env(profile)
    _ensure_backend_loaded()

    smiles = _canonical_smiles(entry.smiles.strip())
    name = (entry.name or "").strip() or smiles

    cif_path = Path(prepare_ligand_from_smiles(smiles, name=name, skip_if_cached=False)).expanduser()
    if not cif_path.exists():
        raise HTTPException(500, f"Ligand CIF not created: {cif_path}")

    lig_hash = cache_utils.compute_hash(smiles)

    paths = _ensure_profile_layout(profile)
    data = _read_json(paths["ligand_cache"], {})
    data[name] = {"smiles": smiles, "hash": lig_hash, "path": str(cif_path)}
    _write_json(paths["ligand_cache"], data)
    return {"ok": True, "name": name, "path": str(cif_path), "hash": lig_hash}

@app.delete("/ligands/{name}")
def delete_ligand(name: str, profile: Optional[str] = None):
    profile = profile or _load_current_profile_name()
    if not profile:
        raise HTTPException(400, "No active profile")

    paths = _ensure_profile_layout(profile)
    data = _read_json(paths["ligand_cache"], {})
    data.pop(name, None)
    _write_json(paths["ligand_cache"], data)
    return {"ok": True}

@app.get("/ligands/{name}/preview")
def ligand_preview(name: str):
    lig_dir = _lig_dir_for(name)
    svg = lig_dir / "LIG.svg"
    if not svg.exists():
        raise HTTPException(404, f"Preview not found: {svg}")
    return FileResponse(svg, media_type="image/svg", filename="LIG.svg")

@app.get("/ligands/{name}/pdb")
def ligand_pdb(name: str):
    lig_dir = _lig_dir_for(name)
    pdb = lig_dir / "LIG.pdb"
    if not pdb.exists():
        raise HTTPException(404, f"PDB not found: {pdb}")
    return FileResponse(pdb, media_type="chemical/x-pdb", filename=f"{name}_LIG.pdb")

# ============================
# RUN PIPELINE
# ============================
@app.post("/run")
def run_job(spec: JobSpec, profile: Optional[str] = None):
    profile = profile or _load_current_profile_name()
    if not profile:
        raise HTTPException(400, "No active profile")

    _activate_profile_env(profile)

    d = spec.model_dump()
    d["jobname"] = _sanitize_jobname(d["jobname"])
    d.setdefault("created_at", _now_iso())

    task_id = _new_task_id("run")
    TASKS[task_id] = {"status": "running", "logs": [], "error": None, "result": None}

    try:
        _task_log(task_id, f"🚀 Starting job: {d['jobname']}")
        json_builder._progress_hook = lambda msg: _task_log(task_id, f"🧬 {msg}")

        # build_input signature in your GUI:
        try:
            json_path = json_builder.build_input(
                jobname=d["jobname"],
                proteins=d.get("proteins", []) or [],
                rna=d.get("rna", []) or [],
                dna=d.get("dna", []) or [],
                ligand=d.get("ligand", {}) or {},
            )
        except TypeError:
            # legacy json_builder
            rna_one = (d.get("rna") or [{}])[0] if isinstance(d.get("rna"), list) else d.get("rna") or {}
            dna_one = (d.get("dna") or [{}])[0] if isinstance(d.get("dna"), list) else d.get("dna") or {}
            json_path = json_builder.build_input(
                jobname=d["jobname"],
                proteins=d.get("proteins", []) or [],
                rna=rna_one,
                dna=dna_one,
                ligand=d.get("ligand", {}) or {},
            )

        _task_log(task_id, f"✅ JSON built: {json_path}")

        # ✅ pass multi_seed through
        runner.run_af3(
            str(json_path),
            job_name=d["jobname"],
            auto_analyze=False,
            multi_seed=bool(d.get("multi_seed", False)),
        )
        _task_log(task_id, "✅ AF3 run finished")

        from af3_pipeline.analysis.post_analysis import run_analysis
        run_analysis(
            d["jobname"],
            model=None,
            multi_seed=bool(d.get("multi_seed", False)),
            skip_rosetta=bool(d.get("skip_rosetta", False)),
        )
        _task_log(task_id, "✅ Post-analysis finished")

        # record runs_history (match GUI behavior)
        paths = _ensure_profile_layout(profile)
        hist = _read_json(paths["runs_history"], [])
        hist = hist if isinstance(hist, list) else []

        rec = dict(d)
        rec["json_path"] = str(json_path)
        rec["finished_at"] = _now_iso()
        # (optional but recommended): if you can determine job_dir here, store it:
        # rec["job_dir"] = "<absolute path to ~/.af3_pipeline/jobs/<...>>"
        hist.insert(0, rec)
        _write_json(paths["runs_history"], hist)

        TASKS[task_id]["status"] = "done"
        TASKS[task_id]["result"] = {"jobname": d["jobname"], "json_path": str(json_path)}
        return {"task_id": task_id}

    except Exception as e:
        msg = str(e)
        # ✅ fix: update the specific task entry, not TASKS dict
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = msg
        TASKS[task_id]["logs"].append("❌ ERROR:")
        TASKS[task_id]["logs"].append(msg)
        raise

@app.get("/tasks/{task_id}")
def task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "Unknown task")
    return TASKS[task_id]

# ============================
# RUNS (single /runs endpoint)
# ============================
@app.get("/runs")
def list_runs():
    p = _runs_history_file()
    if not p.exists():
        return []
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, dict):
            out.append({
                "id": i,
                "jobname": (r.get("jobname") or "").strip(),
                "finished_at": (r.get("finished_at") or r.get("created_at") or "").strip(),
                "record": r,
            })
    return out

@app.get("/runs/{run_id}/metrics_html")
def run_metrics_html(run_id: int):
    runs = list_runs()
    if run_id < 0 or run_id >= len(runs):
        raise HTTPException(404, "Invalid run id")

    rec = runs[run_id]["record"]
    job_dir = _find_job_folder_by_history(rec)
    if not job_dir:
        return JSONResponse({"html": "<i>Could not locate job folder.</i>"})

    metrics = _find_metrics_csv(job_dir)
    if not metrics:
        return JSONResponse({"html": f"<i>No metrics CSV found under {job_dir}</i>"})

    # lazy import to avoid any GUI/Qt import-time surprises
    from apps.gui import runs_metrics

    class _Dummy: pass
    dummy = _Dummy()
    html = runs_metrics._runs_metrics_html(dummy, metrics)
    return {"html": html}

@app.get("/runs/{run_id}/download")
def run_download(run_id: int, kind: str):
    runs = list_runs()
    if run_id < 0 or run_id >= len(runs):
        raise HTTPException(404, "Invalid run id")

    rec = runs[run_id]["record"]
    jobname = (rec.get("jobname") or "").strip()

    job_dir = _find_job_folder_by_history(rec)
    if not job_dir:
        raise HTTPException(404, "Job folder not found")

    kind = (kind or "").strip().lower()

    if kind == "job_folder":
        data = _zip_dir_bytes(job_dir)
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_dir.name}.zip"'},
        )

    if kind == "af_model":
        p = _find_af_model(job_dir, jobname)
        if not p:
            raise HTTPException(404, "AF model not found")
        return FileResponse(p, filename=p.name)

    if kind == "rosetta_model":
        p = _find_latest_rosetta_model(job_dir)
        if not p:
            raise HTTPException(404, "Rosetta model not found")
        return FileResponse(p, filename=p.name)

    raise HTTPException(400, "kind must be job_folder|af_model|rosetta_model")

@app.post("/runs/{run_id}/run_rosetta")
def run_rosetta(run_id: int, multi_seed: bool = False):
    runs = list_runs()
    if run_id < 0 or run_id >= len(runs):
        raise HTTPException(404, "Invalid run id")

    rec = runs[run_id]["record"]
    jobname = (rec.get("jobname") or "").strip()
    if not jobname:
        raise HTTPException(400, "Missing jobname")

    cmd = [sys.executable, "-m", "af3_pipeline.analysis.post_analysis", "--job", jobname]
    if multi_seed:
        cmd.append("--multi_seed")

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise HTTPException(500, f"rosetta failed\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}")

    return {"ok": True, "stdout": p.stdout}
