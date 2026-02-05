#!/usr/bin/env python3
"""
main_window.py (PyQt6)
======================
Drop-in replacement for your redesigned AlphaFold GUI.

Key updates vs old version:
- PyQt6 (not PyQt5)
- Sidebar navigation: navList -> pagesStack
- New Sequences / Ligands / Alphafold / Runs / Config pages
- Dynamic macromolecule lists (Proteins/DNA/RNA) on Alphafold page (scrollable + add buttons)
- Embedded queue on Alphafold page (queueList + move up/down/remove + runQueueButton)
- Separate ligand dropdown on Ligands page (view/save) vs Alphafold run ligand dropdown (runLigandDropdown)
- Keeps + reuses your existing AF3 backend hooks (json_builder, runner, cache_utils, ligand_utils)

NOTE:
- This implementation supports multiple Proteins/DNA/RNA entries in the UI.
  If your current json_builder.build_input() only supports single RNA/DNA dicts,
  you may need to update json_builder to accept lists. The code tries to pass
  lists as-is (preferred), falling back to first item if needed.
"""

from __future__ import annotations

import os
import sys
import json
import random
import subprocess
import traceback
import yaml
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QRect, QRectF, QPoint, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QPen, QFontMetrics, QWheelEvent, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QPlainTextEdit,
    QLineEdit,
    QComboBox,
    QWidget,
    QFrame,
    QLabel,
    QSpinBox,
    QCheckBox,
    QScrollArea,
    QGroupBox,
    QInputDialog,
    QScrollBar,
    QToolButton, 
    QMenu
)

# ==========================
# 🧠 Lazy backend import (after profile env set)
# ==========================
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

    # ✅ Guard: don't load backend until profile config is chosen
    if not (os.environ.get("AF3_PIPELINE_CONFIG") or "").strip():
        raise RuntimeError("Backend loaded before AF3_PIPELINE_CONFIG is set (profile not activated yet).")

    from af3_pipeline.config import cfg as _cfg
    from af3_pipeline import json_builder as _json_builder, runner as _runner, cache_utils as _cache_utils
    from af3_pipeline.ligand_utils import prepare_ligand_from_smiles as _pls, _canonical_smiles as _canon

    cfg = _cfg
    json_builder = _json_builder
    runner = _runner
    cache_utils = _cache_utils
    prepare_ligand_from_smiles = _pls
    _canonical_smiles = _canon


# ==========================
# 📁 Paths & Cache (profile-scoped)
# ==========================
def norm_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()

def _default_user_cfg_dir() -> Path:
    return Path.home() / ".af3_pipeline"

# These are initialized AFTER profile activation
GUI_CACHE_DIR: Path = norm_path(_default_user_cfg_dir() / "users" / "_UNSET_" / "gui_cache")
SEQUENCE_CACHE_FILE: Path = GUI_CACHE_DIR / "sequence_cache.json"
LIGAND_CACHE_FILE: Path   = GUI_CACHE_DIR / "ligand_cache.json"
QUEUE_FILE: Path          = GUI_CACHE_DIR / "job_queue.json"
NOTES_FILE: Path          = GUI_CACHE_DIR / "notes.txt"
RUNS_HISTORY_FILE: Path   = GUI_CACHE_DIR / "runs_history.json"

LIGAND_CACHE: Path        = norm_path(_default_user_cfg_dir() / "users" / "_UNSET_" / "cache" / "ligands")

# ==========================
# ⚙️ Constants
# ==========================
PTM_CHOICES = {
    "None": None,

    # Phosphorylation
    "Phosphoserine (pSer)": "SEP",
    "Phosphothreonine (pThr)": "TPO",
    "Phosphotyrosine (pTyr)": "PTR",
    "Phosphohistidine": "HIP",
    "Phosphocysteine": "CSP",

    # Methylation
    "Monomethyllysine": "MLY",
    "Dimethyllysine": "M2L",
    "Trimethyllysine": "M3L",
    "Monomethylarginine": "MMA",
    "Dimethylarginine": "DMA",

    # Acetylation
    "N6-Acetyllysine": "ALY",
    "N-terminal acetylation": "ACE",
    "O-Acetylserine": "ASE",

    # Hydroxylation
    "Hydroxyproline": "HYP",
    "Hydroxylysine": "HYL",
    "Hydroxycysteine": "CSO",

    # Oxidation / redox
    "Methionine sulfoxide": "MSO",
    "Cysteine sulfinic acid": "SFA",
    "Cysteine sulfonic acid": "CSA",

    # Sulfation
    "Sulfotyrosine": "TYS",
    "Cysteine sulfinic acid": "SFA",
    "Cysteine sulfonic acid": "CSA",

    # Carbonyl / carboxyl related
    "Pyroglutamate": "PCA",
    "Carboxymethyllysine": "CML",
    "Dehydroalanine": "DHA",

    # Seleno / formyl
    "Selenomethionine": "MSE",
    "N-Formylmethionine": "FME",

    # Glycosylated residues (common PDB sugars)
    "N-Acetylglucosamine": "NAG",
    "Mannose": "MAN",
    "Galactose": "GAL",
    "Fucose": "FUC",
    "Sialic acid": "SIA",

    # Lipidations (as PDB small-molecule attachments)
    "Myristoylation": "MYS",
    "Palmitoylation": "PAM",
    "Farnesylation": "FAR",
    "Geranylgeranylation": "GGG",
}
ION_CHOICES       = ["MG", "CA", "ZN", "NA", "K", "CL", "MN", "FE", "CO", "CU"]
COFACTOR_CHOICES  = ["ATP", "ADP", "AMP", "NAD", "NADP", "FAD", "CoA", "SAM", "GTP", "GDP"]

ATOM_MAP = {
    "Cysteine (SG)": "SG",
    "Lysine (NZ)": "NZ",
    "Tyrosine (OH)": "OH",
    "Histidine (ND1)": "ND1",
    "Histidine (NE2)": "NE2",
}

# ==========================
# 🧭 Small helpers
# ==========================
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default

def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

DEFAULT_TZ = "America/Los_Angeles"

def _get_tz_name() -> str:
    """
    cfg is None until a profile is activated (lazy backend import).
    So we must fall back to a stable default here.
    """
    try:
        if cfg is not None:
            return str(cfg.get("timezone", DEFAULT_TZ) or DEFAULT_TZ)
    except Exception:
        pass
    return DEFAULT_TZ

def _now_iso() -> str:
    return datetime.now(ZoneInfo(_get_tz_name())).isoformat(timespec="seconds")

def _sanitize_jobname(s: str) -> str:
    return (s or "").strip().replace("+", "_")

def _msg_info(parent, title, text):
    QMessageBox.information(parent, title, text)

def _msg_warn(parent, title, text):
    QMessageBox.warning(parent, title, text)

def _msg_err(parent, title, text):
    QMessageBox.critical(parent, title, text)

def _msg_yesno(parent, title, text) -> bool:
    res = QMessageBox.question(
        parent,
        title,
        text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    return res == QMessageBox.StandardButton.Yes

def select_multiple(parent, title, items) -> str:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    layout = QVBoxLayout(dlg)
    lst = QListWidget(dlg)
    for val in items:
        itm = QListWidgetItem(val)
        itm.setCheckState(Qt.CheckState.Unchecked)
        lst.addItem(itm)
    layout.addWidget(lst)
    ok = QPushButton("OK", dlg)
    ok.clicked.connect(dlg.accept)
    layout.addWidget(ok)
    if dlg.exec():
        selected = [
            lst.item(i).text()
            for i in range(lst.count())
            if lst.item(i).checkState() == Qt.CheckState.Checked
        ]
        return ",".join(selected)
    return ""

def resource_path(rel: str) -> Path:
    """
    Return absolute path to a resource bundled by PyInstaller,
    or the correct path in dev mode.
    Works for both --onefile and normal execution.
    """
    if getattr(sys, "frozen", False):        # running as packaged EXE
        base = Path(sys._MEIPASS)
    else:                                    # running from source
        # adjust this so it points to apps/gui/assets from main.py
        base = Path(__file__).resolve().parent

    return base / rel

# ==========================
# 👤 Profiles (module-level)
# ==========================
CURRENT_PROFILE_FILE = norm_path(_default_user_cfg_dir() / "current_profile.json")
USERS_ROOT = norm_path(_default_user_cfg_dir() / "users")

def _safe_profile_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w\- ]+", "", name)   # keep letters/numbers/_/-/space
    name = re.sub(r"\s+", "_", name)
    return name[:64].strip("_")

def _list_profiles() -> list[str]:
    if not USERS_ROOT.exists():
        return []
    out = []
    for p in USERS_ROOT.iterdir():
        if p.is_dir():
            out.append(p.name)
    return sorted(out)

def _load_current_profile_name() -> str:
    try:
        if CURRENT_PROFILE_FILE.exists():
            d = json.loads(CURRENT_PROFILE_FILE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return str(d.get("current", "") or "").strip()
    except Exception:
        pass
    return ""

def _save_current_profile_name(name: str) -> None:
    CURRENT_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_PROFILE_FILE.write_text(json.dumps({"current": name}, indent=2), encoding="utf-8")

def _profile_root(name: str) -> Path:
    return USERS_ROOT / name

def _set_active_profile_paths(profile_name: str) -> Path:
    """
    Repoint global cache locations to the active profile.
    Returns profile_root.
    """
    global GUI_CACHE_DIR, SEQUENCE_CACHE_FILE, LIGAND_CACHE_FILE, QUEUE_FILE, NOTES_FILE, RUNS_HISTORY_FILE, LIGAND_CACHE

    profile_name = _safe_profile_name(profile_name)
    root = norm_path(_profile_root(profile_name))
    root.mkdir(parents=True, exist_ok=True)

    # GUI cache
    GUI_CACHE_DIR = norm_path(root / "gui_cache")
    GUI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    SEQUENCE_CACHE_FILE = GUI_CACHE_DIR / "sequence_cache.json"
    LIGAND_CACHE_FILE   = GUI_CACHE_DIR / "ligand_cache.json"
    QUEUE_FILE          = GUI_CACHE_DIR / "job_queue.json"
    NOTES_FILE          = GUI_CACHE_DIR / "notes.txt"
    RUNS_HISTORY_FILE   = GUI_CACHE_DIR / "runs_history.json"

    # Ligand structure cache (per-profile)
    LIGAND_CACHE = norm_path(root / "cache" / "ligands")
    LIGAND_CACHE.mkdir(parents=True, exist_ok=True)

    return root

SHARED_ROOT = norm_path(_default_user_cfg_dir() / "shared")
SHARED_ROOT.mkdir(parents=True, exist_ok=True)

def _shared_dir(*parts: str) -> Path:
    p = SHARED_ROOT
    for part in parts:
        p = p / part
    p = norm_path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ==========================
# 🗒 Notes
# ==========================
class NotesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🗒 Notes")
        self.resize(600, 450)
        layout = QVBoxLayout(self)
        self.text_edit = QPlainTextEdit(self)
        layout.addWidget(self.text_edit)
        if NOTES_FILE.exists():
            self.text_edit.setPlainText(NOTES_FILE.read_text(encoding="utf-8"))
        save_button = QPushButton("Save + Close", self)
        save_button.clicked.connect(self.save_and_close)
        layout.addWidget(save_button)

    def save_and_close(self):
        NOTES_FILE.write_text(self.text_edit.toPlainText(), encoding="utf-8")
        self.accept()

class PtmDialog(QDialog):
    """
    Popout editor for multiple PTMs.
    Returns list of dicts: [{"label": <ui label>, "ccd": <CCD>, "pos": <str>}, ...]
    """
    def __init__(self, parent, ptm_choices: dict[str, Optional[str]], initial: Optional[list[dict[str, str]]] = None):
        super().__init__(parent)
        self.setWindowTitle("PTMs")
        self.resize(520, 360)

        self.ptm_choices = ptm_choices
        self._rows: list[tuple[QComboBox, QLineEdit, QPushButton]] = []

        outer = QVBoxLayout(self)

        # rows container
        self.rows_widget = QWidget(self)
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)
        outer.addWidget(self.rows_widget)

        # controls
        controls = QHBoxLayout()
        self.add_btn = QPushButton("➕ Add PTM", self)
        self.add_btn.clicked.connect(self._add_row)
        controls.addWidget(self.add_btn)
        controls.addStretch(1)
        outer.addLayout(controls)

        # save/cancel
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.save_btn = QPushButton("Save", self)
        self.cancel_btn = QPushButton("Cancel", self)
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.save_btn)
        outer.addLayout(btns)

        # seed initial rows
        if initial:
            for it in initial:
                self._add_row(label=it.get("label"), ccd=it.get("ccd"), pos=it.get("pos"))
        else:
            # start with one row by default (optional)
            self._add_row()

    def _add_row(self, *, label: Optional[str] = None, ccd: Optional[str] = None, pos: Optional[str] = None):
        row = QHBoxLayout()

        dd = QComboBox(self)
        # Build dropdown from PTM_CHOICES
        # (keep "None" as a choice; you can also omit if you want)
        for lab, c in self.ptm_choices.items():
            dd.addItem(lab, c)

        # restore selection by ccd or label
        if ccd is not None:
            for i in range(dd.count()):
                if dd.itemData(i) == ccd:
                    dd.setCurrentIndex(i)
                    break
        elif label:
            idx = dd.findText(label)
            if idx >= 0:
                dd.setCurrentIndex(idx)

        pos_edit = QLineEdit(self)
        pos_edit.setPlaceholderText("Residue position (e.g. 23)")
        if pos:
            pos_edit.setText(str(pos))

        rm = QPushButton("🗑", self)
        rm.setFixedWidth(40)

        def _remove():
            # Remove from layout + delete widgets
            for i, (ddd, ppp, rrr) in enumerate(list(self._rows)):
                if ddd is dd and ppp is pos_edit and rrr is rm:
                    self._rows.pop(i)
                    break
            dd.setParent(None); dd.deleteLater()
            pos_edit.setParent(None); pos_edit.deleteLater()
            rm.setParent(None); rm.deleteLater()
            # also remove the layout item wrapper by rebuilding the rows widget
            self._rebuild_rows()

        rm.clicked.connect(_remove)

        row.addWidget(dd, 3)
        row.addWidget(pos_edit, 2)
        row.addWidget(rm, 0)

        self._rows.append((dd, pos_edit, rm))
        self._rebuild_rows()

    def _rebuild_rows(self):
        # Clear rows_layout
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            if item.layout():
                # layouts are owned, no direct delete needed
                pass

        # Re-add rows
        for dd, pos_edit, rm in self._rows:
            h = QHBoxLayout()
            h.addWidget(dd, 3)
            h.addWidget(pos_edit, 2)
            h.addWidget(rm, 0)
            self.rows_layout.addLayout(h)

        self.rows_layout.addStretch(1)

    def get_ptms(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for dd, pos_edit, _rm in self._rows:
            label = dd.currentText().strip()
            ccd = dd.currentData()

            # Skip empty/None PTMs unless you want to keep explicit "None"
            if ccd in (None, "None") or label.lower() == "none":
                continue

            pos = pos_edit.text().strip()
            if not pos:
                # allow missing pos? I'd recommend requiring it:
                continue

            out.append({
                "label": label,
                "ccd": str(ccd),
                "pos": pos,
            })
        return out

class NewProfileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Profile")
        self.resize(420, 140)

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("Profile name:", self))

        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("e.g. Oliver")
        outer.addWidget(self.name_edit)

        btns = QHBoxLayout()
        btns.addStretch(1)
        save = QPushButton("Save", self)
        cancel = QPushButton("Cancel", self)
        btns.addWidget(save)
        btns.addWidget(cancel)
        outer.addLayout(btns)

        cancel.clicked.connect(self.reject)
        save.clicked.connect(self.accept)

    def profile_name(self) -> str:
        return self.name_edit.text().strip()

# ==========================
# 🧵 Threads
# ==========================
class BuildThread(QThread):
    finished = pyqtSignal(str)
    failed   = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, jobname: str, proteins: list[dict], rna: Any, dna: Any, ligand: dict):
        super().__init__()
        self.jobname = jobname
        self.proteins, self.rna, self.dna, self.ligand = proteins, rna, dna, ligand

    def run(self):
        try:
            _ensure_backend_loaded()
            def progress_hook(msg):  # noqa: ANN001
                self.progress.emit(str(msg))

            json_builder._progress_hook = progress_hook  # type: ignore[attr-defined]

            # Preferred: pass lists through if backend supports it
            try:
                json_path = json_builder.build_input(
                    jobname=self.jobname,
                    proteins=self.proteins,
                    rna=self.rna,
                    dna=self.dna,
                    ligand=self.ligand,
                )
            except TypeError:
                # Fallback: old backend expects single RNA/DNA dicts
                rna_one = self.rna[0] if isinstance(self.rna, list) and self.rna else {"sequence": "", "modification": "", "pos": ""}
                dna_one = self.dna[0] if isinstance(self.dna, list) and self.dna else {"sequence": "", "modification": "", "pos": ""}
                json_path = json_builder.build_input(
                    jobname=self.jobname,
                    proteins=self.proteins,
                    rna=rna_one,
                    dna=dna_one,
                    ligand=self.ligand,
                )

            self.finished.emit(str(json_path))
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if hasattr(json_builder, "_progress_hook"):
                delattr(json_builder, "_progress_hook")


class RunThread(QThread):
    finished = pyqtSignal(str)
    failed   = pyqtSignal(str)

    def __init__(self, json_path: str, job_name: str, auto_analyze: bool = False, multi_seed: bool = False):
        super().__init__()
        self.json_path = json_path
        self.job_name  = job_name
        self.auto_analyze = auto_analyze
        self.multi_seed = multi_seed

    def run(self):
        try:
            _ensure_backend_loaded()
            runner.run_af3(
                self.json_path,
                job_name=self.job_name,
                auto_analyze=self.auto_analyze,
                multi_seed=self.multi_seed,
            )
            self.finished.emit(self.json_path)
        except Exception as e:
            self.failed.emit(str(e))

class DetectConfigWorker(QThread):
    done = pyqtSignal(dict)
    log = pyqtSignal(str)

    def __init__(self, current_platform: str, current_distro: str | None = None):
        super().__init__()
        self.current_platform = (current_platform or "").strip().lower() or "wsl"
        self.current_distro = (current_distro or "").strip() if current_distro else ""

    def run(self):
        try:
            _ensure_backend_loaded()
            suggestions: dict[str, Any] = {}
            if sys.platform.startswith("win"):
                suggestions.update(self._detect_on_windows())
            else:
                suggestions.update(self._detect_on_linux())

            self.done.emit(suggestions)
        except Exception as e:
            self.log.emit(f"❌ Auto-detect failed: {e}")
            self.done.emit({})

    # -------------------------
    # Windows + WSL detection
    # -------------------------
    def _run_cmd(self, cmd: list[str]) -> tuple[int, str, str]:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out, err = p.communicate()

        # ✅ NUL bytes can appear in Windows command output sometimes
        out = (out or "").replace("\x00", "").strip()
        err = (err or "").replace("\x00", "").strip()

        return p.returncode, out, err


    def _detect_on_windows(self) -> dict[str, Any]:
        s: dict[str, Any] = {}
        self.log.emit("🔎 Detecting on Windows…")

        # gui_dir = folder containing this file (best-effort)
        try:
            gui_dir = Path(__file__).resolve().parent
            s["gui_dir"] = str(gui_dir)
        except Exception:
            pass

        s["platform"] = "wsl"  # your main target

        rc, out, err = self._run_cmd(["wsl.exe", "-l", "-v"])
        if rc == 0 and out:
            distro = self._parse_default_wsl_distro(out) or self.current_distro
            distro = (distro or "").replace("\x00", "").strip()
            if distro:
                s["wsl_distro"] = distro
                self.log.emit(f"✅ WSL distro: {distro}")
            else:
                self.log.emit("⚠️ Could not determine WSL default distro from `wsl.exe -l -v` output.")
        else:
            self.log.emit("⚠️ wsl.exe not found or failed. Is WSL installed?")

        # docker_bin: if docker works on Windows, keep "docker"
        rc, _, _ = self._run_cmd(["docker", "version"])
        if rc == 0:
            s["docker_bin"] = "docker"
            self.log.emit("✅ Docker available on Windows")
        else:
            # Some users only have docker inside WSL
            distro = s.get("wsl_distro", self.current_distro)
            if distro:
                rc2, _, _ = self._run_cmd(["wsl.exe", "-d", str(distro), "--", "docker", "version"])
                if rc2 == 0:
                    s["docker_bin"] = "docker"
                    self.log.emit("✅ Docker available inside WSL")
                else:
                    self.log.emit("⚠️ Docker not detected (Windows or WSL).")

        # Now run WSL probes if we have a distro
        distro = (s.get("wsl_distro", self.current_distro) or "").replace("\x00", "").strip()
        if distro:
            s.update(self._detect_inside_wsl(distro))
        s.setdefault("alphafold_docker_image", "alphafold3")

        return s

    def _parse_default_wsl_distro(self, text: str) -> str | None:
        text = (text or "").replace("\x00", "")
        # Try to match the starred default distro line
        for line in text.splitlines():
            line = line.replace("\x00", "").strip()
            m = re.match(r"^\*\s*([A-Za-z0-9_.-]+)\b", line)
            if m:
                return m.group(1)

        # Fallback: if only one distro row exists, take first token
        lines = [
            ln.replace("\x00", "").strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.lower().startswith("name")
        ]
        if len(lines) == 1:
            m2 = re.match(r"^([A-Za-z0-9_.-]+)\b", lines[0])
            return m2.group(1) if m2 else None

        return None



    def _detect_inside_wsl(self, distro: str) -> dict[str, Any]:
        s: dict[str, Any] = {}
        self.log.emit(f"🔎 Detecting inside WSL ({distro})…")

        def wsl(cmd: str) -> tuple[int, str, str]:
            return self._run_cmd(["wsl.exe", "-d", distro, "--", "bash", "-lc", cmd])

        # linux_home_root
        rc, out, _ = wsl("echo $HOME")
        if rc == 0 and out.startswith("/"):
            s["linux_home_root"] = out
            self.log.emit(f"✅ linux_home_root: {out}")

        # af3_dir: try a few common paths, then shallow find
        candidates = [
            "$HOME/Repositories/alphafold",
            "$HOME/repos/alphafold",
            "$HOME/GitHub/alphafold",
            "$HOME/alphafold",
        ]
        for c in candidates:
            rc, out, _ = wsl(f"test -d {c} && echo {c}")
            if rc == 0 and out:
                s["af3_dir"] = out
                self.log.emit(f"✅ af3_dir: {out}")
                break

        if "af3_dir" not in s:
            rc, out, _ = wsl(r"find $HOME -maxdepth 4 -type d -name alphafold -print -quit")
            if rc == 0 and out:
                s["af3_dir"] = out
                self.log.emit(f"✅ af3_dir (found): {out}")

        # msa.db: prefer <af3_dir>/mmseqs_db
        af3 = s.get("af3_dir")
        if af3:
            rc, out, _ = wsl(f"test -d '{af3}/mmseqs_db' && echo '{af3}/mmseqs_db'")
            if rc == 0 and out:
                s.setdefault("msa", {})
                s["msa"]["db"] = out
                self.log.emit(f"✅ msa.db: {out}")

        # rosetta_relax_bin: leave blank unless we confidently find it
        # (Better to prompt user than guess wrong.)
        return s

    # -------------------------
    # Linux-only detection
    # -------------------------
    def _detect_on_linux(self) -> dict[str, Any]:
        s: dict[str, Any] = {}
        self.log.emit("🔎 Detecting on Linux…")

        s["platform"] = "linux"
        s["linux_home_root"] = str(Path.home())
        try:
            s["gui_dir"] = str(Path(__file__).resolve().parent)
        except Exception:
            pass
        # You can add Linux AF3/mmseqs detection later
        return s


class AnalysisWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, job_name: str, multi_seed: bool = False):
        super().__init__()
        self.job_name = job_name
        self.multi_seed = multi_seed

    def run(self):
        try:
            _ensure_backend_loaded()
            # main_window.py lives at: repo/apps/gui/main_window.py
            # -> parents[2] == repo/
            repo_root = Path(__file__).resolve().parents[2]

            # Use the same interpreter as the GUI (conda env, etc.)
            exe = sys.executable

            cmd = [exe, "-m", "af3_pipeline.analysis.post_analysis", "--job", str(self.job_name)]
            if self.multi_seed:
                cmd.append("--multi_seed")

            # Ensure the subprocess can import af3_pipeline regardless of cwd
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["AF3_PIPELINE_CONFIG"] = os.environ.get("AF3_PIPELINE_CONFIG", "")

            self.log.emit(f"▶ Post-AF3 analysis\n$ {' '.join(map(str, cmd))}")

            with subprocess.Popen(
                cmd,
                cwd=str(repo_root),          # ⭐ critical
                env=env,                     # ⭐ critical
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            ) as p:
                assert p.stdout is not None
                for line in p.stdout:
                    self.log.emit(line.rstrip("\n"))
                rc = p.wait()
                if rc != 0:
                    raise RuntimeError(f"post_analysis returned {rc}")

            self.done.emit()

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n{tb}")


# ==========================
# 🧬 Dynamic entry widgets (Alphafold page)
# ==========================
@dataclass
class MacroEntryRefs:
    root: QWidget
    dropdown: QComboBox
    name: QLineEdit
    seq: QPlainTextEdit

    # NEW PTM UI + state
    ptm_button: QPushButton
    ptm_summary: QLabel
    ptms: list[dict[str, str]] = field(default_factory=list)  # [{"ccd":"SEP","label":"Phosphoserine (pSer)","pos":"12"}, ...]

    template: Optional[QLineEdit] = None  # proteins only
    delete_btn: Optional[QPushButton] = None



def _build_entry_widget(
    parent: QWidget,
    *,
    kind: str,
    saved_names: list[str],
    ptm_choices: dict[str, Optional[str]],
    on_delete,
    on_select,
) -> MacroEntryRefs:
    frame = QFrame(parent)
    frame.setFrameShape(QFrame.Shape.StyledPanel)

    outer = QVBoxLayout(frame)
    outer.setContentsMargins(6, 6, 6, 6)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)

    dd = QComboBox(frame)
    dd.addItem("Select saved…")
    dd.addItems(saved_names)

    del_btn = QPushButton("🗑", frame)

    # Hidden name field (kept for compatibility with run caching/loader)
    name = QLineEdit(frame)
    name.hide()
    name.setFixedSize(0, 0)
    name.setObjectName("hiddenNameField")

    row.addWidget(dd, 3)

    template = None
    if kind == "protein":
        template = QLineEdit(frame)
        template.setPlaceholderText("PDB template (optional)")
        row.addWidget(template, 3)

    # --- NEW PTM button + summary ---
    ptm_btn = QPushButton("PTMs…", frame)
    ptm_summary = QLabel("None", frame)
    ptm_summary.setMinimumWidth(160)
    ptm_summary.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

    row.addWidget(ptm_btn, 0)
    row.addWidget(ptm_summary, 2)

    row.addWidget(del_btn, 0)

    outer.addLayout(row)

    seq = QPlainTextEdit(frame)
    seq.hide()

    refs = MacroEntryRefs(
        root=frame,
        dropdown=dd,
        name=name,
        seq=seq,
        ptm_button=ptm_btn,
        ptm_summary=ptm_summary,
        ptms=[],
        template=template,
        delete_btn=del_btn,
    )

    del_btn.clicked.connect(lambda: on_delete(refs))
    dd.currentTextChanged.connect(lambda txt: on_select(refs, txt))

    return refs


def _norm_seq_and_index_map(text: str) -> tuple[str, list[int]]:
    """
    Normalize sequence to only letters (A-Z, a-z). Return:
      - normalized uppercase sequence
      - norm_to_doc: list of document indices (into the ORIGINAL text) for each residue
        such that normalized[i] corresponds to original_text[norm_to_doc[i]].
    """
    norm_chars: list[str] = []
    norm_to_doc: list[int] = []
    for i, ch in enumerate(text or ""):
        if ch.isalpha():
            norm_chars.append(ch.upper())
            norm_to_doc.append(i)
    return "".join(norm_chars), norm_to_doc

def _find_doc_range_for_norm_span(norm_to_doc: list[int], start: int, length: int) -> tuple[int, int]:
    """
    Convert (start,length) in normalized coordinates into (doc_start, doc_end_exclusive)
    in the original editor text.
    """
    if not norm_to_doc or length <= 0:
        return (0, 0)
    start = max(0, min(start, len(norm_to_doc) - 1))
    end_norm = max(start, min(start + length - 1, len(norm_to_doc) - 1))
    doc_start = norm_to_doc[start]
    doc_end = norm_to_doc[end_norm] + 1
    # Expand doc_end across any immediately following whitespace/newlines? Not necessary.
    return (doc_start, doc_end)

def _all_occurrences(haystack: str, needle: str) -> list[int]:
    """Return all start indices of needle in haystack (allow overlaps)."""
    if not needle:
        return []
    out: list[int] = []
    i = 0
    while True:
        j = haystack.find(needle, i)
        if j < 0:
            break
        out.append(j)
        i = j + 1  # allow overlaps
    return out


class SequenceMapCanvas(QWidget):
    """
    A lightweight ruler+highlight canvas. Parent panel provides:
      - sequence length
      - hits and active hit
      - selection region
      - scroll position + zoom
    """
    requestSeek = pyqtSignal(int)                 # normalized index to center view on
    requestSelect = pyqtSignal(int, int)          # (start, length) normalized selection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setMouseTracking(True)

        self._seq_len = 0
        self._px_per_res = 6.0        # zoom
        self._offset_res = 0          # leftmost residue index
        self._hits: list[tuple[int, int]] = []       # list of (start, length)
        self._active_hit_idx: int = -1
        self._sel: tuple[int, int] | None = None     # (start, length)

        # drag selection state
        self._dragging = False
        self._drag_anchor_res: int | None = None

    # ----- model setters -----
    def setSequenceLength(self, n: int) -> None:
        self._seq_len = max(0, int(n))
        self.update()

    def setZoom(self, px_per_res: float) -> None:
        self._px_per_res = float(max(1.5, min(px_per_res, 60.0)))
        self.update()

    def setOffset(self, offset_res: int) -> None:
        self._offset_res = max(0, min(int(offset_res), max(0, self._seq_len - 1)))
        self.update()

    def setHits(self, hits: list[tuple[int, int]], active_idx: int = -1) -> None:
        self._hits = hits[:] if hits else []
        self._active_hit_idx = int(active_idx)
        self.update()

    def setSelection(self, sel: tuple[int, int] | None) -> None:
        self._sel = sel
        self.update()

    # ----- geometry helpers -----
    def _res_at_x(self, x: int) -> int:
        # map x px -> residue index
        res = int(self._offset_res + (x / self._px_per_res))
        return max(0, min(res, max(0, self._seq_len - 1)))

    def _x_at_res(self, res: int) -> float:
        return (res - self._offset_res) * self._px_per_res

    def visible_res_count(self) -> int:
        return int(self.width() / self._px_per_res) + 1

    # ----- painting -----
    def paintEvent(self, _evt):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()

        # Background
        p.fillRect(QRect(0, 0, w, h), self.palette().base())

        # If no sequence, draw hint
        if self._seq_len <= 0:
            p.setPen(self.palette().text().color())
            p.drawText(QRect(0, 0, w, h), int(Qt.AlignmentFlag.AlignCenter), "No sequence")
            return

        fm = QFontMetrics(p.font())
        top = 10
        baseline_y = h // 2
        tick_top = baseline_y - 12
        tick_bot = baseline_y + 12

        # Baseline
        pen_base = QPen(self.palette().text().color())
        pen_base.setWidth(1)
        p.setPen(pen_base)
        p.drawLine(0, baseline_y, w, baseline_y)

        # Highlight hits
        for idx, (s, ln) in enumerate(self._hits):
            if ln <= 0:
                continue
            x1 = self._x_at_res(s)
            x2 = self._x_at_res(s + ln)
            if x2 < 0 or x1 > w:
                continue
            rect = QRectF(max(0.0, x1), baseline_y - 16, min(float(w), x2) - max(0.0, x1), 32)
            if idx == self._active_hit_idx:
                # stronger fill
                p.fillRect(rect, self.palette().highlight())
            else:
                # lighter fill
                c = self.palette().highlight().color()
                c.setAlpha(90)
                p.fillRect(rect, c)

        # Highlight selection (on top)
        if self._sel:
            s, ln = self._sel
            if ln > 0:
                x1 = self._x_at_res(s)
                x2 = self._x_at_res(s + ln)
                if not (x2 < 0 or x1 > w):
                    c = self.palette().alternateBase().color()
                    c.setAlpha(160)
                    p.fillRect(QRectF(max(0.0, x1), baseline_y - 18, min(float(w), x2) - max(0.0, x1), 36), c)

        # Ticks + labels
        # Major tick every 10, label every 50
        first_res = self._offset_res
        last_res = min(self._seq_len - 1, self._offset_res + self.visible_res_count())

        # Start from nearest 10 below first_res (1-based display makes labels nicer)
        start_tick = (first_res // 10) * 10
        for r in range(start_tick, last_res + 1, 10):
            x = int(self._x_at_res(r))
            if x < 0:
                continue
            # Major tick
            p.drawLine(x, tick_top, x, tick_bot)

            # label every 50 residues
            if r % 50 == 0 and r != 0:
                label = str(r)
                tw = fm.horizontalAdvance(label)
                p.drawText(x - tw // 2, top + fm.ascent(), label)

        # End label (sequence length)
        end_label = f"Len: {self._seq_len}"
        p.drawText(8, h - 8, end_label)

    # ----- interaction -----
    def wheelEvent(self, e: QWheelEvent):  # noqa: N802
        # Ctrl+wheel zoom (common convention)
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = e.angleDelta().y()
            if delta > 0:
                self.setZoom(self._px_per_res * 1.15)
            elif delta < 0:
                self.setZoom(self._px_per_res / 1.15)
            e.accept()
            return
        super().wheelEvent(e)

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self._seq_len > 0:
            self._dragging = True
            self._drag_anchor_res = self._res_at_x(e.position().x())
            # single click → caret/selection of length 1
            self.requestSelect.emit(self._drag_anchor_res, 1)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._dragging and self._drag_anchor_res is not None:
            cur = self._res_at_x(e.position().x())
            a = self._drag_anchor_res
            start = min(a, cur)
            end = max(a, cur)
            self.requestSelect.emit(start, end - start + 1)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_anchor_res = None
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            res = self._res_at_x(e.position().x())
            self.requestSeek.emit(res)
            e.accept()
            return
        super().mouseDoubleClickEvent(e)


class SequenceMapPanel(QWidget):
    """
    Composite widget:
      - map canvas
      - horizontal scrollbar
      - search bar + next/prev + hit count
    Wires to a QPlainTextEdit.
    """
    def __init__(self, editor: QPlainTextEdit, parent=None, *, title: str = ""):
        super().__init__(parent)
        self._editor = editor
        self._title = title

        self._seq = ""
        self._norm_to_doc: list[int] = []

        self._hits: list[tuple[int, int]] = []
        self._active_hit = -1

        self._offset_res = 0
        self._px_per_res = 6.0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 0)
        outer.setSpacing(6)

        # Optional mini header row
        head = QHBoxLayout()
        if title:
            lab = QLabel(title, self)

            font = lab.font()
            font.setFamily("Arial")   
            font.setPointSize(10)                     
            font.setWeight(QFont.Weight.Normal)    
            lab.setFont(font)

            head.addWidget(lab)
        head.addStretch(1)

        self.zoomOutBtn = QPushButton("−", self)
        self.zoomOutBtn.setFixedWidth(28)
        self.zoomInBtn = QPushButton("+", self)
        self.zoomInBtn.setFixedWidth(28)
        self.zoomHint = QLabel("Ctrl+Wheel to zoom", self)
        self.zoomHint.setStyleSheet("color: gray;")
        head.addWidget(self.zoomHint)
        head.addWidget(self.zoomOutBtn)
        head.addWidget(self.zoomInBtn)
        outer.addLayout(head)

        self.canvas = SequenceMapCanvas(self)
        outer.addWidget(self.canvas)

        self.hscroll = QScrollBar(Qt.Orientation.Horizontal, self)
        outer.addWidget(self.hscroll)

        # Search row
        row = QHBoxLayout()
        self.searchEdit = QLineEdit(self)
        self.searchEdit.setPlaceholderText("Search motif (e.g. CXXC, KEN, NLS…) — plain text (no regex)")
        self.searchBtn = QPushButton("Search", self)
        self.findPrevBtn = QPushButton("◀ Prev", self)
        self.findNextBtn = QPushButton("Next ▶", self)
        self.clearBtn = QPushButton("Clear", self)
        self.hitsLabel = QLabel("", self)
        self.hitsLabel.setMinimumWidth(90)

        row.addWidget(self.searchEdit, 6)
        row.addWidget(self.searchBtn, 0)
        row.addWidget(self.findPrevBtn, 0)
        row.addWidget(self.findNextBtn, 0)
        row.addWidget(self.clearBtn, 0)
        row.addWidget(self.hitsLabel, 0)
        outer.addLayout(row)

        # --- wiring ---
        self.zoomInBtn.clicked.connect(lambda: self._set_zoom(self._px_per_res * 1.15))
        self.zoomOutBtn.clicked.connect(lambda: self._set_zoom(self._px_per_res / 1.15))

        self.hscroll.valueChanged.connect(self._on_scroll_changed)
        self.canvas.requestSeek.connect(self.center_on_residue)
        self.canvas.requestSelect.connect(self.select_norm_span)

        self.searchBtn.clicked.connect(self.run_search)
        self.searchEdit.returnPressed.connect(self.run_search)
        self.findNextBtn.clicked.connect(lambda: self._step_hit(+1))
        self.findPrevBtn.clicked.connect(lambda: self._step_hit(-1))
        self.clearBtn.clicked.connect(self._clear_search)

        # Debounce editor updates & search typing
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._recompute_from_editor)

        self._editor.textChanged.connect(self._debounced_refresh)

        # Selection sync: reflect editor selection on the map
        try:
            self._editor.selectionChanged.connect(self._sync_selection_from_editor)  # type: ignore[attr-defined]
        except Exception:
            # Not fatal if not available
            pass

        self._recompute_from_editor()

    def run_search(self):
        """
        Run search only when the user explicitly requests it.
        Important: do NOT move focus away from the search box.
        """
        self._recompute_hits()

        # If we have hits, select the first hit in the editor BUT keep typing focus in search box.
        if self._active_hit >= 0 and self._hits:
            s, ln = self._hits[self._active_hit]

            # perform selection without stealing focus
            doc_start, doc_end = _find_doc_range_for_norm_span(self._norm_to_doc, s, ln)
            cursor = self._editor.textCursor()
            cursor.setPosition(doc_start)
            cursor.setPosition(doc_end, cursor.MoveMode.KeepAnchor)
            self._editor.setTextCursor(cursor)

            # update map selection + view (map may scroll, but focus stays)
            self.canvas.setSelection((s, ln))
            self.center_on_residue(s)

        # return focus to search box explicitly
        self.searchEdit.setFocus()


    # ----- public helpers -----
    def _debounced_refresh(self):
        self._debounce.start()

    def _set_zoom(self, z: float):
        self._px_per_res = float(max(1.5, min(z, 60.0)))
        self._update_scrollbar()
        self._push_view()

    def center_on_residue(self, res: int):
        if self._seq_len() <= 0:
            return
        visible = self.canvas.visible_res_count()
        new_off = int(res - visible // 2)
        new_off = max(0, min(new_off, max(0, self._seq_len() - visible)))
        self._offset_res = new_off
        self._update_scrollbar()
        self._push_view()

    def select_norm_span(self, start: int, length: int):
        if self._seq_len() <= 0 or length <= 0:
            return
        start = max(0, min(start, self._seq_len() - 1))
        length = max(1, min(length, self._seq_len() - start))

        # update editor selection
        doc_start, doc_end = _find_doc_range_for_norm_span(self._norm_to_doc, start, length)
        cursor = self._editor.textCursor()
        cursor.setPosition(doc_start)
        cursor.setPosition(doc_end, cursor.MoveMode.KeepAnchor)
        self._editor.setTextCursor(cursor)
        self._editor.setFocus()

        # update map selection
        self.canvas.setSelection((start, length))
        self.center_on_residue(start)

    # ----- internals -----
    def _seq_len(self) -> int:
        return len(self._seq)

    def _recompute_from_editor(self):
        txt = self._editor.toPlainText()
        self._seq, self._norm_to_doc = _norm_seq_and_index_map(txt)
        self._hits = []
        self._active_hit = -1
        self.canvas.setSequenceLength(self._seq_len())
        self._update_scrollbar()
        self._push_view()
        self._recompute_hits()

    def _update_scrollbar(self):
        n = self._seq_len()
        if n <= 0:
            self.hscroll.setRange(0, 0)
            self.hscroll.setPageStep(1)
            self.hscroll.setValue(0)
            return

        visible = max(1, self.canvas.visible_res_count())
        max_off = max(0, n - visible)
        self.hscroll.setRange(0, max_off)
        self.hscroll.setPageStep(visible)
        self._offset_res = max(0, min(self._offset_res, max_off))
        was = self.hscroll.blockSignals(True)
        try:
            self.hscroll.setValue(self._offset_res)
        finally:
            self.hscroll.blockSignals(was)

    def _push_view(self):
        self.canvas.setZoom(self._px_per_res)
        self.canvas.setOffset(self._offset_res)
        self.canvas.setHits(self._hits, self._active_hit)

    def _on_scroll_changed(self, v: int):
        self._offset_res = int(v)
        self._push_view()

    def _recompute_hits(self):
        motif = (self.searchEdit.text() or "").strip().upper()
        if not motif:
            self._hits = []
            self._active_hit = -1
            self.hitsLabel.setText("")
            self.canvas.setHits(self._hits, self._active_hit)
            return

        # plain-string search (user can type 1..N aa)
        starts = _all_occurrences(self._seq, motif)
        self._hits = [(s, len(motif)) for s in starts]
        self._active_hit = 0 if self._hits else -1
        self.hitsLabel.setText(f"{len(self._hits)} hit(s)" if self._hits else "0 hit(s)")
        self.canvas.setHits(self._hits, self._active_hit)

    def _step_hit(self, direction: int):
        if not self._hits:
            return
        self._active_hit = (self._active_hit + direction) % len(self._hits)
        self.canvas.setHits(self._hits, self._active_hit)
        s, ln = self._hits[self._active_hit]
        self.select_norm_span(s, ln)

    def _clear_search(self):
        self.searchEdit.clear()
        self._hits = []
        self._active_hit = -1
        self.hitsLabel.setText("")
        self.canvas.setHits(self._hits, self._active_hit)

    def _sync_selection_from_editor(self):
        # Try to reflect editor selection on the map (best-effort)
        if not self._norm_to_doc:
            self.canvas.setSelection(None)
            return
        cur = self._editor.textCursor()
        a = min(cur.selectionStart(), cur.selectionEnd())
        b = max(cur.selectionStart(), cur.selectionEnd())
        if a == b:
            # caret position → select 1 residue (nearest)
            # Find the nearest normalized residue whose doc index <= a
            # (binary search would be nicer; linear is ok for typical lengths)
            idx = 0
            for i, doc_i in enumerate(self._norm_to_doc):
                if doc_i <= a:
                    idx = i
                else:
                    break
            self.canvas.setSelection((idx, 1))
            return

        # Selection range: take residues with doc index in [a,b)
        start_norm = None
        end_norm = None
        for i, doc_i in enumerate(self._norm_to_doc):
            if start_norm is None and doc_i >= a:
                start_norm = i
            if doc_i < b:
                end_norm = i
            else:
                break

        if start_norm is None or end_norm is None or end_norm < start_norm:
            self.canvas.setSelection(None)
            return

        self.canvas.setSelection((start_norm, end_norm - start_norm + 1))

# ==========================
# 🪟 Main Window
# ==========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- dynamic entries ----
        self.protein_entries = []
        self.dna_entries = []
        self.rna_entries = []

        # Load UI first
        ui_path = resource_path("main_window.ui")
        uic.loadUi(str(ui_path), self)
        self.resize(1500, 1200)
        self.setMinimumSize(900, 600)

        # Profiles UI
        self._install_profiles_row()
        self._install_sequence_map_panels()

        # ✅ Must activate a profile BEFORE any cache/config IO
        self._ensure_profile_first_run()

        # ✅ Now that profile is active, load caches from correct per-profile paths
        self.sequence_cache = _read_json(SEQUENCE_CACHE_FILE, {})
        self.ligand_cache   = _read_json(LIGAND_CACHE_FILE, {})
        self.runs_history   = _read_json(RUNS_HISTORY_FILE, [])

        # ---- state ----
        self.custom_protein_atom: Optional[str] = None
        self.is_running = False
        self.build_thread: Optional[BuildThread] = None
        self.run_thread: Optional[RunThread] = None
        self.analysis_thread: Optional[AnalysisWorker] = None
        self.current_jobname: str = ""

        # ---- wire up ----
        self._connect_nav()
        self._connect_sequences_page()
        self._connect_ligands_page()
        self._connect_alphafold_page()
        self._connect_runs_page()
        self._connect_config_page()

        self._current_ligand_notes_name: str = ""
        self._install_ligand_notes_box()

        # ---- initial populate ----
        self._refresh_all_dropdowns()
        self._refresh_queue_view()
        self._refresh_runs_view()

        # covalent field enable
        self.covalentCheckbox.toggled.connect(self._update_covalent_fields)
        self._update_covalent_fields()
        self.proteinAtomComboBox.currentTextChanged.connect(self._on_protein_atom_change)

        self._ensure_profile_first_run()

        # --- First run detection (only if required fields missing) ---

    def _active_profile_name(self) -> str:
        return _safe_profile_name(getattr(self, "_active_profile", "") or "")

    def _rename_active_profile(self):
        old = self._active_profile_name()
        if not old:
            _msg_warn(self, "No active profile", "Select a profile first.")
            return

        new_raw, ok = QInputDialog.getText(self, "Rename Profile", "New profile name:", text=old)
        if not ok:
            return

        new = _safe_profile_name(new_raw)
        if not new:
            _msg_warn(self, "Invalid name", "Please enter a valid profile name.")
            return
        if new == old:
            return

        profiles = _list_profiles()
        if new in profiles:
            _msg_warn(self, "Already exists", f"A profile named '{new}' already exists.")
            return

        src = _profile_root(old)
        dst = _profile_root(new)
        try:
            if not src.exists():
                _msg_warn(self, "Missing profile", f"Profile folder not found: {src}")
                return
            src.rename(dst)
        except Exception as e:
            _msg_warn(self, "Rename failed", str(e))
            return

        _save_current_profile_name(new)
        self._refresh_profiles_dropdown()
        self.profileDropdown.setCurrentText(new)
        self._activate_profile(new, run_autodetect_if_needed=False)
        self.log(f"✅ Renamed profile '{old}' → '{new}'")

    def _delete_active_profile(self):
        name = self._active_profile_name()
        if not name:
            _msg_warn(self, "No active profile", "Select a profile first.")
            return

        # Optional safety: block deletion while jobs running, if you track that state
        # if getattr(self, "_job_running", False):
        #     _msg_warn(self, "Busy", "Can't delete profile while a job is running.")
        #     return

        reply = QMessageBox.question(
            self,
            "Delete Profile",
            f"Delete profile '{name}'?\n\nThis will remove:\n{_profile_root(name)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        root = _profile_root(name)
        try:
            import shutil
            if root.exists():
                shutil.rmtree(root)
        except Exception as e:
            _msg_warn(self, "Delete failed", str(e))
            return

        self.log(f"🗑️ Deleted profile '{name}'")

        # Choose another profile or force-create a new one
        remaining = _list_profiles()
        if remaining:
            new_active = remaining[0]
            _save_current_profile_name(new_active)
            self._refresh_profiles_dropdown()
            self.profileDropdown.setCurrentText(new_active)
            self._activate_profile(new_active, run_autodetect_if_needed=False)
        else:
            _save_current_profile_name("")
            self._refresh_profiles_dropdown()
            self._ensure_profile_first_run()

    # =======================
    # 👤 Profiles UI + switching
    # =======================
    def _install_profiles_row(self):
        if hasattr(self, "_profilesRow"):
            return

        self._profilesRow = QWidget(self)
        row = QHBoxLayout(self._profilesRow)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.profileDropdown = QComboBox(self._profilesRow)
        self.profileDropdown.setMinimumWidth(170)

        self.newProfileButton = QPushButton("➕", self._profilesRow)
        self.newProfileButton.setFixedWidth(42)
        self.newProfileButton.setToolTip("Create new profile")

        # ✅ NEW: options menu button
        self.profileOptionsButton = QToolButton(self._profilesRow)
        self.profileOptionsButton.setText("⋯")
        self.profileOptionsButton.setFixedWidth(42)
        self.profileOptionsButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        menu = QMenu(self.profileOptionsButton)
        self.actNewProfile = menu.addAction("New profile…")
        self.actRenameProfile = menu.addAction("Rename profile…")
        self.actDeleteProfile = menu.addAction("Delete profile…")
        self.profileOptionsButton.setMenu(menu)

        row.addWidget(self.profileDropdown, 1)
        row.addWidget(self.newProfileButton, 0)
        row.addWidget(self.profileOptionsButton, 0)

        # Insert above navList
        try:
            self.sidebarLayout.insertWidget(1, self._profilesRow)
        except Exception as e:
            self.log(f"⚠️ Failed to insert profile row: {e}")

        self.newProfileButton.clicked.connect(self._create_new_profile)
        self.actNewProfile.triggered.connect(lambda: self._create_new_profile(force=False))
        self.actRenameProfile.triggered.connect(self._rename_active_profile)
        self.actDeleteProfile.triggered.connect(self._delete_active_profile)

        self.profileDropdown.currentTextChanged.connect(self._on_profile_selected)

        self._refresh_profiles_dropdown()

    def _set_profile_dropdown_silent(self, name: str) -> None:
        was = self.profileDropdown.blockSignals(True)
        try:
            self.profileDropdown.setCurrentText(name)
        finally:
            self.profileDropdown.blockSignals(was)

    def _refresh_profiles_dropdown(self):
        profiles = _list_profiles()
        cur = _load_current_profile_name()

        self.profileDropdown.blockSignals(True)
        try:
            self.profileDropdown.clear()
            self.profileDropdown.addItem("Select profile…")
            self.profileDropdown.addItems(profiles)

            if cur and cur in profiles:
                self.profileDropdown.setCurrentText(cur)
            else:
                self.profileDropdown.setCurrentIndex(0)
        finally:
            self.profileDropdown.blockSignals(False)

    def _ensure_profile_first_run(self):
        """
        On first startup: force profile creation before anything else.
        """
        cur = _load_current_profile_name()
        if cur and cur in _list_profiles():
            self._activate_profile(cur, run_autodetect_if_needed=False)
            return

        # No profile exists / not set: force create
        self._create_new_profile(force=True)

    def _create_new_profile(self, force: bool = False):
        dlg = NewProfileDialog(self)
        if force:
            # Make it harder to skip on first run
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)

        if not dlg.exec():
            if force:
                _msg_warn(self, "Profile required", "You must create a profile to continue.")
                return self._create_new_profile(force=True)
            return

        raw = dlg.profile_name()
        name = _safe_profile_name(raw)
        if not name:
            _msg_warn(self, "Invalid name", "Please enter a profile name.")
            return self._create_new_profile(force=force)

        root = _profile_root(name)
        if root.exists() and any(root.iterdir()):
            # Existing profile
            pass
        else:
            root.mkdir(parents=True, exist_ok=True)

        _save_current_profile_name(name)
        self._refresh_profiles_dropdown()
        self._set_profile_dropdown_silent(name)
        self._activate_profile(name, run_autodetect_if_needed=True)

    def _on_profile_selected(self, name: str):
        name = (name or "").strip()
        if not name or name == "Select profile…":
            return
        # Avoid reloading same profile repeatedly
        if getattr(self, "_active_profile", "") == name:
            return
        self._activate_profile(name, run_autodetect_if_needed=False)

    def _activate_profile(self, name: str, *, run_autodetect_if_needed: bool):
        name = _safe_profile_name(name)
        if not name:
            return

        if getattr(self, "_activating_profile", False):
            return
        if getattr(self, "_active_profile", "") == name and hasattr(self, "_profile_root"):
            # already active and initialized
            if not run_autodetect_if_needed:
                return

        self._activating_profile = True
        self._active_profile = name
        _save_current_profile_name(name)
        try:
            # 1) repoint cache locations (globals)
            root = _set_active_profile_paths(name)
            self._profile_root = root
            self.log(f"👤 Active profile: {name}")
            self.log(f"📁 Profile root: {root}")

            # 2) Per-profile env vars
            profile_cfg = root / "config.yaml"
            os.environ["AF3_PIPELINE_CONFIG"] = str(profile_cfg)

            # ✅ Most important missing piece:
            os.environ["AF3_PIPELINE_CACHE_ROOT"] = str(root / "cache")

            # Optional shared caches (fine)
            os.environ["AF3_PIPELINE_MSA_CACHE"] = str(_shared_dir("msa"))
            os.environ["AF3_PIPELINE_TEMPLATE_CACHE"] = str(_shared_dir("templates"))

            # 3) Now load backend (after env set)
            _ensure_backend_loaded()

            # 4) Ensure config exists for this profile
            self._bootstrap_config_yaml_from_template()

            # 4) Reload cfg from this profile config
            try:
                cfg.reload(profile_cfg)
            except Exception as e:
                self.log(f"⚠️ cfg.reload failed: {e}")

            # 5) Reload caches from the *profile* paths
            self.sequence_cache = _read_json(SEQUENCE_CACHE_FILE, {})
            self.ligand_cache   = _read_json(LIGAND_CACHE_FILE, {})
            self.runs_history   = _read_json(RUNS_HISTORY_FILE, [])

            # 6) Refresh UI
            self._refresh_all_dropdowns()
            self._refresh_queue_view()
            self._refresh_runs_view()

            # Also reset ligand notes box (if installed)
            if hasattr(self, "ligandNotesEdit"):
                self._load_ligand_notes_for_name(self.ligandDropdown.currentText().strip())

            # 7) Run autodetect if needed
            if run_autodetect_if_needed or self._config_needs_setup():
                self.pagesStack.setCurrentIndex(0)  # CONFIG page index in your .ui (INTRO=0..CONFIG=5)
                self.navList.setCurrentRow(0)
                self.log("🧭 Running per-profile auto-detect…")
                self._start_autodetect()
        finally:
            self._activating_profile = False

    # =======================
    # 📝 Ligand Notes (cached with ligand)
    # =======================
    def _install_ligand_notes_box(self):
        """
        Add a per-ligand notes editor underneath the 2D preview on the Ligands page.
        No .ui changes required.
        Saves notes to: LIGAND_CACHE/<hash>/NOTES.txt
        """
        # Guard: only install once
        if hasattr(self, "ligandNotesEdit"):
            return

        # Create label + editor
        self.ligandNotesLabel = QLabel("Notes", self)
        self.ligandNotesEdit = QPlainTextEdit(self)
        self.ligandNotesEdit.setPlaceholderText("Notes for this ligand… (autosaved)")
        self.ligandNotesEdit.setMinimumHeight(140)

        # Insert under preview label within existing ligandPreviewLayout
        try:
            # Your UI has: self.ligandPreviewLayout inside ligandPreviewGroup
            # Add after the preview image label
            self.ligandPreviewLayout.addWidget(self.ligandNotesLabel)
            self.ligandPreviewLayout.addWidget(self.ligandNotesEdit)
        except Exception as e:
            self.log(f"⚠️ Failed to add ligand notes box to layout: {e}")
            return

        # Debounced autosave
        self._ligand_notes_save_timer = QTimer(self)
        self._ligand_notes_save_timer.setSingleShot(True)
        self._ligand_notes_save_timer.setInterval(350)
        self._ligand_notes_save_timer.timeout.connect(self._save_current_ligand_notes_silent)

        self.ligandNotesEdit.textChanged.connect(lambda: self._ligand_notes_save_timer.start())

        # Disabled until a ligand is selected
        self.ligandNotesEdit.setEnabled(False)
        self.ligandNotesLabel.setEnabled(False)

    def _ligand_hash_dir_from_entry(self, entry: dict) -> Optional[Path]:
        """
        Return LIGAND_CACHE/<hash> for a ligand cache entry.
        """
        try:
            lig_hash = entry.get("hash")
            if not lig_hash:
                smiles = entry.get("smiles", "") or ""
                lig_hash = cache_utils.compute_hash(smiles)
            lig_dir = LIGAND_CACHE / str(lig_hash)
            lig_dir.mkdir(parents=True, exist_ok=True)
            return lig_dir
        except Exception:
            return None

    def _ligand_notes_path_from_entry(self, entry: dict) -> Optional[Path]:
        lig_dir = self._ligand_hash_dir_from_entry(entry)
        if not lig_dir:
            return None
        return lig_dir / "NOTES.txt"

    def _load_ligand_notes_for_name(self, ligand_name: str):
        """
        Load notes for ligand_name into the notes widget.
        """
        if not hasattr(self, "ligandNotesEdit"):
            return

        ligand_name = (ligand_name or "").strip()
        entry = self.ligand_cache.get(ligand_name)
        if not isinstance(entry, dict):
            # No ligand selected / bad entry
            self._current_ligand_notes_name = ""
            self.ligandNotesEdit.blockSignals(True)
            try:
                self.ligandNotesEdit.setPlainText("")
            finally:
                self.ligandNotesEdit.blockSignals(False)
            self.ligandNotesEdit.setEnabled(False)
            self.ligandNotesLabel.setEnabled(False)
            return

        notes_path = self._ligand_notes_path_from_entry(entry)
        txt = ""
        if notes_path and notes_path.exists():
            try:
                txt = notes_path.read_text(encoding="utf-8")
            except Exception:
                txt = ""

        self._current_ligand_notes_name = ligand_name

        self.ligandNotesEdit.blockSignals(True)
        try:
            self.ligandNotesEdit.setPlainText(txt)
        finally:
            self.ligandNotesEdit.blockSignals(False)

        self.ligandNotesEdit.setEnabled(True)
        self.ligandNotesLabel.setEnabled(True)

    def _save_current_ligand_notes_silent(self):
        """
        Save current notes text to the currently-selected ligand's cache folder.
        Silent (no popups). Safe to call frequently.
        """
        if not hasattr(self, "ligandNotesEdit"):
            return

        name = (self._current_ligand_notes_name or "").strip()
        if not name:
            return

        entry = self.ligand_cache.get(name)
        if not isinstance(entry, dict):
            return

        notes_path = self._ligand_notes_path_from_entry(entry)
        if not notes_path:
            return

        try:
            notes_path.parent.mkdir(parents=True, exist_ok=True)
            notes_path.write_text(self.ligandNotesEdit.toPlainText(), encoding="utf-8")
        except Exception as e:
            # Log once in a while; keep it quiet to avoid annoying the user
            self.log(f"⚠️ Failed to save ligand notes for '{name}': {e}")

    def _install_sequence_map_panels(self):
        """
        Create + insert map/search panels under each sequence editor on the Sequences page.
        No .ui changes needed.
        """
        # Protein
        try:
            self._proteinSeqMapPanel = SequenceMapPanel(self.proteinSeqEditor, self, title="Protein map")
            self.proteinSeqGroupLayout.addWidget(self._proteinSeqMapPanel)
        except Exception as e:
            self.log(f"⚠️ Failed to install protein sequence map: {e}")

        # DNA
        try:
            self._dnaSeqMapPanel = SequenceMapPanel(self.dnaSeqEditor, self, title="DNA map")
            self.dnaSeqGroupLayout.addWidget(self._dnaSeqMapPanel)
        except Exception as e:
            self.log(f"⚠️ Failed to install DNA sequence map: {e}")

        # RNA
        try:
            self._rnaSeqMapPanel = SequenceMapPanel(self.rnaSeqEditor, self, title="RNA map")
            self.rnaSeqGroupLayout.addWidget(self._rnaSeqMapPanel)
        except Exception as e:
            self.log(f"⚠️ Failed to install RNA sequence map: {e}")

    def _set_combo_text_force(self, combo: QComboBox, text: str) -> None:
        text = (text or "").strip()
        if not text:
            combo.setCurrentIndex(0)
            return

        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return

        was_blocked = combo.blockSignals(True)
        try:
            combo.addItem(text)
        finally:
            combo.blockSignals(was_blocked)

        idx2 = combo.findText(text)
        if idx2 >= 0:
            combo.setCurrentIndex(idx2)
        else:
            combo.setCurrentIndex(0)

    def _template_yaml_path(self) -> Path:
        return Path(__file__).resolve().parent / "config_template.yaml"

    def _bootstrap_config_yaml_from_template(self) -> bool:
        """
        If ~/.af3_pipeline/config.yaml does not exist, create it by reading
        config_template.yaml and filling any missing keys with defaults.
        Returns True if we created a new config.
        """
        out = self._config_yaml_path()
        if out.exists():
            return False

        tmpl = self._template_yaml_path()
        data: dict[str, Any] = {}

        user_home = Path.home()               
        linux_home_guess = f"/home/{user_home.name}"
        repo_guess = f"{linux_home_guess}/Repositories/alphafold"

        if tmpl.exists():
            try:
                loaded = yaml.safe_load(tmpl.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}

        # --- Fill/normalize required top-level fields ---
        def put_if_missing(k: str, v: Any):
            if k not in data or data.get(k) in (None, ""):
                data[k] = v

        put_if_missing("wsl_distro", "Ubuntu-22.04")
        put_if_missing("gui_dir", str(Path(__file__).resolve().parent))
        put_if_missing("linux_home_root", linux_home_guess)

        put_if_missing("af3_dir", repo_guess)
        put_if_missing("docker_bin", "docker")
        put_if_missing("alphafold_docker_image", "alphafold3")
        put_if_missing("alphafold_docker_env", {})

        # --- Nested msa defaults ---
        msa = data.get("msa")
        if not isinstance(msa, dict):
            msa = {}
            data["msa"] = msa
        msa.setdefault("threads", 10)
        msa.setdefault("sensitivity", 5.7)
        msa.setdefault("max_seqs", 25)
        msa.setdefault("db", f"{repo_guess}/mmseqs_db")

        # --- Nested ligand defaults ---
        lig = data.get("ligand")
        if not isinstance(lig, dict):
            lig = {}
            data["ligand"] = lig
        lig.setdefault("n_confs", 200)
        lig.setdefault("seed", 0)
        lig.setdefault("prune_rms", 0.25)
        lig.setdefault("keep_charge", False)
        lig.setdefault("require_assigned_stereo", False)
        lig.setdefault("basename", "LIGAND")
        lig.setdefault("name_default", "LIG")
        lig.setdefault("png_size", [1500, 1200])
        lig.setdefault("rdkit_threads", 0)

        # --- Rosetta default (blank is fine) ---
        data.setdefault("rosetta_relax_bin", "")

        # Normalize any POSIX-ish paths to forward slashes
        msa["db"] = self._posixify(str(msa.get("db", "")))
        if isinstance(data.get("af3_dir"), str):
            data["af3_dir"] = self._posixify(data["af3_dir"])
        if isinstance(data.get("linux_home_root"), str):
            data["linux_home_root"] = self._posixify(data["linux_home_root"])
        if isinstance(data.get("rosetta_relax_bin"), str):
            data["rosetta_relax_bin"] = self._posixify(data["rosetta_relax_bin"])

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.log(f"🧾 Bootstrapped config.yaml from template:\n{out}")
        return True

    def _ptm_summary_text(self, ptms: list[dict[str, str]]) -> str:
        if not ptms:
            return "None"
        # e.g. "SEP@12, PTR@98"
        parts = []
        for p in ptms:
            ccd = (p.get("ccd") or "").strip()
            pos = (p.get("pos") or "").strip()
            if ccd and pos:
                parts.append(f"{ccd}@{pos}")
        return ", ".join(parts) if parts else "None"

    def _edit_ptms_for_entry(self, refs: MacroEntryRefs):
        dlg = PtmDialog(self, PTM_CHOICES, initial=refs.ptms)
        if dlg.exec():
            refs.ptms = dlg.get_ptms()
            refs.ptm_summary.setText(self._ptm_summary_text(refs.ptms))
            self.log(f"🧬 PTMs updated: {refs.ptm_summary.text()}")



    # =======================
    # 🧭 Navigation
    # =======================
    def _connect_nav(self):
        self.navList.currentRowChanged.connect(self.pagesStack.setCurrentIndex)

    # =======================
    # 🔁 Cache helpers
    # =======================
    def _names_for(self, kind: str) -> list[str]:
        kind_l = kind.lower()
        out: list[str] = []
        for key, val in self.sequence_cache.items():
            if isinstance(val, dict):
                t = (val.get("type") or "").strip().lower()
                if t == kind_l:
                    out.append(key)
            else:
                # legacy: treat as protein
                if kind_l == "protein":
                    out.append(key)
        return sorted(set(out))

    def _get_seq_from_cache(self, key: str) -> str:
        if key not in self.sequence_cache:
            return ""
        val = self.sequence_cache[key]
        if isinstance(val, dict):
            return str(val.get("sequence", "") or "")
        return str(val)

    def _refresh_all_dropdowns(self):
        # Sequences page dropdowns
        prot = self._names_for("protein")
        dna = self._names_for("dna")
        rna = self._names_for("rna")

        self.proteinSeqDropdown.clear(); self.proteinSeqDropdown.addItem("Select saved…"); self.proteinSeqDropdown.addItems(prot)
        self.dnaSeqDropdown.clear(); self.dnaSeqDropdown.addItem("Select saved…"); self.dnaSeqDropdown.addItems(dna)
        self.rnaSeqDropdown.clear(); self.rnaSeqDropdown.addItem("Select saved…"); self.rnaSeqDropdown.addItems(rna)

        # Ligands page
        self.ligandDropdown.clear()
        self.ligandDropdown.addItem("Select saved…")
        self.ligandDropdown.addItems(sorted(self.ligand_cache.keys()))

        # Alphafold run ligand selector (separate!)
        self.runLigandDropdown.clear()
        self.runLigandDropdown.addItem("Select ligand…")
        self.runLigandDropdown.addItems(sorted(self.ligand_cache.keys()))

        # also refresh dynamic entry dropdown choices
        self._refresh_dynamic_entry_dropdowns()

    def _refresh_dynamic_entry_dropdowns(self):
        prot_names = self._names_for("protein")
        dna_names = self._names_for("dna")
        rna_names = self._names_for("rna")

        def refill(entry: MacroEntryRefs, names: list[str]):
            current = entry.dropdown.currentText()
            entry.dropdown.blockSignals(True)
            entry.dropdown.clear()
            entry.dropdown.addItem("Select saved…")
            entry.dropdown.addItems(names)
            # best-effort restore
            if current and current not in {"Select saved…", "Select saved..."}:
                idx = entry.dropdown.findText(current)
                if idx >= 0:
                    entry.dropdown.setCurrentIndex(idx)
            entry.dropdown.blockSignals(False)

        for e in self.protein_entries:
            refill(e, prot_names)
        for e in self.dna_entries:
            refill(e, dna_names)
        for e in self.rna_entries:
            refill(e, rna_names)

    def _start_autodetect(self):
        # Avoid double-running
        if getattr(self, "_autodetect_in_flight", False):
            return
        self._autodetect_in_flight = True
        if hasattr(self, "_detect_thread") and self._detect_thread and self._detect_thread.isRunning():
            return

        distro = self.cfgWslDistro.text().strip()

        self._detect_thread = DetectConfigWorker("wsl", distro)
        self._detect_thread.log.connect(self.log)
        self._detect_thread.done.connect(self._on_autodetect_done)
        self._detect_thread.start()

    def _save_config_yaml_silent(self):
        out = self._config_yaml_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self._ui_config_dict()
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.log(f"💾 Saved config.yaml (auto): {out}")


    def _on_autodetect_done(self, s: dict):
        self._autodetect_in_flight = False
        if not s:
            self.log("⚠️ Auto-detect returned no suggestions.")
            return

        # Top-level simple fields
        if "wsl_distro" in s:
            self.cfgWslDistro.setText(str(s["wsl_distro"]))
        if "gui_dir" in s:
            self.cfgGuiDir.setText(str(s["gui_dir"]))
        if "linux_home_root" in s:
            self.cfgLinuxHomeRoot.setText(str(s["linux_home_root"]))
        if "af3_dir" in s:
            self.cfgAf3Dir.setText(str(s["af3_dir"]))
        if "docker_bin" in s:
            self.cfgDockerBin.setText(str(s["docker_bin"]))
        if "alphafold_docker_image" in s:
            self.cfgDockerImage.setText(str(s["alphafold_docker_image"]))

        # Nested msa dict
        msa = s.get("msa")
        if isinstance(msa, dict):
            if "db" in msa:
                self.cfgMMseqsDB.setText(self._posixify(str(msa["db"])))

        self._save_config_yaml_silent()
        self._reset_config_fields_from_cfg()
        try:
            _ensure_backend_loaded()
            cfg.reload(self._config_yaml_path())
            self.log(f"🧪 cfg msa.db = {cfg.get('msa.db','')}")
        except Exception as e:
            self.log(f"⚠️ cfg reload after autodetect failed: {e}")

        self.log("✅ Auto-detect applied + config saved.")

    def _config_needs_setup(self) -> bool:
        data = self._load_config_yaml_best_effort()

        # msa.db is required for MSA generation
        msa = data.get("msa", {}) if isinstance(data.get("msa", {}), dict) else {}
        msa_db = (msa.get("db") or "").strip()

        required = [
            (data.get("wsl_distro") or "").strip(),
            (data.get("gui_dir") or "").strip(),
            (data.get("linux_home_root") or "").strip(),
            (data.get("af3_dir") or "").strip(),
            msa_db,
        ]
        return any(not x for x in required)



    # =======================
    # 🧾 Sequences page
    # =======================
    def _connect_sequences_page(self):
        # load on selection
        self.proteinSeqDropdown.currentTextChanged.connect(lambda name: self._load_sequence_into_editor("protein", name))
        self.dnaSeqDropdown.currentTextChanged.connect(lambda name: self._load_sequence_into_editor("dna", name))
        self.rnaSeqDropdown.currentTextChanged.connect(lambda name: self._load_sequence_into_editor("rna", name))

        # save
        self.proteinSeqSaveButton.clicked.connect(lambda: self._save_sequence_from_editor("protein"))
        self.dnaSeqSaveButton.clicked.connect(lambda: self._save_sequence_from_editor("dna"))
        self.rnaSeqSaveButton.clicked.connect(lambda: self._save_sequence_from_editor("rna"))

        # delete
        self.proteinSeqDeleteButton.clicked.connect(lambda: self._delete_selected_sequence("protein"))
        self.dnaSeqDeleteButton.clicked.connect(lambda: self._delete_selected_sequence("dna"))
        self.rnaSeqDeleteButton.clicked.connect(lambda: self._delete_selected_sequence("rna"))

    def _load_sequence_into_editor(self, kind: str, name: str):
        if name in {"Select saved…", "Select saved..."}:
            if kind == "protein":
                self.proteinSeqName.clear(); self.proteinSeqEditor.setPlainText("")
            elif kind == "dna":
                self.dnaSeqName.clear(); self.dnaSeqEditor.setPlainText("")
            else:
                self.rnaSeqName.clear(); self.rnaSeqEditor.setPlainText("")
            return

        seq = self._get_seq_from_cache(name)
        if kind == "protein":
            self.proteinSeqName.setText(name)
            self.proteinSeqEditor.setPlainText(seq)
        elif kind == "dna":
            self.dnaSeqName.setText(name)
            self.dnaSeqEditor.setPlainText(seq)
        else:
            self.rnaSeqName.setText(name)
            self.rnaSeqEditor.setPlainText(seq)

    def _save_sequence_from_editor(self, kind: str):
        if kind == "protein":
            name_w = self.proteinSeqName
            seq_w  = self.proteinSeqEditor
            dd     = self.proteinSeqDropdown
        elif kind == "dna":
            name_w = self.dnaSeqName
            seq_w  = self.dnaSeqEditor
            dd     = self.dnaSeqDropdown
        else:
            name_w = self.rnaSeqName
            seq_w  = self.rnaSeqEditor
            dd     = self.rnaSeqDropdown

        name = name_w.text().strip()
        seq  = seq_w.toPlainText().strip()

        if not name or not seq:
            _msg_warn(self, "Missing input", "Please provide both a name and a sequence.")
            return

        if name in self.sequence_cache:
            if not _msg_yesno(self, "Overwrite?", f"A saved entry named '{name}' already exists.\nOverwrite it?"):
                return

        self.sequence_cache[name] = {"sequence": seq, "type": kind.capitalize()}
        _write_json(SEQUENCE_CACHE_FILE, self.sequence_cache)

        # --- Refresh dropdowns WITHOUT triggering load callbacks ---
        # (refresh triggers currentTextChanged which repopulates the editors)
        was = dd.blockSignals(True)
        try:
            self._refresh_all_dropdowns()
            dd.setCurrentIndex(0)
        finally:
            dd.blockSignals(was)

        # --- Clear editors explicitly (now it will "stick") ---
        name_w.clear()
        seq_w.clear()  # QPlainTextEdit supports clear()

        # Optional: put dropdown back to "Select saved…" so it doesn't look like it's still "active"
        # dd.setCurrentIndex(0)

        self.log(f"💾 Saved {kind.upper()} sequence: {name}")


    def _delete_selected_sequence(self, kind: str):
        if kind == "protein":
            dd = self.proteinSeqDropdown
        elif kind == "dna":
            dd = self.dnaSeqDropdown
        else:
            dd = self.rnaSeqDropdown

        name = dd.currentText().strip()
        if not name or name in {"Select saved…", "Select saved..."}:
            return

        if not _msg_yesno(self, "Delete?", f"Delete saved {kind.upper()} sequence '{name}'?"):
            return

        if name in self.sequence_cache:
            del self.sequence_cache[name]
            _write_json(SEQUENCE_CACHE_FILE, self.sequence_cache)
            self._refresh_all_dropdowns()
            dd.setCurrentIndex(0)
            self.log(f"🗑 Deleted {kind.upper()} sequence: {name}")

    # =======================
    # 💊 Ligands page (view/save)
    # =======================
    def _connect_ligands_page(self):
        self._install_ligand_notes_box()
        self.ligandDropdown.currentTextChanged.connect(self._load_ligand_view)
        self.ligandSaveButton.clicked.connect(self._save_ligand_view)
        self.ligandDeleteButton.clicked.connect(self._delete_ligand_view)

        self.openChimeraXButton.clicked.connect(self._open_ligand_in_chimerax_view)
        self.openPyMOLButton.clicked.connect(self._open_ligand_in_pymol_view)

    def _save_ligand_view(self):
        smiles_raw = self.ligandSmiles.text().strip()
        if not smiles_raw:
            _msg_warn(self, "Missing SMILES", "Please enter a SMILES string before saving.")
            return

        smiles = _canonical_smiles(smiles_raw)
        default_name = ""
        try:
            y = self._load_config_yaml_best_effort()
            default_name = ((y.get("ligand", {}) or {}).get("name_default", "") if isinstance(y.get("ligand", {}), dict) else "")
        except Exception:
            pass
        name = (self.ligandName.text().strip() or default_name or smiles).strip()


        try:
            cif_path = Path(prepare_ligand_from_smiles(smiles, name=name, skip_if_cached=False)).expanduser()
            if not cif_path.exists():
                raise FileNotFoundError(f"Ligand CIF was not created: {cif_path}")

            lig_hash = cache_utils.compute_hash(smiles)

            self.ligand_cache[name] = {
                "smiles": smiles,
                "hash": lig_hash,
                "path": str(cif_path),
            }
            _write_json(LIGAND_CACHE_FILE, self.ligand_cache)
            self._refresh_all_dropdowns()

            # select and preview
            idx = self.ligandDropdown.findText(name)
            if idx >= 0:
                self.ligandDropdown.setCurrentIndex(idx)

            self.log(f"✅ Saved ligand '{name}' → {cif_path}")
            self._update_ligand_preview_from_cache(name)

        except Exception as e:
            _msg_err(self, "Error", f"Ligand generation failed:\n{e}")

    def _load_ligand_view(self, name: str):
        self._save_current_ligand_notes_silent()
        if name in {"Select saved…", "Select saved..."}:
            self.ligandName.clear()
            self.ligandSmiles.clear()
            self.ligandPreviewLabel.setText("(Preview image will appear here)")
            self.ligandPreviewLabel.setPixmap(QPixmap())
            self._load_ligand_notes_for_name("")
            return

        entry = self.ligand_cache.get(name)
        if not entry:
            return

        self.ligandName.setText(name)

        if isinstance(entry, str):
            self.ligandSmiles.setText(entry)
        elif isinstance(entry, dict):
            self.ligandSmiles.setText(entry.get("smiles", "") or "")

        self._update_ligand_preview_from_cache(name)
        self._load_ligand_notes_for_name(name)

    def _delete_ligand_view(self):
        name = self.ligandDropdown.currentText().strip()
        if not name or name in {"Select saved…", "Select saved..."}:
            return

        if not _msg_yesno(self, "Delete?", f"Delete saved ligand '{name}'?"):
            return

        if name in self.ligand_cache:
            del self.ligand_cache[name]
            _write_json(LIGAND_CACHE_FILE, self.ligand_cache)
            self._refresh_all_dropdowns()
            self.ligandDropdown.setCurrentIndex(0)
            self.log(f"🗑 Deleted ligand: {name}")

    def _ligand_entry_to_structure_path(self, entry: dict) -> Optional[Path]:
        """
        Best-effort locate a structure file to open: prefer PDB, else CIF.
        We follow your old cache convention: <LIGAND_CACHE>/<hash>/LIGAND.(pdb|cif)
        """
        try:
            lig_hash = entry.get("hash")
            if not lig_hash:
                smiles = entry.get("smiles", "")
                lig_hash = cache_utils.compute_hash(smiles)
            lig_dir = LIGAND_CACHE / lig_hash
            pdb_candidates = [lig_dir / "LIGAND.pdb", lig_dir / "ligand.pdb"]
            cif_candidates = [lig_dir / "LIGAND.cif", lig_dir / "ligand.cif"]
            for p in pdb_candidates:
                if p.exists():
                    return p
            for p in cif_candidates:
                if p.exists():
                    return p
            # fall back to stored path
            p = Path(entry.get("path", ""))
            return p if p.exists() else None
        except Exception:
            return None

    def _open_ligand_in_chimerax_view(self):
        name = self.ligandDropdown.currentText().strip()
        if not name or name in {"Select saved…", "Select saved..."}:
            _msg_warn(self, "No Ligand", "Select a saved ligand first.")
            return
        entry = self.ligand_cache.get(name)
        if not isinstance(entry, dict):
            _msg_warn(self, "No Cache Entry", "Saved ligand entry is missing metadata; re-save the ligand.")
            return

        target = self._ligand_entry_to_structure_path(entry)
        if not target:
            _msg_warn(self, "Not Found", "No ligand structure file found for this ligand.")
            return

        self.log(f"🧩 Opening {target.name} in ChimeraX…")
        try:
            chimerax_exe = Path(r"C:\Program Files\ChimeraX\bin\ChimeraX.exe")
            if sys.platform.startswith("win") and chimerax_exe.exists():
                subprocess.Popen([str(chimerax_exe), str(target)])
            else:
                subprocess.Popen(["chimerax", str(target)])
        except Exception as e:
            _msg_err(self, "Error", f"Failed to open ChimeraX:\n{e}")

    def _open_ligand_in_pymol_view(self):
        name = self.ligandDropdown.currentText().strip()
        if not name or name in {"Select saved…", "Select saved..."}:
            _msg_warn(self, "No Ligand", "Select a saved ligand first.")
            return
        entry = self.ligand_cache.get(name)
        if not isinstance(entry, dict):
            _msg_warn(self, "No Cache Entry", "Saved ligand entry is missing metadata; re-save the ligand.")
            return

        target = self._ligand_entry_to_structure_path(entry)
        if not target:
            _msg_warn(self, "Not Found", "No ligand structure file found for this ligand.")
            return

        self.log(f"🧪 Opening {target.name} in PyMOL…")
        try:
            # user said backend will be added later; this is best-effort
            subprocess.Popen(["pymol", str(target)])
        except Exception as e:
            _msg_err(self, "Error", f"Failed to open PyMOL:\n{e}")

    def _update_ligand_preview_from_cache(self, ligand_name: str):
        """
        Simple 2D preview hook:
        - If af3_pipeline.ligand_utils later writes an image (e.g. LIGAND_2D.png)
          into the ligand hash folder, we can display it here.
        """
        entry = self.ligand_cache.get(ligand_name)
        if not isinstance(entry, dict):
            return

        lig_hash = entry.get("hash") or cache_utils.compute_hash(entry.get("smiles", ""))
        lig_dir = LIGAND_CACHE / lig_hash
        img_candidates = [
            lig_dir / "LIGAND.svg",
            lig_dir / "LIGAND.png",
        ]
        img = next((p for p in img_candidates if p.exists()), None)
        if not img:
            self.ligandPreviewLabel.setText("(No 2D preview available yet)")
            self.ligandPreviewLabel.setPixmap(QPixmap())
            return

        pix = QPixmap(str(img))
        if pix.isNull():
            self.ligandPreviewLabel.setText("(Failed to load preview image)")
            self.ligandPreviewLabel.setPixmap(QPixmap())
            return

        self.ligandPreviewLabel.setPixmap(pix.scaled(
            self.ligandPreviewLabel.width(),
            self.ligandPreviewLabel.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self.ligandPreviewLabel.setText("")

    # =======================
    # 🧠 Alphafold page
    # =======================
    def _connect_alphafold_page(self):
        # Add entry buttons
        self.addProteinButton.clicked.connect(lambda: self._add_macro_entry("protein"))
        self.addDNAButton.clicked.connect(lambda: self._add_macro_entry("dna"))
        self.addRNAButton.clicked.connect(lambda: self._add_macro_entry("rna"))

        # Ions/Cofactors pickers
        self.editIonsButton.clicked.connect(lambda: self._choose_multi_text(self.ionsInput, "Select Ions", ION_CHOICES))
        self.editCofactorsButton.clicked.connect(lambda: self._choose_multi_text(self.cofactorsInput, "Select Cofactors", COFACTOR_CHOICES))

        # Seeds
        self.autoSeedButton.clicked.connect(self._auto_generate_seeds)

        # Run / queue
        self.runButton.clicked.connect(self._run_now)
        self.addToQueueButton.clicked.connect(self._add_current_to_queue)
        self.runQueueButton.clicked.connect(self._run_queue)
        self.queueMoveUpButton.clicked.connect(self._queue_move_up)
        self.queueMoveDownButton.clicked.connect(self._queue_move_down)
        self.queueRemoveButton.clicked.connect(self._queue_remove_selected)

        # Output
        self.openOutputButton.clicked.connect(self._open_output_directory)

        # Post-analysis
        self.rosettaButton.clicked.connect(self._run_post_af3_analysis)

        # Notes + logs
        self.notesButton.clicked.connect(self._open_notes_dialog)

        # If no entries yet, keep placeholder labels until first add
        # (we remove placeholders on first add)

    def _choose_multi_text(self, target_line_edit: QLineEdit, title: str, items: list[str]):
        selected = select_multiple(self, title, items)
        if selected:
            target_line_edit.setText(selected)

    def _update_covalent_fields(self):
        enabled = self.covalentCheckbox.isChecked()
        self.runLigandDropdown.setEnabled(True)  # ligand selection always enabled
        self.covalentChain.setEnabled(enabled)
        self.covalentResidue.setEnabled(enabled)
        self.proteinAtomComboBox.setEnabled(enabled)
        self.covalentLigandAtom.setEnabled(enabled)

    def _on_protein_atom_change(self, text: str):
        if text == "Custom…":
            atom, ok = QInputDialog.getText(self, "Custom Atom", "Enter atom name (e.g. OG1):")
            self.custom_protein_atom = atom.strip() if ok and atom.strip() else None
        elif text == "Custom...":
            atom, ok = QInputDialog.getText(self, "Custom Atom", "Enter atom name (e.g. OG1):")
            self.custom_protein_atom = atom.strip() if ok and atom.strip() else None
        else:
            self.custom_protein_atom = None

    def _add_macro_entry(self, kind: str):
        # remove placeholder label (if present)
        if kind == "protein":
            self._remove_placeholder_if_present(self.proteinsContainerLayout, "proteinsPlaceholder")
            container_layout = self.proteinsContainerLayout
            names = self._names_for("protein")
            entry_list = self.protein_entries
        elif kind == "dna":
            self._remove_placeholder_if_present(self.dnaContainerLayout, "dnaPlaceholder")
            container_layout = self.dnaContainerLayout
            names = self._names_for("dna")
            entry_list = self.dna_entries
        else:
            self._remove_placeholder_if_present(self.rnaContainerLayout, "rnaPlaceholder")
            container_layout = self.rnaContainerLayout
            names = self._names_for("rna")
            entry_list = self.rna_entries

        def on_delete(refs: MacroEntryRefs):
            self._delete_macro_entry(kind, refs)

        def on_select(refs: MacroEntryRefs, selected_name: str):
            if selected_name in {"Select saved…", "Select saved..."}:
                return
            seq = self._get_seq_from_cache(selected_name)
            if getattr(refs, "name", None):
                refs.name.setText(selected_name)
            refs.seq.setPlainText(seq)

        entry = _build_entry_widget(
            parent=self,
            kind=kind,
            saved_names=names,
            ptm_choices=PTM_CHOICES,
            on_delete=on_delete,
            on_select=on_select,
        )
        entry.ptm_button.clicked.connect(lambda _=False, e=entry: self._edit_ptms_for_entry(e))
        container_layout.addWidget(entry.root)
        entry_list.append(entry)
        self.log(f"➕ Added {kind.upper()} entry")

    def _remove_placeholder_if_present(self, layout, obj_name: str):
        # obj_name label exists in UI; if still there, remove it
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w and w.objectName() == obj_name:
                layout.takeAt(i)
                w.setParent(None)
                w.deleteLater()
                return

    def _delete_macro_entry(self, kind: str, refs: MacroEntryRefs):
        if kind == "protein":
            lst = self.protein_entries
        elif kind == "dna":
            lst = self.dna_entries
        else:
            lst = self.rna_entries

        if refs in lst:
            lst.remove(refs)

        refs.root.setParent(None)
        refs.root.deleteLater()
        self.log(f"🗑 Removed {kind.upper()} entry")

    def _auto_generate_seeds(self):
        n = self.autoSeedSpin.value()
        seeds = random.sample(range(1, 100000), n)
        self.modelSeedInput.setText(", ".join(str(s) for s in seeds))
        self.log(f"🎲 Auto-generated {n} model seeds: {seeds}")

    def _parse_model_seeds(self) -> Optional[list[int]]:
        txt = self.modelSeedInput.text().strip()
        if not txt:
            return None
        out: list[int] = []
        for part in txt.split(","):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return out or None

    def _build_current_job_spec(self) -> dict[str, Any]:
        jobname = _sanitize_jobname(self.jobNameInput.text())
        if not jobname:
            return {}

        # Proteins
        proteins: list[dict[str, Any]] = []
        for e in self.protein_entries:
            seq = e.seq.toPlainText().strip()
            if not seq:
                continue

            name = (e.name.text().strip() or "").strip()
            template = (e.template.text().strip() if e.template else "")

            ptms = list(e.ptms or [])
            # Backward compat fields (use first PTM if present)
            if ptms:
                mod_ccd = ptms[0].get("ccd", "None")
                pos = ptms[0].get("pos", "")
            else:
                mod_ccd = "None"
                pos = ""

            proteins.append({
                "name": name,
                "sequence": seq,
                "template": template,

                # NEW
                "ptms": ptms,

                # OLD/compat
                "modification": mod_ccd,
                "mod_position": pos,
            })

        dna_list: list[dict[str, Any]] = []
        for e in self.dna_entries:
            seq = e.seq.toPlainText().strip()
            if not seq:
                continue

            ptms = list(e.ptms or [])
            if ptms:
                mod_ccd = ptms[0].get("ccd", "None")
                pos = ptms[0].get("pos", "")
            else:
                mod_ccd = "None"
                pos = ""

            dna_list.append({
                "sequence": seq,

                # NEW
                "ptms": ptms,

                # OLD/compat
                "modification": mod_ccd,
                "pos": pos,
            })

        rna_list: list[dict[str, Any]] = []
        for e in self.rna_entries:
            seq = e.seq.toPlainText().strip()
            if not seq:
                continue

            ptms = list(e.ptms or [])
            if ptms:
                mod_ccd = ptms[0].get("ccd", "None")
                pos = ptms[0].get("pos", "")
            else:
                mod_ccd = "None"
                pos = ""

            rna_list.append({
                "sequence": seq,

                # NEW
                "ptms": ptms,

                # OLD/compat
                "modification": mod_ccd,
                "pos": pos,
            })


        # ligand for run: selected from runLigandDropdown
        ligand_name = self.runLigandDropdown.currentText().strip()
        ligand_smiles = ""
        lig_entry = None
        if ligand_name and ligand_name not in {"Select ligand…", "Select ligand...", "Select saved…", "Select saved..."}:
            lig_entry = self.ligand_cache.get(ligand_name)
            if isinstance(lig_entry, str):
                ligand_smiles = lig_entry
            elif isinstance(lig_entry, dict):
                ligand_smiles = lig_entry.get("smiles", "") or ""

        selected_atom_text = self.proteinAtomComboBox.currentText()
        if selected_atom_text in {"Custom…", "Custom..."} and self.custom_protein_atom:
            prot_atom = self.custom_protein_atom
        else:
            prot_atom = ATOM_MAP.get(selected_atom_text, "")

        ligand = {
            "smiles": ligand_smiles.strip(),
            "covalent": self.covalentCheckbox.isChecked(),
            "chain": self.covalentChain.text().strip(),
            "residue": self.covalentResidue.text().strip(),
            "prot_atom": prot_atom,
            "ligand_atom": self.covalentLigandAtom.text().strip(),
            "ions": self.ionsInput.text().strip(),
            "cofactors": self.cofactorsInput.text().strip(),
        }

        seeds = self._parse_model_seeds()
        if seeds:
            ligand["modelSeeds"] = seeds

        spec = {
            "jobname": jobname,
            "proteins": proteins,
            "rna": rna_list,
            "dna": dna_list,
            "ligand": ligand,
            "created_at": _now_iso(),
        }
        return spec

    # =======================
    # 🧺 Queue (embedded)
    # =======================
    def _read_queue(self) -> list[dict[str, Any]]:
        q = _read_json(QUEUE_FILE, [])
        return q if isinstance(q, list) else []

    def _save_queue(self, q: list[dict[str, Any]]):
        # ensure JSON-safe
        def sanitize(o):
            if isinstance(o, dict):
                return {k: sanitize(v) for k, v in o.items()}
            if isinstance(o, list):
                return [sanitize(v) for v in o]
            try:
                json.dumps(o)
                return o
            except TypeError:
                return str(o)

        _write_json(QUEUE_FILE, sanitize(q))

    def _refresh_queue_view(self):
        q = self._read_queue()
        self.queueList.clear()
        for item in q:
            name = (item.get("jobname") or "").strip() or "(unnamed)"
            self.queueList.addItem(name)

    def _add_current_to_queue(self):
        spec = self._build_current_job_spec()
        if not spec:
            _msg_warn(self, "Missing job", "Please enter a job name.")
            return

        q = self._read_queue()
        q.append(spec)
        self._save_queue(q)
        self._refresh_queue_view()
        self.log(f"🧺 Queued job: {spec.get('jobname')}")

    def _queue_remove_selected(self):
        row = self.queueList.currentRow()
        if row < 0:
            return
        q = self._read_queue()
        if 0 <= row < len(q):
            removed = q.pop(row)
            self._save_queue(q)
            self._refresh_queue_view()
            self.log(f"🗑 Removed from queue: {removed.get('jobname','(unnamed)')}")

    def _queue_move_up(self):
        row = self.queueList.currentRow()
        if row <= 0:
            return
        q = self._read_queue()
        if row < len(q):
            q[row - 1], q[row] = q[row], q[row - 1]
            self._save_queue(q)
            self._refresh_queue_view()
            self.queueList.setCurrentRow(row - 1)

    def _queue_move_down(self):
        row = self.queueList.currentRow()
        q = self._read_queue()
        if row < 0 or row >= len(q) - 1:
            return
        q[row + 1], q[row] = q[row], q[row + 1]
        self._save_queue(q)
        self._refresh_queue_view()
        self.queueList.setCurrentRow(row + 1)

    def _run_queue(self):
        if self.is_running:
            self.log("⚠️ A job is already running.")
            return
        q = self._read_queue()
        if not q:
            self.log("ℹ️ Queue is empty.")
            return
        self._run_spec(q.pop(0))
        self._save_queue(q)
        self._refresh_queue_view()

    def _run_now(self):
        if self.is_running:
            self.log("⚠️ A job is already running. Use Add to Queue.")
            return
        spec = self._build_current_job_spec()
        if not spec:
            _msg_warn(self, "Missing job", "Please enter a job name.")
            return
        self._run_spec(spec)

    def _maybe_run_next(self):
        q = self._read_queue()
        if not q:
            return
        nxt = q.pop(0)
        self._save_queue(q)
        self._refresh_queue_view()
        self.log(f"➡️ Starting next queued job: {nxt.get('jobname','(unnamed)')}")
        self._run_spec(nxt)

    # =======================
    # 🏃 Build & Run
    # =======================
    def _run_spec(self, spec: dict[str, Any]):
        jobname = (spec.get("jobname") or "").strip()
        if not jobname:
            _msg_warn(self, "Bad job", "Job spec missing jobname.")
            return

        proteins = spec.get("proteins", []) or []
        rna = spec.get("rna", []) or []
        dna = spec.get("dna", []) or []
        ligand = spec.get("ligand", {}) or {}

        if not proteins and not rna and not dna and not (ligand.get("smiles") or "").strip():
            _msg_warn(self, "No inputs", "No proteins/DNA/RNA/ligand provided.")
            return

        self.current_jobname = jobname
        self.is_running = True
        self.runButton.setEnabled(False)
        self.addToQueueButton.setEnabled(False)
        self.log(f"🚀 Running job: {jobname}")
        self.log("⚙️ Building AF3 JSON and running MSA in background…")

        self.build_thread = BuildThread(jobname, proteins, rna, dna, ligand)
        self.build_thread.progress.connect(lambda msg: self.log(f"🧬 {msg}"))
        self.build_thread.finished.connect(self._on_build_finished)
        self.build_thread.failed.connect(self._on_build_failed)
        self.build_thread.start()

    def _on_build_finished(self, json_path: str):
        self.log(f"✅ JSON build complete: {json_path}")

        auto_flag = bool(self.autoAnalyzeCheckbox.isChecked())
        multi_flag = bool(self.multiSeedCheckbox.isChecked())

        self.run_thread = RunThread(
            json_path=json_path,
            job_name=self.current_jobname or "GUI_job",
            auto_analyze=auto_flag,
            multi_seed=multi_flag,
        )
        self.run_thread.finished.connect(self._on_run_finished)
        self.run_thread.failed.connect(self._on_run_failed)
        self.run_thread.start()

    def _on_build_failed(self, err: str):
        self.log(f"❌ Build failed: {err}")
        self.is_running = False
        self.runButton.setEnabled(True)
        self.addToQueueButton.setEnabled(True)
        self._maybe_run_next()

    def _on_run_finished(self, json_path: str):
        self.log(f"✅ Run finished: {json_path}")
        self.is_running = False
        self.runButton.setEnabled(True)
        self.addToQueueButton.setEnabled(True)

        # record history
        self._record_run_history_from_spec(self._build_current_job_spec(), json_path=json_path)

        # auto-analysis if checked (runner may also do it, but keep UI-trigger too)
        if self.autoAnalyzeCheckbox.isChecked():
            self.log("🧠 Auto-running post-AF3 analysis…")
            self._run_post_af3_analysis()

        self._refresh_runs_view()
        self._maybe_run_next()

    def _on_run_failed(self, err: str):
        self.log(f"❌ Run failed: {err}")
        self.is_running = False
        self.runButton.setEnabled(True)
        self.addToQueueButton.setEnabled(True)
        self._maybe_run_next()

    # =======================
    # 📊 Post-AF3 analysis
    # =======================
    def _run_post_af3_analysis(self):
        job_name = self.rosettaPath.text().strip() if self.rosettaPath.text().strip() else self.current_jobname.strip()
        if not job_name:
            _msg_warn(self, "Missing job", "Please enter a job folder name (or run a job) first.")
            return

        multi_flag = bool(self.multiSeedCheckbox.isChecked())
        self.log(f"🧠 Post-AF3 analysis for job '{job_name}'")

        self.analysis_thread = AnalysisWorker(job_name, multi_flag)
        self.analysis_thread.log.connect(self.log)
        self.analysis_thread.done.connect(lambda: self.log("✅ Post-AF3 analysis complete."))
        self.analysis_thread.error.connect(lambda m: (_msg_err(self, "Analysis Error", m), self.log(f"❌ Analysis failed:\n{m}")))
        self.analysis_thread.start()

    # =======================
    # 🪵 Logging
    # =======================
    def log(self, message: str):
        self.logOutput.appendPlainText(str(message))
        sb = self.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _open_notes_dialog(self):
        try:
            dlg = NotesDialog(self)
            dlg.exec()
        except Exception as e:
            _msg_err(self, "Notes Error", f"Failed to open notes dialog:\n{e}")

    # =======================
    # 📂 Output directory (config-aware)
    # =======================
    def _open_output_directory(self):
        import subprocess as _subprocess

        # Always use the *local* user home, not WSL
        target_dir = Path.home() / ".af3_pipeline" / "jobs"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)  # ensure it exists

            if sys.platform.startswith("win"):
                os.startfile(str(target_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                _subprocess.call(["open", str(target_dir)])
            else:
                _subprocess.call(["xdg-open", str(target_dir)])
        except OSError as e:
            _msg_err(
                self,
                "Error",
                f"Failed to open jobs directory:\n{target_dir}\n\n{e}",
            )


    # =======================
    # 🧾 Runs tab
    # =======================
    def _connect_runs_page(self):
        self.runsList.currentRowChanged.connect(self._show_selected_run_details)
        self.runsOpenOutputButton.clicked.connect(self._open_output_directory)
        self.runsLoadToAlphafoldButton.clicked.connect(self._load_selected_run_to_alphafold)

    def _record_run_history_from_spec(self, spec: dict[str, Any], *, json_path: str):
        if not spec:
            return
        rec = dict(spec)
        rec["json_path"] = json_path
        rec["finished_at"] = _now_iso()
        self.runs_history.insert(0, rec)
        _write_json(RUNS_HISTORY_FILE, self.runs_history)

    def _refresh_runs_view(self):
        self.runs_history = _read_json(RUNS_HISTORY_FILE, [])
        if not isinstance(self.runs_history, list):
            self.runs_history = []

        self.runsList.clear()
        for rec in self.runs_history:
            name = (rec.get("jobname") or "(unnamed)").strip()
            ts = rec.get("finished_at") or rec.get("created_at") or ""
            self.runsList.addItem(f"{name}    {ts}")

        self.runDetailsText.setPlainText("Select a run to view details…")

    def _selected_run_record(self) -> Optional[dict[str, Any]]:
        row = self.runsList.currentRow()
        if row < 0:
            return None
        if row >= len(self.runs_history):
            return None
        rec = self.runs_history[row]
        return rec if isinstance(rec, dict) else None

    def _show_selected_run_details(self, row: int):
        rec = self._selected_run_record()
        if not rec:
            return
        pretty = json.dumps(rec, indent=2)
        self.runDetailsText.setPlainText(pretty)

    def _load_selected_run_to_alphafold(self):
        rec = self._selected_run_record()
        if not rec:
            return

        # switch to alphafold page
        self.pagesStack.setCurrentIndex(3)
        self.navList.setCurrentRow(3)

        # jobname
        self.jobNameInput.setText(rec.get("jobname", ""))

        # ligand selection (binding section ligand dropdown, not Ligands tab dropdown)
        lig = rec.get("ligand", {}) if isinstance(rec.get("ligand", {}), dict) else {}
        lig_smiles = (lig.get("smiles") or "").strip()
        lig_name = self._find_ligand_name_by_smiles(lig_smiles)
        if lig_name:
            idx = self.runLigandDropdown.findText(lig_name)
            if idx >= 0:
                self.runLigandDropdown.setCurrentIndex(idx)

        # covalent + environment
        self.covalentCheckbox.setChecked(bool(lig.get("covalent", False)))
        self.covalentChain.setText(lig.get("chain", "") or "")
        self.covalentResidue.setText(lig.get("residue", "") or "")
        self.covalentLigandAtom.setText(lig.get("ligand_atom", "") or "")
        self.ionsInput.setText(lig.get("ions", "") or "")
        self.cofactorsInput.setText(lig.get("cofactors", "") or "")

        # seeds
        seeds = lig.get("modelSeeds")
        if isinstance(seeds, list) and seeds:
            self.modelSeedInput.setText(", ".join(str(x) for x in seeds))
        else:
            self.modelSeedInput.setText("")

        # clear existing dynamic entries and rebuild
        self._clear_dynamic_entries()

        def _force_select_dropdown(dd: QComboBox, wanted: str) -> None:
            """Select dropdown item by text; if missing, add it temporarily."""
            wanted = (wanted or "").strip()
            if not wanted or wanted in {"Select saved…", "Select saved..."}:
                dd.setCurrentIndex(0)
                return

            idx = dd.findText(wanted)
            if idx >= 0:
                dd.setCurrentIndex(idx)
                return

            # Not present (e.g., saved sequence deleted). Add temporarily so it can be selected.
            was_blocked = dd.blockSignals(True)
            try:
                dd.addItem(wanted)
            finally:
                dd.blockSignals(was_blocked)

            idx2 = dd.findText(wanted)
            dd.setCurrentIndex(idx2 if idx2 >= 0 else 0)

        # -----------------------
        # Proteins
        # -----------------------
        for p in rec.get("proteins", []) or []:
            if not isinstance(p, dict):
                continue
            self._add_macro_entry("protein")
            e = self.protein_entries[-1]

            # dropdown selection (this is what you were missing)
            _force_select_dropdown(e.dropdown, p.get("name", "") or "")

            # keep hidden name in sync if it exists
            if getattr(e, "name", None) is not None:
                e.name.setText(p.get("name", "") or "")

            if e.template:
                e.template.setText(p.get("template", "") or "")

            # store sequence for reproducibility (even if dropdown name isn't saved anymore)
            e.seq.setPlainText(p.get("sequence", "") or "")

            # PTM selection by CCD
            # NEW PTMs restore
            ptms = p.get("ptms")
            if isinstance(ptms, list):
                e.ptms = [x for x in ptms if isinstance(x, dict)]
            else:
                # fallback from old single PTM fields
                mod = p.get("modification")
                pos = p.get("mod_position", "") or ""
                e.ptms = []
                if mod not in (None, "None") and str(pos).strip():
                    e.ptms = [{"label": "", "ccd": str(mod), "pos": str(pos).strip()}]

            e.ptm_summary.setText(self._ptm_summary_text(e.ptms))


        # -----------------------
        # DNA
        # -----------------------
        for d in rec.get("dna", []) or []:
            if not isinstance(d, dict):
                continue
            self._add_macro_entry("dna")
            e = self.dna_entries[-1]

            _force_select_dropdown(e.dropdown, d.get("name", "") or "")
            if getattr(e, "name", None) is not None:
                e.name.setText(d.get("name", "") or "")

            e.seq.setPlainText(d.get("sequence", "") or "")
            self._set_combo_by_data(e.ptm, d.get("modification"))
            e.ptm_pos.setText(d.get("pos", "") or "")

        # -----------------------
        # RNA
        # -----------------------
        for r in rec.get("rna", []) or []:
            if not isinstance(r, dict):
                continue
            self._add_macro_entry("rna")
            e = self.rna_entries[-1]

            _force_select_dropdown(e.dropdown, r.get("name", "") or "")
            if getattr(e, "name", None) is not None:
                e.name.setText(r.get("name", "") or "")

            e.seq.setPlainText(r.get("sequence", "") or "")
            self._set_combo_by_data(e.ptm, r.get("modification"))
            e.ptm_pos.setText(r.get("pos", "") or "")

        self.log(f"↩ Loaded run into Alphafold: {rec.get('jobname','(unnamed)')}")


    def _find_ligand_name_by_smiles(self, smiles: str) -> Optional[str]:
        if not smiles:
            return None
        canon = _canonical_smiles(smiles)
        for name, entry in self.ligand_cache.items():
            if isinstance(entry, str) and _canonical_smiles(entry) == canon:
                return name
            if isinstance(entry, dict) and _canonical_smiles(entry.get("smiles", "")) == canon:
                return name
        return None

    def _set_combo_by_data(self, combo: QComboBox, data_value):
        # find by .itemData
        for i in range(combo.count()):
            if combo.itemData(i) == data_value:
                combo.setCurrentIndex(i)
                return
        # fallback: None -> "None"
        if data_value in {None, "None"}:
            combo.setCurrentIndex(0)

    def _clear_dynamic_entries(self):
        # remove all entry widgets
        for lst in (self.protein_entries, self.dna_entries, self.rna_entries):
            for e in list(lst):
                e.root.setParent(None)
                e.root.deleteLater()
            lst.clear()

        # restore placeholders (simple: add labels back)
        self._restore_placeholder(self.proteinsContainerLayout, "proteinsPlaceholder", "(No proteins added yet — click “Add Protein”)")
        self._restore_placeholder(self.dnaContainerLayout, "dnaPlaceholder", "(No DNA added yet — click “Add DNA”)")
        self._restore_placeholder(self.rnaContainerLayout, "rnaPlaceholder", "(No RNA added yet — click “Add RNA”)")

    def _restore_placeholder(self, layout, obj_name: str, text: str):
        # if no widgets present, add placeholder label
        if layout.count() == 0:
            lab = QLabel(text)
            lab.setObjectName(obj_name)
            layout.addWidget(lab)

    # =======================
    # ⚙️ Config tab (scaffold)
    # =======================
    def _connect_config_page(self):
        # ---- apply/save/reset ----
        self.cfgApplyButton.clicked.connect(self._apply_config_fields)
        self.cfgSaveButton.clicked.connect(self._save_config_yaml)
        self.cfgResetButton.clicked.connect(self._reset_config_fields_from_cfg)
        self.cfgAutoDetectButton.clicked.connect(self._start_autodetect)
        self._reset_config_fields_from_cfg()
    
    def _posixify(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return s

        # If pathlib/Windows turned "/home/..." into "\home\..."
        # force it back to POSIX root form.
        if s.startswith("\\") and not s.startswith("\\\\"):
            s = "/" + s.lstrip("\\")

        # Convert all backslashes to slashes
        s = s.replace("\\", "/")

        # Collapse accidental doubles (optional)
        while "//" in s:
            s = s.replace("//", "/")

        return s

    def _ui_config_dict(self) -> dict[str, Any]:
        # docker env text → dict
        env_text = self.cfgDockerEnv.toPlainText().strip()
        env_dict: dict[str, str] = {}

        if env_text:
            try:
                # allow user to paste YAML mapping
                loaded = yaml.safe_load(env_text)
                if isinstance(loaded, dict):
                    env_dict = {str(k): str(v) for k, v in loaded.items()}
                else:
                    raise ValueError("Docker env must be a YAML mapping.")
            except Exception:
                # fallback: parse KEY=VALUE lines
                for line in env_text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" in line and "=" not in line:
                        # allow KEY: VALUE
                        k, v = line.split(":", 1)
                    elif "=" in line:
                        k, v = line.split("=", 1)
                    else:
                        continue
                    env_dict[k.strip()] = v.strip().strip('"').strip("'")

        return {
            "wsl_distro": self.cfgWslDistro.text().strip(),
            "gui_dir": self.cfgGuiDir.text().strip(),
            "cache_root": str(self._profile_root / "cache") if hasattr(self, "_profile_root") else str(Path.home() / ".af3_pipeline" / "cache"),
            "linux_home_root": self.cfgLinuxHomeRoot.text().strip(),

            "af3_dir": self.cfgAf3Dir.text().strip(),
            "docker_bin": self.cfgDockerBin.text().strip(),
            "alphafold_docker_image": self.cfgDockerImage.text().strip(),
            "alphafold_docker_env": env_dict,

            "msa": {
                "threads": int(self.cfgMsaThreads.value()),
                "sensitivity": float(self.cfgMsaSensitivity.value()),
                "max_seqs": int(self.cfgMsaMaxSeqs.value()),
                "db": self._posixify(self.cfgMMseqsDB.text()),
            },

            "ligand": {
                "n_confs": int(self.cfgLigandNConfs.value()),
                "seed": int(self.cfgLigandSeed.value()),
                "prune_rms": float(self.cfgLigandPruneRms.value()),
                "keep_charge": bool(self.cfgLigandKeepCharge.isChecked()),
                "require_assigned_stereo": bool(self.cfgLigandRequireStereo.isChecked()),
                "basename": self.cfgLigandBasename.text().strip() or "LIGAND",
                "name_default": self.cfgLigandNameDefault.text().strip() or "LIG",
                "png_size": [int(self.cfgLigandPngW.value()), int(self.cfgLigandPngH.value())],
                "rdkit_threads": int(self.cfgLigandRdkitThreads.value()),
            },

            "rosetta_relax_bin": self._posixify(self.cfgRosettaRelaxBin.text()),
        }

    def _apply_config_fields(self):
        try:
            # Write yaml first
            self._save_config_yaml()

            self.log(f"⚙️ Applied config (saved to YAML).")
            _msg_info(
                self,
                "Applied",
                "Config saved.\n\nSome settings (backend runner/ligand/MSA) may require restarting the GUI to fully apply."
            )
        except Exception as e:
            _msg_err(self, "Apply failed", str(e))

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent

    def _config_yaml_path(self) -> Path:
        root = getattr(self, "_profile_root", None)
        if isinstance(root, Path):
            return root / "config.yaml"
        return Path.home() / ".af3_pipeline" / "config.yaml"
    
    def _read_yaml_file(self, p: Path) -> dict[str, Any]:
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _deep_merge_dicts(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = self._deep_merge_dicts(base[k], v)  # type: ignore[index]
            else:
                base[k] = v
        return base



    def _save_config_yaml(self):
        try:
            out = self._config_yaml_path()
            out.parent.mkdir(parents=True, exist_ok=True)

            template = Path(__file__).resolve().parent / "config_template.yaml"
            base = self._read_yaml_file(template) if template.exists() else {}

            ui = self._ui_config_dict()
            merged = self._deep_merge_dicts(base, ui)

            out.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
            self.log(f"💾 Saved config.yaml: {out}")
            _msg_info(self, "Saved", f"Saved config.yaml to:\n{out}")
        except Exception as e:
            _msg_err(self, "Save failed", str(e))



    def _load_config_yaml_best_effort(self) -> dict[str, Any]:
        p = self._config_yaml_path()
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _reset_config_fields_from_cfg(self):
        data = self._load_config_yaml_best_effort()

        def g(key, default=""):
            return (data.get(key, cfg.get(key, default)) if isinstance(data, dict) else cfg.get(key, default))

        self.cfgWslDistro.setText(str(g("wsl_distro", "")))
        self.cfgLinuxHomeRoot.setText(str(g("linux_home_root", "")))

        self.cfgGuiDir.setText(str(g("gui_dir", "")))

        self.cfgAf3Dir.setText(str(g("af3_dir", "")))
        self.cfgDockerBin.setText(str(g("docker_bin", "")))
        self.cfgDockerImage.setText(str(g("alphafold_docker_image", "")))

        env = g("alphafold_docker_env", {})
        if isinstance(env, dict):
            self.cfgDockerEnv.setPlainText(yaml.safe_dump(env, sort_keys=False).strip())
        else:
            self.cfgDockerEnv.setPlainText("")

        msa = data.get("msa", {}) if isinstance(data.get("msa", {}), dict) else {}
        self.cfgMsaThreads.setValue(int(msa.get("threads", cfg.get("msa_threads", 10))))
        self.cfgMsaSensitivity.setValue(float(msa.get("sensitivity", cfg.get("msa_sensitivity", 5.7))))
        self.cfgMsaMaxSeqs.setValue(int(msa.get("max_seqs", cfg.get("msa_max_seqs", 25))))

        lig = data.get("ligand", {}) if isinstance(data.get("ligand", {}), dict) else {}
        self.cfgLigandNConfs.setValue(int(lig.get("n_confs", 200)))
        self.cfgLigandSeed.setValue(int(lig.get("seed", 0)))
        self.cfgLigandPruneRms.setValue(float(lig.get("prune_rms", 0.25)))
        self.cfgLigandKeepCharge.setChecked(bool(lig.get("keep_charge", False)))
        self.cfgLigandRequireStereo.setChecked(bool(lig.get("require_assigned_stereo", False)))
        self.cfgLigandBasename.setText(str(lig.get("basename", "LIGAND")))
        self.cfgLigandNameDefault.setText(str(lig.get("name_default", "LIG")))
        png = lig.get("png_size", [1500, 1200])
        if isinstance(png, list) and len(png) == 2:
            self.cfgLigandPngW.setValue(int(png[0]))
            self.cfgLigandPngH.setValue(int(png[1]))
        self.cfgLigandRdkitThreads.setValue(int(lig.get("rdkit_threads", 0)))

        self.cfgRosettaRelaxBin.setText(str(g("rosetta_relax_bin", "")))
        msa_db = msa.get("db", cfg.get("msa.db", ""))
        self.cfgMMseqsDB.setText(self._posixify(str(msa_db)))


    # =======================
    # 🏁 entry safety
    # =======================
    def closeEvent(self, event):  # noqa: N802
        # optional: persist queue view etc.
        super().closeEvent(event)


# ==========================
# 🏁 Entry
# ==========================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    ico = QIcon(resource_path("assets/custom_icon_2.ico"))
    app.setWindowIcon(ico)

    try:
        pass
    except Exception as e:
        QMessageBox.critical(
            None,
            "Configuration Error",
            f"Failed to load AF3 configuration:\n{e}",
        )
        sys.exit(1)

    window = MainWindow()
    window.setWindowIcon(ico)
    window.show()
    sys.exit(app.exec())
