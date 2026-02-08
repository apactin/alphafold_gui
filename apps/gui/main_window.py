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
import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Optional, List
from typing import Any, Optional
from zoneinfo import ZoneInfo
from typing import Any

import config_tab as _config_tab
import runs_features as _runs_features
import runs_metrics as _runs_metrics
import viewers as _viewers

from PyQt6 import uic
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QRect, QRectF, QTimer, QProcess, QSignalBlocker, QUrl
from PyQt6.QtGui import QPixmap, QPainter, QPen, QFontMetrics, QWheelEvent, QFont, QIcon, QDesktopServices, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
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
    QInputDialog,
    QScrollBar,
    QToolButton, 
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QSizePolicy,
    QFileDialog,
    QDoubleSpinBox, 
    QSpinBox, 
    QDialogButtonBox, 
    QFormLayout,
    QGridLayout,
)


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

def norm_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()

def _default_user_cfg_dir() -> Path:
    return Path.home() / ".af3_pipeline"

GUI_CACHE_DIR: Path = norm_path(_default_user_cfg_dir() / "users" / "_UNSET_" / "gui_cache")
SEQUENCE_CACHE_FILE: Path = GUI_CACHE_DIR / "sequence_cache.json"
LIGAND_CACHE_FILE: Path   = GUI_CACHE_DIR / "ligand_cache.json"
QUEUE_FILE: Path          = GUI_CACHE_DIR / "job_queue.json"
NOTES_FILE: Path          = GUI_CACHE_DIR / "notes.txt"
RUNS_HISTORY_FILE: Path   = GUI_CACHE_DIR / "runs_history.json"
LIGAND_CACHE: Path        = norm_path(_default_user_cfg_dir() / "users" / "_UNSET_" / "cache" / "ligands")

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
    "Histidine (ND1)": "ND1",
    "Histidine (NE2)": "NE2",
    "Tyrosine (OH)": "OH",
}

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

class CompareStructsThread(QThread):
    finished_ok = pyqtSignal(dict, str)   # results, out_path
    finished_err = pyqtSignal(str)

    def __init__(self, pdb1: str, lig1: str, pdb2: str, lig2: str, cutoff: float, out_path: str):
        super().__init__()
        self.pdb1 = pdb1
        self.lig1 = lig1
        self.pdb2 = pdb2
        self.lig2 = lig2
        self.cutoff = cutoff
        self.out_path = out_path

    def run(self):
        try:
            from af3_pipeline.analysis.compare_structs import compare_and_write
            res = compare_and_write(
                pdb1=self.pdb1, lig1=self.lig1,
                pdb2=self.pdb2, lig2=self.lig2,
                cutoff=float(self.cutoff),
                out=self.out_path,
            )
            self.finished_ok.emit(res, self.out_path)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")

def _parse_atomref_text(s: str) -> dict:
    """
    Parse 'ATOM,RES,CHAIN' into {'atom':..., 'res': int, 'chain':...}
    """
    parts = [p.strip() for p in (s or "").split(",")]
    if len(parts) != 3:
        raise ValueError("Atom reference must be 'ATOM,RES,CHAIN' (e.g. SG,79,A)")
    atom, res, chain = parts
    if not atom or not chain:
        raise ValueError("Atom and chain are required (e.g. SG,79,A)")
    if not str(res).lstrip("-").isdigit():
        raise ValueError("Residue must be an integer (e.g. 79)")
    return {"atom": atom, "res": int(res), "chain": chain}

class AtomPairDialog(QDialog):
    def __init__(self, parent=None, *, default_func="HARMONIC", default_x0=1.80, default_sd=0.10):
        super().__init__(parent)
        self.setWindowTitle("Add AtomPair constraint")
        self.setFixedWidth(520)

        self.a = QLineEdit()
        self.a.setPlaceholderText("ATOM,RES,CHAIN   e.g. SG,79,A")
        self.b = QLineEdit()
        self.b.setPlaceholderText("ATOM,RES,CHAIN   e.g. C7,245,B")

        self.func = QComboBox()
        self.func.addItems(["HARMONIC", "FLAT_HARMONIC", "BOUNDED"])
        idx = self.func.findText(default_func)
        if idx >= 0:
            self.func.setCurrentIndex(idx)

        self.x0 = QDoubleSpinBox()
        self.x0.setDecimals(4)
        self.x0.setRange(-999999, 999999)
        self.x0.setValue(float(default_x0))

        self.sd = QDoubleSpinBox()
        self.sd.setDecimals(4)
        self.sd.setRange(0.0001, 999999)
        self.sd.setValue(float(default_sd))

        form = QFormLayout()
        form.addRow("a (atom,res,chain)", self.a)
        form.addRow("b (atom,res,chain)", self.b)
        form.addRow("func", self.func)
        form.addRow("x0", self.x0)
        form.addRow("sd", self.sd)

        self.err = QLabel("")
        self.err.setStyleSheet("color: #b00020;")
        self.err.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.err)
        lay.addWidget(btns)

        self._result = None

    def _accept(self):
        try:
            item = {
                "type": "AtomPair",
                "a": _parse_atomref_text(self.a.text()),
                "b": _parse_atomref_text(self.b.text()),
                "func": self.func.currentText().strip(),
                "x0": float(self.x0.value()),
                "sd": float(self.sd.value()),
            }
            self._result = item
            self.accept()
        except Exception as e:
            self.err.setText(str(e))

    def result_item(self):
        return self._result


class DihedralDialog(QDialog):
    def __init__(self, parent=None, *, default_func="CIRCULARHARMONIC", default_x0=180.0, default_sd=10.0):
        super().__init__(parent)
        self.setWindowTitle("Add Dihedral constraint")
        self.setFixedWidth(520)

        self.a = QLineEdit(); self.a.setPlaceholderText("ATOM,RES,CHAIN  e.g. CA,93,H")
        self.b = QLineEdit(); self.b.setPlaceholderText("ATOM,RES,CHAIN  e.g. CB,93,H")
        self.c = QLineEdit(); self.c.setPlaceholderText("ATOM,RES,CHAIN  e.g. ND1,94,H")
        self.d = QLineEdit(); self.d.setPlaceholderText("ATOM,RES,CHAIN  e.g. S1,1,L")

        self.func = QComboBox()
        self.func.addItems(["CIRCULARHARMONIC", "HARMONIC"])
        idx = self.func.findText(default_func)
        if idx >= 0:
            self.func.setCurrentIndex(idx)

        self.x0 = QDoubleSpinBox()
        self.x0.setDecimals(3)
        self.x0.setRange(-999999, 999999)
        self.x0.setValue(float(default_x0))

        self.sd = QDoubleSpinBox()
        self.sd.setDecimals(3)
        self.sd.setRange(0.0001, 999999)
        self.sd.setValue(float(default_sd))

        form = QFormLayout()
        form.addRow("a (atom,res,chain)", self.a)
        form.addRow("b (atom,res,chain)", self.b)
        form.addRow("c (atom,res,chain)", self.c)
        form.addRow("d (atom,res,chain)", self.d)
        form.addRow("func", self.func)
        form.addRow("x0 (deg)", self.x0)
        form.addRow("sd (deg)", self.sd)

        self.err = QLabel("")
        self.err.setStyleSheet("color: #b00020;")
        self.err.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.err)
        lay.addWidget(btns)

        self._result = None

    def _accept(self):
        try:
            item = {
                "type": "Dihedral",
                "a": _parse_atomref_text(self.a.text()),
                "b": _parse_atomref_text(self.b.text()),
                "c": _parse_atomref_text(self.c.text()),
                "d": _parse_atomref_text(self.d.text()),
                "func": self.func.currentText().strip(),
                "x0": float(self.x0.value()),
                "sd": float(self.sd.value()),
            }
            self._result = item
            self.accept()
        except Exception as e:
            self.err.setText(str(e))

    def result_item(self):
        return self._result
    
class AngleDialog(QDialog):
    def __init__(self, parent=None, *, default_func="HARMONIC", default_x0=109.5, default_sd=5.0):
        super().__init__(parent)
        self.setWindowTitle("Add Angle constraint")
        self.setFixedWidth(520)

        self.a = QLineEdit(); self.a.setPlaceholderText("ATOM,RES,CHAIN  e.g. CB,93,H")
        self.b = QLineEdit(); self.b.setPlaceholderText("ATOM,RES,CHAIN  e.g. ND1,94,H")
        self.c = QLineEdit(); self.c.setPlaceholderText("ATOM,RES,CHAIN  e.g. S1,1,L")

        self.func = QComboBox()
        self.func.addItems(["HARMONIC", "CIRCULARHARMONIC"])
        idx = self.func.findText(default_func)
        if idx >= 0:
            self.func.setCurrentIndex(idx)

        self.x0 = QDoubleSpinBox()
        self.x0.setDecimals(3)
        self.x0.setRange(-999999, 999999)
        self.x0.setValue(float(default_x0))

        self.sd = QDoubleSpinBox()
        self.sd.setDecimals(3)
        self.sd.setRange(0.0001, 999999)
        self.sd.setValue(float(default_sd))

        form = QFormLayout()
        form.addRow("a (atom,res,chain)", self.a)
        form.addRow("b (atom,res,chain)", self.b)
        form.addRow("c (atom,res,chain)", self.c)
        form.addRow("func", self.func)
        form.addRow("x0 (deg)", self.x0)
        form.addRow("sd (deg)", self.sd)

        self.err = QLabel("")
        self.err.setStyleSheet("color: #b00020;")
        self.err.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.err)
        lay.addWidget(btns)

        self._result = None

    def _accept(self):
        try:
            item = {
                "type": "Angle",
                "a": _parse_atomref_text(self.a.text()),
                "b": _parse_atomref_text(self.b.text()),
                "c": _parse_atomref_text(self.c.text()),
                "func": self.func.currentText().strip(),
                "x0": float(self.x0.value()),
                "sd": float(self.sd.value()),
            }
            self._result = item
            self.accept()
        except Exception as e:
            self.err.setText(str(e))

    def result_item(self):
        return self._result


class CoordinateDialog(QDialog):
    """
    Coordinate constraint in your YAML format:
      {"type":"Coordinate", "atom":{atom,res,chain}, "func":"HARMONIC", "x0":0.0, "sd":1.0}

    NOTE: Your exporter should interpret this as "from-input-pose" coordinate constraints
    (ref XYZ comes from the input pose). The GUI just captures the target atom + weights.
    """
    def __init__(self, parent=None, *, default_func="HARMONIC", default_x0=0.0, default_sd=1.0):
        super().__init__(parent)
        self.setWindowTitle("Add Coordinate constraint (from input pose)")
        self.setFixedWidth(520)

        self.atom = QLineEdit()
        self.atom.setPlaceholderText("ATOM,RES,CHAIN  e.g. CA,94,H")

        # In Rosetta coordinate constraints, func/x0/sd are still meaningful; your pipeline
        # may treat x0 as 0.0 for HARMONIC or ignore it depending on implementation.
        self.func = QComboBox()
        self.func.addItems(["HARMONIC", "FLAT_HARMONIC"])
        idx = self.func.findText(default_func)
        if idx >= 0:
            self.func.setCurrentIndex(idx)

        self.x0 = QDoubleSpinBox()
        self.x0.setDecimals(4)
        self.x0.setRange(-999999, 999999)
        self.x0.setValue(float(default_x0))

        self.sd = QDoubleSpinBox()
        self.sd.setDecimals(4)
        self.sd.setRange(0.0001, 999999)
        self.sd.setValue(float(default_sd))

        hint = QLabel(
            "Uses the atom's XYZ from the *input pose* as the reference.\n"
            "So you only specify the target atom + function parameters."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;")

        form = QFormLayout()
        form.addRow("atom (atom,res,chain)", self.atom)
        form.addRow("func", self.func)
        form.addRow("x0", self.x0)
        form.addRow("sd", self.sd)

        self.err = QLabel("")
        self.err.setStyleSheet("color: #b00020;")
        self.err.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addLayout(form)
        lay.addWidget(self.err)
        lay.addWidget(btns)

        self._result = None

    def _accept(self):
        try:
            item = {
                "type": "Coordinate",
                "atom": _parse_atomref_text(self.atom.text()),
                "func": self.func.currentText().strip(),
                "x0": float(self.x0.value()),
                "sd": float(self.sd.value()),
            }
            self._result = item
            self.accept()
        except Exception as e:
            self.err.setText(str(e))

    def result_item(self):
        return self._result


class WSLProcessTable(QWidget):
    """
    Polls WSL `ps` and shows only processes matching a regex (default: alphafold/rosetta/mmseqs).
    Auto-refresh every 2 seconds. Includes a Kill button to terminate the selected PID.
    """
    def __init__(self, parent=None, distro="Ubuntu-22.04"):
        super().__init__(parent)
        self.distro = distro

        self.proc = QProcess(self)
        self.proc.finished.connect(self._on_finished)

        self.kill_proc = QProcess(self)

        self.timer = QTimer(self)
        self.timer.setInterval(2000) 
        self.timer.timeout.connect(self.refresh)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)

        self.title = QLabel("WSL processes (GUI-relevant)", self)
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)

        title_row.addWidget(self.title, 1)
        outer.addLayout(title_row)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 4, 0, 0)
        controls.setSpacing(6)

        self.filterEdit = QLineEdit(self)
        self.filterEdit.setPlaceholderText("filter regex (matches full cmdline)")
        self.filterEdit.setText(r"(alphafold|rosetta|mmseqs)")

        self.btn_kill = QPushButton("Kill selected", self)
        self.btn_kill.setToolTip("Send SIGTERM to selected PID; if it persists, send SIGKILL.")
        self.btn_kill.clicked.connect(self.kill_selected)

        controls.addWidget(QLabel("Filter:", self), 0)
        controls.addWidget(self.filterEdit, 3)
        controls.addWidget(self.btn_kill, 0)

        outer.addLayout(controls)

        self.table = QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["PID", "CPU%", "MEM%", "ELAPSED", "CMD", "ARGS"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        body_font = self.table.font()

        header_font = QFont(body_font)
        header_font.setPointSize(max(8, body_font.pointSize() - 2))

        self.table.horizontalHeader().setFont(header_font)

        self.table.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.table.verticalHeader().setDefaultSectionSize(18)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents) 
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)       

        outer.addWidget(self.table, 1)

        self.filterEdit.returnPressed.connect(self.refresh)

        self.setMinimumHeight(180)
        self.setMaximumHeight(220)

        self.timer.start()
        self.refresh()

    def refresh(self):
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            return

        ps_cmd = r"ps -eo pid=,pcpu=,pmem=,etime=,comm=,args= --sort=-pcpu"
        args = ["-d", self.distro, "--", "bash", "-lc", ps_cmd]
        self.proc.start("wsl.exe", args)

    def _on_finished(self, _code, _status):
        out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")

        if err.strip() and not out.strip():
            self._set_rows([("—", "—", "—", "—", "ps", err.strip())])
            return

        pattern = (self.filterEdit.text() or "").strip()
        try:
            rx = re.compile(pattern, re.IGNORECASE) if pattern else None
        except re.error:
            rx = re.compile(r"(alphafold|rosetta|mmseqs)", re.IGNORECASE)

        rows = []
        for line in (out or "").splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 5)
            if len(parts) < 5:
                continue
            if len(parts) == 5:
                parts.append("")

            pid, cpu, mem, etime, comm, cmdline = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            hay = f"{comm} {cmdline}".strip()
            if rx and not rx.search(hay):
                continue

            rows.append((pid, cpu, mem, etime, comm, cmdline))

        self._set_rows(rows)

    def _set_rows(self, rows):
        self.table.setRowCount(len(rows))
        for r, rowdata in enumerate(rows):
            for c, val in enumerate(rowdata):
                item = QTableWidgetItem(str(val))
                if c in (0, 1, 2):
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
                self.table.setItem(r, c, item)

    def _selected_pid(self) -> str | None:
        sel = self.table.selectionModel()
        if not sel or not sel.hasSelection():
            return None
        row = sel.selectedRows()[0].row()
        pid_item = self.table.item(row, 0)
        pid = (pid_item.text() if pid_item else "").strip()
        return pid if pid.isdigit() else None

    def kill_selected(self):
        pid = self._selected_pid()
        if not pid:
            QMessageBox.information(self, "Kill process", "Select a process row first (PID must be numeric).")
            return
        
        kill_cmd = f"kill -TERM {pid} >/dev/null 2>&1; " \
                   f"sleep 0.2; " \
                   f"kill -0 {pid} >/dev/null 2>&1 && kill -KILL {pid} >/dev/null 2>&1; true"

        args = ["-d", self.distro, "--", "bash", "-lc", kill_cmd]
        self.kill_proc.start("wsl.exe", args)

        QTimer.singleShot(350, self.refresh)

    def closeEvent(self, e):  
        self.timer.stop()
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
            self.proc.waitForFinished(500)
        if self.kill_proc.state() != QProcess.ProcessState.NotRunning:
            self.kill_proc.kill()
            self.kill_proc.waitForFinished(500)
        super().closeEvent(e)

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

        self.rows_widget = QWidget(self)
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)
        outer.addWidget(self.rows_widget)

        controls = QHBoxLayout()
        self.add_btn = QPushButton("➕ Add PTM", self)
        self.add_btn.clicked.connect(self._add_row)
        controls.addWidget(self.add_btn)
        controls.addStretch(1)
        outer.addLayout(controls)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.save_btn = QPushButton("Save", self)
        self.cancel_btn = QPushButton("Cancel", self)
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.save_btn)
        outer.addLayout(btns)

        if initial:
            for it in initial:
                self._add_row(label=it.get("label"), ccd=it.get("ccd"), pos=it.get("pos"))
        else:
            self._add_row()

    def _add_row(self, *, label: Optional[str] = None, ccd: Optional[str] = None, pos: Optional[str] = None):
        row = QHBoxLayout()

        dd = QComboBox(self)
        for lab, c in self.ptm_choices.items():
            dd.addItem(lab, c)

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
            for i, (ddd, ppp, rrr) in enumerate(list(self._rows)):
                if ddd is dd and ppp is pos_edit and rrr is rm:
                    self._rows.pop(i)
                    break
            dd.setParent(None); dd.deleteLater()
            pos_edit.setParent(None); pos_edit.deleteLater()
            rm.setParent(None); rm.deleteLater()
            self._rebuild_rows()

        rm.clicked.connect(_remove)

        row.addWidget(dd, 3)
        row.addWidget(pos_edit, 2)
        row.addWidget(rm, 0)

        self._rows.append((dd, pos_edit, rm))
        self._rebuild_rows()

    def _rebuild_rows(self):
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            if item.layout():
                pass

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

            if ccd in (None, "None") or label.lower() == "none":
                continue

            pos = pos_edit.text().strip()
            if not pos:
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
            def progress_hook(msg):  
                self.progress.emit(str(msg))

            json_builder._progress_hook = progress_hook  

            try:
                json_path = json_builder.build_input(
                    jobname=self.jobname,
                    proteins=self.proteins,
                    rna=self.rna,
                    dna=self.dna,
                    ligand=self.ligand,
                )
            except TypeError:
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

    def __init__(self, json_path: str, job_name: str):
        super().__init__()
        self.json_path = json_path
        self.job_name  = job_name

    def run(self):
        try:
            _ensure_backend_loaded()
            runner.run_af3(
                self.json_path,
                job_name=self.job_name,
                auto_analyze=False,   
                multi_seed=False,   
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

        out = (out or "").replace("\x00", "").strip()
        err = (err or "").replace("\x00", "").strip()

        return p.returncode, out, err


    def _detect_on_windows(self) -> dict[str, Any]:
        s: dict[str, Any] = {}
        self.log.emit("🔎 Detecting on Windows…")
        try:
            gui_dir = Path(__file__).resolve().parent
            s["gui_dir"] = str(gui_dir)
        except Exception:
            pass

        s["platform"] = "wsl" 

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

        rc, _, _ = self._run_cmd(["docker", "version"])
        if rc == 0:
            s["docker_bin"] = "docker"
            self.log.emit("✅ Docker available on Windows")
        else:
            distro = s.get("wsl_distro", self.current_distro)
            if distro:
                rc2, _, _ = self._run_cmd(["wsl.exe", "-d", str(distro), "--", "docker", "version"])
                if rc2 == 0:
                    s["docker_bin"] = "docker"
                    self.log.emit("✅ Docker available inside WSL")
                else:
                    self.log.emit("⚠️ Docker not detected (Windows or WSL).")

        distro = (s.get("wsl_distro", self.current_distro) or "").replace("\x00", "").strip()
        if distro:
            s.update(self._detect_inside_wsl(distro))
        s.setdefault("alphafold_docker_image", "alphafold3")

        return s

    def _parse_default_wsl_distro(self, text: str) -> str | None:
        text = (text or "").replace("\x00", "")

        for line in text.splitlines():
            line = line.replace("\x00", "").strip()
            m = re.match(r"^\*\s*([A-Za-z0-9_.-]+)\b", line)
            if m:
                return m.group(1)

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

        rc, out, _ = wsl("echo $HOME")
        if rc == 0 and out.startswith("/"):
            s["linux_home_root"] = out
            self.log.emit(f"✅ linux_home_root: {out}")

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

        af3 = s.get("af3_dir")
        if af3:
            rc, out, _ = wsl(f"test -d '{af3}/mmseqs_db' && echo '{af3}/mmseqs_db'")
            if rc == 0 and out:
                s.setdefault("msa", {})
                s["msa"]["db"] = out
                self.log.emit(f"✅ msa.db: {out}")

        return s

    def _detect_on_linux(self) -> dict[str, Any]:
        s: dict[str, Any] = {}
        self.log.emit("🔎 Detecting on Linux…")

        s["platform"] = "linux"
        s["linux_home_root"] = str(Path.home())
        try:
            s["gui_dir"] = str(Path(__file__).resolve().parent)
        except Exception:
            pass
        return s
    
class AnalysisWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        job_name: str,
        multi_seed: bool = False,
        skip_rosetta: bool = False,
        skip_rosetta_ligand: bool = False,
        constraints_file: str = "",
    ):
        super().__init__()
        self.job_name = job_name
        self.multi_seed = bool(multi_seed)
        self.skip_rosetta = bool(skip_rosetta)
        self.skip_rosetta_ligand = bool(skip_rosetta_ligand)
        self.constraints_file = (constraints_file or "").strip()

    def run(self):
        try:
            _ensure_backend_loaded()
            repo_root = Path(__file__).resolve().parents[2]
            exe = sys.executable

            cmd = [exe, "-m", "af3_pipeline.analysis.post_analysis", "--job", str(self.job_name)]
            if self.multi_seed:
                cmd.append("--multi_seed")
            if self.skip_rosetta:
                cmd.append("--skip_rosetta") 
            if self.skip_rosetta_ligand:
                cmd.append("--skip_rosetta_ligand")
            if self.constraints_file:
                cmd += ["--constraints", self.constraints_file]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["AF3_PIPELINE_CONFIG"] = os.environ.get("AF3_PIPELINE_CONFIG", "")

            self.log.emit(f"▶ Post-AF3 analysis\n$ {' '.join(map(str, cmd))}")

            with subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
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

try:
    import runs_features as _runs_features
    _runs_features.AnalysisWorker = AnalysisWorker
except Exception:
    pass


@dataclass
class MacroEntryRefs:
    root: QWidget
    dropdown: QComboBox
    name: QLineEdit
    seq: QPlainTextEdit

    ptm_button: QPushButton
    ptm_summary: QLabel
    ptms: list[dict[str, str]] = field(default_factory=list)  

    template: Optional[QLineEdit] = None
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
    frame.setObjectName("MacroEntryCard")
    frame.setFrameShape(QFrame.Shape.StyledPanel)

    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    frame.setMaximumHeight(190)

    outer = QVBoxLayout(frame)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(8)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)

    dd = QComboBox(frame)
    dd.addItem("Select saved…")
    dd.addItems(saved_names)
    dd.setMinimumWidth(220)
    dd.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    name = QLineEdit(frame)
    name.hide()
    name.setFixedSize(0, 0)
    name.setObjectName("hiddenNameField")

    template = None
    if kind == "protein":
        template = QLineEdit(frame)
        template.setPlaceholderText("PDB template (optional)")
        template.setMinimumWidth(220)
        template.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    ptm_btn = QToolButton(frame)
    ptm_btn.setText("PTMs…")
    ptm_btn.setAutoRaise(True)

    ptm_summary = QLabel("None", frame)
    ptm_summary.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    ptm_summary.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    ptm_summary.setMinimumWidth(60)

    del_btn = QToolButton(frame)
    del_btn.setAutoRaise(True)
    del_btn.setText("🗑")
    del_btn.setToolTip("Remove this entry")

    row.addWidget(dd, 3)
    if template is not None:
        row.addWidget(template, 3)

    row.addWidget(ptm_btn, 0)
    row.addWidget(ptm_summary, 0)
    row.addStretch(1)
    row.addWidget(del_btn, 0)

    outer.addLayout(row)

    seq = QPlainTextEdit(frame)
    seq.setPlaceholderText("Sequence…")
    seq.setTabChangesFocus(True)

    seq.setMinimumHeight(80)
    seq.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    outer.addWidget(seq)

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

    del_btn.clicked.connect(lambda _=False: on_delete(refs))
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
        i = j + 1
    return out


class SequenceMapCanvas(QWidget):
    """
    A lightweight ruler+highlight canvas. Parent panel provides:
      - sequence length
      - hits and active hit
      - selection region
      - scroll position + zoom
    """
    requestSeek = pyqtSignal(int)
    requestSelect = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setMouseTracking(True)

        self._seq_len = 0
        self._px_per_res = 6.0       
        self._offset_res = 0       
        self._hits: list[tuple[int, int]] = []     
        self._active_hit_idx: int = -1
        self._sel: tuple[int, int] | None = None    

        self._dragging = False
        self._drag_anchor_res: int | None = None

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

    def _res_at_x(self, x: int) -> int:
        res = int(self._offset_res + (x / self._px_per_res))
        return max(0, min(res, max(0, self._seq_len - 1)))

    def _x_at_res(self, res: int) -> float:
        return (res - self._offset_res) * self._px_per_res

    def visible_res_count(self) -> int:
        return int(self.width() / self._px_per_res) + 1

    def paintEvent(self, _evt):  
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()

        p.fillRect(QRect(0, 0, w, h), self.palette().base())

        if self._seq_len <= 0:
            p.setPen(self.palette().text().color())
            p.drawText(QRect(0, 0, w, h), int(Qt.AlignmentFlag.AlignCenter), "No sequence")
            return

        fm = QFontMetrics(p.font())
        top = 10
        baseline_y = h // 2
        tick_top = baseline_y - 12
        tick_bot = baseline_y + 12

        pen_base = QPen(self.palette().text().color())
        pen_base.setWidth(1)
        p.setPen(pen_base)
        p.drawLine(0, baseline_y, w, baseline_y)

        for idx, (s, ln) in enumerate(self._hits):
            if ln <= 0:
                continue
            x1 = self._x_at_res(s)
            x2 = self._x_at_res(s + ln)
            if x2 < 0 or x1 > w:
                continue
            rect = QRectF(max(0.0, x1), baseline_y - 16, min(float(w), x2) - max(0.0, x1), 32)
            if idx == self._active_hit_idx:
                p.fillRect(rect, self.palette().highlight())
            else:
                c = self.palette().highlight().color()
                c.setAlpha(90)
                p.fillRect(rect, c)

        if self._sel:
            s, ln = self._sel
            if ln > 0:
                x1 = self._x_at_res(s)
                x2 = self._x_at_res(s + ln)
                if not (x2 < 0 or x1 > w):
                    c = self.palette().alternateBase().color()
                    c.setAlpha(160)
                    p.fillRect(QRectF(max(0.0, x1), baseline_y - 18, min(float(w), x2) - max(0.0, x1), 36), c)

        first_res = self._offset_res
        last_res = min(self._seq_len - 1, self._offset_res + self.visible_res_count())

        start_tick = (first_res // 10) * 10
        for r in range(start_tick, last_res + 1, 10):
            x = int(self._x_at_res(r))
            if x < 0:
                continue
            p.drawLine(x, tick_top, x, tick_bot)

            if r % 50 == 0 and r != 0:
                label = str(r)
                tw = fm.horizontalAdvance(label)
                p.drawText(x - tw // 2, top + fm.ascent(), label)

        end_label = f"Len: {self._seq_len}"
        p.drawText(8, h - 8, end_label)

    def wheelEvent(self, e: QWheelEvent): 
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = e.angleDelta().y()
            if delta > 0:
                self.setZoom(self._px_per_res * 1.15)
            elif delta < 0:
                self.setZoom(self._px_per_res / 1.15)
            e.accept()
            return
        super().wheelEvent(e)

    def mousePressEvent(self, e): 
        if e.button() == Qt.MouseButton.LeftButton and self._seq_len > 0:
            self._dragging = True
            self._drag_anchor_res = self._res_at_x(e.position().x())
            self.requestSelect.emit(self._drag_anchor_res, 1)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):  
        if self._dragging and self._drag_anchor_res is not None:
            cur = self._res_at_x(e.position().x())
            a = self._drag_anchor_res
            start = min(a, cur)
            end = max(a, cur)
            self.requestSelect.emit(start, end - start + 1)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):  
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_anchor_res = None
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e): 
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

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._recompute_from_editor)

        self._editor.textChanged.connect(self._debounced_refresh)

        try:
            self._editor.selectionChanged.connect(self._sync_selection_from_editor) 
        except Exception:
            pass

        self._recompute_from_editor()

    def run_search(self):
        """
        Run search only when the user explicitly requests it.
        Important: do NOT move focus away from the search box.
        """
        self._recompute_hits()

        if self._active_hit >= 0 and self._hits:
            s, ln = self._hits[self._active_hit]

            doc_start, doc_end = _find_doc_range_for_norm_span(self._norm_to_doc, s, ln)
            cursor = self._editor.textCursor()
            cursor.setPosition(doc_start)
            cursor.setPosition(doc_end, cursor.MoveMode.KeepAnchor)
            self._editor.setTextCursor(cursor)

            self.canvas.setSelection((s, ln))
            self.center_on_residue(s)

        self.searchEdit.setFocus()


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

def _af3_dir_posix() -> str:
    # cfg stores af3_dir at top level (see _ensure_profile_first_run template)
    return str(PurePosixPath((cfg.get("af3_dir") or "").strip()))

def _constraints_root_posix() -> str:
    """
    Always derive constraints root as: <af3_dir>/constraints
    e.g. /home/olive/Repositories/alphafold/constraints
    """
    af3 = (cfg.get("af3_dir") or "").strip()
    if not af3:
        return ""
    return str(PurePosixPath(af3) / "constraints")

def _constraints_root_host_path() -> Path:
    """
    Host-visible path for the constraints root.
    - On Windows: \\wsl.localhost\<distro>\<posix_path>
    - On Linux/macOS: normal POSIX path
    """
    posix = _constraints_root_posix().strip()
    if not posix:
        return Path()

    if sys.platform.startswith("win"):
        distro = (cfg.get("wsl_distro") or "Ubuntu-22.04").strip()
        return Path(rf"\\wsl.localhost\{distro}\{posix.lstrip('/')}")
    return Path(posix)

def _constraints_root() -> Path | None:
    posix = _constraints_root_posix().strip()
    return Path(posix) if posix else None

def _posix_to_wsl_unc(posix_path: str) -> Path:
    """
    /home/olive/... -> \\wsl.localhost\\Ubuntu-22.04\\home\\olive\\...
    """
    distro = (cfg.get("wsl_distro") or "Ubuntu-22.04").strip()
    p = str(PurePosixPath(posix_path)).lstrip("/")
    return Path(rf"\\wsl.localhost\{distro}\{p}")

def _list_constraint_sets() -> list[str]:
    root = _constraints_root_host_path()
    if not root.exists():
        return []
    files = sorted(root.glob("*_constraints.cst"))
    return [f.name[:-len("_constraints.cst")] for f in files]
    return out

def _constraint_set_to_file(set_name: str) -> Optional[Path]:
    set_name = (set_name or "").strip()
    if not set_name:
        return None

    root = _constraints_root_host_path()
    if not root:
        return None

    p = root / f"{set_name}_constraints.cst"
    return p if p.exists() else None



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
        self.resize(1500, 1175)
        self.setMinimumSize(900, 600)

        # Profiles UI
        self._install_profiles_row()
        self._install_sequence_map_panels()

        self._install_wsl_proc_table_under_logs()
        self._install_config_help()

        self._ensure_profile_first_run()
        self._install_constraints_dropdown_ui()

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
        self._pending_json_path: str = ""
        self._pending_multi_seed: bool = False
        self._pending_skip_rosetta: bool = False
        self._pending_skip_rosetta_ligand: bool = False

        # ---- wire up ----
        self._connect_nav()
        self._connect_sequences_page()
        self._connect_ligands_page()
        self._connect_rosetta_config_page()
        self._connect_alphafold_page()
        self._connect_runs_page()
        self._connect_config_page()

        self._current_ligand_notes_name: str = ""
        self._install_ligand_notes_box()

        # ---- initial populate ----
        self._refresh_all_dropdowns()
        self._refresh_queue_view()
        self._refresh_runs_view()
        self._connect_compare_pdbs_ui()

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

        self._create_new_profile(force=True)

    def _create_new_profile(self, force: bool = False):
        dlg = NewProfileDialog(self)
        if force:
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
        if getattr(self, "_active_profile", "") == name:
            return
        self._activate_profile(name, run_autodetect_if_needed=False)

    def _clear_rosetta_config_ui(self):
        """
        Reset ONLY widgets that still exist in the trimmed .ui.
        """
        try:
            # Stage2 protocol defaults
            if hasattr(self, "rsNeighborDistSpin"):
                self.rsNeighborDistSpin.setValue(6.0)

            if hasattr(self, "rsNstructNoncovSpin"):
                self.rsNstructNoncovSpin.setValue(5)
            if hasattr(self, "rsNstructCovSpin"):
                self.rsNstructCovSpin.setValue(1)

            if hasattr(self, "rsLocalRbEnabledCheck"):
                self.rsLocalRbEnabledCheck.setChecked(True)
            if hasattr(self, "rsRbTranslateAngSpin"):
                self.rsRbTranslateAngSpin.setValue(1.5)
            if hasattr(self, "rsRbTranslateCyclesSpin"):
                self.rsRbTranslateCyclesSpin.setValue(25)
            if hasattr(self, "rsRbRotateDegSpin"):
                self.rsRbRotateDegSpin.setValue(15.0)
            if hasattr(self, "rsRbRotateCyclesSpin"):
                self.rsRbRotateCyclesSpin.setValue(50)
            if hasattr(self, "rsRbSlideTogetherCheck"):
                self.rsRbSlideTogetherCheck.setChecked(True)

            if hasattr(self, "rsHighresCyclesSpin"):
                self.rsHighresCyclesSpin.setValue(6)
            if hasattr(self, "rsHighresRepackSpin"):
                self.rsHighresRepackSpin.setValue(3)

            # Stage2 scoring/minimization controls
            if hasattr(self, "rsSfxnSoftInput"):
                self.rsSfxnSoftInput.setText("ligand_soft_rep")
            if hasattr(self, "rsSfxnHardInput"):
                self.rsSfxnHardInput.setText("ligand")
            if hasattr(self, "rsMinTypeDropdown") and self.rsMinTypeDropdown.count() > 0:
                self.rsMinTypeDropdown.setCurrentIndex(0)
            if hasattr(self, "rsPackCyclesSpin"):
                self.rsPackCyclesSpin.setValue(4)
            if hasattr(self, "rsRampScorefxnCheck"):
                self.rsRampScorefxnCheck.setChecked(True)

            # Constraints generator group (genRosettaParamsGroup)
            if hasattr(self, "grpRefPdbInput"):
                self.grpRefPdbInput.clear()
            if hasattr(self, "grpLigandNameInput"):
                self.grpLigandNameInput.clear()
            if hasattr(self, "grpRefLigandResnameInput"):
                self.grpRefLigandResnameInput.setText("LIG")
            if hasattr(self, "grpChainInput"):
                self.grpChainInput.setText("")
            if hasattr(self, "grpResnumSpin"):
                self.grpResnumSpin.setValue(1)
            if hasattr(self, "grpRestypeInput"):
                self.grpRestypeInput.setText("")
            if hasattr(self, "grpProteinAtomInput"):
                self.grpProteinAtomInput.clear()
            if hasattr(self, "grpLigandAtomTrueInput"):
                self.grpLigandAtomTrueInput.clear()
            if hasattr(self, "grpSdDistSpin"):
                self.grpSdDistSpin.setValue(0.5)
            if hasattr(self, "grpSdAngSpin"):
                self.grpSdAngSpin.setValue(5.0)
            if hasattr(self, "grpSdDihSpin"):
                self.grpSdDihSpin.setValue(10.0)
            if hasattr(self, "grpOutInput"):
                self.grpOutInput.clear()
            if hasattr(self, "grpStatusLabel"):
                self.grpStatusLabel.setText("Status: idle")

            if hasattr(self, "rosettaDictsStatusLabel"):
                self.rosettaDictsStatusLabel.setText("Status: idle")

        except Exception as e:
            try:
                self.log(f"⚠️ _clear_rosetta_config_ui failed: {e}")
            except Exception:
                pass

    def _activate_profile(self, name: str, *, run_autodetect_if_needed: bool):
        name = _safe_profile_name(name)
        if not name:
            return

        if getattr(self, "_activating_profile", False):
            return
        if getattr(self, "_active_profile", "") == name and hasattr(self, "_profile_root"):
            if not run_autodetect_if_needed:
                return

        self._activating_profile = True
        self._active_profile = name
        _save_current_profile_name(name)
        try:
            root = _set_active_profile_paths(name)
            self._profile_root = root
            self.log(f"👤 Active profile: {name}")
            self.log(f"📁 Profile root: {root}")

            profile_cfg = root / "config.yaml"
            os.environ["AF3_PIPELINE_CONFIG"] = str(profile_cfg)
            os.environ["AF3_PIPELINE_CACHE_ROOT"] = str(root / "cache")
            os.environ["AF3_PIPELINE_MSA_CACHE"] = str(_shared_dir("msa"))
            os.environ["AF3_PIPELINE_TEMPLATE_CACHE"] = str(_shared_dir("templates"))

            _ensure_backend_loaded()
            self._bootstrap_config_yaml_from_template()

            try:
                cfg.reload(profile_cfg)
            except Exception as e:
                self.log(f"⚠️ cfg.reload failed: {e}")

            self.sequence_cache = _read_json(SEQUENCE_CACHE_FILE, {})
            self.ligand_cache   = _read_json(LIGAND_CACHE_FILE, {})
            self.runs_history   = _read_json(RUNS_HISTORY_FILE, [])

            self.rosetta_dicts_data = {}
            self.rosetta_dicts_path = None
            self._rs_selected_set_name = ""
            self._clear_rosetta_config_ui()

            self._refresh_all_dropdowns()
            self._refresh_queue_view()
            self._refresh_runs_view()

            if hasattr(self, "ligandNotesEdit"):
                self._load_ligand_notes_for_name(self.ligandDropdown.currentText().strip())

            if run_autodetect_if_needed or self._config_needs_setup():
                self.pagesStack.setCurrentIndex(0) 
                self.navList.setCurrentRow(0)
                self.log("🧭 Running per-profile auto-detect…")
                self._start_autodetect()
        finally:
            self._activating_profile = False

    def _install_wsl_proc_table_under_logs(self):
        if hasattr(self, "wslProcTable"):
            return

        distro = "Ubuntu-22.04"
        try:
            d = (self.cfgWslDistro.text() or "").strip()
            if d:
                distro = d
        except Exception:
            pass

        self.wslProcTable = WSLProcessTable(self, distro=distro)

        try:
            idx = self.sidebarLayout.indexOf(self.logOutput)
            if idx < 0:
                self.sidebarLayout.addWidget(self.wslProcTable)
            else:
                self.sidebarLayout.insertWidget(idx + 1, self.wslProcTable)
        except Exception as e:
            self.log(f"⚠️ Failed to insert WSL proc table: {e}")
            self.sidebarLayout.addWidget(self.wslProcTable)

    def _connect_compare_pdbs_ui(self) -> None:
        if hasattr(self, "browseRefPDBPathButton"):
            self.browseRefPDBPathButton.clicked.connect(self._browse_ref_pdb_for_compare)
        if hasattr(self, "browseQueryPDBPathButton"):
            self.browseQueryPDBPathButton.clicked.connect(self._browse_query_pdb_for_compare)
        if hasattr(self, "runComparisonButton"):
            self.runComparisonButton.clicked.connect(self._run_compare_structs_clicked)

    def _browse_ref_pdb_for_compare(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference PDB", "", "PDB files (*.pdb *.ent);;All files (*)"
        )
        if path:
            self.refPDBPath.setText(path)

    def _browse_query_pdb_for_compare(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select query PDB", "", "PDB files (*.pdb *.ent);;All files (*)"
        )
        if path:
            self.queryPDBPath.setText(path)

    def _run_compare_structs_clicked(self):
        pdb1 = (self.refPDBPath.text() or "").strip()
        lig1 = (self.refLigName.text() or "").strip()
        pdb2 = (self.queryPDBPath.text() or "").strip()
        lig2 = (self.queryLigName.text() or "").strip()

        if not pdb1 or not Path(pdb1).exists():
            _msg_warn(self, "Missing input", "Reference PDB path is missing or invalid.")
            return
        if not pdb2 or not Path(pdb2).exists():
            _msg_warn(self, "Missing input", "Query PDB path is missing or invalid.")
            return
        if not lig1:
            _msg_warn(self, "Missing input", "Reference ligand name is required (e.g. SF4).")
            return
        if not lig2:
            _msg_warn(self, "Missing input", "Query ligand name is required (e.g. LIG).")
            return

        out = (self.comparisonOutPath.text() or "").strip()
        if not out:
            # default: next to query pdb
            out = str(Path(pdb2).with_suffix("").as_posix() + "_compare.json")
            self.comparisonOutPath.setText(out)

        # Optional: you can add a cutoff widget later; for now fixed 5 Å
        cutoff = 5.0

        self.runComparisonButton.setEnabled(False)
        self.runDetailsText.append(f"🧪 Running compare_structs:\n  ref={pdb1} ({lig1})\n  query={pdb2} ({lig2})\n  out={out}")

        self._compare_thread = CompareStructsThread(pdb1, lig1, pdb2, lig2, cutoff, out)
        self._compare_thread.finished_ok.connect(self._on_compare_finished_ok)
        self._compare_thread.finished_err.connect(self._on_compare_finished_err)
        self._compare_thread.start()

    def _on_compare_finished_ok(self, res: dict, out_path: str):
        self.runComparisonButton.setEnabled(True)
        # Show a short summary in Details
        try:
            bc = res.get("best_chain_pair", {})
            lig = res.get("ligand", {})
            pocket = res.get("pocket", {})
            msg = (
                f"✅ Compare complete\n"
                f"Best chain pair: pdb1:{bc.get('pdb1_chain')} <-> pdb2:{bc.get('pdb2_chain')}\n"
                f"Backbone RMSD: {bc.get('final_alignment_rmsd_backbone'):.3f} Å\n"
                f"Ligand RMSD: {lig.get('rmsd_A')}\n"
                f"Ligand centroid distance: {lig.get('ligand_centroid_distance')}\n"
                f"Pocket CA RMSD: {pocket.get('ca_rmsd_A')} (n={pocket.get('mapped_residue_count_used_for_rmsd')})\n"
                f"Wrote: {out_path}\n"
            )
        except Exception:
            msg = f"✅ Compare complete. Wrote: {out_path}\n"
        self.runDetailsText.append(msg)

    def _on_compare_finished_err(self, err: str):
        self.runComparisonButton.setEnabled(True)
        _msg_err(self, "Compare failed", err)
        self.runDetailsText.append(f"❌ Compare failed: {err}")

    def _install_constraints_dropdown_ui(self) -> None:
        """
        Create a 'Custom constraints set' label + dropdown directly under the
        covalent constraints generator section within genRosettaParamsGroup.
        Creates the dropdown widget (constraintsSetNextDropdown) at runtime.
        """

        group = getattr(self, "genRosettaParamsGroup", None)
        if group is None:
            print("WARNING: genRosettaParamsGroup not found; cannot add constraints dropdown.")
            return

        # Avoid duplicates if called twice
        if getattr(self, "constraintsSetNextDropdown", None) is not None:
            return

        lay = group.layout()
        if lay is None:
            # genRosettaParamsGroup should already have a layout in the .ui
            # but if not, create one so we have somewhere to insert widgets.
            from PyQt6.QtWidgets import QVBoxLayout
            lay = QVBoxLayout(group)
            group.setLayout(lay)

        # --- Build row ---
        row = QHBoxLayout()

        lbl = QLabel("Custom constraints set:")
        lbl.setObjectName("constraintsSetNextLabel")
        lbl.setToolTip(
            "Optional: choose a saved constraints folder.\n"
            "Auto = use the default covalent constraints generated from the AF model."
        )

        dd = QComboBox()
        dd.setObjectName("constraintsSetNextDropdown")
        dd.setMinimumWidth(280)
        dd.setToolTip(lbl.toolTip())

        # Store on self so existing code can access it
        self.constraintsSetNextDropdown = dd

        row.addWidget(lbl)
        row.addWidget(dd, stretch=1)

        # --- Insert right under the generator section ---
        #
        # We need an anchor widget that is part of the generator subsection.
        # Common patterns: a button like "genRosettaParamsBtn" or an output/path field.
        #
        # Try several likely anchor names; we insert *after* the first one we find.
        anchor_names = [
            "genRosettaParamsBtn",
            "genRosettaParamsButton",
            "genRosettaParamsOut",
            "genRosettaParamsOutEdit",
            "genRosettaParamsFolderEdit",
            "generateRosettaParamsBtn",
        ]

        inserted = False
        for nm in anchor_names:
            anchor = getattr(self, nm, None)
            if anchor is None:
                continue
            idx = lay.indexOf(anchor)
            if idx >= 0:
                # insert after anchor
                lay.insertLayout(idx + 1, row)
                inserted = True
                break

        if not inserted:
            # Fallback: append to end of group
            lay.addLayout(row)

        self._refresh_constraints_dropdown()
        dd.currentIndexChanged.connect(self._on_constraints_dropdown_changed_any)
        self._on_constraints_dropdown_changed_any("constraintsSetNextDropdown")
        with QSignalBlocker(dd):
            dd.setCurrentIndex(0)   # Auto row
        self._on_constraints_dropdown_changed_any("constraintsSetNextDropdown")

    def _refresh_constraints_dropdown(self) -> None:
        dds = []
        for nm in ("constraintsSetNextDropdown", "constraintsDropdown", "constraintsDropdown_2"):
            dd = getattr(self, nm, None)
            if dd is not None:
                dds.append(dd)
        if not dds:
            return

        sets = _list_constraint_sets()
        cur = (cfg.get("rosetta.constraints_set") or "").strip()

        for dd in dds:
            with QSignalBlocker(dd):
                dd.clear()
                dd.addItem("Auto (no custom constraints)", "")  # data=""
                for s in sets:
                    dd.addItem(s, s)

                # Select current
                if not cur or cur.lower() == "auto":
                    dd.setCurrentIndex(0)
                else:
                    idx = dd.findData(cur)
                    dd.setCurrentIndex(idx if idx >= 0 else 0)

    def _on_constraints_dropdown_changed_any(self, *args) -> None:
        """
        Universal handler for ALL constraints dropdowns.

        Supported call patterns:
        - _on_constraints_dropdown_changed_any()                     # default to constraintsSetNextDropdown
        - _on_constraints_dropdown_changed_any("constraintsDropdown")# explicit widget name
        - Qt signal: currentIndexChanged(int) -> args=(index,)       # ignore index, use sender()
        """

        widget_name = None

        # If first arg is a string, treat it as widget_name
        if args and isinstance(args[0], str):
            widget_name = args[0]

        # If called from a signal, try sender()
        dd = None
        if widget_name:
            dd = getattr(self, widget_name, None)
        if dd is None:
            try:
                dd = self.sender()
            except Exception:
                dd = None

        # Final fallback: use the Rosetta page dropdown if it exists
        if dd is None:
            dd = getattr(self, "constraintsSetNextDropdown", None)

        if dd is None:
            return  # nothing to do

        set_name = (dd.currentData() or "").strip()

        # Normalize: treat "auto" or "" as no custom constraints
        if not set_name or set_name.lower() == "auto":
            cfg.set("rosetta.constraints_set", "auto")
            cfg.set("rosetta.constraints_file", "")
        else:
            cfg.set("rosetta.constraints_set", set_name)
            cst_path = _constraint_set_to_file(set_name)
            cfg.set("rosetta.constraints_file", str(cst_path) if cst_path else "")

        cfg.save()

        # Keep all dropdowns in sync
        self._refresh_constraints_dropdown()


    # =======================
    # 📝 Ligand Notes (cached with ligand)
    # =======================
    def _install_ligand_notes_box(self):
        """
        Add a per-ligand notes editor underneath the 2D preview on the Ligands page.
        No .ui changes required.
        Saves notes to: LIGAND_CACHE/<hash>/NOTES.txt
        """
        if hasattr(self, "ligandNotesEdit"):
            return

        self.ligandNotesLabel = QLabel("Notes", self)
        self.ligandNotesEdit = QPlainTextEdit(self)
        self.ligandNotesEdit.setPlaceholderText("Notes for this ligand… (autosaved)")
        self.ligandNotesEdit.setMinimumHeight(140)

        try:
            self.ligandPreviewLayout.addWidget(self.ligandNotesLabel)
            self.ligandPreviewLayout.addWidget(self.ligandNotesEdit)
        except Exception as e:
            self.log(f"⚠️ Failed to add ligand notes box to layout: {e}")
            return

        self._ligand_notes_save_timer = QTimer(self)
        self._ligand_notes_save_timer.setSingleShot(True)
        self._ligand_notes_save_timer.setInterval(350)
        self._ligand_notes_save_timer.timeout.connect(self._save_current_ligand_notes_silent)

        self.ligandNotesEdit.textChanged.connect(lambda: self._ligand_notes_save_timer.start())
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
        put_if_missing("chimera_path", "")
        put_if_missing("pymol_path", "")
        put_if_missing("models_dir", f"{repo_guess}/models")

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
        lig.setdefault("basename", "LIG")
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

    def _install_constraints_dropdown_existing(self, widget_name: str) -> None:
        dd = getattr(self, widget_name, None)
        if dd is None:
            return

        # Avoid double-connect
        key = f"_constraints_dd_hooked_{widget_name}"
        if getattr(self, key, False):
            return
        setattr(self, key, True)

        # Populate + sync
        self._refresh_constraints_dropdown()

        # When changed, update cfg and sync all dropdowns
        dd.currentIndexChanged.connect(lambda *_: self._on_constraints_dropdown_changed_any(widget_name))

    def _install_grp_ligand_dropdown(self) -> None:
        """
        Replace grpLigandCacheInput (path) with a dropdown selecting a saved ligand
        from self.ligand_cache (same as Ligands/AlphaFold pages).
        Creates:
        - self.grpLigandDropdown (QComboBox)
        - self.grpLigandKeyLabel (optional small label)
        """

        # Only install once
        if hasattr(self, "grpLigandDropdown"):
            return

        # Must exist in UI
        if not self._has_grp_widgets():
            return

        # Anchor: insert next to/under the existing cache input row
        cache_input = getattr(self, "grpLigandCacheInput", None)
        cache_browse = getattr(self, "grpLigandCacheBrowseBtn", None)
        parent = cache_input.parentWidget() if cache_input else None
        lay = parent.layout() if parent else None
        if lay is None:
            # fallback: try group container
            group = getattr(self, "genRosettaParamsGroup", None)
            lay = group.layout() if group else None
        if lay is None:
            print("WARNING: Could not find layout to insert grpLigandDropdown.")
            return

        # Hide old widgets (keep them around so nothing else breaks)
        try:
            cache_input.setVisible(False)
        except Exception:
            pass
        try:
            if cache_browse:
                cache_browse.setVisible(False)
        except Exception:
            pass

        # Create row widget (label + combo)
        roww = QWidget(parent or self)
        row = QHBoxLayout(roww)
        row.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Ligand (saved):")
        lbl.setObjectName("grpLigandDropdownLabel")
        lbl.setToolTip("Choose one of your saved ligands (from the Ligands page).")

        dd = QComboBox()
        dd.setObjectName("grpLigandDropdown")
        dd.setMinimumWidth(260)
        dd.setToolTip(lbl.toolTip())

        row.addWidget(lbl)
        row.addWidget(dd, stretch=1)

        # Store on self
        self.grpLigandDropdown = dd

         # Insert/replace where the old cache input was
        inserted = False

        # ✅ Grid: REPLACE the existing widget in-place (prevents overlap stealing clicks)
        if isinstance(lay, QGridLayout):
            try:
                lay.replaceWidget(cache_input, roww)
                inserted = True
            except Exception:
                inserted = False

        # ✅ Form layout: replace the row cleanly
        elif isinstance(lay, QFormLayout):
            try:
                r = lay.getWidgetPosition(cache_input)[0]
                # Hide the old form label if present
                lab = lay.labelForField(cache_input)
                if lab:
                    lab.setVisible(False)
                lay.removeWidget(cache_input)
                cache_input.setVisible(False)
                if cache_browse:
                    cache_browse.setVisible(False)
                lay.insertRow(r, lbl, dd)
                inserted = True
            except Exception:
                inserted = False

        # ✅ Fallback: linear layouts
        if not inserted:
            idx = lay.indexOf(cache_input)
            if idx >= 0 and hasattr(lay, "insertWidget"):
                lay.insertWidget(idx, roww)
                inserted = True

        if not inserted:
            lay.addWidget(roww)

        # Populate now + whenever dropdown changes
        self._refresh_grp_ligand_dropdown()
        dd.currentIndexChanged.connect(self._on_grp_ligand_selected)

    def _refresh_grp_ligand_dropdown(self) -> None:
        dd = getattr(self, "grpLigandDropdown", None)
        if dd is None:
            return

        # self.ligand_cache shape: {name: {"smiles":..., "hash":..., "path":...}, ...}
        ligs = self.ligand_cache if isinstance(getattr(self, "ligand_cache", {}), dict) else {}

        with QSignalBlocker(dd):
            dd.clear()
            dd.addItem("Select…", "")  # data="" means none selected

            for name in sorted(ligs.keys(), key=lambda s: s.lower()):
                meta = ligs.get(name, {}) if isinstance(ligs.get(name, {}), dict) else {}
                lig_hash = str(meta.get("hash", "") or "").strip()
                # store hash as item data; display name as text
                dd.addItem(name, lig_hash)

            # Try to restore last selection from the generator name field if it matches a ligand
            cur_name = (self.grpLigandNameInput.text() or "").strip()
            if cur_name:
                idx = dd.findText(cur_name)
                if idx >= 0:
                    dd.setCurrentIndex(idx)

    def _on_grp_ligand_selected(self) -> None:
        """
        Keep grpLigandNameInput synced to the dropdown name (optional).
        """
        dd = getattr(self, "grpLigandDropdown", None)
        if dd is None:
            return

        name = (dd.currentText() or "").strip()
        lig_hash = (dd.currentData() or "").strip()

        if not lig_hash:
            return

        # Helpful: set ligand-name input automatically (used as filename hint, etc.)
        try:
            self.grpLigandNameInput.setText(name)
        except Exception:
            pass

    def _init_rosetta_config_state(self):
        """
        Holds the loaded YAML dict (editable). UI no longer contains:
        - rosettaDictsPathInput / Browse / Load / Save
        - covalent_patches table
        - constraints sets/items tables, raw XML editor, weight controls

        So we only keep an in-memory dict + optional path, and drive the Stage2 widgets.
        """
        if not hasattr(self, "rosetta_dicts_data") or not isinstance(getattr(self, "rosetta_dicts_data", None), dict):
            self.rosetta_dicts_data = {}

        # Optional: if other code sets this, we respect it. Otherwise it may remain None.
        if not hasattr(self, "rosetta_dicts_path"):
            self.rosetta_dicts_path = None

        # Make sure Stage2 subtree exists
        if "rosetta_scripts_stage" not in self.rosetta_dicts_data or not isinstance(self.rosetta_dicts_data.get("rosetta_scripts_stage"), dict):
            self.rosetta_dicts_data["rosetta_scripts_stage"] = {}

    def _connect_rosetta_config_page(self):
        # Ensure state exists
        if not hasattr(self, "rosetta_dicts_data"):
            self._init_rosetta_config_state()

        # --- Constraints generator group wiring (still exists) ---
        if getattr(self, "_has_grp_widgets", None) and self._has_grp_widgets():
            if hasattr(self, "grpRefPdbBrowseBtn"):
                self.grpRefPdbBrowseBtn.clicked.connect(self._browse_grp_ref_pdb)
            if hasattr(self, "grpShowCmdBtn"):
                self.grpShowCmdBtn.clicked.connect(self._grp_show_command)
            if hasattr(self, "grpRunBtn"):
                self.grpRunBtn.clicked.connect(self._grp_run_constraints_generator)

            # If you have this helper already, keep using it (it populates grpLigandDropdown)
            if hasattr(self, "_install_grp_ligand_dropdown"):
                self._install_grp_ligand_dropdown()

        # --- Optional: Stage2 UI changes update in-memory dict immediately ---
        # (No YAML save button exists anymore; this just keeps self.rosetta_dicts_data in sync.)
        def _stage2_changed(*_args):
            try:
                if isinstance(self.rosetta_dicts_data, dict):
                    self.rosetta_dicts_data["rosetta_scripts_stage"] = self._rs_read_stage2_from_ui()
            except Exception:
                pass

        for name in [
            "rsNeighborDistSpin",
            "rsNstructNoncovSpin",
            "rsNstructCovSpin",
            "rsLocalRbEnabledCheck",
            "rsRbTranslateAngSpin",
            "rsRbTranslateCyclesSpin",
            "rsRbRotateDegSpin",
            "rsRbRotateCyclesSpin",
            "rsRbSlideTogetherCheck",
            "rsHighresCyclesSpin",
            "rsHighresRepackSpin",
            "rsSfxnSoftInput",
            "rsSfxnHardInput",
            "rsMinTypeDropdown",
            "rsPackCyclesSpin",
            "rsRampScorefxnCheck",
        ]:
            w = getattr(self, name, None)
            if w is None:
                continue

            # Connect the most likely signal type
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(_stage2_changed)
            elif hasattr(w, "toggled"):
                w.toggled.connect(_stage2_changed)
            elif hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(_stage2_changed)
            elif hasattr(w, "editingFinished"):
                w.editingFinished.connect(_stage2_changed)
        
    def _load_rosetta_dicts_into_ui(self):
        """
        Safe: loads dicts from self.rosetta_dicts_path if it exists,
        and fills only the widgets that still exist.
        """
        try:
            p = getattr(self, "rosetta_dicts_path", None)
            if not p:
                # nothing to load; just fill from current in-memory dict
                self._rs_fill_stage2_from_yaml()
                if hasattr(self, "rosettaDictsStatusLabel"):
                    self.rosettaDictsStatusLabel.setText("Status: loaded (memory)")
                return

            p = Path(p)
            if not p.exists():
                _msg_warn(self, "Not Found", f"rosetta_dicts.yaml not found:\n{p}")
                return

            from af3_pipeline.rosetta.rosetta_config import load_rosetta_dicts
            dicts = load_rosetta_dicts(p)
            self.rosetta_dicts_data = dict(dicts.data or {})
            self.rosetta_dicts_path = Path(dicts.path)

            self._rs_fill_stage2_from_yaml()

            if hasattr(self, "rosettaDictsStatusLabel"):
                self.rosettaDictsStatusLabel.setText(f"Status: loaded {self.rosetta_dicts_path}")
            self.log(f"✅ Loaded rosetta_dicts.yaml → {self.rosetta_dicts_path}")

        except Exception as e:
            _msg_err(self, "Load failed", f"{type(e).__name__}: {e}")



    def _rs_get_stage2(self) -> dict:
        d = self.rosetta_dicts_data.get("rosetta_scripts_stage", {})
        return d if isinstance(d, dict) else {}

    def _rs_fill_stage2_from_yaml(self):
        st = self._rs_get_stage2()

        # protocol
        proto = st.get("protocol", {}) if isinstance(st.get("protocol", {}), dict) else {}

        if hasattr(self, "rsNeighborDistSpin"):
            self.rsNeighborDistSpin.setValue(float(proto.get("neighbor_dist", 6.0)))

        nstruct = proto.get("nstruct", {}) if isinstance(proto.get("nstruct", {}), dict) else {}
        if hasattr(self, "rsNstructNoncovSpin"):
            self.rsNstructNoncovSpin.setValue(int(nstruct.get("noncovalent", 5)))
        if hasattr(self, "rsNstructCovSpin"):
            self.rsNstructCovSpin.setValue(int(nstruct.get("covalent", 1)))

        local_rb = proto.get("local_rb", {}) if isinstance(proto.get("local_rb", {}), dict) else {}
        if hasattr(self, "rsLocalRbEnabledCheck"):
            self.rsLocalRbEnabledCheck.setChecked(bool(local_rb.get("enabled", True)))

        tr = local_rb.get("translate", {}) if isinstance(local_rb.get("translate", {}), dict) else {}
        rot = local_rb.get("rotate", {}) if isinstance(local_rb.get("rotate", {}), dict) else {}
        if hasattr(self, "rsRbTranslateAngSpin"):
            self.rsRbTranslateAngSpin.setValue(float(tr.get("angstroms", 1.5)))
        if hasattr(self, "rsRbTranslateCyclesSpin"):
            self.rsRbTranslateCyclesSpin.setValue(int(tr.get("cycles", 25)))
        if hasattr(self, "rsRbRotateDegSpin"):
            self.rsRbRotateDegSpin.setValue(float(rot.get("degrees", 15)))
        if hasattr(self, "rsRbRotateCyclesSpin"):
            self.rsRbRotateCyclesSpin.setValue(int(rot.get("cycles", 50)))
        if hasattr(self, "rsRbSlideTogetherCheck"):
            self.rsRbSlideTogetherCheck.setChecked(bool(local_rb.get("slide_together", True)))

        highres = proto.get("highres", {}) if isinstance(proto.get("highres", {}), dict) else {}
        if hasattr(self, "rsHighresCyclesSpin"):
            self.rsHighresCyclesSpin.setValue(int(highres.get("cycles", 6)))
        if hasattr(self, "rsHighresRepackSpin"):
            self.rsHighresRepackSpin.setValue(int(highres.get("repack_every_nth", 3)))

        # scorefunctions
        sfx = st.get("scorefunctions", {}) if isinstance(st.get("scorefunctions", {}), dict) else {}
        if hasattr(self, "rsSfxnSoftInput"):
            self.rsSfxnSoftInput.setText(str(sfx.get("soft", "ligand_soft_rep")))
        if hasattr(self, "rsSfxnHardInput"):
            self.rsSfxnHardInput.setText(str(sfx.get("hard", sfx.get("hires", "ligand"))))

        # simple knobs (these are now top-level fields in your trimmed UI)
        if hasattr(self, "rsMinTypeDropdown"):
            # try to select by text if present
            min_type = str(st.get("min_type", "dfpmin")).strip()
            idx = self.rsMinTypeDropdown.findText(min_type)
            if idx >= 0:
                self.rsMinTypeDropdown.setCurrentIndex(idx)

        if hasattr(self, "rsPackCyclesSpin"):
            self.rsPackCyclesSpin.setValue(int(st.get("pack_cycles", 4)))

        if hasattr(self, "rsRampScorefxnCheck"):
            self.rsRampScorefxnCheck.setChecked(bool(st.get("ramp_scorefxn", True)))

    def _save_rosetta_dicts_from_ui(self):
        """
        Safe: writes stage2 subtree back into self.rosetta_dicts_data.
        If self.rosetta_dicts_path is set, writes YAML there; otherwise memory-only.
        """
        try:
            y = self.rosetta_dicts_data if isinstance(self.rosetta_dicts_data, dict) else {}
            self.rosetta_dicts_data = y

            y["rosetta_scripts_stage"] = self._rs_read_stage2_from_ui()

            p = getattr(self, "rosetta_dicts_path", None)
            if not p:
                if hasattr(self, "rosettaDictsStatusLabel"):
                    self.rosettaDictsStatusLabel.setText("Status: saved (memory)")
                return

            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)

            try:
                import yaml  # type: ignore
                p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
            except ModuleNotFoundError:
                from ruamel.yaml import YAML  # type: ignore
                yy = YAML()
                yy.default_flow_style = False
                with p.open("w", encoding="utf-8") as f:
                    yy.dump(y, f)

            if hasattr(self, "rosettaDictsStatusLabel"):
                self.rosettaDictsStatusLabel.setText(f"Status: saved {p}")
            self.log(f"💾 Saved rosetta_dicts.yaml → {p}")

        except Exception as e:
            _msg_err(self, "Save failed", f"{type(e).__name__}: {e}")

    def _csv_to_list(self, txt: str) -> list[str]:
        parts = [p.strip() for p in (txt or "").split(",")]
        return [p for p in parts if p]

    def _rs_read_stage2_from_ui(self) -> dict:
        """
        Read ONLY fields present in the trimmed .ui.
        (No constraints sets, no weights, no raw XML editor, no packing_flags input.)
        """
        st: dict[str, Any] = {}

        # protocol
        proto: dict[str, Any] = {}
        proto["neighbor_dist"] = float(self.rsNeighborDistSpin.value()) if hasattr(self, "rsNeighborDistSpin") else 6.0
        proto["nstruct"] = {
            "noncovalent": int(self.rsNstructNoncovSpin.value()) if hasattr(self, "rsNstructNoncovSpin") else 5,
            "covalent": int(self.rsNstructCovSpin.value()) if hasattr(self, "rsNstructCovSpin") else 1,
        }

        local_rb: dict[str, Any] = {
            "enabled": bool(self.rsLocalRbEnabledCheck.isChecked()) if hasattr(self, "rsLocalRbEnabledCheck") else True,
            "translate": {
                "angstroms": float(self.rsRbTranslateAngSpin.value()) if hasattr(self, "rsRbTranslateAngSpin") else 1.5,
                "cycles": int(self.rsRbTranslateCyclesSpin.value()) if hasattr(self, "rsRbTranslateCyclesSpin") else 25,
            },
            "rotate": {
                "degrees": float(self.rsRbRotateDegSpin.value()) if hasattr(self, "rsRbRotateDegSpin") else 15.0,
                "cycles": int(self.rsRbRotateCyclesSpin.value()) if hasattr(self, "rsRbRotateCyclesSpin") else 50,
            },
            "slide_together": bool(self.rsRbSlideTogetherCheck.isChecked()) if hasattr(self, "rsRbSlideTogetherCheck") else True,
        }
        proto["local_rb"] = local_rb

        proto["highres"] = {
            "cycles": int(self.rsHighresCyclesSpin.value()) if hasattr(self, "rsHighresCyclesSpin") else 6,
            "repack_every_nth": int(self.rsHighresRepackSpin.value()) if hasattr(self, "rsHighresRepackSpin") else 3,
        }

        st["protocol"] = proto

        # scorefunctions
        st["scorefunctions"] = {
            "soft": (self.rsSfxnSoftInput.text() or "ligand_soft_rep").strip() if hasattr(self, "rsSfxnSoftInput") else "ligand_soft_rep",
            "hard": (self.rsSfxnHardInput.text() or "ligand").strip() if hasattr(self, "rsSfxnHardInput") else "ligand",
        }

        # remaining single controls
        if hasattr(self, "rsMinTypeDropdown"):
            st["min_type"] = (self.rsMinTypeDropdown.currentText() or "dfpmin").strip()
        st["pack_cycles"] = int(self.rsPackCyclesSpin.value()) if hasattr(self, "rsPackCyclesSpin") else 4
        st["ramp_scorefxn"] = bool(self.rsRampScorefxnCheck.isChecked()) if hasattr(self, "rsRampScorefxnCheck") else True

        return st

    def _has_grp_widgets(self) -> bool:
        """Return True if the covalent constraints generator widgets exist in the loaded .ui."""
        return all(hasattr(self, n) for n in (
            "grpRefPdbInput",
            "grpRefPdbBrowseBtn",
            "grpLigandDropdown",
            "grpLigandNameInput",
            "grpLigandNameInput",
            "grpRefLigResnameInput",
            "grpChainInput",
            "grpResnumSpin",
            "grpRestypeInput",
            "grpProteinAtomInput",
            "grpLigandAtomTrueInput",
            "grpSdDistSpin",
            "grpSdAngSpin",
            "grpSdDihSpin",
            "grpOutInput",
            "grpShowCmdBtn",
            "grpRunBtn",
            "grpStatusLabel",
        ))


    def _browse_grp_ref_pdb(self):
        start = (self.grpRefPdbInput.text() or "").strip()
        start_dir = str(Path(start).expanduser().parent) if start else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference PDB (protein + ligand)",
            start_dir,
            "PDB files (*.pdb *.ent);;All files (*)",
        )
        if path:
            self.grpRefPdbInput.setText(path)


    def _resolve_constraints_generator_script(self) -> Path:
        """
        Best-effort locate the generator script.
        Adjust candidates if your file is named differently.
        """
        here = Path(__file__).resolve().parent
        apps = Path(here.resolve().parent)
        repo = Path(apps.resolve().parent)

        candidates = [
            repo / "af3_pipeline" / "analysis" / "generate_rosetta_constraints.py",
        ]
        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            "Could not find covalent constraints generator script.\n"
            "Tried:\n  " + "\n  ".join(str(p) for p in candidates)
        )


    def _build_grp_constraints_cmd(self) -> list[str]:
        """
        Build the exact CLI argv for your argparse main() (no shell).
        """
        ref_pdb = (self.grpRefPdbInput.text() or "").strip()

        dd = getattr(self, "grpLigandDropdown", None)
        lig_key = (dd.currentData() or "").strip() if dd else ""

        lig_name = (self.grpLigandNameInput.text() or "").strip()
        ref_lig_resname = (self.grpRefLigResnameInput.text() or "").strip()

        chain = (self.grpChainInput.text() or "").strip()
        resnum = int(self.grpResnumSpin.value())
        restype = (self.grpRestypeInput.text() or "").strip()
        prot_atom = (self.grpProteinAtomInput.text() or "").strip()
        lig_atom_true = (self.grpLigandAtomTrueInput.text() or "").strip()

        # ✅ read out_path from the GUI FIRST
        out_path = ""
        try:
            out_path = (self.grpOutInput.text() or "").strip()
        except Exception:
            out_path = ""

        # If user didn't provide an output path, auto-fill one
        if not out_path and lig_name:
            root = _constraints_root_posix()
            if root:
                out_path = str(PurePosixPath(root) / f"{lig_name}_constraints.cst")
                try:
                    self.grpOutInput.setText(out_path)
                except Exception:
                    pass

        # Required field checks (keep it strict, matches argparse)
        missing = []
        if not ref_pdb: missing.append("--ref-pdb")
        if not lig_key: missing.append("--ligand-key")
        if not lig_name: missing.append("--ligand-name")
        if not chain: missing.append("--chain")
        if not restype: missing.append("--restype")
        if not prot_atom: missing.append("--protein-atom")
        if not lig_atom_true: missing.append("--ligand-atom-true")
        if missing:
            raise ValueError("Missing required inputs: " + ", ".join(missing))

        argv = [
            sys.executable,
            "-m",
            "af3_pipeline.analysis.generate_rosetta_constraints",
        ]

        argv += ["--ref-pdb", ref_pdb]
        argv += ["--ligand-key", lig_key]
        argv += ["--ligand-name", lig_name]

        if ref_lig_resname:
            argv += ["--ref-ligand-resname", ref_lig_resname]

        argv += ["--chain", chain]
        argv += ["--resnum", str(resnum)]
        argv += ["--restype", restype]
        argv += ["--protein-atom", prot_atom]
        argv += ["--ligand-atom-true", lig_atom_true]

        argv += ["--sd-dist", str(float(self.grpSdDistSpin.value()))]
        argv += ["--sd-ang", str(float(self.grpSdAngSpin.value()))]
        argv += ["--sd-dih", str(float(self.grpSdDihSpin.value()))]

        if out_path:
            argv += ["--out", out_path]

        return argv



    def _grp_show_command(self):
        try:
            argv = self._build_grp_constraints_cmd()
            # Pretty print: quote args with spaces
            def q(s: str) -> str:
                return f'"{s}"' if (" " in s or "\t" in s) else s
            cmd = " ".join(q(a) for a in argv)
            self.log("🧾 Covalent constraints command:\n" + cmd)
            if hasattr(self, "grpStatusLabel"):
                self.grpStatusLabel.setText("Status: command shown in log")
        except Exception as e:
            QMessageBox.warning(self, "Command build failed", str(e))


    def _grp_run_constraints_generator(self):
        try:
            argv = self._build_grp_constraints_cmd()
        except Exception as e:
            QMessageBox.warning(self, "Missing/invalid inputs", str(e))
            return

        # UI feedback
        try:
            self.grpStatusLabel.setText("Status: running…")
            self.grpRunBtn.setEnabled(False)
        except Exception:
            pass

        self.log("▶ Running covalent constraints generator…")
        self.log(" ".join(argv))

        try:
            # Run and stream output
            proc = subprocess.run(argv, capture_output=True, text=True)

            if proc.stdout:
                self.log(proc.stdout.rstrip())
            if proc.stderr:
                # keep stderr visible too
                self.log(proc.stderr.rstrip())

            if proc.returncode != 0:
                try:
                    self.grpStatusLabel.setText(f"Status: failed (code {proc.returncode})")
                except Exception:
                    pass
                QMessageBox.warning(self, "Generator failed", f"Exit code: {proc.returncode}\n\nSee log for details.")
                return

            try:
                self.grpStatusLabel.setText("Status: done ✅")
            except Exception:
                pass

        except Exception as e:
            try:
                self.grpStatusLabel.setText("Status: error")
            except Exception:
                pass
            QMessageBox.critical(self, "Run error", str(e))
        finally:
            try:
                self.grpRunBtn.setEnabled(True)
            except Exception:
                pass


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

        try:
            self._refresh_grp_ligand_dropdown()
        except Exception:
            pass

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

    def _save_config_yaml_silent(self):
        out = self._config_yaml_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self._ui_config_dict()
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.log(f"💾 Saved config.yaml (auto): {out}")

    def _get_template_from_cache(self, key: str) -> str:
        """
        Return saved template string for a sequence cache entry (protein/dna/rna).
        Backward-compatible with legacy cache entries.
        """
        if key not in self.sequence_cache:
            return ""
        val = self.sequence_cache[key]
        if isinstance(val, dict):
            return str(val.get("template", "") or "")
        return ""

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
                self.proteinSeqName.clear()
                self.proteinSeqEditor.setPlainText("")
                self.proteinTemplate.clear()
            elif kind == "dna":
                self.dnaSeqName.clear()
                self.dnaSeqEditor.setPlainText("")
                self.dnaTemplate.clear()
            else:
                self.rnaSeqName.clear()
                self.rnaSeqEditor.setPlainText("")
                self.rnaTemplate.clear()
            return

        seq = self._get_seq_from_cache(name)
        tmpl = self._get_template_from_cache(name)

        if kind == "protein":
            self.proteinSeqName.setText(name)
            self.proteinSeqEditor.setPlainText(seq)
            self.proteinTemplate.setText(tmpl)
        elif kind == "dna":
            self.dnaSeqName.setText(name)
            self.dnaSeqEditor.setPlainText(seq)
            self.dnaTemplate.setText(tmpl)
        else:
            self.rnaSeqName.setText(name)
            self.rnaSeqEditor.setPlainText(seq)
            self.rnaTemplate.setText(tmpl)

    def _save_sequence_from_editor(self, kind: str):
        if kind == "protein":
            name_w = self.proteinSeqName
            seq_w  = self.proteinSeqEditor
            dd     = self.proteinSeqDropdown
            tmpl_w = self.proteinTemplate
        elif kind == "dna":
            name_w = self.dnaSeqName
            seq_w  = self.dnaSeqEditor
            dd     = self.dnaSeqDropdown
            tmpl_w = self.dnaTemplate
        else:
            name_w = self.rnaSeqName
            seq_w  = self.rnaSeqEditor
            dd     = self.rnaSeqDropdown
            tmpl_w = self.rnaTemplate

        name = name_w.text().strip()
        seq  = seq_w.toPlainText().strip()
        tmpl = (tmpl_w.text() or "").strip()

        if not name or not seq:
            _msg_warn(self, "Missing input", "Please provide both a name and a sequence.")
            return

        if name in self.sequence_cache:
            if not _msg_yesno(self, "Overwrite?", f"A saved entry named '{name}' already exists.\nOverwrite it?"):
                return

        self.sequence_cache[name] = {"sequence": seq, "type": kind.capitalize(), "template": tmpl,}
        _write_json(SEQUENCE_CACHE_FILE, self.sequence_cache)

        was = dd.blockSignals(True)
        try:
            self._refresh_all_dropdowns()
            dd.setCurrentIndex(0)
        finally:
            dd.blockSignals(was)

        name_w.clear()
        seq_w.clear()  
        tmpl_w.clear()

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
        We follow your old cache convention: <LIGAND_CACHE>/<hash>/LIG.(pdb|cif)
        """
        try:
            lig_hash = entry.get("hash")
            if not lig_hash:
                smiles = entry.get("smiles", "")
                lig_hash = cache_utils.compute_hash(smiles)
            lig_dir = LIGAND_CACHE / lig_hash
            pdb_candidates = [lig_dir / "LIG.pdb", lig_dir / "lig.pdb"]
            cif_candidates = [lig_dir / "LIG.cif", lig_dir / "lig.cif"]
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
            fallbacks = []
            if sys.platform.startswith("win"):
                fallbacks = [
                    r"C:\Program Files\ChimeraX\bin\ChimeraX.exe",
                    r"C:\Program Files\ChimeraX\ChimeraX.exe",
                    r"C:\Program Files (x86)\ChimeraX\bin\ChimeraX.exe",
                ]

            exe_argv = self._resolve_viewer_exe(
                key="chimerax",
                ui_widget_name="cfgChimeraPath",
                fallbacks=fallbacks,
            )

            # If user set a path but it's wrong, give a helpful message once
            # (Optional: only warn if they actually typed something)
            if hasattr(self, "cfgChimeraPath"):
                typed = (self.cfgChimeraPath.text() or "").strip()
                if typed and not Path(typed).expanduser().exists():
                    _msg_warn(self, "ChimeraX Not Found", f"Configured ChimeraX path does not exist:\n{typed}\n\nFix it in Config → ChimeraX path.")

            subprocess.Popen(exe_argv + [str(target)])
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
            fallbacks = []
            if sys.platform.startswith("win"):
                # This varies a lot by install, so keep it minimal:
                fallbacks = []

            exe_argv = self._resolve_viewer_exe(
                key="pymol",
                ui_widget_name="cfgPyMolPath",
                fallbacks=fallbacks,
            )

            if hasattr(self, "cfgPyMolPath"):
                typed = (self.cfgPyMolPath.text() or "").strip()
                if typed and not Path(typed).expanduser().exists():
                    _msg_warn(self, "PyMOL Not Found", f"Configured PyMOL path does not exist:\n{typed}\n\nFix it in Config → PyMOL path.")

            subprocess.Popen(exe_argv + [str(target)])
        except Exception as e:
            _msg_err(self, "Error", f"Failed to open PyMOL:\n{e}")

    def _update_ligand_preview_from_cache(self, ligand_name: str):
        entry = self.ligand_cache.get(ligand_name)
        if not isinstance(entry, dict):
            return

        lig_hash = entry.get("hash") or cache_utils.compute_hash(entry.get("smiles", ""))
        lig_dir = LIGAND_CACHE / lig_hash

        svg_path = lig_dir / "LIG.svg"
        png_path = lig_dir / "LIG.png"  # if you later want to prefer PNG

        # Prefer SVG if present
        img = svg_path if svg_path.exists() else (png_path if png_path.exists() else None)

        if not img:
            self.ligandPreviewLabel.setText("(No 2D preview available yet)")
            self.ligandPreviewLabel.setPixmap(QPixmap())
            return

        # ---- SVG: render at widget pixel resolution (HiDPI-safe) ----
        if img.suffix.lower() == ".svg":
            renderer = QSvgRenderer(str(img))
            if not renderer.isValid():
                self.ligandPreviewLabel.setText("(Failed to load SVG preview)")
                self.ligandPreviewLabel.setPixmap(QPixmap())
                return

            # HiDPI support
            screen = QApplication.primaryScreen()
            dpr = screen.devicePixelRatio() if screen else 1.0

            w = max(1, int(self.ligandPreviewLabel.width() * dpr))
            h = max(1, int(self.ligandPreviewLabel.height() * dpr))

            pm = QPixmap(w, h)
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)

            painter = QPainter(pm)
            renderer.render(painter)
            painter.end()

            self.ligandPreviewLabel.setPixmap(pm)
            self.ligandPreviewLabel.setText("")
            return

        # ---- PNG fallback: scale smoothly ----
        pix = QPixmap(str(img))
        if pix.isNull():
            self.ligandPreviewLabel.setText("(Failed to load preview image)")
            self.ligandPreviewLabel.setPixmap(QPixmap())
            return

        self.ligandPreviewLabel.setPixmap(
            pix.scaled(
                self.ligandPreviewLabel.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
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

        # Notes + logs
        self.notesButton.clicked.connect(self._open_notes_dialog)
        self._install_constraints_dropdown_existing("constraintsDropdown_2")

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
            tmpl = self._get_template_from_cache(selected_name)

            if getattr(refs, "name", None):
                refs.name.setText(selected_name)

            refs.seq.setPlainText(seq)

            # NEW: template autofill (if the entry widget has it)
            if getattr(refs, "template", None):
                refs.template.setText(tmpl)

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
            "skip_rosetta": bool(self.skipRosettaCheckbox.isChecked()),
            "skip_rosetta_ligand": bool(self.skipRosettaLigandCheckbox.isChecked()),
            "multi_seed": bool(self.multiSeedCheckbox.isChecked()),
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
        
        self._pending_multi_seed = bool(spec.get("multi_seed", False))
        self._pending_skip_rosetta = bool(spec.get("skip_rosetta", False))
        self._pending_skip_rosetta_ligand = bool(spec.get("skip_rosetta_ligand", False))

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
        self._pending_json_path = json_path

        self.run_thread = RunThread(
            json_path=json_path,
            job_name=self.current_jobname or "GUI_job",
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

    def _on_analysis_done(self):
        self.log("✅ Analysis complete.")
        self.is_running = False
        self.runButton.setEnabled(True)
        self.addToQueueButton.setEnabled(True)

        self._refresh_runs_view()
        self._maybe_run_next()

    def _on_analysis_failed(self, err: str):
        self.log(f"❌ Analysis failed:\n{err}")
        # Still allow queue to continue
        self.is_running = False
        self.runButton.setEnabled(True)
        self.addToQueueButton.setEnabled(True)

        self._refresh_runs_view()
        self._maybe_run_next()

    def _on_run_finished(self, json_path: str):
        self.log(f"✅ AF3 run finished: {json_path}")

        # Record history NOW (or after analysis — your choice). I’d keep it now.
        self._record_run_history_from_spec(self._build_current_job_spec(), json_path=json_path)
        self._refresh_runs_view()

        # Always run analysis next
        job_folder_name = self._guess_selected_job_folder()
        # If you don't have this helper, simplest is: job_folder_name = self.current_jobname
        # BUT you already use folder name in RUNS tab based on filesystem.
        # Recommended: call runner to get job folder name; if not available, fall back to current_jobname.

        job_folder_name = self.current_jobname  # fallback; adjust if your post_analysis expects timestamped folder name

        multi = bool(self._pending_multi_seed)
        skip = bool(self._pending_skip_rosetta)
        skip_ligand = bool(self._pending_skip_rosetta_ligand)

        self.log(f"🧠 Post-analysis (always on): skip_rosetta_relax={skip}, skip_rosetta_ligand={skip_ligand} multi_seed={multi}")

        constraints_set = str(cfg.get("rosetta.constraints_set") or "auto").strip().lower()
        constraints_file = str(cfg.get("rosetta.constraints_file") or "").strip()

        if constraints_set == "auto":
            constraints_file = None
        elif not constraints_file:
            constraints_file = None  # custom selected but no file => treat as auto
        self.analysis_thread = AnalysisWorker(
            job_folder_name,
            multi_seed=multi,
            skip_rosetta=skip,
            skip_rosetta_ligand=skip_ligand,
            constraints_file=constraints_file,
        )
        self.analysis_thread.log.connect(self.log)
        self.analysis_thread.done.connect(self._on_analysis_done)
        self.analysis_thread.error.connect(self._on_analysis_failed)
        self.analysis_thread.start()

    def _on_run_failed(self, err: str):
        self.log(f"❌ Run failed: {err}")
        traceback.print_exc()
        self.is_running = False
        self.runButton.setEnabled(True)
        self.addToQueueButton.setEnabled(True)
        self._maybe_run_next()

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
        """
        Runs/Post-analysis page is now Designer-owned.
        Do NOT dynamically install toolbar/table (runs_features v2 installers).
        """

        # Selection drives details
        if hasattr(self, "runsHistoryTable"):
            self.runsHistoryTable.itemSelectionChanged.connect(
                lambda: self._show_selected_run_details(self._runs_selected_row())
            )
            hdr = self.runsHistoryTable.horizontalHeader()
            hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.runsHistoryTable.horizontalHeader().setMinimumSectionSize(80)
        else:
            # legacy fallback
            self.runsList.currentRowChanged.connect(self._show_selected_run_details)

        # Toolbar row (Designer)
        if hasattr(self, "runsRefreshButton"):
            self.runsRefreshButton.clicked.connect(self._refresh_runs_view)

        if hasattr(self, "runsSearchInput"):
            self.runsSearchInput.textChanged.connect(lambda *_: self._refresh_runs_view())

        if hasattr(self, "runsOpenJSONButton"):
            self.runsOpenJSONButton.clicked.connect(self._runs_open_selected_json)

        if hasattr(self, "runsDeleteHistoryButton"):
            self.runsDeleteHistoryButton.clicked.connect(self._runs_delete_selected_from_history)

        # Existing action buttons (Designer)
        if hasattr(self, "runsLoadToAlphafoldButton"):
            self.runsLoadToAlphafoldButton.clicked.connect(self._load_selected_run_to_alphafold)

        if hasattr(self, "runsOpenSelectedButton"):
            self.runsOpenSelectedButton.clicked.connect(self._runs_open_selected_run_folder)

        if hasattr(self, "runsOpenMetricsButton"):
            self.runsOpenMetricsButton.clicked.connect(self._runs_open_selected_metrics_csv)

        if hasattr(self, "runsAnalyzeButton"):
            self.runsAnalyzeButton.clicked.connect(self._runs_run_selected_analysis)

        # Viewer controls (Designer)
        if hasattr(self, "openAlphafold"):
            self.openAlphafold.clicked.connect(lambda: self._runs_open_structure(which="af"))
        if hasattr(self, "openRosetta"):
            self.openRosetta.clicked.connect(lambda: self._runs_open_structure(which="rosetta"))
        if hasattr(self, "openBoth"):
            self.openBoth.clicked.connect(lambda: self._runs_open_structure(which="both"))

        # Constraints dropdown on Runs page (Designer)
        self._install_constraints_dropdown_existing("constraintsDropdown")

        # Initial populate
        self._refresh_runs_view()


    def _record_run_history_from_spec(self, spec: dict[str, Any], *, json_path: str):
        """
        Record a run-history entry.

        IMPORTANT:
        - We prefer reading the spec back from json_path (per-run file) because any in-memory
        spec you pass may be a shared/mutated object (e.g., UI state updated for the next job).
        - We deep-copy the dict so nested lists/dicts can't be mutated later and "rewrite history".
        """
        if not json_path:
            return

        # Try to read the on-disk spec for THIS run
        disk_spec = _read_json(json_path, None)
        if isinstance(disk_spec, dict) and disk_spec:
            src = disk_spec
        else:
            # Fallback: use provided spec (but still deepcopy)
            src = spec if isinstance(spec, dict) else {}
            if not src:
                return

        rec = copy.deepcopy(src)

        rec["json_path"] = json_path

        # Use created_at for when we recorded it (or queued it); finished_at is misleading here.
        # Keep finished_at if you already rely on it in the UI.
        now = _now_iso()
        rec.setdefault("created_at", now)
        rec["finished_at"] = now

        # Ensure jobname is stable even if spec doesn't contain it.
        # If fold_input.json lives at .../<jobfolder>/input/fold_input.json, take <jobfolder>.
        try:
            p = Path(json_path)
            # parent: input ; parent.parent: <jobfolder>
            job_folder = p.parent.parent.name if p.parent and p.parent.parent else ""
        except Exception:
            job_folder = ""

        rec["jobname"] = (rec.get("jobname") or job_folder or "(unnamed)").strip()

        # Optional: stamp a unique id so selection survives renames
        rec.setdefault("run_id", rec["jobname"])

        self.runs_history.insert(0, rec)
        _write_json(RUNS_HISTORY_FILE, self.runs_history)

    def _refresh_runs_view(self):
        data = _read_json(RUNS_HISTORY_FILE, [])

        # Compatibility: dict => [dict]
        if isinstance(data, dict):
            self.runs_history = [data]
        elif isinstance(data, list):
            self.runs_history = data
        else:
            self.runs_history = []

        # Search filter (Designer)
        q = ""
        if hasattr(self, "runsSearchInput"):
            q = (self.runsSearchInput.text() or "").strip().lower()

        runs = [r for r in self.runs_history if isinstance(r, dict)]
        if q:
            runs = [r for r in runs if q in (r.get("jobname") or "").lower()]

        # Store filtered view so selection index maps correctly
        self._runs_filtered = runs

        # Prefer table
        if hasattr(self, "runsHistoryTable"):
            t = self.runsHistoryTable
            was = t.blockSignals(True)
            try:
                t.setRowCount(len(runs))
                t.setColumnCount(5)
                t.setHorizontalHeaderLabels(["Job", "Finished", "Seed", "Relax", "Ligand"])
                t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
                t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
                t.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
                t.verticalHeader().setVisible(False)

                hdr = t.horizontalHeader()
                hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
                hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

                for r, rec in enumerate(runs):
                    job = (rec.get("jobname") or "(unnamed)").strip()
                    ts = rec.get("finished_at") or rec.get("created_at") or ""
                    ts_pretty = self._format_run_timestamp_for_list(ts)

                    multi = "yes" if bool(rec.get("multi_seed", False)) else "no"
                    relax = "no" if bool(rec.get("skip_rosetta", False)) else "yes"
                    ligand = "no" if bool(rec.get("skip_rosetta_ligand", False)) else "yes"

                    t.setItem(r, 0, QTableWidgetItem(job))
                    t.setItem(r, 1, QTableWidgetItem(ts_pretty))
                    t.setItem(r, 2, QTableWidgetItem(multi))
                    t.setItem(r, 3, QTableWidgetItem(relax))
                    t.setItem(r, 4, QTableWidgetItem(ligand))
            finally:
                t.blockSignals(was)

            t = self.runsHistoryTable
            hdr = t.horizontalHeader()

            # Stop Qt from using weird leftover sizes
            hdr.setStretchLastSection(False)

            # First: size everything to header + contents
            t.resizeColumnsToContents()

            # Then: enforce minimum widths so header labels are readable
            # (tweak numbers if you want)
            min_widths = {
                0: 220,  # Job
                1: 140,  # Finished
                2: 80,   # Seed
                3: 80,   # Relax
                4: 90,   # Ligand
            }
            for col, w in min_widths.items():
                t.setColumnWidth(col, max(t.columnWidth(col), w))

            # Finally: let Job take remaining space, BUT only after others are readable
            hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
            hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
            hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
            hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

            if len(runs) == 0:
                if hasattr(self, "runDetailsText"):
                    self.runDetailsText.setPlainText("No runs found yet.")
                return

            self._runs_set_selected_row(0)
            QTimer.singleShot(0, lambda: self._show_selected_run_details(0))
            return

        # Fallback list (old UI)
        lst = self.runsList
        was = lst.blockSignals(True)
        try:
            lst.clear()
            for rec in runs:
                job = (rec.get("jobname") or "(unnamed)").strip()
                ts = rec.get("finished_at") or rec.get("created_at") or ""
                ts_pretty = self._format_run_timestamp_for_list(ts)
                lst.addItem(f"{job}    {ts_pretty}")
        finally:
            lst.blockSignals(was)

        if lst.count() == 0:
            self.runDetailsText.setPlainText("No runs found yet.")
            return

        if lst.currentRow() < 0:
            lst.setCurrentRow(0)
        QTimer.singleShot(0, lambda: self._show_selected_run_details(lst.currentRow()))

    def _jobs_root_dir(self) -> Path:
        # Match runner._user_jobs_root() default
        return (Path.home() / ".af3_pipeline" / "jobs").resolve()

    def _open_path_in_file_manager(self, p: Path):
        import subprocess as _subprocess
        p = Path(p)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                _subprocess.call(["open", str(p)])
            else:
                _subprocess.call(["xdg-open", str(p)])
        except Exception as e:
            _msg_err(self, "Open failed", f"Failed to open:\n{p}\n\n{e}")

    def _runs_after_analysis_refresh(self, job_dir: Path):
        self.log("✅ RUNS: analysis complete. Refreshing metrics display…")
        row = self._runs_selected_row()
        self._show_selected_run_details(row)

    def _selected_run_record(self) -> Optional[dict[str, Any]]:
        row = self._runs_selected_row()
        if row < 0:
            return None

        runs = getattr(self, "_runs_filtered", None)
        if not isinstance(runs, list):
            runs = self.runs_history if isinstance(self.runs_history, list) else []

        if row >= len(runs):
            return None

        rec = runs[row]
        return rec if isinstance(rec, dict) else None

    def _show_selected_run_details(self, row: int):
        rec = self._selected_run_record()
        if not rec:
            return

        from datetime import datetime
        import html
        import json

        def _format_ts(ts) -> str:
            if not ts:
                return "—"
            if isinstance(ts, datetime):
                dt = ts
            else:
                s = str(ts).strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(s)
                except Exception:
                    return str(ts)

            try:
                if dt.tzinfo is not None:
                    dt = dt.astimezone()
            except Exception:
                pass

            month = dt.strftime("%b")
            day = dt.day
            year = dt.year
            hour24 = dt.hour
            minute = dt.minute
            ampm = "AM" if hour24 < 12 else "PM"
            hour12 = hour24 % 12
            if hour12 == 0:
                hour12 = 12
            return f"{month} {day}, {year} {hour12}:{minute:02d} {ampm}"

        def _pick_ts(rec: dict) -> tuple[str, object]:
            for k in ("started_at", "timestamp", "created_at", "time", "ts", "datetime"):
                if k in rec and rec.get(k):
                    return k, rec.get(k)
            return "timestamp", None

        jobname = str(rec.get("jobname") or "—")
        ts_key, ts_val = _pick_ts(rec)
        ts_pretty = _format_ts(ts_val)

        # Resolve job folder + metrics CSV (define metrics_csv in all paths)
        job_dir = self._guess_selected_job_folder()
        metrics_csv = None
        metrics_block = ""
        metrics_title = "METRICS"

        if job_dir:
            metrics_csv = self._runs_find_metrics_csv(job_dir)
            if metrics_csv:
                try:
                    metrics_block = self._runs_metrics_text(metrics_csv)
                except Exception as e:
                    metrics_block = f"\n\n===== {metrics_title} =====\n(Error reading metrics: {e})\n"
            else:
                metrics_block = f"\n\n===== {metrics_title} =====\n(No metrics CSV found yet under {job_dir})\n"
        else:
            metrics_block = f"\n\n===== {metrics_title} =====\n(Could not locate bundled job folder in ~/.af3_pipeline/jobs)\n"

        raw_pretty = json.dumps(rec, indent=2, ensure_ascii=False)

        # Prefer rich HTML if supported
        if hasattr(self.runDetailsText, "setHtml"):
            def esc(x) -> str:
                return html.escape(str(x))

            rows = [
                ("Job", jobname),
                ("When", f"{ts_pretty}  (from '{ts_key}')"),
            ]

            # Optional keys if present
            for k in ("model", "af_model", "model_name", "seeds", "ligand", "ligand_name"):
                if rec.get(k):
                    rows.append((k.replace("_", " ").title(), rec.get(k)))

            if job_dir:
                rows.append(("Folder", str(job_dir)))

            rows_html = "\n".join(
                "<tr>"
                f"<td style='padding:2px 10px 2px 0; color:#666; white-space:nowrap;'><b>{esc(k)}</b></td>"
                f"<td style='padding:2px 0; word-break:break-all;'>{esc(v)}</td>"
                "</tr>"
                for k, v in rows
            )

            # Metrics: structured HTML if available, else fallback italic
            try:
                if metrics_csv:
                    metrics_html = self._runs_metrics_html(metrics_csv)
                elif job_dir:
                    metrics_html = f"<i>No metrics CSV found under {esc(job_dir)}</i>"
                else:
                    metrics_html = "<i>Could not locate job folder to search for metrics.</i>"
            except Exception as e:
                metrics_html = f"<i>Error rendering metrics:</i> {esc(e)}"

            # Raw record (HTML)
            raw_html = (
                f"<pre style='margin:8px 0 0 0; padding:8px; "
                f"background:rgba(0,0,0,0.04); border-radius:6px; "
                f"white-space:pre-wrap;'>{esc(raw_pretty)}</pre>"
            )

            html_text = f"""
            <div style="font-family:Segoe UI, Arial; font-size:12px;">
            <div style="font-size:13px; margin-bottom:6px;"><b>Run details</b></div>
            <table style="border-collapse:collapse;">
                {rows_html}
            </table>

            <div style="margin-top:10px;"><b>Metrics</b></div>
            {metrics_html}

            <div style="margin-top:10px;"><b>Raw record</b></div>
            {raw_html}
            </div>
            """.strip()

            self.runDetailsText.setHtml(html_text)
            return

        # Fallback: plain text (QPlainTextEdit)
        out = []
        out.append("===== RUN DETAILS =====")
        out.append(f"Job:    {jobname}")
        out.append(f"When:   {ts_pretty}  (from '{ts_key}')")
        if job_dir:
            out.append(f"Folder: {job_dir}")
        out.append("")
        out.append(metrics_block.strip())
        out.append("")
        out.append("===== RAW RECORD =====")
        out.append(raw_pretty)

        self.runDetailsText.setPlainText("\n".join(out))

    def _runs_open_selected_json(self):
        rec = self._selected_run_record()
        if not rec:
            _msg_warn(self, "No selection", "Select a run first.")
            return

        jp = (rec.get("json_path") or "").strip()
        if not jp:
            _msg_warn(self, "Not found", "This run has no json_path recorded in runs_history.json.")
            return

        p = Path(jp)
        if not p.exists():
            _msg_warn(self, "Not found", f"JSON file not found:\n{p}")
            return

        # Open in file manager (and user can double click to open)
        self._open_path_in_file_manager(p.parent)

    def _runs_delete_selected_from_history(self):
        row = self._runs_selected_row()
        runs = getattr(self, "_runs_filtered", None)
        if not isinstance(runs, list) or row < 0 or row >= len(runs):
            _msg_warn(self, "No selection", "Select a run first.")
            return

        rec = runs[row]
        # Remove this exact dict from full history (by identity-ish matching)
        full = self.runs_history if isinstance(self.runs_history, list) else []
        try:
            full.remove(rec)
        except ValueError:
            # fallback: match by jobname+finished_at
            j = rec.get("jobname")
            t = rec.get("finished_at") or rec.get("created_at")
            full = [
                r for r in full
                if not (isinstance(r, dict) and (r.get("jobname") == j) and ((r.get("finished_at") or r.get("created_at")) == t))
            ]

        _write_json(RUNS_HISTORY_FILE, full)
        self._refresh_runs_view()

    def _set_protein_atom_combo_from_prot_atom(self, prot_atom: str) -> None:
        """
        Restore proteinAtomComboBox from a stored prot_atom value (e.g. 'SG', 'NZ', ...).

        - If prot_atom matches one of ATOM_MAP values, select that label in the combo box.
        - Otherwise, select Custom… and store it in self.custom_protein_atom.
        """
        prot_atom = (prot_atom or "").strip()
        if not prot_atom:
            # default to first item (or leave as-is)
            self.proteinAtomComboBox.setCurrentIndex(0)
            self.custom_protein_atom = ""
            return

        # Build inverse: atom -> label
        inv = {}
        for label, atom in (ATOM_MAP or {}).items():
            if atom:
                inv[str(atom)] = str(label)

        label = inv.get(prot_atom)
        if label:
            idx = self.proteinAtomComboBox.findText(label)
            if idx >= 0:
                self.proteinAtomComboBox.setCurrentIndex(idx)
                self.custom_protein_atom = ""
                return

        # Not found -> Custom…
        custom_idx = self.proteinAtomComboBox.findText("Custom…")
        if custom_idx < 0:
            custom_idx = self.proteinAtomComboBox.findText("Custom...")

        if custom_idx >= 0:
            self.proteinAtomComboBox.setCurrentIndex(custom_idx)
        else:
            # As a last resort, just leave current selection
            pass

        self.custom_protein_atom = prot_atom



    def _load_selected_run_to_alphafold(self):
        rec = self._selected_run_record()
        if not rec:
            return

        # switch to alphafold page
        self.pagesStack.setCurrentIndex(4)
        self.navList.setCurrentRow(4)

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
        self._set_protein_atom_combo_from_prot_atom(lig.get("prot_atom", "") or "")

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

    # =======================
    # 🏁 entry safety
    # =======================
    def closeEvent(self, event):  # noqa: N802
        # optional: persist queue view etc.
        super().closeEvent(event)

# ==============================================================
# Attach moved functions back onto MainWindow (no refactor needed)
# ==============================================================

# ---- Config tab functions ----
MainWindow._config_help_defs = _config_tab._config_help_defs
MainWindow._install_config_help = _config_tab._install_config_help
MainWindow._ui_config_dict = _config_tab._ui_config_dict
MainWindow._save_config_yaml = _config_tab._save_config_yaml
MainWindow._load_config_yaml_best_effort = _config_tab._load_config_yaml_best_effort
MainWindow._reset_config_fields_from_cfg = _config_tab._reset_config_fields_from_cfg
MainWindow._start_autodetect = _config_tab._start_autodetect
MainWindow._on_autodetect_done = _config_tab._on_autodetect_done
MainWindow._config_needs_setup = _config_tab._config_needs_setup

# ---- Runs features ----
MainWindow._install_runs_toolbar_v2 = _runs_features._install_runs_toolbar_v2
MainWindow._runs_open_structure = _runs_features._runs_open_structure
MainWindow._runs_open_selected_run_folder = _runs_features._runs_open_selected_run_folder
MainWindow._runs_open_selected_relax_folder = _runs_features._runs_open_selected_relax_folder
MainWindow._runs_open_selected_metrics_csv = _runs_features._runs_open_selected_metrics_csv
MainWindow._runs_run_selected_analysis = _runs_features._runs_run_selected_analysis

MainWindow._guess_selected_job_folder = _runs_features._guess_selected_job_folder
MainWindow._runs_find_relax_dir = _runs_features._runs_find_relax_dir
MainWindow._runs_find_relax_scripts_dir = _runs_features._runs_find_relax_scripts_dir
MainWindow._runs_find_metrics_csv = _runs_features._runs_find_metrics_csv
MainWindow._runs_find_af_structure = _runs_features._runs_find_af_structure
MainWindow._runs_find_rosetta_structure = _runs_features._runs_find_rosetta_structure
MainWindow._format_run_timestamp_for_list = _runs_features._format_run_timestamp_for_list
MainWindow._runs_set_selected_row = _runs_features._runs_set_selected_row
MainWindow._runs_selected_row = _runs_features._runs_selected_row
MainWindow._install_runs_history_table = _runs_features._install_runs_history_table

# ---- Runs metrics ----
MainWindow._runs_metrics_text = _runs_metrics._runs_metrics_text
MainWindow._runs_metrics_html = _runs_metrics._runs_metrics_html

# ---- Viewers ----
MainWindow._resolve_viewer_exe = _viewers._resolve_viewer_exe
MainWindow._open_structures_in_pymol = _viewers._open_structures_in_pymol
MainWindow._open_structures_in_chimerax = _viewers._open_structures_in_chimerax


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
