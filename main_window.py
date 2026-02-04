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
import shutil
import yaml
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
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
)

# === Configuration ===
from af3_pipeline.config import cfg

# === Backend ===
from af3_pipeline import json_builder, runner, cache_utils
from af3_pipeline.ligand_utils import prepare_ligand_from_smiles, _canonical_smiles

# ==========================
# 📁 Paths & Cache (config-driven)
# ==========================
def norm_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()


WSL_DISTRO = cfg.get("wsl_distro", "Ubuntu-22.04")

def _default_user_cfg_dir() -> Path:
    return Path.home() / ".af3_pipeline"

GUI_CACHE_DIR = norm_path(_default_user_cfg_dir() / "gui_cache")
GUI_CACHE_DIR.mkdir(parents=True, exist_ok=True)


SEQUENCE_CACHE_FILE = GUI_CACHE_DIR / "sequence_cache.json"
LIGAND_CACHE_FILE   = GUI_CACHE_DIR / "ligand_cache.json"
QUEUE_FILE          = GUI_CACHE_DIR / "job_queue.json"
NOTES_FILE          = GUI_CACHE_DIR / "notes.txt"
RUNS_HISTORY_FILE   = GUI_CACHE_DIR / "runs_history.json"
cache_root_raw = (cfg.get("cache_root") or "").strip()
if cache_root_raw:
    LIGAND_CACHE = norm_path(Path(cache_root_raw) / "ligands")
else:
    # fallback if cache_root not set (should be rare)
    LIGAND_CACHE = norm_path(_default_user_cfg_dir() / "cache" / "ligands")

LIGAND_CACHE.mkdir(parents=True, exist_ok=True)

# ==========================
# ⚙️ Constants
# ==========================
PTM_CHOICES = {
    "None": None,
    "Phosphoserine (pSer)": "SEP",
    "Phosphothreonine (pThr)": "TPO",
    "Phosphotyrosine (pTyr)": "PTR",
    "Hydroxyproline": "HYP",
    "Hydroxylysine": "HYL",
    "N6-Acetyllysine": "ALY",
    "O-Acetylserine": "ASE",
    "Monomethyllysine": "MLY",
    "Dimethyllysine": "M2L",
    "Trimethyllysine": "M3L",
    "Monomethylarginine": "MMA",
    "Dimethylarginine": "DMA",
    "Citrulline": "CIT",
    "Selenomethionine": "MSE",
    "N-Formylmethionine": "FME",
    "Sulfotyrosine": "TYS",
    "Pyroglutamate": "PCA",
    "Dehydroalanine": "DHA",
    "Carboxymethyllysine": "CML",
    "N-Methylalanine": "NAL",
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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
    Return absolute path to a resource bundled by PyInstaller, or in dev mode.
    Works for both --onedir and --onefile builds.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / rel
    return Path(__file__).resolve().parent / rel


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
        save_button = QPushButton("Save & Close", self)
        save_button.clicked.connect(self.save_and_close)
        layout.addWidget(save_button)

    def save_and_close(self):
        NOTES_FILE.write_text(self.text_edit.toPlainText(), encoding="utf-8")
        self.accept()


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

        # default cache subdir if missing
        if "gui_cache_subdir" not in s:
            s["gui_cache_subdir"] = "cache\\af3_gui_cache"
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
            cmd = ["python", "-m", "af3_pipeline.analysis.post_analysis", "--job", str(self.job_name)]
            if self.multi_seed:
                cmd.append("--multi_seed")

            self.log.emit(f"▶ Post-AF3 analysis\n$ {' '.join(cmd)}")
            with subprocess.Popen(
                cmd,
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
    ptm: QComboBox
    ptm_pos: QLineEdit
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

    ptm = QComboBox(frame)
    for label, ccd in ptm_choices.items():
        ptm.addItem(label, ccd)

    ptm_pos = QLineEdit(frame)
    ptm_pos.setPlaceholderText("PTM pos")

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

    row.addWidget(ptm, 2)
    row.addWidget(ptm_pos, 1)
    row.addWidget(del_btn, 0)

    outer.addLayout(row)

    seq = QPlainTextEdit(frame)
    seq.hide()

    refs = MacroEntryRefs(
        root=frame,
        dropdown=dd,
        name=name,   # ✅ exists, but hidden
        seq=seq,
        ptm=ptm,
        ptm_pos=ptm_pos,
        template=template,
        delete_btn=del_btn,
    )

    del_btn.clicked.connect(lambda: on_delete(refs))
    dd.currentTextChanged.connect(lambda txt: on_select(refs, txt))

    return refs

# ==========================
# 🪟 Main Window
# ==========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(__file__), "main_window.ui")
        ui_path = resource_path("main_window.ui")
        uic.loadUi(str(ui_path), self)
        self.resize(1500, 1200)       # good default
        self.setMinimumSize(900, 600)

        # ---- caches ----
        self.sequence_cache: dict[str, Any] = _read_json(SEQUENCE_CACHE_FILE, {})
        self.ligand_cache: dict[str, Any] = _read_json(LIGAND_CACHE_FILE, {})
        self.runs_history: list[dict[str, Any]] = _read_json(RUNS_HISTORY_FILE, [])

        # ---- state ----
        self.custom_protein_atom: Optional[str] = None
        self.is_running = False
        self.build_thread: Optional[BuildThread] = None
        self.run_thread: Optional[RunThread] = None
        self.analysis_thread: Optional[AnalysisWorker] = None
        self.current_jobname: str = ""

        # ---- dynamic entries ----
        self.protein_entries: list[MacroEntryRefs] = []
        self.dna_entries: list[MacroEntryRefs] = []
        self.rna_entries: list[MacroEntryRefs] = []

        self._bootstrap_config_yaml_from_template()

        # ---- wire up ----
        self._connect_nav()
        self._connect_sequences_page()
        self._connect_ligands_page()
        self._connect_alphafold_page()
        self._connect_runs_page()
        self._connect_config_page()

        # ---- initial populate ----
        self._refresh_all_dropdowns()
        self._refresh_queue_view()
        self._refresh_runs_view()

        # covalent field enable
        self.covalentCheckbox.toggled.connect(self._update_covalent_fields)
        self._update_covalent_fields()

        # atom selection custom
        self.proteinAtomComboBox.currentTextChanged.connect(self._on_protein_atom_change)

        # --- First run detection (only if required fields missing) ---
        if self._config_needs_setup():
            self.pagesStack.setCurrentIndex(4)  # Config tab index (adjust if different)
            self.navList.setCurrentRow(4)
            self.log("🧭 First-time setup: running auto-detect…")
            self._start_autodetect()

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

        put_if_missing("platform", "wsl" if sys.platform.startswith("win") else "linux")
        put_if_missing("wsl_distro", "Ubuntu-22.04")
        put_if_missing("gui_dir", str(Path(__file__).resolve().parent))
        put_if_missing("gui_cache_subdir", "cache/af3_gui_cache")
        put_if_missing("linux_home_root", "/home/olive")

        put_if_missing("af3_dir", "/home/olive/Repositories/alphafold")
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
        msa.setdefault("db", "/home/olive/Repositories/alphafold/mmseqs_db")

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
        if hasattr(self, "_detect_thread") and self._detect_thread and self._detect_thread.isRunning():
            return

        plat = (self.cfgPlatform.currentText() or "wsl").strip()
        distro = self.cfgWslDistro.text().strip()

        self._detect_thread = DetectConfigWorker(plat, distro)
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
        if not s:
            self.log("⚠️ Auto-detect returned no suggestions.")
            return

        # Top-level simple fields
        if "platform" in s:
            self._set_combo_text_force(self.cfgPlatform, str(s["platform"]))
        if "wsl_distro" in s:
            self.cfgWslDistro.setText(str(s["wsl_distro"]))
        if "gui_dir" in s:
            self.cfgGuiDir.setText(str(s["gui_dir"]))
        if "gui_cache_subdir" in s:
            self.cfgGuiCacheSubdir.setText(str(s["gui_cache_subdir"]))
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

        self.log("✅ Auto-detect applied + config saved.")

    def _config_needs_setup(self) -> bool:
        # Required fields to run in your setup
        plat = (self.cfgPlatform.currentText() or "").strip().lower()
        if not plat:
            return True

        if plat == "wsl" and not self.cfgWslDistro.text().strip():
            return True

        if not self.cfgGuiDir.text().strip():
            return True

        if not self.cfgAf3Dir.text().strip():
            return True

        # MSA DB is required for MSA generation
        if not self.cfgMMseqsDB.text().strip():
            return True

        return False



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
        if name in {"Select saved…", "Select saved..."}:
            self.ligandName.clear()
            self.ligandSmiles.clear()
            self.ligandPreviewLabel.setText("(Preview image will appear here)")
            self.ligandPreviewLabel.setPixmap(QPixmap())
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
            mod_ccd = e.ptm.currentData() or "None"
            pos = e.ptm_pos.text().strip()
            proteins.append({
                "name": name,
                "sequence": seq,
                "template": template,
                "modification": mod_ccd,
                "mod_position": pos,
            })

        # DNA/RNA (lists)
        dna_list: list[dict[str, Any]] = []
        for e in self.dna_entries:
            seq = e.seq.toPlainText().strip()
            if not seq:
                continue
            mod_ccd = e.ptm.currentData() or "None"
            pos = e.ptm_pos.text().strip()
            dna_list.append({"sequence": seq, "modification": mod_ccd, "pos": pos})

        rna_list: list[dict[str, Any]] = []
        for e in self.rna_entries:
            seq = e.seq.toPlainText().strip()
            if not seq:
                continue
            mod_ccd = e.ptm.currentData() or "None"
            pos = e.ptm_pos.text().strip()
            rna_list.append({"sequence": seq, "modification": mod_ccd, "pos": pos})

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
        """
        Open <af3_dir>/af_output
        Windows opens UNC: \\wsl.localhost\<distro>\<af3_dir>\af_output
        """
        import subprocess as _subprocess
        from pathlib import PurePosixPath

        af3_dir_cfg = (cfg.get("af3_dir") or "").strip()
        linux_output = PurePosixPath(af3_dir_cfg) / "af_output"

        if sys.platform.startswith("win"):
            distro = cfg.get("wsl_distro") or "Ubuntu-22.04"
            linux_part = str(linux_output).lstrip("/").replace("/", "\\")
            unc_str = rf"\\wsl.localhost\{distro}\{linux_part}"
            output_dir = Path(unc_str)
        else:
            output_dir = Path(str(linux_output))

        try:
            if output_dir.exists():
                if sys.platform.startswith("win"):
                    os.startfile(str(output_dir))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    _subprocess.call(["open", str(output_dir)])
                else:
                    _subprocess.call(["xdg-open", str(output_dir)])
            else:
                _msg_warn(self, "Not found", f"Output directory not found:\n{output_dir}")
        except OSError as e:
            _msg_err(self, "Error", f"Failed to open output directory:\n{output_dir}\n\n{e}")

    # =======================
    # 🧾 Runs tab
    # =======================
    def _connect_runs_page(self):
        self.runsList.currentRowChanged.connect(self._show_selected_run_details)
        self.runsOpenOutputButton.clicked.connect(self._open_selected_run_output)
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

    def _open_selected_run_output(self):
        rec = self._selected_run_record()
        if not rec:
            return
        # reuse the same "Open Output Directory" behavior, but ideally open job subfolder later
        self._open_output_directory()

    def _load_selected_run_to_alphafold(self):
        rec = self._selected_run_record()
        if not rec:
            return

        # switch to alphafold page
        self.pagesStack.setCurrentIndex(2)
        self.navList.setCurrentRow(2)

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
            self._set_combo_by_data(e.ptm, p.get("modification"))
            e.ptm_pos.setText(p.get("mod_position", "") or "")

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
            "platform": (self.cfgPlatform.currentText() or "wsl").strip(),
            "wsl_distro": self.cfgWslDistro.text().strip(),
            "gui_dir": self.cfgGuiDir.text().strip(),
            "gui_cache_subdir": self.cfgGuiCacheSubdir.text().strip(),
            "cache_root": str(Path.home() / ".af3_pipeline" / "cache"),
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

            # Update GUI-local cache directory if gui_dir/subdir changed
            gui_dir = Path(self.cfgGuiDir.text().strip())
            subdir  = self.cfgGuiCacheSubdir.text().strip()
            new_cache = norm_path(gui_dir / subdir)
            new_cache.mkdir(parents=True, exist_ok=True)

            self.log(f"⚙️ Applied config (saved to YAML). GUI cache now: {new_cache}")
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
        # match your config loader preference (repo-local first)
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

        self._set_combo_text_force(self.cfgPlatform, str(g("platform", "")))
        self.cfgWslDistro.setText(str(g("wsl_distro", "")))
        self.cfgLinuxHomeRoot.setText(str(g("linux_home_root", "")))

        self.cfgGuiDir.setText(str(g("gui_dir", "")))
        self.cfgGuiCacheSubdir.setText(str(g("gui_cache_subdir", "")))

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

    try:
        _ = cfg.get("af3_output_dir")
    except Exception as e:
        QMessageBox.critical(
            None,
            "Configuration Error",
            f"Failed to load AF3 configuration:\n{e}",
        )
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
