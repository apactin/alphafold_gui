from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QWidget,
    QLabel,
    QCheckBox,
    QTableWidget,
    QHeaderView,
    QTableWidgetItem,
    QMessageBox,
)

def _msg_info(parent, title, text):
    QMessageBox.information(parent, title, text)

def _msg_warn(parent, title, text):
    QMessageBox.warning(parent, title, text)

def _msg_err(parent, title, text):
    QMessageBox.critical(parent, title, text)

def _guess_selected_job_folder(self) -> Optional[Path]:
    """
    Runs history stores spec['jobname'] (often base name), but the bundled job folder
    under ~/.af3_pipeline/jobs is usually the *timestamped* AF3 output folder name.

    We try:
    1) exact match of a folder named jobname
    2) newest folder that startswith jobname + '_' (timestamped)
    3) newest folder that startswith jobname (fallback)
    """
    rec = self._selected_run_record()
    if not rec:
        return None

    jobname = (rec.get("jobname") or "").strip()
    if not jobname:
        return None

    root = self._jobs_root_dir()
    if not root.exists():
        return None

    # 1) exact match
    exact = root / jobname
    if exact.exists() and exact.is_dir():
        return exact

    # 2) common timestamped naming: <jobname>_YYYYMMDD_HHMMSS
    cand = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(jobname + "_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cand:
        return cand[0]

    # 3) fallback: any folder starting with jobname
    cand2 = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(jobname)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cand2[0] if cand2 else None


def _find_parent_with_layout(w: QWidget) -> Optional[QWidget]:
    """
    Walk upward from widget until we find a QWidget that has a layout().
    This is much more reliable than assuming runDetailsText.parentWidget() has a layout.
    """
    cur: Optional[QWidget] = w
    for _ in range(16):
        if cur is None:
            return None
        try:
            if cur.layout() is not None:
                return cur
        except Exception:
            pass
        cur = cur.parentWidget()
    return None


def _install_runs_toolbar_v2(self) -> None:
    """
    Install a 2-row Runs toolbar without touching the .ui file.

    IMPORTANT: We do NOT reparent existing .ui buttons (that can dump them at (0,0)
    if insertion fails). Instead we:
      - hide the old .ui buttons
      - create new toolbar buttons
      - insert the toolbar into the layout that contains the old load button
    """
    if hasattr(self, "runsSearchInput") or hasattr(self, "runsRefreshButton") or hasattr(self, "viewerDropdown"):
        return
    # If a broken toolbar exists from a previous attempt, remove it cleanly
    old = getattr(self, "_runsToolbarV2", None)
    if old is not None:
        try:
            old.setParent(None)
            old.deleteLater()
        except Exception:
            pass
        self._runsToolbarV2 = None

    # ---------- helpers ----------
    def _hide_widget(attr: str) -> None:
        w = getattr(self, attr, None)
        try:
            if w is not None:
                w.setVisible(False)
        except Exception:
            pass

    def _find_layout_containing(w) -> tuple[Optional[object], Optional[object], int]:
        """
        Walk up parents until we find a parent.layout() that contains widget w.
        Returns (layout, container_widget, index) or (None,None,-1).
        """
        if w is None:
            return (None, None, -1)

        cur = w
        for _ in range(12):
            parent = cur.parentWidget() if hasattr(cur, "parentWidget") else None
            if parent is None:
                break
            lay = parent.layout()
            if lay is not None:
                try:
                    idx = lay.indexOf(w)
                except Exception:
                    idx = -1
                if idx >= 0:
                    return (lay, parent, idx)
            cur = parent
        return (None, None, -1)

    # ---------- hide legacy UI buttons ----------
    # These are the ones that tend to exist in your .ui
    for nm in (
        "runsLoadToAlphafoldButton",
        "runsOpenSelectedButton",
        "runsOpenMetricsButton",
        "runsAnalyzeButton",
        "runsOpenOutputButton",
        "runsOpenRelaxButton",
    ):
        _hide_widget(nm)

    # Also hide the stray checkbox if it's from UI and you want it in the toolbar
    # (we'll create our own checkbox)
    # If you actually want to keep the UI checkbox, comment this out.
    _hide_widget("runsMultiSeedCheckbox")

    # ---------- build toolbar ----------
    self._runsToolbarV2 = QWidget(self)
    self._runsToolbarV2.setObjectName("runsToolbarV2")

    v = QVBoxLayout(self._runsToolbarV2)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(6)

    # Row 1
    row1 = QHBoxLayout()
    row1.setContentsMargins(0, 0, 0, 0)
    row1.setSpacing(8)

    self._runs_btn_load = QPushButton("Load into AlphaFold", self._runsToolbarV2)
    self._runs_btn_load.setObjectName("runsToolbarLoadButton")

    self._runs_btn_open_selected = QPushButton("Open selected run", self._runsToolbarV2)
    self._runs_btn_open_selected.setObjectName("runsToolbarOpenSelectedButton")

    self._runs_btn_open_metrics = QPushButton("Open metrics", self._runsToolbarV2)
    self._runs_btn_open_metrics.setObjectName("runsToolbarOpenMetricsButton")

    self._runs_btn_analyze = QPushButton("Run analysis", self._runsToolbarV2)
    self._runs_btn_analyze.setObjectName("runsToolbarAnalyzeButton")

    self._runs_chk_multiseed = QCheckBox("Multi-seed", self._runsToolbarV2)
    self._runs_chk_multiseed.setObjectName("runsToolbarMultiSeedCheckbox")

    self._runs_chk_relax = QCheckBox("RosettaRelax", self._runsToolbarV2)
    self._runs_chk_relax.setObjectName("runsRelaxCheckBox")
    self._runs_chk_relax.setChecked(True)

    self._runs_chk_ligand = QCheckBox("RosettaLigand", self._runsToolbarV2)
    self._runs_chk_ligand.setObjectName("runsLigandCheckBox")
    self._runs_chk_ligand.setChecked(True)

    row1.addWidget(self._runs_btn_load)
    row1.addWidget(self._runs_btn_open_selected)
    row1.addWidget(self._runs_btn_open_metrics)
    row1.addStretch(1)
    row1.addWidget(QLabel("Analyze run:", self._runsToolbarV2))
    row1.addWidget(self._runs_btn_analyze)
    row1.addWidget(self._runs_chk_multiseed)
    row1.addWidget(self._runs_chk_relax)
    row1.addWidget(self._runs_chk_ligand)
    
    

    # Row 2
    row2 = QHBoxLayout()
    row2.setContentsMargins(0, 0, 0, 0)
    row2.setSpacing(8)

    self.runsViewerDropdown = QComboBox(self._runsToolbarV2)
    self.runsViewerDropdown.setObjectName("runsViewerDropdown")
    self.runsViewerDropdown.addItems(["ChimeraX", "PyMOL"])
    self.runsViewerDropdown.setMinimumWidth(130)

    self.runsOpenAFStructButton = QPushButton("Open AlphaFold", self._runsToolbarV2)
    self.runsOpenAFStructButton.setObjectName("runsOpenAFStructButton")

    self.runsOpenRosettaStructButton = QPushButton("Open Rosetta", self._runsToolbarV2)
    self.runsOpenRosettaStructButton.setObjectName("runsOpenRosettaStructButton")

    self.runsOpenBothStructButton = QPushButton("Open both", self._runsToolbarV2)
    self.runsOpenBothStructButton.setObjectName("runsOpenBothStructButton")

    row2.addWidget(QLabel("Viewer:", self._runsToolbarV2))
    row2.addWidget(self.runsViewerDropdown)
    row2.addSpacing(12)
    row2.addWidget(self.runsOpenAFStructButton)
    row2.addWidget(self.runsOpenRosettaStructButton)
    row2.addWidget(self.runsOpenBothStructButton)
    row2.addStretch(1)

    v.addLayout(row1)
    v.addLayout(row2)

    # ---------- insert into correct layout ----------
    # Anchor on the *existing* UI load button if it exists (even though it's hidden)
    anchor = getattr(self, "runsLoadToAlphafoldButton", None)
    if anchor is None:
        anchor = getattr(self, "runsOpenSelectedButton", None)
    if anchor is None:
        anchor = getattr(self, "runDetailsText", None)

    lay, _container, idx = _find_layout_containing(anchor)

    if lay is None:
        # Fallback: try inserting into runDetailsText parent layout
        try:
            p = self.runDetailsText.parentWidget()
            lay = p.layout() if p is not None else None
            idx = 0
        except Exception:
            lay = None

    if lay is None:
        # Absolute last resort: just add to runs page widget if we can
        # (won't overlap if that page has a layout)
        try:
            self._runsToolbarV2.setParent(self.runsList.parentWidget())
            self.runsList.parentWidget().layout().addWidget(self._runsToolbarV2)  # type: ignore[union-attr]
        except Exception:
            # If even this fails, at least don't leave it floating
            self._runsToolbarV2.setParent(self)
            self._runsToolbarV2.move(10, 10)
    else:
        try:
            lay.insertWidget(max(0, idx), self._runsToolbarV2)
        except Exception:
            # If insertWidget not supported, fall back to addWidget
            try:
                lay.addWidget(self._runsToolbarV2)
            except Exception:
                self._runsToolbarV2.setParent(self)
                self._runsToolbarV2.move(10, 10)

    # ---------- wire actions to your existing methods ----------
    # Load into AlphaFold: call the existing slot (uses selected run record)
    self._runs_btn_load.clicked.connect(self._load_selected_run_to_alphafold)

    # Open selected folder / metrics / analysis: use your existing methods from runs_features
    self._runs_btn_open_selected.clicked.connect(self._runs_open_selected_run_folder)
    self._runs_btn_open_metrics.clicked.connect(self._runs_open_selected_metrics_csv)

    # Store checkbox where your analysis method expects it
    self.runsMultiSeedCheckbox = self._runs_chk_multiseed
    self.runsRelaxCheckBox = self._runs_chk_relax
    self.runsLigandCheckBox = self._runs_chk_ligand

    self._runs_btn_analyze.clicked.connect(self._runs_run_selected_analysis)

    # Viewer open buttons
    self.runsOpenAFStructButton.clicked.connect(lambda: self._runs_open_structure(which="af"))
    self.runsOpenRosettaStructButton.clicked.connect(lambda: self._runs_open_structure(which="rosetta"))
    self.runsOpenBothStructButton.clicked.connect(lambda: self._runs_open_structure(which="both"))



def _runs_open_structure(self, *, which: str) -> None:
    job_dir = self._guess_selected_job_folder()
    if not job_dir:
        _msg_warn(self, "No selection", "Select a run first.")
        return

    af = self._runs_find_af_structure(job_dir)
    ro = self._runs_find_rosetta_structure(job_dir)

    if which == "af":
        if not af:
            _msg_warn(self, "Not found", f"No AlphaFold structure found under:\n{job_dir}")
            return
        paths = [af]
        align = False
    elif which == "rosetta":
        if not ro:
            _msg_warn(self, "Not found", f"No Rosetta relaxed structure found under:\n{job_dir}")
            return
        paths = [ro]
        align = False
    else:
        # both
        if not af or not ro:
            _msg_warn(self, "Not found", f"Need both structures.\n\nAlphaFold: {af}\nRosetta: {ro}")
            return
        paths = [af, ro]
        align = True

    dd = getattr(self, "viewerDropdown", None) or getattr(self, "runsViewerDropdown", None)
    viewer = (dd.currentText() if dd else "").strip().lower()
    if "pymol" in viewer:
        self._open_structures_in_pymol(paths=paths, align=align)
    else:
        self._open_structures_in_chimerax(paths=paths, align=align)


def _runs_open_selected_run_folder(self):
    job_dir = self._guess_selected_job_folder()
    if not job_dir:
        _msg_warn(self, "No selection", "Select a run first (and make sure the bundled job folder exists).")
        return
    self._open_path_in_file_manager(job_dir)


def _runs_open_selected_relax_folder(self):
    job_dir = self._guess_selected_job_folder()
    if not job_dir:
        _msg_warn(self, "No selection", "Select a run first.")
        return
    relax = self._runs_find_relax_dir(job_dir)
    if not relax:
        _msg_warn(self, "No relax folder", f"No rosetta_relax_* folder found in:\n{job_dir}")
        return
    self._open_path_in_file_manager(relax)


def _runs_open_selected_metrics_csv(self):
    job_dir = self._guess_selected_job_folder()
    if not job_dir:
        _msg_warn(self, "No selection", "Select a run first.")
        return
    m = self._runs_find_metrics_csv(job_dir)
    if not m:
        _msg_warn(self, "No metrics found", f"No metrics CSV found under:\n{job_dir}")
        return
    # Open the file in OS default app (Excel, etc.)
    if sys.platform.startswith("win"):
        os.startfile(str(m))  # type: ignore[attr-defined]
    else:
        self._open_path_in_file_manager(m.parent)


def _runs_run_selected_analysis(self):
    job_dir = self._guess_selected_job_folder()
    if not job_dir:
        _msg_warn(self, "No selection", "Select a run first.")
        return

    multi = False
    # Prefer the toolbar-owned checkbox (which we always create if missing)
    try:
        if hasattr(self, "_runs_chk_multiseed") and self._runs_chk_multiseed is not None:
            multi = bool(self._runs_chk_multiseed.isChecked())
        elif hasattr(self, "runsMultiSeedCheckbox"):
            multi = bool(self.runsMultiSeedCheckbox.isChecked())
    except Exception:
        multi = False
        
    do_relax = True
    do_ligand = True

    try:
        if hasattr(self, "runRelaxcheckbox") and self.runRelaxcheckbox is not None:
            do_relax = bool(self.runRelaxcheckbox.isChecked())
        elif hasattr(self, "runsRelaxCheckBox") and self.runsRelaxCheckBox is not None:
            do_relax = bool(self.runsRelaxCheckBox.isChecked())
    except Exception:
        do_relax = True

    try:
        if hasattr(self, "runLigandCheckbox") and self.runLigandCheckbox is not None:
            do_ligand = bool(self.runLigandCheckbox.isChecked())
        elif hasattr(self, "runsLigandCheckBox") and self.runsLigandCheckBox is not None:
            do_ligand = bool(self.runsLigandCheckBox.isChecked())
    except Exception:
        do_ligand = True

    skip_rosetta = (not do_relax)
    skip_rosetta_ligand = (not do_ligand)

    # Run your existing AnalysisWorker, but pass the *job folder name* (what post_analysis expects)
    job_folder_name = job_dir.name
    self.log(
        f"🧠 RUNS: post-analysis for '{job_folder_name}' "
        f"(multi_seed={multi}, relax={do_relax}, ligand={do_ligand})"
    )

    self.analysis_thread = AnalysisWorker(
        job_folder_name,
        multi,
        skip_rosetta=skip_rosetta,
        skip_rosetta_ligand=skip_rosetta_ligand,
    )
    self.analysis_thread.log.connect(self.log)
    self.analysis_thread.done.connect(lambda: self._runs_after_analysis_refresh(job_dir))
    self.analysis_thread.error.connect(
        lambda m: (_msg_err(self, "Analysis Error", m), self.log(f"❌ Analysis failed:\n{m}"))
    )
    self.analysis_thread.start()


def _runs_find_relax_dir(self, job_dir: Path) -> Optional[Path]:
    relax = sorted(
        [p for p in job_dir.glob("rosetta_relax_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return relax[0] if relax else None


def _runs_find_relax_scripts_dir(self, job_dir: Path) -> Optional[Path]:
    relax = sorted(
        [p for p in job_dir.glob("rosetta_scripts_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return relax[0] if relax else None


def _runs_find_metrics_csv(self, job_dir: Path) -> Optional[Path]:
    # Single-job mode
    m1 = job_dir / "metrics_summary.csv"
    if m1.exists():
        return m1

    # Sometimes written under relax dir if you run metrics there
    r = self._runs_find_relax_dir(job_dir)
    if r:
        m2 = r / "metrics_summary.csv"
        if m2.exists():
            return m2

    # Multi-seed mode summary (commonly under rosetta_runs/rosetta_run_*/multi_seed_metrics_summary.csv)
    runs_root = job_dir / "rosetta_runs"
    if runs_root.exists():
        run_folders = sorted(
            [p for p in runs_root.glob("rosetta_run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if run_folders:
            m3 = run_folders[0] / "multi_seed_metrics_summary.csv"
            if m3.exists():
                return m3

    return None


def _runs_find_af_structure(self, job_dir: Path) -> Optional[Path]:
    """
    Best-effort: find the primary AF3 structure in the job root.
    Prefers *model*.cif, then any .cif, then .pdb.
    Ignores rosetta_relax_* and rosetta_runs folders.
    """
    if not job_dir.exists():
        return None

    def allowed(p: Path) -> bool:
        parts = set(p.parts)
        if any("rosetta_relax_" in x for x in parts):
            return False
        if "rosetta_runs" in parts:
            return False
        return True

    cifs = [p for p in job_dir.rglob("*.cif") if p.is_file() and allowed(p)]
    pdbs = [p for p in job_dir.rglob("*.pdb") if p.is_file() and allowed(p)]

    # Prefer "model" cifs
    model_cifs = [p for p in cifs if "model" in p.name.lower()]
    for pool in (model_cifs, cifs, pdbs):
        if pool:
            pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return pool[0]
    return None


def _runs_find_rosetta_structure(self, job_dir: Path) -> Optional[Path]:
    """
    Best-effort: find the minimized/relaxed structure in the newest rosetta_ligand_* folder.
    Prefers *relax*/*.pdb, then any pdb/cif.
    """
    rosettaLigand = self._runs_find_relax_scripts_dir(job_dir)
    rosettaRelax = self._runs_find_relax_dir(job_dir)
    if rosettaLigand:
        pdbs = sorted(rosettaLigand.rglob("*.pdb"), key=lambda p: p.stat().st_mtime, reverse=True)
        cifs = sorted(rosettaLigand.rglob("*.cif"), key=lambda p: p.stat().st_mtime, reverse=True)
    elif rosettaRelax:
        pdbs = sorted(rosettaRelax.rglob("*.pdb"), key=lambda p: p.stat().st_mtime, reverse=True)
        cifs = sorted(rosettaRelax.rglob("*.cif"), key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        return None

    prefer = [p for p in pdbs if "relax" in p.name.lower() or "relaxed" in p.name.lower()]
    if prefer:
        return prefer[0]
    if pdbs:
        return pdbs[0]
    if cifs:
        return cifs[0]
    return None


def _install_runs_history_table(self) -> None:
    """
    Replace runsList (QListWidget) with a QTableWidget at runtime (no .ui edits).
    Keeps selection semantics: selected row index == index into self.runs_history.
    """
    if hasattr(self, "runsHistoryTable"):
        return

    # Find the layout containing runsList
    anchor = getattr(self, "runsList", None)
    if anchor is None:
        return

    parent = anchor.parentWidget()
    lay = parent.layout() if parent else None
    if lay is None:
        return

    idx = lay.indexOf(anchor)
    if idx < 0:
        return

    # Hide the old list (do NOT delete it; other code may still reference it)
    anchor.setVisible(False)

    # Create table
    t = QTableWidget(parent)
    t.setObjectName("runsHistoryTable")
    t.setColumnCount(2)
    t.setHorizontalHeaderLabels(["Job", "Finished"])

    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)

    hdr = t.horizontalHeader()
    hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)           # Job
    hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # Finished

    # Insert into same spot
    lay.insertWidget(idx, t)

    t.setMinimumHeight(175) 

    # Save + wire selection
    self.runsHistoryTable = t
    self.runsHistoryTable.currentCellChanged.connect(
        lambda row, _col, _pr, _pc: self._show_selected_run_details(row)
    )


def _runs_selected_row(self) -> int:
    """
    Unified selection getter for Runs history list/table.
    """
    if hasattr(self, "runsHistoryTable"):
        return int(self.runsHistoryTable.currentRow())
    return int(self.runsList.currentRow())


def _runs_set_selected_row(self, row: int) -> None:
    """
    Unified selection setter.
    """
    if hasattr(self, "runsHistoryTable"):
        if row >= 0:
            self.runsHistoryTable.selectRow(row)
            self.runsHistoryTable.setCurrentCell(row, 0)
        return
    self.runsList.setCurrentRow(row)


def _format_run_timestamp_for_list(self, ts_val) -> str:
    """
    Pretty timestamp for the table column (uses your existing timezone behavior).
    """
    from datetime import datetime

    if not ts_val:
        return "—"

    if isinstance(ts_val, datetime):
        dt = ts_val
    else:
        s = str(ts_val).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            return str(ts_val)

    try:
        if dt.tzinfo is not None:
            dt = dt.astimezone()
    except Exception:
        pass

    # e.g. "Feb 5, 2026 2:16 PM"
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
