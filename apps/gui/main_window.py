#!/usr/bin/env python3
import os
import sys
import json
import random
import subprocess
import traceback
import glob
import re
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QApplication,
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QPlainTextEdit, QLineEdit, QComboBox, QInputDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# === Configuration ===
from af3_pipeline.config import cfg

# === Backend ===
from af3_pipeline import json_builder, runner, cache_utils
from af3_pipeline.ligand_utils import prepare_ligand_from_smiles, _canonical_smiles

# ============================================
# ✅ Always write to the true WSL path (config-aware)
# ============================================
def wsl_path(subpath: str = "") -> Path:
    """Return a Windows-accessible UNC path or native Linux path based on config."""
    af3_base = Path(cfg.get("af3_output_dir", "/home/olive/Repositories/alphafold/af_output")).parent
    base = str(af3_base)

    if cfg.get("platform") == "wsl":
        base_win = base.replace("/", "\\")
        wsl_root = f"\\\\wsl.localhost\\{WSL_DISTRO}{base_win}"
        return Path(wsl_root + ("\\" + subpath.replace("/", "\\")) if subpath else wsl_root)
    else:
        return Path(base) / subpath

# ==========================
# 📁 Paths & Cache (config-driven)
# ==========================
def norm_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()

GUI_CACHE_DIR = norm_path(cfg.get("gui_cache_dir", str(Path.home() / ".af3_gui_cache")))
GUI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_CACHE_FILE = GUI_CACHE_DIR / "sequence_cache.json"
LIGAND_CACHE_FILE   = GUI_CACHE_DIR / "ligand_cache.json"
QUEUE_FILE          = GUI_CACHE_DIR / "job_queue.json"
NOTES_FILE          = GUI_CACHE_DIR / "notes.txt"

AF3_CACHE    = norm_path(cfg.get("ligand_cache_dir", str(Path.home() / ".cache/af3_pipeline")))
LIGAND_CACHE = AF3_CACHE / "ligands"
AF_OUTPUT_DIR = norm_path(cfg.get("af3_output_dir", "/home/olive/Repositories/alphafold/af_output"))

WSL_DISTRO = cfg.get("wsl_distro", "Ubuntu-22.04")

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
ROSETTA_BOND_TYPES   = ["thioether", "amide", "ester"]
ION_CHOICES          = ["MG", "CA", "ZN", "NA", "K", "CL", "MN", "FE", "CO", "CU"]
COFACTORS_CHOICES    = ["ATP", "ADP", "AMP", "NAD", "NADP", "FAD", "CoA", "SAM", "GTP", "GDP"]
WARHEADS = [
    "internal alkyne", "acrylamide", "acrylonitrile", "squarate ester", "vinyl sulfone",
    "vinyl sulfonamide", "maleimide", "chloroacetamide", "bromoacetamide",
    "sulfonyl fluoride", "sulfonyl chloride", "epoxide", "aziridine",
    "β-lactam", "aldehyde", "salicyl aldehyde", "ketone", "nitrile", "isothiocyanate",
    "isocyanate", "boronic acid", "alkyne", "azide"
]

# ==========================
# 🧭 Utilities
# ==========================
def to_windows_path(wsl_path: str | Path) -> Path:
    p = str(wsl_path)
    if not p.startswith("/"):
        return Path(p)
    return Path(f"\\\\wsl.localhost\\{WSL_DISTRO}" + p.replace("/", "\\"))

def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else (default or {})
    except Exception:
        return default or {}

def _write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def select_multiple(parent, title, items):
    dlg = QDialog(parent); dlg.setWindowTitle(title)
    layout = QVBoxLayout(dlg)
    lst = QListWidget(dlg)
    for val in items:
        itm = QListWidgetItem(val); itm.setCheckState(Qt.Unchecked)
        lst.addItem(itm)
    layout.addWidget(lst)
    ok = QPushButton("OK", dlg); ok.clicked.connect(dlg.accept)
    layout.addWidget(ok)
    if dlg.exec_():
        selected = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]
        return ",".join(selected)
    return ""

# ==========================
# 🗒 Notes
# ==========================
class NotesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🗒 Notes")
        self.resize(500, 400)
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

    def __init__(self, jobname, proteins, rna, dna, ligand):
        super().__init__()
        self.jobname = jobname
        self.proteins, self.rna, self.dna, self.ligand = proteins, rna, dna, ligand

    def run(self):
        try:
            def _s(x): return (x or "").strip() if isinstance(x, (str, type(None))) else x
            if isinstance(self.ligand, dict):
                for key in ("smiles","chain","residue","prot_atom","ligand_atom","ions","cofactors"):
                    if key in self.ligand: self.ligand[key] = _s(self.ligand[key])
                self.ligand["covalent"] = bool(self.ligand.get("covalent", False))

            def progress_hook(msg): self.progress.emit(msg)
            json_builder._progress_hook = progress_hook

            json_path = json_builder.build_input(
                jobname=self.jobname,
                proteins=self.proteins,
                rna=self.rna,
                dna=self.dna,
                ligand=self.ligand
            )
            self.finished.emit(json_path)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if hasattr(json_builder, "_progress_hook"):
                del json_builder._progress_hook

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
            # Pass flag to runner — triggers automatic post-AF3 analysis if checked
            runner.run_af3(
                self.json_path,
                job_name=self.job_name,
                auto_analyze=self.auto_analyze,
                multi_seed = self.multi_seed
            )
            self.finished.emit(self.json_path)
        except Exception as e:
            self.failed.emit(str(e))

class AnalysisWorker(QThread):
    """Non-blocking unified post-AF3 analysis (runs post_analysis.py sequence)."""
    log = pyqtSignal(str)
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, job_dir: Path, multi_seed: bool = False):
        super().__init__()
        self.job_dir = Path(job_dir)
        self.multi_seed = multi_seed

    def _run_step(self, cmd, label):
        self.log.emit(f"▶ {label}\n$ {' '.join(cmd)}\n")
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace"  # prevents crash on invalid UTF-8
            ) as p:
                for line in p.stdout:
                    self.log.emit(line.rstrip("\n"))
                rc = p.wait()
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
        except Exception as e:
            raise RuntimeError(f"{label} failed: {e}")

    def run(self):
        try:
            # 🧠 Run unified post-analysis pipeline (build_meta → minimize → metrics)
            cmd_post = [
                "python", "-m", "af3_pipeline.analysis.post_analysis",
                "--job", str(self.job_dir.name),
            ]
            if self.multi_seed:
                cmd_post.append("--multi_seed")
            self._run_step(cmd_post, f"Post-AF3 analysis for job '{self.job_dir.name}'")
            self.done.emit()

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n{tb}")

# ==========================
# 🪟 Main Window
# ==========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), "main_window.ui")
        uic.loadUi(ui_path, self)

        # Caches
        self.sequence_cache = _read_json(SEQUENCE_CACHE_FILE, {})
        self.ligand_cache   = _read_json(LIGAND_CACHE_FILE, {})

        # Atom map & state
        self.ATOM_MAP = {
            "Cysteine (SG)": "SG",
            "Lysine (NZ)": "NZ",
            "Tyrosine (OH)": "OH",
            "Histidine (ND1)": "ND1",
            "Histidine (NE2)": "NE2"
        }
        self.custom_protein_atom = None

        # Signals
        self._connect_all_signals()
        self.notesButton.clicked.connect(self._open_notes_dialog)

        self.covalentCheckbox.toggled.connect(self.update_covalent_fields)
        self.update_covalent_fields()

        # Populate initial UI
        self._populate_dropdowns()
        self._populate_saved_sequences()

        # Threads/state
        self.run_thread = None
        self.build_thread = None
        self.is_running = False
        self._update_run_button_label()

    # =======================
    # 🔗 Signals
    # =======================
    def _connect_all_signals(self):
        # Core buttons
        self.runButton.clicked.connect(self.run_from_queue_or_form)
        self.openOutputButton.clicked.connect(self._open_output_directory)
        self.addToQueueButton.clicked.connect(self.add_to_queue)
        self.openQueueButton.clicked.connect(self.open_queue_dialog)
        self.autoSeedButton.clicked.connect(self._auto_generate_seeds)
        self.rosettaButton.clicked.connect(self._run_post_af3_analysis)
        self.notesButton.clicked.connect(self._open_notes_dialog)

        # Proteins / RNA / DNA
        self._connect_protein_signals()
        self._connect_rna_dna_signals()
        self._connect_delete_buttons()

        # Ligand
        self.proteinAtomComboBox.currentTextChanged.connect(self._on_protein_atom_change)
        self.saveSmiles.clicked.connect(self.save_ligand)
        self.deleteSmiles.clicked.connect(lambda: self._delete_selected_item(self.smilesDropdown, LIGAND_CACHE_FILE))
        self.smilesDropdown.currentTextChanged.connect(self.load_ligand)
        self.openChimeraButton.clicked.connect(self._open_ligand_in_chimerax)
        self.editIonsButton.clicked.connect(lambda: self._choose_multi("ions"))
        self.editCofactorsButton.clicked.connect(lambda: self._choose_multi("cofactors"))

        # Save biomolecule (now on left under DNA)
        self.saveBioButton.clicked.connect(self._save_biomolecule_entry)

    # =======================
    # 🔽 Populate UI
    # =======================
    def _populate_dropdowns(self):
        # PTM dropdowns
        for widget_name in [
            "modificationDropdown1", "modificationDropdown2", "modificationDropdown3",
            "rnaModificationDropdown", "dnaModificationDropdown"
        ]:
            dd = getattr(self, widget_name)
            dd.clear()
            for label, ccd in PTM_CHOICES.items():
                dd.addItem(label, ccd)

        # Ligand smiles dropdown
        self.smilesDropdown.clear()
        self.smilesDropdown.addItem("Select saved…")
        self.smilesDropdown.addItems(sorted(self.ligand_cache.keys()))

        # Ions/Cofactors placeholders
        if hasattr(self, "ionsDropdown") and isinstance(self.ionsDropdown, QLineEdit):
            self.ionsDropdown.setPlaceholderText("e.g. MG,ZN,CA")
        if hasattr(self, "cofactorDropdown") and isinstance(self.cofactorDropdown, QLineEdit):
            self.cofactorDropdown.setPlaceholderText("e.g. ATP,FAD,NAD")


    def _populate_saved_sequences(self):
        cache = self.sequence_cache

        def names_for(kind):
            out = []
            for key, val in cache.items():
                if isinstance(val, dict):
                    t = (val.get("type") or "").strip().lower()
                    if kind.lower() == t:
                        out.append(key)
                else:
                    out.append(key)  # legacy
            return sorted(set(out))

        protein_names = names_for("Protein")
        rna_names     = names_for("RNA")
        dna_names     = names_for("DNA")

        for i in range(1, 4):
            dd = getattr(self, f"proteinDropdown{i}")
            dd.clear(); dd.addItem("Select saved…"); dd.addItems(protein_names)

        self.rnaDropdown.clear(); self.rnaDropdown.addItem("Select saved…"); self.rnaDropdown.addItems(rna_names)
        self.dnaDropdown.clear(); self.dnaDropdown.addItem("Select saved…"); self.dnaDropdown.addItems(dna_names)

    def _sanitize_jobname(self, s: str) -> str:
        return (s or "").strip().replace("+", "_")

    # =======================
    # 🧬 Protein/RNA/DNA signals
    # =======================
    def _connect_protein_signals(self):
        for i in range(1, 4):
            getattr(self, f"proteinDropdown{i}").currentTextChanged.connect(
                lambda text, idx=i: self.load_protein_sequence(idx, text)
            )

    def _connect_rna_dna_signals(self):
        self.rnaDropdown.currentTextChanged.connect(lambda t: self.load_rna_dna_sequence("rna", t))
        self.dnaDropdown.currentTextChanged.connect(lambda t: self.load_rna_dna_sequence("dna", t))

    def update_covalent_fields(self):
        """Enable or disable covalent binding fields based on checkbox state."""
        enabled = self.covalentCheckbox.isChecked()
        self.covalentChain.setEnabled(enabled)
        self.covalentResidue.setEnabled(enabled)
        self.proteinAtomComboBox.setEnabled(enabled)
        self.covalentLigandAtom.setEnabled(enabled)

    # =======================
    # 🗑 Delete handlers
    # =======================
    def _delete_entry_from_cache(self, cache_file: Path, key: str):
        if not cache_file.exists() or not key:
            return
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if key in data:
                del data[key]
                _write_json(cache_file, data)
                # Refresh caches and UI
                if cache_file == SEQUENCE_CACHE_FILE:
                    self.sequence_cache = data
                    self._populate_saved_sequences()
                elif cache_file == LIGAND_CACHE_FILE:
                    self.ligand_cache = data
                    self._populate_dropdowns()
                    # Rosetta ligand dropdown no longer needed
        except Exception as e:
            print(f"⚠️ Failed to delete '{key}' from {cache_file}: {e}")

    def _delete_selected_item(self, dropdown, cache_file: Path):
        name = dropdown.currentText().strip()
        if not name or name in {"Select saved…", "Select saved..."}:
            return
        self._delete_entry_from_cache(cache_file, name)
        dropdown.setCurrentIndex(0)

    def _connect_delete_buttons(self):
        self.deleteProtein1.clicked.connect(lambda: self._delete_selected_item(self.proteinDropdown1, SEQUENCE_CACHE_FILE))
        self.deleteProtein2.clicked.connect(lambda: self._delete_selected_item(self.proteinDropdown2, SEQUENCE_CACHE_FILE))
        self.deleteProtein3.clicked.connect(lambda: self._delete_selected_item(self.proteinDropdown3, SEQUENCE_CACHE_FILE))
        self.deleteRNA.clicked.connect(lambda: self._delete_selected_item(self.rnaDropdown, SEQUENCE_CACHE_FILE))
        self.deleteDNA.clicked.connect(lambda: self._delete_selected_item(self.dnaDropdown, SEQUENCE_CACHE_FILE))

    # =======================
    # 🧬 Sequence I/O
    # =======================
    def _get_seq_from_cache(self, key: str) -> str:
        if key not in self.sequence_cache:
            return ""
        val = self.sequence_cache[key]
        if isinstance(val, dict):
            return val.get("sequence", "")
        return str(val)

    def load_protein_sequence(self, idx, name):
        if name in {"Select saved…", "Select saved..."}:
            getattr(self, f"proteinSeq{idx}").setPlainText("")
            return
        seq = self._get_seq_from_cache(name)
        getattr(self, f"proteinSeq{idx}").setPlainText(seq)

    def load_rna_dna_sequence(self, kind, name):
        if name in {"Select saved…", "Select saved..."}:
            getattr(self, f"{kind}Seq").setPlainText("")
            return
        seq = self._get_seq_from_cache(name)
        getattr(self, f"{kind}Seq").setPlainText(seq)

    # =======================
    # 💊 Ligand
    # =======================
    def save_ligand(self):
        smiles_raw = self.smilesInput.text().strip()
        if not smiles_raw:
            QMessageBox.warning(self, "Missing SMILES", "Please enter a SMILES string before saving.")
            return

        smiles = _canonical_smiles(smiles_raw)
        name   = (self.smilesName.text().strip() or smiles).strip()

        try:
            # Generate ligand CIF (already cached by ligand_utils using config)
            cif_path = Path(prepare_ligand_from_smiles(
                smiles,
                name=name,
                skip_if_cached=False,
            )).expanduser()

            if not cif_path.exists():
                raise FileNotFoundError(f"Ligand CIF was not created: {cif_path}")

            ligand_hash = cache_utils.compute_hash(smiles)

            # Save mapping (new format)
            self.ligand_cache[name] = {
                "smiles": smiles,
                "hash": ligand_hash,
                "path": str(cif_path),
            }
            _write_json(LIGAND_CACHE_FILE, self.ligand_cache)

            self._populate_dropdowns()
            self.log(f"✅ Saved ligand '{name}' → {cif_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ligand generation failed:\n{e}")


    def load_ligand(self, name):
        if name in {"Select saved…", "Select saved..."}:
            self.smilesInput.clear()
            return

        entry = self.ligand_cache.get(name)
        if not entry:
            self.smilesInput.clear()
            return

        # Support legacy cache values: name -> "SMILES"
        if isinstance(entry, str):
            self.smilesInput.setText(entry)
            return

        if isinstance(entry, dict):
            self.smilesInput.setText(entry.get("smiles", "") or "")

    def _open_ligand_in_chimerax(self):
        smiles = self.smilesInput.text().strip()
        if not smiles:
            QMessageBox.warning(self, "No Ligand", "Please enter or select a ligand first.")
            return
        canon = _canonical_smiles(smiles)
        lig_hash = cache_utils.compute_hash(canon)
        lig_dir = LIGAND_CACHE / lig_hash
        if not lig_dir.exists():
            QMessageBox.warning(self, "Not Found", f"No cache folder found for this ligand:\n{lig_dir}")
            return

        pdb_candidates = [lig_dir / "LIGAND.pdb", lig_dir / "ligand.pdb"]
        cif_candidates = [lig_dir / "LIGAND.cif", lig_dir / "ligand.cif"]

        pdb_path = next((p for p in pdb_candidates if p.exists()), None)
        cif_path = next((p for p in cif_candidates if p.exists()), None)

        target_path = pdb_path or cif_path
        if not target_path:
            QMessageBox.warning(self, "Not Found", f"No ligand structure files found in {lig_dir}")
            return

        self.log(f"🧩 Opening {target_path.name} in ChimeraX...")
        try:
            chimerax_exe = Path(r"C:\Program Files\ChimeraX\bin\ChimeraX.exe")
            if chimerax_exe.exists():
                subprocess.Popen([str(chimerax_exe), str(target_path)])
            else:
                subprocess.Popen(["wsl", "chimerax", str(target_path)], shell=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open ChimeraX:\n{e}")

    def _open_notes_dialog(self):
        """Open the persistent notes dialog."""
        try:
            dlg = NotesDialog(self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Notes Error", f"Failed to open notes dialog:\n{e}")

    def _choose_multi(self, which):
        if which == "ions":
            selected = select_multiple(self, "Select Ions", ION_CHOICES)
            if selected and isinstance(self.ionsDropdown, QLineEdit):
                self.ionsDropdown.setText(selected)
        elif which == "cofactors":
            selected = select_multiple(self, "Select Cofactors", COFACTORS_CHOICES)
            if selected and isinstance(self.cofactorDropdown, QLineEdit):
                self.cofactorDropdown.setText(selected)

    # =======================
    # 💾 Save Biomolecule (left)
    # =======================
    def _save_biomolecule_entry(self):
        bio_type = self.saveTypeDropdown.currentText().strip()   # Protein | RNA | DNA
        name     = self.saveNameInput.text().strip()
        sequence = self.saveSequenceInput.toPlainText().strip()

        if not name or not sequence:
            QMessageBox.warning(self, "Missing input", "Please provide both a name and a sequence.")
            return

        cache = _read_json(SEQUENCE_CACHE_FILE, {})
        if name in cache:
            reply = QMessageBox.question(self, "Overwrite?",
                                         f"A saved entry named '{name}' already exists.\nOverwrite it?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        cache[name] = {"sequence": sequence, "type": bio_type}
        _write_json(SEQUENCE_CACHE_FILE, cache)
        self.sequence_cache = cache
        self._populate_saved_sequences()
        self.saveNameInput.clear()
        self.saveSequenceInput.clear()
        QMessageBox.information(self, "Saved", f"{bio_type} '{name}' saved successfully!")
        self.log(f"💾 Saved {bio_type}: {name}")

    # =======================
    # 🎲 Seeds
    # =======================
    def _auto_generate_seeds(self):
        n = self.autoSeedSpin.value()
        seeds = random.sample(range(1, 100000), n)
        self.modelSeedInput.setText(", ".join(str(s) for s in seeds))
        self.log(f"🎲 Auto-generated {n} model seeds: {seeds}")

    def _parse_model_seeds(self):
        txt = self.modelSeedInput.text().strip()
        if not txt:
            return None
        seeds = []
        for part in txt.split(","):
            part = part.strip()
            if part.isdigit():
                seeds.append(int(part))
        return seeds or None

    # =======================
    # 🧮 Normalization
    # =======================
    def _s(self, x): return (x or "").strip()

    def _normalize_ligand(self, lig):
        if not lig: return {}
        lig["smiles"]      = self._s(lig.get("smiles"))
        lig["chain"]       = self._s(lig.get("chain"))
        lig["residue"]     = self._s(lig.get("residue"))
        lig["prot_atom"]   = self._s(lig.get("prot_atom"))
        lig["ligand_atom"] = self._s(lig.get("ligand_atom"))
        lig["ions"]        = self._s(lig.get("ions"))
        lig["cofactors"]   = self._s(lig.get("cofactors"))
        lig["covalent"]    = bool(lig.get("covalent", False))
        return lig

    # =======================
    # 🏃 Build & Run
    # =======================
    def run_pipeline(self):
        if self.is_running:
            self.log("⚠️ A job is already running. Use 'Add to Queue' to queue this one.")
            return

        jobname = self._sanitize_jobname(self.jobNameInput.text())
        if not jobname:
            QMessageBox.warning(self, "Error", "Please enter a job name.")
            return

        self.current_jobname = jobname

        proteins = []
        for i in range(1, 4):
            seq = getattr(self, f"proteinSeq{i}").toPlainText().strip()
            template = getattr(self, f"templatePDB{i}").text().strip()
            mod_ccd = getattr(self, f"modificationDropdown{i}").currentData() or "None"
            pos = getattr(self, f"modPosition{i}").text().strip()
            sel_name = getattr(self, f"proteinDropdown{i}").currentText().strip()
            name = "" if sel_name in {"Select saved…", "Select saved..."} else sel_name
            if seq:
                proteins.append({
                    "name": name, "sequence": seq, "template": template,
                    "modification": mod_ccd, "mod_position": pos
                })

        rna_seq = self.rnaSeq.toPlainText().strip()
        dna_seq = self.dnaSeq.toPlainText().strip()
        rna_mod = self.rnaModificationDropdown.currentData() or "None"
        dna_mod = self.dnaModificationDropdown.currentData() or "None"
        rna_pos = self.rnaModPosition.text().strip()
        dna_pos = self.dnaModPosition.text().strip()

        # protein reactive atom
        selected_text = self.proteinAtomComboBox.currentText()
        if selected_text == "Custom..." and self.custom_protein_atom:
            protein_atom_code = self.custom_protein_atom
        else:
            protein_atom_code = self.ATOM_MAP.get(selected_text, "")

        ion_txt = self.ionsDropdown.text().strip() if isinstance(self.ionsDropdown, QLineEdit) else ""
        cof_txt = self.cofactorDropdown.text().strip() if isinstance(self.cofactorDropdown, QLineEdit) else ""

        ligand = {
            "smiles": self.smilesInput.text().strip(),
            "covalent": self.covalentCheckbox.isChecked(),
            "chain": self.covalentChain.text().strip(),
            "residue": self.covalentResidue.text().strip(),
            "prot_atom": protein_atom_code,
            "ligand_atom": self.covalentLigandAtom.text().strip(),
            "ions": "" if ion_txt == "None" else ion_txt,
            "cofactors": "" if cof_txt == "None" else cof_txt
        }

        seeds = self._parse_model_seeds()
        if seeds:
            ligand["modelSeeds"] = seeds

        if not proteins and not rna_seq and not dna_seq and not ligand.get("smiles"):
            QMessageBox.warning(self, "Error", "No inputs provided (protein/nucleic acid/ligand).")
            return

        self.runButton.setEnabled(False)
        self.is_running = True
        self.log(f"🚀 Running job {jobname}...")
        self.build_thread = BuildThread(
            jobname,
            proteins,
            {"sequence": rna_seq, "modification": rna_mod, "pos": rna_pos},
            {"sequence": dna_seq, "modification": dna_mod, "pos": dna_pos},
            ligand
        )
        self.build_thread.progress.connect(self.on_build_progress)
        self.build_thread.finished.connect(self.on_build_finished)
        self.build_thread.failed.connect(self.on_build_failed)
        self.build_thread.start()
        self.log("⚙️ Building AF3 JSON and running MSA in background...")

    # =======================
    # 🪵 Thread callbacks
    # =======================
    def on_build_progress(self, msg): self.log(f"🧬 {msg}")

    def on_build_finished(self, json_path):
        self.log(f"✅ JSON build complete: {json_path}")

        # Capture checkbox state before launching thread
        auto_analyze = getattr(self, "autoAnalyzeCheckbox", None)
        auto_flag = auto_analyze.isChecked() if auto_analyze else False
        multi_seed = getattr(self, "multiSeedCheckbox", None)
        multi_flag = multi_seed.isChecked() if multi_seed else False

        self.run_thread = RunThread(
            json_path,
            getattr(self, "current_jobname", "GUI_job"),
            auto_analyze=auto_flag,
            multi_seed=multi_flag
        )
        self.run_thread.finished.connect(self.on_run_finished)
        self.run_thread.failed.connect(self.on_run_failed)
        self.run_thread.start()

    def on_build_failed(self, err):
        self.log(f"❌ Build failed: {err}")
        self.is_running = False
        self.runButton.setEnabled(True)
        self._update_run_button_label()
        self._maybe_run_next()

    def on_run_finished(self, json_path):
        self.log(f"✅ Run finished: {json_path}")
        self.is_running = False
        self.runButton.setEnabled(True)
        self._update_run_button_label()

        # Extract job name directly from path
        job_name = Path(json_path).stem.replace("_fold_input", "")
        
        # Auto-run post-AF3 analysis if the checkbox is checked
        if hasattr(self, "autoAnalyzeCheckbox") and self.autoAnalyzeCheckbox.isChecked():
            self.log("🧠 Auto-running post-AF3 analysis...")
            self._run_post_af3_analysis(job_name=job_name)
        self._maybe_run_next()



    def on_run_failed(self, err):
        self.log(f"❌ Run failed: {err}")
        self.is_running = False
        self.runButton.setEnabled(True)
        self._update_run_button_label()
        self._maybe_run_next()

    # =======================
    # 🧺 Queue
    # =======================
    def _read_queue(self):
        try:
            raw = json.loads(QUEUE_FILE.read_text()) if QUEUE_FILE.exists() else []
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    def _save_queue(self, q):
        def _json_safe(obj):
            try:
                json.dumps(obj); return obj
            except TypeError:
                return str(obj)
        def _sanitize(o):
            if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
            if isinstance(o, list): return [_sanitize(v) for v in o]
            return _json_safe(o)
        _write_json(QUEUE_FILE, _sanitize(q))

    def _update_run_button_label(self):
        q_len = len(self._read_queue())
        self.runButton.setText(f"▶ Run ({q_len} queued)" if q_len > 0 else "▶ Run")
        self.runButton.setEnabled(True)

    def _normalize_spec(self, spec):
        spec = dict(spec)
        spec.setdefault("jobname", self._s(spec.get("jobname")))
        spec.setdefault("proteins", spec.get("proteins", []))
        spec.setdefault("rna", {"sequence":"","modification":"","pos":""})
        spec.setdefault("dna", {"sequence":"","modification":"","pos":""})
        spec.setdefault("ligand", {
            "smiles":"", "covalent":False, "chain":"", "residue":"",
            "prot_atom":"", "ligand_atom":"", "ions":"", "cofactors":""
        })
        return spec

    def _ensure_job_dict(self, item):
        if isinstance(item, dict):
            return self._normalize_spec(item)
        return self._normalize_spec({
            "jobname": str(item),
            "proteins": [],
            "rna": {"sequence":"","modification":"","pos":""},
            "dna": {"sequence":"","modification":"","pos":""},
            "ligand": {"smiles":"", "covalent":False, "chain":"", "residue":"",
                       "prot_atom":"", "ligand_atom":"", "ions":"", "cofactors":""},
        })

    def _safe_jobname(self, item):
        if isinstance(item, dict):
            return (item.get("jobname") or "").strip() or "(unnamed)"
        return str(item)

    def _current_job_spec(self):
        jobname = self._sanitize_jobname(self.jobNameInput.text())

        proteins = []
        for i in range(1, 4):
            seq = getattr(self, f"proteinSeq{i}").toPlainText().strip()
            template = getattr(self, f"templatePDB{i}").text().strip()
            mod_ccd = getattr(self, f"modificationDropdown{i}").currentData() or "None"
            pos = getattr(self, f"modPosition{i}").text().strip()
            sel_name = getattr(self, f"proteinDropdown{i}").currentText().strip()
            name = "" if sel_name in {"Select saved…","Select saved..."} else sel_name
            if seq:
                proteins.append({"name":name,"sequence":seq,"template":template,"modification":mod_ccd,"mod_position":pos})

        rna_seq = self.rnaSeq.toPlainText().strip()
        dna_seq = self.dnaSeq.toPlainText().strip()
        rna_mod = self.rnaModificationDropdown.currentData() or "None"
        dna_mod = self.dnaModificationDropdown.currentData() or "None"
        rna_pos = self.rnaModPosition.text().strip()
        dna_pos = self.dnaModPosition.text().strip()

        selected_text = self.proteinAtomComboBox.currentText()
        if selected_text == "Custom..." and self.custom_protein_atom:
            protein_atom_code = self.custom_protein_atom
        else:
            protein_atom_code = self.ATOM_MAP.get(selected_text, "")

        ion_txt = self.ionsDropdown.text().strip() if isinstance(self.ionsDropdown, QLineEdit) else ""
        cof_txt = self.cofactorDropdown.text().strip() if isinstance(self.cofactorDropdown, QLineEdit) else ""

        ligand = {
            "smiles": self.smilesInput.text().strip(),
            "covalent": self.covalentCheckbox.isChecked(),
            "chain": self.covalentChain.text().strip(),
            "residue": self.covalentResidue.text().strip(),
            "prot_atom": protein_atom_code,
            "ligand_atom": self.covalentLigandAtom.text().strip(),
            "ions": "" if ion_txt == "None" else ion_txt,
            "cofactors": "" if cof_txt == "None" else cof_txt
        }

        seeds = self._parse_model_seeds()
        if seeds:
            ligand["modelSeeds"] = seeds

        return self._normalize_spec({
            "jobname": jobname,
            "proteins": proteins,
            "rna": {"sequence": rna_seq, "modification": rna_mod, "pos": rna_pos},
            "dna": {"sequence": dna_seq, "modification": dna_mod, "pos": dna_pos},
            "ligand": ligand
        })

    def _enqueue(self, spec):
        spec = self._normalize_spec(spec)
        q = self._read_queue()
        q.append(spec)
        self._save_queue(q)
        self.log(f"🧺 Queued job: {spec.get('jobname','unnamed')}")

    def _dequeue(self):
        q = self._read_queue()
        if not q:
            self._update_run_button_label()
            return None
        item = q.pop(0)
        self._save_queue(q)
        self._update_run_button_label()
        return self._ensure_job_dict(item)

    def add_to_queue(self):
        spec = self._current_job_spec()
        if not spec["jobname"]:
            QMessageBox.warning(self, "Error", "Please enter a job name before queuing.")
            return
        self._enqueue(spec)
        self._clear_form()

    def run_from_queue_or_form(self):
        q = self._read_queue()
        if q:
            self.log(f"📥 Queue detected — running {len(q)} job(s)...")
            if not self.is_running:
                nxt = self._dequeue()
                if nxt:
                    name = self._safe_jobname(nxt)
                    self.log(f"➡️ Starting queued job: {name}")
                    self.run_pipeline_from_spec(nxt)
                else:
                    self.log("⚠️ Queue was empty after dequeue.")
        else:
            self.run_pipeline()

    def _maybe_run_next(self):
        nxt = self._dequeue()
        if not nxt: return
        self.log(f"➡️ Starting next queued job: {self._safe_jobname(nxt)}")
        self.run_pipeline_from_spec(nxt)

    def run_pipeline_from_spec(self, spec):
        spec = self._normalize_spec(spec)
        if not spec.get("jobname"):
            QMessageBox.warning(self, "Error", "Queued job is missing a job name.")
            return
        if self.is_running:
            self._enqueue(spec); return

        self.current_jobname = spec["jobname"]
        self.log(f"🚀 Running job {self.current_jobname} from queue...")
        self.runButton.setEnabled(False); self.runButton.setText("⏳ Running...")

        self.build_thread = BuildThread(
            spec["jobname"], spec["proteins"], spec["rna"], spec["dna"], spec["ligand"]
        )
        self.build_thread.progress.connect(self.on_build_progress)
        self.build_thread.finished.connect(self.on_build_finished)
        self.build_thread.failed.connect(self.on_build_failed)
        self.build_thread.start()
        self.is_running = True

    def open_queue_dialog(self):
        dlg = QDialog(self); dlg.setWindowTitle("Job Queue")
        layout = QVBoxLayout(dlg)
        lst = QListWidget(dlg)
        q = self._read_queue()
        for item in q:
            lst.addItem(self._safe_jobname(item))
        layout.addWidget(lst)

        btns = QHBoxLayout()
        rm = QPushButton("Remove Selected", dlg)
        clr = QPushButton("Clear All", dlg)
        btns.addWidget(rm); btns.addWidget(clr)
        layout.addLayout(btns)

        def _remove():
            rows = sorted({i.row() for i in lst.selectedIndexes()}, reverse=True)
            q = self._read_queue()
            for r in rows:
                if 0 <= r < len(q): q.pop(r)
            self._save_queue(q)
            lst.clear()
            for item in q: lst.addItem(self._safe_jobname(item))
            self._update_run_button_label()

        def _clear():
            self._save_queue([]); lst.clear(); self._update_run_button_label()

        rm.clicked.connect(_remove); clr.clicked.connect(_clear)
        dlg.exec_()

    # =======================
    # 🧽 Clearing & Logging
    # =======================
    def _clear_form(self):
        for w in self.findChildren(QLineEdit): w.clear()
        for w in self.findChildren(QPlainTextEdit): w.clear()
        for w in self.findChildren(QComboBox): w.setCurrentIndex(0)
        if hasattr(self, "covalentCheckbox"): self.covalentCheckbox.setChecked(False)
        self.log("🧹 Form cleared.")

    def log(self, message):
        self.logOutput.appendPlainText(message)
        self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())

    # =======================
    # 📂 Output directory (config-aware)
    # =======================
    def _open_output_directory(self):
        r"""
        Open the AF3 output directory in a platform-appropriate way.

        On Windows: open the WSL UNC path
        \\wsl.localhost\Ubuntu-22.04\home\olive\Repositories\alphafold\af_output

        On Linux/macOS: open the Linux path directly.
        """
        linux_output_dir = Path(cfg.get("af3_output_dir", "/home/olive/Repositories/alphafold/af_output"))

        if sys.platform.startswith("win"):
            linux_str = linux_output_dir.as_posix() 
            unc_str = rf"\\wsl.localhost\{WSL_DISTRO}" + linux_str.replace("/", "\\")
            windows_output_dir = Path(unc_str)
        else:
            windows_output_dir = linux_output_dir

        if windows_output_dir.exists():
            if sys.platform.startswith("win"):
                os.startfile(windows_output_dir)
            elif sys.platform == "darwin":
                subprocess.call(["open", str(windows_output_dir)])
            else:
                subprocess.call(["xdg-open", str(windows_output_dir)])
        else:
            QMessageBox.warning(self, "Not found", f"Output directory not found:\n{windows_output_dir}")


    # =======================
    # 🧪 Protein reactive atom
    # =======================
    def _on_protein_atom_change(self, text):
        if text == "Custom...": 
            atom, ok = QInputDialog.getText(self, "Custom Atom", "Enter atom name (e.g. OG1):")
            self.custom_protein_atom = atom.strip() if ok and atom.strip() else None
        else:
            self.custom_protein_atom = None

    # =======================
    # 🧠 Post-AF3 Analysis (non-blocking)
    # =======================
    def _run_post_af3_analysis(self, job_name=None):
        """Launch post-AF3 analysis. The analysis script itself will now handle
        resolution of timestamped vs. base job folders."""

        if not job_name:
            job_name = (
                self.rosettaPath.text().strip()
                if hasattr(self, "rosettaPath")
                else self.jobNameInput.text().strip()
            )

        if not job_name:
            QMessageBox.warning(self, "Missing job", "Please enter a job name first.")
            return

        # --- Multi-seed flag ---
        multi_seed = getattr(self, "multiSeedCheckbox", None)
        multi_flag = multi_seed.isChecked() if multi_seed else False
        # --- Logging ---
        self.log(f"🧠 Post-AF3 analysis for job '{job_name}'")
        self.log("▶ Passing job name only; model resolution handled downstream.")

        # --- Launch analysis worker ---
        self.analysis_thread = AnalysisWorker(job_name, multi_flag)
        self.analysis_thread.log.connect(self.log)
        self.analysis_thread.done.connect(lambda: self.log("✅ Post-AF3 analysis complete."))
        self.analysis_thread.error.connect(lambda m: (
            self.log(f"❌ Analysis failed:\n{m}"),
            QMessageBox.critical(self, "Analysis Error", m)
        ))
        self.analysis_thread.start()

# ==========================
# 🏁 Entry
# ==========================
if __name__ == "__main__":
    try:
        _ = cfg["af3_output_dir"]
    except Exception as e:
        QMessageBox.critical(None, "Configuration Error", f"Failed to load AF3 configuration:\n{e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
