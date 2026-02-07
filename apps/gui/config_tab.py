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
import yaml
from pathlib import Path
from typing import Any
from af3_pipeline.config import cfg

from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QMessageBox,
    QToolTip,
    QMessageBox,
)

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

def _install_config_help(self):
    """
    Wire your existing help buttons (helpAF3Dir, helpDockerEnv, etc.)
    to show hover tooltips and click-to-open dialogs.

    Also (optionally) attaches the same tooltip to the target input widget.
    """
    help_defs = self._config_help_defs()

    for btn_name, meta in help_defs.items():
        btn = getattr(self, btn_name, None)
        if btn is None:
            continue  # not present in UI

        title = meta["title"]
        text  = meta["text"]
        tooltip_html = f"<b>{title}</b><br>{text}"

        # Tooltip on the ? button itself
        btn.setToolTip(tooltip_html)

        # Click -> modal dialog
        def _make_click_handler(t=title, msg=text):
            return lambda: QMessageBox.information(self, t, msg)
        btn.clicked.connect(_make_click_handler())

        # Hover -> show immediately at cursor
        old_enter = getattr(btn, "enterEvent", None)
        def _enter_event(evt, b=btn, tip=tooltip_html, old=old_enter):
            try:
                QToolTip.showText(QCursor.pos(), tip, b)
            except Exception:
                pass
            if callable(old):
                old(evt)
        btn.enterEvent = _enter_event  # type: ignore

        # OPTIONAL: also apply tooltip to the *target widget*
        target_name = meta.get("target")
        if target_name:
            w = getattr(self, target_name, None)
            if w is not None:
                w.setToolTip(tooltip_html)

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
            "basename": self.cfgLigandBasename.text().strip() or "LIG",
            "name_default": self.cfgLigandNameDefault.text().strip() or "LIG",
            "png_size": [int(self.cfgLigandPngW.value()), int(self.cfgLigandPngH.value())],
            "rdkit_threads": int(self.cfgLigandRdkitThreads.value()),
        },

        "rosetta_relax_bin": self._posixify(self.cfgRosettaRelaxBin.text()),
        "chimera_path": self.cfgChimeraPath.text().strip(),
        "pymol_path": self.cfgPyMolPath.text().strip(),
    }

def _save_config_yaml(self):
    try:
        out = self._config_yaml_path()
        out.parent.mkdir(parents=True, exist_ok=True)

        template = Path(__file__).resolve().parent / "config_template.yaml"
        tmpl = self._read_yaml_file(template) if template.exists() else {}

        # ✅ read the CURRENT config first (preserve unknown keys)
        existing = self._read_yaml_file(out) if out.exists() else {}

        ui = self._ui_config_dict()

        # Merge order: existing -> template defaults -> ui overrides
        merged = self._deep_merge_dicts(existing, tmpl)
        merged = self._deep_merge_dicts(merged, ui)

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
    self.cfgLigandBasename.setText(str(lig.get("basename", "LIG")))
    self.cfgLigandNameDefault.setText(str(lig.get("name_default", "LIG")))
    png = lig.get("png_size", [1500, 1200])
    if isinstance(png, list) and len(png) == 2:
        self.cfgLigandPngW.setValue(int(png[0]))
        self.cfgLigandPngH.setValue(int(png[1]))
    self.cfgLigandRdkitThreads.setValue(int(lig.get("rdkit_threads", 0)))

    # Viewer executables (host paths; do NOT posixify)
    if hasattr(self, "cfgChimeraPath"):
        self.cfgChimeraPath.setText(str(g("chimera_path", "")))
    if hasattr(self, "cfgPyMolPath"):
        self.cfgPyMolPath.setText(str(g("pymol_path", "")))

    self.cfgRosettaRelaxBin.setText(str(g("rosetta_relax_bin", "")))
    msa_db = msa.get("db", cfg.get("msa.db", ""))
    self.cfgMMseqsDB.setText(self._posixify(str(msa_db)))

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

def _config_help_defs(self) -> dict[str, dict[str, str]]:
    """
    help button objectName -> {title, text, target(optional)}
    'target' is the UI widget that the ? describes (gets the same tooltip).
    """
    return {
        # --- Core paths / platform ---
        "helpCfgWslDistro": {
            "title": "WSL distro",
            "text": "The WSL distribution name used for wsl.exe calls (e.g. Ubuntu-22.04). Affects MSA + AF3 orchestration inside WSL.",
            "target": "cfgWslDistro",
        },
        "helpLinuxHome": {
            "title": "Linux home root",
            "text": "Your Linux $HOME path inside WSL (usually /home/<user>). Used when constructing WSL paths.",
            "target": "cfgLinuxHomeRoot",
        },
        "helpGUIDir": {
            "title": "GUI directory",
            "text": "Path to the GUI folder on the host. Mainly informational.",
            "target": "cfgGuiDir",
        },
        "helpAF3Dir": {
            "title": "AF3 directory (WSL path)",
            "text": "Path inside WSL to your AlphaFold3 repo (e.g. /home/olive/Repositories/alphafold). Used when building/running AF3 jobs.",
            "target": "cfgAf3Dir",
        },
        "helpAFImage": {
            "title": "AlphaFold Docker image",
            "text": "Docker image name/tag used to run AF3 (default: alphafold3). Must match your built/pulled image.",
            "target": "cfgDockerImage",
        },
        "helpDockerBin": {
            "title": "Docker binary",
            "text": "Command used to invoke Docker (usually 'docker'). Used by runner to launch containers.",
            "target": "cfgDockerBin",
        },
        "helpDockerEnv": {
            "title": "Docker environment",
            "text": "Environment variables passed into the AF3 container. You can paste YAML mapping or KEY=VALUE lines.",
            "target": "cfgDockerEnv",
        },

        # --- MMseqs / MSA ---
        "helpMMseqsDir": {
            "title": "MMseqs database directory (WSL path)",
            "text": "Path inside WSL to the mmseqs_db folder. Required for MSA generation.",
            "target": "cfgMMseqsDB",
        },
        "helpMMseqsThreads": {
            "title": "MSA threads",
            "text": "Thread count used for MMseqs2 MSA generation.",
            "target": "cfgMsaThreads",
        },
        "helpMMseqsSensitivity": {
            "title": "MSA sensitivity",
            "text": "MMseqs2 sensitivity. Higher can find more distant homologs but may run slower.",
            "target": "cfgMsaSensitivity",
        },
        "helpMMseqsMaxSequences": {
            "title": "MSA max sequences",
            "text": "Maximum number of sequences kept/returned for MSAs.",
            "target": "cfgMsaMaxSeqs",
        },

        # --- Ligand / RDKit ---
        "helpBasename": {
            "title": "Ligand basename",
            "text": "Residue/ligand basename used in generated files (often 'LIG').",
            "target": "cfgLigandBasename",
        },
        "helpNameDefault": {
            "title": "Default ligand name",
            "text": "Default name used when saving a ligand if the user doesn’t specify one.",
            "target": "cfgLigandNameDefault",
        },
        "helpNConfs": {
            "title": "Ligand conformers (n_confs)",
            "text": "Number of RDKit conformers generated before selecting the lowest-energy conformer.",
            "target": "cfgLigandNConfs",
        },
        "helpSeed": {
            "title": "Ligand seed",
            "text": "Random seed for RDKit embedding. Use a fixed seed for reproducible conformer generation.",
            "target": "cfgLigandSeed",
        },
        "helpPruneRMS": {
            "title": "Prune RMS",
            "text": "RMS threshold used to prune similar conformers. Smaller = keep more diverse conformers.",
            "target": "cfgLigandPruneRms",
        },
        "helpKeepCharge": {
            "title": "Keep charge",
            "text": "Preserve charge handling during ligand preparation.",
            "target": "cfgLigandKeepCharge",
        },
        "helpRequireStereo": {
            "title": "Require stereochemistry",
            "text": "Reject SMILES lacking explicit stereochemistry (helps avoid ambiguous ligand inputs).",
            "target": "cfgLigandRequireStereo",
        },
        "helpRDKitThreads": {
            "title": "RDKit threads",
            "text": "Thread count for RDKit operations where supported (0 means auto).",
            "target": "cfgLigandRdkitThreads",
        },
        "helpPNGSize": {
            "title": "Ligand preview size",
            "text": "Width/height for ligand preview images if your pipeline writes previews.",
            "target": None,  # you have two widgets; see note below
        },

        # --- Rosetta ---
        "helpRosettaDir": {
            "title": "Rosetta relax binary",
            "text": "Path (usually WSL path) to the Rosetta relax executable. Used during post-analysis if Rosetta is enabled.",
            "target": "cfgRosettaRelaxBin",
        },

        # --- Viewers ---
        "helpChimera": {
            "title": "ChimeraX executable",
            "text": "Full path to ChimeraX.exe (Windows). Used by the Ligands tab 'Open in ChimeraX'.",
            "target": "cfgChimeraPath",
        },
        "helpPyMol": {
            "title": "PyMOL executable",
            "text": "Full path to the PyMOL executable (Windows). Used by the Ligands tab 'Open in PyMOL'.",
            "target": "cfgPyMolPath",
        },
    }