#!/usr/bin/env python3
"""
main.py
========
Entry point for the AlphaFold3 GUI Runner.

IMPORTANT DESIGN CHOICE:
We create QApplication *before importing MainWindow* so that
no QWidget can be constructed too early.
"""

import sys
import qdarkstyle
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
from qt_material import apply_stylesheet
from pathlib import Path
import yaml
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _posixify(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith("\\") and not s.startswith("\\\\"):
        s = "/" + s.lstrip("\\")
    s = s.replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s

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

def bootstrap_user_config():
    out = Path.home() / ".af3_pipeline" / "config.yaml"
    if out.exists():
        return

    tmpl = resource_path("config_template.yaml")
    data = {}
    if tmpl.exists():
        try:
            loaded = yaml.safe_load(tmpl.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}

    # Minimal safe defaults (prevent /af_input)
    data.setdefault("platform", "wsl" if sys.platform.startswith("win") else "linux")
    data.setdefault("wsl_distro", "Ubuntu-22.04")
    data.setdefault("af3_dir", "/home/olive/Repositories/alphafold")
    data.setdefault("linux_home_root", "/home/olive")
    data.setdefault("gui_dir", str(Path(__file__).resolve().parent))
    data.setdefault("gui_cache_subdir", "cache/af3_gui_cache")
    data.setdefault("docker_bin", "docker")
    data.setdefault("alphafold_docker_image", "alphafold3")
    data.setdefault("alphafold_docker_env", {})

    msa = data.get("msa")
    if not isinstance(msa, dict):
        msa = {}
        data["msa"] = msa
    msa.setdefault("db", "/home/olive/Repositories/alphafold/mmseqs_db")
    msa.setdefault("threads", 10)
    msa.setdefault("sensitivity", 5.7)
    msa.setdefault("max_seqs", 25)

    # Normalize paths
    data["af3_dir"] = _posixify(str(data.get("af3_dir", "")))
    data["linux_home_root"] = _posixify(str(data.get("linux_home_root", "")))
    msa["db"] = _posixify(str(msa.get("db", "")))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

bootstrap_user_config()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    #qss_path = Path(__file__).parent / "assets" / "Ubuntu.qss"
    #with open(qss_path, "r", encoding="utf-8") as f:
        #app.setStyleSheet(f.read())

    base = qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.LightPalette)

    app.setStyleSheet(base + """
    QMainWindow {
        background-color: #3B59D9;
    }
    QLabel {
        font-weight: 600;
    }
    """)

    ico = QIcon(str(resource_path("assets/custom_icon_2.ico")))
    app.setWindowIcon(ico)

    app.setFont(QFont("Arial", 10))

    from main_window import MainWindow

    window = MainWindow()
    window.setWindowIcon(ico)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
