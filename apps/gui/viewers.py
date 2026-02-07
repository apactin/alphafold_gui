from __future__ import annotations
import sys
import subprocess
from pathlib import Path
import os
import shutil

from PyQt6.QtWidgets import QMessageBox

def _resolve_viewer_exe(
    self,
    *,
    key: str,
    ui_widget_name: str | None = None,
    fallbacks: list[str] | None = None,
) -> list[str]:
    """
    Return argv prefix for launching a viewer.

    Priority:
      1) UI field (path OR command)
      2) YAML key in config (path OR command)
      3) fallback candidates (absolute paths)
      4) PATH lookup for common executable names inferred from `key`
      5) finally: treat the config value / UI value / or inferred command as the argv[0]
    """

    def _clean(s: str) -> str:
        s = (s or "").strip()
        # remove surrounding quotes people paste on Windows
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        return os.path.expandvars(s)

    def _as_argv(s: str) -> list[str] | None:
        """
        If s points to an existing file, return [abs_path].
        If s is a command found on PATH, return [resolved_command].
        Otherwise None.
        """
        s = _clean(s)
        if not s:
            return None

        # Path?
        try:
            p = Path(s).expanduser()
            if p.exists():
                return [str(p.resolve())]
        except Exception:
            pass

        # Command on PATH?
        w = shutil.which(s)
        if w:
            return [w]

        return None

    # 1) UI field
    if ui_widget_name and hasattr(self, ui_widget_name):
        w = getattr(self, ui_widget_name)
        try:
            txt = (w.text() or "").strip()
            got = _as_argv(txt)
            if got:
                return got
        except Exception:
            pass

    # 2) config file
    cfg_val = ""
    try:
        data = self._load_config_yaml_best_effort()
        cfg_val = (data.get(key) or "").strip() if isinstance(data, dict) else ""
        got = _as_argv(cfg_val)
        if got:
            return got
    except Exception:
        pass

    # 3) fallbacks (absolute)
    for fb in (fallbacks or []):
        got = _as_argv(fb)
        if got:
            return got

    # 4) infer common command names for PATH based on key
    # (so we never accidentally try to execute "chimera_path" or "pymol_path")
    key_l = (key or "").lower()
    candidates: list[str] = []
    if "chimera" in key_l:
        candidates = ["ChimeraX.exe", "chimerax.exe", "ChimeraX", "chimerax"]
    elif "pymol" in key_l:
        candidates = ["pymol.exe", "pymol", "PyMOL.exe", "PyMOL"]

    for c in candidates:
        w = shutil.which(c)
        if w:
            return [w]

    # 5) last resort: if cfg_val existed, try to run it literally; else use inferred command
    if cfg_val:
        return [_clean(cfg_val)]
    if candidates:
        return [candidates[-1]]
    return [key]

def _open_structures_in_pymol(self, *, paths: list[Path], align: bool) -> None:

        exe_argv = self._resolve_viewer_exe(
            key="pymol_path",
            ui_widget_name="cfgPyMolPath",
            fallbacks=[],
        )

        p = [x.resolve() for x in paths if x and x.exists()]
        if not p:
            return

        cmds = []
        if len(p) >= 1:
            cmds.append(f'load "{p[0]}", af')
        if len(p) >= 2:
            cmds.append(f'load "{p[1]}", relax')
            if align:
                cmds.append("align relax, af")

        cmd_str = "; ".join(cmds)
        subprocess.Popen(exe_argv + ["-q", "-d", cmd_str])


def _open_structures_in_chimerax(self, *, paths: list[Path], align: bool) -> None:


    fallbacks = []
    if sys.platform.startswith("win"):
        fallbacks = [
            r"C:\Program Files\ChimeraX\bin\ChimeraX.exe",
            r"C:\Program Files\ChimeraX\ChimeraX.exe",
        ]

    exe_argv = self._resolve_viewer_exe(
        key="chimera_path",
        ui_widget_name="cfgChimeraPath",
        fallbacks=fallbacks,
    )

    p = [x.resolve() for x in paths if x and x.exists()]
    if not p:
        return

    cmds = [f'open "{p[0]}"']
    if len(p) >= 2:
        cmds.append(f'open "{p[1]}"')
        if align:
            # first opened becomes #1, second becomes #2
            cmds.append("matchmaker #2 to #1")

    cmd_str = " ; ".join(cmds)
    try:
        subprocess.Popen(exe_argv + ["--cmd", cmd_str])
    except FileNotFoundError:
        QMessageBox.warning(
            self,
            "ChimeraX not found",
            "Could not launch ChimeraX.\n\n"
            "Check Config → ChimeraX Path, or add ChimeraX to your PATH.\n"
            f"Tried: {exe_argv[0]}",
        )