AlphaFold GUI

A Windows graphical interface for running AlphaFold 3 in WSL/Docker.

This application provides a point-and-click front end for building AlphaFold input files and launching AlphaFold jobs that run inside WSL.

The GUI runs natively on Windows, while all heavy computation (AlphaFold, MMseqs2, Docker, and GPUs) continues to run inside your existing WSL installation. The GUI simply orchestrates that environment from Windows.

Requirements (must already be working)

Before using this application, you must already have a functioning AlphaFold 3 environment in WSL:

WSL2 installed (Ubuntu 22.04 recommended)

Docker running inside WSL with GPU access (docker --gpus all works)

A working AlphaFold 3 installation inside WSL, including:

AlphaFold 3 code directory

Model weights (af3.bin)

MMseqs2 databases (e.g., uniref30_2302_db)

Rule of thumb:
If you can already run AlphaFold successfully from the WSL command line, you are ready to use this GUI.

Using the pre-built Windows app (recommended)

Download AlphaFoldGUI.exe from the Releases page

Place it anywhere (e.g., Desktop)

Double-click to run

On first launch, you will be prompted to configure:

Your WSL distro (e.g., Ubuntu-22.04)

AlphaFold installation directory in WSL

Weights directory

MMseqs database path

After configuration, you can:

Create protein / nucleic acid sequences

Prepare ligands

Build AlphaFold inputs

Launch jobs

Monitor logs
—all from Windows.

You do not need Python installed on Windows

The GUI is fully bundled inside the executable.

What the .exe does NOT include

The executable does not ship with:

AlphaFold

Model weights

Databases

Docker

WSL

Those must already exist in your WSL environment.

The GUI is strictly a Windows front end that controls your existing AlphaFold setup.

Development (run from source)

If you want to modify or extend the GUI:

conda activate af3
python apps/gui/main.py

This runs the same interface directly from Python.

Build Windows EXE

From the repository root:

python -m PyInstaller --noconfirm --clean --onefile --windowed --name AlphaFoldGUI `
  --paths "." `
  --exclude-module PySide6 --exclude-module shiboken6 --exclude-module PyQt5 `
  --icon "apps\gui\assets\alphafold.ico" `
  --add-data "apps\gui\main_window.ui;." `
  --add-data "apps\gui\config_template.yaml;." `
  apps\gui\main.py

The final application will be in:

dist/AlphaFoldGUI.exe
Troubleshooting
“Docker permission denied”

Inside WSL:

sudo usermod -aG docker $USER
newgrp docker

or reopen your Ubuntu terminal.

“GPU not visible in Docker”

From WSL, check:

docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi

If this fails, restart WSL:

wsl --shutdown

Then try again.

Installation guide for AlphaFold + WSL

If you still need to set up AlphaFold itself, see:

install/README_SETUP.md

This contains step-by-step instructions for a fresh Ubuntu + CUDA + Docker + AlphaFold install.