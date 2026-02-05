# AlphaFold GUI

A Windows graphical interface for running **AlphaFold 3 in WSL/Docker**.

This application provides a point-and-click front end for building AlphaFold input files and launching AlphaFold jobs that run inside WSL. The GUI itself runs on Windows; AlphaFold continues to run inside your existing WSL + Docker installation.

---

## Requirements

Before using this application you must already have:

- **WSL2** installed (Ubuntu recommended)
- **Docker running inside WSL**
- A working **AlphaFold 3 installation inside WSL**, including:
  - AlphaFold code directory  
  - Model weights  
  - MMseqs2 databases  

If you can already run AlphaFold from the WSL command line, you are ready to use this GUI.

---

## Using the pre-built Windows app (recommended)

1. Download `AlphaFoldGUI.exe` from the **Releases** page  
2. Place it anywhere (e.g., Desktop)  
3. Double-click to run  
4. On first launch, configure:
   - Your WSL distro (e.g., `Ubuntu-22.04`)  
   - AlphaFold installation directory  
   - Weights directory  
   - MMseqs database path  

After configuration, you can create inputs and launch AlphaFold jobs directly from Windows.

> You do **not** need Python installed to run the `.exe`.  
> The GUI is fully bundled inside the executable.

---

## What the `.exe` does *not* include

The executable **does not contain**:

- AlphaFold  
- Model weights  
- Databases  
- Docker  
- WSL  

Those must already exist in your WSL environment.

The GUI is simply a Windows front end that controls your existing AlphaFold installation.

---

## Development (run from source)

If you want to modify the GUI:

```bash
conda activate af3
python apps/gui/main.py
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