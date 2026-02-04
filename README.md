# AlphaFold GUI

## Development
Run locally:
conda activate af3
python main.py


## Build Windows EXE
pyinstaller --noconfirm --clean --onefile --windowed --name AlphaFoldGUI `
  --paths "." `
  --exclude-module PySide6 --exclude-module shiboken6 --exclude-module PyQt5 `
  --icon "apps\gui\assets\alphafold.ico" `
  --add-data "apps\gui\main_window.ui;." `
  --add-data "apps\gui\config_template.yaml;." `
  apps\gui\main.py


The final app will be in `dist/AlphaFoldGUI/AlphaFoldGUI.exe`