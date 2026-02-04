# AlphaFold GUI

## Development
Run locally:
conda activate af3
python main.py


## Build Windows EXE
pyinstaller --noconfirm --clean --windowed --name AlphaFoldGUI
--icon=assets/alphafold.ico
--exclude-module PySide6 --exclude-module shiboken6 --exclude-module PyQt5
--add-data="main_window.ui;."
--add-data="config_template.yaml;."
main.py


The final app will be in `dist/AlphaFoldGUI/AlphaFoldGUI.exe`