@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run fem_gui\fem_app.py
) else (
    streamlit run fem_gui\fem_app.py
)
