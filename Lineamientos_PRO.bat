@echo off
REM Lanza la interfaz de Lineamientos PRO
cd /d "%~dp0"
python -X utf8 gui_pro\app.py %*
if errorlevel 1 pause
