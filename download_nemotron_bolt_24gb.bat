@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON_BIN=python
if exist "C:\Users\tino\Downloads\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" (
    set "PYTHON_BIN=C:\Users\tino\Downloads\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe"
)

echo ===============================================================================
echo   Downloading ^& Preparing NVIDIA Nemotron 3.5 Bolt (74B-A9B 24GB IQ2_XXS)
echo   Python: %PYTHON_BIN%
echo ===============================================================================

"%PYTHON_BIN%" scripts\download_nemotron_bolt_24gb.py %*
