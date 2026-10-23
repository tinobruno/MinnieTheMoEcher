@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON_BIN=python
if exist "C:\Users\tino\Downloads\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" (
    set "PYTHON_BIN=C:\Users\tino\Downloads\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe"
)

echo ===============================================================================
echo   MinnieTheMoEcher — Hugging Face Model Package Uploader
echo   Model: NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4)
echo   Python: %PYTHON_BIN%
echo ===============================================================================

"%PYTHON_BIN%" tools\upload_nemotron_bolt.py %*

echo.
echo ===============================================================================
echo   Process finished.
echo ===============================================================================
pause
