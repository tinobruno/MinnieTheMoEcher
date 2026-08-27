@echo off
cd /d "%~dp0"
title Moecher - Hugging Face Model Uploader

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)

.venv\Scripts\python.exe tools\upload_to_hf.py %*

echo.
echo ===============================================================================
echo   Process finished.
echo ===============================================================================
pause
