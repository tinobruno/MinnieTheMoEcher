@echo off
setlocal enabledelayedexpansion
title Moecher - Hugging Face Model Uploader
cd /d "%~dp0"

echo ===============================================================================
echo   Moecher Inference Engine - Hugging Face Model Uploader
echo   Target Account: TinoBruno
echo ===============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Choose what to upload:
echo   [1] Qwen 3.8 27B INT4 (~18.7 GB)
echo   [2] DeepSeek V4 Flash IQ2 (~81.4 GB)
echo   [3] Both Models
echo.
set /p CHOICE="Enter choice (1, 2, or 3) [default: 1]: "
if "%CHOICE%"=="" set CHOICE=1

set TARGET=qwen
if "%CHOICE%"=="1" set TARGET=qwen
if "%CHOICE%"=="2" set TARGET=deepseek
if "%CHOICE%"=="3" set TARGET=all

echo.
if "%HF_TOKEN%"=="" (
    echo Paste your Hugging Face Write Access Token (from https://huggingface.co/settings/tokens):
    set /p HF_TOKEN="Token (hf_...): "
)

if "%HF_TOKEN%"=="" (
    echo [ERROR] Token cannot be empty.
    pause
    exit /b 1
)

echo.
echo Starting upload for: %TARGET% ...
echo.

.venv\Scripts\python.exe tools\upload_to_hf.py %TARGET% --token %HF_TOKEN%

echo.
echo ===============================================================================
echo   Process finished.
echo ===============================================================================
pause
