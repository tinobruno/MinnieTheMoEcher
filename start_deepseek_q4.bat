@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ================================================================
echo    Moecher - DeepSeek V4 Flash Q4 (8GB GPU Mode)
echo    Bare-Metal DeepSeek-V4 MoE Fast Inference Engine
echo ================================================================
echo.

:: Check for moecher.exe
if not exist "moecher.exe" (
    if exist "build\Release\moecher.exe" (
        copy "build\Release\moecher.exe" . >nul
    ) else (
        echo [ERROR] moecher.exe not found!
        echo Please build the project first: cmake --build build --config Release
        pause
        exit /b 1
    )
)

set MANIFEST=models\deepseek_v4_flash_q4\moecher_manifest.json
if not exist "%MANIFEST%" (
    echo [ERROR] Model manifest not found at %MANIFEST%!
    pause
    exit /b 1
)

echo Starting Moecher with DeepSeek V4 Flash Q4 on http://localhost:8000 ...
echo [Press Ctrl+C to stop]
echo.

moecher.exe --manifest "%MANIFEST%" --port 8000 --max-vram 6 --dram-cache-gb 64 --budget 4096
pause
