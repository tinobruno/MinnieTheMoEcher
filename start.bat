@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo    MinnieTheMoECher (Windows Native v2.05)
echo    Bare-Metal DeepSeek-V4 MoE Fast Inference Engine
echo ================================================================
echo.

:: Check for moecher.exe
if not exist "moecher.exe" (
    if exist "build\Release\moecher.exe" (
        copy "build\Release\moecher.exe" . >nul
    ) else if exist "build\moecher.exe" (
        copy "build\moecher.exe" . >nul
    ) else (
        echo [ERROR] moecher.exe not found!
        echo Please build the project first or run install.ps1
        pause
        exit /b 1
    )
)

:: Check for model manifest
set MANIFEST=moecher_manifest_iq2.json
if not exist "%MANIFEST%" (
    if exist "moecher_manifest.json" (
        set MANIFEST=moecher_manifest.json
    ) else (
        echo [ERROR] No model manifest found ^(moecher_manifest_iq2.json or moecher_manifest.json^)!
        pause
        exit /b 1
    )
)

:: Check for dense weights
if not exist "attention_dense_layers.bin" (
    echo [ERROR] attention_dense_layers.bin not found!
    echo Please ensure the weights binary is placed in this directory.
    pause
    exit /b 1
)

:: Check for expert weights
if not exist "moe_experts_iq2.bin" (
    if not exist "moe_experts.bin" (
        echo [ERROR] moe_experts_iq2.bin not found!
        echo Please ensure the quantized expert weights are placed in this directory.
        pause
        exit /b 1
    )
)

echo [INFO] Using manifest: %MANIFEST%
echo [INFO] Launching MinnieTheMoECher Server on http://localhost:8001 ...
echo [INFO] Open your browser at http://localhost:8001 to access the Web UI.
echo.

:: Default launch configuration (22 GB VRAM for 3090/4090, 48 GB DRAM cache)
moecher.exe --manifest "%MANIFEST%" --max-vram 22 --dram-cache-gb 48 --port 8001 --quiet

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] MinnieTheMoECher exited with error code %ERRORLEVEL%.
    pause
)
