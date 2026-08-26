@echo off
title Moecher - Qwen 3.8 27B INT4 Server
cd /d "%~dp0"

echo ======================================================================
echo   Moecher Inference Engine - Qwen 3.8 27B (INT4 Block-32)
echo ======================================================================
echo.
echo Starting server on http://127.0.0.1:8001 ...
echo.

if not exist "models\qwen3_8_27b_q4\attention_dense_layers_q4.bin" (
    echo [ERROR] Model weight file 'models\qwen3_8_27b_q4\attention_dense_layers_q4.bin' not found!
    echo Please make sure the installation was not corrupted.
    echo.
    pause
    exit /b 1
)

start /high moecher.exe --manifest models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json --port 8001

echo Server launched in high priority window.
echo You can test the API by running 'test_qwen.bat'.
echo.
pause
