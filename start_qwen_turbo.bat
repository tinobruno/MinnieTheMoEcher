@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo ===============================================================================
echo   Starting Moecher Server: Qwen 3.8 27B Turbo Vision (13G Mixed Quant)
echo   Web UI: http://localhost:8001
===============================================================================
.\moecher.exe -m "E:\moecher\models\qwen3_8_turbo_vision_13g\moecher_manifest.json" --port 8001 --ctx 8192 --max-vram 14.0
