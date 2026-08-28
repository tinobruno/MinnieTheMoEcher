@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo ===============================================================================
echo   Starting Moecher Server with Qwen 3.8 27B INT4
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher Qwen Server" /high moecher.exe --manifest models\qwen3_8_27b_q4\moecher_manifest.json --max-vram 0 --dram-cache-gb 0 --quiet
