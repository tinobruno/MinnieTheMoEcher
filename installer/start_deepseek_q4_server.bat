@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo ===============================================================================
echo   Starting Moecher Server with DeepSeek V4 Flash Q4 (8GB GPU Mode)
echo   Web UI: http://localhost:8000
echo ===============================================================================
start "Moecher DeepSeek Q4 Server" /high moecher.exe --manifest models\deepseek_v4_flash_q4\moecher_manifest.json --max-vram 6 --dram-cache-gb 64 --quiet
