@echo off
setlocal
cd /d "%~dp0"
echo ===============================================================================
echo   Starting Moecher Server with DeepSeek V4 Flash IQ2
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher DeepSeek Server" /high moecher.exe --manifest models\deepseek_v4_flash_iq2\moecher_manifest.json --max-vram 0 --dram-cache-gb 64 --quiet
