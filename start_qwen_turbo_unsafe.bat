@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo ===============================================================================
echo   Starting Moecher Server: Qwen 3.8 27B Turbo Vision (13G Mixed Quant)
echo   MODE: FULL AGENTIC UNSAFE TOOLING ENABLED
echo   - Web Scraping ^& Headless Browsing: ENABLED
echo   - Host Shell Command Execution:     ENABLED
echo   - File Read ^& Write Operations:     ENABLED
echo   - Max Tool Execution Rounds:        15
echo   Web UI: http://localhost:8001
===============================================================================
.\moecher.exe -m "E:\moecher\models\qwen3_8_turbo_vision_13g\moecher_manifest.json" --port 8001 --ctx 8192 --max-vram 14.0 --no-mtp --tools --allow-host-exec --allow-file-write --headless-browsing --max-tool-rounds 15
