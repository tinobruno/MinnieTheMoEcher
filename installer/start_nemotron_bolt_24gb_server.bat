@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
echo ===============================================================================
echo   Starting Moecher Server with NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4)
echo   Target: 24GB VRAM (100% In-VRAM Resident Execution)
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher Nemotron Bolt Server" /high moecher.exe -m models\nemotron_3.5_bolt\moecher_manifest_24gb.json --port 8001 %*
