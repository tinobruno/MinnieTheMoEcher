@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
echo ===============================================================================
echo   Starting Moecher Server with NVIDIA Nemotron 3.5 Lightning (30B-A3B NVFP4)
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher Nemotron Server" /high moecher.exe -m models\nemotron_3.5_lightning\moecher_manifest.json --port 8001

