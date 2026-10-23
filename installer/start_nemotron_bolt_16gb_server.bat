@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
echo ===============================================================================
echo   Starting Moecher Server with NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4)
echo   Target: 16GB VRAM (Hybrid 18 VRAM / 22 DRAM Lookahead Streaming)
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher Nemotron Bolt Server" /high moecher.exe -m models\nemotron_3.5_bolt\moecher_manifest_16gb.json --port 8001 %*
