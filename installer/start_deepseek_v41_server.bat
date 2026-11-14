@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "moecher.exe" if exist "..\moecher.exe" cd /d "%~dp0.."
set "MANIFEST=models\deepseek_v41_nvfp4\moecher_manifest_mixed.json"
if not exist "%MANIFEST%" if exist "F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest_mixed.json" set "MANIFEST=F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest_mixed.json"
echo ===============================================================================
echo   Starting Moecher Server: DeepSeek-V4.1-Flash (552B / 14B Active Mixed)
echo   Hardware Target: NVIDIA RTX PRO 6000 96GB + 64GB RAM
echo   Memory Hierarchy: L1 VRAM (Hot NVFP4) + L2 DRAM (Cold IQ2_XXS)
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher DeepSeek-V4.1 Server" /high moecher.exe -m "%MANIFEST%" --mtp-k 4 --port 8001 %*
