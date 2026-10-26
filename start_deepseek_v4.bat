@echo off
setlocal
chcp 65001 >nul
echo ===============================================================================
echo   Starting Moecher Server: DeepSeek-V4-Flash (IQ2 Quantization)
echo   Hardware Target: NVIDIA RTX PRO 6000 96GB
echo   Memory Hierarchy: 11,008 Experts all-resident in VRAM (72.6 GB)
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher DeepSeek-V4 Server" /high moecher.exe -m F:\Moecher\models\deepseek_v4_flash_iq2\moecher_manifest.json --port 8001 %*
