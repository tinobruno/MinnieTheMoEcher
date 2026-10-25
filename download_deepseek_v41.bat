@echo off
setlocal
echo =====================================================================
echo   MinnieTheMoEcher: DeepSeek-V4.1-Flash Official Weights Downloader
echo =====================================================================
echo.
python scripts\download_deepseek_v41.py --repo deepseek-ai/DeepSeek-V4.1-Flash-1017 --dest F:\models\deepseek_v41_raw --workers 8
if %errorlevel% neq 0 (
    echo [ERROR] Download encountered an error.
    exit /b 1
)
echo [SUCCESS] Download finished successfully.
