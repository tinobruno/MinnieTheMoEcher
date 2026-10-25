@echo off
setlocal
echo ==============================================================================
echo                 MinnieBench - Multi-Model Inference Suite
echo ==============================================================================
echo Host GPU: NVIDIA GeForce RTX 5060 Ti 16GB
echo Python: C:\Users\tino\AppData\Local\Programs\Python\Python312\python.exe
echo Engine: MinnieTheMoEcher v2.09
echo ==============================================================================
echo.

"C:\Users\tino\AppData\Local\Programs\Python\Python312\python.exe" "E:\dev\MinnieTheMoEcher\minniebench\bench_runner.py"

echo.
echo ==============================================================================
echo MinnieBench execution complete. Check minniebench/MINNIEBENCH_REPORT.md.
echo ==============================================================================
pause
