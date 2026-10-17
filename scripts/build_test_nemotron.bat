@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 exit /b 1

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

if exist build_nemotron rmdir /s /q build_nemotron

cmake -B build_nemotron -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" ^
    -DCMAKE_CUDA_ARCHITECTURES="86;89;90;100;120"

if %errorlevel% neq 0 exit /b 1

cmake --build build_nemotron --target test_nemotron_ssm
if %errorlevel% neq 0 exit /b 1

echo [INFO] Running test_nemotron_ssm.exe...
build_nemotron\test_nemotron_ssm.exe
