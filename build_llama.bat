@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 exit /b 1

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "NVCC_PREPEND_FLAGS=--allow-unsupported-compiler"

cd /d E:\dev\llama.cpp
if exist build rmdir /s /q build

cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" ^
    -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" ^
    -DCMAKE_CUDA_ARCHITECTURES="120a" ^
    -DGGML_CUDA_FA_ALL_QUANTS=OFF
if %errorlevel% neq 0 exit /b 1

cmake --build build --config Release
if %errorlevel% neq 0 exit /b 1

echo [SUCCESS] llama.cpp build complete!
