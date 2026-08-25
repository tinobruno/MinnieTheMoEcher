# ==============================================================================
# MinnieTheMoECher — Windows Automated 1-Click Installer & Builder
# ==============================================================================
# Requires: Windows 10/11 x64, Visual Studio 2022 / Build Tools (C++), CUDA Toolkit 12.x
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   MinnieTheMoECher Windows Automated Installer (v2.05)       " -ForegroundColor Cyan
Write-Host "   Bare-Metal DeepSeek-V4 MoE Fast Inference Engine            " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check for NVIDIA GPU
Write-Host "[1/5] Checking GPU & CUDA..." -ForegroundColor Yellow
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Write-Host "  Detected GPU: $gpuInfo" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] nvidia-smi not found. Ensure NVIDIA GPU drivers are installed." -ForegroundColor Yellow
}

# 2. Check for CUDA Toolkit (nvcc)
try {
    $nvcc = Get-Command nvcc -ErrorAction Stop
    $cudaVersion = & nvcc --version | Select-String "release"
    Write-Host "  Detected CUDA: $cudaVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] CUDA Toolkit (nvcc) not found in PATH!" -ForegroundColor Red
    Write-Host "  Please install CUDA Toolkit 12.0+ from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    Pause
    Exit 1
}

# 3. Check for CMake
Write-Host "[2/5] Checking CMake..." -ForegroundColor Yellow
$cmakePath = $null

if (Get-Command cmake -ErrorAction SilentlyContinue) {
    $cmakePath = "cmake"
} else {
    # Check standard install locations and Visual Studio embedded CMake
    $possibleCmakePaths = @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )

    foreach ($p in $possibleCmakePaths) {
        if (Test-Path $p) {
            $cmakePath = $p
            break
        }
    }
}

if (-not $cmakePath) {
    Write-Host "  [WARN] CMake not found in PATH or Visual Studio." -ForegroundColor Yellow
    Write-Host "  Attempting to install CMake via winget..." -ForegroundColor Cyan
    try {
        winget install --id Kitware.CMake -e --accept-package-agreements --accept-source-agreements
        $cmakePath = "${env:ProgramFiles}\CMake\bin\cmake.exe"
        if (-not (Test-Path $cmakePath)) { $cmakePath = "cmake" }
    } catch {
        Write-Host "  [ERROR] Could not install CMake automatically." -ForegroundColor Red
        Write-Host "  Please install CMake manually from https://cmake.org/download/" -ForegroundColor Red
        Write-Host "  (Ensure you check 'Add CMake to system PATH' during install)." -ForegroundColor Red
        Pause
        Exit 1
    }
}

$cmakeVersion = & $cmakePath --version | Select-Object -First 1
Write-Host "  Found CMake: $cmakeVersion" -ForegroundColor Green

# 4. Check for Visual Studio C++ Compiler
Write-Host "[3/5] Checking C++ Build Environment..." -ForegroundColor Yellow
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    Write-Host "  Found Visual Studio at: $vsPath" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Visual Studio Installer not detected at default path." -ForegroundColor Yellow
}

# 5. Configure and Build
Write-Host "[4/5] Configuring and Building MinnieTheMoECher with CMake & MSVC..." -ForegroundColor Yellow

$buildDir = "build"
if (!(Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Write-Host "  Running CMake configure..." -ForegroundColor Cyan
if (Test-Path "build\CMakeCache.txt") {
    Remove-Item "build\CMakeCache.txt" -Force
}
& $cmakePath -B build -A x64 -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] CMake configuration failed!" -ForegroundColor Red
    Pause
    Exit 1
}

Write-Host "  Compiling Release binary..." -ForegroundColor Cyan
& $cmakePath --build build --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Compilation failed!" -ForegroundColor Red
    Pause
    Exit 1
}

# Copy compiled executable to root
if (Test-Path "build\Release\moecher.exe") {
    Copy-Item "build\Release\moecher.exe" -Destination ".\moecher.exe" -Force
} elseif (Test-Path "build\moecher.exe") {
    Copy-Item "build\moecher.exe" -Destination ".\moecher.exe" -Force
}

# 6. Success
Write-Host "[5/5] Build Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Installation and compilation successful!" -ForegroundColor Green
Write-Host "  To launch MinnieTheMoECher, run: .\start.bat" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
