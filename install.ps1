# ==============================================================================
# MinnieTheMoECher — Windows Automated 1-Click Installer & Builder
# ==============================================================================
# Requires: Windows 10/11 x64, Visual Studio 2022/2026 (C++), CUDA Toolkit 12.0+ / 13.x
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   MinnieTheMoECher Windows Automated Installer                 " -ForegroundColor Cyan
Write-Host "   Bare-Metal Agentic AI Runtime (CUDA Backend)                 " -ForegroundColor Cyan
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

# 2. Check for CUDA Toolkit (nvcc) — prefer newest installed toolkit in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
$nvccPath = $null
$cudaRoot = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $cudaRoot) {
    $cudaDirs = Get-ChildItem -Path $cudaRoot -Directory | Sort-Object Name -Descending
    foreach ($d in $cudaDirs) {
        $candidate = Join-Path $d.FullName "bin\nvcc.exe"
        if (Test-Path $candidate) {
            $nvccPath = $candidate
            $env:CUDA_PATH = $d.FullName
            $env:PATH = "$(Join-Path $d.FullName 'bin');$env:PATH"
            break
        }
    }
}
if (-not $nvccPath -and (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    $nvccPath = (Get-Command nvcc).Source
}

if (-not $nvccPath) {
    Write-Host "  [ERROR] CUDA Toolkit (nvcc) not found!" -ForegroundColor Red
    Write-Host "  Please install CUDA Toolkit 12.0+ / 13.x from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    exit 1
}

$cudaVersion = & $nvccPath --version | Select-String "release"
Write-Host "  Detected CUDA: $cudaVersion ($nvccPath)" -ForegroundColor Green

# 3. Check for CMake (PATH, standard locations, and Visual Studio 2022/2026)
Write-Host "[2/5] Checking CMake..." -ForegroundColor Yellow
$cmakePath = $null

if (Get-Command cmake -ErrorAction SilentlyContinue) {
    $cmakePath = (Get-Command cmake).Source
} else {
    $possibleCmakePaths = @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\18\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\18\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
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
    Write-Host "  [WARN] CMake not found in PATH or Visual Studio. Attempting winget install..." -ForegroundColor Yellow
    try {
        winget install --id Kitware.CMake -e --accept-package-agreements --accept-source-agreements
        $cmakePath = "${env:ProgramFiles}\CMake\bin\cmake.exe"
        if (-not (Test-Path $cmakePath)) { $cmakePath = "cmake" }
    } catch {
        Write-Host "  [ERROR] Could not install CMake automatically. Install from https://cmake.org/download/" -ForegroundColor Red
        exit 1
    }
}

$cmakeVersion = & $cmakePath --version | Select-Object -First 1
Write-Host "  Found CMake: $cmakeVersion" -ForegroundColor Green

# 4. Embed Web UI assets into src/embedded_web.hpp
Write-Host "[3/5] Embedding Web UI Assets..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python scripts\embed_web.py
    Write-Host "  Embedded Web UI assets into src/embedded_web.hpp" -ForegroundColor Green
}

# 5. Configure and Build
Write-Host "[4/5] Configuring and Building MinnieTheMoECher..." -ForegroundColor Yellow

# Prefer fast Ninja build via build_fast.bat if VS 18/2022 vcvars64 + Ninja are available
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = $null
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath | Select-Object -First 1
}
if (-not $vsPath) {
    foreach ($cand in @("${env:ProgramFiles}\Microsoft Visual Studio\18\Community", "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community")) {
        if (Test-Path $cand) { $vsPath = $cand; break }
    }
}

$nvccForward = $nvccPath -replace '\\', '/'
$vcvars = if ($vsPath) { Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
$ninjaDir = if ($vsPath) { Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja" } else { $null }

if ($vcvars -and (Test-Path $vcvars) -and $ninjaDir -and (Test-Path (Join-Path $ninjaDir "ninja.exe"))) {
    Write-Host "  Using Visual Studio + Ninja fast build pipeline ($vsPath)..." -ForegroundColor Cyan
    $cudaBin = Split-Path -Parent $nvccPath
    $cmakeBin = Split-Path -Parent $cmakePath
    $env:PATH = "$cmakeBin;$ninjaDir;$cudaBin;$env:PATH"
    $cmdLine = "call `"$vcvars`" && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=`"$nvccForward`" -DCMAKE_CUDA_ARCHITECTURES=`"86;89;90;100;120`" && cmake --build build --config Release"
    cmd.exe /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] Compilation failed!" -ForegroundColor Red
        exit 1
    }
} else {
    if (Test-Path "build\CMakeCache.txt") {
        Remove-Item "build\CMakeCache.txt" -Force
    }
    & $cmakePath -B build -A x64 -DCMAKE_CUDA_COMPILER="$nvccForward" -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] CMake configuration failed!" -ForegroundColor Red
        exit 1
    }
    & $cmakePath --build build --config Release --parallel
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] Compilation failed!" -ForegroundColor Red
        exit 1
    }
}

if (Test-Path "build\Release\moecher.exe") {
    Copy-Item "build\Release\moecher.exe" -Destination ".\moecher.exe" -Force
} elseif (Test-Path "build\moecher.exe") {
    Copy-Item "build\moecher.exe" -Destination ".\moecher.exe" -Force
}

Write-Host "[5/5] Build Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Installation and compilation successful!" -ForegroundColor Green
Write-Host "  Run .\moecher.exe to select or download a model and launch" -ForegroundColor Cyan
Write-Host "  the Web UI at http://localhost:8001" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
