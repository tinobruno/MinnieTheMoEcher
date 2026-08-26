# ─────────────────────────────────────────────────────────────────────────────
#  build_installer.ps1 — Build Moecher Windows Setup Package
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Building Moecher Modular Windows Setup Package" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# 1. Locate Inno Setup Compiler (ISCC.exe)
$IsccPaths = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)

$IsccExe = $IsccPaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $IsccExe) {
    Write-Error "ISCC.exe not found! Please ensure Inno Setup is installed."
}

Write-Host "Found Inno Setup Compiler: $IsccExe" -ForegroundColor Green

# 2. Verify all prerequisite files
$RequiredFiles = @(
    "build\Release\moecher.exe",
    "installer\Moecher_Setup.iss",
    "installer\start_qwen_server.bat",
    "installer\test_qwen.bat",
    "graphics\moecher.ico",
    "graphics\installer_wizard.bmp",
    "graphics\installer_small.bmp",
    "web\index.html",
    "models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json",
    "models\qwen3_8_27b_q4\tokenizer.json",
    "models\qwen3_8_27b_q4\attention_dense_layers_q4.bin"
)

foreach ($file in $RequiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "Missing required source file: $file"
    }
}

# 3. Create dist output directory
$DistDir = "D:\dev\minniemoe\MinnieTheMoEcher\dist"
if (-not (Test-Path $DistDir)) {
    New-Item -ItemType Directory -Path $DistDir | Out-Null
}

# 4. Compile Inno Setup package (creates standalone Moecher-Setup.exe)
Write-Host "Compiling Inno Setup package..." -ForegroundColor Cyan
$IssScript = "D:\dev\minniemoe\MinnieTheMoEcher\installer\Moecher_Setup.iss"

& $IsccExe $IssScript

if ($LASTEXITCODE -ne 0) {
    Write-Error "Installer compilation failed with exit code $LASTEXITCODE"
}

# 5. Populate separated models directory in dist/
$DistModelDir = "$DistDir\models\qwen3_8_27b_q4"
if (-not (Test-Path $DistModelDir)) {
    New-Item -ItemType Directory -Path $DistModelDir -Force | Out-Null
}

Write-Host "Syncing separated model files to dist\models\qwen3_8_27b_q4..." -ForegroundColor Cyan
$modelFiles = @(
    "moecher_manifest_qwen_q4.json",
    "tokenizer.json",
    "attention_dense_layers_q4.bin"
)

foreach ($mf in $modelFiles) {
    $src = "models\qwen3_8_27b_q4\$mf"
    $dst = "$DistModelDir\$mf"
    if (-not (Test-Path $dst)) {
        # Create hardlink if on same NTFS volume to avoid copying 18.7 GB, or copy if needed
        try {
            New-Item -ItemType HardLink -Path $dst -Target (Resolve-Path $src) -ErrorAction Stop | Out-Null
            Write-Host "  Linked $mf (instant hardlink)" -ForegroundColor Green
        } catch {
            Copy-Item $src $dst -Force
            Write-Host "  Copied $mf" -ForegroundColor Green
        }
    } else {
        Write-Host "  $mf already present in dist\models\qwen3_8_27b_q4" -ForegroundColor Gray
    }
}

$setupExe = "$DistDir\Moecher-Setup.exe"
$setupItem = Get-Item $setupExe
$setupMB = [math]::Round($setupItem.Length / 1MB, 2)

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Moecher Setup Package Created Successfully!" -ForegroundColor Green
Write-Host "  Installer: $setupExe ($setupMB MB)" -ForegroundColor Green
Write-Host "  Model Dir: $DistModelDir (~18.7 GB separate)" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
