# ─────────────────────────────────────────────────────────────────────────────
#  Build Windows Installer for Moecher & Qwen 3.8 27B INT4
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$WorkspaceRoot = "D:\dev\minniemoe\MinnieTheMoEcher"
Set-Location $WorkspaceRoot

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Building Moecher Qwen 3.8 27B INT4 Windows Installer" -ForegroundColor Cyan
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
    "installer\Moecher_Qwen_Setup.iss",
    "installer\start_qwen_server.bat",
    "installer\test_qwen.bat",
    "models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json",
    "models\qwen3_8_27b_q4\tokenizer.json",
    "models\qwen3_8_27b_q4\attention_dense_layers_q4.bin",
    "web\index.html"
)

foreach ($file in $RequiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "Missing required source file: $file"
    }
}

# 3. Create dist output directory
$DistDir = Join-Path $WorkspaceRoot "dist"
if (-not (Test-Path $DistDir)) {
    New-Item -ItemType Directory -Path $DistDir | Out-Null
}

# 4. Compile Installer
Write-Host "Compiling Inno Setup package (packaging ~18.7 GB model)..." -ForegroundColor Yellow
$IssScript = Join-Path $WorkspaceRoot "installer\Moecher_Qwen_Setup.iss"

& "$IsccExe" /Qp "$IssScript"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Installer compilation failed with exit code $LASTEXITCODE"
}

$InstallerPath = Join-Path $DistDir "Moecher-Qwen3.8-Setup.exe"
if (Test-Path $InstallerPath) {
    $Item = Get-Item $InstallerPath
    $SizeGB = [math]::Round($Item.Length / 1GB, 2)
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  Installer Created Successfully!" -ForegroundColor Green
    Write-Host "  Path: $InstallerPath ($SizeGB GB)" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
} else {
    Write-Error "Installer output file not found at $InstallerPath"
}
