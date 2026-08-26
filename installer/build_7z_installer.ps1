# ─────────────────────────────────────────────────────────────────────────────
#  build_7z_installer.ps1 — Build 64-bit Single-Executable Installer for Moecher
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

$SevenZipExe = "C:\Program Files\7-Zip\7z.exe"
$SevenZipSfx = "C:\Program Files\7-Zip\7z.sfx"

if (-not (Test-Path $SevenZipExe) -or -not (Test-Path $SevenZipSfx)) {
    Write-Error "7-Zip 64-bit not found in standard installation directory!"
}

# 1. Prepare staging directory
$StageDir = "D:\dev\minniemoe\MinnieTheMoEcher\dist\staging"
$DistDir  = "D:\dev\minniemoe\MinnieTheMoEcher\dist"

if (Test-Path $StageDir) {
    Remove-Item -Path $StageDir -Recurse -Force
}
New-Item -ItemType Directory -Path $StageDir | Out-Null
New-Item -ItemType Directory -Path "$StageDir\models\qwen3_8_27b_q4" | Out-Null
New-Item -ItemType Directory -Path "$StageDir\web" | Out-Null

Write-Host "Staging core files..." -ForegroundColor Cyan

# Copy binaries & DLLs
Copy-Item "build\Release\moecher.exe" "$StageDir\moecher.exe" -Force
if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublas64_13.dll") {
    Copy-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublas64_13.dll" "$StageDir\cublas64_13.dll" -Force
    Copy-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublasLt64_13.dll" "$StageDir\cublasLt64_13.dll" -Force
}

# Copy launch scripts & icon
Copy-Item "installer\start_qwen_server.bat" "$StageDir\start_qwen_server.bat" -Force
Copy-Item "installer\test_qwen.bat" "$StageDir\test_qwen.bat" -Force
Copy-Item "graphics\moecher.ico" "$StageDir\moecher.ico" -Force
Copy-Item "web\*" "$StageDir\web\" -Recurse -Force

# Create Desktop Shortcut helper script
$ShortcutScript = @"
`$WshShell = New-Object -comObject WScript.Shell
`$Shortcut = `$WshShell.CreateShortcut("`$([Environment]::GetFolderPath('Desktop'))\Moecher Qwen Server.lnk")
`$Shortcut.TargetPath = "`$PSScriptRoot\start_qwen_server.bat"
`$Shortcut.WorkingDirectory = "`$PSScriptRoot"
`$Shortcut.IconLocation = "`$PSScriptRoot\moecher.ico,0"
`$Shortcut.Description = "Launch Moecher Inference Engine"
`$Shortcut.Save()
Write-Host "Desktop shortcut created successfully." -ForegroundColor Green
"@
Set-Content -Path "$StageDir\create_shortcut.ps1" -Value $ShortcutScript

# Copy Model Assets (Hard link or copy if same volume)
Write-Host "Linking model assets (~18.7 GB)..." -ForegroundColor Cyan
Copy-Item "models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json" "$StageDir\models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json" -Force
Copy-Item "models\qwen3_8_27b_q4\tokenizer.json" "$StageDir\models\qwen3_8_27b_q4\tokenizer.json" -Force
Copy-Item "models\qwen3_8_27b_q4\attention_dense_layers_q4.bin" "$StageDir\models\qwen3_8_27b_q4\attention_dense_layers_q4.bin" -Force

# 2. Create 7-Zip Archive (-mx0 for maximum packaging speed, store mode)
$Archive7z = "$DistDir\payload.7z"
if (Test-Path $Archive7z) { Remove-Item $Archive7z -Force }

Write-Host "Creating 64-bit archive payload..." -ForegroundColor Cyan
& $SevenZipExe a -t7z -mx0 "$Archive7z" "$StageDir\*"

# 3. Create SFX Config
$SfxConfig = "$DistDir\sfx_config.txt"
$ConfigContent = ";!@Install@!UTF-8!
Title=""Moecher Inference Engine - Qwen 3.8 27B INT4 Setup""
BeginPrompt=""Do you want to install Moecher Inference Engine (Qwen 3.8 27B INT4)?""
ExtractTitle=""Installing Moecher Inference Engine...""
InstallPath=""D:\\Moecher""
GUIFlags=""8""
ExecuteFile=""powershell.exe""
ExecuteParameters=""-ExecutionPolicy Bypass -File create_shortcut.ps1""
;!@InstallEnd@!"
[System.IO.File]::WriteAllText($SfxConfig, $ConfigContent, [System.Text.Encoding]::UTF8)

# 4. Concatenate SFX + Config + 7z payload into single standalone Setup.exe
$FinalExe = "$DistDir\Moecher-Qwen3.8-Setup.exe"
Write-Host "Assembling 64-bit standalone installer: $FinalExe..." -ForegroundColor Cyan

cmd.exe /c "copy /b `"$SevenZipSfx`" + `"$SfxConfig`" + `"$Archive7z`" `"$FinalExe`""

# 5. Clean up temporary stage files
Remove-Item $Archive7z -Force -ErrorAction SilentlyContinue
Remove-Item $SfxConfig -Force -ErrorAction SilentlyContinue
Remove-Item $StageDir -Recurse -Force -ErrorAction SilentlyContinue

$exeItem = Get-Item $FinalExe
$exeSizeGB = [math]::Round($exeItem.Length / 1GB, 2)
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Successfully created 64-bit Standalone Setup Executable!" -ForegroundColor Green
Write-Host "  Path: $FinalExe ($exeSizeGB GB)" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
