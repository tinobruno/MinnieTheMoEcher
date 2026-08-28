; ─────────────────────────────────────────────────────────────────────────────
;  Inno Setup Script: Moecher Inference Engine (Online Installer)
;  With hardware parameter configuration and dual model shortcuts
; ─────────────────────────────────────────────────────────────────────────────

#define MyAppName "Moecher Inference Engine"
#define MyAppVersion "2.05"
#define MyAppPublisher "MinnieTheMoEcher Project"
#define MyAppExeName "moecher.exe"

[Setup]
AppId={{C8E9A4B2-7104-4F93-A65E-91F26815B3DE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={sd}\Moecher
DefaultGroupName=Moecher
AllowNoIcons=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=D:\dev\minniemoe\MinnieTheMoEcher\dist
OutputBaseFilename=Moecher-Setup
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile=D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher.ico
WizardImageFile=D:\dev\minniemoe\MinnieTheMoEcher\graphics\installer_wizard.bmp
WizardSmallImageFile=D:\dev\minniemoe\MinnieTheMoEcher\graphics\installer_small.bmp
UninstallDisplayIcon={app}\{#MyAppExeName}
DisableWelcomePage=no
DisableDirPage=no

[Types]
Name: "full"; Description: "Standard Installation (Server, Web UI, and Model selection)"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Moecher Engine Core & Web UI (~40 MB)"; Types: full custom; Flags: fixed
Name: "dl_qwen"; Description: "Download Qwen 3.8 27B INT4 (~18.7 GB) from Hugging Face"; Types: full custom
Name: "dl_deepseek"; Description: "Download DeepSeek V4 Flash IQ2 - Standard (~81.4 GB) from Hugging Face"; Types: custom
Name: "dl_deepseek_q4"; Description: "Download DeepSeek V4 Flash Q4 - 8GB GPU Mode (~78.8 GB) from Hugging Face"; Types: custom

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Core Binaries & CUDA 13 Runtime
Source: "D:\dev\minniemoe\MinnieTheMoEcher\build\Release\moecher.exe"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublas64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: core
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublasLt64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: core

; Launchers, Scripts, Icons & Web UI
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\start_qwen_server.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\start_deepseek_server.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\start_deepseek_q4_server.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\test_qwen.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\download_model.ps1"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher.ico"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\web\*"; DestDir: "{app}\web"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: core

[Icons]
; Start Menu Shortcuts
Name: "{group}\Moecher (Qwen 3.8 27B)"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"
Name: "{group}\Moecher (DeepSeek V4 Flash Standard)"; Filename: "{app}\start_deepseek_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"
Name: "{group}\Moecher (DeepSeek V4 Flash 8GB GPU)"; Filename: "{app}\start_deepseek_q4_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"
Name: "{group}\Test API"; Filename: "{app}\test_qwen.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall Moecher"; Filename: "{uninstallexe}"

; Desktop Shortcuts
Name: "{autodesktop}\Moecher (Qwen 3.8 27B)"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"; Tasks: desktopicon
Name: "{autodesktop}\Moecher (DeepSeek V4 Flash Standard)"; Filename: "{app}\start_deepseek_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"; Tasks: desktopicon
Name: "{autodesktop}\Moecher (DeepSeek V4 Flash 8GB GPU)"; Filename: "{app}\start_deepseek_q4_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"; Tasks: desktopicon

[Code]
var
  HardwarePage: TInputQueryWizardPage;

procedure InitializeWizard;
begin
  // Create Custom Hardware Configuration Page
  HardwarePage := CreateInputQueryPage(
    wpSelectComponents,
    'Hardware & Memory Configuration',
    'Configure GPU VRAM and CPU DRAM limits',
    'Specify memory allocation for Moecher engine (you can edit these in .bat files later):'
  );
  
  HardwarePage.Add('Max GPU VRAM (GB) [0 = Auto/all available]:', False);
  HardwarePage.Values[0] := '0';
  
  HardwarePage.Add('DRAM / CPU Cache limit (GB) [0 = Default, 64 for DeepSeek]:', False);
  HardwarePage.Values[1] := '0';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  MaxVram: String;
  DramCache: String;
  QwenBatContent: String;
  DeepSeekBatContent: String;
  DeepSeekQ4BatContent: String;
  AppDir: String;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    AppDir := ExpandConstant('{app}');
    MaxVram := HardwarePage.Values[0];
    if MaxVram = '' then MaxVram := '0';
    
    DramCache := HardwarePage.Values[1];
    if DramCache = '' then DramCache := '0';
    
    // Customize start_qwen_server.bat
    QwenBatContent :=
      '@echo off' + #13#10 +
      'setlocal' + #13#10 +
      'chcp 65001 >nul' + #13#10 +
      'cd /d "%~dp0"' + #13#10 +
      'echo ===============================================================================' + #13#10 +
      'echo   Starting Moecher Server with Qwen 3.8 27B INT4' + #13#10 +
      'echo   Web UI: http://localhost:8001' + #13#10 +
      'echo ===============================================================================' + #13#10 +
      'start "Moecher Qwen Server" /high moecher.exe --manifest models\qwen3_8_27b_q4\moecher_manifest.json --max-vram ' + MaxVram + ' --dram-cache-gb ' + DramCache + ' --quiet' + #13#10;
    SaveStringToFile(AppDir + '\start_qwen_server.bat', QwenBatContent, False);
    
    // Customize start_deepseek_server.bat
    DeepSeekBatContent :=
      '@echo off' + #13#10 +
      'setlocal' + #13#10 +
      'chcp 65001 >nul' + #13#10 +
      'cd /d "%~dp0"' + #13#10 +
      'echo ===============================================================================' + #13#10 +
      'echo   Starting Moecher Server with DeepSeek V4 Flash IQ2' + #13#10 +
      'echo   Web UI: http://localhost:8001' + #13#10 +
      'echo ===============================================================================' + #13#10 +
      'start "Moecher DeepSeek Server" /high moecher.exe --manifest models\deepseek_v4_flash_iq2\moecher_manifest.json --max-vram ' + MaxVram + ' --dram-cache-gb ' + DramCache + ' --quiet' + #13#10;
    SaveStringToFile(AppDir + '\start_deepseek_server.bat', DeepSeekBatContent, False);

    // Customize start_deepseek_q4_server.bat
    DeepSeekQ4BatContent :=
      '@echo off' + #13#10 +
      'setlocal' + #13#10 +
      'chcp 65001 >nul' + #13#10 +
      'cd /d "%~dp0"' + #13#10 +
      'echo ===============================================================================' + #13#10 +
      'echo   Starting Moecher Server with DeepSeek V4 Flash Q4 (8GB GPU Mode)' + #13#10 +
      'echo   Web UI: http://localhost:8000' + #13#10 +
      'echo ===============================================================================' + #13#10 +
      'start "Moecher DeepSeek Q4 Server" /high moecher.exe --manifest models\deepseek_v4_flash_q4\moecher_manifest.json --max-vram ' + MaxVram + ' --dram-cache-gb ' + DramCache + ' --quiet' + #13#10;
    SaveStringToFile(AppDir + '\start_deepseek_q4_server.bat', DeepSeekQ4BatContent, False);
    
    // Trigger model download if selected
    if WizardIsComponentSelected('dl_qwen') and WizardIsComponentSelected('dl_deepseek') then
    begin
      Exec('powershell.exe', '-ExecutionPolicy Bypass -File "' + AppDir + '\download_model.ps1" -Model both -DestDir "' + AppDir + '\models"', AppDir, SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode);
    end
    else if WizardIsComponentSelected('dl_qwen') then
    begin
      Exec('powershell.exe', '-ExecutionPolicy Bypass -File "' + AppDir + '\download_model.ps1" -Model qwen -DestDir "' + AppDir + '\models"', AppDir, SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode);
    end
    else if WizardIsComponentSelected('dl_deepseek') then
    begin
      Exec('powershell.exe', '-ExecutionPolicy Bypass -File "' + AppDir + '\download_model.ps1" -Model deepseek -DestDir "' + AppDir + '\models"', AppDir, SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode);
    end
    else if WizardIsComponentSelected('dl_deepseek_q4') then
    begin
      Exec('powershell.exe', '-ExecutionPolicy Bypass -File "' + AppDir + '\download_model.ps1" -Model deepseek_q4 -DestDir "' + AppDir + '\models"', AppDir, SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;
