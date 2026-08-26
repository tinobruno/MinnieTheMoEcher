; ─────────────────────────────────────────────────────────────────────────────
;  Inno Setup Script: Moecher Inference Engine & Web UI + Single Model .bin File
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
DiskSpanning=yes
DiskSliceSize=max
SlicesPerDisk=1
Compression=lzma2/fast
SolidCompression=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile=D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher.ico
WizardImageFile=D:\dev\minniemoe\MinnieTheMoEcher\graphics\installer_wizard.bmp
WizardSmallImageFile=D:\dev\minniemoe\MinnieTheMoEcher\graphics\installer_small.bmp
UninstallDisplayIcon={app}\{#MyAppExeName}
DisableWelcomePage=no
DisableDirPage=no

[Types]
Name: "full"; Description: "Full Installation (Server, Web UI, and Qwen 3.8 27B INT4 Model)"
Name: "serveronly"; Description: "Server & Web UI only (Reinstall/Update server without model .bin)"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Moecher Inference Engine & Web UI (Main executable)"; Types: full serveronly custom; Flags: fixed
Name: "model_qwen"; Description: "Qwen 3.8 27B INT4 Model (Stored in Moecher-Setup-1.bin, ~18.7 GB)"; Types: full custom

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Core Binaries, CUDA Runtime, & Web UI (Installed from main Setup.exe)
Source: "D:\dev\minniemoe\MinnieTheMoEcher\build\Release\moecher.exe"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublas64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: core
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublasLt64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\start_qwen_server.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\test_qwen.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher.ico"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\web\*"; DestDir: "{app}\web"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: core

; Model, Tokenizer, and Manifest Files (Stored in single Moecher-Setup-1.bin, nocompression prevents 32-bit CRC overflow)
Source: "D:\dev\minniemoe\MinnieTheMoEcher\models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json"; DestDir: "{app}\models\qwen3_8_27b_q4"; Flags: ignoreversion; Components: model_qwen
Source: "D:\dev\minniemoe\MinnieTheMoEcher\models\qwen3_8_27b_q4\tokenizer.json"; DestDir: "{app}\models\qwen3_8_27b_q4"; Flags: ignoreversion; Components: model_qwen
Source: "D:\dev\minniemoe\MinnieTheMoEcher\models\qwen3_8_27b_q4\attention_dense_layers_q4.bin"; DestDir: "{app}\models\qwen3_8_27b_q4"; Flags: ignoreversion nocompression; Components: model_qwen

[Icons]
Name: "{group}\Start Moecher Qwen Server"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"
Name: "{group}\Test Qwen API"; Filename: "{app}\test_qwen.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall Moecher"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Moecher Qwen Server"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\start_qwen_server.bat"; Description: "Launch Moecher Qwen Server now"; Flags: nowait postinstall skipifsilent shellexec
