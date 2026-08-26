; ─────────────────────────────────────────────────────────────────────────────
;  Inno Setup Script for Moecher Inference Engine + Qwen 3.8 27B INT4
; ─────────────────────────────────────────────────────────────────────────────

#define MyAppName "Moecher Inference Engine (Qwen 3.8 27B INT4)"
#define MyAppVersion "2.05"
#define MyAppPublisher "MinnieTheMoEcher Project"
#define MyAppExeName "moecher.exe"

[Setup]
AppId={{C8E9A4B2-7104-4F93-A65E-91F26815B3DE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={sd}\Moecher
DefaultGroupName=Moecher Qwen Server
AllowNoIcons=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=D:\dev\minniemoe\MinnieTheMoEcher\dist
OutputBaseFilename=Moecher-Qwen3.8-Setup
DiskSpanning=yes
DiskSliceSize=max
SlicesPerDisk=1
Compression=none
SolidCompression=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}
DisableWelcomePage=no
DisableDirPage=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Core Binaries & CUDA Runtime
Source: "D:\dev\minniemoe\MinnieTheMoEcher\build\Release\moecher.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublas64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublasLt64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; Launcher & Utility Scripts
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\start_qwen_server.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\test_qwen.bat"; DestDir: "{app}"; Flags: ignoreversion

; Web UI Assets
Source: "D:\dev\minniemoe\MinnieTheMoEcher\web\*"; DestDir: "{app}\web"; Flags: ignoreversion recursesubdirs createallsubdirs

; Model Assets (Qwen 3.8 27B INT4 block-32)
Source: "D:\dev\minniemoe\MinnieTheMoEcher\models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json"; DestDir: "{app}\models\qwen3_8_27b_q4"; Flags: ignoreversion
Source: "D:\dev\minniemoe\MinnieTheMoEcher\models\qwen3_8_27b_q4\tokenizer.json"; DestDir: "{app}\models\qwen3_8_27b_q4"; Flags: ignoreversion
Source: "D:\dev\minniemoe\MinnieTheMoEcher\models\qwen3_8_27b_q4\attention_dense_layers_q4.bin"; DestDir: "{app}\models\qwen3_8_27b_q4"; Flags: ignoreversion

[Icons]
Name: "{group}\Start Moecher Qwen Server"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"
Name: "{group}\Test Qwen API"; Filename: "{app}\test_qwen.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall Moecher"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Moecher Qwen Server"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\start_qwen_server.bat"; Description: "Launch Moecher Qwen Server now"; Flags: nowait postinstall skipifsilent shellexec
