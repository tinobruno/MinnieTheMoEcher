; ─────────────────────────────────────────────────────────────────────────────
;  Inno Setup Script: Moecher Inference Engine + Web UI
;  Supports optional / external model installation
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
Name: "full"; Description: "Full Installation (Server, Web UI, and Qwen 3.8 27B Model)"
Name: "serveronly"; Description: "Server & Web UI only (Skip model files / Quick update)"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Moecher Inference Engine & Web UI"; Types: full serveronly custom; Flags: fixed
Name: "model_qwen"; Description: "Qwen 3.8 27B INT4 Model Files (copies models\qwen3_8_27b_q4 from setup folder)"; Types: full custom

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Core Binaries & CUDA 13 Runtime
Source: "D:\dev\minniemoe\MinnieTheMoEcher\build\Release\moecher.exe"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublas64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: core
Source: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64\cublasLt64_13.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist; Components: core

; Launchers, Scripts & Icon
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\start_qwen_server.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\installer\test_qwen.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher.ico"; DestDir: "{app}"; Flags: ignoreversion; Components: core

; Web UI Frontend
Source: "D:\dev\minniemoe\MinnieTheMoEcher\web\*"; DestDir: "{app}\web"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: core

; External Model Files (Separated from installer exe, copied directly from sibling models/ folder if selected)
Source: "models\qwen3_8_27b_q4\*"; DestDir: "{app}\models\qwen3_8_27b_q4"; Flags: external skipifsourcedoesntexist recursesubdirs; Components: model_qwen

[Icons]
Name: "{group}\Start Moecher Qwen Server"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"
Name: "{group}\Test Qwen API"; Filename: "{app}\test_qwen.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall Moecher"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Moecher Qwen Server"; Filename: "{app}\start_qwen_server.bat"; WorkingDir: "{app}"; IconFilename: "{app}\moecher.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\start_qwen_server.bat"; Description: "Launch Moecher Qwen Server now"; Flags: nowait postinstall skipifsilent shellexec
