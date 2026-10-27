#define AppName "SignBridge"
#define AppVersion "1.0.0"
#define AppPublisher "SignBridge"

[Setup]
AppId={{6B6E5C31-7B0C-4B2D-9F31-5D6B39F9D5A1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\SignBridge
DefaultGroupName=SignBridge
OutputDir=output
OutputBaseFilename=SignBridge-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\SignBridge.exe

[Files]
Source: "dist\SignBridge\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\SignBridge"; Filename: "{app}\SignBridge.exe"
Name: "{autodesktop}\SignBridge"; Filename: "{app}\SignBridge.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "إنشاء اختصار على سطح المكتب"; GroupDescription: "اختصارات إضافية:"

[Run]
Filename: "{app}\SignBridge.exe"; Description: "تشغيل SignBridge"; Flags: postinstall nowait skipifsilent
