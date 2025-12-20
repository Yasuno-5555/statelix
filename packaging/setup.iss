[Setup]
AppName=Statelix
AppVersion=2.3
DefaultDirName={autopf}\Statelix
DefaultGroupName=Statelix
OutputBaseFilename=Statelix_Setup_v2.3
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\dist

[Files]
Source: "..\dist\Statelix\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Statelix"; Filename: "{app}\Statelix.exe"
Name: "{group}\{cm:UninstallProgram,Statelix}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\Statelix"; Filename: "{app}\Statelix.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
Filename: "{app}\Statelix.exe"; Description: "{cm:LaunchProgram,Statelix}"; Flags: nowait postinstall skipifsilent
