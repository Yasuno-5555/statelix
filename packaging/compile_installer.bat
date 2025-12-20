@echo off
REM Statelix Installer Compilation Script v2
REM More robust ISCC detection

set "ISCC_PATH_X86=C:\Program Files (x86)\Inno Setup 6\iscc.exe"
set "ISCC_PATH_X64=C:\Program Files\Inno Setup 6\iscc.exe"
set "ISCC_PATH_LOCAL=%LOCALAPPDATA%\Programs\Inno Setup 6\iscc.exe"

echo Checking for Inno Setup...

if exist "%ISCC_PATH_X86%" (
    set "ISCC=%ISCC_PATH_X86%"
    goto :FOUND
)
if exist "%ISCC_PATH_X64%" (
    set "ISCC=%ISCC_PATH_X64%"
    goto :FOUND
)
if exist "%ISCC_PATH_LOCAL%" (
    set "ISCC=%ISCC_PATH_LOCAL%"
    goto :FOUND
)

REM Try Checking PATH
where iscc >nul 2>nul
if %errorlevel%==0 (
    set "ISCC=iscc"
    goto :FOUND
)

goto :NOTFOUND

:FOUND
echo Found ISCC at: "%ISCC%"
echo Compiling setup.iss...
"%ISCC%" "%~dp0setup.iss"
if errorlevel 1 (
    echo [ERROR] Compilation Failed!
    pause
    exit /b 1
) else (
    echo.
    echo ==========================================
    echo Installer Created Successfully!
    echo Location: ..\dist\Statelix_Setup_v2.3.exe
    echo ==========================================
    pause
    exit /b 0
)

:NOTFOUND
echo.
echo [ERROR] Inno Setup Compiler (iscc.exe) not found.
echo.
echo Checked:
echo  - %ISCC_PATH_X86%
echo  - %ISCC_PATH_X64%
echo  - %ISCC_PATH_LOCAL%
echo  - SYSTEM PATH
echo.
echo Please install Inno Setup 6 from https://jrsoftware.org/isdl.php
pause
exit /b 1
