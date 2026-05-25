@echo off
rem Build helper for MSVC on Windows.
rem Usage: msvc_build.bat (configure|build) (debug|relwithdebinfo)
rem
rem Clears MSVC environment variables inherited from any VS 2019 Developer
rem Command Prompt before initializing a clean VS 2022 environment.
rem This prevents STL header contamination when both VS versions are installed.

set ACTION=%1
set PRESET=windows-msvc-%2

rem Clear MSVC/SDK env vars that may be inherited from a VS 2019 dev shell
set INCLUDE=
set LIB=
set LIBPATH=
set VSCMD_VER=
set VSCMD_ARG_TGT_ARCH=
set VSCMD_ARG_HOST_ARCH=
set VSCMD_ARG_app_plat=
set VisualStudioVersion=
set VSINSTALLDIR=
set VCToolsInstallDir=
set VCToolsRedistDir=
set VCToolsVersion=

rem Minimal PATH: VS Installer (for vswhere), System32, PowerShell, Git
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer"
set "PATH=%PATH%;C:\Windows\System32"
set "PATH=%PATH%;C:\Windows\System32\WindowsPowerShell\v1.0"
set "PATH=%PATH%;C:\Windows"
set "PATH=%PATH%;C:\Program Files\Git\cmd"
set "PATH=%PATH%;C:\Program Files\Git\bin"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo [msvc_build] vcvarsall.bat failed
    exit /b 1
)

set CMAKE="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if "%ACTION%"=="configure" goto :configure
if "%ACTION%"=="build" goto :build
echo [msvc_build] Unknown action: %ACTION%
exit /b 1

:configure
%CMAKE% --preset %PRESET%
exit /b %ERRORLEVEL%

:build
%CMAKE% --build --preset %PRESET%
exit /b %ERRORLEVEL%
