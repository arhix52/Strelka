@echo off
cd /d "%~dp0..\build\Release"
set OPTIX_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0
if not exist StrelkaEditor.exe (
    echo StrelkaEditor.exe not found. Build first: Ctrl+Shift+B
    exit /b 1
)
start "" StrelkaEditor.exe
