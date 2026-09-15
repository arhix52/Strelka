# Run StrelkaEditor without a debugger.
param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildDir = Join-Path $Root "build\Release"
$Exe = Join-Path $BuildDir "StrelkaEditor.exe"
$BuildScript = Join-Path $PSScriptRoot "build_editor_win.ps1"

if (-not $env:OPTIX_DIR) {
    $env:OPTIX_DIR = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
}

if ($Build) {
    & $BuildScript
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path $Exe)) {
    Write-Error @"
StrelkaEditor not built: $Exe
Build first: Ctrl+Shift+B  (task: Build StrelkaEditor)
Or: Terminal -> Run Task -> Run StrelkaEditor  (build + run)
"@
}

Write-Host "Starting StrelkaEditor..."
if ($args.Count -gt 0) {
    Start-Process -FilePath $Exe -WorkingDirectory $BuildDir -ArgumentList $args
} else {
    Start-Process -FilePath $Exe -WorkingDirectory $BuildDir
}
exit 0
