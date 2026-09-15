# Build StrelkaEditor (Release) for F5 / VS Code tasks on Windows.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Build = Join-Path $Root "build\Release"
if (-not $env:OPTIX_DIR) {
    $env:OPTIX_DIR = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
}

if (-not (Test-Path (Join-Path $Build "build.ninja"))) {
    Write-Error "configure Release first: $Build\build.ninja is missing"
}

function Normalize-CMakePath {
    param([string]$Path)
    if (-not $Path) { return $null }
    return ([System.IO.Path]::GetFullPath($Path)).ToLowerInvariant()
}

function Find-CMake {
    $conanCmake = Get-ChildItem (Join-Path $env:USERPROFILE ".conan2\p\cmake*\p\bin\cmake.exe") -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($conanCmake) { return $conanCmake.FullName }

    $onPath = Get-Command cmake -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $candidates = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }

    Write-Error "cmake not found (install VS CMake or run conan install)"
}

function Import-VcVars {
    if (Get-Command cl -ErrorAction SilentlyContinue) { return }

    $vcvars = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if (-not $vcvars) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $install = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($install) {
                $candidate = Join-Path $install "VC\Auxiliary\Build\vcvarsall.bat"
                if (Test-Path $candidate) { $vcvars = $candidate }
            }
        }
    }

    if (-not $vcvars) {
        Write-Error 'vcvarsall.bat not found; install "Desktop development with C++"'
    }

    cmd /c "`"$vcvars`" amd64 >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match "^(?<name>[^=]+?)=(?<value>.*)$") {
            Set-Item -Path "env:$($Matches.name)" -Value $Matches.value
        }
    }

    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        Write-Error "cl.exe not on PATH after vcvarsall"
    }
}

$cmake = Find-CMake
Import-VcVars

$conanEnv = Join-Path $Build "generators\conanbuildenv-release-x86_64.bat"
if (Test-Path $conanEnv) {
    cmd /c "`"$conanEnv`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match "^(?<name>[^=]+?)=(?<value>.*)$") {
            Set-Item -Path "env:$($Matches.name)" -Value $Matches.value
        }
    }
}

$cacheFile = Join-Path $Build "CMakeCache.txt"
$cachedCmake = $null
if (Test-Path $cacheFile) {
    $match = Select-String -Path $cacheFile -Pattern "^CMAKE_COMMAND:.*?=(.+)$" | Select-Object -First 1
    if ($match) {
        $cachedCmake = $match.Matches[0].Groups[1].Value.Trim()
    }
}

$needsReconfigure = $false
if (-not $cachedCmake -or -not (Test-Path $cachedCmake)) {
    $needsReconfigure = $true
} elseif ((Normalize-CMakePath $cachedCmake) -ne (Normalize-CMakePath $cmake)) {
    Write-Host "build: cmake changed ($cachedCmake -> $cmake)"
    $needsReconfigure = $true
}

if ($needsReconfigure) {
    Write-Host "build: refreshing CMake cache (previous cmake path missing)"
    $ninja = (Get-Command ninja).Source
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if (-not $pythonCmd) {
        Write-Error "python not found on PATH (needed for OpenPBR header generation)"
    }
    $python = $pythonCmd.Source
    $toolchain = Join-Path $Build "generators\conan_toolchain.cmake"
    & $cmake -S $Root -B $Build -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
        "-DCMAKE_MAKE_PROGRAM=$ninja" `
        "-DPython3_EXECUTABLE=$python"
}

& $cmake --build $Build --target StrelkaEditor -j 24
exit $LASTEXITCODE
