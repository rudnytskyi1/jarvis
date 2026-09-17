<#
.SYNOPSIS
    Installs the environment for the Jarvis room client (the PC at the TV).

.DESCRIPTION
    1. Finds the conda env `jarvis` (or creates it with Python 3.11; if conda is
       missing entirely, falls back to a plain venv in .venv at the repo root).
    2. Installs the dependencies from client\requirements.txt.
    3. Downloads the Vosk wake-word model (scripts\download-models.ps1).
    4. Creates config.yaml from config.example.yaml when it does not exist yet,
       then prints the audio device list and the Bluetooth status.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-client.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-client.ps1 -SkipModels
#>
[CmdletBinding()]
param(
    [string]$EnvName = "jarvis",
    [string]$PythonVersion = "3.11",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent $PSScriptRoot
# Known interpreter of the `jarvis` env: anaconda3 on the brain PC, miniconda3
# variants on the room PC. Checked before asking conda itself (much faster).
$KnownPython = "C:\Users\Anton\anaconda3\envs\jarvis\python.exe"

function Write-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Find-KnownEnvPython {
    param([string]$Name)
    if ($Name -ne "jarvis") { return $null }
    $candidates = @(
        $KnownPython,
        "$env:USERPROFILE\anaconda3\envs\$Name\python.exe",
        "$env:USERPROFILE\miniconda3\envs\$Name\python.exe",
        "$env:USERPROFILE\Anaconda3\envs\$Name\python.exe",
        "$env:USERPROFILE\Miniconda3\envs\$Name\python.exe",
        "$env:LOCALAPPDATA\anaconda3\envs\$Name\python.exe",
        "$env:LOCALAPPDATA\miniconda3\envs\$Name\python.exe",
        "C:\ProgramData\anaconda3\envs\$Name\python.exe",
        "C:\ProgramData\miniconda3\envs\$Name\python.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Find-CondaExe {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\Users\Anton\anaconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Get-CondaEnvPython {
    param([string]$CondaExe, [string]$Name)
    $lines = & $CondaExe env list
    foreach ($line in $lines) {
        if ($line -match '^\s*#') { continue }
        if ($line -match '^(\S+)\s+\*?\s*(\S.*?)\s*$') {
            if ($matches[1] -eq $Name) {
                $candidate = Join-Path $matches[2] "python.exe"
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }
    return $null
}

function Resolve-Python {
    $known = Find-KnownEnvPython -Name $EnvName
    if ($known) {
        Write-Host "conda env '$EnvName' found: $known"
        return $known
    }

    $conda = Find-CondaExe
    if ($conda) {
        Write-Host "conda: $conda"
        $python = Get-CondaEnvPython -CondaExe $conda -Name $EnvName
        if (-not $python) {
            Write-Step "Creating conda env '$EnvName' with Python $PythonVersion"
            & $conda create -y -n $EnvName "python=$PythonVersion"
            if ($LASTEXITCODE -ne 0) { throw "conda create failed with exit code $LASTEXITCODE" }
            $python = Get-CondaEnvPython -CondaExe $conda -Name $EnvName
        }
        if ($python) {
            Write-Host "Environment Python: $python"
            return $python
        }
        Write-Warning "Could not determine python.exe of env '$EnvName'; falling back to venv."
    }
    else {
        Write-Warning "conda not found - using a plain venv (.venv at the repo root)."
    }

    $venv = Join-Path $RepoRoot ".venv"
    $venvPython = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Step "Creating a venv in $venv"
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            & py "-$PythonVersion" -m venv $venv
            if ($LASTEXITCODE -ne 0) { & py -3 -m venv $venv }
        }
        else {
            $sys = Get-Command python -ErrorAction SilentlyContinue
            if (-not $sys) {
                throw "Neither conda nor python was found. Install Miniconda or Python $PythonVersion and run this script again."
            }
            & python -m venv $venv
        }
    }
    if (-not (Test-Path $venvPython)) { throw "Failed to create the venv in $venv" }
    Write-Host "Environment Python: $venvPython"
    return $venvPython
}

Write-Step "Jarvis: installing the client side (mic, wake word, speakers, devices)"
Write-Host "Repository: $RepoRoot"

$requirements = Join-Path $RepoRoot "client\requirements.txt"
if (-not (Test-Path $requirements)) {
    throw "$requirements not found - the repository was not downloaded completely."
}

$python = Resolve-Python

Write-Step "Upgrading pip"
& $python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip install --upgrade pip failed with exit code $LASTEXITCODE" }

Write-Step "Installing dependencies from client\requirements.txt"
& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "pip install -r $requirements failed with exit code $LASTEXITCODE" }

if ($SkipModels) {
    Write-Step "Skipping the Vosk model download (-SkipModels)"
}
else {
    Write-Step "Downloading the Vosk wake-word model"
    & (Join-Path $PSScriptRoot "download-models.ps1")
}

Write-Step "Checking the config"
$config = Join-Path $RepoRoot "config.yaml"
$example = Join-Path $RepoRoot "config.example.yaml"
if (-not (Test-Path $config)) {
    Copy-Item $example $config
    Write-Host "Created $config from config.example.yaml - review it before the first run." -ForegroundColor Yellow
}
else {
    Write-Host "config.yaml already exists - leaving it alone."
}

Write-Step "Audio devices (index/name for client.audio.input_device and output_device)"
& $python -c "import sounddevice; print(sounddevice.query_devices())"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "sounddevice could not list the devices - check that the microphone is plugged in and allowed in the Windows privacy settings."
}

Write-Step "Checking Bluetooth (needed for the SwitchBot Bot)"
try {
    $bt = Get-PnpDevice -Class Bluetooth -ErrorAction Stop | Where-Object { $_.Status -eq "OK" }
    if ($bt) {
        Write-Host "Bluetooth adapter found:"
        $bt | Select-Object -First 3 -Property FriendlyName, Status | Format-Table -AutoSize | Out-String | Write-Host
    }
    else {
        Write-Warning "No working Bluetooth adapter found. A SwitchBot Bot needs a USB BLE dongle (Bluetooth 4.0+ / BLE)."
    }
}
catch {
    Write-Warning "Could not query the Bluetooth devices: $($_.Exception.Message)"
}

Write-Step "Done"
Write-Host "Next steps:"
Write-Host "  1. Review config.yaml (the client section): server_url, devices, apps, microphone."
Write-Host "  2. A SwitchBot MAC can be found like this (the device advertises itself as WoHand):"
Write-Host "     `"$python`" -c `"import asyncio; from bleak import BleakScanner; print('\n'.join(f'{d.address}  {d.name}' for d in asyncio.run(BleakScanner.discover(timeout=8))))`""
Write-Host "  3. Start the client:  scripts\run-client.ps1  (or double-click start-jarvis-client.bat)"
