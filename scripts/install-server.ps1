<#
.SYNOPSIS
    Installs the environment for the Jarvis brain server (the RTX 5090 PC).

.DESCRIPTION
    1. Finds the conda env `jarvis` (or creates it with Python 3.11; if conda is
       missing entirely, falls back to a plain venv in .venv at the repo root).
    2. Installs the dependencies from hub\requirements.txt.
    3. Creates config.yaml from config.example.yaml when it does not exist yet.
    4. Checks that Ollama is installed and prints the next steps.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-server.ps1
#>
[CmdletBinding()]
param(
    [string]$EnvName = "jarvis",
    [string]$PythonVersion = "3.11"
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
            # conda-forge with --override-channels: the default anaconda channels
            # require an interactive Terms-of-Service acceptance on fresh installs.
            & $conda create -y -n $EnvName "python=$PythonVersion" -c conda-forge --override-channels
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

Write-Step "Jarvis: installing the server side (STT + LLM client + TTS)"
Write-Host "Repository: $RepoRoot"

$requirements = Join-Path $RepoRoot "hub\requirements.txt"
if (-not (Test-Path $requirements)) {
    throw "$requirements not found - the repository was not downloaded completely."
}

$python = Resolve-Python

Write-Step "Upgrading pip"
& $python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip install --upgrade pip failed with exit code $LASTEXITCODE" }

Write-Step "Installing dependencies from hub\requirements.txt"
& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "pip install -r $requirements failed with exit code $LASTEXITCODE" }

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

Write-Step "Checking CUDA/torch (if torch is already installed)"
& $python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "torch does not import. Whisper will run on CPU only (config.yaml: stt.device=cpu, compute_type=int8)."
}

Write-Step "Checking Ollama"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "ollama: $($ollama.Source)"
    Write-Host "Pull the models if you have not done it yet:"
    Write-Host "  ollama pull qwen3:30b        # chat + tool calling"
    Write-Host "  ollama pull qwen3-vl:30b     # screen vision (look_at_screen)"
}
else {
    Write-Warning "Ollama not found. Install it from https://ollama.com/download, then run: ollama pull qwen3:30b and ollama pull qwen3-vl:30b"
}

Write-Step "Done"
Write-Host "Next steps:"
Write-Host "  1. Review config.yaml (the server section)."
Write-Host "  2. Open port 8765 in the firewall (once, from an elevated PowerShell) if the client connects over the LAN:"
Write-Host '     New-NetFirewallRule -DisplayName "Jarvis 8765" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765'
Write-Host "  3. Start the server:  scripts\run-server.ps1  (or double-click start-jarvis-server.bat for server + ngrok)"
