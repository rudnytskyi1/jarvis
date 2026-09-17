<#
.SYNOPSIS
    Starts the Jarvis brain server (STT + LLM + TTS) from the repo root.

.DESCRIPTION
    Finds the Python of the `jarvis` environment (conda or .venv) and runs
    `python -m server.main --config <config.yaml>` with the working directory set
    to the repo root, so that the common/ package is importable on both sides.
    Ctrl+C stops the server gracefully.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run-server.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run-server.ps1 -Config D:\jarvis\config.yaml
#>
[CmdletBinding()]
param(
    [string]$Config,
    [string]$EnvName = "jarvis",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent $PSScriptRoot
# Known interpreter of the `jarvis` env on the brain PC.
$KnownPython = "C:\Users\Anton\anaconda3\envs\jarvis\python.exe"

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
    if ($EnvName -eq "jarvis" -and (Test-Path $KnownPython)) { return $KnownPython }

    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $conda = Find-CondaExe
    if ($conda) {
        $python = Get-CondaEnvPython -CondaExe $conda -Name $EnvName
        if ($python) { return $python }
    }
    if (Test-Path $venvPython) { return $venvPython }

    throw "Environment '$EnvName' not found. Run scripts\install-server.ps1 first."
}

if (-not $Config) { $Config = Join-Path $RepoRoot "config.yaml" }
if (-not (Test-Path $Config)) {
    throw "Config $Config not found. Copy config.example.yaml to config.yaml and adjust it."
}

$python = Resolve-Python
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Jarvis server: $python -m server.main --config $Config" -ForegroundColor Cyan

Push-Location $RepoRoot
try {
    if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
        & $python -m server.main --config $Config @ExtraArgs
    }
    else {
        & $python -m server.main --config $Config
    }
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($null -eq $code) { $code = 0 }
exit $code
