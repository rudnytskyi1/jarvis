<#
.SYNOPSIS
    Starts the Jarvis room client (mic, wake word, speakers, devices).

.DESCRIPTION
    Finds the Python of the `jarvis` environment (conda or .venv) and runs
    `python -m client.main --config <config.yaml>` with the working directory set
    to the repo root, so that the common/ package and the relative Vosk model path
    (models\vosk-model-small-en-us-0.15) resolve correctly.
    Ctrl+C stops the client gracefully.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run-client.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run-client.ps1 -Config D:\jarvis\config.yaml
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
# Known interpreter of the `jarvis` env on the brain PC; on the room PC the conda
# lookup below finds the miniconda3 copy of the same env.
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

    throw "Environment '$EnvName' not found. Run scripts\install-client.ps1 first."
}

if (-not $Config) { $Config = Join-Path $RepoRoot "config.yaml" }
if (-not (Test-Path $Config)) {
    throw "Config $Config not found. Copy config.example.yaml to config.yaml and adjust it."
}

$python = Resolve-Python
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# Секреты клиента живут в окружении, а не в конфиге (ТЗ 4.3). Владелец
# 2026-09-23: комната подключалась к хабу без токена вообще, поэтому у хаба не
# было её ``home_id`` — и облачное чтение реплики, запись лиц, тела и убеждений,
# облачный взгляд на кадр молча выключались. Токен выдаёт хаб один раз
# (`scripts/issue-client-token.py`), на ПК комнаты он попадает строкой
# ``ROWAN_CLIENT_TOKEN=...`` в ``.env`` рядом с конфигом; имя переменной берётся
# из ``client.token_env``. Уже заданная в окружении переменная важнее файла.
$jarvisDotEnv = Join-Path $RepoRoot '.env'
if (Test-Path -LiteralPath $jarvisDotEnv) {
    $jarvisLoaded = @()
    foreach ($line in [IO.File]::ReadAllLines($jarvisDotEnv)) {
        $text = $line.Trim()
        if (-not $text -or $text.StartsWith('#')) { continue }
        $split = $text.IndexOf('=')
        if ($split -lt 1) { continue }
        $name = $text.Substring(0, $split).Trim()
        if ($name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { continue }
        $value = $text.Substring($split + 1).Trim().Trim('"').Trim("'")
        if (-not $value) { continue }
        if (-not [Environment]::GetEnvironmentVariable($name)) {
            Set-Item -Path "env:$name" -Value $value
            $jarvisLoaded += $name
        }
    }
    if ($jarvisLoaded.Count -gt 0) {
        Write-Host ('Loaded from .env: ' + ($jarvisLoaded -join ', ')) -ForegroundColor Green
    }
}

$voskDir = Join-Path $RepoRoot "models"
if (-not (Test-Path $voskDir)) {
    Write-Warning "The models\ directory is missing - the Vosk model was not downloaded. Run: scripts\download-models.ps1"
}

Write-Host "Jarvis client: $python -m client.main --config $Config" -ForegroundColor Cyan

Push-Location $RepoRoot
try {
    if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
        & $python -m client.main --config $Config @ExtraArgs
    }
    else {
        & $python -m client.main --config $Config
    }
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($null -eq $code) { $code = 0 }
exit $code
