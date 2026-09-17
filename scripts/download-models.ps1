<#
.SYNOPSIS
    Downloads the Vosk model used for wake-word detection ("rowan") into models\.

.DESCRIPTION
    By default it downloads vosk-model-small-en-us-0.15 (~40 MB) and unpacks it
    into models\vosk-model-small-en-us-0.15 - exactly the path configured in
    config.example.yaml (client.wakeword.vosk_model).
    The wake word is English, so the model is the small English one: it runs
    continuously and has to be fast.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Force
#>
[CmdletBinding()]
param(
    [string]$ModelUrl = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    [string]$ModelsDir,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ModelsDir) { $ModelsDir = Join-Path $RepoRoot "models" }

function Write-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

$zipName = [System.IO.Path]::GetFileName($ModelUrl)
$modelName = [System.IO.Path]::GetFileNameWithoutExtension($zipName)
$targetDir = Join-Path $ModelsDir $modelName

Write-Step "Vosk model: $modelName"
Write-Host "Models directory: $ModelsDir"

if ((Test-Path $targetDir) -and (-not $Force)) {
    Write-Host "The model is already in place: $targetDir (re-download with -Force)" -ForegroundColor Green
    Write-Host "In config.yaml: client.wakeword.vosk_model: models/$modelName"
    return
}

if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
}
if ((Test-Path $targetDir) -and $Force) {
    Write-Host "Removing the old copy: $targetDir"
    Remove-Item -Recurse -Force $targetDir
}

$zipPath = Join-Path $ModelsDir $zipName

Write-Step "Downloading $ModelUrl"
try {
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
}
catch {
    Write-Verbose "Could not set TLS 1.2: $($_.Exception.Message)"
}
$progress = $ProgressPreference
$ProgressPreference = "SilentlyContinue"   # without this Invoke-WebRequest downloads very slowly
try {
    Invoke-WebRequest -Uri $ModelUrl -OutFile $zipPath -UseBasicParsing
}
finally {
    $ProgressPreference = $progress
}

$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "Downloaded: $zipPath ($sizeMb MB)"

Write-Step "Unpacking"
Expand-Archive -Path $zipPath -DestinationPath $ModelsDir -Force
Remove-Item $zipPath -Force

if (-not (Test-Path $targetDir)) {
    # The archive unpacked under a different name - locate the model directory by its conf\ folder.
    $found = Get-ChildItem -Path $ModelsDir -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "conf") } |
        Select-Object -First 1
    if ($found) {
        $targetDir = $found.FullName
        $modelName = $found.Name
    }
    else {
        throw "The archive was unpacked, but no model directory was found in $ModelsDir"
    }
}

Write-Step "Done"
Write-Host "Model: $targetDir" -ForegroundColor Green
Write-Host "Put this into config.yaml:"
Write-Host "  client:"
Write-Host "    wakeword:"
Write-Host "      vosk_model: models/$modelName"
