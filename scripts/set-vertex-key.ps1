<#
.SYNOPSIS
    Put a Google Cloud service-account key where Vertex image generation finds it.

.DESCRIPTION
    Владелец 2026-09-23: «для генерации картинок теперь используй vertexai api
    (у меня бесплатные 300$ credits)». This script copies the JSON key that
    Google Cloud downloaded into data\vertex-credentials.json (a git-ignored
    directory), checks that the file really is a credential, and - when
    -Project is given - writes provider/project/location into the active config
    so no hand editing is needed.

    The private key is never printed and never leaves the machine.

.EXAMPLE
    pwsh -File scripts\set-vertex-key.ps1 -KeyPath "$env:USERPROFILE\Downloads\rowan-images-1a2b.json" -Project rowan-images-482301
.EXAMPLE
    pwsh -File scripts\set-vertex-key.ps1 -KeyPath D:\keys\vertex.json -Project my-project -Location us-central1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$KeyPath,
    [string]$Destination,
    [string]$Project,
    [string]$Location
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\Anton\anaconda3\envs\jarvis\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }

if (-not $Destination) { $Destination = Join-Path $repoRoot 'data\vertex-credentials.json' }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "Key file not found: $KeyPath" }

$info = Get-Item -LiteralPath $KeyPath
if ($info.Length -gt 65536) { throw "That does not look like a service-account key ($($info.Length) bytes)." }

try {
    $payload = Get-Content -LiteralPath $KeyPath -Raw | ConvertFrom-Json
} catch {
    throw "The file is not JSON: $($_.Exception.Message)"
}
$kind = [string]$payload.type
if ($kind -ne 'service_account' -and $kind -ne 'authorized_user') {
    throw "The file is type '$kind'; a service-account key has type 'service_account'."
}
if ($kind -eq 'service_account' -and -not $payload.client_email) {
    throw 'The service-account file has no client_email.'
}

$destinationFolder = Split-Path -Parent $Destination
if (-not (Test-Path -LiteralPath $destinationFolder)) {
    New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
}
if (Test-Path -LiteralPath $Destination) {
    $backup = "$Destination.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
    Move-Item -LiteralPath $Destination -Destination $backup
    Write-Host "Replaced the previous key (kept a copy at $backup)." -ForegroundColor Yellow
}
Copy-Item -LiteralPath $KeyPath -Destination $Destination -Force
Write-Host "Vertex credential saved to $Destination" -ForegroundColor Green
if ($kind -eq 'service_account') {
    Write-Host ("Service account: " + $payload.client_email)
    if ($payload.project_id) { Write-Host ("Key belongs to project: " + $payload.project_id) }
}

function Set-JarvisImageGenerationKey {
    <# Replace or add one ``server.image_generation`` key, keeping the file valid. #>
    param([string]$Path, [string]$Name, [string]$Value)
    $lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $Path)
    $blockIndex = -1
    $blockIndent = 0
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index] -match '^(\s*)image_generation:\s*(#.*)?$') {
            $blockIndex = $index
            $blockIndent = $matches[1].Length
            break
        }
    }
    if ($blockIndex -lt 0) { throw "No 'image_generation:' section in $Path." }
    $entryIndent = ' ' * ($blockIndent + 2)
    $end = $lines.Count
    for ($index = $blockIndex + 1; $index -lt $lines.Count; $index++) {
        if ($lines[$index] -match '^\s*$') { continue }
        if ($lines[$index] -match '^(\s*)\S' -and $matches[1].Length -le $blockIndent) {
            $end = $index
            break
        }
    }
    for ($index = $blockIndex + 1; $index -lt $end; $index++) {
        if ($lines[$index] -match "^$entryIndent$Name\s*:") {
            $lines[$index] = "$entryIndent$Name`: $Value"
            return $lines
        }
    }
    $lines.Insert($blockIndex + 1, "$entryIndent$Name`: $Value")
    return $lines
}

if ($Project -or $Location) {
    $config = Join-Path $repoRoot 'config.openai.yaml'
    if (-not (Test-Path -LiteralPath $config)) { $config = Join-Path $repoRoot 'config.yaml' }
    if (-not (Test-Path -LiteralPath $config)) { throw 'No config.openai.yaml or config.yaml to update.' }
    $backup = "$config.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
    Copy-Item -LiteralPath $config -Destination $backup
    $edited = Set-JarvisImageGenerationKey -Path $config -Name 'provider' -Value 'vertex'
    if ($Project) { $edited = Set-JarvisImageGenerationKey -Path $config -Name 'vertex_project' -Value $Project }
    if ($Location) { $edited = Set-JarvisImageGenerationKey -Path $config -Name 'vertex_location' -Value $Location }
    Set-Content -LiteralPath $config -Value $edited -Encoding UTF8
    Push-Location $repoRoot
    try {
        & $python -c "from common.config import load_config; g = load_config(r'$config').server.image_generation; print('provider=%s project=%s location=%s' % (g.provider, g.vertex_project, g.vertex_location))"
        if ($LASTEXITCODE -ne 0) {
            Copy-Item -LiteralPath $backup -Destination $config -Force
            throw 'The edited config does not load; the previous one was restored.'
        }
    } finally {
        Pop-Location
    }
    Write-Host "Config updated: $config (previous copy at $backup)" -ForegroundColor Green
}

Write-Host ''
Write-Host 'Next: restart the hub (start-jarvis-openai.bat), then check with' -ForegroundColor Cyan
Write-Host '  python scripts\vertex_image_probe.py' -ForegroundColor Cyan
if (-not $Project) {
    Write-Host 'Set server.image_generation.vertex_project to the Google Cloud project id' -ForegroundColor Yellow
    Write-Host '(or rerun this script with -Project <id>).' -ForegroundColor Yellow
}
