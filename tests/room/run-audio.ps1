<#
Запускает room_audio.py на комнатном ПК тем же python, которым работает клиент.

Лежит рядом с room_audio.py и копируется туда вместе с ним: удалённая команда
должна быть одним словом без кавычек и каналов, иначе ssh -> cmd -> powershell
их не передаёт.

  powershell -NoProfile -ExecutionPolicy Bypass -File <root>\tests\room\run-audio.ps1 -Command mute
  powershell -NoProfile -ExecutionPolicy Bypass -File <root>\tests\room\run-audio.ps1 -Command unmute
  powershell -NoProfile -ExecutionPolicy Bypass -File <root>\tests\room\run-audio.ps1 -Command volume_set -Value 30
#>
[CmdletBinding()]
param(
    [string]$Root,
    [string]$Python,
    [ValidateSet('mute', 'unmute', 'volume_set')]
    [string]$Command = 'mute',
    [int]$Value = 0
)

if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
if (-not $Python) {
    $running = Get-Process python -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty Path
    if ($running -and (Test-Path -LiteralPath $running)) { $Python = $running }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
    foreach ($candidate in @("$Root\.rowan-python\python.exe",
                             "$env:USERPROFILE\anaconda3\envs\jarvis\python.exe",
                             "$env:USERPROFILE\miniconda3\envs\jarvis\python.exe",
                             "$env:USERPROFILE\anaconda3\envs\rowanai\python.exe",
                             "$env:USERPROFILE\miniconda3\envs\rowanai\python.exe",
                             'C:\ProgramData\miniconda3\envs\jarvis\python.exe',
                             'C:\Users\Anton\anaconda3\envs\jarvis\python.exe',
                             "$env:USERPROFILE\anaconda3\python.exe")) {
        if (Test-Path -LiteralPath $candidate) { $Python = $candidate; break }
    }
}
if (-not $Python) { $Python = 'python' }
Write-Host "python: $Python"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
& $Python (Join-Path $PSScriptRoot 'room_audio.py') --command $Command --value $Value
exit $LASTEXITCODE
