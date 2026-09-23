<#
Запускает room_audit.py на комнатном ПК тем же python, которым работает клиент.

Лежит рядом с проверками и копируется туда вместе с ними: удалённая команда
должна быть одним словом без кавычек и каналов.
#>
[CmdletBinding()]
param(
    [string]$Root,
    [string]$Python,
    [switch]$Extended,
    [double]$Pause = 0.15
)

if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
if (-not $Python) {
    # Самый честный источник: интерпретатор уже запущенного клиента.
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
$report = Join-Path $Root 'data\room-audit.json'
$arguments = @((Join-Path $PSScriptRoot 'room_audit.py'), '--json', $report, '--pause', $Pause)
if ($Extended) { $arguments += '--extended' }
& $Python @arguments
exit $LASTEXITCODE
