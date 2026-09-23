<#
Запускает room_checks.py тем же python, которым работает клиент этого ПК.

Лежит рядом с проверками и копируется на комнатный ПК, чтобы удалённая команда
была одним словом без кавычек и каналов: ssh -> cmd -> powershell не умеет
их аккуратно передавать.
#>
[CmdletBinding()]
param([string]$Root, [string]$Python)

if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
if (-not $Python) {
    $resolver = Join-Path $Root 'scripts\client-python.ps1'
    if (Test-Path $resolver) {
        . $resolver
        try { $Python = Resolve-RowanPython -Root $Root } catch { $Python = '' }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
if ($Python -eq 'python') {
    # Самый честный источник: интерпретатор уже запущенного клиента.
    $running = Get-Process python -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty Path
    if ($running -and (Test-Path -LiteralPath $running)) { $Python = $running }
}
if ($Python -eq 'python' -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    # Антон-ПК: клиент живёт в conda-окружении jarvis, а «python» из PATH — заглушка
    # Microsoft Store. Ищем то, чем реально запускается клиент.
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
Write-Host "python: $Python"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$report = Join-Path $Root 'data\room-checks.json'
& $Python (Join-Path $PSScriptRoot 'room_checks.py') --json $report
exit $LASTEXITCODE
