<#
Запускает аудит комнатного ПК В ИНТЕРАКТИВНОЙ СЕССИИ пользователя.

Важно и неочевидно: ssh на Windows попадает в отдельную сессию, и оттуда не
видно ни одного окна рабочего стола — проверено 2026-09-23: `EnumWindows`
вернул 0 видимых окон, хотя Chrome был открыт, и драйвер браузера честно
ответил «No matching ordinary browser window is open». Клиент же работает в
сессии вошедшего пользователя (задача JarvisRoomClient/RowanRoomClient), и
только там переходы, окна и нажатия имеют смысл.

Поэтому аудит ставится разовой задачей `schtasks /IT` (интерактивной), ждёт
отчёт `data\room-audit.json`, печатает его и задачу убирает за собой.

  powershell -NoProfile -ExecutionPolicy Bypass -File <root>\tests\room\audit_task.ps1 -Root <root>
#>
[CmdletBinding()]
param(
    [string]$Root,
    [switch]$Extended,
    [double]$Pause = 0.15,
    [int]$TimeoutSeconds = 420
)

$ErrorActionPreference = 'Continue'
if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$task = 'RowanRoomAudit'
$report = Join-Path $Root 'data\room-audit.json'
$runner = Join-Path $Root 'tests\room\run-audit.ps1'
$extra = if ($Extended) { ' -Extended' } else { '' }
$command = "powershell -NoProfile -ExecutionPolicy Bypass -File $runner -Pause $Pause$extra"

# Прошлый прогон мог оставить отчёт и задачу: отчёт убираем, задачу пересоздаём.
if (Test-Path -LiteralPath $report) { Remove-Item -LiteralPath $report -Force }
& schtasks /Delete /TN $task /F 2>&1 | Out-Null
$created = & schtasks /Create /TN $task /TR $command /SC ONCE /ST 00:00 /IT /F 2>&1
Write-Host ($created -join ' ')
if ($LASTEXITCODE -ne 0) {
    Write-Host 'не удалось создать интерактивную задачу: аудит не запускался' -ForegroundColor Red
    exit 1
}

& schtasks /Run /TN $task 2>&1 | ForEach-Object { Write-Host $_ }
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$reportText = ''
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if (Test-Path -LiteralPath $report) {
        $reportText = Get-Content -LiteralPath $report -Raw -Encoding UTF8
        if ($reportText -match '"summary"') { break }
    }
}

$state = & schtasks /Query /TN $task /FO LIST 2>&1 | Out-String
& schtasks /Delete /TN $task /F 2>&1 | Out-Null

if ($reportText) {
    Write-Host $reportText
} else {
    Write-Host 'отчёта нет: задача не отработала' -ForegroundColor Red
    Write-Host $state
    exit 1
}
exit 0
