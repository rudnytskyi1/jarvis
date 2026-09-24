<#
.SYNOPSIS
    Поднять клиента Rowan на комнатных ПК, которые уже включены.

.DESCRIPTION
    Владелец: «на buro pc запусти». `update-room-pcs.ps1` обновляет код и
    перезапускает клиента, но для «просто запусти» этого много: скрипт ходит по
    инвентарю `deploy/room-pcs.json`, печатает состояние задачи клиента ДО и
    ПОСЛЕ, поднимает её через `Start-ScheduledTask`, если она стоит, и отдельно
    говорит, если задачи нет или ПК не отвечает по SSH. Ничего не «дорисовывает»:
    что реально напечатал ПК, то и в отчёте.

    Задача клиента на AntonDorm называется `JarvisRoomClient`, на ПК-клоне
    публичного релиза (buro) — `RowanRoomClient`; скрипт берёт имя из инвентаря
    (`task`) и пробует его, а если имени нет — оба известных.

.EXAMPLE
    pwsh -File scripts\start-room-client.ps1 -Only buro
.EXAMPLE
    pwsh -File scripts\start-room-client.ps1
#>
[CmdletBinding()]
param(
    [string]$Inventory,
    [string[]]$Only = @(),
    # Перезапуск: остановить клиента и поднять заново (владелец: «запусти на buro pc
    # (рестарт)»). Останавливается ТОЛЬКО процесс самого клиента: на комнатном ПК
    # живут и чужие python-программы (на buro — camwatch друга), и глушить их
    # нельзя.
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Inventory) { $Inventory = Join-Path $repo 'deploy\room-pcs.json' }
if (-not (Test-Path -LiteralPath $Inventory)) { throw "Inventory not found: $Inventory" }

$sshExe = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
if (-not (Test-Path -LiteralPath $sshExe)) { throw "OpenSSH client not found: $sshExe" }

# Что делает ПК: смотрит именованную задачу, поднимает её и печатает правду.
$remoteScript = @'
$ErrorActionPreference = 'Continue'
$names = @('__TASK__')
$task = $null
foreach ($name in $names) {
    $found = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($found) { $task = $found; break }
}
if (-not $task) {
    Write-Output ('no-task: ' + ($names -join '/'))
    Write-Output ('python-running: ' + [bool](Get-Process python -ErrorAction SilentlyContinue))
    exit 0
}
Write-Output ('task=' + $task.TaskName)
Write-Output ('state-before=' + $task.State)
$info = Get-ScheduledTaskInfo -TaskName $task.TaskName -ErrorAction SilentlyContinue
if ($info) {
    Write-Output ('last-run=' + $info.LastRunTime + ' result=' + $info.LastTaskResult)
}
if ($task.State -eq 'Disabled') {
    # Аудит и запуски «в фоне» умеют выключать задачу; без этого Start молча
    # падает с 0x80041326, и выглядит это как «ПК не запускается».
    try {
        Enable-ScheduledTask -TaskName $task.TaskName | Out-Null
        Write-Output 'enabled=yes'
    } catch {
        Write-Output ('enable-failed: ' + $_.Exception.Message)
    }
    $task = Get-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
}
if ($task.State -ne 'Running') {
    try {
        Start-ScheduledTask -TaskName $task.TaskName
        Start-Sleep -Seconds 5
    } catch {
        Write-Output ('start-failed: ' + $_.Exception.Message)
    }
    $task = Get-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
}
Write-Output ('state-after=' + $(if ($task) { $task.State } else { 'gone' }))
$loggedIn = (query user) 2>$null
Write-Output ('logged-in=' + $(if ($loggedIn) { 'yes' } else { 'no' }))
$process = Get-Process python -ErrorAction SilentlyContinue | Select-Object -First 1
Write-Output ('python-started=' + $(if ($process) { $process.StartTime.ToString('HH:mm:ss') } else { 'none' }))
Write-Output ('hostname=' + $env:COMPUTERNAME)
'@

# Остановка клиента перед перезапуском: своя задача, свой процесс, ничего чужого.
$restartScript = @'
$ErrorActionPreference = 'Continue'
$task = '__TASK__'
Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
# Только сам клиент Rowan: `python -m client.main --config ...`. Фильтр по
# имени каталога проекта здесь не годится — «rowanai» в пути поймало бы любую
# чужую программу, положенную рядом (владелец 2026-09-23: «другие процессы
# питона не трогай»).
$mine = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match '(?i)client[\\/\.]main' })
foreach ($item in $mine) {
    Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Output ('stopped-client-processes=' + $mine.Count)
Write-Output ('python-others-still-running=' + ([bool](Get-Process python -ErrorAction SilentlyContinue)))
Start-Sleep -Seconds 2
'@

$inventoryItems = Get-Content -LiteralPath $Inventory -Raw | ConvertFrom-Json
$failed = 0
foreach ($pc in $inventoryItems) {
    if ($Only.Count -gt 0 -and $Only -notcontains $pc.name) { continue }
    Write-Host "=== $($pc.name) ($($pc.host)) ===" -ForegroundColor Cyan
    $taskNames = @()
    if ($pc.task) { $taskNames += [string]$pc.task }
    foreach ($fallback in 'JarvisRoomClient', 'RowanRoomClient') {
        if ($taskNames -notcontains $fallback) { $taskNames += $fallback }
    }
    $script = $remoteScript.Replace("'__TASK__'", (($taskNames | ForEach-Object { "'$_'" }) -join ', '))
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    $arguments = @('-i', [string]$pc.key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
                   '-o', 'ConnectTimeout=10', "$($pc.user)@$($pc.host)",
                   "powershell -NoProfile -EncodedCommand $encoded")
    $output = @()
    if ($Restart) {
        $stop = $restartScript.Replace("'__TASK__'", "'$([string]$pc.task)'")
        $stopEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($stop))
        $stopArguments = @('-i', [string]$pc.key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
                           '-o', 'ConnectTimeout=10', "$($pc.user)@$($pc.host)",
                           "powershell -NoProfile -EncodedCommand $stopEncoded")
        $output += & $sshExe @stopArguments 2>&1
    }
    $output += & $sshExe @arguments 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "  ПК не ответил по SSH (код $code)" -ForegroundColor Red
        $output | ForEach-Object { "  $_" }
        $failed++
        continue
    }
    $output | ForEach-Object { "  $_" }
    $report = ($output | ForEach-Object { "$_" }) -join ' '
    if ($report -match 'state-after=Running') {
        Write-Host '  клиент работает' -ForegroundColor Green
    } else {
        Write-Host '  клиент НЕ работает (см. строки выше)' -ForegroundColor Yellow
        $failed++
    }
}

Write-Host ''
Write-Host 'Подключение к хабу видно в логе: data\server.log (строки Client ... connected).' -ForegroundColor Cyan
exit ([int]($failed -gt 0))
