<#
Аудит действий НА КАЖДОМ комнатном ПК (десятки настоящих вызовов).

Берёт инвентарь `deploy/room-pcs.json`, копирует `tests/room/room_audit.py` и
запускает его тем python, которым на этом ПК работает клиент. Отчёты
складываются в `data/room-eval/audit-<имя>.json`.

Живой стенд (`scripts/live-eval.py`) проверяет понимание запросов на мозговом
ПК. Этот скрипт проверяет исполнение: громкость, буфер, окна, настоящий
браузер, камера, экран и отказы на опасные адреса и команды.

  pwsh -File scripts\room-audit.ps1
  pwsh -File scripts\room-audit.ps1 -Extended      # плюс медиа-клавиши

Звук после прогона остаётся выключенным, как просил владелец на время
аудита: вернуть его — `pwsh -File scripts\room-audio.ps1 unmute`.
#>
[CmdletBinding()]
param(
    [string]$Root,
    [switch]$Extended,
    [double]$Pause = 0.15
)

if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
$inventory = Get-Content (Join-Path $Root 'deploy\room-pcs.json') -Raw | ConvertFrom-Json
$audit = Join-Path $Root 'tests\room\room_audit.py'
$runner = Join-Path $Root 'tests\room\run-audit.ps1'
$wrapper = Join-Path $Root 'tests\room\audit_task.ps1'
$reports = Join-Path $Root 'data\room-eval'
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$failed = 0
$extra = if ($Extended) { ' -Extended' } else { '' }

foreach ($pc in $inventory) {
    Write-Host "=== $($pc.name) ===" -ForegroundColor Cyan
    $ssh = @('-i', $pc.key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
             '-o', 'ConnectTimeout=8', "$($pc.user)@$($pc.host)")
    $remote = "$($pc.root)\tests\room"
    $destination = "$($pc.user)@$($pc.host):" + ($remote -replace '\\', '/')
    & ssh @ssh ('mkdir "' + $remote + '"') 2>&1 | Out-Null
    & scp -i $pc.key -o BatchMode=yes "$audit" "$destination/room_audit.py" 2>&1 | Out-Null
    & scp -i $pc.key -o BatchMode=yes "$runner" "$destination/run-audit.ps1" 2>&1 | Out-Null
    & scp -i $pc.key -o BatchMode=yes "$wrapper" "$destination/audit_task.ps1" 2>&1 | Out-Null
    # Десктоп-действия возможны только в сессии вошедшего пользователя: ssh
    # видит ноль окон. Обёртка ставит разовую интерактивную задачу и печатает
    # её отчёт.
    $output = & ssh @ssh "powershell -NoProfile -ExecutionPolicy Bypass -File $remote\audit_task.ps1 -Root $($pc.root) -Pause $Pause$extra" 2>&1
    $output | ForEach-Object { "  $_" }
    $text = & ssh @ssh "type $($pc.root)\data\room-audit.json" 2>&1
    $json = ($text -join "`n").Trim()
    if (-not $json) {
        Write-Host "  отчёта нет (ПК не ответил или python не запустился)" -ForegroundColor Red
        $failed++
        continue
    }
    $json | Out-File -FilePath (Join-Path $reports "audit-$($pc.name).json") -Encoding utf8
    try {
        $report = $json | ConvertFrom-Json
        Write-Host ("  итог: {0}/{1} действий прошло" -f $report.passed, $report.total) `
            -ForegroundColor $(if ($report.passed -eq $report.total) { 'Green' } else { 'Yellow' })
        foreach ($item in $report.results) {
            if (-not $item.passed) {
                Write-Host ("    провал {0}: {1} {2}" -f $item.id, $item.tool, $item.detail) `
                    -ForegroundColor Red
            }
        }
        $failed += ($report.total - $report.passed)
    } catch {
        Write-Host "  отчёт не разобрался: $($_.Exception.Message)" -ForegroundColor Red
        $failed++
    }
}

Write-Host "всего не прошло: $failed" -ForegroundColor $(if ($failed) { 'Yellow' } else { 'Green' })
exit ([int]($failed -gt 0))
