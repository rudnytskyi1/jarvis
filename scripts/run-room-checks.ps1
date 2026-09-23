<#
Прогон проверок комнатного клиента НА КАЖДОМ комнатном ПК.

Скрипт берёт инвентарь `deploy/room-pcs.json`, копирует туда
`tests/room/room_checks.py`, запускает его тем python, который стоит на этом
ПК, и складывает отчёты в `data/room-eval/<имя>.json`.

Живой стенд `scripts/live-eval.py` проверяет хаб и подражает клиенту на
мозговом ПК. Этот скрипт проверяет то, что видно только на комнатном ПК:
камеру, микрофон, HUD, устройства, браузер и собственный журнал клиента.

  pwsh -File scripts\run-room-checks.ps1
#>
[CmdletBinding()]
param(
    [string]$Root
)

if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
$inventory = Get-Content (Join-Path $Root 'deploy\room-pcs.json') -Raw | ConvertFrom-Json
$checks = Join-Path $Root 'tests\room\room_checks.py'
$runner = Join-Path $Root 'tests\room\run-checks.ps1'
$reports = Join-Path $Root 'data\room-eval'
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$failed = 0

foreach ($pc in $inventory) {
    Write-Host "=== $($pc.name) ===" -ForegroundColor Cyan
    $ssh = @('-i', $pc.key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
             '-o', 'ConnectTimeout=8', "$($pc.user)@$($pc.host)")
    $remote = "$($pc.root)\tests\room"
    $remoteReport = "$($pc.root)\data\room-checks.json"
    # scp создаёт каталог сам, а удалённая команда остаётся одним словом:
    # ssh -> cmd -> powershell не передаёт кавычки и каналы надёжно.
    $mkdir = 'mkdir "' + $remote + '"'
    & ssh @ssh $mkdir 2>&1 | Out-Null
    $destination = "$($pc.user)@$($pc.host):" + ($remote -replace '\\', '/')
    & scp -i $pc.key -o BatchMode=yes "$checks" "$destination/room_checks.py" 2>&1 | Out-Null
    & scp -i $pc.key -o BatchMode=yes "$runner" "$destination/run-checks.ps1" 2>&1 | Out-Null
    $output = & ssh @ssh "powershell -NoProfile -ExecutionPolicy Bypass -File $remote\run-checks.ps1" 2>&1
    $output | ForEach-Object { "  $_" }
    $text = & ssh @ssh "type $remoteReport" 2>&1
    $json = ($text -join "`n").Trim()
    if ($json) {
        $json | Out-File -FilePath (Join-Path $reports "$($pc.name).json") -Encoding utf8
        try {
            $report = $json | ConvertFrom-Json
            Write-Host ("  итог: {0}/{1} проверок прошло" -f $report.passed, $report.total) `
                -ForegroundColor $(if ($report.passed -eq $report.total) { 'Green' } else { 'Yellow' })
            $failed += ($report.total - $report.passed)
        } catch {
            Write-Host "  отчёт не разобрался: $($_.Exception.Message)" -ForegroundColor Red
            $failed++
        }
    } else {
        Write-Host "  отчёта нет (ПК не ответил или python не запустился)" -ForegroundColor Red
        $failed++
    }
}

Write-Host "всего не прошло: $failed" -ForegroundColor $(if ($failed) { 'Yellow' } else { 'Green' })
exit ([int]($failed -gt 0))
