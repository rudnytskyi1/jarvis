<#
Звук на КАЖДОМ комнатном ПК: глушение перед аудитом и возврат после него.

Владелец просил выключить звук на ПК, где идут проверки, чтобы он не мешал
(2026-09-22). Скрипт берёт инвентарь `deploy/room-pcs.json`, копирует рядом с
проверками `tests/room/room_audio.py` и `tests/room/run-audio.ps1`, зовёт на
каждом ПК тот же код, которым звук меняет сам ассистент
(`client/actions/pc.py`), и печатает честный итог по каждому ПК.

  pwsh -File scripts\room-audio.ps1 mute
  pwsh -File scripts\room-audio.ps1 unmute
  pwsh -File scripts\room-audio.ps1 volume_set -Value 30

Отчёт складывается в `data/room-eval/audio-<имя>.json`.
#>
[CmdletBinding()]
param(
    [ValidateSet('mute', 'unmute', 'volume_set')]
    [string]$Command = 'mute',
    [int]$Value = 0,
    [string]$Root
)

if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
$inventory = Get-Content (Join-Path $Root 'deploy\room-pcs.json') -Raw | ConvertFrom-Json
$audio = Join-Path $Root 'tests\room\room_audio.py'
$runner = Join-Path $Root 'tests\room\run-audio.ps1'
$reports = Join-Path $Root 'data\room-eval'
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$failed = 0

foreach ($pc in $inventory) {
    Write-Host "=== $($pc.name) ===" -ForegroundColor Cyan
    $ssh = @('-i', $pc.key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
             '-o', 'ConnectTimeout=8', "$($pc.user)@$($pc.host)")
    $remote = "$($pc.root)\tests\room"
    $destination = "$($pc.user)@$($pc.host):" + ($remote -replace '\\', '/')
    & ssh @ssh ('mkdir "' + $remote + '"') 2>&1 | Out-Null
    & scp -i $pc.key -o BatchMode=yes "$audio" "$destination/room_audio.py" 2>&1 | Out-Null
    & scp -i $pc.key -o BatchMode=yes "$runner" "$destination/run-audio.ps1" 2>&1 | Out-Null
    $output = & ssh @ssh "powershell -NoProfile -ExecutionPolicy Bypass -File $remote\run-audio.ps1 -Command $Command -Value $Value" 2>&1
    $output | ForEach-Object { "  $_" }
    $json = ($output | Where-Object { "$_".TrimStart().StartsWith('{') } | Select-Object -Last 1)
    if ($json) {
        $json.Trim() | Out-File -FilePath (Join-Path $reports "audio-$($pc.name).json") -Encoding utf8
        try {
            $parsed = $json | ConvertFrom-Json
            if ($parsed.ok) {
                Write-Host ("  {0}: ok ({1})" -f $Command, $parsed.detail) -ForegroundColor Green
            } else {
                Write-Host ("  {0}: НЕ вышло — {1}" -f $Command, $parsed.error) -ForegroundColor Red
                $failed++
            }
        } catch {
            Write-Host "  ответ не разобрался: $($_.Exception.Message)" -ForegroundColor Red
            $failed++
        }
    } else {
        Write-Host "  нет ответа (ПК не ответил или python не запустился)" -ForegroundColor Red
        $failed++
    }
}

Write-Host "не вышло: $failed" -ForegroundColor $(if ($failed) { 'Yellow' } else { 'Green' })
exit ([int]($failed -gt 0))
