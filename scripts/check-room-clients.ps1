<#
Что РЕАЛЬНО происходит на комнатных ПК, а не на этом (мозговом).

Владелец справедливо заметил, что живой стенд `scripts/live-eval.py` гоняет
запросы через хаб на этом ПК и подражает клиенту здесь же: экран, браузер и
клики в нём - здешние. Этот скрипт смотрит на настоящие комнатные ПК по SSH:

  * жив ли процесс клиента;
  * что клиент писал последним (его собственный data\client.log);
  * совпадает ли client\camera.py с версией в этом рабочем каталоге.

  pwsh -File scripts\check-room-clients.ps1
#>
[CmdletBinding()]
param([string]$Root)

if (-not $Root) {
    $Root = Split-Path -Parent $PSScriptRoot
}
$inventory = Get-Content (Join-Path $Root 'deploy\room-pcs.json') -Raw | ConvertFrom-Json
$local = Join-Path $Root 'client\camera.py'
$localHash = if (Test-Path $local) { (Get-FileHash $local -Algorithm SHA256).Hash } else { '' }

foreach ($pc in $inventory) {
    Write-Host "=== $($pc.name) ($($pc.user)@$($pc.host)) ===" -ForegroundColor Cyan
    $ssh = @('-i', $pc.key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
             '-o', 'ConnectTimeout=8', "$($pc.user)@$($pc.host)")
    # Без /fi: кавычки не переживают ssh + cmd + PowerShell. Фильтруем здесь.
    $process = & ssh @ssh 'tasklist /fo csv /nh' 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ПК не отвечает: $process" -ForegroundColor Red
        continue
    }
    $running = ($process | Select-String 'python' -SimpleMatch).Count
    Write-Host "  процессы python: $running" -ForegroundColor $(if ($running) { 'Green' } else { 'Red' })

    $log = Join-Path $pc.root 'data\client.log'
    $tail = & ssh @ssh "powershell -NoProfile -Command Get-Content -Tail 6 '$log'" 2>&1
    Write-Host "  последние строки $log :"
    $tail | ForEach-Object { "    $_" }

    $hash = & ssh @ssh "powershell -NoProfile -Command (Get-FileHash '$($pc.root)\client\camera.py' -Algorithm SHA256).Hash" 2>&1
    $remoteHash = ($hash | Select-Object -First 1).ToString().Trim()
    $same = $remoteHash -eq $localHash
    Write-Host ("  client\camera.py: {0} (здесь {1})" -f $remoteHash,
        $(if ($same) { 'то же' } else { 'ДРУГОЙ - нужен scripts\update-room-pcs.ps1' })) `
        -ForegroundColor $(if ($same) { 'Green' } else { 'Yellow' })
}

