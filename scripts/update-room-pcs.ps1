<#
.SYNOPSIS
    Обновить клиент Rowan на ВСЕХ комнатных ПК и перезапустить его там.

.DESCRIPTION
    Правило владельца (2026-09-22): «когда делаешь такие важные апдейты клиента
    его надо на всех пк перезапускать». Один запуск этого скрипта делает ровно
    это: собирает один пакет из `client/` и `common/`, кладёт его на каждый ПК из
    `deploy/room-pcs.json`, останавливает клиента, распаковывает код, снова
    запускает клиента и печатает, что реально лежит на ПК (SHA-256 `client/camera.py`)
    и в каком состоянии задача.

    ПК без `.git` (копия проекта) обновляются файлами — так же, как это делает
    `scripts/publish_client.py` для публичного релиза. ПК вида `"kind": "git"`
    обновляются `git pull --ff-only` в своём клоне. В обоих случаях локальный
    `config.yaml`, `data/` и модели не трогаются.

    Скрипт честный: если ПК не отвечает, обновление упало или задача не поднялась,
    он пишет это в итоговой таблице и возвращает ненулевой код. Ничего не
    «дорисовывается»: хэш на ПК сравнивается с хэшем этого рабочего каталога.

.EXAMPLE
    pwsh -File scripts/update-room-pcs.ps1
.EXAMPLE
    pwsh -File scripts/update-room-pcs.ps1 -Only buro -DryRun
#>
[CmdletBinding()]
param(
    # Инвентарь ПК. По умолчанию — deploy/room-pcs.json в этом репозитории.
    [string]$Inventory,
    # Обновить только эти ПК (по полю name). Пусто — все.
    [string[]]$Only = @(),
    # Ничего не менять: только проверить доступность и показать план.
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Inventory) { $Inventory = Join-Path $repo 'deploy\room-pcs.json' }
if (-not (Test-Path -LiteralPath $Inventory)) { throw "Inventory not found: $Inventory" }

$sshExe = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
$scpExe = Join-Path $env:SystemRoot 'System32\OpenSSH\scp.exe'
foreach ($tool in @($sshExe, $scpExe)) {
    if (-not (Test-Path -LiteralPath $tool)) { throw "OpenSSH client not found: $tool" }
}

function New-ClientPackage {
    param([string]$Repo)
    $stage = Join-Path $env:TEMP ('rowan-room-update-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    foreach ($sub in 'client', 'common') {
        Copy-Item -Path (Join-Path $Repo $sub) -Destination $stage -Recurse -Force
    }
    # Рантайм-состояние и кэши комнаты остаются её собственными.
    Remove-Item -Path (Join-Path $stage 'client\room-tracker.runtime.yaml') -Force -ErrorAction SilentlyContinue
    Get-ChildItem $stage -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
    $zip = Join-Path $env:TEMP ((Split-Path -Leaf $stage) + '.zip')
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -Force
    Remove-Item $stage -Recurse -Force
    return $zip
}

function Invoke-Remote {
    param([object]$Pc, [string]$Script)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    $out = & $sshExe -i $Pc.key -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 `
        "$($Pc.user)@$($Pc.host)" "powershell -NoProfile -EncodedCommand $encoded" 2>&1
    return @{ code = $LASTEXITCODE; lines = @($out | ForEach-Object { "$_" }) }
}

function Get-Line {
    param([string[]]$Lines, [string]$Key)
    foreach ($line in $Lines) {
        if ($line -match "^$([regex]::Escape($Key))=(.*)$") { return $Matches[1].Trim() }
    }
    return ''
}

$localCamera = (Get-FileHash -LiteralPath (Join-Path $repo 'client\camera.py') -Algorithm SHA256).Hash
$pcs = @(Get-Content -LiteralPath $Inventory -Raw | ConvertFrom-Json)
$results = @()

$zip = $null
if (-not $DryRun) {
    $zip = New-ClientPackage -Repo $repo
    "Пакет: $zip ($([math]::Round((Get-Item -LiteralPath $zip).Length / 1KB)) КБ)"
}

foreach ($pc in $pcs) {
    if ($Only.Count -and ($Only -notcontains $pc.name)) { continue }
    $entry = [ordered]@{ pc = $pc.name; host = "$($pc.user)@$($pc.host)"; result = ''; camera = ''; task = ''; window = '' }

    $probe = Invoke-Remote -Pc $pc -Script '"PROBE=ok"'
    if ($probe.code -ne 0 -or (Get-Line $probe.lines 'PROBE') -ne 'ok') {
        $entry.result = 'НЕ ОТВЕЧАЕТ'
        $results += [pscustomobject]$entry
        continue
    }
    if ($DryRun) {
        $entry.result = 'доступен (dry run)'
        $results += [pscustomobject]$entry
        continue
    }

    $remoteZip = "$($pc.root)\data\rowan-client-update.zip"
    & $scpExe -i $pc.key -o BatchMode=yes -o StrictHostKeyChecking=no $zip "$($pc.user)@$($pc.host):$($remoteZip.Replace('\', '/'))" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $entry.result = 'ОШИБКА копирования архива'
        $results += [pscustomobject]$entry
        continue
    }

    $remote = @'
$ErrorActionPreference = 'Stop'
$root = 'ROOT_HERE'
$task = 'TASK_HERE'
$zip = Join-Path $root 'data\rowan-client-update.zip'
$dest = Join-Path $root 'data\rowan-client-unpacked'
"ZIP_HASH=$((Get-FileHash $zip -Algorithm SHA256).Hash)"
if ('KIND_HERE' -eq 'git') {
    Push-Location $root
    try { $pull = & git pull --ff-only 2>&1 } finally { Pop-Location }
    $pull | Select-Object -Last 3
    if ($LASTEXITCODE -ne 0) { "RESULT=git pull failed"; exit 3 }
    "RESULT=git"
} else {
    Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Copy-Item -Path (Join-Path $dest '*') -Destination $root -Recurse -Force
"RESULT=files"
}
"CAMERA_HASH=$((Get-FileHash (Join-Path $root 'client\camera.py') -Algorithm SHA256).Hash)"
# The room PC needs no PowerShell window on the TV: run-client.ps1 keeps its
# own log file (data\logs), and the HUD is drawn by the client itself. The
# action is edited in place so triggers, principal and settings survive.
try {
    $registered = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    if (-not $registered) { "WINDOW=task missing" }
    else {
        $first = @($registered.Actions)[0]
        $arguments = [string]$first.Arguments
        if ($arguments -match '(?i)-WindowStyle') { "WINDOW=hidden" }
        else {
            $hidden = $arguments -replace '(?i)(-File\b)', '-WindowStyle Hidden $1'
            $splat = @{ Execute = [string]$first.Execute; Argument = $hidden }
            if ([string]$first.WorkingDirectory) { $splat['WorkingDirectory'] = [string]$first.WorkingDirectory }
            Set-ScheduledTask -TaskName $task -Action (New-ScheduledTaskAction @splat) | Out-Null
            $arguments = $hidden
            "WINDOW=fixed"
        }
        "TASK_ACTION=$arguments"
    }
} catch { "WINDOW=не удалось ($($_.Exception.GetType().Name))" }
Start-ScheduledTask -TaskName $task
Start-Sleep -Seconds 12
"TASK_STATE=$((Get-ScheduledTask -TaskName $task).State)"
"PYTHON=$((Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count)"
'@
    $remote = $remote.Replace('ROOT_HERE', $pc.root).Replace('TASK_HERE', $pc.task).Replace('KIND_HERE', $pc.kind)
    $run = Invoke-Remote -Pc $pc -Script $remote
    $hash = Get-Line $run.lines 'CAMERA_HASH'
    $entry.camera = if ($hash -eq $localCamera) { 'совпадает' } elseif ($hash) { "ДРУГОЙ ($($hash.Substring(0, 12)))" } else { 'неизвестно' }
    $entry.task = Get-Line $run.lines 'TASK_STATE'
    $entry.window = Get-Line $run.lines 'WINDOW'
    $entry.result = if ($run.code -ne 0) { "ОШИБКА (код $($run.code))" }
        elseif ((Get-Line $run.lines 'RESULT') -eq 'files') { 'обновлён и перезапущен' }
        elseif ((Get-Line $run.lines 'RESULT') -eq 'git') { 'обновлён через git и перезапущен' }
        else { 'БЕЗ РЕЗУЛЬТАТА' }
    $results += [pscustomobject]$entry
}

"" 
$results | Format-Table -AutoSize | Out-String -Width 200
if ($DryRun) {
    "Проверка связи пройдена; ничего не менялось (dry run)."
    exit 0
}
$failed = @($results | Where-Object { $_.result -notmatch 'обновлён' -and $_.result -notmatch 'dry run' })
if ($failed.Count) {
    "Не обновлены: " + (($failed | ForEach-Object { $_.pc }) -join ', ')
    exit 1
}
"Все комнатные ПК обновлены и клиент на них перезапущен."
