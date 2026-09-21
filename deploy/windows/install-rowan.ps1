# Автозапуск Rowan на Windows (ТЗ 4.9): планировщик задач, при наличии — NSSM.
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install-rowan.ps1 -Hub
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install-rowan.ps1 -Client
#
# Хаб: скрытая задача, перезапуск при падении. Клиент: обычное окно (там HUD),
# запуск при входе пользователя в систему.
param(
    [switch]$Hub,
    [switch]$Client,
    [string]$Python = 'C:\Users\Anton\anaconda3\envs\jarvis\python.exe',
    [string]$Repo = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$Config = 'config.yaml'
)

$ErrorActionPreference = 'Stop'
if (-not ($Hub -or $Client)) { throw 'Choose -Hub, -Client, or both.' }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $Repo 'config.yaml'))) { throw "Not a Rowan checkout: $Repo" }

$logs = Join-Path $Repo 'data\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Install-RowanTask {
    param([string]$Name, [string]$Module, [switch]$Interactive)

    $arguments = "-m $Module --config `"$Config`""
    $action = New-ScheduledTaskAction -Execute $Python -Argument $arguments -WorkingDirectory $Repo
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 0)
    if ($Interactive) {
        $trigger = New-ScheduledTaskTrigger -AtLogOn
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    } else {
        $trigger = New-ScheduledTaskTrigger -AtStartup
        $trigger.Delay = 'PT30S'
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
    }
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Force | Out-Null
    Write-Host "Installed scheduled task $Name -> $Python $arguments (working dir $Repo)"
}

function Install-WithNssm {
    param([string]$Name, [string]$Module, [string]$LogName)
    $nssm = Get-Command nssm.exe -ErrorAction SilentlyContinue
    if (-not $nssm) { return $false }
    & $nssm.Source install $Name $Python "-m $Module --config `"$Config`"" | Out-Null
    & $nssm.Source set $Name AppDirectory $Repo | Out-Null
    & $nssm.Source set $Name AppStdout (Join-Path $logs "$LogName.log") | Out-Null
    & $nssm.Source set $Name AppStderr (Join-Path $logs "$LogName.log") | Out-Null
    & $nssm.Source set $Name AppExit Default Restart | Out-Null
    & $nssm.Source start $Name | Out-Null
    Write-Host "Installed NSSM service $Name -> $Python -m $Module"
    return $true
}

if ($Hub) {
    if (-not (Install-WithNssm -Name 'RowanHub' -Module 'hub.main' -LogName 'hub')) {
        Install-RowanTask -Name 'RowanHub' -Module 'hub.main'
    }
}
if ($Client) {
    if (-not (Install-WithNssm -Name 'RowanClient' -Module 'client.main' -LogName 'client')) {
        Install-RowanTask -Name 'RowanClient' -Module 'client.main' -Interactive
    }
}
Write-Host 'Done. Start now with: schtasks /Run /TN RowanHub'
