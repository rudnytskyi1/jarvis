<#
.SYNOPSIS
    Поднять (или остановить) локальный мониторинг Rowan: Prometheus + Grafana.

.DESCRIPTION
    Оба сервиса живут ВНЕ репозитория (по умолчанию
    ``C:\Users\Anton\Desktop\monitoring``), а конфигурация — в репозитории
    (``deploy\monitoring``), чтобы её правки попадали в git.

    Prometheus собирает ``/metrics`` хаба (ТЗ F-707), Grafana показывает
    дашборд ``deploy\grafana\rowan-dashboard.json``. Порт Grafana — 3001: на
    3000 у владельца занято другим сервисом. Дашборд и источник данных
    прописаны заранее, поэтому открывать и импортировать вручную ничего не
    нужно — достаточно открыть http://localhost:3001.

    Пароль админа Grafana берётся из переменной ``GRAFANA_ADMIN_PASSWORD``
    (её читает ``.env``; если её нет — ``rowan``). Логин: ``admin``.

.EXAMPLE
    pwsh -File scripts\start-monitoring.ps1
.EXAMPLE
    pwsh -File scripts\start-monitoring.ps1 -Stop
#>
[CmdletBinding()]
param(
    [string]$Root = 'C:\Users\Anton\Desktop\monitoring',
    [string]$PrometheusPort = '9090',
    [string]$GrafanaPort = '3001',
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$prometheusExe = Join-Path $Root 'prometheus\prometheus-3.14.0.windows-amd64\prometheus.exe'
$grafanaExe = Join-Path $Root 'grafana\grafana-13.2.2\bin\grafana.exe'
$grafanaHome = Split-Path -Parent (Split-Path -Parent $grafanaExe)
$prometheusConfig = Join-Path $repo 'deploy\monitoring\prometheus.yml'
$grafanaProvisioning = Join-Path $repo 'deploy\monitoring\grafana\provisioning'
$prometheusData = Join-Path $Root 'data\prometheus'
$logDir = Join-Path $Root 'logs'

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*#') { continue }
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $name, $value = $matches[1], $matches[2].Trim()
            if (-not [Environment]::GetEnvironmentVariable($name)) {
                [Environment]::SetEnvironmentVariable($name, $value)
            }
        }
    }
}

function Stop-RowanMonitoring {
    foreach ($name in 'prometheus', 'grafana') {
        Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host ("  stopping {0} (pid {1})" -f $name, $_.Id)
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

if ($Stop) {
    Stop-RowanMonitoring
    Write-Host 'Мониторинг остановлен.'
    exit 0
}

Import-DotEnv -Path (Join-Path $repo '.env')
$password = $env:GRAFANA_ADMIN_PASSWORD
if (-not $password) { $password = 'rowan' }

if (-not (Test-Path -LiteralPath $prometheusExe)) { throw "Prometheus not found: $prometheusExe" }
if (-not (Test-Path -LiteralPath $grafanaExe)) { throw "Grafana not found: $grafanaExe" }
if (-not (Test-Path -LiteralPath $prometheusConfig)) { throw "Prometheus config not found: $prometheusConfig" }
New-Item -ItemType Directory -Force -Path $prometheusData, $logDir | Out-Null

# Уже запущено — не поднимаем второй экземпляр (иначе занятый порт и мусор).
foreach ($name in 'prometheus', 'grafana') {
    if (Get-Process -Name $name -ErrorAction SilentlyContinue) {
        Write-Host "  $name уже запущен"
    }
}

if (-not (Get-Process -Name prometheus -ErrorAction SilentlyContinue)) {
    $args = @(
        "--config.file=$prometheusConfig",
        "--storage.tsdb.path=$prometheusData",
        "--web.listen-address=127.0.0.1:$PrometheusPort",
        "--storage.tsdb.retention.time=30d"
    )
    Start-Process -FilePath $prometheusExe -ArgumentList $args -WorkingDirectory (Split-Path -Parent $prometheusExe) `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'prometheus.out.log') `
        -RedirectStandardError (Join-Path $logDir 'prometheus.err.log')
    Write-Host "  prometheus поднят на 127.0.0.1:$PrometheusPort"
}

if (-not (Get-Process -Name grafana -ErrorAction SilentlyContinue)) {
    $env:GF_SERVER_HTTP_PORT = $GrafanaPort
    $env:GF_SERVER_HTTP_ADDR = '127.0.0.1'
    $env:GF_SECURITY_ADMIN_USER = 'admin'
    $env:GF_SECURITY_ADMIN_PASSWORD = $password
    $env:GF_USERS_ALLOW_SIGN_UP = 'false'
    $env:GF_ANALYTICS_REPORTING_ENABLED = 'false'
    $env:GF_ANALYTICS_CHECK_FOR_UPDATES = 'false'
    $env:GF_PATHS_PROVISIONING = $grafanaProvisioning
    # Grafana 13 — это CLI из подкоманд: сервер поднимает `grafana server`,
    # а домашний каталог задаётся флагом --homepath (раньше он был -homepath).
    Start-Process -FilePath $grafanaExe -ArgumentList 'server', "--homepath=$grafanaHome" `
        -WorkingDirectory $grafanaHome -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir 'grafana.out.log') `
        -RedirectStandardError (Join-Path $logDir 'grafana.err.log')
    Write-Host "  grafana поднята на http://127.0.0.1:$GrafanaPort (admin / $password)"
}

Start-Sleep -Seconds 20
try {
    $targets = Invoke-RestMethod -Uri "http://127.0.0.1:$PrometheusPort/api/v1/targets" -TimeoutSec 10
    $rowan = $targets.data.activeTargets | Where-Object { $_.labels.job -eq 'rowan' }
    "prometheus: цель rowan -> $($rowan.health)"
} catch { "prometheus не ответил: $($_.Exception.Message)" }

try {
    $auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:$password"))
    $search = Invoke-RestMethod -Uri "http://127.0.0.1:$GrafanaPort/api/search?type=dash-db" `
        -Headers @{ Authorization = "Basic $auth" } -TimeoutSec 10
    "grafana: дашбордов найдено $($search.Count) -> " + (($search | ForEach-Object { $_.title }) -join ', ')
} catch {
    "grafana не ответила: $($_.Exception.Message)"
    "смотрите лог: " + (Join-Path $logDir 'grafana.err.log')
}
