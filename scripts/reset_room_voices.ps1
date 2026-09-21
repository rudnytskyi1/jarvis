$ErrorActionPreference = 'Stop'
$rowanRoot = Split-Path -Parent $PSScriptRoot
$rowanPython = 'C:\Users\Anton\anaconda3\envs\jarvis\python.exe'
$rowanKeyPath = Join-Path $env:LOCALAPPDATA 'Jarvis\openai-api-key.dpapi'
if (-not (Test-Path -LiteralPath $rowanKeyPath)) { throw 'Saved server key is unavailable; server unchanged' }
$rowanProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match '-m hub.main' -and $_.CommandLine.Contains($rowanRoot)
})
if ($rowanProcesses.Count -ne 1) { throw 'Expected exactly one Rowan brain server; nothing changed' }
$rowanStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$rowanLog = Join-Path $rowanRoot "data\brain-voice-reset-$rowanStamp"
Stop-Process -Id $rowanProcesses[0].ProcessId -Force
try {
    & $rowanPython (Join-Path $PSScriptRoot 'reset_voice_profiles.py')
    if ($LASTEXITCODE -ne 0) { throw 'Voice reset failed' }
} finally {
    $rowanLauncher = Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'run-openai-server.ps1')) -WorkingDirectory $rowanRoot -WindowStyle Hidden -RedirectStandardOutput "$rowanLog.out.log" -RedirectStandardError "$rowanLog.err.log" -PassThru
    Write-Output "Rowan launcher PID: $($rowanLauncher.Id). Log: $rowanLog.err.log"
}
