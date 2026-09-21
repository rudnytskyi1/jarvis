$ErrorActionPreference = 'Stop'
$rowanRoot = 'C:\Users\Anton\Desktop\jarvis'
$rowanCheck = Get-Content -LiteralPath (Join-Path $rowanRoot 'data\microphone-setup.json') -Raw | ConvertFrom-Json
if (-not $rowanCheck.ok) { throw 'Audio configuration check failed; client unchanged' }
Stop-ScheduledTask -TaskName 'JarvisRoomClient'
try {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -match '-m client.main' -and $_.CommandLine.Contains($rowanRoot)
    } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
} finally {
    Start-ScheduledTask -TaskName 'JarvisRoomClient'
}
Write-Output 'Restarted only the Rowan room client with the verified audio configuration.'
