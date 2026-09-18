# Restart the Jarvis room client INSIDE the interactive desktop session.
#
# A process started over SSH lands in its own Windows session and cannot draw
# on the console session's desktop, so the HUD overlay would never appear on
# the TV. A scheduled task registered with -LogonType Interactive runs as the
# logged-on user, on their desktop, which is exactly what the overlay needs.
$ErrorActionPreference = "Stop"
$Root = "C:\Users\Anton\Desktop\jarvis"
$TaskName = "JarvisRoomClient"

Write-Host "== stopping the running client =="
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("  killing pid {0}" -f $_.Id)
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

Write-Host "== (re)registering the interactive task =="
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ("-NoProfile -ExecutionPolicy Bypass -File `"{0}\scripts\run-client.ps1`"" -f $Root) `
    -WorkingDirectory $Root
$principal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\Anton" -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "== starting it on the desktop =="
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 6
Get-Process python -ErrorAction SilentlyContinue |
    Select-Object Id, SessionId, StartTime | Format-Table -AutoSize
