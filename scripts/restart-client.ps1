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
# Only the room client, never every python of this PC: the hub runs from the
# same folder and in the same interpreter, so a blanket "kill python" here
# would take the brain down with the client (2026-09-22). The client is the
# process whose command line names client.main.
Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*client.main*' -or $_.CommandLine -like '*client\main.py*' } |
    ForEach-Object {
        Write-Host ("  killing client pid {0}" -f $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 2

Write-Host "== (re)registering the interactive task =="
# -WindowStyle Hidden: run-client.ps1 keeps its own log file, and the owner
# does not want a PowerShell window parked on the room TV (the HUD is a
# separate window the client draws itself).
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ("-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"{0}\scripts\run-client.ps1`"" -f $Root) `
    -WorkingDirectory $Root
$principal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\Anton" -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "== starting it on the desktop =="
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 6
Get-Process python -ErrorAction SilentlyContinue |
    Select-Object Id, SessionId, StartTime | Format-Table -AutoSize
