$ErrorActionPreference = 'Stop'
$rowanRoot = 'C:\Users\Anton\Desktop\jarvis'
$rowanName = 'JarvisRequestDiagnostics'
$rowanClientTask = Get-ScheduledTask -TaskName 'JarvisRoomClient'
$rowanAction = New-ScheduledTaskAction -Execute 'C:\Users\Anton\miniconda3\envs\jarvis\pythonw.exe' -Argument 'C:\Users\Anton\Desktop\jarvis\scripts\verify_app_requests.py --output' -WorkingDirectory $rowanRoot
$rowanSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $rowanName -Action $rowanAction -Principal $rowanClientTask.Principal -Settings $rowanSettings -Force | Out-Null
Start-ScheduledTask -TaskName $rowanName
Write-Output 'Started the private window and browser inventory check in the client desktop session.'
