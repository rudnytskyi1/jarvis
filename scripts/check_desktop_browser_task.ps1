param([switch]$Verify, [switch]$YouTube)
$ErrorActionPreference = 'Stop'
$rowanRoot = 'C:\Users\Anton\Desktop\jarvis'
$rowanClientTask = Get-ScheduledTask -TaskName 'JarvisRoomClient'
$rowanScript = if ($YouTube) { 'verify_youtube_browser.py' } elseif ($Verify) { 'verify_desktop_browser_runtime.py' } else { 'inspect_desktop_browser.py' }
$rowanAction = New-ScheduledTaskAction -Execute 'C:\Users\Anton\miniconda3\envs\jarvis\pythonw.exe' -Argument (Join-Path $rowanRoot ('scripts\' + $rowanScript)) -WorkingDirectory $rowanRoot
$rowanSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName 'JarvisDesktopBrowserDiagnostics' -Action $rowanAction -Principal $rowanClientTask.Principal -Settings $rowanSettings -Force | Out-Null
Start-ScheduledTask -TaskName 'JarvisDesktopBrowserDiagnostics'
Write-Output 'Started ordinary-browser check in the interactive Rowan desktop.'
