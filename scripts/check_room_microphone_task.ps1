param([switch]$Configure, [switch]$Capture)
$ErrorActionPreference = 'Stop'
$rowanRoot = 'C:\Users\Anton\Desktop\jarvis'
$rowanName = 'JarvisMicrophoneDiagnostics'
$rowanClientTask = Get-ScheduledTask -TaskName 'JarvisRoomClient'
$rowanArguments = 'C:\Users\Anton\Desktop\jarvis\scripts\inspect_room_audio.py --output'
if ($Capture) { $rowanArguments += ' --capture' }
if ($Configure) {
    $rowanArguments = 'C:\Users\Anton\Desktop\jarvis\scripts\configure_room_audio.py --input-device "onn Gaming USB MME" --output-device "Roku TV MME"'
}
$rowanAction = New-ScheduledTaskAction -Execute 'C:\Users\Anton\miniconda3\envs\jarvis\pythonw.exe' -Argument $rowanArguments -WorkingDirectory $rowanRoot
$rowanSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $rowanName -Action $rowanAction -Principal $rowanClientTask.Principal -Settings $rowanSettings -Force | Out-Null
Start-ScheduledTask -TaskName $rowanName
Write-Output 'Started microphone check in the Rowan desktop session.'
