param([string]$Package = '')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$rowanRoot = Split-Path -Parent $PSScriptRoot
$rowanPython = 'C:\Users\Anton\miniconda3\envs\jarvis\python.exe'
if (-not $Package) { $Package = Join-Path $rowanRoot 'data\speech-recording-update.zip' }
$rowanStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$rowanStage = Join-Path $rowanRoot "data\speech-recording-stage-$rowanStamp"
$rowanBackup = Join-Path $rowanRoot "data\speech-recording-backup-$rowanStamp"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$rowanZip = [IO.Compression.ZipFile]::OpenRead($Package)
try {
    foreach ($entry in $rowanZip.Entries) {
        $resolved = [IO.Path]::GetFullPath((Join-Path $rowanStage $entry.FullName))
        if (-not $resolved.StartsWith($rowanStage + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid package path' }
    }
} finally { $rowanZip.Dispose() }
Expand-Archive -LiteralPath $Package -DestinationPath $rowanStage
$manifest = Get-Content -LiteralPath (Join-Path $rowanStage 'manifest.json') -Raw | ConvertFrom-Json
foreach ($entry in $manifest.PSObject.Properties) {
    $destination = [IO.Path]::GetFullPath((Join-Path $rowanRoot $entry.Name))
    if (-not $destination.StartsWith($rowanRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid destination path' }
    if ((Get-FileHash -LiteralPath (Join-Path $rowanStage $entry.Name)).Hash -ne $entry.Value) { throw "Hash mismatch: $($entry.Name)" }
}
& $rowanPython -m compileall -q (Join-Path $rowanStage 'client') (Join-Path $rowanStage 'common') (Join-Path $rowanStage 'scripts')
if ($LASTEXITCODE -ne 0) { throw 'Syntax check failed; client unchanged' }
& $rowanPython (Join-Path $rowanStage 'scripts\configure_recording.py') --config (Join-Path $rowanRoot 'config.openai.yaml') --target room --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Configuration validation failed; client unchanged' }
$rowanFiles = @($manifest.PSObject.Properties.Name) + @('config.openai.yaml')
foreach ($name in $rowanFiles) {
    $source = Join-Path $rowanRoot $name
    if (Test-Path -LiteralPath $source) {
        $saved = Join-Path $rowanBackup $name
        New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $saved
    }
}
Stop-ScheduledTask -TaskName 'JarvisRoomClient'
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match '-m client.main' -and $_.CommandLine.Contains($rowanRoot)
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
try {
    foreach ($entry in $manifest.PSObject.Properties) {
        $destination = Join-Path $rowanRoot $entry.Name
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $rowanStage $entry.Name) -Destination $destination -Force
    }
    & $rowanPython (Join-Path $rowanRoot 'scripts\update_room_speech_config.py')
    if ($LASTEXITCODE -ne 0) { throw 'Speech settings update failed' }
    & $rowanPython (Join-Path $rowanRoot 'scripts\configure_recording.py') --target room
    if ($LASTEXITCODE -ne 0) { throw 'Recording settings update failed' }
    Start-ScheduledTask -TaskName 'JarvisRoomClient'
    Write-Output "Update installed; client restart requested. Backup: $rowanBackup"
} catch {
    foreach ($name in $rowanFiles) {
        $saved = Join-Path $rowanBackup $name
        if (Test-Path -LiteralPath $saved) { Copy-Item -LiteralPath $saved -Destination (Join-Path $rowanRoot $name) -Force }
    }
    Start-ScheduledTask -TaskName 'JarvisRoomClient'
    throw
}
