# Mute (or unmute) the loudspeakers of a room PC. Run it ON the room PC:
#     pwsh -File scripts\mute-room-sound.ps1 -Action mute
# The owner asked for silence on every PC Rowan is connected to; this is the
# same pycaw call the room client itself uses for its "mute" command, so the
# room keeps working while its speakers stay off.
[CmdletBinding()]
param([ValidateSet('mute', 'unmute')][string]$Action = 'mute')
$ErrorActionPreference = 'Stop'

function Resolve-RoomPython {
    $candidates = @(
        'C:\Users\Anton\anaconda3\envs\jarvis\python.exe',
        (Join-Path $env:USERPROFILE 'anaconda3\envs\jarvis\python.exe'),
        (Join-Path $env:USERPROFILE 'anaconda3\envs\rowanai\python.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\envs\jarvis\python.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\envs\rowanai\python.exe'),
        (Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    $conda = @(
        (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($conda) {
        foreach ($name in @('jarvis', 'rowanai', 'rowan')) {
            $match = (& $conda env list 2>$null) |
                Where-Object { $_ -match "^\s*$name\s+(\S+)" } | Select-Object -First 1
            if ($match -and $matches[1]) {
                $python = Join-Path $matches[1] 'python.exe'
                if (Test-Path -LiteralPath $python) { return $python }
            }
        }
    }
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    throw 'No Python with pycaw found on this PC.'
}

$python = Resolve-RoomPython
$muted = if ($Action -eq 'mute') { 1 } else { 0 }
$scalar = if ($Action -eq 'mute') { '0.0' } else { '1.0' }
$code = @'
import os
from pycaw.pycaw import AudioUtilities

device = AudioUtilities.GetSpeakers()
volume = getattr(device, "EndpointVolume", device)
volume.SetMute(MUTED, None)
volume.SetMasterVolumeLevelScalar(SCALAR, None)
print(os.environ.get("COMPUTERNAME", "this PC"), "muted", volume.GetMute(),
      "volume", round(float(volume.GetMasterVolumeLevelScalar()), 3))
'@
$code = $code.Replace('MUTED', $muted).Replace('SCALAR', $scalar)
$code | & $python -
