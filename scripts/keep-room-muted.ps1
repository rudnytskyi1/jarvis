# Keep the speakers of a room PC at zero - always, not once.
#
# Owner's rule (2026-09-24): "только громкость на нуле всегда должна быть в
# комнатах". Muting once is not enough: Windows restores the level after a device
# change, and anything on the PC (the room client's own "громче", a game, a
# browser tab) can raise it again. This script is the answer to "always": a small
# loop that reads the default speakers every few seconds and puts mute + 0.0 back
# the moment they move, plus a scheduled task so the loop comes back after every
# logon.
#
# Run it ON the room PC:
#     pwsh -File scripts\keep-room-muted.ps1 -Action status   # what is it now
#     pwsh -File scripts\keep-room-muted.ps1 -Action once     # silence it now
#     pwsh -File scripts\keep-room-muted.ps1 -Action watch    # keep it silent
#     pwsh -File scripts\keep-room-muted.ps1 -Action autostart  # survive logons
#     pwsh -File scripts\keep-room-muted.ps1 -Action install    # task, needs admin
#
# `autostart` is the mode that works without an administrator: it drops a one-line
# launcher into the user's Startup folder, so the loop comes back at every logon.
# `install` registers a logon task instead (nicer, but Windows asks for elevation
# on these PCs, so it falls back to `autostart` when the task cannot be created).
# The loop writes what it did into `room-mute.log` beside this file.
[CmdletBinding()]
param(
    [ValidateSet('status', 'once', 'watch', 'autostart', 'install', 'uninstall')][string]$Action = 'status',
    [int]$IntervalSeconds = 5
)
$ErrorActionPreference = 'Stop'
$TaskName = 'RowanKeepRoomsSilent'
$LogPath = Join-Path $PSScriptRoot 'room-mute.log'

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

# The whole job is one Python process: pycaw is the same library the room client
# uses for its own mute command, and keeping the loop inside it means one COM
# connection instead of a new interpreter every few seconds.
$pythonCode = @'
import os
import sys
import time
from datetime import datetime

from pycaw.pycaw import AudioUtilities

ACTION = "__ACTION__"
INTERVAL = max(1, __INTERVAL__)
LOG = r"__LOG__"


def say(line):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = "%s %s %s" % (stamp, os.environ.get("COMPUTERNAME", "this PC"), line)
    print(text, flush=True)
    if LOG:
        try:
            with open(LOG, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass


def endpoint():
    device = AudioUtilities.GetSpeakers()
    return getattr(device, "EndpointVolume", device)


def state(volume):
    return bool(volume.GetMute()), float(volume.GetMasterVolumeLevelScalar())


def silence(volume):
    volume.SetMute(1, None)
    volume.SetMasterVolumeLevelScalar(0.0, None)


try:
    speakers = endpoint()
    muted, level = state(speakers)
except Exception as error:
    say("could not reach the speakers (%s: %s)" % (type(error).__name__, error))
    raise SystemExit(2)

if ACTION == "status":
    say("muted %d volume %.3f" % (int(muted), level))
    raise SystemExit(0)

if ACTION == "once":
    if not muted or level > 0.0:
        silence(speakers)
    muted, level = state(speakers)
    say("silent: muted %d volume %.3f" % (int(muted), level))
    raise SystemExit(0)

if not muted or level > 0.0:
    silence(speakers)
    say("silenced at startup (was muted=%d volume=%.3f)" % (int(muted), level))
muted, level = state(speakers)
say("watching every %d s (muted %d volume %.3f)" % (INTERVAL, int(muted), level))

restored = 0
refreshed = time.time()
while True:
    try:
        if time.time() - refreshed > 120:
            # A different output device (headphones, a TV over HDMI) needs its own
            # endpoint object, otherwise the loop would police a speaker nobody
            # hears any more.
            speakers = endpoint()
            refreshed = time.time()
        muted, level = state(speakers)
        if not muted or level > 0.0:
            silence(speakers)
            restored += 1
            say("restored silence (was muted=%d volume=%.3f), %d time(s)"
                % (int(muted), level, restored))
    except Exception as error:
        say("lost the speakers (%s: %s); reconnecting" % (type(error).__name__, error))
        try:
            speakers = endpoint()
            refreshed = time.time()
        except Exception:
            pass
    time.sleep(INTERVAL)
'@

$pythonCode = $pythonCode.Replace('__ACTION__', $Action)
$pythonCode = $pythonCode.Replace('__INTERVAL__', "$IntervalSeconds")
$pythonCode = $pythonCode.Replace('__LOG__', $LogPath)

if ($Action -in @('status', 'once', 'watch')) {
    $python = Resolve-RoomPython
    $pythonCode | & $python -
    exit $LASTEXITCODE
}

if ($Action -eq 'uninstall') {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    $launcher = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup\RowanKeepRoomsSilent.cmd'
    Remove-Item -LiteralPath $launcher -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*keep-room-muted.ps1*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "Removed the '$TaskName' task and the Startup launcher."
    exit 0
}

function Start-RoomMuteWatcher {
    Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
        '-Action', 'watch', '-IntervalSeconds', "$IntervalSeconds"
    )
}

# Silence it now, then make sure the loop comes back after every logon.
& $PSCommandPath -Action once -IntervalSeconds $IntervalSeconds

if ($Action -eq 'install') {
    try {
        $arguments = '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -Action watch -IntervalSeconds {1}' -f `
            $PSCommandPath, $IntervalSeconds
        $taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
        $trigger = New-ScheduledTaskTrigger -AtLogOn
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
        Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
            -Settings $settings -Force `
            -Description 'Keeps the room speakers muted at zero (owner rule, 2026-09-24)' | Out-Null
        Start-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 2
        $installed = Get-ScheduledTask -TaskName $TaskName
        Write-Host ("{0} is {1}; the room stays at zero." -f $TaskName, $installed.State)
        exit 0
    } catch {
        Write-Warning ("The scheduled task needs administrator rights ({0}); using the Startup folder instead." -f $_.Exception.Message.Trim())
    }
}

$startup = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup'
$launcher = Join-Path $startup 'RowanKeepRoomsSilent.cmd'
New-Item -ItemType Directory -Force -Path $startup | Out-Null
Set-Content -LiteralPath $launcher -Encoding ASCII -Value @(
    '@echo off',
    'rem Rowan: the room speakers stay at zero (owner rule, 2026-09-24).',
    ('start "" powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -Action watch -IntervalSeconds {1}' -f `
        $PSCommandPath, $IntervalSeconds)
)
Start-RoomMuteWatcher
Write-Host ("autostart launcher: {0}" -f $launcher)
