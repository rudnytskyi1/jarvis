@echo off
rem Jarvis: room client (PC at the TV, RTX 3060). Double-click and it listens for "rowan".
chcp 65001 >nul
cd /d "%~dp0"

if not exist config.yaml (
    copy config.example.yaml config.yaml >nul
    echo [i] config.yaml created from config.example.yaml - fill in your devices (LED strip, SwitchBot)
)

echo [i] Starting the Jarvis client in its own window so you can watch the log.
echo     Logs also go to: data\client.log      Stop it by closing that window.
start "Jarvis client" powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-client.ps1"
timeout /t 3 >nul
