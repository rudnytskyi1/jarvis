@echo off
rem Jarvis: room client (PC at the TV, RTX 3060). Double-click and it listens for "rowan".
chcp 65001 >nul
cd /d "%~dp0"

if not exist config.yaml (
    copy config.example.yaml config.yaml >nul
    echo [i] config.yaml created from config.example.yaml - fill in your devices (LED strip, SwitchBot)
)

powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-client.ps1
pause
