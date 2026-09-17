@echo off
rem Jarvis: brain server (RTX 5090 PC) + ngrok tunnel. Double-click and everything comes up.
chcp 65001 >nul
cd /d "%~dp0"

set "NGROK=C:\Users\Anton\Desktop\ngrok.exe"
if not exist "%NGROK%" set "NGROK=ngrok"

if not exist config.yaml (
    copy config.example.yaml config.yaml >nul
    echo [i] config.yaml created from config.example.yaml - check the settings
)

echo [i] Starting ngrok: https://dorm-smart-un-iversity-of-nebr-omaha.ngrok.app -^> localhost:8770
echo [i] If the ngrok window shows ERR_NGROK_334 "already online", the tunnel is already up
echo     (for example from your always-on agent via default.internal). That is NOT an error,
echo     just close that window - the server will work through the existing tunnel.
start "Jarvis ngrok" cmd /k "%NGROK%" http --url=dorm-smart-un-iversity-of-nebr-omaha.ngrok.app 8770

powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-server.ps1

echo.
echo [i] Server stopped. Close the ngrok window separately if it is still open.
pause
