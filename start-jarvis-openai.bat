@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting budgeted OpenAI server. Keep the existing ngrok tunnel running if used.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-openai-server.ps1"
pause
