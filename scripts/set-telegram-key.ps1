<# Run on the brain PC; the room client never needs this token. #>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'openai-key-store.ps1')
$rowanTelegramKey = $null
try {
    $rowanTelegramKey = Read-Host 'Telegram bot token (hidden)' -AsSecureString
    if ($rowanTelegramKey.Length -eq 0) { throw 'The token is empty; no changes were made.' }
    $rowanTelegramPath = Join-Path $env:LOCALAPPDATA 'Jarvis\telegram-bot-token.dpapi'
    Save-JarvisApiKey -Path $rowanTelegramPath -Key $rowanTelegramKey
    Write-Host 'Telegram token saved encrypted. Restart start-jarvis-openai.bat to load it.' -ForegroundColor Green
} finally {
    if ($null -ne $rowanTelegramKey) { $rowanTelegramKey.Dispose() }
}
