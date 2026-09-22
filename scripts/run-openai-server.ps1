<# Start the budgeted profile; cache an interactively entered key using Windows DPAPI. #>
[CmdletBinding()]
param([string]$Config, [switch]$ReplaceKey)
$ErrorActionPreference = 'Stop'
# Resolve script-relative defaults after parameter binding (Windows PowerShell 5.1).
if (-not $Config) {
    $Config = Join-Path (Split-Path -Parent $PSScriptRoot) 'config.openai.yaml'
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw 'Create config.openai.yaml first: python scripts/configure_openai.py (inside the jarvis environment).'
}
$jarvisPreviousKey = $env:OPENAI_API_KEY
$jarvisPreviousGeminiKey = $env:GEMINI_API_KEY
$jarvisPreviousTelegramToken = $env:TELEGRAM_BOT_TOKEN
. (Join-Path $PSScriptRoot 'openai-key-store.ps1')
$jarvisRepoRoot = Split-Path -Parent $PSScriptRoot
# Keys may also live in a git-ignored ``.env`` next to the config: the file is
# read first, and anything already set in the environment keeps priority.
$jarvisFromDotEnv = Import-JarvisDotEnv -Path (Join-Path $jarvisRepoRoot '.env')
if ($jarvisFromDotEnv.Count -gt 0) {
    Write-Host ('Loaded from .env: ' + ($jarvisFromDotEnv -join ', ')) -ForegroundColor Green
}
$jarvisKeyPath = Get-JarvisKeyPath 'openai-api-key.dpapi'
$jarvisSecureKey = $null
$jarvisGeminiSecureKey = $null
$jarvisTelegramSecureKey = $null
try {
    if ($ReplaceKey -or -not $env:OPENAI_API_KEY) {
        if (-not $ReplaceKey) { $jarvisSecureKey = Read-JarvisApiKey -Path $jarvisKeyPath }
        if ($null -eq $jarvisSecureKey) {
            Write-Host "No saved OpenAI key at $jarvisKeyPath - entering one now." -ForegroundColor Yellow
            $jarvisSecureKey = Read-Host 'OpenAI API key (hidden; saved encrypted for future starts)' -AsSecureString
            if ($jarvisSecureKey.Length -eq 0) { throw 'API key is empty.' }
            try {
                Save-JarvisApiKey -Path $jarvisKeyPath -Key $jarvisSecureKey
                Write-Host 'Key saved encrypted for your Windows account. Future starts will use it automatically.' -ForegroundColor Green
            } catch {
                Write-Warning 'Could not save the encrypted key. Starting this session anyway; a future start may ask again.'
            }
        } else {
            Write-Host 'Using the saved OpenAI key.' -ForegroundColor Green
        }
        $jarvisCredential = [System.Management.Automation.PSCredential]::new('jarvis', $jarvisSecureKey)
        $env:OPENAI_API_KEY = $jarvisCredential.GetNetworkCredential().Password
        $jarvisCredential = $null
    }
    if (-not $env:OPENAI_API_KEY) { throw 'API key is empty.' }
    # Optional provider: never prompt or prevent ordinary chat if unconfigured.
    if (-not $env:GEMINI_API_KEY) {
        $jarvisGeminiSecureKey = Read-JarvisApiKey -Path (Get-JarvisKeyPath 'gemini-api-key.dpapi')
        if ($null -ne $jarvisGeminiSecureKey) {
            $jarvisGeminiCredential = [System.Management.Automation.PSCredential]::new('jarvis', $jarvisGeminiSecureKey)
            $env:GEMINI_API_KEY = $jarvisGeminiCredential.GetNetworkCredential().Password
            $jarvisGeminiCredential = $null
            Write-Host 'Using the saved Gemini key for Nano Banana.' -ForegroundColor Green
        }
    }
    if (-not $env:TELEGRAM_BOT_TOKEN) {
        $jarvisTelegramSecureKey = Read-JarvisApiKey -Path (Get-JarvisKeyPath 'telegram-bot-token.dpapi')
        if ($null -ne $jarvisTelegramSecureKey) {
            $jarvisTelegramCredential = [System.Management.Automation.PSCredential]::new('jarvis', $jarvisTelegramSecureKey)
            $env:TELEGRAM_BOT_TOKEN = $jarvisTelegramCredential.GetNetworkCredential().Password
            $jarvisTelegramCredential = $null
            Write-Host 'Using the saved Telegram bot token.' -ForegroundColor Green
        }
    }
    & (Join-Path $PSScriptRoot 'run-server.ps1') -Config $Config
}
finally {
    $env:OPENAI_API_KEY = $jarvisPreviousKey
    $env:GEMINI_API_KEY = $jarvisPreviousGeminiKey
    $env:TELEGRAM_BOT_TOKEN = $jarvisPreviousTelegramToken
    if ($null -ne $jarvisSecureKey) { $jarvisSecureKey.Dispose() }
    if ($null -ne $jarvisGeminiSecureKey) { $jarvisGeminiSecureKey.Dispose() }
    if ($null -ne $jarvisTelegramSecureKey) { $jarvisTelegramSecureKey.Dispose() }
}
