<# Run on the brain PC. Uses Windows DPAPI, never YAML, shell arguments or logs. #>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'openai-key-store.ps1')
$rowanGeminiKey = $null
try {
    Write-Host 'Create a Gemini API key at https://aistudio.google.com/apikey (billing required for image generation).'
    Write-Host 'The key stays encrypted under your Windows account on this PC.'
    $rowanGeminiKey = Read-Host 'Gemini API key (hidden)' -AsSecureString
    if ($rowanGeminiKey.Length -eq 0) { throw 'The key is empty; no changes were made.' }
    $rowanGeminiPath = Join-Path $env:LOCALAPPDATA 'Jarvis\gemini-api-key.dpapi'
    Save-JarvisApiKey -Path $rowanGeminiPath -Key $rowanGeminiKey
    Write-Host 'Gemini key saved. Restart the brain server using start-jarvis-openai.bat to load it.' -ForegroundColor Green
    Write-Host 'Normal server starts will reuse this key without asking again.'
} finally {
    if ($null -ne $rowanGeminiKey) { $rowanGeminiKey.Dispose() }
}
