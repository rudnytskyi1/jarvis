# Windows DPAPI: encrypted for the current Windows user on this PC.
# Call .NET directly: Start-Process can inherit an incompatible PowerShell 7
# module path, breaking Windows PowerShell's Security module autoload.
#
# The type is loaded AND checked explicitly, because that failure is otherwise
# silent: a window whose parent exported a PowerShell 7 ``PSModulePath`` fails
# with "Unable to find type [Security.Cryptography.ProtectedData]", the read
# returns ``$null``, and the operator is asked for a key that is in fact already
# saved. Checking costs nothing and turns a confusing prompt into a real answer.
function Initialize-JarvisDpapi {
    if ('System.Security.Cryptography.ProtectedData' -as [type]) { return $true }
    [System.Reflection.Assembly]::LoadWithPartialName('System.Security') | Out-Null
    if ('System.Security.Cryptography.ProtectedData' -as [type]) { return $true }
    Add-Type -AssemblyName System.Security -ErrorAction SilentlyContinue
    return [bool]('System.Security.Cryptography.ProtectedData' -as [type])
}

# The folder that holds the encrypted keys. Launchers normally inherit a valid
# ``LOCALAPPDATA``, but a window opened by a tool that replaced or dropped it
# would look in the wrong place, find nothing, and ask for a key that IS saved
# (the read cannot tell "no key yet" from "wrong folder"). The user profile is
# therefore checked as well, and the caller can print the folder it settled on.
function Get-JarvisKeyDirectory {
    $candidates = @()
    if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA 'Jarvis') }
    if ($env:USERPROFILE) { $candidates += (Join-Path $env:USERPROFILE 'AppData\Local\Jarvis') }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    foreach ($candidate in $candidates) { return $candidate }
    return 'Jarvis'
}

function Get-JarvisKeyPath {
    param([Parameter(Mandatory=$true)][string]$Name)
    return (Join-Path (Get-JarvisKeyDirectory) $Name)
}

# Optional plain-text fallback: ``.env`` in the repository root (git-ignored).
# Windows DPAPI stays the preferred store, but an operator who would rather keep
# the keys in a file can put ``OPENAI_API_KEY=...`` there and the launcher uses
# it as-is. A value that is already in the environment wins, so an explicit
# ``set OPENAI_API_KEY=...`` in the calling window still overrides the file.
function Import-JarvisDotEnv {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not [IO.File]::Exists($Path)) { return @() }
    $imported = @()
    foreach ($line in [IO.File]::ReadAllLines($Path)) {
        $text = $line.Trim()
        if (-not $text -or $text.StartsWith('#')) { continue }
        $split = $text.IndexOf('=')
        if ($split -lt 1) { continue }
        $name = $text.Substring(0, $split).Trim()
        if ($name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { continue }
        $value = $text.Substring($split + 1).Trim().Trim('"').Trim("'")
        if (-not $value) { continue }
        if (-not [Environment]::GetEnvironmentVariable($name)) {
            Set-Item -Path "env:$name" -Value $value
            $imported += $name
        }
    }
    return $imported
}

function Read-JarvisApiKey {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not [IO.File]::Exists($Path)) { return $null }
    $plainBytes = $null
    try {
        if (-not (Initialize-JarvisDpapi)) {
            throw 'DPAPI is unavailable in this PowerShell session'
        }
        # Same DPAPI hex format as ConvertFrom-SecureString, without importing it.
        $hex = [IO.File]::ReadAllText($Path).Trim()
        if ($hex.Length -eq 0 -or $hex.Length % 2 -ne 0 -or $hex -match '[^0-9a-fA-F]') {
            throw 'Invalid encrypted key format'
        }
        $encrypted = [byte[]]::new($hex.Length / 2)
        for ($i = 0; $i -lt $encrypted.Length; $i++) {
            $encrypted[$i] = [Convert]::ToByte($hex.Substring($i * 2, 2), 16)
        }
        $plainBytes = [Security.Cryptography.ProtectedData]::Unprotect(
            $encrypted, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        if ($plainBytes.Length -eq 0 -or $plainBytes.Length % 2 -ne 0) { throw 'Empty or invalid key' }
        $secure = [Security.SecureString]::new()
        for ($i = 0; $i -lt $plainBytes.Length; $i += 2) {
            $secure.AppendChar([char][BitConverter]::ToUInt16($plainBytes, $i))
        }
        $secure.MakeReadOnly()
        return $secure
    } catch {
        Write-Warning "The saved Jarvis key could not be unlocked: $($_.Exception.Message). Enter it again to replace the saved copy."
        return $null
    } finally {
        if ($null -ne $plainBytes) { [Array]::Clear($plainBytes, 0, $plainBytes.Length) }
    }
}

function Save-JarvisApiKey {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][Security.SecureString]$Key
    )
    if ($Key.Length -eq 0) { throw 'API key is empty.' }
    if (-not (Initialize-JarvisDpapi)) {
        throw 'DPAPI is unavailable in this PowerShell session'
    }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path)) | Out-Null
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($Key)
    $plainBytes = [byte[]]::new($Key.Length * 2)
    try {
        [Runtime.InteropServices.Marshal]::Copy($pointer, $plainBytes, 0, $plainBytes.Length)
        $encrypted = [Security.Cryptography.ProtectedData]::Protect(
            $plainBytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        $hex = [BitConverter]::ToString($encrypted).Replace('-', '').ToLowerInvariant()
        [IO.File]::WriteAllText($Path, $hex)
    } finally {
        [Array]::Clear($plainBytes, 0, $plainBytes.Length)
        [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($pointer)
    }
}
