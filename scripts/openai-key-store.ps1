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
