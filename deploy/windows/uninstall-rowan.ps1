# Снятие автозапуска Rowan (ТЗ 4.9). Ничего не удаляет, кроме самих задач.
$ErrorActionPreference = 'Continue'
foreach ($name in @('RowanHub', 'RowanClient')) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Removed scheduled task $name"
    }
    if (Get-Command nssm.exe -ErrorAction SilentlyContinue) {
        nssm stop $name 2>$null | Out-Null
        nssm remove $name confirm 2>$null | Out-Null
    }
}
Write-Host 'Rowan is no longer started automatically. Configuration and data were kept.'
