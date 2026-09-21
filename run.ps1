for ($i = 1; $i -le 50; $i++) {
  if (Test-Path DONE) { Write-Host "ГОТОВО"; break }
  if (Test-Path BLOCKED.md) { Write-Host "ЗАСТРЯЛ, см. BLOCKED.md"; break }
  "=== запуск $i : $(Get-Date)" | Tee-Object -FilePath loop.log -Append
  codex exec -a never --sandbox workspace-write `
    "Прочитай AGENTS.md, ТЗ и PROGRESS.md. Продолжай выполнять задачи по порядку и не останавливайся, пока не создашь DONE или BLOCKED.md." `
    2>&1 | Tee-Object -FilePath loop.log -Append
  Start-Sleep -Seconds 10
}