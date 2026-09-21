[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
for ($i = 1; $i -le 50; $i++) {
  if (Test-Path DONE) { Write-Host "DONE - all tasks finished"; break }
  if (Test-Path BLOCKED.md) { Write-Host "BLOCKED - see BLOCKED.md"; break }
  "=== run $i : $(Get-Date)" | Tee-Object -FilePath loop.log -Append
  codex exec --sandbox workspace-write "Read AGENTS.md, the spec and PROGRESS.md. Continue executing tasks in order and do not stop until you create DONE or BLOCKED.md." 2>&1 | Tee-Object -FilePath loop.log -Append
  git add -A
  git commit -m "auto: run $i $(Get-Date -Format s)" 2>&1 | Out-Null
  Start-Sleep -Seconds 10
}