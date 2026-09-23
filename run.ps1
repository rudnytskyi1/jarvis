#
# Ночной цикл основного плана: каждый круг Codex доводит задания из PROGRESS.md.
# После круга изменения коммитятся И отправляются в origin/main: владелец
# не раз жаловался, что работа осталась только на этом ПК.
#
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
for ($i = 1; $i -le 50; $i++) {
  if (Test-Path DONE) { Write-Host "DONE - all tasks finished"; break }
  if (Test-Path BLOCKED.md) { Write-Host "BLOCKED - see BLOCKED.md"; break }
  "=== run $i : $(Get-Date)" | Out-File -FilePath loop.log -Append -Encoding utf8
  # Tee-Object -FilePath писал UTF-16 в UTF-8-журнал, и лог нельзя было
  # прочитать; поэтому вывод идёт в окно, а в файл пишется явным UTF-8.
  $block = @()
  codex exec --sandbox workspace-write --color never "Read AGENTS.md, the spec and PROGRESS.md. Continue executing tasks in order and do not stop until you create DONE or BLOCKED.md." 2>&1 | Tee-Object -Variable block
  $block | Out-File -FilePath loop.log -Append -Encoding utf8
  git add -A
  git commit -m "auto: run $i $(Get-Date -Format s)" 2>&1 | Out-Null
  git push origin master:main *>> loop.log
  Start-Sleep -Seconds 10
}
