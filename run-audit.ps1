# Ночной цикл массового аудита: каждый круг Codex доводит очередную задачу из
# PROGRESS_AUDIT.md до конца (правка -> проверки -> отметка -> commit -> push),
# чтобы к утру список был закрыт.
#
#   pwsh -File run-audit.ps1              # до 14 кругов
#   pwsh -File run-audit.ps1 -Rounds 3
#   pwsh -File run-audit.ps1 -Model deepseek-flash
#
# Отличие от run.ps1 (фазы основного ТЗ) и run-understanding.ps1 (понимание
# запросов и Jev): здесь массовый аудит — три слоя из docs/AUDIT_MASS.md и
# PROGRESS_AUDIT.md.
[CmdletBinding()]
param(
    [int]$Rounds = 14,
    [string]$Model = "deepseek-flash",
    [int]$PauseSeconds = 5
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$repo = $PSScriptRoot
Set-Location $repo

$log = Join-Path $repo "loop-audit.log"

for ($i = 1; $i -le $Rounds; $i++) {
    "=== audit run $i : $(Get-Date) ===" | Out-File -FilePath $log -Append -Encoding utf8

    $remaining = (Select-String -Path (Join-Path $repo "PROGRESS_AUDIT.md") `
        -Pattern "^- \[ \]" -ErrorAction SilentlyContinue).Count
    if (-not $remaining) {
        Write-Host "Все задачи PROGRESS_AUDIT.md закрыты" -ForegroundColor Green
        "all audit tasks done at $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
        Write-Host "Не забудь вернуть звук на комнатных ПК:" -ForegroundColor Yellow
        Write-Host "  pwsh -File scripts\room-audio.ps1 unmute" -ForegroundColor Yellow
        break
    }
    Write-Host "круг $i : осталось задач $remaining, Codex ($Model) берёт следующую..." -ForegroundColor Cyan

    # ВАЖНО: в тексте промпта не должно быть двойных кавычек: Windows
    # PowerShell 5.1 при запуске .exe пересобирает строку аргумента и рвёт её
    # на кавычках (проверено на run-understanding.ps1 2026-09-22).
    $prompt = @'
Ты работаешь в C:\Users\Anton\Desktop\jarvis над массовым аудитом Rowan.

Сделай ровно это, по порядку:
1. Прочитай PROGRESS_AUDIT.md и возьми ПЕРВУЮ незакрытую задачу (строка
   начинается с дефиса и пустых квадратных скобок).
2. Прочитай её основания: docs/AUDIT_MASS.md (числа и классы поломок),
   DECISIONS.md (AUDIT-01…AUDIT-07), PROGRESS_UNDERSTANDING.md.
3. Выполни ИМЕННО ЭТУ задачу: правь код, гоняй настоящие прогоны, НЕ выдумывай
   числа. Живые вызовы делай с --workers 6 и пиши отчёт в
   data/audit/runs/<имя>.jsonl (ключи в .env: DEEPSEEK_API_KEY, JEV_API_KEY).
   Модель в конфиге — deepseek-flash; хаб после правок hub/ перезапускай
   скриптом scripts/run-openai-server.ps1.
4. Прогони проверки ровно так:
   & C:\Users\Anton\anaconda3\envs\jarvis\python.exe -m pytest tests -q
   & C:\Users\Anton\anaconda3\envs\jarvis\python.exe -m ruff check .
   & C:\Users\Anton\anaconda3\envs\jarvis\python.exe -m mypy common
   Известные чужие падения (tests/test_guess_who.py, 15 тестов незавершённой
   чужой фичи) своими не считать.
5. Отметь задачу в PROGRESS_AUDIT.md закрытой ([x]) и допиши рядом строку
   Проверено: что именно запускалось и с каким результатом.
6. Закоммить только свои файлы и запушь:
   git add по списку файлов задачи, git commit -m с ID задачи и сутью,
   git push origin master:main.
7. Если задача не сходится — запиши причину в DECISIONS.md и переходи к
   следующей; после трёх падений подряд создай BLOCKED_AUDIT.md и остановись.
8. Если правил client/ или common/ — обнови ВСЕ комнатные ПК:
   pwsh -File scripts\update-room-pcs.ps1, затем pwsh -File scripts\room-audit.ps1.

Работай до конца задачи и не задавай вопросов: владелец спит, значения по
умолчанию бери из DECISIONS.md и ТЗ.
'@

    $block = @()
    codex exec --sandbox workspace-write --color never --model $Model $prompt 2>&1 |
        Tee-Object -Variable block
    $block | Out-File -FilePath $log -Append -Encoding utf8
    "exit code: $LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8

    Start-Sleep -Seconds $PauseSeconds
}
