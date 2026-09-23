# Ночной цикл «понимание запросов»: каждый круг Codex доводит очередную задачу
# из PROGRESS_UNDERSTANDING.md до конца (правка -> make test -> отметка ->
# commit -> push), чтобы к утру список был закрыт.
#
#   pwsh -File run-understanding.ps1                 # до 20 кругов
#   pwsh -File run-understanding.ps1 -Rounds 3
#   pwsh -File run-understanding.ps1 -Model deepseek-flash
#
# Отличие от run.ps1: там фазы основного ТЗ, здесь — понимание запросов, Jev и
# DeepSeek (docs/PLAN_UNDERSTANDING.md, PROGRESS_UNDERSTANDING.md).
[CmdletBinding()]
param(
    [int]$Rounds = 20,
    [string]$Model = "deepseek-flash",
    [int]$PauseSeconds = 5
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$repo = $PSScriptRoot
Set-Location $repo

$python = "C:\Users\Anton\anaconda3\envs\jarvis\python.exe"
$log = Join-Path $repo "loop-understanding.log"

for ($i = 1; $i -le $Rounds; $i++) {
    "=== understanding run $i : $(Get-Date) ===" | Out-File -FilePath $log -Append -Encoding utf8

    $remaining = (Select-String -Path (Join-Path $repo "PROGRESS_UNDERSTANDING.md") -Pattern "^- \[ \]" -ErrorAction SilentlyContinue).Count
    if (-not $remaining) {
        Write-Host "Все задачи PROGRESS_UNDERSTANDING.md закрыты" -ForegroundColor Green
        "all tasks done at $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
        break
    }
    Write-Host "круг $i : осталось задач $remaining, Codex ($Model) берёт следующую..." -ForegroundColor Cyan

    # ВАЖНО: в тексте промпта не должно быть двойных кавычек. Windows
    # PowerShell 5.1 при запуске .exe пересобирает строку аргумента и разрывает
    # её на кавычках: codex получал огрызок и падал с unexpected argument '['.
    # Проверено на первом запуске 2026-09-22 23:54.
    $prompt = @'
Ты работаешь в C:\Users\Anton\Desktop\jarvis над пониманием запросов Rowan.

Сделай ровно это, по порядку:
1. Прочитай PROGRESS_UNDERSTANDING.md и возьми ПЕРВУЮ незакрытую задачу (строка
   начинается с дефиса и пустых квадратных скобок).
2. Прочитай её основания: docs/PLAN_UNDERSTANDING.md, docs/REQUESTS_AUDIT.md,
   docs/ADMIN_PANEL.md, DECISIONS.md (API-01, API-02, API-03) и при
   необходимости PDF Rowan — интеграция TypeSafe Jev (идеи JV-xx уже
   перенесены в задачи как UG-xx).
3. Выполни ИМЕННО ЭТУ задачу целиком: правь код, добавляй тесты, НЕ выдумывай
   результаты. Нужны живые вызовы — делай их (ключи в .env: DEEPSEEK_API_KEY,
   JEV_API_KEY) и записывай реальные числа.
4. Прогони проверки ровно так:
   & C:\Users\Anton\anaconda3\envs\jarvis\python.exe -m pytest tests -q
   & C:\Users\Anton\anaconda3\envs\jarvis\python.exe -m ruff check .
   & C:\Users\Anton\anaconda3\envs\jarvis\python.exe -m mypy common
   Известные чужие падения (15 тестов tests/test_guess_who.py, незавершённая
   чужая фича) считать НЕ своими: их не трогать и не подгонять тесты под них.
5. Отметь задачу в PROGRESS_UNDERSTANDING.md закрытой ([x]) и допиши рядом
   строку Проверено: что именно запускалось и с каким результатом.
6. Закоммить только свои файлы и запушь: git add по списку файлов задачи,
   git commit -m с ID задачи и сутью, git push origin master:main
   (ветка master отслеживает origin/main).
7. Если задача не сходится — запиши причину в DECISIONS.md и переходи к
   следующей; после трёх падений подряд создай BLOCKED_UNDERSTANDING.md
   и остановись.

Работай до конца задачи и не задавай вопросов: владелец спит, значение по
умолчанию берётся из PDF и DECISIONS.md.
'@

    $block = @()
    codex exec --sandbox workspace-write --color never --model $Model $prompt 2>&1 |
        Tee-Object -Variable block
    $block | Out-File -FilePath $log -Append -Encoding utf8
    "exit code: $LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8

    Start-Sleep -Seconds $PauseSeconds
}

Write-Host "Цикл понимания закончен. Лог: $log" -ForegroundColor Green
