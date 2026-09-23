# Автоцикл проверок Rowan на живом ПК: прогон -> разбор -> правка -> прогон.
#
# Это тот же приём, что у run.ps1, но задача одна: сделать так, чтобы живой
# ассистент адекватно выполнял реальные просьбы (VOICE_EVAL.md). Каждый круг
# запускает Codex на модели deepseek-flash, даёт ему отчёт стенда и логи, и
# просит починить самое важное падение, а не подправить тест.
#
#   pwsh -File run-eval.ps1                 # до 25 кругов
#   pwsh -File run-eval.ps1 -Rounds 3       # короче
#   pwsh -File run-eval.ps1 -NoActions      # только выбор инструментов, без действий
#   pwsh -File run-eval.ps1 -ThenRunMain    # после кругов сразу передать ход run.ps1
#
[CmdletBinding()]
param(
    [int]$Rounds = 25,
    [string]$Model = "deepseek-flash",
    [switch]$NoActions,
    [switch]$NoCommit,
    [switch]$ThenRunMain
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$repo = $PSScriptRoot
Set-Location $repo

$python = "C:\Users\Anton\anaconda3\envs\jarvis\python.exe"
$report = Join-Path $repo "data\live-eval\last.json"
$log = Join-Path $repo "loop-eval.log"

# ВАЖНО: аргументы собираются в типизированный массив и раскрываются через
# @pyArgs. Со splat-ом обычного массива строк Windows PowerShell 5.1 при
# запуске через -File разбирает "--actions" на отдельные символы, и
# live-eval.py падает с "unrecognized arguments: - - a c t i o n s".
$pyArgs = [string[]]@((Join-Path $repo "scripts\live-eval.py"))
if (-not $NoActions) { $pyArgs += "--actions" }
$pyArgs += @("--json", $report)

for ($i = 1; $i -le $Rounds; $i++) {
    "=== eval run $i : $(Get-Date) ===" | Out-File -FilePath $log -Append -Encoding utf8

    # 1. Настоящий прогон на живом ПК: настоящая модель, настоящие инструменты.
    Write-Host "круг $i : прогоняю живой стенд на этом ПК..." -ForegroundColor Cyan
    $block = @()
    & $python @pyArgs 2>&1 | Tee-Object -Variable block
    $failed = $LASTEXITCODE
    $block | Out-File -FilePath $log -Append -Encoding utf8
    "live-eval.py exit code: $failed" | Out-File -FilePath $log -Append -Encoding utf8

    if ($failed -eq 0) {
        Write-Host "Все проверки VOICE_EVAL.md прошли - цикл закончен" -ForegroundColor Green
        break
    }

    # 2. Codex правит причину: отчёт стенда, логи хаба, панель /admin/turns.
    $prompt = @'
Ты работаешь по VOICE_EVAL.md в C:\Users\Anton\Desktop\jarvis.

Сделай ровно это, по порядку:
1. Прочитай VOICE_EVAL.md и отчёт прогона data/live-eval/last.json.
2. Возьми САМОЕ ВАЖНОЕ падение (то, из-за которого живой ассистент не может
   выполнить обычную просьбу голосом) и найди причину в коде: смотри
   data/server.log, data/live-eval/last.json и таблицу turn_events через
   http://127.0.0.1:8770/admin/turns (пароль в .env, ROWAN_ADMIN_PASSWORD).
3. Почини причину в коде, а не в тесте. Если падение вызвано тем, что стенд
   неправильно подражает клиенту - почини стенд, но напиши это в VOICE_EVAL.md
   и не выдавай за исправление продукта.
4. Добавь или поправь модульный тест на найденную поломку.
5. Прогони make test (или: pytest tests -q, ruff check ., mypy common) и убедись,
   что не сломал остальное. Падения в tests/test_guess_who.py - чужая задача P5-21,
   их не трогай.
6. Обнови VOICE_EVAL.md: статусы проверок, дата и итог прогона.
7. Не трогай файлы вне задачи, не коммить hub/guess_who.py.

Не отчитывайся вопросами: при неоднозначности выбери вариант, который делает
живого ассистента честнее, и запиши выбор в DECISIONS.md.
'@
    Write-Host "круг $i : Codex ($Model) правит причину падения..." -ForegroundColor Cyan
    $block = @()
    codex exec --sandbox workspace-write --color never --model $Model $prompt 2>&1 |
        Tee-Object -Variable block
    $block | Out-File -FilePath $log -Append -Encoding utf8

    if (-not $NoCommit) {
        # hub/guess_who.py - чужая незаконченная задача (P5-21), не коммитим.
        git add -A -- . ':!hub/guess_who.py'
        git commit -m "auto: eval run $i $(Get-Date -Format s)" 2>&1 | Out-Null
    }
    Start-Sleep -Seconds 5
}

if ($ThenRunMain) {
    "=== eval loop done, handing over to run.ps1 : $(Get-Date) ===" |
        Out-File -FilePath $log -Append -Encoding utf8
    Write-Host "Цикл проверок закончен - запускаю run.ps1" -ForegroundColor Cyan
    & (Join-Path $repo "run.ps1")
}
