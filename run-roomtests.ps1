<#
Ночной цикл проверок КОМНАТНЫХ ПК: прогон -> разбор -> правка -> прогон.

Тот же приём, что у run.ps1 и run-eval.ps1, но задача одна: чтобы клиент Rowan
на настоящих комнатных ПК (Anton, buro, дальше - остальные из
deploy/room-pcs.json) работал. Каждый круг:

  1. `scripts/run-room-checks.ps1` - SSH на каждый комнатный ПК, проверки
     камеры, микрофона, HUD, устройств, браузера и журнала клиента;
  2. Codex получает отчёт `data/room-eval/*.json` и чинит САМУЮ ВАЖНУЮ
     поломку в коде клиента, а не в проверке;
  3. коммит и push (владелец просил, чтобы работа не оставалась только здесь).

  pwsh -File run-roomtests.ps1                # до 20 кругов
  pwsh -File run-roomtests.ps1 -Rounds 3      # короче
  pwsh -File run-roomtests.ps1 -NoPush        # без push
#>
[CmdletBinding()]
param(
    [int]$Rounds = 20,
    [string]$Model = '',
    [switch]$NoPush
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$repo = $PSScriptRoot
Set-Location $repo

$log = Join-Path $repo 'loop-roomtests.log'
$prompt = @'
Ты работаешь по НАСТОЯЩИМ комнатным ПК в C:\Users\Anton\Desktop\jarvis.

Сделай ровно это, по порядку:
1. Прочитай отчёты data/room-eval/*.json (их пишет scripts/run-room-checks.ps1)
   и tests/room/room_checks.py.
2. Возьми САМУЮ ВАЖНУЮ поломку - ту, из-за которой живой клиент на комнатном
   ПК не может выполнить обычную просьбу - и найди причину В КОДЕ КЛИЕНТА
   (client/, common/, scripts/). Отчёт выше - это факты с ПК, а не гипотезы.
3. Почини код клиента. Если проверка сама врёт (например, ловит занятую
   камеру во время работы клиента) - почини проверку и напиши это прямо.
4. Добавь или поправь модульный тест на найденную поломку.
5. Прогони: python -m pytest tests -q, ruff check ., mypy common.
   Падения в tests/test_guess_who.py - чужая задача P5-21, их не трогай.
6. Если правка в client/ или common/ - обнови и перезапусти клиентов на всех
   комнатных ПК: pwsh -File scripts/update-room-pcs.ps1.
7. Не трогай hub/guess_who.py и не коммить его.

Вопросов не задавай: при неоднозначности выбери вариант, который делает живого
клиента честнее, и запиши выбор в DECISIONS.md.
'@

for ($i = 1; $i -le $Rounds; $i++) {
    "=== roomtests run $i : $(Get-Date) ===" | Out-File -FilePath $log -Append -Encoding utf8
    Write-Host "круг $i : проверяю комнатные ПК по SSH..." -ForegroundColor Cyan
    $block = @()
    & (Join-Path $repo 'scripts\run-room-checks.ps1') 2>&1 | Tee-Object -Variable block
    $failed = $LASTEXITCODE
    $block | Out-File -FilePath $log -Append -Encoding utf8
    "run-room-checks.ps1 exit code: $failed" | Out-File -FilePath $log -Append -Encoding utf8

    if ($failed -eq 0) {
        Write-Host "Все проверки комнатных ПК прошли - цикл закончен" -ForegroundColor Green
        break
    }

    Write-Host "круг $i : Codex правит причину падения..." -ForegroundColor Cyan
    $block = @()
    if ($Model) {
        codex exec --sandbox workspace-write --color never --model $Model $prompt 2>&1 |
            Tee-Object -Variable block
    } else {
        codex exec --sandbox workspace-write --color never $prompt 2>&1 |
            Tee-Object -Variable block
    }
    $block | Out-File -FilePath $log -Append -Encoding utf8

    git add -A -- . ':!hub/guess_who.py'
    git commit -m "auto: room tests $i $(Get-Date -Format s)" 2>&1 | Out-Null
    if (-not $NoPush) { git push origin master:main *>> $log }
    Start-Sleep -Seconds 5
}
