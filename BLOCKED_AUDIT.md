# BLOCKED_AUDIT — цикл массового аудита остановлен

Дата: 23.09.2026, 09:40 (America/Chicago). Правило: `AGENTS.md`
«ПК не отвечает — это НЕ закрытая задача» и инструкция владельца «после трёх
падений подряд создай `BLOCKED_AUDIT.md` и остановись».

## Задача

**AU-22 — `RA-035`: `_navigate` печатает адрес в ещё грузящуюся страницу**
(`PROGRESS_AUDIT.md`, раздел «Остаток аудита»; обоснование — `DECISIONS.md`,
AUDIT-19e и AUDIT-23).

## Что именно запускалось

1. `pwsh -File scripts\update-room-pcs.ps1` — оба ПК: «обновлён и перезапущен»,
   SHA-256 `client/camera.py` совпал, задача `Running`, окно `hidden`.
2. `pwsh -File scripts\room-audit.ps1` (прогон 1):
   * AntonDorm — **57/57**, 34.4 с, `RA-035` прошёл
     (`https://www.google.com/`, 1.2 с), файл
     `data/room-eval/audit-AntonDorm.json`;
   * buro — ssh к интерактивной задаче `RowanRoomAudit` повис на ~19 минут
     (процесс `ssh` жив, CPU 0.02 с; отчёт `data\room-audit.json` на ПК не
     появился), прогон убит.
3. `pwsh -File scripts\room-audit.ps1` (прогон 2):
   * AntonDorm — **57/57**, 37.8 с;
   * buro — `ssh: connect to host 100.67.114.67 port 22: Connection timed out`,
     отчёт не прочитан (`Conversion from JSON failed…` — вместо JSON пришло
     сообщение об отсутствии файла).
4. Пробы связи: `ssh -o ConnectTimeout=8 -o BatchMode=yes … hostname` ×3 —
   три раза `Connection timed out`;
   `Test-NetConnection 100.67.114.67 -Port 22` → `False`;
   `Test-NetConnection 100.126.102.69 -Port 22` (AntonDorm) → `True`;
   `/health` хаба → `status=ok`, `outbound.clients=1` (клиент buro не на связи).

## Почему это блокирует задачу

Приёмка AU-22 — **57/57 на ОБОИХ комнатных ПК**. AntonDorm проверен дважды, а
buro не отвечает по сети, поэтому его прогон невозможен ни через
`room-audit.ps1`, ни вручную по ssh. Правка при этом уже развёрнута на buro
(в 09:09 ПК ответил «обновлён и перезапущен», хэш совпал, задача `Running`),
то есть после возвращения ПК достаточно аудита, повторное обновление не нужно.

## Что уже проверено и не требует повтора

* Код: `client/actions/browser_desktop.py` — ожидание спокойной страницы
  (`_settled_page`, 2.5 с), повтор всего ввода (`NAVIGATE_ROUNDS` = 2),
  повтор ввода при ненайденной адресной строке; тесты
  `tests/test_browser_desktop.py` (53 passed вместе с `test_browser_control.py`).
* `pytest tests -q` → **9357 passed, 15 failed**, 11 skipped (все 15 — чужой
  `tests/test_guess_who.py`, незавершённая чужая фича);
  `ruff check .` → All checks passed; `mypy common` → Success.
* АнтонDorm: 57/57 дважды, `RA-035` (та самая гонка) прошёл.

## Как продолжить (для владельца или следующего запуска)

1. Убедиться, что buro включён и в сети
   (`ssh -i C:\Users\Anton\Desktop\keyburo\buro user@100.67.114.67 hostname`).
2. `pwsh -File scripts\room-audit.ps1` — дождаться 57/57 на ОБОИХ ПК.
3. Отметить AU-22 как `[x]` в `PROGRESS_AUDIT.md` со строкой «Проверено: …» и
   продолжить цикл с AU-23 (она ждёт в `PROGRESS_AUDIT.md` ниже).

Цикл остановлен здесь по правилу трёх падений: дальше задачи не берутся, пока
условие не изменится (ПК не вернётся в сеть).

## Обновление 23.09.2026, 10:24 — buro всё ещё не в сети, цикл продолжен

Новый запуск ночного цикла взял ту же первую незакрытую задачу (AU-22) и снова
не смог её закрыть: ПК buro по-прежнему недоступен.

* `Test-NetConnection 100.67.114.67 -Port 22` → **False** (дважды);
* `Test-Connection 100.67.114.67` (ICMP) → **False**;
* `ssh -i C:\Users\Anton\Desktop\keyburo\buro -o ConnectTimeout=8 -o BatchMode=yes
  user@100.67.114.67 hostname` → `Connection timed out` (три пробы за полчаса:
  09:54, 10:05, 10:24);
* AntonDorm (`100.126.102.69:22`) — **True**, `/health` хаба —
  `outbound.clients: 1` (комната buro не подключена).

Правка `client/actions/browser_desktop.py` на buro уже развёрнута (обновление
09:09: «обновлён и перезапущен», SHA-256 совпал, задача `Running`), поэтому
после возвращения ПК достаточно прогона `pwsh -File scripts\room-audit.ps1`.

По правилу владельца «если задача не сходится — запиши причину в `DECISIONS.md`
и переходи к следующей» (`DECISIONS.md`, AUDIT-23c) цикл переходит к AU-23 —
останавливать весь аудит из-за одного выключенного ПК значило бы не сделать
ни одной из трёх оставшихся задач. AU-22 остаётся `[ ]`.
