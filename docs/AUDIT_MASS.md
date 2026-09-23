# Массовый аудит запросов Rowan

> Здесь первый прогон (до правок). Числа «после» — в `docs/AUDIT_MASS_AFTER.md`;
> сводная таблица «до/после» — в конце этого файла.

Прогон: 2026-09-23 00:23 · файл `mass-01.jsonl`
Всего сценариев: **1106**, прошло **812**, не прошло **294** (73.4 %).

Каждый сценарий идёт через ту же цепочку, что живая комната: Jev читает
реплику один раз (`hub.app.Connection._understand_turn`) и сужает набор
инструментов, затем модель отвечает и зовёт инструменты. Вердикт пишется
по строке на сценарий, поэтому отчёт можно перечитать после каждой правки.

## Итоги по семействам

| семейство | прошло | всего | доля |
|---|---:|---:|---:|
| devices | 0 | 64 | 0.0 % |
| skills | 0 | 2 | 0.0 % |
| people | 23 | 81 | 28.4 % |
| notify | 9 | 26 | 34.6 % |
| russian | 8 | 15 | 53.3 % |
| vision | 71 | 116 | 61.2 % |
| noisy | 22 | 34 | 64.7 % |
| multi | 15 | 20 | 75.0 % |
| browser | 329 | 375 | 87.7 % |
| media | 101 | 115 | 87.8 % |
| memory | 80 | 90 | 88.9 % |
| pc | 148 | 162 | 91.4 % |
| chat | 2 | 2 | 100.0 % |
| injection | 4 | 4 | 100.0 % |

## Классы поломок

| класс | сколько | пример реплики |
|---|---:|---|
| первым вызван не тот инструмент | 262 | Rowan, open google.com in the browser |
| инструмент не вызван вовсе | 252 | Rowan, open google.com in the browser |
| в аргументах нет нужного слова | 218 | Rowan, open google.com in the browser |
| Jev-сужение скрыло нужный инструмент | 80 | I want you to open spotify |
| модель не ответила (ключ, сеть, бюджет) | 4 | find dorm rules |
| the model called none of recall_conversation | 4 | hey rowan, do you remember what I said about the exam |
| the model called none of browser_control, pc_control | 2 | Rowan, click the Videos tab |
| the model called none of pc_control | 1 | Rowan, read my clipboard |
| Jev-сужение не оставило ни одного из вариантов | 1 | Rowan, read my clipboard |
| the model called none of show_photo, say_in_room | 1 | Rowan, show me the camera |

## Что починить в первую очередь

### первым вызван не тот инструмент — 262

- AU-0026: Rowan, open google.com in the browser
- AU-0037: Could you open gmail, please?
- AU-0055: Can you please open github?
- AU-0076: Rowan, open chatgpt
- AU-0073: Rowan, open chatgpt now

### инструмент не вызван вовсе — 252

- AU-0026: Rowan, open google.com in the browser
- AU-0037: Could you open gmail, please?
- AU-0055: Can you please open github?
- AU-0076: Rowan, open chatgpt
- AU-0073: Rowan, open chatgpt now

### в аргументах нет нужного слова — 218

- AU-0026: Rowan, open google.com in the browser
- AU-0037: Could you open gmail, please?
- AU-0055: Can you please open github?
- AU-0076: Rowan, open chatgpt
- AU-0073: Rowan, open chatgpt now

### Jev-сужение скрыло нужный инструмент — 80

- AU-0094: I want you to open spotify
- AU-0095: Could you go to spotify, please?
- AU-0091: Hey Rowan, can you open spotify?
- AU-0109: open twitch
- AU-0127: Rowan, open netflix

### модель не ответила (ключ, сеть, бюджет) — 4

- AU-0264: find dorm rules
- AU-0789: Could you find the conversation where we discussed the dorm, please?
- AU-0787: Rowan, what did we talk about yesterday
- AU-1080: hey rowan, um, could you maybe what did we talk about yesterday please?

### the model called none of recall_conversation — 4

- AU-0788: hey rowan, do you remember what I said about the exam
- AU-0791: do you remember what I said about the exam
- AU-0794: Rowan, do you remember what I said about the exam
- AU-1081: hey rowan, um, could you maybe do you remember what I said about the exam please?


## Первый прогон и правки по нему (2026-09-23)

| что нашлось | сколько | что сделано |
|---|---:|---|
| сайт открывают командой оболочки (`run_command`) вместо `browser_control` | 262 | защита в хабе: `hub/tools.py::opening_a_web_page`, отказ называет верный инструмент (`hub/app.py`), подтверждение по F-113 для такого вызова больше не спрашивается, стенд видит ту же защиту (`scripts/live-eval.py`) |
| Jev-сужение спрятало нужный инструмент | 80 | значения семейств переписаны (`hub/tools.py::TOOL_FAMILY_MEANINGS`): «кто в комнате», «что на экране», «кто ты знаешь», «расскажи про погоду» теперь читаются как `vision`/`people`/`pc` |
| адрес без схемы и `youtube .com` | десятки | `client/actions/browser.py::web_url` дополняет `https://`, читает пробел вокруг точки как опечатку (проверено на двух комнатных ПК: RA-036 открыл `https://www.youtube.com/`) |
| результат действия приходил без слов (`ok: true`, `output: null`) | 11 | `client/actions/dispatcher.py` отдаёт хаб-у `detail`: «volume 30%» |
| буфер обмена падал с `OverflowError` | 3 | `client/actions/pc.py::_kernel32` с `argtypes` (64-битные `HGLOBAL`) |
| лишний аргумент отменял весь вызов | десятки | `hub/tool_args.py`: объявленные поля строго, чужое поле отбрасывается, опечатка поля возвращается модели с подсказкой |
| `update-room-pcs.ps1` не работал в Windows PowerShell 5.1 | — | BOM у `.ps1` с русским текстом + `foreach` вместо `@(… | ConvertFrom-Json)` |
| устройства и скиллы в стенде непроверяемы | 66 | помечены `bench_skip` с причиной (AUDIT-07); задача AU-05 — собирать в стенде живой префикс `[home: …]` |

Проверка исполнения на реальных ПК (`scripts/room-audit.ps1`, 57 действий на
каждый): **AntonDorm 57/57**, **buro 57/57** — громкость и её ограничение,
буфер обмена, открытие/закрытие приложения, переходы Chrome/Firefox по семи
адресам (включая `youtube .com` и `example.com`), камера, экран и 12 отказов
(`file://`, `javascript:`, адрес с паролем, неизвестная команда, пустой
`run_command`, инструмент, которого нет).
