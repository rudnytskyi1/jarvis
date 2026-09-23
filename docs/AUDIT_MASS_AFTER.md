# Массовый аудит запросов Rowan

Прогон: 2026-09-23 00:47 · файл `mass-02.jsonl`
Всего сценариев: **1048**, прошло **845**, не прошло **203** (80.6 %).

Каждый сценарий идёт через ту же цепочку, что живая комната: Jev читает
реплику один раз (`hub.app.Connection._understand_turn`) и сужает набор
инструментов, затем модель отвечает и зовёт инструменты. Вердикт пишется
по строке на сценарий, поэтому отчёт можно перечитать после каждой правки.

## Итоги по семействам

| семейство | прошло | всего | доля |
|---|---:|---:|---:|
| devices | 0 | 8 | 0.0 % |
| people | 31 | 81 | 38.3 % |
| notify | 11 | 26 | 42.3 % |
| vision | 79 | 116 | 68.1 % |
| russian | 11 | 15 | 73.3 % |
| multi | 15 | 20 | 75.0 % |
| noisy | 26 | 34 | 76.5 % |
| browser | 331 | 375 | 88.3 % |
| memory | 80 | 90 | 88.9 % |
| media | 103 | 115 | 89.6 % |
| pc | 152 | 162 | 93.8 % |
| chat | 2 | 2 | 100.0 % |
| injection | 4 | 4 | 100.0 % |

## Классы поломок

| класс | сколько | пример реплики |
|---|---:|---|
| инструмент не вызван вовсе | 156 | Rowan, open chatgpt |
| в аргументах нет нужного слова | 146 | Rowan, open chatgpt |
| первым вызван не тот инструмент | 118 | Rowan, open chatgpt |
| Jev-сужение скрыло нужный инструмент | 50 | I want you to open spotify |
| the model called none of recall_conversation | 4 | hey rowan, do you remember what I said about the exam |
| модель не ответила (ключ, сеть, бюджет) | 3 | Can you please find python asyncio? |
| the model called none of browser_control, pc_control | 2 | I want you to go back to the previous page |
| the model called none of pc_control | 2 | I want you to put this text in the clipboard |
| Jev-сужение не оставило ни одного из вариантов | 2 | Rowan, read my clipboard |
| the model called none of run_skill, look_at_screen | 2 | whats the weather |
| the model called none of show_photo, say_in_room | 1 | Rowan, show me the camera |
| the model called none of look_at_camera, list_people | 1 | who is that |

## Что починить в первую очередь

### инструмент не вызван вовсе — 156

- AU-0076: Rowan, open chatgpt
- AU-0084: Rowan, open chatgpt.com
- AU-0094: I want you to open spotify
- AU-0091: Hey Rowan, can you open spotify?
- AU-0092: go to spotify

### в аргументах нет нужного слова — 146

- AU-0076: Rowan, open chatgpt
- AU-0084: Rowan, open chatgpt.com
- AU-0094: I want you to open spotify
- AU-0091: Hey Rowan, can you open spotify?
- AU-0092: go to spotify

### первым вызван не тот инструмент — 118

- AU-0076: Rowan, open chatgpt
- AU-0084: Rowan, open chatgpt.com
- AU-0094: I want you to open spotify
- AU-0091: Hey Rowan, can you open spotify?
- AU-0092: go to spotify

### Jev-сужение скрыло нужный инструмент — 50

- AU-0094: I want you to open spotify
- AU-0091: Hey Rowan, can you open spotify?
- AU-0127: Rowan, open netflix
- AU-0130: Can you please open netflix?
- AU-0279: Rowan, find weather in omaha now

### the model called none of recall_conversation — 4

- AU-0788: hey rowan, do you remember what I said about the exam
- AU-0791: do you remember what I said about the exam
- AU-0794: Rowan, do you remember what I said about the exam
- AU-1081: hey rowan, um, could you maybe do you remember what I said about the exam please?

### модель не ответила (ключ, сеть, бюджет) — 3

- AU-0253: Can you please find python asyncio?
- AU-0787: Rowan, what did we talk about yesterday
- AU-0823: I want you to remember this face as Alex

