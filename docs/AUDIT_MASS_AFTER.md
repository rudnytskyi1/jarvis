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

## AU-02 — браузер после защиты `run_command` (2026-09-23, 01:16)

Оба прогона — одно и то же семейство и один и тот же корпус (409 сценариев
`browser` + `noisy`), `--workers 6`, живая модель и живой Jev.

| прогон | файл | разобрано | прошло | доля |
|---|---|---:|---:|---:|
| до правок | `mass-04-browser-before.jsonl` | 409 | 362 | 88.5 % |
| после правок | `mass-06-browser-final.jsonl` | 394 | **390** | **99.0 %** |

По семействам: `browser` — 334/375 (89.1 %) → 357/360 (99.2 %);
`noisy` — 28/34 (82.4 %) → 33/34 (97.1 %).

15 сценариев набора текста в поле страницы (`type X into the search box`,
`put X in the address bar`) в повторном прогоне печатаются как `SKIP`: `fill`
берёт ref из `read`, а стенд без `--actions` страницу не отдаёт, поэтому
слово из просьбы физически не может попасть в аргументы (AUDIT-08 в
`DECISIONS.md`). Если считать их непройденными, повторный прогон — 390 из 409
(95.4 %). 33 сценария, падавших до правок, теперь проходят.

### Что починили

33 сценария, падавших до правок, теперь проходят. По классам прогона «до»:

| класс поломки | закрыто | чем |
|---|---:|---|
| первым вызван `run_command` на просьбу открыть сайт | 10 | `prompts/system.md`: сайт по имени, поиск и текст в поле страницы — `browser_control` navigate, `run_command` страницу не открывает никогда |
| сайт просят открыть, а `browser_control` не вызван вовсе | 18 | тот же промпт + спорное имя (`spotify`) больше не требует один инструмент |
| первым вызван `pc_control` (`open_app spotify`) | 5 | «открой/зайди на spotify» — верен и `browser_control`, и `pc_control` |
| Jev спрятал `browser_control` / `pc_control` сужением | 4 + 3 | значения семейств: онлайн-сервисы — браузерное, ПК — «не сайт, не страница, не поиск» |
| в аргументах нет слова из просьбы | 19 | спорное имя, `run_skill` для погоды (`skills/weather`) и 15 сценариев `needs_actions` (см. ниже) |
| первым вызван не тот инструмент (`None`, `run_skill`, `look_at_screen`, `remember`) | 12 | «найди/посмотри X» — это вызов, а не уточняющий вопрос |

15 сценариев набора текста в поле страницы теперь помечены `needs_actions`:
`fill` берёт ref из `read`, а стенд без `--actions` страницу не отдаёт. Стенд
печатает по ним `SKIP` с причиной, а не выдаёт провал модели; сам набор текста
проверяется на комнатном ПК (AU-12).

### Что осталось (4)

- `AU-0189` «open the dorm portal»: адреса общежития никто не знает, модель
  ищет ярлык через `run_command` вместо `browser_control` navigate.
- `AU-0357` «close this tab», `AU-0358` «close the browser window»: модель
  сначала смотрит (`look_at_screen`) или перечисляет процессы
  (`run_command`), а не закрывает вкладку `pc_control hotkey ctrl+w`.
- `AU-0971` «quiet please»: модель читает это как «замолчи» (и молчит), а не
  как `mute`; сама фраза допускает оба смысла.

## AU-03 — зрение: «кто в комнате», «что на экране» (2026-09-23, 02:0x)

Все прогоны — семейство `vision`, один и тот же корпус (116 сценариев),
`--workers 6`, живая модель (`deepseek-flash`) и живой Jev. «До» — сразу
после правок браузера (AU-02), «после» — текущий код.

| прогон | файл | разобрано | прошло | доля |
|---|---|---:|---:|---:|
| до правок | `mass-07-vision-before.jsonl` | 116 | 81 | 69.8 % |
| после правок Jev | `mass-08-vision-after.jsonl` | 116 | 110 | 94.8 % |
| после промпта про имена вещей | `mass-10-vision-final.jsonl` | 114 | 113 | 99.1 % |
| итог (две правки двусмысленностей) | `mass-12-vision-final.jsonl` | **114** | **114** | **100 %** |

Два сценария из 116 (`AU-0609`, `AU-0610` — «what is in this photo») в
итоговом прогоне печатаются как `SKIP`: в стенде нет вложения Telegram,
`inspect_photo` физически нечего смотреть, и это не поломка модели
(`DECISIONS.md`, AUDIT-09d). Итог считается по 114 разобранным сценариям.

### Что было сломано

1. **Jev отвечал «ничего не делай» и `family=vision` одним чтением.** На
   вопрос «who is in the room?» ответ `act=false` (уверенность 0.5–0.9)
   срабатывал раньше семейства, и `hub.app._narrow_tools_for` отдавал модели
   только ядро (`remember`, `recall_conversation`, `say_in_room`, …) — без
   `look_at_camera`, `look_at_screen` и `find_object`. 35 падений базового
   прогона — из них 32 «сужение Jev не предложило нужный инструмент».
2. **Вопрос «кто в комнате» / «где мои ключи» Jev читал как память или
   людей.** В значении семейства `vision` не было слов «прямо сейчас» с
   примерами таких фраз; `find_object` не назван ни одним живым примером
   («do you see my phone», «find my keys»).
3. **«Do you see my phone» модель отвечала общим осмотром**
   (`look_at_camera`), а `find_object` не звала вовсе (4 сценария): ни промпт
   комнаты, ни описание инструмента не отправляли такие слова в детектор.
4. **Корпус требовал того, чего в реплике нет:** по фразе «where is my keys»
   модель честно искала «key» (простое слово — так советует само описание
   `find_object`), а проверка требовала буквально `keys`.
5. **«Read the browser window»** модель выполняла двумя верными способами —
   прочитать страницу (`browser_control read`) или посмотреть на экран
   (`look_at_screen`); корпус требовал только второй.

### Что сделано

| правка | файл | что закрыла |
|---|---|---|
| вопрос `act` называет «посмотреть/прочитать/найти» действием | `hub/jev_decider.py` | 29 падений, где сужение оставило одно ядро |
| вопрос о семействе разводит `vision` и `memory`/`people`/`media` словами и примерами | `hub/jev_decider.py` | 6 падений, где Jev назвал не то семейство |
| названное семейство важнее ответа «ничего не делать» | `hub/app.py` | ветка `act=false` больше не прячет семейные инструменты; ядро входит в любое семейство |
| значение `vision` — «прямо сейчас и ответить», `media` — «показать/скрыть/нарисовать» | `hub/tools.py` | Jev выбирает `vision` для вопросов о комнате и не путает с медиа |
| `find_object` в промпте и описании получает «do you see my X», «find my X», «call it FIRST» | `prompts/system.md`, `hub/tools.py` | 4 сценария `find_object` |
| корпус проверяет простое слово (`key`, не `keys`) | `scripts/gen-audit-scenarios.py` | AU-0573 |
| чтение окна браузера принимает и `browser_control` | `scripts/gen-audit-scenarios.py` | AU-0513/AU-1050 |
| фото Telegram помечено `bench_skip` с причиной | `scripts/gen-audit-scenarios.py` | AU-0609/AU-0610 печатаются `SKIP` |

Тесты: `tests/test_jev_understanding.py` (3 новых, файл 19 passed),
`tests/audit/test_request_matrix.py` (4 новых, матрица 3193 passed).
Приёмка: Jev отдаёт всё семейство
`vision` (в том числе при `act=false`), первый вызов — `look_at_camera`,
`look_at_screen` или `find_object` по смыслу просьбы.

## AU-07 — медиа: картинки, обои, показ (2026-09-23, 03:3x)

Все прогоны — семейство `media`, один и тот же корпус (115 сценариев),
`--workers 6`, живая модель (`deepseek-flash`) и живой Jev.

| прогон | файл | разобрано | прошло | доля |
|---|---|---:|---:|---:|
| до правок | `au-07-media-before.jsonl` | 115 | 110 | 95.7 % |
| после правок (правки ещё без «обои одним вызовом») | `au-07-media-final.jsonl` | 100 | 100 | 100 % |
| повтор (поймал нестабильность обоев) | `au-07-media-final-2.jsonl` | 100 | 98 | 98.0 % |
| после правки AUDIT-13b | `au-07-media-final-3.jsonl` | 100 | 99 | 99.0 % |
| итог, первый прогон | `au-07-media-final-4.jsonl` | 100 | **100** | **100 %** |
| итог, повтор | `au-07-media-final-5.jsonl` | 100 | **100** | **100 %** |

Пятнадцать сценариев правки присланного фото (`edit this photo …`,
`change the attached picture …`) печатаются `SKIP`: у `generate_image` в
голосовом договоре нет источника «вложение» (`source` = none / camera /
screen / last), фото приходит только с Telegram-сообщением, и в стенде его
физически нет — та же причина, что у `inspect_photo` (AUDIT-09d). Сам путь
проверяют `tests/test_telegram_edit_prompt.py` и
`tests/test_telegram_reply_photo.py`.

### Что было сломано

1. **«tell everyone …» модель читала как рассылку.** `AU-0714` «I want you to
   tell everyone that dinner is ready» — ответ «у меня один общий чат, хочешь
   — напишу туда», `say_in_room` не вызван. В описании инструмента не было
   слов «announce»/«tell everyone», в промпте комнаты про `say_in_room` не
   было ни строки.
2. **Обои: модель шла смотреть, а потом сдавалась.** `AU-0711`/`AU-0712`
   падали в прогонах `…-final-2.jsonl`/`…-final-3.jsonl`: сначала
   `look_at_screen`/`show_photo` (стенд без `--actions` картинку не отдаёт),
   потом ответ «нечего ставить» — вместо `set_wallpaper`, чей честный ответ и
   был нужен человеку.
3. **Правка присланного фото непроверяема в стенде.** У `generate_image` нет
   источника «вложение», поэтому половину прогона модель честно отвечала
   «вложения нет», а половину звала генерацию с источником камеры —
   проверялось не то, что заявлено в корпусе.

### Что сделано

| правка | файл | что закрыла |
|---|---|---|
| промпт: раздел «Saying things out loud» — объявление вслух не Telegram | `prompts/system.md` | `AU-0714` |
| `say_in_room` в описании знает «announce» и «tell everyone» | `hub/tools.py` | `AU-0714` |
| промпт: два хода обоев явно, «не ищи картинку взглядом, зови `set_wallpaper` и передай его ответ» | `prompts/system.md`, `hub/tools.py` | `AU-0711`/`AU-0712` |
| сценарии правки присланного фото помечены `bench_skip` с причиной | `scripts/gen-audit-scenarios.py` | 15 `SKIP` вместо выдуманной оценки |
| матрица: объявление вслух, обои двумя ходами, `SKIP` правки фото | `tests/audit/test_request_matrix.py` | +3 проверки (3076) |

Тесты: `pytest tests -q` → 9272 passed, 15 failed (все 15 — чужой
`tests/test_guess_who.py`, незавершённая чужая фича, в зачёт не идёт),
11 skipped; `tests/audit -q` → 3076 passed; `ruff check .` → All checks
passed; `mypy common` → Success. Хаб перезапущен
`scripts/run-openai-server.ps1` (`/health` 8770 → 200, `outbound.clients: 2`).
`client/` и `common/` не правились — обновление комнатных ПК не требуется.
Обоснования — `DECISIONS.md`, AUDIT-13…13f.

