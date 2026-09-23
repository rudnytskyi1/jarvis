# План работ: массовый аудит запросов (ночной цикл `run-audit.ps1`)

Владелец просил аудит с несколькими тысячами сценариев, проверку всего, что
сломано, и запуск через `.ps1`, чтобы работа шла, пока он спит
(2026-09-23). Задачи берутся строго по порядку: выполнить → прогнать
проверки → отметить `[x]` с записью «Проверено: …» → следующая.

Инструменты аудита:

| Что | Как |
|---|---|
| Корпус реплик | `python scripts/gen-audit-scenarios.py` → `data/audit/scenarios.jsonl` |
| Живой прогон | `python scripts/live-eval.py --scenarios data/audit/scenarios.jsonl --workers 6 --jsonl data/audit/runs/<имя>.jsonl --quiet` |
| Одно семейство | то же + `--family browser` (можно несколько раз) |
| Сводка | `python scripts/audit-summary.py --runs data/audit/runs/<имя>.jsonl --write docs/AUDIT_MASS.md` |
| Офлайн-матрица | `python -m pytest tests/audit -q` |
| Комнатные ПК | `pwsh -File scripts/room-audit.ps1` (запускает `tests/room/room_audit.py` в интерактивной сессии ПК через `scripts/restart-client.ps1`-подобную разовую задачу) |
| Звук на ПК | `pwsh -File scripts/room-audio.ps1 mute` / `… unmute` |

Правила: не выдумывать результаты; каждая правка — с тестом; после правки
клиента — `pwsh -File scripts/update-room-pcs.ps1` (обновить ВСЕ комнатные ПК);
`git add` нужных файлов + `git commit` + `git push origin master:main`.
Обоснования решений — `DECISIONS.md`, раздел «Массовый аудит 2026-09-23».

## Сделано

- [x] **AU-00 — корпус и стенд.** Проверено: `scripts/gen-audit-scenarios.py`
  даёт 1106 сценариев (browser 375, pc 162, vision 116, media 115, memory 90,
  people 81, devices 64, noisy 34, notify 26, multi 20, russian 15, injection 4,
  skills 2, chat 2); `scripts/live-eval.py` получил `--workers`, `--scenarios`,
  `--jsonl`, `--family`, `--no-understanding` и шаг Jev внутри прогона;
  `tests/audit/test_request_matrix.py` — 3216 офлайн-проверок, зелёные.
- [x] **AU-00b — звук на комнатных ПК выключен** (просьба владельца на время
  аудита). Проверено: `scripts/room-audio.ps1 mute` → «sound muted»,
  `volume_set 0` → «volume 0%» на AntonDorm и buro.
- [x] **AU-00c — первый массовый прогон.** Проверено: 1106 сценариев,
  812 прошло (73.4 %). Классы поломок — в `docs/AUDIT_MASS.md`.

## Задачи

- [x] **AU-01 — повторный прогон после правок.** Проверено: `mass-02.jsonl` —
  845 из 1048 прошло (80.6 %) против 812 из 1106 (73.4 %) в первом прогоне
  (`mass-01.jsonl`); числа и таблица правок — в `docs/AUDIT_MASS.md`,
  семейные итоги — в `docs/AUDIT_MASS_AFTER.md`. Остаток провалов в основном
  не поломки, а честные отказы стенда («действия на ПК выключены», «личность
  выключена», «устройств нет») — это записано задач AU-05/AU-06.
- [x] **AU-02 — браузер: остатки после защиты `run_command`.** Проверено:
  живой прогон `--family browser --family noisy` с `--workers 6`
  (`data/audit/runs/mass-04-browser-before.jsonl`, 409 сценариев) — 362
  прошло, 47 падений. Главная причина нашлась в промпте: `prompts/system.md`
  сам учил «For sites and online services … call `run_command` with
  Start-Process and the address» (10 падений «первым вызван `run_command`») и
  читал «открой chrome/spotify» как программу (5 «первым вызван
  `pc_control`»); значение семейства Jev `pc` не исключало сайты (6
  «сужение скрыло `browser_control`»). Сделано: промпт переписан на
  `browser_control` navigate для сайта по имени, поиска и текста в поле
  страницы; `TOOL_FAMILY_MEANINGS` называет онлайн-сервисы (YouTube, Netflix,
  Twitch, Gmail, Reddit, Spotify) в браузерном семействе и «не сайт, не
  страница, не поиск» — в ПК; слово из просьбы проверяется в том вызове,
  который модель сделала, когда верны два инструмента (`spotify` есть и
  сайтом, и программой — `DECISIONS.md` AUDIT-08); «найди погоду» принимает
  и `run_skill` (`skills/weather`); сценарии набора текста помечены
  `needs_actions`. Повторный прогон тех же 409 сценариев
  (`data/audit/runs/mass-06-browser-final.jsonl`) — **390 из 394** разобранных
  прошло (98.98 %), 33 прежде падавших сценария закрыто, 15 печатаются как
  `SKIP` (`fill` берёт ref из `read`, стенд без `--actions` страницу не
  отдаёт). Осталось 4: `AU-0189` — «the dorm portal» без известного адреса
  модель ищет ярлык через `run_command`; `AU-0357`/`AU-0358` — закрытие
  вкладки/окна начинается с `look_at_screen`/`run_command`; `AU-0971` —
  «quiet please» модель читает как «замолчи», а не как `mute`. Тесты:
  `tests/audit/test_request_matrix.py` (матрица + вердикт стенда),
  `tests/test_web_page_guard.py` (промпт и защита хаба говорят одно и то же);
  `pytest tests -q` → 9368 passed, 15 failed и все 15 — чужой
  `tests/test_guess_who.py` (незавершённая чужая фича, в зачёт не идёт);
  `ruff check .` → All checks passed; `mypy common` → Success. Хаб перезапущен
  `scripts/run-openai-server.ps1` (`/health` → 200). Коммит/пуш из песочницы
  невозможны (`.git` только для чтения, у remote нет креденшелов) — список
  файлов для внешнего скрипта в `DECISIONS.md`, AUDIT-08f; `hub/tools.py`,
  этот файл и `DECISIONS.md` уже попали в чужие коммиты `7373aab`/`bd3323a`.
- [x] **AU-03 — зрение: «кто в комнате», «что на экране».** Проверено: живой
  прогон `--family vision --workers 6` (`data/audit/runs/mass-07-vision-before.jsonl`,
  116 сценариев) — 81 прошло (69.8 %), 32 падения из 35 — Jev не предложил
  нужный инструмент. Причина: Jev отвечал `act=false` и `family=vision` одним
  чтением, а ветка «просто вопрос» отдавала только ядро; плюс значения
  семейств не называли живые фразы. Сделано: вопрос `act` считает
  «посмотреть/прочитать/найти» действием; названное семейство важнее
  `act=false` (`hub/app.py`); `vision`/`media` в значениях и в вопросе Jev
  разведены примерами (`hub/tools.py`, `hub/jev_decider.py`); `find_object`
  получил триггеры «do you see my X»/«find my X» в промпте и описании; корпус
  проверяет простое слово и принимает чтение окна браузера обоими
  инструментами; фото Telegram помечено `bench_skip`. Итог:
  `data/audit/runs/mass-12-vision-final.jsonl` — 114 из 114 разобранных
  (100 %), `AU-0609`/`AU-0610` — `SKIP` (нет вложения). Тесты
  `tests/test_jev_understanding.py` (+3), `tests/audit/test_request_matrix.py`
  (+4, матрица 3193); `ruff check .` → All checks passed; `mypy common` →
  Success; `pytest tests -q` → 9378 passed, 24 failed — все чужие (15
  `test_guess_who.py` + 9 незавершённый офлайн-клиент), в зрении и Jev ноль.
  Хаб перезапущен `scripts/run-openai-server.ps1` (`/health` → 200, комнаты
  переподключились). `git add`/`push` из песочницы невозможны (`.git` только
  для чтения, у remote нет креденшелов) — список файлов для внешнего коммита
  в `DECISIONS.md`, AUDIT-09h. `client/` и `common/` не правились, обновление
  комнатных ПК не требуется.
- [ ] **AU-04 — люди: запись лица и голоса.** `--family people`. «Сохрани это
  лицо как X» обязан дойти до `enroll_face` (взгляд на кадр перед этим
  допустим); «кто ты знаешь» — до `list_people`.
- [ ] **AU-05 — уведомления: групповое сообщение и правило присутствия.**
  `--family notify`. Личное имя без канала честно не отправляется (в конфиге
  один `chat_id`); групповое сообщение — `telegram_send`; «скажи, когда кто-то
  войдёт» — `create_rule` (правило presence следит камерой комнаты).
- [ ] **AU-06 — стенд должен собирать живой префикс `[home: …]`.** Сейчас
  `scripts/live-eval.py` шлёт только системный промпт, а живой ход добавляет
  `[at … | speaker: …] [home: …] [memory: …]` (`hub/speaker_context.py`).
  Из-за этого семейства devices и skills в стенде непроверяемы (AUDIT-07).
  Сделать: собирать тот же префикс (устройства и скиллы из живого дома),
  снять `bench_skip` с устройств и скиллов, прогнать их.
- [ ] **AU-07 — медиа: картинки, обои, показ.** `--family media`. Проверить
  `generate_image` (явная просьба нарисовать/отредактировать), `set_wallpaper`,
  `show_photo`, `save_photo`, `say_in_room`; отдельно — правка присланного
  фото и обои из готовой картинки.
- [ ] **AU-08 — память.** `--family memory`: `remember`, `forget_fact`,
  `list_memory`, `recall_conversation`; сверить, что сказанное действительно
  сохранено (таблицы `facts`), а «что ты помнишь обо мне» отвечает списком.
- [ ] **AU-09 — ПК: громкость, приложения, клавиши.** `--family pc`,
  `--family russian`; проверить `volume_set/down/up`, `open_app/close_app`
  (в т.ч. имя в `target`), медиа-клавиши, `clipboard_read/write/paste`,
  `hotkey` для «свернуть всё», `minimize_app`.
- [ ] **AU-10 — две просьбы в одной реплике (UG-08).** `--family multi`:
  «открой ютуб и сделай громче» должно дать ОБА вызова. Если сужение по
  семейству мешает — вернуть Jev четвёртый вопрос `single` и не сужать при
  «несколько» (см. `PROGRESS_UNDERSTANDING.md`).
- [ ] **AU-11 — латентность хода.** По `mass-*.jsonl` посчитать долю ходов
  дольше 4 с и первый звук позже 2.5 с (ТЗ 15.1, F-101); найти, что именно
  столько занимает (STT, Jev, раунды модели), и записать замеры до/после в
  `docs/TZ_STATUS.md`.
- [ ] **AU-12 — комнатные ПК после каждой правки клиента.** Прогнать
  `scripts/update-room-pcs.ps1`, затем `scripts/room-audit.ps1`; 57/57
  действий на каждом ПК — приёмка. Записать числа в отчёт.
- [ ] **AU-13 — итог.** Обновить `docs/AUDIT_MASS.md`, `docs/TZ_STATUS.md`,
  `README.md` (какие слои аудита есть и как их запускать), `DECISIONS.md`.
  После закрытия — вернуть звук на ПК: `pwsh -File scripts/room-audio.ps1 unmute`.
- [x] **AU-14 — лимит шагов computer_use снят (просьба владельца 23.09.2026).**
  «the step limit of 15 steps is reached… убери эту фигню. никаких лимитов: все
  что его попросили — делает». Проверено: `max_steps: 0` = лимита нет,
  `["*"]` = любое приложение, `max_tool_rounds: 40`; `pytest
  tests/test_computer_use.py tests/test_computer_use_audit.py -q` → 37 passed;
  `ruff check .` → All checks passed; `mypy common` → Success. Обоснование —
  `DECISIONS.md`, «Лимит шагов computer_use снят». **Чего нет:** живого прогона
  длиннее пятнадцати шагов (появится после перезапуска хаба).
- [x] **AU-15 — бюджет времени хода снят (CU-LIMIT-02).** Владелец: «никаких
  лимитов». `server.timeouts.reply_ms` накрывал весь раунд модели вместе с
  инструментами (20 с по умолчанию) — длинная задача обрывалась словами «не
  успел придумать ответ». Проверено: `0` в бюджете = без предела
  (`hub.app.UNBOUNDED_BUDGET_S` = 86400 с), `pytest tests/test_stage_timeouts.py
  -q` → 17 passed; `Config.model_validate(config.openai.yaml)` → `reply_ms 0`.
  Обоснование — `DECISIONS.md`, «CU-LIMIT-02».
- [x] **AU-16 — счётчик раундов инструментов снят (CU-LIMIT-03).** `0` в
  `server.llm.max_tool_rounds` = без счётчика (`UNLIMITED_TOOL_ROUNDS` = 1000
  как предохранитель от зацикливания). Проверено: `pytest tests/test_multi_step.py
  tests/test_llm_vllm_provider.py tests/test_config.py tests/test_stage_timeouts.py
  -q` → 60 passed; `Config.model_validate(config.openai.yaml)` →
  `max_tool_rounds 0`. Обоснование — `DECISIONS.md`, «CU-LIMIT-03».
- [x] **AU-17 — разговор без обрывов и ранний старт действий (CU-PAUSE-01,
  CU-EARLY-01).** Пауза 1.5 с не закрывает реплику, а после слова-связки
  («и», «and») запись терпит ещё 3.5 с (`client.vad.unfinished_hold_ms`,
  `sentence_unfinished`); раунд по черновику получил право НАЧАТЬ обратимое
  «открыть приложение/страницу» (`may_run_early`), остальное ждёт
  подтверждённого транскрипта. Проверено: `pytest tests/test_vad_machine.py
  tests/test_vad_endpoint.py -q` → 14 passed; `pytest tests/test_early_start.py
  -q` → 33 passed (в том числе «приложение открылось, пока STT ещё считает»).
- [x] **AU-18 — TypeSafe Jev: отчёт числами и живой вызов (JE-01).** Проверено:
  `python scripts/jev_probe.py` → ответ за 0.62 с (`family=devices 0.98`);
  `python scripts/jev_usage_report.py` → 5061 решение, все `rules`; в
  `turn_events` нет события `understanding`. Бюджет чтения поднят 900 → 2000 мс.
- [ ] **AU-19 — Jev в Telegram-чате.** Открытая задача из JE-01: чтение Jev
  (`_understand_turn`) стоит только на голосовом пути, а Telegram-чат идёт
  через `hub/telegram_chat.py` → `llm.reply_text` без чтения и без сужения
  инструментов. Владелец просил, чтобы в Telegram были те же возможности, что
  и у голосового ассистента; сюда же — второй провайдер `jev` в цепочке решений
  (сейчас всегда отвечает `rules`).
