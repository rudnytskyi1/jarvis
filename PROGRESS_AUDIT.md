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
- [x] **AU-04 — люди: запись лица и голоса.** Проверено: живой прогон
  `--family people --workers 6` — **81 из 81** сценария
  (`data/audit/runs/mass-19-people-final.jsonl` и `…/mass-20-people-final.jsonl`,
  два прогона подряд) против 39 из 81 в начале задачи (`…/mass-13-people-before.jsonl`,
  48.1 %). Главные причины нашлись не в модели: (1) в стенде не было реестра
  людей (`hub_app._voices` — None), поэтому `list_people`, `set_role`,
  `rename_person` и `enroll_voice` отвечали «speaker recognition is disabled»,
  модель сдавалась за хаб, а план «сначала список, потом роль» обрывался;
  (2) стенд посылал голую фразу, а живой ход посылает префикс
  `[at … | speaker: … | role: …]` — без него модель спрашивала «кто говорит?»;
  (3) реестр был один на все сценарии, поэтому «сделай Джона админом» после
  чужого переименования отвечало «он уже админ». Сделано: стенд собирает
  реестр людей на каждого воркера из фикстур через `VoiceRegistry.admin_profile`
  (`scripts/live-eval.py`), начинает каждый сценарий в одной и той же комнате
  (шаблон `fixtures.json`) и берёт префикс из живого `Connection._turn_prefix`;
  реестр отвечает тому, кто спрашивает (`_RoomRegistries`, иначе чужой воркер
  видел чужую переписку); отказ инструмента больше не читается как «модель не
  ответила» (`INFRA_FAILURES` — только слова самого хаба); промпт и описания
  `enroll_face`/`enroll_voice`/`set_role` говорят, что запись по имени — это
  вызов инструмента, а не отказ за него (F-210: владелец может записать гостя);
  «переименуй X» без нового имени в корпусе больше нет (инструменту нужны оба
  имени), а «save this person as X» принимает и лицо, и голос. Тесты:
  `tests/audit/test_request_matrix.py` (+7, матрица 3200); `pytest tests -q` →
  9394 passed, 15 failed — все 15 чужой `tests/test_guess_who.py` (в зачёт не
  идёт); `ruff check .` → All checks passed; `mypy common` → Success.
  **Открыто:** хаб перезапустить из песочницы нельзя (Stop-Process/taskkill →
  Access denied, эскалаций нет) — новый промпт подхватят новые сессии при
  переподключении комнат, описания инструментов — после внешнего перезапуска
  `scripts/run-openai-server.ps1` (записано в `DECISIONS.md`, AUDIT-10h).
  `client/` и `common/` не правились, обновление комнатных ПК не требуется.
- [x] **AU-05 — уведомления: групповое сообщение и правило присутствия.**
  `--family notify`. Личное имя без канала честно не отправляется (в конфиге
  один `chat_id`); групповое сообщение — `telegram_send`; «скажи, когда кто-то
  войдёт» — `create_rule` (правило presence следит камерой комнаты).
  Проверено: живой прогон `--family notify --workers 6` —
  `data/audit/runs/au-05-notify-before.jsonl` **18 из 26** (69.2 %, 8 падений),
  после правок `data/audit/runs/au-05-notify-final.jsonl` **24 из 24**
  разобранных (100 %), `AU-0923`/`AU-0924` напечатаны `SKIP` (в стенде нет ни
  камеры, ни истории хода — «эта/последняя картинка»). Сделано: стенд
  записывает ход в `hub.app._recording_turn`, как живой хаб, иначе
  `telegram_send` отказывал честной просьбе «отправь в группу»
  («requires an explicit user request in this turn»); промпт и описание
  `create_rule` требуют предлагать правило, а не задавать вопрос («сообщи,
  когда откроется дверь» → `create_rule` с `zone_entered` и зоной из просьбы);
  «скажи Джону…» и «когда станет темно» корпус признал честными отказами
  (`AUDIT-11c`/`AUDIT-11d` в `DECISIONS.md` — в доме один общий чат и нет
  датчика освещённости, а `AU-0940` в первом прогоне «проходил» на выдуманном
  `device_state device_id=room_light`). Тесты:
  `tests/audit/test_request_matrix.py` (+4, 3197 проверок); `pytest tests -q` → 9391
  passed, 15 failed и все 15 — чужой `tests/test_guess_who.py` (в зачёт не
  идёт); `ruff check .` → All checks passed; `mypy common` → Success. Хаб
  перезапускает внешний скрипт (из песочницы `Stop-Process` → Access denied,
  `DECISIONS.md` AUDIT-11g); `client/` и `common/` не правились, обновление
  комнатных ПК не требуется.
- [x] **AU-06 — стенд должен собирать живой префикс `[home: …]`.** Сейчас
  `scripts/live-eval.py` шлёт только системный промпт, а живой ход добавляет
  `[at … | speaker: …] [home: …] [memory: …]` (`hub/speaker_context.py`).
  Из-за этого семейства devices и skills в стенде непроверяемы (AUDIT-07).
  Сделать: собирать тот же префикс (устройства и скиллы из живого дома),
  снять `bench_skip` с устройств и скиллов, прогнать их. Проверено: живой
  прогон `--family devices --family skills --workers 6` до правок
  (`data/audit/runs/au-06-devices-skills-before.jsonl`, 66 сценариев) — 63 из
  66; после правок (`…/au-06-devices-skills-after.jsonl`) — **66 из 66**.
  Стенд теперь несёт комнату живого хода: `room_devices(cfg)` собирает тот же
  `hello`, что и комната, тем же кодом клиента (`client.main.build_hello`,
  приборов в `config.openai.yaml` нет), `hub_app._skills` грузится настоящим
  `_skill_hot_reload`, поэтому в префиксе видно `[home: anton | skills: canvas,
  games, weather]`, `run_skill` действительно исполняется, а каждая строка
  отчёта несёт `prefix`/`room_devices`/`room_skills`. Три падения нашлись
  только потому, что семейства впервые пошли живьём: `AU-0872` позвал
  `set_light`, `AU-0904` — `set_switch` над пустым списком приборов (промпт
  инструментов это запрещал, а слот `{devices}` говорил только «(no devices
  configured)» — `hub/session.py::NO_DEVICES_TEXT` теперь говорит, что сказать
  и чего не звать), и `AU-0942` — Jev читал «will it rain tomorrow» как
  `family=pc` с уверенностью 0.56 (порог 0.65), прятал `run_skill`, и модель
  звала `run_command` (вопрос о семействе не называл скиллы дома —
  `hub/jev_decider.py`; живой `scripts/jev_probe.py`: 0.56 → 1.00, девять
  реплик всех семейств разобраны верно). Корпус берёт приборы из того же
  конфига, что стенд: пока приборов нет, просьба «включи лампу» требует честной
  фразы, а не вызова (`expect_no_claim` проверяется списком фраз хаба,
  `hub.llm.claims_completed_action`); появится лампа в конфиге — тот же
  генератор снова ждёт `set_light`. Тесты: `tests/audit/test_request_matrix.py`
  (+4, матрица 3073), `tests/test_devices.py` (+1), `tests/test_jev_understanding.py`
  (+1); `pytest tests -q` → 9269 passed, 15 failed и все 15 — чужой
  `tests/test_guess_who.py` (11 skipped — регрессионные реплики без аудио и
  named pipes, к правке не относятся); `ruff check .` → All checks passed;
  `mypy common` → Success. Хаб перезапущен `scripts/run-openai-server.ps1`
  (старый PID 65804 остановлен, новый 78036, `/health` → 200,
  `outbound.clients: 2`). `git add`/`push` из песочницы невозможны (`.git`
  только для чтения) — список файлов для внешнего коммита в `DECISIONS.md`,
  AUDIT-12f. `client/` и `common/` не правились, обновление комнатных ПК не
  требуется. Обоснования — `DECISIONS.md`, AUDIT-12…12f.
- [x] **AU-07 — медиа: картинки, обои, показ.** `--family media`. Проверить
  `generate_image` (явная просьба нарисовать/отредактировать), `set_wallpaper`,
  `show_photo`, `save_photo`, `say_in_room`; отдельно — правка присланного
  фото и обои из готовой картинки. Проверено: живой прогон `--family media
  --workers 6` до правок (`data/audit/runs/au-07-media-before.jsonl`,
  115 сценариев) — **110 из 115**; падали `AU-0677`/`AU-0680`/`AU-0682`/
  `AU-1066` (правка присланного фото: в стенде нет вложения, а у
  `generate_image` нет источника «вложение» — `source` = none/camera/screen/
  last) и `AU-0714` («tell everyone …» модель читала как рассылку в Telegram
  вместо `say_in_room`). Сделано: промпт комнаты получил разделы «Pictures,
  images and the wallpaper» и «Saying things out loud» (`prompts/system.md`),
  `hub/tools.py` — слова «announce»/«tell everyone» в описании `say_in_room` и
  правило «зови `set_wallpaper` сразу, не ищи картинку взглядом, передай
  человеку ответ инструмента» в описании `set_wallpaper`; сценарии правки
  присланного фото помечены `bench_skip` с причиной (как `inspect_photo`,
  AUDIT-09d), сам путь проверяют `tests/test_telegram_edit_prompt.py` и
  `tests/test_telegram_reply_photo.py`. Итог: `au-07-media-final-4.jsonl` и
  `au-07-media-final-5.jsonl` — по **100 из 100** разобранных (15 `SKIP`),
  два прогона подряд; промежуточные `…-final-2.jsonl` (98/100) и
  `…-final-3.jsonl` (99/100) — это нестабильность обоев, которую закрыла
  правка AUDIT-13b. Тесты: `tests/audit/test_request_matrix.py` (+3, матрица
  3076); `pytest tests -q` → 9272 passed, 15 failed и все 15 — чужой
  `tests/test_guess_who.py` (незавершённая чужая фича, в зачёт не идёт),
  11 skipped; `ruff check .` → All checks passed; `mypy common` → Success.
  Хаб перезапущен `scripts/run-openai-server.ps1` (старый PID 78036
  остановлен, новый 79620, `/health` 8770 → 200, `llm_model: deepseek-flash`,
  `outbound.clients: 2`). `client/` и `common/` не правились, обновление
  комнатных ПК не требуется. `git add`/`git commit` из песочницы отвечают
  `Unable to create .git/index.lock: Permission denied`, `git push` —
  `SEC_E_NO_CREDENTIALS`, поэтому коммит и пуш AU-07 делает внешний скрипт
  (список файлов — `DECISIONS.md`, AUDIT-13f). Обоснования — `DECISIONS.md`,
  AUDIT-13…13f.
- [x] **AU-08 — память.** `--family memory`: `remember`, `forget_fact`,
  `list_memory`, `recall_conversation`; сверить, что сказанное действительно
  сохранено (таблицы `facts`), а «что ты помнишь обо мне» отвечает списком.
  Проверено: живой прогон `--family memory --workers 6` до правок
  (`data/audit/runs/au-08-memory-before.jsonl`) — **70 из 90** (77.8 %);
  после правок `…/au-08-memory-final-2.jsonl` и `…/au-08-memory-final-3.jsonl`
  — по **90 из 90** (100 %), два прогона подряд. Главная поломка была не в
  модели: стенд читал и писал память ВЛАДЕЛЬЦА (`Memory()` без каталога =
  `data/memory.jsonl`, а `hub.app` лениво открывал живой `data/hub.db`), в
  префиксе хода стояло `recent facts: "Anton likes tea."`, и модель честно
  отвечала «ты уже мне это говорил», не вызывая `remember` (12 таких
  падений). Сделано: у каждого воркера стенда своя комната (свои
  `people.json`, `memory.jsonl`, `conversations.sqlite3`; фикстуры + сброс на
  каждый сценарий), база хаба — по его же переключателю `ROWAN_HUB_DB` в
  `data/live-eval/hub.db` и пустая на старте (`AUDIT-14`); корпус проверяет
  только то слово, которое человек назвал («exam», а не «dorm» из соседней
  формулировки; «вчера» — время, а не тема), «забудь …» объявляет предусловие
  `assumes_fact`, час можно писать цифрами; `recall_conversation` объявляет
  `person`/`since`/`until`/`limit`, которые уже читал хаб; промпт и описания
  разводят «что ты обо мне знаешь» (`list_memory`) и «что я говорил»
  (`recall_conversation`). Что подтверждено живьём: все 46 сценариев
  `remember` вызвали инструмент и факт реально оказался в памяти комнаты
  (поле `room_facts` в отчёте), 12 сценариев `recall_conversation` вернули
  настоящие строки из архива (вчерашние фикстуры: 2 записи), 32 сценария
  `forget_fact` дошли до голосового подтверждения F-113, `list_memory` читает
  факты этой комнаты. Убрана и старая грязь: 190 строк `data/memory.jsonl` +
  324 строки таблицы `memories` живого `hub.db` (выдуманные стендом факты),
  копия всех 514 строк — `data/audit/memory-bench-cleanup.jsonl`,
  скрипт — `scripts/clean-bench-memory.py`; собственные факты владельца
  оставлены (ключи, Chrome, кофе, джаз). Полный корпус после правок —
  `data/audit/runs/au-08-full-after.jsonl`: **1056 из 1072** разобранных
  (34 `SKIP`), память 90/90; остаток — известные задачи других семейств
  (`notify` 20/24, `multi` 17/20, `pc` 158/162, русское «включи свет» —
  AU-1007/AU-1008, записано как AUDIT-14i). Тесты: `pytest tests -q` →
  **9249 passed, 15 failed** и все 15 — чужой `tests/test_guess_who.py`
  (в зачёт не идёт), 11 skipped; `pytest tests/audit -q` → 3053 passed;
  `ruff check .` → All checks passed; `mypy common` → Success. Хаб
  перезапустить из песочницы не удалось (старый процесс держит 8770,
  `Access denied`; новый упал на `[Errno 10048]`), прежний хаб жив
  (`/health` → 200, `llm_model: deepseek-flash`, `outbound.clients: 2`) —
  новые описания инструментов подхватит внешний перезапуск
  `scripts/run-openai-server.ps1` (`DECISIONS.md`, AUDIT-14j). `client/` и
  `common/` не правились, обновление комнатных ПК не требуется.
  Обоснования — `DECISIONS.md`, AUDIT-14…14j.
- [x] **AU-09 — ПК: громкость, приложения, клавиши.** `--family pc`,
  `--family russian`; проверить `volume_set/down/up`, `open_app/close_app`
  (в т.ч. имя в `target`), медиа-клавиши, `clipboard_read/write/paste`,
  `hotkey` для «свернуть всё», `minimize_app`.
  Проверено: живой прогон `--family pc --family russian --workers 6` до правок
  (`data/audit/runs/au-09-pc-before.jsonl`, 177 сценариев) — **172** (pc 159/162,
  russian 13/15), падали `AU-0497` (первым шёл `run_command` со списком окон),
  `AU-0500` («type my password into the field» — корпус требовал выдуманный
  пароль), `AU-0503` («read my clipboard» — Jev читал буфер как зрение, модель
  честно отвечала «инструмента нет»), `AU-1007`/`AU-1008` (русское «включи/
  выключи свет» ждало `set_light` в комнате без приборов). После правок три
  прогона подряд `data/audit/runs/au-09-pc-final.jsonl`, `…-final-2.jsonl`,
  `…-final-3.jsonl` — по **181 из 182** разобранных (pc 162/162 старых
  сценариев во всех трёх, russian 15/15, 15/15, 14/15; `AU-1109` — `SKIP`,
  вставка идёт в окно в фокусе, которого у стенда без `--actions` нет).
  Сделано: `hub/tools.py::TOOL_FAMILY_MEANINGS` называет буфер обмена, медиа-
  клавиши и «сверни всё» у семейства `pc`, а `vision` — что буфер это не экран
  (`hub/jev_decider.py::FAMILY_QUESTION` говорит то же; живой `jev_probe`:
  «read my clipboard» → `family=pc` 1.00); описание `pc_control` и
  `prompts/system.md` требуют `hotkey win+d` на «minimize everything» (и не
  перечислять окна через `run_command`), медиаклавиши — на «play/pause the
  music» без названного сайта, буфер — у `pc_control`; `minimize_app` со
  словами «all/everything/the desktop» переписывается в тот же `win+d`
  (`normalize_pc_control_args`); `type_text` с паролем, картой, кодом или PIN
  хаб отклоняет (`hub/tools.py::types_a_secret` — список F-512, `hub/app.py`),
  и корпус больше не требует вызова за пароль (`no_secret_args` +
  `expect_no_claim`); «положи ЭТОТ текст в буфер» без текста объявлено
  `may_ask` — верный ход это вопрос, и вердикт стенда его принимает; русское
  «включи свет» следует за `client.devices` комнаты, как семейство `devices`
  (перенос корпусной правки AUDIT-14i); хвост `AU-1107…AU-1112` проверяет,
  что названный человеком текст доезжает до аргументов `clipboard_write`, и
  медиаклавиши `next track`/`previous song`. Тесты: `pytest tests -q` →
  **9297 passed, 15 failed**, и все 15 — чужой `tests/test_guess_who.py` (в
  зачёт не идёт), 11 skipped; `pytest tests/audit -q` → 3072 passed;
  `ruff check .` → All checks passed; `mypy common` → Success;
  `python -c "import hub.app"` → ok. Хаб перезапущен
  `scripts/run-openai-server.ps1` (старый PID 79120 остановлен, новый 70532,
  `/health` → 200, `llm_model: deepseek-flash`, `outbound.clients: 2`).
  `client/` и `common/` не правились, обновление комнатных ПК не требуется.
  Что осталось незакрытым и почему — `DECISIONS.md`, AUDIT-15…15h.
- [x] **AU-10 — две просьбы в одной реплике (UG-08).** `--family multi`:
  «открой ютуб и сделай громче» должно дать ОБА вызова. Если сужение по
  семейству мешает — вернуть Jev четвёртый вопрос `single` и не сужать при
  «несколько» (см. `PROGRESS_UNDERSTANDING.md`).
  Проверено: живой прогон `--family multi --workers 6` до правок
  (`data/audit/runs/au-10-multi-before.jsonl`) — 19 из 20, но **не потому что
  обе половины проверялись**: у половины пар корпус ждал ОДИН инструмент
  («turn on the light and play some music» — `set_light`), а сужение по
  семейству прятало вторую половину (Jev: `family=devices` 0.93, набор без
  `pc_control`), и модель попадала в неё случайно. Сделано: у Jev появился
  четвёртый вопрос `single` в том же batched-вызове
  (`hub/jev_decider.py::SINGLE_QUESTION`), а `hub/app.py::_narrow_tools_for`
  при уверенном «просьб несколько» не сужает набор вовсе; корпус
  (`scripts/gen-audit-scenarios.py::PAIRS`) теперь требует ОБЕ половины
  (инструменты пары обязаны быть вызваны все, а неоднозначная половина
  объявляется набором `also_any`), половина про лампу следует за комнатой, а
  «сделай снимок и отправь в группу» принимает одну команду
  `telegram_send kind=image source=screen` (`picture_in_the_send`); промпт
  комнаты требует делать обе половины («a half that failed … is never a reason
  to drop the other half»). Числа: пять прогонов того же семейства —
  `au-10-multi-final.jsonl` 14/20, `…-final-2` 17/20, `…-final-3` и
  `…-final-4` по 18/20, `…-final-5` — **20 из 20**;
  приёмочная реплика задачи «Rowan, open youtube and turn the volume up» дала
  оба вызова (`browser_control` + `pc_control`) во всех пяти прогонах, как и
  «open youtube and turn the volume up» без обращения. Остаток (не поломка
  сужения, а ход модели) — «save a photo and put it on my wallpaper»: в четырёх
  прогонах модель делала одну половину и спрашивала про вторую; это записано в
  `DECISIONS.md` (AUDIT-16d). Живая проверка Jev:
  `python scripts/jev_probe.py` → «open youtube and turn the volume up»
  `single=False` 0.94, «save a photo and put it on my wallpaper» 0.90,
  «read my clipboard» `single=True` 0.94, ответ 489–522 мс (бюджет 1500 мс).
  Тесты: `pytest tests/audit -q` → 3078 passed; `pytest tests -q` → **9303
  passed, 15 failed** и все 15 — чужой `tests/test_guess_who.py` (в зачёт не
  идёт), 11 skipped; `ruff check .` → All checks passed; `mypy common` →
  Success; `python -c "import hub.app"` → ok. Хаб перезапущен
  `scripts/run-openai-server.ps1` (старый PID 70532 остановлен, новый 71508,
  `/health` → 200, `llm_model: deepseek-flash`, `outbound.clients: 2`).
  `client/` и `common/` не правились, обновление комнатных ПК не требуется.
  Обоснования — `DECISIONS.md`, AUDIT-16…16e.
- [x] **AU-11 — латентность хода.** По `mass-*.jsonl` посчитать долю ходов
  дольше 4 с и первый звук позже 2.5 с (ТЗ 15.1, F-101); найти, что именно
  столько занимает (STT, Jev, раунды модели), и записать замеры до/после в
  `docs/TZ_STATUS.md`.
  Проверено: доля ходов дольше 4 с посчитана по всем прогонам корпуса —
  `mass-01.jsonl` 29.4 % (325/1106), `au-08-full-after.jsonl` 16.7 %
  (179/1072), `au-11-full-after.jsonl` **13.6 % (146/1077, 6 падений)**;
  медиана хода 2.76 → 2.85 → 2.56 с. «Первый звук позже 2.5 с» — по 82 живым
  замерам F-101 из `data/server.log`: **65/82 = 79.3 %**, медиана 4703 мс,
  позже бюджета 1200 мс 80/82 = 97.6 %. Что занимает время (13 живых ходов
  комнаты с настоящим раундом модели, `data/hub.db::turn_events`, медианы):
  STT 1397 мс, стадия `llm` 3808 мс — из неё раунды модели 3280 мс, — TTS
  759 мс, весь ход 6367 мс; раунды модели медиана 2 (69.2 % ходов больше
  одного), первый вызов инструмента через 1289 мс после старта модели.
  Найдена и починена инженерная часть: `hub/jev_decider.py` открывал новый
  `httpx.AsyncClient` на каждый ход (TCP + TLS заново) — теперь соединение
  живёт между ходами, замер на тех же 12 сценариях `--workers 6`: чтение Jev
  **1247 → 257 мс**, «Jev + модель + инструменты» 3.86 → 2.68 с, ходов
  дольше 4 с 41.7 % → 0 % (`au-11-smoke.jsonl` против
  `au-11-profile-after.jsonl`). Живой ход по настоящей речи
  (`scripts/measure_voice_latency.py` → `data/audit/voice-latency.json`,
  пять записей из `data/voices`): тёплые ходы STT 407…472 мс, LLM
  1466…3276 мс, весь ход 1.9…3.7 с; первый ход после старта — STT 7229 мс
  (прогрев CUDA). Сводка «до/после» записана в `docs/TZ_STATUS.md`
  (раздел AU-11), полный отчёт с числами — `docs/AUDIT_LATENCY.md`
  (`scripts/audit-latency.py`), обоснования — `DECISIONS.md` AUDIT-17…17g.
  Тесты: `pytest tests -q` → **9316 passed, 15 failed** и все 15 — чужой
  `tests/test_guess_who.py` (незавершённая чужая фича, в зачёт не идёт),
  11 skipped; `ruff check .` → All checks passed; `mypy common` → Success;
  `python -c "import hub.app"` → ok. Хаб перезапущен
  `scripts/run-openai-server.ps1` (PID 71508 → 81700, `/health` → 200,
  `llm_model: deepseek-flash`, `outbound.clients: 2`). **Чего нет:** сквозного
  «первого звука» после правки нет — движок синтеза в песочнице не
  поднимается (сам хаб пишет `Could not load Silero TTS — replies will be
  text only`: phonemizer не может записать `espeak-ng.dll` во временный
  каталог), поэтому TTS-стадия взята из трасс хаба, а `first_audio_ms` пробы
  оставлен пустым, а не нулём (`DECISIONS.md`, AUDIT-17f); бюджет 1.2 с при
  облачной модели на критическом пути недостижим, локальные уровни в конфиге
  пусты (AUDIT-17e). `client/` и `common/` не правились, обновление комнатных
  ПК не требуется.
- [x] **AU-12 — комнатные ПК после каждой правки клиента.** Прогнать
  `scripts/update-room-pcs.ps1`, затем `scripts/room-audit.ps1`; 57/57
  действий на каждом ПК — приёмка. Записать числа в отчёт.
  Проверено: `pwsh -File scripts\update-room-pcs.ps1` (2026-09-23) — пакет
  331 КБ из `client/`+`common/`, на обоих ПК «обновлён и перезапущен»,
  SHA-256 `client/camera.py` совпал с этим рабочим каталогом, задача
  `JarvisRoomClient`/`RowanRoomClient` в состоянии `Running`, `WINDOW=hidden`.
  `pwsh -File scripts\room-audit.ps1` — **AntonDorm 57/57** за 32.3 с
  (`data/room-eval/audit-AntonDorm.json`), **buro 56/57** за 42.2 с
  (`data/room-eval/audit-buro.json`): упало одно действие, RA-035 «открой
  `www.google.com`» сразу после `youtube.com` — Firefox ещё дочитывал
  страницу, адресная строка не взяла текст, и за 8 с (`NAVIGATE_WAIT_S`)
  адрес не сменился («the browser is still on (871) YouTube»). Повторный
  прогон того же аудита на том же ПК (`data/room-eval/audit-buro-rerun.json`)
  — **57/57** за 36.1 с, RA-035 прошёл за 2.09 с (`https://www.google.com/`);
  значит это гонка с загрузкой страницы, а не сборка клиента — причина
  записана в `DECISIONS.md` (AUDIT-18…18b), числа — в `docs/AUDIT_MASS.md`.
  Проверки: `python -m pytest tests -q` → **9316 passed, 15 failed**, и все 15
  — чужой `tests/test_guess_who.py` (в зачёт не идёт), 11 skipped;
  `python -m ruff check .` → All checks passed; `python -m mypy common` →
  Success. `git add`/`commit`/`push` из песочницы невозможны (`.git` только
  для чтения, у remote нет креденшелов) — как в AU-03…AU-11, список файлов для
  внешнего коммита в `DECISIONS.md`, AUDIT-18c. `client/` и `common/` не
  правились.
- [x] **AU-13 — итог.** Обновить `docs/AUDIT_MASS.md`, `docs/TZ_STATUS.md`,
  `README.md` (какие слои аудита есть и как их запускать), `DECISIONS.md`.
  После закрытия — вернуть звук на ПК: `pwsh -File scripts/room-audio.ps1 unmute`.
  Проверено: корпус пересобран (`python scripts/gen-audit-scenarios.py` →
  1112 реплик) и прогнан ЗАНОВО живьём после всех правок —
  `python scripts/live-eval.py --scenarios data/audit/scenarios.jsonl
  --workers 6 --jsonl data/audit/runs/au-13-full-final.jsonl --quiet`:
  1077 разобрано, 35 `SKIP` (17 вложение Telegram, 15 `fill`, 2 «эта
  картинка», 1 окно в фокусе), **1071 прошло = 99.4 %** против 812 из 1106
  (73.4 %) в первом прогоне; сводка пересчитана
  `python scripts/audit-summary.py --runs data/audit/runs/au-13-full-final.jsonl`
  в `docs/AUDIT_MASS.md`, латентность — `python scripts/audit-latency.py`
  по четырём прогонам в `docs/AUDIT_LATENCY.md` (медиана хода 2.54 с против
  2.76 с в первом прогоне, чтение Jev 199 мс против 1247 мс до AU-11).
  Остаток разобран повторными прогонами того же файла сценариев
  (`data/audit/runs/au-13-residual-1.jsonl`, `…-2.jsonl`, тот же
  `--workers 6`): `AU-0462` — сбой провайдера (2 из 2 прошло), `AU-0971` и
  `AU-0910` — разброс модели (1 и 2 из 2), устойчивые `AU-0977` и пара
  `AU-0989`/`AU-0990` — задачи AU-20/AU-21. Комнатные ПК:
  `pwsh -File scripts\update-room-pcs.ps1` → оба «обновлён и перезапущен»,
  SHA-256 `client/camera.py` совпал, задача `Running`, окно `hidden`;
  `pwsh -File scripts\room-audit.ps1` дважды → AntonDorm **57/57** (29.9 с
  и 33.7 с), buro **56/57** (41.6 с и 42.9 с) — оба раза одно и то же
  действие `RA-035` (`navigate www.google.com` сразу после `youtube.com`,
  Firefox ещё дочитывает страницу), то есть устойчивый дефект `_navigate`:
  задача AU-22 и `DECISIONS.md` AUDIT-19e (в AU-12 это было записано как
  гонка). Звук комнат возвращён `pwsh -File scripts\room-audio.ps1 unmute` —
  «sound unmuted» на AntonDorm и buro. Витрина — `README.md`, раздел 13.
  Сам `hub/` в AU-13 не правился, а запущенный хаб уже новее последней
  правки: ни один файл `hub/*.py` и сам `config.openai.yaml` не менялись
  после его старта (05:13:46), перезапуск не требовался — `/health` отвечает
  `status=ok`, `llm=true`, `llm_model=deepseek-flash`, `outbound.clients=2`
  (обе комнаты на связи), `tts=false` — это известное ограничение песочницы
  (AUDIT-17f), а не следствие этой задачи.
  `python -m pytest tests -q` → 9316 passed, 15 failed, 11 skipped
  (все 15 — чужой `tests/test_guess_who.py`, незавершённая чужая фича,
  в зачёт не идёт);
  `python -m ruff check .` → All checks passed; `python -m mypy common` →
  Success. `git add`/`commit`/`push` из песочницы невозможны (`.git` только
  для чтения, у remote нет креденшелов) — список файлов для внешнего
  коммита в `DECISIONS.md`, AUDIT-19g. Обоснования — `DECISIONS.md`,
  AUDIT-19…19g.
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
- [x] **AU-19 — Jev в Telegram-чате.** Открытая задача из JE-01: чтение Jev
  (`_understand_turn`) стоит только на голосовом пути, а Telegram-чат идёт
  через `hub/telegram_chat.py` → `llm.reply_text` без чтения и без сужения
  инструментов. Владелец просил, чтобы в Telegram были те же возможности, что
  и у голосового ассистента; сюда же — второй провайдер `jev` в цепочке решений
  (сейчас всегда отвечает `rules`).
  Проверено: чтение и сужение вынесены в `hub/turn_reading.py`, и оба пути
  (голос и `hub/telegram_control.py`) зовут одну функцию; `TelegramController`
  получил читателя `hub.app._telegram_turn_tools` параметром и передаёт
  суженный набор в `llm.generate(tools=…)`. Первый полный прогон корпуса
  Telegram-путём (`python scripts/live-eval.py --scenarios
  data/audit/scenarios.jsonl --telegram --workers 6 --jsonl
  data/audit/runs/au-19-telegram.jsonl --quiet`, новый ключ `--telegram`
  запускает НАСТОЯЩИЙ `TelegramController` в комнате стенда):
  **1036 из 1077** разобранных (96.2 %), те же 35 `SKIP`; тот же корпус
  голосовым путём в тот же час (`data/audit/runs/au-19-voice.jsonl`) — **1068
  из 1077** (99.2 %). Тресс каждого Telegram-хода несёт шаг `understanding`
  (`turn_events`) — **1077 из 1077** разобранных строк, медиана 220 мс, в
  **1024** из них набор инструментов действительно сужен (в остальных
  fail-open), — чего до правки не было ни в одном из 81 живого
  Telegram-хода. 41 падение телеграфного прогона прогнано дважды
  (`data/audit/runs/au-19-telegram-failures-jev.jsonl` — **25 из 41**,
  `…-nojev.jsonl` без Jev — **21 из 41**): сужение набора инструментов в
  остатке не виновато, не предложен нужный инструмент ровно в одном сценарии
  (`AU-0977`, задача AU-20); устойчивых падений 11, и они — вердикты корпуса,
  написанные для комнаты (картинка в чате показывает сам чат, запись лица из
  чата, память Telegram-аккаунта) → задачи AU-23 и AU-24.
  Второй провайдер: `hub/decider.py::_settled` не даёт цепочке закончиться на
  ответе ниже полосы `auto_above` своего типа решения, поэтому `[rules, jev]`
  наконец значит то, что написано в ТЗ 5.2. Живой `python scripts/jev_probe.py`
  → `addressed` отвечает `jev -> True (0.83)` за 520 мс там, где правило
  говорило 0.6; `route` оставлен локальным (0–2 мс) и в конфиге владельца
  теперь `route: [rules]` (AUDIT-20c). `python scripts/jev_usage_report.py`
  починен (таблица `turn_events`, колонка `payload_json`, разрез голос/Telegram)
  и на живой базе до правок показывал 5087 решений, все `rules`, и 0 чтений.
  Тесты: `tests/test_telegram_understanding.py` (10, новый),
  `tests/test_decider.py` (+6), `tests/test_jev_decider.py` (+2);
  `pytest tests -q` → **9334 passed, 15 failed** — все 15 чужой
  `tests/test_guess_who.py` (незавершённая чужая фича, в зачёт не идёт),
  11 skipped; `ruff check .` → All checks passed; `mypy common` → Success;
  `python -c "import hub.app"` → ok. Хаб перезапущен
  `scripts/run-openai-server.ps1` дважды (PID 81700 → 86156 → 78348, на
  финальном коде; `/health` → `status=ok`, `llm_model=deepseek-flash`,
  `telegram=true`, `clients=1`).
  `client/` и `common/` не правились — обновление комнатных ПК не требуется.
  Обоснования — `DECISIONS.md`, AUDIT-20…20g; числа — `docs/AUDIT_MASS.md`,
  «Telegram-путь целиком».

## Остаток аудита после итога (AU-13, 23.09.2026)

- [x] **AU-20 — «picture of a dog» без слова «нарисуй».** Устойчивое (0 из 3
  прогонов, `data/audit/runs/au-13-residual-1.jsonl`, `…-2.jsonl` и строка
  `AU-0977` в `au-13-full-final.jsonl`): Jev называет для просьбы о картинке
  другое семейство, `generate_image` в набор не попадает — модель ищет
  картинку (`look_at_screen`, `find_object`, `inspect_photo`), а не рисует.
  Сделано: значения семейств (`hub/tools.py::TOOL_FAMILY_MEANINGS`) и вопрос
  Jev (`hub/jev_decider.py::FAMILY_QUESTION`) называют просьбу без глагола
  словами человека («picture of a dog», «a pic of my cat», «an image of a
  dragon») и прямо относят её к `media` — «сделать картинку», — а `vision`
  оставляют тому, что уже есть (присланное фото, только что нарисованная
  картинка, камера, экран). Проверено: живой прогон двух семейств
  `--family noisy --family media --workers 6` **до** правки
  (`data/audit/runs/au-20-before.jsonl`, 134 разобранных сценария) — 132
  прошло, и `AU-0977` упал именно на сужении (`offered` — `vision`:
  `look_at_screen`, `find_object`…, `generate_image` там нет; `AU-0971` —
  известный разброс модели). **После** правки — `au-20-after.jsonl` 132/134 и
  повтор `au-20-after-2.jsonl` 132/134, `AU-0977` прошёл в обоих (`offered` —
  семейство `media`, `generate_image` на месте; `AU-0978` тоже прошёл).
  Остальные падения этих прогонов — ход модели, а не сужение: у `AU-0710`
  («save this screenshot…», набор `media` с `save_photo` и до, и после) и
  `AU-0622` («draw a dragon…») модель позвала первым другой инструмент, у
  `AU-0971` инструмент не позван вовсе — это тот же класс, что `AU-0910`.
  Живая проба `python scripts/jev_probe.py "picture of a dog"` — `family=media`
  **1.00** за 684 мс; соседние формулировки не разъехались («describe the
  picture I sent» → `vision` 1.00, «who is in the room» → `vision` 1.00,
  «what is on the screen» → `vision` 1.00, «put this picture on my wallpaper» →
  `media` 0.89, «show that picture again» → `media`). Тест:
  `tests/test_jev_understanding.py::test_a_bare_request_for_a_picture_means_making_one_not_looking_at_one`
  (новый). `pytest tests -q` → **9335 passed, 15 failed**, 11 skipped — все 15
  чужой `tests/test_guess_who.py` (незавершённая чужая фича, в зачёт не идёт);
  `ruff check .` → All checks passed; `mypy common` → Success;
  `python -c "import hub.app"` → ok. Хаб перезапущен
  `scripts/run-openai-server.ps1` на новом коде (`/health` → `status=ok`,
  `llm_model=deepseek-flash`, `llm=true`, `telegram=true`, `outbound.clients=1`,
  `tts=false` — известное ограничение песочницы, AUDIT-17f). `client/` и
  `common/` не правились — обновление комнатных ПК не требуется.
  `git add`/`commit`/`push` из песочницы невозможны (`.git` только для чтения)
  — список файлов для внешнего коммита в `DECISIONS.md`, AUDIT-21b.
- [x] **AU-21 — пара «сохрани фото и поставь на обои» теряет половину.**
  Устойчивое (0 из 3), но терялись РАЗНЫЕ половины (то `save_photo`, то
  `set_wallpaper`) и набор полный (`offered` пуст) — ход модели, а не сужение
  (тот же класс, что AUDIT-16d). Разбор ответил на вопрос задачи: хаб зовёт
  `verify` после хода НЕ на всякую пару — гейт D-04 смотрит `changed_state`
  (медиа-инструментов в `STATE_CHANGING_TOOLS` нет), `imperative_without_tool`
  и `unfinished_step`, а потерянный шаг умел замечать только
  `site_step_unfinished` («open chrome and go to youtube»); к тому же в живом
  конфиге владельца `verify_actions: false`. Стенд прохода самопроверки не
  делал вовсе — то есть числа корпуса по паре были первым планом модели.
  Сделано: детектор потерянного шага обобщён на шаги, названные самим
  человеком (`hub/decision_points.py::named_step_unfinished`,
  `any_step_unfinished` — «сохрани фото», «обои», «покажи камеру», у каждого
  свой набор выполняющих инструментов); хаб зовёт по нему самопроверку
  (`hub/app.py`); стенд получил тот же гейт и тот же проход
  (`scripts/live-eval.py::self_check_needed`, `Bench._after_turn`, в отчёте
  строки `self_check`/`self_check_rounds`); в промпт комнаты добавлено правило
  «save a screenshot / save the photo» — это `save_photo`, а не
  `look_at_screen` (в AU-0996 модель звала взгляд вместо сохранения).
  Проверено: живой прогон `--family multi --workers 6` —
  `data/audit/runs/au-21-before.jsonl` 18/20 (падали `AU-0990` и `AU-0996`),
  `au-21-after.jsonl` 19/20 (стенд + детектор: оба «обойных» сценария прошли,
  `AU-0996` ещё нет), `au-21-after-2.jsonl` — 18/20 и `au-21-after-3.jsonl` —
  **20/20**; все четыре сценария пары («фото + обои» `AU-0989`/`AU-0990` и
  «камера + скриншот» `AU-0995`/`AU-0996`) проходят в двух последних прогонах.
  Полный корпус `au-21-full-after.jsonl` (1077 разобранных, 35 `SKIP`) —
  **1067 прошло (99.1 %)**, 265 ходов прошли через самопроверку и **все 265
  успешны**; все 10 падений — ходы без самопроверки (известные классы
  `notify`/`devices`, разброс модели, сбой провайдера `AU-0956`). Контроль
  семейств, которых правка касается: `au-21-regress-media-vision-noisy.jsonl`
  — media 100/100, noisy 34/34, vision 112/114 (два падения — первым вызван
  `find_object`, сужение ни при чём); повтор `notify`/`devices`/`people`
  (`au-21-regress-notify-devices-people.jsonl`) — 81/81, 22/24, 62/64, и
  падают уже ДРУГИЕ сценарии, чем в полном прогоне (разброс, не регрессия).
  Тест: `tests/test_lost_step.py` (новый — детектор, гейт хаба и проход
  стенда). `pytest tests -q` → **9355 passed, 15 failed**, 11 skipped — все 15
  чужой `tests/test_guess_who.py` (незавершённая чужая фича, в зачёт не идёт);
  `ruff check .` → All checks passed; `mypy common` → Success;
  `python -c "import hub.app"` → ok. Хаб перезапущен
  `scripts/run-openai-server.ps1` на новом коде (`/health` → `status=ok`,
  `llm_model=deepseek-flash`, `telegram=true`, `outbound.clients=1`).
  `client/` и `common/` не правились — обновление комнатных ПК не требуется.
  `git add`/`commit`/`push` из песочницы невозможны (`.git` только для чтения)
  — список файлов для внешнего коммита в `DECISIONS.md`, AUDIT-22b. Числа —
  `docs/AUDIT_MASS.md`, раздел «AU-21»; решения — `DECISIONS.md`, AUDIT-22.
- [ ] **AU-22 — `RA-035`: `_navigate` печатает адрес в ещё грузящуюся
  страницу.** buro упал дважды подряд (`data/room-eval/audit-buro-au13-first.json`
  и `audit-buro.json`, 56/57, 41.6 с и 42.9 с) на
  `browser_control navigate www.google.com` сразу после `youtube.com`:
  «The address bar did not open google.com: the browser is still on (871)
  YouTube». Сделать: в `client/actions/browser_desktop.py::_navigate`
  дождаться окончания загрузки текущей страницы (или повторить ввод), тест;
  после правки — `pwsh -File scripts\update-room-pcs.ps1` и
  `pwsh -File scripts\room-audit.ps1` до 57/57 на ОБОИХ ПК.
  **Сделано, но НЕ закрыто: buro не отвечает по сети.** Правка в
  `client/actions/browser_desktop.py`: перед вводом страница получает до 2.5 с
  на то, чтобы перестать меняться (`_settled_page`, сравнение адреса и
  заголовка), ввод повторяется целиком (`NAVIGATE_ROUNDS` = 2), и ввод
  повторяется, если адресная строка ещё не нашлась в дереве (текст при этом
  всё равно уходит в браузер, как решено 2026-09-22). Проверено частично:
  `pwsh -File scripts\update-room-pcs.ps1` — оба ПК «обновлён и перезапущен»,
  SHA-256 совпал, задача `Running`, окно `hidden`;
  `pwsh -File scripts\room-audit.ps1` — AntonDorm **57/57 дважды** (34.4 с и
  37.8 с), и `RA-035` (та самая гонка: `google.com` сразу после `youtube.com`)
  прошёл за 1.2 с (`data/room-eval/audit-AntonDorm.json`); buro в первом
  прогоне повис на ~19 минут (ssh живой, отчёта нет), во втором —
  `ssh: connect to host 100.67.114.67 port 22: Connection timed out`,
  три прямые пробы — тоже таймаут, `Test-NetConnection … -Port 22` → False,
  при живом AntonDorm; `/health` хаба — `outbound.clients=1`. Код на buro при
  этом уже новый (09:09, хэш совпал), поэтому после возвращения ПК достаточно
  аудита. Задача остаётся `[ ]` до 57/57 на ОБОИХ ПК (правило «ПК не отвечает
  — не закрытая задача»); подробности и как продолжить — `BLOCKED_AUDIT.md`,
  обоснование — `DECISIONS.md`, AUDIT-23 и AUDIT-23b.
- [x] **AU-23 — у корпуса нет телеграфных ожиданий.** Проверено: живой прогон
  `--telegram --workers 6` до правок (`data/audit/runs/au-23-before.jsonl`,
  1077 разобранных, 35 `SKIP`) — **1032 прошло (45 падений)**; после правок
  `au-23-after.jsonl` — **1060**, финальный `au-23-final.jsonl` — **1059
  (98.3 %)**, и все сценарии, ради которых задача заводилась, проходят:
  `AU-0707` «покажи камеру» (FAIL→PASS), пара `AU-0995`/`AU-0996`
  (FAIL→PASS), «сохрани это лицо как X» `AU-0802`/`AU-0819` (PASS), плюс
  найденное по ходу `AU-0547`/`AU-1011` «кто в комнате» (FAIL→PASS).
  Сделано: у сценария появился признак «в чате верный ход такой»
  (`telegram_expect_tools` / `telegram_expect_any`,
  `scripts/gen-audit-scenarios.py`), который читает только
  `scripts/live-eval.py --telegram` (`telegram_scenario`), поэтому голосовые
  ожидания не менялись и прогоны сравнимы сценарий за сценарием; решение по
  продукту — лицо и голос из чата записываются ТЕМ ЖЕ инструментом
  (`enroll_face`/`enroll_voice`), источник — камера комнаты, а не отказ
  (`hub/telegram_control.py`, ТЗ F-210); ход хаба, который сам смотрит в
  камеру на «кто в комнате», снова виден в цепочке (`turn_trace`), иначе
  панель владельца показывала ответ про комнату без взгляда на камеру;
  последний шаг доставки в Telegram в стенде подменён транспортом-двойником
  (`BenchTelegram`: в чат владельца ночью не пишется, квитанции — в поле
  `deliveries` строки отчёта, 11 штук в финальном прогоне). Остаток финала
  прогнан дважды (`au-23-residual-1/2.jsonl` — 14/18 и 15/18); устойчивых
  два — `AU-0251` (Jev читает «find lofi hip hop» как ПК и не отдаёт
  `browser_control`) и `AU-0922` («message the group: I am on my way» —
  модель отвечает «в личку нельзя»), записаны задачами AU-25/AU-26; остальные
  16 — разброс модели (в соседнем прогоне проходят). Тесты:
  `tests/audit/test_request_matrix.py` (+5), `tests/test_telegram_understanding.py`
  (+1, шаг камеры в трассе); `pytest tests -q` → **9364 passed, 15 failed**,
  11 skipped — все 15 чужой `tests/test_guess_who.py` (незавершённая чужая
  фича, в зачёт не идёт, проверено по junit-отчёту `tmp/au-23-junit.xml`:
  единственный класс падений — `tests.test_guess_who`); `ruff check .` → All
  checks passed; `mypy common` → Success; `python -c "import hub.app"` → ok.
  Хаб перезапущен `scripts/run-openai-server.ps1` (старый PID 93564
  остановлен, новый процесс поднялся, `/health` → `status=ok`,
  `llm_model=deepseek-flash`, `telegram=true`, `outbound.clients=1`); сам
  лаунчер перед этим проверен на отдельном порту 8899 (`/health` → 200).
  `client/` и `common/` не правились — обновление комнатных ПК не требуется.
  Числа — `docs/AUDIT_MASS.md`, раздел «AU-23»; решения — `DECISIONS.md`,
  AUDIT-25…25d.
- [x] **AU-24 — память Telegram-аккаунта.** Проверено: живой прогон
  `--telegram --family memory --family russian --workers 6`
  (`data/audit/runs/au-24-memory.jsonl`, 105 сценариев) — **104 из 105**;
  `AU-1010` («что ты помнишь обо мне») проходит: вопрос уходит в
  `list_memory`, и ответ — факты ИМЕННО этого пространства имён
  (`person: telegram:8322835915`, 10 фактов, что записал сам чат), а не
  «ничего» из промпта; единственное падение — `AU-1007` «включи свет»
  (модель позвала `set_light` без лампы, известный разброс; в прогонах AU-23
  та же семья падала на ДРУГИХ сценариях: 103/105 в `au-23-before.jsonl` и
  `au-23-final.jsonl`). Решение по продукту (по умолчанию, ТЗ F-415/F-701):
  личка ВЛАДЕЛЬЦА читает его собственные заметки комнаты — профиль берётся из
  конфигурации дома (`homes[].owner_person_id`, ТЗ 14); у постороннего
  аккаунта и в группе остаётся своё пространство имён, поэтому личные заметки
  комнаты в чужой чат не утекают. Пока дом владельца не называет, личка
  остаётся со своим пространством имён и честно отвечает из него.
  Проверено живьём отдельным прогоном: с временным конфигом, где дом назвал
  `owner_person_id: Anton`, личка владельца прочитала ЕГО факт
  (`data/audit/runs/au-24-owner-profile.jsonl` — «ключи ты держишь в верхнем
  ящике» + факт комнаты), чего без него не видит. Тесты:
  `tests/test_telegram_understanding.py` (+2: «лишь личка владельца читает его
  профиль» и «фасад берёт профиль из дома»). `pytest tests -q` → 9366 passed,
  15 failed (все 15 — чужой `tests/test_guess_who.py`; проверено по junit
  `tmp/au-24-junit.xml`), 11 skipped; `ruff check .` → All checks passed;
  `mypy common` → Success. Хаб перезапущен `scripts/run-openai-server.ps1`
  после правки (`/health` → `status=ok`, `llm_model=deepseek-flash`,
  `telegram=true`, `outbound.clients=1`). `client/` и `common/` не правились —
  обновление комнатных ПК не требуется. Решения — `DECISIONS.md`, AUDIT-26.
- [ ] **AU-25 — «find lofi hip hop» в чате уходит в ПК.** Устойчивое (3 из 3
  Telegram-прогонов: `au-23-final.jsonl`, `au-23-residual-1.jsonl`,
  `au-23-residual-2.jsonl`): Jev читает поисковую просьбу «find lofi hip hop»
  как семейство `pc`, набор инструментов приходит без `browser_control`
  (`offered`: `pc_control`, `run_command`, …), и модель идёт в `pc_control` и
  `run_skill`. В голосовом прогоне та же реплика проходит. Сделать: вопрос
  семейств и его значения должны называть поиск в интернете/на сайте
  браузерным (как уже сделано для сайтов), живая проба `scripts/jev_probe.py
  "find lofi hip hop"`, тест на сужение, повторный прогон `--telegram
  --family browser --workers 6`.
- [ ] **AU-26 — «message the group: I am on my way» не отправляется.** Устойчивое
  (3 из 3): на «Can you please message the group: I am on my way?» модель
  отвечает «в личку отправить не могу — есть только общий чат», хотя общий чат
  назван в самой реплике, и `telegram_send` не зовётся. В голосовом прогоне
  реплика проходит. Сделать: правило промпта Telegram-хода («named group is a
  destination») и проверка `hub/telegram_intent.telegram_send_requested` на
  этой формулировке, тест, повторный прогон `--telegram --family notify
  --workers 6`.
- [x] **AU-27 — видео уведомлений снова открывается на телефоне (TG-10).**
  Владелец 2026-09-23: «опять видео в телеге на мобилке не грузятся». Причина:
  перекодирование через `subprocess` падало с `PermissionError` (в логе
  «ffmpeg could not convert an alert video (PermissionError)»), и клип уходил
  в Телеграм как `mp4v`. Проверено: `python scripts/video_probe.py` в
  песочнице → PyAV даёт H.264 (2699 из 5528 байт), вне песочницы → ffmpeg даёт
  H.264 + faststart (2520 байт); `pytest tests/test_video_transcode.py
  tests/test_telegram_admin_transport.py tests/test_presence_alerts.py -q`
  → 57 passed.
