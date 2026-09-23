# Статус выполнения ТЗ

Обновляется по мере работы. Статус ставится только по факту проверки:
«готово» означает, что код есть и тесты/проверки зелёные.

**Фаза 0 — подготовка (идёт):**

| Пункт ТЗ | Статус | Доказательство |
|---|---|---|
| Структура `hub/`, `client/`, `common/`, `skills/`, `migrations/` | готово | `server/` → `hub/`, `pytest tests` зелёный |
| Pydantic-схемы протокола v2 + совместимость v1 | готово | `common/protocol.py`, `tests/test_protocol_v2.py` |
| Конфиги хаба и клиента | готово | `config.hub.example.yaml`, `config.client.example.yaml`, `tests/test_hub_config.py`, `tests/test_client_config_v2.py` |
| SQLite + миграции | готово | `migrations/0001_init.py`, `hub/migrations_runner.py`, `tests/test_migrations.py` |
| CI (ruff, mypy, pytest) | готово, проверено локально | `ruff check .` — чисто; `mypy common` (strict) — чисто; pytest 2397+ зелёных |
| Регрессионный набор реплик | готово: 112 реплик (из них 102 проверяются текстовым роутером, порог 95 % держится) | `tests/regress/fixtures.jsonl`, `tests/regress/README.md` |
| Каркас скиллов | частично: манифест + генератор, без реестра | `hub/skills_runtime.py`, `hub/skill_scaffold.py` |

**Вне ТЗ, по прямой просьбе заказчика (сделано):**

| Изменение | Статус | Доказательство |
|---|---|---|
| Кулдаун уведомлений: минимум 10 с → 1 с; диапазоны вынесены в один источник | готово | `hub/presence_alerts.py` (`RULE_RANGES`), тесты в `tests/test_presence_alerts.py` |
| Пресеты числовых полей уведомлений в Telegram (без ввода текста) | готово | `hub/telegram_admin.py` (`_ALERT_PRESETS`), `tests/test_telegram_admin.py` |
| Компактная клавиатура `/tools`: по две кнопки в ряд, кулдаун виден в списке правил | готово | `hub/telegram_admin.py`, `tests/test_telegram_admin.py` |

**Фаза 1 — ядро (закрыта):**

Критерии приёмки фазы закрыты: две комнаты на одном хабе работают
одновременно и изолированно; реплика одной не задерживает другую дольше
1,5 с; токены работают (при этом найдена и исправлена реальная ошибка:
любой v2-клиент отвергался кодом 4401, см. `DECISIONS.md` P1-44); сцена
«кино» выполняется по голосовой команде end-to-end. Подробности — в строках
P1-44…P1-47 ниже и в `PROGRESS.md`.

| Пункт | Статус | Доказательство |
|---|---|---|
| Схема БД и миграции (4.6) | готово | `migrations/0001_init.py` |
| Векторы в sqlite-vec: подключение расширения и виртуальные таблицы (4.6) | готово | `hub/vectors.py`, `hub/vendor/vec0.dll` (0.1.9), `tests/test_vectors.py` — 24 passed; `make test` 2543 passed, 2 skipped |
| Импорт унаследованных `people.json`, `memory.jsonl`, `dialogs/*.jsonl` в БД (4.6) | код и тесты готовы, коммит заблокирован песочницей (`git` не может писать `.git`) | `hub/legacy_migrate.py`, `hub/main.py`, `tests/test_legacy_migrate.py` |
| Слой решений Decider: интерфейс, Rules-провайдер, цепочка с таймаутом, политики уверенности (раздел 5) | частично: нет LocalLLM и Jev; точки D-02–D-09 ещё не переведены | `hub/decider.py`, `tests/test_decider.py`, `tests/test_decision_log.py` |
| Решения пишутся в таблицу `decisions` (5.3) | готово | `hub/decision_log.py`, `tests/test_decision_log.py` |
| Очередь GPU: классы приоритета, fair share по домам, таймауты, оценка ожидания (4.5) | готово: подключена к STT/LLM/лицам/vision/SAM3 | `hub/gpu_queue.py`, `tests/test_gpu_queue.py`, `tests/test_gpu_queue_wiring.py` |
| Auth клиентов по токену, привязка сессии к `home_id`, rate limit (4.3) | готово | `hub/auth.py`, `hub/gateway.py`, тесты |
| Gateway: приём `hello`, реестр комнат, квоты на реплики и кадры (4.4) | готово | `hub/gateway.py`, `tests/test_gateway.py` |
| Backpressure: буфер на сессию, фон отбрасывается первым, метрики (4.4) | готово в коде | `hub/outbound.py`, `server.outbound.queue_capacity`, `/health.outbound`, `tests/test_outbound.py` — 7 passed |
| STT батчинг 2–4 реплики в один вызов (4.4, 15.1) | готово в коде, включён конфигом `server.stt.batch_size` (по умолчанию 4) | `hub/stt.py::SttBatcher`/`transcribe_batch`, `tests/test_stt_batching.py` — 8 passed |
| Авторизация подключена к живому `hello`: v1 проходит без токена, v2 без токена закрывается 4401 (4.3) | готово | `hub/app.py::Connection._authorize`, `tests/test_hello_auth.py` |
| Хаб-БД поднимается при старте: миграции + сид комнат, отказ не роняет сервер (4.6, 4.7) | готово | `hub/main.py::_prepare_hub_database` |
| Дома из конфига: сид таблицы `homes`, `config_rev` (4.2, 4.7) | готово | `hub/homes.py`, `tests/test_homes.py` |
| Горячая перезагрузка настроек дома + `config_update` клиенту (4.7) | готово в коде, вызов из хаба; форма владельца — P1-40 | `hub/config_reload.py`, `hub/app.py::reload_room_configs`, `tests/test_config_reload.py` — 11 passed |
| vLLM-провайдер: OpenAI-совместимый API, tool calling, guided JSON (F-402) | готово в коде | `hub/llm.py` (`PROVIDER_VLLM`, `structured_json`), `tests/test_llm_vllm_provider.py` |
| Роутер моделей: уровни `local_fast`/`local_strong`/`local_vision`/`cloud_cheap`/`cloud_strong`, D-10 (F-401) | готово в коде, выключено конфигом (`models.enabled: false`). Короткая реплика (≤ `models.routing.short_chars`, по умолчанию 120 символов) уходит в `local_fast` и большую модель не трогает — в том числе когда в ней ОДИН вызов инструмента (P3-04); реплика с признаком кода/рассуждения или длиннее `strong_chars` идёт в сильную модель, а пустая `local_fast.model` честно деградирует до неё, а не молчит | `hub/model_router.py` (правило `pick`: перелив → картинка → сложность → длина), `hub/app.py::_reply_model`, `tests/test_model_router.py`, `tests/test_local_fast.py` — 7 passed (настоящий ход `Connection._handle_utterance` с настоящими роутером и `LevelPool`, подставлены только клиенты уровней) |
| Перелив при перегрузке очереди (F-403) | готово в коде, включается `models.routing.cloud_fallback`. Перелив ВИДЕН СНАРУЖИ: `/health.models` публикует уровни, пороги, мгновенное ожидание очереди (`queue_wait_s`), счётчики (в т.ч. сколько ходов ушло в облако) и последние решения (`уровень`, `причина`, `ожидание`, `overflow`). Проверено на живом пути: занятая настоящая `GpuQueue` + разрешённое облако + бюджет → ответ даёт `cloud_cheap` с причиной `queue_overflow`; спокойная очередь, запрет владельца и исчерпанный бюджет оставляют ответ локальным. Правило уровня теперь ОДНО: `RulesDecider` стоит за `ModelRouter.pick` (передаётся как `heuristic`), а не выводит уровень второй раз по `looks_complex` | `hub/model_router.py` (`pick`/`choose`/`RoutingReport`), `hub/decider.py::RulesDecider.choose`, `hub/app.py::_models_snapshot` + `/health.models`, `hub/gpu_queue.py::wait_estimate`, `tests/test_models_health.py` — 8 passed, `tests/test_model_router.py`, `tests/test_image_difficulty.py` |
| `utterance_id` (ULID) на клиенте, сквозь сообщения, логи, БД и метрики (4.5) | готово | `common/ids.py`, `hub/utterances.py`, `hub/app.py::_store_dialog_turns`/`/health.utterances`, `tests/test_ulid.py`, `tests/test_utterance_id.py` — 30 passed |
| `event_id` на каждое фоновое событие камеры (4.5) | готово: кадр/скриншот/клип несут один id через сообщения, логи, архив и метрики | `hub/camera_events.py`, `hub/app.py` (запросы кадров), `hub/camera_clip_receiver.py`, `client/camera.py`, `client/camera_clips.py`, `tests/test_camera_events.py` — 13 passed |
| Таймауты по стадиям с деградацией, а не молчание (4.5, 15.1) | готово | `common/config.py::StageTimeouts` (`server.timeouts`), `hub/app.py::_stage_budget`/`_degrade`/`_plain_transcript`, `tests/test_stage_timeouts.py` — 10 passed |
| Порог перелива F-403 внесён в конфиг по замеру (F-403, 15.1) | готово с оговоркой: порог = бюджет 15.1 (0.55 с), замер на реальной очереди с оценочным временем реплики 2.5 с (живого стенда с тремя комнатами в песочнице нет) | `scripts/measure_overflow.py`, `models.routing.overflow_wait_s: 0.55` в `config.yaml`, `tests/test_measure_overflow.py` — 10 passed, `DECISIONS.md` (P1-24) |
| `LocalLLMDecider`: structured output, logprobs первого токена для yes/no (5.2) | готово в коде, включается через `server.decider.order` | `hub/decider_local.py`, `hub/llm.py::first_token_probabilities`, `tests/test_decider_local.py` — 12 passed |
| Порядок провайдеров по типу решения и таймаут 400 мс задаются конфигом (5.2) | готово | `common/config.py::DeciderConfig` (`server.decider`), `hub/app.py::_decision_settings`/`_decision_providers`, `tests/test_decider_config.py` — 12 passed |
| Точки D-02, D-03, D-04, D-05, D-07, D-09, D-11 переведены на Decider (5.3) | готово; D-02/D-11 решаются и пишутся хабом, а исполняются wake-гейтом и окном follow-up (F-103, задача P2-03 — сделано) | `hub/decision_points.py`, `hub/decider.py::HEURISTIC_TYPES`, `hub/app.py::_decide`/`_permission_check`, `tests/test_decision_points.py` — 25 passed |
| Кэш решений по хешу входа на 60 с (5.4) | готово | `hub/decider_cache.py`, `common/config.py::DeciderConfig.cache_ttl_s`, `hub/app.py::_decision_cache`, `tests/test_decision_cache.py` — 13 passed |
| Недельный отчёт калибровки: доля ошибок по типу и провайдеру (5.4) | готово; «ошибка» — решение, которое опроверг сам ход: роутер обещал локальную команду, а её не нашлось, либо self-check нашёл работу, которой не было. Типы, которые хаб не проверяет, показываются как непроверенные, а не как успешные (см. `DECISIONS.md`) | `migrations/0002_decision_observations.py`, `hub/decision_log.py::observe`/`calibration`, `hub/app.py::_observe`, `hub/admin_backend.py` (`calibration.list`), страница «Calibration» в Telegram-админке, `tests/test_calibration_report.py` — 26 passed |
| Абстракция устройств: модель `Device`, capability-инструменты `device.set`/`device.get` (F-501, 10.1) | готово в коде; адаптеры подключаются реестром, без установленного адаптера реплика честная («адаптер не установлен»), а не выдуманный успех | `hub/devices.py`, `tests/test_devices.py` — 48 passed, единицы значений в `DECISIONS.md` (P1-34) |
| Адаптеры устройств: MQTT, Home Assistant REST/WS, Roku/Android TV, Spotify Connect, BLE/Tuya/MagicHome/HDMI-CEC (F-502) | готово в коде; библиотеки железa (`bleak`, `tinytuya`, `flux_led`, `cec`, `androidtv`) в песочнице не установлены — адаптер честно выпадает из реестра с именем пакета, HTTP/MQTT-адаптеры проверены на подставных транспортах; живого железа нет | `hub/adapters/`, `tests/test_device_adapters.py` — 23 passed, `DECISIONS.md` (P1-35) |
| ESP32-переключатели: прошивка, MQTT-топики `set`/`state`/`config`, Mosquitto, калибровка углов через админку (F-503) | готово в коде; живого железа нет — прошивка проверена на синтаксис и на своей арифметике (углы, PWM, отпускание серво), публикация калибровки — на подставном MQTT-клиенте; без брокера углы всё равно сохраняются, реплика говорит «переключателю не сообщено» | `firmware/esp32_switch/`, `firmware/mosquitto/mosquitto.conf`, `hub/switch_calibration.py`, `hub/adapters/mqtt.py::publish_config`, `hub/admin_backend.py` (`devices.list`/`devices.calibrate`), страница «Wall switches», `tests/test_esp32_switches.py` — 18 passed |
| Мастер добавления устройств: BLE, mDNS, Tuya discovery, порты; выбор адаптера, тест «мигни», имя и алиасы в hotwords (F-504) | готово в коде; сканы библиотек (`bleak`, `zeroconf`) честно сообщают о недоступности, TCP-скан и Tuya-broadcast проверены без железа (локальный сервер и подставной сокет) | `hub/discovery.py`, `hub/device_wizard.py`, `hub/admin_backend.py` (`devices.scan`/`add`/`blink`/`remove`), `tests/test_device_wizard.py` — 16 passed |
| Сцены: модель, шаги `device|pc|say|delay`, предустановки «кино/учёба/сон/гости/ушёл», запуск из админки (F-506) | готово в коде; предустановки создаются один раз и не перезаписывают правки владельца, неудавшийся шаг попадает в отчёт, а не в тишину | `hub/scenes.py`, `migrations/0003_scene_presets.py`, `hub/admin_backend.py` (`scenes.list`/`run`/`delete`), страница «Scenes», `tests/test_scenes.py` — 22 passed |
| Сцена по голосовой команде и создание голосом (F-506) | готово: «кино»/«cinema»/«включи кино» запускают предустановку в комнате, «запомни как сцену „вечер“» сохраняет действия прошлого хода (устройства и ПК), предложение со словом внутри уходит модели | `hub/scenes.py::match_scene`/`steps_from_actions`/`REMEMBER_SCENE`, `hub/app.py::Connection._scene_turn`, `tests/test_scene_voice.py` — 19 passed |
| Реестр скиллов, изоляция по дому, таймаут 20 с (F-405, F-406) | готово | `hub/skills_registry.py`, `tests/test_skills_registry.py` |
| Hot reload скиллов по изменению файла только в dev-режиме (F-405) | готово: `server.skills.dev_reload` выключен по умолчанию; сломанный файл оставляет рабочую копию | `common/config.py::SkillReloadConfig`, `hub/skills_registry.py::SkillWatcher`, `hub/app.py::_skill_hot_reload`, `tests/test_skill_reload.py` — 14 passed |
| Веб-админка владельца (F-705) | готово в коде: FastAPI + Jinja на том же приложении, доступ только из overlay-сети (LAN получает 404 даже на форму логина), пароль из окружения, клиенты показаны отпечатком хеша токена; разделы «дома», «клиенты (токены/статус/версия)», «люди и членство» | `common/config.py::WebAdminConfig`, `hub/web_admin.py`, `hub/templates/admin/`, `hub/app.py::_mount_web_admin`, `tests/test_web_admin.py` — 19 passed |
| Аудит привилегированных действий, изменений настроек и удалений (F-706) | готово: кто, что, когда, дом, результат; отказ тоже пишется, чтения — нет, секреты в `detail` заменяются на `[redacted]` | `hub/audit.py`, `hub/admin_backend.py` (`AUDITED`, `_home_of`), страница `/admin/audit`, `tests/test_audit.py` — 12 passed |
| Автозапуск сервисов и единые цели (4.9) | готово в коде: Task Scheduler/NSSM на Windows (хаб — скрыто, с перезапуском; клиент — при входе пользователя), systemd на Linux; `make test` = pytest + ruff + mypy, как в `AGENTS.md`. Установка на живом стенде не выполнялась (в песочнице нет прав вне каталога) | `deploy/`, `Makefile`, `tests/test_deploy.py` — 10 passed |
| OTA клиента по git-тегу хаба (4.9) | готово в коде: тег из `server.client_release`, проверка раз в час, fetch/checkout, миграция конфига, рестарт, откат на прошлый тег, если новый не прожил `healthy_after_s` (60 с). Живого репозитория и второго ПК в песочнице нет — git и рестарт подставные в тестах | `common/client_config.py::ClientOTAConfig`, `hub/ota.py`, `client/ota.py`, `client/main.py`, `tests/test_client_ota.py` — 18 passed |
| Регрессионный набор реплик доведён до 100 фраз (15.6) | готово: 112 записей (en/ru/es), 102 из них проверяются текстовым роутером с порогом 95 % (сейчас 100 %), 10 реплик болтовни/«не ко мне» ждут записанной речи и пропускаются с причиной | `tests/regress/fixtures.jsonl`, `tests/regress/test_regression.py`, `tests/regress/README.md` — `pytest tests/regress -q` → 104 passed, 10 skipped |
| Две комнаты на одном хабе: одновременно и изолированно (16, фаза 1) | готово в коде: два настоящих `Connection` через настоящий `Gateway` с двумя токенами; оба дома отвечают одновременно и не мешают друг другу, ответы и `actions`-кадры уходят только на свой сокет, `dialog_turns` пишутся под своим `home_id`, устройства и сцены одного дома не видны другому, токен одного дома не принимается на другой (4401), отключение комнаты не трогает соседнюю. **Исправлено по ходу:** `_authorize`/`_send_room_config` уводили соединение `data/hub.db` в рабочий поток и ловили `sqlite3.ProgrammingError` — любой v2-клиент получал 4401, а кадр настроек комнаты не отправлялся; теперь чтение токена и строки дома идёт на цикле (см. `DECISIONS.md` P1-44). Живого стенда с двумя ПК в песочнице нет — проверка идёт по настоящему коду хаба с подставными STT/LLM/TTS | `hub/app.py` (auth, `_send_room_config`), `tests/test_two_rooms.py` — 6 passed |
| Реплика одной комнаты не задерживает другую дольше 1,5 с (16, 15.1) | замер выполнен на настоящем коде хаба (`scripts/measure_cross_room_delay.py`): собственное планирование хаба добавляет **−0,002 с** (быстрая комната: 0,218 с в одиночку против 0,215 с рядом с медленной), очередь GPU при трёх активных домах даёт медиану 0 и p95 0, а без перелива F-403 — разовый максимум 2,59 с; с включённым F-403 (порог 0,55 с) хвост срезается: максимум 0,0 с, 2 из 36 реплик ушли на облачный уровень. Живого стенда (RTX 5090 + 3 комнаты) в песочнице нет, поэтому время одной реплики — параметр (2,5 с, как в P1-24); см. `DECISIONS.md` (P1-45) | `scripts/measure_cross_room_delay.py`, `tests/test_cross_room_delay.py` — 4 passed |
| Сцена «кино» по голосовой команде end-to-end (F-506) | готово: реплика идёт по настоящему пайплайну (`Connection._handle_utterance`) до адаптеров и кадра `say`; «Rowan, кино» / «Rowan AI, cinema» запускают предустановку (свет off, лента `#221100`, ТВ on) и отвечают «Cinema mode.» **без обращения к модели**; нехватка устройства не выдумывает успех — шаг попадает в ответ словами. С фразой самого сценария 1 («выключи свет и включи фильм», ru/en/es) то же самое плюс секундомер — см. строку «Сценарий 1» ниже | `tests/test_scene_e2e.py` — 6 passed |

**Фаза 2 — речь, идентичность, присутствие (закрыта):**

Критерии приёмки фазы (раздел 16) закрыты: сценарий 1 («Rowan, выключи свет и
включи фильм» → сцена «кино» за < 2 с — замер на настоящем ходу: медиана
0,006 с), сценарий 2 (приветствие по имени после спины к камере — 0,179 с),
сценарий 4 (незнакомец → фото в Telegram за < 5 с — 2,18 с вместе с
требованием «стабильно в кадре»), сценарий 7 (комната без хаба продолжает
работать сама и говорит об этом), сценарий 9 (двое говорят одновременно —
вопрос «повторите по одному», смесь не выполняется). Бюджеты 15.1/15.3
проверены замерами (`scripts/measure_scene_latency.py`,
`scripts/measure_presence_latency.py`; F-101/F-102/F-403 — таблица замеров
ниже). Набор идентичности 15.6 готов форматом, метриками и прогоном, но его
числа (точность, полнота, «спина после фронта ≥ 80 %») требуют записей и
insightface — они получаются командой `make test-identity` на стенде и
отмечены ниже как **не измеренные**, а не как пройденные. Подробности по
задачам — в `PROGRESS.md` (P2-01…P2-40).

| Пункт | Статус | Доказательство |
|---|---|---|
| F-101 Стриминговый ответ: первое предложение звучит сразу | готово в коде: первое предложение синтезируется отдельной группой (короткий ответ больше не ждёт синтеза целиком), бюджет «конец речи → первый звук» (1,2 с из 15.1) измеряется на настоящем пайплайне и пишется в лог хода; старт по промежуточному транскрипту (P2-41) реализован — модель получает черновик `LiveTranscript` параллельно финальному STT, ранний ответ уходит в комнату только после сверки с финальным текстом (расхождение — честный откат к обычному ходу), флаг `server.streaming_reply.early_start` выключен по умолчанию до замера на стенде; живого выигрыша в песочнице не мерили (модели подставные) | `hub/streaming_reply.py`, `hub/early_start.py`, `hub/live_transcript.py::draft_text`, `hub/app.py` (`_stream_tts_unlocked`, `_report_first_audio`, `_start_speculative_reply`, `_reconcile_speculative`), `common/config.py::StreamingReplyConfig`, `tests/test_streaming_reply.py` — 21 passed, `tests/test_early_start.py` — 21 passed |
| F-104 Динамические hotwords | готово: имена людей, устройства и их алиасы, сцены и их алиасы и приложения роутера собираются из БД (не из одной формы админки), слова владельца идут первыми и не вытесняются ограничением, повторы по регистру схлопываются; движок получает одну строку, обновление — на старте хаба и после добавления/удаления устройства, удаления сцены и переименования человека | `hub/hotwords.py`, `hub/app.py::_refresh_hotwords`, `hub/admin_backend.py::_sync_hotwords`, `tests/test_hotwords.py` — 9 passed |
| F-102 Barge-in | готово в коде: речь поверх ответа глушит воспроизведение (бюджет 15.1 — 200 мс; секундомер считает от начала речи до возврата `abort()`, то есть до реальной тишины, включая буфер устройства) и записывается как новая реплика — с уже услышанными словами и без подтверждающего бипа; без работающего WebRTC AEC функция выключена (микрофон даже не опрашивается), в HUD остаётся предупреждение, а wake word по-прежнему прерывает ответ. Живой звуковой карты в песочнице нет — замер на стенде даст строку `Barge-in (speech): silence … ms` в логе клиента | `client/barge_in.py`, `client/audio.py::AudioOutput.abort`, `client/audio_processing.py` (`aec_active`), `client/main.py` (`_apply_barge_in_state`, `_cut_playback_for_speech`, `_barge_loop`), `client/overlay.py` + `overlay_web/hud.html`/`chat.html`, `common/client_config.py::AudioConfig.barge_in`, `tests/test_barge_in.py` — 22 passed |
| F-103 Follow-up без wake word | готово: клиент держит микрофон открытым 6 с после ответа (`attention_mode: window`, `followup_window_s: 6` в шаблонах), в HUD видно окно с оставшимися секундами (закрывается первым же записанным звуком и в конце хода), признак окна уходит на хаб в `utterance_start.followup`; адресность решают D-02 и D-11 — реплика внутри окна отвечается без имени, разговор людей между собой вне окна игнорируется с кадром `ignored` (и с записью в трейс), а клиент, который окна не объявляет, сохраняет прежнее поведение (`server.followup.gate_unaddressed` по умолчанию выключен). **Исправлено по ходу:** RulesDecider для `addressed` отвечал только по wake-префиксу и перекрывал ответ пайплайна — из-за этого цепочка решений принимала «соседскую» реплику за обращённую | `common/protocol.py::UtteranceStart.followup`, `common/config.py::FollowupConfig`, `hub/app.py` (`_client_followup`, `_in_followup`, развилка D-02), `hub/decider.py` (`addressed`), `client/main.py`, `client/overlay.py`, `client/overlay_web/hud.html`, `config*.yaml`, `tests/test_followup.py` — 11 passed, `tests/test_overlay_followup.py` — 6 passed |
| F-105 Фильтр галлюцинаций v2 | готово: D-03 (знаков в секунду выше порога, пол — полсекунды аудио) + два новых правила — транскрипт, состоящий только из стоп-фраз ru/en/es (включая кредитные строки «субтитры сделал …», «subtitles by …»), и зацикленный повтор n-граммы (1–4 слова, ≥3 повторов, ≥60 % транскрипта, ≥4 слова; одно слово требует 4 повторов, чтобы «нет, нет, нет» не срезалось). Правила настраиваются в `server.stt.hallucination`, причина отказа пишется в лог, транскрипт до модели не доходит. **Исправлено по ходу:** RulesDecider для `hallucination` считал только «знаки в секунду» и перекрывал вердикт пайплайна — теперь возвращает его | `hub/noise_filter.py`, `common/config.py::HallucinationConfig`, `hub/app.py`, `hub/decider.py`, `tests/test_noise_filter.py` — 30 passed |
| F-106 Язык по говорящему | готово: `persons.preferred_language` — одна точка записи обновляет и строку в БД, и копию в реестре голосов (`data/people.json`); известный говорящий получает ответ на своём языке (инструкция «Answer in …» едет в префиксе хода, а не в системном промпте, чтобы не ломать кэш промпта), а со СЛЕДУЮЩЕГО хода Whisper получает его как `language`-подсказку (говорящий и язык определяются в одном проходе — раньше подсказки быть не может, это и есть «после идентификации» из ТЗ). Незнакомец — автоопределение внутри `server.stt.allowed_languages`; язык вне whitelist падает в `server.stt.language`. Коды нормализуются (ru-RU, «Русский», English…), любой двухбуквенный код из whitelist пропускается в Whisper, опечатка вида «klingon» отвергается. В трейс хода пишутся `reply_language` и `preferred_language`. Запись из админки — `profiles.language` | `hub/languages.py`, `hub/speaker.py` (`language_of`/`set_language`), `hub/app.py` (`_whisper_language`, `_preferred_language`, префикс хода), `hub/admin_backend.py`, `hub/session.py` (префикс), `tests/test_language_per_speaker.py` — 27 passed |
| F-108 Одновременная речь | готово: диаризация отдаёт интервалы, где говорили одновременно, хаб измеряет их долю от длины реплики; больше 40 % (`server.diarization.overlap_limit`, 0 выключает) — ход НЕ выполняется: комната слышит «I heard voices talking over each other. Please say Rowan AI and repeat one at a time», инструменты и модель не вызываются, в трейс пишутся `overlap.seconds/ratio/limit`. Ровно 40 % — ещё не превышение. Регистрация голоса не изменилась: она и раньше шла строгим режимом и при перекрытии просит прочитать фразу заново, не принимая смешанную запись. Тест фазы 1 на ослабленный режим теперь явно ставит `overlap_limit: 0` | `hub/overlap.py`, `hub/diarization.py` (`AttributedUtterance.overlaps`), `common/config.py::DiarizationConfig.overlap_limit`, `hub/app.py`, `hub/enrollment.py::clarification`, `tests/test_overlap_speech.py` — 11 passed |
| F-113 Подтверждение опасных действий | готово в коде: опасный вызов (инструмент `run_command` целиком, кроме read-only команд — `dir`, `uptime`, `echo`/`write-output`, …; и `pc_control` `sleep`/`suspend`/`hibernate`/`shutdown`/`power_off`/`reboot`/`restart`/`logoff`/`logout`/`close_app`) не уходит на клиент: комната слышит вопрос хаба («Say yes within 8 seconds to run the command 'shutdown /s'…anything else cancels it»), а не ответ модели. «Да» (en/ru/es, wake-слово и «пожалуйста» снимаются) в течение 8 с (`server.confirmations.window_s`) выполняет ровно тот же вызов; «нет», молчание/просроченное окно и любая другая просьба отменяют его (другая просьба при этом обрабатывается как обычно), и каждое решение пишется в `audit` (`action=confirm.dangerous`, `result` ok/denied/failed). Вопрос задаётся ПОСЛЕ проверки прав: «нельзя» важнее «подтверди». Список и слова настраиваются в `server.confirmations`; полный выбор «по умолчанию» — в `DECISIONS.md` (P2-08). Живого микрофона в песочнице нет — голосовой замер на стенде не делали | `hub/confirmations.py`, `common/config.py::ConfirmationsConfig`, `hub/app.py` (`_confirmation_needed`/`_request_confirmation`/`_resolve_confirmation`/`_audit_confirmation`), `config.yaml`, `config.example.yaml`, `tests/test_confirmations.py` — 62 passed |
| F-114 Многошаговые команды | готово: список действий приходит ОДНИМ structured output (несколько tool-call'ов в одном ответе либо JSON `{"steps": [{"tool": …, "arguments": {…}}, …]}` — принимаются также `actions`/`plan`/`commands` и голый массив, имя инструмента сверяется с настоящим `TOOL_NAMES`, поэтому прозаический JSON не исполняется; план короче двух шагов планом не считается, больше 8 — отвергается). Шаги выполняются строго по порядку через обычный клиентский протокол (`a1`, `a2`, …), падение одного не останавливает остальные, и в трейс хода идёт результат каждого шага. Реплика хода с ≥2 действиями дополняется отчётом по КАЖДОМУ упавшему шагу («1 of 2 steps are done. Step 2 of 2 (pc control) failed: the player is not running»); ход с одним действием сохраняет прежнее поведение. JSON плана вырезается из произносимого текста | `hub/multi_step.py`, `hub/llm.py` (`find_plan`→tool-call'ы, `LlmResult.plan_steps`), `hub/app.py::_multi_step_report`, `prompts/system.md`, `tests/test_multi_step.py` — 25 passed |
| F-117 Локальные команды на клиенте | готово в коде: `client/local_commands.py` — один словарь фраз ru/en/es и `LocalRunner`, который исполняет команду СВОИМИ руками клиента, без хаба: громкость (`громче`/`тише`/`громкость 30`/mute/unmute), медиа (следующий/предыдущий трек, пауза), «стоп/замолчи» (собственный путь тишины), «повтори» (проигрывание последнего ответа из локального кэша PCM, 30 с), устройства (`включи/выключи <устройство>` по списку устройств этого ПК), сцены (шаги приходят в `config_update` и кэшируются; выполняются по порядку с отчётом о каждом упавшем шаге) и таймеры (считает сам клиент, заголовок + бип). Громкость/медиа/устройства идут через НАСТОЯЩИЙ `client.actions.dispatcher` (pycaw + media-keys), то есть тем же кодом, что и команды от хаба. Применяется и в `_wait_for_wakeword` (когда играет ответ или когда связи нет), и в `_barge_loop`. Команда без своей руки (нет диспетчера, нет локального голоса для `say`-шага сцены, сцена не кэширована) отвечает честно «без хаба не могу», а не изображает успех. Живого микрофона и локального TTS в песочнице нет: словарь и раннер проверены тестами, а голосовой замер (и локальный распознаватель faster-whisper small) — задача P2-35 | `client/local_commands.py`, `client/main.py`, `client/voice_controls.py::SilenceDetector.accept_phrase`, `hub/config_reload.py` (`scenes` в `config_update`), `hub/app.py`, `scripts/export_client.py`, `tests/test_local_commands.py` — 61 passed |
| F-201 Трекинг людей на клиенте | готово в коде: YOLO11x + BoT-SORT остаются за Ultralytics (`model.track(..., persist=True)`), но буфер трекера BoT-SORT считается в КАДРАХ, поэтому «re-association до 30 с» выведено из реального fps (`track_buffer = max(30, 30 с × fps)`: 150 кадров при 5 fps, 900 при 30) и пишется в `room-tracker.runtime.yaml` (пороги берутся из поставленного `room-tracker.yaml`, меняется только буфер). Поверх этого `client/tracking.py::TrackRegistry` держит id потерянного трека 30 с и сообщает `entered`/`returned`/`active` с величиной разрыва, а `track_id` у человека в кадре сохраняется, пока он виден. В протокол уходят именно ТРЕКИ: отдельное сообщение `tracks` (`track_id`, `bbox`, `conf`, `zone`, `since`) при смене набора треков (движение — с дебаунсом 2 с), `camera_state` со счётчиком остаётся для клиентов без этого сообщения; `hub/room_state.py` читает обе формы записи (`{id, box}` v1.4 и `{track_id, bbox}` F-201), поэтому комната не слепнет на переходе. Живой камеры и GPU в песочнице нет: логика проверена на реестре, а замер 30-секундного re-association — в задаче P2-36 (набор идентичности) | `client/tracking.py`, `client/camera.py` (runtime-конфиг трекера, `_publish_tracks`), `common/protocol.py::MSG_TRACKS`, `common/client_config.py::CameraConfig.tracks_message`, `hub/app.py::_on_tracks`, `hub/room_state.py::normalise_tracks`, `tests/test_tracking.py` — 20 passed |
| F-202 Кропы тела в хаб | готово в коде: кроп тела уходит в хаб при появлении трека, затем не чаще раза в 2 с и дополнительно при смене ракурса (аспект bbox изменился на ≥ 15 %). JPEG — полная фигура по bbox с полем 4 %, высотой не больше 640 px, q90, с `track_id`, `home_id`, `client_id`, `ts`. Хаб проверяет JPEG по SOF-маркеру, заводит ряд трека (`ensure_track`) под FK `body_crops→tracks` и кладёт файл через `MediaStore` как kind `crop` в `data/homes/<home>/media/<date>/`, поэтому TTL медиа (F-304) удалит кроп вместе с остальными файлами. Кропы включаются тем же флагом `client.camera.tracks_message`. Живого GPU/камеры в песочнице нет: кропы и хранение проверены на настоящих JPEG через cv2, а замер «узнавание со спины» — задача P2-36 | `common/body_crops.py` (`CropSchedule`, `jpeg_height`/`crop_is_valid`), `client/body_crops.py::encode_crop`, `client/camera.py::_publish_body_crops`/`_send_body_crop`, `common/protocol.py::MSG_BODY_CROP`, `hub/body_crops.py::BodyCropStore`, `hub/app.py::_on_body_crop_header`/`_deliver_body_crop`, `migrations/0004_body_crops.py`, `tests/test_body_crops.py` — 20 passed |
| F-203 ReID-эмбеддинги тела | **готово в коде, модель без замера.** `hub/reid.py`: OSNet из torchreid (`osnet_x1_0`, `osnet_ain_x1_0` через `server.identity.reid.model`, предобученные веса) даёт L2-нормированный 512-d вектор на кроп; предобработка как в примере torchreid (resize 256×128, BGR→RGB, статистики ImageNet) вынесена в чистую функцию, поэтому проверяется без torch. Строка `body_embeddings` несёт `track_id`, `session_day` (локальный день дома — F-206 сравнивает тела только внутри дня) и `person_id`, когда он известен (`tracks.person_id` либо совпадение того же дня при cos ≥ 0,5 с отрывом 0,05 — порога в ТЗ нет, поэтому это выбор исполнителя, см. `DECISIONS.md` P2-13); `person_id` всегда сверяется с `persons`. Одна строка на кроп, без усреднения (кластеры дня — F-209/P2-19). Вектор считается фоновой задачей GPU-очереди (`PRIORITY_BACKGROUND`, `asyncio.to_thread`), кроп уже сохранён, реплику это не задерживает. Вектор обязан быть 512-d: другая размерность — отказ с причиной в логе. **Чего нет:** torchreid в песочнице не установлен, поэтому реального инференса не было — модуль лениво импортирует torch/torchreid, при недоступности честно не пишет ни одной строки и не подставляет заглушку; замер точности (в т. ч. «со спины после одного фронтального кадра ≥ 80 %») — задача P2-36 | `hub/reid.py` (`ReidEngine`, `BodyEmbeddingStore`, `match_day`, `normalise`, `preprocess_image`, `session_day_of`), `common/config.py::IdentityConfig`/`ReidConfig` + `config.yaml`/`config.example.yaml` (`server.identity.reid`), `hub/app.py::_reid_engine`/`_body_embedding_store`/`_embed_body_crop`/`_schedule_reid`, `migrations/0005_body_embedding_track.py`, `tests/test_reid.py` — 43 passed |
| F-204 Лица привязываются к треку | **готово в коде, модель без замера.** `hub/face_tracks.py`: лицо хранится в НАСТОЯЩЕЙ таблице `face_embeddings` (сх. 14) со своим `track_id`, `dim`, `quality` и `person_id`, когда он известен. Правило «лицо внутри bbox трека» не дублируется — это `hub/room_state.py::enclosing_track` (в двух перекрывающихся телах ответ «не знаю», а не догадка). Камера отдаёт ~5 кадров/с, поэтому `observe` хранит первый ракурс трека и затем только действительно новые (cos < 0,9): таблица получает «фронтальный кадр + поворот», а не поток дубликатов (частоты в ТЗ нет — выбор исполнителя, `DECISIONS.md` P2-14). Распознанное лицо распространяется на ВЕСЬ трек: `spread` проставляет `tracks.person_id`, линкует все лица трека и векторы тела ТОГО ЖЕ дня через `BodyEmbeddingStore.link_track` — так спина и бок считаются человеком, чьё лицо видели один раз; вчерашние векторы не трогаются (внешность дня — F-209). Трек, уже названный другим человеком, не переписывается: конфликт возвращается флагом (смену решённой личности решает гистерезис F-207). Имя → `person_id`: правило хаба `hub/utterances.resolve_person_id` плюс регистронезависимая сверка в Python (SQLite `lower()` не сворачивает кириллицу); незарегистрированное имя не создаёт `persons`, лицо хранится без ссылки. Вызов — сразу после `resolve_faces` в `_match_presence`: GPU не нужен (эмбеддинг уже посчитан для presence), БД-работа идёт на цикле (DECISIONS P1-44). **Чего нет:** insightface в песочнице не запускался — проверены привязка, хранение и распространение на настоящей схеме; замеры спина/бок и 80 % — P2-36 | `hub/face_tracks.py` (`FaceTrackStore`, `observe`/`record`/`spread`), `hub/app.py::_face_track_store`/`_person_id_of`/`_record_track_faces`/`_observe_track_face`, `tests/test_face_tracks.py` — 21 passed |
| F-205 Голос привязывается к треку | **готово в коде, модель без замера.** `hub/voice_tracks.py`: связь ставится ровно в двух случаях из ТЗ — активный трек один, либо среди нескольких ровно один «лицом к камере с движением губ»; всё остальное (двое говорят, никто не смотрит, лица не видно) остаётся без связи, потому что ошибка здесь попала бы в права и историю человека на много ходов вперёд. «Лицом» и «губы» считаются по 5 точкам insightface (нос между глазами, отношение 0,2…0,8; размах расстояния уголков рта от носа в межзрачковых единицах за окно 12 кадров, порог 0,06 — числа калибруются на стенде, `DECISIONS.md` P2-15). `hub/face.py::_landmarks` теперь отдаёт `kps` (или `[]` — выдуманного направления лица нет), `hub/app.py::_note_mouth` держит состояние по трекам, `_bind_speaker_to_track` вызывается сразу после узнавания голоса в ходе. Голос хранится строкой `voice_embeddings` (сх. 14) со `track_id`, `dim` реального энкодера и `person_id`; дальше связывание общее с лицом через `hub/identity_link.py::link_track_to_person` (трек, лица трека, векторы тела ТОГО ЖЕ дня; уже названный другим трек не переписывается — решает F-207). `hub/speaker.py::identify_ex` отдаёт вектор, который уже посчитан для распознавания, поэтому второго прохода ECAPA в ходу нет; `identify` остался трёхэлементным, а реестр без `identify_ex` (тесты, старые сборки) поддерживается. **Чего нет:** живого микрофона/камеры и insightface в песочнице нет — правило и хранение проверены на синтетических landmarks и настоящей схеме; пороги и замеры «кто говорил» — на стенде, P2-36 | `hub/voice_tracks.py` (`choose_track`, `facing_camera`, `mouth_signal`, `mouth_is_moving`, `VoiceTrackStore`), `hub/identity_link.py`, `hub/face.py::_landmarks`, `hub/speaker.py::identify_ex`, `hub/app.py::_voice_track_store`/`_identify_with_vector`/`_note_mouth`/`_bind_speaker_to_track`, `tests/test_voice_tracks.py` — 25 passed |
| F-206 Слияние сигналов → person | **готово в коде, модель без замера.** `hub/identity_fusion.py`: пороги — буквально из ТЗ (лицо cos ≥ 0,45; голос ≥ 0,40 и отрыв 0,15 от второго ЧЕЛОВЕКА; тело — `server.identity.reid.threshold` и только по векторам ТЕКУЩЕГО дня, `body_signal` никогда не смотрит на другой день). Каждый сигнал даёт уверенность `(score − порог)/(1 − порог)`, шов — `p = 1 − Π(1 − c)`: один фронтальный кадр cos 0,90 даёт p 0,82 (порог «узнан» F-207 достижим с одного лица), два слабых сигнала складываются, больше 1 не выходит. Проход — раз в секунду (`FUSION_INTERVAL_S`), из живых сигналов (распознанное лицо, связанный голос; старше 15 с не доказательство) плюс векторов тела из БД. Ничья ближе `AMBIGUITY_MARGIN` уходит в D-06: `hub/decision_points.py::identity_heuristic` решает по контексту ТЗ (кто член дома из `memberships`, кто уже в комнате по `tracks.person_id`), а если он не различает ровно одного — никто не назван (`person_id = NULL`), хотя числа сохраняются для объяснения. Хранение — одна текущая строка на трек в новой таблице `identity_belief` (`track_id` PK, `home_id`, `person_id`, `p`, `sources_json` с сырыми косинусами, `at`; миграция 0006), `sources_json` — основа ответа F-215 («голос 0,71, лицо 0,90»). F-206 только считает и хранит: права и `tracks.person_id` от прямых свидетельств остаются у F-204/F-205, смену решённой личности решает F-207. **Чего нет:** GPU/камеры/микрофона в песочнице нет — слияние, D-06 и хранение проверены на настоящей схеме; калибровка чисел и замер сценариев — P2-36 | `hub/identity_fusion.py` (`fuse`, `Signal`, `Belief`, `BeliefStore`, `body_signal`, `thresholds_for`), `hub/decision_points.py::identity_heuristic` (D-06), `hub/app.py::_identity_belief_store`/`_identity_context`/`_fuse_identities`/`_identity_loop`/`_start_identity_task`, `migrations/0006_identity_belief.py`, `tests/test_identity_fusion.py` — 25 passed |
| F-207 Гистерезис узнавания | готово в коде: `IdentityHysteresis` считает серии проходов F-206 по каждому треку — «узнан» только когда p ≥ 0,8 случилось дважды ПОДРЯД для одного и того же человека (смена кандидата перезапускает счёт, слабый проход рвёт серию), «потерян» — когда p < 0,5 трижды подряд (сильный проход сбрасывает счётчик потери). Промежуточные состояния (`pending`, `fading`) различимы в логе, а `_fuse_identities` действует только на вердикт: `_confirm_track_identity` пишет `tracks.person_id` — смену решённой личности ТЗ поручает именно этому слою, поэтому одна случайная рамка F-204 переименовать человека не может, а исправить ошибку система способна; `_lose_track_identity` снимает имя, НО не трогает его, пока есть свежее прямое свидетельство (распознанное лицо или связанный голос моложе `LIVE_SIGNAL_TTL_S`): три тихих прохода значат «сигналов не было», а не «пришёл другой». Флаг визита (`should_greet`/`greeted`) выдаёт «да» ровно один раз на трек за визит, поглощает ответ и сбрасывается, если подтвердился другой человек; само приветствие — F-302 (P2-28), как и разделяет ТЗ. Гистерезисное состояние ушедшего трека забывается вместе с belief. **Чего нет:** живого GPU/камеры в песочнице нет — правила и запись проверены на настоящей схеме; озвучивание приветствия по подтверждённой личности — P2-28 | `hub/identity_fusion.py::IdentityHysteresis`, `hub/app.py::_confirm_track_identity`/`_lose_track_identity` (вердикт в `_fuse_identities`), `tests/test_identity_fusion.py` — 35 passed |
| F-208 Admin-порог и голосовой PIN | готово в коде: привилегированное действие (admin-роль + `HIGH_CONFIDENCE_TOOLS`/global `remember`) требует голос ≥ 0,65 И второй свидетель — лицо ≥ 0,55 (свежий матч лица того трека, который назван этим человеком) ИЛИ вектор тела ТОГО ЖЕ дня, уже привязанный к нему (`link_track_to_person` F-204/F-205); если ничего нет — отказ с причиной, а не догадка. Проверка стоит там же, где роль: в `_permission_check` ПОСЛЕ D-07, то есть ни один путь вызова инструмента не может её пропустить; роли trusted/user/guest решает D-07, как и раньше. Для телефона (`session.identity.kind == 'phone'`) — свой порог `phone_admin_threshold` и голосовой PIN вместо камеры: 4–8 цифр (понимаются цифры и слова ru/en/es), хранится ТОЛЬКО PBKDF2-хеш со случайной солью в `persons.settings_json.admin_pin`, сравнение constant-time; три неверные попытки — блокировка на 5 минут (счётчик в памяти процесса, не подключения), верный PIN сбрасывает счётчик, а у человека без PIN админ-действия с телефона невозможны (отказ `no_pin`). Вопрос задаёт сам хаб (say-шаг вместо ответа модели, как в F-113), следующий речевой ход разрешается `_resolve_pin`: неверный PIN можно повторить внутри окна (20 с), третий закрывает вопрос; цифры не попадают ни в лог, ни в диалоги, ни в аудит — только вердикт (`confirm.pin`); пендинг снимается при dismiss и close. Настройки — `server.identity.{admin_voice_threshold, admin_face_threshold, phone_admin_threshold, phone_pin_required, pin_window_s, pin_max_failures, pin_lockout_s}`. **Чего нет:** телефона-клиента (F-711/F-712, фаза 4) в песочнице нет — проверены правило, хранение, блокировка и разбор ответа; живой замер — на стенде | `hub/identity_fusion.py` (`admin_gate`, `VoicePin`, `AdminDecision`), `hub/app.py::_permission_check`/`_admin_strength_check`/`_admin_evidence`/`_track_of_person`/`_client_channel`/`_request_pin`/`_resolve_pin`/`_audit_pin`/`_voice_pin`, `common/config.py::IdentityConfig` + шаблоны, `tests/test_identity_fusion.py` — 48 passed |
| F-209 Дневные сессии ReID | готово в коде: `hub/identity_lifecycle.py::consolidate` — ночной проход по одному дню (по умолчанию вчерашний, локальная дата дома). Кластеры тела БЕЗ `person_id` удаляются (незачем хранить галерею незнакомцев), привязанные усредняются по человеку в один единичный вектор (`average_vectors`: единичные векторы, среднее, перенормировка — как центроид голоса, чтобы один кадр не утащил день) и пишутся в новую таблицу `daily_appearance` (`person_id`+`day` PK, `home_id`, `vector`, `dim`, `samples`, `created_at`, `expires_at`). «Внешность дня» живёт неделю (`server.identity.appearance_retention_days = 7`), срок хранится в строке и удаляется тем же проходом (`purge_expired`); `appearance_of`/`appearances_for`/`matches_day` — единственные читатели для статистики и объяснимости. Проход идемпотентен, умеет ограничиваться одним домом и запускается при старте хаба (`hub/main.py::_consolidate_identity`, как TTL медиа F-304), не мешая старту при ошибке. Миграция 0007. **Чего нет:** запуск по расписанию (а не при старте) — задача P2-30 (F-304) того же блока; живого GPU/камеры в песочнице нет | `hub/identity_lifecycle.py`, `hub/main.py::_consolidate_identity`, `migrations/0007_daily_appearance.py`, `common/config.py::IdentityConfig.appearance_retention_days` + шаблоны, `tests/test_identity_lifecycle.py` — 10 passed |
| F-210 Регистрация гостя | готово в коде: владелец (роли `admin`/`trusted`, `DECISIONS.md` P2-20) говорит «это Макс, друг» — распознаётся тремя языками детерминированным шаблоном, без модели; без имени хаб переспрашивает, а не записывает говорящего. Дальше поток F-210: гость читает фразу (≥ 3 с чистой речи, предложение сверяется по словам, поэтому «включи свет» образцом не считается), камера отдаёт два бёрста по 5 кадров, а кадр годится при лице ≥ 12 % высоты кадра, без смаза (дисперсия лапласиана ≥ 40) и с поворотом ≤ 55° (угол считается по kps insightface — та же мера, что «лицом к камере» у F-205); бёрст принимается при 5–10 кадрах и ≥ 2 ракурсах, каждый шаг повторяется один раз. Затем фиксируется «тело дня» (кропы текущего дня трека — `body_embeddings`). Гость слышит короткое уведомление о том, что хранится (голос, фото лица, внешность дня; как удалить — «забудь меня») и подтверждает голосом тем же «да», что F-113 (ТЗ 15.4). Подтверждение владельца в Telegram обязательно: хаб посылает вопрос с одноразовыми кнопками (`guest:<токен>:yes|no`, токен привязан к дому, чужое нажатие ничего не делает, просроченный вопрос не регистрирует никого). Всё собранное до этого лежит ТОЛЬКО в памяти подключения; после «да» одна транзакция (`commit_guest`) пишет человека, членство `role='guest'`, векторы голоса/лица/тела и имя живого трека через общий путь F-204/F-205; при «нет», отмене голосом, второй неудаче с кадрами, истечении окна или отсутствии Telegram не сохраняется ничего (честная фраза в комнату + запись в лог). Роль `guest` добавлена в реестр людей (`hub/speaker.py`), подтверждённый гость дописывается в него (`enroll` + лица + `set_role guest`), но уже существующий человек роль не теряет. Новых таблиц не нужно — схема 14 уже разрешает `memberships.role='guest'`. **Чего нет:** живого микрофона, камеры и Telegram в песочнице нет — проверены слова, качество кадров, поток, транзакция и кнопки на настоящей схеме БД и настоящем `Connection` (подставлены камера/лицо-движок/Telegram); живой прогон с гостем — на стенде (P2-38). «Забудь меня» (F-213) — задача P2-23, здесь гость только сообщает о ней | `hub/guest_registration.py` (`declaration`, `assess_burst`, `Shot`, `GuestRegistration`, `OwnerConfirmations`, `commit_guest`, `revoke_guest`), `hub/speaker.py::ROLE_GUEST`, `hub/app.py::_guest_turn`/`_guest_start`/`_guest_voice_step`/`_guest_face_capture`/`_guest_consent_step`/`_guest_open_owner_question`/`_guest_owner_decision`/`_guest_commit`/`_TelegramHandlers`/`_guest_shot`, `common/config.py::GuestConfig` + `server.identity.guest` в обоих шаблонах, `tests/test_guest_registration.py` — 59 passed |
| F-211 Адаптивное дообучение | готово в коде: два предела ТЗ (до 12 векторов лица, до 8 голоса, тело — по дням, `body_per_day = 4`) и два условия (`p ≥ 0,9`, сильнее порога «узнан» 0,8 из F-207, и отсутствие конфликта с другими людьми) живут в `hub/adaptive_learning.py` и настраиваются в `server.identity.learning`. Привязка трека к человеку (F-204/F-205/F-210) проставляет `person_id` сразу — этого требует F-208 («тело того же дня, уже привязанное к лицу»), — поэтому F-211 в этой схеме надзирающий проход: подтверждение при p ≥ 0,9 запускает `review`, который удаляет из профиля вектор, неотличимый от вектора ДРУГОГО человека (cos ≥ 0,6: он делает следующий матч лотереей), и всё сверх пределов, начиная с самого старого (профиль — окно, а не стена: человек, сменивший причёску, должен оставаться узнаваемым). Дни тела не смешиваются, чужой профиль не трогается, `learn` добавляет НОВЫЙ вектор человека по тем же правилам (человек обязан быть в `persons`, неизвестный `track_id` не пишется вместо FK), а пределы реестра людей (`hub/speaker.py`, было 10/30 из фазы 1) приведены к числам ТЗ 12/8, чтобы профиль хаба и профиль реестра не расходились. Границы p, конфликт раньше пределов, дубликат своего вектора, окно 12/8, тело по дням и хук в `_confirm_track_identity` покрыты тестами. **Чего нет:** живого GPU/камеры в песочнице нет — правила и запись проверены на настоящей схеме БД; влияния дообучения на точность (набор 15.6) — P2-36 | `hub/adaptive_learning.py` (`AdaptiveLearning`, `Candidate`, `Verdict`, `decide`, `cap_for`, `cosine`, `caps_summary`), `hub/app.py::_adaptive_learning`/`_review_profile`, `hub/speaker.py` (пределы 12/8), `common/config.py::LearningConfig` + `server.identity.learning` в шаблонах, `tests/test_adaptive_learning.py` — 28 passed |
| F-212 Общий профиль между домами | готово в коде: согласие человека живёт во флаге `memberships.share_identity` (схема 14) и проверяется одним правилом `hub/shared_identity.py::visible_in`: в своём доме (самое раннее членство — там его зарегистрировали) узнают всегда; в другом доме хаба, где он в членстве, — только если он разрешил; дом, где членства нет, не видит его никогда (согласие не создаёт членство); строка `persons` без членств сохраняет поведение фаз до F-212, потому что флага у неё нет. Скрытый человек не становится именем трека (F-204), не привязывает голос к телу (F-205) и не даёт сигнала в слиянии (F-206) — лицо/голос продолжают храниться как эвиденция без `person_id`, а «ожидаемые здесь» для D-06 — это члены дома, прошедшие проверку. Разрешение и отказ человек говорит сам («разреши/не узнавай меня в других домах», ru/en/es, отказ побеждает), меняет флаг только распознанный говорящий и только у себя, изменение идёт в аудит (`identity.share`/`identity.unshare`), повторное значение — не событие; `set_share_identity` принимает только существующее членство. Ошибка чтения БД = «не разрешено» (согласие, которое нельзя доказать, не согласие), а хаб без БД сохраняет поведение фазы 1. **Чего нет:** кнопки в Telegram-панели (придёт с админкой F-216/P2-26) и живого стенда из двух домов с камерами — правила проверены на настоящей схеме БД и настоящем `Connection` | `hub/shared_identity.py` (`visible_in`, `visible_people`, `member_ids`, `primary_home`, `shared_with`, `set_share_identity`, `share_command`, `describe`), `hub/app.py::_person_visible_here`/`_share_identity_turn`/`_identity_context`/`_record_track_faces`/`_bind_speaker_to_track`/`_fuse_identities`, `tests/test_shared_identity.py` — 18 passed |
| F-213 «Забудь меня» | готово в коде: хаб спрашивает своим текстом и удаляет только по «да» следующей репликой — тот же механизм, что F-113 (`server.confirmations.window_s`, 8 с по умолчанию), поэтому вопрос произносит хаб, а не модель: необратимое действие нельзя перефразировать. Следующая реплика разбирается как ОТВЕТ (`_resolve_forget` вызывается сразу после STT, рядом с `_resolve_confirmation`/`_resolve_pin`) и в модель не уходит; «нет» и любой другой ответ отменяют, истёкшее окно не удаляет ничего («окно кончилось, я ничего не удалил»), а просить может только РАСПОЗНАННЫЙ говорящий — аутентификация здесь голос, незнакомец слышит «я запомню тебя, когда узнаю» и ничего не удаляется. Слова понимаются на ru/en/es, и «не забывай меня» / «don't forget me» / «no me olvides» просьбой НЕ считается. `hub/forget_me.py::erase` стирает человека по `person_id` (не по имени: имя может произнести кто угодно) одной транзакцией, ровно как требует раздел 14: `voice_embeddings`, `face_embeddings`, `body_embeddings`, `daily_appearance` (F-209), упоминания в `presence_events` и `dialog_turns`, кропы `body_crops` вместе с JPEG-файлами и записями `media`, членства и строку `persons`; `tracks.person_id` обнуляется — трек это наблюдение камеры, а не данные человека, и он остаётся безымянным. Затем то, что хранят хранилища фаз 1–2: память человека (`Memory`), архив диалогов (`Conversations.forget`), локальный архив обучения (`TrainingArchive.forget` — строки и папки) и реестр людей. Никакой корзины: данные не помечаются удалёнными, а удаляются; ошибка БД откатывает транзакцию и оставляет человека на месте (`ok=False`, честная фраза «не получилось»), а каждый успех пишет `identity.forget` в аудит с числами (`detail.vectors`, кропы, файлы, упоминания, память) — «сколько именно» важнее, чем «готово», и именно эти числа произносит отчёт. `_pending_forget` снимается при dismiss и close. **Чего нет:** половина «или в Telegram» — запрос из чата владельца и уведомление об удалении придут с F-701 (P2-32, многодомность Telegram), как и кнопка F-212; живого микрофона, камеры и Telegram в песочнице нет — слова, вопрос, транзакция и ход проверены на настоящей схеме БД и настоящем `Connection` (подставлены TTS и сокет), живой прогон — на стенде | `hub/forget_me.py` (`forget_requested`, `question`, `cancelled`, `confirmation`, `Inventory`, `inventory`, `erase`, `ForgetReport`, `_erase_crops`, `_unregister_media`, `_erase_stores`), `hub/conversations.py::Conversations.forget`, `hub/training_archive.py::TrainingArchive.forget`, `hub/app.py::_forget_turn`/`_resolve_forget`/`_erase_person`/`_forget_window`/`_pending_forget`, `tests/test_forget_me.py` — 15 passed |
| F-214 Anti-spoofing | готово в коде: два разных механизма ТЗ живут в `hub/anti_spoofing.py`. **Лицо** — бёрст последних кадров трека (то, что клиент и так присылает; `_note_liveness` собирает его и судит в рабочем потоке) проверяется тремя признаками, названными в ТЗ: *муар* (экран даёт в спектре кадра узкие пики; решает произведение «доля высоких частот × пиковость», поэтому шум сенсора — высокочастотный, но плоский — за экран не считается; на синтетике кожа 0,01–0,04, шум 0,10–0,13, решётка 1,00 при пороге 0,35), *микродвижения* (у живого лица между кадрами всегда меняются соотношения пяти ключевых точек — дыхание, мимика, моргание; у распечатки или фотографии на экране они застывают; считается и по точкам, и по пикселям, причём кроп приводится к общему размеру, поэтому дрожание камеры движением лица не считается) и *плоскость* (проекция любой плоскости под любым движением камеры есть ровно гомография, а у живого 3D-лица нос выдвинут из плоскости и остаток подгонки растёт: на синтетике фотография даёт остаток 0,000 при сдвиге 3,9 % своего размера, лицо с носом — 0,185 при 7,8 %; поэтому признак работает только когда в бёрсте что-то заметно сдвинулось). Бёрста короче пяти кадров недостаточно — «судить не о чем» это не «живой». Вердикт «спуф» пишется в аудит (`identity.spoof`) с признаками, снимает с трека имя и выкидывает лицо и голос из свидетелей F-206/F-208: кадр спуфа хранится как эвиденция без `person_id`, а спуф не может быть вторым свидетелем привилегированного вызова. **Голос** — challenge-слово для привилегированных действий: хаб сам выбирает случайное слово из восьми (ru/en/es) на каждый вызов, человек его произносит, и проверяются два независимых факта — что произнесено именно это слово (транскрипт, с допуском на ошибку whisper в одну букву) и что это ТОТ ЖЕ голос (ECAPA-сходство с собственным профилем человека, а не с населением комнаты, порог `challenge_voice_threshold` 0,5). Ход повторяет путь F-208/F-213: следующая реплика — ОТВЕТ (разбирается сразу после STT, в модель не уходит), окно 20 с, отказы различимы (`word`, `voice`, `no_voice`, истёкшее окно), вердикт с числами идёт в аудит `confirm.challenge`, а выполняется ровно тот вызов, который ждал. Режим `server.identity.anti_spoofing.challenge`: `off` (поведение фазы 2 — F-208 отказывает сам), `on_missing_witness` (по умолчанию — спросить слово, когда свидетелей F-208 нет) или `always` (спрашивать всегда, даже когда лицо подтверждено); `face` выключает проверку бёрста. **Чего нет:** нейросетевой liveness-модели ТЗ (insightface anti-spoof / MiniFASNet) в сборке нет — и это честно, а не заглушка: `load_model` бросает `SpoofModelUnavailable`, а `require_model` (по умолчанию `false`) делает отсутствие модели ОТКАЗОМ (`no_model`), записанным в раздел 17 ТЗ; живого микрофона и камеры в песочнице тоже нет — признаки, вердикты и ход проверены настоящей математикой по синтетическим кадрам и настоящим `Connection`; пороги признаков сняты на синтетике и калибруются на стенде (P2-36) | `hub/anti_spoofing.py` (`moire_score`, `motion_score`, `planarity`, `assess_burst`, `LivenessVerdict`, `words`/`choose_word`/`challenge`/`ask`/`heard`, `speaker_similarity`, `verify`, `ChallengeVerdict`, `load_model`, `SpoofModelUnavailable`), `hub/app.py::_note_liveness`/`_prune_liveness`/`_report_spoof`/`_is_spoofed`/`_anti_spoofing`/`_challenge_mode`/`_request_challenge`/`_audit_challenge`/`_resolve_challenge`/`_model_of_liveness` + правки `_record_track_faces`/`_admin_evidence`/`_admin_strength_check`/`_handle_utterance`/`_match_presence`, `hub/speaker.py::VoiceRegistry.vectors_of`, `common/config.py::AntiSpoofingConfig` + `server.identity.anti_spoofing` в обоих шаблонах, `tests/test_anti_spoofing.py` — 37 passed |
| F-215 Объяснимость | готово в коде: `hub/explain.py` собирает ответ ТОЛЬКО из полей belief, который записал F-206, поэтому досочинять нечего — каждое предложение либо число из строки `identity_belief`, либо честное «этого я не видел». Сигналы, которые были, называются с их числами («голос 0,71, лицо 0,9»), сигналы, которых не было, — как отсутствующие («лицо не видно», «тела того же дня нет»), уверенность берётся из `p`, время решения — из `at` («12 секунд назад», «только что», «2 минуты назад», с правильными формами ru/en/es), числа пишутся так, как пишет сам ТЗ (запятая в ru/es, точка в en). Ничья (`ambiguous`) и решающий контекст дома (`context`) называются отдельными фразами; причина отказа, которую пишет F-206/D-06, переводится на язык человека вместо английской строки из БД. Вопрос `why_question` понимается на ru/en/es и различает «почему ты решил, что это Макс» (имя извлекается) и «почему ты думаешь, что это я» (про самого говорящего); обычная речь вопросом не считается, поэтому перехвата нет. `hub/app.py::_explain_turn` ищет решение по живому треку человека (F-207), а если трека нет — по последнему belief этого человека в доме, и отвечает числами; про человека, которого хаб не знает вовсе, честно говорит «этого я не решал». Модель в этом ходе не участвует совсем: объяснение идентичности — это не текст, который модель может переписать. **Чего нет:** живого GPU/камеры в песочнице нет — ответ проверен на настоящей схеме БД и настоящем `Connection`, озвучка идёт обычным TTS-путём | `hub/explain.py` (`why_question`, `WhyQuestion`, `explain`, `seen_and_missing`, `age_phrase`, `number`, `KIND_LABELS`, `REASON_TEXT`), `hub/app.py::_explain_turn`/`_belief_for`, `tests/test_explain.py` — 15 passed |
| F-216 Ручная разметка треков | готово в коде: очередь неопознанных треков за день живёт в админке владельца (`/admin/tracks`), а клик владельца привязывает трек к человеку. Метка — это данные, а не только исправление: строка `identity_labels` (миграция 0008) хранит, кто, когда, в каком доме и по какому кропу сказал, что этот трек — этот человек, поэтому по меткам можно пересчитать пороги слияния и собрать проверочный набор 15.6, не разбирая аудит задним числом. Клик делает три вещи и честно говорит, что именно: ставит `tracks.person_id` (F-204/F-205 после этого не гадают), привязывает к человеку нераспределённые сэмплы ТОГО ЖЕ дня (лицо и голос — все, тело — по `session_day`: вчерашняя одежда к сегодняшней метке отношения не имеет) и пишет строку метки; успех и провал идут в аудит `identity.label` с числами, а ошибка БД откатывается — метка не появляется. В очереди владелец видит не только миниатюру: кропы того же дня, числа лица/тела/голоса, лучшее качество и belief F-206 с его числами (`p`, сигналы), поэтому решение принимается по эвиденции, а F-215 после клика объясняет уже новое решение. Кроп отдаётся только из `data` (путь из БД не может увести веб-сервер в чужую папку), а сама разметка остаётся за overlay-гейтом и сессией, как вся панель F-705. **Чего нет:** живого клика в браузере на стенде (панель поднимается в деплое) и кнопки привязки в Telegram — она придёт с F-701/P2-32; пороги по накопленным меткам пересчитываются задачей 15.6 (P2-36) | `hub/labelling.py` (`queue`, `label`, `labels`, `summary`, `crop_file`, `day_of`, `day_window`), `migrations/0008_identity_labels.py`, `hub/web_admin.py` (`WebAdminData.unknown_tracks`/`labelling_people`/`save_label`/`crop_path`, `/admin/tracks`, `POST /admin/label`, `/admin/crop/{id}.jpg`), `hub/templates/admin/tracks.html` + `base.html`/`dashboard.html`, `tests/test_labelling.py` — 10 passed, `tests/test_web_admin.py` — 24 passed |
| F-301 Состояние присутствия дома | готово в коде: хаб держит `presence(home_id)` — кто в комнате (человек или безымянный трек), с какого времени (`since`, то есть когда тело появилось, а не когда его узнали), какой кадр был последним (`last_seen`) и в какой зоне он стоит, — и пишет из этого ровно четыре события ТЗ в настоящую таблицу `presence_events` (схема 14): `person_entered`, `person_left`, `unknown_appeared`, `zone_entered`. Решения по умолчанию там, где ТЗ молчит (`DECISIONS.md` P2-27): трек, который сначала был незнакомцем, а потом получил имя, даёт `person_entered` (F-302 здоровается именно по нему, а «незнакомец появился» остаётся правдой о прошлом); про ушедшего безымянного трека события нет, потому что ТЗ называет четыре вида и «незнакомец вышел» в них не входит; `zone_entered` пишется при СМЕНЕ зоны, а первое появление несёт зону в событии входа; сколько секунд без кадров считать выходом — `server.presence.absence_s` (30 с), потому что число зависит от комнаты и камеры, а не от кода. Вопросы F-301 отвечаются из записей, и модель в них не участвует: «кто дома» берёт живое состояние (имена с временем входа плюс число незнакомцев), «Макс заходил сегодня» — события дня человека (во сколько зашёл, во сколько вышел и сколько был, сколько визитов, «и сейчас здесь»; вчера тоже понимается), «сколько я был за столом» — интервалы зоны, собранные из `zone_entered` до следующего перехода или до `person_left`; имя зоны сопоставляется со словами владельца плюс синонимы ru/en/es. Честность важнее полноты: нет кадров — «кадров с камеры нет», свежий `camera_state` с нулём — «никого не вижу», зоны не размечены — «их размечает владелец», спрашивающий не узнан — «я тебя не узнал», чужое имя — «не знаю человека по имени». **Чего нет:** живого стенда с камерой и нарисованными зонами (зоны — F-309, фаза 3; проверено на синтетических событиях и на контракте `tracks`) и потребителей событий: правила и уведомления — F-702 (P2-33), приветствие по `person_entered` — F-302 (P2-28) | `hub/presence_state.py` (`Sighting`, `Occupant`, `PresenceEvent`, `PresenceState.observe`/`occupants`/`known`/`unknown`/`forget`, `PresenceLog.record`/`day`/`zones`, `zone_spans`, `day_of`, `day_bounds`), `hub/presence_questions.py` (`parse`, `Question`, `Facts`, `answer`, `duration`, `match_zone`, `clock_of`, тексты ru/en/es), `hub/app.py::_presence_state`/`_presence_log`/`_remember_zones`/`_presence_sightings`/`_observe_presence`/`_sight_status`/`_presence_turn` + вызов в `_match_presence` и в цепочке хода, `common/config.py::PresenceConfig` + `server.presence` в обоих шаблонах, `tests/test_presence_state.py` — 37 passed |
| F-302 Приветствия и прощания | готово в коде: приветствие произносит сам хаб по событию входа (`person_entered` из F-301), пока человек стоит перед камерой, поэтому строки фиксированные и на модель не опираются. Персонально: по имени, на языке человека (язык говорящего, а до первого слова — язык комнаты); незнакомцу имени не дают вовсе — он слышит знакомство Rowan и имена тех, кого хаб видит рядом. С учётом времени суток: утро 05–12, день 12–18, вечер 18–23, ночь 23–05 в МЕСТНОМ времени дома (`homes.tz`), по две формулировки на кусок, чтобы человек не слышал одно и то же слово каждый раз. С учётом тихих часов: окно `HH:MM` в поясе дома (умеет через полночь), в нём хаб молчит. Число ТЗ «не чаще одного раза в 20 минут на человека» — это `server.greeting.cooldown_s` (1200 с): он поднимает порог отсутствия для знакомого лица (`max(server.face.greeting_cooldown_known_s, server.greeting.cooldown_s)`), а `server.greeting.enabled: false` возвращает ровно поведение фазы 2 (900 с, без тихих часов) — новые правила включаются флагом, а не заменой старого. Из TTS-кэша: `hub/tts.py::TtsCache` хранит готовый PCM коротких фиксированных реплик (по тексту и частоте, ограничен по записям и байтам, потокобезопасный), `_stream_tts(..., cache=True)` берёт приветствие и прощание оттуда — фраза звучит сразу, а не после синтеза на CPU; счётчики видны в `/health` (`tts_cache`). Прощания: по событию `person_left` (F-301) хаб называет ушедшего по имени (имя берётся из `persons`, а не из догадки), держит то же правило 20 минут на человека, молчит в тихие часы и отключается флагом `farewell: false`. **Чего нет:** живого стенда с камерой и колонкой — строки, окна времени, порог и кэш проверены на подставном синтезе и настоящем `_may_greet`; живая озвучка — стенд (P2-36…P2-38), уведомления по событиям F-301 — F-702 (P2-33) | `hub/greetings.py` (`greeting`, `farewell`, `stranger_greeting`, `part_of_day`, `in_quiet_hours`, `window`, `COOLDOWN_S`, тексты ru/en/es), `hub/tts.py::TtsCache` (`get`/`put`/`clear`/`stats`), `hub/app.py::_scripted_greeting`/`_greeting_language`/`_home_timezone`/`_in_quiet_hours`/`_greet_config`/`_may_say_bye`/`_schedule_farewell`/`_say_farewell` + `_stream_tts(cache=…)` + `/health`, `common/config.py::GreetingConfig` + `server.greeting` в обоих шаблонах, `tests/test_greetings.py` — 19 passed |
| F-303 Privacy-режим камеры | готово в коде: «Rowan, перестань смотреть» перестаёт отдавать кадры, «смотри снова» возвращает их — и то и другое голосом, при работающем микрофоне (иначе камеру нельзя было бы вернуть тем же голосом). Флаг живёт НА КЛИЕНТЕ (`client/privacy.py`), потому что кадры уходят именно с него: в приватном режиме не отправляется ни присутствие, ни треки, ни кропы, а запрос хаба на кадр получает честный `camera_error` («камера выключена»), не пустой кадр; рабочий поток детекции в этом режиме вообще не запускает YOLO — камера перестаёт смотреть, а не просто молчит. Программный индикатор обязателен и есть: постоянный красный бейдж с перечёркнутой камерой в углу HUD (`OverlayHUD.camera_privacy` → `window.hudState({camera_off})`), который не гаснет, пока режим включён, и учитывается в видимости оверлея. Хаб (`hub/privacy.py`, `_privacy_turn`) понимает просьбу на ru/en/es и отличает её от обычной просьбы посмотреть кадр («посмотри, кто там» — это не приватность): отправляет клиенту `privacy`, помнит состояние комнаты, отбрасывает кадры присутствия, даже если клиент их всё же пришлёт, и пишет аудит `privacy.on`/`privacy.off`. Клиент говорит о своём состоянии сам — `hello.privacy` и `camera_state.privacy` после переподключения, — поэтому хаб верит факту, а не своим воспоминаниям. **Чего нет:** аппаратного индикатора (USB-реле или светодиод) — это P2 в ТЗ; живого стенда с камерой и микрофоном (проверено на настоящих классах клиента, протоколе и `Connection`) | `client/privacy.py` (`PrivacyMode`, `INDICATOR`, `confirmation`), `client/camera.py::set_privacy`/`announce_privacy` + проверки в `_publish_state`/`_publish_tracks`/`_publish_body_crops`/`_maybe_push_presence`/`serve_request`/`_infer_loop`, `client/overlay.py::camera_privacy`, `client/overlay_web/hud.html` (`#camera-off`), `client/main.py` (`_apply_privacy`, `_announce_privacy`, `build_hello(privacy=…)`), `hub/privacy.py` (`privacy_command`, `confirmation`, `looks_like_look_request`), `hub/app.py::_privacy_turn` + отказ в `_buffer_presence_frame`, `common/protocol.py::MSG_PRIVACY`/`Privacy`/`Hello.privacy`, `scripts/export_client.py`, `tests/test_privacy_mode.py` — 41 passed |
| F-304 TTL медиа | готово в коде: TTL живёт задачей планировщика, а не разовым проходом при старте. `hub/scheduler.py` — сам планировщик хаба (`Job` с именем, интервалом и признаком `threaded`; `Scheduler` держит по циклу на задачу, считает проходы, хранит последний отчёт, зовёт `on_error`/`on_report`, переживает падение отдельного прохода и останавливается без следов). `hub/media.py::MediaTtlTask` (имя `media.ttl`) удаляет просроченные файлы и их строки в `media`: кадры и кропы через `media_ttl_days` (3 дня), клипы через 7 (числа ТЗ 15.4 «кропы и кадры 3 дня, клипы 7 дней»); **эмбеддинги (`body_embeddings`, `face_embeddings`) и `presence_events` не трогаются вовсе** — ТЗ 15.4: «эмбеддинги и события — до „забудь меня“»; файл вне `<home>/media` не удаляется никогда (такая строка видна в отчёте как `unsafe_skipped`). Отчёт: одна строка аудита на проход, в котором было что удалять (`expired_rows`, `deleted_files`, `missing_files`, `unsafe_skipped`, оба TTL), пустой проход `audit` не трогает; упавший проход пишется как `result='failed'` с текстом ошибки. Хаб собирает задачу в `_media_ttl_scheduler` и держит её в lifespan рядом с watcher'ом навыков; период — `server.media.cleanup_interval_s` (3600 с, `0` = только проход при старте), состояние видно в `/health.scheduler`. Проход идёт на цикле событий, потому что `data/hub.db` открыт на нём (DECISIONS P1-44). **Чего нет:** живого стенда (медиа немного, файлы удаляются на месте) | `hub/scheduler.py`, `hub/media.py::MediaTtlTask`, `hub/app.py::_media_ttl_scheduler`/`_scheduled_job_failed`/`_scheduler_snapshot` + запуск в lifespan, `hub/main.py::_cleanup_media` (тот же отчёт на старте), `common/config.py::MediaConfig.cleanup_interval_s` + `config.yaml`/`config.example.yaml`, `tests/test_scheduler.py` — 10 passed, `tests/test_media_ttl_job.py` — 11 passed |
| F-312 Ускорение клиента | готово в коде: «Экспорт YOLO11x в TensorRT (FP16) для 3060 Ti; профиль для слабых ПК: YOLO11s и треки 10 FPS. Автовыбор профиля по измеренной задержке при старте клиента». Экспорт — `scripts/export_tensorrt.py` (реальный `model.export(format="engine", half=True)` + замер движка настоящими выводами и JSON-отчёт; `--check` только докладывает о машине; без ultralytics/CUDA/TensorRT скрипт печатает, чего не хватает, и выходит с кодом 2 — «успех» без движка не выдаётся). Профили — `client.camera.profiles` в шаблонах клиента: `tensorrt` (yolo11x.engine, FP16, бюджет 30 мс), `gpu` (yolo11x.pt, FP16, 90 мс), `weak` (yolo11s.pt, 10 FPS, без FP16 — ровно то, что называет ТЗ). Автовыбор — `client/vision_profile.py` + `client/camera.py`: при старте клиента каждый кандидат грузится и меряется на одном кадре (`measure_ms`: прогрев не считается, дальше медиана — задержка никогда не берётся из имени модели), выбирается первый уложившийся в свой бюджет; профиль без собранного на этой машине `.engine` пропускается с причиной, упавший замер (нет cuDNN/CUDA) пропускается, а не считается быстрым; если измерить нечего — клиент остаётся на `model`/`fps`/`half` из конфига, то есть без камеры не остаётся. Флаг `auto_profile` по умолчанию `false`: выключенный автовыбор оставляет поведение фазы 2, включённый не заменяет конфиг, а выбирает из него. Выбор, задержка и причины пропусков видны в логе и в `data/camera-performance.json` (`profile`, `profile_latency_ms`, `profile_reason`, `profile_attempts`). **Чего нет:** живого GPU в песочнице (в env нет ultralytics и TensorRT) — логика выбора, замер и интеграция проверены на настоящем коде камеры с подставным замером; сам экспорт и живой замер на 3060 Ti выполняются одной командой | `client/vision_profile.py`, `client/camera.py::_select_profile`/`_probe_frame` + `stats()`, `scripts/export_tensorrt.py`, `common/client_config.py::VisionProfileConfig`/`default_vision_profiles` + `config.client.example.yaml`, `distribution/client/config.example.yaml`, `scripts/export_client.py` (`RUNTIME_FILES`), `tests/test_vision_profile.py` — 21 passed |
| F-701 Многодомность в Telegram | готово в коде: владелец дома — это Telegram-аккаунт, названный в конфиге (`homes[].telegram_user_id`) или выданный админом хаба со страницы «Homes and owners» (хранится в `telegram_home_owners`, засев из конфига идемпотентен и никого не отзывает). Владелец открывает /tools в СВОЁМ приватном чате, и панель показывает только его дома: `workplaces.list` фильтруется по дому рабочего места (живое подключение и запись при hello), чужой дом отбивается словами, действие без явного дома относится к единственному дому владельца, общехабные разделы (настройки, люди, память, аккаунты Telegram, аудит, калибровка) ему не выдаются и в меню, и в бэкенде; группы по-прежнему только для админа хаба. Глобальный админ (`server.telegram.control_user_id`) видит ВСЕ дома и раздаёт их по Telegram ID. Кнопки панели остались прежними: одноразовые, с TTL 15 минут и привязкой к сообщению/поколению (тест на истечение есть). **Чего нет:** живого бота и настоящих чатов в песочнице (маршрутизация проверена на настоящих классах с подставным транспортом); раздачи домов из веб-админки F-705 нет — это конфиг и панель Telegram | `hub/telegram_homes.py`, `hub/telegram_admin_state.py` (`telegram_home_owners`, `homes_of`/`grant_home`/`revoke_home`/`owners_of`), `hub/telegram_admin.py` (`_may_panel`/`_route`/страница `homes`, меню по области видимости), `hub/admin_backend.py` (`get_scope`/`get_workplace_home`, `HOME_SCOPED`, фильтрация списков, `homes.*`), `hub/app.py` (`_workplaces` с `home_id`, `_workplace_home`, `HomeOwners` в lifespan), `common/config.py::HomeConfig.telegram_user_id` + `config.example.yaml`, `tests/test_telegram_multihome.py` — 18 passed |
| F-702 Уведомления по правилам | готово в коде: правило уведомления слушает СОБЫТИЕ, а не только «человек в кадре». Виды событий: прежнее `presence` (поведение по умолчанию), события F-301 (`person_entered`, `person_left`, `unknown_appeared`, `zone_entered`), F-109 (`sound_event` — клиент присылает метку и уверенность, аудио не уходит; хаб принимает сообщение по протоколу `MSG_SOUND_EVENT` и отбрасывает слабые сигналы ниже 0.5) и F-311 (`object` — появление объекта внимания; событие рождается на переходе «метки не было — метка появилась», а не каждый кадр). Событийный путь (`PresenceAlerts.observe_event`) не требует стабильности в кадре, но держит ту же очередь, тот же кулдаун и те же тихие часы, что и присутствие. «Кулдаун и тихие часы — на дом»: тихие часы дома (`homes[].quiet_hours` в поясе дома) молчат сильнее правила, `homes[].alert_cooldown_s` — минимум кулдауна дома (правило может быть реже, но не чаще). «Выбор канала»: `telegram` (как раньше, с подтверждением доставки), `hud` (подпись на экране комнаты существующим сообщением `status`, без медиа) и `push` (транспорта нет до F-712 — доставка честно помечается `skipped` с причиной, а не «отправлено»). **Чего нет:** детектора звука на клиенте (F-109 — фаза 3) и пуша (F-712 — фаза 3): путь хаба готов и проверен синтетическими событиями; зоны F-311 (F-309) — фаза 3, поле `zone` уже проверяется | `hub/presence_alerts.py` (`EVENT_KINDS`, `CHANNELS`, `DEFAULT_RULE.event/channel/home_id/zone`, `observe_event`, `_event_match`, `home_settings`/`home_quiet_now`/`home_cooldown`, `_deliver_hud`/`_deliver_push`), `hub/app.py::_on_sound_event`/`_observe_objects`/`_alert_home` + события F-301 в `_observe_presence`, `common/protocol.py::MSG_SOUND_EVENT`, `common/config.py::HomeConfig.alert_cooldown_s` + `config.example.yaml`, `hub/telegram_admin.py`/`hub/telegram_admin_view.py` (поля редактора), `tests/test_alert_rules.py` — 14 passed |
| F-708 HUD v2 | готово в коде: экран комнаты показывает состояние самого Rowan — «слушаю / думаю / говорю» (существующие состояния), имя узнанного говорящего (`speaker`), строку живого транскрипта ТЕКУЩЕЙ реплики (черновой текст курсивом; в конце хода и при «тишине» строка снимается), окно follow-up с оставшимися секундами (F-103) и постоянный значок выключенной камеры (F-303). Статус хаба — новое сообщение `hub_status` (`online` / `queue` / `offline` + число ожидающих задач): хаб считает его по СВОЕЙ очереди GPU (`queue` = «работа ждёт видеокарту», идущая задача оставляет `online`) и присылает комнате при подключении и при каждом изменении очереди (`GpuQueue(on_change=…)`), повторы одного состояния не отправляются, кадр объявлен background-классом; `offline` хаб не выдумывает — это вывод клиента по разорванному сокету. «Имена над треками по запросу „покажи камеру“»: фраза понимается на ru/en/es и НЕ путается с вопросом о кадре («посмотри, кто там» — это `look_at_camera`, F-314); хаб тянет настоящий кадр, подписи берёт из треков ЭТОГО кадра, имена — только из уже известных имён треков комнаты (F-201/F-204), безымянный трек остаётся без подписи, модель в ходу не участвует вовсе. Кадр уходит на экран комнаты сообщением `image_show` с полем `tracks`, клиент рисует подписи ровно столько, сколько живёт снимок, и считает их по рамке кадра (полноэкранный снимок держит пропорции). Новые хуки страницы (`hudHub`, `hudTracks`, `hudTranscript({clear})`) есть на обеих страницах HUD — `hud.html` и `chat.html`, которая и загружается в WebView. **Чего нет:** живого экрана (WebView) и камеры в песочнице — проверены настоящие классы (страницы, `OverlayHUD`, `Connection`, `GpuQueue`), но не картинка на телевизоре | `common/protocol.py::MSG_HUB_STATUS` + background-класс кадров, `hub/app.py::hub_status_frame`/`broadcast_hub_status`/`Connection.publish_hub_status`/`_hub_queue_changed`/`_track_names`/`_show_camera_turn`/`_send_image_show(tracks=…)`, `hub/gpu_queue.py::GpuQueue(on_change=…)`, `hub/camera_view.py` (`show_camera_requested`, `track_labels`, тексты ru/en/es), `client/overlay.py::hub_state`/`tracks`, `client/main.py` (маршрут `hub_status` в обоих режимах, `_show_track_labels`/`_clear_track_labels`, снятие транскрипта в конце хода), `client/overlay_web/hud.html` + `chat.html`, `tests/test_hud_v2.py` — 46 passed |
| Деградация клиента без хаба (4.8) | готово в коде: связь ищется ФОНОМ (комната не стоит в очереди на переподключение), по ТЗ «потеря соединения дольше 3 с» отдельный сторож показывает на экране «мозг оффлайн» (`hub_status: offline`, статус F-708) и произносит заранее принесённую хабом строку «Хаб недоступен, работаю локально» (ru/en/es) — своей TTS у комнатного ПК нет, поэтому `MSG_TTS_PREFETCH` просит хаб синтезировать нужные фразы при подключении, а `client/tts_cache.py` держит PCM на диске этого ПК (чего синтезировать нечем — честная ошибка без звука, а строка всё равно видна на экране). Пока хаба нет, комната продолжает работать сама: локальные команды и сцены (F-117) выполняются как раньше, а для произвольных фраз поднимается локальный faster-whisper base/small на своём GPU/CPU (`client/local_stt.py`: ленивая загрузка, `auto` пробует CUDA и честно отступает на CPU, модель не загрузилась — комната говорит об этом, а не делает вид, что поняла). Реконнект — экспоненциальный (1, 2, 4 … до 30 с; `client/offline.py::backoff_delay`), политика задана `client.offline`, транспорт без этих аргументов сохраняет прежние фиксированные 3 с. «После восстановления — досылка накопленных событий присутствия»: `client/presence_buffer.py` копит состояния присутствия (счётчик/треки) с меткой времени клиента, после подключения они уходят хабу по порядку с `replay: true`; хаб ставит их в свою шкалу по метке клиента (правдоподобное прошлое, иначе «сейчас»), восстанавливает состояние комнаты и НЕ поднимает уведомления по событиям давности — «кто-то вошёл» пять минут назад не новость; кадры не копятся вовсе (старый JPEG — это ложь о том, как комната выглядит теперь). Уходя на обслуживание, хаб сам предупреждает комнаты (`MSG_OFFLINE_HINT` в lifespan), чтобы те перешли в локальный режим до разрыва связи. **Чего нет:** живого железа (микрофона, отсутствующего хаба, CUDA) в песочнице нет — правила, кэш, буфер, ход хаба и локальный ход клиента проверены на настоящих классах; распознанный за минуты простоя участок присутствия не восстановить (кадров лиц за это время нет) — это записано честно | `client/offline.py`, `client/local_stt.py`, `client/tts_cache.py`, `client/presence_buffer.py`, `client/ws_client.py` (`backoff`/`before_retry`), `client/main.py` (`_reconnect_loop`, `_offline_loop`, `_enter_offline_mode`, `_say_offline_notice`, `_flush_presence`, `_request_phrases`, `_on_tts_phrase`, `_on_phrase_audio`, `_on_offline_hint`, `_local_turn`, `_local_transcript`), `client/camera.py::_keep_for_replay`, `common/protocol.py::MSG_TTS_PREFETCH`/`MSG_TTS_PHRASE`/`MSG_OFFLINE_HINT`, `common/client_config.py::OfflineConfig`, `hub/app.py::_on_tts_prefetch`/`announce_offline_hint`/`_frame_timestamp` + replay в `_on_camera_state`, шаблоны клиента, `scripts/export_client.py`, `tests/test_offline_mode.py` — 43 passed |
| Сценарий 1: сцена «кино» по голосовой команде за < 2 с | готово: фраза сценария («Rowan, выключи свет и включи фильм», а также en/es-варианты) узнаётся САМА, без круга модели: `hub/scenes.py::cinema_request` требует три слова сразу — что свет назван, что его просят выключить и что назван фильм, — и `match_scene` отдаёт по ней пресет «кино» этого дома. Поэтому «включи свет» и «расскажи про кино» остаются обычным ходом, как и раньше. Замер идёт по НАСТОЯЩЕМУ ходу (`scripts/measure_scene_latency.py`: настоящие `SceneStore`-пресеты, устройства, `Connection._handle_utterance`; модель подставлена так, что её вызов валит замер). Итог: хаб отвечает за **6 мс** (медиана; p95 33 мс) против бюджета 2,0 с, объявленные движки комнаты (`--stt-ms`/`--tts-ms`, время стенда) входят в отчёт отдельным слагаемым, а превышение стадийного бюджета хаба (ТЗ 15.1) видно строкой и кодом 1, а не удобным числом. **Чего нет:** живых STT/TTS и 5090 в песочнице — время движков комнаты параметр, на стенде команда повторяется с фактическими временами (`DECISIONS.md` P2-37) | `scripts/measure_scene_latency.py`, `hub/scenes.py::cinema_request`, `make measure-scene`, `tests/test_measure_scene_latency.py` — 15 passed, `tests/test_scene_e2e.py` — 6 passed |
| Сценарий 2: друг заходит и слышит приветствие по имени, даже стоя сначала спиной | готово: пока лица нет, хаб молчит — спина это не имя и НЕ незнакомец (незнакомцем считается только свежее лицо, которое движок видел и не сопоставил, иначе отвернувшийся друг слышал бы «здравствуйте, незнакомец»). Когда человек поворачивается, лицо опознаётся, личность удерживается на ТОМ ЖЕ треке (F-204), и цикл приветствий (`_greeting_loop`) произносит имя скриптовым приветствием (F-302, без модели). Замер по настоящему циклу: **0,179 с** (медиана), p95 0,186 с против бюджета 2,0 с (ТЗ 15.3). **Чего нет:** живого стенда — проверены настоящие `Connection._match_presence`, `RoomState`, `PresenceTracker` и БД хаба, подставлены движок лиц и TTS (`DECISIONS.md` P2-38) | `scripts/measure_presence_latency.py`, `make measure-presence`, `tests/test_presence_scenarios.py` — 7 passed |
| Сценарий 4: незнакомец в отсутствие владельца → фото в Telegram за < 5 с | готово: незнакомое лицо в комнате идёт настоящим путём — трек, свежее непознанное лицо, правило уведомления дома (`target: unknown`, `destination: owner`) и настоящий `PresenceAlerts` с рабочим циклом; владелец получает фото (JPEG) с подписью «Rowan · <комната>: камера заметила — неопознанный человек» и временем. Замер: **2,18 с** (медиана), худшее 2,20 с против бюджета 5,0 с, и в это число входит требование ТЗ «человек стабильно в кадре» (`min_stable_s`, 2 с по умолчанию); замер с 5,5 с стабильности честно выходит кодом 1. **Чего нет:** живого Telegram и камеры — транспорт подставной и признаётся в доставке; отдельного признака «владельца нет дома» нет до телефон-клиента (F-711/F-712, фаза 4) | `scripts/measure_presence_latency.py`, `tests/test_presence_scenarios.py` — 7 passed |
| Сценарий 7: хаб недоступен — комната работает сама и говорит об этом | готово: связь пропала → через 3 с (`client.offline.after_s`, ТЗ 4.8) на экране «мозг оффлайн» и произносится строка, принесённая хабом ЗАРАНЕЕ (`PhraseCache`; звука нет — строку видно на экране, тишина за фразу не выдаётся). Дальше «выключи свет» распознаётся локальным распознавателем, разбирается настоящим `parse_local_command` и выполняется настоящим `LocalRunner` собственными руками клиента — хаб не участвует; нелокальная фраза и выключатель, которого клиент не знает, честно заканчиваются строкой про хаб, а не выдуманным выполнением. **Чего нет:** живого железа — подставлены микрофон, распознавание и само устройство, но не логика клиента; прогон «выдернуть сеть» — стенд | `tests/test_scenarios_offline.py` — 4 passed, `tests/test_offline_mode.py` — 43 passed, `tests/test_local_commands.py` |
| F-404 Vision-уровень и облачный fallback | готово в коде: «один локальный мультимодальный уровень для скриншотов и кадров плюс облачный fallback для сложных изображений при флаге дома `cloud_vision: true`». Зрение теперь уровень модели — `models.levels.local_vision` с провайдером, endpoint и моделью, а `models.routing.vision_level` указывает, какой уровень смотрит (по умолчанию `local_vision`); пока схема уровней выключена, отвечает классический `server.llm.vision_model`, и это записано в логе, так что обновление ничего не отнимает у работающего хаба. Облако открывается тремя замками: `models.routing.cloud_vision`, разрешение САМОГО дома (`homes[].cloud_vision`, по умолчанию `false`) и место в бюджете (`api_usage`); неизвестный бюджет = «нет». Уровень выбирает роутер (`ModelRouter.vision_pick` → `vision`/`vision_cloud`/`None`), локальный взгляд идёт через GPU-очередь как раньше, облачный — сетевой и GPU не занимает; нет ничего — инструменты честно говорят «модель зрения не загружена». Расход облака пишется в тот же журнал, что и текст: бронь до сети, сверка после, неудачный запрос бронь не возвращает. Сложность картинки (P3-03) решается ФАКТАМИ о самой картинке и о вопросе — `hub/vision_levels.py::image_is_hard`: несколько лиц в кадре, большой скриншот, мелкий текст, ИЗМЕРЕННЫЙ по перепадам строк (порог 0,10 откалиброван на синтетике: страница текста 0,16, фото 0,01, простое окно 0,02, градиент 0,00), вопрос про текст/таблицу/код на трёх языках; спорит об этом Decider типа `vision_level` (`ModelRouter.vision_choose`), но облачный уровень попадает в список вариантов только после всех трёх разрешений, поэтому провайдер не может отправить кадр из комнаты наружу самовольно. **Чего нет:** живого ключа и облака в песочнице — транспорт подставной (`httpx.MockTransport`), настоящая отправка картинки — стенд, а порог «мелкого текста» — первый замер на синтетике, его стоит перепроверить на настоящих скриншотах (`DECISIONS.md` P3-01…P3-03) | `hub/vision_levels.py` (`build_vision`, `image_is_hard`, `text_score`), `hub/vision_cloud.py`, `hub/model_router.py::vision_pick`/`vision_choose`, `hub/openai_responses.py::describe_image`, `hub/app.py::_describe_image`/`_vision_route`/`_cloud_vision_allowed`, `common/config.py` (`local_vision`, `vision_cloud_level`, `HomeConfig.cloud_vision`) + шаблоны, `tests/test_vision_levels.py` — 11 passed, `tests/test_vision_cloud.py` — 19 passed, `tests/test_image_difficulty.py` — 12 passed |
| Сценарий 9: двое говорят одновременно — Rowan просит повторить, а не выполняет смесь | готово (сделано в P2-08, F-108, проверено здесь заново): диаризация отдаёт интервалы одновременной речи, хаб считает их долю; больше 40 % (`server.diarization.overlap_limit`) — вопрос «повторите по одному» от самого хаба, при этом НИ ОДНОГО действия не выполнено и модель не вызвана, а смешанные слова не исполняются. Чистая реплика идёт обычным путём (ответ, а не жалоба) | `hub/overlap.py`, `hub/diarization.py`, `hub/app.py`, `tests/test_overlap_speech.py` — 14 passed (в наборе с локальными путями — 119 passed) |

## Замеры латентности (раздел 15.1)

| Что мерили | Команда | Результат |
|---|---|---|
| Порог перелива F-403 при трёх активных домах | `python scripts/measure_overflow.py --rate 4 --service-s 2.5 --per-home 20 --time-scale 10` | ожидание класса 0: медиана 0,00 с, p95 0,79–0,93 с, худшее 2,6 с; порог 0,55 с записан в `config.yaml` (P1-24) |
| Задержка, которую одна комната создаёт другой | `python scripts/measure_cross_room_delay.py --repeats 3` | планирование хаба: −0,002 с (быстрая комната 0,218 с одна, 0,215 с рядом с медленной, которая отвечает 2,0 с); очередь GPU: медиана 0, p95 0, max 2,59 с без F-403 и max 0,0 с при включённом F-403 (порог 0,55 с, 2 из 36 реплик ушли в облако). Порог приёмки 1,5 с выдержан |
| Сцена «кино» по голосовой команде (сценарий 1, бюджет 2 с) | `python scripts/measure_scene_latency.py --repeats 5` | хаб отвечает медиана **0,006 с**, p95 0,033 с, худшее 0,033 с — запас 1,97 с на распознавание и синтез комнаты; с объявленными движками комнаты (`--stt-ms 300 --tts-ms 200`) медиана 1,02 с, худшее 1,04 с. Модель в ходу не участвует (её вызов валит замер) |
| Вход человека → приветствие по имени (сценарий 2, бюджет 2 с) | `python scripts/measure_presence_latency.py --repeats 3` | друг входит спиной к камере (приветствия нет), поворачивается — приветствие по имени: медиана **0,179 с**, p95 0,186 с, худшее 0,186 с. В секундомер входит настоящий цикл приветствий (опрос 0,15 с) |
| Незнакомец → фото в Telegram (сценарий 4, бюджет 5 с) | `python scripts/measure_presence_latency.py --repeats 3` | медиана **2,18 с**, p95 2,20 с, худшее 2,20 с; из них 2 с — требование ТЗ «человек стабильно в кадре» (`min_stable_s`), фото уходит владельцу с подписью. Транспорт подставной (живого Telegram в песочнице нет) |

Оба замера идут по настоящим реализациям (очередь GPU, fair share, оценка
ожидания, планирование комнат), но время одной реплики на GPU — параметр:
стенда с RTX 5090 и тремя комнатами в песочнице нет, см. `DECISIONS.md`
(P1-24, P1-45). На стенде замер повторяется с фактическим временем реплики.

**Фаза 2 (речь, идентичность, присутствие) — начата.** Задачи P2-01…P2-40 в
`PROGRESS.md`, план — в `PLAN.md`. Ниже — карта функций ТЗ, чтобы видеть объём.

| Блок | Функции | Статус |
|---|---|---|
| Сценарии приёмки фазы 2 (раздел 2) | 1, 2, 4, 7, 9 | закрыты: см. строки «Сценарий 1/2/4/7/9» выше в таблице фазы 2 (замеры 0,006 с / 0,179 с / 2,18 с + сценарии 7 и 9) |
| Речь и диалог | F-101 … F-120 | фаза 2: F-101–F-106, F-113, F-114, F-117 (задачи P2-01…P2-10); остальное — фазы 3–5 |
| Идентичность людей | F-201 … F-216 | фаза 2: F-201–F-216 (P2-11…P2-26); набор и метрики 15.6 — P2-36 (числа — стенд) |
| Зрение и присутствие | F-301 … F-314 | фаза 2: F-301–F-304 и F-312 готовы (P2-27…P2-31); F-305–F-309, F-311 — фазы 3–5 |
| LLM, агент, память, интеграции | F-401 … F-424 | фаза 1: F-401–F-406, F-409–F-411 (Decider, роутер, скиллы, guards). Фаза 3, закрыто: **F-401** — уровни `local_fast`/`local_strong`/`local_vision`/`cloud_cheap`/`cloud_strong`, правило «короткая реплика (≤ `short_chars`) идёт в `local_fast`, в том числе с одним инструментом» ОДНО для хаба и цепочки решений, уровень ответа виден в `/health.models` и в трассе хода `/health.utterances` (P3-04…P3-06); **F-403** — перелив виден снаружи (`/health.models`: уровень, причина, ожидание очереди, счётчики переливов), проверен на настоящей `GpuQueue` (P3-05); **F-404** — зрение как уровень + облачный fallback и правило сложности картинки (P3-01…P3-03); **F-409** — `hub/tool_args.py`: аргументы инструментов валидируются Pydantic ДО выполнения по ТОЙ ЖЕ схеме, что получила модель, невалидный вызов возвращается модели текстом ошибки (максимум 2 повтора на инструмент) и не считается действием (P3-07); **F-410** — guard `action_claim` в `hub/action_completion.py`: «сделал» стоит успешного результата инструмента (учитывается ПОСЛЕДНИЙ результат, признание неудачи не наказывается, обещание поверх неудачи получает одну поправку, а выдуманный успех заменяется честной строкой) (P3-08); **F-411** — `hub/untrusted.py`: внешний текст помечен источником и обёрнут явными разделителями `<<<UNTRUSTED … UNTRUSTED>>>` при попадании в промпт, чужие разделители обезвреживаются, хаб читает те же результаты через `strip()`, а помечены все четыре источника ТЗ — скриншот/кадр (F-308), веб-страница, история чата Telegram (текущая просьба владельца — нет, она и есть команда) и скилл с `reads_internet: true` (P3-09…P3-10). **F-411/D-09** — инструменты по командам из недоверенного текста не вызываются: ход, прочитавший скриншот, кадр, страницу, историю Telegram или ответ скилла, отдаёт модели этот текст как данные, а ДЕЙСТВУЮЩИЕ инструменты (`pc_control`, `run_command`, `click_screen`, `set_light`, `set_switch`, `remember`, `show_photo`, `save_photo`, `generate_image`, `set_wallpaper`, `telegram_send`, `enroll_voice`, `enroll_face`, `set_role`, `rename_person`) в этом ходу не запускаются: `Connection._untrusted_reads`/`_untrusted_this_turn` собирают прочитанное всеми читающими инструментами и историей Telegram, `untrusted_text()` разбирает ВЛОЖЕННЫЕ значения тем же обходом, что и обёртка (до этой правки вложенный ответ скилла и строки контекста Telegram не проверялись вовсе), а читающие инструменты осознанно остаются разрешёнными (P3-11). **F-412** — контекст говорящего и состояние дома: профиль (имя, язык, роль, ключевые предпочтения, три новейших факта о человеке) и дом (кто в комнате и сколько незнакомых, светильники, тихие часы вместе с тем, действуют ли они сейчас, время по часам дома) едут в префиксе ХОДА (`hub/speaker_context.py`, `Connection._turn_prefix`), а не в системном промпте — тот обязан оставаться байт-в-байт ради кэша префикса; состояние света клиент пока не сообщает, и блок честно говорит «on/off not reported by the room» вместо выдуманного «off» (P3-12). **F-413** — уточняющие вопросы: «включи свет» в комнате с тремя светильниками получает РОВНО один вопрос (`hub/clarifications.py`, решение D-12 через цепочку решений), а ответ читается следующим ходом в ТОМ ЖЕ окне, что у F-113 (`server.confirmations.window_s`): ответ по имени или по номеру запускает настоящий `set_light`/`set_switch` через обычный `_execute_tool` (права, подтверждение и guard внешнего текста остаются в силе), опоздавший ответ не выполняется и объявляется просроченным, а реплика, не назвавшая кандидата, закрывает вопрос без второго (P3-13…P3-14). **F-414** — факт памяти стал типизированной строкой таблицы `memories` (схема 14): `hub/memories.py` — строгая pydantic-модель `MemoryFact` (id-ULID, тип `person_fact`/`home_fact`/`preference`/`todo`/`event`, область `person`/`home`/`hub`, вес 0…1, вектор float32 вместе с размерностью, `expires_at`/`created_at`), где TTL приходит секундами и становится моментом, факт без TTL не истекает, а просроченный при создании отвергается (пустой и слишком длинный тоже — длинный режется до 500 знаков); `MemoryIndex` пишет (INSERT OR REPLACE), читает по области/владельцу/типу, удаляет и вычищает истёкшие, а эмбеддинг best-effort зеркалится в `vec_memories` (нет sqlite-vec — строка в `memories` всё равно записана, она и есть источник истины). В хабе это write-through рядом с файловым архивом: `remember` с ключом даёт `preference` говорящего, «для всех» (`scope=global` или `about: room/everyone/…`) — `home_fact` дома (`self.home_id`), остальное — `person_fact` названного человека; файл `data/memory.jsonl` остаётся источником промпта до P3-16/P3-17, поэтому поведение комнаты не изменилось (P3-15). **F-414/гибридный поиск** — факты ищутся BM25 ВМЕСТЕ с вектором и в промпт идут `top_k` (по умолчанию 8) самых релевантных фактов для конкретной реплики: `hub/memory_search.py` (unicode-токенизация, BM25, косинус, слияние `(1-w)·лексика + w·вектор`, обе половины видны в `MemoryHit`), блок `[memory: "…" (person: …)]` едет в префиксе хода после состояния дома — и в голосовом ходу, и в раннем старте, и в Telegram-контроллере; факты без вектора оцениваются словами, нерелевантное не отдаётся вовсе, квадратные скобки в факте обезвреживаются (иначе факт подделал бы блок). Эмбеддер — `hub/embeddings.py`: модель ТЗ 9.4 (`multilingual-e5-small` по умолчанию, `bge-m3` как альтернатива) на CPU, только из локальной папки `models/…` (скачивание по HF id — лишь при `server.memory.allow_download: true`), через `sentence-transformers` или `transformers`+`torch`; модель отсутствует — честный `EmbedderUnavailable`, ОДНО предупреждение в лог и поиск по словам, а не выдуманный вектор. `remember` считает вектор факта и кладёт его в строку `memories` (P3-16). **F-414/диалоги** — история и `recall_conversation` читаются из таблицы `dialog_turns` за флагом `server.memory.dialogs_from_db` (по умолчанию выключен): `hub/dialog_turns.py` собирает пару «вопрос + ответ» из двух строк (по `utterance_id`, иначе по соседству), выбирает человека по `person_id`, отдаёт строки той же формы, что прежнее архивное хранилище, и честно отвечает пустой историей без схемы. Миграция 0009 добавила индексы по человеку и по реплике. Миграция хвоста архива — тот же `hub/legacy_migrate.py::migrate_dialogs` на каждом старте: строка jsonl с уже известным `utterance_id` — читаемый двойник строк живого писателя (ТЗ 4.5) и не импортируется (иначе каждая реплика удваивалась), число в отчёте — реально добавленные строки, а хвост привязывается к существующему человеку, а не к второй строке `persons` с тем же именем. Владелец личного факта в `memories` — имя человека (в таблице нет `person_id`); по этой же причине «забудь меня» F-213 теперь чистит личные факты из таблицы и векторный индекс, а факты дома не трогает (P3-17). **F-415/изоляция** — кто в комнате свой, решает правило F-212 (`hub/shared_identity.py::visible_in`): личные факты человека доступны в любом его доме (и только ему), факт дома — только в своём доме и только своему, факт хаба — всем. Гость (незнакомый голос или placeholder-имя) не получает ни факта дома, ни чужого личного; закрыты ОБА канала промпта — поиск (`[memory: …]`) и `{memory}` системного промпта, где гостю остаются ключевые настройки комнаты (как отвечать) и его собственные факты. Именованный профиль без строки `persons` (данные до базы идентичности) остаётся известным, поэтому включение памяти v2 не отнимает факты у комнат без членств. Набор `tests/test_memory_isolation.py` проходит при значениях флагов по умолчанию — изоляция доказана ДО включения памяти v2 (P3-18). Остальное — задачи P3-19…P3-40: память F-414…F-418 (ночная консолидация, напоминания), правила и скиллы F-419…F-421, JevDecider | `PLAN.md`, `PROGRESS.md` |
| Управление комнатой и ПК | F-501 … F-515 | частично в фазе 1: F-501–F-504, F-506; фаза 3 берёт F-505, F-507, F-511 (P3-30…P3-35); остальное — фазы 3–5 |
| Социальные функции и комнаты | F-601 … F-612 | закрыто в фазе 4 (P4-01…P4-40): согласия F-602 (взаимное подтверждение, отзыв, блокировка, `share_presence`), интерком F-601 (слова автора звучат в комнате получателя, очередь «когда придёт», ответ «передай ему: …», тихие часы, аудит и карточка на HUD), опросы F-604, гостевой режим F-606, персонализация F-607. Сценарии 5, 8, 10 пройдены настоящими ходами (`tests/test_phase4_scenarios.py`, 5 тестов), межкомнатное без взаимного согласия невозможно: отказ на настоящем пути (`intercom.send` → `denied` в аудите) и ни одного сообщения в очереди. F-603 (частота) — тот же гейт в P4-06; F-608 (игры) и остальное P2 — фаза 5 |
| Telegram, админка, HUD, мобильный | F-701 … F-713 | фаза 1: F-705, F-706; фаза 2: F-701, F-702, F-708; фаза 4: F-704 (ежедневный дайджест владельцу по часам дома, идемпотентный, с честным разделом «что не удалось»), F-707 (`/metrics` для Prometheus + дашборд Grafana и README в `deploy/grafana/`), F-711/F-712 (телефон как клиент, пуш и очередь). Остальное — фазы 3–5 |
| Решения (Decider) | D-01 … D-12 | D-01…D-11 переведены в фазе 1; D-06 и D-12 придут с идентичностью и уточнениями (фаза 2–3) |

## Что уже сделано из не-ТЗ

Строка таблицы «LLM, агент, память, интеграции» уточняется журналом ниже:
**F-416** (ночная консолидация, P3-19) и **F-418** (явное управление памятью,
P3-20) закрыты; из памяти F-414…F-418 остаются напоминания F-417
(P3-21…P3-23).

- Удобная Telegram-клавиатура (по две кнопки в ряд), пресеты числовых полей
  и диапазоны уведомлений — сделаны (см. таблицу «Вне ТЗ» в начале файла).
- Фаза 2 добавила **три замера приёмки** (`make measure-scene`,
  `make measure-presence`, `make test-identity`): первые два дают числа в
  песочнице, третий честно требует записей и insightface со стенда.

## Открытые вопросы (раздел 17 ТЗ)

Реализуются вариантом «по умолчанию», пока заказчик не ответит:
overlay-сеть Tailscale, клиенты на Windows, хаб на Linux, LLM Qwen 35B-A3B под
vLLM, бюджет $18/мес общий на хаб, до 4 комнат, Jev за флагом, Home Assistant
в Фазе 5.

Отдельным пунктом: **модели эмбеддингов ТЗ 9.4 в сборке нет.** `models/multilingual-e5-small` (или `models/bge-m3`) заказчик кладёт сам; скачивание по HF id по умолчанию выключено. Без модели хаб не выдумывает вектор: `hub/embeddings.py` честно отказывает, а поиск памяти идёт по словам (BM25), и это видно в логе одной строкой на модель. Числа стенда по гибридному поиску (качество и время) — за P3-40.

Отдельным пунктом: **весов anti-spoofing-модели ТЗ (insightface anti-spoof /
MiniFASNet) в сборке нет.** ТЗ F-214 называет модель, но ни весов, ни лицензии
в поставке нет, а подделывать вердикт чужой модели правилами проекта
запрещено. Поэтому `hub/anti_spoofing.py::load_model` честно бросает
`SpoofModelUnavailable`, признаки ТЗ (муар, микродвижения, плоскость) считаются
без неё, а `server.identity.anti_spoofing.require_model = true` превращает
отсутствие модели в отказ. Решение о поставке весов — за заказчиком.

Отдельным пунктом: **боевой клиент комнаты ещё говорит по протоколу v1, и это
видно в данных.** Клиент на DormPC — сборка до переименования `server/` →
`hub/`: он не присылает `proto`/`token`, поэтому сессия живёт без привязки к
строке дома (`Connection.home_id` остаётся пустым — так и описано в
`_send_room_config`: «v1 connections are not bound to a room row»). Следствия
на живом хабе: `/health.utterances` и `/health.camera_events` считают всё в
`home_id: ""`, очередь GPU уводит такие задания в `default_home_id()`, а
`hub/face_tracks.py` и `hub/identity_fusion.py` честно отказываются сохранять
лица и убеждения («track … is new and home '' is unknown») — то есть узнавание
лиц работает в моменте, но не накапливается, и память уровня дома для этого
клиента не наполняется. Лечится выдачей клиентского токена ТЗ 4.3
(`hub/auth.py::ClientTokenStore.issue`) и полем `home_id` в конфиге клиента —
это шаг обновления боевой комнаты, а не правка хаба; на текущий сеанс это
оставлено как есть, чтобы не менять рабочую комнату в тот же вечер.

## Журнал фазы 3 (закрытые задачи)

- **Живая регрессия очереди GPU в распознавании речи: найдена и закрыта (21.09.2026).**
  На боевом хабе каждая реплика падала с `TypeError: object list can't be used in
  'await' expression`: батчер STT (`hub/stt.py::SttBatcher`) отдаёт в `runner`
  БЛОКИРУЮЩУЮ функцию, а `GpuQueue.submit` ждёт нуль-аргументную корутину, и
  прежняя обёртка `await`-ила результат декода. Теперь
  `hub/app.py::_stt_batch_job` декодирует батч в рабочем потоке
  (`asyncio.to_thread`) и держит слот очереди на всё время декодирования: один
  батч — один слот, а цикл продолжает обслуживать другие комнаты. Проверено:
  `pytest tests/test_gpu_queue_wiring.py tests/test_stt_batching.py -q` → 26
  passed; полный `pytest tests -q` → 4354 passed, 10 skipped; `ruff check .` и
  `mypy common` чисто. Вживую на хабе синтетическая реплика прошла путь STT за
  242 мс (`/health.utterances` → `stages_ms.stt=242`, `/health.stt_batching` →
  `batches=1`), ошибка больше не появляется.
- **P3-19 (F-416) — ночная консолидация памяти: сделано.** `hub/memory_consolidation.py`
  проходит по настоящей таблице `memories` (схема 14) по расписанию КАЖДОГО дома:
  04:00 по поясу дома (`consolidation_hour`/`consolidation_minute`), отметка о
  пройденной ночи — в `homes.settings_json.memory_consolidated_on`, поэтому ночь не
  повторяется, а пропущенная (хаб был выключен) догоняется; задача живёт в
  планировщике хаба под именем `memory.consolidate` и проверяется раз в 900 с.
  Проход удаляет истёкшее, снимает вес у старых (`decay_per_day`, но не ниже
  `decay_floor`), сливает дубли (тот же текст или косинус ≥ `duplicate_similarity`,
  только внутри одного владельца и только когда вектор есть у обоих), догоняет
  отсутствующие векторы в рабочем потоке и пишет сводку дня в 5–10 фактов СТРОГО
  ответом модели (Guided JSON); без модели или при невалидном ответе сводки нет
  вовсе — только честный `summarizer="none"` с причиной в отчёте, а отчёт уходит в
  `audit`. Сводка — факт дома (`scope=home`), гостю она не видна (F-415).
  Проверено: `pytest tests/test_memory_consolidation.py -q` → 32 passed;
  `make test` → 4351 passed, 10 skipped, `ruff check .` и `mypy common` чисто;
  на живом хабе `/health.scheduler` = `media.ttl`, `memory.consolidate`.
  **Чего нет:** сумматор и эмбеддер проверены подставными (встроенной модели
  эмбеддингов ТЗ 9.4 в сборке нет — раздел 17), качество сводки и время прохода
  замеряются на стенде (P3-40).

## Операционные исправления по живой эксплуатации (21–22.09.2026)

- **P3-20 (F-418) — явное управление памятью: сделано.** Три реплики ТЗ
  («запомни, что …», «забудь, что …», «что ты обо мне знаешь?») распознаёт сам
  хаб (`hub/memory_admin.py`) до модели, на трёх языках, и ведёт их через
  инструменты `remember`/`forget_fact`/`list_memory`. Хранящие факт и
  удаляющие его вызовы ждут устного «да» F-113 (окно `window_s`, флаги
  `server.memory.confirm_remember`/`confirm_forget`), «нет» и истечение окна
  отвечают честно и ничего не меняют, каждый вопрос уходит в `audit`. Удаление
  выбирает ровно ОДИН факт по словам запроса и при близких кандидатах задаёт
  вопрос вместо догадки; «забудь меня» (F-213) идёт своим необратимым потоком.
  Проверено: `pytest tests/test_memory_admin.py -q` → 50 passed;
  `pytest tests -q` → 4498 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто. **Чего нет:** напоминания F-417 — P3-21…P3-23.

- **P3-21 (F-417) — разбор времени и запись напоминаний: сделано.**
  `hub/reminders.py` читает «через 20 минут», «через полчаса», «в пятницу в
  9», «в девять вечера», «завтра», «in 2 hours», «en media hora» на ru/en/es
  и считает срок по часам ДОМА (`homes.tz`), а в `reminders` (схема 14)
  ложится момент UTC: одни и те же слова в Чикаго и в Киеве дают разные
  мгновения. Реплика идёт через `hub/app.py::_reminder_turn` (пишет строку и
  `reminder.scheduled` в `audit`, отвечает названным местным временем) и
  честно отказывает без срока, без узнанного человека, без базы и сверх
  `server.reminders.max_pending_per_person`. Проверено:
  `pytest tests/test_reminders.py -q` → 45 passed; `pytest tests -q` → 4558
  passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:**
  доставка (озвучка в нужном доме) — P3-22; «когда приду домой» — P3-23.

- **P3-22 (F-417) — доставка напоминания в нужную комнату: сделано.**
  `hub/reminders.py::ReminderDeliveryTask` (задача планировщика
  `reminder.deliver`) отдаёт наступившие строки по ПРИСУТСТВИЮ: человек
  слышит напоминание там, где его сейчас видно (F-301), независимо от того, в
  каком доме он его просил. Миграция `0010` добавила `delivery_state`,
  `delivery_note`, `attempts`, поэтому три исхода названы честно: `spoken`,
  `person_absent` (сказать некому, пуш F-712 — фаза 4, строки ждут в
  `stale_unspoken()`) и `waiting_client` (повтор без спама в аудите). Проверено:
  `pytest tests/test_reminder_delivery.py tests/test_reminders.py tests/test_migrations.py -q`
  → 59 passed; `pytest tests -q` → 4573 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** «когда приду домой» — P3-23.

- **P3-23 (F-417) — «напомни, когда приду домой»: сделано.** Напоминание без
  срока живёт в той же таблице `reminders`: миграция `0011` пересобрала её
  (`due_at` необязателен, `trigger_kind` = `time`/`person_entered`), и
  `hub/reminders.py::parse_arrival` читает просьбу на ru/en/es, а
  `arm_arrivals` оживляет строки ровно того человека на событии F-301
  `person_entered` (`hub/app.py::_observe_presence`). Дальше работает обычная
  доставка P3-22: человек только что вошёл в комнату и слышит напоминание
  здесь же. Проверено: `pytest tests/test_reminder_arrival.py
  tests/test_reminder_delivery.py tests/test_reminders.py tests/test_migrations.py -q`
  → 86 passed; `pytest tests -q` → 4605 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** правила F-419 — P3-24…P3-27.

- **P3-24 (F-419) — модель правила: сделано.** `hub/automation.py` держит
  строгие модели триггера (присутствие F-301, время «HH:MM» с днями недели,
  звук с порогом, состояние устройства по capability F-501), условий (роль,
  кто в комнате, тихие часы) и действий (сцена, say, уведомление, скилл) и
  пишет их в `rules` ТОЛЬКО через Pydantic — испорченная строка логируется и
  пропускается. Миграция `0012` добавила `name` и `last_fired_at`; правило
  говорит словами на ru/en/es. Проверено:
  `pytest tests/test_automation_model.py -q` → 43 passed. **Чего нет:**
  создание голосом — P3-26, панель — P3-27.

- **P3-25 (F-419) — исполнение правил: сделано.** `RuleEngine` прогоняет
  триггер → условия → действия с теми же правами, что у голосовой команды:
  действует запустивший правило человек (для правил по времени — автор,
  правило без автора не исполняется), say и сцена требуют `user`,
  уведомление — `trusted`, скилл — своей роли из манифеста; в тихие часы
  F-115 проходит только критичное уведомление. Каждое решение, включая отказ,
  уходит в `audit`, правило по времени помечается `last_fired_at` и
  срабатывает раз в сутки. Правила по событиям запускаются из
  `_observe_presence`, правила по времени — задачей `rule.time` в
  планировщике (`server.rules.check_interval_s`). Проверено:
  `pytest tests/test_rule_engine.py -q` → 24 passed; `pytest tests -q` → 4673
  passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:**
  создание правила голосом — P3-26, панель — P3-27.

- **P3-26 (F-419) — правило голосом: сделано.** Хаб сам узнаёт реплику-правило
  («когда я приду после 22:00, включи тёплый свет»), локальная модель
  составляет правило через structured output (`RuleDraft`), а дом, автор и
  включённость приходят от хаба; правило начинает работать только после
  устного «да» (F-113), поэтому `create_rule` в списке опасных навсегда.
  Инструмент `create_rule` доступен и модели, но включает правило всё равно
  человек. Проверено: `pytest tests/test_rule_voice.py -q` → 26 passed;
  `pytest tests -q` → 4701 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** панель правил — P3-27.

- **P3-27 (F-419) — правила в панели: сделано.** Раздел «Room rules» показывает
  правила дома словами (`describe`): «✓ Тёплый свет: Когда: кто-то входит
  (Anton); если: Anton дома; включить сцену «вечер»». Ни одного JSON-поля на
  экране нет — `rules.list` собирает `words` из строгих моделей, а не из
  сырых колонок. Включение, выключение и удаление идут через подтверждение
  (как остальные опасные кнопки), требуют дом из области видимости
  (владелец дома не тронет чужое правило), и каждая правка попадает в аудит
  F-706 (`rules.` в `AUDITED`), потому что выключенное или удалённое правило
  меняет поведение комнаты. Проверено: `pytest tests/test_admin_rules.py -q`
  → 9 passed; `pytest tests -q` → 4711 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. Попутно прогон нашёл две настоящие болячки, обе
  закрыты: ветки страниц правил были вложены в ветку `alerts` и не
  открывались вовсе; выбор «показанной» картинки гостя в
  `Connection._latest_generated` проигрывал ничью по времени сохранения
  собственной картинке владельца (тест падал через раз) — теперь ничья
  отдаётся показанной картинке. **Чего нет:** утренний брифинг F-420 —
  P3-28.

- **P3-28 (F-420) — утренний брифинг: сделано.** `hub/briefing.py` собирает
  пять разделов ТЗ (погода, первая пара или встреча, дедлайны, напоминания,
  статус устройств) из ИСТОЧНИКОВ, а модель только пересказывает их: она не
  ходит ни в погоду, ни в календарь и не может выдумать факт, которого нет в
  данных. Раздел без источника не исчезает молча — он говорит, чего не
  хватает («источник погоды не подключён»); отсутствие модели тоже не
  фальшивка: комната слышит те же факты простыми фразами. Поводы — час дома
  из конфига и вход в комнату за утро (событие F-301), один брифинг в сутки
  на человека и дом, часы считаются по `homes.tz`. Живой источник сейчас один
  — напоминания F-417; погода (P3-29), Canvas (P3-30), календарь (P3-31) и
  состояние устройств (P3-32) подключаются снаружи, без правок модуля.
  Проверено: `pytest tests/test_briefing.py -q` → 69 passed;
  `pytest tests -q` → 4780 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** перечисленных источников — они и есть
  P3-29…P3-32; брифинг выключен по умолчанию (`server.briefing.enabled`).

- **P3-29 (F-421) — скилл погоды и скиллы как инструменты: сделано.**
  `skills/weather/` — настоящий Open-Meteo без ключа (геокодирование + прогноз
  на сегодня и завтра, пояс комнаты), таймаут 8 с и честные отказы: обрыв
  сети, таймаут, `5xx`, «не данные», незнакомое место и отсутствие места —
  это `ok=False` с причиной, а не выдуманная температура. Коды WMO
  переводятся в слова в самом скилле (ru/en/es). Скиллы дошли до модели:
  инструмент `run_skill` зовёт скилл по имени (проверка дома, `enabled` и
  роли говорящего по настоящему реестру), имена доступных скиллов идут в
  `[home: ...]`, а ответ помечается как недоверенный текст ровно тогда, когда
  манифест объявил `reads_internet` (F-411) — обещание P3-10 наконец стало
  путём в промпт. Погода подключена к брифингу F-420 через раздел-скилл;
  место комнаты — `homes[].weather_location`. Проверено:
  `pytest tests/test_weather_skill.py tests/test_skill_tools.py -q` →
  46 passed; `pytest tests -q` → 4828 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** живого интернета в песочнице нет
  (транспорт подставной) — настоящий ответ Open-Meteo виден только на стенде;
  Canvas (P3-30) и календарь (P3-31) ещё не подключены.

- **P3-30 (F-421) — Canvas LMS по токену человека: сделано.** Скилл `canvas`
  отвечает на сценарий 3 («что мне сдавать на этой неделе?») по настоящим
  данным Canvas: ближайшие события с настоящими сроками в окне недели,
  список «сделать», курсы и текущие оценки; срок говорится по часам комнаты,
  на языке человека, с баллами. Токен — по человеку: в конфиге только ИМЯ
  переменной окружения (`token_env`, `person_tokens`), сам секрет — в
  окружении хаба, и тест проверяет, что он не утекает ни в ответ, ни в
  запрос. Отказы раздельные и честные: нет адреса, нет токена, токен
  отвергнут, таймаут, обрыв, `5xx`, «не данные», «не список». Дедлайны
  Canvas подключены к утреннему брифингу F-420. Проверено:
  `pytest tests/test_canvas_skill.py -q` → 25 passed; `pytest tests -q` →
  4853 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** живого Canvas нет (транспорт подставной), настоящий токен
  проверяется на стенде; календарь Google — P3-31.

- **P3-31 (F-421) — Google Calendar за флагом: сделано.** Скилл `calendar`
  читает ближайшую встречу, день и неделю и умеет записать НОВОЕ событие в
  календарь того, кто попросил (OAuth: refresh-токен человека меняется на
  access-токен Google). Флаг `server.skills.calendar.enabled` по умолчанию
  выключен: при выключенном флаге скилла нет ни в списке для модели, ни в
  `run_skill`, ни в брифинге. Секретов в конфиге нет — только имена переменных
  окружения, у каждого человека может быть свой refresh-токен. Отказы честные
  и раздельные: нет доступа, Google отверг разрешение, нет календаря, таймаут,
  обрыв, `5xx`, «не данные», ответ без списка событий; создание подтверждается
  только настоящим `id` из ответа Google. Ничего не удаляется. Первая встреча
  подключена к брифингу F-420. Проверено:
  `pytest tests/test_calendar_skill.py -q` → 34 passed; `pytest tests -q` →
  4887 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** живого Google нет (транспорт подставной); OAuth-согласие
  внутри Rowan не получается — токены кладёт в окружение тот, кто ставит хаб
  (открытый вопрос раздела 17, записан в `DECISIONS.md`).

- **P3-32 (F-505) — состояние устройств: сделано.** Новый
  `hub/device_state.py`. Текущее состояние устройств теперь типизировано:
  `DeviceStateStore.read/summary/by_name` отдают `DeviceState` (значения, дом,
  имя, вид, момент и источник отчёта; служебные `_at`/`_source` наружу не
  протекают). Подписка — это отчёты самой комнаты: `parse_device_report`
  разбирает ответ клиента в способности F-501 (`on_off`, `brightness` с
  зажимом 0–100, `color_rgb`), строка без состояния не учит ничему, чужое имя и
  чужой дом игнорируются. Опрос — вторая половина ТЗ и отдельная задача
  планировщика `device.state` (`DeviceStateTask`): спрашивает у адаптера только
  объявленные способности, записывает только настоящий ответ, считает молчащие
  адаптеры и устройства без адаптера, включается лишь при заданном
  `server.device_state.poll_interval_s` (по умолчанию 0). Контекст LLM:
  `[home: ...]` называет состояние — «lights: Ceiling lamp (off)» вместо
  «on/off not reported by the room». Правило F-505:
  `Connection._device_command_is_already_done` не отправляет команду, если
  последний отчёт о том же состоянии свежий (`set_light`/`set_switch` с «off»
  при известном «off» → `{"ok": true, "already": true, ...}`, ни одного кадра в
  комнату); частичные изменения (яркость, цвет) и `press` не считаются, а
  старое состояние (`server.device_state.stale_after_s`, по умолчанию 900 с) не
  блокирует команду. Заодно `_record_device_report` пишет состояние после
  успешной команды. Проверено: `pytest tests/test_device_state.py -q` → 36
  passed; `pytest tests -q` → 4923 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто; `python -c "import hub.app"` → ok.
  **Чего нет:** живых лампочек и BLE/Tuya нет (адаптеры подставные); история
  состояний — следующая задача P3-33.

- **P3-33 (F-505) — история состояний устройств: сделано.** Миграция
  `0014_device_state_history.py` (схема 15) добавила таблицу
  `device_state_events(event_id, device_id→devices ON DELETE CASCADE, home_id,
  capability, value_json, source, ts)` с двумя индексами. `devices.state_json`
  по-прежнему хранит «как сейчас», а история — каждое ИЗМЕНЕНИЕ: `record`
  кладёт строку только на реально сменившуюся способность, поэтому повторный
  отчёт не плодит записей, а первый отчёт записывается даже «off». История
  пишется и от отчёта комнаты (`room`), и от опроса адаптера (`adapter`), и не
  стоит состояния: база без таблицы истории честно логируется, состояние всё
  равно обновляется. Чтение — `history(home_id, device=, capability=, since=,
  until=, limit=)` (новые впереди, окно, лимит 1…1000) и `last_change(...)`;
  дом ищется внутри своего дома, чужой дом истории не видит. Размер хранимого
  ограничен `keep_per_device` (по умолчанию 5000 событий на устройство).
  Проверено: `pytest tests/test_device_state.py -q` → 48 passed;
  `pytest tests -q` → 4935 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто; `python -c "import hub.app"` → ok.
  **Чего нет:** удержания по времени (только по числу событий); живого стенда
  с лампами; потребителя истории — правила и статистика F-515 придут дальше.

- **P3-34 (F-507) — presence-автоматика: сделано.** Новый
  `hub/presence_automation.py`. `PresenceAutomation` выдаёт два события: `left`
  (с последнего живого трека прошло `server.presence.left_after_s`, по
  умолчанию 600 с = 10 минут ТЗ) и `returned` (пока дом «ушёл», снова
  увидели человека с ролью `admin` — admin-порог F-208). Дом без единого
  наблюдения не «уходит» никогда; `left` выдаётся один раз на уход;
  безымянный трек — «в комнате кто-то есть», но не возврат владельца.
  `PresenceAutomationTask` (`presence.home`, интервал
  `server.presence.check_interval_s`) обходит дома по расписанию. В хабе:
  `_observe_presence` кормит автоматику каждым наблюдением, `left` выполняет
  сцену ухода `left_scene` (по умолчанию пресет «ушёл») тем же `SceneRunner`,
  что и правило F-419 (устройства, ПК-lock через настоящий `_run_client_action`,
  реплика пресета), пишет аудит `presence.away` и ставит дом в охрану
  (`/health.away_homes`); `returned` снимает охрану, пишет `presence.returned`
  и метит вернувшегося владельца «пора здороваться», поэтому приветствие F-302
  звучит по возврату. Конфиг `server.presence`: `left_after_s`, `left_scene`,
  `guard_enabled`, `check_interval_s` — в обоих шаблонах.
  Проверено: `pytest tests/test_presence_automation.py -q` → 17 passed;
  `pytest tests -q` → 4952 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто; `python -c "import hub.app"` → ok.
  **Чего нет:** живого стенда с камерой и ПК (треки и сцена подставные);
  камера переводится в охрану состоянием хаба и аудитом, а фото/клип в
  Telegram (F-510, фаза 5) и разблокировка ПК по лицу и голосу (P3-35) ещё
  не сделаны; состояние автоматики живёт в памяти и после перезапуска хаба
  начинается заново.

- **P3-35 (F-507) — разблокировка ПК по лицу и голосу: сделано.** Разрешение
  живёт у дома — новое `homes[].pc_unlock` (**false по умолчанию**, как велит
  открытый вопрос раздела 17); без него команда не уходит на ПК. Права:
  `pc_control unlock` — только admin (trusted/user/guest получают отказ).
  Идентичность: `_needs_admin_identity` добавил разблокировку к вызовам F-208
  (`admin_gate`: голос ≥ `admin_voice_threshold` и лицо ≥
  `admin_face_threshold`, тело того же дня или PIN с телефона) — «по лицу и
  голосу», а не по одному голосу. Каждая попытка пишется в аудит `pc.unlock`.
  Клиент `client/actions/pc.py`: команды `lock` (Win32 `LockWorkStation`) и
  `unlock` — Windows Hello не эмулируется, поэтому реализован разрешённый ТЗ
  вариант «авто-ввод PIN из локального хранилища»: PIN только из окружения
  клиента (`ROWAN_PC_UNLOCK_PIN`), 4–12 цифр, ввод на экране блокировки плюс
  Enter; без PIN или при недоступном input desktop клиент честно отказывает.
  Проверено: `pytest tests/test_pc_unlock.py -q` → 13 passed;
  `pytest tests -q` → 4965 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто; `python -c "import hub.app"` → ok.
  **Чего нет:** живого Windows-стенда (secure desktop в песочнице недоступен —
  ввод PIN проверен на подставленном вызове); ответ клиента говорит «PIN
  набран», потому что отпирание отсюда не проверить.

- **P3-36 (F-511) — ПК v2: сделано.** `type_text`, медиаклавиши и прокрутка
  были и раньше; добавлены окно на монитор N, громкость приложения и буфер
  обмена. `move_to_monitor` нумерует мониторы через
  `EnumDisplayMonitors`/`GetMonitorInfoW` (порядок исполнителя: слева направо,
  затем сверху вниз), находит окно приложения по процессу или заголовку и
  переносит его `SetWindowPos` с `SWP_NOZORDER|SWP_NOACTIVATE`, центрируя и
  сохраняя размер; без `target` двигается окно в фокусе, несуществующий
  монитор отвергается. `app_volume` (target = приложение, value = 0…100)
  ищет аудиосессию процесса через pycaw и ставит `SetMasterVolume`; молчащее
  приложение называется словами, а не выдумывается. Буфер обмена — три
  команды: `clipboard_read` (CF_UNICODETEXT, лимит 4000), `clipboard_write`
  и `clipboard_paste` (при необходимости пишет текст, потом Ctrl+V).
  Схема `pc_control` получила новый аргумент `target` (`string|null`) и новые
  значения `command`; Pydantic-контракт F-409 собирается из той же схемы, а
  диспетчер передаёт `target` на клиент. Новые команды требуют admin или
  trusted (как `type_text`). Проверено: `pytest tests/test_pc_v2.py -q` → 23
  passed; `pytest tests -q` → 4988 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто; `python -c "import hub.app, client.actions.pc"` → ok.
  **Чего нет:** живого Windows-стенда (мониторы, аудиосессии и буфер в
  песочнице подставные); DPI-масштабирования при переносе окна нет.

- **P3-37 (F-511) — lock/sleep/shutdown и скриншот области: сделано.**
  `lock` добавлен в `DEFAULT_DANGEROUS_PC_COMMANDS` (и в оба шаблона конфига),
  поэтому уход в lock/shutdown ждёт устного «да» F-113, а вопрос говорит
  человеческими словами («lock the PC», «put the PC to sleep», «shut the PC
  down»). Клиент получил настоящую команду `shutdown` (`shutdown /s /t 0` с
  честным отчётом об отказе). Закрыт пробел: `lock`, `unlock` и `shutdown`
  не были в enum `pc_control`, то есть модель не могла их вызвать — теперь
  они в схеме. Скриншот области — новый `hub/screen_regions.py`
  (`parse_region`: слова en/ru/es и «x, y, width, height» в долях экрана;
  `crop_jpeg`; `region_words`), аргумент `region` у `look_at_screen`,
  обрезка кадра до отправки модели зрения и честный отказ на непонятую
  область без запроса скриншота. Уход по сцене «ушёл» подтверждения не
  спрашивает: сцену заранее настроил владелец, а в комнате никого нет
  (`DECISIONS.md` P3-37). Проверено: `pytest tests/test_pc_power.py -q` →
  29 passed; `pytest tests -q` → 5017 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто; `python -c "import hub.app, client.actions.pc"` → ok.
  **Чего нет:** живого Windows-стенда (кроп проверен на настоящем JPEG из
  синтетической картинки, сам захват подставной); областей в пикселях —
  только доли экрана и слова; `click_screen` по области не умеет.

- **P3-38 (JevDecider) — сделано.** Новый `hub/jev_decider.py` — тонкий
  HTTP-клиент TypeSafe AI Jev на те же три операции `Decider` (`yes_no` с
  вероятностью, `choose` только из предложенных вариантов, `score` внутри
  шкалы). Флаг `server.decider.providers.jev` (по умолчанию **false**,
  `base_url`, `path`, `api_key_env`) — ключ только из переменной окружения
  (раздел 1). Без ключа/адреса/флага провайдера нет, и хаб честно решает
  локальными провайдерами. Порядок адресата в обоих конфигах —
  `addressed: [jev, local_llm, rules]` (D-02 ТЗ); недоступный провайдер
  цепочка пропускает, таймаут — общий `server.decider.timeout_ms` (400 мс).
  Флаг дома `homes[].cloud_decisions` (по умолчанию false, ТЗ 5.5) выключает
  облако для комнаты: `Connection._decide` кладёт `home_id` в каждое
  решение, и без разрешения запрос не уходит вовсе. `_privacy_context`
  отправляет только скаляры и короткие списки — кадры, звук и эмбеддинги не
  уходят никогда (ТЗ 5.5). Все отказы (сеть, таймаут, HTTP ≠ 200, битый JSON,
  нет уверенности, значение вне вариантов/шкалы) — `DecisionUnavailable`.
  Проверено: `pytest tests/test_jev_decider.py -q` → 22 passed;
  `pytest tests -q` → 5039 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто; `python -c "import hub.app"` → ok.
  **Чего нет:** ключа раннего доступа TypeSafe — живого Jev нет; форма
  запроса/ответа собрана по интерфейсу ТЗ и уточняется по документации после
  получения доступа (открытый вопрос раздела 17, `DECISIONS.md`).

- **P3-38 (JevDecider) — уточнено 22.09.2026: доступ получен, форма реальная.**
  Ключ раннего доступа пришёл, и провайдер переписан на настоящий System One
  (`typesafe-sdk` 0.7.1): `POST {base_url}/v1/systemone`, тело
  `{state, model, questions}` с типизированными вопросами `noul` / `choice` /
  `score`, ответ `{"answers": {...}}` — придуманная ручка `/v1/decide` убрана.
  Работает и через OpenRouter (`base_url: https://openrouter.ai/api`, ключ
  `JEV_API_KEY` в `.env`), и напрямую с TypeSafe. У облачного провайдера теперь
  **свой бюджет** времени (`providers.jev.timeout_ms`, по умолчанию 1500 мс):
  400 мс таблицы 15.1 остаются локальному пути, иначе облако обрезалось бы
  таймаутом всегда. Порядок в конфигах — `[rules, jev]`: правила отвечают
  мгновенно, Jev подхватывает те вопросы, где правил нет (тип вне
  `HEURISTIC_TYPES`, а `score` правилам недоступен вовсе). Живой замер:
  `addressed` 612 мс / `route` 475 мс / `score` 448 мс на вопрос через
  OpenRouter (модель под именем `jev-latest` отвечает `typesafe/jev-1.13`).
  Проверено: `pytest tests/test_jev_decider.py -q` → 31 passed;
  `pytest tests/test_decider.py tests/test_decider_config.py
  tests/test_decider_local.py tests/test_metrics.py tests/test_grafana_dashboard.py
  tests/test_config.py -q` → 60 passed.
  **Чего нет:** batching — System One умеет задавать несколько вопросов одним
  запросом, а интерфейс `Decider` спрашивает по одному (стоит один сетевой
  круг на вопрос); это задел на будущее, а не ошибка.

- **P3-39 (F-305, сценарий 6) — память объектов: сделано наполовину, честно.**
  Чтение и ответ реализованы по-настоящему: `hub/object_memory.py` —
  `ObjectMemoryStore` над таблицей `objects_index` схемы 14 (`record`,
  `sightings` с окном 48 часов, `last_seen`, `known_labels`), `where_question`
  разбирает «где мои ключи?» / «where are my keys?» / «¿dónde están mis
  llaves?», `answer_for` говорит «на столе в 14:30» по часам дома и упоминает
  сохранённый кадр (`media_ref`). Миграция `0015_objects_index_zone.py`
  (схема 16) добавила `objects_index.zone` — место из ответа ТЗ, которое
  называет клиентская зона F-309. `Connection._where_turn` отвечает из памяти
  без модели и честно говорит «Я не видела ключи за последние 48 часов»,
  когда записей нет. Проверено: `pytest tests/test_object_memory.py -q` → 30
  passed; `pytest tests -q` → 5069 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто.
  **Чего нет:** детектора и CLIP-эмбеддера (F-305/F-311) — индексатор, который
  наполняет `objects_index`, стоит в фазе 5 по плану ТЗ («объекты (F-305)»),
  поэтому в живом хабе ответы пока честно говорят «не видела»; кадр в HUD или
  Telegram из ответа и поиск по вектору тоже ещё не сделаны.

- **P3-40 (F-305/F-421/F-417/5.4) — итог фазы 3: сделано.** Приёмочные
  критерии фазы прогнаны настоящим кодом в новом `tests/test_phase3_scenarios.py`
  (5 тестов). Сценарий 3 идёт целиком: `Connection._handle_utterance` →
  настоящий `run_skill` → настоящий реестр `skills/` → настоящий скилл
  `skills/canvas/` (подставлена только сеть Canvas, `httpx.MockTransport`),
  комната слышит «Эссе по истории … 100 баллов», задание на 30-й день в
  недельное окно не попадает, а ответ интернет-скилла помечен внешним текстом
  (F-411). Без токена Canvas комната слышит честный отказ про токен. Смена
  `days: 31` реально уезжает в запрос. Напоминания проверены на настоящем
  `ReminderDeliveryTask`: сказано в том доме, где человек стоит (создано в
  `livingroom`, человек в `kyiv` → озвучено в `kyiv`), а напоминание, которое
  некому отдать, честно помечено `absent`. Калибровочный отчёт 5.4 читается из
  настоящей таблицы `decisions` и содержит провайдера `jev` наравне с `rules`.
  Проверено: `pytest tests/test_phase3_scenarios.py -q` → 5 passed;
  `pytest tests -q` → 5074 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто. **Чего нет:** живого Canvas/Telegram/камеры и живого LLM-уровня —
  тест доказывает путь инструмента и скилла, а не выбор модели; индексатор
  объектов F-305 и ключ Jev остаются на стенд (фаза 5 и раздел 17).

- **EXP-01 — реплика больше не глохнет из-за бюджета стадии: сделано.**
  `hub/app.py::Connection._wait_for_stage` ждёт стадию сначала бюджет 15.1,
  затем (с отметкой деградации) абсолютный предохранитель. Причина: на живом
  хабе отмена на 700-й мс выбрасывала готовый транскрипт, клиент получал
  `stt timed out`, `/health.utterances.last.note = stt_timeout`, комната
  слышала пустое TTS. Проверено: `pytest tests/test_stage_timeouts.py
  tests/test_measure_scene_latency.py -q` → 30 passed; полный
  `pytest tests -q` → 4382 passed, 10 skipped; `ruff check .` и `mypy common`
  чисто.
- **EXP-03 — клиент без дома не пишет личность: сделано.**
  `hub/app.py::Connection._identity_storage_ready` закрывает путь записи лиц,
  кропов тела и привязок для v1-`hello`, оставляя одну строку в логе вместо
  предупреждений на каждый кадр. Покрыто `tests/test_stage_timeouts.py`.
- **EXP-04/EXP-05 — оверлей комнаты: сделано.** Постоянный значок
  `online`/`queue` убран из обоих экранов HUD (осталась только надпись
  `brain offline` при потере связи), а бейджи F-102/F-303 держат прозрачное
  окно не дольше `BADGE_HOLD_S` = 6 с (`client/overlay.py::_visibility_now`).
  Проверено: `pytest tests/test_overlay.py tests/test_hud_v2.py
  tests/test_overlay_followup.py tests/test_privacy_mode.py
  tests/test_barge_in.py -q` → 178 passed.

## Журнал фазы 4 (закрытые задачи)

- **P4-01 (F-602) — контакты и взаимное согласие: сделано.** Миграция
  `0016_contacts_consent.py` (схема 17) добавляет к таблице `contacts` схемы 14
  четыре поля, и каждое — приватность, а не удобство: `requested_by` (кто
  позвал: иначе нельзя ни подтвердить чужое приглашение, ни запретить подпись
  за другого), `confirmed_at` (когда согласие стало взаимным), `blocked_by`
  (снять блокировку может только поставивший) и два раздельных флага
  присутствия `share_presence_a`/`share_presence_b` («да» Антона за Макса не
  считается). Новый `hub/contacts.py`: строгая модель `Contact` и
  `ContactStore` — каноническая пара, идемпотентное приглашение (встречные
  приглашения = два «да»), подтверждение только со стороны приглашённого,
  отказ/отзыв/блокировка/разблокировка, `set_share_presence`/`presence_shared`
  и главная точка `contact_gate`/`allowed`. Проверено:
  `pytest tests/test_contacts.py tests/test_migrations.py -q` → 31 passed;
  `pytest tests -q` → 5101 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто. **Чего нет:** голосового приглашения (P4-02), гейта в живом ходе
  (P4-04), вопроса «Макс дома?» (P4-05) и кулдауна (P4-06).

- **P4-02 (F-602) — приглашение и подтверждение голосом: сделано.** Новый
  `tests/test_contact_voice.py` (32 теста) и разбор фразы в `hub/contacts.py`:
  «добавь Макса в контакты» / «add Max to my contacts» / «agrega a Max a mis
  contactos» и подтверждение на трёх языках; чужие реплики не перехватываются.
  Падеж снимается неизменяемой частью имени («Макса» → «макс») с требованием
  единственного совпадения, а ответ называет имя из `persons`, а не форму из
  реплики. `Connection._contact_turn` даёт согласие только распознанному
  человеку: за другого подписать нельзя, «да» приходит из его комнаты, каждое
  изменение и каждый отказ ложатся в `audit` (`contact.invite`,
  `contact.confirm`) — F-706. Проверено: `pytest tests/test_contact_voice.py
  -q` → 32 passed; `pytest tests -q` → 5133 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** отзыва/блокировки голосом и флага
  присутствия (P4-03), живого межкомнатного гейта (P4-04), «Макс дома?»
  (P4-05), кулдауна (P4-06).

- **P4-03 (F-602) — отзыв, блокировка и флаг присутствия голосом: сделано.**
  Семь намерений и 34 выражения на ru/en/es («убери Макса из контактов»,
  «block Max», «desbloquea a Max», «разреши Максу видеть, что я дома»,
  «stop sharing my presence with Max»); вопросы о присутствии и другие ходы не
  перехватываются. Блокировка отменяет ожидающее приглашение, стирает
  `confirmed_at`, обнуляет оба флага присутствия и запрещает новые приглашения;
  снять её может только поставивший. Флаг присутствия у каждого свой, поэтому
  «да» Антона не показывает Макса. Все семь действий и все отказы пишутся в
  `audit` с понятной причиной, ответы — на языке говорящего. Проверено:
  `pytest tests/test_contact_voice.py -q` → 64 passed; `pytest tests -q` →
  5165 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** единой точки гейта (P4-04), «Макс дома?» через дома (P4-05),
  кулдауна (P4-06).

- **P4-04 (F-602) — одна точка межкомнатного: сделано.** `gate_reason` отдаёт
  стабильный код причины (`self`/`strangers`/`pending`/`blocked`), а
  `Connection._interhome_gate` и `_interhome_send` — единственный путь, которым
  что-либо уходит в другую комнату: доставка вызывается ТОЛЬКО после согласия
  обеих сторон, а отказ звучит на ru/en/es. Тест на семи случаях доказывает,
  что без согласия подставная доставка не вызывается ни разу (критерий
  приёмки фазы 4 «межкомнатное невозможно без взаимного согласия»). Проверено:
  `pytest tests/test_contact_voice.py -q` → 71 passed; `pytest tests -q` →
  5172 passed, 11 skipped. **Чего нет:** самого интеркома (P4-07…P4-13) и
  опросов (P4-14…P4-19) — они подключаются к этому гейту.

- **P4-07 (F-601) — разбор интеркома: сделано.** `hub/intercom.py`:
  `intercom_request` («скажи Максу, что я иду», «tell Max that I am coming»,
  «pass on to Max: ok», «dile a Max que voy en camino») и `intercom_reply`
  («передай ему: ок»), строгие модели `IntercomRequest` и `IntercomMessage`
  (дом получателя, дом отправителя, статус очереди). Разбор не угадывает:
  «скажи мне, что делать», «tell me a joke» и «скажи время» остаются модели.
  Тесты `tests/test_intercom_parsing.py` (30). Проверено:
  `pytest tests/test_intercom_parsing.py -q` → 30 passed; `pytest tests -q` →
  5236 passed, 11 skipped. **Чего нет:** таблицы, доставки и очереди
  (P4-08…P4-13).

- **P4-08 (F-601) — очередь интеркома: сделано.** Миграция
  `0017_intercom_messages.py` (схема 18) и `IntercomStore`: дом получателя,
  дом отправителя, статусы queued/spoken/replied/expired, ссылка на исходное
  сообщение, пределы очереди (`server.intercom.queue_limit` — переполнение
  помечает старое `expired`, а не теряет молча). Забытый человек уносит свои
  сообщения (`CASCADE`), забытый автор оставляет слова без имени (`SET NULL`).
  Тесты `tests/test_intercom_store.py` (11). Проверено:
  `pytest tests/test_intercom_store.py tests/test_migrations.py -q` → 15
  passed; `pytest tests -q` → 5247 passed, 11 skipped.
- **P4-09 (F-601) — доставка в комнату получателя: сделано.**
  `Connection._intercom_turn`: согласие → лимит → очередь дома получателя →
  громкая речь, если человек в комнате; иначе сообщение ждёт и отправитель
  слышит «Передам при появлении». Речь идёт на языке получателя
  (`preferred_language`), слова отправителя не переводятся — называется автор
  и приводится сказанное. Дом человека ищется по живому присутствию, затем по
  членству. Тесты `tests/test_intercom_delivery.py` (11). Проверено:
  `pytest tests/test_intercom_delivery.py -q` → 11 passed; `pytest tests -q` →
  5258 passed, 11 skipped. **Чего нет:** автодоставки при входе (P4-10),
  ответа «передай ему: ок» (P4-11), тихих часов (P4-12), аудита и карточки
  (P4-13).

- **P4-10 (F-601) — «передам, когда придёт»: сделано.**
  `IntercomDeliveryTask` (`intercom.deliver`, интервал
  `server.intercom.check_interval_s`) отдаёт очередь дома тому, кто в нём
  появился: от старого к новому, сказанное — `spoken` и в аудит, неслышное
  остаётся `queued`; одна молчащая комната или недоступная камера не мешают
  остальным. Тесты `tests/test_intercom_queue_delivery.py` (8). Проверено:
  `pytest tests/test_intercom_queue_delivery.py -q` → 8 passed; `pytest tests
  -q` → 5266 passed, 11 skipped.
- **P4-11 (F-601) — ответ «передай ему: ок»: сделано.** `_intercom_reply_turn`
  отвечает автору последнего прозвучавшего сообщения (`last_delivered`), тем
  же путём через согласие и лимит, со ссылкой `reply_to`; исходное сообщение
  помечается `replied`. Без полученных сообщений — честное «я не получала для
  вас сообщений». Тесты в `tests/test_intercom_delivery.py` (5 новых, всего
  16). Проверено: `pytest tests -q` → 5271 passed, 11 skipped.

- **P4-12 (F-601) — тихие часы интеркома: сделано.** Ночью сообщение ждёт
  («просил не беспокоить в тихие часы»), потому что по умолчанию хаб не будит;
  личное «разреши интерком ночью» (`persons.settings_json.intercom_quiet_ok`,
  голосом распознанного человека) пропускает сообщения, а `audit` фиксирует
  `intercom.quiet`. Задача доставки получила `quiet(home, person)` и отчёт
  `quiet`. Тесты `tests/test_intercom_quiet.py` (21). Проверено:
  `pytest tests/test_intercom_quiet.py -q` → 21 passed; `pytest tests -q` →
  5292 passed, 11 skipped.
- **P4-13 (F-601/F-706/F-709) — аудит интеркома и карточка на HUD: сделано.**
  `MSG_CARD` в протоколе (фоновый кадр с ttl), карточка уходит в комнату
  получателя вместе с произнесённым сообщением (офлайн — ждёт вместе с
  очередью), клиент показывает её подписью HUD и гасит по ttl. Аудит:
  `intercom.send` (spoken/queued/quiet/hidden) с причиной при отказе,
  `intercom.deliver`, `intercom.quiet`. Тесты `tests/test_intercom_cards.py`
  (13). Проверено: `pytest tests/test_intercom_cards.py
  tests/test_intercom_delivery.py -q` → 29 passed; `pytest tests -q` → 5305
  passed, 11 skipped. **Чего нет:** очереди нескольких карточек и ручного
  скрытия (F-709 в части «очередь карточек» — карточка живёт как подпись HUD).

- **P4-14 (F-604) — опросы в базе: сделано.** Миграция `0018_polls_flow.py`
  (схема 19): аудитория, статус/`closed_at`, свои варианты, комната ответа.
  `hub/polls.py::PollStore` — создание, ответ (только для спрошенного и только
  из предложенных вариантов; второй ответ требует явной замены), свод с
  молчащими, дедлайн и закрытие. Тесты `tests/test_polls.py` (12).
  Проверено: `pytest tests/test_polls.py tests/test_migrations.py -q` → 16
  passed; `pytest tests -q` → 5317 passed, 11 skipped. **Чего нет:** разбора
  фразы (P4-15), задания вопроса по появлению (P4-16), голосового сбора
  (P4-17), свода автору (P4-18), гейта контактов и тихих часов (P4-19).

- **P4-15 (F-604) — разбор вопроса к компании: сделано.** `poll_request`
  понимает «кто в баскетбол в 6?» / «who is up for basketball at 6?» / «¿quién
  juega al baloncesto a las 6?» и берёт срок тем же разбором, что напоминания
  (`parse_when`) — по часам дома автора; без названного времени опрос живёт два
  часа. «Кто дома?» и «кто ты?» остаются присутствию и разговору. Тесты
  `tests/test_polls.py` (31). Проверено: `pytest tests/test_polls.py -q` → 31
  passed; `pytest tests -q` → 5336 passed, 11 skipped.

- **P4-05 (F-602) — «Макс дома?» через комнаты: сделано.** Новый вид вопроса
  `ASK_HOME` («Макс дома?», «Is Max at home?», «¿Max está en casa?») и ответы
  `answer_home` на ru/en/es: да / не вижу кадров / «не разрешил говорить,
  дома ли он» / «нужно узнать ваш голос» / «не знаю такого человека».
  `_home_question_turn` ищет человека по всем домам хаба через живое
  `presence.occupants`: в своей комнате ответ идёт без гейта (там присутствие и
  так видно, F-301), в другой — только при `share_presence` самого человека и
  подтверждённом контакте; блокировка закрывает и это. Чужая комната не
  называется — только «дома». Тесты `tests/test_interhome_presence.py` (19).
  Проверено: `pytest tests/test_interhome_presence.py
  tests/test_presence_state.py -q` → 56 passed; `pytest tests -q` → 5191
  passed, 11 skipped. **Чего нет:** кулдауна (P4-06), интеркома (P4-07+).

- **P4-06 (F-602/F-603) — частота межкомнатных сообщений: сделано.** Секция
  `server.intercom` в `common/config.py` и обоих шаблонах (`cooldown_s` 600 —
  число ТЗ «не чаще раза в 10 минут на человека», `max_messages`, `queue_limit`
  для очереди «когда придёт»); новый `hub/interhome.py::InterhomeLimiter`
  (скользящее окно на пару «человек + дом», потокобезопасно) и ответ
  «Слишком часто: следующее сообщение можно через N мин.» на ru/en/es.
  Отказ по согласию лимит НЕ тратит — он расходуется только на состоявшуюся
  доставку. Тесты `tests/test_interhome_limit.py` (15). Проверено:
  `pytest tests/test_interhome_limit.py tests/test_config.py
  tests/test_hub_config.py -q` → 35 passed; `pytest tests -q` → 5206 passed,
  11 skipped. **Чего нет:** интеркома (P4-07+); счёт в памяти процесса.

- **P4-16 (F-604) — вопрос задаётся при появлении, а не всем сразу: сделано.**
  Миграция `0019_poll_asks.py` (схема 20, таблица `poll_asks`) и в
  `hub/polls.py` — `ask_line` на трёх языках, `mark_asked`/`asked_at`/`asked`
  («спросили» ≠ «ответили»), `pending_for` отдаёт только тех, кого ещё не
  спрашивали, и задача `PollAskTask` (`poll.ask`), которая закрывает просроченные
  опросы и спрашивает каждого участника в первой комнате, где его видно, отмечая
  вопрос только после того, как он прозвучал. Тесты `tests/test_poll_asking.py`
  (9). Проверено: `pytest tests/test_poll_asking.py tests/test_migrations.py -q`
  → 15 passed; `pytest tests -q` → 5345 passed, 11 skipped.
- **P4-17 (F-604) — ответы голосом «да/нет/позже»: сделано.**
  `hub/polls.py::answer_command` на ru/en/es; ответ принимается только целиком,
  замена — только по явным словам («передумал: да»), ответ пишется с комнатой и
  в `audit` (`poll.answer`), чужие варианты («да» у опроса «пицца или суши») не
  принимаются, «да» в обычном разговоре остаётся разговором. Тесты
  `tests/test_poll_answers.py` (42). Проверено: `pytest
  tests/test_poll_answers.py -q` → 42 passed; `pytest tests -q` → 5387 passed,
  11 skipped.
- **P4-18 (F-604) — свод автору после дедлайна: сделано.** Миграция
  `0020_poll_summary.py` (схема 21, `polls.summarized_at`), `summary_line`
  (цифры по вариантам + промолчавшие по именам) и `PollSummaryTask`
  (`poll.summary`), который закрывает просроченные опросы и говорит свод в
  комнате автора ровно один раз. Тесты `tests/test_poll_summary.py` (7).
  Проверено: `pytest tests/test_poll_summary.py tests/test_migrations.py -q` →
  11 passed; `pytest tests -q` → 5394 passed, 11 skipped.
- **P4-19 (F-604) — опрос уважает контакты и тихие часы: сделано.**
  `PollAskTask` получил две двери: `allowed(person, author)` через тот же
  `ContactStore.allowed` (F-602) и `quiet(home, person)` через те же тихие часы,
  что у интеркома; упавшая проверка согласия означает тишину, а не вопрос, и
  отчёт прохода различает `denied`, `quiet` и `left`. Тесты
  `tests/test_poll_asking.py` (12). Проверено: `pytest
  tests/test_poll_asking.py -q` → 12 passed; `pytest tests -q` → 5397 passed,
  11 skipped.
- **P4-20 (F-606) — матрица прав гостя в одной точке: сделано.** Новый
  `hub/guest_access.py`: таблица `MATRIX` из восьми строк ТЗ (время, погода,
  свет, выключатель — можно; ПК, память, `restricted: true`, интерком — нельзя),
  `tool_denial`/`intercom_denial` и отказы на ru/en/es. Матрица применяется в
  `Connection._permission_check` перед ролевой таблицей D-07 (строго — к роли
  `guest`, незнакомому голосу — только запреты, сохраняя общие команды комнаты и
  свои заметки F-415) и в едином гейте `_interhome_gate`, где её получают
  интерком, опрос и вопрос о присутствии. Живёт под выключателем
  `server.permissions_enabled`. Тесты `tests/test_guest_access.py` (14).
  Проверено: `pytest tests/test_guest_access.py -q` → 14 passed; `pytest tests
  -q` → 5411 passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего
  нет:** строка `restricted` подключается к реальным устройствам в P4-21;
  выдачу доступа по слову владельца даёт P4-22.
- **P4-21 (F-606) — `restricted: true` у устройств и сцен: сделано.** Флаг
  переносится с клиента (`common/client_config.py::DeviceConfig.restricted`,
  первым классом, а не через свободную сумку `params`) в БД хаба
  (`hub/devices.py::Device.restricted`; колонка `devices.restricted` была в
  схеме 14 и до сих пор не читалась) и обратно. `Connection._device_restricted`
  спрашивает оба источника по имени устройства — список `hello` этой комнаты и
  строку `devices` этого дома, — а `_permission_check` отбивает гостя фразой про
  хозяина, не «устройство не найдено». Сцена ограничена, если хоть один её
  device-шаг называет ограниченное устройство; владелец дома получает настоящий
  исход сцены. Тесты `tests/test_restricted_devices.py` (8). Проверено:
  `pytest tests/test_restricted_devices.py -q` → 8 passed; `pytest tests -q` →
  5419 passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:**
  живых адаптеров устройств в песочнице нет.
- **P4-22 (F-606) — «разреши ему музыку»: окно доступа гостю: сделано.**
  Миграция `0021_guest_grants.py` (схема 22) и `GuestGrantStore`: окно живёт
  до `expires_at` и по его наступлении исчезает САМ, поэтому отзывать нечего.
  `Connection._guest_grant_turn` пускает к выдаче только хозяина дома, берёт
  названного гостя или единственного видимого, а на двух гостях без имени
  спрашивает; окно открывает ровно музыку (`device_set media_play` и медиа-команды
  `pc_control`), и это же право проверяет `_permission_check`. Выдача и отказ
  пишутся в `audit` (`guest.grant`). Длина окна — `server.identity.guest.
  grant_window_s` (30 минут). Тесты `tests/test_guest_grants.py` (27).
  Проверено: `pytest tests/test_guest_grants.py -q` → 27 passed; `pytest tests
  -q` → 5446 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** других предметов, кроме музыки, окно не открывает; немедленного
  отзыва голосом нет — окно кончается по TTL.
- **P4-23 (F-606/F-415/F-212) — гость не получает память дома и общий профиль:
  сделано.** Гость, зарегистрированный комнатой (F-210: членство с ролью
  `guest`), теперь считается гостем в `Connection._home_guest()` — раньше его
  собственное членство делало его «своим», и он получал память дома. Заодно
  исправлена настоящая ошибка: `_prompt_memory` вызывался в рабочем потоке, а
  он спрашивает БД хаба про говорящего (DECISIONS P1-44) — вопрос падал, и
  гость получал память комнаты; теперь память читается на цикле. «Общий профиль»
  (имена жильцов) скрыт от гостя и в per-turn префиксе, и в `presence_text`:
  говорится, сколько людей в комнате, а не кто. Тесты
  `tests/test_guest_isolation.py` (4), плюс 107 в смежных наборах. Проверено:
  `pytest tests/test_guest_isolation.py -q` → 4 passed; `pytest tests -q` →
  5450 passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:**
  живой камеры и микрофона в песочнице нет.
- **P4-24 (F-607) — настройки человека: сделано.** Миграция
  `0022_person_preferences.py` (схема 23) и `PersonPreferencesStore`: язык,
  голос ответа, wake-фраза и стиль одной строкой на человека, `ON DELETE
  CASCADE` по человеку. Незнакомый стиль — ошибка с внятным списком
  (`default|brief|formal|playful`), а не тихое «по умолчанию»; язык пишется и
  в каноническое `persons.preferred_language` (F-106), и в реестр голосов;
  `set` пишет только заданные поля и валидирует до записи. Тесты
  `tests/test_person_preferences.py` (12). Проверено: `pytest
  tests/test_person_preferences.py tests/test_migrations.py -q` → 15 passed;
  `pytest tests -q` → 5461 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто. **Чего нет:** настройки пока никто не читает в живом ходе — это
  P4-25…P4-28.
- **P4-25 (F-607) — голос ответа едет с человеком: сделано.** `TtsEngine`
  получил `voices()`/`knows_voice()`/`with_voice()`: копия движка с другим
  `speaker` делит загруженную модель, а недоступный голос оставляет голос
  комнаты (иначе комната услышала бы тишину). `Connection._reply_voice`
  подставляет голос говорящего в ответ, а `_say_proactive(name=...)` — в
  приветствие по имени, напоминание, вопрос и свод опроса, интерком и
  брифинг. Тесты `tests/test_reply_voice.py` (11). Проверено: `pytest
  tests/test_reply_voice.py -q` → 11 passed; `pytest tests -q` → 5472 passed,
  11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:** живого
  Silero/Kokoro в песочнице нет.
- **P4-26 (F-607) — wake-фраза и стиль ответа из профиля: сделано.**
  `_person_wake_word` добавляет личную фразу человека к словам комнаты, а
  `style_instruction` даёт модели строку стиля (`default` — пусто, `brief`,
  `formal`, `playful`); стиль едет в per-turn префиксе рядом с языком.
  Незнакомый стиль отвергается при записи, а плохой стиль, найденный в БД,
  попадает в лог как ошибка и не подменяется молча. Тесты
  `tests/test_person_wake_and_style.py` (7). Проверено: `pytest
  tests/test_person_wake_and_style.py -q` → 7 passed; `pytest tests -q` →
  5479 passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:**
  клиент по-прежнему слушает фразы своего конфига — личная фраза принимается
  хабом, а не переключает микрофон клиента.
- **P4-27 (F-607) — любимые сцены человека: сделано.** Миграция
  `0023_person_scenes.py` (схема 24) — таблица `person_scene_favourites` с
  `ON DELETE CASCADE` (уходит человек — уходят и его сцены). Три новых метода
  `PersonPreferencesStore`: `favourite_scenes`, `add_favourite_scene` (имя
  сравнивается без регистра в Python — SQLite складывает только ASCII, поэтому
  «ВЕЧЕР» и «вечер» иначе стали бы двумя сценами), `remove_favourite_scene`.
  `hub/scenes.py::usual_scene_request` узнаёт «как обычно / как всегда / мою
  любимую сцену / the usual / my favourite scene / la de siempre / mi escena
  favorita», а `Connection._favourite_scene_turn` ищет ИМЯ человека среди сцен
  ТОЙ комнаты, где он стоит, и запускает первой ту, что здесь есть. Права
  решает сам дом: сцена с `restricted: true` устройством и гость получают
  гостевой отказ (`guest_access.denial(RESTRICTED)`), а не «не нашёл».
  Нераспознанный голос, пустой список и отсутствие сцены честно
  проговариваются на языке человека (`favourite_line`, ru/en/es). Тесты
  `tests/test_favourite_scenes.py` (20): девять «как обычно» и четыре чужие
  фразы; дедупликация без регистра; сцена человека запускается в его комнате
  и в другой, где она есть; комната без этой сцены говорит об этом; права дома
  решают (`restricted` гость → «хозяин»); без любимых — честный ответ;
  нераспознанный голос не запускает чужую сцену. Проверено: `pytest
  tests/test_favourite_scenes.py -q` → 20 passed; `pytest tests -q` → 5499
  passed, 11 skipped; `ruff check .` и `mypy common` чисто; `python -c "import
  hub.app"` ok. **Чего нет:** панели, где человек отметил бы сцену
  «любимой» — сейчас её можно только назвать голосом; личная фраза «включи
  любимую» принимается хабом, а не будит микрофон клиента.
- **P4-28 (F-607) — настройки голосом, аудит и F-212: сделано.** Новый
  `hub/preference_commands.py` строго разбирает «отвечай по-английски»,
  «говори кратко», «говори голосом X», «просыпайся на слово X» на трёх
  языках (и «speak English», «be brief», «responde en español»), а
  `Connection._preference_turn` пишет их в `person_preferences` и кладёт
  каждый реально изменившийся ключ в `audit` (`preference.language` и т. д.) —
  повторная команда не событие. «Говори медленнее» — темп речи, которого в
  F-607 нет: хаб честно отвечает, что это настройка дома, и предлагает то,
  что умеет. `Connection._profile_here` (`shared_identity.visible_in`) закрывает
  и запись, и чтение профиля: в чужом доме без `share_identity` настройки не
  меняются и не читаются (уточнение к P4-25…P4-27). Тесты
  `tests/test_preference_voice.py` (34). Проверено: `pytest
  tests/test_preference_voice.py -q` → 34 passed; `pytest tests -q` → 5534
  passed, 11 skipped; `ruff check .` и `mypy common` чисто; `python -c "import
  hub.app, hub.preference_commands"` ok. **Чего нет:** панели настроек
  человека — всё меняется голосом; живого TTS-движка в песочнице нет.
- **P4-29 (F-711) — телефон как клиент: сделано.** `client.kind` едет в
  `hello`, телефон не рекламирует камеру, а хаб отвергает от него кадры
  камеры/экрана (`PHONE_FORBIDDEN_INPUTS`) и команды ПК
  (`PHONE_FORBIDDEN_TOOLS` — `pc_control`, `run_command`, `browser_control`)
  до отправки. Тесты `tests/test_phone_client.py` (11). Проверено: `pytest
  tests/test_phone_client.py -q` → 11 passed; `pytest tests -q` → 5545 passed,
  11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:** P4-30 —
  готовая реплика/транскрипт от телефона; приложения-телефона в репозитории
  нет, проверено протокольное поведение хаба.
- **P4-30 (F-711) — готовый транскрипт и тот же ход: сделано.** Новое
  сообщение `utterance_text` (`UtteranceText`) несёт слова, которые телефон
  распознал сам; `Connection._on_utterance_text` проводит их через ту же
  турн-машину (`_handle_utterance(..., transcript=...)` пропускает STT и
  D-03), ответ идёт как обычно — `say` + TTS, а PCM-путь (локальный VAD)
  остаётся нетронутым. Тесты `tests/test_phone_voice_turn.py` (6) на настоящих
  ходах. Проверено: `pytest tests -q` → 5551 passed, 11 skipped; `ruff check .`
  и `mypy common` чисто. **Чего нет:** P4-31 — сценарий 8 целиком.
- **P4-31 (F-711, сценарий 8) — «в машине»: сделано.** `tests/test_phase4_scenarios.py`
  проверяет настоящий ход: телефон (`kind: phone`) стримит PCM, хаб узнаёт
  голос владельца и отвечает его личной памятью скриптовым роутером (модель
  не нужна), ответ идёт кадрами `transcript`/`say` + TTS. Проверено: `pytest
  tests/test_phase4_scenarios.py -q` → 1 passed; `pytest tests -q` → 5552
  passed, 11 skipped.
- **P4-32 (F-712) — подписки телефона и очередь пуша: сделано.** Миграция
  `0024_push_subscriptions.py` (схема 25) добавила `push_subscriptions`,
  `push_outbox` и `clients.person_id`; новый `hub/push.py` держит подписки,
  очередь и транспорт за флагом: без ключа (или без `pywebpush`) сообщение не
  «отправляется», а честно ложится в очередь (`PushResult.queued`). Тесты
  `tests/test_push.py` (19 с миграционными) — 15 в файле. Проверено: `pytest
  tests/test_push.py tests/test_migrations.py -q` → 19 passed; `pytest tests
  -q` → 5567 passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего
  нет:** P4-33 — доставка напоминаний/интеркома в очередь и выдача при
  подключении телефона.
- **P4-33 (F-712) — напоминание и интерком уходят на телефон: сделано.**
  Телефон закреплён за человеком (`clients.person_id`, миграция
  `0025_intercom_push.py`), напоминание человеку не дома и интерком адресату
  не в комнате уходят пушем или честной очередью (`DeliveryState.PUSHED`/
  `QUEUED`, `intercom_messages.pushed_at` — пуш один раз, сообщение всё ещё
  ждёт «когда придёт»), а при подключении телефона `_on_hello` →
  `_flush_push_outbox` выдаёт очередь (`say` + TTS) и только тогда помечает
  `delivered`. Тесты `tests/test_push_delivery.py` (15). Проверено: `pytest
  tests -q` → 5578 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** реального провайдера пуша в песочнице; P4-34…P4-36.
- **P4-34 (F-704) — отчёт собран из настоящих источников: сделано.**
  `hub/digest.py::collect` берёт день по часам ДОМА (`day_window`, часовой
  пояс `homes.tz`) и читает настоящие `presence_events`, `audit`,
  `dialog_turns` и расход API. Расход больше не пустая таблица: `ApiBudget`
  пишет `created_at` каждого запроса, а `digest._read_ledger` считает день и
  месяц из журнала `data/api_usage.sqlite3` (`amount` — консервативно, вместе
  с не подтверждёнными резервациями). Строка без отметки времени честно
  входит только в месяц. Пустой день говорит «за сутки записей нет»,
  сломанный источник назван в `missing` и в тексте отчёта. Тесты
  `tests/test_digest.py` (9). Проверено: `pytest tests -q` → 5591 passed,
  11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:** расписания
  и канала Telegram (P4-35), деградаций хода в отчёте (P4-36).
- **P4-35 (F-704) — расписание, часы дома и Telegram: сделано.**
  `hub/app.py::_digest_task` строит задачу из конфига домов, `_hub_scheduler`
  ставит job `digest.daily`, а `_send_digest` отправляет отчёт в личный чат
  владельцев дома (`HomeOwners`). «Ровно один отчёт в день» держит таблица
  `digest_runs`: строка занимается до отправки, повторный запуск (в том числе
  вторым процессом на той же базе) не отправляет второй отчёт, а неудачная
  отправка строку освобождает. Время — по часам ДОМА, поэтому Киев и Чикаго
  получают отчёты за разные сутки. Тесты `tests/test_digest_schedule.py` (9).
  Проверено: `pytest tests -q` → 5600 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** живого Telegram; деградации хода в
  отчёте (P4-36).
- **P4-36 (F-704) — «что не удалось» названо, а не спрятано: сделано.**
  Миграция `0027_dialog_turn_degraded.py` добавила `dialog_turns.degraded`
  (стадии, которые не успели внутри хода: STT, диаризация, узнавание голоса,
  раунд модели), `record_dialog_turns` пишет их вместе с ходом, а
  `Connection._store_dialog_turns` передаёт их из `self._degradations` —
  теперь деградация переживает перезапуск хаба. Отчёт показывает неполные
  ходы отдельным разделом, отказы и провалы действий — с целью и причиной,
  заголовок раздела называет истинное число записей, а усечённый перечень
  говорит «и ещё K». Старая база без колонки честно называет её недоступным
  источником. Тесты `tests/test_digest_problems.py` (6). Проверено: `pytest
  tests -q` → 5606 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** живого стенда; метрики Prometheus (P4-37…P4-39).
- **P4-37 (F-707) — `/metrics` для Prometheus: сделано.** Новый
  `hub/metrics.py` печатает текстовую экспозицию: стадии хода по домам
  (гистограмма с корзинами 50–3000 мс), очередь GPU (длина, отказы,
  ожидание по классам, занятость по домам), память видеокарты из
  `torch.cuda.mem_get_info()` (без импорта torch ради метрики), ошибки
  (неудачные ходы, отказ токена, непонятное сообщение, отброшенные кадры,
  упавшие задачи планировщика) и расход бюджета API из настоящего журнала.
  Метрики хода заполняет сам хаб в `_finish_utterance`, поэтому `/metrics` и
  `/health.utterances` не расходятся. Секция `server.metrics.enabled`
  объявлена в обоих конфигах. Тесты `tests/test_metrics.py` (8). Проверено:
  `pytest tests -q` → 5614 passed, 11 skipped; `ruff check .` и `mypy common`
  чисто. **Чего нет:** живого Prometheus/Grafana; дашборд (P4-38) и проверка
  «в метриках нет секретов» (P4-39).
- **P4-38 (F-707) — дашборд Grafana в репозитории: сделано.**
  `deploy/grafana/rowan-dashboard.json` (uid `rowan-hub`, 7 панелей: стадии,
  дома, очередь GPU, VRAM, ошибки и деградации, бюджет API, уровни моделей) и
  `deploy/grafana/README.md` с настройкой scrape и импортом. Каждая метрика из
  запросов дашборда сверяется тестом с настоящим экспортом хаба
  (`tests/test_grafana_dashboard.py`, 4 теста), поэтому переименованная метрика
  не оставит молча пустую панель. Проверено: `pytest tests -q` → 5618 passed,
  11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:** живых
  Prometheus и Grafana в песочнице.
- **P4-39 (F-707) — метрики без секретов и на том же входе, что `/health`:
  сделано.** Белый список меток (`home`, `stage`, `class`, `kind`, `job`,
  `level`, `state`, `le`) закреплён тестом на живом ответе; имени человека,
  текста реплики и токенов в `/metrics` нет (проверено с выставленными в
  окружении секретами и настоящим ходом, где говорил «Максим Петров»), а сам
  `hub/metrics.py` не читает окружение ни разу. Endpoint живёт в том же
  приложении и на том же порту, что `/health`; своего host/port у метрик нет
  (`MetricsConfig` знает только `enabled`), а выключенные метрики отвечают
  404. Тесты `tests/test_metrics_privacy.py` (4). Проверено: `pytest tests
  -q` → 5622 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** живой Grafana.

## Фаза 4 закрыта (P4-01…P4-40)

Сценарии приёмки раздела 2 ТЗ пройдены настоящими ходами:

- **Сценарий 5 «скажи Максу, что я иду»** — слова автора звучат в комнате
  Макса в ДРУГОМ доме того же хаба (другая общага), сообщение лежит в его
  доме (`intercom_messages.home_id`), автор назван. Пока обе стороны не
  подтвердили знакомство, доставки нет: отказ на настоящем пути и строка
  `denied` в аудите.
- **Сценарий 8 «владелец в машине»** — телефон (`client.kind = phone`) шлёт
  PCM после своего VAD, хаб отвечает его ЛИЧНОЙ памятью тем же протоколом и
  тем же голосом.
- **Сценарий 10 «скилл друга»** — скилл `coffee` лежит в
  `data/homes/<его дом>/skills/`, отвечает настоящим ходом в его доме и
  ЧЕСТНО отсутствует в соседнем доме того же хаба.

Что появилось за фазу: `hub/contacts.py` (F-602), `hub/intercom.py` (F-601),
`hub/polls.py` (F-604), `hub/guest_access.py` (F-606),
`hub/person_preferences.py` + `hub/preference_commands.py` (F-607),
`hub/push.py` (F-712), `client/phone` = `client.kind: phone` (F-711),
`hub/digest.py` (F-704), `hub/metrics.py` + `deploy/grafana/` (F-707), шесть
миграций (`0016_contacts_consent` … `0021_push_subscriptions`, `0026_digest_runs`,
`0027_dialog_turn_degraded`), протокол v2 (`intercom_reply`, `poll_answer`,
`push_token`, `card`, `utterance_text`, `camera_clip`-ответы).

  **Чего нет:** живого стенда (5090, две общаги, настоящий Telegram, Grafana) —
  всё, что требует железа, помечено в строках P4-xx и в `DECISIONS.md`;
  примерочные замеры латентности фазы 4 не проводились, потому что железо
  появится вместе со стендом.

Задачи P4-01…P4-40 закрыты, доказательство каждой — в `PROGRESS.md` (рядом с
каждым `[x]` записано, что именно запускалось и с каким результатом).
Итоговая проверка фазы: `pytest tests -q` → 5626 passed, 11 skipped;
`ruff check .` и `mypy common` чисто; `python -c "import hub.app"` → ok.
Дальше — фаза 5 (расширения, раздел 16 ТЗ): задачи P5-01…P5-32 в
`PROGRESS.md`, план — в `PLAN.md`.

## Telegram: доп. администраторы и назначения уведомлений (22.09.2026)

- **TG-01 — доступ к `/tools` для названных аккаунтов: сделано.** Новое поле
  `server.telegram.admin_user_ids` (`common/config.py`): аккаунты получают те
  же права, что владелец хаба — панель в личном чате и в группе, все
  возможности, личные уведомления; владелец остаётся единственным, кого
  нельзя изменить. Право проверяется на каждый клик по конфигу, поэтому
  удалённый из конфига аккаунт теряет панель сразу (`_hub_admin_now`).
  На этом хабе в список внесены `8928749210`, `6617808228`, `1328190425`.
  Проверено: `pytest tests/test_telegram_admin.py
  tests/test_telegram_admin_state.py -q` → 41 passed (в полном прогоне
  `pytest tests -q` → 4391 passed, 10 skipped; `ruff check .` и `mypy common`
  чисто).
- **TG-02 — «Private chat (everyone)»: сделано.** Назначение `owner` теперь
  рассылает уведомление в личные чаты всех, у кого есть доступ
  (`TelegramAdminState.private_recipients` → `PresenceAlerts._private_recipients`),
  частичная доставка помечается `uncertain` и не повторяется. На этом хабе
  получателей четверо (владелец + три админа). Проверено:
  `pytest tests/test_presence_alerts.py -q` → 28 passed.
- **TG-03 — группы в «Destination of notifications»: сделано.** `destination`
  принимает `owner` или `group:<chat_id>`; хаб запоминает группы из апдейтов
  (`TelegramChat._remember_chat`, максимум 20), панель показывает их
  названиями, а доставка идёт явным `group_chat_id` (`hub/telegram.py`).
  Группа заказчика `-1003570242441` («RowanAI Notifications», подтверждена
  `getChat`) внесена в список. Проверено: `pytest tests/test_telegram_chat.py
  tests/test_alert_rules.py -q` → 68 passed.

## Обращения заказчика 22.09.2026 (вторая часть)

- **TG-04 — фото/экран по имени ПК в самой просьбе: сделано.** `_named_workplace`
  ищет в тексте id рабочего места, его имя или имя камеры (регистр и пробелы
  не важны), `_selected_telegram_room` уводит на него весь ход, не меняя
  сохранённый выбор; названный и выключенный ПК получает ответ
  `the computer "…" is not connected`. Домашний селектор «Home» собран из
  конфига и домов подключённых ПК, а не из базы с тестовыми домами.
  Проверено: `pytest tests/test_telegram_admin.py tests/test_admin_workplaces.py
  tests/test_presence_alerts.py -q` → 120 passed (полный прогон
  `pytest tests -q` → 4398 passed, 10 skipped; `ruff check .` и `mypy common`
  чисто).
- **TG-05 — «снимать, пока человек не выйдет из кадра» кусками по ≤60 с:
  сделано.** Новое поле правила `record_until_clear` (панель: *Keep recording
  while the person stays*), потолок `EPISODE_MAX_PARTS` = 20 видео на
  срабатывание, комната считается пустой после 6 с без кадров присутствия
  (`source_presence` пишется на каждом кадре, `_people_now` читает). Длина
  одного видео `clip_seconds` поднята до 3…60 в протоколе
  (`CameraClipRequest`, `CameraRequest`), в клиенте (`client/camera_clips.py`)
  и в правиле. Проверено: `pytest tests/test_alert_episode.py
  tests/test_camera_clips.py tests/test_presence_alerts.py -q` → 60 passed.
- **TG-06 — половина многошаговой просьбы больше не теряется: сделано.**
  `site_step_unfinished` (ТЗ 5.3, D-04) видит, что в просьбе назван сайт или
  адрес, а ни одно действие хода до него не дошло, и включает self-check даже
  при `verify_actions: false`; предложение запомнить браузер
  (`remember_offer`) молчит, пока в той же фразе есть шаг с сайтом. В промпте
  добавлено правило про «одна фраза — несколько шагов». Проверено:
  `pytest tests/test_site_step.py tests/test_browser_choice_memory.py -q` →
  64 passed.
- **TG-07 — журнал «кто что менял»: сделано.** `hub/telegram_audit.py` пишет
  `data/telegram/audit.log` (ID аккаунта, имя, действие, дом, значения,
  результат) рядом с прежними SQLite-записями; `_audit` панели теперь
  сохраняет и **значения**, а не только имена полей. Проверено:
  `pytest tests/test_telegram_audit_log.py tests/test_admin_backend.py -q` →
  49 passed.
- **TG-08 — уведомления на английском: сделано.** `_event_text` и подпись
  целиком переведены (`the camera spotted a person`), имя человека не
  переводится. Проверено: `pytest tests/test_alert_rules.py -q`.

## Журнал фазы 5 (расширения, раздел 16 ТЗ)

- **P5-01 (F-305) — индексатор кадра: сделано.** `hub/object_index.py`:
  `YoloDetector` (пакет `ultralytics` и веса грузятся ЛЕНИВО, на первом
  кадре; нет пакета или битые веса — `DetectorUnavailable` с названием
  причины), `DetectedObject` (label, bbox, confidence — строгая модель) и
  `SceneIndexer.index(home_id, frame, ts=…, media_ref=…)`, пишущий настоящие
  строки в `objects_index` через `ObjectMemoryStore.record`. Главное правило —
  «не смотрела» ≠ «не нашла»: нет кадра или детектора — `ok=False`, таблица
  НЕ трогается; пустой список находок при работающем детекторе — `ok=True` и
  «в кадре этого нет». Метки YOLO английские, а вопрос звучит на языке
  человека, поэтому `hub/object_memory.py` получил `LABEL_GROUPS`
  (ключи/keys/llaves и ещё 15 групп), а `normalize_label` сводит их к группе.
  Секция `server.objects` (`enabled: false`) в конфиге. Проверено:
  `pytest tests/test_object_index.py -q` → 11 passed; `pytest tests -q`
  → 5637 passed, 11 skipped; `ruff check .` и `mypy common` чисто.
  **Чего нет:** живого `ultralytics`/камеры в песочнице — детектор подставной.
- **P5-02 (F-305) — CLIP-эмбеддинги регионов: сделано.**
  `hub/object_embed.py`: `ClipEmbedder` (`open_clip`/`torch` лениво),
  `crop_box` (бокс за краем кадра обрезается; вырезка тоньше двух пикселей не
  существует — `None`, а не «вектор пустой картинки»), `vector_dim`.
  `SceneIndexer(embedder=…)` считает вектор региона и кладёт его в строку
  `objects_index` (`vector`+`dim`), а `ObjectMemoryStore.record` best-effort
  зеркалит его в `vec_objects_index` (ошибка расширения — в лог; строка
  остаётся источником истины). Без CLIP объект всё равно записывается (его
  видно глазами), просто без вектора, и это названо в отчёте (`embedded`,
  `embed_error`). Проверено: `pytest tests/test_object_embed.py -q` → 7 passed;
  `pytest tests -q` → 5644 passed, 11 skipped. **Чего нет:** живого `open_clip`
  и GPU — эмбеддер подставной.
- **P5-03 (F-305) — расписание индексации: сделано.** `SceneIndexTask` умеет
  синхронный и асинхронный провайдер кадра и принимает `(кадр, момент)`, тогда
  находка получает время СЪЁМКИ, а не время прохода; `changed(home, frame)` —
  «сцена изменилась» (хэш последнего ПРОИНДЕКСИРОВАННОГО кадра), `on_indexed`
  вызывается только после удачного прохода; неизменившийся кадр честно идёт в
  отчёт как `unchanged`. Хаб: job `objects.index` в `_hub_scheduler`,
  `_home_frame_for_indexing` берёт кадр, который комната УЖЕ прислала (камеру
  ради фоновой работы не будим), нет живого клиента — `no_frame`.
  `_keep_indexed_frame` кладёт кадр в настоящий медиах дома
  (`MediaStore.save_bytes`, kind `frame`, TTL 3 дня), поэтому ответ «Кадр
  сохранён.» обещает настоящий файл. Проверено:
  `pytest tests/test_object_index_task.py -q` → 7 passed; `pytest tests -q`
  → 5651 passed, 11 skipped.
- **P5-04 (F-305) — показ найденного кадра: сделано.** `_media_bytes(media_ref)`
  достаёт картинку по настоящей ссылке из таблицы `media` (нет файла/истёк
  TTL — `None`, и обещание кадра исчезает, слова остаются);
  `Connection._show_found_frame` показывает кадр в комнате существующим путём
  `_send_image_show` и отправляет его владельцам дома в Telegram
  (`_send_photo_to_home_owners`); неудача Telegram не отменяет и не меняет
  ответ в комнате. `_where_turn` после честного текста «ключи — стол в 14:30.
  Кадр сохранён.» реально показывает кадр. Проверено:
  `pytest tests/test_object_frame.py -q` → 6 passed; `pytest tests -q`
  → 5657 passed, 11 skipped. **Чего нет:** живого Telegram и HUD — приёмники
  подставные, JPEG настоящий (Pillow).
- **P5-05 (F-309) — зоны кадра и маска «не анализировать»: сделано.**
  `common/config.py::FrameZone` (`name`, `points`, `mask`) — полигон в
  НОРМАЛИЗОВАННЫХ координатах кадра (0…1), не меньше трёх точек, каждая точка
  `[x, y]` внутри кадра; `HomeConfig.zones` плюс валидатор уникальных имён.
  `hub/zones.py`: `Zone.contains` (ray casting), `FrameZones.zone_at` (зона по
  ЦЕНТРУ бокса; маска сюда не попадает — это два разных ответа),
  `FrameZones.masked_at`, `FrameZones.describe` и `zones_for_homes`.
  `SceneIndexer` принимает `zone_of(home, bbox, size)` и `masked(home, bbox,
  size)`: находка в маске НЕ пишется вовсе (обещание «эту область не смотрят»
  важнее полноты памяти), а её число честно видно в отчёте полем `masked`;
  ошибка `zone_of` не теряет объект (зона остаётся пустой). `jpeg_size(frame)`
  читает размер кадра из SOF-заголовков JPEG без декодирования. Хаб:
  `_frame_zones`/`_zones_of`/`_zone_of_home`/`_masked_in_home`, подстановка в
  `_object_indexer`; `homes.list` админки отдаёт `zones` через `describe()`,
  поэтому владелец видит настроенные зоны и признак маски, не читая yaml.
  `config.example.yaml` — пример `zones:` в блоке `homes:`. Проверено:
  `pytest tests/test_zones.py -q` → 7 passed; `pytest tests -q` → 5664 passed,
  11 skipped; `ruff check .` → All checks passed; `mypy common` → Success;
  `python -c "import hub.app"` → ok. **Чего нет:** редактора полигонов в
  админке (пока только чтение) и вырезания маски на КЛИЕНТЕ до отправки
  кадра — это P5-06; зоны рисует владелец, поэтому живой камеры с размеченным
  кадром в песочнице нет.

Итог промежуточной проверки фазы: `pytest tests -q` → 5664 passed, 11 skipped;
`ruff check .` и `mypy common` чисто; `python -c "import hub.app"` → ok.

- **P5-06 (F-309) — маска вырезается на клиенте до отправки кадра: сделано.**
  Общий модуль `common/frame_zones.py`: `FrameZone` (переехала из
  `common/config.py`), `parse_zones`, `mask_polygons`, `zones_rev`/`masks_rev`
  (отпечаток масок в 12 hex, не зависящий от порядка зон). Клиент
  (`client/camera.py`): зоны приходят своим конфигом (`client.camera.zones`)
  и патчем хаба (`set_zones`); `_mask_frame` закрашивает маску ЧЁРНЫМ на
  КОПИИ кадра до `cv2.imencode` (JPEG не умеет дырок), поэтому кадр уходит
  уже без области, а исходный кадр камеры не портится; маскируются и кропы
  тела до вырезки; заголовки `camera_frame`/`body_crop` несут `masked` и
  `zones_rev`; не удалось закрасить — кадр не отправляется вовсе. Хаб:
  `_frame_mask_problem` + отказ в `_deliver_image` (камера) и
  `_deliver_body_crop`: кадр дома с масками без `masked: true` и совпадающего
  `zones_rev` не идёт ни в присутствие, ни в индексатор F-305, ни в копию для
  обучения; ждавший `camera_request` получает ошибку с домом и ожидаемым
  отпечатком; отказ виден в логе, в `/health.unmasked_frames` и строкой
  `camera.frame_unmasked` в аудите. Зоны едут комнате тем же `config_update`
  (`home_patch`, `current_room_frame`), а миграция `0028_home_zones_rev.py`
  (схема 28) хранит `homes.zones_rev`, поэтому правка ТОЛЬКО маски в конфиге
  больше не теряется. Проверено: `pytest tests/test_frame_mask.py -q` → 14
  passed; `pytest tests -q` → 5678 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто; `python -c "import hub.app"` → ok. **Чего нет:** живой
  камеры и редактора полигонов в админке.

- **P5-08 (F-306) — жесты руки, ладонь останавливает речь: сделано.**
  `client/gestures.py`: `MediaPipeHands` (ленивый импорт `mediapipe` на CPU
  комнаты, отсутствие пакета — названная причина), `recognize` для трёх
  жестов ТЗ (открытая ладонь, большой палец вверх, указательный палец) по
  расстояниям до запястья (жест не зависит от поворота кисти), `GestureHold`
  («ладонь дольше 1 с» срабатывает один раз, пока руку не уберут) и
  `GestureService` (флаг дома, ограничение частоты, тишина без mediapipe).
  Кадр уже пойман камерой (`CameraService.set_frame_listener`), наружу уходит
  только событие жеста. Ладонь обрывает приветствие и чистит очередь
  динамика, а флаг `_stopped_by_gesture` гасит остаток потока синтеза до
  `tts_end`. Флаг включается НА ДОМ: `homes[].settings.gestures` (едет
  комнате тем же `config_update`, что и остальные настройки дома) или своя
  секция `client.gestures`. Проверено: `pytest tests/test_gestures.py -q` →
  10 passed; `pytest tests -q` → 5699 passed, 11 skipped; `ruff check .` и
  `mypy common` чисто. **Чего нет:** живого `mediapipe`/камеры в песочнице
  (21 точка подставная); подтверждение большим пальцем — P5-09, указание —
  P5-10.

- **P5-09 (F-306/F-113) — большой палец вверх подтверждает вместо «да»:
  сделано.** `_handle_voice_confirmation` публикует на время вопроса тот же
  thread-safe ответ, что и кнопка HUD (`self._confirmation_resolve`), а
  `JarvisClient._on_gesture('thumb_up')` отвечает им `True` — только если хаб
  правда что-то спросил, иначе жест молчит и это видно в логе. Ответ уходит
  хабу обычным `voice_confirmation_result` с тем же `id`, поэтому путь один и
  тот же, что у устного «да» и кнопки. Проверено:
  `pytest tests/test_gestures.py -q` → 12 passed; `pytest tests -q` → 5701
  passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:**
  живого `mediapipe`.

- **P5-10 (F-306) — «что это?» по направлению указания: сделано.** Клиент
  считает точку ЗА кончиком указательного пальца (`point_hint`: линия
  «сустав у ладони → кончик», продолженная на 1.5 длины пальца — человек
  показывает НА предмет, а не на свой палец) и отправляет `point_event` с
  одним лишь направлением `{x, y, at_ms, event_id}`: кадр остаётся в комнате.
  Хаб держит точку рядом с соединением (`Connection._on_point_event`,
  `_point_hint`, TTL 10 с) и в `look_at_camera` описывает вырезанную вокруг
  указанной точки часть кадра (`_crop_around_point`, 45 % меньшей стороны),
  а не всю комнату; лица в вырезанной части не ищутся. Проверено:
  `pytest tests/test_point_hint.py -q` → 7 passed; `pytest tests -q` → 5708
  passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего нет:**
  живого `mediapipe`/SAM3; у здешнего `Sam3Engine` промпт текстовый, поэтому
  вырезанная по направлению часть идёт vision-модели, а SAM3 получит её,
  когда вопрос назовёт предмет.

- **P5-07 (F-311) — объекты внимания со зоной и уведомлением: сделано.**
  Общая таблица слов вынесена в `common/object_labels.py` (`LABEL_GROUPS`,
  `normalize_label`; `hub/object_memory.py` их ре-экспортирует), а
  `common/attention_objects.py` знает классы внимания ТЗ (кошка/собака/
  посылка) со словами ru/en/es, синонимы детектора и строку «где объект»:
  «посылка у дверь» / «dog at door». Имя зоны не переводится (это слова
  владельца), фраза уведомления английская (TG-08). Клиент считает зону
  находки по полигонам дома (F-309) общим ray casting
  (`common.frame_zones.zone_at`, тот же код у хаба) и шлёт `object_event`
  только на ПЕРЕХОДЕ пары (объект, зона) и только для объектов внимания;
  находка в области «не анализировать» не сообщается, приватный режим
  молчит. Хаб принимает событие (`Connection._on_object_event`) и уходит
  правилам F-702 путём звука F-109 — с зоной и уверенностью; правило,
  написанное словами владельца («посылка»), ловится по группе, а
  уведомление называет объект и зону. Проверено:
  `pytest tests/test_attention_objects.py -q` → 11 passed; `pytest tests -q`
  → 5689 passed, 11 skipped; `ruff check .` и `mypy common` чисто. **Чего
  нет:** живого MediaPipe/YOLO (модель подставная) и переключателя по
  человеку вместо дома.

- **P5-11 (F-307) — поза: «лежит неподвижно» без капризов поворота:
  сделано.** `client/posture.py`: `PoseDetector` (ленивый импорт
  `ultralytics`, `yolo11n-pose.pt`), `posture_of` (лежит/сидит/стоит по
  ТУЛОВИЩУ и ноге, а не по «прямоугольник выше, чем шире»), `StillnessWatch`
  (10 минут без смещения центра тела → `sleep` один раз; переход из лежачего
  положения в сидя/стоя → `awake`; пропажа из кадра подъёмом НЕ считается),
  `PostureService` (1 кадр в 5 с, флаг дома `homes[].settings.posture` плюс
  секция `client.posture`). Кадр берётся тот же, что у жестов и YOLO, и
  никуда не уходит. Режим сна рождается только в тихие часы дома; подъём
  замечается всегда. Проверено: `pytest tests/test_posture_sleep.py -q` →
  10 passed; `pytest tests -q` → 5739 passed, 11 skipped; `ruff check .` →
  All checks passed; `mypy common` → Success; `python -c "import client.main"`
  → ok. **Чего нет:** живого `ultralytics` и камеры в песочнице (весов нет,
  разбор позы проверен на настоящих числах 17 COCO-точек) — названо в
  `DECISIONS.md` (P5-11).

- **P5-12 (F-307/F-420) — режим сна дома и утренний подъём: сделано.**
  `migrations/0029_home_modes.py` (`home_modes`, `home_wakeups`) и
  `hub/home_modes.py::HomeModes`: режим сна переживает рестарт хаба, подъёмы
  помнятся по дням. `Connection._on_posture_event` → `_sleep_home` (режим
  `asleep` + сцена `homes[].settings.sleep_scene` настоящим `SceneRunner`;
  сцены нет — устройства не трогаются) и `_wake_home` (режим `awake` +
  подъём за узнанным человеком F-201; безымянный подъём честно назван в
  логе). Спящий дом не звонит в Telegram: уведомление идёт подписью на HUD,
  а `_briefing_entries` делает сегодняшние подъёмы поводом утренней рутины
  F-420. Проверено: `pytest tests/test_posture_sleep.py -q` → 10 passed;
  `pytest tests -q` → 5739 passed, 11 skipped; `ruff check .` → All checks
  passed; `mypy common` → Success; `python -c "import hub.app"` → ok. **Чего
  нет:** живого сценария на настоящем железе (устройства/Telegram
  подставные) — названо в `DECISIONS.md` (P5-12).

- **P5-13 (F-308) — OCR экрана вместе с vision-моделью: сделано.**
  `hub/ocr.py`: `OcrEngine` (RapidOCR — пакеты `rapidocr`/`rapidocr_onnxruntime`
  — или PaddleOCR; ленивая сборка, названный отказ без пакета/весов, один
  движок на хаб), `rapidocr_lines`/`paddleocr_lines` (обе настоящие формы
  ответов, включая `rec_texts` из PaddleOCR 3.x), `ScreenText`/`OcrLine`.
  `Connection._read_screen_text` читает ТУ ЖЕ область, что уходит
  vision-модели, в отдельном потоке (OCR CPU-шный, слот GPU-очереди 4.5 не
  занимает), а `_run_look_at_screen` возвращает `ocr_text`/`ocr_lines` вместе
  с описанием модели и укладывает строки в запрос к ней обёрнутыми
  `<<<UNTRUSTED … UNTRUSTED>>>`. Текст со скрина недоверенный (F-411/D-09):
  результат `look_at_screen` уже помечен внешним, поэтому D-09 сканирует и
  строки OCR, а инструменты по ним не запускаются. Флаг `server.ocr`
  (`enabled: false` по умолчанию). Проверено: `pytest tests/test_screen_ocr.py
  -q` → 13 passed; `pytest tests -q` → 5752 passed, 11 skipped; `ruff check .`
  → All checks passed; `mypy common` → Success; `python -c "import hub.app"` →
  ok. **Чего нет:** живых `rapidocr`/`paddleocr` и весов в песочнице (нет
  ни того, ни другого) — разбор проверен на настоящих формах ответов, отказ —
  на настоящем отсутствии пакета; названо в `DECISIONS.md` (P5-13).

- **P5-14 (F-512) — ограничения computer-use: сделано.** Правила лежат в
  `common/computer_use.py` (одна копия на хаб и клиента): `MAX_STEPS` = 15,
  allow-list приложений (пустой — запрещено всё), запрет ввода паролей,
  одноразовых кодов, карт, банковских реквизитов, сид-фраз и PIN словами
  ru/en/es, строгая модель шага. `hub/computer_use.py` ведёт прогон
  (`ComputerUseRun`/`ComputerUseRuns`): отказы видны в отчёте, отказ не
  занимает шаг, один живой прогон на дом, `stop` закрывает все.
  `client/actions/computer_use.py::ComputerUseExecutor` — ленивый `pyautogui`,
  повторная проверка каждого шага перед действием, сверка активного окна с
  allow-list (неизвестное окно = отказ), счётчик реально выполненных шагов.
  Флаг `server.computer_use` (выключен, allow-list пуст по умолчанию).
  Проверено: `pytest tests/test_computer_use.py -q` → 21 passed;
  `pytest tests -q` → 5773 passed, 11 skipped; `ruff check .` → All checks
  passed; `mypy common` → Success; `python -c "import hub.app"` и
  `import client.main` → ok. **Чего нет:** живого `pyautogui` и рабочего стола
  в песочнице (исполнитель проверен на подставном `pyautogui`); стоп-слово,
  ладонь и оверлей — P5-15, аудит и F-113 — P5-16; названо в `DECISIONS.md`
  (P5-14).

- **P5-15 (F-512/F-306) — «стоп» словом и ладонью, оверлей «Rowan управляет»:
  сделано.** Новые кадры `computer_use` (значок, хаб→комната) и
  `computer_use_step` (отчёт о шаге и о стопе, комната→хаб; телефону
  запрещён). `hub/computer_use.py::is_stop_command` узнаёт «стоп»/«stop»/
  «detente» и вежливые формы трёх языков, но не считает стопом вопрос со
  словом внутри; `Connection._computer_use_stop_turn` стоит до wake-проверки,
  закрывает прогон, снимает значок, говорит «Остановил.» и пишет аудит. Ладонь
  (F-306) останавливает прогон в комнате и отправляет `stopped` хабу, поэтому
  стоп слышен на обоих концах. Значок — новый стоячий бейдж `OverlayHUD.control`
  (не гаснет по `BADGE_HOLD_S`, держится весь прогон) с янтарным элементом в
  `hud.html`. Шаги ходят действием `computer_use_step` с политикой от хаба.
  Проверено: `pytest tests/test_computer_use_stop.py -q` → 17 passed;
  `pytest tests -q` → 5790 passed, 11 skipped; `ruff check .` → All checks
  passed; `mypy common` → Success; `python -c "import hub.app"` и
  `import client.main` → ok. **Чего нет:** живого `pyautogui`/камеры (жест и
  исполнитель на подставных); запуск прогона моделью и F-113 — P5-16; названо
  в `DECISIONS.md` (P5-15).

- **P5-16 (F-512/F-113/F-706) — аудит шага и подтверждение опасного:
  сделано.** Инструмент `computer_use` (один вызов = один шаг, `finish`
  закрывает задачу) объявлен модели и живёт на сервере; прогон стартует
  первым шагом, а выключенный дом честно отказывает. КАЖДЫЙ шаг — строка
  `computer_use.step` в `audit` (ok/denied, run_id, index, причина). Шаги,
  меняющие систему (Alt+F4, Ctrl+Alt+Del, Win+L/R, Ctrl+Shift+Esc),
  распознаются `common/computer_use.changes_system` и ждут голосового «да» по
  F-113: до ответа команда в комнату не уходит, «да» выполняет ровно
  отложенный шаг, «нет» ничего не трогает. `computer_use` добавлен в
  `GUARDED_TOOLS`, поэтому команда из кадра/страницы/чата агента не запускает.
  Проверено: `pytest tests/test_computer_use_audit.py -q` → 11 passed;
  `pytest tests/test_tools_contract.py tests/test_decision_points.py
  tests/test_injection_guard.py -q` → 51 passed; `pytest tests -q` → 5803
  passed, 11 skipped; `ruff check .` → All checks passed; `mypy common` →
  Success; python-импорты `hub.app`/`client.main` → ok. **Чего нет:** живого
  `pyautogui`/рабочего стола (исполнитель проверен на подставном) — названо в
  `DECISIONS.md` (P5-16).

- **P5-17 (F-111) — клонированный голос владельца: сделано (провайдер
  недоступен и назван).** `hub/voice_clone.py`: согласие человека по дому
  (`ConsentStore`, переживает рестарт), окно референса 10–20 с по настоящему
  WAV, ленивый провайдер F5-TTS/Chatterbox с названным отказом, кэш синтеза на
  диске с уборкой, `snapshot()` для `/health.voice_clone`. Флаг
  `server.voice_clone` (выключен по умолчанию), без согласия владельца клон не
  синтезирует ничего. Проверено: `pytest tests/test_voice_clone.py -q` → 15
  passed; `pytest tests -q` → 5818 passed, 11 skipped; `ruff check .` → All
  checks passed; `mypy common` → Success; `python -c "import hub.app"` → ok.
  **Чего нет:** ни F5-TTS, ни Chatterbox, ни GPU в песочнице нет — синтез
  проверен на подставном провайдере, а настоящий провайдер честно отвечает
  `VoiceCloneUnavailable` (подмена обычным голосом запрещена правилом «никаких
  фейков»); живой прогон — задача стенда, названо в `DECISIONS.md` (P5-17).

- **P5-18 (F-112) — эмоция в голосе: сделано (веса недоступны и названы).**
  `hub/emotions.py`: метки IEMOCAP с синонимами, превращение настоящего PCM
  (48 кГц) в 16 кГц float32, ленивый `SpeechBrainEmotion` с ЛОКАЛЬНОЙ папкой
  весов (HF id — только по `server.emotion.allow_download`), бюджет
  `timeout_ms` и порог уверенности. Эмоция едет в персональный префикс хода
  (`SpeakerProfile.emotion`) строкой «how the person sounds: … (… it never
  changes what you do)», то есть меняет СТИЛЬ ответа и не читается ни одним
  инструментом. Классификация идёт через GPU-очередь 4.5; опоздала,
  недоступна или упала — ход продолжается. `/health.emotion` показывает флаг,
  модель и счётчики. Проверено: `pytest tests/test_emotions.py -q` → 13
  passed; `pytest tests -q` → 5831 passed, 11 skipped; `ruff check .` → All
  checks passed; `mypy common` → Success; `python -c "import hub.app"` → ok.
  **Чего нет:** локальных весов `emotion-wav2vec2-IEMOCAP` и GPU в песочнице
  (`speechbrain` 1.1.1 есть, весов нет — хаб не качает их сам) — названо в
  `DECISIONS.md` (P5-18).

- **P5-19 (F-605) — общий календарь группы: сделано.** Migration
  `0030_shared_events` (событие и отметка «напомнили» переживают рестарт),
  `hub/shared_events.py` — `SharedEvent`/`SharedEventStore` (`create`,
  `upcoming`, `between`, `due_for_reminder`, `mark_reminded`, `cancel`), разбор
  речи `parse_shared_event` со словами события по ГРАНИЦАМ слов (встреча, игра,
  поход и их варианты), время берётся тем же парсером, что у напоминаний F-417;
  ответы `created_answer`/`missing_time_answer`/`list_answer`/`reminder_line`
  на ru/en/es, `SharedEventReminderTask` напоминает в каждой комнате участника и
  пишет аудит. Хаб: `_shared_events_store`, `_shared_events_task` в планировщике,
  `_calendar_question`/`_calendar_homes` (участники — дома названных людей по
  журналу присутствия, иначе все дома), `Connection._shared_event_turn` до
  wake-проверки и `_speak_calendar`. **Найдено и исправлено по пути:**
  подстрочный поиск «trip» ловил «the strip» и крал обычную реплику — заменено
  на границы слов; `_shared_events_store` через `_hub_gateway()` подменял уже
  открытое соединение и уводил запись в другую базу. Секция
  `server.shared_events` в `common/config.py`, `config.yaml` и
  `config.example.yaml`. Проверено: `pytest tests/test_shared_events.py -q` →
  13 passed; `pytest tests -q` → 5844 passed, 11 skipped; `ruff check .` → All
  checks passed; `mypy common` → Success. **Чего нет:** живого Telegram/голоса в
  песочнице (комнаты подставные); названо в `DECISIONS.md` (P5-19).

- **F-608 (P5-20) — игры между комнатами: квиз по темам сделан, «угадай, кто
  сказал» и таймер-соревнование — P5-21.** Каркас — скилл с состоянием F-407:
  `skills/games/` (`start`/`answer`/`score`/`stop`) поверх `hub/games.py`
  (`QuizEngine` с `SkillStateStore` и `SkillScheduler`, `LlmQuizGenerator`
  через `structured_json`, `answers_match` по границам слов). Партия живёт в
  состоянии ХАБА (`hub:games`), поэтому счёт по домам виден из каждой
  комнаты-участника и из `/health.games`; вопрос и счёт слышат все комнаты
  партии, очко получает та, где ответили; опоздавший ответ называет правильный
  ответ, но очка не даёт; темы нет или модели нет — хаб честно спрашивает
  тему/называет причину отказа и не выдумывает ни вопросов, ни темы. Секция
  `server.games` (`enabled: false`) в `common/config.py`, `config.yaml` и
  `config.example.yaml`. **Найдено и исправлено по пути:** движок и скилл
  держали состояние в разных местах (хаб писал в `hub:games`, скилл — в
  `livingroom:games`); `_game_engine()` через `_audit_log()` подменял уже
  открытое соединение хаба (та же ловушка, что у F-605); `_game_homes()` не
  видел комнаты из таблицы `homes`; тест F-605 проходил только до 18:00 UTC;
  подставная `_skill_context` в тестах сценария 3 не принимала параметр
  `skill=`. Проверено: `pytest tests/test_games.py -q` → 27 passed;
  `pytest tests -q` → 5918 passed, 11 skipped; `ruff check .` → All checks
  passed; `mypy common` → Success; `python -c "import hub.app, client.main"` →
  ok. **Чего нет:** живого микрофона и голоса в песочнице (комнаты подставные,
  генератор вопросов подставной, настоящий путь проверен на подставной
  модели); межкомнатный голос «угадай, кто сказал» — P5-21.

- **F-705 — панель владельца включена, и в ней видна цепочка каждого запроса
  (в том числе из Telegram).** Вопрос владельца был прямой: «хочу админ-панель,
  где все запросы, в том числе из ТГ, и все промпты ИИ, чтобы я понимал, что
  происходит». Сделано: таблица `turn_events` (миграция 0031) и
  `hub/turn_trace.py` пишут по строке на шаг хода; `turn_id` комнаты — прежний
  `utterance_id`, у Telegram — `telegram:<чат>:<сообщение>`; в цепочку попадают
  `turn` (что услышали, кто это был, что ответили, стадии stt/llm/tts),
  `decision` (rules/Jev, значение, уверенность, задержка), `tool` (аргументы,
  результат, задержка), `llm` (раунд модели и её текст), `prompt` (что именно
  ушло модели — сообщения раунда, а для картинок ещё и промпт Nano Banana),
  `image` (отказ Google с причиной) и `say` (что произнесла комната). Страницы:
  `/admin/turns` (список) и `/admin/turns/<turn_id>` (пошаговая цепочка).
  `server.web_admin.enabled: true` в `config.yaml` и `config.openai.yaml`,
  пароль — `ROWAN_ADMIN_PASSWORD` в `.env`. Панель отвечает на порту хаба
  (8770) и только под своими именами: `localhost` или IP-литерал из
  `allowed_networks`; публичный туннель ngrok получает 404, хотя запрос оттуда
  приходит с loopback. Prometheus и Grafana остаются отдельно: они про здоровье
  системы во времени (`/metrics` без текста, ТЗ 15.5). **Чего нет:** полного
  живого хода через новую панель — хаб нужно перезапустить, чтобы он поднял
  `server.web_admin.enabled: true` и прочитал `ROWAN_ADMIN_PASSWORD` из `.env`
  (владелец просил пока не запускать `run.ps1`, поэтому живой хаб не трогали).
  Проверено: `pytest tests/test_turn_trace.py tests/test_web_admin.py
  tests/test_audit.py tests/test_migrations.py tests/test_image_generation.py
  tests/test_decision_log.py -q` → 108 passed; `pytest tests -q` → 6047 passed,
  11 skipped, 15 failed (все пятнадцать — чужая незакоммиченная задача P5-21,
  `hub/guess_who.py`: `sqlite3.ProgrammingError` про соединение из другого
  потока); `ruff check .` → All checks passed; `mypy common` → Success;
  `python -c "import hub.app"` → ok. Отдельно прогнано на **копии живого**
  `data/hub.db` (4988 решений): миграция уже применена, схема на версии 31,
  цепочка читается, `/admin/turns` и `/admin/turns/<id>` отдают 200 с реальными
  данными, а запрос с именем публичного туннеля получает 404.

- **IMG-ROMANCE-01 — разобран отказ по картинке «поцелуй двух людей».** По
  логу `data/server.log` отказ пришёл не от Google и не от кода хаба:
  инструмент `generate_image` вообще не вызывался, отказала сама чат-модель
  (Luna) в первом раунде, хотя тот же запрос в 18:06 сгенерировал картинку.
  Причина — формулировка в `prompts/cloud.md`; она уточнена: запрошенный
  поцелуй, объятие или парная поза записанных людей это обычная романтика (её и
  раньше разрешал тот же файл), подменять её на «friendly hug» запрещено, а
  сексуализированные или раздетые правки реальных людей по-прежнему
  отклоняются. **Чего нет:** живого прогона нового промпта на облачной модели
  (нужен перезапуск хаба).

- **F-705 — панель открыта и приведена в читаемый вид.** Хаб перезапущен
  (18:39), `/admin` отвечает 200, `/admin/turns` работает; пароль —
  `ROWAN_ADMIN_PASSWORD` из `.env`. Список запросов теперь показывает, что
  именно спросили и что ответили, а страница запроса — карточки шагов с
  читаемой строкой (кто решил и с какой уверенностью, что вызвано и что
  вернулось, какой раунд модели, что ушло в неё, что услышала комната) и сырым
  JSON шага под «raw event»; список сам обновляется раз в 15 секунд и
  фильтруется строкой поиска. **Найдено и исправлено по пути:** прогон тестов
  писал события ходов в живую `data/hub.db` (191 тест добавлял 175 строк) —
  путь базы стал переопределяемым (`ROWAN_HUB_DB`), `tests/conftest.py` уводит
  его в sandbox, и это закреплено тестом; накопленный мусор из живой базы
  убран (удалено 430 + 214 строк тестовых фикстур, таблица начинается заново).
  Проверено: `pytest tests/test_web_admin.py tests/test_turn_trace.py
  tests/test_audit.py tests/test_telegram_control.py -q` → 129 passed;
  `ruff check .` → All checks passed; живой `/admin/turns` → 200 и чистый пустой
  список. **Чего нет:** живого запроса в новой панели — первая настоящая цепочка
  появится после следующей реплики в комнате или сообщения в Telegram.

- **CU-LIMIT-01 (F-512) — потолок в 15 шагов у computer-use снят (23.09.2026).**
  Владелец: «the step limit of 15 steps is reached… убери эту фигню. никаких
  лимитов: все что его попросили — делает». Что изменено: `MAX_STEPS` больше не
  потолок, добавлены `DEFAULT_MAX_STEPS = 0` и `UNLIMITED_STEPS = 0`
  (`common/computer_use.py`), `0` значит «лимита нет», положительное число —
  предел владельца; `ComputerUsePolicy.unlimited`; в `common/config.py`
  `max_steps: int = Field(default=0, ge=0)` вместо `ge=1, le=15`; allow-list
  `["*"]` = любое приложение (`ANY_APP`); `hub/computer_use.py::remaining`
  отдаёт `-1` при снятом лимите, лог пишет `no limit`; ответ на шаг в
  `hub/app.py` — «step N done; no step limit»; описание инструмента
  `computer_use` в `hub/tools.py` больше не обещает «At most 15 steps»;
  `config.example.yaml` и `config.openai.yaml` — `max_steps: 0`,
  `max_tool_rounds: 40`. Живое включение: `computer_use.enabled: true`,
  `allowed_apps: ["*"]` в `config.openai.yaml` (локальный, в git не попадает).
  Проверено: `pytest tests/test_computer_use.py tests/test_computer_use_audit.py
  -q` → 37 passed; `ruff check .` → All checks passed; `mypy common` → Success.
  **Чего нет:** живого прогона длиннее пятнадцати шагов — он появится при
  первой настоящей многошаговой просьбе после перезапуска хаба.
  Обоснование решения — `DECISIONS.md`, «Лимит шагов computer_use снят».

- **CU-LIMIT-02 — бюджет времени хода тоже снят (23.09.2026).** Потолок шагов
  был не единственной стеной: `server.timeouts.reply_ms` (по умолчанию 20 с)
  накрывает весь раунд модели вместе с раундами инструментов, поэтому длинная
  задача обрывалась и комната слышала «не успел придумать ответ». Что
  изменено: `0` в любом бюджете стадии значит «предела нет» —
  `hub/app.py::_stage_budget` возвращает `UNBOUNDED_BUDGET_S` (сутки) вместо
  прежних 0.05 с, `common/config.py::StageTimeouts` принимает `ge=0` для всех
  четырёх бюджетов, живой `config.openai.yaml` ставит
  `server.timeouts.reply_ms: 0`, `config.example.yaml` и `config.yaml`
  объясняют этот ноль в комментарии. Бюджеты `stt_ms`/`diarization_ms`/
  `speaker_ms` остались на 700 мс: они стоят подписи голоса, а не отказа в
  работе. Проверено: `pytest tests/test_stage_timeouts.py -q` → 17 passed;
  `Config.model_validate` на живом `config.openai.yaml` → `reply_ms 0`;
  `hub.app.UNBOUNDED_BUDGET_S` → 86400.0. **Чего нет:** живого длинного хода
  после перезапуска хаба — он и покажет, что модель доводит дело до конца.

- **CU-LIMIT-03 — счётчик раундов инструментов снят (23.09.2026).**
  `server.llm.max_tool_rounds` обрывал реплику после N раундов «модель →
  инструмент» и этим повторял ту же стену, что и лимит шагов (40 раундов ≈ 40
  шагов агента). Что изменено: `0` (или отрицательное) значит «счётчика нет» —
  `hub/llm.py::UNLIMITED_TOOL_ROUNDS` = 1000 как предохранитель от зацикленной
  модели, поле в `common/config.py` принимает `ge=0`, живой конфиг ставит
  `max_tool_rounds: 0`, `config.example.yaml` объясняет ноль в комментарии.
  Проверено: `pytest tests/test_multi_step.py tests/test_llm_vllm_provider.py
  tests/test_config.py tests/test_stage_timeouts.py -q` → 60 passed, среди них
  новый тест «двенадцать шагов подряд, ни один не потерян»;
  `Config.model_validate(config.openai.yaml)` → `max_tool_rounds 0`.
  **Чего нет:** живого длинного хода после перезапуска хаба.

- **CU-PAUSE-01 / CU-EARLY-01 — паузы и ранний старт действий (23.09.2026).**
  Владелец: «если я на секунду даже перестану говорить, то уже запись
  остановится» и «люди делают так, чтобы он открывал приложения уже во время
  разговора». Что изменено: `client/vad.py` получил удержание по смыслу
  (`hold_while` + `UNFINISHED_TAIL_WORDS`, `client.vad.unfinished_hold_ms`),
  окно тишины в живом конфиге 700 → 1500 мс, `max_utterance_s` 15 → 30 с;
  раунд по черновику получил свой исполнитель (`hub/early_start.py::
  may_run_early`) и может начать только обратимое «открыть приложение или
  страницу», когда объект уже прозвучал; отложенный вызов откатывает слова
  раннего раунда, а `Connection._early_actions_note` говорит настоящему ходу,
  что уже начато. Живой конфиг: `streaming_reply.early_start: true`.
  Проверено: `pytest tests/test_vad_machine.py tests/test_vad_endpoint.py -q`
  → 14 passed; `pytest tests/test_early_start.py -q` → 33 passed.

- **JE-01 — TypeSafe Jev: проверен живьём.** `python scripts/jev_probe.py` →
  ответ за 0.62 с («включи свет на кухне»: `act=True 0.94`,
  `family=devices 0.98`, `followup=False 0.89`);
  `python scripts/jev_usage_report.py` → 5061 решение Decider'а, все от
  `rules`, и ни одного события `understanding` в `turn_events` (901 запись).
  Бюджет чтения поднят 900 → 2000 мс: 900 мс не покрывали холодный старт, и
  чтение молча отбрасывалось. **Чего нет:** в Telegram-чате чтения Jev нет
  вовсе (`hub/telegram_chat.py` → `llm.reply_text`) — задача AU-19.

- **AU-03 — зрение и чтение Jev проверены живым прогоном (23.09.2026).**
  Массовый аудит семейства `vision` (`--workers 6`, живая модель и живой Jev):
  до правок `data/audit/runs/mass-07-vision-before.jsonl` — 81 из 116
  (69.8 %), 32 падения из 35 — сужение Jev; после — 
  `data/audit/runs/mass-12-vision-final.jsonl` — 114 из 114 разобранных
  (100 %), `AU-0609`/`AU-0610` печатаются `SKIP` (в стенде нет вложения
  Telegram, `inspect_photo` нечего смотреть; живой ход с фото — в комнате).
  Что подтверждено: Jev выбирает семейство `vision` для «кто в комнате»,
  «что на экране», «где мои ключи», «do you see my phone» и отдаёт модели
  весь набор зрения даже при `act=false`; первый вызов — `look_at_camera`,
  `look_at_screen` или `find_object` по смыслу просьбы. Что изменено:
  `hub/jev_decider.py` (вопрос `act` считает взгляд действием, вопрос о
  семействе разводит `vision`/`media`/`memory`/`people` примерами),
  `hub/app.py::_narrow_tools_for` (названное семейство важнее ответа «ничего
  не делать»), `hub/tools.py` (значения семейств и триггеры `find_object`),
  `prompts/system.md`, `scripts/gen-audit-scenarios.py`. Тесты:
  `tests/test_jev_understanding.py` (+3), `tests/audit/test_request_matrix.py`
  (+4); отдельно прогнаны `pytest tests/audit/test_request_matrix.py -q` →
  3193 passed и `pytest tests/test_jev_understanding.py -q` → 19 passed.
  **Чего нет:** живого фото из
  Telegram и комнатной камеры в песочнице — сценарии с вложением помечены
  `bench_skip`, а глаз комнаты проверяется на ПК (AU-12).

- **AU-08 — память проверена живым прогоном (23.09.2026).** Массовый аудит
  семейства `memory` (`--workers 6`, живая модель и живой Jev): до правок
  `data/audit/runs/au-08-memory-before.jsonl` — 70 из 90 (77.8 %), после —
  `data/audit/runs/au-08-memory-final-2.jsonl` и `…/au-08-memory-final-3.jsonl`
  по 90 из 90 (100 %). Что подтверждено живьём: `remember` вызывается на
  «remember that …», «don't forget that …», «keep in mind that …» (46
  сценариев, и каждый факт реально лёг в память комнаты —
  `hub.storage.Memory`, поле `room_facts` отчёта), `list_memory` отвечает
  списком сохранённых фактов этой комнаты, `forget_fact` находит
  объявленный предусловием факт и доходит до голосового подтверждения F-113
  (32 сценария), `recall_conversation` читает архив разговоров
  (`hub/conversations.py`) и возвращает настоящие строки: «what did we talk
  about yesterday» — две вчерашние записи, «about the exam» — одну-две.
  Что изменено: стенд больше не пишет в живые данные владельца (свои
  `memory.jsonl`/`conversations.sqlite3` на воркера и своя база хаба по
  `ROWAN_HUB_DB`, `scripts/live-eval.py`), корпус проверяет слово из самой
  реплики и объявляет предусловия (`scripts/gen-audit-scenarios.py`),
  `recall_conversation` объявляет `person`/`since`/`until`/`limit`, которые
  хаб уже читал, а промпт и описания разводят сохранённый факт и разговор
  (`hub/tools.py`, `prompts/system.md`). Тесты: `pytest tests -q` →
  9249 passed, 15 failed (все 15 — чужой `tests/test_guess_who.py`),
  11 skipped; `pytest tests/audit -q` → 3053 passed; `ruff check .` →
  All checks passed; `mypy common` → Success. Полный корпус после правок:
  `data/audit/runs/au-08-full-after.jsonl` — 1056 из 1072 разобранных
  (34 `SKIP`). **Чего нет:** живого голосового «да» в стенде (удаление по
  F-113 проверено офлайн, `tests/test_memory_admin.py`), а новые описания
  инструментов подхватит внешний перезапуск хаба — из песочницы старый
  процесс на порту 8770 не останавливается (`DECISIONS.md`, AUDIT-14j).

- **AU-11 — латентность хода измерена и объяснена (23.09.2026).** Полный
  отчёт с числами — `docs/AUDIT_LATENCY.md` (его печатает
  `scripts/audit-latency.py` по трём реальным источникам сразу: отчёты стенда
  `data/audit/runs/*.jsonl`, база живого хаба `data/hub.db` — таблица
  `turn_events`, и лог `data/server.log` — строки `First audio N ms after the
  end of speech`). **Что занимает время** (живой хаб, 13 ходов с настоящим
  раундом модели, медианы): STT 1397 мс, стадия `llm` 3808 мс — из неё
  раунды самой модели 3280 мс, — TTS 759 мс, весь ход 6367 мс; раунды модели
  медиана 2, у 9 из 13 ходов больше одного (первый раунд — вызов инструмента,
  второй — ответ). Второй раунд — это и есть плата за действие: 92.8 % ходов
  полного корпуса требуют больше одного раунда, первый вызов инструмента идёт
  в среднем через 1289 мс после старта модели. **Первый звук** (82 живых
  замера F-101 в логе): медиана 4703 мс, p95 19390 мс; позже бюджета ТЗ 15.1
  (1200 мс) — 80 из 82 (97.6 %), позже 2,5 с — 65 из 82 (79.3 %).
  **Правка:** `hub/jev_decider.py` открывал НОВЫЙ `httpx.AsyncClient` на каждый
  ход и закрывал его вместе с пулом соединений, поэтому каждый ход платил за
  TCP + TLS до облака; теперь соединение с Jev живёт между ходами
  (`JevDecider._pooled_client`, `aclose` на выключении хаба — `hub/app.py`).
  Замер на тех же 12 сценариях (6 воркеров, живые ключ и модель): чтение Jev
  1247 → 257 мс (медиана, −79 %), ход целиком «Jev + модель + инструменты»
  3.86 → 2.68 с, доля ходов дольше 4 с 41.7 % → 0 %; полный корпус
  (`data/audit/runs/au-11-full-after.jsonl`, 1077 ходов, 6 падений) — медиана
  2.56 с против 2.85 с в `au-08-full-after.jsonl` и 2.76 с в первом прогоне
  `mass-01.jsonl`, доля ходов дольше 4 с 13.6 % против 16.7 % и 29.4 %.
  **Живой ход по настоящей речи** (`scripts/measure_voice_latency.py`,
  `data/audit/voice-latency.json`, пять записей из `data/voices`): тёплые
  ходы — STT 407…472 мс, LLM 1466…3276 мс, весь ход 1.9…3.7 с; первый ход
  после старта — STT 7229 мс (прогрев CUDA). **Чего нет и почему это
  записано, а не нарисовано:** движок синтеза в этой песочнице не
  поднимается — сам хаб пишет `Could not load Silero TTS — replies will be
  text only` (phonemizer не может скопировать `espeak-ng.dll` во временный
  каталог: песочница запрещает запись в новые временные подпапки и вне
  рабочего каталога), поэтому TTS-стадия и сквозной «первый звук» после
  правки измерены не этим стендом, а трассами самого хаба (медиана 759 мс);
  `first_audio_ms` в отчёте пробы поэтому пуст, а не ноль (`DECISIONS.md`,
  AUDIT-17f). Бюджет 1.2 с при облачной модели на критическом пути
  недостижим — в конфиге `local_*` уровни пусты и каждый ход уходит в
  `deepseek-flash` (ТЗ 15.1 говорит про локальные модели; это задача стенда,
  записано в `DECISIONS.md`, AUDIT-17e).

## Массовый аудит запросов (23.09.2026, AU-01…AU-13)

Владелец просил аудит на несколько тысяч сценариев и правку всего, что
сломано, — ночной цикл шёл по `PROGRESS_AUDIT.md`. Итог с числами и
командами запуска — `docs/AUDIT_MASS.md`, латентность — `docs/AUDIT_LATENCY.md`,
обоснования решений — `DECISIONS.md` (AUDIT-01…AUDIT-19).

**Слои аудита.** (1) Корпус `scripts/gen-audit-scenarios.py` →
`data/audit/scenarios.jsonl`, сегодня 1112 реплик (browser 375, pc 168,
vision 116, media 115, memory 90, people 81, devices 64, noisy 34, notify 26,
multi 20, russian 15, injection 4, skills 2, chat 2). (2) Живой прогон
`scripts/live-eval.py --scenarios … --workers 6 --jsonl … --quiet` — та же
цепочка, что живая комната: Jev читает реплику и сужает набор инструментов,
отвечает настоящая модель (`deepseek-flash`), инструменты исполняются
по-настоящему. (3) Офлайн-матрица `pytest tests/audit` — те же реплики без
сети и модели (наличие инструмента, сужение семейства, нормализация
аргументов). (4) Комнатные ПК `scripts/update-room-pcs.ps1` +
`scripts/room-audit.ps1` — 57 настоящих действий на рабочем столе.
Сценарий, который стенд проверить не может, печатается `SKIP` с причиной,
а не «прошёл».

**Числа.** Первый прогон корпуса — **812 из 1106** (73.4 %). Финальный
прогон после правок — `data/audit/runs/au-13-full-final.jsonl` (2026-09-23
06:15, `--workers 6`, 1112 сценариев): **1077 разобрано, 35 `SKIP`,
1071 прошло = 99.4 %**. По семействам (первый прогон → финал): browser
329/375 → 360/360, devices 0/64 → 63/64, people 23/81 → 81/81, vision
71/116 → 114/114, notify 9/26 → 24/24, media 101/115 → 100/100, memory
80/90 → 90/90, pc 148/162 → 166/167, russian 8/15 → 15/15, multi 15/20 →
18/20, noisy 22/34 → 32/34. `SKIP` — 17 сценариев с вложением Telegram,
15 `fill` без `--actions`, 2 «эта картинка», 1 вставка в окно в фокусе.
**Остаток — шесть строк**: `AU-0462` (сбой провайдера: в обоих повторных
прогонах прошёл), `AU-0971` и `AU-0910` (разброс модели: 1 и 2 прогона из
2 прошли), `AU-0977` «picture of a dog» (0 из 3 — Jev не отдаёт
`generate_image`, задача AU-20) и пара `AU-0989`/`AU-0990` «сохрани фото и
поставь на обои» (0 из 3, теряется то одна, то другая половина — тот же
класс, что AUDIT-16d, задача AU-21).

**Латентность** (та же полная выборка, `docs/AUDIT_LATENCY.md`): ход стенда
без STT и TTS — медиана 2.54 с против 2.76 с в первом прогоне, ходов
дольше 4 с 12.7 % против 29.4 %; чтение Jev — 199 мс медиана (было
1247 мс до AU-11, когда соединение поднималось на каждый ход). Живой хаб:
STT 1397 мс, стадия `llm` 3808 мс (в ней раунды модели 3280), TTS 759 мс,
весь ход 6367 мс (медианы по 13 ходам с настоящим раундом модели);
«первый звук» по 82 живым замерам F-101 — 79.3 % позже 2.5 с, медиана
4703 мс. Бюджет ТЗ 15.1 (первый звук 1.2 с) при облачной модели на
критическом пути не берётся: локальные уровни в конфиге пусты (AUDIT-17e).

**Комнатные ПК (AU-13).** `scripts/update-room-pcs.ps1` — оба ПК обновлены
и клиент перезапущен, SHA-256 `client/camera.py` совпал, задачи `Running`,
окно `hidden`. `scripts/room-audit.ps1` — AntonDorm **57/57** (29.9 с и
33.7 с в двух прогонах), buro **56/57** (41.6 с и 42.9 с): оба раза упало
одно действие `RA-035` (`navigate www.google.com` сразу после
`youtube.com`, Firefox ещё дочитывает страницу). В AU-12 это же действие
упало один раз и прошло на повторе и было записано как гонка; повтор
означает устойчивый дефект `client/actions/browser_desktop.py::_navigate` —
отдельная задача AU-22 и `DECISIONS.md` AUDIT-19e. Звук комнат после
аудита возвращён (`scripts/room-audio.ps1 unmute` — «sound unmuted» на
обоих ПК).

## Jev в Telegram-чате (AU-19, 23.09.2026)

ТЗ 5.3–5.5 и запрос владельца 23.09: «в Telegram должны быть те же
возможности, что и у голосового ассистента». Проверено и сделано:

* **Чтение реплики.** Голосовой ход и запрос из Telegram-чата читает один и
  тот же код (`hub/turn_reading.py`): один batched-вопрос к Jev (`act`,
  `family`, `single`, `followup`) и сужение набора инструментов по
  уверенности (`server.decider.understanding.min_confidence`). Раньше чтение
  стояло только в `hub/app.py::Connection._understand_turn`, а чат в Telegram
  уходил в модель со всеми инструментами и **без чтения вовсе**: в
  `data/hub.db` из 81 Telegram-хода ноль событий `understanding`.
* **Доказательство.** `data/audit/runs/au-19-telegram.jsonl` — первый полный
  прогон корпуса (1112 реплик, `--workers 6`), который спрашивает хаб так,
  как это делает Telegram: настоящий `hub.telegram_control.TelegramController`
  с тем же `get_llm`, тем же читателем Jev и настоящими инструментами;
  **1036 из 1077** разобранных (96.2 %), те же 35 `SKIP`. В трессе каждого
  Telegram-хода теперь есть шаг `understanding` (`--telegram` в
  `scripts/live-eval.py`, `turn_events`).
* **Второй провайдер в цепочке решений.** `[rules, jev]` (ТЗ 5.2) до этой
  задачи означало «всегда отвечает `rules`»: правила закрывали вопрос своей
  догадкой (0.6–0.7), и Jev не спрашивали ни разу — в живой базе было 5087
  решений и **все** от `rules`. Теперь цепочка заканчивается только на
  ответе в полосе `auto_above` типа решения; ниже него следующий провайдер
  получает тот же вопрос, а побеждает более уверенный ответ
  (`hub/decider.py::_settled`). Живая проверка: `python scripts/jev_probe.py`
  — `addressed` отвечает `jev -> True (0.83)` за 520 мс там, где правило
  говорило 0.6; `route` остался локальным (0–2 мс), потому что это факт о
  регулярках роутера, а не о смысле реплики (`config.openai.yaml`,
  `DECISIONS.md` AUDIT-20).
* **Цена.** Второе мнение стоит сетевого вызова только на неуверенных
  ответах; у типов с уверенным правилом облако по-прежнему не зовётся
  (`tests/test_jev_decider.py`, `tests/test_decider.py`).

Проверки: `pytest tests -q` → 9334 passed, 15 failed (все 15 — чужой
`tests/test_guess_who.py`, в зачёт не идёт), 11 skipped; `ruff check .` →
All checks passed; `mypy common` → Success. Хаб перезапущен
`scripts/run-openai-server.ps1` (`/health` → `status=ok`, `llm_model=
deepseek-flash`, `telegram=true`).

## Аудит AU-23: Telegram-путь целиком, с ожиданиями для чата (23.09.2026)

Первый прогон Telegram-пути (AU-19) показал 1036 из 1077 — и часть падений
была не поломкой, а вердиктом, написанным для комнаты: в чате «покажи камеру»
присылает фото в разговор (`telegram_send kind=image`), а не ставит картинку
на экран комнаты; запись лица из чата идёт через камеру комнаты.

* **Верный ход в чате назван в корпусе.** У сценария есть признаки
  `telegram_expect_tools` / `telegram_expect_any`
  (`scripts/gen-audit-scenarios.py`), которые читает только `--telegram`
  (`scripts/live-eval.py::telegram_scenario`); голосовые ожидания не
  меняются, прогоны сравнимы сценарий за сценарием.
* **Запись лица/голоса из чата выполняется, а не отклоняется** (ТЗ F-210):
  промпт Telegram-хода прямо говорит, что источник — камера ОБЩЕЙ комнаты и
  что имя берётся слово в слово («my roommate» — это имя).
* **Шаг хаба «кто в комнате» виден в цепочке хода.** Telegram-ход отвечает на
  этот вопрос свежим кадром до модели, и раньше этот шаг не попадал в
  `turn_events`: панель владельца показывала ответ про комнату без взгляда на
  камеру.
* **Стенд не пишет в чат владельца ночью.** Последний шаг (API Telegram)
  подменён транспортом-двойником `BenchTelegram`, квитанции о доставке
  попадают в строку отчёта (`deliveries`).

Числа: `data/audit/runs/au-23-before.jsonl` — 1032 из 1077 (95.8 %),
`au-23-after.jsonl` — 1060, `au-23-final.jsonl` — **1059 (98.3 %)**; все
сценарии задачи (`AU-0707`, `AU-0995`, `AU-0996`, `AU-0802`, `AU-0819`) и
найденные по ходу `AU-0547`/`AU-1011` проходят. Остаток разобран повторными
прогонами (`au-23-residual-1/2.jsonl` — 14/18 и 15/18): устойчивых два
(`AU-0251`, `AU-0922` — задачи AU-25/AU-26), остальное разброс модели.

Проверки: `pytest tests -q` → 9364 passed, 15 failed (все 15 — чужой
`tests/test_guess_who.py`; проверено по junit `tmp/au-23-junit.xml`), 11
skipped; `ruff check .` → All checks passed; `mypy common` → Success;
`python -c "import hub.app"` → ok. Хаб перезапущен
`scripts/run-openai-server.ps1` (`/health` → `status=ok`,
`llm_model=deepseek-flash`, `telegram=true`, `outbound.clients=1`).
`client/` и `common/` не правились — обновление комнатных ПК не требуется.
Решения — `DECISIONS.md`, AUDIT-25…25d; числа — `docs/AUDIT_MASS.md`, «AU-23».

## Аудит AU-24: память Telegram-аккаунта (23.09.2026)

«Что ты помнишь обо мне» в Telegram-чате отвечало «ничего»: вопрос не уходил в
`list_memory`/`recall_conversation`. Теперь вопрос доходит до инструментов, а
по продукту решено так (ТЗ F-415/F-701): личка ВЛАДЕЛЬЦА дома читает его
собственные заметки комнаты — профиль назван в конфигурации дома
(`homes[].owner_person_id`, ТЗ 14) и совпадает с тем, под которым владельца
узнаёт ход в комнате (`hub/telegram_control.py::owner_memory_profile`);
посторонний аккаунт и группа остаются на своём пространстве имён
(`telegram:<чат>:<пользователь>`), поэтому личные заметки комнаты в чужой чат
не утекают. Пока дом владельца не называет, личка остаётся со своим
пространством имён и отвечает из него честно.

Числа: `--telegram --family memory --family russian --workers 6`
(`data/audit/runs/au-24-memory.jsonl`) — **104 из 105**; `AU-1010` проходит
(`list_memory` вернул 10 фактов этого чата), падает только `AU-1007` «включи
свет» (известный разброс). Решение проверено живьём отдельным прогоном с
временным конфигом (`data/audit/runs/au-24-owner-profile.jsonl`): личка
владельца прочитала его собственный факт, чего без поля не видит.

Проверки: `pytest tests -q` → 9366 passed, 15 failed (все 15 — чужой
`tests/test_guess_who.py`; проверено по junit `tmp/au-24-junit.xml`), 11
skipped; `ruff check .` → All checks passed; `mypy common` → Success.
`client/` и `common/` не правились — обновление комнатных ПК не требуется.
Решения — `DECISIONS.md`, AUDIT-26/26b; числа — `docs/AUDIT_MASS.md`, «AU-24».

- **API-04 — месячного предела расходов нет по умолчанию (23.09.2026).**
  Владелец: «monthly api allowance убери нахер у меня чатбот не работает». В логе
  живого хаба за 14:31 стоит `monthly allowance $18.00` и сразу за ним
  `Cloud turn stopped: Monthly API allowance reached; local commands remain
  available.` — то есть каждый облачный ход отклонялся, и ассистент молчал.
  Что изменено: умолчание `server.llm.monthly_budget_usd` в `common/config.py`
  стало `0.0` («предела нет, расход считается и виден»), оба запасных
  `getattr(..., 18.0)` в `hub/app.py` тоже `0.0`, `config.example.yaml` — `0`,
  а применение потолка в живом хабе теперь пишет WARNING
  (`hub/admin_settings.py`). Сам потолок как функция остался: число, введённое
  владельцем в панели, работает — но его видно в логе с момента применения.
  Проверено: `python scripts/llm_probe.py` → «предел расходов: нет (0)»,
  живой ответ 863 мс («Yes, I'm working.»); `pytest tests/test_config.py
  tests/test_admin_settings.py tests/test_api_budget.py
  tests/test_image_generation.py tests/test_metrics.py tests/test_digest.py -q`
  → 109 passed; `pytest tests -q` → 9371 passed, 11 skipped, 15 failed (все
  пятнадцать — чужой незаконченный `tests/test_guess_who.py`).

- **API-05 — потолок расходов нигде не отказывает (23.09.2026).** Владелец:
  «api allowance reached убери это, я не хочу». Убрано насовсем: `ApiBudget.reserve`
  больше не бросает `BudgetExceeded` (превышение — одна строка WARNING на пару
  «месяц, сумма», расход считается дальше), `hub/app.py::cloud_budget_allows`
  отвечает `True` всегда, ветка «The monthly API budget is exhausted.» из
  `hub/telegram_chat.py` удалена, умолчание `ApiBudget.monthly_usd` — `0`,
  стартовая строка лога говорит «(reporting only, no request is refused)».
  Проверено: `pytest tests/test_api_budget.py tests/test_openai_responses.py
  tests/test_image_generation.py tests/test_digest.py tests/test_models_health.py
  tests/test_browser_recovery.py -q` → 116 passed; `python scripts/llm_probe.py`
  → предел 0, живой ответ 863 мс.

- **API-06 — потолка нет и в живом хабе: сохранённое значение сброшено
  (23.09.2026).** Владелец: «api allowance reached убери это, я не хочу».
  Проверка живого хаба показала вторую половину причины: 15:28:59 владелец сам
  поставил `server.llm.monthly_budget_usd = 100.0` в панели Telegram, значение
  сохранилось в `data/telegram/admin.sqlite3`
  (`telegram_settings.config:server.llm.monthly_budget_usd`) и применялось
  при каждом старте (`hub/admin_settings.py::restore_overrides`), а процесс,
  запущенный в 14:43, держал его в памяти. Что сделано: сохранённое значение
  сброшено на `0`, хаб перезапущен на закоммиченном коде — в логе
  `LLM: deepseek-flash via OpenAI Responses; monthly allowance none
  (reporting only, no request is refused)`, после 15:39 ни одной строки
  `allowance reached`; последняя пользовательская фраза со словом allowance
  («This image is too large for the configured API allowance.») заменена на
  «This picture is too large to send to the cloud. Send a smaller or cropped
  photo.» (`hub/openai_responses.py`). Проверено: `python scripts/llm_probe.py`
  → «предел расходов: нет (0)», живой ответ 843 мс («Yes, I'm working.»);
  `/health` → 200, `telegram: true`, `telegram_error: null`, `llm: true`;
  `pytest tests/test_openai_responses.py tests/test_api_budget.py
  tests/test_vision_cloud.py -q` → 60 passed; `ruff check hub/openai_responses.py`
  → All checks passed. `client/` и `common/` не правились — обновление
  комнатных ПК не требуется. Решения — `DECISIONS.md`, API-06.
