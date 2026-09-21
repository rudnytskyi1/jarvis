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
| Роутер моделей: уровни `local_fast`/`local_strong`/`cloud_cheap`/`cloud_strong`, D-10 (F-401) | готово в коде, выключено конфигом (`models.enabled: false`) | `hub/model_router.py`, `tests/test_model_router.py` |
| Перелив при перегрузке очереди (F-403) | готово в коде, включается `models.routing.cloud_fallback` | `hub/model_router.py`, `hub/gpu_queue.py::wait_estimate` |
| `utterance_id` (ULID) на клиенте, сквозь сообщения, логи, БД и метрики (4.5) | готово | `common/ids.py`, `hub/utterances.py`, `hub/app.py::_store_dialog_turns`/`/health.utterances`, `tests/test_ulid.py`, `tests/test_utterance_id.py` — 30 passed |
| `event_id` на каждое фоновое событие камеры (4.5) | готово: кадр/скриншот/клип несут один id через сообщения, логи, архив и метрики | `hub/camera_events.py`, `hub/app.py` (запросы кадров), `hub/camera_clip_receiver.py`, `client/camera.py`, `client/camera_clips.py`, `tests/test_camera_events.py` — 13 passed |
| Таймауты по стадиям с деградацией, а не молчание (4.5, 15.1) | готово | `common/config.py::StageTimeouts` (`server.timeouts`), `hub/app.py::_stage_budget`/`_degrade`/`_plain_transcript`, `tests/test_stage_timeouts.py` — 10 passed |
| Порог перелива F-403 внесён в конфиг по замеру (F-403, 15.1) | готово с оговоркой: порог = бюджет 15.1 (0.55 с), замер на реальной очереди с оценочным временем реплики 2.5 с (живого стенда с тремя комнатами в песочнице нет) | `scripts/measure_overflow.py`, `models.routing.overflow_wait_s: 0.55` в `config.yaml`, `tests/test_measure_overflow.py` — 10 passed, `DECISIONS.md` (P1-24) |
| `LocalLLMDecider`: structured output, logprobs первого токена для yes/no (5.2) | готово в коде, включается через `server.decider.order` | `hub/decider_local.py`, `hub/llm.py::first_token_probabilities`, `tests/test_decider_local.py` — 12 passed |
| Порядок провайдеров по типу решения и таймаут 400 мс задаются конфигом (5.2) | готово | `common/config.py::DeciderConfig` (`server.decider`), `hub/app.py::_decision_settings`/`_decision_providers`, `tests/test_decider_config.py` — 12 passed |
| Точки D-02, D-03, D-04, D-05, D-07, D-09, D-11 переведены на Decider (5.3) | готово; D-02/D-11 решаются и пишутся хабом, но исполняются клиентом (wake-гейт и окно follow-up, F-113 — фаза 2, см. `DECISIONS.md`) | `hub/decision_points.py`, `hub/decider.py::HEURISTIC_TYPES`, `hub/app.py::_decide`/`_permission_check`, `tests/test_decision_points.py` — 25 passed |
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
| Сцена «кино» по голосовой команде end-to-end (F-506) | готово: реплика идёт по настоящему пайплайну (`Connection._handle_utterance`) до адаптеров и кадра `say`; «Rowan, кино» / «Rowan AI, cinema» запускают предустановку (свет off, лента `#221100`, ТВ on) и отвечают «Cinema mode.» **без обращения к модели**; нехватка устройства не выдумывает успех — шаг попадает в ответ словами | `tests/test_scene_e2e.py` — 3 passed |

**Фаза 2 — речь, идентичность, присутствие (начата):**

| Пункт | Статус | Доказательство |
|---|---|---|
| F-101 Стриминговый ответ: первое предложение звучит сразу | частично готово: первое предложение синтезируется отдельной группой (короткий ответ больше не ждёт синтеза целиком), бюджет «конец речи → первый звук» (1,2 с из 15.1) измеряется на настоящем пайплайне и пишется в лог хода; старт по промежуточному транскрипту — задача P2-41 | `hub/streaming_reply.py`, `hub/app.py` (`_stream_tts_unlocked`, `_report_first_audio`), `common/config.py::StreamingReplyConfig`, `tests/test_streaming_reply.py` — 21 passed |
| F-104 Динамические hotwords | готово: имена людей, устройства и их алиасы, сцены и их алиасы и приложения роутера собираются из БД (не из одной формы админки), слова владельца идут первыми и не вытесняются ограничением, повторы по регистру схлопываются; движок получает одну строку, обновление — на старте хаба и после добавления/удаления устройства, удаления сцены и переименования человека | `hub/hotwords.py`, `hub/app.py::_refresh_hotwords`, `hub/admin_backend.py::_sync_hotwords`, `tests/test_hotwords.py` — 9 passed |

## Замеры латентности (раздел 15.1)

| Что мерили | Команда | Результат |
|---|---|---|
| Порог перелива F-403 при трёх активных домах | `python scripts/measure_overflow.py --rate 4 --service-s 2.5 --per-home 20 --time-scale 10` | ожидание класса 0: медиана 0,00 с, p95 0,79–0,93 с, худшее 2,6 с; порог 0,55 с записан в `config.yaml` (P1-24) |
| Задержка, которую одна комната создаёт другой | `python scripts/measure_cross_room_delay.py --repeats 3` | планирование хаба: −0,002 с (быстрая комната 0,218 с одна, 0,215 с рядом с медленной, которая отвечает 2,0 с); очередь GPU: медиана 0, p95 0, max 2,59 с без F-403 и max 0,0 с при включённом F-403 (порог 0,55 с, 2 из 36 реплик ушли в облако). Порог приёмки 1,5 с выдержан |

Оба замера идут по настоящим реализациям (очередь GPU, fair share, оценка
ожидания, планирование комнат), но время одной реплики на GPU — параметр:
стенда с RTX 5090 и тремя комнатами в песочнице нет, см. `DECISIONS.md`
(P1-24, P1-45). На стенде замер повторяется с фактическим временем реплики.

**Фаза 2 (речь, идентичность, присутствие) — начата.** Задачи P2-01…P2-40 в
`PROGRESS.md`, план — в `PLAN.md`. Ниже — карта функций ТЗ, чтобы видеть объём.

| Блок | Функции | Статус |
|---|---|---|
| Речь и диалог | F-101 … F-120 | фаза 2: F-101–F-106, F-113, F-114, F-117 (задачи P2-01…P2-10); остальное — фазы 3–5 |
| Идентичность людей | F-201 … F-216 | фаза 2: F-201–F-208, F-210–F-213 (P2-11…P2-26); F-214–F-216 — P1 в той же фазе |
| Зрение и присутствие | F-301 … F-314 | фаза 2: F-301–F-304 (P2-27…P2-30), F-312 (P2-31); F-305–F-309, F-311 — фазы 3–4 |
| LLM, агент, память, интеграции | F-401 … F-424 | частично в фазе 1: F-401–F-406, F-409–F-411 (Decider, роутер, скиллы, guards); остальное — фаза 3 |
| Управление комнатой и ПК | F-501 … F-515 | частично в фазе 1: F-501–F-504, F-506; остальное — фазы 3–5 |
| Социальные функции и комнаты | F-601 … F-612 | фаза 4 |
| Telegram, админка, HUD, мобильный | F-701 … F-713 | частично в фазе 1: F-705, F-706; фаза 2: F-701, F-702, F-708; остальное — фазы 3–4 |
| Решения (Decider) | D-01 … D-12 | D-01…D-11 переведены в фазе 1; D-06 и D-12 придут с идентичностью и уточнениями (фаза 2–3) |

## Что уже сделано из не-ТЗ

- Удобная Telegram-клавиатура и диапазоны уведомлений ещё **не** сделаны —
  это F-701/F-702 и они попадают в Фазы 1–2.

## Открытые вопросы (раздел 17 ТЗ)

Реализуются вариантом «по умолчанию», пока заказчик не ответит:
overlay-сеть Tailscale, клиенты на Windows, хаб на Linux, LLM Qwen 35B-A3B под
vLLM, бюджет $18/мес общий на хаб, до 4 комнат, Jev за флагом, Home Assistant
в Фазе 5.
