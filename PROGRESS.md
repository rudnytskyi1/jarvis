# PROGRESS — задачи по фазам ТЗ

Порядок обязателен: задача берётся сверху вниз, закрывается, коммитится с её
ID, и только потом начинается следующая (см. `AGENTS.md`, «Режим цикла»).

Статус `[x]` ставится только после реально прогнанной проверки, и рядом
записывается, что именно запускалось.

## Фаза 0 — подготовка (закрыта)

| Пункт | Доказательство |
|---|---|
| Структура `hub/`, `client/`, `common/`, `skills/`, `migrations/` | `server/` → `hub/`, импорты обновлены, `pytest tests` зелёный |
| Pydantic-схемы протокола v2 + совместимость v1 | `common/protocol.py`, `tests/test_protocol_v2.py` |
| Конфиги хаба и клиента, шаблоны, `extra='forbid'` | `config.hub.example.yaml`, `config.client.example.yaml`, `tests/test_hub_config.py`, `tests/test_client_config_v2.py` |
| SQLite + миграции | `migrations/0001_init.py`, `hub/migrations_runner.py`, `tests/test_migrations.py` |
| CI: ruff, mypy (strict на `common`), pytest | `.github/workflows/ci.yml`, `ruff check .` и `mypy common` локально чисто |
| Регрессионный набор реплик | `tests/regress/fixtures.jsonl` — 38 реплик из 100 записанных |
| Каркас скиллов (манифест + генератор) | `hub/skills_runtime.py`, `hub/skill_scaffold.py`, `make skill` |

## Фаза 1 — многодомность и ядро (текущая)

Содержание фазы по ТЗ (раздел 16): разделы 4.2–4.7 и 4.9; SQLite и миграция
`people.json`/`memory`/`dialogs`; auth клиентов; GPU queue + `utterance_id`;
vLLM-провайдер (F-402); Decider с Rules и LocalLLM (раздел 5); скиллы
(F-405, F-406); устройства и сцены (F-501–F-504, F-506); админка минимум
(F-705); аудит (F-706).

Критерии приёмки фазы: две комнаты на одном хабе работают одновременно и
изолированно; реплика в одной не задерживает другую дольше 1,5 с; токены
работают; сцена «кино» по голосовой команде.

### Хранилище и сущности

- [x] **P1-01 (4.6)** Схема БД хаба: `migrations/0001_init.py` (26 таблиц, FK, WAL), `hub/migrations_runner.py`, таблица `schema_version`. Проверено: `pytest tests/test_migrations.py -q` → 4 passed.
- [x] **P1-02 (4.6)** Миграция `data/people.json` → `persons` + `voice_embeddings`/`face_embeddings` скриптом, старый файл остаётся бэкапом. Проверено: `pytest tests/test_legacy_migrate.py -q` → 4 passed; `make test` → 2513 passed, 2 skipped.
- [x] **P1-03 (4.6)** Миграция `data/memory.jsonl` → `memories` (тип, область, TTL). Проверено: `pytest tests/test_legacy_migrate.py -q` → 4 passed; `make test` → 2513 passed, 2 skipped.
- [x] **P1-04 (4.6)** Миграция `data/dialogs/*.jsonl` → `dialog_turns`. Проверено: `pytest tests/test_legacy_migrate.py -q` → 4 passed; `make test` → 2513 passed, 2 skipped.
- [x] **P1-05 (4.6)** Медиа по правилу `data/homes/<home_id>/media/YYYY-MM-DD/` + TTL и удаление из `media`. Проверено: `pytest tests/test_media.py tests/test_config.py tests/test_hub_config.py -q` → 26 passed; `make test` → 2519 passed, 2 skipped.
- [x] **P1-06 (4.6)** sqlite-vec: сборка/подключение и таблицы векторов (фолбэк — LanceDB, решение записать в `DECISIONS.md`). Проверено: `pytest tests/test_vectors.py -q` → 24 passed; `make test` → 2543 passed, 2 skipped; `ruff check .` и `mypy common` чисто. `vec0.dll` 0.1.9 подключён из `hub/vendor/`, виртуальные таблицы `vec_voice_embeddings`, `vec_face_embeddings`, `vec_body_embeddings`, `vec_memories`, `vec_objects_index` создаются на старте хаба.
- [x] **P1-07 (4.2)** Сущности `home`: таблица `homes`, сид из `homes:` конфига, `config_rev`. Проверено: `pytest tests/test_homes.py -q` → 4 passed.
- [x] **P1-08 (4.2, 4.7)** Pydantic-конфиги хаба и клиента (`HomeConfig`, `GpuQueueConfig`, `ModelsConfig`, `homes:`, `models:`), шаблоны примеров. Проверено: `pytest tests/test_config.py tests/test_hub_config.py tests/test_client_config_v2.py -q` → 20 passed.
- [x] **P1-09 (4.7)** Горячая перезагрузка настроек дома без рестарта + сообщение `config_update` клиенту. Проверено: `pytest tests/test_config_reload.py -q` → 11 passed; `make test` → 2554 passed, 2 skipped; `ruff check .` и `mypy common` чисто. `hub/config_reload.py` + `hub/app.py::reload_room_configs`/`broadcast_config_update`, клиент хранит `room_config_rev`/`room_config`.

### Сеть, auth, gateway

- [x] **P1-10 (4.3)** Токены клиентов: выдача, ротация, отзыв, хранение только хеша. Проверено: `pytest tests/test_auth.py -q` → 10 passed.
- [x] **P1-11 (4.3)** `hello`: сессия привязана к `home_id`, `home_id` из тела сообщения не принимается, неверный токен → закрытие 4401. Проверено: `pytest tests/test_hello_auth.py -q` → 3 passed.
- [x] **P1-12 (4.3)** Rate-limit на клиента (реплики в минуту, кадры в секунду) из конфига. Проверено: `pytest tests/test_gateway.py -q` → 10 passed.
- [x] **P1-13 (4.4)** Gateway: реестр комнат и клиентов, квоты, `rejection()`. Проверено: `pytest tests/test_gateway.py -q` → 10 passed.
- [x] **P1-14 (4.4)** Backpressure: медленный клиент не задерживает рассылку остальным (буфер на сессию + метрика отброшенного). Проверено: `pytest tests/test_outbound.py -q` → 7 passed; `make test` → 2561 passed, 2 skipped; `ruff check .` и `mypy common` чисто. `hub/outbound.py` (буфер на сессию, фон отбрасывается первым, реплики/PCM — никогда), метрики в `/health` (`outbound`), конфиг `server.outbound.queue_capacity`.
- [x] **P1-15 (4.4, 15.1)** STT батчингом 2–4 реплики: `faster-whisper` принимает батч, а не по одной. Проверено: `pytest tests/test_stt_batching.py -q` → 8 passed; `make test` → 2569 passed, 2 skipped; `ruff check .` и `mypy common` чисто. `hub/stt.py::SttBatcher` (окно `server.stt.batch_window_ms`, размер `server.stt.batch_size` 1–4) + `SttEngine.transcribe_batch` через `BatchedInferencePipeline`, один слот GPU-очереди на батч.

### Очередь GPU и трассировка

- [x] **P1-16 (4.5)** `GpuQueue`: четыре класса приоритета, fair share 50 % по домам, таймауты, отказ при переполнении, оценка ожидания класса. Проверено: `pytest tests/test_gpu_queue.py -q` → 15 passed.
- [x] **P1-17 (4.5)** Очередь подключена к пайплайну: STT (в т.ч. диаризация и wake-префикс), LLM-раунд и verify, идентификация голоса, лица (presence/enrollment), vision и SAM3 вместе с их VRAM-свопом. Проверено: `pytest tests/test_gpu_queue_wiring.py -q` → 15 passed.
- [ ] **P1-18 (4.5)** `utterance_id` (ULID) генерируется клиентом и проходит через все сообщения, логи, записи БД и метрики.
- [ ] **P1-19 (4.5)** `event_id` для каждого фонового события камеры.
- [ ] **P1-20 (4.5, 15.1)** Таймауты по стадиям с деградацией (ответ без diarization/ReID), а не молчание.

### Модели и Decider

- [x] **P1-21 (F-402)** vLLM как отдельный провайдер на OpenAI-совместимом API: tool calling, `response_format`/`guided_json` со сменой режима при отказе, `extra_body`; Ollama сохранён. Проверено: `pytest tests/test_llm_vllm_provider.py -q` → 9 passed.
- [x] **P1-22 (F-401)** Уровни моделей `local_fast`, `local_strong`, `cloud_cheap`, `cloud_strong` + роутер по D-10 (сложность, изображение, загрузка очереди), `LevelPool` с ленивым клиентом на уровень. Проверено: `pytest tests/test_model_router.py -q` → 22 passed.
- [x] **P1-23 (F-403)** Перелив при перегрузке: оценка ожидания класса 0 из очереди → `cloud_cheap` только при `cloud_fallback: true` и разрешении бюджета, с записью в лог. Проверено: `pytest tests/test_model_router.py tests/test_gpu_queue.py -q` → 37 passed.
- [ ] **P1-24 (F-403)** Проверить порог перелива на живых замерах трёх активных домов (таблица 15.1) и внести фактический порог в конфиг.
- [ ] **P1-25 (5.2)** `LocalLLMDecider`: локальная модель через vLLM со structured output (JSON Schema), `logprobs` первого токена для yes/no, если провайдер их отдаёт.
- [ ] **P1-26 (5.2)** Порядок провайдеров по типу решения и таймаут на провайдера (400 мс) задаются конфигом, а не кодом.
- [x] **P1-27 (5.3)** Таблица `decisions` заполняется из цепочки Decider: рекордер, политика уверенности `auto_above`/`ask_below`, хеш входа вместо самого текста. Проверено: `pytest tests/test_decision_log.py -q` → 7 passed.
- [ ] **P1-28 (5.3)** Точки D-02, D-03, D-04, D-05, D-07, D-09, D-11 переведены на Decider (адресат, галлюцинация, результат действия, sight/action guard, admin-права, инъекции, follow-up).
- [ ] **P1-29 (5.4)** Кэш решений по хешу входа на 60 с.
- [ ] **P1-30 (5.4)** Недельный отчёт калибровки (доля ошибок по типу и провайдеру) в админке.

### Скиллы

- [x] **P1-31 (F-405)** Реестр скиллов: `manifest.yaml` (имя, описание, область, роль, caps), Pydantic-схема аргументов, `async run(ctx, args) -> SkillResult`, автосборка tool-схем для LLM. Проверено: `pytest tests/test_skills_registry.py -q` → 6 passed.
- [x] **P1-32 (F-406)** Изоляция по дому: скиллы дома лежат в `data/homes/<home_id>/skills/` и видны только в нём; ошибка скилла не роняет хаб, таймаут 20 с. Проверено: `pytest tests/test_skills_registry.py -q` → 6 passed.
- [ ] **P1-33 (F-405)** Hot reload скиллов по изменению файла только в dev-режиме.

### Устройства и сцены

- [ ] **P1-34 (F-501)** Абстракция устройств: Pydantic-модель `Device(id, home_id, name, aliases, zone, kind, capabilities, adapter, adapter_config)` и capability-инструменты `device.set(device, capability, value)`.
- [ ] **P1-35 (F-502)** Адаптеры: обернуть BLE/Tuya/MagicHome и добавить MQTT, Home Assistant REST/WS, HDMI-CEC, Roku/Android TV, Spotify Connect.
- [ ] **P1-36 (F-503)** ESP32-переключатели: прошивка, MQTT-топики `home/<home_id>/switch/<n>/set|state`, брокер Mosquitto, калибровка углов через админку.
- [ ] **P1-37 (F-504)** Мастер добавления устройств: сканирование BLE и локальной сети (mDNS, Tuya discovery), выбор адаптера, тест «мигни», имя и алиасы сразу в hotwords.
- [ ] **P1-38 (F-506)** Сцены: модель `Scene(home_id, name, aliases, steps[])` (устройства, действия ПК, `say`, задержки) и предустановки «кино», «учёба», «сон», «гости», «ушёл».
- [ ] **P1-39 (F-506)** Сцена по голосовой команде и создание голосом («запомни как сцену „вечер“»).

### Админка и аудит

- [ ] **P1-40 (F-705)** Веб-админка минимум (FastAPI + Jinja), доступ только из overlay-сети, логин владельца: дома, клиенты (токены, статус, версия), люди и членство.
- [ ] **P1-41 (F-706)** Аудит: запись в таблицу `audit` (кто, что, когда, из какого дома, результат) для привилегированных действий, изменений настроек и удалений данных.

### Развёртывание

- [ ] **P1-42 (4.9)** Автозапуск сервисов (NSSM или Task Scheduler на Windows, systemd на Linux) + единые цели `make hub`, `make client`, `make test`, `make migrate`, `make skill`.
- [ ] **P1-43 (4.9)** OTA клиента: проверка git-тега из конфига хаба раз в час, `git fetch`/`checkout`, миграция конфига, рестарт и откат на предыдущий тег при падении в первые 60 с.

### Критерии приёмки фазы

- [ ] **P1-44** Две комнаты на одном хабе работают одновременно и изолированно (интеграционный тест на двух клиентах).
- [ ] **P1-45** Реплика одной комнаты не задерживает другую дольше 1,5 с (замер на живом стенде, запись в `docs/TZ_STATUS.md`).
- [ ] **P1-46** Сцена «кино» выполняется по голосовой команде end-to-end.
- [ ] **P1-47** Регрессионный набор реплик доведён до 100 записанных фраз.
- [ ] **P1-48** Итог фазы: обновлены `docs/TZ_STATUS.md` и `PLAN.md`, следующая фаза разбита в этот файл.
