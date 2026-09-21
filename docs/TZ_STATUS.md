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
| Регрессионный набор реплик | частично: 38 текстовых реплик из 100 записанных | `tests/regress/fixtures.jsonl`, `tests/regress/README.md` |
| Каркас скиллов | частично: манифест + генератор, без реестра | `hub/skills_runtime.py`, `hub/skill_scaffold.py` |

**Вне ТЗ, по прямой просьбе заказчика (сделано):**

| Изменение | Статус | Доказательство |
|---|---|---|
| Кулдаун уведомлений: минимум 10 с → 1 с; диапазоны вынесены в один источник | готово | `hub/presence_alerts.py` (`RULE_RANGES`), тесты в `tests/test_presence_alerts.py` |
| Пресеты числовых полей уведомлений в Telegram (без ввода текста) | готово | `hub/telegram_admin.py` (`_ALERT_PRESETS`), `tests/test_telegram_admin.py` |
| Компактная клавиатура `/tools`: по две кнопки в ряд, кулдаун виден в списке правил | готово | `hub/telegram_admin.py`, `tests/test_telegram_admin.py` |

**Фаза 1 — ядро (начата):**

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
| Устройства и сцены (F-501–F-504, F-506) | не начато | — |
| Реестр скиллов, изоляция по дому, таймаут 20 с (F-405, F-406) | готово | `hub/skills_registry.py`, `tests/test_skills_registry.py` |
| Аудит (F-706) | не начато | таблица `audit` в схеме |

**Фазы 2–5 — не начаты.** Ниже — карта функций ТЗ, чтобы видеть объём.

| Блок | Функции | Статус |
|---|---|---|
| Речь и диалог | F-101 … F-120 | не начато |
| Идентичность людей | F-201 … F-216 | не начато |
| Зрение и присутствие | F-301 … F-314 | не начато |
| LLM, агент, память, интеграции | F-401 … F-424 | не начато |
| Управление комнатой и ПК | F-501 … F-515 | не начато |
| Социальные функции и комнаты | F-601 … F-612 | не начато |
| Telegram, админка, HUD, мобильный | F-701 … F-713 | не начато |
| Решения (Decider) | D-01 … D-12 | не начато (сейчас — регулярки и пороги) |

## Что уже сделано из не-ТЗ

- Удобная Telegram-клавиатура и диапазоны уведомлений ещё **не** сделаны —
  это F-701/F-702 и они попадают в Фазы 1–2.

## Открытые вопросы (раздел 17 ТЗ)

Реализуются вариантом «по умолчанию», пока заказчик не ответит:
overlay-сеть Tailscale, клиенты на Windows, хаб на Linux, LLM Qwen 35B-A3B под
vLLM, бюджет $18/мес общий на хаб, до 4 комнат, Jev за флагом, Home Assistant
в Фазе 5.
