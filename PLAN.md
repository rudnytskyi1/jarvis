# План фазы 1 — многодомность и ядро

Фаза уже частично выполнена (см. `PROGRESS.md`). Этот план описывает оставшуюся
часть: файлы, которые будут созданы/изменены, модели, таблицы, сообщения
протокола и риски. План выполняется сразу, без ожидания подтверждения
(раздел 1 ТЗ, пункт 2 отключён).

## Оставшиеся задачи

1. **P1-02..P1-04** — импорт унаследованных данных.
   - Новый модуль `hub/legacy_migrate.py`.
   - Подключение в `hub/main.py::_prepare_hub_database` после миграций и сида домов.
   - Тест `tests/test_legacy_migrate.py`.
2. **P1-05** — медиа и TTL.
   - Новый модуль `hub/media.py`; конфиг `server.media` (`media_ttl_days`,
     `clip_ttl_days`) в `common/config.py`.
   - Тест `tests/test_media.py`.
3. **P1-06** — sqlite-vec: сначала пробуем подключить расширение; если не
   соберётся — фолбэк LanceDB (решение в `DECISIONS.md`).
4. Далее строго по `PROGRESS.md`: горячая перезагрузка, backpressure, STT
   батчинг, `utterance_id`/`event_id`, таймауты с деградацией, LocalLLMDecider,
   провайдеры по конфигу, точки Decider, кэш, отчёт калибровки, hot reload
   скиллов, устройства, адаптеры, ESP32, мастер добавления, сцены, админка,
   аудит, автозапуск/OTA и приёмочные критерии.

## Файлы

| Действие | Файл |
|---|---|
| создать | `hub/legacy_migrate.py` |
| создать | `hub/media.py` |
| создать | `tests/test_legacy_migrate.py` |
| создать | `tests/test_media.py` |
| изменить | `hub/main.py` (запуск импорта legacy + медиа) |
| изменить | `common/config.py` (секция `server.media`) |
| изменить | `config.example.yaml` (шаблон `server.media`) |

## Модели

- `MediaConfig(media_ttl_days=3, clip_ttl_days=7)` — Pydantic v2, `extra='forbid'`.
- Legacy-импорт читает только уже описанные JSON/JSONL форматы; новые
  Pydantic-модели не вводятся, потому что унаследованные файлы не
  валидируются как строгие схемы (это данные, а не контракт).

## Таблицы

Используются существующие таблицы схемы `0001_init.py`: `persons`,
`voice_embeddings`, `face_embeddings`, `memberships`, `memories`,
`dialog_turns`, `media`, `homes`. Новые таблицы не создаются.

## Сообщения протокола

Для этих задач новых сообщений нет. Сообщение `config_update` появится в
задаче P1-09.

## Риски

- Импорт реальных `data/*` файлов в тестовые временные БД не должен происходить:
  поэтому legacy-импорт вынесен из нумерованных миграций в отдельный вызов.
- Идемпотентность важна: повторный запуск хаба не должен дублировать строки.
- `sqlite-vec` может не собраться на этой машине; тогда включается LanceDB за
  флагом/фолбэком.

## Выполнено (обновляется по ходу фазы 1)

- **P1-06** — sqlite-vec собран и подключён: `hub/vendor/vec0.dll` 0.1.9,
  `hub/vectors.py` создаёт `vec0`-таблицы на старте, фолбэк на LanceDB не нужен
  (см. `DECISIONS.md`). Тесты: `tests/test_vectors.py` (24).
- **P1-09** — горячая перезагрузка настроек дома без рестарта: `hub/config_reload.py`,
  `hub/app.py::reload_room_configs`/`broadcast_config_update`, клиент хранит
  `room_config_rev`; сообщение `config_update` (v2). Тесты: `tests/test_config_reload.py` (11).
- **P1-14** — backpressure: буфер на сессию `hub/outbound.py`, фон отбрасывается
  первым (реплики и PCM — никогда), метрики в `/health.outbound`.
  Тесты: `tests/test_outbound.py` (7).
- **P1-15** — STT батчинг 2–4 реплики: `hub/stt.py::SttBatcher` +
  `SttEngine.transcribe_batch` (BatchedInferencePipeline), один слот GPU-очереди
  на батч. Тесты: `tests/test_stt_batching.py` (8).

Следующая задача по порядку `PROGRESS.md`: **P1-18** (`utterance_id`).
