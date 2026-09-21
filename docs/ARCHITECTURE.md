# Архитектура Rowan (текущее состояние)

Документ описывает **то, что реально есть в коде сейчас**, после Фазы 0 ТЗ.
Целевую (многокомнатную) архитектуру см. в самом ТЗ; здесь — только факты.

## Из чего состоит репозиторий

| Каталог | Назначение |
|---|---|
| `hub/` | «Мозг» хаба: WebSocket-сервер, речь, идентичность, память, Telegram, админ-панель |
| `client/` | Клиент комнаты: микрофон, камера, HUD, действия на ПК и устройствах |
| `common/` | Общий контракт: протокол (v1 + v2), конфиги, модели тарифов |
| `migrations/` | Нумерованные миграции SQLite (`NNNN_name.py`) |
| `skills/` | Скиллы: `manifest.yaml` + `skill.py` (каркас Фазы 0) |
| `tests/` | pytest: юнит-тесты, регрессионные реплики (`tests/regress`) |
| `scripts/` | Установка, запуск, диагностика, публикация клиента |
| `data/` | Локальные данные: `hub.db`, `people.json`, память, диалоги, медиа |

В Фазе 0 пакет `server/` переименован в `hub/`; код, конфиги и скрипты обновлены,
поведение системы не изменилось (`pytest tests` — 2284+ теста зелёные).

## Как проходит голосовая реплика

1. Клиент слушает wake word (Vosk, grammar-режим) и пишет реплику через
   webrtcvad, включая `pre_roll_ms` аудио до срабатывания.
2. PCM (16 кГц, моно, s16le) уходит на хаб кадрами по 30 мс.
3. Хаб: `hub/stt.py` (faster-whisper) → `hub/diarization.py` (pyannote) →
   `hub/speaker.py` (ECAPA) → выбор адресата.
4. Быстрые команды решает `hub/local_commands.py` без модели; остальное —
   `hub/llm.py` (Ollama `ollama_native`, `openai`, `openai_responses`).
5. Инструменты выполняет клиент, реальный `action_result` возвращается модели.
6. `say` + TTS (Silero, опционально Kokoro) озвучиваются в комнате.
7. Между репликами хаб продолжает читать сокет: приветствия лиц, кадры камеры,
   статусы HUD.

## Транспорт

WebSocket (FastAPI/uvicorn, `/ws`). Текстовые JSON-кадры плюс бинарные
(PCM/JPEG/MP4); бинарь всегда объявлен предшествующим заголовком.

- **v1** — исторические константы `common/protocol.py`, поддерживаются.
- **v2** (Фаза 0) — Pydantic-модели в том же файле: общий конверт
  (`type`, `proto=2`, `ts`, `home_id`, `client_id`, `seq`, `utterance_id`/
  `event_id`), discriminated union по `type`, `parse_message()` отдаёт v1-кадры
  старой ветке кода и сообщает об неизвестном типе как о `ProtocolError`.

## Данные

Хаб держит одну SQLite (WAL) `data/hub.db`: дома, люди, членство, треки,
эмбеддинги, присутствие, устройства, сцены, правила, скиллы, диалоги, память,
напоминания, опросы, решения, аудит, медиа. Схему применяет
`python -m hub.migrations_runner --db data/hub.db`; версия пишется в
`schema_version`. Векторы пока хранятся как BLOB, `sqlite-vec` — следующий шаг.

## Комнатный клиент

Wake word — Vosk; VAD — webrtcvad; звук — sounddevice; камера — OpenCV +
YOLO11x; HUD — PySide6/QtWebEngine; действия — pycaw (громкость),
comtypes + psutil (приложения, браузер), bleak (SwitchBot), flux_led (Magic
Home), tinytuya (Tuya), keyboard.

## Конфигурация

Один `config.yaml` на машину. Шаблоны: `config.example.yaml` (один дом),
`config.hub.example.yaml` (секция `homes:`), `config.client.example.yaml`
(`home_id`, `hub_url`, `token_env`, `caps`). Всё — Pydantic v2 со
`extra='forbid'`; токен клиента хранится только в переменной окружения.

## Запуск и проверки

```
make hub          # хаб
make client       # клиент комнаты
make test         # полный pytest
make test-regress # регрессия реплик
make migrate      # миграции БД
make skill name=x # каркас нового скилла
```

Полный прогон требует окружения `jarvis` (CUDA-стек). В CI без GPU идут
`ruff`, `mypy common` и `tests/regress`.
