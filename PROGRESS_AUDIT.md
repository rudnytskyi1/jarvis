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
- [ ] **AU-02 — браузер: остатки после защиты `run_command`.** Защита уже в
  хабе (`opening_a_web_page`), нужно проверить живьём:
  `--family browser --family noisy`, починить остатки (первый вызов не
  `browser_control`, отсутствие слова из просьбы в аргументах), дописать тесты.
- [ ] **AU-03 — зрение: «кто в комнате», «что на экране».** Живой прогон
  `--family vision`; цель — Jev выбирает семейство `vision` и модель зовёт
  `look_at_camera`/`look_at_screen` первым вызовом. Проверять и `offered`.
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
