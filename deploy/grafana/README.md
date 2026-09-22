# Дашборд Rowan для Grafana (ТЗ F-707)

`rowan-dashboard.json` — готовый дашборд хаба: латентность по стадиям и домам,
очередь видеокарты, её память, ошибки и расход бюджета API. Он читает только
метрики `/metrics`, ничего не настраивает в хабе и не ходит никуда, кроме
Prometheus.

## Как подключить

1. **Включить метрики.** Они включены по умолчанию
   (`server.metrics.enabled: true` в `config.yaml`); выключенный endpoint
   честно отвечает `404`, а не пустым текстом.
2. **Проверить руками**, что хаб отдаёт метрики — `/metrics` живёт на том же
   порту и с той же видимостью, что `/health` (по умолчанию `8765`), то есть
   внутри сети хаба:

   ```powershell
   Invoke-WebRequest http://127.0.0.1:8765/metrics | Select-Object -Expand Content | Select-Object -First 20
   ```

3. **Настроить scrape в Prometheus** (`prometheus.yml`), добавив задание:

   ```yaml
   scrape_configs:
     - job_name: rowan
       metrics_path: /metrics
       scrape_interval: 15s
       static_configs:
         - targets: ["127.0.0.1:8765"]   # адрес хаба в overlay-сети
   ```

4. **Импортировать дашборд.** В Grafana: *Dashboards → New → Import → Upload
   JSON file* → выбрать `rowan-dashboard.json`. Дашборд приносит переменную
   `datasource` («Prometheus») — в мастере импорта укажите свой источник
   Prometheus; UID дашборда `rowan-hub`, поэтому повторный импорт обновляет
   тот же дашборд, а не плодит копии.

## Что на панелях

| Панель | Метрика | Зачем |
|---|---|---|
| Turn latency by stage (p95) | `rowan_turn_stage_milliseconds` | Бюджеты стадий ТЗ 15.1 (STT/диаризация/голос — по 0,7 с); гистограмма считает p50 и p95 |
| Turn latency by home (p95) | `rowan_turn_stage_milliseconds{stage="total"}` | «Реплика одной комнаты не задерживает другую» — сравниваются дома |
| GPU queue | `rowan_gpu_queue_jobs`, `rowan_gpu_queue_wait_seconds`, `rowan_gpu_queue_dropped_total` | Хвост очереди и отказы: растут раньше, чем люди это почувствуют |
| Video memory | `rowan_gpu_vram_bytes` | Что драйвер CUDA отдаёт прямо сейчас (`used`/`free`/`total`) |
| Turns, incomplete turns and errors | `rowan_turns_total`, `rowan_turn_degraded_total`, `rowan_turn_errors_total`, `rowan_outbound_dropped_total`, `rowan_scheduler_failures_total` | ТЗ F-704: неполные ходы видно отдельно от удач |
| Cloud API budget | `rowan_api_budget_usd`, `rowan_api_unsettled_requests` | Консервативный учёт хаба: резервация считается расходом, пока не подтверждена |
| Which model level answered | `rowan_turn_level_total` | Сколько ходов ушло в облако, а сколько осталось локальным (F-401/F-403) |

## Приватность

В метках есть только дом (идентификатор комнаты из конфига), стадия, класс
очереди, уровень модели, вид ошибки и имя задачи планировщика. Имени человека,
текста реплики, кадров, токенов и ключей в `/metrics` нет — это проверяется
тестом `tests/test_metrics.py` и требованием ТЗ 15.5. Grafana и Prometheus
должны быть доступны так же, как сам хаб (overlay-сеть), потому что
`/metrics` намеренно имеет ту же видимость, что `/health`.
