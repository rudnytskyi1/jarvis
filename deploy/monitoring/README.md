# Локальный мониторинг Rowan (Prometheus + Grafana)

Метрики хаба Rowan (ТЗ F-707) и его дашборд (ТЗ F-707, `deploy/grafana/`).
Это про **здоровье системы во времени**: латентность стадий и комнат, очередь
видеокарты, VRAM, ходы и ошибки, расход бюджета API, уровень модели. Текстов,
промптов и цепочки конкретного запроса здесь нет — в `/metrics` их не бывает
по требованию ТЗ 15.5 (никаких имён, реплик и кадров). Что именно произошло в
одном запросе, показывает панель владельца: `/admin/turns`.

## Как запустить

```powershell
pwsh -File scripts\start-monitoring.ps1          # поднять
pwsh -File scripts\start-monitoring.ps1 -Stop    # остановить
```

Затем открыть:

| Что | Адрес | Логин |
|---|---|---|
| Grafana (дашборд «Rowan hub») | http://localhost:3001 | `admin` / пароль из `GRAFANA_ADMIN_PASSWORD` в `.env` |
| Prometheus (сырые серии, targets) | http://localhost:9090 | без входа, только localhost |

## Панель владельца (там же, где цепочка запроса)

| Что | Адрес | Логин |
|---|---|---|
| Панель владельца: дома, клиенты, люди, разметка | http://127.0.0.1:8770/admin | только пароль из `ROWAN_ADMIN_PASSWORD` в `.env` |
| Список запросов (голос и Telegram) | http://127.0.0.1:8770/admin/turns | то же |
| Цепочка одного запроса: решения, инструменты, промпты, что сказали | http://127.0.0.1:8770/admin/turns/&lt;turn_id&gt; | то же |

Панель живёт в том же приложении, что хаб, поэтому её порт — порт хаба
(`server.port`, 8770). Она отвечает только под своими именами (`localhost` или
адрес из `allowed_networks`, например Tailscale `100.x.y.z`): публичный туннель
ngrok получает 404, хотя запрос оттуда приходит с loopback. `turn_id` комнаты —
это `utterance_id` хода, у Telegram — `telegram:<чат>:<сообщение>`.

Порт Grafana — **3001**, потому что 3000 занят другим сервисом владельца.
Дашборд и источник данных прописаны заранее (`provisioning/`), импортировать
вручную ничего не нужно; UID дашборда `rowan-hub`, UID источника `prometheus`
— на них смотрит `rowan-dashboard.json`.

## Где что лежит

| Что | Путь |
|---|---|
| Бинарники (не в git) | `C:\Users\Anton\Desktop\monitoring\prometheus\...`, `...\grafana\grafana-13.2.2\` |
| Конфиг Prometheus | `deploy/monitoring/prometheus.yml` |
| Провisioning Grafana | `deploy/monitoring/grafana/provisioning/` |
| Дашборд | `deploy/grafana/rowan-dashboard.json` |
| Логи сервисов | `C:\Users\Anton\Desktop\monitoring\logs\` |
| Данные Prometheus (30 дней) | `C:\Users\Anton\Desktop\monitoring\data\prometheus` |

## Порт хаба

`prometheus.yml` смотрит на `127.0.0.1:8770` — это порт владельца
(`config.yaml` / `config.openai.yaml`). Шаблон `config.example.yaml` слушает
8765; если порт сменится, правится адрес цели в `prometheus.yml` и Prometheus
перезапускается.
