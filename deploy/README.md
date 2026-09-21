# Автозапуск Rowan (ТЗ 4.9)

Хаб и клиент запускаются как сервисы: хаб перезапускается сам после падения,
клиент комнаты поднимается при входе пользователя в систему.

| ОС | Чем | Файлы |
|---|---|---|
| Windows | Планировщик задач (или NSSM, если он есть) | `deploy/windows/install-rowan.ps1` |
| Linux | systemd | `deploy/systemd/rowan-hub.service`, `rowan-client.service`, `install.sh` |

## Windows

```powershell
# от имени владельца ПК; PY по умолчанию — conda-env jarvis
powershell -ExecutionPolicy Bypass -File deploy\windows\install-rowan.ps1 -Hub
powershell -ExecutionPolicy Bypass -File deploy\windows\install-rowan.ps1 -Client
```

Скрипт создаёт по задаче на сервис: запуск от текущего пользователя, повтор при
падении (хаб — каждую минуту, до 999 раз), лог в `data\logs\`. Хаб запускается
скрыто (окно не мешает), клиент — в обычном окне: у него HUD, и его должно быть
видно. Удаление — `deploy\windows\uninstall-rowan.ps1`.

## Linux

```bash
sudo deploy/systemd/install.sh           # копирует юниты и запускает хаб
```

Юниты ждут рабочую копию в `/opt/rowan`, конфиг — `/opt/rowan/config.yaml`,
интерпретатор — `/opt/rowan/.venv/bin/python`. Правьте `WorkingDirectory` и
`ExecStart`, если пути другие; после правки — `sudo systemctl daemon-reload`.

## Единые цели

| Цель | Что делает |
|---|---|
| `make hub` | запускает хаб (`python -m hub.main --config config.yaml`) |
| `make client` | запускает клиент комнаты |
| `make test` | pytest + ruff + mypy (то же, что описано в `AGENTS.md`) |
| `make test-regress` | только регрессия реплик |
| `make migrate` | применяет миграции к `data/hub.db` |
| `make skill name=X` | создаёт скилл в `skills/X` |

Секретов в этих файлах нет: ключи и пароли — только в окружении
(`ROWAN_ADMIN_PASSWORD`, `ROWAN_SPOTIFY_TOKEN`, ключи API).
