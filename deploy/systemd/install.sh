#!/usr/bin/env bash
# Установка юнитов Rowan (ТЗ 4.9): копирует их в /etc/systemd/system и запускает.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target=/etc/systemd/system
install -d "$target"
install -m 644 "$here/rowan-hub.service" "$target/rowan-hub.service"
install -m 644 "$here/rowan-client.service" "$target/rowan-client.service"
systemctl daemon-reload
systemctl enable rowan-hub.service
systemctl restart rowan-hub.service
echo "Hub started. For a room PC also run: systemctl enable --now rowan-client.service"
systemctl --no-pager --lines=5 status rowan-hub.service || true
