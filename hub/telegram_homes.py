"""Who owns which home in Telegram, and what that account may see (ТЗ F-701).

"Многодомность в Telegram. У каждого владельца дома свой чат; /tools показывает
только его дома; глобальный админ хаба видит все. Существующие expiring
callback-токены сохранить."

Two sources, one answer:

* ``homes:`` in the hub config may name a ``telegram_user_id`` per home — that is
  the deployment's own statement of who owns a room, and :meth:`HomeOwners.seed`
  writes it into the access store once (idempotent), so a config edit does not
  need the panel;
* the access store (``telegram_home_owners``) holds the grants the hub admin made
  at run time, and it is what every lookup reads afterwards.

The hub admin (``server.telegram.control_user_id``) is *not* a home owner: their
scope is ``None``, which every caller reads as "all homes" — exactly what the ТЗ
asks for ("глобальный админ хаба видит все").
"""
from __future__ import annotations

from typing import Any


class HomeOwners:
    """The homes of every Telegram account, and the grant/revoke rules."""

    def __init__(self, access: Any, homes: Any = ()) -> None:
        self.access = access
        #: ``home_id -> telegram user id`` as the deployment configured it; a
        #: value of 0 (the config default) means "nobody yet". Every configured
        #: home is kept, owner or not: the panel must still be able to grant it.
        self.configured: dict[str, int] = {}
        for home in homes or ():
            home_id = str(getattr(home, "home_id", "") or "").strip()
            if home_id:
                self.configured[home_id] = max(0, int(getattr(home, "telegram_user_id", 0) or 0))

    # --- reading ------------------------------------------------------------

    def scope(self, user_id: Any) -> frozenset[str] | None:
        """Homes this account may see; ``None`` means the whole hub."""
        if self.access is None:
            return None
        if self.access.is_owner(user_id):
            return None
        return self.access.homes_of(user_id)

    def may_use_panel(self, user_id: Any) -> bool:
        """ТЗ F-701: the hub admin, or an account that owns at least one home."""
        if self.access is None:
            return False
        if self.access.is_owner(user_id):
            return True
        return self.access.is_home_owner(user_id)

    def home_ids(self) -> tuple[str, ...]:
        """Every home the hub knows through config or grants."""
        known = set(self.configured)
        for homes in self.access.home_owners().values() if self.access is not None else ():
            known.update(homes)
        return tuple(sorted(known))

    def owners(self) -> dict[str, tuple[int, ...]]:
        """``home_id -> telegram ids`` (config seed plus run-time grants)."""
        result: dict[str, list[int]] = {home_id: [] for home_id in self.home_ids()}
        for home_id, user_id in self.configured.items():
            if user_id > 0:
                result[home_id].append(user_id)
        for user_id, homes in (self.access.home_owners().items() if self.access is not None else ()):
            for home_id in homes:
                result.setdefault(home_id, [])
                if user_id not in result[home_id]:
                    result[home_id].append(user_id)
        return {home_id: tuple(sorted(ids)) for home_id, ids in sorted(result.items())}

    def accounts(self) -> dict[int, tuple[str, ...]]:
        """``telegram id -> homes`` for the panel, config seed included."""
        result: dict[int, set[str]] = {}
        for home_id, user_id in self.configured.items():
            if user_id > 0:
                result.setdefault(user_id, set()).add(home_id)
        if self.access is not None:
            for user_id, homes in self.access.home_owners().items():
                result.setdefault(user_id, set()).update(homes)
        return {user_id: tuple(sorted(homes)) for user_id, homes in sorted(result.items())}

    # --- writing (ТЗ F-701: только глобальный админ хаба) -------------------

    def _require_admin(self, actor: Any) -> None:
        if self.access is None or not self.access.is_owner(actor):
            raise ValueError("Only the hub administrator can change home owners.")

    def grant(self, actor: Any, user_id: Any, home_id: Any) -> bool:
        """Give ``home_id`` to one Telegram account; returns True when new."""
        self._require_admin(actor)
        return self.access.grant_home(user_id, home_id)

    def revoke(self, actor: Any, user_id: Any, home_id: Any) -> bool:
        """Take a home back; the account keeps the homes it was granted."""
        self._require_admin(actor)
        return self.access.revoke_home(user_id, home_id)

    # --- startup -----------------------------------------------------------

    def seed(self) -> int:
        """Write the configured owners into the access store; returns how many.

        Idempotent, and it never revokes: removing a grant is the hub admin's
        decision, not the side effect of editing a config file.
        """
        granted = 0
        for home_id, user_id in sorted(self.configured.items()):
            if user_id <= 0:
                continue
            try:
                granted += 1 if self.access.grant_home(user_id, home_id) else 0
            except Exception:  # noqa: BLE001 - a bad home id in config is not fatal
                continue
        return granted

    def visible(self, user_id: Any, rows: Any, key: str = "home_id") -> list[Any]:
        """Filter workplace rows to the account's homes (ТЗ F-701: /tools)."""
        scope = self.scope(user_id)
        if scope is None:
            return list(rows)
        return [row for row in rows if str(row.get(key) or "") in scope]


__all__ = ["HomeOwners"]
