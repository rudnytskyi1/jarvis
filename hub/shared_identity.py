"""Общий профиль между домами (ТЗ F-212).

Один хаб обслуживает несколько комнат, и один человек может жить в двух. ТЗ
даёт ему право решать самому: флаг ``memberships.share_identity``. Пока флаг
снят, лицо и голос человека узнаются ТОЛЬКО в его собственном доме; если он
разрешил — во всех домах хаба, где он в членстве.

«Свой дом» — самое раннее членство человека (``created_at``): именно там его
зарегистрировали, и именно оно остаётся последней комнатой, где его узнают,
если он никому не разрешил делиться профилем. Дом, в членстве которого флаг
поднят, видит человека; дом, которого нет в членстве, не видит его никогда —
флаг не создаёт членство.

Модуль отвечает на один вопрос («виден ли этот человек здесь?») и умеет
переключать флаг, записывая это в аудит: согласие — это действие, а не
настройка в файле.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("jarvis.server.shared_identity")

#: «разреши узнавать меня в других домах» / «не узнавай меня в других домах».
_ALLOW = re.compile(
    r"\bshare my (?:identity|profile|face)\b|\bshare me with\b"
    r"|\b(?:let|allow)\b[^.]{0,40}\bother (?:homes?|rooms?)\b"
    r"|\b(?:разреши|разрешаю)\b[^.]{0,40}\b(?:узнавать|делиться)\b[^.]{0,40}\b(?:друг|дом|комнат)\w*\b"
    r"|\bделись моим профилем\b"
    r"|\bcomparte mi (?:identidad|perfil)\b",
    re.IGNORECASE,
)
_DENY = re.compile(
    r"\bstop sharing\b|\bdo not share\b|\bdon't share\b|keep my (?:identity|profile) (?:here|local)\b"
    r"|\bне (?:узнавай|делись)\b[^.]{0,40}\b(?:друг|дом|комнат)\w*\b"
    r"|\bзапре(?:ти|щаю)\b[^.]{0,40}\b(?:узнавать|делиться)\b"
    r"|\bno compartas\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Membership:
    """One row of ``memberships`` (схема 14)."""

    person_id: str
    home_id: str
    role: str
    share_identity: bool = False
    share_presence: bool = False
    created_at: str = ""


def memberships_of(conn: sqlite3.Connection, person_id: str) -> list[Membership]:
    """Every room this person belongs to, oldest membership first."""
    if not person_id:
        return []
    rows = conn.execute(
        "SELECT person_id, home_id, role, share_identity, share_presence, created_at"
        " FROM memberships WHERE person_id=? ORDER BY created_at, rowid",
        (str(person_id),)).fetchall()
    return [Membership(person_id=str(row[0]), home_id=str(row[1]), role=str(row[2]),
                       share_identity=bool(row[3]), share_presence=bool(row[4]),
                       created_at=str(row[5] or "")) for row in rows]


def primary_home(conn: sqlite3.Connection, person_id: str) -> str | None:
    """The home this person was registered in first ("своё" of ТЗ F-212)."""
    found = memberships_of(conn, person_id)
    return found[0].home_id if found else None


def shared_with(conn: sqlite3.Connection, person_id: str) -> list[str]:
    """Every room where the person allowed their profile to be recognized."""
    return sorted({row.home_id for row in memberships_of(conn, person_id) if row.share_identity})


def visible_in(conn: sqlite3.Connection, person_id: str, home_id: str) -> bool:
    """May this person's face and voice be recognized in ``home_id`` (F-212)?

    * a room where the person is not a member never sees them - consent does
      not create membership;
    * their own (first) room always does;
    * any other room only when at least one of their memberships carries
      ``share_identity``.

    A ``persons`` row that has no membership anywhere is the one exception:
    F-212 sets permissions on a membership, and a person without one is not a
    member of any house, so there is no flag to consult. Such a row (imported
    data, a person registered before memberships existed) keeps the behaviour
    of the phases before this one and is recognized. As soon as they have a
    membership, the rule above decides.
    """
    if not person_id or not home_id:
        return False
    found = memberships_of(conn, person_id)
    if not found:
        return True
    if not any(row.home_id == str(home_id) for row in found):
        return False
    if found[0].home_id == str(home_id):
        return True
    return any(row.share_identity for row in found)


def visible_people(conn: sqlite3.Connection, home_id: str) -> set[str]:
    """``person_id`` of everybody this room may recognize (one query per home)."""
    if not home_id:
        return set()
    allowed = {person_id for person_id in member_ids(conn, home_id)
               if visible_in(conn, person_id, str(home_id))}
    # A person without any membership has no flag to consult (see visible_in).
    allowed.update(str(row[0]) for row in conn.execute(
        "SELECT person_id FROM persons WHERE person_id NOT IN"
        " (SELECT person_id FROM memberships)"))
    return allowed


def member_ids(conn: sqlite3.Connection, home_id: str) -> set[str]:
    """``person_id`` of everybody who has a membership in ``home_id``.

    This is the "who is expected here" list of D-06, before F-212 is applied -
    a person who is a member of the house but did not share their profile is
    still a member of it, just not recognizable through biometrics.
    """
    if not home_id:
        return set()
    return {str(row[0]) for row in conn.execute(
        "SELECT DISTINCT person_id FROM memberships WHERE home_id=?", (str(home_id),))}


def filter_visible(conn: sqlite3.Connection, home_id: str,
                   person_ids: Iterable[str]) -> set[str]:
    """The subset of ``person_ids`` this room may recognize."""
    allowed = visible_people(conn, home_id)
    return {str(person_id) for person_id in person_ids if str(person_id) in allowed}


def set_share_identity(conn: sqlite3.Connection, person_id: str, home_id: str, shared: bool, *,
                       audit: Any = None, actor: str = "") -> bool:
    """Turn the flag of F-212 on or off for one membership.

    Only an existing membership can be changed: a person who is not a member of
    the room has nothing to share there. Returns ``True`` when the value did
    change (an idempotent call is not an event worth auditing twice).
    """
    if not person_id or not home_id:
        return False
    current = conn.execute(
        "SELECT share_identity FROM memberships WHERE person_id=? AND home_id=?",
        (str(person_id), str(home_id))).fetchone()
    if current is None:
        log.info("No membership of %s in %s - nothing to share", person_id, home_id)
        return False
    wanted = 1 if shared else 0
    if int(current[0]) == wanted:
        return False
    conn.execute("UPDATE memberships SET share_identity=? WHERE person_id=? AND home_id=?",
                 (wanted, str(person_id), str(home_id)))
    conn.commit()
    log.info("Profile sharing of %s in %s is now %s", person_id, home_id, bool(shared))
    if audit is not None:
        try:
            audit.record(action="identity.share" if shared else "identity.unshare",
                         actor=str(actor or person_id), target=str(person_id),
                         home_id=str(home_id), result="ok",
                         detail={"share_identity": bool(shared)})
        except Exception:  # noqa: BLE001 - the consent stands even if auditing fails
            log.warning("Could not audit the sharing change of %s", person_id)
    return True


def share_command(text: Any) -> bool | None:
    """Read the person's own words: allow sharing, refuse it, or neither.

    ``None`` means this utterance is not about sharing - the hub then leaves the
    settings of the person alone, because a command nobody asked for is not
    consent. Refusal wins when both patterns appear ("share... no, don't").
    """
    said = " ".join(str(text or "").split())
    if not said:
        return None
    if _DENY.search(said):
        return False
    if _ALLOW.search(said):
        return True
    return None


def describe(conn: sqlite3.Connection, person_id: str) -> dict[str, Any]:
    """What F-212 currently says about one person, for the admin view and logs."""
    rooms = memberships_of(conn, person_id)
    return {"person_id": str(person_id), "primary_home": primary_home(conn, person_id),
            "shared_with": shared_with(conn, person_id),
            "rooms": [{"home_id": row.home_id, "role": row.role,
                       "share_identity": row.share_identity} for row in rooms]}


__all__ = [
    "Membership",
    "describe",
    "filter_visible",
    "member_ids",
    "memberships_of",
    "primary_home",
    "set_share_identity",
    "share_command",
    "shared_with",
    "visible_in",
    "visible_people",
]
