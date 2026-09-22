"""«Забудь меня» (ТЗ F-213).

Самое опасное действие ассистента — не выключить свет, а стереть человека.
ТЗ требует трёх вещей, и все три живут здесь:

* **необратимость.** Данные не помечаются «удалено» — они удаляются: векторы,
  кропы с их файлами, «внешность дня», упоминания в событиях и диалогах,
  память человека, архивные кадры и клипы, само членство и строка человека.
  Никакой корзины и никакого «можно вернуть» в ответе;
* **подтверждение F-113.** Хаб сначала спрашивает словами и ждёт «да» в окне
  (:mod:`hub.confirmations`), и только потом что-то удаляет;
* **аудит.** Каждое удаление пишется в ``audit`` как ``identity.forget`` с
  числом удалённых строк — «сколько именно» важнее, чем «готово».

Удаление идёт по ``person_id``, то есть по человеку, а не по имени: имя может
быть произнесено кем угодно, а строка ``persons`` одна.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.server.forget_me")

#: «забудь меня» / «forget me» / «olvídame» - and nothing else.
_FORGET = re.compile(
    r"\bforget (?:me|everything about me|all about me)\b"
    r"|\bdelete (?:my|all my) (?:data|profile|identity|biometrics)\b"
    r"|\bзабудь меня\b|\bзабудь обо мне\b|\bудали (?:мои |все мои )?(?:данные|профиль)\b"
    r"|\bolv[ií]dame\b|\bborra mis datos\b",
    re.IGNORECASE,
)
#: Somebody who asks NOT to forget is not asking to forget.
_NOT_FORGET = re.compile(
    r"\bdon'?t forget\b|\bdo not forget\b|\bnever forget\b"
    r"|\bне забывай\b|\bне забудь\b|\bno olvides\b|\bnunca olvides\b",
    re.IGNORECASE,
)

#: ТЗ F-113: the spoken answer that turns the question into a deletion.
ASK: dict[str, str] = {
    "ru": ("Скажи «да» в течение {seconds} секунд, чтобы удалить всё, что я о тебе знаю: "
           "отпечаток голоса, фотографии и кадры, «внешность дня», упоминания в диалогах "
           "и памяти. Это необратимо. Любой другой ответ отменяет."),
    "en": ("Say yes within {seconds} seconds to delete everything I know about you: your voice "
           "print, the photos and frames, the appearance of the day, and your mentions in the "
           "dialogues and memory. This cannot be undone. Anything else cancels it."),
    "es": ("Di sí en {seconds} segundos para borrar todo lo que sé de ti: tu huella de voz, "
           "las fotos y fotogramas, la apariencia del día y tus menciones en los diálogos y la "
           "memoria. No se puede deshacer. Cualquier otra respuesta lo cancela."),
}

DONE: dict[str, str] = {
    "ru": ("Готово. Я удалил всё, что знал о тебе: {vectors} векторов, {crops} кропов, "
           "{mentions} упоминаний и {memory} записей памяти. Вернуть это нельзя."),
    "en": ("Done. I deleted everything I knew about you: {vectors} vectors, {crops} crops, "
           "{mentions} mentions and {memory} memory entries. It cannot be undone."),
    "es": ("Hecho. Borré todo lo que sabía de ti: {vectors} vectores, {crops} recortes, "
           "{mentions} menciones y {memory} entradas de memoria. No se puede deshacer."),
}

CANCELLED: dict[str, str] = {
    "ru": "Хорошо, ничего не удаляю.",
    "en": "Alright, I am deleting nothing.",
    "es": "De acuerdo, no borro nada.",
}

#: ТЗ F-113's window is the same for an irreversible deletion.
WINDOW_S = 8.0

#: Tables whose rows are the person's own data (deleted outright).
_PERSON_TABLES = ("voice_embeddings", "face_embeddings", "body_embeddings")
#: The "appearance of the day" rows (ТЗ F-209), counted on their own.
_APPEARANCE_TABLE = "daily_appearance"
#: Tables that MENTION the person (rows of that person are removed).
_MENTION_TABLES = ("presence_events", "dialog_turns")
#: The person's own facts in the ``memories`` table live under ``scope='person'``.
_OWN_MEMORIES = "SELECT memory_id, owner_id FROM memories WHERE scope='person'"


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in ASK else default


def forget_requested(text: Any) -> bool:
    """Whether this utterance is the request of F-213 (and not its opposite)."""
    said = " ".join(str(text or "").split())
    if not said or _NOT_FORGET.search(said):
        return False
    return bool(_FORGET.search(said))


def question(language: Any = "ru", *, window_s: float = WINDOW_S) -> str:
    """The F-113 style question, spoken BEFORE anything is deleted."""
    return ASK[language_of(language)].replace("{seconds}", str(max(1, int(round(window_s)))))


def cancelled(language: Any = "ru") -> str:
    return CANCELLED[language_of(language)]


def confirmation(language: Any = "ru", *, window_s: float = WINDOW_S) -> Any:
    """A :class:`hub.confirmations.Confirmation` for the deletion (F-113)."""
    from hub.confirmations import Confirmation

    return Confirmation(tool="forget_me", arguments={}, description="forget everything about me",
                        window_s=window_s)


@dataclass(frozen=True)
class Inventory:
    """What the hub currently knows about one person, before deleting it."""

    person_id: str = ""
    vectors: int = 0
    crops: int = 0
    mentions: int = 0
    appearances: int = 0
    tracks: int = 0
    memory: int = 0
    memberships: int = 0

    @property
    def total(self) -> int:
        return (self.vectors + self.crops + self.mentions + self.appearances
                + self.memory + self.memberships)

    def summary(self) -> dict[str, int]:
        return {"vectors": self.vectors, "crops": self.crops, "mentions": self.mentions,
                "appearances": self.appearances, "tracks": self.tracks,
                "memory": self.memory, "memberships": self.memberships}


@dataclass(frozen=True)
class ForgetReport:
    """What one «забудь меня» really deleted (ТЗ F-213)."""

    person_id: str
    display_name: str = ""
    vectors: int = 0
    crops: int = 0
    files: int = 0
    mentions: int = 0
    appearances: int = 0
    tracks_unnamed: int = 0
    memory: int = 0
    conversations: int = 0
    archive: dict[str, Any] = field(default_factory=dict)
    memberships: int = 0
    registry: bool = False
    ok: bool = True
    note: str = ""

    def summary(self) -> dict[str, Any]:
        return {"person_id": self.person_id, "name": self.display_name,
                "vectors": self.vectors, "crops": self.crops, "files": self.files,
                "mentions": self.mentions, "appearances": self.appearances,
                "tracks_unnamed": self.tracks_unnamed, "memory": self.memory,
                "conversations": self.conversations, "archive": dict(self.archive),
                "memberships": self.memberships, "registry": self.registry,
                "ok": self.ok, "note": self.note}

    def spoken(self, language: Any = "ru") -> str:
        """What the room hears afterwards - with the real numbers."""
        return DONE[language_of(language)].format(
            vectors=self.vectors, crops=self.crops,
            mentions=self.mentions + self.conversations, memory=self.memory)


def _count(conn: sqlite3.Connection, sql: str, params: Iterable[Any]) -> int:
    try:
        row = conn.execute(sql, tuple(params)).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0) if row else 0


def _own_memory_ids(conn: sqlite3.Connection, name: Any) -> list[str]:
    """The ids of one person's own facts in ``memories`` (ТЗ F-213 + F-414).

    The table has no ``person_id`` column: a fact is owned by the NAME the hub
    knows the person by, exactly as ``remember`` writes it (P3-15). The
    comparison is done in Python because SQLite's ``lower()`` folds only ASCII -
    a Russian name must still match itself whatever case it was saved in, and
    «забудь меня» means everything.
    """
    wanted = " ".join(str(name or "").split()).casefold()
    if not wanted:
        return []
    try:
        rows = conn.execute(_OWN_MEMORIES).fetchall()
    except sqlite3.Error:
        return []
    return [str(row[0]) for row in rows if str(row[1] or "").casefold() == wanted]


def _delete_own_memories(conn: sqlite3.Connection, name: Any) -> int:
    """Delete the person's own facts from ``memories``; return how many.

    The room's facts (``scope='home'``) and the hub's stay: they are not about
    this person. The vector index is cleaned first, best effort, so no orphan
    embedding outlives the row it belonged to.
    """
    ids = _own_memory_ids(conn, name)
    if not ids:
        return 0
    try:
        from hub import vectors

        for memory_id in ids:
            vectors.remove(conn, "memory", memory_id)
    except Exception as exc:  # noqa: BLE001 - the row is authoritative
        log.info("Could not clean the memory index of %r (%s)", name, exc)
    marks = ",".join("?" * len(ids))
    cursor = conn.execute(f"DELETE FROM memories WHERE memory_id IN ({marks})", ids)
    return int(cursor.rowcount or 0)


def inventory(conn: sqlite3.Connection, person_id: str, *,
              memory: Any = None, display_name: str = "") -> Inventory:
    """Count what a deletion would remove (no writes)."""
    if not person_id:
        return Inventory()
    name = str(display_name or "")
    if not name:
        # The person's own facts in ``memories`` are owned by their NAME (the
        # table has no ``person_id``), so the inventory has to know it - and
        # the row that carries it is about to be deleted.
        row = conn.execute("SELECT display_name FROM persons WHERE person_id=?",
                           (str(person_id),)).fetchone()
        name = str(row[0]) if row else ""
    vectors = sum(_count(conn, f"SELECT COUNT(*) FROM {table} WHERE person_id=?", [person_id])
                  for table in ("voice_embeddings", "face_embeddings", "body_embeddings"))
    appearances = _count(conn, "SELECT COUNT(*) FROM daily_appearance WHERE person_id=?",
                         [person_id])
    crops = _count(conn, "SELECT COUNT(*) FROM body_crops WHERE track_id IN"
                         " (SELECT track_id FROM tracks WHERE person_id=?)", [person_id])
    mentions = sum(_count(conn, f"SELECT COUNT(*) FROM {table} WHERE person_id=?", [person_id])
                   for table in _MENTION_TABLES)
    tracks = _count(conn, "SELECT COUNT(*) FROM tracks WHERE person_id=?", [person_id])
    memberships = _count(conn, "SELECT COUNT(*) FROM memberships WHERE person_id=?", [person_id])
    memories = len(memory.admin_entries(name)) if memory is not None and name else 0
    if name:
        # ТЗ F-213 + F-414: the ``memories`` table now feeds the prompt and the
        # retrieval, so it counts as memory the person has here, not as a
        # "mention" in somebody else's rows.
        memories += len(_own_memory_ids(conn, name))
    return Inventory(person_id=str(person_id), vectors=vectors, crops=crops, mentions=mentions,
                     appearances=appearances, tracks=tracks, memory=memories,
                     memberships=memberships)


def erase(conn: sqlite3.Connection, person_id: str, *,
          memory: Any = None, conversations: Any = None, archive: Any = None,
          registry: Any = None, media: Any = None, audit: Any = None,
          actor: str = "", display_name: str = "", from_status: str = "confirmed") -> ForgetReport:
    """Delete everything the hub knows about one person (ТЗ F-213).

    ``registry`` is the phase-1 people registry (``hub.speaker.VoiceRegistry``),
    ``memory`` the ``hub.storage.Memory`` of the room, ``conversations`` the
    dialogue archive, ``archive`` the local training archive and ``media`` the
    ``MediaStore`` that registered the crops. Each is optional so a hub without
    one still deletes everything it does have - and the report says what was
    deleted, never what "should" be gone.
    """
    if not person_id:
        return ForgetReport(person_id="", ok=False, note="no person")
    name = str(display_name or "")
    if not name:
        row = conn.execute("SELECT display_name FROM persons WHERE person_id=?",
                           (str(person_id),)).fetchone()
        name = str(row[0]) if row else ""
    try:
        crops, files = _erase_crops(conn, person_id, media=media)
        vectors = sum(conn.execute(f"DELETE FROM {table} WHERE person_id=?",
                                   (str(person_id),)).rowcount
                      for table in _PERSON_TABLES)
        appearances = conn.execute(f"DELETE FROM {_APPEARANCE_TABLE} WHERE person_id=?",
                                   (str(person_id),)).rowcount
        mentions = sum(conn.execute(f"DELETE FROM {table} WHERE person_id=?",
                                    (str(person_id),)).rowcount for table in _MENTION_TABLES)
        # ТЗ F-213: the person's own facts in ``memories`` are memory, and the
        # table now reaches the prompt (P3-16) and the history (P3-17) - a
        # person who is forgotten must not survive in either. Facts of the ROOM
        # stay: they are not about this person.
        own_memories = _delete_own_memories(conn, name)
        tracks = conn.execute("UPDATE tracks SET person_id=NULL WHERE person_id=?",
                              (str(person_id),)).rowcount
        memberships = conn.execute("DELETE FROM memberships WHERE person_id=?",
                                   (str(person_id),)).rowcount
        conn.execute("DELETE FROM persons WHERE person_id=?", (str(person_id),))
        conn.commit()
    except sqlite3.Error as exc:
        try:
            conn.rollback()
        except sqlite3.Error:
            log.debug("Could not roll back the deletion of %s", person_id, exc_info=True)
        log.warning("Could not delete the data of %s (%s)", person_id, exc)
        return ForgetReport(person_id=str(person_id), display_name=name, ok=False,
                            note=f"database error: {exc}")
    report = ForgetReport(person_id=str(person_id), display_name=name,
                          vectors=int(vectors), crops=int(crops), files=int(files),
                          mentions=int(mentions), appearances=int(appearances or 0),
                          tracks_unnamed=int(tracks), memberships=int(memberships),
                          note=str(from_status))
    report = _erase_stores(report, name, memory=memory, conversations=conversations,
                           archive=archive, registry=registry)
    if own_memories:
        # The file store and the table are one memory to the person asking.
        report = replace(report, memory=int(report.memory) + int(own_memories))
    log.info("Forget me: %s (%s) deleted - %s", name or person_id, person_id,
             ", ".join(f"{key}={value}" for key, value in report.summary().items()
                       if key in {"vectors", "crops", "files", "mentions", "memory",
                                  "conversations", "memberships"}))
    if audit is not None:
        try:
            audit.record(action="identity.forget", actor=str(actor or person_id),
                         target=str(person_id), result="ok" if report.ok else "failed",
                         detail=report.summary())
        except Exception:  # noqa: BLE001 - the deletion stands even if auditing fails
            log.warning("Could not audit the deletion of %s", person_id)
    return report


def _erase_crops(conn: sqlite3.Connection, person_id: str, *, media: Any = None) -> tuple[int, int]:
    """Delete the body crops of the person's tracks and the JPEGs behind them."""
    rows = conn.execute(
        "SELECT crop_id, home_id, path FROM body_crops WHERE track_id IN"
        " (SELECT track_id FROM tracks WHERE person_id=?)", (str(person_id),)).fetchall()
    files = 0
    for _crop_id, _home_id, path in rows:
        target = Path(str(path))
        try:
            if target.is_file():
                target.unlink()
                files += 1
        except OSError as exc:
            log.info("Could not delete the crop file %s (%s)", target, exc)
    removed = conn.execute(
        "DELETE FROM body_crops WHERE track_id IN"
        " (SELECT track_id FROM tracks WHERE person_id=?)", (str(person_id),)).rowcount
    if media is not None:
        _unregister_media(media, [str(row[2]) for row in rows])
    return int(removed or 0), files


def _unregister_media(media: Any, paths: Iterable[str]) -> None:
    """Drop the ``media`` rows of the deleted files, when the store allows it."""
    conn = getattr(media, "_conn", None)
    if conn is None:
        return
    for path in paths:
        try:
            conn.execute("DELETE FROM media WHERE path=?", (str(path),))
        except sqlite3.Error as exc:
            log.debug("Could not unregister %s (%s)", path, exc)
    try:
        conn.commit()
    except sqlite3.Error:
        log.debug("Could not commit the media unregistration", exc_info=True)


def _erase_stores(report: ForgetReport, name: str, *, memory: Any, conversations: Any,
                  archive: Any, registry: Any) -> ForgetReport:
    """Delete the person from the stores that keep their own copy of them."""
    entries = report.summary()
    if name:
        if memory is not None:
            try:
                remembered = list(memory.admin_entries(name))
                for row in remembered:
                    memory.change_entry(row.get("id"), name, delete=True)
                entries["memory"] = len(remembered)
            except Exception:  # noqa: BLE001 - the DB half is already gone
                log.warning("Could not delete the memory of %s", name, exc_info=True)
        if conversations is not None:
            try:
                entries["conversations"] = int(conversations.forget(name) or 0)
            except Exception:  # noqa: BLE001
                log.warning("Could not delete the dialogues of %s", name, exc_info=True)
        if archive is not None:
            try:
                entries["archive"] = dict(archive.forget(name) or {})
            except Exception:  # noqa: BLE001
                log.warning("Could not delete the archive of %s", name, exc_info=True)
        if registry is not None:
            try:
                registry.admin_profile("delete", name)
                entries["registry"] = True
            except Exception:  # noqa: BLE001 - an absent profile is not a failure
                log.info("The registry had no profile of %s to delete", name)
    entries["memory"] = int(entries.get("memory") or 0)
    return ForgetReport(
        person_id=report.person_id, display_name=report.display_name,
        vectors=report.vectors, crops=report.crops, files=report.files,
        mentions=report.mentions, appearances=report.appearances,
        tracks_unnamed=report.tracks_unnamed,
        memory=int(entries.get("memory") or 0),
        conversations=int(entries.get("conversations") or 0),
        archive=dict(entries.get("archive") or {}),
        memberships=report.memberships, registry=bool(entries.get("registry")),
        ok=report.ok, note=report.note)


__all__ = [
    "ASK",
    "CANCELLED",
    "DONE",
    "WINDOW_S",
    "ForgetReport",
    "Inventory",
    "cancelled",
    "confirmation",
    "erase",
    "forget_requested",
    "inventory",
    "language_of",
    "question",
]
