"""Общий список покупок и дел (ТЗ F-610).

ТЗ F-610: «Общий список покупок и дел. Список на группу домов; добавление
голосом из любой комнаты, чтение в Telegram и на HUD».

Три вещи здесь и есть весь список: разбор реплики (что человек просит —
добавить, прочитать, вычеркнуть), хранилище строк (``shopping_items``) и
строки ответов на ru/en/es. Список живёт у ГРУППЫ (``group_id``, по умолчанию
весь хаб), поэтому добавленное из одной комнаты читают и другие, и Telegram.

Чего хаб НЕ делает: не угадывает пункт по «похожести» (вычеркнуть можно только
то, что в списке есть), не придумывает список, когда он пуст, и не считает
вопросом любую фразу со словом «список».
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.ids import new_ulid

log = logging.getLogger("jarvis.server.shopping")

#: Список по умолчанию принадлежит всему хабу (одна группа домов).
DEFAULT_GROUP = "hub"
#: Сколько пунктов показывать в ответе и на карточке.
DEFAULT_LIST_LIMIT = 15
#: Длина одного пункта: «молоко» — пункт, абзац — уже не пункт.
MAX_ITEM_LENGTH = 120
#: Языки, на которых хаб говорит (раздел 1 ТЗ).
LANGUAGES = ("ru", "en", "es")


class ShoppingError(RuntimeError):
    """С пунктом списка нельзя ничего сделать: пусто, не найдено, слишком длинно."""


class ItemStatus(StrEnum):
    """Пункт либо ждёт покупки, либо уже вычеркнут."""

    OPEN = "open"
    DONE = "done"


class ShoppingAction(StrEnum):
    """Что человек просит сделать со списком."""

    ADD = "add"
    LIST = "list"
    DONE = "done"
    REMOVE = "remove"
    CLEAR = "clear"


class ShoppingRequest(BaseModel):
    """Разобранная реплика: действие и что именно (для add/done/remove)."""

    model_config = ConfigDict(extra="forbid")

    action: ShoppingAction = ShoppingAction.LIST
    text: str = Field(default="", max_length=MAX_ITEM_LENGTH)
    matched: str = Field(default="", max_length=200)


class ShoppingItem(BaseModel):
    """Один пункт общего списка (ТЗ F-610)."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(default_factory=new_ulid)
    group_id: str = DEFAULT_GROUP
    text: str = ""
    status: ItemStatus = ItemStatus.OPEN
    added_by: str = ""
    home_id: str = ""
    created_at: float = 0.0
    done_at: float = 0.0

    @property
    def open(self) -> bool:
        return self.status is ItemStatus.OPEN


# ---------------------------------------------------------------------------
# разбор реплики
# ---------------------------------------------------------------------------

#: Как называется список: «список покупок», «список дел», «the shopping list»,
#: «la lista de la compra». Английское слово-определение идёт ПЕРЕД ``list``,
#: русское и испанское — после, поэтому варианты названы целиком.
_LIST_WORD = (r"(?:(?i:список|списке|списка)"
              r"(?:\s+(?i:покупок|дел|покупки|дел\s+и\s+покупок))?"
              r"|(?i:(?:shopping|to-?do)\s+list|list)"
              r"|(?i:lista(?:\s+de\s+(?:la\s+)?(?:compra|compras|tareas))?))")
_ADD_VERB = (r"(?i:добав\w+|добавь|запиш\w+|запис\w+|внеси|внест\w+|полож\w+|"
             r"add|append|put|write|a[ñn]ade|agrega|apunta|pon)")
_DONE_VERB = (r"(?i:куп\w+|купил\w*|куплен\w*|вычеркн\w*|отметь|отмет\w+|зачеркн\w+|"
              r"готово|сдел\w*|bought|bought\s+it|tick\w*|check\w*\s+off|done|compr\w+|"
              r"marca\w*|tacha\w*|hecho)")
_REMOVE_VERB = (r"(?i:убери|убрат\w+|удали|удалит\w+|remove|delete|borra|quita)")
_CLEAR_VERB = (r"(?i:очист\w+|очисть|сотри|стереть|выброси|clear|wipe|empty|limpi\w+|borra\s+todo)")
_LIST_ASK = (r"(?i:(?:что\s+(?:купить|нужно\s+купить|в)|покаж\w+|прочит\w+|"
             r"what(?:'s|\s+is|\s+are)?|read|qu[ée]\s+(?:hay\s+en|comprar)|lee)"
             r"\s+(?:(?:on|in|en)\s+)?(?:(?:the|la|el)\s+)?)")

#: Пункт после предлога: «добавь В СПИСОК молоко», «add TO THE LIST milk».
_INTO_LIST = (r"(?:в|во|к|for|to|on|en|a)\s+(?:(?i:the|la|el|los|las|my|мой|наш|our)\s+)?"
              + _LIST_WORD)
_PREPOSITION = r"(?:(?i:в|во|на|к|for|to|into|on|en|a)\s+)?"

_PATTERNS: tuple[tuple[ShoppingAction, re.Pattern[str], str], ...] = (
    # «добавь в список покупок молоко» / «добавь молоко в список»
    (ShoppingAction.ADD,
     re.compile(r"^(?:" + _ADD_VERB + r")\s+" + _INTO_LIST
                + r"(?:\s+|:\s*)(?P<item>.+?)\s*[.!?]?$"), "item"),
    (ShoppingAction.ADD,
     re.compile(r"^(?:" + _ADD_VERB + r")\s+(?P<item>.+?)\s+"
                + _INTO_LIST + r"\s*[.!?]?$"), "item"),
    # «add milk to the shopping list»
    (ShoppingAction.ADD,
     re.compile(r"^(?:" + _ADD_VERB + r")\s+(?P<item>.+?)\s+(?:(?i:to|on|en|a)\s+)"
                r"(?:(?i:the|my|our|la|el)\s+)?" + _LIST_WORD + r"\s*[.!?]?$"), "item"),
    # «купил молоко» / «вычеркни молоко» / «отметь хлеб» — пункт выполнен.
    (ShoppingAction.DONE,
     re.compile(r"^(?:" + _DONE_VERB + r")\s+(?P<item>.+?)\s*[.!?]?$"), "item"),
    # «убери молоко из списка» / «remove milk from the list»
    (ShoppingAction.REMOVE,
     re.compile(r"^(?:" + _REMOVE_VERB + r")\s+(?P<item>.+?)"
                r"(?:\s+(?:(?i:из|из\s+списка|from|of|de)\s+" + _PREPOSITION
                + _LIST_WORD + r")?)\s*[.!?]?$"), "item"),
    # «очисти список покупок» / «clear the shopping list»
    (ShoppingAction.CLEAR,
     re.compile(r"^(?:(?i:очист\w+|очисть|сотри|стереть|выброси|clear|wipe|empty|limpi\w+))\s+"
                + _PREPOSITION + _LIST_WORD + r"\s*[.!?]?$"), ""),
    # «что в списке покупок?» / «покажи список дел» / «what is on the list?»
    (ShoppingAction.LIST,
     re.compile(r"^" + _LIST_ASK + _LIST_WORD + r"\s*[.!?]?$"), ""),
)


def shopping_command(text: Any) -> ShoppingRequest | None:
    """Разобрать реплику про общий список; ``None`` — это про другое."""
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    for action, pattern, group in _PATTERNS:
        found = pattern.match(phrase)
        if found is None:
            continue
        item = ""
        if group:
            item = " ".join(str(found.group(group) or "").split())
            item = item.strip(" ,.!?«»—-\t")
            # «добавь в список покупок» без самого пункта — это не добавление.
            if not item or item.casefold() in _NOT_AN_ITEM:
                continue
        return ShoppingRequest(action=action, text=item[:MAX_ITEM_LENGTH],
                               matched=phrase[:200])
    return None


#: Служебные слова, которые не бывают пунктом списка.
_NOT_AN_ITEM = frozenset((
    "список", "списке", "списка", "покупки", "покупок", "дел", "list", "lista",
    "the list", "la lista", "to the list", "в список", "в списке",
))


# ---------------------------------------------------------------------------
# строки для комнаты
# ---------------------------------------------------------------------------


def _language_of(language: Any) -> str:
    code = str(language or "")[:2].casefold()
    return code if code in LANGUAGES else "ru"


def added_line(item: ShoppingItem, *, language: Any = "ru") -> str:
    text = item.text
    if _language_of(language) == "ru":
        return f"Добавила в список: {text}."
    if _language_of(language) == "es":
        return f"Añadido a la lista: {text}."
    return f"Added to the list: {text}."


def empty_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Список пуст — скажи, что добавить."
    if _language_of(language) == "es":
        return "La lista está vacía: dime qué añadir."
    return "The list is empty — tell me what to add."


def list_line(items: Iterable[ShoppingItem], *, language: Any = "ru",
              more: int = 0) -> str:
    rows = list(items)
    if not rows:
        return empty_line(language=language)
    body = "; ".join(item.text for item in rows)
    tail = ""
    if more > 0:
        tail = {"ru": f" и ещё {more}", "es": f" y {more} más",
                "en": f" and {more} more"}[_language_of(language)]
    if _language_of(language) == "ru":
        return f"В списке: {body}{tail}."
    if _language_of(language) == "es":
        return f"En la lista: {body}{tail}."
    return f"On the list: {body}{tail}."


def done_line(item: ShoppingItem, *, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return f"Вычеркнула: {item.text}."
    if _language_of(language) == "es":
        return f"Tachado: {item.text}."
    return f"Ticked off: {item.text}."


def removed_line(item: ShoppingItem, *, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return f"Убрала из списка: {item.text}."
    if _language_of(language) == "es":
        return f"Quitado de la lista: {item.text}."
    return f"Removed from the list: {item.text}."


def cleared_line(*, language: Any = "ru", removed: int = 0) -> str:
    if _language_of(language) == "ru":
        return f"Список очищен ({int(removed)} пункт(ов))."
    if _language_of(language) == "es":
        return f"Lista vaciada ({int(removed)} elementos)."
    return f"The list is cleared ({int(removed)} item(s))."


def not_found_line(text: str, *, language: Any = "ru") -> str:
    what = " ".join(str(text or "").split())
    if _language_of(language) == "ru":
        return f"В списке нет «{what}» — не буду вычёркивать то, чего не вижу."
    if _language_of(language) == "es":
        return f"En la lista no hay «{what}»: no tacho lo que no veo."
    return f"There is no «{what}» on the list — I will not tick off what I cannot see."


def unavailable_line(reason: str, *, language: Any = "ru") -> str:
    reason = " ".join(str(reason or "").split()) or "the reason is unknown"
    if _language_of(language) == "ru":
        return f"Со списком не получилось: {reason}."
    if _language_of(language) == "es":
        return f"No pude con la lista: {reason}."
    return f"The list did not work out: {reason}."


def card_title(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Список покупок и дел"
    if _language_of(language) == "es":
        return "Lista de la compra"
    return "Shopping and to-do list"


def card_text(items: Iterable[ShoppingItem], *, language: Any = "ru",
              more: int = 0) -> str:
    """Текст карточки HUD (F-709): пункты списка построчно, вычеркнутое — ``—``."""
    rows = list(items)
    if not rows:
        return {"ru": "пусто", "es": "vacía", "en": "empty"}[_language_of(language)]
    lines = [f"• {item.text}" for item in rows]
    if more > 0:
        lines.append({"ru": f"и ещё {more}", "es": f"y {more} más",
                      "en": f"and {more} more"}[_language_of(language)])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# хранилище
# ---------------------------------------------------------------------------

_COLUMNS = "item_id, group_id, text, status, added_by, home_id, created_at, done_at"


def _row(row: Any) -> ShoppingItem:
    keys = tuple(name.strip() for name in _COLUMNS.split(","))
    data = dict(zip(keys, row, strict=False))
    for number in ("created_at", "done_at"):
        data[number] = float(data.get(number) or 0.0)
    for field in ("group_id", "text", "added_by", "home_id"):
        data[field] = str(data.get(field) or "")
    return ShoppingItem.model_validate(data)


class ShoppingStore:
    """Общий список покупок и дел в базе хаба (ТЗ F-610, ``shopping_items``)."""

    def __init__(self, conn: sqlite3.Connection, *, group_id: str = DEFAULT_GROUP,
                 list_limit: int = DEFAULT_LIST_LIMIT,
                 clock: Callable[[], float] = time.time) -> None:
        self.conn = conn
        self.group_id = str(group_id or DEFAULT_GROUP)
        self.list_limit = max(1, int(list_limit))
        #: ``None`` is "use the real clock": a caller may pass an unset override.
        self.clock = clock or time.time

    def add(self, text: Any, *, added_by: str = "", home_id: str = "",
            group_id: str = "", now: float | None = None) -> ShoppingItem:
        """Добавить пункт; пустой и слишком длинный — отказ, а не строка."""
        item_text = " ".join(str(text or "").split())
        if not item_text:
            raise ShoppingError("an empty item is not added")
        if len(item_text) > MAX_ITEM_LENGTH:
            raise ShoppingError(
                f"the item is longer than {MAX_ITEM_LENGTH} characters")
        moment = float(self.clock() if now is None else now)
        item = ShoppingItem(group_id=str(group_id or self.group_id), text=item_text,
                            added_by=str(added_by or ""), home_id=str(home_id or ""),
                            created_at=moment)
        self.conn.execute(
            f"INSERT INTO shopping_items({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?)",
            # Пустой автор — это NULL, а не строка "": иначе внешний ключ на
            # ``persons`` отверг бы пункт, добавленный без узнанного человека.
            (item.item_id, item.group_id, item.text, str(item.status),
             item.added_by or None, item.home_id, item.created_at, item.done_at))
        self.conn.commit()
        return item

    def get(self, item_id: str) -> ShoppingItem | None:
        row = self.conn.execute(
            f"SELECT {_COLUMNS} FROM shopping_items WHERE item_id=?",
            (str(item_id or ""),)).fetchone()
        return _row(row) if row is not None else None

    def open_items(self, *, group_id: str = "", limit: int | None = None
                   ) -> tuple[list[ShoppingItem], int]:
        """Открытые пункты (старые сверху) и сколько их всего."""
        group = str(group_id or self.group_id)
        cap = self.list_limit if limit is None else max(1, int(limit))
        total = int(self.conn.execute(
            "SELECT COUNT(*) FROM shopping_items WHERE group_id=? AND status='open'",
            (group,)).fetchone()[0])
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM shopping_items"
            " WHERE group_id=? AND status='open' ORDER BY created_at, item_id LIMIT ?",
            (group, cap)).fetchall()
        return [_row(row) for row in rows], total

    def done_items(self, *, group_id: str = "", limit: int = 20) -> list[ShoppingItem]:
        group = str(group_id or self.group_id)
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM shopping_items"
            " WHERE group_id=? AND status='done' ORDER BY done_at DESC, item_id LIMIT ?",
            (group, max(1, int(limit)))).fetchall()
        return [_row(row) for row in rows]

    def find(self, text: Any, *, group_id: str = "") -> list[ShoppingItem]:
        """Пункты, которые совпадают с названными словами (по границам слов)."""
        wanted = " ".join(str(text or "").split()).casefold()
        if not wanted:
            return []
        group = str(group_id or self.group_id)
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM shopping_items"
            " WHERE group_id=? AND status='open' ORDER BY created_at, item_id",
            (group,)).fetchall()
        pattern = re.compile(rf"(?<!\w){re.escape(wanted)}(?!\w)")
        hits = [item for item in map(_row, rows) if pattern.search(item.text.casefold())]
        if hits:
            return hits
        words = wanted.split()
        if len(words) == 1:
            return []
        # «купил молоко и хлеб» — каждый названный пункт должен найтись.
        return [item for item in map(_row, rows)
                if any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", item.text.casefold())
                       for word in words)]

    def mark_done(self, item_id: str, *, now: float | None = None) -> bool:
        moment = float(self.clock() if now is None else now)
        cursor = self.conn.execute(
            "UPDATE shopping_items SET status='done', done_at=?"
            " WHERE item_id=? AND status='open'",
            (moment, str(item_id or "")))
        self.conn.commit()
        return bool(cursor.rowcount)

    def remove(self, item_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM shopping_items WHERE item_id=?",
                                   (str(item_id or ""),))
        self.conn.commit()
        return bool(cursor.rowcount)

    def clear(self, *, group_id: str = "", only_done: bool = False) -> int:
        """Убрать пункты группы: ``only_done`` — только вычеркнутые."""
        group = str(group_id or self.group_id)
        sql = "DELETE FROM shopping_items WHERE group_id=?"
        params: list[Any] = [group]
        if only_done:
            sql += " AND status='done'"
        cursor = self.conn.execute(sql, tuple(params))
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def counts(self, *, group_id: str = "") -> dict[str, int]:
        group = str(group_id or self.group_id)
        result = {str(status): 0 for status in ItemStatus}
        for status, count in self.conn.execute(
                "SELECT status, COUNT(*) FROM shopping_items WHERE group_id=? GROUP BY status",
                (group,)):
            result[str(status)] = int(count)
        return result


__all__ = [
    "DEFAULT_GROUP",
    "DEFAULT_LIST_LIMIT",
    "ItemStatus",
    "LANGUAGES",
    "MAX_ITEM_LENGTH",
    "ShoppingAction",
    "ShoppingError",
    "ShoppingItem",
    "ShoppingRequest",
    "ShoppingStore",
    "added_line",
    "card_text",
    "card_title",
    "cleared_line",
    "done_line",
    "empty_line",
    "list_line",
    "not_found_line",
    "removed_line",
    "shopping_command",
    "unavailable_line",
]
