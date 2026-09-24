"""P5-23 (F-610): общий список покупок и дел — голос, Telegram и HUD."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config, ShoppingConfig
from hub import app as hub_app
from hub import migrations_runner
from hub import shopping as shopping_mod
from hub.homes import ensure_home
from hub.session import Session
from hub.tools import SERVER_TOOLS, TOOL_FAMILIES, TOOL_NAMES
from hub.utterances import UtteranceMetrics

AMY = "p-amy"


# --- разбор реплики ---------------------------------------------------------


@pytest.mark.parametrize("text,action,item", [
    ("добавь в список покупок молоко", "add", "молоко"),
    ("добавь молоко в список", "add", "молоко"),
    ("запиши в список дел полить цветы", "add", "полить цветы"),
    ("add milk to the shopping list", "add", "milk"),
    ("añade leche a la lista de la compra", "add", "leche"),
    ("что в списке покупок?", "list", ""),
    ("покажи список дел", "list", ""),
    ("what is on the list?", "list", ""),
    ("купил молоко", "done", "молоко"),
    ("отметь хлеб", "done", "хлеб"),
    ("убери молоко из списка", "remove", "молоко"),
    ("очисти список покупок", "clear", ""),
])
def test_the_list_phrases_are_understood(text, action, item):
    request = shopping_mod.shopping_command(text)
    assert request is not None, text
    assert request.action.value == action
    assert request.text == item


@pytest.mark.parametrize("text", [
    "привет, как дела?",
    "включи музыку",
    "добавь в список покупок",       # пункт не назван
    "что нового?",
    "список песен включи",
])
def test_other_phrases_are_not_list_commands(text):
    assert shopping_mod.shopping_command(text) is None


def test_the_tool_is_declared_and_has_a_family():
    assert "shopping_list" in TOOL_NAMES
    assert "shopping_list" in SERVER_TOOLS, "инструмент выполняет хаб, а не комната"
    assert "shopping_list" in TOOL_FAMILIES["lists"]


# --- хранилище --------------------------------------------------------------


def _store(tmp_path, **kwargs):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.commit()
    return conn, shopping_mod.ShoppingStore(conn, **kwargs)


def test_items_are_shared_oldest_first_and_do_not_vanish_when_done(tmp_path):
    conn, store = _store(tmp_path)
    try:
        milk = store.add("молоко", added_by=AMY, home_id="kitchen", now=1_000.0)
        bread = store.add("хлеб", added_by=AMY, home_id="office", now=1_001.0)
        items, total = store.open_items()
        assert [item.text for item in items] == ["молоко", "хлеб"]
        assert total == 2
        assert store.mark_done(milk.item_id) is True
        assert store.mark_done(milk.item_id) is False, "второй раз не вычёркивается"
        items, total = store.open_items()
        assert [item.text for item in items] == ["хлеб"] and total == 1
        assert [item.text for item in store.done_items()] == ["молоко"]
        assert store.get(milk.item_id).done_at == pytest.approx(store.get(milk.item_id).done_at)
        assert bread.item_id != milk.item_id
    finally:
        conn.close()


def test_the_item_is_found_by_whole_words_and_not_by_similarity(tmp_path):
    conn, store = _store(tmp_path)
    try:
        store.add("молоко", now=1_000.0)
        store.add("молоток", now=1_001.0)
        assert [item.text for item in store.find("молоко")] == ["молоко"]
        assert [item.text for item in store.find("молоток")] == ["молоток"]
        assert store.find("моло") == [], "по части слова пункт не угадывается"
        assert store.find("сыр") == []
        assert [item.text for item in store.find("молоко и хлеб")] == ["молоко"]
    finally:
        conn.close()


def test_empty_and_too_long_items_are_refused(tmp_path):
    conn, store = _store(tmp_path)
    try:
        with pytest.raises(shopping_mod.ShoppingError, match="empty"):
            store.add("   ")
        with pytest.raises(shopping_mod.ShoppingError, match="longer"):
            store.add("м" * (shopping_mod.MAX_ITEM_LENGTH + 1))
        assert store.counts()["open"] == 0
    finally:
        conn.close()


def test_clearing_the_list_can_keep_the_ticked_items(tmp_path):
    conn, store = _store(tmp_path)
    try:
        first = store.add("молоко", now=1_000.0)
        store.add("хлеб", now=1_001.0)
        store.mark_done(first.item_id)
        assert store.clear(only_done=True) == 1
        assert [item.text for item in store.open_items()[0]] == ["хлеб"]
        assert store.clear() == 1
        assert store.counts() == {"open": 0, "done": 0}
    finally:
        conn.close()


# --- ход хаба ---------------------------------------------------------------


def _hub(tmp_path, monkeypatch, *, enabled=True, homes=("kitchen", "office"),
         list_limit=15):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    for home in homes:
        ensure_home(conn, home, name=home, tz="UTC")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.commit()
    cfg = Config(homes=[{"home_id": home, "name": home, "tz": "UTC"} for home in homes])
    cfg.server.shopping = ShoppingConfig(enabled=enabled, list_limit=list_limit)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_shopping", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    connection = _connection(cfg, home=homes[0], speaker="Антон")
    other = _connection(cfg, home=homes[1], speaker="Дрю")
    monkeypatch.setattr(hub_app, "_connections", [connection, other])
    return conn, connection, other


def _connection(cfg, *, home, speaker):
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.session = Session(client_id=f"pc-{home}", devices=[], history_turns=4)
    connection.home_id = home
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._speaker_name = speaker
    connection._speaker_role = "owner"
    connection.sample_rate = 16000
    connection.send_json = AsyncMock()
    connection.send_bytes = AsyncMock()
    connection._stream_tts = AsyncMock()
    connection._log_dialog = AsyncMock()
    return connection


def _say(connection, text) -> bool:
    return asyncio.run(connection._shopping_turn(
        text, "ru", None, 100.0, connection.session, 40))


def _said(connection) -> list[str]:
    return [str(call.args[0].get("text") or "") for call in connection.send_json.await_args_list]


def _cards(connection) -> list[dict]:
    return [call.args[0] for call in connection.send_json.await_args_list
            if call.args[0].get("type") == "card"]


def test_an_item_added_in_one_room_is_read_in_another(tmp_path, monkeypatch):
    conn, kitchen, office = _hub(tmp_path, monkeypatch)
    try:
        assert _say(kitchen, "добавь в список покупок молоко") is True
        assert any("молоко" in line for line in _said(kitchen))
        cards = _cards(office), _cards(kitchen)
        assert all(any("молоко" in str(card.get("text")) for card in group)
                   for group in cards), "карточка списка видна в обеих комнатах группы"
        assert _say(office, "что в списке покупок?") is True
        assert any("молоко" in line for line in _said(office))
        assert shopping_mod.ShoppingStore(conn).counts()["open"] == 1
    finally:
        conn.close()


def test_a_bought_item_is_ticked_off_and_a_stranger_is_refused(tmp_path, monkeypatch):
    conn, kitchen, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(kitchen, "добавь в список покупок молоко")
        assert _say(kitchen, "купил молоко") is True
        assert any("Вычеркнула" in line for line in _said(kitchen))
        assert shopping_mod.ShoppingStore(conn).counts() == {"open": 0, "done": 1}
        before = len(_said(kitchen))
        assert _say(kitchen, "купил сыр") is True
        assert any("нет «сыр»" in line for line in _said(kitchen)[before:])
        assert shopping_mod.ShoppingStore(conn).counts() == {"open": 0, "done": 1}, \
            "незнакомый пункт не заводится вычеркиванием"
    finally:
        conn.close()


def test_clearing_the_list_says_how_many_left(tmp_path, monkeypatch):
    conn, kitchen, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(kitchen, "добавь в список покупок молоко")
        _say(kitchen, "добавь в список покупок хлеб")
        assert _say(kitchen, "очисти список покупок") is True
        assert any("очищен" in line.lower() for line in _said(kitchen))
        assert shopping_mod.ShoppingStore(conn).counts() == {"open": 0, "done": 0}
    finally:
        conn.close()


def test_a_plain_phrase_is_left_to_the_rest_of_the_hub(tmp_path, monkeypatch):
    conn, kitchen, _ = _hub(tmp_path, monkeypatch)
    try:
        assert _say(kitchen, "какая сегодня погода?") is False
        assert _say(kitchen, "включи музыку") is False
        assert shopping_mod.ShoppingStore(conn).counts()["open"] == 0
    finally:
        conn.close()


def test_the_flag_turns_the_list_off(tmp_path, monkeypatch):
    conn, kitchen, _ = _hub(tmp_path, monkeypatch, enabled=False)
    try:
        assert _say(kitchen, "добавь в список покупок молоко") is False
        assert shopping_mod.ShoppingStore(conn).counts()["open"] == 0
    finally:
        conn.close()


def test_the_list_limit_is_honest_about_what_it_does_not_show(tmp_path, monkeypatch):
    conn, kitchen, _ = _hub(tmp_path, monkeypatch, list_limit=2)
    try:
        for item in ("молоко", "хлеб", "сыр", "яйца"):
            _say(kitchen, f"добавь в список покупок {item}")
        before = len(_said(kitchen))
        _say(kitchen, "что в списке покупок?")
        answer = _said(kitchen)[before]
        assert "молоко" in answer and "хлеб" in answer
        assert "и ещё 2" in answer, "остаток списка назван, а не спрятан"
    finally:
        conn.close()


def test_the_tool_reads_and_writes_the_same_list(tmp_path, monkeypatch):
    conn, kitchen, _ = _hub(tmp_path, monkeypatch)
    try:
        added = kitchen._shopping_tool({"action": "add", "item": "молоко"})
        assert added["ok"] is True and "молоко" in added["spoken"]
        listed = kitchen._shopping_tool({"action": "list"})
        assert listed["ok"] is True and listed["list"] == ["молоко"]
        assert _say(kitchen, "что в списке покупок?") is True
        assert any("молоко" in line for line in _said(kitchen)), \
            "голос и Telegram читают один и тот же список"
        bad = kitchen._shopping_tool({"action": "sing"})
        assert bad["ok"] is False and "action must be" in bad["error"]
        empty = kitchen._shopping_tool({"action": "add"})
        assert empty["ok"] is False and "item is needed" in empty["error"]
    finally:
        conn.close()


def test_the_health_snapshot_names_the_list(tmp_path, monkeypatch):
    conn, kitchen, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(kitchen, "добавь в список покупок молоко")
        snapshot = hub_app._shopping_snapshot()
        assert snapshot["enabled"] is True
        assert snapshot["counts"]["open"] == 1
        assert snapshot["homes"] == ["kitchen", "office"]
    finally:
        conn.close()
