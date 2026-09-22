"""ТЗ F-606: матрица прав гостя — по строке на тест, как просит ТЗ."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import guest_access
from hub.session import Session


def _connection(*, role: str = guest_access.ROLE_GUEST, name: str = "Кай",
                language: str = "ru"):
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = "livingroom"
    conn.cfg = Config()
    conn.session = Session(client_id="room-pc", devices=[], history_turns=2)
    conn._speaker_role = role
    conn._speaker_name = name
    conn._speaker_score = 0.9
    conn._reply_language = language
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    return conn


def _matrix_row(capability: str) -> bool:
    """The single row of the matrix for one capability (fails on a missing row)."""
    rows = dict(guest_access.MATRIX)
    assert capability in rows, f"the guest matrix has no row for {capability!r}"
    return rows[capability]


# --- строка «время» -------------------------------------------------------


def test_a_guest_may_ask_the_time(monkeypatch):
    assert _matrix_row(guest_access.TIME) is True
    # Время хаб знает сам: права на инструмент здесь нет и быть не должно.
    assert guest_access.TIME not in guest_access.TOOL_CAPABILITY.values()


# --- строка «погода» ------------------------------------------------------


def test_a_guest_may_ask_the_weather(monkeypatch):
    assert _matrix_row(guest_access.WEATHER) is True
    assert guest_access.WEATHER not in guest_access.TOOL_CAPABILITY.values()
    assert guest_access.WEATHER not in guest_access.DEVICE_TOOLS


# --- строка «свет» --------------------------------------------------------


def test_a_guest_may_turn_the_light_on(monkeypatch):
    assert _matrix_row(guest_access.LIGHT) is True
    assert guest_access.tool_denial("set_light", {"device": "Лампа", "state": "on"}) is None


# --- строка «ПК» ----------------------------------------------------------


def test_a_guest_may_not_use_the_room_pc(monkeypatch):
    assert _matrix_row(guest_access.PC) is False
    for command in ("open_app", "volume_set", "unlock", "shutdown"):
        refusal = guest_access.tool_denial("pc_control", {"command": command})
        assert refusal and "компьютер" in refusal
    assert guest_access.tool_denial("run_command", {"command": "dir"})
    assert guest_access.tool_denial("browser_control", {"command": "navigate"})


def test_the_guest_pc_refusal_reaches_the_live_permission_check(monkeypatch):
    conn = _connection()
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    refusal = asyncio.run(conn._permission_check(
        "pc_control", {"command": "volume_set", "value": 30}))
    assert refusal and "компьютер" in refusal, (
        "a registered guest is refused even the shared room basics F-606 calls PC"
    )


# --- строка «память» ------------------------------------------------------


def test_a_guest_may_not_use_the_hub_memory(monkeypatch):
    assert _matrix_row(guest_access.MEMORY) is False
    for tool in ("remember", "forget_fact", "list_memory", "recall_memory", "show_photo"):
        assert guest_access.tool_denial(tool, {"text": "что-нибудь"}), tool


def test_the_unknown_voice_keeps_its_own_notes_but_not_the_room_memory(monkeypatch):
    # ТЗ F-415: гость сохраняет СВОИ факты — «their own facts».
    assert guest_access.tool_denial(
        "remember", {"text": "я люблю чай", "about": "me"},
        role=guest_access.ROLE_UNKNOWN) is None
    assert guest_access.tool_denial(
        "remember", {"text": "в комнате всегда тихо", "scope": "global"},
        role=guest_access.ROLE_UNKNOWN) is not None


# --- строка «устройства с restricted: true» -------------------------------


def test_a_guest_may_not_touch_a_restricted_device(monkeypatch):
    assert _matrix_row(guest_access.RESTRICTED) is False
    refusal = guest_access.tool_denial(
        "set_switch", {"device": "Сервер", "action": "press"}, restricted=True)
    assert refusal and "хозяин" in refusal
    refusal = guest_access.tool_denial(
        "set_light", {"device": "Лампа", "state": "off"}, restricted=True)
    assert refusal and "хозяин" in refusal


# --- строка «интерком» ----------------------------------------------------


def test_a_guest_may_not_use_the_intercom(monkeypatch):
    assert _matrix_row(guest_access.INTERCOM) is False
    refusal = guest_access.intercom_denial(guest=True, stranger=False)
    assert refusal and "гостевом режиме" in refusal
    # Незнакомый голос отбивает сам гейт F-602 своим кодом причины, а не эта
    # строка, поэтому здесь она молчит и решения за гейт не принимает.
    assert guest_access.intercom_denial(guest=False, stranger=True) is None


def test_the_intercom_gate_refuses_a_registered_guest(monkeypatch):
    conn = _connection()
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    refusal = conn._interhome_gate("p-max", "ru")
    assert "гостевом режиме" in refusal


@pytest.mark.parametrize("language,needle", [
    ("ru", "гостевом"),
    ("en", "guest mode"),
    ("es", "invitado"),
])
def test_the_refusal_is_spoken_in_the_persons_language(language, needle):
    assert needle in guest_access.denial(guest_access.PC, language)
    assert needle in guest_access.denial(guest_access.INTERCOM, language)


def test_an_unknown_capability_is_never_allowed():
    assert guest_access.allows("something-new") is False
    assert guest_access.capability_of("unheard-of-tool") is None
