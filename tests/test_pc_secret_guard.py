"""Пароль не набирают за человека: `pc_control type_text` держит правило F-512.

Массовый аудит 2026-09-23 (AU-09, PROGRESS_AUDIT.md) нашёл сценарий «type my
password into the field», на котором корпус требовал вызов ``pc_control`` с
``type_text`` — то есть выдуманный пароль, — а модель отказывалась его делать.
ТЗ F-512 запрещает ввод паролей и платёжных данных в computer-use
(``common/computer_use.SENSITIVE_RULES``); для ``type_text`` правило то же, и
причина та же: текст приходит из голоса комнаты и остаётся в транскрипте, в
логе и в обучающем архиве, а секрет, набранный один раз, уже утёк.

«Сверни всё» — второй случай того же рода: слова называют ВСЕ окна, а не
приложение, и ПК отвечал «no installed application matches 'all'». Win+D делает
это одним нажатием, поэтому хаб переписывает вызов, а не отправляет его на
отказ (ТЗ F-511, docs/REQUESTS_AUDIT.md).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub.session import Session
from hub.tools import ALL_WINDOWS_WORDS, normalize_pc_control_args, types_a_secret

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "system.md"


@pytest.mark.parametrize("secret", [
    "my password",
    "hunter2 is my passwd",
    "код подтверждения 1234",
    "the card number is 4111",
    "its PIN-code",
])
def test_a_secret_in_the_typed_text_is_recognised(secret):
    assert types_a_secret({"command": "type_text", "value": secret})


@pytest.mark.parametrize("args", [
    {"command": "type_text", "value": "hello world"},
    {"command": "volume_set", "value": "my password"},
    {"command": "clipboard_write", "value": "the dorm address"},
    {"command": "type_text"},
])
def test_ordinary_text_is_left_alone(args):
    assert types_a_secret(args) == ""


def test_a_secret_in_the_other_slot_is_still_seen():
    """Имя приложения и значение приходят в двух слотах — секрет виден в обоих."""
    assert types_a_secret({"command": "type_text", "target": "my password"})


@pytest.mark.parametrize("word", sorted(ALL_WINDOWS_WORDS))
def test_every_window_word_becomes_one_hotkey(word):
    assert normalize_pc_control_args({"command": "minimize_app", "value": word}) == {
        "command": "hotkey", "value": "win+d"}


def test_a_named_application_is_not_rewritten():
    cleaned = normalize_pc_control_args({"command": "minimize_app", "value": "chrome"})
    assert cleaned == {"command": "minimize_app", "value": "chrome"}


def _connection() -> object:
    cfg = Config()
    cfg.server.identity.enabled = False
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.home_id = "livingroom"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn._speaker_name = "Anton"
    conn._speaker_role = "admin"
    conn._speaker_score = 0.9
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={"ok": True, "output": "done"})
    return conn


def test_the_hub_refuses_to_type_a_password():
    conn = _connection()
    result = asyncio.run(conn._execute_tool(
        "pc_control", {"command": "type_text", "value": "my password"}))
    assert result["ok"] is False
    assert "password" in result["error"]
    conn._run_client_action.assert_not_awaited()


def test_the_hub_still_types_ordinary_text():
    conn = _connection()
    result = asyncio.run(conn._execute_tool(
        "pc_control", {"command": "type_text", "value": "hello world"}))
    assert result["ok"] is True
    conn._run_client_action.assert_awaited()


def test_the_prompt_and_the_guard_say_the_same_thing():
    prompt = " ".join(PROMPT.read_text(encoding="utf-8").split()).casefold()
    assert "never type a password" in prompt
    assert "win+d" in prompt
