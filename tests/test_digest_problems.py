"""Отчёт дня перечисляет то, что НЕ получилось (ТЗ F-704, P4-36).

Деградация хода (опоздавший диаризатор, не узнанный говорящий, оборванный
раунд модели) хранится в самой строке диалога, а отказ или провал действия —
в `audit`. Проверяется, что отчёт называет и то и другое, и что усечённый
перечень примеров честно говорит, сколько строк в него не влезло.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from common.config import Config
from hub import app as hub_app
from hub.digest import collect, digest_text
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.utterances import record_dialog_turns

HOME = "livingroom"
AMY = "p-amy"
#: 21:30 10 мая в Чикаго — отчёт за этот день уже пора собирать.
MOMENT = datetime(2026, 5, 11, 2, 30, tzinfo=UTC)


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.commit()
    yield conn, tmp_path
    conn.close()


def _day_start():
    # Окно дня в Чикаго: 10 мая 00:00 местного = 05:00 UTC.
    return datetime(2026, 5, 10, 5, 0, tzinfo=UTC).timestamp()


def _audit(conn, *, entry_id, action, result, target=""):
    conn.execute(
        "INSERT INTO audit(id, ts, actor_person_id, home_id, action, target, result,"
        " detail_json) VALUES (?,?,?,?,?,?,?,?)",
        (entry_id, _day_start() + 60, AMY, HOME, action, target, result, "{}"))
    conn.commit()


def test_a_degraded_turn_is_stored_with_the_stages_that_were_skipped(hub_db):
    conn, _tmp_path = hub_db
    record_dialog_turns(conn, home_id=HOME, utterance_id="u1", question="включи свет",
                        answer="включаю", ts=_day_start() + 10, person_id=AMY,
                        degraded=("stt", "diarization"))
    rows = conn.execute("SELECT role, degraded FROM dialog_turns ORDER BY role").fetchall()
    assert rows == [("assistant", "stt,diarization"), ("user", "stt,diarization")]
    # Ход без деградаций остаётся пустым: это не «неизвестно», а «полный ход».
    record_dialog_turns(conn, home_id=HOME, utterance_id="u2", question="привет",
                        answer="привет", ts=_day_start() + 20, person_id=AMY)
    assert conn.execute("SELECT degraded FROM dialog_turns WHERE utterance_id='u2'"
                        " AND role='user'").fetchone()[0] == ""


def test_the_hub_writes_the_degradations_it_recorded(hub_db, monkeypatch):
    conn, tmp_path = hub_db
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_hub_db_path", lambda: tmp_path / "hub.db")
    turn = SimpleNamespace(
        utterance_id="u-live",
        home_id=HOME,
        _speaker_name="Антон",
        _degradations=["stt", "llm"],
        cfg=Config(),
    )
    asyncio.run(hub_app.Connection._store_dialog_turns(
        turn, datetime(2026, 5, 10, 6, 0, tzinfo=UTC), "найди ключи", "не помню"))
    row = conn.execute("SELECT degraded FROM dialog_turns WHERE utterance_id='u-live'"
                       " AND role='user'").fetchone()
    assert row is not None and row[0] == "stt,llm"


def test_the_report_names_incomplete_turns_and_refusals(hub_db):
    conn, tmp_path = hub_db
    _audit(conn, entry_id="a1", action="intercom.send", result="denied", target="Макс")
    _audit(conn, entry_id="a2", action="pc_control", result="failed", target="laptop")
    record_dialog_turns(conn, home_id=HOME, utterance_id="u1", question="включи свет",
                        answer="включаю", ts=_day_start() + 10, person_id=AMY,
                        degraded=("stt", "diarization"))
    record_dialog_turns(conn, home_id=HOME, utterance_id="u2", question="привет",
                        answer="привет", ts=_day_start() + 20, person_id=AMY)
    data = collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT)
    assert data.turns == 2
    assert data.degraded_turns == 1
    assert data.degraded == [("stt,diarization", 1)]
    assert data.audit_problems == 2
    text = digest_text(data)
    assert "degraded 1" in text
    assert "Что не удалось (3):" in text
    assert "- Неполные ходы: stt,diarization (1)" in text
    assert "- intercom.send (denied) -> Макс" in text
    assert "- pc_control (failed) -> laptop" in text
    assert "и ещё" not in text


def test_a_truncated_list_says_how_many_were_left_out(hub_db):
    conn, tmp_path = hub_db
    for number in range(3):
        _audit(conn, entry_id=f"a{number}", action="device.set", result="failed",
               target=f"tv{number}")
    data = collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT,
                   history_limit=1)
    text = digest_text(data)
    assert data.audit_problems == 3 and len(data.problems) == 1
    assert "Что не удалось (3):" in text
    assert "- и ещё 2" in text


def test_a_day_of_only_degradations_is_not_a_day_of_luck(hub_db):
    conn, _tmp_path = hub_db
    record_dialog_turns(conn, home_id=HOME, utterance_id="u1", question="что там",
                        answer="не разобрала", ts=_day_start() + 10, person_id=AMY,
                        degraded=("llm",))
    text = digest_text(collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT))
    assert "неудач не записано" not in text
    assert "Что не удалось (1):" in text
    assert "Неполные ходы: llm (1)" in text


def test_an_older_database_names_the_missing_column(hub_db):
    conn, _tmp_path = hub_db
    conn.execute("ALTER TABLE dialog_turns DROP COLUMN degraded")
    conn.commit()
    data = collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT)
    assert "dialog_turns.degraded" in data.missing
    assert "источник недоступен: dialog_turns.degraded" in digest_text(data)
