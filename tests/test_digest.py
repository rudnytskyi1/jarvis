"""Дайджест дня читается из настоящих источников (ТЗ F-704, P4-34).

Ни одна проверка здесь не подставляет готовый текст отчёта: строки берутся из
`presence_events`, `audit`, `dialog_turns` и журнала бюджета, а затем
проверяется, что отчёт говорит ровно то, что лежит в этих таблицах — включая
честное «записей нет» и имя сломанного источника.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from hub import migrations_runner
from hub.api_budget import ApiBudget
from hub.digest import (
    DigestError,
    DigestRuns,
    collect,
    day_window,
    digest_due,
    digest_text,
    timezone_of,
)
from hub.homes import ensure_home

HOME = "livingroom"
OTHER = "kyivflat"
AMY = "p-amy"
MAX = "p-max"
#: Полдень UTC — середина дня и в Чикаго, и в Киеве.
MOMENT = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def hub(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    ensure_home(conn, OTHER, name="Kyiv flat", tz="Europe/Kyiv")
    for person_id, name in ((AMY, "Антон"), (MAX, "Макс")):
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)",
                     (person_id, name))
    conn.commit()
    yield conn, tmp_path
    conn.close()


def _presence(conn, *, event_id, person_id, ts, home=HOME, kind="entered"):
    conn.execute(
        "INSERT INTO presence_events(event_id, home_id, kind, person_id, ts)"
        " VALUES (?,?,?,?,?)", (event_id, home, kind, person_id, ts))
    conn.commit()


def _audit(conn, *, entry_id, action, result, ts, home=HOME, target="", detail="{}"):
    conn.execute(
        "INSERT INTO audit(id, ts, actor_person_id, home_id, action, target, result,"
        " detail_json) VALUES (?,?,?,?,?,?,?,?)",
        (entry_id, ts, AMY, home, action, target, result, detail))
    conn.commit()


def _turn(conn, *, turn_id, role, text, ts, home=HOME, person_id=AMY):
    conn.execute(
        "INSERT INTO dialog_turns(turn_id, home_id, person_id, utterance_id, role, text, ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (turn_id, home, person_id, turn_id.split(":")[0], role, text, ts))
    conn.commit()


def _ledger(tmp_path, rows):
    """Create the real budget journal and put exact requests into it."""
    path = tmp_path / "api_usage.sqlite3"
    ApiBudget(path, monthly_usd=18.0)
    conn = sqlite3.connect(path)
    for row in rows:
        conn.execute("INSERT INTO requests(id, month, amount, model, created_at)"
                     " VALUES (?,?,?,?,?)",
                     (row["id"], row.get("month", "2026-05"), row["amount"],
                      row.get("model", "gpt-5.4-mini"), row.get("created_at", "")))
    conn.commit()
    conn.close()
    return path


def _window_start(tz="America/Chicago"):
    start, _end, _day = day_window(tz, moment=MOMENT)
    return start.timestamp()


def test_the_day_belongs_to_the_home_clock():
    start, end, day = day_window("America/Chicago", moment=MOMENT)
    assert day == "2026-05-10"
    # 10 мая в Чикаго начинается 10 мая в 05:00 UTC.
    assert start == datetime(2026, 5, 10, 5, 0, tzinfo=UTC)
    assert end == datetime(2026, 5, 11, 5, 0, tzinfo=UTC)
    # Тот же момент — тот же день в Киеве, но окно считается по его часам.
    kyiv_start, _kyiv_end, kyiv_day = day_window("Europe/Kyiv", moment=MOMENT)
    assert kyiv_day == "2026-05-10"
    assert kyiv_start == datetime(2026, 5, 9, 21, 0, tzinfo=UTC)
    # Плохая зона не теряет отчёт: она честно превращается в UTC.
    assert str(timezone_of("Nowhere/None")) == "UTC"


def test_the_report_is_read_from_the_real_sources(hub):
    conn, tmp_path = hub
    lo = _window_start()
    _presence(conn, event_id="e1", person_id=MAX, ts=lo + 60)
    _presence(conn, event_id="e2", person_id=MAX, ts=lo + 120)
    _presence(conn, event_id="e3", person_id=AMY, ts=lo + 180)
    _presence(conn, event_id="e4", person_id=AMY, ts=lo + 240, kind="left")
    # Событие чужого дня и чужого дома в отчёт не попадает.
    _presence(conn, event_id="e5", person_id=AMY, ts=lo - 600)
    _presence(conn, event_id="e6", person_id=AMY, ts=lo + 300, home=OTHER)
    _audit(conn, entry_id="a1", action="light.on", result="ok", ts=lo + 10)
    _audit(conn, entry_id="a2", action="intercom.send", result="denied", ts=lo + 20,
           target="Макс")
    _audit(conn, entry_id="a3", action="device.set", result="failed", ts=lo + 30,
           target="tv", detail='{"why": "no adapter"}')
    _turn(conn, turn_id="t1:user", role="user", text="привет", ts=lo + 40)
    _turn(conn, turn_id="t1:assistant", role="assistant", text="привет", ts=lo + 41)
    ledger = _ledger(tmp_path, [
        {"id": "r1", "amount": 2_500_000, "created_at": "2026-05-10T06:00:00+00:00"},
        {"id": "r2", "amount": 500_000, "created_at": "2026-05-10T07:00:00+00:00",
         "model": "gpt-5.4"},
        # Чужой день — в дневную строку не входит, в месячную входит.
        {"id": "r3", "amount": 1_000_000, "created_at": "2026-05-09T07:00:00+00:00"},
        # Строка без момента: день неизвестен, поэтому только месяц.
        {"id": "r4", "amount": 4_000_000, "created_at": ""},
    ])

    data = collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT,
                   ledger_path=ledger)

    assert data.day == "2026-05-10"
    assert data.visitors == [("Макс", 2), ("Антон", 1)]
    assert data.presence_total == 3
    assert (data.audit_total, data.audit_ok, data.audit_denied, data.audit_failed) == (3, 1, 1, 1)
    assert data.turns == 1
    assert len(data.problems) == 2
    assert any(item == "intercom.send (denied) -> Макс" for item in data.problems)
    assert any("device.set (failed) -> tv" in item and "no adapter" in item
               for item in data.problems)
    assert (data.api_requests, data.api_amount_micro) == (2, 3_000_000)
    assert data.api_source == "ledger"
    assert data.api_month_micro == 8_000_000
    assert data.missing == []

    text = digest_text(data)
    assert text.splitlines()[0] == "Rowan — livingroom, 2026-05-10"
    assert "Макс (2)" in text and "Антон (1)" in text
    assert "1 request(s), audit 3 (1 ok, 1 denied, 1 failed)" in text
    assert "Расход API: 2 request(s), $3.0000" in text
    assert "3.0000" in text and "8.0000" in text
    assert "device.set (failed) -> tv" in text


def test_a_day_with_nothing_says_so_and_shows_the_month_spend(hub):
    conn, tmp_path = hub
    ledger = _ledger(tmp_path, [
        {"id": "r1", "amount": 1_500_000, "created_at": "2026-05-01T06:00:00+00:00"},
        {"id": "r2", "amount": 2_500_000, "created_at": ""},
    ])
    data = collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT,
                   ledger_path=ledger)
    text = digest_text(data)
    assert "за сутки записей нет" in text
    # Дневных запросов нет, но месяц истратил 4 доллара — это видно.
    assert data.api_requests == 0 and data.api_month_micro == 4_000_000
    assert "месяц" in text and "4.0000" in text
    assert "неудач не записано" in text


def test_the_api_usage_table_is_read_when_there_is_no_ledger(hub):
    conn, _tmp_path = hub
    conn.execute(
        "INSERT INTO api_usage(request_id, month, model, amount_micro, settled, created_at)"
        " VALUES (?,?,?,?,?,?)",
        ("u1", "2026-05", "gpt-5.4-mini", 750_000, 1, "2026-05-10T06:30:00+00:00"))
    conn.execute(
        "INSERT INTO api_usage(request_id, month, model, amount_micro, settled, created_at)"
        " VALUES (?,?,?,?,?,?)",
        ("u2", "2026-04", "gpt-5.4-mini", 900_000, 1, "2026-04-10T06:30:00+00:00"))
    conn.commit()
    data = collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT)
    assert (data.api_requests, data.api_amount_micro) == (1, 750_000)
    assert data.api_source == "api_usage"
    assert data.api_models == [("gpt-5.4-mini", 1)]


def test_a_broken_source_is_named_and_not_hidden(hub):
    conn, _tmp_path = hub
    conn.execute("DROP TABLE audit")
    conn.commit()
    data = collect(conn, home_id=HOME, tz="America/Chicago", moment=MOMENT)
    assert "audit" in data.missing
    text = digest_text(data)
    assert "источник недоступен: audit" in text


def test_a_report_needs_a_home():
    with pytest.raises(DigestError):
        collect(sqlite3.connect(":memory:"), home_id="", tz="UTC", moment=MOMENT)


def test_digest_due_follows_the_time_of_day_in_the_home():
    settings = SimpleNamespace(enabled=True, time="21:00")
    assert digest_due(settings, moment=datetime(2026, 5, 11, 1, 0, tzinfo=UTC),
                      tz="America/Chicago") is False    # 20:00 в Чикаго — рано
    assert digest_due(settings, moment=datetime(2026, 5, 11, 1, 0, tzinfo=UTC),
                      tz="Europe/Kyiv") is False        # 04:00 в Киеве — тоже рано
    assert digest_due(settings, moment=datetime(2026, 5, 11, 18, 30, tzinfo=UTC),
                      tz="Europe/Kyiv") is True         # 21:30 в Киеве — пора
    assert digest_due(settings, moment=datetime(2026, 5, 11, 18, 30, tzinfo=UTC),
                      tz="America/Chicago") is False    # 13:30 в Чикаго — рано
    assert digest_due(settings, moment=datetime(2026, 5, 11, 2, 5, tzinfo=UTC),
                      tz="America/Chicago") is True     # 21:05 в Чикаго — пора
    assert digest_due(SimpleNamespace(enabled=False, time="21:00"),
                      moment=MOMENT, tz="UTC") is False
    # Плохое время не молчит: отчёт уходит в 21:00 по умолчанию.
    assert digest_due(SimpleNamespace(enabled=True, time="25:99"),
                      moment=datetime(2026, 5, 11, 2, 5, tzinfo=UTC),
                      tz="America/Chicago") is True


def test_one_report_per_day_is_held_by_the_database(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    runs = DigestRuns(conn)
    assert runs.claim(HOME, "2026-05-10") is True
    assert runs.claim(HOME, "2026-05-10") is False
    runs.done(HOME, "2026-05-10", lines=4)
    assert runs.sent(HOME, "2026-05-10") is True
    assert runs.claim(HOME, "2026-05-10") is False
    runs.release(HOME, "2026-05-10", note="delivery failed")
    assert runs.sent(HOME, "2026-05-10") is True  # отправленный отчёт не отменяется
    runs2 = DigestRuns(conn)
    assert runs2.claim(HOME, "2026-05-11") is True
    runs2.release(HOME, "2026-05-11", note="no telegram")
    assert runs2.sent(HOME, "2026-05-11") is False
    assert runs2.claim(HOME, "2026-05-11") is True
    conn.close()


def test_the_ledger_records_the_moment_of_every_request(tmp_path):
    ledger = ApiBudget(tmp_path / "api_usage.sqlite3", monthly_usd=18.0)
    ledger.reserve(1_000, 500)
    conn = sqlite3.connect(ledger.path)
    row = conn.execute("SELECT month, created_at FROM requests").fetchone()
    conn.close()
    assert row[0] == datetime.now(UTC).strftime("%Y-%m")
    # Момент записан в том же виде, в котором его сравнивает дайджест.
    assert row[1].startswith(datetime.now(UTC).strftime("%Y-%m-%dT"))
    assert row[1].endswith("+00:00")
