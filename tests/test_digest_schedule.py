"""Расписание дайджеста: часы дома, Telegram и «один отчёт в день» (ТЗ F-704).

Проверяется настоящая `DigestTask` с настоящей таблицей `digest_runs`: второй
проход того же дня не отправляет второй отчёт, неудачная отправка день не
съедает, а время берётся по часам ДОМА — у Киева и Чикаго оно разное.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from common.config import Config
from hub import app as hub_app
from hub.digest import DigestData, DigestRuns, DigestTask
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate

HOME = "livingroom"
KYIV = "kyivflat"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    ensure_home(conn, KYIV, name="Kyiv flat", tz="Europe/Kyiv")
    yield conn
    conn.close()


def _settings(**overrides):
    values = dict(enabled=True, time="21:00", channel="telegram", language="ru",
                  history_limit=5, max_chars=3000, check_interval_s=300.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def _collect(*, home_id, tz, moment, language, history_limit, ledger_path=None):
    return DigestData(home_id=home_id, day="2026-05-11", language=language,
                      turns=2, api_month_micro=1_000_000)


class _Audit:
    def __init__(self):
        self.rows = []

    def record(self, **entry):
        self.rows.append(entry)
        return dict(entry)


def _task(conn, homes=((HOME, "America/Chicago"),), *, send, collect=_collect,
          audit=None, **overrides):
    return DigestTask(_settings(**overrides), homes=homes, collect=collect, send=send,
                      runs=DigestRuns(conn), audit=audit)


def test_exactly_one_report_per_day(hub_db):
    sent = []

    async def send(text, *, home_id):
        sent.append((home_id, text))
        return True

    task = _task(hub_db, send=send)
    # 21:30 10 мая в Чикаго — это 02:30 11 мая по UTC; день берётся местный.
    moment = datetime(2026, 5, 11, 2, 30, tzinfo=UTC)
    first = asyncio.run(task.run(now=moment))
    second = asyncio.run(task.run(now=moment))

    assert (first["due"], first["sent"], first["skipped"]) == (1, 1, 0)
    assert (second["due"], second["sent"], second["skipped"]) == (1, 0, 1)
    assert len(sent) == 1
    assert sent[0][0] == HOME and "livingroom" in sent[0][1]
    assert DigestRuns(hub_db).sent(HOME, "2026-05-10") is True
    # Ещё один «процесс» на той же базе — тоже не второй отчёт.
    again = DigestTask(_settings(), homes=((HOME, "America/Chicago"),),
                       collect=_collect, send=send, runs=DigestRuns(hub_db))
    assert asyncio.run(again.run(now=moment))["sent"] == 0
    assert len(sent) == 1


def test_a_failed_delivery_is_retried_the_same_day(hub_db):
    attempts = []

    async def send(text, *, home_id):
        attempts.append(home_id)
        return len(attempts) > 1  # первый раз Telegram недоступен

    task = _task(hub_db, send=send)
    moment = datetime(2026, 5, 11, 2, 30, tzinfo=UTC)
    assert asyncio.run(task.run(now=moment))["failed"] == 1
    assert DigestRuns(hub_db).sent(HOME, "2026-05-10") is False
    second = asyncio.run(task.run(now=moment))
    assert (second["sent"], len(attempts)) == (1, 2)
    assert DigestRuns(hub_db).sent(HOME, "2026-05-10") is True


def test_the_clock_of_the_home_decides_not_the_server(hub_db):
    sent = []

    async def send(text, *, home_id):
        sent.append(home_id)
        return True

    task = _task(hub_db, homes=((HOME, "America/Chicago"), (KYIV, "Europe/Kyiv")),
                 send=send)
    # 18:30 UTC — это 13:30 в Чикаго (рано) и 21:30 в Киеве (пора).
    report = asyncio.run(task.run(now=datetime(2026, 5, 11, 18, 30, tzinfo=UTC)))
    assert report["due"] == 1 and report["sent"] == 1
    assert sent == [KYIV]
    assert report["homes"] == {KYIV: "2026-05-11"}
    # К 02:30 UTC пора уже Чикаго, а киевский отчёт за этот день уже ушёл.
    later = asyncio.run(task.run(now=datetime(2026, 5, 12, 2, 30, tzinfo=UTC)))
    assert later["homes"] == {HOME: "2026-05-11"}
    assert sent == [KYIV, HOME]


def test_one_broken_home_does_not_stop_the_others(hub_db):
    def collect(**kwargs):
        if kwargs["home_id"] == HOME:
            raise RuntimeError("the audit table is gone")
        return _collect(**kwargs)

    sent = []

    async def send(text, *, home_id):
        sent.append(home_id)
        return True

    # Часы домов — настройка: у обоих домов в этом тесте уже прошло 21:00.
    task = _task(hub_db, homes=((KYIV, "UTC"), (HOME, "Europe/London")),
                 send=send, collect=collect)
    report = asyncio.run(task.run(now=datetime(2026, 5, 11, 21, 30, tzinfo=UTC)))
    assert (report["due"], report["sent"], report["failed"]) == (2, 1, 1)
    assert sent == [KYIV]
    # Неудачный дом не оставил за собой занятую строку: попытка будет снова.
    assert DigestRuns(hub_db).sent(HOME, "2026-05-11") is False


def test_the_hub_writes_both_outcomes_to_the_audit(hub_db):
    audit = _Audit()
    answers = [False, True]

    async def send(text, *, home_id):
        return answers.pop(0)

    task = _task(hub_db, send=send, audit=audit)
    moment = datetime(2026, 5, 11, 2, 30, tzinfo=UTC)
    asyncio.run(task.run(now=moment))
    asyncio.run(task.run(now=moment))
    actions = [row["action"] for row in audit.rows]
    assert actions == ["digest.failed", "digest.sent"]
    assert audit.rows[0]["result"] == "failed" and audit.rows[0]["target"] == HOME
    assert audit.rows[1]["result"] == "ok"
    assert audit.rows[1]["detail"]["day"] == "2026-05-10"


class _Owners:
    def __init__(self, mapping):
        self._mapping = mapping

    def owners(self):
        return self._mapping


class _Telegram:
    def __init__(self, *, ready=True, fail=()):
        self.ready = ready
        self.fail = set(fail)
        self.messages = []

    async def send_text(self, text, *, private_reply_to_user_id=None, **_kwargs):
        if private_reply_to_user_id in self.fail:
            raise RuntimeError("telegram refused")
        self.messages.append((private_reply_to_user_id, text))
        return {"ok": True, "chat_id": private_reply_to_user_id, "message_id": 1}


def test_the_report_goes_to_the_owners_of_the_home(hub_db, monkeypatch):
    provider = _Telegram()
    monkeypatch.setattr(hub_app, "_telegram", provider)
    monkeypatch.setattr(hub_app, "_home_owners", _Owners({HOME: (7, 8)}))
    monkeypatch.setattr(hub_app, "_config",
                        Config(server={"digest": {"enabled": True, "channel": "telegram"}}))
    assert asyncio.run(hub_app._send_digest("Rowan — livingroom", home_id=HOME)) is True
    assert provider.messages == [(7, "Rowan — livingroom"), (8, "Rowan — livingroom")]


def test_a_partial_delivery_is_still_todays_one_report(hub_db, monkeypatch):
    provider = _Telegram(fail=(8,))
    monkeypatch.setattr(hub_app, "_telegram", provider)
    monkeypatch.setattr(hub_app, "_home_owners", _Owners({HOME: (7, 8)}))
    monkeypatch.setattr(hub_app, "_config",
                        Config(server={"digest": {"enabled": True, "channel": "telegram"}}))
    # Второй раз первый владелец получил бы отчёт дважды — «ровно один» важнее.
    assert asyncio.run(hub_app._send_digest("отчёт", home_id=HOME)) is True
    assert provider.messages == [(7, "отчёт")]


def test_the_report_waits_when_there_is_no_channel(hub_db, monkeypatch):
    provider = _Telegram(ready=False)
    monkeypatch.setattr(hub_app, "_telegram", provider)
    monkeypatch.setattr(hub_app, "_home_owners", _Owners({HOME: (7,)}))
    monkeypatch.setattr(hub_app, "_config",
                        Config(server={"digest": {"enabled": True, "channel": "telegram"}}))
    assert asyncio.run(hub_app._send_digest("отчёт", home_id=HOME)) is False
    assert provider.messages == []
    # Канал, которого нет, честно говорит «не отправлено», а не «ушло».
    monkeypatch.setattr(hub_app, "_config",
                        Config(server={"digest": {"enabled": True, "channel": "push"}}))
    monkeypatch.setattr(hub_app, "_telegram", _Telegram())
    assert asyncio.run(hub_app._send_digest("отчёт", home_id=HOME)) is False
    # Дом без владельца в Telegram — тоже не отправка.
    monkeypatch.setattr(hub_app, "_config",
                        Config(server={"digest": {"enabled": True, "channel": "telegram"}}))
    monkeypatch.setattr(hub_app, "_home_owners", _Owners({}))
    assert asyncio.run(hub_app._send_digest("отчёт", home_id=HOME)) is False


def test_the_hub_schedules_the_digest_job(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    assert hub_app._digest_task(Config(), audit=None) is None
    cfg = Config(server={"digest": {"enabled": True, "time": "20:15"}},
                 homes=[{"home_id": HOME, "name": "Living room",
                         "tz": "America/Chicago", "telegram_user_id": 7}])
    task = hub_app._digest_task(cfg, audit=None)
    assert task is not None and task.name == "digest.daily"
    assert task.interval_s == 300.0
    assert task.homes == [(HOME, "America/Chicago")]
    scheduler = hub_app._hub_scheduler(cfg, audit=None)
    assert scheduler is not None and scheduler.get("digest.daily") is not None
