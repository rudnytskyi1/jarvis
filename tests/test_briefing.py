"""P3-28 (F-420): утренний брифинг — факты из источников, слова от модели."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from common.config import Config
from hub import app as hub_app
from hub import briefing
from hub.briefing import (
    BriefingData,
    BriefingSection,
    BriefingSettings,
    BriefingUnavailable,
    MorningBriefingTask,
    OnceADay,
    ReminderSource,
    SectionKind,
    SourceUnavailable,
    briefing_messages,
    clip_sentences,
    collect_sections,
    facts_text,
    llm_briefing,
    morning_due,
    parse_clock,
)
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.presence_state import PresenceEvent, PresenceLog
from hub.reminders import ReminderStore, TriggerKind

CHICAGO = "America/Chicago"
#: 12:30 UTC — 07:30 в Чикаго (летом), то есть ровно час брифинга.
NOW = datetime(2026, 9, 21, 12, 30, tzinfo=UTC)


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-drew', 'Drew')")
    conn.commit()
    yield conn
    conn.close()


class _Source:
    def __init__(self, kind, lines=(), *, reason="", boom=None):
        self.kind, self.lines, self.reason, self.boom = kind, list(lines), reason, boom
        self.calls: list[dict] = []

    async def collect(self, *, person_id, home_id, moment, language):
        self.calls.append({"person_id": person_id, "home_id": home_id,
                           "moment": moment, "language": language})
        if self.boom is not None:
            raise self.boom
        if self.reason:
            raise SourceUnavailable(self.reason)
        return BriefingSection(kind=self.kind, ok=True, lines=self.lines)


def _data(**kwargs) -> BriefingData:
    base = {"home_id": "livingroom", "person_id": "p-anton", "person": "Anton",
            "language": "ru", "moment": NOW, "reason": "time",
            "sections": [BriefingSection(kind=SectionKind.REMINDERS, ok=True,
                                         lines=["напоминание «сдать лабу» — сегодня в 18:30"])]}
    base.update(kwargs)
    return BriefingData(**base)


# --- настройки и часы -------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [("07:30", (7, 30)), ("7:05", (7, 5)),
                                          ("23:59", (23, 59)), ("00:00", (0, 0))])
def test_a_clock_is_read_as_the_home_writes_it(raw, expected):
    value = parse_clock(raw)
    assert (value.hour, value.minute) == expected


@pytest.mark.parametrize("raw", ["25:00", "07:60", "утро", "", "7", "07:30:00"])
def test_a_broken_clock_is_a_config_error(raw):
    with pytest.raises(briefing.BriefingError):
        parse_clock(raw)
    with pytest.raises(ValidationError):
        BriefingSettings(time=raw)


def test_the_briefing_is_off_until_the_owner_turns_it_on():
    assert BriefingSettings().enabled is False
    assert Config().server.briefing.enabled is False


def test_the_morning_window_must_not_be_empty():
    with pytest.raises(ValidationError):
        BriefingSettings(window_start="11:00", window_end="11:00")
    with pytest.raises(ValidationError):
        BriefingSettings(window_start="12:00", window_end="09:00")


def test_the_settings_come_from_the_config_without_inventing_values():
    settings = BriefingSettings.from_config(SimpleNamespace(enabled=True, time="06:15"))
    assert settings.enabled is True and settings.time == "06:15"
    assert settings.window_start == "05:00" and settings.max_chars == 700


def test_the_morning_window_follows_the_rooms_clock():
    settings = BriefingSettings(time="07:30")
    # 12:30 UTC — 07:30 в Чикаго, 19:30 в Киеве.
    chicago = NOW.astimezone(briefing.timezone_of(CHICAGO))
    kyiv = NOW.astimezone(briefing.timezone_of("Europe/Kyiv"))
    assert settings.moment_is_morning(chicago) and settings.hour_has_come(chicago)
    assert not settings.moment_is_morning(kyiv) and settings.hour_has_come(kyiv)


def test_a_foreign_time_zone_does_not_break_the_briefing():
    assert briefing.timezone_of("Nowhere/Nothing") is not None
    assert briefing.timezone_of("") is not None


# --- сбор фактов ------------------------------------------------------------


def test_the_sections_of_the_spec_arrive_in_the_spec_order():
    weather = _Source(SectionKind.WEATHER, ["+7, дождь после обеда"])
    reminders = _Source(SectionKind.REMINDERS, ["напоминание — сегодня в 18:30"])
    sections = asyncio.run(collect_sections([reminders, weather], person_id="p-anton",
                                            home_id="livingroom", moment=NOW, language="ru"))
    assert [item.kind for item in sections] == [SectionKind(kind) for kind in briefing.KIND_ORDER]
    assert sections[0].lines == ["+7, дождь после обеда"]
    assert sections[3].lines == ["напоминание — сегодня в 18:30"]


def test_a_missing_source_is_said_out_loud_instead_of_disappearing():
    sections = asyncio.run(collect_sections([], home_id="livingroom", moment=NOW, language="ru"))
    by_kind = {item.kind: item for item in sections}
    assert by_kind[SectionKind.WEATHER].ok is False
    assert "источник погоды не подключён" in by_kind[SectionKind.WEATHER].reason
    assert by_kind[SectionKind.DEVICES].ok is False
    assert all(item.lines == [] for item in sections if not item.ok)


def test_a_missing_source_speaks_the_language_of_the_person():
    sections = asyncio.run(collect_sections([], home_id="livingroom", moment=NOW, language="es"))
    weather = next(item for item in sections if item.kind is SectionKind.WEATHER)
    assert "no hay fuente" in weather.reason


def test_a_source_that_refuses_keeps_its_own_reason():
    source = _Source(SectionKind.FIRST_EVENT, reason="Canvas answers slowly today")
    sections = asyncio.run(collect_sections([source], home_id="h", moment=NOW))
    first = next(item for item in sections if item.kind is SectionKind.FIRST_EVENT)
    assert first.ok is False and first.reason == "Canvas answers slowly today"


def test_a_broken_source_does_not_take_the_other_sections_with_it():
    broken = _Source(SectionKind.WEATHER, boom=RuntimeError("the internet is gone"))
    good = _Source(SectionKind.DEADLINES, ["эссе по истории — пятница"])
    sections = asyncio.run(collect_sections([broken, good], home_id="h", moment=NOW,
                                            language="en"))
    by_kind = {item.kind: item for item in sections}
    assert by_kind[SectionKind.WEATHER].ok is False
    assert "unavailable" in by_kind[SectionKind.WEATHER].reason
    assert by_kind[SectionKind.DEADLINES].lines == ["эссе по истории — пятница"]


def test_a_source_that_names_no_known_section_is_not_guessed():
    class _Weird:
        kind = "horoscope"

        async def collect(self, **kwargs):  # pragma: no cover - сюда не дойдёт
            raise AssertionError("an unknown section must never be called")

    sections = asyncio.run(collect_sections([_Weird()], home_id="h", moment=NOW))
    assert len(sections) == len(briefing.KIND_ORDER)


def test_a_section_keeps_only_real_facts():
    section = BriefingSection(kind=SectionKind.DEVICES, ok=True,
                              lines=["  свет в комнате выключен  ", "", "   ", "x" * 500])
    assert section.lines == ["свет в комнате выключен", "x" * 300]
    assert BriefingSection(kind=SectionKind.DEVICES, ok=True).available is False
    assert BriefingSection(kind=SectionKind.DEVICES, ok=True, lines=["свет выключен"]).available is True
    with pytest.raises(ValidationError):
        BriefingSection(kind=SectionKind.DEVICES, temperature=5)


# --- слова ------------------------------------------------------------------


def test_the_facts_alone_read_as_a_briefing():
    text = facts_text(_data(), max_chars=700)
    assert text.startswith("Доброе утро, Anton!")
    assert "напоминание «сдать лабу»" in text


@pytest.mark.parametrize("language,lead", [("ru", "Доброе утро, Anton!"),
                                           ("en", "Good morning, Anton!"),
                                           ("es", "¡Buenos días, Anton!")])
def test_the_greeting_is_the_persons_language(language, lead):
    assert facts_text(_data(language=language)).startswith(lead)


def test_a_briefing_without_a_name_still_greets():
    assert facts_text(_data(person="")).startswith("Доброе утро!")


def test_facts_without_a_model_still_say_what_is_missing():
    data = _data(sections=list(asyncio.run(collect_sections([], home_id="h", moment=NOW,
                                                            language="ru"))))
    text = facts_text(data)
    assert "источник погоды не подключён" in text


def test_the_briefing_never_runs_long():
    data = _data(sections=[BriefingSection(kind=SectionKind.REMINDERS, ok=True,
                                           lines=["факт " + "очень длинный текст " * 5])])
    assert len(facts_text(data, max_chars=120)) <= 121


@pytest.mark.parametrize("text,limit", [("Коротко.", 100), ("Раз. Два.", 100)])
def test_clipping_keeps_short_answers_whole(text, limit):
    assert clip_sentences(text, max_chars=limit) == text


def test_clipping_cuts_on_a_sentence_and_not_mid_word():
    text = "Первое предложение целиком. Второе предложение уже обрезано вот здесь"
    assert clip_sentences(text, max_chars=40) == "Первое предложение целиком."
    knot = "слово " * 20
    assert clip_sentences(knot, max_chars=30).endswith("…")


def test_the_model_is_given_facts_and_forbidden_to_invent():
    messages = briefing_messages(_data())
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    assert "Never invent" in messages[0]["content"]
    assert "Russian" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["person_id"] == "p-anton" and payload["language"] == "ru"
    assert payload["sections"][0]["lines"] == _data().sections[0].lines


class _Llm:
    def __init__(self, answer="Доброе утро, Антон. Сегодня плюс семь и дождь.", *, fail=None):
        self.answer, self.fail, self.seen = answer, fail, []

    async def reply_text(self, messages):
        self.seen.append(messages)
        if self.fail is not None:
            raise self.fail
        return self.answer


def test_the_model_writes_the_words_over_the_facts_it_was_given():
    llm = _Llm()
    text = asyncio.run(llm_briefing(llm, _data(), max_chars=700))
    assert text == llm.answer
    assert "сдать лабу" in llm.seen[0][1]["content"]


@pytest.mark.parametrize("answer", ["", "   ", '{"weather": "+7"}', '{"ok": true}'])
def test_a_data_answer_is_not_spoken_as_a_briefing(answer):
    with pytest.raises(BriefingUnavailable):
        asyncio.run(llm_briefing(_Llm(answer), _data()))


def test_a_broken_model_is_an_honest_refusal_not_a_briefing():
    with pytest.raises(BriefingUnavailable):
        asyncio.run(llm_briefing(_Llm(fail=RuntimeError("no GPU")), _data()))
    with pytest.raises(BriefingUnavailable):
        asyncio.run(llm_briefing(None, _data()))


def test_a_long_model_answer_is_clipped_before_it_is_spoken():
    llm = _Llm("Раз. " * 200)
    text = asyncio.run(llm_briefing(llm, _data(), max_chars=200))
    assert len(text) <= 201 and text.endswith(".")


# --- когда брифинг случается ------------------------------------------------


def test_the_gate_lets_one_briefing_in_per_person_and_day():
    gate = OnceADay()
    assert gate.claim("livingroom", "p-anton", "2026-09-21") is True
    assert gate.claim("livingroom", "p-anton", "2026-09-21") is False
    assert gate.claim("livingroom", "p-drew", "2026-09-21") is True
    assert gate.claim("livingroom", "p-anton", "2026-09-22") is True


def test_the_gate_forgets_the_days_that_are_over():
    gate = OnceADay()
    gate.claim("livingroom", "p-anton", "2026-09-20")
    gate.claim("livingroom", "p-anton", "2026-09-21")
    gate.forget_others("2026-09-21")
    assert gate.seen("livingroom", "p-anton", "2026-09-21") is True
    assert gate.seen("livingroom", "p-anton", "2026-09-20") is False


def test_the_hour_of_the_home_calls_the_briefing():
    due = morning_due(BriefingSettings(time="07:30"), home_id="livingroom", moment=NOW,
                      occupants=["p-anton"], tz=CHICAGO)
    assert [item.person_id for item in due] == ["p-anton"]
    assert due[0].reason == "time" and due[0].day == "2026-09-21"


def test_the_briefing_does_not_come_before_its_hour():
    early = NOW - timedelta(hours=2)  # 05:30 в Чикаго
    assert morning_due(BriefingSettings(time="07:30"), home_id="livingroom", moment=early,
                       occupants=["p-anton"], tz=CHICAGO) == []


def test_a_person_who_is_not_in_the_room_hears_nothing():
    assert morning_due(BriefingSettings(time="07:30"), home_id="livingroom", moment=NOW,
                       occupants=[], tz=CHICAGO) == []


def test_waking_up_earlier_still_brings_the_briefing():
    awake = NOW - timedelta(minutes=90)  # 06:00 в Чикаго
    due = morning_due(BriefingSettings(time="07:30"), home_id="livingroom", moment=awake,
                      occupants=["p-anton"], entries={"p-anton": awake.timestamp()},
                      tz=CHICAGO)
    assert [item.reason for item in due] == ["woke"]
    # Тот же человек, но без входа сегодня, до своего часа молчит.
    assert morning_due(BriefingSettings(time="07:30"), home_id="livingroom", moment=awake,
                       occupants=["p-anton"], entries={}, tz=CHICAGO) == []


def test_the_briefing_window_closes_before_the_afternoon():
    noon = NOW + timedelta(hours=4)  # 11:30 в Чикаго
    assert morning_due(BriefingSettings(time="07:30"), home_id="livingroom", moment=noon,
                       occupants=["p-anton"], tz=CHICAGO) == []


def test_the_gate_keeps_the_briefing_once_even_if_the_pass_repeats():
    gate = OnceADay()
    gate.claim("livingroom", "p-anton", "2026-09-21")
    assert morning_due(BriefingSettings(), home_id="livingroom", moment=NOW,
                       occupants=["p-anton"], gate=gate, tz=CHICAGO) == []


# --- источники --------------------------------------------------------------


def test_the_reminders_of_the_person_are_a_real_section(hub_db):
    store = ReminderStore(hub_db)
    store.add(text="сдать лабу", due_at=datetime(2026, 9, 21, 23, 30, tzinfo=UTC),
              person_id="p-anton", home_id="livingroom")
    store.add(text="", due_at=datetime(2026, 9, 22, 14, 0, tzinfo=UTC),
              person_id="p-anton", home_id="livingroom")
    store.add(text="выключить утюг", person_id="p-anton", home_id="livingroom",
              trigger=TriggerKind.PERSON_ENTERED)
    store.add(text="чужая задача", due_at=datetime(2026, 9, 21, 20, 0, tzinfo=UTC),
              person_id="p-drew", home_id="livingroom")
    section = asyncio.run(ReminderSource(store, tz_of=lambda home: CHICAGO).collect(
        person_id="p-anton", home_id="livingroom", moment=NOW, language="ru"))
    assert section.ok is True
    assert "сдать лабу" in section.lines[0] and "18:30" in section.lines[0]
    assert any("без текста" in line for line in section.lines)
    assert any("когда придёшь домой" in line for line in section.lines)
    assert all("чужая задача" not in line for line in section.lines)


def test_the_reminders_speak_the_persons_language(hub_db):
    store = ReminderStore(hub_db)
    store.add(text="buy bread", due_at=datetime(2026, 9, 21, 23, 30, tzinfo=UTC),
              person_id="p-anton", home_id="livingroom")
    section = asyncio.run(ReminderSource(store, tz_of=lambda home: CHICAGO).collect(
        person_id="p-anton", home_id="livingroom", moment=NOW, language="en"))
    assert section.lines[0].startswith("reminder ")


def test_reminders_without_a_database_are_said_to_be_missing():
    sections = asyncio.run(collect_sections([ReminderSource(None)], person_id="p-anton",
                                            home_id="h", moment=NOW, language="ru"))
    reminders = next(item for item in sections if item.kind is SectionKind.REMINDERS)
    assert reminders.ok is False and "недоступны" in reminders.reason


def test_a_broken_reminder_store_does_not_raise_through_the_briefing():
    class _Broken:
        def pending(self, **kwargs):
            raise RuntimeError("database is locked")

    sections = asyncio.run(collect_sections([ReminderSource(_Broken())], home_id="h",
                                            person_id="p-anton", moment=NOW, language="ru"))
    reminders = next(item for item in sections if item.kind is SectionKind.REMINDERS)
    assert reminders.ok is False and "database is locked" in reminders.reason


# --- задача планировщика ----------------------------------------------------


class _Audit:
    def __init__(self):
        self.rows: list[dict] = []

    def record(self, **row):
        self.rows.append(row)


def _task(*, occupants=None, entries=None, sections=None, speak=None, audit=None,
          settings=None, gate=None, homes=("livingroom",), names=None):
    spoken: list[tuple[str, BriefingData]] = []

    async def default_speak(home_id, data):
        spoken.append((home_id, data))
        return True

    async def default_sections(home_id, person_id, moment):
        return [BriefingSection(kind=SectionKind.REMINDERS, ok=True,
                                lines=["напоминание «сдать лабу» — сегодня в 18:30"])]

    task = MorningBriefingTask(
        settings or BriefingSettings(time="07:30"),
        homes=list(homes),
        occupants=lambda home: (occupants or {"livingroom": ["p-anton"]})[home],
        entries=lambda home, moment: (entries or {}).get(home, {}),
        sections_for=sections or default_sections,
        speak=speak or default_speak,
        gate=gate, audit=audit, tz_of=lambda home: CHICAGO,
        name_of=names or (lambda person_id: "Anton"),
        language=lambda person_id: "ru")
    return task, spoken


def test_the_task_says_the_briefing_once_and_reports_it():
    audit = _Audit()
    task, spoken = _task(audit=audit)
    assert task.name == "briefing.morning" and task.interval_s == 60.0
    report = asyncio.run(task.run(now=NOW))
    assert report == {"homes": 1, "due": 1, "spoken": 1, "missing_client": 0, "failed": 0}
    assert len(spoken) == 1
    home_id, data = spoken[0]
    assert home_id == "livingroom" and data.person == "Anton"
    assert data.language == "ru" and data.reason == "time"
    assert data.facts() == ["напоминание «сдать лабу» — сегодня в 18:30"]
    assert [row["action"] for row in audit.rows] == ["briefing.spoken"]
    # Второй проход того же утра молчит: человек слышал брифинг один раз.
    again = asyncio.run(task.run(now=NOW + timedelta(minutes=5)))
    assert again["due"] == 0 and again["spoken"] == 0 and len(spoken) == 1


def test_a_room_without_a_live_client_keeps_the_briefing_for_later():
    audit = _Audit()
    heard: list[str] = []

    async def silent(home_id, data):
        return bool(heard)

    task, _ = _task(speak=silent, audit=audit)
    first = asyncio.run(task.run(now=NOW))
    assert first["missing_client"] == 1 and first["spoken"] == 0
    assert [row["action"] for row in audit.rows] == ["briefing.waiting_client"]
    # Клиент появился — тот же брифинг всё ещё ждёт человека.
    heard.append("the room is online now")
    second = asyncio.run(task.run(now=NOW + timedelta(minutes=5)))
    assert second["spoken"] == 1


def test_a_broken_briefing_does_not_stop_the_next_person():
    order: list[str] = []

    async def sections(home_id, person_id, moment):
        order.append(person_id)
        if person_id == "p-drew":
            raise RuntimeError("no weather source")
        return [BriefingSection(kind=SectionKind.REMINDERS, ok=True, lines=["факт"])]

    audit = _Audit()
    task, spoken = _task(sections=sections, audit=audit,
                         occupants={"livingroom": ["p-drew", "p-anton"]})
    report = asyncio.run(task.run(now=NOW))
    assert order == ["p-drew", "p-anton"]
    assert report["failed"] == 1 and report["spoken"] == 1
    assert [row["action"] for row in audit.rows] == ["briefing.failed", "briefing.spoken"]
    assert spoken[0][1].person_id == "p-anton"


def test_a_broken_home_does_not_stop_the_other_room():
    def occupants(home_id):
        if home_id == "kyiv":
            raise RuntimeError("this home is broken")
        return ["p-anton"]

    async def sections(home_id, person_id, moment):
        return [BriefingSection(kind=SectionKind.REMINDERS, ok=True, lines=["факт"])]

    async def speak(home_id, data):
        return True

    task = MorningBriefingTask(
        BriefingSettings(time="07:30"), homes=["kyiv", "livingroom"], occupants=occupants,
        entries=lambda home, moment: {}, sections_for=sections, speak=speak,
        tz_of=lambda home: CHICAGO, name_of=lambda person_id: "Anton",
        language=lambda person_id: "ru")
    report = asyncio.run(task.run(now=NOW))
    assert report["homes"] == 1 and report["spoken"] == 1 and report["failed"] == 1


def test_the_task_speaks_nothing_outside_the_morning():
    task, spoken = _task()
    report = asyncio.run(task.run(now=NOW + timedelta(hours=6)))  # 13:30 в Чикаго
    assert report["due"] == 0 and spoken == []


# --- проводка в хабе --------------------------------------------------------


def test_the_hub_schedules_the_morning_briefing(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    cfg = Config(server={"briefing": {"enabled": True}},
                 homes=[{"home_id": "livingroom", "name": "Living room", "tz": CHICAGO}])
    task = hub_app._morning_briefing_task(cfg, conn=hub_db, audit=None)
    assert task is not None and task.name == "briefing.morning"
    assert hub_app._morning_briefing_task(Config(), conn=hub_db, audit=None) is None


def test_the_briefing_switch_and_interval_come_from_the_config(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    fast = hub_app._morning_briefing_task(
        Config(server={"briefing": {"enabled": True, "check_interval_s": 15}}),
        conn=hub_db, audit=None)
    assert fast is not None and fast.interval_s == 15.0
    off = hub_app._morning_briefing_task(Config(server={"briefing": {"enabled": False}}),
                                         conn=hub_db, audit=None)
    assert off is None


def test_a_broken_briefing_config_does_not_stop_the_hub(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    cfg = Config(server={"briefing": {"enabled": True, "time": "07:30"}})
    cfg.server.briefing.time = "утро"  # опечатка, которую конфиг бы отверг
    assert hub_app._morning_briefing_task(cfg, conn=hub_db, audit=None) is None


def test_the_hub_reads_the_clock_of_the_room(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    assert hub_app._home_timezone_of("livingroom") == CHICAGO
    assert hub_app._home_timezone_of("nobody") == ""


def test_the_entry_that_counts_as_waking_up_comes_from_presence(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    log_ = PresenceLog(hub_db)
    log_.record(PresenceEvent(home_id="livingroom", kind="person_entered",
                              person_id="p-anton", ts=NOW.timestamp() - 600))
    log_.record(PresenceEvent(home_id="livingroom", kind="person_entered",
                              person_id="p-drew", ts=NOW.timestamp() - 300))
    log_.record(PresenceEvent(home_id="livingroom", kind="person_left",
                              person_id="p-anton", ts=NOW.timestamp() - 100))
    entries = hub_app._briefing_entries("livingroom", NOW)
    assert sorted(entries) == ["p-anton", "p-drew"]
    assert entries["p-anton"] < entries["p-drew"]


def test_the_briefing_language_follows_the_person(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    hub_db.execute("UPDATE persons SET preferred_language='es' WHERE person_id='p-anton'")
    hub_db.commit()
    assert hub_app._briefing_language("p-anton") == "es"
    assert hub_app._briefing_language("") in {"ru", "en", "es"}


class _Conn:
    """Комната без сети: записывает то, что Rowan произнесла бы."""

    def __init__(self, home_id, *, heard=True):
        self.home_id, self.session, self.said, self.heard = home_id, object(), [], heard
        self.name, self.role = "Anton", "admin"

    async def _say_proactive(self, text, *, name: str = ""):
        self.said.append(text)
        return self.heard


def _live(monkeypatch, *connections):
    monkeypatch.setattr(hub_app, "_connections", list(connections))


def test_the_hub_speaks_the_facts_when_no_model_is_loaded(monkeypatch):
    _live(monkeypatch, _Conn("livingroom"))
    monkeypatch.setattr(hub_app, "_llm", None)
    heard = asyncio.run(hub_app._speak_briefing_in_home("livingroom", _data()))
    said = hub_app._connections[0].said
    assert heard is True and len(said) == 1
    assert said[0].startswith("Доброе утро, Anton!") and "сдать лабу" in said[0]


def test_the_model_rewrites_the_facts_and_the_room_hears_that(monkeypatch):
    _live(monkeypatch, _Conn("livingroom"))
    llm = _Llm("Доброе утро! Сегодня в 18:30 сдать лабу.")
    monkeypatch.setattr(hub_app, "_llm", llm)
    assert asyncio.run(hub_app._speak_briefing_in_home("livingroom", _data())) is True
    assert hub_app._connections[0].said == [llm.answer]


def test_a_model_answer_full_of_data_is_replaced_by_the_facts(monkeypatch):
    _live(monkeypatch, _Conn("livingroom"))
    monkeypatch.setattr(hub_app, "_llm", _Llm('{"sections": []}'))
    assert asyncio.run(hub_app._speak_briefing_in_home("livingroom", _data())) is True
    said = hub_app._connections[0].said[0]
    assert said.startswith("Доброе утро, Anton!") and "{" not in said


def test_a_room_without_a_client_is_not_counted_as_spoken(monkeypatch):
    _live(monkeypatch)
    monkeypatch.setattr(hub_app, "_llm", None)
    assert asyncio.run(hub_app._speak_briefing_in_home("livingroom", _data())) is False


def test_another_rooms_client_is_not_used_for_this_home(monkeypatch):
    _live(monkeypatch, _Conn("kyiv"))
    monkeypatch.setattr(hub_app, "_llm", None)
    assert asyncio.run(hub_app._speak_briefing_in_home("livingroom", _data())) is False
    assert hub_app._connections[0].said == []


def test_a_source_can_be_added_without_touching_the_briefing_module(hub_db, monkeypatch):
    """F-421/F-505 подключают свои разделы; модуль брифинга не меняется."""

    class _Weather:
        kind = SectionKind.WEATHER

        async def collect(self, *, person_id, home_id, moment, language):
            return BriefingSection(kind=self.kind, ok=True, lines=["+7, дождь после обеда"])

    monkeypatch.setattr(hub_app, "_briefing_sources", [])
    hub_app.register_briefing_source(_Weather())
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    cfg = Config(server={"briefing": {"enabled": True}},
                 homes=[{"home_id": "livingroom", "name": "Living room", "tz": CHICAGO}])
    task = hub_app._morning_briefing_task(cfg, conn=hub_db, audit=None)
    sections = asyncio.run(task.sections_for("livingroom", "p-anton", NOW))
    weather = next(item for item in sections if item.kind is SectionKind.WEATHER)
    assert weather.lines == ["+7, дождь после обеда"]
