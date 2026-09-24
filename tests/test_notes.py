"""P5-22 (F-609): голосовые заметки друг другу, TTL 7 дней, выдача по приходу."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config, NotesConfig
from hub import app as hub_app
from hub import migrations_runner
from hub import notes as notes_mod
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.interhome import InterhomeLimiter
from hub.media import MediaStore
from hub.session import Session
from hub.utterances import UtteranceMetrics

RATE = 16000
AMY = "p-amy"
MAX = "p-max"


def _pcm(seconds: float = 2.0) -> bytes:
    return b"\x01\x02" * int(RATE * seconds)


# --- разбор просьбы ---------------------------------------------------------


@pytest.mark.parametrize("text,name", [
    ("оставь Максу голосовое", "Максу"),
    ("Оставь Максу голосовое!", "Максу"),
    ("запиши голосовую заметку для Макса", "Макса"),
    ("запиши заметку от Марии Петровне", "Марии Петровне"),
    ("leave Max a voice note", "Max"),
    ("send a voice message to Max", "Max"),
    ("deja un mensaje de voz a Max", "Max"),
    ("graba una nota de voz para Max", "Max"),
])
def test_a_note_phrase_names_the_recipient(text, name):
    request = notes_mod.note_request(text)
    assert request is not None, text
    assert request.to == name


@pytest.mark.parametrize("text", [
    "привет, как дела?",
    "включи музыку",
    "поставь голосовое сообщение",       # «поставь» — это проиграть, а не оставить
    "покажи камеру",
    "какой у меня голос?",
    "оставь меня в покое",               # нет слова «голосовое»/«заметка»
    "оставь мне голосовое",              # «мне» — не имя получателя
])
def test_other_phrases_are_not_note_requests(text):
    assert notes_mod.note_request(text) is None


# --- хранилище --------------------------------------------------------------


def _store(tmp_path, **kwargs):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "office", name="Кабинет", tz="UTC")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.commit()
    return conn, notes_mod.VoiceNoteStore(conn, **kwargs)


def test_a_note_waits_oldest_first_and_is_played_once(tmp_path):
    conn, store = _store(tmp_path)
    try:
        first = store.add(to_person=MAX, home_id="office", media_ref="media-1",
                          seconds=1.0, from_person=AMY)
        second = store.add(to_person=MAX, home_id="office", media_ref="media-2",
                           seconds=2.0, from_person=AMY)
        assert [note.note_id for note in store.queued("office")] == \
            [first.note_id, second.note_id]
        assert store.mark_played(first.note_id, home_id="office") is True
        assert store.mark_played(first.note_id, home_id="office") is False, \
            "второй раз та же заметка не «проигрывается»"
        assert [note.note_id for note in store.queued("office")] == [second.note_id]
        assert store.get(first.note_id).status is notes_mod.NoteStatus.PLAYED
        assert store.counts("office")["played"] == 1
    finally:
        conn.close()


def test_a_note_lives_seven_days_and_an_overdue_one_is_marked(tmp_path):
    conn, store = _store(tmp_path)
    try:
        note = store.add(to_person=MAX, home_id="office", media_ref="media-1",
                         seconds=1.0, from_person=AMY, now=1_000.0)
        assert note.expires_at == pytest.approx(1_000.0 + 7 * 86400)
        assert store.expire_overdue(now=note.expires_at - 1) == 0
        assert store.queued("office"), "до срока заметка ещё ждёт"
        assert store.expire_overdue(now=note.expires_at) == 1
        assert store.get(note.note_id).status is notes_mod.NoteStatus.EXPIRED
        assert store.queued("office") == []
    finally:
        conn.close()


def test_the_queue_does_not_grow_without_a_limit(tmp_path):
    conn, store = _store(tmp_path, queue_limit=2)
    try:
        for index in range(4):
            store.add(to_person=MAX, home_id="office", media_ref=f"media-{index}",
                      seconds=1.0, from_person=AMY, now=1_000.0 + index)
        waiting = store.queued("office")
        assert len(waiting) == 2, "очередь ограничена, а не растёт вечно"
        assert [note.media_ref for note in waiting] == ["media-2", "media-3"]
        assert store.counts("office")["expired"] == 2
    finally:
        conn.close()


# --- служба записи ----------------------------------------------------------


def _service(tmp_path, conn, settings=None, media=None):
    store = notes_mod.VoiceNoteStore(conn)
    return notes_mod.NoteService(
        store=store,
        media=media if media is not None else MediaStore(
            conn, tmp_path / "data", media_ttl_days=3, clip_ttl_days=7, note_ttl_days=7),
        settings=settings or NotesConfig(), clock=lambda: 1_000.0)


def test_the_note_is_a_real_wav_kept_for_seven_days(tmp_path):
    conn, _ = _store(tmp_path)
    try:
        service = _service(tmp_path, conn)
        note = service.record(from_person=AMY, to_person=MAX, home_id="office",
                              audio_pcm=_pcm(2.0), sample_rate=RATE, origin_home="livingroom")
        row = conn.execute("SELECT path, kind, ts, expires_at FROM media WHERE media_ref=?",
                           (note.media_ref,)).fetchone()
        assert row is not None and row[1] == "note"
        data = Path(row[0]).read_bytes()
        assert data.startswith(b"RIFF"), "заметка — настоящий WAV, а не след в памяти"
        assert notes_mod.wav_seconds(data) == pytest.approx(2.0, abs=0.05)
        from datetime import datetime

        kept = datetime.fromisoformat(str(row[3])).timestamp() - float(row[2])
        assert kept == pytest.approx(7 * 86400, abs=1), \
            "медиаха держит заметку 7 дней (F-609), а не 3, как кадр"
        assert note.seconds == pytest.approx(2.0, abs=0.05)
        assert service.snapshot()["recorded"] == 1
    finally:
        conn.close()


def test_a_short_recording_is_refused_and_a_long_one_is_trimmed(tmp_path):
    conn, _ = _store(tmp_path)
    try:
        service = _service(tmp_path, conn, settings=NotesConfig(max_note_s=5.0))
        with pytest.raises(notes_mod.NoteUnavailable, match="too short"):
            service.record(from_person=AMY, to_person=MAX, home_id="office",
                           audio_pcm=_pcm(0.2), sample_rate=RATE)
        assert notes_mod.VoiceNoteStore(conn).queued("office") == []
        note = service.record(from_person=AMY, to_person=MAX, home_id="office",
                              audio_pcm=_pcm(9.0), sample_rate=RATE)
        assert note.truncated is True
        assert note.seconds == pytest.approx(5.0, abs=0.05)
    finally:
        conn.close()


def test_without_a_media_store_the_note_is_refused_not_pretended(tmp_path):
    conn, _ = _store(tmp_path)
    try:
        service = _service(tmp_path, conn, media=None)
        service.media = None
        with pytest.raises(notes_mod.NoteUnavailable, match="cannot keep"):
            service.record(from_person=AMY, to_person=MAX, home_id="office",
                           audio_pcm=_pcm(2.0), sample_rate=RATE)
        assert notes_mod.VoiceNoteStore(conn).counts("office")["queued"] == 0
    finally:
        conn.close()


# --- доставка ---------------------------------------------------------------


class _Playing:
    """Подставное воспроизведение: помнит, что и где звучало, и умеет молчать."""

    def __init__(self, *, mute: bool = False) -> None:
        self.played: list[tuple[str, str]] = []
        self.mute = mute

    async def __call__(self, home_id: str, note) -> bool:
        if self.mute:
            return False
        self.played.append((home_id, note.note_id))
        return True


def _task(store, play, *, present=None, homes=("office",), audit=None):
    return notes_mod.VoiceNoteDeliveryTask(
        store, play=play, present=present or (lambda home: ()), homes=homes,
        audit=audit, clock=lambda: 2_000.0)


def test_the_note_sounds_when_the_recipient_appears(tmp_path):
    conn, store = _store(tmp_path)
    try:
        note = store.add(to_person=MAX, home_id="office", media_ref="media-1",
                         seconds=3.0, from_person=AMY, now=1_000.0)
        play = _Playing()
        away = asyncio.run(_task(store, play, present=lambda home: (AMY,)).run())
        assert away["played"] == 0 and away["left"] == 1 and play.played == []
        home = asyncio.run(_task(store, play, present=lambda home: (MAX,)).run())
        assert home["played"] == 1 and home["homes"] == {"office": 1}
        assert play.played == [("office", note.note_id)]
        assert store.get(note.note_id).status is notes_mod.NoteStatus.PLAYED
    finally:
        conn.close()


def test_a_room_that_could_not_play_keeps_the_note_for_the_next_pass(tmp_path):
    conn, store = _store(tmp_path)
    try:
        note = store.add(to_person=MAX, home_id="office", media_ref="media-1",
                         seconds=3.0, from_person=AMY, now=1_000.0)
        mute = _Playing(mute=True)
        first = asyncio.run(_task(store, mute, present=lambda home: (MAX,)).run())
        assert first["played"] == 0 and first["left"] == 1
        assert store.get(note.note_id).status is notes_mod.NoteStatus.QUEUED
        mute.mute = False
        second = asyncio.run(_task(store, mute, present=lambda home: (MAX,)).run())
        assert second["played"] == 1
    finally:
        conn.close()


def test_an_overdue_note_is_not_played_at_all(tmp_path):
    conn, store = _store(tmp_path)
    try:
        note = store.add(to_person=MAX, home_id="office", media_ref="media-1",
                         seconds=3.0, from_person=AMY, now=1_000.0)
        play = _Playing()
        task = _task(store, play, present=lambda home: (MAX,))
        task.clock = lambda: note.expires_at + 1
        report = asyncio.run(task.run())
        assert report["expired"] == 1 and report["played"] == 0
        assert play.played == [], "просроченная заметка не звучит"
        assert store.get(note.note_id).status is notes_mod.NoteStatus.EXPIRED
    finally:
        conn.close()


def test_a_broken_room_does_not_stop_the_others(tmp_path):
    conn, store = _store(tmp_path)
    try:
        ensure_home(conn, "livingroom", name="Гостиная", tz="UTC")
        store.add(to_person=MAX, home_id="office", media_ref="media-1", seconds=1.0,
                  from_person=AMY, now=1_000.0)
        store.add(to_person=MAX, home_id="livingroom", media_ref="media-2", seconds=1.0,
                  from_person=AMY, now=1_000.0)
        play = _Playing()

        def present(home):
            if home == "livingroom":
                raise RuntimeError("presence is broken here")
            return (MAX,)

        report = asyncio.run(_task(store, play, present=present,
                                   homes=("livingroom", "office")).run())
        assert report["played"] == 1 and play.played == [("office", play.played[0][1])]
    finally:
        conn.close()


def test_the_delivery_leaves_an_audit_row(tmp_path):
    conn, store = _store(tmp_path)
    try:
        from hub.audit import AuditLog

        audit = AuditLog(conn)
        note = store.add(to_person=MAX, home_id="office", media_ref="media-1",
                         seconds=1.0, from_person=AMY, now=1_000.0)
        asyncio.run(_task(store, _Playing(), present=lambda home: (MAX,),
                          audit=audit).run())
        row = conn.execute("SELECT action, actor_person_id, target FROM audit"
                           " WHERE action='note.play'").fetchone()
        assert row == ("note.play", AMY, MAX)
        assert store.get(note.note_id).status is notes_mod.NoteStatus.PLAYED
    finally:
        conn.close()


# --- ход хаба ---------------------------------------------------------------


def _hub(tmp_path, monkeypatch, *, enabled=True, speaker="Антон", contacts=True):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Гостиная", tz="UTC")
    ensure_home(conn, "office", name="Кабинет", tz="UTC")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Максим"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (MAX, "office", "user"))
    conn.commit()
    if contacts:
        store = ContactStore(conn)
        store.invite(AMY, MAX)
        store.confirm(MAX, AMY)
        monkeypatch.setattr(hub_app, "_contacts", store)
    else:
        monkeypatch.setattr(hub_app, "_contacts", None)
    cfg = Config()
    cfg.server.notes = NotesConfig(enabled=enabled)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_interhome_limits", InterhomeLimiter(enabled=False))
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    media = MediaStore(conn, tmp_path / "data", media_ttl_days=3, clip_ttl_days=7,
                       note_ttl_days=7)
    monkeypatch.setattr(hub_app, "_media", media)
    service = notes_mod.NoteService(store=notes_mod.VoiceNoteStore(conn), media=media,
                                    settings=cfg.server.notes)
    monkeypatch.setattr(hub_app, "_notes_service_cache", service)
    connection = _connection(cfg, home="livingroom", speaker=speaker)
    monkeypatch.setattr(hub_app, "_connections", [connection])
    return conn, connection, service


def _connection(cfg, *, home, speaker):
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.session = Session(client_id=f"pc-{home}", devices=[], history_turns=4)
    connection.home_id = home
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._speaker_name = speaker
    connection._speaker_role = "owner"
    connection.sample_rate = RATE
    connection.send_json = AsyncMock()
    connection.send_bytes = AsyncMock()
    connection._stream_tts = AsyncMock()
    connection._log_dialog = AsyncMock()
    return connection


def _say(connection, text, *, pcm=b"") -> bool:
    return asyncio.run(connection._notes_turn(
        text, "ru", None, 100.0, connection.session, 40, pcm))


def _said(connection) -> list[str]:
    return [str(call.args[0].get("text") or "") for call in connection.send_json.await_args_list]


def test_the_turn_arms_then_records_the_next_phrase(tmp_path, monkeypatch):
    conn, connection, service = _hub(tmp_path, monkeypatch)
    try:
        assert _say(connection, "оставь Максиму голосовое") is True
        assert any("Записываю" in line for line in _said(connection))
        assert notes_mod.VoiceNoteStore(conn).queued("office") == [], \
            "сама просьба в заметку не попадает"
        before = len(_said(connection))
        assert _say(connection, "я буду дома после восьми", pcm=_pcm(3.0)) is True
        lines = _said(connection)[before:]
        assert any("Записала" in line and "7" in line for line in lines)
        waiting = notes_mod.VoiceNoteStore(conn).queued("office")
        assert len(waiting) == 1
        assert waiting[0].from_person == AMY and waiting[0].to_person == MAX
        assert waiting[0].origin_home == "livingroom"
        assert service.snapshot()["recorded"] == 1
    finally:
        conn.close()


def test_without_contact_consent_the_note_is_refused(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch, contacts=False)
    try:
        assert _say(connection, "оставь Максиму голосовое") is True
        assert notes_mod.VoiceNoteStore(conn).counts("office")["queued"] == 0
        assert getattr(connection, "_note_arm", 0.0) == 0.0, "окно записи не открылось"
        assert not any("Записываю" in line for line in _said(connection))
    finally:
        conn.close()


def test_an_unknown_person_or_yourself_is_not_a_recipient(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        assert _say(connection, "оставь Кларе голосовое") is True
        assert any("Не знаю человека" in line for line in _said(connection))
        assert _say(connection, "оставь Антону голосовое") is True
        assert any("Это ты" in line for line in _said(connection))
        assert notes_mod.VoiceNoteStore(conn).counts("office")["queued"] == 0
    finally:
        conn.close()


def test_a_short_phrase_keeps_the_window_open(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(connection, "оставь Максиму голосовое")
        assert _say(connection, "угу", pcm=_pcm(0.1)) is True
        assert any("Слишком коротко" in line for line in _said(connection))
        assert getattr(connection, "_note_arm", 0.0) > 0.0
        assert _say(connection, "уже иду домой", pcm=_pcm(2.0)) is True
        assert len(notes_mod.VoiceNoteStore(conn).queued("office")) == 1
    finally:
        conn.close()


def test_another_command_in_the_window_is_not_recorded_as_a_note(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(connection, "оставь Максиму голосовое")
        assert _say(connection, "угадай, кто сказал", pcm=_pcm(2.0)) is False
        assert notes_mod.VoiceNoteStore(conn).counts("office")["queued"] == 0
        assert getattr(connection, "_note_arm", 0.0) == 0.0
    finally:
        conn.close()


def test_a_plain_phrase_is_left_to_the_rest_of_the_hub(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        assert _say(connection, "какая сегодня погода?", pcm=_pcm(1.0)) is False
        assert _say(connection, "включи музыку", pcm=_pcm(1.0)) is False
        assert notes_mod.VoiceNoteStore(conn).counts()["queued"] == 0
    finally:
        conn.close()


def test_the_flag_turns_the_notes_off(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch, enabled=False)
    try:
        assert _say(connection, "оставь Максиму голосовое") is False
        assert notes_mod.VoiceNoteStore(conn).counts()["queued"] == 0
    finally:
        conn.close()


def test_the_health_snapshot_names_the_notes(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(connection, "оставь Максиму голосовое")
        _say(connection, "позвоню вечером", pcm=_pcm(2.0))
        snapshot = hub_app._notes_snapshot()
        assert snapshot["enabled"] is True
        assert snapshot["recorded"] == 1
        assert snapshot["ttl_days"] == 7
        assert snapshot["counts"]["queued"] == 1
    finally:
        conn.close()


def test_the_room_hears_the_real_recording_of_the_note(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(connection, "оставь Максиму голосовое")
        _say(connection, "я буду дома после восьми", pcm=_pcm(3.0))
        note = notes_mod.VoiceNoteStore(conn).queued("office")[0]
        office = SimpleNamespace(
            language="ru", _say_proactive=AsyncMock(return_value=True),
            _send_play_audio=AsyncMock())
        monkeypatch.setattr(hub_app, "_home_connection",
                            lambda home: office if home == "office" else None)
        assert asyncio.run(hub_app._play_note_in_home("office", note)) is True
        played = office._send_play_audio.await_args
        assert played is not None, "запись действительно уходит в комнату"
        pcm, rate = bytes(played.args[0]), int(played.args[1])
        assert rate == RATE
        assert len(pcm) == len(_pcm(3.0)), "звучит настоящая запись человека, а не пересказ"
        # Перед записью комната слышит, от кого она.
        spoken = str(office._say_proactive.await_args.args[0])
        assert "Антон" in spoken
    finally:
        conn.close()


def test_a_note_whose_recording_is_gone_is_not_pretended(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _say(connection, "оставь Максиму голосовое")
        _say(connection, "я буду дома после восьми", pcm=_pcm(2.0))
        note = notes_mod.VoiceNoteStore(conn).queued("office")[0]
        office = SimpleNamespace(language="ru", _say_proactive=AsyncMock(),
                                 _send_play_audio=AsyncMock())
        monkeypatch.setattr(hub_app, "_home_connection", lambda home: office)
        monkeypatch.setattr(hub_app, "_media_bytes", lambda ref: None)
        assert asyncio.run(hub_app._play_note_in_home("office", note)) is False
        assert office._send_play_audio.await_count == 0
        assert notes_mod.VoiceNoteStore(conn).get(note.note_id).status is \
            notes_mod.NoteStatus.QUEUED
    finally:
        conn.close()
