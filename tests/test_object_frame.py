"""«Где мои ключи?» показывает кадр, а не только обещает его (F-305, P5-04).

Кадр уходит на HUD комнаты (`image_show`, SPEC v1.6) и владельцам дома в
Telegram; если файл истёк или медиах недоступен, ответ остаётся словами, а
обещание кадра — нет.
"""
from __future__ import annotations

import asyncio
import io
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub.homes import ensure_home
from hub.media import MediaStore
from hub.migrations_runner import connect, migrate
from hub.object_memory import ObjectMemoryStore, ObjectSighting

HOME = "livingroom"
MOMENT = datetime(2026, 9, 22, 14, 30, tzinfo=UTC).timestamp()
def _jpeg(width: int = 8, height: int = 6) -> bytes:
    """Настоящий JPEG: хаб меряет его размер из заголовка, а не угадывает."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


JPEG = _jpeg()


@pytest.fixture()
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    yield conn, tmp_path
    conn.close()


def _kept_frame(conn, tmp_path, *, home=HOME) -> str:
    store = MediaStore(conn, tmp_path, media_ttl_days=3, clip_ttl_days=7)
    media_ref, _path = store.save_bytes(home, "frame", JPEG, ts=MOMENT)
    return media_ref


def _sighting(media_ref: str) -> ObjectSighting:
    return ObjectSighting(home_id=HOME, label="keys", ts=MOMENT, zone="стол",
                          bbox=[1, 2, 3, 4], media_ref=media_ref)


def test_a_saved_frame_is_read_from_the_media_table(hub_db, monkeypatch):
    conn, tmp_path = hub_db
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    media_ref = _kept_frame(conn, tmp_path)
    assert hub_app._media_bytes(media_ref) == JPEG
    assert hub_app._media_bytes("media-missing") is None
    assert hub_app._media_bytes("") is None


def test_the_room_shows_the_frame_and_the_owners_get_it_in_telegram(hub_db, monkeypatch):
    conn, tmp_path = hub_db
    media_ref = _kept_frame(conn, tmp_path)
    memory = ObjectMemoryStore(conn)
    memory.record(_sighting(media_ref))

    class _Telegram:
        ready = True

        def __init__(self):
            self.photos = []

        async def send_image(self, data, mime, caption, filename, **kwargs):
            self.photos.append((data, mime, caption, kwargs.get("private_reply_to_user_id")))
            return {"ok": True, "chat_id": kwargs.get("private_reply_to_user_id"),
                    "message_id": 5}

    provider = _Telegram()
    connection = hub_app.Connection(SimpleNamespace(client=None), Config())
    connection.home_id = HOME
    connection._reply_language = "ru"
    connection._speaker_name = "Антон"
    connection._send_image_show = AsyncMock()
    connection._home_timezone = lambda: "America/Chicago"
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_object_memory_store", lambda: memory)
    monkeypatch.setattr(hub_app, "_telegram", provider)
    monkeypatch.setattr(hub_app, "_home_owners",
                        SimpleNamespace(owners=lambda: {HOME: (7, 8)}))
    monkeypatch.setattr(hub_app, "_person_id_of", lambda name: None)

    answer = asyncio.run(connection._where_turn("где мои ключи?", "ru"))
    assert "ключи — стол в" in answer and "Кадр сохранён." in answer
    # HUD комнаты: тот же кадр, что лежит в медиах, и подпись-ответ.
    connection._send_image_show.assert_awaited()
    call = connection._send_image_show.await_args
    assert call.args[0] == JPEG and call.args[1] == 8 and call.args[2] == 6
    assert call.args[3] == answer
    # Telegram: владельцам дома, а не группе.
    assert [photo[3] for photo in provider.photos] == [7, 8]
    assert all(photo[0] == JPEG and photo[1] == "image/jpeg" for photo in provider.photos)


def test_nothing_is_shown_when_the_picture_is_gone(hub_db, monkeypatch):
    conn, tmp_path = hub_db
    memory = ObjectMemoryStore(conn)
    memory.record(_sighting("media-expired"))       # ссылка есть, файла нет
    connection = hub_app.Connection(SimpleNamespace(client=None), Config())
    connection.home_id = HOME
    connection._reply_language = "ru"
    connection._send_image_show = AsyncMock()
    connection._home_timezone = lambda: "America/Chicago"
    provider = SimpleNamespace(ready=True, send_image=AsyncMock())
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_object_memory_store", lambda: memory)
    monkeypatch.setattr(hub_app, "_telegram", provider)
    monkeypatch.setattr(hub_app, "_home_owners", SimpleNamespace(owners=lambda: {}))
    monkeypatch.setattr(hub_app, "_person_id_of", lambda name: None)

    answer = asyncio.run(connection._where_turn("где мои ключи?", "ru"))
    assert "ключи" in answer
    connection._send_image_show.assert_not_awaited()
    provider.send_image.assert_not_awaited()


def test_a_question_about_a_person_is_not_an_object_question(hub_db, monkeypatch):
    conn, _tmp_path = hub_db
    connection = hub_app.Connection(SimpleNamespace(client=None), Config())
    connection.home_id = HOME
    connection._reply_language = "ru"
    connection._send_image_show = AsyncMock()
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_object_memory_store", lambda: ObjectMemoryStore(conn))
    monkeypatch.setattr(hub_app, "_person_id_of", lambda name: "p-max")
    assert asyncio.run(connection._where_turn("где Макс?", "ru")) is None


def test_the_photo_helper_reports_how_many_owners_got_it(hub_db, monkeypatch):
    conn, _tmp_path = hub_db

    class _Telegram:
        ready = True

        def __init__(self):
            self.seen = []

        async def send_image(self, data, mime, caption, filename, **kwargs):
            user = kwargs.get("private_reply_to_user_id")
            self.seen.append(user)
            if user == 8:
                raise RuntimeError("chat not found")
            return {"ok": True, "chat_id": user, "message_id": 1}

    provider = _Telegram()
    monkeypatch.setattr(hub_app, "_telegram", provider)
    monkeypatch.setattr(hub_app, "_home_owners",
                        SimpleNamespace(owners=lambda: {HOME: (7, 8)}))
    sent = asyncio.run(hub_app._send_photo_to_home_owners(HOME, JPEG, "ключи"))
    assert sent == 1 and provider.seen == [7, 8]
    # Бот не готов — ноль отправок, без исключения.
    monkeypatch.setattr(hub_app, "_telegram", SimpleNamespace(ready=False))
    assert asyncio.run(hub_app._send_photo_to_home_owners(HOME, JPEG, "ключи")) == 0


def test_the_indexer_keeps_a_frame_the_answer_can_show(hub_db, monkeypatch):
    """Сквозная связка F-305: индексатор сохранил кадр → ответ его показал."""
    conn, tmp_path = hub_db
    from hub.media import MediaStore
    from hub.object_index import DetectedObject, SceneIndexer

    media = MediaStore(conn, tmp_path, media_ttl_days=3, clip_ttl_days=7)
    memory = ObjectMemoryStore(conn)

    class _Detector:
        def detect(self, frame):
            return [DetectedObject(label="keys", bbox=[1, 2, 3, 4])]

    monkeypatch.setattr(hub_app, "_media_store", lambda cfg=None: media)
    indexer = SceneIndexer(_Detector(), memory)
    media_ref = hub_app._keep_indexed_frame(HOME, JPEG)
    result = indexer.index(HOME, JPEG, ts=MOMENT, media_ref=media_ref)
    assert result.ok is True and result.sightings[0].media_ref == media_ref
    connection = hub_app.Connection(SimpleNamespace(client=None), Config())
    connection.home_id = HOME
    connection._reply_language = "ru"
    connection._send_image_show = AsyncMock()
    connection._home_timezone = lambda: "America/Chicago"
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_object_memory_store", lambda: memory)
    monkeypatch.setattr(hub_app, "_telegram", None)
    monkeypatch.setattr(hub_app, "_person_id_of", lambda name: None)
    answer = asyncio.run(connection._where_turn("где мои ключи?", "ru"))
    assert "Кадр сохранён." in answer
    assert connection._send_image_show.await_args.args[0] == JPEG
