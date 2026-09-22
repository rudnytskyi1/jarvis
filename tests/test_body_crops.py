"""Кропы тела: появление, раз в 2 с, смена ракурса (ТЗ F-202).

Real JPEGs are cut here with the client's own OpenCV, so the size rule ("up to
640 px tall") is checked against bytes rather than against a mock; the hub's
side is checked with the same bytes it would receive from the socket, and the
storage is checked through the real media rule (``data/homes/<home>/media/…``)
and the real ``body_crops`` row.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from client.body_crops import encode_crop
from common.body_crops import (
    ASPECT_TOLERANCE,
    INTERVAL_S,
    MAX_HEIGHT,
    CropSchedule,
    crop_box,
    crop_is_large_enough,
    crop_is_valid,
    jpeg_height,
    scaled_size,
)
from hub import migrations_runner
from hub.body_crops import BodyCropStore
from hub.media import MediaStore

# --- the client cuts it -----------------------------------------------------


def _frame(width=1280, height=720):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = 40
    return frame


def test_the_crop_is_full_height_and_at_most_640_px():
    # ТЗ F-202: JPEG высотой до 640 px.
    assert MAX_HEIGHT == 640
    data = encode_crop(_frame(), (0.1, 0.05, 0.5, 0.99), cv2)
    assert data is not None
    jpeg, width, height = data
    assert height == MAX_HEIGHT
    assert jpeg_height(jpeg) == MAX_HEIGHT
    assert width < height, "the crop keeps the person's aspect ratio"
    assert jpeg.startswith(b"\xff\xd8\xff")


def test_a_short_person_is_not_blown_up():
    data = encode_crop(_frame(640, 400), (0.4, 0.4, 0.6, 0.6), cv2)
    assert data is not None and data[2] <= 400


def test_a_speck_of_noise_is_not_a_crop():
    assert encode_crop(_frame(), (0.5, 0.5, 0.5001, 0.5001), cv2) is None


def test_a_frame_that_is_not_an_image_is_not_a_crop():
    assert encode_crop(None, (0.1, 0.1, 0.5, 0.9), cv2) is None
    assert encode_crop('not a frame', (0.1, 0.1, 0.5, 0.9), cv2) is None
    assert encode_crop(_frame(), (0.1, 0.1, 0.5, 0.9), None) is None


def test_the_box_is_clamped_to_the_frame():
    assert crop_box((-0.5, -0.5, 1.5, 1.5), 100, 50) == (0, 0, 100, 50)
    # 4 % of air around the box on every side, then the pixels are clamped.
    assert crop_box((0.2, 0.2, 0.4, 0.6), 100, 50) == (19, 9, 41, 31)
    assert crop_is_large_enough((0, 0, 30, 30)) is True
    assert crop_is_large_enough((0, 0, 10, 40)) is False


def test_the_size_helper_only_ever_shrinks():
    assert scaled_size(320, 640) == (320, 640)
    assert scaled_size(3200, 6400) == (320, 640)
    assert scaled_size(100, 200, max_height=100) == (50, 100)


# --- when the client sends it ----------------------------------------------


def test_the_schedule_is_appearance_then_every_two_seconds():
    # ТЗ F-202: при появлении, затем раз в 2 с.
    assert INTERVAL_S == 2.0
    schedule = CropSchedule()
    assert schedule.should_send('a:1', 0.5, now=0.0) is True, 'появление'
    assert schedule.should_send('a:1', 0.5, now=1.9) is False
    assert schedule.should_send('a:1', 0.5, now=2.0) is True
    assert schedule.should_send('a:1', 0.5, now=2.5) is False
    assert schedule.should_send('a:1', 0.5, now=4.0) is True


def test_a_person_who_turns_gets_a_crop_immediately():
    """ТЗ F-202: и при смене ракурса (по изменению аспекта bbox)."""
    schedule = CropSchedule()
    assert schedule.should_send('a:1', 0.40, now=0.0) is True
    assert schedule.should_send('a:1', 0.40 + ASPECT_TOLERANCE - 0.01, now=0.2) is False
    assert schedule.should_send('a:1', 0.40 + ASPECT_TOLERANCE, now=0.3) is True
    assert schedule.should_send('a:1', 0.40 + ASPECT_TOLERANCE, now=0.4) is False


def test_each_track_has_its_own_schedule():
    schedule = CropSchedule()
    assert schedule.should_send('a:1', 0.5, now=0.0) is True
    assert schedule.should_send('a:2', 0.5, now=0.0) is True, 'a second person is new'
    schedule.forget('a:1')
    assert schedule.live() == ['a:2']
    assert schedule.should_send('a:1', 0.5, now=0.1) is True, 'a return is an appearance'


# --- the hub keeps it -------------------------------------------------------


def _jpeg(height=640, width=320):
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    return encoded.tobytes()


def test_a_crop_the_hub_would_store_passes_the_check():
    assert crop_is_valid(_jpeg()) == (True, '')


@pytest.mark.parametrize('jpeg', [b'', b'not a jpeg at all'])
def test_a_crop_that_is_not_a_jpeg_is_refused(jpeg):
    ok, reason = crop_is_valid(jpeg)
    assert ok is False and reason


def test_a_full_sized_frame_is_refused_with_a_reason():
    ok, reason = crop_is_valid(_jpeg(height=1080, width=1920))
    assert ok is False and '1080' in reason and '640' in reason


def test_the_hub_stores_the_crop_as_room_media(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    media = MediaStore(conn, tmp_path / 'data', media_ttl_days=3)
    store = BodyCropStore(conn, tmp_path / 'data', media=media)
    saved = store.save(home_id='livingroom', client_id='pc-1', track_id='a:1', jpeg=_jpeg())
    assert saved is not None and saved.track_id == 'a:1' and saved.height == 640
    path = store.path_of(saved)
    assert path.is_file() and 'livingroom' in path.as_posix() and 'media' in path.as_posix()
    assert path.read_bytes().startswith(b'\xff\xd8\xff')
    kinds = {row[0] for row in conn.execute("SELECT kind FROM media")}
    assert kinds == {'crop'}, 'ТЗ F-304 deletes crops by the media row'
    row = conn.execute("SELECT height, ts FROM body_crops").fetchone()
    assert row[0] == 640 and row[1] > 0


def test_a_refused_crop_is_not_stored(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    store = BodyCropStore(conn, tmp_path / 'data', media=MediaStore(conn, tmp_path / 'data'))
    assert store.save(home_id='livingroom', client_id='pc-1', track_id='a:1',
                      jpeg=_jpeg(height=1080)) is None
    assert conn.execute("SELECT COUNT(*) FROM body_crops").fetchone()[0] == 0
    assert not (tmp_path / 'data' / 'homes').exists(), 'nothing is written for a refusal'


def test_the_latest_crops_of_a_track_come_back(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    store = BodyCropStore(conn, tmp_path / 'data')
    for index in range(3):
        store.save(home_id='livingroom', client_id='pc-1', track_id='a:1',
                   jpeg=_jpeg(), ts=100.0 + index)
    store.save(home_id='livingroom', client_id='pc-1', track_id='a:2', jpeg=_jpeg(), ts=200.0)
    latest = store.latest('a:1', limit=2)
    assert [crop.ts for crop in latest] == [102.0, 101.0]
    assert store.latest('a:2')[0].track_id == 'a:2'


# --- the hub receives it ----------------------------------------------------


def test_the_hub_turns_a_crop_header_plus_bytes_into_a_stored_crop(tmp_path, monkeypatch):
    from hub import app as hub_app

    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_body_crops", None)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)

    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = 'pc-1:5100'
    connection.home_id = 'livingroom'
    connection.session = None
    connection._expect_body_crop = None
    connection._on_body_crop_header({'type': 'body_crop', 'track_id': 'a:1',
                                     'kind': 'body', 'w': 320, 'h': 640})
    assert connection._expect_body_crop is not None
    connection._deliver_body_crop(_jpeg())
    assert connection._expect_body_crop is None
    rows = conn.execute("SELECT track_id, height FROM body_crops").fetchall()
    assert rows == [('a:1', 640)]
    assert (tmp_path / 'data' / 'homes' / 'livingroom' / 'media').is_dir()


def test_a_crop_without_a_track_is_ignored(tmp_path):
    from hub import app as hub_app

    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = 'pc-1:5100'
    connection._expect_body_crop = None
    connection._on_body_crop_header({'type': 'body_crop', 'track_id': ''})
    assert connection._expect_body_crop is None


def test_a_binary_frame_that_is_not_a_crop_is_dropped_quietly(tmp_path, monkeypatch):
    from hub import app as hub_app

    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_body_crops", None)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = 'pc-1:5100'
    connection.home_id = 'livingroom'
    connection.session = None
    connection._on_body_crop_header({'track_id': 'a:1'})
    connection._deliver_body_crop(b'not a jpeg')
    assert conn.execute("SELECT COUNT(*) FROM body_crops").fetchone()[0] == 0


def test_the_protocol_declares_the_crop_message():
    from common.protocol import CLIENT_MESSAGE_TYPES, MSG_BODY_CROP, BodyCropHeader

    header = BodyCropHeader(track_id='a:1', kind='body', w=320, h=640)
    assert header.type == MSG_BODY_CROP == 'body_crop'
    assert MSG_BODY_CROP in CLIENT_MESSAGE_TYPES
    assert json.dumps({'type': MSG_BODY_CROP}) is not None
    assert Path(__file__).name  # keep the import of Path meaningful for readers
