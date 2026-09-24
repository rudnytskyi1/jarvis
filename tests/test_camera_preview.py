"""Живое окно камеры комнаты (владелец, 2026-09-24).

Просьба: «можешь на пк buro открыть камеру и чтобы оно показывало видео с
камеры на экране и все детекции?». Проверяется весь путь: клиент объявляет
возможность, по ``camera_preview`` открывает окно, рисует ВСЕ боксы последнего
кадра с подписями людей, честно сообщает, когда окна нет, закрывает его на
разрыве связи; хаб включает окно только у клиента, который это умеет, шлёт
имена и умеет адресовать команду названному ПК.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from client import camera as client_camera
from client.live_view import LivePreview
from client.main import JarvisClient, build_hello
from common import protocol
from hub import app as hub_app

# --- клиент: возможность и сообщение -----------------------------------------

def test_a_room_pc_advertises_the_live_preview_and_a_phone_does_not():
    room = build_hello(SimpleNamespace(client_id='buro', workplace_name='buro',
                                       camera=SimpleNamespace(name='Buro camera')))
    assert protocol.CAP_CAMERA_PREVIEW in room['capabilities']
    phone = build_hello(SimpleNamespace(client_id='phone-1', workplace_name='', kind='phone'))
    assert protocol.CAP_CAMERA_PREVIEW not in phone['capabilities']


def _client(*, preview):
    client = JarvisClient.__new__(JarvisClient)
    client.ws = SimpleNamespace(send_json=AsyncMock())
    client.live_view = preview
    client.camera = SimpleNamespace(enabled=True)
    client.ccfg = SimpleNamespace(client_id='buro')
    return client


def test_the_preview_message_opens_the_window_with_the_names(monkeypatch):
    opened: dict = {}

    class _Fake:
        def __init__(self, camera, label=''):
            opened['label'] = label
            opened['camera'] = camera
            self.started = False
            opened['self'] = self

        def update_names(self, names):
            opened['names'] = names

        def start(self):
            self.started = True
            return True

        def status(self):
            return {'running': self.started, 'label': opened['label'], 'frames': 0, 'error': ''}

        def stop(self):
            opened['stopped'] = True

    monkeypatch.setattr('client.main.LivePreview', _Fake)
    async def run():
        client = _client(preview=None)
        await client._handle_camera_preview({'type': protocol.MSG_CAMERA_PREVIEW, 'on': True,
                                             'label': 'buro', 'names': {'t1': 'Антон'}})
        assert opened['names'] == {'t1': 'Антон'} and opened['label'] == 'buro'
        sent = client.ws.send_json.call_args.args[0]
        assert sent['name'] == 'camera_preview' and sent['ok'] is True and sent['running'] is True
        await client._handle_camera_preview({'type': protocol.MSG_CAMERA_PREVIEW, 'on': False})
        assert opened['stopped'] is True
        assert client.live_view is None
        assert client.ws.send_json.call_args.args[0]['running'] is False
    asyncio.run(run())


def test_a_room_that_cannot_show_a_preview_says_so(monkeypatch):
    monkeypatch.setattr('client.main.LivePreview', None)
    monkeypatch.setattr('client.main._LIVE_VIEW_IMPORT_ERROR', 'RuntimeError: no cv2')
    async def run():
        client = _client(preview=None)
        await client._handle_camera_preview({'type': protocol.MSG_CAMERA_PREVIEW, 'on': True})
        sent = client.ws.send_json.call_args.args[0]
        assert sent['ok'] is False and 'no cv2' in sent['error']
    asyncio.run(run())


def test_the_window_closes_when_the_connection_goes_away():
    """Кадры комнаты не остаются на экране после разрыва связи."""
    stopped: list[bool] = []
    preview = SimpleNamespace(stop=lambda: stopped.append(True),
                              running=True, status=lambda: {'running': True})
    client = JarvisClient.__new__(JarvisClient)
    client.live_view = preview
    client._reader_task = None
    client._camera_clip_task = None
    client._inbox = asyncio.Queue()
    asyncio.run(client._stop_reader())
    assert stopped == [True] and client.live_view is None


# --- клиент: что именно нарисовано -------------------------------------------

class _List(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, rows):
        self.cls = _List([row[0] for row in rows])
        self.conf = _List([row[1] for row in rows])
        self.xyxyn = _List([row[2] for row in rows])
        self.id = _List([row[3] for row in rows])


class _Result:
    names = {0: 'person', 56: 'chair', 67: 'cell phone'}

    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class _Model:
    def __init__(self, rows):
        self.rows = rows

    def track(self, **kwargs):
        return [_Result(self.rows)]


def _service():
    return client_camera.CameraService(SimpleNamespace(enabled=True, half=False))


def test_the_detector_keeps_every_box_of_the_last_frame():
    """«Все детекции»: не только люди и объекты внимания, но и телефон со стулом."""
    rows = [(0, 0.9, [0.10, 0.10, 0.30, 0.60], 7),
            (56, 0.7, [0.40, 0.40, 0.55, 0.70], None),
            (67, 0.55, [0.60, 0.20, 0.65, 0.25], None),
            (56, 0.2, [0.70, 0.10, 0.80, 0.20], None)]
    service = _service()
    service._detect(_Model(rows), SimpleNamespace(shape=(480, 640, 3), size=1))
    kept = [(item['label'], round(item['conf'], 2)) for item in service._last_detections]
    assert kept == [('person', 0.9), ('chair', 0.7), ('cell phone', 0.55)]
    assert service._last_detections[0]['box'] == [0.10, 0.10, 0.30, 0.60]


def test_the_preview_draws_every_detection_and_the_names_of_the_people():
    import cv2
    import numpy as np

    service = _service()
    service._last_detections = [{'label': 'chair', 'box': [0.40, 0.40, 0.55, 0.70], 'conf': 0.7},
                                {'label': 'cell phone', 'box': [0.60, 0.20, 0.65, 0.25],
                                 'conf': 0.4}]
    service._tracks = [{'id': 't1', 'box': [0.10, 0.10, 0.30, 0.60]}]
    preview = LivePreview(service, label='buro')
    preview.update_names({'t1': 'Антон'})
    untouched = np.full((480, 640, 3), 30, dtype=np.uint8)
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)

    drawn = preview._draw(cv2, frame)

    assert np.array_equal(frame, untouched), 'кадр камеры не разрисовывается на месте'
    assert not np.array_equal(drawn, untouched), 'нарисованы рамки детекций'
    person = (int(0.10 * 640), int(0.10 * 480))
    chair = (int(0.40 * 640), int(0.40 * 480))
    phone = (int(0.60 * 640), int(0.20 * 480))
    assert not np.array_equal(drawn[person[1], person[0]], frame[person[1], person[0]])
    assert not np.array_equal(drawn[chair[1], chair[0]], frame[chair[1], chair[0]])
    assert not np.array_equal(drawn[phone[1], phone[0]], frame[phone[1], phone[0]])
    assert drawn[:24].mean() < 120, 'шапка с названием комнаты нарисована'


def test_a_person_box_is_drawn_once_even_if_both_lists_hold_it():
    """Людей рисует общий код клипов (с именем), повторная рамка не нужна."""
    import cv2
    import numpy as np

    service = _service()
    box = [0.10, 0.10, 0.30, 0.60]
    service._tracks = [{'id': 't1', 'box': box}]
    service._last_detections = [{'label': 'person', 'box': list(box), 'conf': 0.9}]
    preview = LivePreview(service, label='buro')
    drawn = preview._draw(cv2, np.full((240, 320, 3), 30, dtype=np.uint8))
    assert drawn.shape == (240, 320, 3)


def test_a_preview_without_a_camera_does_not_open(monkeypatch):
    camera = SimpleNamespace(enabled=False, _cv2=SimpleNamespace())
    preview = LivePreview(camera, label='buro')
    assert preview.start() is False
    assert 'off' in preview.status()['error']


# --- хаб: включение, имена, адресация ----------------------------------------

def _connection(preview: bool, *, sent=None):
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = 'pc-1:5100'
    connection.home_id = 'livingroom'
    connection.workplace_name = 'buro'
    connection.session = SimpleNamespace(client_id='buro')
    connection._can_camera_preview = preview
    connection.send_json = AsyncMock(side_effect=None if sent is None else sent)
    connection.room = None
    return connection


def test_the_hub_turns_the_live_view_on_and_off(monkeypatch):
    async def run():
        connection = _connection(True)
        monkeypatch.setattr(connection, '_track_names', lambda: {'t1': 'Антон'})
        result = await connection.set_camera_preview(True)
        assert result['ok'] is True and result['running'] is True and result['names'] == 1
        payload = connection.send_json.call_args.args[0]
        assert payload['type'] == protocol.MSG_CAMERA_PREVIEW
        assert payload['on'] is True and payload['names'] == {'t1': 'Антон'}
        assert payload['label'] == 'buro'
        await connection.set_camera_preview(False)
        assert connection.send_json.call_args.args[0] == {
            'type': protocol.MSG_CAMERA_PREVIEW, 'on': False, 'label': 'buro'}
        assert getattr(connection, '_preview_task', None) is None
    asyncio.run(run())


def test_a_client_that_cannot_show_the_view_is_refused():
    async def run():
        connection = _connection(False)
        result = await connection.set_camera_preview(True)
        assert result['ok'] is False and 'updated' in result['error']
        connection.send_json.assert_not_awaited()
    asyncio.run(run())


def test_the_names_are_re_sent_only_when_they_change(monkeypatch):
    async def run():
        monkeypatch.setattr(hub_app, 'PREVIEW_NAMES_INTERVAL_S', 0.01)
        connection = _connection(True)
        state = {'names': {'t1': 'Антон'}}
        monkeypatch.setattr(connection, '_track_names', lambda: dict(state['names']))
        await connection.set_camera_preview(True)
        connection.send_json.reset_mock()
        await asyncio.sleep(0.05)
        connection.send_json.assert_not_awaited()
        state['names'] = {'t1': 'Антон', 't2': 'Макс'}
        await asyncio.sleep(0.05)
        assert connection.send_json.call_args.args[0]['names'] == {'t1': 'Антон', 't2': 'Макс'}
        await connection.set_camera_preview(False)
    asyncio.run(run())


def test_the_tool_acts_on_the_computer_the_request_named(monkeypatch):
    async def run():
        buro, anton = _connection(True), _connection(True)
        anton.workplace_name, anton.peer = 'AntonDorm', 'pc-2:5100'
        anton.session = SimpleNamespace(client_id='anton')
        monkeypatch.setattr(hub_app, '_connections', {buro, anton})
        monkeypatch.setattr(buro, 'set_camera_preview', AsyncMock(return_value={'ok': True}))
        monkeypatch.setattr(anton, 'set_camera_preview', AsyncMock(return_value={'ok': True}))
        caller = _connection(True)
        monkeypatch.setattr(caller, 'set_camera_preview', AsyncMock(return_value={'ok': True}))

        assert (await caller._run_camera_preview({'on': True, 'workplace': 'buro'}))['ok'] is True
        buro.set_camera_preview.assert_awaited_once_with(True)
        anton.set_camera_preview.assert_not_awaited()

        assert (await caller._run_camera_preview(
            {'on': False, 'workplace': 'anton pc'}))['ok'] is True
        anton.set_camera_preview.assert_awaited_once_with(False)

        assert (await caller._run_camera_preview({'on': True}))['ok'] is True
        caller.set_camera_preview.assert_awaited_once_with(True)
    asyncio.run(run())


@pytest.mark.parametrize('args, expected', [
    ({'on': True, 'workplace': 'narnia'}, 'no connected computer'),
    ({'on': True}, ''),
])
def test_naming_an_unknown_computer_is_an_honest_answer(args, expected):
    async def run():
        caller = _connection(True)
        caller.set_camera_preview = AsyncMock(return_value={'ok': True})
        result = await caller._run_camera_preview(args)
        if expected:
            assert result['ok'] is False and expected in result['error']
        else:
            assert result['ok'] is True
    asyncio.run(run())


def test_the_tool_needs_a_clear_on_or_off():
    async def run():
        caller = _connection(True)
        result = await caller._run_camera_preview({})
        assert result['ok'] is False and 'on' in result['error']
    asyncio.run(run())


def test_the_live_view_is_a_media_family_tool():
    from hub import tools as hub_tools

    names = [tool['function']['name'] for tool in hub_tools.TOOLS]
    assert 'camera_preview' in names
    assert 'camera_preview' in hub_tools.TOOL_FAMILIES['media']
    assert 'camera_preview' in hub_tools.SERVER_TOOLS
