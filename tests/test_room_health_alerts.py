"""A broken room must retry forever and say so in the notification group.

Владелец 2026-09-23: «оно должно постоянно ретраить и в тг увед слать в группу
уведов если чет не работает» + «там несколько камер». Two halves are tested
here: the client's capture loop reopens a local camera instead of disabling it
for the rest of the process, and the hub turns ``room_health`` into one message
in the notification group (never one per retry).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from common import protocol as proto
from hub import app as hub_app
from hub.presence_alerts import PresenceAlerts

# --- the hub half ----------------------------------------------------------


class FakeProvider:
    """A Telegram stand-in that records the destination of every sentence."""

    ready = True

    def __init__(self):
        self.sent: list[tuple[str, dict]] = []

    async def send_text(self, text, **kwargs):
        self.sent.append((text, kwargs))
        return {'message_id': len(self.sent)}


def alerts(tmp_path, provider, *, group=-1003570242441, private=()):
    return PresenceAlerts(tmp_path / 'alerts', get_provider=lambda: provider, get_room=lambda _: {},
                          owner_id=111, group_id=group,
                          get_private_recipients=lambda: tuple(private))


def test_a_broken_camera_goes_to_the_notification_group_once_per_episode(tmp_path):
    provider = FakeProvider()
    service = alerts(tmp_path, provider)

    async def run():
        first = await service.notify_system('Rowan · buro: the camera is not working.',
                                            key='room-health:buro:camera:failed', cooldown_s=900.0)
        again = await service.notify_system('Rowan · buro: the camera is not working.',
                                            key='room-health:buro:camera:failed', cooldown_s=900.0)
        assert first is True and again is False, 'ретраи клиента не должны спамить группу'
        # A recovery is worth a line while the failure is still fresh…
        assert service.system_notice_sent('room-health:buro:camera:failed', within_s=900.0)
        # …а после долгой паузы о новой поломке сообщат снова.
        assert not service.system_notice_sent('room-health:buro:camera:failed', within_s=-1.0)

    asyncio.run(run())
    assert len(provider.sent) == 1, 'повторная поломка внутри кулдауна молчит'
    assert provider.sent[0][1] == {'group_chat_id': -1003570242441}


def test_without_a_group_the_notice_goes_to_every_private_chat_with_access(tmp_path):
    provider = FakeProvider()
    service = alerts(tmp_path, provider, group=None, private=(111, 222))

    async def run():
        assert await service.notify_system('Rowan · buro: the camera is not working.') is True

    asyncio.run(run())
    assert [target for _, target in provider.sent] == [
        {'private_reply_to_user_id': 111}, {'private_reply_to_user_id': 222}]


def test_a_missing_telegram_only_logs_and_never_pretends_it_was_sent(tmp_path):
    service = alerts(tmp_path, None)

    async def run():
        assert await service.notify_system('Rowan · buro: the camera is not working.') is False

    asyncio.run(run())


def test_the_hub_turns_room_health_into_one_sentence_for_the_group(tmp_path, monkeypatch):
    provider = FakeProvider()
    service = alerts(tmp_path, provider)
    monkeypatch.setattr(hub_app, '_presence_alerts', service)
    connection = hub_app.Connection(SimpleNamespace(client=None), SimpleNamespace())
    connection.workplace_name = 'buro'
    connection.home_id = 'livingroom'

    async def run():
        await connection._on_room_health({'kind': 'camera', 'ok': False, 'camera': 'usb:0',
                                          'detail': 'the camera stopped delivering frames'})
        await connection._on_room_health({'kind': 'camera', 'ok': True,
                                          'detail': 'the camera is delivering frames again'})

    asyncio.run(run())
    assert len(provider.sent) == 2
    broken, recovered = provider.sent[0][0], provider.sent[1][0]
    assert 'buro' in broken and 'camera is not working' in broken and 'usb:0' in broken
    assert 'retrying' in broken
    assert 'working again' in recovered
    assert all(target == {'group_chat_id': -1003570242441} for _, target in provider.sent)


def test_a_recovered_camera_leaves_the_next_failure_free_to_be_reported(tmp_path, monkeypatch):
    provider = FakeProvider()
    service = alerts(tmp_path, provider)
    monkeypatch.setattr(hub_app, '_presence_alerts', service)
    connection = hub_app.Connection(SimpleNamespace(client=None), SimpleNamespace())
    connection.workplace_name = 'buro'

    async def run():
        # Flapping: down, up, down again, up again — one episode, two lines.
        for ok in (False, True, False, True):
            await connection._on_room_health({'kind': 'camera', 'ok': ok, 'detail': 'frames'})

    asyncio.run(run())
    texts = [text for text, _ in provider.sent]
    assert len(texts) == 2, texts
    assert texts[0].endswith('voice still works in that room.')
    assert 'working again' in texts[1]


def test_a_recovery_alone_is_not_worth_a_message(tmp_path, monkeypatch):
    """Nothing was said about a failure, so nothing has to be taken back."""
    provider = FakeProvider()
    service = alerts(tmp_path, provider)
    monkeypatch.setattr(hub_app, '_presence_alerts', service)
    connection = hub_app.Connection(SimpleNamespace(client=None), SimpleNamespace())
    connection.workplace_name = 'buro'

    async def run():
        await connection._on_room_health({'kind': 'camera', 'ok': True, 'detail': 'frames'})

    asyncio.run(run())
    assert provider.sent == []


# --- the client half -------------------------------------------------------


def bare_service(camera_module):
    """A CameraService without the heavy parts: only what the loop touches."""
    service = camera_module.CameraService.__new__(camera_module.CameraService)
    service.index = 0
    service.stream_url = ''
    service._stop_event = camera_module.threading.Event()
    service._warned = False
    service._enabled = True
    service._send_json = None
    service._loop = None
    return service


def test_a_local_camera_is_reopened_instead_of_disabled_for_good(monkeypatch):
    """The old behaviour ended the capture thread; the room stayed blind."""
    from client import camera as camera_module

    monkeypatch.setattr(camera_module, 'CAMERA_RETRY_MIN_S', 0.01)
    attempts: list[int] = []

    class Service(camera_module.CameraService):
        def _open_capture(self, cv2):
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:  # the other program still holds the device
                raise camera_module.CameraUnavailable('busy')
            return object()

    service = Service.__new__(Service)
    service.index, service.stream_url = 0, ''
    service._stop_event = camera_module.threading.Event()
    assert service._reconnect_capture(None) is not None
    assert len(attempts) == 3, 'устройство должно открываться заново, а не выключаться навсегда'


def test_retrying_stops_the_moment_the_client_shuts_down():
    from client import camera as camera_module

    service = bare_service(camera_module)
    service._stop_event.set()
    assert service._reconnect_capture(None) is None


def test_the_client_tells_the_hub_that_the_camera_is_gone_and_back():
    from client import camera as camera_module

    service = bare_service(camera_module)
    frames: list[dict] = []

    async def run():
        service._loop = asyncio.get_running_loop()

        async def send_json(payload):
            frames.append(payload)

        service._send_json = send_json
        service._report_health(False, 'the camera stopped delivering frames')
        service._report_health(True, 'the camera is delivering frames again')
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert [frame['type'] for frame in frames] == [proto.MSG_ROOM_HEALTH] * 2
    assert [frame['ok'] for frame in frames] == [False, True]
    assert frames[0]['kind'] == 'camera' and frames[0]['camera'] == 'usb:0'
    assert 'stopped delivering' in frames[0]['detail']
