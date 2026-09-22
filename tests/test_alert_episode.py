"""ТЗ F-702: record until the person leaves, in videos no longer than asked.
The owner wants a visit covered end to end rather than one five-second clip:
the rule keeps sending videos while somebody is in the room, and each video is
``clip_seconds`` long (up to a minute) so Telegram never gets a file it would
refuse. These tests drive the loop with the room's own presence answer.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hub import presence_alerts
from hub.presence_alerts import PresenceAlerts


def provider():
    ack = AsyncMock(return_value={'ok': True, 'chat_id': 123, 'message_id': 5})
    return SimpleNamespace(ready=True,
                           send_image=AsyncMock(return_value={'ok': True, 'chat_id': 123, 'message_id': 4}),
                           send_video=ack)


def make_room(source='a', clip=None):
    clip = clip or AsyncMock(return_value={'data': b'mp4-bytes'})
    return SimpleNamespace(workplace_name='Place ' + source, receiving=False,
                           _task=None, _telegram_control_task=None, _enroll_face_task=None,
                           session=SimpleNamespace(client_id=source),
                           _request_camera_clip=clip)


async def play_episode(tmp_path, monkeypatch, *, people, max_parts=3, clip_seconds=60):
    api, camera = provider(), make_room()
    alerts = PresenceAlerts(tmp_path, lambda: api, lambda source=None: camera, 123, -456)
    # The room's answer to "is somebody still there?" is the one fact the loop
    # uses, so the test states it instead of waiting for real presence frames.
    monkeypatch.setattr(PresenceAlerts, '_people_now',
                        lambda self, source_id: (people, presence_alerts.time.time()))
    monkeypatch.setattr(presence_alerts, 'EPISODE_MAX_PARTS', max_parts)
    alerts.save_rule({'enabled': True, 'media': 'video', 'record_until_clear': True,
                      'clip_seconds': clip_seconds, 'min_stable_s': 0, 'min_frames': 1})
    alerts.start()
    alerts.observe(persons=1, source_id='a', jpeg=b'jpeg')
    await alerts.drain()
    return api, camera, alerts


def test_a_staying_person_is_recorded_video_after_video(tmp_path, monkeypatch):
    async def run():
        api, camera, alerts = await play_episode(tmp_path, monkeypatch, people=1)
        try:
            assert api.send_video.await_count == 3, "one video per part until the ceiling"
            captions = [call.args[2] for call in api.send_video.await_args_list]
            assert 'part' not in captions[0]
            assert 'part 2' in captions[1] and 'part 3' in captions[2]
            delivery = alerts.status()['deliveries'][0]
            assert delivery['status'] == 'sent' and delivery['detail'].startswith('3 videos')
        finally:
            await alerts.close()
    asyncio.run(run())


def test_recording_stops_as_soon_as_the_room_is_empty(tmp_path, monkeypatch):
    async def run():
        api, camera, alerts = await play_episode(tmp_path, monkeypatch, people=0)
        try:
            assert api.send_video.await_count == 1
            assert 'part' not in api.send_video.await_args.args[2]
        finally:
            await alerts.close()
    asyncio.run(run())


def test_each_video_is_as_long_as_the_rule_asks(tmp_path, monkeypatch):
    async def run():
        api, camera, alerts = await play_episode(tmp_path, monkeypatch, people=1,
                                                 max_parts=2, clip_seconds=30)
        try:
            assert api.send_video.await_count == 2
            assert [call.kwargs['seconds'] for call in camera._request_camera_clip.call_args_list] == [30, 30]
        finally:
            await alerts.close()
    asyncio.run(run())


def test_the_room_presence_is_kept_per_room(tmp_path):
    async def run():
        alerts = PresenceAlerts(tmp_path, lambda: provider(), lambda source=None: make_room(), 123, -456)
        alerts.start()
        alerts.observe(persons=2, source_id='a')
        alerts.observe(persons=0, source_id='b')
        await alerts.drain()
        assert alerts._people_now('a')[0] == 2
        assert alerts._people_now('b')[0] == 0
        assert alerts._people_now('unknown')[0] == 0
        await alerts.close()
    asyncio.run(run())


def test_the_clip_is_converted_before_it_reaches_telegram(tmp_path, monkeypatch):
    """ТЗ F-702: what is sent is the version a phone can open."""
    seen = []

    def convert(data):
        seen.append(data)
        return b'converted-for-phones'

    monkeypatch.setattr(presence_alerts, 'phone_ready_mp4', convert)

    async def run():
        api, camera, alerts = await play_episode(tmp_path, monkeypatch, people=0)
        try:
            assert seen == [b'mp4-bytes'], 'the room clip goes in'
            assert api.send_video.await_args.args[0] == b'converted-for-phones'
        finally:
            await alerts.close()
    asyncio.run(run())


def test_without_a_conversion_the_recording_still_arrives(tmp_path, monkeypatch):
    monkeypatch.setattr(presence_alerts, 'phone_ready_mp4', lambda data: None)

    async def run():
        api, camera, alerts = await play_episode(tmp_path, monkeypatch, people=0)
        try:
            assert api.send_video.await_args.args[0] == b'mp4-bytes'
        finally:
            await alerts.close()
    asyncio.run(run())

