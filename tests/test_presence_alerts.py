import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hub.presence_alerts import PresenceAlerts, quiet_now, validate_rule
from hub.telegram import TelegramError


def provider():
    return SimpleNamespace(ready=True, send_image=AsyncMock(return_value={'ok': True, 'chat_id': 123, 'message_id': 1}),
                           send_video=AsyncMock(return_value={'ok': True, 'chat_id': 123, 'message_id': 2}))


def engine(tmp_path, service=None, room=None):
    return PresenceAlerts(tmp_path, lambda: service, lambda: room, 123, -456)


@pytest.mark.parametrize('patch', [
    {'enabled': 'yes'}, {'chat_id': 999}, {'target': 'all'}, {'destination': 'other'},
    {'target': 'person', 'name': ''}, {'cooldown_s': 0}, {'min_stable_s': float('nan')},
    {'absence_s': False}, {'media': 'audio'}, {'clip_seconds': 11},
    {'quiet_start': '22:00'}, {'quiet_start': '22:00', 'quiet_end': '22:00'},
    {'quiet_start': '25:00', 'quiet_end': '07:00'}, {'timezone': '../etc/passwd'},
])
def test_rule_validation_is_strict(patch):
    with pytest.raises(ValueError):
        validate_rule(patch)


def test_a_known_group_destination_is_accepted():
    """A rule may target any group the bot has met, not only the fixed one."""
    rule = validate_rule({'destination': 'group:-1003570242441'})
    assert rule['destination'] == 'group:-1003570242441'
    with pytest.raises(ValueError):
        validate_rule({'destination': 'group:not-a-number'})


def test_private_chat_everyone_writes_to_each_account(tmp_path):
    """ТЗ F-702: "Private chat (everyone)" - the owner and the named admins."""
    async def run():
        api = provider()
        # The provider answers with the chat it was asked for, exactly as the
        # real one does (an ack for a different chat is never a delivery).
        async def send_image(data, mime, caption='', filename='', **kwargs):
            return {'ok': True, 'chat_id': kwargs['private_reply_to_user_id'], 'message_id': 7}
        api.send_image = AsyncMock(side_effect=send_image)
        people = [123, 8928749210]
        alerts = PresenceAlerts(tmp_path, lambda: api, lambda: None, 123, -456,
                                get_private_recipients=lambda: people)
        alerts.save_rule({'enabled': True, 'destination': 'owner', 'min_stable_s': 0})
        alerts.start()
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        assert api.send_image.await_count == 2, "every account with access gets it"
        assert {call.kwargs['private_reply_to_user_id'] for call in api.send_image.await_args_list} == set(people)
        assert alerts.status()['deliveries'][0]['status'] == 'sent'
        await alerts.close()
    asyncio.run(run())


def test_an_unconfirmed_recipient_stops_the_rest(tmp_path):
    """A partial fan-out is reported as uncertain and never retried."""
    async def run():
        api = provider()
        api.send_image = AsyncMock(side_effect=[
            {'ok': True, 'chat_id': 123, 'message_id': 7},
            {'ok': True, 'chat_id': 123},
        ])
        alerts = PresenceAlerts(tmp_path, lambda: api, lambda: None, 123, -456,
                                get_private_recipients=lambda: [123, 8928749210])
        alerts.save_rule({'enabled': True, 'destination': 'owner', 'min_stable_s': 0})
        alerts.start()
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        assert api.send_image.await_count == 2
        delivery = alerts.status()['deliveries'][0]
        assert delivery['status'] == 'uncertain' and '1 of 2' in delivery['detail']
        await alerts.close()
    asyncio.run(run())


def test_rules_are_disabled_by_default_and_patch_survives_restart(tmp_path):
    async def run():
        alerts = engine(tmp_path)
        assert alerts.list_rules() == []
        saved = alerts.save_rule({'target': 'person', 'name': 'Anton'})
        assert not saved['enabled'] and saved['destination'] == 'owner'
        changed = alerts.save_rule({'media': 'video'}, saved['id'])
        assert changed['name'] == 'Anton' and changed['clip_seconds'] == 5
        await alerts.close()
        restored = engine(tmp_path)
        assert restored.list_rules()[0]['media'] == 'video'
        assert restored.remove_rule(saved['id']) and restored.list_rules() == []
        await restored.close()
    asyncio.run(run())


def test_workplace_filters_and_stability_are_per_source(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])
    async def run():
        api = provider()
        rooms = {key: SimpleNamespace(workplace_name='Place ' + key) for key in ('a', 'b')}
        alerts = PresenceAlerts(tmp_path, lambda: api, lambda source=None: rooms.get(source), 123, -456)
        alerts.save_rule({'enabled': True, 'min_stable_s': 2, 'target': 'person', 'name': 'Anton'})
        alerts.start()
        alerts.observe(names=['Anton'], source_id='a', jpeg=b'a')
        await alerts.drain()
        clock[0] = 1002.
        alerts.observe(names=['Anton'], source_id='b', jpeg=b'b')
        await alerts.drain()
        api.send_image.assert_not_awaited()
        clock[0] = 1002.1
        alerts.observe(names=['Anton'], source_id='a', jpeg=b'a')
        await alerts.drain()
        api.send_image.assert_awaited_once()
        assert api.send_image.call_args.args[0] == b'a'
        assert 'Place a' in api.send_image.call_args.args[2]
        filtered = alerts.save_rule({'enabled': True, 'workplace_id': 'b', 'min_stable_s': 0})
        clock[0] += 1
        alerts.observe(persons=1, source_id='a', jpeg=b'a')
        await alerts.drain()
        assert api.send_image.await_count == 1
        clock[0] += 1
        alerts.observe(persons=1, source_id='b', jpeg=b'b')
        await alerts.drain()
        assert api.send_image.await_count == 2
        assert api.send_image.call_args.args[0] == b'b'
        assert next(rule for rule in alerts.list_rules() if rule['id'] == filtered['id'])['workplace_id'] == 'b'
        await alerts.close()
    asyncio.run(run())


def test_midnight_quiet_hours_use_configured_timezone():
    rule = validate_rule({'quiet_start': '22:00', 'quiet_end': '07:00'})
    assert quiet_now(rule, datetime.fromisoformat('2026-09-21T04:00:00+00:00').timestamp())
    assert quiet_now(rule, datetime.fromisoformat('2026-09-21T11:59:00+00:00').timestamp())
    assert not quiet_now(rule, datetime.fromisoformat('2026-09-21T12:00:00+00:00').timestamp())


def test_short_burst_does_not_meet_stability_and_continuous_presence_sends_once(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])
    async def run():
        api = provider()
        alerts = engine(tmp_path, api)
        alerts.save_rule({'enabled': True, 'min_stable_s': 2, 'cooldown_s': 10})
        alerts.start()
        for stamp in (1000., 1000.25, 1000.5):
            clock[0] = stamp
            assert alerts.observe(persons=1, jpeg=b'jpeg')
            await alerts.drain()
        api.send_image.assert_not_awaited()
        for stamp in (1002.1, 1003., 1012.):
            clock[0] = stamp
            alerts.observe(persons=1, jpeg=b'jpeg')
            await alerts.drain()
        api.send_image.assert_awaited_once()
        assert api.send_image.call_args.kwargs == {'private_reply_to_user_id': 123}
        assert alerts.status()['deliveries'][0]['status'] == 'sent'
        await alerts.close()
    asyncio.run(run())


def test_restart_and_uncertain_delivery_never_retry_same_episode(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])
    async def run():
        api = provider()
        api.send_image.side_effect = TelegramError('Unconfirmed connection', uncertain=True)
        alerts = engine(tmp_path, api)
        alerts.save_rule({'enabled': True, 'min_stable_s': 0, 'cooldown_s': 10})
        alerts.start()
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        assert alerts.status()['deliveries'][0]['status'] == 'uncertain'
        await alerts.close()
        alerts = engine(tmp_path, api)
        alerts.start()
        clock[0] += 1
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        api.send_image.assert_awaited_once()
        await alerts.close()
    asyncio.run(run())


def test_reappearance_requires_absence_and_reserved_cooldown(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])
    async def run():
        api = provider()
        alerts = engine(tmp_path, api)
        alerts.save_rule({'enabled': True, 'min_stable_s': 0, 'cooldown_s': 10, 'absence_s': 2})
        alerts.start()
        for stamp, people in [(1000., 1), (1001., 0), (1003., 1), (1004., 0), (1011., 1)]:
            clock[0] = stamp
            alerts.observe(persons=people, jpeg=b'jpeg')
            await alerts.drain()
        assert api.send_image.await_count == 2
        await alerts.close()
    asyncio.run(run())


def test_named_rule_uses_only_fresh_direct_names_not_room_history(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])
    async def run():
        api = provider()
        alerts = engine(tmp_path, api, SimpleNamespace(room=SimpleNamespace(tracks={'old': {'name': 'Anton'}})))
        alerts.save_rule({'enabled': True, 'target': 'person', 'name': 'Anton', 'min_stable_s': 0})
        alerts.start()
        for names in (None, ['Drew'], ['Anton']):
            clock[0] += 1
            alerts.observe(persons=1, names=names, jpeg=b'jpeg')
            await alerts.drain()
        api.send_image.assert_awaited_once()
        assert 'Anton' in api.send_image.call_args.args[2]
        await alerts.close()
    asyncio.run(run())


def test_unknown_rule_requires_explicit_fresh_unknown_evidence(tmp_path):
    async def run():
        api = provider()
        alerts = engine(tmp_path, api)
        alerts.save_rule({'enabled': True, 'target': 'unknown', 'min_stable_s': 0})
        alerts.start()
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        api.send_image.assert_not_awaited()
        alerts.observe(unknown_count=1, jpeg=b'jpeg')
        await alerts.drain()
        api.send_image.assert_awaited_once()
        await alerts.close()
    asyncio.run(run())


def test_video_uses_clip_transport_and_fixed_group_destination(tmp_path):
    async def run():
        api = provider()
        api.send_video.return_value = {'ok': True, 'chat_id': -456, 'message_id': 3}
        room = SimpleNamespace(receiving=False, session=SimpleNamespace(client_id='room'),
                               _request_camera_clip=AsyncMock(return_value=b'mp4'))
        alerts = engine(tmp_path, api, room)
        alerts.save_rule({'enabled': True, 'media': 'video', 'destination': 'group', 'min_stable_s': 0})
        alerts.start()
        alerts.observe(persons=1, source_id='room')
        await alerts.drain()
        room._request_camera_clip.assert_awaited_once()
        api.send_video.assert_awaited_once()
        assert api.send_video.call_args.args[:2] == (b'mp4', 'video/mp4')
        # The group is named explicitly since a rule may target any group the
        # bot has met (`group:<id>`), not only the one in the config.
        assert api.send_video.call_args.kwargs == {'group_chat_id': -456}
        assert alerts.status()['deliveries'][0]['status'] == 'sent'
        await alerts.close()
    asyncio.run(run())


def test_busy_room_does_not_start_clip_or_retry(tmp_path):
    async def run():
        api = provider()
        room = SimpleNamespace(receiving=True, _request_camera_clip=AsyncMock())
        alerts = engine(tmp_path, api, room)
        alerts.save_rule({'enabled': True, 'media': 'video', 'min_stable_s': 0})
        alerts.start()
        alerts.observe(persons=1)
        await alerts.drain()
        room._request_camera_clip.assert_not_awaited()
        api.send_video.assert_not_awaited()
        assert alerts.status()['deliveries'][0]['status'] == 'failed'
        await alerts.close()
    asyncio.run(run())


def test_missing_ack_is_uncertain_and_disabled_rule_does_not_send(tmp_path):
    async def run():
        api = provider()
        api.send_image.return_value = {'ok': True}
        alerts = engine(tmp_path, api)
        alerts.save_rule({})
        alerts.start()
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        api.send_image.assert_not_awaited()
        alerts.save_rule({'enabled': True, 'min_stable_s': 0})
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        assert alerts.status()['deliveries'][0]['status'] == 'uncertain'
        await alerts.close()
    asyncio.run(run())
