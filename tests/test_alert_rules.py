"""ТЗ F-702: правила на события F-301/F-109/F-311, дом и канал доставки."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from common.config import Config
from hub import app as hub_app
from hub.presence_alerts import (
    CHANNELS,
    EVENT_KINDS,
    PresenceAlerts,
    home_cooldown,
    home_quiet_now,
    home_settings,
    quiet_now,
    validate_rule,
)

OWNER, GROUP = 8322835915, -10012345678
NOW = datetime(2026, 9, 21, 22, 30, tzinfo=ZoneInfo("America/Chicago"))


class Provider:
    """Telegram stand-in that acknowledges every photo it is handed."""

    ready = True

    def __init__(self):
        self.sent = []

    async def send_image(self, data, mime, caption, filename, **kwargs):
        self.sent.append({'kind': 'image', 'caption': caption, 'chat_id': kwargs.get(
            'private_reply_to_user_id', GROUP), 'message_id': len(self.sent) + 5})
        return {'ok': True, 'chat_id': self.sent[-1]['chat_id'], 'message_id': self.sent[-1]['message_id']}

    async def send_video(self, data, mime, caption, filename, **kwargs):
        self.sent.append({'kind': 'video', 'caption': caption,
                          'chat_id': kwargs.get('private_reply_to_user_id', GROUP),
                          'message_id': len(self.sent) + 5})
        return {'ok': True, 'chat_id': self.sent[-1]['chat_id'], 'message_id': self.sent[-1]['message_id']}


class Room:
    """A connected room whose HUD the hub can put a caption on."""

    workplace_name = 'Зал'

    def __init__(self):
        self.captions = []

    async def send_json(self, payload):
        self.captions.append(payload)


def service(tmp_path, *, home=None, provider=None, room=None):
    alerts = PresenceAlerts(tmp_path, lambda: provider, lambda client_id=None: room,
                            OWNER, GROUP, get_home=lambda home_id: home)
    alerts.start()
    return alerts


# --- the rule model ----------------------------------------------------------


def test_the_events_and_channels_of_the_tz_are_the_ones_the_rules_accept():
    assert set(EVENT_KINDS) == {'presence', 'person_entered', 'person_left',
                                'unknown_appeared', 'zone_entered', 'sound_event', 'object'}
    assert set(CHANNELS) == {'telegram', 'push', 'hud'}
    rule = validate_rule({'event': 'sound_event', 'channel': 'hud', 'home_id': 'livingroom'})
    assert rule['event'] == 'sound_event' and rule['channel'] == 'hud'
    assert rule['home_id'] == 'livingroom'
    assert validate_rule({})['event'] == 'presence'  # прежнее поведение по умолчанию
    for bad in ({'event': 'doorbell'}, {'channel': 'sms'}, {'home_id': '../etc'}, {'event': 'object'}):
        with pytest.raises(ValueError):
            validate_rule(bad)


def test_an_object_rule_keeps_its_label_and_a_zone_is_normalised():
    rule = validate_rule({'event': 'object', 'name': '  посылка ', 'zone': '  у  двери '})
    assert rule['name'] == 'посылка'
    assert rule['zone'] == 'у двери'
    # Правило на человека без имени по-прежнему отвергается.
    with pytest.raises(ValueError):
        validate_rule({'target': 'person'})


# --- quiet hours and cooldown of the HOME ------------------------------------


def test_the_home_window_is_read_and_crosses_midnight():
    home = {'quiet_hours': {'start': '23:00', 'end': '07:00'}, 'timezone': 'America/Chicago'}
    assert home_settings(home)['quiet_start'] == '23:00'
    late = datetime(2026, 9, 21, 23, 30, tzinfo=ZoneInfo('America/Chicago')).timestamp()
    morning = datetime(2026, 9, 21, 9, 30, tzinfo=ZoneInfo('America/Chicago')).timestamp()
    assert home_quiet_now(home, late) is True
    assert home_quiet_now(home, morning) is False
    # Дом без окна молчит только по правилу.
    assert home_quiet_now({}, late) is False
    assert home_settings(None) == {}


def test_the_home_can_make_a_rule_rarer_but_not_faster():
    home = {'cooldown_s': 3600}
    assert home_cooldown(home, 300) == 3600
    assert home_cooldown(home, 7200) == 7200
    assert home_cooldown({}, 300) == 300
    # Правило и дом действуют вместе: своё окно правила никто не отменяет.
    rule = validate_rule({'quiet_start': '13:00', 'quiet_end': '14:00'})
    assert quiet_now(rule, NOW.timestamp()) is False
    assert home_quiet_now({'quiet_hours': {'start': '22:00', 'end': '23:00'},
                           'timezone': 'America/Chicago'}, NOW.timestamp()) is True


# --- the service -------------------------------------------------------------


def test_an_event_rule_fires_on_the_event_with_its_own_words(tmp_path):
    async def scenario():
        provider = Provider()
        alerts = service(tmp_path, provider=provider)
        try:
            alerts.save_rule(dict(enabled=True, event='person_entered', target='person',
                                  name='Макс', media='photo', destination='owner'))
            alerts.observe(persons=1, names=('Макс',), source_id='living', jpeg=b'jpeg-bytes')
            alerts.observe_event('person_entered', name='Макс', source_id='living', home_id='livingroom')
            await alerts.drain()
            assert len(provider.sent) == 1
            assert 'пришёл' in provider.sent[0]['caption'] and 'Макс' in provider.sent[0]['caption']
            deliveries = alerts.status()['deliveries']
            assert [(row['status'], row['detail']) for row in deliveries][0][0] == 'sent'
        finally:
            await alerts.close()

    asyncio.run(scenario())


def test_a_rule_of_one_home_never_fires_for_another(tmp_path):
    async def scenario():
        provider, alerts = Provider(), service(tmp_path)
        try:
            alerts.save_rule(dict(enabled=True, event='unknown_appeared', target='unknown',
                                  home_id='office', media='photo'))
            alerts.observe(persons=1, unknown_count=1, source_id='living', jpeg=b'jpeg-bytes')
            alerts.observe_event('unknown_appeared', source_id='living', home_id='livingroom')
            await alerts.drain()
            assert provider.sent == []
            assert alerts.list_rules()[0]['home_id'] == 'office'
        finally:
            await alerts.close()

    asyncio.run(scenario())


def test_a_quiet_home_silences_the_rule_even_when_the_rule_has_no_window(tmp_path):
    async def scenario():
        night = datetime.now(ZoneInfo('UTC'))
        home = {'quiet_hours': {'start': (night - timedelta(minutes=5)).strftime('%H:%M'),
                                'end': (night + timedelta(minutes=5)).strftime('%H:%M')},
                'timezone': 'UTC'}
        provider, alerts = Provider(), service(tmp_path, home=home)
        try:
            alerts.save_rule(dict(enabled=True, event='sound_event', media='photo', timezone='UTC'))
            assert alerts.observe_event('sound_event', label='стук', source_id='living',
                                        home_id='livingroom') is True
            await alerts.drain()
            assert provider.sent == []
            assert alerts.status()['deliveries'] == []
        finally:
            await alerts.close()

    asyncio.run(scenario())


def test_a_home_cooldown_holds_the_second_event_back(tmp_path):
    async def scenario():
        provider = Provider()
        alerts = service(tmp_path, provider=provider, home={'cooldown_s': 3600})
        try:
            alerts.save_rule(dict(enabled=True, event='sound_event', media='photo', cooldown_s=1))
            alerts.observe(persons=1, source_id='living', jpeg=b'jpeg-bytes')
            alerts.observe_event('sound_event', label='стук', source_id='living')
            await alerts.drain()
            time.sleep(1.1)
            alerts.observe_event('sound_event', label='стук', source_id='living')
            await alerts.drain()
            assert len(provider.sent) == 1  # дом держит минимум в час
        finally:
            await alerts.close()

    asyncio.run(scenario())


def test_the_hud_channel_puts_a_caption_on_the_room(tmp_path):
    async def scenario():
        room = Room()
        provider, alerts = Provider(), service(tmp_path, provider=Provider(), room=room)
        try:
            alerts.save_rule(dict(enabled=True, event='object', name='посылка',
                                  channel='hud', media='photo'))
            alerts.observe_event('object', label='посылка', source_id='living', home_id='livingroom')
            await alerts.drain()
            assert provider.sent == []          # канал HUD не трогает Telegram
            assert len(room.captions) == 1
            assert room.captions[0]['type'] == 'status'
            assert 'посылка' in room.captions[0]['text']
            assert alerts.status()['deliveries'][0]['status'] == 'sent'
        finally:
            await alerts.close()

    asyncio.run(scenario())


def test_the_push_channel_says_it_has_no_transport_yet(tmp_path):
    async def scenario():
        provider, alerts = Provider(), service(tmp_path)
        try:
            alerts.save_rule(dict(enabled=True, event='sound_event', label='',
                                  channel='push', media='photo'))
            alerts.observe_event('sound_event', label='звонок', source_id='living')
            await alerts.drain()
            assert provider.sent == []
            delivery = alerts.status()['deliveries'][0]
            assert delivery['status'] == 'skipped' and 'F-712' in delivery['detail']
        finally:
            await alerts.close()


def test_an_event_is_refused_when_it_cannot_be_trusted(tmp_path):
    async def scenario():
        alerts = service(tmp_path)
        try:
            assert alerts.observe_event('presence', source_id='living') is False
            assert alerts.observe_event('doorbell', source_id='living') is False
            stale = time.time() - 600
            assert alerts.observe_event('sound_event', label='стук', observed_at=stale) is False
            assert alerts.observe_event('sound_event', label='стук') is True
        finally:
            await alerts.close()
    asyncio.run(scenario())


# --- the hub feeds the rules --------------------------------------------------


class Recorder:
    def __init__(self):
        self.events = []

    def observe(self, **payload):
        self.events.append(('presence', payload))
        return True

    def observe_event(self, kind, **payload):
        self.events.append((kind, payload))
        return True


def connection(alerts, **attributes):
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = 'pc-1:5100'
    conn.session = SimpleNamespace(client_id='living')
    conn.home_id = 'livingroom'
    conn.camera_state = None
    conn.presence = SimpleNamespace(note_persons=lambda count: None, reconcile=lambda count: None)
    for name, value in attributes.items():
        setattr(conn, name, value)
    return conn


def test_a_sound_event_from_the_client_reaches_the_rules(monkeypatch):
    alerts = Recorder()
    monkeypatch.setattr(hub_app, '_presence_alerts', alerts)
    conn = connection(alerts)
    conn._on_sound_event({'label': 'стук', 'conf': 0.9, 'at_ms': 120})
    assert alerts.events == [('sound_event', {'label': 'стук', 'confidence': 0.9,
                                             'source_id': 'living', 'home_id': 'livingroom'})]
    # Слабый сигнал до правил не доходит: детектор живёт на клиенте, а правило
    # не должно срабатывать на «может быть, кашель».
    conn._on_sound_event({'label': 'кашель', 'conf': 0.2})
    conn._on_sound_event({'label': '  ', 'conf': 0.9})
    assert len(alerts.events) == 1


def test_a_new_object_in_the_frame_becomes_one_event(monkeypatch):
    alerts = Recorder()
    monkeypatch.setattr(hub_app, '_presence_alerts', alerts)
    conn = connection(alerts)
    conn._on_camera_state({'persons': 0, 'objects': {'cat': 1}})
    assert [payload['label'] for kind, payload in alerts.events if kind == 'object'] == ['cat']
    conn._on_camera_state({'persons': 0, 'objects': {'cat': 1}})       # та же метка
    assert len([kind for kind, _ in alerts.events if kind == 'object']) == 1
    conn._on_camera_state({'persons': 0, 'objects': {'cat': 1, 'package': 1}})
    assert [payload['label'] for kind, payload in alerts.events if kind == 'object'] == ['cat', 'package']
    assert all(payload['home_id'] == 'livingroom' for kind, payload in alerts.events if kind == 'object')


def test_the_home_of_a_rule_is_read_from_the_config(monkeypatch):
    cfg = Config.model_validate({"homes": [{"home_id": "livingroom", "name": "Living",
                                            "tz": "America/Chicago",
                                            "quiet_hours": {"start": "23:00", "end": "07:00"},
                                            "alert_cooldown_s": 900}]})
    monkeypatch.setattr(hub_app, 'get_config', lambda: cfg)
    home = hub_app._alert_home('livingroom')
    assert home['timezone'] == 'America/Chicago'
    assert home['cooldown_s'] == 900
    assert home['quiet_hours'] == {'start': '23:00', 'end': '07:00'}
    assert hub_app._alert_home('unknown') == {}
    assert hub_app._alert_home('') == {}
