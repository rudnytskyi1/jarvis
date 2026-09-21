"""Real temporary admin stores with fake engines/cameras; no model or network."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common.config import Config
from hub.admin_backend import AdminBackend
from hub.speaker import FACE_KEY, PEOPLE_FILENAME, VOICE_KEY, VOICE_MODEL_ID, VoiceRegistry
from hub.storage import Memory
from hub.telegram_admin_state import TelegramAdminState

OWNER, GROUP = 8322835915, -100


def services(tmp_path, rooms=None):
    cfg = Config()
    cfg.server.telegram.control_user_id, cfg.server.telegram.chat_id = OWNER, GROUP
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    directory = tmp_path / 'registry'
    directory.mkdir()
    (directory / PEOPLE_FILENAME).write_text(json.dumps({'voice_model': VOICE_MODEL_ID, 'people': {
        'Anton': {'role': 'admin', VOICE_KEY: [[1., 0.]], FACE_KEY: [[0., 1.]]},
        'Alice': {'role': 'user', VOICE_KEY: [], FACE_KEY: []}}}), encoding='utf-8')
    registry = VoiceRegistry(directory)
    memory = Memory(tmp_path / 'memory')
    values = {'voices': registry, 'memory': memory, 'face': SimpleNamespace(threshold=.45)}
    rooms = rooms or {}
    def get_room(identifier=None):
        if identifier is not None:
            return rooms.get(identifier)
        return next(iter(rooms.values())) if len(rooms) == 1 else None
    workplaces = lambda: [dict(id=key, name=key.upper(), camera_name='Camera ' + key, connected=True)
                          for key in rooms]
    provider = SimpleNamespace(send_image=AsyncMock(return_value={'ok': True, 'message_id': 8}))
    rename = AsyncMock()
    backend = AdminBackend(cfg, access, runtime=lambda: values, get_room=get_room,
        get_alerts=lambda: None, rename_profile=rename, get_workplaces=workplaces, get_provider=lambda: provider)
    return SimpleNamespace(backend=backend, cfg=cfg, access=access, registry=registry, memory=memory,
                           runtime=values, rooms=rooms, provider=provider, rename=rename)


def room():
    return SimpleNamespace(_task=None, _telegram_control_task=None, _enroll_face_task=None,
        _enroll_pending=None, _face_selection=None, receiving=False, _control_tasks=set(),
        _reply_lock=asyncio.Lock(), presence=SimpleNamespace(clear=Mock()),
        room=SimpleNamespace(tracks={1: {'name': 'Anton'}}),
        _request_camera_frame_full=AsyncMock(return_value=SimpleNamespace(jpeg=b'camera-jpeg')))


@pytest.mark.parametrize('actor', [17, str(OWNER), True, None])
def test_backend_cannot_be_called_by_delegated_admin_or_forged_owner(tmp_path, actor):
    fixture = services(tmp_path)
    fixture.access.set_user(17, 'admin')
    fixture.backend.runtime = Mock(side_effect=AssertionError('unauthorized runtime access'))
    result = asyncio.run(fixture.backend.call('settings.set', {'key': 'server.permissions_enabled', 'value': False}, actor))
    assert result['ok'] is False and fixture.cfg.server.permissions_enabled
    fixture.backend.runtime.assert_not_called()
    assert fixture.access.events() == []


@pytest.mark.parametrize('payload', [None, {'token': 'synthetic'}, {'text': 'password=synthetic'},
                                    {'text': 'sk-' + 'x' * 30}])
def test_secrets_are_rejected_before_storage_or_audit(tmp_path, payload):
    fixture = services(tmp_path)
    result = asyncio.run(fixture.backend.call('memory.add', payload, OWNER))
    assert result['ok'] is False
    assert not fixture.memory.path.exists() and not fixture.access.events()


def test_settings_live_and_pending_changes_have_distinct_effects_and_survive_reload(tmp_path):
    async def run():
        fixture = services(tmp_path)
        old_voice = fixture.cfg.server.tts.speaker
        live = await fixture.backend.call('settings.set', {'key': 'server.face.threshold', 'value': '.63'}, OWNER)
        pending = await fixture.backend.call('settings.set', {'key': 'server.tts.speaker', 'value': 'en_2'}, OWNER)
        assert live['ok'] and not live['requires_restart']
        assert fixture.cfg.server.face.threshold == fixture.runtime['face'].threshold == .63
        assert pending['ok'] and pending['requires_restart']
        assert fixture.cfg.server.tts.speaker == old_voice
        listing = await fixture.backend.call('settings.list', {'category': 'tts'}, OWNER)
        assert next(row for row in listing['settings'] if row['key'] == 'server.tts.speaker')['pending_value'] == 'en_2'
        reloaded = TelegramAdminState(fixture.access.path, OWNER)
        assert reloaded.get_setting('config:server.tts.speaker') == {'value': 'en_2'}
        assert len(reloaded.events()) == 2
    asyncio.run(run())


@pytest.mark.parametrize('key,value', [('server.telegram.chat_id', -999),
    ('server.telegram.control_user_id', 17), ('server.telegram.enabled', True),
    ('client.camera.index', 8), ('server.face.threshold', 2), ('server.llm.monthly_budget_usd', float('nan'))])
def test_invalid_setting_cannot_change_config_or_leave_an_override(tmp_path, key, value):
    fixture = services(tmp_path)
    before = fixture.cfg.model_dump()
    result = asyncio.run(fixture.backend.call('settings.set', {'key': key, 'value': value}, OWNER))
    assert result['ok'] is False and fixture.cfg.model_dump() == before
    assert fixture.access.get_setting('config:' + key) is None and fixture.access.events() == []


def test_memory_edit_and_delete_update_keyed_preferences_without_resurrecting_old_values(tmp_path):
    async def run():
        fixture = services(tmp_path)
        fixture.memory.add('Old browser', '', key='browser', value='Old')
        fixture.memory.add('Use Firefox', '', key='apps.browser', value='Firefox')
        fixture.memory.add('Private Safari', 'Anton', key='browser', value='Safari')
        listing = await fixture.backend.call('memory.list', {'scope': 'shared'}, OWNER)
        assert len(listing['items']) == 1
        identifier = listing['items'][0]['id']
        changed = await fixture.backend.call('memory.edit', {'id': identifier, 'scope': 'shared',
            'text': 'Use Edge', 'value': 'Edge'}, OWNER)
        assert changed['ok']
        reloaded = Memory(fixture.memory.path.parent)
        assert reloaded.preference('browser', 'Anton') == {'value': 'Edge', 'scope': 'global'}
        assert reloaded.admin_entries()[0]['id'] == identifier
        original = fixture.memory.path.read_text(encoding='utf-8')
        deleted = await fixture.backend.call('memory.delete', {'id': identifier, 'scope': 'shared'}, OWNER)
        assert deleted['ok'] and fixture.memory.path.read_text(encoding='utf-8').startswith(original)
        assert Memory(fixture.memory.path.parent).admin_entries() == []
        assert reloaded.preference('browser', 'Anton') == {'value': 'Safari', 'scope': 'personal'}
        assert (await fixture.backend.call('memory.edit', {'id': identifier, 'text': 'stale'}, OWNER))['ok'] is False
    asyncio.run(run())


def test_memory_scope_and_identifier_prevent_cross_profile_edits(tmp_path):
    async def run():
        fixture = services(tmp_path)
        fixture.memory.add('A private fact', 'Anton')
        identifier = fixture.memory.admin_entries('Anton')[0]['id']
        before = fixture.memory.path.read_bytes()
        for payload in ({'id': identifier, 'scope': 'shared'},
                        {'id': identifier, 'scope': 'personal', 'owner_id': 'Missing'},
                        {'id': identifier, 'scope': 'personal', 'owner_id': 'Alice'}):
            assert (await fixture.backend.call('memory.delete', payload, OWNER))['ok'] is False
        assert fixture.memory.path.read_bytes() == before
    asyncio.run(run())


def test_legacy_memory_ids_and_tombstones_survive_rename_and_restart(tmp_path):
    memory = Memory(tmp_path)
    memory.path.write_text('\n'.join([json.dumps('Legacy shared fact'), '{invalid json',
        json.dumps({'person': 'Anton', 'fact': 'Old private fact', 'author': 'Anton'}),
        json.dumps({'person': 'Anton', 'fact': 'Use Firefox', 'key': 'browser', 'value': 'Firefox'})]) + '\n', encoding='utf-8')
    shared_id = memory.admin_entries()[0]['id']
    note, setting = memory.admin_entries('Anton')
    memory.change_entry(note['id'], 'Anton', text='Corrected private fact')
    memory.rename('Anton', 'Anthony')
    reloaded = Memory(tmp_path)
    assert reloaded.admin_entries()[0]['id'] == shared_id
    assert reloaded.admin_entries('Anton') == []
    assert {row['id'] for row in reloaded.admin_entries('Anthony')} == {note['id'], setting['id']}
    assert reloaded.admin_entries('Anthony')[0]['fact'] == 'Corrected private fact'
    reloaded.change_entry(setting['id'], 'Anthony', delete=True)
    assert Memory(tmp_path).preference('browser', 'Anthony') is None
    assert '{invalid json' in memory.path.read_text(encoding='utf-8')


def test_profile_crud_changes_active_registry_but_preserves_source_archives(tmp_path):
    async def run():
        active = room()
        fixture = services(tmp_path, {'a': active})
        source = fixture.registry.audio_dir / 'Anton' / 'sample.wav'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'original audio')
        archive = tmp_path / 'training_archive' / 'Anton' / 'original.jpg'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b'original camera photo')
        fixture.memory.add('Retained person fact', 'Anton')
        created = await fixture.backend.call('profiles.create', {'name': 'Bob', 'role': 'trusted'}, OWNER)
        assert created['ok'] and fixture.registry.people()['Bob'] == 'trusted'
        assert (await fixture.backend.call('profiles.reset_voice', {'id': 'Anton'}, OWNER))['ok']
        assert 'Anton' not in fixture.registry.voice_profiles() and fixture.registry.face_profiles()['Anton']
        assert (await fixture.backend.call('profiles.reset_face', {'id': 'Anton'}, OWNER))['ok']
        assert 'Anton' not in fixture.registry.face_profiles()
        assert (await fixture.backend.call('profiles.delete', {'id': 'Anton'}, OWNER))['ok']
        assert 'Anton' not in VoiceRegistry(fixture.registry.path.parent).people()
        assert source.read_bytes() == b'original audio' and archive.read_bytes() == b'original camera photo'
        assert fixture.memory.facts('Anton') == ['Retained person fact']
        assert list((fixture.registry.path.parent / 'profile_backups').glob('*.json'))
        assert active.presence.clear.call_count == 4 and active.room.tracks == {}
    asyncio.run(run())


@pytest.mark.parametrize('busy', ['_enroll_pending', '_face_selection', '_enroll_face_task'])
def test_enrollment_in_any_workplace_prevents_profile_mutation(tmp_path, busy):
    async def run():
        first, second = room(), room()
        setattr(second, busy, SimpleNamespace(done=lambda: False) if busy.endswith('_task') else {'name': 'Anton'})
        fixture = services(tmp_path, {'a': first, 'b': second})
        before = fixture.registry.path.read_bytes()
        result = await fixture.backend.call('profiles.delete', {'id': 'Anton'}, OWNER)
        assert result['ok'] is False and fixture.registry.path.read_bytes() == before
        assert 'Anton' in fixture.registry.people()
        first.presence.clear.assert_not_called()
        second.presence.clear.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize('action,name', [('create', 'Bob'), ('delete', 'Anton'),
                                      ('reset_voice', 'Anton'), ('reset_face', 'Anton')])
def test_failed_profile_save_restores_active_state(tmp_path, monkeypatch, action, name):
    fixture = services(tmp_path)
    snapshot = {key: fixture.registry.profile_snapshot(key) for key in fixture.registry.people()}
    original = fixture.registry.path.read_bytes()
    monkeypatch.setattr(fixture.registry, '_save_locked', Mock(side_effect=OSError('synthetic disk failure')))
    result = asyncio.run(fixture.backend.call('profiles.' + action, {'name': name, 'id': name}, OWNER))
    assert result['ok'] is False and fixture.registry.path.read_bytes() == original
    assert {key: fixture.registry.profile_snapshot(key) for key in fixture.registry.people()} == snapshot


def test_profile_rename_calls_coordinated_service_and_never_implicitly_merges(tmp_path):
    async def run():
        fixture = services(tmp_path)
        result = await fixture.backend.call('profiles.rename', {'id': 'Anton', 'name': 'Anthony'}, OWNER)
        assert result['ok']
        fixture.rename.assert_awaited_once_with('Anton', 'Anthony')
        before = fixture.registry.path.read_bytes()
        with pytest.raises(ValueError, match='already belongs'):
            fixture.registry.rename_person('Anton', 'Alice', allow_merge=False)
        assert fixture.registry.path.read_bytes() == before
    asyncio.run(run())


def test_workplace_selection_is_route_scoped_and_photo_never_uses_arbitrary_recipient(tmp_path):
    async def run():
        fixture = services(tmp_path, {'a': room(), 'b': room()})
        listing = await fixture.backend.call('workplaces.list', {'chat_id': OWNER}, OWNER)
        assert listing['selected_id'] is None
        assert (await fixture.backend.call('workplaces.select', {'chat_id': OWNER, 'id': 'b'}, OWNER))['ok']
        assert fixture.access.get_setting(f'workplace:{OWNER}:{OWNER}') == 'b'
        assert (await fixture.backend.call('workplaces.list', {'chat_id': GROUP}, OWNER))['selected_id'] is None
        for destination in (OWNER, GROUP):
            result = await fixture.backend.call('workplaces.photo', {'chat_id': destination, 'id': 'b'}, OWNER)
            assert result['ok']
            expected = {'private_reply_to_user_id': OWNER} if destination == OWNER else {}
            fixture.provider.send_image.assert_awaited_with(b'camera-jpeg', 'image/jpeg', caption='B · Camera b', **expected)
            assert fixture.rooms['b']._telegram_control_task is None
        assert fixture.rooms['b']._request_camera_frame_full.await_count == 2
        fixture.rooms['a']._request_camera_frame_full.assert_not_awaited()
        before = fixture.provider.send_image.await_count
        assert (await fixture.backend.call('workplaces.photo', {'chat_id': 17, 'id': 'b'}, OWNER))['ok'] is False
        assert fixture.provider.send_image.await_count == before
    asyncio.run(run())


def test_busy_or_failed_workplace_photo_releases_reservation_without_delivery(tmp_path):
    async def run():
        active = room()
        fixture = services(tmp_path, {'a': active})
        active.receiving = True
        assert (await fixture.backend.call('workplaces.photo', {'id': 'a'}, OWNER))['ok'] is False
        active._request_camera_frame_full.assert_not_awaited()
        active.receiving = False
        async def fail(identifier):
            assert active._telegram_control_task is asyncio.current_task()
            raise RuntimeError('synthetic camera failure')
        active._request_camera_frame_full.side_effect = fail
        assert (await fixture.backend.call('workplaces.photo', {'id': 'a'}, OWNER))['ok'] is False
        assert active._telegram_control_task is None
        fixture.provider.send_image.assert_not_awaited()
    asyncio.run(run())
