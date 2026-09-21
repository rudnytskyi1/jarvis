"""Delegated tool capabilities and Telegram photo inspection use mocks only."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_control import authorized_message, tool_capabilities
from tests.test_telegram_chat import png
from tests.test_telegram_control import USER, FakeBrain, FakeRoom, config, message, setup


def policy(tmp_path, role='operator', capabilities=None):
    state = TelegramAdminState(tmp_path / 'access.sqlite3', USER)
    state.set_user(17, role, capabilities)
    return state


def delegated_message(private=False):
    value = message(private=private)
    value['from']['id'] = 17
    if private:
        value['chat']['id'] = 17
    return value


@pytest.mark.parametrize('private', [False, True])
def test_explicit_delegated_sender_is_valid_only_in_own_route(tmp_path, private):
    access = policy(tmp_path)
    current = delegated_message(private)
    assert authorized_message(config(), current, access)
    current['chat']['id'] += 1
    assert not authorized_message(config(), current, access)
    current = delegated_message(private)
    current['from']['id'] = '17'
    assert not authorized_message(config(), current, access)
    access.set_user(17, 'blocked')
    assert not authorized_message(config(), delegated_message(private), access)
    assert authorized_message(config(), message(private=private), access)


@pytest.mark.parametrize('name,args,expected', [
    ('generate_image', {'source': 'none'}, {'images'}),
    ('generate_image', {'source': 'camera', 'target': 'wallpaper', 'reference_people': ['Person']},
     {'images', 'camera', 'pc', 'profiles'}),
    ('generate_image', {'source': 'screen'}, {'images', 'pc'}),
    ('telegram_send', {'kind': 'image', 'source': 'annotated'}, {'chat', 'camera'}),
    ('telegram_send', {'kind': 'image', 'source': 'screen'}, {'chat', 'pc'}),
    ('show_photo', {}, {'camera'}),
    ('show_photo', {'which': 'hide'}, {'pc'}),
    ('find_object', {'source': 'screen'}, {'pc'}),
    ('enroll_face', {}, {'profiles', 'camera'}),
    ('inspect_photo', {}, {'images'}),
    ('recall_conversation', {}, {'memory'}),
    ('invented_future_tool', {}, None),
])
def test_source_and_secondary_action_permission_mapping(name, args, expected):
    assert tool_capabilities(name, args) == expected


def test_delegate_denied_pc_never_reaches_transport_even_when_server_permissions_off(tmp_path):
    async def run():
        access = policy(tmp_path, 'member')
        controller, cfg, _, room, brain, _, _, _ = setup(access=access)
        cfg.server.permissions_enabled = False
        result = await controller([], delegated_message(), 'Set the volume to 20')
        assert 'permission denied' in result
        assert brain.results[-1]['ok'] is False and not room.sent
    asyncio.run(run())


def test_permission_is_rechecked_after_model_starts(tmp_path):
    async def run():
        access = policy(tmp_path)
        controller, _, _, room, brain, _, _, _ = setup(access=access)
        brain.before_tool = lambda: access.set_user(17, 'operator', {'pc': False})
        await controller([], delegated_message(), 'Set the volume to 20')
        assert brain.results[-1]['ok'] is False and not room.sent
    asyncio.run(run())


def test_unknown_tools_fail_closed_for_delegates(tmp_path):
    async def run():
        access = policy(tmp_path, 'admin')
        controller, _, _, room, brain, _, _, _ = setup(access=access, brain=FakeBrain('future_tool'))
        await controller([], delegated_message(), 'Run future tool')
        assert brain.results[-1]['ok'] is False and not room.sent
    asyncio.run(run())


def test_memory_facts_follow_private_and_group_keys_and_require_capability(tmp_path):
    async def run():
        access = policy(tmp_path)
        memory = SimpleNamespace(effective=Mock(return_value=[]))
        controller, _, _, _, _, _, _, _ = setup(access=access, get_memory=lambda: memory, brain=FakeBrain(None))
        await controller([], delegated_message(), 'Hello')
        memory.effective.assert_not_called()
        access.set_user(17, 'admin')
        await controller([], delegated_message(), 'Hello')
        memory.effective.assert_called_with(f'telegram:{delegated_message()["chat"]["id"]}:17')
        await controller([], delegated_message(True), 'Hello')
        memory.effective.assert_called_with('telegram:17')
    asyncio.run(run())


def test_selected_workplace_receives_actions_and_global_default_does_not(tmp_path):
    async def run():
        selected, default = FakeRoom(), FakeRoom()
        controller, _, current, _, _, _, _, _ = setup(room=default, select_room=lambda msg: selected)
        await controller([], current, 'Set volume to 20')
        assert selected.sent and not default.sent
        assert selected._telegram_control_task is None
    asyncio.run(run())


def test_missing_workplace_selection_keeps_upload_tools_available(tmp_path):
    async def run():
        inspector = AsyncMock(return_value={'ok': True, 'answer': 'A person.', '_annotation': b'annotated jpg'})
        reference = AsyncMock(return_value=(png(), 'image/png'))
        brain = FakeBrain('inspect_photo', {'query': 'Who is in the photo?', 'target': 'person'})
        controller, _, current, room, _, provider, _, _ = setup(brain=brain, select_room=lambda msg: None,
            get_image_reference=reference, inspect_photo=inspector)
        current['photo'] = [{'file_id': 'current-photo', 'width': 32, 'height': 24}]
        await controller([], current, 'Who is in this photo?')
        inspector.assert_awaited_once()
        assert inspector.call_args.args[0].png == png()
        assert '_annotation' not in brain.results[-1]
        assert brain.results[-1]['telegram_delivery']['ok']
        provider.send_image.assert_awaited_once_with(b'annotated jpg', 'image/jpeg',
                                                   caption='Photo analysis', reply_to_message_id=42)
        assert not room.sent
        room._request_image.assert_not_awaited()
    asyncio.run(run())


def test_inspection_never_substitutes_old_cached_or_room_photo(tmp_path):
    async def run():
        inspector, reference = AsyncMock(), AsyncMock()
        brain = FakeBrain('inspect_photo', {'query': 'Describe it'})
        controller, _, current, room, _, _, _, _ = setup(brain=brain,
            get_image_reference=reference, inspect_photo=inspector)
        await controller([], current, 'Describe the photo')
        assert brain.results[-1]['ok'] is False
        inspector.assert_not_awaited()
        reference.assert_not_awaited()
        room._request_image.assert_not_awaited()
    asyncio.run(run())
