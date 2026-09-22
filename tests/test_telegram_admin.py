import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hub.telegram_admin import TelegramAdmin
from hub.telegram_admin_state import TelegramAdminState

OWNER, GROUP, BOT = 8322835915, -10012345678, 777


class Provider:
    def __init__(self):
        self.sent, self.edited, self.answers = [], [], []
        self.latest = None

    async def send_text(self, text, **kwargs):
        record = {'text': text, 'message_id': len(self.sent) + 100, **deepcopy(kwargs)}
        self.sent.append(record)
        self.latest = record
        return {'ok': True, 'message_id': record['message_id']}

    async def edit_text(self, text, **kwargs):
        self.latest = {'text': text, **deepcopy(kwargs)}
        self.edited.append(self.latest)
        return {'ok': True, 'message_id': kwargs['message_id']}

    async def answer_callback(self, callback_query_id, text='', show_alert=False):
        self.answers.append((callback_query_id, text, show_alert))


class Backend:
    def __init__(self):
        self.calls = []

    async def __call__(self, action, payload, actor_id):
        self.calls.append((action, deepcopy(payload), actor_id))
        if action == 'memory.list':
            return {'ok': True, 'items': [{'id': 'fact1', 'text': 'private fact'}],
                    'owner': payload.get('owner_id', f'telegram:{actor_id}')}
        if action == 'settings.list':
            return {'ok': True, 'settings': [{'key': 'server.test', 'label': 'Test setting',
                    'value': 1, 'type': 'int', 'editable': True, 'requires_restart': True}]}
        if action == 'workplaces.list':
            return {'ok': True, 'selected_id': 'living', 'items': [
                {'id': 'living', 'name': 'Зал', 'connected': True, 'camera_name': 'Вход'},
                {'id': 'office', 'name': 'Кабинет', 'connected': True, 'camera_name': 'Стол'}]}
        return {'ok': True, 'items': []}


def setup(tmp_path):
    state = TelegramAdminState(tmp_path / 'access.sqlite3', OWNER)
    provider, backend, now = Provider(), Backend(), [10.0]
    cfg = SimpleNamespace(control_user_id=OWNER, chat_id=GROUP)
    admin = TelegramAdmin(provider, cfg, state, backend, clock=lambda: now[0])
    admin.set_identity(BOT, 'RowanBot')
    return admin, provider, backend, state, now


def message(text, *, sender=OWNER, chat=GROUP, reply=None):
    value = {'message_id': 50, 'text': text, 'from': {'id': sender, 'is_bot': False},
             'chat': {'id': chat, 'type': 'private' if chat > 0 else 'supergroup'}}
    if reply is not None:
        value['reply_to_message'] = {'message_id': reply, 'from': {'id': BOT, 'is_bot': True}}
    return {'message': value}


def button(provider, label):
    return next(item['callback_data'] for row in provider.latest['reply_markup']['inline_keyboard']
                for item in row if item['text'].startswith(label))


def callback(provider, token, *, sender=OWNER, chat=GROUP, message_id=None):
    return {'callback_query': {'id': 'query-' + token, 'data': token,
            'from': {'id': sender, 'is_bot': False}, 'message': {
                'message_id': message_id or provider.latest['message_id'],
                'chat': {'id': chat, 'type': 'private' if chat > 0 else 'supergroup'},
                'from': {'id': BOT, 'is_bot': True}}}}


def test_a_named_admin_gets_the_panel_and_the_groups_it_has_met(tmp_path):
    """``admin_user_ids`` opens /tools; a known group becomes a destination."""
    extra, notifications = 8928749210, -1003570242441

    async def run():
        state = TelegramAdminState(tmp_path / 'access.sqlite3', OWNER, [extra])
        state.set_setting('chats', {str(notifications): {
            'id': notifications, 'title': 'RowanAI Notifications', 'type': 'supergroup', 'seen': 1.0}})
        provider, backend, now = Provider(), Backend(), [10.0]
        admin = TelegramAdmin(provider, SimpleNamespace(control_user_id=OWNER,
                                                        admin_user_ids=[extra], chat_id=GROUP),
                              state, backend, clock=lambda: now[0])
        admin.set_identity(BOT, 'RowanBot')
        assert await admin.handle_update(message('/tools', sender=extra, chat=extra))
        assert provider.sent, "a named admin opens the panel in their own chat"
        await admin.handle_update(callback(provider, button(provider, 'Notifications'),
                                           sender=extra, chat=extra))
        await admin.handle_update(callback(provider, button(provider, 'New rule'),
                                           sender=extra, chat=extra))
        await admin.handle_update(callback(provider, button(provider, 'Destination'),
                                           sender=extra, chat=extra))
        labels = [item['text'] for row in provider.latest['reply_markup']['inline_keyboard']
                  for item in row]
        assert 'Private chat (everyone) · 2' in labels, labels
        # The group the bot has met is offered by name, and it is what the rule
        # stores when picked.
        group_button = button(provider, 'RowanAI Notifications')
        await admin.handle_update(callback(provider, group_button, sender=extra, chat=extra))
        await admin.handle_update(callback(provider, button(provider, 'Save rule'),
                                           sender=extra, chat=extra))
        await admin.handle_update(callback(provider, button(provider, 'Confirm'),
                                           sender=extra, chat=extra))
        saved = next(call[1] for call in backend.calls if call[0] == 'alerts.create')
        assert saved['destination'] == f'group:{notifications}'

    asyncio.run(run())


def test_the_owner_can_still_be_revoked_in_the_config(tmp_path):
    """A named admin is a config fact, not a panel grant."""

    async def run():
        state = TelegramAdminState(tmp_path / 'access.sqlite3', OWNER, [8928749210])
        provider, backend, now = Provider(), Backend(), [10.0]
        cfg = SimpleNamespace(control_user_id=OWNER, admin_user_ids=[], chat_id=GROUP)
        admin = TelegramAdmin(provider, cfg, state, backend, clock=lambda: now[0])
        admin.set_identity(BOT, 'RowanBot')
        # The owner is gone from the config, the extra admin is gone with it.
        cfg.control_user_id = 1
        # The command is still claimed (so the chat handler leaves it alone),
        # but no panel is opened for either account.
        await admin.handle_update(message('/tools', sender=OWNER))
        await admin.handle_update(message('/tools', sender=8928749210, chat=8928749210))
        assert not provider.sent, "the panel does not open for a revoked account"

    asyncio.run(run())


def test_panel_is_owner_only_and_exact_bot_command(tmp_path):
    async def run():
        admin, provider, _, state, _ = setup(tmp_path)
        state.set_user(42, 'admin')
        assert await admin.handle_update(message('/tools', sender=42))
        assert not provider.sent
        assert not await admin.handle_update(message('/tools@OtherBot'))
        assert await admin.handle_update(message('/tools@RowanBot'))
        assert provider.sent
        assert await admin.handle_update(message('/tools', chat=OWNER))
        assert provider.sent[-1]['private_reply_to_user_id'] == OWNER
        await admin.close()
    asyncio.run(run())


def test_callback_binds_owner_chat_message_and_is_one_use(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        token = button(provider, 'Status')
        backend.calls.clear()
        assert len(token.encode()) <= 64
        for changes in ({'sender': 42}, {'chat': OWNER}, {'message_id': 999}):
            await admin.handle_update(callback(provider, token, **changes))
        assert not backend.calls
        request = callback(provider, token)
        await admin.handle_update(request)
        await admin.handle_update(request)
        assert [item[0] for item in backend.calls] == ['status']
        assert provider.answers[-1][2] is True
    asyncio.run(run())


def test_expired_and_replaced_panel_buttons_do_not_work(tmp_path):
    async def run():
        admin, provider, backend, _, now = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        expired = callback(provider, button(provider, 'Status'))
        backend.calls.clear()
        now[0] += 901
        await admin.handle_update(expired)
        assert not backend.calls
        await admin.handle_update(message('/tools'))
        old = callback(provider, button(provider, 'Status'))
        await admin.handle_update(message('/tools'))
        backend.calls.clear()
        await admin.handle_update(old)
        assert not backend.calls
    asyncio.run(run())


def test_memory_input_only_consumes_owner_reply_and_rejects_secrets(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Shared memory')))
        await admin.handle_update(callback(provider, button(provider, 'Add entry')))
        prompt_id = provider.latest['message_id']
        assert not await admin.handle_update(message('normal conversation'))
        assert not await admin.handle_update(message('malicious', sender=42, reply=prompt_id))
        assert await admin.handle_update(message('token=123456789:abcdefghijklmnopqrstuvwxyz1234', reply=prompt_id))
        assert not any(call[0] == 'memory.add' for call in backend.calls)
        assert await admin.handle_update(message('Remember tea', reply=provider.latest['message_id']))
        assert next(call for call in backend.calls if call[0] == 'memory.add')[1]['text'] == 'Remember tea'
        assert '123456789:' not in str(provider.edited)
    asyncio.run(run())


def test_personal_memory_is_only_rendered_to_owner_dm(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Personal memory')))
        private = [item for item in provider.sent if item.get('private_reply_to_user_id') == OWNER]
        assert private and 'private fact' in str(private)
        public = [item for item in provider.sent + provider.edited if item.get('private_reply_to_user_id') is None]
        assert 'private fact' not in str(public)
        assert any(call[0] == 'memory.list' and call[1]['scope'] == 'personal' for call in backend.calls)
    asyncio.run(run())


def test_delete_memory_requires_fresh_confirmation(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Shared memory')))
        await admin.handle_update(callback(provider, button(provider, 'private fact')))
        await admin.handle_update(callback(provider, button(provider, 'Delete entry')))
        assert not any(call[0] == 'memory.delete' for call in backend.calls)
        confirmation = callback(provider, button(provider, 'Confirm'))
        await admin.handle_update(confirmation)
        await admin.handle_update(confirmation)
        assert len([call for call in backend.calls if call[0] == 'memory.delete']) == 1
    asyncio.run(run())


def test_settings_input_has_confirmation_and_restart_notice(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Settings')))
        await admin.handle_update(callback(provider, button(provider, 'Test setting')))
        assert 'restart' in provider.latest['text']
        await admin.handle_update(callback(provider, button(provider, 'Edit')))
        await admin.handle_update(message('3', reply=provider.latest['message_id']))
        assert not any(call[0] == 'settings.set' for call in backend.calls)
        await admin.handle_update(callback(provider, button(provider, 'Confirm')))
        assert ('settings.set', {'key': 'server.test', 'value': 3}, OWNER) in backend.calls
    asyncio.run(run())


def test_new_alert_draft_does_not_send_or_enable_until_saved(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Notifications')))
        await admin.handle_update(callback(provider, button(provider, 'New rule')))
        await admin.handle_update(callback(provider, button(provider, 'Attachment')))
        await admin.handle_update(callback(provider, button(provider, 'Video')))
        assert not any(call[0] == 'alerts.create' for call in backend.calls)
        await admin.handle_update(callback(provider, button(provider, 'Save rule')))
        await admin.handle_update(callback(provider, button(provider, 'Confirm')))
        saved = next(call[1] for call in backend.calls if call[0] == 'alerts.create')
        assert saved['enabled'] is False and saved['media'] == 'video'
        assert saved['destination'] == 'owner'
    asyncio.run(run())


def test_revoked_config_owner_cannot_confirm_existing_action(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Shared memory')))
        await admin.handle_update(callback(provider, button(provider, 'private fact')))
        await admin.handle_update(callback(provider, button(provider, 'Delete entry')))
        pending = callback(provider, button(provider, 'Confirm'))
        admin.cfg.control_user_id = 42
        await admin.handle_update(pending)
        assert not any(call[0] == 'memory.delete' for call in backend.calls)
    asyncio.run(run())


def test_workplace_selection_and_photo_are_explicit_and_route_bound(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Computers and cameras')))
        assert ('workplaces.list', {'chat_id': GROUP}, OWNER) in backend.calls
        assert not any(call[0] == 'workplaces.photo' for call in backend.calls)
        await admin.handle_update(callback(provider, button(provider, 'Select: Кабинет')))
        assert ('workplaces.select', {'id': 'office', 'chat_id': GROUP}, OWNER) in backend.calls
        photo = callback(provider, button(provider, 'Photo: Кабинет'))
        await admin.handle_update(photo)
        await admin.handle_update(photo)
        assert [call for call in backend.calls if call[0] == 'workplaces.photo'] == [
            ('workplaces.photo', {'id': 'office', 'chat_id': GROUP}, OWNER)]
    asyncio.run(run())


def test_old_prompt_reply_cannot_fill_a_later_form(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Settings')))
        await admin.handle_update(callback(provider, button(provider, 'Test setting')))
        await admin.handle_update(callback(provider, button(provider, 'Edit')))
        old_prompt = provider.latest['message_id']
        await admin.handle_update(message('/tools'))
        await admin.handle_update(callback(provider, button(provider, 'Shared memory')))
        await admin.handle_update(callback(provider, button(provider, 'Add entry')))
        new_prompt = provider.latest['message_id']
        assert old_prompt != new_prompt
        assert not await admin.handle_update(message('5', reply=old_prompt))
        assert not any(call[0] == 'memory.add' for call in backend.calls)
        assert await admin.handle_update(message('Remember tea', reply=new_prompt))
        assert next(call for call in backend.calls if call[0] == 'memory.add')[1]['text'] == 'Remember tea'
    asyncio.run(run())


def test_all_computers_remain_reachable_and_refresh_updates_online_state(tmp_path):
    async def run():
        admin, provider, _, _, _ = setup(tmp_path)
        computers = [dict(id=f'pc-{index}', name=f'Computer {index:02}',
                          camera_name='Main camera', connected=True) for index in range(20)]
        async def service(action, payload, actor):
            assert action == 'workplaces.list'
            return dict(ok=True, items=deepcopy(computers), selected_id=None)
        admin.backend = service
        await admin.handle_update(message('/tools'))
        assert 'Computers online: 20' in provider.latest['text']
        await admin.handle_update(callback(provider, button(provider, 'Computers and cameras')))
        seen = set()
        while True:
            buttons = [item for row in provider.latest['reply_markup']['inline_keyboard'] for item in row]
            seen.update(item['text'][len('Select: '):] for item in buttons if item['text'].startswith('Select: '))
            next_page = next((item for item in buttons if item['text'] == 'Next ›'), None)
            if next_page is None:
                break
            assert len(provider.latest['text']) < 3800
            await admin.handle_update(callback(provider, next_page['callback_data']))
        assert seen == {row['name'] for row in computers}
        for row in computers:
            row['connected'] = False
        await admin.handle_update(callback(provider, button(provider, 'Refresh')))
        assert 'Online: 0' in provider.latest['text']
        assert not any(item['text'].startswith('Photo: ') for row in provider.latest['reply_markup']['inline_keyboard'] for item in row)
    asyncio.run(run())


def test_panel_status_and_rule_use_readable_text_without_json(tmp_path):
    async def run():
        admin, provider, _, _, _ = setup(tmp_path)
        async def service(action, payload, actor):
            if action == 'status':
                return dict(ok=True, profiles=5, llm_model='test-model', permissions_enabled=False,
                    workplaces=[dict(name='Living room', connected=True)],
                    api_usage=dict(accounted_usd=2.5, settled_estimate_usd=1.5, reserved_usd=1, limit_usd=18,
                        unsettled_requests=2, providers=[dict(provider='OpenAI', settled_estimate_usd=1.5,
                            reserved_usd=1, unsettled_requests=2)]), notifications=dict(running=True, pending=0))
            return dict(ok=True, items=[])
        admin.backend = service
        await admin.handle_update(message('/tools', chat=OWNER))
        await admin.handle_update(callback(provider, button(provider, 'Status'), chat=OWNER))
        text = provider.latest['text']
        assert 'AI model: test-model' in text and 'Budget counted incl. reserves: $2.50 / $18.00' in text
        assert 'OpenAI: $1.50 estimated' in text and 'Reserved / unconfirmed: $1.00' in text
        assert 'not provider billing' in text and '$2.50 used' not in text
        assert 'Living room' in text and 'Computers online: 1' in text
        assert not any(character in text for character in '{}[]')
        await admin.handle_update(message('/tools', chat=OWNER))
        await admin.handle_update(callback(provider, button(provider, 'Notifications'), chat=OWNER))
        await admin.handle_update(callback(provider, button(provider, 'New rule'), chat=OWNER))
        text = provider.latest['text']
        assert 'Who to detect: Any person' in text and 'State: Disabled' in text
        assert not any(character in text for character in '{}[]')
    asyncio.run(run())


def test_alert_presets_are_inside_the_real_ranges():
    """Every preset the panel offers must pass the real validator."""
    from hub.presence_alerts import RULE_RANGES, validate_rule
    from hub.telegram_admin import _ALERT_FIELDS, _ALERT_PRESETS

    assert set(_ALERT_PRESETS) <= set(_ALERT_FIELDS), "a preset without a panel field is dead code"
    for key, values in _ALERT_PRESETS.items():
        for value in values:
            assert validate_rule({key: value})[key] == value
        if key in RULE_RANGES:
            low, high = RULE_RANGES[key]
            for value in values:
                assert low <= value <= high, f"{key} preset {value} is outside {low}..{high}"
    # clip_seconds has its own integer rule instead of a RULE_RANGES entry.
    assert 'clip_seconds' in _ALERT_PRESETS
    with pytest.raises(ValueError):
        validate_rule({'clip_seconds': 11})


def test_notification_cooldown_floor_is_one_second():
    from hub.presence_alerts import validate_rule

    assert validate_rule({'cooldown_s': 1})['cooldown_s'] == 1
    with pytest.raises(ValueError):
        validate_rule({'cooldown_s': 0})
    with pytest.raises(ValueError):
        validate_rule({'min_stable_s': 301})
    assert validate_rule({'absence_s': 0})['absence_s'] == 0


def test_notification_field_offers_presets_and_saves_one(tmp_path):
    async def run():
        admin, provider, backend, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools', chat=OWNER))
        await admin.handle_update(callback(provider, button(provider, 'Notifications'), chat=OWNER))
        await admin.handle_update(callback(provider, button(provider, 'New rule'), chat=OWNER))
        await admin.handle_update(callback(provider, button(provider, 'Notification cooldown'), chat=OWNER))
        labels = [item['text'] for row in provider.latest['reply_markup']['inline_keyboard'] for item in row]
        assert '1' in labels, 'the one-second preset must be offered'
        assert any(label.startswith('Enter exact value') for label in labels)
        assert 'Allowed range: 1' in provider.latest['text']
        await admin.handle_update(callback(provider, button(provider, '1'), chat=OWNER))
        assert 'Notification cooldown, seconds: 1' in provider.latest['text']
        await admin.handle_update(callback(provider, button(provider, 'Save rule'), chat=OWNER))
        await admin.handle_update(callback(provider, button(provider, 'Confirm'), chat=OWNER))
        saved = next(call for call in backend.calls if call[0] == 'alerts.create')[1]
        assert saved['cooldown_s'] == 1
        await admin.close()

    asyncio.run(run())


def test_home_panel_pairs_buttons_into_rows(tmp_path):
    async def run():
        admin, provider, _, _, _ = setup(tmp_path)
        await admin.handle_update(message('/tools', chat=OWNER))
        rows = provider.latest['reply_markup']['inline_keyboard']
        sizes = [len(row) for row in rows]
        assert max(sizes) == 2, 'the home menu must not be one button per row'
        assert sum(sizes) >= 9
        await admin.close()

    asyncio.run(run())
