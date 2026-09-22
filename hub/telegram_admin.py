"""Owner-only Telegram settings UI with expiring, single-use action handles."""
from __future__ import annotations

import asyncio
import re
import secrets
import time
from dataclasses import dataclass, field

from hub.presence_alerts import RULE_RANGES
from hub.telegram_admin_state import CAPABILITIES, ROLES, contains_secret
from hub.telegram_admin_view import (
    audit_text,
    automation_rule_text,
    calibration_text,
    profile_text,
    rule_text,
    rule_value,
    rules_text,
    scenes_text,
    status_text,
    switches_text,
    user_text,
)
from hub.telegram_admin_view import display as _display

TTL_SECONDS = 900
_COMMAND = re.compile(r'^/(tools|cancel|tools_input)(?:@([A-Za-z0-9_]+))?(?:\s+([\s\S]*))?$')
_ALERT_FIELDS = {
    'workplace_id': ('Workplace and camera', 'workplace'),
    # ТЗ F-702: правило слушает событие (F-301/F-109/F-311), а не только кадр.
    'event': ('What to watch', ('presence', 'person_entered', 'person_left',
                                'unknown_appeared', 'zone_entered', 'sound_event', 'object')),
    'target': ('Who to detect', ('any', 'unknown', 'person')),
    'name': ('Profile name (for person)', 'str'),
    'zone': ('Zone (empty = any)', 'str'),
    'media': ('Attachment', ('photo', 'video')),
    'destination': ('Destination', ('owner', 'group')),
    # ТЗ F-702: канал доставки — Telegram, пуш на телефон или подпись в HUD.
    'channel': ('Delivery channel', ('telegram', 'push', 'hud')),
    # ТЗ F-701/F-702: «дом» — это комната, которая владеет людьми, памятью и
    # правилами; компьютер внутри неё — рабочее место. Поле выбирается из
    # настоящих домов, а не вводится строкой.
    'home_id': ('Home', 'home'),
    'cooldown_s': ('Notification cooldown, seconds', 'float'),
    'min_stable_s': ('Stable presence, seconds', 'float'),
    'min_frames': ('Stable presence, frames', 'int'),
    'absence_s': ('Absence before a new entry, seconds', 'float'),
    'quiet_start': ('Quiet hours start: HH:MM or -', 'str'),
    'quiet_end': ('Quiet hours end: HH:MM or -', 'str'),
    'timezone': ('IANA time zone', 'str'),
    'clip_seconds': ('Video length, seconds', 'int'),
    # ТЗ F-702: «снимать, пока человек не выйдет из кадра». Одно видео всё
    # равно ограничено (clip_seconds), а следующее начинается, пока человек
    # ещё в комнате.
    'record_until_clear': ('Keep recording while the person stays', 'bool'),
}
_ALERT_DEFAULTS = dict(enabled=False, workplace_id='', target='any', name='', media='photo', destination='owner',
                       cooldown_s=300, min_stable_s=0, min_frames=2, absence_s=15,
                       quiet_start='', quiet_end='',
                       # Кадры вместо секунд по умолчанию: быстрый проход перед
                       # камерой — это уже присутствие (см. hub/presence_alerts.py).
                       timezone='America/Chicago', clip_seconds=5, event='presence',
                       channel='telegram', home_id='', zone='', record_until_clear=False)
#: One-tap values for the numeric alert settings, so a rule no longer needs a
#: reply-based text input for the common cases. The ranges come from
#: hub.presence_alerts.RULE_RANGES, so a preset can never be rejected.
_ALERT_PRESETS = {
    'cooldown_s': (1, 5, 15, 30, 60, 300),
    'min_stable_s': (0, 0.5, 1, 2, 5, 10),
    'min_frames': (1, 2, 3, 5),
    'absence_s': (0, 5, 15, 30, 60, 300),
    'clip_seconds': (5, 10, 30, 60),
}


def _clip(value, maximum=3800):
    return str(value).encode('utf-16-le', errors='replace')[:maximum * 2].decode('utf-16-le', errors='ignore')


@dataclass
class _Panel:
    owner: int
    chat: int
    private: bool
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    message_id: int = 0
    expires: float = 0
    generation: int = 0
    pending: dict | None = None
    draft: dict | None = None
    draft_id: str | None = None


class TelegramAdmin:
    """No LLM or arbitrary action names are involved in panel routing.

    ``backend(action, payload, actor_id)`` is an async validated service call.
    The access store is synchronous; disk calls run off the asyncio loop.
    """

    def __init__(self, provider, cfg, access, backend, *, clock=time.monotonic, homes=None):
        self.provider, self.cfg, self.access, self.backend = provider, cfg, access, backend
        #: ТЗ F-701: кто ещё может открыть панель — владельцы домов. ``None``
        #: оставляет прежнее поведение «панель только у админа хаба».
        self.homes = homes
        self.clock = clock
        self.bot_id, self.username = None, ''
        self._panels, self._tokens = {}, {}
        self._lock = asyncio.Lock()
        self._closed = False

    def set_identity(self, bot_id, username):
        if type(bot_id) is int and bot_id > 0 and isinstance(username, str):
            self.bot_id, self.username = bot_id, username

    @property
    def _cfg(self):
        return getattr(getattr(self.cfg, 'server', None), 'telegram', self.cfg)

    def _owner(self, sender):
        return (isinstance(sender, dict) and type(sender.get('id')) is int
                and sender.get('is_bot') is False and self.access.is_owner(sender['id'])
                and sender['id'] == getattr(self._cfg, 'control_user_id', None))

    def _may_panel(self, user_id):
        """ТЗ F-701: админ хаба — везде, владелец дома — в своём чате."""
        if type(user_id) is not int or user_id <= 0:
            return False
        # The owner and the accounts named in ``admin_user_ids`` share the
        # panel; the owner is still the one whose row cannot be changed.
        if self._hub_admin_now(user_id):
            return True
        if self.homes is None:
            return False
        try:
            return bool(self.homes.may_use_panel(user_id)) and not self.access.is_hub_admin(user_id)
        except Exception:  # noqa: BLE001 - unreadable grants are not a permission
            return False

    def _hub_admin_now(self, user_id) -> bool:
        """Is this account a hub admin *by the configuration in force now*.

        The config is the source of truth for both lists: an account the owner
        removed from ``control_user_id`` or from ``admin_user_ids`` loses the
        panel at the next click, instead of keeping it until the process
        restarts. The access store still has to know the account, so a stale
        panel handle alone never confers rights.
        """
        if not self.access.is_hub_admin(user_id):
            return False
        if user_id == getattr(self.access, 'owner_id', None):
            return getattr(self._cfg, 'control_user_id', None) == user_id
        return user_id in set(getattr(self._cfg, 'admin_user_ids', ()) or ())

    def _scope(self, user_id):
        """``None`` для админа хаба, иначе — дома этого аккаунта (F-701)."""
        if self.access.is_hub_admin(user_id) or self.homes is None:
            return None
        try:
            return self.homes.scope(user_id)
        except Exception:  # noqa: BLE001 - без грантов аккаунт видит только себя
            return frozenset()

    # --- ТЗ F-702: куда уходит уведомление -------------------------------

    def _known_chats(self) -> dict:
        """Group chats the bot has seen, as ``{'group:<id>': title}``.

        A Telegram bot cannot ask for the list of its groups, so the hub keeps
        the ones it has actually met (see ``TelegramChat._remember_chat``).
        The panel offers exactly those, which is how a group created later
        becomes a selectable destination.
        """
        if self.access is None:
            return {}
        chats = self.access.get_setting('chats', {})
        if not isinstance(chats, dict):
            return {}
        configured = str(getattr(self._cfg, 'chat_id', '') or '')
        named: dict[str, str] = {}
        for key, value in chats.items():
            chat_id = str(key).strip()
            if not chat_id.lstrip('-').isdigit() or chat_id == configured:
                continue
            title = value.get('title') if isinstance(value, dict) else ''
            named['group:' + chat_id] = str(title or ('Group ' + chat_id))[:60]
        return dict(sorted(named.items(), key=lambda item: item[1].casefold()))

    def _destination_options(self) -> tuple:
        """``owner`` first, then the configured group, then every known group."""
        options = ['owner']
        if isinstance(getattr(self._cfg, 'chat_id', None), int) and self._cfg.chat_id < 0:
            options.append('group')
        options += list(self._known_chats())
        return tuple(options)

    def _alert_fields(self) -> dict:
        """The static rule fields with the destination list filled in."""
        fields = dict(_ALERT_FIELDS)
        caption, _ = fields['destination']
        fields['destination'] = (caption, self._destination_options())
        return fields

    def _destination_label(self, value) -> str:
        people = 0
        if self.access is not None and str(value or '') == 'owner':
            people = len(self.access.private_recipients())
        return rule_value('destination', value, private_to=people, chats=self._known_chats())

    def _field_label(self, key, value) -> str:
        """One field's current value in words (destinations know the groups)."""
        if key == 'destination':
            return self._destination_label(value)
        return rule_value(key, value)

    def _home_options(self) -> dict:
        """The homes this hub serves, as ``{'<home_id>': 'label'}``.

        A home (ТЗ F-701) is the room that owns people, memories and rules; a
        workplace is one computer inside it. Only homes the owner actually has
        are offered: the ones the config names and the ones the connected
        computers are bound to. The database keeps test homes from the earlier
        smoke runs, and those have no business in a picker.
        """
        named: dict[str, str] = {}
        for home in getattr(self.cfg, 'homes', None) or []:
            home_id = str(getattr(home, 'home_id', '') or '').strip()
            if home_id:
                named[home_id] = str(getattr(home, 'name', '') or home_id)
        return named

    async def _homes_of_workplaces(self, panel) -> dict:
        """Homes the connected computers are bound to (ТЗ F-701)."""
        named: dict[str, str] = {}
        result = await self._backend(panel, 'workplaces.list')
        for item in (result or {}).get('items', []) if isinstance(result, dict) else []:
            home_id = str(item.get('home_id') or '').strip()
            if home_id:
                named.setdefault(home_id, home_id)
        return named

    def _allowed_sender(self, sender):
        """A human account allowed to use the panel: hub admin or home owner."""
        return (isinstance(sender, dict) and type(sender.get('id')) is int
                and sender.get('is_bot') is False and self._may_panel(sender['id']))

    def _route(self, chat, owner):
        if not isinstance(chat, dict) or type(chat.get('id')) is not int:
            return False
        if chat.get('type') == 'private':
            # ТЗ F-701: у владельца дома свой чат — свой, а не только у админа.
            return self._may_panel(chat['id']) and chat['id'] == owner
        return (chat.get('type') in {'group', 'supergroup'}
                and chat['id'] == self._cfg.chat_id and self._hub_admin_now(owner))

    def _prune(self):
        now = self.clock()
        self._tokens = {key: value for key, value in self._tokens.items() if value[0].expires > now}
        self._panels = {key: value for key, value in self._panels.items() if value.expires > now}

    def _invalidate(self, panel):
        panel.generation += 1
        self._tokens = {key: value for key, value in self._tokens.items() if value[0] is not panel}

    def _button(self, panel, caption, action):
        token = 'adm:' + secrets.token_urlsafe(18)
        self._tokens[token] = (panel, panel.generation, action)
        return {'text': _clip(caption, 60), 'callback_data': token}

    def _back(self, panel, page='home', **payload):
        return [self._button(panel, '‹ Back', dict(kind='page', page=page, **payload))]

    async def _send(self, panel, text, rows):
        panel.expires = self.clock() + TTL_SECONDS
        kwargs = {'private_reply_to_user_id': panel.owner if panel.private else None,
                  'reply_markup': {'inline_keyboard': rows}}
        if panel.message_id:
            await self.provider.edit_text(_clip(text), message_id=panel.message_id, **kwargs)
        else:
            result = await self.provider.send_text(_clip(text), **kwargs)
            message_id = result.get('message_id') if isinstance(result, dict) else None
            if type(message_id) is not int or message_id <= 0:
                raise RuntimeError('Panel delivery was not confirmed.')
            panel.message_id = message_id

    async def _backend(self, panel, action, payload=None):
        if self._closed or not self._may_panel(panel.owner):
            return {'ok': False, 'error': 'Owner access is no longer confirmed.'}
        try:
            target = self.backend.call if hasattr(self.backend, 'call') else self.backend
            data = dict(payload or {})
            if action.startswith('workplaces.'):
                data['chat_id'] = panel.chat
            result = await target(action, data, panel.owner)
            return result if isinstance(result, dict) else {'ok': False, 'error': 'The service returned an invalid response.'}
        except asyncio.CancelledError:
            raise
        except Exception:
            return {'ok': False, 'error': 'Could not complete the action. Check the service status.'}

    async def _audit(self, panel, event, details=None):
        await asyncio.to_thread(self.access.audit, panel.owner, event, details or {})

    async def _new_panel(self, owner, chat, private):
        old = self._panels.get((owner, chat))
        if old:
            self._invalidate(old)
            old.pending = None
        panel = _Panel(owner=owner, chat=chat, private=private, expires=self.clock() + TTL_SECONDS)
        self._panels[(owner, chat)] = panel
        return panel

    async def handle_update(self, update):
        if self._closed or not isinstance(update, dict):
            return False
        async with self._lock:
            self._prune()
            callback = update.get('callback_query')
            if isinstance(callback, dict):
                data = callback.get('data')
                if not isinstance(data, str) or not data.startswith('adm:'):
                    return False
                await self._callback(callback)
                return True
            message = update.get('message')
            if not isinstance(message, dict):
                return False
            text = message.get('text')
            if not isinstance(text, str):
                return False
            command = _COMMAND.fullmatch(text.strip())
            if command and command[2] and command[2].casefold() != self.username.casefold():
                return False
            sender, chat = message.get('from'), message.get('chat')
            if (not self._allowed_sender(sender) or not self._route(chat, sender['id'])
                    or message.get('sender_chat') or message.get('forward_origin') or message.get('forward_date')):
                return bool(command)
            panel = self._panels.get((sender['id'], chat['id']))
            if command and command[1] == 'tools' and not command[3]:
                panel = await self._new_panel(sender['id'], chat['id'], chat['type'] == 'private')
                await self._audit(panel, 'panel.open')
                await self._page(panel, 'home')
                return True
            if command and command[1] == 'cancel':
                if panel:
                    panel.pending = panel.draft = None
                    await self._page(panel, 'home', notice='Input and unsaved draft cancelled.')
                return True
            if panel and panel.pending:
                parent = message.get('reply_to_message')
                parent_sender = parent.get('from') if isinstance(parent, dict) else None
                reply = (isinstance(parent, dict) and type(parent.get('message_id')) is int
                         and parent['message_id'] == panel.message_id and isinstance(parent_sender, dict)
                         and type(parent_sender.get('id')) is int and parent_sender['id'] == self.bot_id
                         and parent_sender.get('is_bot') is True)
                explicit = command and command[1] == 'tools_input' and command[3]
                if explicit:
                    nonce, separator, text = command[3].partition(' ')
                    reply = bool(separator and secrets.compare_digest(nonce, panel.nonce))
                if reply:
                    await self._input(panel, text)
                    return True
            return bool(command)

    async def _callback(self, callback):
        sender, message = callback.get('from'), callback.get('message')
        chat = message.get('chat') if isinstance(message, dict) else None
        token = self._tokens.get(callback.get('data'))
        valid = (self._allowed_sender(sender) and isinstance(message, dict) and self._route(chat, sender['id'])
                 and type(message.get('message_id')) is int and token is not None)
        if valid:
            panel, generation, action = token
            valid = (panel.owner == sender['id'] and panel.chat == chat['id']
                     and panel.message_id == message['message_id'] and panel.generation == generation
                     and self._panels.get((panel.owner, panel.chat)) is panel
                     and panel.expires > self.clock())
        callback_id = callback.get('id')
        if not valid:
            if isinstance(callback_id, str):
                await self.provider.answer_callback(callback_id, 'This button is unavailable. The owner can reopen /tools.', show_alert=True)
            return
        # Claim before any await: a replay cannot execute the action twice.
        self._tokens.pop(callback['data'], None)
        self._invalidate(panel)
        if isinstance(callback_id, str):
            await self.provider.answer_callback(callback_id)
        panel.pending = None
        await self._dispatch(panel, action)

    async def _dispatch(self, panel, action):
        kind = action['kind']
        if kind == 'page':
            await self._page(panel, action['page'], **{k: v for k, v in action.items() if k not in {'kind', 'page'}})
        elif kind == 'prompt':
            await self._prompt(panel, action)
        elif kind == 'confirm':
            await self._confirm(panel, action['action'], action.get('label', 'Confirm this action.'))
        elif kind == 'execute':
            await self._execute(panel, action)
        elif kind == 'draft':
            if panel.draft is None:
                await self._page(panel, 'alerts', notice='This draft has expired.')
            else:
                panel.draft[action['key']] = action['value']
                await self._page(panel, 'alert_draft')

    async def _prompt(self, panel, action):
        self._invalidate(panel)
        panel.nonce = secrets.token_urlsafe(12)
        panel.pending = dict(action)
        text = action.get('label', 'Enter a value.')
        text += '\n\nUse Reply on this message to enter a value. Regular chat messages will not change settings.'
        text += f'\nAlternatively: /tools_input {panel.nonce} value\nCancel: /cancel. Do not send keys or passwords here.'
        # A Telegram reply carries only message_id, not the edited version of a
        # message. Every form therefore needs a fresh bot message, even retries.
        panel.message_id = 0
        await self._send(panel, text, [self._back(panel, action.get('back', 'home'), **action.get('back_payload', {}))])

    async def _confirm(self, panel, action, label):
        self._invalidate(panel)
        await self._send(panel, label + '\n\nThe change will only take effect after confirmation.', [
            [self._button(panel, 'Confirm', dict(action, kind='execute'))], self._back(panel)])

    async def _input(self, panel, text):
        action, panel.pending = panel.pending, None
        if not isinstance(text, str) or not text.strip() or len(text) > 4000 or contains_secret(text):
            await self._prompt(panel, dict(action, label='Enter non-empty text up to 4000 characters, without keys or passwords.'))
            return
        value = text.strip()
        try:
            value_type = action.get('type', 'str')
            if value_type == 'int':
                value = int(value)
            elif value_type in {'float', 'number'}:
                value = float(value)
                if not -float('inf') < value < float('inf'):
                    raise ValueError()
            elif value_type == 'bool':
                if value.casefold() not in {'true', 'false', 'да', 'нет', '1', '0'}:
                    raise ValueError()
                value = value.casefold() in {'true', 'да', '1'}
            if action.get('draft_key'):
                if panel.draft is None:
                    raise ValueError()
                panel.draft[action['draft_key']] = '' if value == '-' else value
                await self._page(panel, 'alert_draft')
                return
            payload = dict(action.get('payload', {}))
            if action['action'] == 'users.add':
                user_id, _, label = str(value).partition(' ')
                payload.update(user_id=int(user_id), label=label.strip(), role='member')
            else:
                payload[action.get('field', 'text')] = value
            execute = dict(kind='execute', action=action['action'], payload=payload,
                           back=action.get('back', 'home'), back_payload=action.get('back_payload', {}))
            if action.get('confirm'):
                await self._confirm(panel, execute, action.get('confirm_label', 'Save this change?'))
            else:
                await self._execute(panel, execute)
        except (ValueError, TypeError, OverflowError):
            await self._prompt(panel, dict(action, label='Invalid format. ' + action.get('label', '')))

    async def _execute(self, panel, action):
        if self._closed or not self._may_panel(panel.owner):
            return
        name, payload = action['action'], dict(action.get('payload', {}))
        if name.startswith('users.'):
            # ТЗ F-701: аккаунты Telegram — общие для хаба, их ведёт админ хаба.
            if not self.access.is_owner(panel.owner):
                result = {'ok': False, 'error': 'Only the hub administrator can change Telegram users.'}
            else:
                result = await self._execute_users(panel, name, payload)
        else:
            result = await self._backend(panel, name, payload)
            await self._audit(panel, name, {'ok': result.get('ok') is True, 'id': payload.get('id'),
                                          'scope': payload.get('scope'), 'key': payload.get('key')})
        notice = _display(result.get('message', 'Saved.')) if result.get('ok') is True else _display(result.get('error', 'Could not save.'))
        if result.get('ok') is True and name.startswith('alerts.'):
            panel.draft = panel.draft_id = None
        await self._page(panel, action.get('back', 'home'), notice=notice, **action.get('back_payload', {}))

    async def _execute_users(self, panel, name, payload):
        try:
            if name == 'users.remove':
                await asyncio.to_thread(self.access.remove_user, payload['user_id'])
            else:
                await asyncio.to_thread(self.access.set_user, payload['user_id'], payload['role'],
                                        payload.get('capabilities'), payload.get('label', ''))
            await self._audit(panel, name, payload)
            result = {'ok': True}
        except (ValueError, KeyError, TypeError):
            result = {'ok': False, 'error': 'Invalid user, role or permissions. The owner cannot be changed.'}
        return result

    async def _page(self, panel, page, *, notice='', offset=0, **payload):
        self._invalidate(panel)
        panel.pending = None
        rows, text = [], ''
        button = lambda caption, **action: self._button(panel, caption, action)
        if page == 'home':
            scope = self._scope(panel.owner)
            if scope is None:
                text = 'Rowan control panel · owner\nOnly you can use these buttons. They expire after 15 minutes.\nPersonal memory opens in your private chat, even from a group.'
            else:
                # ТЗ F-701: владелец дома видит только свои дома.
                text = ('Rowan control panel · ' + ', '.join(sorted(scope)) + '\n'
                        'Only the homes named above are shown here. Buttons expire after 15 minutes.')
            places = await self._backend(panel, 'workplaces.list')
            online = [item for item in places.get('items', []) if item.get('connected')]
            text += f'\n\nComputers online: {len(online)}'
            text += ''.join('\n• ' + _clip(item.get('name') or item['id'], 80) for item in online[:8])
            if len(online) > 8:
                text += '\nOpen Computers and cameras for the full list.'
            if places.get('ok') is False:
                text += '\n' + _display(places.get('error'))
            menu = [('Status', 'status'), ('Settings', 'settings'), ('Shared memory', 'memory'),
                    ('Personal memory', 'personal'), ('People profiles', 'profiles'),
                    ('Telegram users', 'users'), ('Computers and cameras', 'workplaces'),
                    ('Notifications', 'alerts'), ('Audit log', 'audit'),
                    ('Calibration', 'calibration'), ('Wall switches', 'switches'),
                    ('Scenes', 'scenes'), ('Room rules', 'rules'),
                    ('Homes and owners', 'homes')]
            if scope is not None:
                # Домашняя панель: то, что относится к дому, и ничего чужого.
                menu = [row for row in menu
                        if row[1] in {'status', 'workplaces', 'switches', 'scenes', 'rules'}]
            # Two buttons per row keeps the whole panel one screen tall on a phone.
            for index in range(0, len(menu), 2):
                rows.append([button(label, kind='page', page=name) for label, name in menu[index:index + 2]])
        elif page == 'personal':
            if not panel.private:
                private = await self._new_panel(panel.owner, panel.owner, True)
                try:
                    await self._page(private, 'memory', scope='personal')
                    text = 'Personal memory is open in your private chat with the bot.'
                except Exception:
                    text = 'Open a private chat with the bot and send /tools first. Personal data is not shown in the group.'
                rows.append(self._back(panel))
            else:
                await self._page(panel, 'memory', scope='personal')
                return
        elif page == 'status':
            result = await self._backend(panel, 'status')
            text = status_text(result)
            rows = [[button('Refresh', kind='page', page='status')], self._back(panel)]
        elif page == 'homes':
            # ТЗ F-701: админ хаба видит все дома и раздаёт их владельцам.
            result = await self._backend(panel, 'homes.list')
            text = ('Homes and their Telegram owners\n'
                    'An owner opens /tools in their own private chat and sees only their homes.')
            if result.get('ok') is False:
                text += '\n' + _display(result.get('error'))
            items = result.get('items', [])
            if not items:
                text += ('\n\nNo homes are known yet. Add a home to the hub config, or let a '
                         'client connect, then tap Refresh.')
            for item in items:
                home_id = str(item.get('home_id') or '')
                owners = ', '.join(str(value) for value in item.get('owners', [])) or 'nobody yet'
                text += '\n\n' + _clip(f"{item.get('name') or home_id} ({home_id})\nOwners: {owners}", 240)
                rows.append([button('Grant: ' + str(item.get('name') or home_id), kind='prompt',
                                    label=f'Enter the Telegram ID that owns {home_id}',
                                    type='int', action='homes.grant', field='user_id',
                                    payload={'home_id': home_id}, back='homes')])
                for user_id in item.get('owners', []):
                    rows.append([button(f'Revoke {user_id} from {home_id}', kind='execute',
                                        action='homes.revoke', payload={'home_id': home_id,
                                                                        'user_id': user_id},
                                        back='homes')])
            rows.append([button('Refresh', kind='page', page='homes'), self._back(panel)])
        elif page == 'workplaces':
            result = await self._backend(panel, 'workplaces.list')
            selected = result.get('selected_id')
            items = sorted(result.get('items', []), key=lambda item: (not item.get('connected'), str(item.get('name') or item['id']).casefold()))
            active = next((item for item in items if item['id'] == selected), None)
            text = 'Computers and cameras\nYour selection is saved separately for this Telegram chat.\nSelected computer: ' + _display((active.get('name') or active['id']) if active else selected or 'None')
            text += f'\nOnline: {sum(bool(item.get("connected")) for item in items)} · Known computers: {len(items)}'
            if result.get('ok') is False:
                text += '\n' + _display(result.get('error'))
            if not items:
                text += '\n\nNo computers have connected yet. Start Rowan on a PC connected to this server, then tap Refresh.'
            offset = max(0, min(int(offset), max(0, (len(items) - 1) // 7 * 7)))
            for item in items[offset:offset + 7]:
                label = str(item.get('name') or item['id'])
                camera = str(item.get('camera_name') or 'Camera')
                status = 'online' if item.get('connected') else 'offline'
                text += '\n\n' + _clip(f'{label} — {status}\nCamera: {camera}', 240)
                rows.append([button(('✓ ' if item['id'] == selected else '') + 'Select: ' + label,
                                    kind='execute', action='workplaces.select', payload={'id': item['id']}, back='workplaces')])
                if item.get('connected'):
                    rows.append([button('Photo: ' + label + ' / ' + camera, kind='execute',
                                        action='workplaces.photo', payload={'id': item['id']}, back='workplaces')])
            navigation = []
            if offset:
                navigation.append(button('‹ Previous', kind='page', page='workplaces', offset=offset - 7))
            if offset + 7 < len(items):
                navigation.append(button('Next ›', kind='page', page='workplaces', offset=offset + 7))
            if navigation:
                rows.append(navigation)
                text += f'\n\nPage {offset // 7 + 1}/{(len(items) + 6) // 7}'
            rows += [[button('Refresh', kind='page', page='workplaces', offset=offset)], self._back(panel)]
        elif page == 'settings':
            result = await self._backend(panel, 'settings.list', payload)
            category_name = next((item['label'] for item in result.get('categories', []) if item['id'] == payload.get('category')), 'Choose a category.')
            text = 'Settings\n' + category_name
            if result.get('ok') is False:
                text += '\n' + _display(result.get('error'))
            for category in result.get('categories', []):
                rows.append([button(category.get('label', category['id']), kind='page', page='settings', category=category['id'])])
            for item in result.get('settings', result.get('items', [])):
                rows.append([button(item.get('label', item['key']), kind='page', page='setting', item=item, category=payload.get('category'))])
            rows.append(self._back(panel))
        elif page == 'setting':
            item = payload['item']
            text = f"{item.get('label', item['key'])}\n{item.get('description', '')}\nCurrent value: {_display(item.get('value'))}"
            if 'pending_value' in item:
                text += '\nSaved for the next start: ' + _display(item['pending_value'])
            if item.get('requires_restart'):
                text += '\nApplies after the next server restart.'
            if item.get('editable', True):
                choices = item.get('choices')
                if item.get('type') in {'bool', 'boolean'}:
                    choices = [True, False]
                common = dict(action='settings.set', back='settings', back_payload={'category': payload.get('category')})
                if choices:
                    for value in choices:
                        rows.append([button(_display(value), kind='confirm', label=f"Change {item.get('label', item['key'])} to {_display(value)}?",
                                            action=dict(**common, payload={'key': item['key'], 'value': value}))])
                else:
                    label = 'New value. ' + item.get('description', '')
                    if item.get('min') is not None or item.get('max') is not None:
                        label += f" Range: {item.get('min', '—')} … {item.get('max', '—')}."
                    rows.append([button('Edit', kind='prompt', label=label, type=item.get('type', 'str'),
                                        field='value', payload={'key': item['key']}, confirm=True, **common)])
            rows.append(self._back(panel, 'settings', category=payload.get('category')))
        elif page in {'memory', 'memory_item'}:
            scope = payload.get('scope', 'shared')
            if scope == 'personal' and not panel.private:
                await self._page(panel, 'personal')
                return
            context = {key: payload[key] for key in ('scope', 'owner_id') if key in payload}
            context.setdefault('scope', scope)
            if page == 'memory':
                result = await self._backend(panel, 'memory.list', context)
                text = ('Personal memory' if scope == 'personal' else 'Shared memory')
                if result.get('ok') is False:
                    text += '\n' + _display(result.get('error'))
                owner_id = result.get('owner_id', result.get('owner'))
                if owner_id:
                    context['owner_id'] = owner_id
                    text += '\nMemory owner: ' + _display(owner_id)
                for owner in result.get('owners', []) if scope == 'personal' else []:
                    rows.append([button(owner.get('label', owner['id']), kind='page', page='memory', scope='personal', owner_id=owner['id'])])
                for item in result.get('items', []):
                    rows.append([button(_display(item.get('text', item.get('value', item['id'])))[:55], kind='page', page='memory_item', item=item, **context)])
                rows.append([button('Add entry', kind='prompt', action='memory.add', label='Enter the memory text.',
                                    payload=context, back='memory', back_payload=context)])
                rows.append(self._back(panel))
            else:
                item = payload['item']
                text = _display(item.get('text', item.get('value', item)))
                mutation = dict(context, id=item['id'])
                field = 'value' if item.get('key') else 'text'
                if item.get('key'):
                    mutation['key'] = item['key']
                    text = str(item['key']) + ': ' + _display(item.get('value'))
                rows = [[button('Edit', kind='prompt', action='memory.edit', label='Enter the new memory text.',
                                field=field, payload=mutation, back='memory', back_payload=context)],
                        [button('Delete entry', kind='confirm', label='Delete the selected memory entry?',
                                action=dict(action='memory.delete', payload=mutation, back='memory', back_payload=context))],
                        self._back(panel, 'memory', **context)]
        elif page in {'profiles', 'profile'}:
            if page == 'profiles':
                result = await self._backend(panel, 'profiles.list')
                text = 'People profiles · voice / face\nDeleting a profile keeps its observation archive.'
                for item in result.get('items', []):
                    rows.append([button(item.get('name', item['id']), kind='page', page='profile', item=item)])
                rows += [[button('Add profile', kind='prompt', action='profiles.create', field='name',
                                 label='Enter the new profile name.', payload={'role': 'user'}, back='profiles')], self._back(panel)]
            else:
                item = payload['item']
                text = profile_text(item)
                rows.append([button('Rename', kind='prompt', action='profiles.rename', field='name',
                                    label='Enter the new name. This will not merge it with an existing profile.',
                                    payload={'id': item['id']}, back='profiles', confirm=True)])
                for role in ('user', 'trusted', 'admin'):
                    rows.append([button('Role: ' + role, kind='confirm', label=f'Assign the {role} role to this profile?',
                                        action=dict(action='profiles.role', payload={'id': item['id'], 'role': role}, back='profiles'))])
                for label, verb in [('Reset voice', 'reset_voice'), ('Reset face', 'reset_face'), ('Delete profile', 'delete')]:
                    rows.append([button(label, kind='confirm', label=f"{label}: {item['id']}? The observation archive will be kept.",
                                        action=dict(action='profiles.' + verb, payload={'id': item['id']}, back='profiles'))])
                rows.append(self._back(panel, 'profiles'))
        elif page in {'users', 'user'}:
            users = await asyncio.to_thread(self.access.users)
            if page == 'users':
                text = 'Telegram users\nowner: all permissions, cannot be changed. admin: all capabilities. operator: chat, images, camera and PC. member: chat and images. blocked: no access.\nOnly the owner can use this panel.'
                for user in users:
                    rows.append([button(f"{user.get('label') or user['user_id']} · {user['role']}", kind='page', page='user', user_id=user['user_id'])])
                rows += [[button('Add by Telegram ID', kind='prompt', action='users.add', label='Enter a numeric Telegram user ID, optionally followed by a space and a label.', confirm=True, back='users')], self._back(panel)]
            else:
                user = next((row for row in users if row['user_id'] == payload['user_id']), None)
                if user is None:
                    await self._page(panel, 'users', notice='This user no longer exists.')
                    return
                capabilities = {key: self.access.allows(user['user_id'], key) for key in CAPABILITIES}
                text = user_text(user, capabilities)
                if not self.access.is_owner(user['user_id']):
                    for role in ROLES:
                        rows.append([button('Role: ' + role, kind='confirm', label=f"Set the role to {role} for {user['user_id']}?",
                                            action=dict(action='users.set', payload=dict(user_id=user['user_id'], role=role, capabilities={}, label=user.get('label', '')), back='users'))])
                    for capability in CAPABILITIES:
                        allowed = capabilities[capability]
                        overrides = dict(user.get('capabilities', {}), **{capability: not allowed})
                        rows.append([button(f"{'✓' if allowed else '—'} {capability}: toggle", kind='confirm', label=f"{'Disable' if allowed else 'Enable'} {capability} for {user.get('label') or user['user_id']}?",
                                            action=dict(action='users.set', payload=dict(user_id=user['user_id'], role=user['role'], capabilities=overrides, label=user.get('label', '')), back='users'))])
                    rows.append([button('Remove granted access', kind='confirm', label='Remove granted access? Basic member permissions will remain in the group. Choose blocked to deny all access.',
                                        action=dict(action='users.remove', payload={'user_id': user['user_id']}, back='users'))])
                rows.append(self._back(panel, 'users'))
        elif page in {'rules', 'rule'}:
            if page == 'rules':
                result = await self._backend(panel, 'rules.list', {'home': payload.get('home')})
                text = rules_text(result)
                for item in result.get('items', []):
                    mark = '✓' if item.get('enabled') else '—'
                    rows.append([button(f"{mark} {item.get('name') or item.get('id')}",
                                        kind='page', page='rule', item=item)])
                rows += [[button('Refresh', kind='page', page='rules')], self._back(panel)]
            else:
                item = payload['item']
                text = automation_rule_text(item)
                toggle = 'Disable' if item.get('enabled') else 'Enable'
                rows = [[button(toggle, kind='confirm',
                                label=f'{toggle} this rule?',
                                action=dict(action='rules.update',
                                            payload={'id': item['id'],
                                                     'enabled': not item.get('enabled')},
                                            back='rules'))],
                        [button('Delete rule', kind='confirm',
                                label='Delete this rule? The room stops doing it at once.',
                                action=dict(action='rules.delete',
                                            payload={'id': item['id']}, back='rules'))],
                        self._back(panel, 'rules')]
        elif page in {'alerts', 'alert', 'alert_draft', 'alert_field'}:
            if page == 'alerts':
                result = await self._backend(panel, 'alerts.list')
                text = 'Presence notifications\nNew rules are disabled. Configure and save a rule, then enable it.'
                for item in result.get('items', []):
                    mark = '✓' if item.get('enabled') else '—'
                    cooldown = f"{float(item.get('cooldown_s') or 0):g}s"
                    rows.append([button(f"{mark} {rule_value('target', item.get('target'))} {item.get('name', '')} "
                                        f"· {rule_value('media', item.get('media'))} · {cooldown}",
                                        kind='page', page='alert', item=item)])
                rows += [[button('New rule', kind='page', page='alert_draft', new=True)], self._back(panel)]
            elif page == 'alert':
                item = payload['item']
                text = rule_text(item, self._alert_fields(), chats=self._known_chats())
                rows = [[button('Edit settings', kind='page', page='alert_draft', item=item)],
                        [button('Disable' if item.get('enabled') else 'Enable', kind='confirm',
                                label='Change this notification rule? An enabled rule sends photos or videos automatically.',
                                action=dict(action='alerts.update', payload={'id': item['id'], 'enabled': not item.get('enabled')}, back='alerts'))],
                        [button('Delete rule', kind='confirm', label='Delete this notification rule?',
                                action=dict(action='alerts.delete', payload={'id': item['id']}, back='alerts'))], self._back(panel, 'alerts')]
            elif page == 'alert_draft':
                if payload.get('new') or 'item' in payload:
                    item = payload.get('item', {})
                    panel.draft = {key: item.get(key, value) for key, value in _ALERT_DEFAULTS.items()}
                    panel.draft_id = item.get('id')
                if panel.draft is None:
                    await self._page(panel, 'alerts', notice='This draft has expired.')
                    return
                text = rule_text(panel.draft, self._alert_fields(), draft=True,
                                 chats=self._known_chats())
                fields = list(self._alert_fields().items())
                for index in range(0, len(fields), 2):
                    rows.append([button(label + ': ' + self._field_label(key, panel.draft[key]),
                                        kind='page', page='alert_field', key=key)
                                 for key, (label, _) in fields[index:index + 2]])
                data = dict(panel.draft)
                if panel.draft_id:
                    data['id'] = panel.draft_id
                else:
                    data['enabled'] = False
                rows += [[button('Save rule', kind='confirm', label='Save these notification settings?',
                                 action=dict(action='alerts.update' if panel.draft_id else 'alerts.create', payload=data, back='alerts'))], self._back(panel, 'alerts')]
            else:
                key = payload['key']
                label, value_type = self._alert_fields()[key]
                text = label
                if value_type == 'workplace':
                    result = await self._backend(panel, 'workplaces.list')
                    items = result.get('items', []) if isinstance(result, dict) else []
                    rows = [[button('Any workplace', kind='draft', key=key, value='')]]
                    # Only what is connected right now: a rule bound to a PC that
                    # is switched off would look armed and be unable to fire. A
                    # previously chosen one stays reachable, marked as offline.
                    current = str(panel.draft.get(key) or '')
                    for item in items:
                        if item.get('connected'):
                            rows.append([button(item.get('name') or item['id'], kind='draft',
                                                key=key, value=item['id'])])
                    for item in items:
                        if str(item.get('id')) == current and not item.get('connected'):
                            rows.append([button('Offline · ' + str(item.get('name') or item['id']),
                                                kind='draft', key=key, value=item['id'])])
                    if not any(item.get('connected') for item in items):
                        text += '\n\nNo computer is connected right now; a rule that names one will wait for it.'
                    rows.append(self._back(panel, 'alert_draft'))
                elif value_type == 'home':
                    named = {**self._home_options(), **(await self._homes_of_workplaces(panel))}
                    rows = [[button('Every home', kind='draft', key=key, value='')]]
                    for home_id, label in sorted(named.items()):
                        rows.append([button(label, kind='draft', key=key, value=home_id)])
                    text += ('\n\nA home is the room that owns people, memories and rules '
                             '(ТЗ F-701); a workplace is one computer inside it.')
                    rows.append(self._back(panel, 'alert_draft'))
                elif isinstance(value_type, tuple):
                    rows = [[button(self._field_label(key, value), kind='draft', key=key, value=value)]
                            for value in value_type]
                    rows.append(self._back(panel, 'alert_draft'))
                elif value_type == 'bool':
                    # On/Off carry real booleans, so the draft cannot turn a
                    # setting into the string "False" and fail validation later.
                    rows = [[button('On', kind='draft', key=key, value=True),
                             button('Off', kind='draft', key=key, value=False)]]
                    if key == 'record_until_clear':
                        text += ('\n\nOn keeps one video after another going while '
                                 'somebody is still in the room; each video is '
                                 '"Video length" seconds long.')
                    rows.append(self._back(panel, 'alert_draft'))
                else:
                    presets = _ALERT_PRESETS.get(key)
                    if not presets:
                        await self._prompt(panel, dict(kind='prompt', label=label, type=value_type,
                                                       draft_key=key, back='alert_draft'))
                        return
                    low, high = RULE_RANGES.get(key, (None, None))
                    if low is not None:
                        text += f'\nAllowed range: {low:g} … {high:g}'
                    text += '\nPick a preset, or enter an exact value.'
                    row: list = []
                    for value in presets:
                        row.append(button(_display(value), kind='draft', key=key, value=value))
                        if len(row) == 3:
                            rows.append(row)
                            row = []
                    if row:
                        rows.append(row)
                    rows.append([button('Enter exact value', kind='prompt', label=label, type=value_type,
                                        draft_key=key, back='alert_draft')])
                    rows.append(self._back(panel, 'alert_draft'))
        elif page == 'audit':
            events = await asyncio.to_thread(self.access.events, 20)
            text = audit_text(events)
            rows = [[button('Refresh', kind='page', page='audit')], self._back(panel)]
        elif page == 'calibration':
            result = await self._backend(panel, 'calibration.list')
            text = calibration_text(result)
            rows = [[button('Refresh', kind='page', page='calibration')], self._back(panel)]
        elif page == 'switches':
            result = await self._backend(panel, 'devices.list')
            text = switches_text(result)
            for item in (result.get('items') or [])[:8]:
                rows.append([button('Set angles: ' + str(item.get('name')), kind='prompt',
                                    action='devices.calibrate', payload={'device_id': item['id']},
                                    label=f"Send the servo angles of {item.get('name')} as "
                                          'closed,open in degrees, for example 0,90',
                                    type='str', back='switches')])
            rows += [[button('Refresh', kind='page', page='switches')], self._back(panel)]
        elif page == 'scenes':
            result = await self._backend(panel, 'scenes.list', payload)
            text = scenes_text(result)
            for item in (result.get('items') or [])[:8]:
                rows.append([button(f"Run: {item.get('name')}", kind='confirm',
                                    label=f"Run the scene {item.get('name')}?",
                                    action=dict(action='scenes.run',
                                                payload={'scene_id': item['scene_id']},
                                                back='scenes'))])
            rows += [[button('Refresh', kind='page', page='scenes')], self._back(panel)]
        else:
            text, rows = 'This section is no longer available.', [self._back(panel)]
        if len(rows) > 18 and page != 'workplaces':
            count = len(rows)
            offset = max(0, min(int(offset), ((count - 1) // 16) * 16))
            rows = rows[offset:offset + 16]
            navigation = []
            if offset:
                navigation.append(button('‹ Previous', kind='page', page=page, offset=max(0, offset - 16), **payload))
            if offset + 16 < count:
                navigation.append(button('Next ›', kind='page', page=page, offset=offset + 16, **payload))
            rows.extend([navigation, self._back(panel)])
            text += f'\nPage {offset // 16 + 1}/{(count + 15) // 16}.'
        if notice:
            text = _clip(notice, 500) + '\n\n' + text
        await self._send(panel, text, rows)

    async def close(self):
        self._closed = True
        self._tokens.clear()
        self._panels.clear()
