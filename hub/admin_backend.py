"""Owner-only administrative operations on real Rowan stores, never an LLM."""
from __future__ import annotations

import asyncio
import math

from hub.admin_settings import CATEGORIES, LIVE, apply_live, catalogue, validated_value
from hub.telegram_admin_state import contains_secret


class AdminBackend:
    def __init__(self, cfg, access, *, runtime, get_room, get_alerts, rename_profile=None,
                 get_workplaces=None, get_provider=None):
        self.cfg, self.access = cfg, access
        self.runtime, self.get_room, self.get_alerts = runtime, get_room, get_alerts
        self.rename_profile = rename_profile
        self.get_workplaces = get_workplaces or (lambda: [])
        self.get_provider = get_provider or (lambda: None)
        self.get_rooms = lambda: [room for row in self.get_workplaces()
                                 if (room := self.get_room(row['id'])) is not None]
        self._lock = asyncio.Lock()

    async def call(self, action, payload, actor_id):
        if not self.access.is_owner(actor_id):
            return {'ok': False, 'error': 'Only the owner can access this panel.'}
        if not isinstance(payload, dict) or contains_secret(payload):
            return {'ok': False, 'error': 'Do not send API keys or passwords to this panel.'}
        async with self._lock:
            try:
                result = await self._call(action, payload, actor_id)
                if not action.endswith('.list') and action != 'status' and result.get('ok'):
                    await asyncio.to_thread(self.access.audit, actor_id, action,
                        {key: value for key, value in payload.items() if key not in {'text', 'value'}})
                return result
            except (ValueError, TypeError) as exc:
                # Validation messages may contain input. Return only bounded,
                # nonsensitive errors; never a full Pydantic config dump.
                from pydantic import ValidationError
                line = 'Invalid setting value.' if isinstance(exc, ValidationError) else str(exc)[:180]
                return {'ok': False, 'error': line if not contains_secret(line) else 'Invalid value.'}
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Owner panel operation failed: %s', action)
                return {'ok': False, 'error': 'Could not complete the action. Check Rowan\'s status.'}

    async def _call(self, action, payload, actor):
        runtime = self.runtime()
        if action.startswith('workplaces.'):
            chat = payload.get('chat_id', actor)
            if type(chat) is not int or chat not in {actor, self.cfg.server.telegram.chat_id}:
                raise ValueError('Invalid panel chat.')
            key = f'workplace:{chat}:{actor}'
            # Connection membership belongs to the event loop; state reads are
            # in-memory snapshots, so no worker thread or I/O is needed here.
            items = self.get_workplaces()
            selected = await asyncio.to_thread(self.access.get_setting, key)
            if action == 'workplaces.list':
                online = [row['id'] for row in items if row.get('connected')]
                return {'ok': True, 'items': items, 'selected_id': selected or (online[0] if len(online) == 1 else None)}
            identifier = str(payload.get('id') or '')
            item = next((row for row in items if row['id'] == identifier), None)
            if item is None:
                raise ValueError('Unknown workplace.')
            if action == 'workplaces.select':
                await asyncio.to_thread(self.access.set_setting, key, identifier)
                return {'ok': True, 'message': 'Selected workplace: ' + item['name']}
            if action == 'workplaces.photo':
                import uuid

                from hub.telegram_control import room_busy
                room = self.get_room(identifier)
                if room is None or room_busy(room):
                    raise ValueError('This client is busy or disconnected.')
                current = asyncio.current_task()
                room._telegram_control_task = current
                try:
                    frame = await room._request_camera_frame_full('panel-' + uuid.uuid4().hex)
                    if isinstance(frame, str):
                        raise ValueError('The camera could not take a photo.')
                    provider = self.get_provider()
                    if provider is None:
                        raise ValueError('Telegram is unavailable.')
                    kwargs = {'private_reply_to_user_id': actor} if chat == actor else {}
                    await provider.send_image(frame.jpeg, 'image/jpeg',
                        caption=item['name'] + ' · ' + item['camera_name'], **kwargs)
                finally:
                    if room._telegram_control_task is current:
                        room._telegram_control_task = None
                return {'ok': True, 'message': 'Photo sent.'}
            raise ValueError('Unknown camera operation.')
        if action == 'status':
            room, alerts = self.get_room(), self.get_alerts()
            workplaces = self.get_workplaces()
            llm = runtime.get('llm')
            budget = getattr(getattr(llm, '_responses', None), 'budget', None)
            usage = await asyncio.to_thread(budget.status) if budget is not None else None
            people = await asyncio.to_thread(runtime['voices'].people) if runtime.get('voices') else {}
            return {'ok': True, 'owner_id': self.access.owner_id,
                    'room_connected': room is not None or any(row.get('connected') for row in workplaces),
                    'workplaces': workplaces,
                    'camera': getattr(room, 'camera_state', None), 'profiles': len(people),
                    'llm_model': getattr(llm, 'model', None), 'api_usage': usage,
                    'notifications': await asyncio.to_thread(alerts.status) if alerts else {'enabled': False},
                    'permissions_enabled': self.cfg.server.permissions_enabled,
                    'message': 'Panel owner: Telegram ID ' + str(self.access.owner_id)}
        if action == 'settings.list':
            items = catalogue(self.cfg)
            for row in items:
                saved = await asyncio.to_thread(self.access.get_setting, 'config:' + row['key'])
                if isinstance(saved, dict) and saved.get('value') != row['value']:
                    row['pending_value'] = saved.get('value')
            category = payload.get('category')
            return {'ok': True, 'categories': [{'id': key, 'label': label} for key, label in CATEGORIES.items()],
                    'settings': [row for row in items if not category or row['category'] == category]}
        if action == 'settings.set':
            key = str(payload.get('key') or '')
            value = validated_value(self.cfg, key, payload.get('value'))
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('The value must be a finite number.')
            await asyncio.to_thread(self.access.set_setting, 'config:' + key, {'value': value})
            if key in LIVE:
                apply_live(self.cfg, key, value, runtime)
            return {'ok': True, 'value': value, 'requires_restart': key not in LIVE,
                    'message': 'Applied.' if key in LIVE else 'Saved. Takes effect after the server restarts.'}
        if action.startswith('memory.'):
            memory = runtime.get('memory')
            if memory is None:
                raise ValueError('Memory storage is unavailable.')
            scope = payload.get('scope', 'shared')
            if scope not in {'shared', 'personal'}:
                raise ValueError('Unknown memory scope.')
            owners = await asyncio.to_thread(memory.people)
            if runtime.get('voices'):
                owners += list(await asyncio.to_thread(runtime['voices'].people))
            owners += ['telegram:' + str(actor)]
            owners += ['telegram:' + str(row['user_id']) for row in await asyncio.to_thread(self.access.users)]
            owners = list(dict.fromkeys(owners))
            owner = '' if scope == 'shared' else str(payload.get('owner_id') or 'telegram:' + str(actor))
            if owner and owner not in owners:
                raise ValueError('Select an existing memory profile.')
            if action == 'memory.list':
                rows = await asyncio.to_thread(memory.admin_entries, owner)
                return {'ok': True, 'owner_id': owner, 'scope': scope,
                        'owners': [{'id': name, 'label': name} for name in owners],
                        'items': [{**row, 'text': row['fact']} for row in rows]}
            if action == 'memory.add':
                await asyncio.to_thread(memory.add, payload.get('text'), owner,
                                        key=payload.get('key', ''), value=payload.get('value'), author=str(actor))
            elif action in {'memory.edit', 'memory.delete'}:
                await asyncio.to_thread(memory.change_entry, str(payload.get('id')), owner,
                    text=payload.get('text'), delete=action == 'memory.delete', value=payload.get('value', ...))
            else:
                raise ValueError('Unknown memory operation.')
            return {'ok': True, 'message': 'Memory updated. Shared settings take priority over personal settings.'}
        if action.startswith('profiles.'):
            registry = runtime.get('voices')
            if registry is None:
                raise ValueError('The profile registry is unavailable.')
            profiles = await asyncio.to_thread(registry.people)
            if action == 'profiles.list':
                voices = await asyncio.to_thread(registry.voice_profiles)
                faces = await asyncio.to_thread(registry.face_profiles)
                return {'ok': True, 'items': [dict(id=name, name=name, role=role,
                    voice_samples=voices.get(name, 0), face_samples=len(faces.get(name, [])))
                    for name, role in profiles.items()]}
            rooms = self.get_rooms()
            if not rooms and self.get_room() is not None:
                rooms = [self.get_room()]
            if any(getattr(room, '_enroll_pending', None) or
                    getattr(room, '_face_selection', None) or
                    (getattr(room, '_enroll_face_task', None) is not None and not room._enroll_face_task.done())
                    for room in rooms):
                raise ValueError('Finish the current face or voice enrollment first.')
            name = str(payload.get('id') or payload.get('name') or '')
            if action != 'profiles.create' and name not in profiles:
                raise ValueError('This profile has already been changed or deleted.')
            if action == 'profiles.rename':
                new = str(payload.get('name') or '')
                if self.rename_profile is None:
                    raise ValueError('Renaming is currently unavailable.')
                await self.rename_profile(name, new)
            elif action == 'profiles.role':
                await asyncio.to_thread(registry.set_role, name, payload.get('role'))
            elif action in {'profiles.create', 'profiles.delete', 'profiles.reset_voice', 'profiles.reset_face'}:
                await asyncio.to_thread(registry.admin_profile, action.split('.')[1], name,
                                        role=payload.get('role', 'user'))
                for room in rooms:
                    room.presence.clear()
                    room.room.tracks.clear()
            else:
                raise ValueError('Unknown profile operation.')
            return {'ok': True, 'message': 'Active profile updated. Original photos, audio and archive history have been preserved.'}
        if action.startswith('alerts.'):
            alerts = self.get_alerts()
            if alerts is None:
                raise ValueError('The notification service is unavailable.')
            if action == 'alerts.list':
                return {'ok': True, 'items': await asyncio.to_thread(alerts.list_rules)}
            if action in {'alerts.create', 'alerts.update'}:
                patch = {key: value for key, value in payload.items() if key != 'id'}
                rule = await asyncio.to_thread(alerts.save_rule, patch, rule_id=payload.get('id'))
                return {'ok': True, 'rule': rule, 'message': 'Notification rule saved.'}
            if action == 'alerts.delete':
                await asyncio.to_thread(alerts.remove_rule, payload.get('id'))
                return {'ok': True, 'message': 'Rule deleted.'}
            raise ValueError('Unknown notification operation.')
        if action == 'audit.list':
            return {'ok': True, 'items': await asyncio.to_thread(self.access.events, min(50, int(payload.get('limit', 20))))}
        raise ValueError('Unknown panel action.')
