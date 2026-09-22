"""Owner-only administrative operations on real Rowan stores, never an LLM."""
from __future__ import annotations

import asyncio
import math

from hub import automation, telegram_audit
from hub.admin_settings import CATEGORIES, LIVE, apply_live, catalogue, validated_value
from hub.telegram_admin_state import contains_secret


class AdminBackend:
    #: Actions worth a row in the hub's ``audit`` table (ТЗ F-706): privileged
    #: operations, settings changes and deleting data.
    AUDITED = ('settings.', 'profiles.', 'memory.', 'devices.', 'scenes.', 'alerts.', 'users.',
               'rules.')
    #: Раздача домов владельцам (ТЗ F-701) — тоже привилегированное действие.
    AUDITED = AUDITED + ('homes.',)
    #: Что доступно владельцу дома (ТЗ F-701): его комнаты и привязанное к ним.
    #: Всё остальное — настройки хаба, люди, память, аккаунты Telegram, аудит,
    #: калибровка — относится ко всему хабу целиком: у этих данных нет дома, и
    #: «показать только своё» означало бы показать чужое под своим именем.
    HOME_SCOPED = ('workplaces.', 'devices.', 'scenes.', 'rules.')

    def __init__(self, cfg, access, *, runtime, get_room, get_alerts, rename_profile=None,
                 get_workplaces=None, get_provider=None, get_decisions=None, get_switches=None,
                 get_wizard=None, get_scenes=None, get_tools=None, get_audit=None,
                 get_scope=None, get_workplace_home=None, get_home_owners=None,
                 get_rules=None):
        self.cfg, self.access = cfg, access
        self.runtime, self.get_room, self.get_alerts = runtime, get_room, get_alerts
        self.rename_profile = rename_profile
        self.get_workplaces = get_workplaces or (lambda: [])
        self.get_provider = get_provider or (lambda: None)
        self.get_decisions = get_decisions or (lambda: None)
        self.get_switches = get_switches or (lambda: None)
        self.get_wizard = get_wizard or (lambda: None)
        self.get_scenes = get_scenes or (lambda: None)
        self.get_tools = get_tools or (lambda: None)
        self.get_audit = get_audit or (lambda: None)
        #: ``actor -> None (весь хаб) | frozenset(дома)`` (ТЗ F-701). По
        #: умолчанию — весь хаб: панель без многодомности работает как раньше.
        self.get_scope = get_scope or (lambda actor: None)
        self.get_workplace_home = get_workplace_home or (lambda client_id: None)
        self.get_home_owners = get_home_owners or (lambda: None)
        #: ТЗ F-419: таблица ``rules`` для панели — словами, а не JSON-ом.
        self.get_rules = get_rules or (lambda: None)
        self.get_rooms = lambda: [room for row in self.get_workplaces()
                                 if (room := self.get_room(row['id'])) is not None]
        self._lock = asyncio.Lock()

    async def call(self, action, payload, actor_id):
        scope = self.get_scope(actor_id)
        # ``None`` scope means the whole hub, and that is exactly what a hub
        # admin gets: the owner and the accounts named in
        # ``server.telegram.admin_user_ids`` (ТЗ F-701). Refusing everyone but
        # the owner here was what left the named admins with an empty panel.
        if scope is None and not self.access.is_hub_admin(actor_id):
            return {'ok': False, 'error': 'Only the owner can access this panel.'}
        if scope is not None and not scope:
            # Аккаунт без домов: панель открыта, но показывать нечего.
            return {'ok': False, 'error': 'No homes are assigned to this account yet.'}
        if not isinstance(payload, dict) or contains_secret(payload):
            return {'ok': False, 'error': 'Do not send API keys or passwords to this panel.'}
        if scope is not None:
            # ТЗ F-701: «у каждого владельца дома свой чат» — если дом у него
            # один, действие без явного дома относится к нему; с несколькими
            # домами нужен явный home_id, иначе панель гадает.
            if (len(scope) == 1 and str(action).startswith(self.HOME_SCOPED)
                    and not self._home_of(action, payload)):
                payload = {**payload, 'home_id': next(iter(scope))}
            refusal = self._scope_refusal(action, payload, scope)
            if refusal:
                return {'ok': False, 'error': refusal}
        async with self._lock:
            try:
                result = await self._call(action, payload, actor_id)
                if scope is not None and result.get('ok'):
                    result = self._scope_result(action, result, scope)
                self._audit(action, payload, actor_id, 'ok' if result.get('ok') else 'failed')
                if not action.endswith('.list') and action != 'status' and result.get('ok'):
                    await asyncio.to_thread(self.access.audit, actor_id, action,
                        {key: value for key, value in payload.items() if key not in {'text', 'value'}})
                return result
            except (ValueError, TypeError) as exc:
                self._audit(action, payload, actor_id, 'failed', {'error': type(exc).__name__})
                # Validation messages may contain input. Return only bounded,
                # nonsensitive errors; never a full Pydantic config dump.
                from pydantic import ValidationError
                line = 'Invalid setting value.' if isinstance(exc, ValidationError) else str(exc)[:180]
                return {'ok': False, 'error': line if not contains_secret(line) else 'Invalid value.'}
            except Exception:
                self._audit(action, payload, actor_id, 'failed', {'error': 'unexpected'})
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
                # F-104: the room calls people by name, so a rename has to
                # reach the recogniser as well.
                self._sync_hotwords()
            elif action == 'profiles.role':
                await asyncio.to_thread(registry.set_role, name, payload.get('role'))
            elif action == 'profiles.language':
                # ТЗ F-106: the field whisper and the model follow. The DB row
                # is canonical (section 14) and the registry keeps the copy the
                # voice pipeline reads, so both are written here. The two live
                # in different threads: the registry is blocking file I/O (a
                # worker thread), the hub's SQLite connection belongs to the
                # event loop (DECISIONS.md, P1-44).
                from hub.languages import set_preferred_language

                code = payload.get('language')
                written = await asyncio.to_thread(set_preferred_language, registry, None,
                                                  name, code)
                set_preferred_language(None, runtime.get('hub_conn'), name, code)
                return {'ok': True,
                        'message': 'Preferred language saved.',
                        'language': written}
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
        if action == 'calibration.list':
            # ТЗ 5.4: the weekly report is read out of decisions, one row per
            # (type, provider). The window is the configured report period; a
            # caller may ask for a shorter one, never a wider one.
            decisions = self.get_decisions()
            if decisions is None:
                raise ValueError('The decision log is unavailable.')
            configured = int(getattr(self.cfg.server.decider, 'report_days', 7))
            days = int(payload.get('days') or configured)
            if not 1 <= days <= configured:
                raise ValueError(f'Choose between 1 and {configured} days.')
            # One aggregate query on the hub's own connection, so it stays on
            # this thread (see ``hub.decision_log.DecisionLog.calibration``).
            report = decisions.calibration(window_s=days * 86400.0)
            return {'ok': True, 'days': days, **report}
        if action.startswith('rules.'):
            # ТЗ F-419: правила в панели — списком, словами, с включением и
            # удалением. JSON-поля читает и проверяет `hub/automation.py`.
            rules = self.get_rules()
            if rules is None:
                raise ValueError('The rule store is unavailable.')
            home_id = payload.get('home_id')
            if action == 'rules.list':
                items = [{'id': rule.rule_id, 'home_id': rule.home_id,
                          'name': rule.name, 'enabled': rule.enabled,
                          'words': automation.describe(rule, 'ru')}
                         for rule in rules.all(home_id=home_id)]
                return {'ok': True, 'items': items}
            rule_id = str(payload.get('id') or '')
            if not rule_id:
                raise ValueError('Choose a rule first.')
            if action == 'rules.update':
                rule = rules.read(rule_id)
                if rule is None:
                    raise ValueError('That rule is gone.')
                if home_id is not None and str(rule.home_id) != str(home_id):
                    raise ValueError('That rule belongs to another home.')
                enabled = bool(payload.get('enabled'))
                if rules.set_enabled(rule_id, enabled) is False:
                    raise ValueError('That rule is gone.')
                return {'ok': True, 'message': 'Rule enabled.' if enabled else 'Rule disabled.'}
            if action == 'rules.delete':
                if rules.remove(rule_id) is False:
                    raise ValueError('That rule is gone.')
                return {'ok': True, 'message': 'Rule deleted.'}
            raise ValueError('Unknown rule operation.')
        if action.startswith('devices.'):
            # ТЗ F-503: the ESP32 wall switches and the servo angles that make
            # "on" mean on. The panel is the only place they are calibrated.
            switches = self.get_switches()
            if switches is None:
                raise ValueError('The device registry is unavailable.')
            if action == 'devices.list':
                # The device store is the hub's own connection, so it stays on
                # this thread (see hub.decision_log.DecisionLog.calibration).
                items = switches.switches(payload.get('home_id'))
                return {'ok': True, 'items': items}
            if action == 'devices.calibrate':
                device_id = str(payload.get('device_id') or payload.get('id') or '')
                if not device_id:
                    raise ValueError('Choose a switch first.')
                if 'value' in payload:
                    closed, opened = _two_angles(payload.get('value'))
                else:
                    closed, opened = payload.get('closed_angle'), payload.get('open_angle')
                result = switches.calibrate(device_id, closed_angle=closed, open_angle=opened,
                                            dwell_s=payload.get('dwell_s'))
                return {'ok': True, **result}
            if action == 'devices.scan':
                # ТЗ F-504: the scan itself is the wizard's, so the panel only
                # passes on what could and could not be looked at.
                wizard = self.get_wizard()
                if wizard is None:
                    raise ValueError('The device wizard is unavailable.')
                report = await wizard.scan()
                return {'ok': True,
                        'found': [item.model_dump() for item in report['found']],
                        'unavailable': report['unavailable']}
            if action == 'devices.add':
                wizard = self.get_wizard()
                if wizard is None:
                    raise ValueError('The device wizard is unavailable.')
                address = str(payload.get('address') or '')
                candidate = payload.get('found') or {'address': address, 'name': payload.get('name'),
                                                     'source': payload.get('source', 'network'),
                                                     'adapter': payload.get('adapter')}
                from hub.discovery import FoundDevice

                found = FoundDevice.model_validate({**candidate,
                    'name': payload.get('name') or candidate.get('name') or address})
                device = wizard.adopt(
                    found, home_id=str(payload.get('home_id') or ''),
                    name=payload.get('name'), aliases=_aliases(payload.get('aliases')),
                    kind=payload.get('kind'), capabilities=payload.get('capabilities'),
                    zone=payload.get('zone'), adapter=payload.get('adapter'))
                added = await self._add_hotwords(wizard.hotwords_for(device))
                # F-104: the name the owner typed and the aliases the wizard
                # proposed are only useful if the recogniser learns them now.
                self._sync_hotwords()
                return {'ok': True, 'device': device.model_dump(), 'hotwords': added,
                        'message': f"{device.name} added. Say its name to control it."}
            if action == 'devices.blink':
                wizard = self.get_wizard()
                if wizard is None:
                    raise ValueError('The device wizard is unavailable.')
                device = self._device_for(payload)
                return {'ok': True, **await wizard.blink(device.id, home_id=device.home_id)}
            if action in {'devices.remove', 'devices.delete'}:
                wizard = self.get_wizard()
                if wizard is None:
                    raise ValueError('The device wizard is unavailable.')
                device = self._device_for(payload)
                wizard.remove(device.id)
                self._sync_hotwords()
                return {'ok': True, 'message': f'{device.name} removed from this home.'}
            raise ValueError('Unknown device operation.')
        if action.startswith('scenes.'):
            # ТЗ F-506: the five presets are created the first time a home looks
            # at its scenes, and the owner runs them from here or by voice.
            store = self.get_scenes()
            if store is None:
                raise ValueError('Scenes are unavailable.')
            home_id = str(payload.get('home_id') or '')
            if action == 'scenes.list':
                created = store.ensure_presets(home_id)
                return {'ok': True, 'items': [_scene_row(scene) for scene in store.scenes(home_id)],
                        'created': [scene.name for scene in created]}
            if action == 'scenes.run':
                scene = store.resolve(home_id, payload.get('scene_id') or payload.get('name'))
                if scene is None:
                    raise ValueError('Unknown scene.')
                runner = self._scene_runner(store, payload)
                report = await runner.run(scene, home_id=home_id or scene.home_id)
                # ``ok`` here means "the panel did its job"; whether the scene
                # itself ran is the report's own ``ok`` (see DECISIONS.md).
                return {'ok': True, 'scene': report, 'message': report['message']}
            if action in {'scenes.delete', 'scenes.remove'}:
                scene = store.resolve(home_id, payload.get('scene_id') or payload.get('name'))
                if scene is None:
                    raise ValueError('Unknown scene.')
                store.delete(scene.scene_id)
                self._sync_hotwords()
                return {'ok': True, 'message': f'Scene {scene.name} deleted.'}
            raise ValueError('Unknown scene operation.')
        if action.startswith('homes.'):
            # ТЗ F-701: домами владеют Telegram-аккаунты. Раздавать их может
            # только админ хаба (у владельца дома для этого нет прав), а сам
            # владелец видит свои дома там, где ему положено: в /tools.
            owners = self.get_home_owners()
            if owners is None:
                raise ValueError('Home ownership is unavailable.')
            names = {str(getattr(home, 'home_id', '')): str(getattr(home, 'name', '') or '')
                     for home in (getattr(self.cfg, 'homes', None) or [])}
            if action == 'homes.list':
                return {'ok': True, 'items': [
                    {'home_id': home_id, 'name': names.get(home_id) or home_id, 'owners': list(ids)}
                    for home_id, ids in owners.owners().items()]}
            if action == 'homes.grant':
                home_id = str(payload.get('home_id') or '')
                try:
                    user_id = int(payload.get('user_id'))
                except (TypeError, ValueError):
                    raise ValueError('Enter the numeric Telegram ID of the owner.') from None
                added = owners.grant(actor, user_id, home_id)
                return {'ok': True, 'added': added,
                        'message': (f'Home {home_id} is now owned by Telegram ID {user_id}.'
                                    if added else f'Telegram ID {user_id} already owns {home_id}.')}
            if action == 'homes.revoke':
                home_id = str(payload.get('home_id') or '')
                try:
                    user_id = int(payload.get('user_id'))
                except (TypeError, ValueError):
                    raise ValueError('Enter the numeric Telegram ID of the owner.') from None
                removed = owners.revoke(actor, user_id, home_id)
                return {'ok': True, 'removed': removed,
                        'message': (f'Telegram ID {user_id} no longer owns {home_id}.'
                                    if removed else f'Telegram ID {user_id} did not own {home_id}.')}
            raise ValueError('Unknown home ownership operation.')
        raise ValueError('Unknown panel action.')

    def _scene_runner(self, store, payload):
        """How a panel-run scene reaches the room: devices, the PC, and a voice."""
        from hub.scenes import SceneRunner

        tools = self.get_tools()
        room = self.get_room()

        async def run_pc(tool, args):
            if room is None:
                return {'ok': False, 'error': 'no room computer is connected'}
            return await room._run_client_action(tool, dict(args))

        async def say(text):
            # The panel has no room speaker; the owner hears it in Telegram and
            # reads it in the report (see DECISIONS.md, P1-38).
            provider = self.get_provider()
            if provider is None:
                return None
            chat = payload.get('chat_id')
            kwargs = {'private_reply_to_user_id': chat} if isinstance(chat, int) else {}
            await provider.send_text(text, **kwargs)
            return None

        return SceneRunner(store, set_device=(tools.set if tools is not None else None),
                           run_pc=run_pc, say=say)

    def _device_for(self, payload):
        """The device an action names, or a clear refusal."""
        switches = self.get_switches()
        if switches is None:
            raise ValueError('The device registry is unavailable.')
        device_id = str(payload.get('device_id') or payload.get('id') or '')
        device = switches.store.get(device_id)
        if device is None:
            raise ValueError('Unknown device.')
        return device

    async def _add_hotwords(self, words):
        """Put a new device name into the speech recogniser's hotwords (F-104)."""
        if not words:
            return []
        from hub.admin_settings import validated_value

        key = 'server.stt.hotwords'
        current = list(getattr(self.cfg.server.stt, 'hotwords', []) or [])
        merged = list(current)
        for word in words:
            if word.casefold() not in {item.casefold() for item in merged}:
                merged.append(word)
        value = validated_value(self.cfg, key, merged)
        await asyncio.to_thread(self.access.set_setting, 'config:' + key, {'value': value})
        apply_live(self.cfg, key, value, self.runtime())
        return list(value)

    def _sync_hotwords(self) -> list[str]:
        """Rebuild the recogniser's hotwords from the database (ТЗ F-104).

        Called after every change that can introduce a new name: people,
        devices and scenes are the sources, so an edit made here has to reach
        the recogniser even when the owner did not type the word themselves.
        """
        from hub.hotwords import sync

        runtime = self.runtime() or {}
        conn = runtime.get('hub_conn')
        if conn is None:
            return []
        return sync(self.cfg, conn, runtime.get('stt'))

    def _audit(self, action, payload, actor, result, detail=None):
        """One row in the hub's audit table (ТЗ F-706).

        Reads are not audited — ``.list`` actions and ``status`` only look — and
        a panel that cannot reach the audit table still does its job: the log is
        a record, not a gate.

        Two records are written, of the same change: the row in the hub's
        ``audit`` table, and a line in ``data/telegram/audit.log`` with the
        account, the values it changed and the outcome, so the owner can read
        who changed what without opening a database.
        """
        if action.endswith('.list') or action == 'status':
            return
        if not str(action).startswith(self.AUDITED):
            return
        target = str(payload.get('id') or payload.get('device_id') or payload.get('scene_id')
                     or payload.get('user_id') or payload.get('key') or '')
        home_id = payload.get('home_id') or self._home_of(action, payload)
        # The chat the panel is being used from is not a change; everything
        # else the account sent is what "what" means in "who changed what".
        values = {str(key): value for key, value in payload.items() if key != 'chat_id'}
        audit = self.get_audit()
        if audit is not None:
            audit.record(action=action, actor=actor, target=target,
                         home_id=home_id, result=result,
                         detail=detail or {'fields': sorted(payload), 'values': values})
        telegram_audit.record(actor, action, values, result,
                              label=self._actor_label(actor), home_id=str(home_id or ''),
                              error=str((detail or {}).get('error') or ''))

    def _actor_label(self, actor):
        """How the account that made the change is called, when it is known."""
        lookup = getattr(self.access, 'label', None)
        try:
            return str(lookup(actor) or '') if callable(lookup) else ''
        except Exception:  # noqa: BLE001 - a nameless audit line is still a line
            return ''

    def _home_of(self, action, payload):
        """Which home a panel action belongs to, when it can be told."""
        scenes = self.get_scenes()
        if scenes is not None and action.startswith('scenes.'):
            scene = scenes.resolve(str(payload.get('home_id') or ''), payload.get('scene_id'))
            if scene is not None:
                return scene.home_id
        switches = self.get_switches()
        if switches is not None and action.startswith('devices.'):
            device = switches.store.get(str(payload.get('device_id') or payload.get('id') or ''))
            if device is not None:
                return device.home_id
        home_id = payload.get('home_id')
        if not home_id and action.startswith('workplaces.'):
            # ТЗ F-701: у рабочего места дома нет в имени, и взять его можно
            # только у подключения (или у того, что записано при hello).
            home_id = self.get_workplace_home(str(payload.get('id') or ''))
        return home_id

    def _scope_refusal(self, action, payload, scope):
        """Why this home owner may not run this action (``''`` when they may)."""
        if action == 'status':
            return ''
        if not str(action).startswith(self.HOME_SCOPED):
            return ('This setting belongs to the whole hub; only the hub administrator '
                    'can change it.')
        home_id = self._home_of(action, payload)
        if not home_id:
            if action in {'workplaces.list', 'workplaces.'}:
                return ''
            return 'This operation does not name a home, so it cannot be checked.'
        if str(home_id) not in scope:
            return 'This home belongs to another owner.'
        return ''

    def _scope_result(self, action, result, scope):
        """Drop everything outside the account's homes from a list (F-701)."""
        items = result.get('items')
        if isinstance(items, list) and action == 'workplaces.list':
            kept = []
            for row in items:
                if not isinstance(row, dict):
                    continue
                home_id = str(row.get('home_id') or '')
                if not home_id and action == 'workplaces.list':
                    home_id = str(self.get_workplace_home(str(row.get('id') or '')) or '')
                if home_id and home_id in scope:
                    kept.append(row)
            result = {**result, 'items': kept}
        if action == 'status':
            # Бюджет API — деньги хаба, а не дома: владельцу комнаты он не виден.
            result = {key: value for key, value in result.items()
                      if key not in {'api_usage', 'profiles'}}
            result['scoped_homes'] = sorted(scope)
            places = result.get('workplaces')
            if isinstance(places, list):
                kept = []
                for row in places:
                    if not isinstance(row, dict):
                        continue
                    home_id = str(row.get('home_id')
                                  or self.get_workplace_home(str(row.get('id') or '')) or '')
                    if home_id and home_id in scope:
                        kept.append(row)
                result = {**result, 'workplaces': kept}
        if action == 'workplaces.list' and result.get('selected_id') is not None:
            selected = str(result['selected_id'])
            allowed = {str(row.get('id')) for row in result.get('items', [])}
            if selected not in allowed:
                result = {**result, 'selected_id': None}
        return result


def _aliases(value):
    """The panel sends aliases as one comma-separated line."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or '')
    return [part.strip() for part in text.replace(';', ',').split(',') if part.strip()]


def _scene_row(scene):
    """One scene as the panel shows it."""
    return {'scene_id': scene.scene_id, 'home_id': scene.home_id, 'name': scene.name,
            'aliases': list(scene.aliases), 'preset': scene.preset,
            'steps': len(scene.steps),
            'summary': '; '.join(step.describe() for step in scene.steps[:4])}


def _two_angles(value):
    """Parse the panel's ``closed,open`` answer into two numbers."""
    from hub.switch_calibration import parse_calibration

    calibration = parse_calibration(value)
    return calibration.closed_angle, calibration.open_angle
