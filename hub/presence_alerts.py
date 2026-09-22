"""Persistent opt-in room alerts; observations never wait for disk or Telegram."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from common.protocol import CAMERA_CLIP_MAX_BYTES, DEFAULT_STATUS_TTL_S, MSG_STATUS

log = logging.getLogger(__name__)
DEFAULT_RULE = dict(enabled=False, target='any', name='', media='photo', destination='owner',
                    cooldown_s=300, min_stable_s=2, absence_s=15, quiet_start='', quiet_end='',
                    timezone='America/Chicago', clip_seconds=5, workplace_id='',
                    # ТЗ F-702: правило слушает событие, а не только «человек в
                    # кадре»; канал доставки выбирается; дом может дать свои
                    # тихие часы и свой минимум кулдауна.
                    event='presence', channel='telegram', home_id='', zone='')
#: ТЗ F-702: события F-301 (`person_entered`…), F-109 (`sound_event`), F-311
#: (`object`) плюс прежнее `presence` — «человек стабильно в кадре».
EVENT_KINDS = ('presence', 'person_entered', 'person_left', 'unknown_appeared',
               'zone_entered', 'sound_event', 'object')
#: ТЗ F-702: «выбор канала (Telegram, пуш на телефон, HUD)».
CHANNELS = ('telegram', 'push', 'hud')
#: Allowed ``(min, max)`` for the numeric alert settings. The Telegram panel
#: reads this too, so its presets and this validation can never drift apart.
#: The cooldown floor is 1 s (it used to be 10 s): short rules are legitimate,
#: and a room panel should not force a ten-second minimum.
RULE_RANGES = {'cooldown_s': (1.0, 86400.0), 'min_stable_s': (0.0, 300.0), 'absence_s': (0.0, 3600.0)}
_ID = re.compile(r'alert-[0-9a-f]{32}')
_CLOCK = re.compile(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]')


def validate_rule(patch, existing=None):
    if not isinstance(patch, dict) or set(patch) - DEFAULT_RULE.keys():
        raise ValueError('Unknown alert setting.')
    rule = {**DEFAULT_RULE, **(existing or {}), **patch}
    if type(rule['enabled']) is not bool:
        raise ValueError('enabled must be a boolean.')
    for key, allowed in [('target', {'any', 'unknown', 'person'}), ('media', {'photo', 'video'}),
                         ('destination', {'owner', 'group'})]:
        if not isinstance(rule[key], str) or rule[key] not in allowed:
            raise ValueError(f'Invalid {key}.')
    if rule['event'] not in EVENT_KINDS:
        raise ValueError('Unknown alert event.')
    if rule['channel'] not in CHANNELS:
        raise ValueError('Unknown alert channel.')
    if not isinstance(rule['name'], str) or len(rule['name']) > 120 or any(ord(c) < 32 for c in rule['name']):
        raise ValueError('Provide a person name of at most 120 characters.')
    # Имя — это человек для `target: person` и МЕТКА объекта для события F-311;
    # во всех остальных случаях оно не хранится, чтобы правило не выглядело так,
    # будто оно кого-то ищет.
    keep_name = rule['target'] == 'person' or rule['event'] == 'object'
    rule['name'] = ' '.join(rule['name'].split()) if keep_name else ''
    rule['home_id'] = str(rule['home_id'] or '').strip()
    if rule['home_id'] and not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', rule['home_id']):
        raise ValueError('Provide a home id, or leave it empty for every home.')
    if not isinstance(rule['zone'], str) or len(rule['zone']) > 120 or any(ord(c) < 32 for c in rule['zone']):
        raise ValueError('Provide a zone name of at most 120 characters.')
    rule['zone'] = ' '.join(rule['zone'].split())
    if rule['event'] == 'object' and not rule['name']:
        raise ValueError('An object alert needs the object label to watch for.')
    if rule['target'] == 'person' and not rule['name']:
        raise ValueError('A person alert needs an enrolled person name.')
    if (not isinstance(rule['workplace_id'], str) or len(rule['workplace_id']) > 100
            or any(ord(c) < 32 for c in rule['workplace_id'])):
        raise ValueError('Provide a workplace client ID of at most 100 characters.')
    rule['workplace_id'] = rule['workplace_id'].strip()
    for key, (low, high) in RULE_RANGES.items():
        value = rule[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be between {low:g} and {high:g}.')
    if type(rule['clip_seconds']) is not int or not 3 <= rule['clip_seconds'] <= 10:
        raise ValueError('clip_seconds must be between 3 and 10.')
    for key in ('quiet_start', 'quiet_end'):
        if not isinstance(rule[key], str) or (rule[key] and not _CLOCK.fullmatch(rule[key])):
            raise ValueError('Quiet hours must use HH:MM.')
    if bool(rule['quiet_start']) != bool(rule['quiet_end']):
        raise ValueError('Set both quiet-hour boundaries or leave both empty.')
    if rule['quiet_start'] and rule['quiet_start'] == rule['quiet_end']:
        raise ValueError('Quiet-hour start and end must differ.')
    try:
        if not isinstance(rule['timezone'], str) or len(rule['timezone']) > 100:
            raise ValueError
        ZoneInfo(rule['timezone'])
    except (ValueError, ZoneInfoNotFoundError):
        raise ValueError('Provide a valid IANA timezone, such as America/Chicago.') from None
    return rule


def quiet_now(rule, timestamp):
    if not rule['quiet_start']:
        return False
    clock = datetime.fromtimestamp(timestamp, ZoneInfo(rule['timezone'])).strftime('%H:%M')
    start, end = rule['quiet_start'], rule['quiet_end']
    return start <= clock < end if start < end else clock >= start or clock < end


def home_settings(home):
    """A home's alert window as a plain dict (ТЗ F-702, «на дом»)."""
    if not isinstance(home, dict):
        return {}
    quiet = home.get('quiet_hours') if isinstance(home.get('quiet_hours'), dict) else {}
    start = str(home.get('quiet_start') or quiet.get('start') or '').strip()
    end = str(home.get('quiet_end') or quiet.get('end') or '').strip()
    try:
        cooldown = float(home.get('cooldown_s') or 0)
    except (TypeError, ValueError):
        cooldown = 0.0
    return {'quiet_start': start, 'quiet_end': end,
            'timezone': str(home.get('timezone') or '').strip(),
            'cooldown_s': max(0.0, cooldown) if math.isfinite(cooldown) else 0.0}


def home_quiet_now(home, timestamp, fallback_timezone='UTC'):
    """Тихие часы ДОМА: правило может молчать само, дом — тоже (ТЗ F-702)."""
    window = home_settings(home)
    start, end = window.get('quiet_start') or '', window.get('quiet_end') or ''
    if not start or not end or start == end:
        return False
    try:
        zone = ZoneInfo(window.get('timezone') or fallback_timezone)
    except (ValueError, ZoneInfoNotFoundError):
        zone = ZoneInfo('UTC')
    clock = datetime.fromtimestamp(timestamp, zone).strftime('%H:%M')
    return start <= clock < end if start < end else clock >= start or clock < end


def home_cooldown(home, rule_cooldown):
    """Дом может сделать правило РЕЖЕ (свой минимум), но не чаще (F-702)."""
    return max(float(rule_cooldown), float(home_settings(home).get('cooldown_s') or 0.0))


class PresenceAlerts:
    """Callbacks are synchronous; CRUD/status use disk and belong in to_thread.

    ``observe`` runs on the event loop and only enqueues bounded observations.
    ``observed_at`` is Unix wall time, never a camera's monotonic timestamp.
    ``names`` must contain fresh direct face matches from this image, not
    remembered room identities. ``unknown_count`` also requires fresh evidence.
    """

    def __init__(self, folder, get_provider, get_room, owner_id, group_id, *,
                 get_home=None, get_push=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.get_provider, self.get_room = get_provider, get_room
        #: ``home_id -> {'quiet_start','quiet_end','timezone','cooldown_s'}``
        #: (ТЗ F-702): тихие часы и минимум кулдауна берутся у дома.
        self.get_home = get_home or (lambda home_id: None)
        #: Push-транспорт (F-712) или ``None``: без него канал `push` честно
        #: сообщает, что доставлять нечем, а не притворяется отправленным.
        self.get_push = get_push or (lambda: None)
        self.owner_id, self.group_id = owner_id, group_id
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.folder / 'alerts.sqlite3', check_same_thread=False, isolation_level=None)
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.execute('CREATE TABLE IF NOT EXISTS rules (id TEXT PRIMARY KEY, settings TEXT NOT NULL, state TEXT NOT NULL)')
        self._db.execute('CREATE TABLE IF NOT EXISTS rule_sources (rule_id TEXT NOT NULL, source_id TEXT NOT NULL, '
                         'state TEXT NOT NULL, PRIMARY KEY(rule_id,source_id))')
        self._db.execute('CREATE TABLE IF NOT EXISTS deliveries (id TEXT PRIMARY KEY, rule_id TEXT NOT NULL, '
                         'at REAL NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT \'\')')
        # An interrupted upload may have reached Telegram: never retry it.
        self._db.execute("UPDATE deliveries SET status='uncertain', detail='Server stopped before delivery acknowledgement' "
                         "WHERE status='pending'")
        self._queue = asyncio.Queue(maxsize=8)
        self._worker = None
        self._deliveries = set()
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._latest_photo = None
        self.dropped = 0

    def list_rules(self):
        with self._lock:
            return [dict(id=row[0], **{**DEFAULT_RULE, **json.loads(row[1])}, last_attempt=json.loads(row[2]).get('last_attempt'))
                    for row in self._db.execute('SELECT id, settings, state FROM rules ORDER BY rowid')]

    def save_rule(self, patch, rule_id=None):
        with self._lock:
            previous = None
            if rule_id is not None:
                if not isinstance(rule_id, str) or not _ID.fullmatch(rule_id):
                    raise ValueError('Invalid alert rule ID.')
                previous = self._db.execute('SELECT settings,state FROM rules WHERE id=?', (rule_id,)).fetchone()
                if previous is None:
                    raise ValueError('Alert rule was not found.')
            elif self._db.execute('SELECT COUNT(*) FROM rules').fetchone()[0] >= 32:
                raise ValueError('At most 32 alert rules can be saved.')
            rule = validate_rule(patch, json.loads(previous[0]) if previous else None)
            destination = self.owner_id if rule['destination'] == 'owner' else self.group_id
            if rule['enabled'] and (type(destination) is not int or
                    (destination <= 0 if rule['destination'] == 'owner' else destination >= 0)):
                raise ValueError('Configure the selected Telegram destination before enabling this rule.')
            rule_id = rule_id or 'alert-' + uuid.uuid4().hex
            old_state = json.loads(previous[1]) if previous else {}
            # Edits can start a new episode, but cannot erase a reserved cooldown.
            state = {'last_attempt': old_state.get('last_attempt', 0)}
            self._db.execute('INSERT OR REPLACE INTO rules VALUES (?,?,?)',
                             (rule_id, json.dumps(rule), json.dumps(state)))
            self._db.execute('DELETE FROM rule_sources WHERE rule_id=?', (rule_id,))
            return dict(id=rule_id, **rule, last_attempt=state['last_attempt'] or None)

    def remove_rule(self, rule_id):
        with self._lock:
            self._db.execute('DELETE FROM rule_sources WHERE rule_id=?', (str(rule_id),))
            return bool(self._db.execute('DELETE FROM rules WHERE id=?', (str(rule_id),)).rowcount)

    def status(self):
        with self._lock:
            recent = [dict(id=r[0], rule_id=r[1], at=r[2], status=r[3], detail=r[4]) for r in
                      self._db.execute('SELECT * FROM deliveries ORDER BY at DESC LIMIT 30')]
        return dict(running=bool(self._worker and not self._worker.done()), queued=self._queue.qsize(),
                    pending=len(self._deliveries), dropped=self.dropped, deliveries=recent)

    def start(self):
        if not self._closed and (self._worker is None or self._worker.done()):
            self._worker = asyncio.create_task(self._run(), name='presence-alerts')

    def observe(self, *, persons=None, names=None, unknown_count=None, jpeg=None, source_id='', observed_at=None):
        if self._closed or self._worker is None or self._worker.done():
            return False
        now = time.time()
        timestamp = now if observed_at is None else observed_at
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or abs(now - timestamp) > 10:
            return False
        if names is not None:
            if not isinstance(names, (list, tuple, set)):
                return False
            names = tuple(name.strip() for name in names if isinstance(name, str) and 0 < len(name.strip()) <= 120)[:24]
        persons = persons if type(persons) is int and 0 <= persons <= 100 else None
        unknown_count = unknown_count if type(unknown_count) is int and 0 <= unknown_count <= 100 else None
        jpeg = jpeg if isinstance(jpeg, bytes) and 0 < len(jpeg) <= 10_000_000 else None
        source_id = str(source_id)[:100]
        if jpeg:
            self._latest_photo = (timestamp, source_id, jpeg)
        event = dict(at=float(timestamp), persons=persons, names=names, unknown_count=unknown_count,
                     jpeg=jpeg, source_id=source_id, kind='presence', home_id='',
                     label='', zone='', confidence=None)
        if self._queue.full():
            self.dropped += 1
            return False
        self._queue.put_nowait(event)
        return True

    def observe_event(self, kind, *, source_id='', home_id='', name='', label='', zone='',
                      confidence=None, observed_at=None):
        """ТЗ F-702: событие для правил — F-301, F-109 (звук), F-311 (объект).

        ``kind`` — один из :data:`EVENT_KINDS` без ``presence``; остальное —
        подробности события. Возвращает ``False``, когда событие не принято
        (сервис не запущен, время не похоже на настоящее, вид неизвестен), и
        никогда не бросает: правило не может ломать ход комнаты.
        """
        if self._closed or self._worker is None or self._worker.done():
            return False
        if kind not in EVENT_KINDS or kind == 'presence':
            return False
        now = time.time()
        timestamp = now if observed_at is None else observed_at
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or abs(now - timestamp) > 30:
            return False
        try:
            confidence = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence = None
        event = dict(at=float(timestamp), persons=None, names=((str(name).strip(),) if str(name or '').strip() else ()),
                     unknown_count=None, jpeg=None, source_id=str(source_id)[:100],
                     kind=kind, home_id=str(home_id)[:64], label=str(label)[:120],
                     zone=str(zone)[:120], confidence=confidence)
        if self._queue.full():
            self.dropped += 1
            return False
        self._queue.put_nowait(event)
        return True

    def _home_for(self, rule, event):
        """Дом правила или дома события — как словарь для тихих часов."""
        home_id = str(rule.get('home_id') or event.get('home_id') or '')
        if not home_id:
            return {}, ''
        try:
            return (self.get_home(home_id) or {}), home_id
        except Exception as exc:  # noqa: BLE001 - дом без настроек всё равно дом
            log.debug('Alert home %s is unreadable (%s)', home_id, exc)
            return {}, home_id

    @staticmethod
    def _event_match(rule, event):
        """Does this non-presence event satisfy this rule? (ТЗ F-702)"""
        kind = str(event.get('kind') or '')
        if rule.get('event') != kind:
            return False
        label = str(event.get('label') or '')
        if kind == 'object' and rule.get('name', '').casefold() != label.casefold():
            return False
        if kind in {'person_entered', 'person_left', 'unknown_appeared'} and rule.get('target') == 'person':
            wanted = rule.get('name', '').casefold()
            if wanted and wanted not in {str(name).casefold() for name in (event.get('names') or ())}:
                return False
        zone = rule.get('zone') or ''
        return not zone or zone == str(event.get('zone') or '')

    def _reserve(self, event):
        """Persist each claim and cooldown BEFORE starting any media/network work."""
        ready = []
        now = event['at']
        with self._lock:
            self._db.execute('BEGIN IMMEDIATE')
            try:
                rows = self._db.execute('SELECT id,settings,state FROM rules').fetchall()
                for rule_id, raw, saved in rows:
                    rule, global_state = {**DEFAULT_RULE, **json.loads(raw)}, json.loads(saved)
                    if not rule['enabled'] or (rule['workplace_id'] and rule['workplace_id'] != event['source_id']):
                        continue
                    # ТЗ F-702: правило дома применяется только к событиям дома.
                    event_home = str(event.get('home_id') or '')
                    if rule['home_id'] and rule['home_id'] != event_home:
                        continue
                    home, _ = self._home_for(rule, event)
                    cooldown = home_cooldown(home, rule['cooldown_s'])
                    source = self._db.execute('SELECT state FROM rule_sources WHERE rule_id=? AND source_id=?',
                                              (rule_id, event['source_id'])).fetchone()
                    state = json.loads(source[0]) if source else (
                        dict(global_state) if global_state.get('source_id') == event['source_id'] else {})
                    # Каждый ВИД наблюдения помнит своё время: на Windows
                    # ``time.time()`` идёт шагом ~16 мс, поэтому событие F-301
                    # вполне может иметь ту же метку, что и кадр присутствия, и
                    # одно не должно глушить другое.
                    seen_key = ('observed' if str(event.get('kind') or 'presence') == 'presence'
                                else 'observed:' + str(event.get('kind')))
                    if now <= state.get(seen_key, 0):
                        continue
                    state[seen_key] = now
                    if rule['event'] != 'presence':
                        # Событие F-301/F-109/F-311: правилу не нужна «стабильность»
                        # в кадре — событие уже произошло, и оно либо подходит,
                        # либо нет. Кулдаун и тихие часы те же, что у присутствия.
                        if (self._event_match(rule, event)
                                and now - global_state.get('last_attempt', 0) >= cooldown
                                and not quiet_now(rule, now)
                                and not home_quiet_now(home, now, rule['timezone'])):
                            global_state['last_attempt'] = now
                            delivery = dict(id=uuid.uuid4().hex, rule_id=rule_id, rule=rule, event=event)
                            self._db.execute('INSERT INTO deliveries(id,rule_id,at,status) VALUES (?,?,?,?)',
                                             (delivery['id'], rule_id, now, 'pending'))
                            ready.append(delivery)
                        self._db.execute('UPDATE rules SET state=? WHERE id=?', (json.dumps(global_state), rule_id))
                        self._db.execute('INSERT OR REPLACE INTO rule_sources VALUES (?,?,?)',
                                         (rule_id, event['source_id'], json.dumps(state)))
                        continue
                    signal = (event['persons'] > 0 if event['persons'] is not None else None) if rule['target'] == 'any' else (
                        event['unknown_count'] > 0 if event['unknown_count'] is not None else None) if rule['target'] == 'unknown' else (
                        rule['name'].casefold() in {name.casefold() for name in event['names']} if event['names'] is not None else None)
                    last = state.get('last_match', 0)
                    if now - last >= rule['absence_s'] or state.get('source_id') != event['source_id']:
                        state.update(since=0, observations=0, episode_sent=False)
                    if signal is False and not state.get('episode_sent'):
                        state.update(since=0, observations=0)
                    if signal:
                        if not state.get('since'):
                            state['since'] = now
                        state.update(last_match=now, source_id=event['source_id'],
                                     observations=state.get('observations', 0) + 1)
                        stable = (now - state['since'] >= rule['min_stable_s']
                                  and (rule['min_stable_s'] == 0 or state['observations'] >= 2))
                        if (stable and not state.get('episode_sent')
                                and now - global_state.get('last_attempt', 0) >= cooldown
                                and not quiet_now(rule, now)
                                and not home_quiet_now(home, now, rule['timezone'])):
                            state['episode_sent'] = True
                            global_state['last_attempt'] = now
                            delivery = dict(id=uuid.uuid4().hex, rule_id=rule_id, rule=rule, event=event)
                            self._db.execute('INSERT INTO deliveries(id,rule_id,at,status) VALUES (?,?,?,?)',
                                             (delivery['id'], rule_id, now, 'pending'))
                            ready.append(delivery)
                    self._db.execute('UPDATE rules SET state=? WHERE id=?', (json.dumps(global_state), rule_id))
                    self._db.execute('INSERT OR REPLACE INTO rule_sources VALUES (?,?,?)',
                                     (rule_id, event['source_id'], json.dumps(state)))
                self._db.execute('DELETE FROM deliveries WHERE id NOT IN (SELECT id FROM deliveries ORDER BY at DESC LIMIT 500)')
                self._db.execute('COMMIT')
            except BaseException:
                self._db.execute('ROLLBACK')
                raise
        return ready

    def _finish(self, delivery, status, detail=''):
        with self._lock:
            self._db.execute('UPDATE deliveries SET status=?,detail=? WHERE id=?',
                             (status, str(detail)[:500], delivery['id']))

    def _unchanged(self, delivery):
        with self._lock:
            row = self._db.execute('SELECT settings FROM rules WHERE id=?', (delivery['rule_id'],)).fetchone()
            return row is not None and {**DEFAULT_RULE, **json.loads(row[0])} == delivery['rule']

    async def _run(self):
        while True:
            event = await self._queue.get()
            try:
                for delivery in await asyncio.to_thread(self._reserve, event):
                    if len(self._deliveries) >= 8:
                        await asyncio.to_thread(self._finish, delivery, 'skipped', 'Notification queue is busy')
                        continue
                    task = asyncio.create_task(self._deliver(delivery), name='presence-alert-delivery')
                    self._deliveries.add(task)
                    task.add_done_callback(self._deliveries.discard)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error('Presence alert observation failed (%s)', type(exc).__name__)
            finally:
                self._queue.task_done()

    @staticmethod
    def _idle(room):
        if room is None or getattr(room, 'receiving', False):
            return False
        return not any(task is not None and not task.done() for task in
                       (getattr(room, key, None) for key in ('_task', '_telegram_control_task', '_enroll_face_task')))

    def _room(self, source_id):
        try:
            return self.get_room(source_id or None)
        except TypeError:
            # Existing single-room integrations can keep their zero-arg hook.
            return self.get_room()

    async def _media(self, delivery):
        rule, event = delivery['rule'], delivery['event']
        if rule['media'] == 'photo' and event['jpeg']:
            return event['jpeg']
        if rule['media'] == 'photo':
            # A state-only event often precedes its same-camera presence JPEG.
            for _ in range(10):
                photo = self._latest_photo
                if photo and photo[1] == event['source_id'] and photo[0] >= event['at'] - .5 and time.time() - photo[0] <= 5:
                    return photo[2]
                await asyncio.sleep(.1)
        room = self._room(event['source_id'])
        if not self._idle(room):
            raise RuntimeError('Room camera is busy or offline; no automatic retry.')
        client_id = getattr(getattr(room, 'session', None), 'client_id', '')
        if event['source_id'] and client_id != event['source_id']:
            raise RuntimeError('The observed room is no longer connected.')
        if rule['media'] == 'photo':
            frame = await asyncio.wait_for(room._request_camera_frame_full('alert-' + delivery['id']), 35)
            data = getattr(frame, 'jpeg', None)
        else:
            result = await asyncio.wait_for(room._request_camera_clip('alert-' + delivery['id'],
                seconds=rule['clip_seconds'], fps=8), rule['clip_seconds'] + 20)
            data = result.get('data') if isinstance(result, dict) else result
        if not isinstance(data, bytes) or not 0 < len(data) <= CAMERA_CLIP_MAX_BYTES:
            raise RuntimeError('The camera did not return usable alert media.')
        return data

    async def _deliver(self, delivery):
        attempted = False
        try:
            async with self._send_lock:
                if not await asyncio.to_thread(self._unchanged, delivery):
                    await asyncio.to_thread(self._finish, delivery, 'skipped', 'Rule changed or was removed')
                    return
                if time.time() - delivery['event']['at'] > 30:
                    raise RuntimeError('Observation expired while waiting for delivery.')
                rule = delivery['rule']
                home, _ = self._home_for(rule, delivery['event'])
                if (quiet_now(rule, time.time())
                        or home_quiet_now(home, time.time(), rule['timezone'])):
                    await asyncio.to_thread(self._finish, delivery, 'skipped',
                                            'Quiet hours started before delivery')
                    return
                # ТЗ F-702: «выбор канала (Telegram, пуш на телефон, HUD)».
                if rule['channel'] == 'hud':
                    await self._deliver_hud(delivery)
                    return
                if rule['channel'] == 'push':
                    await self._deliver_push(delivery)
                    return
                provider = self.get_provider()
                if provider is None or not provider.ready:
                    raise RuntimeError('Telegram is unavailable.')
                data = await self._media(delivery)
                if not await asyncio.to_thread(self._unchanged, delivery):
                    await asyncio.to_thread(self._finish, delivery, 'skipped', 'Rule changed or was removed')
                    return
                if quiet_now(rule, time.time()) or home_quiet_now(home, time.time(), rule['timezone']):
                    await asyncio.to_thread(self._finish, delivery, 'skipped', 'Quiet hours started before delivery')
                    return
                caption = self._caption(rule, delivery['event'])
                kwargs = {'private_reply_to_user_id': self.owner_id} if rule['destination'] == 'owner' else {}
                attempted = True
                if rule['media'] == 'video':
                    ack = await provider.send_video(data, 'video/mp4', caption, 'presence.mp4', **kwargs)
                else:
                    ack = await provider.send_image(data, 'image/jpeg', caption, 'presence.jpg', **kwargs)
                destination = self.owner_id if rule['destination'] == 'owner' else self.group_id
                if (not isinstance(ack, dict) or ack.get('ok') is not True or ack.get('chat_id') != destination
                        or type(ack.get('message_id')) is not int or ack['message_id'] <= 0):
                    await asyncio.to_thread(self._finish, delivery, 'uncertain', 'Delivery acknowledgement was missing')
                else:
                    await asyncio.to_thread(self._finish, delivery, 'sent', f"Telegram message {ack['message_id']}")
        except asyncio.CancelledError:
            await asyncio.to_thread(self._finish, delivery, 'uncertain' if attempted else 'cancelled', 'Delivery interrupted; no automatic retry')
            raise
        except Exception as exc:
            from hub.telegram import _safe_text
            status = 'uncertain' if getattr(exc, 'uncertain', False) or (attempted and isinstance(exc, (TimeoutError, OSError))) else 'failed'
            await asyncio.to_thread(self._finish, delivery, status, _safe_text(str(exc)))
            log.warning('Presence alert %s (%s); no automatic retry', status, type(exc).__name__)

    def _event_text(self, rule, event):
        """Что случилось, словами (ТЗ F-702: у каждого вида события своя фраза)."""
        kind = str(event.get('kind') or 'presence')
        name = str(rule.get('name') or '')
        label = str(event.get('label') or '')
        zone = str(event.get('zone') or '')
        if kind == 'person_entered':
            who = name or ', '.join(str(item) for item in (event.get('names') or ())) or 'человек'
            return f'пришёл — {who}.'
        if kind == 'person_left':
            who = name or ', '.join(str(item) for item in (event.get('names') or ())) or 'человек'
            return f'ушёл — {who}.'
        if kind == 'unknown_appeared':
            return 'появился незнакомый человек.'
        if kind == 'zone_entered':
            return f'человек в зоне: {zone}.' if zone else 'человек перешёл в другую зону.'
        if kind == 'sound_event':
            return f'звук: {label or "событие"}.'
        if kind == 'object':
            return f'камера видит: {label or name}.'
        target = ('неопознанный человек' if rule['target'] == 'unknown'
                  else rule['name'] if rule['target'] == 'person' else 'человек')
        return f'камера заметила — {target}.'

    def _caption(self, rule, event):
        when = datetime.fromtimestamp(event['at'], ZoneInfo(rule['timezone'])).strftime('%Y-%m-%d %H:%M:%S %Z')
        source_id = event['source_id']
        room = self._room(source_id)
        workplace = ' '.join(str(getattr(room, 'workplace_name', '') or source_id or 'Комната').split())[:100]
        return f'Rowan · {workplace}: {self._event_text(rule, event)}\n{when}'

    async def _deliver_hud(self, delivery):
        """ТЗ F-702: канал HUD — подпись на экране комнаты, без медиа.

        Медиа здесь не запрашивается осознанно: подпись видит тот, кто и так в
        комнате, и дергать камеру ради неё значило бы мешать ходу.
        """
        rule, event = delivery['rule'], delivery['event']
        room = self._room(event['source_id'])
        if room is None or not self._idle(room):
            raise RuntimeError('The room is busy or offline; no automatic retry.')
        sender = getattr(room, 'send_json', None)
        if sender is None:
            raise RuntimeError('This room has no HUD channel.')
        await sender({'type': MSG_STATUS, 'text': self._caption(rule, event),
                      'ttl_s': float(DEFAULT_STATUS_TTL_S)})
        await asyncio.to_thread(self._finish, delivery, 'sent', 'HUD caption')

    async def _deliver_push(self, delivery):
        """ТЗ F-702: канал «пуш на телефон»; транспорт даёт F-712 (фаза 3)."""
        push = self.get_push()
        if push is None:
            await asyncio.to_thread(self._finish, delivery, 'skipped',
                                    'Push transport is not configured yet (ТЗ F-712)')
            return
        payload = {'title': 'Rowan', 'text': self._caption(delivery['rule'], delivery['event']),
                   'rule_id': delivery['rule_id'], 'source_id': delivery['event']['source_id'],
                   'at': delivery['event']['at'], 'kind': delivery['event'].get('kind', 'presence')}
        ack = await push(payload)
        if isinstance(ack, dict) and ack.get('ok') is False:
            raise RuntimeError(str(ack.get('error') or 'The push transport refused the message.'))
        await asyncio.to_thread(self._finish, delivery, 'sent', 'push')

    async def drain(self):
        """Wait for currently queued work; useful during a controlled shutdown."""
        await self._queue.join()
        if self._deliveries:
            await asyncio.gather(*tuple(self._deliveries), return_exceptions=True)

    async def close(self):
        self._closed = True
        tasks = [task for task in [self._worker, *self._deliveries] if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with self._lock:
            self._db.close()
