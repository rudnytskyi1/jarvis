"""Durable Telegram authorization, non-secret settings and redacted audit data."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

CAPABILITIES = ('chat', 'images', 'camera', 'pc', 'memory', 'profiles')
ROLES = ('admin', 'operator', 'member', 'blocked')
_DEFAULTS = {'owner': set(CAPABILITIES), 'admin': set(CAPABILITIES),
             'operator': {'chat', 'images', 'camera', 'pc'},
             'member': {'chat', 'images'}, 'blocked': set()}
_SECRET_KEY = re.compile(r'(?:^|[.\-_:])(?:token|secret|password|credentials?|api[_-]?key|'
                         r'access[_-]?token|bot[_-]?token|authorization)(?:$|[.\-_:])', re.I)
_SECRET_VALUE = re.compile(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b|\bsk-[A-Za-z0-9_-]{16,}\b|'
                           r'-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+\S+', re.I)
#: Home ids look exactly like the ``homes.home_id`` pattern of section 14.
_HOME_ID = re.compile(r'^[a-z0-9][a-z0-9_-]{0,63}$')


def contains_secret(value):
    """Panel input is not a credential provisioning interface."""
    if isinstance(value, dict):
        return any(_SECRET_KEY.search(str(key)) or contains_secret(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_secret(item) for item in value)
    return bool(_SECRET_VALUE.search(str(value)) or re.search(
        r'\b(?:password|token|secret|api[_ -]?key|пароль|токен)\s*[:=]\s*\S+', str(value), re.I))


def _safe(value):
    if isinstance(value, dict):
        return {str(k)[:100]: '[redacted]' if _SECRET_KEY.search(str(k)) else _safe(v)
                for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value[:40]]
    if isinstance(value, str):
        return '[redacted]' if contains_secret(value) else value[:2000]
    return value if value is None or type(value) in (bool, int, float) else str(type(value).__name__)


def _user_id(value):
    if type(value) is not int or not 0 < value < 2 ** 63:
        raise ValueError('Telegram user ID must be a positive integer.')
    return value


class TelegramAdminState:
    """Writes commit before publishing snapshots; authorization reads never do I/O.

    Construct once off-loop and share this instance with the runtime. Settings
    and permission reads use immutable snapshots, so a worker writing SQLite
    cannot block the voice event loop. Persistence changes use this API; a new
    process reloads both snapshots from the database.
    """

    def __init__(self, path, owner_id):
        self.path = Path(path)
        self.owner_id = _user_id(owner_id)
        self._write_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS telegram_access (
                    user_id INTEGER PRIMARY KEY, role TEXT NOT NULL DEFAULT 'member',
                    capabilities TEXT NOT NULL DEFAULT '{}', label TEXT NOT NULL DEFAULT '',
                    explicit INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS telegram_settings (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS telegram_admin_audit (
                    id INTEGER PRIMARY KEY, ts REAL NOT NULL, actor INTEGER NOT NULL,
                    event TEXT NOT NULL, details TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS telegram_home_owners (
                    user_id INTEGER NOT NULL, home_id TEXT NOT NULL, granted REAL NOT NULL,
                    PRIMARY KEY (user_id, home_id));
            ''')
            self._users_cache = {row['user_id']: dict(row) for row in db.execute('SELECT * FROM telegram_access')}
            self._settings_cache = {row['key']: row['value'] for row in db.execute('SELECT * FROM telegram_settings')}
            owners: dict[int, set[str]] = {}
            for row in db.execute('SELECT user_id, home_id FROM telegram_home_owners'):
                owners.setdefault(int(row['user_id']), set()).add(str(row['home_id']))
            self._homes_cache = {user_id: frozenset(homes) for user_id, homes in owners.items()}

    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def is_owner(self, user_id):
        return type(user_id) is int and user_id == self.owner_id

    def _row(self, user_id):
        if type(user_id) is not int or user_id <= 0:
            return None
        row = self._users_cache.get(user_id)
        return dict(row) if row else None

    def role(self, user_id):
        if self.is_owner(user_id):
            return 'owner'
        if type(user_id) is not int or user_id <= 0:
            return 'blocked'
        row = self._row(user_id)
        return row['role'] if row else 'member'

    def allows(self, user_id, capability):
        if capability not in CAPABILITIES:
            return False
        if type(user_id) is not int or user_id <= 0:
            return False
        if self.is_owner(user_id):
            return True
        row = self._row(user_id)
        role = row['role'] if row else 'member'
        if role == 'blocked':
            return False
        overrides = json.loads(row['capabilities']) if row else {}
        return overrides.get(capability, capability in _DEFAULTS[role]) is True

    def can_chat(self, user_id, private=False):
        if self.is_owner(user_id):
            return True
        if type(user_id) is not int or user_id <= 0:
            return False
        row = self._row(user_id)
        role = row['role'] if row else 'member'
        overrides = json.loads(row['capabilities']) if row else {}
        return (role != 'blocked' and overrides.get('chat', 'chat' in _DEFAULTS[role]) is True
                and (not private or bool(row and row['explicit'])))

    def users(self):
        with self._db() as db:
            rows = db.execute('SELECT * FROM telegram_access ORDER BY explicit DESC,label,user_id').fetchall()
        values = [{'user_id': self.owner_id, 'role': 'owner', 'label': 'Owner',
                   'explicit': True, 'capabilities': {cap: True for cap in CAPABILITIES}}]
        for row in rows:
            if row['user_id'] == self.owner_id:
                continue
            value = dict(row)
            value['capabilities'] = json.loads(value['capabilities'])
            value['explicit'] = bool(value['explicit'])
            values.append(value)
        return values

    # --- ТЗ F-701: чьи дома видит этот Telegram-аккаунт ---------------------

    @staticmethod
    def _home_id(home_id):
        value = str(home_id or '').strip()
        if not _HOME_ID.match(value):
            raise ValueError('Unknown home.')
        return value

    def homes_of(self, user_id):
        """The homes granted to this account (empty when none).

        The hub admin is NOT handled here: ``homes_of`` answers about grants
        only, so a caller that wants "all homes" asks the admin first (see
        ``hub/telegram_homes.py``). This keeps the store honest about what was
        actually granted.
        """
        if type(user_id) is not int or user_id <= 0:
            return frozenset()
        return self._homes_cache.get(user_id, frozenset())

    def is_home_owner(self, user_id):
        """True when this account owns at least one home (ТЗ F-701)."""
        return bool(self.homes_of(user_id))

    def grant_home(self, user_id, home_id):
        """Give one home to one account; returns True when it is new."""
        _user_id(user_id)
        value = self._home_id(home_id)
        with self._write_lock:
            current = self._homes_cache.get(user_id, frozenset())
            if value in current:
                return False
            with self._db() as db:
                db.execute('INSERT OR REPLACE INTO telegram_home_owners VALUES(?,?,?)',
                           (user_id, value, time.time()))
            self._homes_cache = {**self._homes_cache, user_id: frozenset(current | {value})}
        return True

    def revoke_home(self, user_id, home_id):
        """Take one home back; returns True when the grant existed."""
        _user_id(user_id)
        value = self._home_id(home_id)
        with self._write_lock:
            current = self._homes_cache.get(user_id, frozenset())
            if value not in current:
                return False
            left = frozenset(current - {value})
            with self._db() as db:
                db.execute('DELETE FROM telegram_home_owners WHERE user_id=? AND home_id=?',
                           (user_id, value))
            updated = dict(self._homes_cache)
            if left:
                updated[user_id] = left
            else:
                updated.pop(user_id, None)
            self._homes_cache = updated
        return True

    def owners_of(self, home_id):
        """Every account that owns this home, smallest id first."""
        value = self._home_id(home_id)
        return tuple(sorted(user_id for user_id, homes in self._homes_cache.items() if value in homes))

    def home_owners(self):
        """All grants as ``{user_id: (home_id, ...)}`` for the panel."""
        return {user_id: tuple(sorted(homes)) for user_id, homes in sorted(self._homes_cache.items())}

    def set_user(self, user_id, role, capabilities=None, label=''):
        _user_id(user_id)
        if self.is_owner(user_id):
            raise ValueError('The owner\'s access cannot be changed.')
        if role not in ROLES:
            raise ValueError('Unknown Telegram role.')
        if capabilities is not None and (not isinstance(capabilities, dict) or any(
                key not in CAPABILITIES or type(value) is not bool for key, value in capabilities.items())):
            raise ValueError('Permissions must be a dictionary of known capabilities with true/false values.')
        if not isinstance(label, str) or len(label) > 120 or contains_secret(label):
            raise ValueError('Enter a short label without secrets.')
        with self._write_lock:
            old = self._row(user_id)
            overrides = capabilities if capabilities is not None else (json.loads(old['capabilities']) if old else {})
            row = dict(user_id=user_id, role=role, capabilities=json.dumps(overrides),
                       label=label or (old['label'] if old else ''), explicit=1, updated=time.time())
            with self._db() as db:
                db.execute('INSERT INTO telegram_access VALUES(?,?,?,?,1,?) ON CONFLICT(user_id) DO UPDATE SET '
                           'role=excluded.role,capabilities=excluded.capabilities,label=excluded.label,'
                           'explicit=1,updated=excluded.updated',
                           (user_id, role, row['capabilities'], row['label'], row['updated']))
            self._users_cache = {**self._users_cache, user_id: row}

    def remove_user(self, user_id):
        _user_id(user_id)
        if self.is_owner(user_id):
            raise ValueError('The owner cannot be removed.')
        with self._write_lock:
            with self._db() as db:
                db.execute('DELETE FROM telegram_access WHERE user_id=?', (user_id,))
                # ТЗ F-701: дом без владельца лучше, чем дом у удалённого аккаунта.
                db.execute('DELETE FROM telegram_home_owners WHERE user_id=?', (user_id,))
            self._users_cache = {key: row for key, row in self._users_cache.items() if key != user_id}
            self._homes_cache = {key: homes for key, homes in self._homes_cache.items()
                                 if key != user_id}

    def observe_user(self, sender):
        if (not isinstance(sender, dict) or type(sender.get('id')) is not int
                or not 0 < sender['id'] < 2 ** 63 or sender.get('is_bot') is not False):
            return
        label = ' '.join(str(sender.get(key, '')).strip() for key in ('first_name', 'last_name')).strip()
        if not label:
            label = str(sender.get('username', ''))
        label = '[redacted]' if contains_secret(label) else label[:120]
        with self._write_lock:
            with self._db() as db:
                db.execute('INSERT OR IGNORE INTO telegram_access VALUES(?,\'member\',\'{}\',?,0,?)',
                           (sender['id'], label, time.time()))
                row = db.execute('SELECT * FROM telegram_access WHERE user_id=?', (sender['id'],)).fetchone()
            self._users_cache = {**self._users_cache, sender['id']: dict(row)}

    def get_setting(self, key, default=None):
        value = self._settings_cache.get(key)
        return json.loads(value) if value is not None else default

    def set_setting(self, key, value):
        if (not isinstance(key, str) or not key or len(key) > 160 or _SECRET_KEY.search(key)
                or contains_secret(value)):
            raise ValueError('Secrets cannot be saved through the settings panel.')
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded) > 32000:
            raise ValueError('The setting value is too large.')
        with self._write_lock:
            with self._db() as db:
                db.execute('INSERT INTO telegram_settings VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET '
                           'value=excluded.value,updated=excluded.updated', (key, encoded, time.time()))
            self._settings_cache = {**self._settings_cache, key: encoded}

    def audit(self, actor, event, details=None):
        _user_id(actor)
        with self._db() as db:
            db.execute('INSERT INTO telegram_admin_audit(ts,actor,event,details) VALUES(?,?,?,?)',
                       (time.time(), actor, str(event)[:120], json.dumps(_safe(details or {}), ensure_ascii=False)))

    def events(self, limit=30):
        limit = max(1, min(100, int(limit)))
        with self._db() as db:
            rows = db.execute('SELECT * FROM telegram_admin_audit ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        return [{**dict(row), 'details': json.loads(row['details'])} for row in rows]

    def update_mapping_setting(self, key, item, value):
        """Atomically merge one entry; concurrent client hellos cannot lose peers."""
        with self._write_lock:
            current = self.get_setting(key, {})
            if not isinstance(current, dict):
                raise ValueError('Setting must be a mapping.')
            current[item] = value
            self.set_setting(key, current)

    def close(self):
        """There is no retained connection to close."""


TelegramAccess = TelegramAdminState
