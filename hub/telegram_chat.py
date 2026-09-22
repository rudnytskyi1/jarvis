"""Group mentions/replies and shared history, with durable claims before work.

The Telegram ID namespace never shares voice profiles, roles or room history.
Only polling retries automatically; uncertain work/delivery is never replayed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from hub.api_budget import BudgetExceeded
from hub.conversations import Conversations
from hub.image_prompt import action_revoked, is_image_request, visual_request
from hub.telegram import TelegramError
from hub.untrusted import TELEGRAM_SOURCE
from hub.untrusted import wrap as wrap_untrusted

log = logging.getLogger(__name__)
PROMPT = Path(__file__).resolve().parents[1] / 'prompts' / 'telegram.md'
MAX_PHOTO_BYTES = 8_000_000
MAX_CONTEXT_BYTES = 40_000
_OTHER_MEDIA = ('document', 'video', 'animation', 'sticker', 'audio', 'voice', 'video_note')
_REQUEST_START = (r'''^[^\w"'“”‘’«»`]*'''
                  r'(?:(?:can|could|would|will)\s+you\s+|(?:можешь|можете)\s+|'
                  r'(?:please|pls|пожалуйста)\b[\s,:!]+|'
                  r'I\s+(?:want|would\s+like)\s+(?:you\s+)?to\s+){0,3}')
_NEW_IMAGE = re.compile(_REQUEST_START +
                        r'(?:draw|paint|sketch|illustrate|render|generate|create|'
                        r'make\s+(?:a|an)\s+(?:image|picture|photo|portrait|wallpaper)|'
                        r'сделай\s+(?:картинку|изображение|фото|портрет)|'
                        r'нарисуй|нарисовать|сгенерируй|сгенерировать|изобрази)\b', re.I)
_ATTACHED_EDIT = re.compile(
    _REQUEST_START +
    r'(?:(?:edit|modify|change|transform|add|remove|replace|put|give)\b|'
    r'(?:make|turn)\s+(?:me|us|him|her|them|it|this|that|the\s+(?:photo|picture|image))\b|'
    r'(?:сделай|измени|измените|преврати|превратите|отредактируй|добавь|убери|замени|поставь|надень)\b)', re.I)
_VISUAL_COMMAND = re.compile(
    _REQUEST_START +
    r'(?P<verb>draw|paint|sketch|illustrate|render|generate|create|edit|modify|change|remove|'
    r'replace|add|put|give|make|turn|transform|нарисуй|нарисовать|изобрази|изобразить|'
    r'сгенерируй|сгенерировать|дорисуй|дорисовать|отредактируй|сделай|сделать|измени|измените|'
    r'преврати|превратить|превратите|добавь|добавить|надень|надеть|поставь|убери|замени)\b'
    r'(?P<detail>[\s\S]*)', re.I)
_NONVISUAL_DRAW = re.compile(
    r'^\s*(?:(?:me|us)\s+)?(?:(?:a|an|the|some|any|no|your|our|my)\s+)?'
    r'(?:conclusions?|comparisons?|analogies|analogy|inferences?|distinctions?|attention|'
    r'lots|cards|money|blood|water|breath|curtains)\b|'
    r'^\s*(?:on|upon)\s+|^\s*up\s+(?:(?:a|the)\s+)?(?:contract|agreement|plan)\b', re.I)


def current_image_request(text, *, has_photo=False):
    """Require a present affirmative visual command before spending on images.

    A visual verb mentioned in a complaint, quotation, explanation or ordinary
    question is not permission to generate. Classification never rewrites the
    prompt that is later sent to the image provider.
    """
    value = visual_request(text)
    if action_revoked(value):
        return False
    command = _VISUAL_COMMAND.match(value)
    if command is None or not re.search(r'\w', command['detail']):
        return False
    if command['verb'].casefold() == 'draw' and _NONVISUAL_DRAW.search(command['detail']):
        return False
    if command['verb'].casefold() in {'create', 'generate', 'render', 'sketch'} and re.match(
        r'^\s*(?:out\s+)?(?:(?:a|an|the|some)\s+)?'
        r'(?:conclusions?|code|functions?|scripts?|programs?|verdicts?|judgments?|arguments?|strategies|strategy)\b',
        command['detail'], re.I):
        return False
    return bool(is_image_request(value) or (has_photo and _ATTACHED_EDIT.match(value)))


class TelegramInputError(RuntimeError):
    """A local, deliberately user-facing input error with no transport details."""


def _integer(value):
    return type(value) is int


def addressed_text(message, bot_id, username):
    """Accept a real mention or reply to this bot, using Telegram sender IDs."""
    parent = message.get('reply_to_message')
    parent_sender = parent.get('from') if isinstance(parent, dict) else None
    reply_to_bot = (isinstance(parent_sender, dict) and _integer(bot_id) and bot_id > 0
                    and _integer(parent_sender.get('id')) and parent_sender['id'] == bot_id)
    text = message.get('text')
    entities = message.get('entities')
    if not isinstance(text, str):
        text, entities = message.get('caption'), message.get('caption_entities')
    if not isinstance(text, str):
        return None
    if not isinstance(entities, list):
        entities = []
    try:
        encoded = text.encode('utf-16-le')
    except UnicodeError:
        return None
    spans = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        start, length = entity.get('offset'), entity.get('length')
        if not _integer(start) or not _integer(length) or start < 0 or length <= 0:
            continue
        left, right = start * 2, (start + length) * 2
        if right > len(encoded):
            continue
        try:
            value = encoded[left:right].decode('utf-16-le')
        except UnicodeError:
            continue
        direct = entity.get('type') == 'mention' and value.casefold() == '@' + username.casefold()
        user = entity.get('user')
        named = (entity.get('type') == 'text_mention' and isinstance(user, dict)
                 and _integer(user.get('id')) and user['id'] == bot_id)
        if direct or named:
            spans.append((left, right))
    if not spans:
        return text.strip() if reply_to_bot else None
    # Ignore overlapping/malformed entities rather than deleting other words.
    accepted = []
    for start, end in sorted(set(spans)):
        if not accepted or start >= accepted[-1][1]:
            accepted.append((start, end))
    for start, end in reversed(accepted):
        encoded = encoded[:start] + encoded[end:]
    return encoded.decode('utf-16-le').strip()


def _clip_text(text, maximum=4000):
    # Telegram limits are UTF-16 code units, not Python code points.
    raw = str(text).encode('utf-16-le', errors='replace')
    return raw[:maximum * 2].decode('utf-16-le', errors='ignore')


def _update_chats(update):
    """Every group chat one update carries (ТЗ F-702 destination list).

    A Telegram bot cannot ask for the list of its groups, so the hub learns
    them from the updates it receives: a message, being added to a group
    (``my_chat_member``), an edit, or a button press. Private chats are not
    collected here - the panel knows those accounts already.
    """
    if not isinstance(update, dict):
        return []
    sources = [update.get('message'), update.get('edited_message'),
               update.get('channel_post'), update.get('my_chat_member')]
    callback = update.get('callback_query')
    if isinstance(callback, dict):
        sources.append(callback.get('message'))
    chats = []
    for item in sources:
        chat = item.get('chat') if isinstance(item, dict) else None
        if isinstance(chat, dict) and chat.get('type') in {'group', 'supergroup'}:
            chats.append(chat)
    return chats


def _chat_title(chat, fallback=''):
    """A group name fit for a button: one line, no control characters."""
    raw = str(chat.get('title') or fallback or chat.get('id') or '')
    clean = ' '.join(''.join(ch if ch >= ' ' else ' ' for ch in raw).split())
    return clean[:80]


#: How often a known group's "seen" stamp is refreshed (seconds). The cap on
#: the destination list drops the least recently used group, and this is the
#: resolution of "recently": one settings write per group per hour, not one per
#: message in the group.
_CHAT_REFRESH_S = 3600.0


class TelegramChat:
    def __init__(self, provider, cfg, reply, image_generator=None, image_store=None,
                 folder=Path('data/telegram'), *, control_reply=None, admin_handler=None, access=None):
        self.provider, self.cfg, self.reply = provider, cfg, reply
        self.control_reply = control_reply
        self.admin_handler, self.access = admin_handler, access
        self.image_generator, self.image_store = image_generator, image_store
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.database = self.folder / 'updates.sqlite3'
        self.history = Conversations(self.folder / 'history')
        self.group_owner = f'telegram:{self.cfg.chat_id}'
        self.system_prompt = PROMPT.read_text(encoding='utf-8')
        self._task = None
        self._process_lock = asyncio.Lock()
        self._initialized = False
        self._started_at = time.time()
        self.bot_id, self.username = None, ''
        self.last_error = None
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY CHECK(id=1), next_id INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO cursor VALUES(1,0)')
            db.execute('CREATE TABLE IF NOT EXISTS updates (update_id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, '
                       'message_id INTEGER NOT NULL, sender_id INTEGER NOT NULL, state TEXT NOT NULL, '
                       'UNIQUE(chat_id,message_id))')
            db.execute('CREATE TABLE IF NOT EXISTS admin_dispatch ('
                       'update_id INTEGER PRIMARY KEY, event_key TEXT UNIQUE NOT NULL, state TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS telegram_photo_replies ('
                       'chat_id INTEGER NOT NULL, sender_id INTEGER NOT NULL, message_id INTEGER NOT NULL, '
                       'photos TEXT NOT NULL, PRIMARY KEY(chat_id,sender_id,message_id))')
        with self.history._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS telegram_turn_metadata ('
                       'turn_id INTEGER PRIMARY KEY, metadata TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS telegram_messages ('
                       'chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, '
                       'ts TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL, '
                       'addressed INTEGER NOT NULL, PRIMARY KEY(chat_id,message_id))')

    def _control_sender(self, message):
        allowed = getattr(self.cfg, 'control_user_id', None)
        sender = message.get('from')
        if (not isinstance(sender, dict) or not _integer(sender.get('id')) or sender['id'] <= 0
                or sender.get('is_bot') is True or message.get('sender_chat')):
            return False
        if _integer(allowed) and allowed > 0 and sender['id'] == allowed:
            return True
        return (self.access is not None and sender.get('is_bot') is False
                and self._can_chat(message)
                and (any(self._allows(sender['id'], cap) for cap in ('camera', 'pc', 'memory', 'profiles'))
                     or (isinstance(message.get('photo'), list) and bool(message['photo'])
                         and self._allows(sender['id'], 'images'))))

    def _allows(self, user_id, capability):
        allowed = getattr(self.cfg, 'control_user_id', None)
        if _integer(allowed) and allowed > 0 and user_id == allowed:
            return True
        if self.access is None:
            return capability in {'chat', 'images'}
        try:
            return self.access.allows(user_id, capability) is True
        except Exception:
            return False

    def _can_chat(self, message):
        sender, chat = message.get('from'), message.get('chat')
        if (not isinstance(sender, dict) or not _integer(sender.get('id')) or sender['id'] <= 0
                or sender.get('is_bot') is True or not isinstance(chat, dict)
                or not _integer(chat.get('id')) or message.get('sender_chat')):
            return False
        private = chat.get('type') == 'private'
        if (private and chat['id'] != sender['id']) or (not private and (
                chat.get('type') not in {'group', 'supergroup'} or chat['id'] != self.cfg.chat_id)):
            return False
        owner = getattr(self.cfg, 'control_user_id', None)
        if _integer(owner) and owner > 0 and sender['id'] == owner:
            return True
        if self.access is None:
            return not private
        try:
            return sender.get('is_bot') is False and self.access.can_chat(sender['id'], private=private) is True
        except Exception:
            return False

    def _control_message(self, message):
        if not self._control_sender(message):
            return False
        chat = message.get('chat')
        if not isinstance(chat, dict) or not _integer(chat.get('id')):
            return False
        if chat.get('type') == 'private':
            return chat['id'] == message['from']['id']
        return chat.get('type') in {'group', 'supergroup'} and chat['id'] == self.cfg.chat_id

    def _history_owner(self, message):
        if message['chat']['type'] == 'private':
            return f'telegram:dm:{message["from"]["id"]}'
        return self.group_owner

    def _image_owner(self, message):
        owner = self._history_owner(message)
        return owner if message['chat']['type'] == 'private' else f'{owner}:{message["from"]["id"]}'

    def _with_reply_photo(self, message):
        """Expose a verified bot reply's photo without changing the requester."""
        if message.get('photo') or any(message.get(key) for key in _OTHER_MEDIA):
            return message  # A current attachment wins, including invalid input.
        parent = message.get('reply_to_message')
        if not isinstance(parent, dict):
            return message
        sender, photos = parent.get('from'), parent.get('photo')
        if (not _integer(self.bot_id) or self.bot_id <= 0 or not isinstance(sender, dict)
                or not _integer(sender.get('id')) or sender['id'] != self.bot_id
                or sender.get('is_bot') is not True
                or parent.get('sender_chat') or parent.get('is_automatic_forward')
                or any(parent.get(key) for key in ('forward_origin', 'forward_date', 'forward_from',
                                                   'forward_from_chat', 'forward_sender_name'))):
            return message
        if 'chat' in parent:
            parent_chat, current_chat = parent['chat'], message.get('chat')
            if (not isinstance(parent_chat, dict) or not isinstance(current_chat, dict)
                    or not _integer(parent_chat.get('id')) or not _integer(current_chat.get('id'))
                    or parent_chat['id'] != current_chat['id']):
                return message
        if not isinstance(photos, list) or not photos:
            current_sender, current_chat = message.get('from'), message.get('chat')
            if (not isinstance(current_sender, dict) or not _integer(current_sender.get('id'))
                    or not isinstance(current_chat, dict) or not _integer(current_chat.get('id'))
                    or not _integer(parent.get('message_id'))):
                return message
            with self._db() as db:
                row = db.execute('SELECT photos FROM telegram_photo_replies WHERE chat_id=? AND sender_id=? '
                                 'AND message_id=?', (current_chat['id'], current_sender['id'], parent['message_id'])).fetchone()
            photos = json.loads(row[0]) if row else None
            if not isinstance(photos, list) or not photos:
                return message
        # Keep author, destination and reply ID from the current message. Only
        # photo descriptors are copied; downloading still requires an image edit.
        return {**message, 'photo': [dict(photo) if isinstance(photo, dict) else photo for photo in photos]}

    def _remember_photo_reply(self, message, receipt):
        if (not isinstance(receipt, dict) or receipt.get('ok') is not True
                or not _integer(receipt.get('message_id')) or receipt['message_id'] <= 0):
            return
        photos = message.get('photo')
        if not isinstance(photos, list) or not 1 <= len(photos) <= 20:
            return
        encoded = json.dumps(photos, ensure_ascii=False)
        if len(encoded.encode('utf-8')) > 16000:
            return
        with self._db() as db:
            db.execute('INSERT OR IGNORE INTO telegram_photo_replies VALUES(?,?,?,?)',
                       (message['chat']['id'], message['from']['id'], receipt['message_id'], encoded))

    def _delivery_kwargs(self, message):
        if not self._can_chat(message):
            raise TelegramInputError('This Telegram account is not authorized for this conversation.')
        values = {'reply_to_message_id': message['message_id']}
        if message['chat']['type'] == 'private':
            values['private_reply_to_user_id'] = message['from']['id']
        return values

    def _author_metadata(self, message):
        sender = message['from']
        name = ' '.join(str(sender.get(key) or '') for key in ('first_name', 'last_name')).strip()
        name = _clip_text(' '.join(name.split()), 150)
        metadata = {'sender_id': sender['id'], 'author': name or f'Telegram user {sender["id"]}',
                    'message_id': message['message_id']}
        username = sender.get('username')
        if isinstance(username, str) and re.fullmatch(r'[A-Za-z0-9_]{1,64}', username):
            metadata['username'] = username
        parent = message.get('reply_to_message')
        if isinstance(parent, dict) and isinstance(parent.get('from'), dict):
            parent_sender = parent['from']
            if (_integer(self.bot_id) and _integer(parent_sender.get('id'))
                    and parent_sender['id'] == self.bot_id
                    and _integer(parent.get('message_id'))):
                content = parent.get('text', parent.get('caption', ''))
                metadata['reply_to_bot_message'] = {
                    'message_id': parent['message_id'],
                    'text': _clip_text(content, 2000) if isinstance(content, str) else '',
                }
        return metadata

    def _remember_message(self, message, addressed):
        text = message.get('text', message.get('caption'))
        if not isinstance(text, str) or not text.strip():
            return
        stamp = datetime.fromtimestamp(message['date'], UTC).isoformat()
        metadata = json.dumps(self._author_metadata(message), ensure_ascii=False)
        with self.history._db() as db:
            db.execute('INSERT OR IGNORE INTO telegram_messages VALUES(?,?,?,?,?,?)',
                       (message['chat']['id'], message['message_id'], stamp, text, metadata, int(addressed)))

    def _begin_group_turn(self, message, stamp, text):
        metadata = json.dumps(self._author_metadata(message), ensure_ascii=False)
        with self.history._db() as db:
            turn = db.execute('INSERT INTO turns(person,ts,question,answer) VALUES(?,?,?,?)',
                              (self._history_owner(message), stamp, text or '[Mention]',
                               '[Request interrupted or answer not completed.]')).lastrowid
            db.execute('INSERT INTO telegram_turn_metadata VALUES(?,?)', (turn, metadata))
        return turn

    @staticmethod
    def _context_text(text, stamp, metadata):
        # JSON preserves untrusted display names/text as data, with the Telegram
        # numeric sender ID explicitly separate from the current requester.
        return json.dumps({**metadata, 'timestamp': stamp, 'text': text}, ensure_ascii=False)

    @classmethod
    def _wrapped_context_text(cls, text, stamp, metadata):
        """ТЗ F-411: chat history is text from outside the room — marked as data.

        The CURRENT request is not wrapped: it is the message the sender wrote
        and the caller has already checked who they are. Prior turns and other
        members' lines are exactly what must not read as instructions.
        """
        return wrap_untrusted(cls._context_text(text, stamp, metadata),
                              source=TELEGRAM_SOURCE)

    def _recent_group_context(self, message=None):
        """Read one route's exchanges; private history never includes the group."""
        private = message is not None and message['chat']['type'] == 'private'
        history_owner = self._history_owner(message) if message is not None else self.group_owner
        where, params = ('t.person=?', [history_owner])
        if not private:
            where += ' OR t.person GLOB ?'
            params.append(self.group_owner + ':[0-9]*')
        with self.history._db() as db:
            turns = db.execute(
                'SELECT t.id,t.person,t.ts,t.question,t.answer,m.metadata FROM turns t '
                'LEFT JOIN telegram_turn_metadata m ON m.turn_id=t.id '
                'WHERE ' + where + ' '
                'ORDER BY julianday(t.ts) DESC,t.id DESC LIMIT 25',
                params).fetchall()
            ambient = [] if private else db.execute(
                'SELECT message_id,ts,text,metadata FROM telegram_messages '
                'WHERE chat_id=? AND addressed=0 '
                'ORDER BY julianday(ts) DESC,message_id DESC LIMIT 25',
                (self.cfg.chat_id,)).fetchall()
        timeline = []
        for identifier, owner, stamp, question, answer, raw in turns:
            if raw:
                metadata = json.loads(raw)
            else:
                # Previous versions keyed each member separately. Reading them
                # together preserves their attribution and all original rows.
                suffix = owner.removeprefix('telegram:dm:' if private else self.group_owner + ':')
                sender_id = int(suffix) if suffix.isdecimal() else None
                metadata = {'sender_id': sender_id,
                            'author': f'Telegram user {sender_id}' if sender_id else 'Unknown Telegram author'}
            block = [{'role': 'user', 'content': self._wrapped_context_text(question, stamp, metadata)},
                     {'role': 'assistant', 'content': answer}]
            timeline.append((stamp, metadata.get('message_id', identifier), block))
        for identifier, stamp, text, raw in ambient:
            metadata = json.loads(raw)
            metadata['group_context_only'] = True
            timeline.append((stamp, identifier,
                             [{'role': 'user',
                               'content': self._wrapped_context_text(text, stamp, metadata)}]))
        def chronological(item):
            try:
                moment = datetime.fromisoformat(item[0].replace('Z', '+00:00')).timestamp()
            except (ValueError, OverflowError):
                moment = 0
            return moment, item[1]
        timeline.sort(key=chronological)
        # Keep exchange pairs together when a busy group exceeds input budget.
        remaining, selected = 0, []
        for _, _, block in reversed(timeline):
            size = sum(len(message['content'].encode('utf-8')) for message in block)
            if remaining + size > MAX_CONTEXT_BYTES:
                break
            selected.append(block)
            remaining += size
        return [message for block in reversed(selected) for message in block]

    def _db(self):
        return sqlite3.connect(self.database, timeout=10)

    def _offset(self):
        with self._db() as db:
            return db.execute('SELECT next_id FROM cursor WHERE id=1').fetchone()[0]

    def _advance(self, update_id):
        with self._db() as db:
            db.execute('UPDATE cursor SET next_id=MAX(next_id,?) WHERE id=1', (update_id + 1,))

    def _claim(self, update_id, message):
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            offset = db.execute('SELECT next_id FROM cursor WHERE id=1').fetchone()[0]
            if update_id < offset:
                return False
            db.execute('UPDATE cursor SET next_id=? WHERE id=1', (update_id + 1,))
            row = db.execute('INSERT OR IGNORE INTO updates VALUES(?,?,?,?,?)',
                             (update_id, message['chat']['id'], message['message_id'],
                              message['from']['id'], 'claimed'))
            return row.rowcount == 1

    #: ТЗ F-702: how many groups the notification destination list remembers.
    CHAT_LIMIT = 20

    def _remember_chat(self, update):
        """Remember a group the bot has met, so the panel can offer it.

        The settings row grows only when a chat is new or its title changed,
        and it is capped at :data:`CHAT_LIMIT` groups (oldest seen first out),
        so a bot added to a hundred groups cannot grow one row without a bound.
        """
        if self.access is None:
            return
        chats = _update_chats(update)
        if not chats:
            return
        known = self.access.get_setting('chats', {})
        if not isinstance(known, dict):
            known = {}
        updated = dict(known)
        changed = False
        now = time.time()
        for chat in chats:
            chat_id, title = chat.get('id'), _chat_title(chat)
            if not _integer(chat_id) or chat_id >= 0 or not title:
                continue
            key = str(chat_id)
            entry = dict(updated.get(key) or {})
            same = entry.get('title') == title and entry.get('type') == chat.get('type')
            # "seen" is refreshed at most once an hour: enough for the cap to
            # drop the least recently used group, without a write per message.
            stale = now - float(entry.get('seen') or 0.0) >= _CHAT_REFRESH_S
            if not same or stale:
                entry.update(id=chat_id, title=title,
                             type=str(chat.get('type') or 'group'), seen=now)
                updated[key] = entry
                changed = True
        if len(updated) > self.CHAT_LIMIT:
            # Ties are real: three groups met inside one clock tick share the
            # same ``seen`` stamp, and the dictionary order is then the only
            # thing that says which of them is newer (the last one met).
            order = sorted(
                enumerate(updated.items()),
                key=lambda item: (float((item[1][1] or {}).get('seen') or 0.0), item[0]),
                reverse=True,
            )
            updated = dict(entry for _, entry in order[:self.CHAT_LIMIT])
            changed = True
        if changed:
            self.access.set_setting('chats', updated)

    def _state(self, update_id, state):
        with self._db() as db:
            db.execute('UPDATE updates SET state=? WHERE update_id=?', (state, update_id))

    def _claim_admin(self, update):
        query, message = update.get('callback_query'), update.get('message')
        key = f'update:{update["update_id"]}'
        if isinstance(query, dict) and isinstance(query.get('id'), str) and 0 < len(query['id']) <= 256:
            key = 'callback:' + query['id']
        elif isinstance(message, dict) and isinstance(message.get('chat'), dict):
            chat_id, message_id = message['chat'].get('id'), message.get('message_id')
            if _integer(chat_id) and _integer(message_id):
                key = f'message:{chat_id}:{message_id}'
        with self._db() as db:
            return db.execute('INSERT OR IGNORE INTO admin_dispatch VALUES(?,?,?)',
                              (update['update_id'], key, 'claimed')).rowcount == 1

    def _admin_state(self, update_id, state):
        with self._db() as db:
            db.execute('UPDATE admin_dispatch SET state=? WHERE update_id=?', (state, update_id))

    def start(self):
        if self._task is None or self._task.done():
            self._initialized = False
            self._started_at = time.time()
            self._task = asyncio.create_task(self.run())
        return self._task

    async def stop(self):
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._task = None

    async def initialize(self):
        """Read identity and acknowledge startup backlog without any replies."""
        if self._initialized:
            return
        webhook = await self.provider.get_webhook_info()
        if webhook.get('url'):
            raise TelegramError('This Telegram bot already has a webhook; polling was not started.')
        bot = await self.provider.get_me()
        if (not isinstance(bot, dict) or not _integer(bot.get('id')) or bot['id'] <= 0
                or not isinstance(bot.get('username'), str)
                or re.fullmatch(r'[A-Za-z0-9_]{1,64}', bot['username']) is None):
            raise TelegramError('Telegram did not provide a usable bot identity.')
        self.bot_id, self.username = bot['id'], bot['username']
        panel = getattr(self.admin_handler, '__self__', self.admin_handler)
        identify = getattr(panel, 'set_identity', None)
        if callable(identify):
            identify(self.bot_id, self.username)
        # A bot being enabled/restarted is not permission to answer old chat.
        for _ in range(20):
            updates = await self.provider.get_updates(offset=await asyncio.to_thread(self._offset), timeout=0)
            ids = [item['update_id'] for item in updates
                   if isinstance(item, dict) and _integer(item.get('update_id')) and item['update_id'] >= 0]
            if ids:
                await asyncio.to_thread(self._advance, max(ids))
            if len(updates) < 100:
                self._started_at = time.time()
                self._initialized = True
                return
        raise TelegramError('Telegram startup backlog is still draining; replies remain paused.')

    async def run(self):
        delay = 1.0
        while True:
            try:
                await self.initialize()
                updates = await self.provider.get_updates(offset=await asyncio.to_thread(self._offset),
                                                         timeout=int(getattr(self.cfg, 'poll_timeout_s', 20)))
                delay = 1.0
                self.last_error = None
                for update in updates:
                    await self.process_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Exception strings may carry transport URLs/tokens. Log only type.
                self.last_error = type(exc).__name__
                log.warning('Telegram polling paused (%s)', self.last_error)
                retry_after = getattr(exc, 'retry_after', None)
                wait = retry_after if _integer(retry_after) and retry_after > 0 else delay
                await asyncio.sleep(min(30.0, max(1.0, wait)))
                delay = min(30.0, delay * 2)

    def _group_message(self, update):
        message = update.get('message')
        if not isinstance(message, dict):
            return None
        chat, sender = message.get('chat'), message.get('from')
        if (not isinstance(chat, dict) or not _integer(chat.get('id'))
                or not isinstance(sender, dict) or not _integer(sender.get('id')) or sender['id'] <= 0
                or sender.get('is_bot') is True or message.get('sender_chat')
                or not _integer(message.get('message_id')) or message['message_id'] <= 0
                or not _integer(message.get('date')) or message['date'] < int(self._started_at)
                or message.get('forward_origin') or message.get('forward_date')):
            return None
        if self._can_chat(message):
            return message
        return None

    def _eligible(self, update):
        message = self._group_message(update)
        if message is None:
            return None
        text = addressed_text(message, self.bot_id, self.username)
        if message['chat']['type'] == 'private' and text is None:
            text = message.get('text', message.get('caption'))
            if isinstance(text, str):
                text = text.strip()
            else:
                text = None
        if text is None and isinstance(message.get('photo'), list) and message['photo']:
            parent = message.get('reply_to_message')
            sender = parent.get('from') if isinstance(parent, dict) else None
            reply = (isinstance(sender, dict) and _integer(sender.get('id'))
                     and sender['id'] == self.bot_id and sender.get('is_bot') is True)
            if message['chat']['type'] == 'private' or reply:
                text = ''
        return (message, text) if text is not None else None

    async def process_update(self, update):
        if not isinstance(update, dict) or not _integer(update.get('update_id')) or update['update_id'] < 0:
            return False
        update_id = update['update_id']
        async with self._process_lock:
            if update_id < await asyncio.to_thread(self._offset):
                return False
            # ТЗ F-702: the destination list learns every group the bot meets.
            await asyncio.to_thread(self._remember_chat, update)
            if callable(self.admin_handler):
                if not await asyncio.to_thread(self._claim_admin, update):
                    await asyncio.to_thread(self._advance, update_id)
                    return False
                try:
                    handled = await self.admin_handler(update) is True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # An administrative side effect may already have completed.
                    # Preserve the claim instead of replaying or entering chat.
                    log.warning('Telegram admin request stopped (%s)', type(exc).__name__)
                    await asyncio.to_thread(self._admin_state, update_id, 'failed')
                    await asyncio.to_thread(self._advance, update_id)
                    return True
                await asyncio.to_thread(self._admin_state, update_id, 'handled' if handled else 'ignored')
                if handled:
                    await asyncio.to_thread(self._advance, update_id)
                    return True
            message = self._group_message(update)
            observe = getattr(self.access, 'observe_user', None)
            if message is not None and callable(observe):
                await asyncio.to_thread(observe, message['from'])
            selected = self._eligible(update) if message is not None else None
            if message is not None:
                await asyncio.to_thread(self._remember_message, message, selected is not None)
            if selected is None:
                await asyncio.to_thread(self._advance, update_id)
                return False
            message, text = selected
            if not await asyncio.to_thread(self._claim, update_id, message):
                return False
            await self._answer(update_id, message, text)
            return True

    async def _answer(self, update_id, message, text):
        if not self._can_chat(message):
            await asyncio.to_thread(self._state, update_id, 'access_revoked')
            return
        message = await asyncio.to_thread(self._with_reply_photo, message)
        # Images retain their sender-specific provenance; textual chat is shared.
        owner = self._image_owner(message)
        stamp = datetime.fromtimestamp(message['date'], UTC).isoformat()
        recent = await asyncio.to_thread(self._recent_group_context, message)
        turn = await asyncio.to_thread(self._begin_group_turn, message, stamp, text)
        sending = False
        try:
            system_prompt = self.system_prompt
            if message['chat']['type'] == 'private':
                system_prompt += ('\nThis request is a private Telegram conversation with the current sender. '
                                  'The supplied history belongs only to this private conversation; '
                                  'no group history is provided or accessible here. '
                                  'References above to shared group history do not apply to this route.')
            messages = [{'role': 'system', 'content': system_prompt}, *recent,
                        {'role': 'user', 'content': self._context_text(
                            text or 'Hello.', stamp, self._author_metadata(message))}]
            has_photo = isinstance(message.get('photo'), list) and bool(message['photo'])
            if has_photo and not text.strip():
                answer = 'Что сделать с фотографией: описать, найти объект, распознать человека или изменить изображение?'
                sending = True
                receipt = await self.provider.send_text(answer, **self._delivery_kwargs(message))
                await asyncio.to_thread(self._remember_photo_reply, message, receipt)
            elif self._control_message(message) and callable(self.control_reply):
                # This callback is installed by the server and repeats the ID
                # check there. Group content, names and quoted senders confer no
                # authorization to enter the tool-enabled execution path.
                answer = _clip_text(await self.control_reply(messages, message, text)).strip()
                if not answer:
                    answer = "I couldn't produce an answer to that request."
                sending = True
                await self.provider.send_text(answer, **self._delivery_kwargs(message))
            elif current_image_request(text, has_photo=has_photo):
                if not self._allows(message['from']['id'], 'images'):
                    raise TelegramInputError('Image generation is not enabled for this Telegram account.')
                if self.image_generator is None or self.image_store is None:
                    raise TelegramInputError('Image generation is not configured here yet.')
                self.image_generator.check_ready()
                prompt = visual_request(text)
                reference, mime = await self._image_reference(message, prompt, owner)
                if not self._can_chat(message) or not self._allows(message['from']['id'], 'images'):
                    raise TelegramInputError('Image generation permission was revoked.')
                generated = await self.image_generator.generate(prompt, reference, mime)
                await asyncio.to_thread(self.image_store.save, owner, generated, self.image_generator.cfg.model)
                await asyncio.to_thread(self._state, update_id, 'generated')
                sending = True
                await self.provider.send_image(generated.png, 'image/png', **self._delivery_kwargs(message))
                answer = '[Generated image sent.]'
            else:
                answer = _clip_text(await self.reply(messages)).strip()
                if not answer:
                    answer = "I couldn't produce an answer to that message."
                sending = True
                await self.provider.send_text(answer, **self._delivery_kwargs(message))
            await asyncio.to_thread(self.history.finish, turn, answer)
            await asyncio.to_thread(self._state, update_id, 'sent')
        except asyncio.CancelledError:
            # The committed claim/offset already prevents replay after restart.
            raise
        except Exception as exc:
            log.warning('Telegram request stopped (%s, sending=%s)', type(exc).__name__, sending)
            state = 'delivery_uncertain' if sending else 'failed'
            answer = '[Reply delivery was not confirmed; no automatic retry.]' if sending else '[Request failed; no automatic retry.]'
            if not sending:
                line = ('Не получилось выполнить запрос. Автоматически повторять его не буду.'
                        if re.search('[А-Яа-яЁё]', text) else
                        "I couldn't finish that request. I won't retry it automatically.")
                if isinstance(exc, TelegramInputError):
                    line = str(exc)
                elif isinstance(exc, BudgetExceeded):
                    line = 'The monthly API budget is exhausted.'
                try:
                    await self.provider.send_text(line, **self._delivery_kwargs(message))
                except Exception:
                    state = 'delivery_uncertain'
            await asyncio.to_thread(self.history.finish, turn, answer)
            await asyncio.to_thread(self._state, update_id, state)

    async def _image_reference(self, message, prompt, owner):
        message = await asyncio.to_thread(self._with_reply_photo, message)
        photos = message.get('photo')
        if (not isinstance(photos, list) or not photos) and any(message.get(key) for key in _OTHER_MEDIA):
            raise TelegramInputError('Attach the image as a Telegram photo. This attachment type is not supported for image edits.')
        if isinstance(photos, list) and photos:
            candidates = [p for p in photos if isinstance(p, dict) and isinstance(p.get('file_id'), str)
                          and _integer(p.get('width')) and _integer(p.get('height'))
                          and p['width'] > 0 and p['height'] > 0
                          and (p.get('file_size') is None or
                               (_integer(p['file_size']) and 0 < p['file_size'] <= MAX_PHOTO_BYTES))]
            if not candidates:
                raise TelegramInputError('The attached photo exceeds the image input limit.')
            selected = max(candidates, key=lambda p: p['width'] * p['height'])
            return await self.provider.download_photo(selected['file_id'])
        if _NEW_IMAGE.search(prompt):
            return None, 'image/jpeg'
        previous = await asyncio.to_thread(self.image_store.last, owner)
        if previous is None:
            raise TelegramInputError('Attach a photo to your edit request, or generate an image first.')
        return previous[0].png, 'image/png'
