"""Telegram transport for the configured group and explicitly allowed users.

No automatic retries, arbitrary recipients, or credential
URLs in HTTP logs. The stdlib transport runs in a worker and exposes only
sanitized failures; it never enables urllib/http.client request debugging.
Polling methods are passive operations invoked only by the optional runtime.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from urllib import error, request

from PIL import Image

from common.config import TelegramConfig

MAX_TEXT_LENGTH = 4096
MAX_CAPTION_LENGTH = 1024
MAX_PHOTO_BYTES = 10_000_000
MAX_DOCUMENT_BYTES = 50_000_000
MAX_VIDEO_BYTES = 20_000_000
MAX_RESPONSE_BYTES = 1_000_000
_METHODS = frozenset({'getMe', 'getChat', 'sendMessage', 'sendPhoto', 'sendDocument',
                      'getUpdates', 'getWebhookInfo', 'getFile', 'sendVideo',
                      'editMessageText', 'answerCallbackQuery'})
_FORMATS = {'PNG': ('image/png', 'png'), 'JPEG': ('image/jpeg', 'jpg'), 'WEBP': ('image/webp', 'webp')}
_TOKEN = re.compile(r'[0-9]+:[A-Za-z0-9_-]{20,}')


class TelegramError(RuntimeError):
    """Safe-to-display error; uncertain sends must not be retried automatically."""

    def __init__(self, message, *, code=None, uncertain=False, retry_after=None):
        super().__init__(message)
        self.code = code
        self.uncertain = bool(uncertain)
        self.retry_after = retry_after


@dataclass(frozen=True)
class TelegramUpload:
    field: str
    filename: str
    mime: str
    data: bytes = field(repr=False)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A token-bearing request may never follow a server-supplied URL.
        return None


def _safe_text(value, token=''):
    text = str(value)
    if token:
        text = text.replace(token, '[redacted]')
    text = re.sub(r'https?://api\.telegram\.org/[^\s"<>]+', '[Telegram endpoint]', text, flags=re.I)
    text = _TOKEN.sub('[redacted]', text)
    return ''.join(c for c in text if c in '\n\t' or ord(c) >= 32)[:500]


def _valid_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _utf16_length(value):
    try:
        return len(value.encode('utf-16-le')) // 2
    except UnicodeError:
        raise TelegramError('Text contains an invalid Unicode character.') from None


def _multipart(payload, upload):
    boundary = 'rowan-' + uuid.uuid4().hex
    pieces = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        pieces.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                       str(value).encode('utf-8'), b'\r\n'])
    pieces.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="{upload.field}"; '
                   f'filename="{upload.filename}"\r\nContent-Type: {upload.mime}\r\n\r\n'.encode('ascii'),
                   upload.data, f'\r\n--{boundary}--\r\n'.encode()])
    return b''.join(pieces), f'multipart/form-data; boundary={boundary}'


def _keyboard(value):
    """Accept bounded callback-only keyboards; never pass through extra fields."""
    if (not isinstance(value, dict) or set(value) != {'inline_keyboard'}
            or not isinstance(value['inline_keyboard'], list) or len(value['inline_keyboard']) > 20):
        raise TelegramError('Provide a valid inline keyboard.')
    rows, count = [], 0
    for row in value['inline_keyboard']:
        if not isinstance(row, list) or not 1 <= len(row) <= 8:
            raise TelegramError('Inline keyboard rows must contain one to eight buttons.')
        buttons = []
        for button in row:
            if (not isinstance(button, dict) or set(button) != {'text', 'callback_data'}
                    or not isinstance(button['text'], str) or not button['text'].strip()
                    or _utf16_length(button['text']) > 128
                    or not isinstance(button['callback_data'], str)):
                raise TelegramError('Provide a valid callback button.')
            try:
                size = len(button['callback_data'].encode('utf-8'))
            except UnicodeError:
                size = 0
            if not 1 <= size <= 64:
                raise TelegramError('Callback data must contain one to 64 UTF-8 bytes.')
            buttons.append(dict(button))
            count += 1
        rows.append(buttons)
    if count > 100:
        raise TelegramError('The inline keyboard contains too many buttons.')
    return {'inline_keyboard': rows}


def _mp4(data):
    """Check bounded ISO-BMFF boxes without decoding or loading a media model."""
    if not isinstance(data, bytes) or not 32 <= len(data) <= MAX_VIDEO_BYTES:
        return False
    offset, seen, boxes = 0, set(), 0
    while offset < len(data) and boxes < 10000:
        if len(data) - offset < 8:
            return False
        size, kind = int.from_bytes(data[offset:offset + 4], 'big'), data[offset + 4:offset + 8]
        header = 8
        if size == 1:
            if len(data) - offset < 16:
                return False
            size, header = int.from_bytes(data[offset + 8:offset + 16], 'big'), 16
        elif size == 0:
            size = len(data) - offset
        if size < header or offset + size > len(data):
            return False
        if boxes == 0 and (kind != b'ftyp' or size < header + 8):
            return False
        if kind in {b'moov', b'mdat'} and size <= header:
            return False
        seen.add(kind)
        offset += size
        boxes += 1
    return offset == len(data) and {b'ftyp', b'moov', b'mdat'} <= seen


class TelegramProvider:
    """Destinations are the configured group or the exact configured controller.

    Optional test transport: ``async transport(method, payload, upload)`` returns
    a parsed Bot API envelope, where upload is a TelegramUpload or None.
    """

    def __init__(self, cfg: TelegramConfig, *, transport=None, private_recipient_allowed=None):
        self._enabled = bool(cfg.enabled)
        self._chat_id = cfg.chat_id
        self._control_user_id = getattr(cfg, 'control_user_id', None)
        self._timeout = float(cfg.timeout_s)
        self._token = os.environ.get(cfg.api_key_env, '').strip()
        self._transport = transport
        self._private_recipient_allowed = private_recipient_allowed

    @property
    def ready(self):
        return (self._enabled and _valid_int(self._chat_id) and self._chat_id < 0
                and _TOKEN.fullmatch(self._token) is not None)

    def _require_ready(self):
        if not self._enabled:
            raise TelegramError('Telegram is disabled in the brain configuration.')
        if not _valid_int(self._chat_id) or self._chat_id >= 0:
            raise TelegramError('Set the fixed Telegram group chat_id in the brain configuration.')
        if _TOKEN.fullmatch(self._token) is None:
            raise TelegramError('The Telegram bot token is missing or invalid; save it on the brain PC and restart.')

    def _request_sync(self, method, payload, upload, *, timeout=None):
        if method not in _METHODS:
            raise TelegramError('Unsupported Telegram operation.')
        if upload is None:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            content_type = 'application/json; charset=utf-8'
        else:
            body, content_type = _multipart(payload, upload)
        req = request.Request(f'https://api.telegram.org/bot{self._token}/{method}',
                              data=body, method='POST', headers={'Content-Type': content_type})
        opener = request.build_opener(_NoRedirect(), request.HTTPSHandler(debuglevel=0))
        status = 200
        try:
            with opener.open(req, timeout=self._timeout if timeout is None else timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                status = response.status
        except error.HTTPError as response:
            status = response.code
            with response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise TelegramError('Telegram returned an oversized response; delivery is unconfirmed.',
                                code=status, uncertain=method.startswith('send'))
        try:
            envelope = json.loads(raw)
        except (ValueError, UnicodeError):
            raise TelegramError('Telegram returned an unreadable response; delivery is unconfirmed.',
                                code=status, uncertain=method.startswith('send')) from None
        if not isinstance(envelope, dict):
            raise TelegramError('Telegram returned an invalid response; delivery is unconfirmed.',
                                code=status, uncertain=method.startswith('send'))
        if not 200 <= status < 300:
            envelope['ok'] = False
            envelope.setdefault('error_code', status)
        return envelope

    async def _call(self, method, payload, upload=None, *, timeout=None):
        self._require_ready()
        request_timeout = self._timeout if timeout is None else timeout
        try:
            if self._transport is not None:
                response = await asyncio.wait_for(self._transport(method, payload, upload), request_timeout + 1)
            else:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._request_sync, method, payload, upload, timeout=request_timeout),
                    request_timeout + 1)
        except asyncio.CancelledError:
            # A worker already writing to the network may still complete.
            raise
        except TelegramError as exc:
            # Also sanitize exceptions supplied by an injected transport.
            raise TelegramError(_safe_text(str(exc), self._token), code=exc.code,
                                uncertain=exc.uncertain, retry_after=exc.retry_after) from None
        except Exception:
            # urllib errors can include the entire token-bearing URL. Never
            # propagate their text, exception chain, repr, or traceback details.
            raise TelegramError('Telegram connection failed; delivery is unconfirmed. Do not retry automatically.',
                                uncertain=method.startswith('send')) from None
        if not isinstance(response, dict) or not isinstance(response.get('ok'), bool):
            raise TelegramError('Telegram returned an invalid response; delivery is unconfirmed.',
                                uncertain=method.startswith('send'))
        if response['ok'] is not True:
            code = response.get('error_code')
            code = code if _valid_int(code) else None
            parameters = response.get('parameters')
            parameters = parameters if isinstance(parameters, dict) else {}
            retry_after = parameters.get('retry_after')
            retry_after = retry_after if _valid_int(retry_after) and retry_after >= 0 else None
            description = _safe_text(response.get('description', 'Telegram rejected the request.'), self._token)
            message = f'Telegram rejected the request{f" ({code})" if code else ""}: {description}'
            if 'migrate_to_chat_id' in parameters:
                message += ' The group migrated; update its configured chat_id. No destination was changed.'
            raise TelegramError(message, code=code, retry_after=retry_after,
                                uncertain=method.startswith('send') and (code is None or code >= 500))
        result = response.get('result')
        if method == 'answerCallbackQuery':
            if result is not True:
                raise TelegramError('Telegram did not confirm the callback response.')
            return True
        if method == 'getUpdates':
            if isinstance(result, list) and all(isinstance(item, dict) for item in result):
                return result
            raise TelegramError('Telegram returned an invalid updates list.')
        if not isinstance(result, dict):
            raise TelegramError('Telegram did not return a usable result; delivery is unconfirmed.',
                                uncertain=method.startswith('send'))
        return result

    def _destination(self, private_reply_to_user_id, group_chat_id=None):
        """Where this call goes: the configured group, one named group, or a DM.

        ТЗ F-702: a notification rule may target any group the bot has been
        added to, not only the fixed ``chat_id`` of the brain. The id still has
        to be a real group id (negative), and the response must come back with
        that same chat, so a typo cannot silently deliver elsewhere.
        """
        if group_chat_id is not None:
            if private_reply_to_user_id is not None:
                raise TelegramError('Pick one Telegram destination for this message.')
            if not _valid_int(group_chat_id) or group_chat_id >= 0:
                raise TelegramError('A Telegram group destination must be a negative chat id.')
            return group_chat_id
        if private_reply_to_user_id is None:
            return self._chat_id
        if not _valid_int(private_reply_to_user_id) or not 0 < private_reply_to_user_id < 2 ** 63:
            raise TelegramError('Private Telegram replies are restricted to authorized numeric accounts.')
        owner = (_valid_int(self._control_user_id) and self._control_user_id > 0
                 and private_reply_to_user_id == self._control_user_id)
        allowed = False
        if not owner and callable(self._private_recipient_allowed):
            try:
                allowed = self._private_recipient_allowed(private_reply_to_user_id) is True
            except Exception:
                allowed = False
        if not owner and not allowed:
            raise TelegramError('Private Telegram replies are restricted to authorized accounts.')
        return private_reply_to_user_id

    def _sent(self, result, kind, *, destination=None):
        destination = self._chat_id if destination is None else destination
        chat = result.get('chat')
        message_id = result.get('message_id')
        if (not isinstance(chat, dict) or not _valid_int(chat.get('id')) or chat['id'] != destination
                or not _valid_int(message_id) or message_id <= 0):
            raise TelegramError('Telegram did not confirm delivery to the requested destination. Do not retry automatically.',
                                uncertain=True)
        return dict(ok=True, chat_id=destination, message_id=message_id, kind=kind)

    @staticmethod
    def _reply(payload, reply_to_message_id):
        if reply_to_message_id is not None:
            if not _valid_int(reply_to_message_id) or reply_to_message_id <= 0:
                raise TelegramError('A Telegram reply needs a valid message ID.')
            payload['reply_parameters'] = dict(message_id=reply_to_message_id, allow_sending_without_reply=True)
        return payload

    async def send_text(self, text, *, reply_to_message_id=None, private_reply_to_user_id=None,
                        group_chat_id=None, reply_markup=None):
        self._require_ready()
        destination = self._destination(private_reply_to_user_id, group_chat_id)
        if not isinstance(text, str) or not text.strip():
            raise TelegramError('Provide a nonempty Telegram message.')
        if _utf16_length(text) > MAX_TEXT_LENGTH:
            raise TelegramError('Telegram messages must fit within 4096 characters; shorten this message.')
        payload = self._reply({'chat_id': destination, 'text': text}, reply_to_message_id)
        if reply_markup is not None:
            payload['reply_markup'] = _keyboard(reply_markup)
        result = await self._call('sendMessage', payload)
        return self._sent(result, 'text', destination=destination)

    async def edit_text(self, text, *, message_id, private_reply_to_user_id=None, reply_markup=None):
        self._require_ready()
        destination = self._destination(private_reply_to_user_id)
        if not _valid_int(message_id) or message_id <= 0:
            raise TelegramError('Editing a Telegram message needs a valid message ID.')
        if not isinstance(text, str) or not text.strip() or _utf16_length(text) > MAX_TEXT_LENGTH:
            raise TelegramError('Provide a nonempty Telegram message within 4096 characters.')
        payload = {'chat_id': destination, 'message_id': message_id, 'text': text}
        if reply_markup is not None:
            payload['reply_markup'] = _keyboard(reply_markup)
        result = await self._call('editMessageText', payload)
        receipt = self._sent(result, 'text', destination=destination)
        if receipt['message_id'] != message_id:
            raise TelegramError('Telegram did not confirm the requested message edit.', uncertain=True)
        return receipt

    async def answer_callback(self, callback_query_id, text='', show_alert=False):
        if (not isinstance(callback_query_id, str) or not 1 <= len(callback_query_id) <= 256
                or not isinstance(text, str) or _utf16_length(text) > 200 or type(show_alert) is not bool):
            raise TelegramError('Provide a valid callback ID and a response within 200 characters.')
        return await self._call('answerCallbackQuery', {'callback_query_id': callback_query_id,
                               'text': text, 'show_alert': show_alert, 'cache_time': 0})

    async def send_video(self, data, mime='video/mp4', caption='', filename='presence.mp4', *,
                         reply_to_message_id=None, private_reply_to_user_id=None, group_chat_id=None):
        self._require_ready()
        destination = self._destination(private_reply_to_user_id, group_chat_id)
        if mime != 'video/mp4' or not _mp4(data):
            raise TelegramError('Provide a complete MP4 video within 20 MB.')
        if not isinstance(caption, str) or _utf16_length(caption) > MAX_CAPTION_LENGTH:
            raise TelegramError('Telegram video captions must fit within 1024 characters.')
        raw_name = str(filename).replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
        safe_name = (re.sub(r'[^A-Za-z0-9_-]', '_', raw_name)[:80] or 'presence') + '.mp4'
        payload = self._reply({'chat_id': destination}, reply_to_message_id)
        if caption:
            payload['caption'] = caption
        result = await self._call('sendVideo', payload, TelegramUpload('video', safe_name, mime, data))
        return self._sent(result, 'video', destination=destination)

    async def send_image(self, data, mime, caption='', filename='image.png', *, reply_to_message_id=None,
                         private_reply_to_user_id=None, group_chat_id=None):
        self._require_ready()
        destination = self._destination(private_reply_to_user_id, group_chat_id)
        if not isinstance(caption, str) or _utf16_length(caption) > MAX_CAPTION_LENGTH:
            raise TelegramError('Telegram image captions must fit within 1024 characters.')
        if not isinstance(data, bytes) or not data or len(data) > MAX_DOCUMENT_BYTES:
            raise TelegramError('Provide a nonempty image no larger than 50 MB.')
        if mime not in {entry[0] for entry in _FORMATS.values()}:
            raise TelegramError('Telegram image sending supports PNG, JPEG and WebP.')
        try:
            with Image.open(BytesIO(data)) as picture:
                image_format = picture.format
                if image_format not in _FORMATS or _FORMATS[image_format][0] != mime:
                    raise ValueError('Image MIME mismatch')
                width, height = picture.size
                if width <= 0 or height <= 0 or getattr(picture, 'n_frames', 1) != 1:
                    raise ValueError('Invalid or animated image')
                picture.verify()
        except Exception:
            raise TelegramError('The selected image could not be validated as a static PNG, JPEG or WebP.') from None
        # Choose exactly one API method before any upload. There is no retry or
        # sendDocument fallback after an ambiguous/failed sendPhoto response.
        as_photo = (image_format in {'PNG', 'JPEG'} and len(data) <= MAX_PHOTO_BYTES
                    and width + height <= 10000 and max(width, height) / min(width, height) <= 20)
        kind = 'photo' if as_photo else 'document'
        raw_name = str(filename).replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
        safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', raw_name)[:80] or 'image'
        safe_name += '.' + _FORMATS[image_format][1]
        upload = TelegramUpload(kind, safe_name, mime, data)
        payload = {'chat_id': destination}
        if caption:
            payload['caption'] = caption
        self._reply(payload, reply_to_message_id)
        result = await self._call('sendPhoto' if as_photo else 'sendDocument', payload, upload)
        return self._sent(result, kind, destination=destination)

    async def check_connection(self):
        """Read bot/chat identity; does not send a test message or prove delivery."""
        bot = await self.get_me()
        chat = await self._call('getChat', {'chat_id': self._chat_id})
        if chat.get('id') != self._chat_id or chat.get('type') not in {'group', 'supergroup'}:
            raise TelegramError('Telegram did not confirm the configured group chat.')
        return dict(ok=True,
                    bot={key: _safe_text(bot[key], self._token) if key != 'id' else bot[key]
                         for key in ('id', 'username', 'first_name') if key in bot},
                    chat={key: _safe_text(chat[key], self._token) if key != 'id' else chat[key]
                          for key in ('id', 'type', 'title') if key in chat})

    async def get_me(self):
        bot = await self._call('getMe', {})
        if not _valid_int(bot.get('id')) or bot['id'] <= 0 or bot.get('is_bot') is not True:
            raise TelegramError('Telegram did not confirm the configured bot identity.')
        return {key: bot[key] if key in {'id', 'is_bot'} else _safe_text(bot[key], self._token)
                for key in ('id', 'is_bot', 'username', 'first_name') if key in bot}

    async def get_updates(self, offset=None, timeout=20):
        if offset is not None and not _valid_int(offset):
            raise TelegramError('Telegram polling offset must be an integer.')
        if not _valid_int(timeout) or not 0 <= timeout <= 50:
            raise TelegramError('Telegram polling timeout must be between 0 and 50 seconds.')
        payload = dict(timeout=timeout, limit=100, allowed_updates=['message', 'callback_query'])
        if offset is not None:
            payload['offset'] = offset
        return await self._call('getUpdates', payload, timeout=self._timeout + timeout)

    async def get_webhook_info(self):
        result = await self._call('getWebhookInfo', {})
        # The destination URL can itself contain a secret. The runtime only
        # needs to detect a conflicting webhook, never expose or delete it.
        return dict(url='[configured webhook]' if result.get('url') else '',
                    pending_update_count=result.get('pending_update_count')
                    if _valid_int(result.get('pending_update_count')) else None)

    def _download_sync(self, file_path, max_bytes):
        req = request.Request(f'https://api.telegram.org/file/bot{self._token}/{file_path}', method='GET')
        opener = request.build_opener(_NoRedirect(), request.HTTPSHandler(debuglevel=0))
        with opener.open(req, timeout=self._timeout) as response:
            length = response.headers.get('Content-Length')
            if length and length.isdigit() and int(length) > max_bytes:
                raise TelegramError('The Telegram image exceeds the download limit.')
            raw = response.read(max_bytes + 1)
        if not raw or len(raw) > max_bytes:
            raise TelegramError('The Telegram image exceeds the download limit or is empty.')
        return raw

    async def download_photo(self, file_id, max_bytes=8_000_000):
        """Download one referenced Telegram photo; no caller-controlled URL/path."""
        if not isinstance(file_id, str) or not file_id or len(file_id) > 512:
            raise TelegramError('A Telegram photo requires a valid file ID.')
        if not _valid_int(max_bytes) or not 1 <= max_bytes <= MAX_PHOTO_BYTES:
            raise TelegramError('The Telegram image download limit must be at most 10 MB.')
        result = await self._call('getFile', {'file_id': file_id})
        file_path, size = result.get('file_path'), result.get('file_size')
        if (not isinstance(file_path, str) or not re.fullmatch(r'[A-Za-z0-9_./-]{1,512}', file_path)
                or file_path.startswith('/') or '..' in file_path.split('/')):
            raise TelegramError('Telegram returned an invalid image location.')
        if _valid_int(size) and size > max_bytes:
            raise TelegramError('The Telegram image exceeds the download limit.')
        try:
            raw = await asyncio.wait_for(asyncio.to_thread(self._download_sync, file_path, max_bytes), self._timeout + 1)
            with Image.open(BytesIO(raw)) as image:
                if image.format not in _FORMATS or getattr(image, 'n_frames', 1) != 1:
                    raise TelegramError('The Telegram attachment is not a supported static image.')
                mime = _FORMATS[image.format][0]
                image.verify()
            return raw, mime
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            raise TelegramError(_safe_text(str(exc), self._token)) from None
        except Exception:
            raise TelegramError('The Telegram image could not be downloaded or validated.') from None

    async def close(self):
        """No persistent HTTP client or background polling to close."""
