"""Validated non-secret settings overrides for the Telegram owner panel."""
from __future__ import annotations

import copy

from common.config import Config

CATEGORIES = {
    'llm': 'AI and budget', 'stt': 'Speech recognition', 'diarization': 'Speaker separation',
    'speaker': 'Voice recognition', 'face': 'Faces and greetings', 'tts': 'Response voice',
    'image_generation': 'Image generation', 'segment': 'Object detection',
    'audio_recording': 'Request recordings', 'camera_request_recording': 'Request photos',
    'training_archive': 'Training archive', 'access': 'Room access',
    'telegram': 'Telegram', 'client': 'Camera, microphone and client',
}
EXCLUDE = {'api_key', 'api_key_env', 'base_url', 'vision_base_url', 'prompt_file',
           'path', 'checkpoint', 'model_path', 'kokoro_model_path', 'kokoro_voices_path',
           'host', 'port', 'server_url', 'stream_url', 'password', 'token', 'secret', 'devices', 'apps', 'vosk_model'}
IMMUTABLE = {'server.telegram.enabled', 'server.telegram.chat_id', 'server.telegram.control_user_id'}
LIVE = {'server.permissions_enabled', 'server.llm.verify_actions',
        'server.llm.monthly_budget_usd', 'server.stt.language', 'server.stt.allowed_languages',
        'server.stt.hotwords', 'server.stt.live_transcript', 'server.stt.live_interval_s',
        'server.stt.live_window_s', 'server.speaker.threshold', 'server.speaker.margin',
        'server.speaker.admin_threshold', 'server.speaker.min_speech_s', 'server.speaker.enabled',
        'server.face.threshold', 'server.face.greetings_enabled', 'server.face.appearance_enabled',
        'server.face.adaptive_recognition', 'server.face.greeting_llm', 'server.face.greet_after_s',
        'server.face.greeting_cooldown_s', 'server.face.greeting_cooldown_known_s',
        'server.face.burst_size', 'server.face.enroll_bursts', 'server.telegram.poll_timeout_s'}


def _lookup(cfg, key):
    value = cfg
    for part in key.split('.'):
        value = getattr(value, part)
    return value


def _assign(cfg, key, value):
    parts = key.split('.')
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], copy.deepcopy(value))


def catalogue(cfg):
    rows = []
    def walk(obj, prefix, category):
        for name, field in type(obj).model_fields.items():
            if name in EXCLUDE:
                continue
            value, key = getattr(obj, name), prefix + '.' + name
            if hasattr(type(value), 'model_fields'):
                walk(value, key, name if prefix == 'server' else category)
                continue
            if not (value is None or type(value) in (str, int, float, bool, list)):
                continue
            # Models and local device parameters are displayed only as data.
            # Client controls require their own acknowledgement/persistence path.
            editable = key not in IMMUTABLE and key.startswith('server.')
            schema = type(obj).model_json_schema()['properties'][name]
            variants = schema.get('anyOf', [schema])
            scalar = next((item for item in variants if item.get('type') != 'null'), schema)
            kind = scalar.get('type', 'string')
            kind = {'integer': 'int', 'number': 'float', 'boolean': 'bool', 'array': 'list'}.get(kind, kind)
            row = dict(key=key, label=name.replace('_', ' '), value=value, type=kind,
                       category='access' if key == 'server.permissions_enabled' else category,
                       description=field.description or '', editable=editable,
                       requires_restart=editable and key not in LIVE)
            if key.startswith('client.'):
                row['description'] = 'Living-room PC setting. This is the value from the server configuration; change it on the client.'
            for dest, source in [('min', 'minimum'), ('max', 'maximum'), ('choices', 'enum')]:
                if source in scalar:
                    row[dest] = scalar[source]
            rows.append(row)
    walk(cfg.server, 'server', 'access')
    walk(cfg.client, 'client', 'client')
    return rows


def validated_value(cfg, key, value):
    spec = next((row for row in catalogue(cfg) if row['key'] == key and row['editable']), None)
    if spec is None:
        raise ValueError('This setting cannot be changed in Telegram.')
    if isinstance(value, str):
        text = value.strip()
        if spec['type'] == 'bool':
            if text.casefold() not in {'true', 'false', '1', '0', 'да', 'нет', 'on', 'off'}:
                raise ValueError('Enter true or false.')
            value = text.casefold() in {'true', '1', 'да', 'on'}
        elif spec['type'] == 'int':
            value = int(text)
        elif spec['type'] == 'float':
            value = float(text)
        elif spec['type'] == 'list':
            import json
            value = json.loads(text) if text.startswith('[') else [part.strip() for part in text.split(',') if part.strip()]
        elif text.casefold() == 'null':
            value = None
    data = cfg.model_dump()
    obj = data
    parts = key.split('.')
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = value
    candidate = Config.model_validate(data)
    return _lookup(candidate, key)


def restore_overrides(cfg, state):
    """Only reviewed, valid server keys are read before initializing engines."""
    for row in catalogue(cfg):
        if not row['editable']:
            continue
        saved = state.get_setting('config:' + row['key'])
        if not isinstance(saved, dict) or 'value' not in saved:
            continue
        try:
            _assign(cfg, row['key'], validated_value(cfg, row['key'], saved['value']))
        except (ValueError, TypeError):
            continue  # A stale override cannot prevent the owner panel starting.


def apply_live(cfg, key, value, runtime):
    """Update the same config instance and copied runtime fields explicitly."""
    _assign(cfg, key, value)
    if key.startswith('server.speaker.'):
        voice = runtime.get('voices')
        attribute = key.rsplit('.', 1)[1]
        if voice is not None and attribute != 'admin_threshold':
            setattr(voice, attribute, value)
    if key == 'server.face.threshold' and runtime.get('face') is not None:
        runtime['face'].threshold = value
    if key.startswith('server.stt.') and runtime.get('stt') is not None:
        stt = runtime['stt']
        if key.endswith('.language'):
            stt.default_language = value
        elif key.endswith('.allowed_languages'):
            stt.allowed_languages = list(value)
        elif key.endswith('.hotwords'):
            stt.hotwords = ', '.join(value)[:1024]
    if key == 'server.llm.monthly_budget_usd':
        from decimal import Decimal
        llm = runtime.get('llm')
        for target in (getattr(llm, '_responses', None), runtime.get('image_generator')):
            budget = getattr(target, 'budget', None)
            if budget is not None:
                budget.limit = int(Decimal(str(value)) * 1_000_000)
