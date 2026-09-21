"""Current room occupants from one requested frame, without a vision LLM."""
from __future__ import annotations

import re
import uuid

from hub.image_prompt import action_revoked
from hub.room_state import valid_tracks
from hub.telegram_intent import _POLITE, _without_quotes

_CURRENT_PEOPLE = re.compile(
    r'^(?:кто\s+(?:(?:сейчас|там|есть|находится)\s+){0,3}(?:в\s+(?:комнате|зале)|у\s+нас)'
    r'(?:\s+(?:сейчас|сегодня|там))?|'
    r'who(?:\s+is|[’\x27]s)\s+(?:(?:currently|now)\s+)?(?:in\s+(?:the\s+)?(?:room|living\s+room)|here)'
    r'(?:\s+(?:right\s+)?now)?)\s*[?!.]*$', re.I)


def current_people_question(text):
    """Recognize the direct present-room question, never history or quotations."""
    if not isinstance(text, str) or action_revoked(text):
        return False
    return _CURRENT_PEOPLE.fullmatch(_POLITE.sub('', _without_quotes(text), count=1).strip()) is not None


async def inspect_current_people(connection, query):
    """Use existing transport and face matching; keep every fact frame-local."""
    record = {'tool': 'look_at_camera', 'args': {'query': query}}
    connection._utterance_actions.append(record)
    frame = await connection._request_camera_frame_full('people-' + uuid.uuid4().hex)
    if isinstance(frame, str):
        result = {'ok': False, 'error': frame}
    else:
        matches = await connection._camera_frame_people(frame)
        tracks = getattr(frame, 'tracks', None)
        tracks_available = isinstance(tracks, list)
        faces = matches.get('faces_in_frame', [])
        count = (max(len(valid_tracks(tracks)), len(faces)) if tracks_available else
                 len(faces) if matches.get('face_positions_available') else None)
        result = {'ok': True, **matches, 'fresh': True,
            'visible_people_count': count, 'count_is_lower_bound': not tracks_available,
            'identity_source': 'face matching on this newly requested camera frame',
            'note': 'Only names in faces_in_frame were matched in this fresh image. '
                'Unmatched faces and visible bodies remain unknown; never guess names. '
                'A face-only count is a lower bound, and no visible faces does not prove an empty room. '
                'No old presence state or vision-model identity guesses were used.'}
    record['result'] = result
    return result


def current_people_reply(result, text):
    """Describe only observed names/counts, including failed or partial views."""
    russian = bool(re.search('[А-Яа-яЁё]', str(text)))
    if not isinstance(result, dict) or result.get('ok') is not True:
        return ('Не удалось получить свежие данные с камеры. Сейчас не могу подтвердить, кто в комнате.'
                if russian else 'I could not get a fresh camera observation, so I cannot confirm who is in the room.')
    faces = result.get('faces_in_frame') or []
    names = list(dict.fromkeys(' '.join(str(face['name']).split()) for face in faces
                 if isinstance(face, dict) and isinstance(face.get('name'), str) and face['name'].strip()))
    count = result.get('visible_people_count')
    count = count if type(count) is int and count >= 0 else None
    if names:
        line = ('На свежем кадре распознал: ' if russian else 'In the fresh camera image I recognized: ') + ', '.join(names) + '.'
        unknown = max(0, count - len(names)) if count is not None else 0
        if unknown:
            minimum = ('как минимум ' if russian else 'at least ') if result.get('count_is_lower_bound') else ''
            line += (f' Ещё не распознано людей: {minimum}{unknown}.' if russian else
                     f' Another {minimum}{unknown} visible person(s) remain unidentified.')
        return line
    if count:
        minimum = ('как минимум ' if russian else 'at least ') if result.get('count_is_lower_bound') else ''
        return (f'На свежем кадре видно людей: {minimum}{count}, но я их не распознал.' if russian else
                f'The fresh camera image shows {minimum}{count} person(s), but I could not identify them.')
    if count == 0 and not result.get('count_is_lower_bound', True):
        return 'На свежем кадре людей не обнаружено.' if russian else 'No people were detected in the fresh camera image.'
    return ('На свежем кадре не удалось распознать людей. Не могу подтвердить, кто сейчас в комнате.' if russian else
            'I could not identify anyone in the fresh camera image. I cannot confirm who is currently in the room.')
