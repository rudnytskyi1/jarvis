"""Choose saved identity references from literal requests, never room memory."""
from __future__ import annotations

import re

from hub.image_prompt import action_revoked, person_reference_requested

_SELF = ('me', 'myself', 'меня')
_POSSESSIVE = re.compile(
    r'\b(?P<word>my|мой|мою|моей|моём|моем|моя|мое|моё|мои)\s+'
    r'(?:(?:own|собственн\w*)\s+)?(?:face|head|body|hair|eyes|nose|mouth|lips|beard|'
    r'appearance|look|outfit|clothes|лиц\w*|голов\w*|тел\w*|волос\w*|глаз\w*|нос\w*|'
    r'рот\w*|губ\w*|бород\w*|внешност\w*|одежд\w*)\b', re.I)
_RECIPIENT = re.compile(
    r'\b(?:draw|paint|generate|create|show|give|send|sketch|render)\s+me\s+'
    r'(?:a|an|the|some|this|that)\b|'
    r'\bmake\s+me\s+(?:a|an|the)\s+(?:picture|image|photo|portrait|drawing)\b', re.I)


def _canonical_names(names):
    result = {}
    for name in names or []:
        if isinstance(name, str) and name.strip():
            clean = ' '.join(name.split())
            result.setdefault(clean.casefold(), clean)
    return result


def _person_text(text, name, names):
    """A short profile name must not also match inside another full name."""
    value = text
    for other in names:
        if len(other) > len(name) and re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', other, re.I):
            value = re.sub(r'(?<!\w)' + re.escape(other) + r'(?!\w)',
                           lambda match: ' ' * len(match.group()), value, flags=re.I)
    return value


def _self_requested(text):
    # "Draw me a cat" names the recipient, not a depicted person. Removing
    # only that me still allows a later explicit "put me beside it".
    value = _RECIPIENT.sub(lambda match: re.sub(r'\bme\b', '  ', match.group(), flags=re.I), text)
    if any(person_reference_requested(value, word) for word in _SELF):
        return True
    if any(person_reference_requested(value, match['word']) for match in _POSSESSIVE.finditer(value)):
        return True
    # Russian dative "мне" is ordinarily a recipient; explicit body edits
    # select the speaker, unlike "нарисуй мне кота".
    if re.search(r'\b(?:надень|добавь|поставь|убери)\s+мне\b|\bмне\s+на\s+голов\w*', value, re.I):
        return person_reference_requested(value, 'мне')
    return False


def select_image_subjects(prompt, explicit_references, registered_names, speaker_name='', *,
                          source='camera', faces_in_frame=None):
    """Return requested/reference/visible/ambiguous canonical person lists.

    ``faces_in_frame`` must come from the exact source image; recent presence or
    track labels are not substitutes. Explicit tool arguments cannot authorize
    an unnamed/excluded profile. More than two results are returned unchanged
    so the caller can ask for a choice instead of silently picking identities.
    """
    result = dict(requested_people=[], reference_people=[], visible_people=[], ambiguous_people=[])
    if not isinstance(prompt, str) or not prompt.strip() or action_revoked(prompt):
        return result
    canonical = _canonical_names(registered_names)
    names = list(canonical.values())
    selected = []
    positive = {}
    for name in names:
        text = _person_text(prompt, name, names)
        positive[name.casefold()] = person_reference_requested(text, name)
    speaker = canonical.get(' '.join(str(speaker_name or '').split()).casefold())
    self_requested = _self_requested(prompt)
    if speaker and self_requested:
        # An explicit exclusion of the same named person takes precedence over
        # an earlier or contradictory pronoun. Never upload an excluded face.
        speaker_text = _person_text(prompt, speaker, names)
        named = re.search(r'(?<!\w)' + re.escape(speaker) + r'(?!\w)', speaker_text, re.I)
        if not named or positive[speaker.casefold()]:
            positive[speaker.casefold()] = True

    def add(value):
        if not isinstance(value, str):
            return
        key = ' '.join(value.split()).casefold()
        if key in {'me', 'myself', 'меня', 'мне'}:
            key = speaker.casefold() if speaker and self_requested else ''
        name = canonical.get(key)
        if name and positive.get(key) and name not in selected:
            selected.append(name)

    if isinstance(explicit_references, (list, tuple)):
        for name in explicit_references:
            add(name)
    # Source wording is authoritative even when the model forgot a tool arg.
    for name in names:
        if positive[name.casefold()]:
            add(name)
    result['requested_people'] = selected
    if source != 'camera':
        result['reference_people'] = list(selected)
        return result
    counts = {}
    for face in faces_in_frame or []:
        if isinstance(face, dict) and isinstance(face.get('name'), str):
            key = face['name'].casefold()
            counts[key] = counts.get(key, 0) + 1
    for name in selected:
        count = counts.get(name.casefold(), 0)
        result['visible_people' if count == 1 else 'ambiguous_people' if count > 1 else 'reference_people'].append(name)
    return result
