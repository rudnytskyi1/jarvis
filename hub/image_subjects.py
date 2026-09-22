"""Choose saved identity references from literal requests, never room memory."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from hub.image_prompt import action_revoked, person_reference_requested


def _fold_name(value: Any) -> str:
    """A name reduced to comparable letters: "John the system" -> "johnthesystem".

    Cyrillic is transliterated first, because the owner says "Антон" while the
    enrolled profile is spelled "Anton"; both have to meet in the same key.
    """
    return _name_key(value)


#: Cyrillic letters as the hub writes the same sound in the Latin alphabet.
_TRANSLIT = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
             'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'i', 'к': 'k', 'л': 'l', 'м': 'm',
             'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
             'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
             'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'}
_WORD = re.compile(r'[^\W_]+', re.UNICODE)
_CYRILLIC = re.compile('[а-яё]', re.I)
#: Shortest form that may stand for a longer enrolled name ("john" for "john the system").
_MIN_FORM = 4
#: Longest Russian ending an inflected form may add: "антона" for "anton".
_MAX_TAIL = 3


def _name_key(value: Any) -> str:
    """The comparable key of one name: transliterated letters and digits only."""
    letters = []
    for char in str(value or '').casefold():
        char = _TRANSLIT.get(char, char)
        if char.isalnum():
            letters.append(char)
    return ''.join(letters)


def _name_words(value: Any) -> list[str]:
    return [match.group(0) for match in _WORD.finditer(str(value or ''))]


def _form_score(word: Any, name: Any) -> int:
    """2 when ``word`` is the name, 1 when it is a short or inflected form of it."""
    word_key, name_key_value = _name_key(word), _name_key(name)
    if not word_key or not name_key_value:
        return 0
    if word_key == name_key_value:
        return 2
    if name_key_value.startswith(word_key) and len(word_key) >= _MIN_FORM:
        # The short form of a longer name: "John" for "John the system".
        return 1
    if (_CYRILLIC.search(str(word)) and word_key.startswith(name_key_value)
            and len(name_key_value) >= _MIN_FORM):
        # A Russian case ending: "антона" is the enrolled "Anton". Only Cyrillic
        # wording inflects like that, so a longer Latin word is a different name
        # ("Antonia", "AntonDorm") instead of an inflection.
        return 1 if len(word_key) - len(name_key_value) <= _MAX_TAIL else 0
    return 0


def expand_person_names(text: Any, names: Iterable[Any]) -> str:
    """``text`` with unambiguous short or inflected forms of enrolled names spelled out.

    The owner uses the name the way they know it - "Антон" and "антона" for the
    enrolled "Anton", "John" for "John the system" - and the hub answered that
    the person "was not requested in this image". Rewriting the matched form as
    the full enrolled name keeps every existing rule (negations, exclusions,
    quoted captions) working on the same sentence. A form that fits more than
    one enrolled person is left alone: guessing between two identities is worse
    than asking.
    """
    value = str(text or '')
    everyone = [str(name) for name in names or () if str(name or '').strip()]
    if not value or not everyone:
        return value
    spelled = {name.casefold() for name in everyone}
    spans: list[tuple[int, int, str]] = []
    for match in _WORD.finditer(value):
        word = match.group(0)
        if word.casefold() in spelled:
            continue  # The name itself, exactly as enrolled: nothing to rewrite.
        matches = {name for name in everyone if _form_score(word, name)}
        if len(matches) == 1:
            spans.append((match.start(), match.end(), matches.pop()))
    if not spans:
        return value
    rebuilt, cursor = [], 0
    for start, end, name in spans:
        rebuilt.append(value[cursor:start])
        rebuilt.append(name)
        cursor = end
    rebuilt.append(value[cursor:])
    return ''.join(rebuilt)


def person_named(text: Any, name: Any, names: Iterable[Any] = ()) -> bool:
    """Does the message name this person, allowing their short or inflected form?"""
    if not isinstance(text, str) or not text.strip() or not str(name or '').strip():
        return False
    if person_reference_requested(text, str(name)):
        return True
    registry = [*names, name] if names else []
    expanded = expand_person_names(text, registry)
    return expanded != text and person_reference_requested(expanded, str(name))


def resolve_named_person(asked: Any, candidates: Iterable[Any]) -> str | None:
    """Which of the message's own people the model meant by ``asked``.

    The chat model shortens names - it asked to add "John" while the message
    said "John the system" - and the hub then refused the call as "not
    requested in this image" even though the owner had named that person. When
    exactly one candidate matches (equal after punctuation is dropped, or one
    name is the beginning of the other) the call follows the message instead
    of failing. Two plausible candidates mean no guess at all.
    """
    wanted = _fold_name(asked)
    if len(wanted) < 3:
        return None
    names = [str(name) for name in candidates or () if str(name or '').strip()]
    exact = [name for name in names if _fold_name(name) == wanted]
    if len(exact) == 1:
        return exact[0]
    partial = [name for name in names if _form_score(asked, name)]
    return partial[0] if len(partial) == 1 else None

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
    # "Добавь антона" and "add John" name enrolled people "Anton" and
    # "John the system": the wording is read with those short and inflected
    # forms spelled out, so the source request authorizes the identity without
    # demanding one exact spelling from the owner.
    wished = expand_person_names(prompt, names)
    selected = []
    positive = {}
    for name in names:
        text = _person_text(wished, name, names)
        positive[name.casefold()] = person_reference_requested(text, name)
    speaker = canonical.get(' '.join(str(speaker_name or '').split()).casefold())
    self_requested = _self_requested(prompt)
    if speaker and self_requested:
        # An explicit exclusion of the same named person takes precedence over
        # an earlier or contradictory pronoun. Never upload an excluded face.
        speaker_text = _person_text(wished, speaker, names)
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
        if name is None:
            # The chat model shortens or inflects the name it passes as a tool
            # argument; resolve it against the enrolled people, never invent one.
            name = resolve_named_person(value, names)
            key = name.casefold() if name else ''
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
