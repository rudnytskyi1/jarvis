"""Bounded ASR recovery after a room client has already confirmed its wake.

This is a second-stage server check, never a local wake detector. The two
substitutions below were observed in real Whisper transcripts; they must not
be added to the client's accepted wake phrases.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from common.voice_commands import ROWAN_AI_PHRASES, has_wake_prefix, normalize

log = logging.getLogger(__name__)
_AI_PHRASES = frozenset(normalize(phrase) for phrase in ROWAN_AI_PHRASES)
_LEGACY_NAMES = frozenset({'rowan', 'roan', 'rowen'})
# Require direct address: a name mentioned later in a sentence, in quotes,
# after a negation, or as reported speech cannot gain recovery. Keep this
# narrower than the existing strict configured-phrase check.
_ADDRESS_START = (
    r'^\s*(?:(?:hey|hi|hello|okay|ok|well|um|uh|please)\b[\s,;:!?.-]+){0,5}'
)


def _recovery_pattern(phrases: Iterable[str]) -> re.Pattern[str] | None:
    """Keep the server gate and speaker selection on the same ASR aliases."""
    configured = {normalize(phrase) for phrase in phrases if isinstance(phrase, str)}
    aliases = []
    if configured & _AI_PHRASES:
        aliases.append(r'(?:roman|ruin)\s+(?:ai|a[.\s]+i)')
    if configured & _LEGACY_NAMES:
        aliases.append(r'roman')
    if not aliases:
        return None
    return re.compile(_ADDRESS_START + r'(?P<alias>' + '|'.join(aliases) + r')\b', re.I)


def server_wake_pattern(phrases: Iterable[str]) -> re.Pattern[str] | None:
    """Match an addressed speaker turn, including the same bounded recovery.

    Unlike the final STT gate, diarization requires the address at the start
    of a turn. Keep configured phrases and recovered aliases anchored so a
    bystander's later mention cannot select their words as the request.
    """
    phrases = tuple(phrases)
    aliases = [re.escape(phrase.strip()) for phrase in phrases
               if isinstance(phrase, str) and phrase.strip()]
    patterns = []
    if aliases:
        # Allow address punctuation, but never consume a quote delimiter.
        patterns.append(r'^[\s,;:!?.-]*(?:(?:hey|okay|ok|эй)[\s,;:!?.-]+)?(?:'
                        + '|'.join(aliases) + r')\b')
    recovered = _recovery_pattern(phrases)
    if recovered is not None:
        patterns.append(recovered.pattern)
    return re.compile('|'.join(f'(?:{pattern})' for pattern in patterns), re.I) if patterns else None


def server_has_wake(text: str, phrases: Iterable[str] = ()) -> bool:
    """Confirm unchanged STT text only after ``utterance_start.verify_wake``.

    Explicit configured phrases remain authoritative. Rowan AI requires the
    spoken AI suffix even when Whisper substitutes Roman or Ruin. Only an
    explicitly configured legacy bare Rowan permits the observed bare Roman
    substitution; an empty list does not enable any ASR recovery.
    """
    phrases = tuple(phrases)
    if has_wake_prefix(text, phrases):
        return True
    recovered = _recovery_pattern(phrases)
    match = recovered.match(str(text)) if recovered is not None else None
    if match is None:
        return False
    log.info('Recovered locally confirmed wake from ASR spelling %s', match['alias'])
    return True
