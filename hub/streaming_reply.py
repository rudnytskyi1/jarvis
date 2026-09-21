"""Streaming answers (ТЗ F-101).

Two things make the first sound of a reply late, and this module owns both:

* the synthesis started only after the *whole* reply was written, so a room
  heard nothing while the model was still writing its second sentence, and
* nobody measured «the first sound must arrive within 1.2 s of the end of
  speech» (ТЗ 15.1), so the budget could not be kept or reported.

``reply_groups`` puts the first sentence in a group of its own: the hub sends
it to the synthesizer while the rest of the reply is still being grouped, and
the client starts speaking it. ``FirstAudioBudget`` is the instrument of the
acceptance criterion - it measures the delay the room actually experienced and
says whether it stayed inside the budget.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

#: ТЗ F-101: the first sound of the answer, after the end of speech.
DEFAULT_FIRST_AUDIO_BUDGET_S = 1.2
#: The first group is one sentence; a monster sentence is cut at a clause.
FIRST_GROUP_MAX_CHARS = 140
#: Below this a comma is not a clause boundary worth starting speech on.
CLAUSE_MIN_CHARS = 40
#: A draft transcript shorter than this is not worth a model round.
DEFAULT_DRAFT_MIN_CHARS = 12
DEFAULT_DRAFT_MIN_WORDS = 3

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
#: What a draft may not end with: a half-spoken word is worse than waiting.
_WORDISH = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")


def first_group(text: str, *, max_chars: int = FIRST_GROUP_MAX_CHARS) -> tuple[str, str]:
    """Split off the group the synthesizer should receive first.

    The first sentence is the natural unit: it is a complete thought, so
    speaking it while the rest is being grouped never cuts a word in half. A
    sentence longer than ``max_chars`` is cut at the first clause boundary past
    :data:`CLAUSE_MIN_CHARS`, which is still a place a speaker may pause.
    """
    text = " ".join(str(text or "").split())
    if not text:
        return "", ""
    head = _SENTENCE_END.split(text, maxsplit=1)
    first, rest = head[0], (head[1] if len(head) > 1 else "")
    if len(first) > max_chars:
        clauses = _CLAUSE_END.split(first)
        taken = ""
        while clauses and len(taken) < max_chars:
            candidate = (taken + " " + clauses[0]).strip()
            if taken and len(candidate) > max_chars and len(taken) >= CLAUSE_MIN_CHARS:
                break
            taken = candidate
            clauses.pop(0)
        if clauses:
            rest = (" ".join(clauses) + " " + rest).strip()
            first = taken
        if len(first) > max_chars:
            # No clause boundary helped: cut at the last space instead of
            # sending a group the synthesizer would read as one long breath.
            cut = first.rfind(" ", 0, max_chars)
            if cut > 0:
                rest = (first[cut:] + " " + rest).strip()
                first = first[:cut].strip()
    return first.strip(), rest.strip()


def reply_groups(text: str, *, max_chars: int = 220) -> list[str]:
    """The reply as synthesis groups, the first sentence first (ТЗ F-101).

    ``hub.tts.split_text`` keeps a short reply as ONE group, which is exactly
    what makes a room wait for the whole text before the first sound. Here the
    first sentence is always sent alone; only the remainder is grouped.
    """
    from hub.tts import split_text

    first, rest = first_group(text)
    if not first:
        return []
    groups = [first]
    if rest:
        groups.extend(split_text(rest, max_chars=max_chars))
    return [group for group in groups if group]


def usable_draft(text: str, *, min_chars: int = DEFAULT_DRAFT_MIN_CHARS,
                 min_words: int = DEFAULT_DRAFT_MIN_WORDS) -> bool:
    """Could this interim transcript carry a model round on its own?

    A draft is only usable when it is long enough, has enough words, and does
    not end on punctuation: a trailing comma (or a bare wake word) means the
    speaker is in the middle of a sentence, and acting on half a sentence is
    worse than answering a little later. Whether the last word itself is
    complete is not something a text gate can know - that judgement belongs to
    the reconciliation of P2-41.
    """
    text = " ".join(str(text or "").split())
    if len(text) < min_chars or len(text.split()) < min_words:
        return False
    return bool(_WORDISH.search(text[-1]))


@dataclass
class FirstAudioBudget:
    """The delay between the end of speech and the first sound we send.

    ``mark_first_audio`` is called by the TTS stream the moment the first audio
    frame is on the wire, which is what the room actually experiences; the
    budget itself is the ТЗ 15.1 value (1.2 s by default, configurable through
    ``server.streaming_reply.first_audio_budget_ms``).
    """

    budget_s: float = DEFAULT_FIRST_AUDIO_BUDGET_S
    speech_end_at: float = field(default_factory=time.monotonic)
    first_audio_at: float | None = None

    def mark_first_audio(self, *, now: float | None = None) -> float:
        """Record the first audio frame; the first call wins."""
        if self.first_audio_at is None:
            self.first_audio_at = time.monotonic() if now is None else now
        return self.delay_s()

    def delay_s(self, *, now: float | None = None) -> float:
        """Seconds from the end of speech to the first sound (``0`` before)."""
        end = self.first_audio_at
        if end is None:
            return 0.0
        return max(0.0, end - self.speech_end_at)

    def met(self, *, now: float | None = None) -> bool:
        return self.first_audio_at is not None and self.delay_s() <= self.budget_s

    def delay_ms(self) -> int:
        return int(round(self.delay_s() * 1000.0))
