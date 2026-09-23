"""Старт хода по промежуточному транскрипту (ТЗ F-101, задача P2-41).

``LiveTranscript`` already decodes the room's speech while it is being spoken
(``transcript_partial`` frames).  Until now the hub only *displayed* that
preview: the model was asked for a reply once, after the final, punctuated
transcript arrived.  This module lets the model work on the draft instead, in
parallel with the final STT pass, and decides whether that early answer may be
spoken at all.

Two pieces:

* :class:`SpeculativeRound` - one model round that started on the draft before
  the final transcript existed.  It is *held*, never sent: nothing it produced
  reaches the room until :func:`reconcile` says the draft and the final
  transcript are the same request.
* :func:`reconcile` - the gate.  A preview is a moving hypothesis, so the
  early answer is used only when the finished transcript did not change the
  request: the draft must be a prefix of the final wording (its last word may
  still have been half-spoken), and whatever the final wording added on top
  may only be filler ("please", "пожалуйста").  A content word that arrived
  late - "turn the light off in" -> "... in the kitchen" - is exactly the case
  the early answer must not answer, because the model never saw "the kitchen".
  A divergence is a rollback: the early answer is dropped and the ordinary
  turn runs, with the cost of one model round honestly logged.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hub.streaming_reply import usable_draft

log = logging.getLogger("jarvis.server.app")

#: How long the turn waits for a speculative round that is still running when
#: the final transcript is already reconciled.  The round is paid for either
#: way, so waiting a little is cheaper than starting a second one - but only a
#: little: a round that is still cold must not eat the answer's budget.
EARLY_START_HOLD_S = 0.7

#: How many finishing words the final transcript may add to the draft.
MAX_TRAILING_TOKENS = 3

#: Words that carry no new request: the sentence may grow by these after the
#: draft without the early answer becoming an answer to another question.
NEUTRAL_TAIL = frozenset({
    "a", "an", "and", "already", "away", "for", "go", "here", "it", "me",
    "now", "ok", "okay", "please", "right", "rowan", "thanks", "thank", "that",
    "the", "then", "you",
    "а", "вот", "да", "давай", "же", "и", "ну", "пожалуйста", "сейчас", "уже",
})

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

#: Что РАЗРЕШЕНО начать по черновику, пока человек ещё говорит (владелец
#: 2026-09-23: «люди делают так, чтобы он открывал приложения уже во время
#: разговора»). Только «открыть/показать», и только эти команды: такое действие
#: обратимо (лишняя вкладка, лишнее окно) и не двигает ничего в комнате.
#: Закрытие, блокировка, печать текста, деньги, звонки, свет и громкость числом
#: ждут подтверждённого транскрипта: у них смысл стоит в КОНЦЕ фразы («громче
#: на 30»), а не в начале, и ранешний запуск был бы угадыванием.
EARLY_ACTION_TOOLS: dict[str, frozenset[str]] = {
    "browser_control": frozenset({"navigate"}),
    "pc_control": frozenset({"open_app"}),
}

#: Где в аргументах лежит объект действия («что открыть»). Пока это слово —
#: последнее в черновике, оно ещё может досказываться («youtube» из «you»),
#: поэтому такой вызов ждёт следующего слова. Ранний старт срабатывает ровно
#: тогда, когда человек уже пошёл дальше по фразе.
EARLY_OBJECT_ARGS: dict[str, tuple[str, ...]] = {
    "browser_control": ("url",),
    "pc_control": ("app", "target", "value"),
}


def words(text: str) -> list[str]:
    """The comparable words of a transcript (case, punctuation and pauses out)."""
    return [word.lower() for word in _WORD.findall(str(text or ""))]


def may_run_early(draft: str, name: str, args: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    """Можно ли ВЫПОЛНИТЬ этот вызов по черновику, не дожидаясь конца фразы.

    Возвращает ``(можно, причина)``. Три условия, и все три — про то, чтобы
    раннее действие было обратимым и уже однозначным:

    1. инструмент и его команда в списке ``EARLY_ACTION_TOOLS``;
    2. объект действия назван в аргументах (открывать «что-то» нельзя);
    3. объект НЕ последнее слово черновика — значит, человек его уже
       проговорил и продолжает фразу. Если он всё-таки замолчал на объекте,
       реплика закроется сама, и обычный ход сделает то же самое.

    Причина возвращается человеческой строкой: она попадает в отказ модели и в
    лог, чтобы было видно, почему именно действие не началось раньше.
    """
    tool = str(name or "")
    commands = EARLY_ACTION_TOOLS.get(tool)
    if commands is None:
        return False, f"{tool or 'that tool'} waits for the finished sentence"
    arguments = dict(args or {})
    command = str(arguments.get("command") or "").strip().lower()
    if command and command not in commands:
        return False, f"{tool} {command} waits for the finished sentence"
    draft_words = words(draft)
    if not draft_words:
        return False, "there is no draft to act on yet"
    last = draft_words[-1]
    object_value = ""
    for key in EARLY_OBJECT_ARGS.get(tool, ()):
        candidate = str(arguments.get(key) or "").strip()
        if candidate:
            object_value = candidate
            break
    if not object_value:
        return False, f"{tool} has no object to open yet"
    spoken = words(object_value)
    # «open you…» — слово ещё договаривают: объект, который НАЧИНАЕТСЯ с
    # последнего услышанного слова, считается недоговорённым («you» → «youtube»),
    # как и в :func:`reconcile`.
    if spoken and (spoken[-1] == last or spoken[-1].startswith(last)):
        return False, "the object is still the last word being spoken"
    return True, "the object is said and the person has moved on"


@dataclass(frozen=True)
class Reconciliation:
    """Verdict of comparing an interim draft with the finished transcript."""

    matched: bool
    reason: str
    #: What the final transcript added after the draft (empty when it added
    #: nothing), so the log line can say *why* a rollback happened.
    trailing: tuple[str, ...] = ()

    def __bool__(self) -> bool:  # so ``if reconcile(...):`` reads naturally
        return self.matched


def reconcile(draft: str, final: str, *,
              max_trailing: int = MAX_TRAILING_TOKENS) -> Reconciliation:
    """May the answer produced for ``draft`` be spoken for ``final``?

    The rule is deliberately narrow.  An interim transcript is a guess, so a
    wrong early answer is worse than a slightly later right one; only a draft
    the finished transcript confirms word for word (plus filler) passes.
    """
    draft_words, final_words = words(draft), words(final)
    if not draft_words:
        return Reconciliation(False, "the draft is empty")
    if not final_words:
        return Reconciliation(False, "the final transcript is empty")
    if len(final_words) < len(draft_words):
        return Reconciliation(False, "the final transcript lost words the draft had")
    for index, draft_word in enumerate(draft_words):
        final_word = final_words[index]
        if draft_word == final_word:
            continue
        # The last word of a preview is the one still being spoken: "kitchen"
        # arrives as "kit".  A prefix there is confirmation, not divergence.
        if index == len(draft_words) - 1 and final_word.startswith(draft_word):
            continue
        return Reconciliation(False, f"the final transcript says {final_word!r} "
                                      f"where the draft had {draft_word!r}")
    trailing = tuple(final_words[len(draft_words):])
    if not trailing:
        return Reconciliation(True, "the final transcript confirms the draft")
    if len(trailing) > max_trailing:
        return Reconciliation(False, f"the final transcript continued for "
                                      f"{len(trailing)} more word(s)", trailing)
    late = [word for word in trailing if word not in NEUTRAL_TAIL]
    if late:
        return Reconciliation(False, f"the final transcript added {late[0]!r}", trailing)
    return Reconciliation(True, f"only filler followed the draft "
                                 f"({' '.join(trailing)})", trailing)


@dataclass
class SpeculativeRound:
    """One model round started on the draft, before the final transcript.

    ``task`` resolves to ``(client, result)``: the model client that answered
    (the self-check later continues on it) and its outcome.  Nothing here is
    sent to the room - ``hub/app.py`` either adopts the result after
    :func:`reconcile` accepted the draft, or calls :meth:`discard`.
    """

    draft: str
    task: asyncio.Task
    started_at: float
    label: str = "llm-reply-early"

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def done(self) -> bool:
        return self.task.done()

    def discard(self, reason: str) -> None:
        """Drop the early round: the draft and the final transcript diverged."""
        if not self.task.done():
            self.task.cancel()
        log.info("Early start rolled back (%s); the ordinary turn answers instead", reason)

    async def take(self, *, timeout: float = EARLY_START_HOLD_S) -> tuple[Any, Any] | None:
        """Wait a bounded moment for the round and return ``(client, result)``.

        ``None`` means the round did not deliver anything usable in time; the
        caller then runs the ordinary turn.  A failure inside the round is
        reported at debug level and treated the same way - an optional
        latency trick must never be the reason a room gets no answer.
        """
        try:
            outcome = await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
        except TimeoutError:
            log.info("Early answer not ready after %.2f s; the ordinary turn answers", timeout)
            self.discard("the speculative round was still running")
            return None
        except asyncio.CancelledError:
            if self.task.cancelled():
                return None
            raise
        except Exception as exc:  # noqa: BLE001 - see the docstring
            log.debug("The speculative round failed (%s)", exc)
            return None
        if not outcome:
            return None
        return outcome


def early_result_or_none(client: Any, result: Any, *,
                         allow_tool_calls: bool = False) -> tuple[Any, Any] | None:
    """Is this early result something the room may hear?

    A draft round has no tools behind it (nothing may be executed for a request
    the speaker may still be finishing), so a model that answered with a tool
    call did not answer the request at all.  Such a round is rolled back: the
    ordinary turn runs the tool and speaks its real result.

    ``allow_tool_calls`` — исключение ровно для одного случая: раунд по
    черновику сам выполнил разрешённые ранние действия (``may_run_early``) и
    ничего не отложил. Тогда его слова отвечают на уже сделанное дело, и
    подменять их нельзя — иначе комната услышит «не смог открыть» про
    вкладку, которая в этот момент открывается.
    """
    if result is None:
        return None
    if getattr(result, "tool_calls", None) and not allow_tool_calls:
        log.info("Early answer rolled back: the draft needed %d tool call(s)",
                 len(result.tool_calls))
        return None
    if not str(getattr(result, "text", "") or "").strip():
        log.info("Early answer rolled back: the draft round produced no words")
        return None
    return client, result


__all__ = [
    "EARLY_START_HOLD_S",
    "EARLY_ACTION_TOOLS",
    "EARLY_OBJECT_ARGS",
    "MAX_TRAILING_TOKENS",
    "NEUTRAL_TAIL",
    "Reconciliation",
    "SpeculativeRound",
    "early_result_or_none",
    "may_run_early",
    "reconcile",
    "usable_draft",
    "words",
]
