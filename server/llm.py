"""LLM with tool calling: Ollama native or OpenAI-compatible (SPEC §3).

Two providers, selected by ``cfg.server.llm.provider``:

* ``"ollama_native"`` (default) — ``POST {base}/api/chat`` via ``httpx`` with
  ``stream: false``, ``think: cfg.llm.think`` (off by default, which disables
  Qwen3 reasoning) and ``options: {num_predict, temperature}``. Native tool calls
  already carry their ``arguments`` as an object.
* ``"openai"`` — the ``openai`` package against ``cfg.llm.base_url``; there tool
  arguments arrive as a JSON string and are parsed defensively.

Tool loop: up to ``cfg.llm.max_tool_rounds`` rounds. Each round executes the
model's tool calls in order through the :data:`ToolExecutor` callback supplied by
``server/app.py``, appends the assistant message plus one ``role: "tool"``
message per call with the REAL result, and asks for the next completion. The loop
stops at the first reply without tool calls; if the round cap is hit, one final
completion is requested with no tools at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

import httpx

from server.tools import TOOL_NAMES, TOOLS

log = logging.getLogger("jarvis.server.llm")

PROVIDER_OLLAMA_NATIVE = "ollama_native"
PROVIDER_OPENAI = "openai"

#: Network timeout for one completion (Ollama on a 30B model can be slow).
#: No SDK-level retries: a hung request is reported instead of doubled.
REQUEST_TIMEOUT_S = 180.0
MAX_RETRIES = 0

#: Result handed to the model when no tool executor is wired up.
NO_EXECUTOR_RESULT: dict[str, Any] = {
    "ok": False,
    "error": "tool execution is not available",
}

#: Executes one tool call and returns its result as a JSON-serializable dict.
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
#: Malformed inline tool-call syntax the model sometimes writes INTO its text
#: instead of a structured call ("<toolcall <function=enrollface <parameter=…").
#: It must never be spoken aloud, and a round that produced it is retried once.
_TOOL_ARTIFACT_RE = re.compile(
    r"<\s*/?\s*(?:tool_?call|function(?:=[^\s<>]*)?|parameter(?:=[^\s<>]*)?)\s*/?>?",
    re.IGNORECASE,
)
_MARKDOWN_CHARS_RE = re.compile(r"[*_`#>|]+")
_WHITESPACE_RE = re.compile(r"\s+")
#: Space left in front of punctuation after markdown was stripped ("Done , sir").
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:…])")

# -- BUG 3: the model must never SAY it saw something it did not look at ---
#: Tools that actually look at the room/screen this turn. A reply making a
#: first-person sight claim without one of these having run is fabrication —
#: see :func:`contains_sight_claim` and :meth:`LlmClient.generate`.
VISION_TOOLS: frozenset[str] = frozenset({"look_at_camera", "look_at_screen", "find_object"})

#: Phrase list for :data:`_SIGHT_CLAIM_RE` — kept as its own constant so it is
#: easy to read, test and extend independently of the compiled pattern.
SIGHT_CLAIM_PHRASES: tuple[str, ...] = (
    r"\bi\s+can\s+see\b",
    r"\bi\s+see\b",
    r"\bi(?:'m| am)\s+seeing\b",
    r"\byou(?:'re| are)\s+holding\b",
    r"\bon\s+(?:the|your)\s+screen\b",
    r"\bin\s+your\s+hand\b",
    r"\byou(?:'re| are)\s+wearing\b",
    r"\bi\s+notice\s+(?:a|an|the)\b",
    r"\blooking\s+at\s+(?:the|your)\s+(?:screen|camera|room)\b",
    r"\bthere\s+(?:is|are|'s)\s+[\w\s]{0,40}?\b"
    r"(?:bottle|person|people|cup|phone|laptop|chair|bag|backpack|book|dog|cat)\b",
)
_SIGHT_CLAIM_RE = re.compile("|".join(SIGHT_CLAIM_PHRASES), re.IGNORECASE)
#: A sight-claim phrase preceded closely by one of these is a DISCLAIMER, not a
#: claim ("I haven't seen anything on the screen yet") — see
#: :func:`contains_sight_claim`.
_NEGATION_RE = re.compile(
    r"\b(?:not|n't|cannot|can't|no|never|didn't|doesn't|isn't|aren't|haven't|hasn't)\b",
    re.IGNORECASE,
)
#: How many characters before a sight-claim match are scanned for a negation.
_NEGATION_WINDOW_CHARS = 30

#: BUG 3: the one-off correction injected as a user turn when the reply claims
#: sight without having looked this turn (SPEC: one forced retry, ever).
FORCE_LOOK_MESSAGE = (
    "[system: you described something you did not actually look at. Call "
    "look_at_camera or look_at_screen NOW, then answer only from the result.]"
)

#: A reply that ANNOUNCES a future action ("I'll open it", "let me do that now")
#: while calling no tool this round is a broken promise: the action never
#: happens. Detected by these phrases and forced to actually act, once.
FUTURE_INTENT_PHRASES: tuple[str, ...] = (
    r"\bi(?:'ll| will|'m going to| am going to| shall)\b",
    r"\blet me (?:do|open|close|search|check|find|play|start|type|click|look|run|set|turn)\b",
    r"\bi(?:'m| am) (?:going to|about to|now) \w+ing\b",
    r"\bone (?:moment|sec|second)\b",
    r"\bhold on\b",
    r"\bright away\b",
    r"\bdoing (?:that|this|it) now\b",
)
_FUTURE_INTENT_RE = re.compile("|".join(FUTURE_INTENT_PHRASES), re.IGNORECASE)
#: Injected once when the model promised an action but ran no tool this turn.
FORCE_ACT_MESSAGE = (
    "[system: you announced an action but did not call any tool, so nothing "
    "happened. Do it NOW by calling the right tool. If it truly cannot be done, "
    "say so plainly instead of promising.]"
)

#: The self-check ("judge") turn, appended after an action reply. The model
#: re-examines whether it actually completed the request and its own promises,
#: finishes anything missing with tools, then gives the final spoken reply.
VERIFY_MESSAGE = (
    "[system self-check: before this is spoken to the user, verify you ACTUALLY "
    "did everything they asked and everything you said you would. Look at the "
    "real tool results above, not your intentions. If anything the user "
    "requested or you promised did NOT happen, do it NOW by calling the right "
    "tools. When everything is truly done, reply with the final one or two "
    "spoken sentences for the user - do not mention this self-check.]"
)


def announces_undone_action(text: str) -> bool:
    """True when ``text`` promises a future action (see FUTURE_INTENT_PHRASES)."""
    if not text:
        return False
    return _FUTURE_INTENT_RE.search(text) is not None


#: A reply that REPORTS an action in the past ("the photo has been hidden",
#: "I've closed it", "volume is now at thirty") while no tool ran at all is a
#: lie of the same family as the broken promise - the thing never happened.
#: The state words a reply uses to assert the world already changed. Kept as
#: one list so the frames below all cover the same family at once.
_STATE_WORDS = (
    r"closed|hidden|open|opened|gone|removed|deleted|off|on|up|down|gray|grey|"
    r"gone\s+from\s+the\s+screen|gone\s+now|gone\s+off|"
    r"muted|unmuted|paused|playing|stopped|started|running|"
    r"minimi[sz]ed|maximi[sz]ed|full\s*screen|visible|displayed|showing|gone\s+away"
)
DONE_CLAIM_PHRASES: tuple[str, ...] = (
    r"\b(?:has|have)\s+been\s+\w+(?:ed|en|ut|one)\b",
    r"\bi(?:'ve| have)\s+(?:closed|hidden|opened|set|muted|unmuted|started|stopped|"
    r"typed|clicked|removed|deleted|saved|changed|paused|played)\b",
    # The COPULA FRAME: "is/are" + an optional polarity adverb + a state word.
    # Matching the frame instead of one spelling of the verb is deliberate. A
    # phrase list is an arms race the model keeps winning: after "the photo has
    # been hidden" was caught it answered "the photo is no longer displayed on
    # the screen", which meant exactly the same thing and slipped straight
    # through. "no longer displayed", "is now closed" and "is gone" are one
    # frame, and this catches all of them.
    r"\b(?:is|are|it'?s|they'?re)\s+"
    r"(?:now|already|no\s+longer|not|n't|currently|back|all)?\s*"
    r"(?:" + _STATE_WORDS + r")\b",
    r"\b(?:closed|hidden|removed|opened|minimi[sz]ed|maximi[sz]ed)\s+(?:it|that|the)\b",
    r"\bi\s+(?:just\s+)?(?:closed|hid|opened|set|muted|clicked|typed|removed)\b",
    r"^\s*(?:done|okay,? done|all set)\b",
)
_DONE_CLAIM_RE = re.compile("|".join(DONE_CLAIM_PHRASES), re.IGNORECASE)

#: Injected once when the reply reports a completed action but nothing ran.
FORCE_DONE_MESSAGE = (
    "[system: you told the user something was already done, but you called no "
    "tool this turn, so it did NOT happen. Do it NOW with the right tool. If it "
    "cannot be done, say that plainly instead.]"
)


def claims_completed_action(text: str) -> bool:
    """True when ``text`` reports an action as already completed."""
    if not text:
        return False
    return _DONE_CLAIM_RE.search(text) is not None


#: Verbs that ask for a CHANGE, in imperative position. Every other guard in
#: this module reads the ASSISTANT's wording, and that side of the conversation
#: re-words itself the moment a phrase list catches it - which is how "the photo
#: has been hidden" became "the photo is no longer displayed" and slipped
#: through. The USER's side is the stationary one: he says "close the photo"
#: however Rowan answers. Matching there turns an arms race into a fixed target.
#: Perception verbs (look, find, see) are left out deliberately - a turn that
#: was only asked to LOOK is already fulfilled by the vision tools.
COMMAND_VERBS: str = (
    r"open|close|shut|hide|show|display|dismiss|put|take|remove|delete|clear|"
    r"turn|switch|set|make|move|scroll|go|play|pause|resume|stop|start|launch|"
    r"run|execute|restart|reboot|lock|unlock|enable|disable|"
    r"mute|unmute|click|press|push|type|write|send|skip|next|previous|"
    r"rewind|minimi[sz]e|maximi[sz]e|dim|brighten|raise|lower|increase|decrease|"
    r"remember|forget|rename|bring|drop|kill|quit|exit"
)
#: Wake word, politeness and "can you" wrappers that sit in front of the verb.
_REQUEST_PREFIX: str = (
    r"(?:\b(?:ok(?:ay)?|hey|hi|yo|so|now|well|please|rowan|jarvis|"
    r"can\s+you|could\s+you|would\s+you|will\s+you|i\s+want\s+you\s+to|"
    r"i\s+need\s+you\s+to|you\s+can|let'?s|lets|just|go\s+ahead\s+and|"
    r"пожалуйста|слушай|эй|ну|давай|а)\b[\s,]*)*"
)
_IMPERATIVE_EN_RE = re.compile(
    r"^[\s,.!?\-]*" + _REQUEST_PREFIX + r"(?:" + COMMAND_VERBS + r")\b",
    re.IGNORECASE,
)
#: Russian needs no verb list: the imperative is morphological (закрой, выключи,
#: покажи, убери, поставь, нажми) - a suffix rule rather than a lexicon.
_IMPERATIVE_RU_RE = re.compile(
    r"^[\s,.!?\-]*" + _REQUEST_PREFIX + r"[а-яё]{2,}"
    r"(?:ай|яй|ей|ой|уй|ди|ни|ти|чи|жи|ши|ри|ли|ми|си|зи|би|ви|пи|ки|ги|хи|"
    r"ь|ьте|ите|йте|айте)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;]+|\band\s+|\bthen\s+|\bи\s+", re.IGNORECASE)


def is_imperative_request(text: str | None) -> bool:
    """True when the user ORDERED a state change, in any clause of ``text``.

    Deliberately generous: a false positive costs one extra self-check round,
    a false negative costs a silent lie to the owner's face. Pure string
    matching - never raises, safe on ``None``.
    """
    if not text:
        return False
    for clause in _CLAUSE_SPLIT_RE.split(str(text)):
        clause = clause.strip()
        if clause and (
            _IMPERATIVE_EN_RE.match(clause) or _IMPERATIVE_RU_RE.match(clause)
        ):
            return True
    return False


def contains_sight_claim(text: str) -> bool:
    """True when ``text`` makes a first-person sight claim (BUG 3).

    Matches :data:`SIGHT_CLAIM_RE`'s phrases, but skips a match that is really
    a disclaimer: when one of :data:`_NEGATION_RE`'s words appears in the
    :data:`_NEGATION_WINDOW_CHARS` characters right before the match, e.g. "I
    haven't seen anything on the screen" must NOT trigger the guard the way "I
    can see on the screen" does. Pure string matching — never raises, and safe
    to call on an empty or ``None`` string.
    """
    if not text:
        return False
    for match in _SIGHT_CLAIM_RE.finditer(text):
        window_start = max(0, match.start() - _NEGATION_WINDOW_CHARS)
        if _NEGATION_RE.search(text[window_start : match.start()]):
            continue
        return True
    return False


@dataclass
class ToolCall:
    """Normalized tool call from the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = "{}"


@dataclass
class LlmResult:
    """Outcome of one utterance: the spoken text plus every tool call executed."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    rounds: int = 0
    #: The full message history the loop ended with (system + turns + every
    #: tool call and result). Fed to :meth:`LlmClient.verify` so the self-check
    #: continues from the real state instead of redoing work.
    history: list[dict[str, Any]] = field(default_factory=list)


def native_base_url(base_url: str) -> str:
    """Ollama's native API root: the configured base URL without a trailing ``/v1``."""
    base = str(base_url or "").strip().rstrip("/")
    if base.lower().endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return base or "http://127.0.0.1:11434"


def _strip_thinking(text: str) -> str:
    """Drop ``<think>`` blocks a reasoning model may leak into its content."""
    cleaned = _THINK_BLOCK_RE.sub(" ", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned


def clean_reply(text: str | None) -> str:
    """Strip reasoning and markdown noise — the reply is spoken aloud, not rendered."""
    if not text:
        return ""
    # Delete the markers instead of replacing them: "**Done**, sir" must not
    # become "Done , sir" — the text is spoken and stored in the history.
    cleaned = _strip_thinking(str(text))
    cleaned = _TOOL_ARTIFACT_RE.sub("", cleaned)
    cleaned = _MARKDOWN_CHARS_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip()


def _arguments_to_dict(raw: Any) -> tuple[dict[str, Any] | None, str]:
    """Return ``(args_dict, raw_json_string)`` for a tool call's arguments.

    ``args_dict`` is ``None`` when the arguments could not be parsed — such a call
    must be dropped, not executed with empty arguments.
    """
    if isinstance(raw, dict):
        try:
            return dict(raw), json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            return dict(raw), "{}"
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, "{}"
    if not isinstance(raw, str):
        log.warning("Unknown tool argument type: %r", type(raw).__name__)
        return None, "{}"
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        log.warning("The model sent unparsable tool arguments: %r", raw)
        return None, raw
    if isinstance(parsed, dict):
        return parsed, raw
    log.warning("Tool arguments are not an object: %r", raw)
    return None, raw


def normalize_tool_calls(tool_calls: Iterable[Any] | None) -> list[ToolCall]:
    """Convert SDK tool-call objects (or native dicts) into :class:`ToolCall` records."""
    result: list[ToolCall] = []
    if not tool_calls:
        return result

    for index, call in enumerate(tool_calls, start=1):
        if isinstance(call, ToolCall):
            result.append(call)
            continue
        if isinstance(call, dict):
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name") or call.get("name")
            raw_args = function.get("arguments") if "arguments" in function else call.get("arguments")
            call_id = call.get("id")
        else:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None) if function is not None else getattr(call, "name", None)
            raw_args = (
                getattr(function, "arguments", None)
                if function is not None
                else getattr(call, "arguments", None)
            )
            call_id = getattr(call, "id", None)

        if not isinstance(name, str) or not name.strip():
            log.warning("Skipping a tool call without a name: %r", call)
            continue
        args, raw = _arguments_to_dict(raw_args)
        if args is None:
            log.warning("Skipping call to %s: could not parse arguments %r", name.strip(), raw)
            continue
        result.append(
            ToolCall(
                id=str(call_id) if call_id else f"call_{index}",
                name=name.strip(),
                arguments=args,
                raw_arguments=raw,
            )
        )
    return result


def looks_like_unfinished_reasoning(raw_content: str, truncated: bool) -> bool:
    """True when the content is a reasoning monologue that never reached an answer.

    With ``think: false`` Ollama stops parsing reasoning, so a Qwen3 model that
    reasons anyway writes it into ``content`` and closes it with ``</think>``
    before the real reply (:func:`clean_reply` keeps only that tail). If the
    token budget runs out first there is no closing tag and no answer at all —
    that text must never be read aloud.
    """
    return bool(truncated and raw_content.strip() and "</think>" not in raw_content)


def _result_to_content(result: Any) -> str:
    """Serialize a tool result for the ``role: "tool"`` message."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps({"ok": False, "error": "unserializable tool result"})


class LlmClient:
    """Chat client with tool calling for both supported providers.

    :meth:`generate` is a coroutine: the HTTP request itself runs in a worker
    thread, and tool calls are awaited on the event loop through the executor.
    """

    def __init__(self, cfg_llm: Any) -> None:
        self.provider = str(getattr(cfg_llm, "provider", PROVIDER_OLLAMA_NATIVE) or PROVIDER_OLLAMA_NATIVE).strip().lower()
        if self.provider not in {PROVIDER_OLLAMA_NATIVE, PROVIDER_OPENAI}:
            log.warning(
                "Unknown llm.provider=%r — falling back to %s",
                self.provider,
                PROVIDER_OLLAMA_NATIVE,
            )
            self.provider = PROVIDER_OLLAMA_NATIVE
        self.model = str(cfg_llm.model)
        self.base_url = str(cfg_llm.base_url)
        self.native_url = native_base_url(self.base_url)
        self.think = bool(getattr(cfg_llm, "think", False))
        try:
            self.temperature = float(cfg_llm.temperature)
        except (TypeError, ValueError):
            self.temperature = 0.6
        try:
            self.max_tokens = int(cfg_llm.max_tokens)
        except (TypeError, ValueError):
            self.max_tokens = 1024
        try:
            self.max_tool_rounds = max(1, int(getattr(cfg_llm, "max_tool_rounds", 4)))
        except (TypeError, ValueError):
            self.max_tool_rounds = 4
        self.api_key = str(getattr(cfg_llm, "api_key", "") or "ollama")
        #: How long Ollama keeps the chat model loaded after a request. The
        #: default 5 m would make the first command after a quiet spell pay a
        #: full ~15 s model reload (native provider only).
        self.keep_alive = str(getattr(cfg_llm, "keep_alive", "4h") or "4h")
        try:
            self.num_ctx = max(1024, int(getattr(cfg_llm, "num_ctx", 8192)))
        except (TypeError, ValueError):
            self.num_ctx = 8192

        #: Cleared when the server rejects the "think" field (older Ollama builds).
        self._send_think = True
        #: With reasoning parsing off the model may write its reasoning into the
        #: content; with think=true Ollama keeps it in a separate field instead.
        self._reasoning_may_leak = (self.provider == PROVIDER_OPENAI) or not self.think
        self._http: httpx.Client | None = None
        self._client: Any = None

        if self.provider == PROVIDER_OLLAMA_NATIVE:
            self._http = httpx.Client(timeout=REQUEST_TIMEOUT_S)
            log.info(
                "LLM: %s via %s/api/chat (native, think=%s, max_tool_rounds=%d)",
                self.model,
                self.native_url,
                self.think,
                self.max_tool_rounds,
            )
        else:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=REQUEST_TIMEOUT_S,
                max_retries=MAX_RETRIES,
            )
            log.info(
                "LLM: %s via %s (openai-compatible, max_tool_rounds=%d)",
                self.model,
                self.base_url,
                self.max_tool_rounds,
            )

    # ------------------------------------------------------------ provider calls

    def _chat_native(
        self, messages: list[dict[str, Any]], with_tools: bool
    ) -> tuple[str, list[ToolCall]]:
        """Blocking ``POST /api/chat`` against Ollama's own API."""
        http = self._http
        if http is None:
            raise RuntimeError("the native LLM client is closed")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if self._send_think:
            payload["think"] = self.think
        if with_tools:
            payload["tools"] = TOOLS

        url = f"{self.native_url}/api/chat"
        try:
            response = http.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            if self._send_think and "think" in body.lower():
                # Older Ollama builds reject the field for non-reasoning models.
                log.warning("Ollama rejected the 'think' field — retrying without it")
                self._send_think = False
                payload.pop("think", None)
                response = http.post(url, json=payload)
                response.raise_for_status()
            else:
                log.error("Ollama returned HTTP %s: %s", exc.response.status_code, body[:300])
                raise

        data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            log.warning("Ollama returned no message object")
            return "", []
        raw_content = str(message.get("content") or "")
        text = clean_reply(raw_content)
        truncated = str(data.get("done_reason") or "") == "length"
        if text and self._reasoning_may_leak and looks_like_unfinished_reasoning(raw_content, truncated):
            log.warning(
                "The model spent the whole %d-token budget on reasoning — dropping it. "
                "Raise server.llm.max_tokens or turn server.llm.think on.",
                self.max_tokens,
            )
            text = ""
        calls = normalize_tool_calls(message.get("tool_calls"))
        return text, calls

    def _chat_openai(
        self, messages: list[dict[str, Any]], with_tools: bool
    ) -> tuple[str, list[ToolCall]]:
        """Blocking completion through the OpenAI-compatible endpoint."""
        if self._client is None:
            raise RuntimeError("the OpenAI-compatible LLM client is closed")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if with_tools:
            kwargs["tools"] = TOOLS
        completion = self._client.chat.completions.create(**kwargs)
        choices = getattr(completion, "choices", None) or []
        if not choices:
            log.warning("The LLM returned no choices")
            return "", []
        message = choices[0].message
        raw_content = str(getattr(message, "content", None) or "")
        text = clean_reply(raw_content)
        truncated = str(getattr(choices[0], "finish_reason", "") or "") == "length"
        if text and self._reasoning_may_leak and looks_like_unfinished_reasoning(raw_content, truncated):
            log.warning(
                "The model spent the whole %d-token budget on reasoning — dropping it. "
                "Raise server.llm.max_tokens.",
                self.max_tokens,
            )
            text = ""
        calls = normalize_tool_calls(getattr(message, "tool_calls", None))
        return text, calls

    async def _chat(
        self, messages: list[dict[str, Any]], with_tools: bool
    ) -> tuple[str, list[ToolCall]]:
        """One completion, executed in a worker thread."""
        if self.provider == PROVIDER_OLLAMA_NATIVE:
            return await asyncio.to_thread(self._chat_native, messages, with_tools)
        return await asyncio.to_thread(self._chat_openai, messages, with_tools)

    # ---------------------------------------------------------- message building

    def _assistant_message(self, text: str, calls: list[ToolCall]) -> dict[str, Any]:
        """Rebuild the assistant turn (with its tool calls) for the next round."""
        if self.provider == PROVIDER_OLLAMA_NATIVE:
            tool_calls = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in calls
            ]
        else:
            tool_calls = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in calls
            ]
        return {"role": "assistant", "content": text or "", "tool_calls": tool_calls}

    def _tool_message(self, call: ToolCall, result: Any) -> dict[str, Any]:
        """One ``role: "tool"`` message carrying the real execution result."""
        message: dict[str, Any] = {
            "role": "tool",
            "name": call.name,
            "content": _result_to_content(result),
        }
        if self.provider == PROVIDER_OLLAMA_NATIVE:
            message["tool_name"] = call.name
        else:
            message["tool_call_id"] = call.id
        return message

    # ------------------------------------------------------------------ tool loop

    async def _run_tool(self, executor: ToolExecutor | None, call: ToolCall) -> dict[str, Any]:
        """Execute one tool call, converting any failure into a result dict."""
        if call.name not in TOOL_NAMES:
            log.warning("The model called an unknown tool %r", call.name)
            return {"ok": False, "error": f"unknown tool: {call.name}"}
        if executor is None:
            return dict(NO_EXECUTOR_RESULT)
        try:
            result = await executor(call.name, dict(call.arguments))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Tool %s failed", call.name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, dict):
            return result
        return {"ok": True, "result": result}

    async def verify(
        self,
        history: list[dict[str, Any]],
        reply: str,
        executor: ToolExecutor | None = None,
    ) -> LlmResult:
        """Self-check pass: did the model do everything asked/promised?

        Continues from the finished conversation ``history`` (which already
        holds every tool call and result of the turn), appends the reply and a
        verifier instruction, and runs the tool loop once more. If work was
        missing, the model finishes it here; otherwise it just confirms. The
        returned text replaces the spoken reply.
        """
        continued = list(history)
        continued.append({"role": "assistant", "content": reply})
        continued.append({"role": "user", "content": VERIFY_MESSAGE})
        return await self.generate(continued, executor)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        executor: ToolExecutor | None = None,
    ) -> LlmResult:
        """Run one utterance through the tool loop and return the spoken reply."""
        history: list[dict[str, Any]] = list(messages)
        executed: list[ToolCall] = []

        artifact_retried = False
        # BUG 3: caps the forced "you didn't actually look" retry to exactly
        # ONE per utterance (per call to generate()), so a model that keeps
        # fabricating sight claims after being corrected still gets a final
        # answer instead of looping forever.
        forced_look_retried = False
        # Same idea for a promised-but-undone action ("I'll open it"): force it
        # to act once instead of only talking about it.
        forced_act_retried = False
        for round_index in range(1, self.max_tool_rounds + 1):
            text, calls = await self._chat(history, with_tools=True)
            if (
                not calls
                and text
                and _TOOL_ARTIFACT_RE.search(text)
                and not artifact_retried
            ):
                # The model wrote a broken tool call into its text instead of a
                # structured one - the tool never ran and the garbage would be
                # spoken. One re-ask usually produces a proper call.
                artifact_retried = True
                log.warning("Malformed inline tool call in the reply - retrying the round")
                text, calls = await self._chat(history, with_tools=True)
            log.info(
                "LLM round %d/%d: text %r, %d tool call(s) (%s)",
                round_index,
                self.max_tool_rounds,
                text,
                len(calls),
                ", ".join(call.name for call in calls) or "-",
            )
            if not calls:
                # BUG 3: the reply is about to be spoken as final — this is
                # where result.text gets finalized, so it is the last chance to
                # catch "I see a bottle on your desk" when no vision tool ran
                # this turn. round_index < max_tool_rounds guarantees there is
                # a round left for the forced retry to use.
                if (
                    not forced_look_retried
                    and round_index < self.max_tool_rounds
                    and contains_sight_claim(text)
                    and not any(call.name in VISION_TOOLS for call in executed)
                ):
                    forced_look_retried = True
                    log.warning(
                        "Reply claims to have seen something without a vision "
                        "tool call this turn - forcing one retry: %r",
                        text,
                    )
                    history.append({"role": "assistant", "content": text})
                    history.append({"role": "user", "content": FORCE_LOOK_MESSAGE})
                    continue
                # A promised action ("I'll do that now") or one REPORTED as done
                # ("the photo has been hidden") with NO tool executed this whole
                # turn never happened: force one real attempt.
                if (
                    not forced_act_retried
                    and round_index < self.max_tool_rounds
                    and not executed
                    and (announces_undone_action(text) or claims_completed_action(text))
                ):
                    forced_act_retried = True
                    promised = announces_undone_action(text)
                    log.warning(
                        "Reply %s an action but ran no tool - forcing one retry: %r",
                        "promises" if promised else "claims",
                        text,
                    )
                    history.append({"role": "assistant", "content": text})
                    history.append(
                        {
                            "role": "user",
                            "content": FORCE_ACT_MESSAGE if promised else FORCE_DONE_MESSAGE,
                        }
                    )
                    continue
                history.append({"role": "assistant", "content": text})
                return LlmResult(
                    text=text, tool_calls=executed, rounds=round_index, history=history
                )

            history.append(self._assistant_message(text, calls))
            for call in calls:
                result = await self._run_tool(executor, call)
                log.info("Tool %s%s -> %s", call.name, call.arguments, result)
                history.append(self._tool_message(call, result))
                executed.append(call)

        # Round cap hit: ask for the spoken reply with no tools available.
        log.info("Tool round cap (%d) reached — asking for a final reply", self.max_tool_rounds)
        text, _ = await self._chat(history, with_tools=False)
        log.info("LLM final reply: %r", text)
        history.append({"role": "assistant", "content": text})
        return LlmResult(
            text=text, tool_calls=executed, rounds=self.max_tool_rounds, history=history
        )

    def close(self) -> None:
        """Release the HTTP resources of whichever provider is in use."""
        for target in (self._http, self._client):
            closer = getattr(target, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    log.debug("Could not close the LLM HTTP client", exc_info=True)
        self._http = None
        self._client = None


__all__ = [
    "LlmClient",
    "LlmResult",
    "ToolCall",
    "ToolExecutor",
    "clean_reply",
    "looks_like_unfinished_reasoning",
    "native_base_url",
    "normalize_tool_calls",
    "contains_sight_claim",
    "SIGHT_CLAIM_PHRASES",
    "VISION_TOOLS",
    "FORCE_LOOK_MESSAGE",
    "PROVIDER_OLLAMA_NATIVE",
    "PROVIDER_OPENAI",
    "REQUEST_TIMEOUT_S",
]
