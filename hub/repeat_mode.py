"""Echo mode: "repeat after me" repeats the next messages out loud.

Владелец 2026-09-23: «я просил в телеге чтобы он за мной повторял (следующие
сообщения) а он не смог». В логе живого хаба видно, что происходило: на «Repeats
after me» модель отвечала «I'm not playing echo. Give me the actual words»,
потому что режима повтора не существовало — каждая реплика обрабатывалась как
отдельный самостоятельный запрос, а просьба «повторяй за мной» не была ни
состоянием, ни инструментом.

Режим живёт здесь, вне модели: включение и выключение ловятся фразами, а пока
режим включён, текст собеседника произносится дословно и без обращения к LLM —
поэтому он работает и когда модель молчит, и когда она отвечает не то.

Два предохранителя, чтобы включённый по ошибке повтор не съел разговор:
режим закрывается сам через ``WINDOW_S`` без новых реплик и после
``MAX_MESSAGES`` повторённых сообщений.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

#: Сколько режим живёт без новых реплик и сколько сообщений он повторит.
WINDOW_S = 600.0
MAX_MESSAGES = 60

#: Реплика, которая включает повтор. Формы для en/ru/es: владелец говорит и
#: пишет на трёх языках, а перевод просьбы не должен её ломать.
_START = (
    r'\brepeat (?:it )?after me\b',
    r'\brepeat (?:what|everything|all) i (?:say|said|type|write|tell you)\b',
    r'\brepeat (?:my|the) (?:messages?|words?|sentences?|phrases?)\b',
    r'\brepeat (?:every|each) (?:message|sentence|phrase|word)\b',
    r'\bsay (?:what|everything) i (?:say|type|write)\b',
    r'\b(?:be|become|play|start|turn on) (?:my |a |the )?(?:parrot|echo)\b',
    r'\b(?:parrot|echo) mode\b',
    r'\bповторяй(?:те)? (?:за мной|за мною|вслед за мной)\b',
    r'\bповтори(?:те)? (?:за мной|за мною|вслед за мной)\b',
    r'\bповторяй(?:те)? (?:мои|все мои|все|моё|мое)? ?(?:сообщения|слова|фразы|реплики)\b',
    r'\bповторяй(?:те)? (?:всё|все|то)?,? что я (?:скажу|говорю|напишу|пишу)\b',
    r'\bповторяй(?:те)? каждое (?:моё|мое)? ?(?:сообщение|слово)\b',
    r'\bрежим повтора\b',
    r'\brep[ií]te(?:lo)? (?:despu[eé]s de m[ií]|lo que (?:digo|diga))\b',
)

#: Реплика, которая выключает повтор.
_STOP = (
    r'\bstop repeat(?:ing|s)?\b',
    r'\bstop the repeat(?:ing)?\b',
    r'\bstop (?:the )?echo\b',
    r'\bstop (?:the )?parrot\b',
    r"\bdon'?t repeat\b",
    r'\bno more repeat(?:ing|s)?\b',
    r'\b(?:cancel|end|turn off) (?:the )?repeat(?:ing| mode)?\b',
    r'\bхватит повтор(?:ять|а|ы)?\b',
    r'\bперестань повторять\b',
    r'\bпрекрати повторять\b',
    r'\bне повторяй\b',
    r'\bне надо повторять\b',
    r'\b(?:останови|выключи|отмени) повтор(?:ение)?\b',
    r'\bстоп повтор\b',
    r'\bdeja de repetir\b',
    r'\bpara de repetir\b',
)

#: Короткое «хватит/стоп» тоже закрывает повтор — но только когда он включён.
#: Владелец 2026-09-23 написал «да все хвтаит уже», и режим не выключился: в
#: списке выше требуется слово «повтор». Опознаётся и опечатка, поэтому шаблон
#: свободный: любое короткое сообщение, начинающееся с «хватит» в любой
#: раскладке букв, со «стоп», «stop», «отмена», «cancel» или «замолчи».
#: Короткое «хватит/стоп» тоже закрывает повтор — но только когда он включён.
#: Владелец 2026-09-23 написал «да все хвтаит уже», и режим не выключился: в
#: списке выше требуется слово «повтор». Поэтому перед стоп-словом разрешены
#: короткие вводные («да», «ну», «всё»), а само слово ловится свободно — вместе
#: с опечаткой: «хватит» в любой раскладке букв, «стоп», «stop», «отмена»,
#: «cancel», «замолчи».
_SOFT_STOP = re.compile(
    r'^(?:(?:да|ну|всё|все|ok|okay|ладно)[,\s]+)*'
    r'(?:хва\w+|хв[аеоиу]?\w*т\w*|стоп\w*|stop\w*|cancel|отмена|замолчи)',
    re.IGNORECASE)

#: Слова про повтор ВСЛУХ (комната) и про повтор ТЕКСТОМ (этот чат). Оба
#: намерения ищутся вместе со словом «повтор», иначе «поставь это в комнату»
#: переключало бы режим вместо того, чтобы быть повторённым. Владелец
#: 2026-09-23: «повторяй за мной в чате а не озвучивай в комнате», и через
#: минуту — «а можешь повторять озвучкой в комнату?»: пока режим включён, такая
#: просьба ДОЛЖНА переключать режим, а не повторяться.
_ROOM = (
    r'\bв комнат\w*', r'\bвслух\b', r'\bозвуч\w*', r'\bголосом\b', r'\bчерез колонк',
    r'\bпо громкой\b', r'\bin the room\b', r'\bout loud\b', r'\baloud\b',
    r'\bthrough the speakers?\b', r'\ben la habitaci[oó]n\b', r'\ben voz alta\b',
)
_CHAT = (
    r'\bв чат\w*', r'\bв телег\w*', r'\bтекстом\b', r'\bздесь же\b',
    r'\bin the chat\b', r'\btext only\b', r'\bas text\b',
)
#: Слово, по которому видно, что речь вообще про режим повтора.
_REPEAT_WORD = re.compile(r'повтор\w*|повт\w*|rep(?:ea|i|e|í)t\w*|echo\b|parrot',
                          re.IGNORECASE)
_ROOM_RE = re.compile('|'.join(_ROOM), re.IGNORECASE)
_CHAT_RE = re.compile('|'.join(_CHAT), re.IGNORECASE)

#: Что считать «просьбой», а не фразой, которую просят повторить: короткое
#: сообщение и слова-просьбы. Фраза «можешь повторять озвучкой в комнату?»
#: включает режим, а «скажи это в комнату» — просто повторяется.
_REQUEST = re.compile(r'можешь|можно|давай|включи|включай|сделай|can you|could you|'
                      r'please|start|turn on|switch|хочу|будешь', re.IGNORECASE)

_START_RE = re.compile('|'.join(_START), re.IGNORECASE)
_STOP_RE = re.compile('|'.join(_STOP), re.IGNORECASE)

#: Кавычки, скобки и знаки на краях сообщения: «хватит» и "хватит" — то же
#: слово, что и хватит (владелец отправил «хватит» в кавычках, и режим не
#: выключился).
_EDGES = "«»„“”‘’\"'()[]{}.,!?:;*#—–-…"


def _clean(text: str) -> str:
    """Одна строка без краёв-пунктуации, в нижнем регистре."""
    return ' '.join(str(text or '').split()).strip().strip(_EDGES).strip().casefold()


def request(text: str) -> str:
    """What one message asks for: ``'start'``, ``'stop'`` or ``''``.

    Выключение проверяется первым: «хватит повторять» и «stop repeating» должны
    закрывать режим, а явный стоп никогда не включает его снова.
    """
    cleaned = ' '.join(str(text or '').split()).strip()
    if not cleaned or len(cleaned) > 200:
        return ''
    lowered = _clean(cleaned)
    if _STOP_RE.search(lowered):
        return 'stop'
    if _START_RE.search(lowered):
        return 'start'
    return ''


def wants_room_speech(text: str) -> bool:
    """Просили ли повтор ВСЛУХ. Иначе повтор идёт текстом в тот же чат."""
    cleaned = _clean(text)
    return bool(_REPEAT_WORD.search(cleaned) and _ROOM_RE.search(cleaned))


def wants_chat_speech(text: str) -> bool:
    """Просили ли повтор ТЕКСТОМ (сюда же, в этот чат)."""
    cleaned = _clean(text)
    return bool(_REPEAT_WORD.search(cleaned) and _CHAT_RE.search(cleaned))


def is_repeat_request(text: str) -> bool:
    """Похоже ли сообщение на просьбу «повторяй, как я скажу» (а не на фразу).

    Только такие просьбы включают и переключают режим; всё остальное, пока режим
    включён, повторяется дословно.
    """
    cleaned = _clean(text)
    if not cleaned or len(cleaned) > 120:
        return False
    return bool(_REPEAT_WORD.search(cleaned) and
                (_ROOM_RE.search(cleaned) or _CHAT_RE.search(cleaned) or _REQUEST.search(cleaned)))


def soft_stop(text: str) -> bool:
    """Короткое «хватит»/«стоп» — стоп для включённого повтора, не для чата.

    Стопом считается только КОРОТКОЕ сообщение (до четырёх слов): владелец
    написал «хватит, можешь озвучкой в комнату повторять?» — это просьба
    переключить режим в комнату, а не выключить его, и она переключает.
    """
    cleaned = _clean(text)
    if not cleaned or len(cleaned) > 40 or len(cleaned.split()) > 4:
        return False
    return bool(_SOFT_STOP.match(cleaned))


class RepeatMode:
    """Which conversations repeat right now, and for how much longer.

    ``scope`` — одна переписка одного человека (в телеге это пара
    «чат + автор»), поэтому повтор в личке не распространяется на группу и
    наоборот.
    """

    def __init__(self, window_s: float = WINDOW_S, max_messages: int = MAX_MESSAGES,
                 *, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = float(window_s)
        self.max_messages = int(max_messages)
        self._clock = clock
        self._open: dict[str, dict[str, float]] = {}

    def start(self, scope: str, *, spoken: bool = False) -> None:
        """Начать повтор: ``spoken=False`` — текстом в чат (по умолчанию)."""
        self._open[str(scope)] = {'until': self._clock() + self.window_s, 'count': 0.0,
                                  'spoken': 1.0 if spoken else 0.0}

    def spoken(self, scope: str) -> bool:
        """Повторять ли вслух в комнате (только если владелец попросил это)."""
        entry = self._open.get(str(scope))
        return bool(entry and entry.get('spoken'))

    def set_spoken(self, scope: str, spoken: bool) -> bool:
        """Переключить «вслух/текстом» у включённого режима; ``False`` — он не включён."""
        entry = self._open.get(str(scope))
        if entry is None:
            return False
        entry['spoken'] = 1.0 if spoken else 0.0
        return True

    def stop(self, scope: str) -> bool:
        """Turn the mode off; ``True`` when it was on."""
        return self._open.pop(str(scope), None) is not None

    def active(self, scope: str) -> bool:
        """Is this conversation repeating? An expired mode closes itself."""
        key = str(scope)
        entry = self._open.get(key)
        if entry is None:
            return False
        if self._clock() >= entry['until'] or entry['count'] >= self.max_messages:
            self._open.pop(key, None)
            return False
        return True

    def note(self, scope: str) -> int:
        """Count one repeated message and keep the mode alive a while longer."""
        entry = self._open.get(str(scope))
        if entry is None:
            return 0
        entry['count'] += 1
        entry['until'] = self._clock() + self.window_s
        return int(entry['count'])

    def snapshot(self) -> dict[str, Any]:
        """The open modes, for the panel and the tests."""
        return {scope: {'count': int(entry['count']), 'until': entry['until']}
                for scope, entry in self._open.items()}
