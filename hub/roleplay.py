"""Explicit, temporary parody state; never inferred from archived conversations."""

import re
import time
from dataclasses import dataclass

from common.voice_commands import WAKE_ADDRESS_PATTERN, normalize

PERSONAS = {
    'putin': (
        'Putin',
        'Use a fictional satire of Vladimir Putin: dry official rhetoric, '
        'elaborate bureaucratic understatement and deadpan absurdity about the requested topic.',
    ),
    'genghis_khan': (
        'Genghis Khan',
        'Use a fictional historical parody of Genghis Khan: grandiose steppe-emperor '
        'rhetoric and exaggerated boasts about harmless everyday matters.',
    ),
}
DURATION_S = 300


def roleplay_command(text: str) -> str | None:
    """Match a whole explicit mode command, not a quotation or combined task."""
    text = normalize(text)
    text = re.sub(
        r'^(?:(?:hey|okay|ok|so|yo) )*(?:' + WAKE_ADDRESS_PATTERN + r'\b *)+', '', text,
    )
    text = re.sub(r'^(?:can you |could you |please |пожалуйста )', '', text)
    text = text.removesuffix(' please').removesuffix(' пожалуйста')
    if text in {
        'stop roleplay', 'stop role play', 'stop the parody', 'stop parody',
        'normal mode', 'be rowan again', 'выключи пародию', 'обычный режим',
        'перестань играть роль', 'отмени роль',
    }:
        return 'off'
    text = re.sub(r' (?:for (?:the next )?(?:5|five) minutes|на (?:5|пять) минут)$', '', text)
    names = {
        r'(?:vladimir )?putin|(?:владимир(?:а|ом)? )?путин(?:а|ым)?': 'putin',
        r'genghis khan|чингисхан(?:а|ом)?|чингис хан': 'genghis_khan',
    }
    for name, persona in names.items():
        if re.fullmatch(
            r'(?:(?:answer|reply|speak|act) (?:as|like)|impersonate|pretend to be|'
            r'roleplay as|отвечай как|говори как|изобрази|побудь|сыграй) (?:' + name + r')', text,
        ):
            return persona
    return None


@dataclass(frozen=True)
class Mode:
    persona: str
    expires: float


class RoleplayModes:
    """One mode per recognized speaker on this connection; guests share a bucket."""

    def __init__(self):
        self._modes: dict[str, Mode] = {}

    def set(self, owner: str, persona: str, *, now: float | None = None) -> None:
        owner = owner.casefold()
        if persona == 'off':
            self._modes.pop(owner, None)
        elif persona in PERSONAS:
            self._modes[owner] = Mode(persona, (time.monotonic() if now is None else now) + DURATION_S)
        else:
            raise ValueError('Unsupported parody persona')

    def current(self, owner: str, *, now: float | None = None) -> str | None:
        owner = owner.casefold()
        mode = self._modes.get(owner)
        if mode is None:
            return None
        if (time.monotonic() if now is None else now) >= mode.expires:
            del self._modes[owner]
            return None
        return mode.persona


def roleplay_prompt(persona: str | None) -> str:
    if persona not in PERSONAS:
        return ''
    label, direction = PERSONAS[persona]
    return (
        '\n\nEXPLICIT TEMPORARY PARODY MODE: ' + label + '\n' + direction
        + ' This mode was explicitly activated by the speaker, independently of chat history. '
        'It overrides the default delivery style only. Keep it across questions until the server ends the mode. '
        'Perform the answer directly without offering to perform it later. '
        'You remain Rowan using the installed voice, not the actual public figure. '
        'The server labels spoken replies as parody. Never claim invented quotes are real statements '
        'or that a fictional operation/action actually happened. Political satire is fiction; '
        'a request for real facts or a serious explanation still gets an accurate answer, with uncertainty acknowledged. '
        'Do not endorse hatred, real violence, threats against people or racial slurs. '
        'All tool permissions, consent, privacy and content rules still apply. '
        'A mode change is not authorization for tools or destructive actions. '
        'Do not invent a restriction against ordinary historical or political parody.'
    )


def label_reply(text: str) -> str:
    """Label both spoken and displayed output without depending on model compliance."""
    return text if text.casefold().startswith('parody:') else 'Parody: ' + text
