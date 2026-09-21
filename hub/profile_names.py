"""Explicit spoken profile renaming. Names are data, never instructions."""
import re

from common.voice_commands import WAKE_ADDRESS_PATTERN


def rename_request(text):
    text = re.sub(r'^\s*(?:(?:hey|okay|ok)\s+)?' + WAKE_ADDRESS_PATTERN + r'\b[\s,.!?]*', '', text, flags=re.I)
    name = r"([\w'’-]+(?:\s+[\w'’-]+){0,2})"
    for pattern in (
        rf'(?:change|correct)\s+my\s+(?:saved\s+)?name\s+to\s+{name}',
        rf'rename\s+me\s+to\s+{name}',
        rf'(?:измени|исправь|поменяй)\s+мо[её]\s+имя\s+на\s+{name}',
        rf'переименуй\s+меня\s+в\s+{name}',
    ):
        match = re.fullmatch(pattern + r'[.!?]*', text.strip(), re.I)
        if match:
            return {'old_name': '', 'new_name': match.group(1)}
    for pattern in (rf'rename\s+{name}\s+to\s+{name}',
                    rf'(?:change|correct)\s+(?:the\s+)?name\s+(?:from\s+)?{name}\s+to\s+{name}',
                    rf'переименуй\s+{name}\s+в\s+{name}'):
        match = re.fullmatch(pattern + r'[.!?]*', text.strip(), re.I)
        if match:
            return {'old_name': match.group(1), 'new_name': match.group(2)}
    return None


def reserved(name, aliases):
    return name.strip().casefold() in {str(alias).strip().casefold() for alias in aliases}
