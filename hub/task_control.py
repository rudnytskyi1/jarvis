"""Conservative parsing for confirmation of task cancellation."""
import re

from common.voice_commands import WAKE_ADDRESS_PATTERN


def decision(text):
    cleaned = re.sub(r"[^\w\s']", ' ', str(text).casefold())
    cleaned = re.sub(r'^\s*(?:(?:hey|okay|ok)\s+)?' + WAKE_ADDRESS_PATTERN + r'\b\s*', '', cleaned)
    cleaned = ' '.join(cleaned.split()).removeprefix('please ').removesuffix(' please')
    if cleaned in {'no', 'continue', 'keep going', 'no continue', 'do not cancel', "don't cancel", 'продолжай', 'не отменяй', 'нет'}:
        return False
    if cleaned in {'yes', 'cancel', 'cancel the task', 'yes cancel', 'yes cancel the task', 'stop the task', 'отмени', 'прерви', 'да'}:
        return True
    return None
