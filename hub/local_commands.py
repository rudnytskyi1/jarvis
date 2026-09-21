"""Exact, bounded shortcuts. Ambiguous/multi-action utterances go to the LLM.

Returns tool arguments only; the caller MUST use the normal permission gate.
"""
from __future__ import annotations

import re


def direct_command(text: str, wake_words=()) -> tuple[dict, str] | None:
    text = text.strip().lower().rstrip(".!?")
    for word in sorted(wake_words, key=len, reverse=True):
        text = re.sub(r"^" + re.escape(word.lower()) + r"[\s,:]+", "", text, count=1)
    text = re.sub(r"^(?:please\s+)", "", text)
    text = re.sub(r"(?:,?\s+please)$", "", text)
    volume = re.fullmatch(r"(?:set (?:the )?volume to|volume|громкость|установи громкость(?: на)?)\s+(\d{1,3})(?:\s*%| percent| процентов)?", text)
    if volume and 0 <= int(volume[1]) <= 100:
        value = int(volume[1])
        return {"command": "volume_set", "value": value}, f"Volume set to {value} percent."
    exact = {
        "mute": ("mute", "Muted."), "выключи звук": ("mute", "Muted."),
        "unmute": ("unmute", "Sound on."), "включи звук": ("unmute", "Sound on."),
        "next track": ("media_next", "Next track."), "следующий трек": ("media_next", "Next track."),
        "previous track": ("media_prev", "Previous track."), "предыдущий трек": ("media_prev", "Previous track."),
        "volume up": ("volume_up", "Volume increased."), "громче": ("volume_up", "Volume increased."),
        "volume down": ("volume_down", "Volume decreased."), "тише": ("volume_down", "Volume decreased."),
    }
    if text in exact:
        command, reply = exact[text]
        return {"command": command}, reply
    app = re.fullmatch(r"(?:open|открой) (chrome|firefox|spotify|steam|notepad|calculator)", text)
    if app:
        return {"command": "open_app", "value": app[1]}, f"Opened {app[1]}."
    # A play/pause toggle cannot truthfully guarantee a PAUSED state; leave it
    # to a state-aware tool rather than incorrectly treating 'pause' as toggle.
    return None
