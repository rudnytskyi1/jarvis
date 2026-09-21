"""Deterministic enrollment intent and prompts, independent of an LLM."""
import re

from hub.speaker import ENROLL_MIN_SAMPLES

PHRASES = (
    "Rowan AI, please remember how my normal voice sounds when I speak in this room.",
    "Rowan AI, today I would like to talk with my friends about music and movies.",
    "Rowan AI, please recognize my voice the next time I ask you a question.",
    "Rowan AI, I sometimes speak quietly, and sometimes I ask questions from across the room.",
    "Rowan AI, when I finish this recording, I would like you to remember my natural speaking voice.",
    "Rowan AI, please help me find information, open applications, and answer questions throughout the day.",
)


def requested(text):
    if re.search(r"(?:(?:do|can) you|have you|did you)\s+(?:know|remember|recognize|save)", text, re.I):
        return False
    if re.search(r"\b(?:don't|do not|never)\s+(?:enroll|register|remember|record|save|add|update)\b", text, re.I):
        return False
    additional = (
        r"\b(?:add|record|save|collect)\s+"
        r"(?:(?:a|some|more|additional|extra|new|another|few)\s+)*"
        r"(?:(?:my\s+)?voice\s+samples?|samples?\s+(?:to|for|of)\s+my\s+(?:voice|profile))\b"
        r"|\band\s+roll\s+my\s+voice\b"
        r"|(?:дозаписать|добавить|дозапишем|дозапишешь).{0,80}(?:голос|образц)"
    )
    return bool(re.search(additional, text, re.I) or re.search(r"\b(?:enroll|enrol|register)\b|(?:remember|learn|save|recognize|update|improve|record)\s+my\s+(?:voice|face)|remember (?:me|who I am)|(?:запомни|сохрани|добавь|выучи|дозапиши|обнови).*(?:голос|лицо|меня)|регистрац", text, re.I))


def extract_name(text, *, answering=False):
    # Prefer an explicitly spelled name when Whisper also guessed its spelling.
    spelling = re.search(r"\b([A-Za-z](?:\s*[-.]\s*[A-Za-z]){2,})\b", text)
    if spelling:
        return re.sub(r"[^A-Za-z]", "", spelling.group(1)).capitalize()
    # Whisper may punctuate a hesitation as "my name, is Anton".
    matches = list(re.finditer(r"(?:my[\s,]+name[\s,]+is|меня[\s,]+зовут|name[\s,]+is)[\s,:]+([\w'’-]+(?:\s+[\w'’-]+){0,2})", text, re.I))
    match = matches[-1] if matches else None  # The person's last correction wins.
    if not match and (answering or requested(text)):
        matches = list(re.finditer(r"\b(?:I am|I'm|I’m|this is)\s+([\w'’-]+)\b", text, re.I))
        match = matches[-1] if matches else None
        if match and match.group(1).casefold() in {'hungry', 'tired', 'sorry', 'ready', 'trying', 'speaking', 'talking', 'eating', 'done', 'not', 'a', 'the', 'here'}:
            match = None
    if not match:
        return None
    words = match.group(1).split()
    stop = {"and", "please", "remember", "update", "record", "register", "enroll", "recognize", "i", "my", "voice", "face", "и", "запомни"}
    clean = []
    for word in words:
        if word.casefold() in stop:
            break
        clean.append(word)
    return " ".join(clean).strip() or None


def initiation(segments):
    """Recover ONLY a registration intent/name across one voice's pauses.

    Never append a later side conversation or a PC command. The caller must
    first reject overlap, uncertain words and recordings with several voices.
    """
    texts = [s['text'] for s in segments]
    if not any(requested(t) for t in texts):
        return None
    parts = [t for t in texts if requested(t)]
    name = next((extract_name(t, answering=True) for t in reversed(texts)
                 if extract_name(t, answering=True)), None)
    if name and extract_name(' '.join(parts)) != name:
        parts.append(f'My name is {name}.')
    return ' '.join(parts)


def clarification(reason, registering=False):
    if reason == 'overlapping_speech':
        return 'I heard voices talking over each other. Please read the sentence while nobody else speaks.' if registering else 'I heard voices talking over each other. Please say Rowan AI and repeat one at a time.'
    if registering:
        return "I couldn't get a clear recording. Your voice does not need to be recognized yet. Please read the sentence again, alone and a little closer to the microphone."
    if reason == 'uncertain_attribution':
        return "I couldn't make out the recording clearly. Please move a little closer and repeat your request."
    return "I heard separate parts of a conversation. Please say Rowan AI and repeat your request in one sentence."


def prompt(pending):
    step = int(pending.get("samples", 0)) + 1
    progress = f"Sentence {step} of {ENROLL_MIN_SAMPLES}. " if step <= ENROLL_MIN_SAMPLES else 'A little more speech is needed. '
    return progress + f"{pending['name']}, read the sentence on screen. Start with Rowan."


def caption(pending):
    step = min(int(pending.get("samples", 0)), len(PHRASES) - 1)
    return PHRASES[step]
