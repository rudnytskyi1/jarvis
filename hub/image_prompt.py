"""Extract literal visual wording without inventing a different image request.

This module only removes narrow transport/workflow phrases. It never translates,
fixes speech recognition, changes a depicted object or appends creative advice.
Ambiguous clauses stay intact, especially negations and later visual constraints.
"""
from __future__ import annotations

import re

_DEFAULT_WAKE_WORDS = ('rowan', 'roan', 'rowen', 'роуэн', 'роуен', 'роуан', 'роан', 'рован')


def _without_wake(text: str, wake_words=()) -> str:
    value = str(text or '').strip()
    aliases = {word.casefold().strip() for word in (*_DEFAULT_WAKE_WORDS, *wake_words)
               if isinstance(word, str) and word.strip()}
    names = '|'.join(re.escape(word) for word in sorted(aliases, key=len, reverse=True))
    pattern = re.compile(r'^(?:(?:hey|hi|okay|ok|yo|эй|привет)\s*[,!.:-]?\s+)?'
                         r'(?:' + names + r')(?!\w)[\s,;:!?\.\-]*', re.I)
    for _ in range(2):
        shorter = pattern.sub('', value, count=1)
        if shorter == value:
            break
        value = shorter.lstrip()
    return value


_CAPTURE_PREFIX = re.compile(
    r'^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?'
    r'(?:take|capture|snap)\s+(?:a\s+|the\s+)?(?:picture|photo|screenshot)'
    r'(?:\s+(?:of\s+(?:me|us)|(?:from|through|with)\s+(?:the\s+)?camera))?'
    r'\s*(?:,\s*(?:and\s+)?|(?:and|then)\s+)', re.I)
_CAPTURE_PREFIX_RU = re.compile(
    r'^(?:(?:можешь|можете)\s+)?(?:пожалуйста[, ]+)?'
    r'(?:сделай|сделать|сними|снять)\s+(?:фото|фотографию|снимок|скриншот)'
    r'(?:\s+(?:меня|нас|с\s+камеры))?\s*(?:,\s*(?:и\s+)?|(?:и|затем)\s+)', re.I)

_REFERENT = r'(?:it|this|that|(?:this|that|the)\s+(?:picture|photo|image|screenshot|result))'
_BACKGROUND = r'(?:(?:my|the|a|your)\s+)?(?:desktop\s+(?:background|wallpaper)|background(?:\s+(?:picture|image))?|wallpaper)'
_ON_PC = r'(?:\s+(?:on|for)\s+(?:(?:this|the|my|your)\s+)?(?:computer|pc|desktop))?'
_EN_WALLPAPER = (
    r'(?:(?:set|put|use|apply)\s+' + _REFERENT + r'\s+(?:as\s+|for\s+)' + _BACKGROUND
    + r'|make\s+' + _REFERENT + r'\s+(?:as\s+)?' + _BACKGROUND + r')' + _ON_PC)
_EN_SAVE_OPEN = (
    r'(?:save|open|download)\s+' + _REFERENT
    + r'(?:\s+(?:on|to|in|with)\s+(?:(?:this|the|my|your)\s+)?'
      r'(?:desktop|computer|pc|pictures\s+folder|default\s+(?:photo\s+)?(?:viewer|app)))?')
_EN_SEND = (
    r'(?:send|share|post)\s+' + _REFERENT + r'\s+(?:to|in|on)\s+'
    r'(?:telegram(?:\s+(?:chat|group))?|(?:(?:our|the|my)\s+)?(?:telegram\s+)?group\s+chat'
    r'(?:\s+on\s+telegram)?)')
_RU_REFERENT = r'(?:это|его|её|ее|(?:это|эту|эту\s+самую|эту\s+же)\s+(?:фото|фотографию|картинку|изображение))'
_RU_WALLPAPER = (
    r'(?:поставь|поставить|установи|установить|сделай|сделать|используй)\s+' + _RU_REFERENT
    + r'\s+(?:(?:как|на|в\s+качестве)\s+)?(?:фон(?:ом)?|обои|обоями)'
      r'(?:\s+(?:рабочего\s+стола|на\s+(?:(?:этом|моём|моем)\s+)?(?:компьютере|пк)))?')
_RU_SAVE_OPEN = (
    r'(?:сохрани|сохранить|открой|открыть|скачай|скачать)\s+' + _RU_REFERENT
    + r'(?:\s+(?:на|в)\s+(?:рабочий\s+стол|рабочем\s+столе|компьютер|компьютере|пк))?')
_RU_SEND = (
    r'(?:отправь|отправить|отправьте|пришли|перешли|скинь)\s+' + _RU_REFERENT + r'\s+в\s+'
    r'(?:телеграм(?:м)?|telegram|(?:(?:наш|мой|этот|общий)\s+)?(?:групповой\s+)?чат'
    r'(?:\s+в\s+(?:телеграм(?:м)?|telegram))?)')
_CONNECTOR = r'(?:and(?:\s+also)?|then|also|а\s+потом|а\s+затем|и(?:\s+потом)?|затем)'
_SEPARATOR = r'(?:\s*[,;]\s*(?:' + _CONNECTOR + r'\s+)?|\s+' + _CONNECTOR + r'\s+)'
_WORKFLOW_SUFFIX = re.compile(
    _SEPARATOR + r'(?:' + _EN_WALLPAPER + r'|' + _EN_SAVE_OPEN + r'|' + _EN_SEND + r'|'
    + _RU_WALLPAPER + r'|' + _RU_SAVE_OPEN + r'|' + _RU_SEND + r')'
    + r'(?:\s*,?\s*(?:please|пожалуйста))?\s*[.!?]*\s*$', re.I)
_PURE_WORKFLOW = re.compile(r'^(?:' + _EN_WALLPAPER + r'|' + _EN_SAVE_OPEN + r'|' + _EN_SEND + r'|'
                            + _RU_WALLPAPER + r'|' + _RU_SAVE_OPEN + r'|' + _RU_SEND + r')[.!?\s]*$', re.I)
_CAPTION_INTRO = re.compile(
    r'\b(?:words|text|caption|phrase|message|says|saying|reads|reading|'
    r'надпись|надписью|текст|текстом|слова|словами|фразу|фразой)\b', re.I)


def _possibly_literal_caption(prefix: str) -> bool:
    """Unclosed quotes/text-on-image wording makes a delivery suffix ambiguous."""
    quote = None
    closed_quotes = []
    closers = {'"': '"', '\u201c': '\u201d', '\u00ab': '\u00bb', "'": "'"}
    for index, char in enumerate(prefix):
        if char == "'" and index and index + 1 < len(prefix) and prefix[index - 1].isalnum() and prefix[index + 1].isalnum():
            continue  # don't / person's are not quoted captions
        if quote is not None and char == quote:
            quote = None
            closed_quotes.append(index)
        elif quote is None and char in closers:
            quote = closers[char]
    if quote is not None:
        return True
    introductions = list(_CAPTION_INTRO.finditer(prefix))
    if not introductions:
        return False
    # Without a closing quote, "draw the words ... and send it to Telegram"
    # may be exactly the caption. Keep it instead of truncating the artwork.
    return not any(index >= introductions[-1].end() for index in closed_quotes)


def visual_request(text: str, wake_words=()) -> str:
    """Return original visual words, minus clear wake/capture/PC-only clauses."""
    value = _without_wake(text, wake_words)
    value = _CAPTURE_PREFIX.sub('', value, count=1)
    value = _CAPTURE_PREFIX_RU.sub('', value, count=1)
    # Only suffixes are removed. An OS instruction followed by a visual detail
    # is deliberately retained rather than swallowing the user's later words.
    while True:
        match = _WORKFLOW_SUFFIX.search(value)
        if match is None:
            break
        if _possibly_literal_caption(value[:match.start()]):
            break
        value = value[:match.start()].rstrip()
    return value


_VISUAL_ACTION = re.compile(
    r'\b(?:draw|paint|sketch|illustrate|render)\b|'
    r'\b(?:generate|create)\b[^.!?]{0,90}\b(?:image|picture|photo|portrait|art|illustration|wallpaper)\b|'
    r'\bmake\s+(?:me|him|her|us|them|[A-Z][a-z]+)\s+look\s+like\b|'
    r'\b(?:turn|transform)\b[^.!?]{0,60}\binto\b|'
    r'\b(?:make|turn|change)\s+(?:it|that|this)\s+(?:look\s+)?(?:more\s+)?'
    r'(?:red|blue|green|yellow|black|white|purple|orange|realistic|photorealistic|cartoon|bigger|smaller)\b|'
    r'\b(?:edit|modify|change|remove|replace|add|put|give|make)\b[^.!?]{0,100}'
    r'\b(?:picture|image|photo|portrait|face|head|hat|cap|hair|clothes|shirt|costume|sky|eyes|glasses|crown|wallpaper|background|next\s+to\s+me)\b|'
    r'\b(?:нарисуй|нарисовать|изобрази|изобразить|сгенерируй|сгенерировать|дорисуй|дорисовать|отредактируй)\b|'
    r'\b(?:сделай|сделать|преврати|превратить|добавь|добавить|надень|надеть|поставь|убери|замени|измени)\b'
    r'[^.!?]{0,100}\b(?:меня|его|её|ее|нас|изображение|картинку|фото|шляпу|шапку|голову|голове|лицо|фон|рядом)\b', re.I)
_NEGATIVE_PREFIX = re.compile(r'(?:\b(?:do\s+not|don[’\x27]t|never|не)\s+)$', re.I)
_EXISTING_BACKGROUND = re.compile(
    r'\b(?:make|set|use|put|apply)\s+(?:the|this|that|my)\s+(?:desktop\s+)?'
    r'(?:background(?:\s+picture)?|wallpaper)\b[^.!?]*\b(?:use|using)\s+(?:it|that|this)\b', re.I)


def is_image_request(text: str) -> bool:
    """Recognize an explicit new visual edit, not viewing/installing an image."""
    value = visual_request(text)
    value = re.sub(r'^(?:(?:can|could|would|will)\s+you\s+|(?:можешь|можете)\s+)?(?:please\s+)?', '', value, count=1, flags=re.I)
    # An existing-result workflow is not authorization to generate again.
    if not value or _PURE_WORKFLOW.fullmatch(value) or _EXISTING_BACKGROUND.search(value):
        return False
    for match in _VISUAL_ACTION.finditer(value):
        if not _NEGATIVE_PREFIX.search(value[max(0, match.start() - 20):match.start()]):
            return True
    return False


def is_existing_image_workflow(text: str) -> bool:
    """A request to deliver an existing result must not buy a new image."""
    value = visual_request(text)
    value = re.sub(r'^(?:(?:can|could|would|will)\s+you\s+|(?:можешь|можете)\s+)?'
                   r'(?:please\s+|пожалуйста[, ]+)?', '', value, count=1, flags=re.I)
    value = re.sub(r'[,\s]+(?:please|пожалуйста)[.!?\s]*$', '', value, flags=re.I)
    return bool(_PURE_WORKFLOW.fullmatch(value) or _EXISTING_BACKGROUND.search(value))


def person_reference_requested(text: str, name: str) -> bool:
    """Names in exclusions or quoted artwork are not permission to upload portraits."""
    matches = list(re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I))
    allowed = False
    for match in matches:
        prefix = text[:match.start()]
        if _possibly_literal_caption(prefix):
            continue
        clause = re.split(r'[.!?;,]|\b(?:but|however|но|зато)\b', prefix, flags=re.I)[-1]
        after = re.split(r'[.!?;,]|\b(?:but|however|но|зато)\b', text[match.end():], flags=re.I)[0]
        if (_NEGATED_WORKFLOW.search(clause)
                or re.search(r'\b(?:except|exclude|excluding|avoid|кроме|исключи)\b', clause, re.I)
                or re.match(r"\s+(?:should\s+not|must\s+not|shouldn[’']t|mustn[’']t|не)\b", after, re.I)):
            return False
        allowed = True
    return allowed


def is_image_clarification(text: str) -> bool:
    """Only short affirmative/target answers; caller must bind a pending request."""
    value = _without_wake(text).strip().rstrip('.!?').strip()
    if not value or len(value) > 100 or len(value.split()) > 12 or is_image_request(value):
        return False
    if re.fullmatch(
        r'(?:yes|yeah|yep|correct|exactly|okay|ok|sure|do\s+it|go\s+ahead|'
        r'that[’\x27]s\s+right|да|ага|верно|именно|давай|сделай|делай)'
        r'(?:[, ]+(?:please|do\s+it|go\s+ahead|пожалуйста))?', value, re.I):
        return True
    if re.fullmatch(
        r'(?:(?:the\s+)?(?:one|person|guy|man|woman)\s+)?(?:on\s+(?:the\s+)?)?'
        r'(?:left|right|middle|center)|'
        r'(?:(?:the\s+)?(?:one|person|guy|man|woman)\s+)(?:in|wearing)\s+[\w -]{1,45}|'
        r'(?:number\s+)?(?:one|two|three|four|[1-9])|'
        r'(?:тот|та|человек|парень|девушка)?\s*(?:слева|справа|посередине|в\s+центре)|'
        r'(?:номер\s+)?(?:один|два|три|первый|второй|третий)', value, re.I):
        return True
    # A proper-name answer is useful after "Which person?". Avoid treating
    # ordinary commands or lowercase chit-chat as somebody's identity.
    if re.fullmatch(r'(?:it[’\x27]s|that[’\x27]s|это)\s+[\w -]{1,50}', value, re.I):
        return True
    words = value.split()
    reserved = {'thanks', 'thank', 'hello', 'stop', 'cancel', 'no', 'wait', 'weather',
                'open', 'close', 'show', 'run', 'load', 'start', 'save', 'draw', 'make', 'change',
                'спасибо', 'привет', 'стоп', 'отмена', 'нет', 'подожди'}
    return (1 <= len(words) <= 3 and not any(word.casefold() in reserved for word in words)
            and all(word[0].isupper() and all(c.isalpha() or c in "-'" for c in word) for word in words))


_EN_DIRECT_WALLPAPER = (
    r'(?:change|set|replace|update)\s+(?:(?:my|the|this|your)\s+)?'
    r'(?:desktop\s+(?:background|wallpaper)|wallpaper)\b')
_EN_EXISTING_BACKGROUND = (
    r'(?:make|set)\s+(?:the|my)\s+background(?:\s+picture)?\s*[,;]\s*'
    r'use\s+(?:that|this|the)\s+(?:picture|image|photo)\b')
_RU_DIRECT_WALLPAPER = (
    r'(?:измени|изменить|поменяй|поменять|замени|заменить|установи|поставь)\s+'
    r'(?:(?:мои|эти|мой|этот)\s+)?(?:обои|фон\s+рабочего\s+стола)\b')
_RU_PICTURE_WALLPAPER = (
    r'(?:сделай|сделать|поставь|поставить|установи|установить)\s+'
    r'(?:(?:это|эту|мою|эту\s+же)\s+)?(?:фото|картинку|изображение|фотографию)\s+'
    r'(?:(?:как|на|в\s+качестве)\s+)?(?:фон(?:ом)?(?:\s+рабочего\s+стола)?|обои|обоями)\b')
_WALLPAPER_CHANGE = re.compile(
    r'\b(?:' + '|'.join((_EN_WALLPAPER, _EN_DIRECT_WALLPAPER, _EN_EXISTING_BACKGROUND,
                         _RU_WALLPAPER, _RU_DIRECT_WALLPAPER, _RU_PICTURE_WALLPAPER)) + r')', re.I)
_NEGATED_WORKFLOW = re.compile(
    r'\b(?:not|never|without|no|cannot|cant|dont|didnt|can[’\x27]t|don[’\x27]t|didn[’\x27]t|'
    r'wouldn[’\x27]t|shouldn[’\x27]t|не|нельзя|нет|без)\b', re.I)
_DISCUSSED_WORKFLOW = re.compile(
    r'\b(?:why|how|when|whether|if|unless|explain|describe|discuss|translate|'
    r'said|saying|asked|told|earlier|previously|yesterday|history|last\s+time|before|'
    r'почему|зачем|как|когда|если|объясни|объяснить|расскажи|переведи|'
    r'сказал|сказала|просил|просила|раньше|вчера|истории)\b', re.I)
_IN_ARTWORK = re.compile(
    r'^\s+(?:(?:in|inside|within|of|for)\s+(?:(?:the|this|that|a|my)\s+)?'
    r'(?:picture|photo|image|screenshot|scene|room)|behind\s+(?:me|him|her|the\s+person)|'
    r'(?:on|for)\s+(?:(?:the|my|a)\s+)?'
    r'(?:wall|walls|bedroom|room)|'
    r'(?:в|на)\s+(?:(?:этой|этом|моём|моем)\s+)?'
    r'(?:картинке|фото|изображении|сцене|стене|стенах|комнате))\b', re.I)


def action_revoked(text: str) -> bool:
    """Recognize a trailing cancellation, ignoring text intended as a caption."""
    pattern = re.compile(
        r'\b(?:actually|but|no|wait|on\s+second\s+thought|хотя|нет|погоди|подожди)'
        r'[,\s]+(?:please\s+)?(?:don[’\x27]t(?:\s+bother)?|do\s+not|не\s+надо|не\s+нужно)'
        r'\s*[.!?]*\s*$|'
        r'\b(?:cancel\s+(?:that|it)|never\s*mind|forget\s+(?:that|it)|'
        r'отмена|отмени\s+это|передумал(?:а)?)\b', re.I)
    return any(not _possibly_literal_caption(text[:match.start()]) for match in pattern.finditer(text))


def wallpaper_change_requested(text: str) -> bool:
    """Require a positive wallpaper action in this request, never in history.

    Image content such as a background, physical wallpaper or a quoted command
    is insufficient. Ambiguous discussion/negation is deliberately denied;
    callers should keep generation/display available without changing Windows.
    """
    value = _without_wake(text)
    if action_revoked(value):
        return False
    for match in _WALLPAPER_CHANGE.finditer(value):
        prefix, after = value[:match.start()], value[match.end():]
        if _possibly_literal_caption(prefix) or _IN_ARTWORK.search(after):
            continue
        # Separate sentences and an explicit "but" can introduce a new positive
        # command. A plain "and" does not erase a preceding negation's scope.
        clause = re.split(r'[.!?;]|\b(?:but|however|но|зато)\b', prefix, flags=re.I)[-1]
        if _NEGATED_WORKFLOW.search(clause) or _DISCUSSED_WORKFLOW.search(clause):
            continue
        # "You set it as wallpaper" reports an action. "Can you set ...?" is
        # a current request and remains accepted after its polite prefix.
        direct = re.sub(r'^\s*(?:(?:can|could|would|will)\s+you\s+|'
                        r'(?:можешь|можете)\s+)?(?:please\s+|пожалуйста[, ]+)?',
                        '', clause, count=1, flags=re.I).strip()
        if re.fullmatch(r'(?:you|he|she|they|we|ты|вы|он|она|они|мы)'
                        r'(?:\s+(?:already|just|уже|только\s+что))?', direct, re.I):
            continue
        # A correction revoking this very action takes precedence over its
        # earlier positive wording. "Don't change anything else" does not.
        if re.search(r'\b(?:but|actually|no|но|нет)\b[^.!?]{0,30}'
                     r'\b(?:don[’\x27]t|do\s+not|не)\s+'
                     r'(?:set|apply|change|touch|меняй|ставь|устанавливай)\b', after, re.I):
            continue
        return True
    return False
