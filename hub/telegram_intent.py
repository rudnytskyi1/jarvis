"""Current group-send requests, excluding reports, quotes and revocations."""
import re

from common.voice_commands import WAKE_ADDRESS_PATTERN
from hub.image_prompt import _possibly_literal_caption

_DESTINATION = r'(?:telegram|tg|group(?:\s+chat)?|телеграм\w*|телег[ауе]|тг|групп\w*|чат\w*)'
_ACTION = re.compile(
    r'\b(?:send|post|share|message(?=\s+(?:(?:the|our|this)\s+)?(?:group|telegram)\b)|'
    r'отправь(?:те)?|отправить|отошли(?:те)?|пришли(?:те)?|перешли(?:те)?|переслать|'
    r'скинь(?:те)?|скинуть|'
    r'напиши(?:те)?|написать|запости|опубликуй)\b', re.I)
_TARGET_AFTER = re.compile(
    r'\b(?:to|in|into|on|with|в|во)\s+'
    r'(?:(?:the|our|this|that|my|наш\w*|эт\w*|мо\w*|общ\w*)\s+){0,2}'
    + _DESTINATION + r'\b', re.I)
_OWN_NAMED_TARGET = re.compile(
    r'\b(?:to|in|into|on|with|в|во)\s+(?:our|наш\w*)\s+'
    r'(?P<qualifiers>(?:[\w-]+\s+){1,4})'
    r'(?:group(?:\s+chat)?|групп\w*|чат\w*)\b', re.I)
_GROUP_QUALIFIER_BOUNDARY = re.compile(
    r'\b(?:and|or|but|then|not|never|without|no|except|other|another|'
    r'his|her|their|your|its|и|или|но|потом|не|нет|без|кроме|'
    r'его|её|ее|их|ваш\w*|чуж\w*|друг\w*)\b', re.I)
_EXCLUDED_TARGET = re.compile(r'\b(?:not|never|except(?:\s+for)?|не|кроме)\s*$', re.I)
_NEGATED = re.compile(
    r'\b(?:not|never|without|no|cannot|cant|dont|didnt|can[’\x27]t|don[’\x27]t|'
    r'didn[’\x27]t|wouldn[’\x27]t|shouldn[’\x27]t|не|нельзя|нет|без)\b', re.I)
_DISCUSSION = re.compile(
    r'\b(?:why|how|when|what|whether|if|unless|explain|describe|discuss|translate|'
    r'said|saying|asked|told|earlier|previously|yesterday|history|last\s+time|before|'
    r'did\s+you|have\s+you|should\s+(?:i|we|you)|'
    r'почему|зачем|как|когда|если|объясни|объяснить|расскажи|переведи|'
    r'сказал\w*|просил\w*|раньше|вчера|истори\w*)\b', re.I)
_REPORT_SUBJECT = re.compile(
    r'\b(?:i|you|he|she|they|we|someone|somebody|nobody|я|ты|вы|он|она|они|мы|кто-то)'
    r'(?:\s+(?:already|just|usually|always|will|would|should|can|might|may|must|'
    r'is\s+going\s+to|уже|обычно|всегда|будет|может|только\s+что))?\s*$', re.I)
_REPORTED_PLAN = re.compile(
    r'\b(?:wants?|wanted|plans?|planned|intends?|intended|tried|trying)\s+to\s*$|'
    r'\b(?:хочет|хотел\w*|собирается|планирует)\s*$', re.I)
_POLITE = re.compile(
    r'^\s*(?:' + WAKE_ADDRESS_PATTERN + r'[,\s:]+)?'
    r'(?:(?:please|kindly|пожалуйста|плиз)[,\s]+)?'
    r'(?:(?:(?:can|could|would|will)\s+you|'
    r'i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to|'
    r'(?:ты\s+|вы\s+)?(?:можешь|можете)(?:\s+ли)?|'
    r'я\s+хочу[,]?\s+чтобы\s+ты)\s+)?'
    r'(?:(?:please|kindly|пожалуйста|плиз)[,\s]+)?', re.I)
_REVOCATION = re.compile(
    r'\b(?:(?:do\s+not|don[’\x27]t|dont|never)\s+(?:send|post|share|message)|'
    r'не\s+(?:отправляй\w*|отправь\w*|отправля\w*|скидывай\w*|скинь\w*|'
    r'пиши\w*|публикуй\w*|посылай\w*)|не\s+надо\s+отправлять)\b|'
    r'\b(?:actually|but|no|wait|on\s+second\s+thought|вообще|хотя|нет|погоди|подожди)'
    r'[,\s]+(?:(?:please|пожалуйста)[,\s]+)?'
    r'(?:don[’\x27]t|dont|do\s+not|не\s+надо|не\s+нужно|отмена|стоп|cancel|stop|wait)'
    r'(?=\s*(?:[.!?;,]|$))|'
    r'[.;,]\s*(?:don[’\x27]t|do\s+not|no|cancel|stop|не\s+надо|не\s+нужно|стоп)'
    r'(?=\s*(?:[.!?;,]|$))|'
    r'\b(?:actually|but|wait|но|хотя)[,\s]+(?:don[’\x27]t|do\s+not)\s+'
    r'(?:bother|do\s+(?:it|that))(?=\s*(?:[.!?;,]|$))|'
    r'\b(?:never\s*mind|cancel\s+(?:that|it|the\s+send)|forget\s+(?:that|it)|'
    r'отмена|отмени(?:\s+(?:это|отправку))?|передумал(?:а)?)\b', re.I)


def _without_quotes(text):
    """Mask quoted text while retaining positions and normal apostrophes."""
    result, quote = list(text), None
    closers = {'"': '"', '“': '”', '«': '»', "'": "'", '`': '`'}
    for index, char in enumerate(text):
        if quote is not None:
            result[index] = ' '
            if char == quote:
                quote = None
        elif (char == "'" and index and index + 1 < len(text)
              and text[index - 1].isalnum() and text[index + 1].isalnum()):
            continue
        elif char in closers:
            quote = closers[char]
            result[index] = ' '
    return ''.join(result)


def _own_named_target(text):
    """Accept a short name of our configured group, not another destination.

    Speech recognition may turn "Telegram" into a proper name. An explicit
    "our <name> group chat" still identifies the current shared group. Keep
    qualifiers within a single noun phrase so they cannot bridge instructions.
    """
    for target in _OWN_NAMED_TARGET.finditer(text):
        if (_GROUP_QUALIFIER_BOUNDARY.search(target.group('qualifiers'))
                or _EXCLUDED_TARGET.search(text[:target.start()])):
            continue
        return target
    return None


#: Anything the owner may call a picture they want sent.
_PICTURE = re.compile(
    r'\b(?:photo|picture|image|snapshot|screenshot|selfie|'
    r'фото|фотк\w*|фотографи\w*|снимок|скриншот|картинк\w*|изображени\w*|селфи)\b', re.I)
#: Destinations that mean the one chat Rowan posts to, or the sender themselves.
_CURRENT_CHAT = {'me', 'us', 'myself', 'here', 'this', 'the', 'my', 'our', 'your',
                 'telegram', 'tg', 'group', 'chat', 'rowan',
                 'меня', 'мне', 'нас', 'нам', 'себя', 'сюда', 'этот', 'эту', 'это', 'эти',
                 'наш', 'наша', 'наше', 'наши', 'телеграм', 'телеграмм', 'телеге', 'тг',
                 'группа', 'группу', 'группы', 'групповой', 'чат', 'чате', 'чата'}
#: Words that follow a picture object without naming a receiver.
_NOT_A_RECIPIENT = {'которое', 'который', 'которую', 'которые', 'которого', 'которой',
                    'что', 'кто', 'где', 'как', 'когда', 'если', 'пожалуйста', 'сейчас'}
#: "Here" in a room belongs to the room first: only the Telegram route itself
#: may read it as this chat (hub.telegram_control.current_chat_send_requested).
_ROOM_HERE = re.compile(r'\b(?:here|сюда)\b', re.I)
_PREPOSITION = re.compile(r"\b(?:to|for|into|on|at|with|в|во|для|к)\s+([\w'-]+)(?:\s+([\w'-]+))?", re.I)
#: Determiners carry no receiver of their own: "to my email" is the email.
_DETERMINER = {'my', 'our', 'your', 'his', 'her', 'their', 'the', 'this', 'that',
               'a', 'an', 'some', 'any', 'мой', 'мою', 'моё', 'мое', 'наш', 'наша',
               'наше', 'твой', 'ваш', 'его', 'её', 'ее', 'их', 'этот', 'эту', 'тот'}
_EN_RECIPIENT = re.compile(
    r'\b(?:send|share|post)\s+(?:(?:me|us|them|it|this|that|the|my|our|your|a|an|some)\s+)*'
    r'(?P<word>[A-Za-z][A-Za-z\'-]{1,})', re.I)
_PICTURE_RECIPIENT = re.compile(_PICTURE.pattern + r'\s*,?\s+(?P<word>[А-Яа-яЁё]{3,12})', re.I)


def _names_other(word):
    """Is this word somebody or somewhere else than the one Telegram chat?"""
    value = str(word or '').casefold()
    return bool(value) and not (
        value in _CURRENT_CHAT or value in _NOT_A_RECIPIENT or _PICTURE.fullmatch(value))


def _other_recipient(value):
    """Does the send clause promise the picture to somebody else?

    "Отправь фото маме" must never be read as "post it in Rowan's chat".
    """
    for match in _PREPOSITION.finditer(value):
        first, second = match.group(1), match.group(2)
        if _names_other(first):
            return True
        if second is not None and first.casefold() in _DETERMINER and _names_other(second):
            return True
    for match in _EN_RECIPIENT.finditer(value):
        if _names_other(match['word']):
            return True
    for match in _PICTURE_RECIPIENT.finditer(value):
        if _names_other(match['word']):
            return True
    return False


def picture_send_requested(text):
    """A current request to send a picture when the destination is implied.

    Rowan has exactly one Telegram chat to post to, so "отправь фото" and
    "send me the photo" already name the action and its object; the owner does
    not have to spell out the chat. The guards of
    :func:`telegram_send_requested` still apply: complaints, quotations,
    reports and revocations never authorize a send, and a picture promised to
    somebody else ("отправь фото маме") is not a request to post it here.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    plain = _without_quotes(text)
    for match in _ACTION.finditer(plain):
        prefix, after = plain[:match.start()], plain[match.end():]
        clause = re.split(r'[.!?;\n]|\b(?:but|however|но|зато)\b', prefix, flags=re.I)[-1]
        if (_NEGATED.search(clause) or _DISCUSSION.search(clause)
                or _possibly_literal_caption(text[:match.start()])):
            continue
        direct = _POLITE.sub('', clause, count=1).strip()
        if _REPORT_SUBJECT.search(direct) or _REPORTED_PLAN.search(direct):
            continue
        if _REVOCATION.search(after):
            continue
        send_clause = re.split(r'[.!?;\n]', after, maxsplit=1)[0]
        # The verb itself belongs to the clause, so "send Anton the photo" is
        # read as promising the picture to Anton, not to this chat.
        sentence = ' '.join((clause, match.group(0), send_clause[:400]))
        if (_PICTURE.search(sentence) and not _other_recipient(sentence)
                and not _ROOM_HERE.search(sentence)):
            return True
    return False


def telegram_send_requested(text):
    if not isinstance(text, str):
        return False
    plain = _without_quotes(text)
    for match in _ACTION.finditer(plain):
        prefix, after = plain[:match.start()], plain[match.end():]
        # Plain 'and' must not discard a preceding report/negation's scope.
        clause = re.split(r'[.!?;\n]|\b(?:but|however|но|зато)\b', prefix, flags=re.I)[-1]
        if (_NEGATED.search(clause) or _DISCUSSION.search(clause)
                or _possibly_literal_caption(text[:match.start()])):
            continue
        direct = _POLITE.sub('', clause, count=1).strip()
        if _REPORT_SUBJECT.search(direct) or _REPORTED_PLAN.search(direct):
            continue
        # The destination must belong to this send, not another sentence or a
        # quoted message. Missing context is insufficient authorization.
        send_clause = re.split(r'[.!?;\n]', after, maxsplit=1)[0]
        target = _TARGET_AFTER.search(send_clause[:400])
        if target is None:
            target = _own_named_target(send_clause[:400])
        if match.group(0).casefold() == 'message':
            target = re.match(r'\s+(?:(?:the|our|this)\s+)?' + _DESTINATION + r'\b', send_clause, re.I)
        if target is None or _REVOCATION.search(after):
            continue
        return True
    return False
