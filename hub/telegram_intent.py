"""Current group-send requests, excluding reports, quotes and revocations."""
import re

from common.voice_commands import WAKE_ADDRESS_PATTERN
from hub.image_prompt import _possibly_literal_caption

_DESTINATION = r'(?:telegram|tg|group(?:\s+chat)?|телеграм\w*|телег[ауе]|тг|групп\w*|чат\w*)'
_ACTION = re.compile(
    r'\b(?:send|post|share|message(?=\s+(?:(?:the|our|this)\s+)?(?:group|telegram)\b)|'
    r'отправь(?:те)?|отправить|отошли|пришли|скинь(?:те)?|скинуть|'
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
