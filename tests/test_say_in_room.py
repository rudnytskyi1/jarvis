"""tools: ``say_in_room`` makes the room say a phrase out loud.

The owner asked the bot in Telegram to say "TEST HELLO" on his room PC and got
"I have no separate function for speaking text through the anton client": the
protocol message (``say``) and the hub's own proactive speech existed, but no
TOOL was ever exposed, so neither the voice path nor Telegram could ask for it.
"""
import asyncio

from hub.app import Connection


def _connection(spoken=True):
    conn = Connection.__new__(Connection)
    conn.said = []

    async def _say_proactive(text, *, name=''):
        conn.said.append((text, name))
        return spoken

    conn._say_proactive = _say_proactive
    return conn


def test_the_exact_words_are_spoken():
    conn = _connection()
    result = asyncio.run(conn._run_say_in_room({'text': '  TEST   HELLO ', 'person': 'Anton'}))
    assert result['ok'] is True
    assert conn.said == [('TEST HELLO', 'Anton')]


def test_empty_text_is_refused_instead_of_silence():
    conn = _connection()
    result = asyncio.run(conn._run_say_in_room({'text': '   '}))
    assert result['ok'] is False and not conn.said


def test_a_room_without_tts_is_reported_not_pretended():
    conn = _connection(spoken=False)
    result = asyncio.run(conn._run_say_in_room({'text': 'hello'}))
    assert result['ok'] is False
    assert 'no voice' in result['error']
    assert conn.said == [('hello', '')]


def test_a_broken_room_returns_an_error_and_never_raises():
    conn = Connection.__new__(Connection)

    async def _boom(text, *, name=''):
        raise RuntimeError('the socket is gone')

    conn._say_proactive = _boom
    result = asyncio.run(conn._run_say_in_room({'text': 'hello'}))
    assert result['ok'] is False and 'RuntimeError' in result['error']
