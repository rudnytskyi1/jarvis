import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from common import protocol
from common.config import Config
from hub import app
from hub.diarization import AttributedUtterance, Word
from hub.live_transcript import LiveTranscript


def result(text='hello', name='Anton', words=None, reason=''):
    return AttributedUtterance(text=text, name=name, role='admin', score=.8,
                              words=words or [Word(.1, .9, text)], reason=reason)


def test_drafts_replace_words_and_keep_completed_window():
    async def run():
        send = AsyncMock()
        recognize = AsyncMock(side_effect=[result('hello'), result('hello there'), result('again')])
        live = LiveTranscript('one', recognize, send, window=2)
        for seconds in (2, 2, 4):
            live.next_at = 0
            live.feed(b'\0\1' * 16000 * seconds)
            await live.task
        assert [c.args[0]['text'] for c in send.call_args_list] == ['hello', 'hello there', 'hello there again']
        assert all(len(c.args[0]) <= 16000 * 4 for c in recognize.call_args_list)
    asyncio.run(run())


def test_stop_discards_inflight_draft_and_does_not_queue_audio():
    async def run():
        complete = asyncio.Event()
        async def recognize(pcm):
            await complete.wait()
            return result()
        send = AsyncMock()
        live = LiveTranscript('one', recognize, send)
        live.feed(b'\0\1' * 32000)
        task = live.task
        live.next_at = 0
        live.feed(b'\0\1' * 48000)
        assert live.task is task
        live.stop()
        complete.set()
        await task
        send.assert_not_awaited()
    asyncio.run(run())


def test_overlapping_preview_does_not_display_identity_or_command():
    async def run():
        send = AsyncMock()
        live = LiveTranscript('one', AsyncMock(return_value=result('delete everything', reason='overlapping_speech')), send)
        live.feed(b'\0\1' * 32000)
        await live.task
        payload = send.call_args.args[0]
        assert payload['uncertain'] and not payload['person'] and not payload['text']
    asyncio.run(run())


def test_preview_never_sets_authenticated_identity(monkeypatch):
    async def run():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.cfg.server.diarization.enabled = True
        conn._can_live_transcribe = True
        conn._recognize_diarized = AsyncMock(return_value=result())
        conn.send_json = AsyncMock()
        conn._on_utterance_start({'sr': 16000, 'utterance_id': 'active'})
        conn._on_binary(b'\0\1' * 32000)
        await conn._live_preview.task
        assert conn._speaker_name == conn._speaker_role == 'unknown'
        assert conn._speaker_score == 0 and not conn._current_pcm
        assert conn.send_json.call_args.args[0]['person'] == 'Anton'
        conn._live_preview.stop()
    asyncio.run(run())


def test_final_recognition_waits_for_native_preview_even_when_caller_cancelled(monkeypatch):
    async def run():
        entered, release = threading.Event(), threading.Event()
        calls = []
        def recognize(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 1:
                entered.set()
                assert release.wait(3)
            return result()
        monkeypatch.setattr(app, '_diarizer', SimpleNamespace(recognize=recognize))
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_preview=Mock()))
        conn = app.Connection(SimpleNamespace(client=None), Config())
        preview = asyncio.create_task(conn._recognize_diarized(b'preview', 16000, preview=True))
        await asyncio.to_thread(entered.wait, 2)
        preview.cancel()
        await asyncio.gather(preview, return_exceptions=True)
        assert await conn._recognize_diarized(b'skipped', 16000, preview=True) is None
        final = asyncio.create_task(conn._recognize_diarized(b'final', 16000))
        await asyncio.sleep(.01)
        assert calls == [b'preview'] and not final.done()
        release.set()
        await final
        assert calls == [b'preview', b'final']
    asyncio.run(run())


def test_client_displays_partial_during_recording_and_ignores_stale_turns():
    from tests.test_silence import make_client
    async def run():
        client = make_client()
        client._recording_live = True
        client._live_turn_id = 'active'
        partial = {'type': protocol.MSG_TRANSCRIPT_PARTIAL, 'utterance_id': 'active', 'text': 'hello'}
        await client._route_message(partial)
        client.overlay.transcript.assert_called_once_with(partial)
        assert client._inbox.empty()
        await client._route_message({**partial, 'utterance_id': 'old'})
        client._recording_live = False
        await client._route_message(partial)
        assert client.overlay.transcript.call_count == 1
    asyncio.run(run())
