"""Live protocol check: one saved false wake and one synthetic ready request.

The ready request uses the configured text API and shared budget. No camera,
image generation or computer actions are requested or executed.
"""
import asyncio
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import websockets

from common.config import load_config


async def exchange(pcm, url):
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({'type': 'hello', 'client_id': 'diagnostic-room-reliability', 'devices': []}))
        assert json.loads(await ws.recv())['type'] == 'ready'
        await ws.send(json.dumps({'type': 'utterance_start', 'sr': 16000,
                                 'format': 'pcm_s16le', 'channels': 1, 'verify_wake': True}))
        for start in range(0, len(pcm), 960):
            await ws.send(pcm[start:start + 960])
        await ws.send(json.dumps({'type': 'utterance_end'}))
        messages, audio_bytes = [], 0
        async with asyncio.timeout(45):
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    audio_bytes += len(raw)
                    continue
                row = json.loads(raw)
                assert row['type'] not in {'error', 'actions', 'camera_request', 'screenshot_request'}, row
                messages.append(row)
                if row['type'] == 'tts_end':
                    break
        return messages, audio_bytes


async def main():
    cfg = load_config(ROOT / 'config.openai.yaml')
    url = f'ws://127.0.0.1:{cfg.server.port}/ws'
    turns = [json.loads(line) for line in (ROOT / 'data/dialogs/2026-09-20.jsonl').read_text(encoding='utf-8').splitlines()]
    turn = next(row for row in turns if row.get('transcript') == 'Oh' and row.get('audio_recording'))
    recorded = (ROOT / 'data/request_audio' / turn['audio_recording']['path']).resolve()
    assert recorded.is_relative_to((ROOT / 'data/request_audio').resolve())
    with wave.open(str(recorded), 'rb') as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1
        pcm = source.readframes(source.getnframes())
    messages, count = await exchange(pcm, url)
    assert any(row.get('ignored') for row in messages), messages
    assert count == 0 and not any(row['type'] == 'say' for row in messages)
    print('PASS: saved false wake completes silently, without a spoken reply.', flush=True)

    from hub.tts import TtsEngine
    voice_cfg = cfg.server.tts.model_copy(update={'sample_rate': 16000})
    voice = TtsEngine(voice_cfg)
    assert await asyncio.to_thread(voice.load), 'Local test voice could not load'
    pcm = await asyncio.to_thread(voice.synth, 'Rowan, reply with just the word ready.')
    assert pcm
    messages, count = await exchange(pcm, url)
    transcript = next(row['text'] for row in messages if row['type'] == 'transcript')
    reply = next(row['text'] for row in messages if row['type'] == 'say')
    assert 'ready' in reply.casefold() and count > 1000, (messages, count)
    report = {'false_wake_silent': True, 'synthetic_request': transcript,
              'reply': reply, 'tts_bytes': count, 'physical_microphone_test': False}
    (ROOT / 'data/room-reliability-live-check.json').write_text(json.dumps(report), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
