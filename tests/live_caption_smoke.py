"""Send speech in real time; assert drafts arrive BEFORE utterance_end."""
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import websockets

from common import protocol as p
from common.config import load_config
from hub.stt import pcm_to_float32, resample
from hub.tts import TtsEngine, float32_to_pcm16


async def main():
    cfg = load_config(ROOT / 'config.openai.yaml')
    tts = TtsEngine(cfg.server.tts)
    await asyncio.to_thread(tts.load)
    raw = await asyncio.to_thread(tts.synth,
        'Rowan, remember my voice. Before we start, I want to make sure you can show these words on the screen while I am speaking to you.')
    pcm = float32_to_pcm16(resample(pcm_to_float32(raw), tts.sample_rate, 16000))
    before = hashlib.sha256((ROOT / 'data/people.json').read_bytes()).hexdigest()
    drafts, final, started, ended = [], {}, time.perf_counter(), asyncio.Event()
    async with websockets.connect(f'ws://127.0.0.1:{cfg.server.port}/ws', max_size=None) as ws:
        await ws.send(json.dumps({'type': p.MSG_HELLO, 'client_id': 'live-caption-smoke', 'devices': [],
                                  'capabilities': ['live_transcript']}))
        assert json.loads(await asyncio.wait_for(ws.recv(), 10))['type'] == p.MSG_READY
        async def send():
            await ws.send(json.dumps({'type': p.MSG_UTTERANCE_START, 'sr': 16000, 'utterance_id': 'smoke-caption'}))
            for offset in range(0, len(pcm), 960):
                await ws.send(pcm[offset:offset+960])
                await asyncio.sleep(.03)
            await ws.send(json.dumps({'type': p.MSG_UTTERANCE_END}))
            ended.set()
        sender = asyncio.create_task(send())
        async with asyncio.timeout(60):
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg['type'] == p.MSG_TRANSCRIPT_PARTIAL:
                    row = dict(seconds=round(time.perf_counter()-started, 2), before_end=not ended.is_set(),
                               text=msg['text'], person=msg.get('person'), uncertain=msg.get('uncertain'))
                    drafts.append(row)
                    print(json.dumps(row), flush=True)
                elif msg['type'] == p.MSG_TRANSCRIPT:
                    final['transcript'] = msg['text']
                elif msg['type'] == p.MSG_SAY:
                    final['reply'] = msg['text']
                elif msg['type'] in (p.MSG_ACTIONS, p.MSG_ERROR, p.MSG_VOICE_CONFIRMATION):
                    raise AssertionError(msg['type'])
                elif msg['type'] == p.MSG_TTS_END:
                    break
        await sender
    assert len([r for r in drafts if r['before_end'] and r['text']]) >= 2
    assert 'name' in final.get('reply', '').lower()
    assert hashlib.sha256((ROOT / 'data/people.json').read_bytes()).hexdigest() == before
    print(json.dumps(final))
    print('PASS: live text before end of speech; profiles unchanged.')


if __name__ == '__main__':
    asyncio.run(main())
