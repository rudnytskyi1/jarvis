"""Live local STT/diarization/registration check; never supplies training samples."""
import asyncio
import hashlib
import json
import sys
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
    async def audio(text):
        raw = await asyncio.to_thread(tts.synth, text)
        return float32_to_pcm16(resample(pcm_to_float32(raw), tts.sample_rate, 16000))
    clips = [await audio('Rowan. Remember my voice.') + b'\0\0' * 22400 +
             await audio('I need a better mic.'),
             await audio('Rowan, my name is Verification Sample.'),
             await audio('Rowan, cancel registration.')]
    original = hashlib.sha256((ROOT / 'data/people.json').read_bytes()).hexdigest()
    results = []
    async with websockets.connect(f'ws://127.0.0.1:{cfg.server.port}/ws', max_size=None) as ws:
        await ws.send(json.dumps({'type': p.MSG_HELLO, 'client_id': 'voice-registration-smoke', 'devices': []}))
        assert json.loads(await asyncio.wait_for(ws.recv(), 10))['type'] == p.MSG_READY
        for pcm in clips:
            await ws.send(json.dumps({'type': p.MSG_UTTERANCE_START, 'sr': 16000, 'format': p.AUDIO_FORMAT, 'channels': 1}))
            await ws.send(pcm)
            await ws.send(json.dumps({'type': p.MSG_UTTERANCE_END}))
            result = {}
            async with asyncio.timeout(60):
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    kind = msg['type']
                    if kind == p.MSG_TRANSCRIPT:
                        result['transcript'] = msg.get('text')
                        result['clarification'] = msg.get('clarification')
                    elif kind == p.MSG_SAY:
                        result['reply'] = msg.get('text')
                        result['caption'] = msg.get('enrollment_sentence')
                    elif kind in (p.MSG_ACTIONS, p.MSG_VOICE_CONFIRMATION, p.MSG_ERROR):
                        raise AssertionError(f'Unexpected operation in registration start/cancel: {kind}')
                    elif kind == p.MSG_TTS_END:
                        break
            results.append(result)
            print(json.dumps(result), flush=True)
    assert all(not r['clarification'] for r in results)
    assert 'name' in results[0]['reply'].lower() or results[0]['caption']
    assert results[1]['caption'] and 'verification' in results[1]['reply'].lower()
    assert 'cancelled' in results[2]['reply'].lower()
    assert hashlib.sha256((ROOT / 'data/people.json').read_bytes()).hexdigest() == original
    print('PASS: live registration starts across a pause, takes a name, cancels; profiles unchanged.')


if __name__ == '__main__':
    asyncio.run(main())
