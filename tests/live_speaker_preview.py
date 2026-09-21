"""Replay a consented enrollment clip for display only; never end/execute a turn."""
import argparse
import asyncio
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import websockets

from common.config import load_config


async def main(person='Anton'):
    candidates = []
    folder = (ROOT / 'data/voices' / person).resolve()
    if folder.parent != (ROOT / 'data/voices').resolve():
        raise ValueError('Expected one enrolled profile name')
    for path in folder.glob('*.wav'):
        with wave.open(str(path), 'rb') as f:
            assert f.getframerate() == 16000 and f.getnchannels() == 1
            candidates.append(f.readframes(f.getnframes()))
    if not candidates:
        raise RuntimeError(f'No saved enrollment audio for {person}')
    pcm = max(candidates, key=len)
    cfg = load_config(ROOT / 'config.openai.yaml')
    before = hashlib.sha256((ROOT / 'data/people.json').read_bytes()).hexdigest()
    started, identified = time.perf_counter(), []
    async with websockets.connect(f'ws://127.0.0.1:{cfg.server.port}/ws', max_size=None) as ws:
        await ws.send(json.dumps({'type': 'hello', 'client_id': 'speaker-preview-smoke',
                                 'devices': [], 'capabilities': ['live_transcript']}))
        assert json.loads(await ws.recv())['type'] == 'ready'
        await ws.send(json.dumps({'type': 'utterance_start', 'sr': 16000, 'utterance_id': 'speaker-preview'}))
        async def listen():
            async for raw in ws:
                assert not isinstance(raw, bytes), 'Preview must not speak or execute a request'
                msg = json.loads(raw)
                assert msg['type'] == 'transcript_partial', msg['type']
                row = dict(seconds=round(time.perf_counter()-started, 2), person=msg['person'], score=msg['score'])
                identified.append(row)
                print(json.dumps(row), flush=True)
        listener = asyncio.create_task(listen())
        for offset in range(0, len(pcm), 960):
            await ws.send(pcm[offset:offset+960])
            await asyncio.sleep(.03)
        await asyncio.sleep(1)
        await ws.close()  # No utterance_end: final pipeline/LLM/tools never run.
        await listener
    assert any(r['person'] == person for r in identified), identified
    assert identified[-1]['person'] == person, 'Identity was lost as more speech arrived'
    assert hashlib.sha256((ROOT / 'data/people.json').read_bytes()).hexdigest() == before
    print(f'PASS: {person} recognized during streaming; no executed turn or profile writes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--person', default='Anton')
    asyncio.run(main(parser.parse_args().person))
