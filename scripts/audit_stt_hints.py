"""Compare local spelling hints on saved enrollment clips; never uploads audio."""
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config
from hub.stt import SttEngine


def main():
    import torch
    torch.set_num_threads(4)
    engine = SttEngine(load_config(ROOT / 'config.openai.yaml').server.stt)
    rows = []
    for directory in sorted((ROOT / 'data' / 'voices').iterdir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('*.wav'))[:2]:
            with wave.open(str(path), 'rb') as audio:
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                    continue
                pcm, rate = audio.readframes(audio.getnframes()), audio.getframerate()
            row = {'person': directory.name, 'file': path.name}
            for hint in ('', 'Rowan'):
                engine.hotwords = hint
                start = time.monotonic()
                result = engine.transcribe_detailed(pcm, rate)
                row[hint or 'baseline'] = {'text': result.text, 'seconds': round(time.monotonic() - start, 2)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    engine.hotwords = 'Rowan'
    silence = engine.transcribe_detailed(b'\0\0' * 16000 * 3, 16000)
    report = {'clips': rows, 'silence_text': silence.text}
    (ROOT / 'data' / 'stt-hints-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'clips_checked': len(rows), 'silence_text': silence.text}), flush=True)


if __name__ == '__main__':
    main()
