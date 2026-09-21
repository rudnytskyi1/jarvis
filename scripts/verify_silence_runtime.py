"""Replay synthetic local test clips through real Vosk; no mic, TTS or API calls."""
import json
import sys
import wave
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from client.voice_controls import SilenceDetector
from client.wakeword import WakeWordDetector

wake = WakeWordDetector(root / 'models/vosk-model-small-en-us-0.15', ['rowan'])
detector = SilenceDetector(wake, root / 'models/vosk-model-small-ru-0.22')
processor = None
if '--denoise' in sys.argv:
    import numpy as np
    from pywebrtc_audio import AudioProcessor
    processor = AudioProcessor(sample_rate=16000, echo_cancellation=True,
                               noise_suppression=True, ns_level=1)
assert len(detector._recognizers) == 2, 'Russian silence model is missing'
result = []
cases = json.loads((root / 'data/silence-smoke/cases.json').read_text(encoding='utf-8-sig'))
for case in cases:
    with wave.open(str(root / 'data/silence-smoke' / case['file']), 'rb') as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        pcm = wav.readframes(wav.getnframes()) + b'\0' * 64000
    detector.reset()
    if processor:
        processor.reset()
    hit = False
    for index in range(0, len(pcm), 960):
        frame = pcm[index:index + 960]
        if processor:
            near = np.frombuffer(frame, np.int16)
            frame = processor.process(near, np.zeros(len(near), np.int16)).tobytes()
        if detector.accept_frame(frame):
            hit = True
            break
    result.append(dict(phrase=case['text'], expected=case['stop'], detected=hit))
print(json.dumps(result, ensure_ascii=False))
(root / 'data/silence-smoke/results.json').write_text(json.dumps(result), encoding='utf-8')
if any(row['expected'] != row['detected'] for row in result):
    sys.exit(1)
