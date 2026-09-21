"""Read-only recognition check against recorded enrollment clips; no API calls."""
import json
import sys
import wave
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from common.config import load_config
from hub.speaker import VoiceRegistry

cfg = load_config(root / 'config.openai.yaml').server.speaker
registry = VoiceRegistry(data_dir=root / 'data', threshold=cfg.threshold, margin=cfg.margin)
rows = []
denoise = '--denoise' in sys.argv
if denoise:
    import numpy as np
    from pywebrtc_audio import AudioProcessor
for path in sorted((root / 'data' / 'voices' / 'unknown').glob('*.wav')):
    with wave.open(str(path), 'rb') as f:
        pcm, sr = f.readframes(f.getnframes()), f.getframerate()
    if denoise:
        ap = AudioProcessor(sample_rate=sr, echo_cancellation=True,
                            noise_suppression=True, ns_level=1)
        near = np.frombuffer(pcm, np.int16)
        pcm = ap.process(near, np.zeros(len(near), np.int16)).tobytes()
    name, role, score = registry.identify(pcm, sr)
    rows.append(dict(clip=path.name, identified=name, score=round(score, 3)))
(root / 'data' / ('voice-denoise-check.json' if denoise else 'voice-repair-check.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(json.dumps(rows))
