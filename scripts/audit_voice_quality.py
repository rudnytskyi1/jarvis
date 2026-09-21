"""Read-only local enrollment audit. No cloud calls, audio uploads or profile edits."""
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config
from hub.speaker import VoiceRegistry


def main():
    import torch
    torch.set_num_threads(4)
    cfg = load_config(ROOT / 'config.openai.yaml').server.speaker
    registry = VoiceRegistry(ROOT / 'data', threshold=cfg.threshold, margin=cfg.margin, save_audio=False)
    clips = []
    for path in sorted((ROOT / 'data' / 'voices').glob('*/*.wav')):
        if path.parent.name not in registry.people():
            continue
        with wave.open(str(path), 'rb') as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                continue
            pcm, rate = source.readframes(source.getnframes()), source.getframerate()
        samples = np.frombuffer(pcm, '<i2').astype(np.float32) / 32768
        vector = registry._embed(pcm, rate)
        if vector is None:
            continue
        variants = {'full': vector}
        for seconds in (1.5, 2.5):
            size = int(seconds * rate) * 2
            if len(pcm) >= size:
                start = ((len(pcm) - size) // 4) * 2
                variants[str(seconds)] = registry._embed(pcm[start:start + size], rate)
        clips.append({'name': path.parent.name, 'clip': path.name, 'vectors': variants,
                      'seconds': round(len(samples) / rate, 2),
                      'rms_dbfs': round(20 * np.log10(max(1e-8, float(np.sqrt(np.mean(samples**2))))), 1),
                      'clipped_percent': round(float(np.mean(np.abs(samples) >= .999)) * 100, 3)})
    results = []
    for clip in clips:
        rows = {key: clip[key] for key in ('name', 'clip', 'seconds', 'rms_dbfs', 'clipped_percent')}
        rows['matches'] = {}
        for length, vector in clip['vectors'].items():
            board = []
            for name in sorted({item['name'] for item in clips}):
                # Hold out this entire recording, including all its shorter crops.
                enrolled = [item['vectors']['full'].tolist() for item in clips if item is not clip and item['name'] == name]
                centre = registry._centroid(enrolled)
                if centre is not None:
                    board.append((float(np.dot(centre, vector)), name))
            board.sort(reverse=True)
            score, name = board[0]
            gap = score - board[1][0] if len(board) > 1 else 1.
            rows['matches'][length] = {'best': name, 'score': round(score, 3), 'margin': round(gap, 3),
                                       'accepted': score >= cfg.threshold and gap >= cfg.margin}
        results.append(rows)
    summary = {}
    for length in ('full', '1.5', '2.5'):
        values = [(row['name'], row['matches'][length]) for row in results if length in row['matches']]
        summary[length] = {'total': len(values),
                           'correct': sum(match['accepted'] and match['best'] == owner for owner, match in values),
                           'wrong': sum(match['accepted'] and match['best'] != owner for owner, match in values),
                           'unknown': sum(not match['accepted'] for _, match in values)}
    path = ROOT / 'data' / 'voice-quality-audit.json'
    path.write_text(json.dumps({'summary': summary, 'clips': results}, indent=2), encoding='utf-8')
    print(json.dumps({'summary': summary, 'clips': results, 'report': str(path)}))


if __name__ == '__main__':
    main()
