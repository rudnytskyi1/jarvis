"""Read-only local debugging of speaker attribution on saved enrollment audio."""
import argparse
import json
import sys
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config
from hub.diarization import DiarizationEngine
from hub.speaker import VoiceRegistry
from hub.stt import SttEngine


def main(person, seconds):
    import torch
    torch.set_num_threads(4)
    cfg = load_config(ROOT / 'config.openai.yaml')
    folder = (ROOT / 'data/voices' / person).resolve()
    if folder.parent != (ROOT / 'data/voices').resolve():
        raise ValueError('Expected an enrolled profile name')
    clips = []
    for path in folder.glob('*.wav'):
        with wave.open(str(path), 'rb') as source:
            if source.getframerate() == 16000 and source.getnchannels() == 1:
                clips.append(source.readframes(source.getnframes()))
    pcm = max(clips, key=len)[:int(seconds * 16000) * 2]
    engine = DiarizationEngine(cfg.server.diarization)
    engine.load()
    spans = engine.diarize(pcm, 16000)
    engine.diarize = lambda *args: spans
    stt = SttEngine(cfg.server.stt)
    voices = VoiceRegistry(ROOT / 'data', threshold=cfg.server.speaker.threshold,
                           margin=cfg.server.speaker.margin, save_audio=False)
    transcripts = []
    def decode(*args):
        transcript = stt.transcribe_preview(*args)
        transcripts.append(asdict(transcript))
        return transcript
    result = engine.recognize(SimpleNamespace(transcribe_detailed=decode), pcm, 16000,
                              cfg.server.stt.language, voices, ['rowan', 'roan', 'rowen'], True, allow_pauses=True)
    print(json.dumps({'person': result.name, 'score': result.score, 'reason': result.reason,
                      'note': result.attribution_note, 'segments': result.segments,
                      'spans': [asdict(span) for span in spans], 'transcripts': transcripts}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--person', default='Theodric')
    parser.add_argument('--seconds', type=float, default=4.68)
    args = parser.parse_args()
    main(args.person, args.seconds)
