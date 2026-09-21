"""Manual, offline synthetic two-voice smoke; no API calls or device actions."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from common.config import load_config
from hub.diarization import DiarizationEngine
from hub.stt import SttEngine, pcm_to_float32, resample
from hub.tts import TtsEngine, float32_to_pcm16


def main():
    cfg = load_config(ROOT / "config.openai.yaml")
    tts = TtsEngine(cfg.server.tts)
    tts.load()
    speech = []
    for speaker, text in [("en_0", "Rowan, please set the volume to thirty percent."),
                          ("en_29", "Hey everyone, are we going to dinner together tonight?")]:
        tts.speaker = speaker
        audio = tts.synth(text)
        assert audio, "TTS failed"
        speech.append(resample(pcm_to_float32(audio), tts.sample_rate, 16000))
    engine = DiarizationEngine(cfg.server.diarization)
    engine.load()
    stt = SttEngine(cfg.server.stt)
    serial = np.concatenate([np.zeros(8000), speech[0], np.zeros(4800), speech[1], np.zeros(8000)])
    mixed = np.zeros(max(len(speech[0]), len(speech[1])) + 16000)
    mixed[8000:8000 + len(speech[0])] += speech[0] * .65
    mixed[8000:8000 + len(speech[1])] += speech[1] * .65
    reports = []
    for label, audio in [("sequential", serial), ("overlap", mixed)]:
        pcm = float32_to_pcm16(audio)
        started = time.perf_counter()
        spans = engine.diarize(pcm, 16000)
        diarization_s = time.perf_counter() - started
        result = engine.recognize(stt, pcm, 16000, "en", None, ["rowan", "roan", "rowen"])
        reports.append(dict(case=label, audio_s=len(audio) / 16000,
                            diarization_s=round(diarization_s, 3), total_s=round(time.perf_counter() - started, 3),
                            clusters=len({s.speaker for s in spans}), selected=result.text, reason=result.reason,
                            segments=result.segments))
    output = ROOT / "data" / "diarization-smoke.json"
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=True, indent=2))
    assert reports[0]["clusters"] == 2, "Two synthetic voices were not separated"
    assert reports[0]["selected"] and "dinner" not in reports[0]["selected"].lower(), "Command was not isolated"
    assert any(s['speaker'] == 'Overlapping voices' for s in reports[1]['segments']), 'Overlap was not detected'
    if cfg.server.diarization.reject_mixed_speech:
        assert reports[1]['reason'] == 'overlapping_speech'
    else:
        assert not reports[1]['reason'] and reports[1]['selected'], 'Relaxed mode rejected overlapping speech'


if __name__ == "__main__":
    main()
