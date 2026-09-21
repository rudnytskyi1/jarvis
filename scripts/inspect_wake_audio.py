"""Compare raw and processed wake input locally; save metrics, never audio."""
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import sounddevice as sd
    import webrtcvad

    from client.audio_processing import AudioPreprocessor
    from client.wakeword import WakeWordDetector
    from common.client_config import load_client_config

    cfg = load_client_config(ROOT / 'config.openai.yaml').client
    model = Path(cfg.wakeword.vosk_model)
    if not model.is_absolute():
        model = ROOT / model
    phrases = cfg.wakeword.phrases or [cfg.wakeword.word]
    detectors = {key: WakeWordDetector(model, phrases) for key in ('raw', 'processed')}
    vads = {key: webrtcvad.Vad(2) for key in detectors}
    metrics = {key: dict(frames=0, samples=0, square_sum=0., peak=0,
                        speech_frames=0, wake_hits=0) for key in detectors}
    processor = AudioPreprocessor(16000, echo_cancellation=cfg.audio.echo_cancellation,
                                 noise_suppression=cfg.audio.noise_suppression,
                                 ns_level=cfg.audio.noise_suppression_level,
                                 output_device=cfg.audio.output_device)
    processor.start()
    try:
        with sd.RawInputStream(device=cfg.audio.input_device, samplerate=16000,
                               channels=1, dtype='int16', blocksize=480) as stream:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                data, overflow = stream.read(480)
                raw = bytes(data)
                processed = processor.process(raw, time.monotonic() - .03)
                for key, pcm in (('raw', raw), ('processed', processed)):
                    samples = np.frombuffer(pcm, dtype=np.int16).astype(float)
                    row = metrics[key]
                    row['frames'] += 1
                    row['samples'] += len(samples)
                    row['square_sum'] += float(samples @ samples)
                    row['peak'] = max(row['peak'], int(np.max(np.abs(samples))))
                    row['speech_frames'] += int(vads[key].is_speech(pcm, 16000))
                    if detectors[key].accept_frame(pcm):
                        row['wake_hits'] += 1
                        detectors[key].reset()
        for row in metrics.values():
            rms = math.sqrt(row.pop('square_sum') / max(1, row['samples']))
            row['rms_dbfs'] = round(20 * math.log10(max(rms, .001) / 32768), 1)
        metrics.update(ts=time.time(), audio_saved=False, input_device=cfg.audio.input_device)
        return metrics
    finally:
        processor.close()


if __name__ == '__main__':
    try:
        result = main()
    except Exception:
        result = dict(error=traceback.format_exc())
    (ROOT / 'data/wake-audio-check.json').write_text(json.dumps(result), encoding='utf-8')
