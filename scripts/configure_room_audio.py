"""Verify capture in the desktop session, then save explicit audio endpoints."""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import sounddevice as sd
import yaml

from client.audio import AudioInput
from client.audio_processing import AudioPreprocessor
from common.config import Config


def configure(args):
    path = ROOT / 'config.openai.yaml'
    original = path.read_bytes()
    data = yaml.safe_load(original.decode('utf-8-sig'))
    settings = data.setdefault('client', {}).setdefault('audio', {})
    settings.update(input_device=args.input_device, output_device=args.output_device)
    cfg = Config.model_validate(data).client.audio
    microphone = sd.query_devices(cfg.input_device, 'input')
    speakers = sd.query_devices(cfg.output_device, 'output')
    sd.check_input_settings(device=cfg.input_device, samplerate=16000, channels=1, dtype='int16')
    sd.check_output_settings(device=cfg.output_device, samplerate=16000, channels=1, dtype='int16')
    processor = AudioPreprocessor(16000, echo_cancellation=cfg.echo_cancellation,
                                 noise_suppression=cfg.noise_suppression,
                                 ns_level=cfg.noise_suppression_level, output_device=cfg.output_device)
    capture = AudioInput(device=cfg.input_device, processor=processor)
    frames = peak = 0
    try:
        capture.start()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            pcm = capture._get_blocking(.1)
            if pcm:
                if len(pcm) != 960:
                    raise RuntimeError('Unexpected microphone frame length')
                frames += 1
                peak = max(peak, int(np.max(np.abs(np.frombuffer(pcm, np.int16).astype(np.int32)))))
        if frames < 80:
            raise RuntimeError(f'Only {frames} microphone frames in four seconds')
        if processor.processor is None:
            raise RuntimeError('Audio processing did not start')
        if cfg.echo_cancellation and processor.reference is None:
            raise RuntimeError('Speaker reference did not start')
        aec = processor.reference is not None
    finally:
        capture.close()
    # Verify the output can actually open without playing any sound.
    with sd.RawOutputStream(device=cfg.output_device, samplerate=16000,
                            channels=1, dtype='int16') as output:
        output.write(bytes(3200))
    if path.read_bytes() != original:
        raise RuntimeError('Configuration changed during verification; refusing to overwrite')
    backup = ROOT / 'data' / f'audio-config-backup-{datetime.now():%Y%m%d-%H%M%S-%f}.yaml'
    with backup.open('xb') as handle:
        handle.write(original)
    pending = path.with_suffix('.audio-pending.yaml')
    with pending.open('x', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    pending.replace(path)
    return dict(ok=True, microphone=microphone['name'], speakers=speakers['name'],
                input_device=cfg.input_device, output_device=cfg.output_device,
                frames=frames, peak=peak, aec_reference=aec, audio_saved=False,
                backup=str(backup))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-device', required=True)
    parser.add_argument('--output-device', required=True)
    args = parser.parse_args()
    try:
        result = configure(args)
    except Exception as exc:
        result = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
    (ROOT / 'data' / 'microphone-setup.json').write_text(json.dumps(result), encoding='utf-8')
    sys.exit(0 if result['ok'] else 1)
