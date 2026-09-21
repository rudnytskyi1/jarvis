"""Check DSP and capture health; never save audio or send it to a server."""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from client.audio import AudioInput
from client.audio_processing import AudioPreprocessor
from common.config import load_config

logging.basicConfig(level=logging.INFO)
cfg = load_config(Path(__file__).resolve().parents[1] / 'config.openai.yaml').client.audio
processor = AudioPreprocessor(16000, echo_cancellation=True, noise_suppression=True,
                             ns_level=1, output_device=cfg.output_device)
capture = AudioInput(device=cfg.input_device, processor=processor)
try:
    capture.start()
    deadline = time.monotonic() + 4
    frames = 0
    peak = 0
    while time.monotonic() < deadline:
        pcm = capture._get_blocking(.1)
        if pcm:
            assert len(pcm) == 960
            frames += 1
            peak = max(peak, int(np.max(np.abs(np.frombuffer(pcm, np.int16).astype(np.int32)))))
    assert frames > 50, f'Only {frames} microphone frames'
    assert processor.processor is not None
    assert processor.reference is not None, 'WASAPI reference did not start'
    print(json.dumps({'ok': True, 'frames': frames, 'peak': peak,
                      'aec_reference': True, 'audio_saved': False,
                      'dsp_ms_per_frame': round(processor.processing_ms / max(1, processor.frames), 3)}))
finally:
    capture.close()
