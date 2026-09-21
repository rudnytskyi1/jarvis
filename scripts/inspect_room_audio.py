"""Read-only audio inventory; no recordings, credentials or config writes."""
import json
import sys
import time
import traceback
from pathlib import Path


def report_error(kind, value, tb):
    (Path(__file__).resolve().parents[1] / 'data' / 'microphone-inventory-error.txt').write_text(
        ''.join(traceback.format_exception(kind, value, tb)), encoding='utf-8')


sys.excepthook = report_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sounddevice as sd

from common.config import load_config

cfg = load_config(Path(__file__).resolve().parents[1] / 'config.openai.yaml').client.audio
hosts = sd.query_hostapis()
rows = []
for index, device in enumerate(sd.query_devices()):
    row = dict(index=index, name=device['name'], host=hosts[device['hostapi']]['name'],
               inputs=device['max_input_channels'], outputs=device['max_output_channels'],
               default_rate=device['default_samplerate'])
    if row['inputs']:
        try:
            sd.check_input_settings(device=index, samplerate=16000, channels=1, dtype='int16')
            row['input_16k_mono'] = True
        except Exception as exc:
            row['input_16k_mono'] = str(exc)
    rows.append(row)
details = dict(config_input=cfg.input_device, config_output=cfg.output_device,
               default_devices=list(sd.default.device), devices=rows)
if '--capture' in sys.argv:
    import numpy as np
    from pycaw.pycaw import AudioUtilities
    endpoints = []
    for device in AudioUtilities.GetAllDevices(data_flow=1, device_state=1):
        volume = device.EndpointVolume
        endpoints.append(dict(name=device.FriendlyName, mute=bool(volume.GetMute()),
                              volume=volume.GetMasterVolumeLevelScalar(),
                              channels=[volume.GetChannelVolumeLevelScalar(i)
                                        for i in range(volume.GetChannelCount())]))
    details['capture_endpoints'] = endpoints
    peak = frames = 0
    with sd.RawInputStream(device=cfg.input_device, samplerate=16000, channels=1,
                           dtype='int16', blocksize=480) as stream:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            pcm, overflow = stream.read(480)
            peak = max(peak, int(np.max(np.abs(np.frombuffer(pcm, np.int16).astype(np.int32)))))
            frames += 1
    details['raw_capture'] = dict(peak=peak, frames=frames, audio_saved=False)
    # Native stereo capture distinguishes a source problem from mono resampling/DSP.
    native_devices = [row for row in rows if row['inputs'] and
                      'onn gaming' in row['name'].casefold() and row['host'] == 'Windows WASAPI']
    if len(native_devices) == 1:
        device = native_devices[0]
        rate = int(device['default_rate'])
        levels = np.zeros(2, dtype=np.int32)
        frames = 0
        with sd.RawInputStream(device=device['index'], samplerate=rate, channels=2,
                               dtype='int16', blocksize=rate // 10) as stream:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                pcm, overflow = stream.read(rate // 10)
                samples = np.frombuffer(pcm, np.int16).astype(np.int32).reshape(-1, 2)
                levels = np.maximum(levels, np.abs(samples).max(axis=0))
                frames += 1
        details['native_capture'] = dict(rate=rate, channels=2, peak=levels.tolist(),
                                         frames=frames, audio_saved=False)
result = json.dumps(details, ensure_ascii=True)
if '--output' in sys.argv:
    (Path(__file__).resolve().parents[1] / 'data' / 'microphone-inventory.json').write_text(result, encoding='utf-8')
else:
    print(result)
