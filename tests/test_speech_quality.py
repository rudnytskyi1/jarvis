import json

import numpy as np
import pytest
import yaml

from common.config import Config
from hub.audio_quality import pcm_stats
from scripts.update_room_speech_config import update


def test_audio_diagnostics_detect_silence_and_clipping_without_int16_overflow():
    silence = pcm_stats(b'\0\0' * 16000, 16000)
    assert silence['seconds'] == 1 and silence['rms_dbfs'] == -160
    loud = pcm_stats(np.array([-32768, 32767, 0, 0], dtype='<i2').tobytes(), 16000)
    assert loud['clipped_percent'] == 50 and loud['peak_dbfs'] == 0
    assert loud['rms_dbfs'] == pytest.approx(-3.0)
    json.dumps(silence, allow_nan=False)


def test_room_config_update_is_scoped_backed_up_and_idempotent(tmp_path):
    data = Config().model_dump()
    data['client']['audio']['input_device'] = 'onn Gaming USB Microphone, MME'
    data['client']['vad'].update(aggressiveness=3, silence_ms=700)
    data['server']['permissions_enabled'] = False
    data['server']['speaker'].update(threshold=.23, admin_threshold=.23)
    path = tmp_path / 'config.openai.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    original = path.read_bytes()
    result = update(path)
    from pathlib import Path
    assert Path(result['backup']).read_bytes() == original
    changed = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert changed['server'] == data['server']
    assert changed['client']['audio'] == data['client']['audio']
    assert changed['client']['vad']['silence_ms'] == 1100
    assert changed['client']['vad']['aggressiveness'] == 2
    again = update(path)
    assert again['before'] == again['after'] and 'backup' not in again


def test_room_config_dry_run_writes_nothing_and_preserves_longer_pauses(tmp_path):
    data = Config().model_dump()
    data['client']['vad'].update(aggressiveness=1, silence_ms=1800, max_utterance_s=40)
    path = tmp_path / 'config.openai.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    before = path.read_bytes()
    result = update(path, dry_run=True)
    assert result['after']['aggressiveness'] == 1 and result['after']['silence_ms'] == 1800
    assert result['after']['max_utterance_s'] == 40
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]
