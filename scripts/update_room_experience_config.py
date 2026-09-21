"""Update only camera cadence; preserve this PC's devices and server address."""
import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from common.config import Config

path = root / 'config.openai.yaml'
data = yaml.safe_load(path.read_text(encoding='utf-8'))
data['client']['camera'].update(fps=8, face_check_interval_s=.5)
data['client']['audio'].update(echo_cancellation=True, noise_suppression=True,
                               noise_suppression_level=1)
data['server']['llm'].update(history_turns=25, max_input_bytes=128000)
data['server']['speaker'].update(threshold=.40, margin=.15, admin_threshold=.65)
data['server']['tts'].update(engine='kokoro', language='en', speaker='am_michael')
Config.model_validate(data)
path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
print('Room camera cadence and local audio processing enabled; device selections preserved.')
