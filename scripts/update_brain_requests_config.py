"""Apply the reviewed room-request settings, keeping secrets and devices intact."""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from common.config import Config

path = root / 'config.openai.yaml'
data = yaml.safe_load(path.read_text(encoding='utf-8'))
data['server']['llm'].update(history_turns=25, max_input_bytes=128000)
data['server']['speaker'].update(threshold=.40, margin=.15, admin_threshold=.65)
data['server']['tts'].update(engine='kokoro', language='en', speaker='am_michael')
Config.model_validate(data)
backup = root / 'data' / ('brain-config-before-requests-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.yaml')
shutil.copy2(path, backup)
temporary = path.with_suffix('.yaml.tmp')
temporary.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
temporary.replace(path)
print('Applied: 25-turn history, stricter speaker matching, local American male Kokoro voice. Config backed up.')
