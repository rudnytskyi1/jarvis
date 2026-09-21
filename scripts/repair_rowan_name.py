"""Repair the specific, logged 2026-09-18 Anton -> Rowan enrollment mistake.

Run only with the brain server stopped. Creates a complete metadata backup.
"""
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hub.conversations import Conversations
from hub.speaker import VoiceRegistry
from hub.storage import Memory

data = ROOT / 'data'
people = json.loads((data / 'people.json').read_text(encoding='utf-8'))['people']
if 'Rowan' not in people:
    print('No mistaken Rowan profile remains; nothing changed.')
    raise SystemExit(0)
assert people['Rowan']['role'] == 'user' and not people['Rowan'].get('face_embeddings')
assert len(people['Rowan']['voice_embeddings']) == 3 and 'Anton' in people
logs = [json.loads(line) for line in (data / 'dialogs/2026-09-18.jsonl').read_text(encoding='utf-8').splitlines()]
assert any(r.get('ts') == '2026-09-18T17:40:22' and r.get('speaker') == 'Anton'
           and r.get('transcript') == 'My name is Rowan. My name is Anton.' for r in logs)
assert any(r.get('ts') == '2026-09-18T17:42:00' and r.get('speaker') == 'Anton'
           and 'Rowan, your voice samples are saved.' in r.get('reply', '') for r in logs)
backup = data / ('name-repair-backup-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
backup.mkdir()
for name in ('people.json', 'memory.jsonl'):
    if (data / name).exists():
        shutil.copy2(data / name, backup / name)
with sqlite3.connect(data / 'conversations.sqlite3') as source:
    with sqlite3.connect(backup / 'conversations.sqlite3') as target:
        source.backup(target)
registry = VoiceRegistry(data)
registry.rename_person('Rowan', 'Anton')
Memory(data).rename('Rowan', 'Anton')
Conversations(data).rename('Rowan', 'Anton')
assert 'Rowan' not in registry.people()
assert registry.role_of('Anton') == people['Anton']['role']
assert registry._people['Anton']['face_embeddings'] == people['Anton']['face_embeddings']
print(f'Repaired the mistaken Rowan profile into Anton. Backup: {backup}')
print('Anton voice samples:', len(registry._people['Anton']['voice_embeddings']))
