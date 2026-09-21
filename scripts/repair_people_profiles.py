"""Requested repair: remove Diodrek and quarantine invalid placeholder profiles.

Run with the brain stopped so an in-memory registry cannot overwrite the edit.
Preserves the existing Theodric profile; never guesses ownership of bad samples.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'data' / 'people.json'
data = json.loads(path.read_text(encoding='utf-8'))
stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
shutil.copy2(path, path.with_name('people.json.before-room-repair-' + stamp))
removed = {}
for name in list(data['people']):
    if name.casefold() in {'diodrek', 'unknown', 'guest', 'user', 'friend'}:
        removed[name] = data['people'].pop(name)
if removed:
    quarantine = root / 'data' / ('quarantined-profiles-' + stamp + '.json')
    quarantine.write_text(json.dumps(removed, ensure_ascii=False), encoding='utf-8')
    temporary = path.with_suffix('.repair.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
print('Removed from active recognition:', ', '.join(removed))
print('Preserved profiles:', ', '.join(data['people']))
