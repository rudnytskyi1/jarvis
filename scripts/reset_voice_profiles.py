"""Reset active voice vectors only; run with the brain server stopped."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def checked(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(DATA.resolve()) or resolved == DATA.resolve():
        raise RuntimeError('Reset target must stay inside the project data directory')
    return resolved


def reset():
    registry = checked(DATA / 'people.json')
    original = registry.read_bytes()
    data = json.loads(original.decode('utf-8'))
    people = data['people']
    count = sum(len(person.get('voice_embeddings', [])) for person in people.values())
    protected = {name: {key: value for key, value in person.items()
                        if key not in {'voice_embeddings', 'embeddings'}}
                 for name, person in people.items()}
    preserved_files = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (DATA / 'memory.jsonl', DATA / 'conversations.sqlite3')
                       if path.is_file()}
    backup = checked(DATA / f'voice-reset-backup-{datetime.now():%Y%m%d-%H%M%S-%f}')
    backup.mkdir()
    (backup / 'people.json').write_bytes(original)
    for person in people.values():
        person['voice_embeddings'] = []
        if 'embeddings' in person:
            person['embeddings'] = []
    pending = checked(DATA / 'people.voice-reset-pending.json')
    with pending.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False)
    pending.replace(registry)
    # Retire old enrollment clips and the legacy registry from active locations.
    # Every move target is resolved and checked before the operation.
    moved = []
    try:
        for source in (DATA / 'voices', DATA / 'voices.json'):
            source = checked(source)
            if source.exists():
                destination = checked(backup / source.name)
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
        (DATA / 'voices').mkdir(exist_ok=True)
        actual = json.loads(registry.read_text(encoding='utf-8'))['people']
        assert all(not person.get('voice_embeddings') and not person.get('embeddings')
                   for person in actual.values())
        assert protected == {name: {key: value for key, value in person.items()
                                   if key not in {'voice_embeddings', 'embeddings'}}
                             for name, person in actual.items()}
        for path, before in preserved_files.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before, path.name
    except Exception:
        registry.write_bytes(original)
        for source, destination in reversed(moved):
            if source.is_dir() and not any(source.iterdir()):
                source.rmdir()
            shutil.move(str(checked(destination)), str(checked(source)))
        raise
    print(json.dumps(dict(reset_voice_samples=count, people_preserved=len(people),
                          active_voice_samples=0, backup=str(backup))))


if __name__ == '__main__':
    reset()
