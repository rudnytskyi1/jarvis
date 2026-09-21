"""Package the matching client side of the reviewed room-request update."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = ['client/main.py', 'client/actions/dispatcher.py', 'client/actions/apps.py',
         'client/actions/app_control.py', 'client/actions/photos.py', 'common/config.py',
         'scripts/update_room_experience_config.py', 'scripts/verify_room_runtime.py',
         'scripts/verify_app_requests.py', 'scripts/check_room_requests_task.ps1']
manifest = {}
with zipfile.ZipFile(root / 'data/room-experience.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        data = (root / name).read_bytes()
        manifest[name] = hashlib.sha256(data).hexdigest()
        archive.writestr(name, data)
    archive.writestr('manifest.json', json.dumps(manifest))
print('Packaged', len(files), 'client files with SHA256 manifest.')
