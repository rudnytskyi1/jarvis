"""Package the offline room-PC update without configs, keys or recordings."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['client/main.py', 'client/audio_processing.py', 'client/overlay_web/chat.html', 'client/actions/pc.py', 'client/actions/app_control.py',
         'client/actions/dispatcher.py', 'client/actions/photos.py', 'client/actions/wallpaper.py',
         'client/camera.py', 'client/frame_recording.py', 'common/__init__.py', 'common/config.py', 'common/client_config.py',
         'common/voice_commands.py',
         'common/recording.py', 'common/openai_models.py', 'common/image_models.py',
         'scripts/configure_recording.py', 'scripts/update_room_speech_config.py',
         'docs/SPEECH_QUALITY_UPDATE.md']


if __name__ == '__main__':
    manifest = {}
    target = ROOT / 'data/speech-recording-update.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            payload = (ROOT / name).read_bytes()
            manifest[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
        archive.writestr('manifest.json', json.dumps(manifest))
    print(f'Packaged {len(FILES)} files: {target}')
