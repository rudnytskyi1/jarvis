"""Room-PC smoke check; never calls OpenAI or changes people's profiles."""
import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)

if '--ui' in sys.argv:
    import logging
    logging.basicConfig(filename=root / 'data' / 'overlay-check.log', level=logging.INFO)
    from client.overlay import OverlayHUD
    from client.screen import capture_jpeg
    hud = OverlayHUD({'enabled': True})
    hud.start()
    time.sleep(3)
    hud.set_state('thinking')
    hud.chat({'person': 'Preview', 'messages': [], 'question': 'Checking the new Rowan interface'})
    hud.chat_reply('The chat is ready. This is a local display test.')
    time.sleep(3)
    hidden = hud.suspend_capture()
    hud.set_status('This update must not reveal the window during capture')
    time.sleep(.3)
    capture = capture_jpeg() if hidden else None
    if capture:
        (root / 'data' / 'overlay-hidden-check.jpg').write_bytes(capture.jpeg)
    result = dict(enabled=hud.enabled, hide_ack=hidden, stayed_hidden=not hud._mapped, captured=bool(capture))
    hud.resume_capture()
    time.sleep(.4)
    result['restored'] = hud._mapped
    (root / 'data' / 'overlay-check.json').write_text(json.dumps(result), encoding='utf-8')
    hud.stop()
    # PySide owns native resources on its UI thread; avoid interpreter teardown
    # racing those resources in this disposable diagnostic subprocess.
    os._exit(0 if all(result.values()) else 1)
else:
    import importlib.metadata as md

    import numpy as np
    import torch
    from ultralytics import YOLO

    from common.config import load_config
    cfg = load_config(root / 'config.openai.yaml')
    from client.camera import CameraService
    camera = CameraService(cfg.client.camera)
    model = YOLO(camera.model_name)
    started = time.perf_counter()
    for _ in range(2):
        camera._detect(model, np.zeros((1080, 1920, 3), dtype=np.uint8))
    print(json.dumps({'ultralytics': md.version('ultralytics'), 'qt': md.version('PySide6'),
                      'cuda': torch.cuda.is_available(), 'tracking_initialized': True,
                      'seconds': round(time.perf_counter() - started, 2)}))
