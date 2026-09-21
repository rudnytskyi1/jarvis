"""Download the local Kokoro v1 model; no paid speech API is involved."""
import hashlib
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'models' / 'kokoro'
root.mkdir(parents=True, exist_ok=True)
for name in ('kokoro-v1.0.onnx', 'voices-v1.0.bin'):
    path = root / name
    if not path.is_file():
        temporary = path.with_suffix('.part')
        urllib.request.urlretrieve('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/' + name + '?download=' + str(int(time.time())), temporary)
        temporary.replace(path)
    print(name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
