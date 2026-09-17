"""Live end-to-end test against a running Jarvis server on localhost:8765.

Not collected by pytest (filename on purpose): run it directly.
Emulates the room client: synthesizes spoken phrases with Silero, streams them
as mic audio, answers ``actions`` and ``screenshot_request`` like the real
client would, and asserts on the replies.
"""
import asyncio
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import websockets
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

URL = "ws://127.0.0.1:8765/ws"

_tts_model = None


def synth_16k(text: str) -> bytes:
    global _tts_model
    if _tts_model is None:
        _tts_model, _ = torch.hub.load(
            "snakers4/silero-models", "silero_tts", language="en", speaker="v3_en"
        )
    audio = _tts_model.apply_tts(text=text, speaker="en_0", sample_rate=48000)
    # Proper anti-aliased resampling: naive [::3] decimation garbles Whisper.
    import torchaudio.functional as F

    pcm16k = F.resample(audio.unsqueeze(0), 48000, 16000).squeeze(0).numpy()
    return (np.clip(pcm16k, -1, 1) * 32767).astype(np.int16).tobytes()


def fake_screenshot() -> tuple[bytes, int, int]:
    img = Image.new("RGB", (1024, 576), (25, 25, 35))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1024, 60], fill=(50, 50, 70))
    d.text((30, 20), "BLUE WHALE DOCUMENTARY - YouTube - Google Chrome", fill=(240, 240, 240))
    d.text((60, 150), "Blue Whale Documentary  |  Ocean Films  |  12M views", fill=(220, 220, 220))
    d.text((60, 250), "Deep Sea Mysteries  |  Nature HD  |  3.4M views", fill=(220, 220, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue(), 1024, 576


class FakeClient:
    def __init__(self, ws):
        self.ws = ws
        self.screenshot_requests = 0
        self.executed_actions = []

    async def utterance(self, pcm: bytes, timeout: float = 240.0) -> dict:
        """Send one utterance; drive the protocol until say/tts_end/error."""
        await self.ws.send(json.dumps({"type": "utterance_start", "sr": 16000,
                                       "format": "pcm_s16le", "channels": 1}))
        for i in range(0, len(pcm), 960):
            await self.ws.send(pcm[i:i + 960])
        await self.ws.send(json.dumps({"type": "utterance_end"}))

        out = {"transcript": None, "say": None, "tts_bytes": 0, "error": None,
               "actions": []}
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout)
            if isinstance(raw, bytes):
                out["tts_bytes"] += len(raw)
                continue
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "transcript":
                out["transcript"] = msg.get("text")
            elif mtype == "actions":
                for item in msg.get("items", []):
                    out["actions"].append(item)
                    self.executed_actions.append(item)
                    output = None
                    if item.get("tool") == "run_command":
                        output = "e2e-fake-output"
                    await self.ws.send(json.dumps({
                        "type": "action_result", "id": item["id"],
                        "ok": True, "error": None, "output": output,
                    }))
            elif mtype == "screenshot_request":
                self.screenshot_requests += 1
                jpeg, w, h = fake_screenshot()
                await self.ws.send(json.dumps({
                    "type": "screenshot", "id": msg["id"], "format": "jpeg",
                    "w": w, "h": h, "screen_w": 1920, "screen_h": 1080,
                }))
                await self.ws.send(jpeg)
            elif mtype == "say":
                out["say"] = msg.get("text")
            elif mtype == "tts_end":
                return out
            elif mtype == "error":
                out["error"] = msg.get("message")
                return out


async def main() -> int:
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"{'OK ' if ok else 'FAIL'} {name} {detail}")

    async with websockets.connect(URL, max_size=None) as ws:
        client = FakeClient(ws)
        await ws.send(json.dumps({"type": "hello", "client_id": "e2e", "devices": []}))
        ready = json.loads(await asyncio.wait_for(ws.recv(), 15))
        check("hello->ready", ready.get("type") == "ready")

        # 1. Silence -> empty transcript error, nothing spoken.
        r = await client.utterance(b"\x00" * 32000)
        check("silence -> empty-transcript error", r["error"] == "empty transcript", str(r["error"]))

        # 2. Plain chat: a reply and audible TTS, no actions.
        r = await client.utterance(synth_16k("Please just say the word hello back to me."))
        check("chat reply spoken", bool(r["say"]) and r["tts_bytes"] > 0,
              f"say={r['say']!r} tts={r['tts_bytes']}")

        # 3. A PC action: mute -> a pc_control mute action must arrive.
        r = await client.utterance(synth_16k("Mute the sound on the computer."))
        commands = [a.get("args", {}).get("command") for a in r["actions"]
                    if a.get("tool") == "pc_control"]
        check("mute triggers pc_control mute", "mute" in commands,
              f"transcript={r['transcript']!r} actions={r['actions']}")
        check("mute is confirmed only after result", bool(r["say"]), repr(r["say"]))

        # 4. Screen question: screenshot requested, answer reflects the fake image.
        r = await client.utterance(synth_16k("What is on my screen right now?"))
        check("screenshot requested", client.screenshot_requests >= 1)
        said = (r["say"] or "").lower()
        check("vision content reaches the spoken reply",
              any(word in said for word in ("whale", "documentary", "youtube", "ocean")),
              repr(r["say"]))

    print(("PASSED" if all(checks) else "FAILED") + f" {sum(checks)}/{len(checks)}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
