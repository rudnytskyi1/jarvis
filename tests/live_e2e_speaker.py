"""Live e2e for speaker recognition and roles against localhost:8765.

Two Silero voices play two people: en_0 enrolls as TestOwner (first person =>
admin), en_29 is a stranger whose run_command must be denied server-side.
Cleans data/voices.json afterwards (the server must be restarted after this
script so its in-memory registry is fresh again).
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch
import websockets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
URL = "ws://127.0.0.1:8765/ws"
VOICES = REPO / "data" / "voices.json"

_model = None


def synth(text: str, speaker: str) -> bytes:
    global _model
    if _model is None:
        _model, _ = torch.hub.load(
            "snakers4/silero-models", "silero_tts", language="en", speaker="v3_en"
        )
    import torchaudio.functional as F

    audio = _model.apply_tts(text=text, speaker=speaker, sample_rate=48000)
    pcm = F.resample(audio.unsqueeze(0), 48000, 16000).squeeze(0).numpy()
    return (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()


async def utter(ws, pcm: bytes) -> dict:
    await ws.send(json.dumps({"type": "utterance_start", "sr": 16000,
                              "format": "pcm_s16le", "channels": 1}))
    for i in range(0, len(pcm), 960):
        await ws.send(pcm[i:i + 960])
    await ws.send(json.dumps({"type": "utterance_end"}))
    out = {"say": None, "error": None, "actions": []}
    while True:
        raw = await asyncio.wait_for(ws.recv(), 240)
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        t = msg.get("type")
        if t == "actions":
            for item in msg.get("items", []):
                out["actions"].append(item)
                output = "C:\\fake-dir" if item.get("tool") == "run_command" else None
                await ws.send(json.dumps({"type": "action_result", "id": item["id"],
                                          "ok": True, "error": None, "output": output}))
        elif t == "say":
            out["say"] = msg.get("text")
        elif t == "tts_end":
            return out
        elif t == "error":
            out["error"] = msg.get("message")
            return out


async def main() -> int:
    if VOICES.exists():
        print("ABORT: data/voices.json already exists - not touching real profiles")
        return 2

    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"{'OK ' if ok else 'FAIL'} {name} {detail}")

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client_id": "e2e-voice", "devices": []}))
        await asyncio.wait_for(ws.recv(), 15)

        r = await utter(ws, synth(
            "Jarvis, please remember my voice. My name is Test Owner.", "en_0"))
        check("enroll starts", bool(r["say"]), repr(r["say"]))

        await utter(ws, synth("The quick brown fox jumps over the lazy dog.", "en_0"))
        r = await utter(ws, synth("I really enjoy listening to music in the evening.", "en_0"))
        check("enrollment completes", bool(r["say"]), repr(r["say"]))

        data = json.loads(VOICES.read_text(encoding="utf-8"))
        person = data.get("people", {}).get("Test Owner", {})
        check("Test Owner is admin", person.get("role") == "admin", str(person.get("role")))
        check("3 voice samples stored", len(person.get("embeddings", [])) == 3,
              str(len(person.get("embeddings", []))))

        # Stranger asks for a command: must be denied server-side.
        r = await utter(ws, synth(
            "Run a command to show me the current folder please.", "en_29"))
        ran = [a for a in r["actions"] if a.get("tool") == "run_command"]
        check("stranger's run_command denied", not ran, str(r["actions"]))
        check("denial explained politely", bool(r["say"]), repr(r["say"]))

        # The admin voice asks the same: must go through.
        r = await utter(ws, synth(
            "Run a command to show me the current folder please.", "en_0"))
        ran = [a for a in r["actions"] if a.get("tool") == "run_command"]
        check("admin's run_command allowed", bool(ran), str(r["actions"]))

    # Show who the dialog log attributed each phrase to.
    dialogs = sorted((REPO / "data" / "dialogs").glob("*.jsonl"))[-1]
    tail = [json.loads(line) for line in
            dialogs.read_text(encoding="utf-8").splitlines()[-6:]]
    for entry in tail:
        print(f"   log: {entry.get('speaker')} ({entry.get('speaker_score')}) "
              f"<- {entry.get('transcript', '')[:60]!r}")

    VOICES.unlink(missing_ok=True)
    print("cleaned data/voices.json")
    print(("PASSED" if all(checks) else "FAILED") + f" {sum(checks)}/{len(checks)}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
