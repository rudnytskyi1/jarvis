"""Manual voice -> running server -> OpenAI -> audio check, no device actions.

Uses the running server's key and budget, never reads or prints the API key.
This sends one small paid conversation request through the normal pipeline.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import websockets

from common import protocol as p
from common.config import load_config
from hub.stt import pcm_to_float32, resample
from hub.tts import TtsEngine, float32_to_pcm16


async def main():
    cfg = load_config(ROOT / "config.openai.yaml")
    voice = TtsEngine(cfg.server.tts)
    await asyncio.to_thread(voice.load)
    audio = await asyncio.to_thread(voice.synth, "Rowan, what is two plus two? Just say the answer.")
    assert audio, "Synthetic test audio was not generated"
    pcm = float32_to_pcm16(resample(pcm_to_float32(audio), voice.sample_rate, 16000))
    report = {"reply": "", "audio_bytes": 0, "actions": 0}
    async with websockets.connect(f"ws://127.0.0.1:{cfg.server.port}/ws", max_size=None) as ws:
        await ws.send(json.dumps({"type": p.MSG_HELLO, "client_id": "deployment-smoke", "devices": []}))
        ready = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert ready["type"] == p.MSG_READY
        started = time.perf_counter()
        await ws.send(json.dumps({"type": p.MSG_UTTERANCE_START, "sr": 16000, "format": "pcm_s16le", "channels": 1}))
        await ws.send(pcm)
        await ws.send(json.dumps({"type": p.MSG_UTTERANCE_END}))
        async with asyncio.timeout(90):
            async for raw in ws:
                if isinstance(raw, bytes):
                    report["audio_bytes"] += len(raw)
                    continue
                msg = json.loads(raw)
                kind = msg["type"]
                if kind == p.MSG_TRANSCRIPT:
                    report["transcript"] = msg.get("text", "")
                    report["clarification"] = msg.get("clarification", "")
                elif kind == p.MSG_SAY:
                    report["reply"] = msg.get("text", "")
                elif kind == p.MSG_ACTIONS:
                    for action in msg.get("items", []):
                        report["actions"] += 1
                        await ws.send(json.dumps({"type": p.MSG_ACTION_RESULT, "id": action["id"],
                                                  "ok": False, "error": "Actions disabled in deployment smoke test"}))
                elif kind in (p.MSG_SCREENSHOT_REQUEST, p.MSG_CAMERA_REQUEST):
                    await ws.send(json.dumps({"type": p.MSG_SCREENSHOT_ERROR if kind == p.MSG_SCREENSHOT_REQUEST else p.MSG_CAMERA_ERROR,
                                              "id": msg.get("id"), "error": "No image capture in smoke test"}))
                elif kind == p.MSG_ERROR:
                    report["error"] = msg.get("message")
                    break
                elif kind == p.MSG_TTS_END:
                    break
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
    (ROOT / "data" / "deployment-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    assert not report.get("error") and not report.get("clarification"), "Voice pipeline failed"
    assert "four" in report["reply"].lower() or "4" in report["reply"], "Expected the answer four from OpenAI"
    assert report["audio_bytes"] > 0 and report["actions"] == 0


if __name__ == "__main__":
    asyncio.run(main())
