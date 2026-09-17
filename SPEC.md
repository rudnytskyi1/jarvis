# Jarvis — Dorm Voice Assistant. Technical Specification v1

This document is the **contract**. All modules must match it exactly: message types,
config keys, module APIs, file ownership. If code and spec disagree, the spec wins.

## 1. Overview

Two Windows 11 PCs on the same LAN in a student dorm:

| Machine | GPU | Role | Runs |
|---|---|---|---|
| **Brain server** (this PC) | RTX 5090 (32 GB) | AI inference | `server/` — STT (faster-whisper), LLM (Ollama, OpenAI-compatible API), TTS (Silero) |
| **Room client** (living-room PC, connected to TV) | RTX 3060 | Ears, voice, hands | `client/` — mic capture, wake word ("rowan", configurable), VAD, audio playback, device/PC actions |

Flow: client hears wake word → beeps → records utterance until silence (VAD) →
streams PCM to server over WebSocket → server transcribes (Whisper) → LLM with tool
calling decides actions + spoken reply → server sends `actions` (client executes:
LED strip, SwitchBot buttons, PC volume/media/apps) and streams TTS audio back →
client plays it. Optional follow-up window: after the reply, listen again briefly
without requiring the wake word.

Language policy (v1.1): users speak mostly **English** (Russian may still occur —
Whisper auto-detects). TTS default is English (Silero `v3_en`); the persona
(Jarvis-style butler, concise, replies in English) lives in `prompts/system.md`.
**All source comments, log messages, scripts, and docs are in English.**

v1.1 capabilities on top of the voice→action loop:
- **Screen vision**: a `look_at_screen` tool — server requests a screenshot from
  the client, runs it through a vision model (Ollama), feeds the answer back to
  the LLM as a real tool result.
- **Full PC control**: `run_command` (arbitrary PowerShell on the client PC with
  output returned to the LLM), `pc_control` extended with `type_text`/`hotkey`,
  and `open_app` that finds ANY installed app by itself (Start Menu / UWP index),
  not only the ones listed in config.
- **Real tool results**: the server now waits for the client's `action_result`s
  (per-action timeout) and gives them to the LLM, instead of assuming success.
- **Dialog log**: every exchange appended to `data/dialogs/YYYY-MM-DD.jsonl` on
  the server (5090) machine.
- **Persistent memory**: a `remember` tool appends facts to `data/memory.jsonl`;
  all saved facts are injected into the system prompt each session.
- Physical devices (`set_light`/`set_switch`) remain implemented but there are
  currently no devices configured (`client.devices: []`) — the persona must not
  advertise them when the device list is empty.

Both machines check out the same repo. One `config.yaml` (copied from
`config.example.yaml`) with `server:` and `client:` sections; each process reads its
own section. Python 3.11+ (conda env `jarvis` exists on the brain PC:
`C:\Users\Anton\anaconda3\envs\jarvis\python.exe`).

## 2. Repository layout & file ownership

Each implementation worker owns ONLY its files. Never create or edit files owned by
another worker.

```
jarvis/
  SPEC.md, config.example.yaml          # pre-written (do not edit)
  CLAUDE.md                             # pre-written (do not edit)
  README.md                             # W4 (Russian)
  .gitignore                            # W4
  common/                               # W4
    __init__.py
    config.py                           # config loading (API in §6)
    protocol.py                         # message-type constants (§4)
  server/                               # W1
    __init__.py, main.py, app.py, stt.py, llm.py, tts.py, tools.py, session.py
    vision.py, storage.py               # v1.1: screen vision, dialogs+memory
    requirements.txt
  prompts/system.md                     # English persona + rules (pre-written)
  data/                                 # runtime storage on the server PC (gitignored):
                                        #   dialogs/YYYY-MM-DD.jsonl, memory.jsonl
  client/                               # W2 (core) and W3 (actions/devices)
    __init__.py, main.py                # W2
    audio.py, wakeword.py, vad.py, ws_client.py   # W2
    requirements.txt                    # W2 (includes W3's deps — W3 lists them in its report)
    actions/                            # W3
      __init__.py, dispatcher.py, pc.py
      apps.py                           # v1.1: installed-app index + fuzzy resolver
    screen.py                           # v1.1: screenshot capture (W2 side)
    devices/                            # W3
      __init__.py, base.py, registry.py, magichome.py, tuya.py, switchbot.py
  scripts/                              # W4
    install-server.ps1, install-client.ps1, download-models.ps1
    run-server.ps1, run-client.ps1
  models/                               # downloaded at install time (gitignored), e.g. vosk model
```

All code imports `common/` with repo root on `sys.path` (entry points run as
`python -m server.main` / `python -m client.main` from repo root — scripts do this).

## 3. Server (W1)

- `server/main.py` — argparse `--config` (default `config.yaml` in repo root),
  loads config via `common.config.load_config`, starts uvicorn programmatically.
- `server/app.py` — FastAPI app; WebSocket endpoint at path **`/ws`**. One
  connection = one session. Handles the protocol in §4. Loads STT/LLM/TTS once at
  startup (module-level singletons initialized in FastAPI startup/lifespan).
- `server/stt.py` — `faster_whisper.WhisperModel(cfg.stt.model, device=cfg.stt.device,
  compute_type=cfg.stt.compute_type)`. `transcribe_pcm(pcm_s16le_bytes, sample_rate,
  language) -> (text, detected_language)`. Convert int16 bytes → float32 numpy / 32768.
  If `language` is null/empty use auto-detect. Use `vad_filter=True`.
- `server/llm.py` — two providers, selected by `cfg.llm.provider`:
  - `"ollama_native"` (default): POST `{base}/api/chat` (base = `cfg.llm.base_url`
    with a trailing `/v1` stripped) via `httpx`, `stream: false`,
    `think: cfg.llm.think` (default false — disables Qwen3.x reasoning),
    `tools=` from `server/tools.py`, `options: {num_predict: cfg.llm.max_tokens,
    temperature: cfg.llm.temperature}`. Native tool_calls carry `arguments` as an
    object already.
  - `"openai"`: the `openai` package against `cfg.llm.base_url` as before
    (arguments arrive as a JSON string — parse defensively).
  **Tool loop (v1.1)**: up to `cfg.llm.max_tool_rounds` (default 4) rounds. Each
  round: if the reply has tool calls, execute them IN ORDER via the ToolExecutor
  callback provided by `server/app.py` (see §5 execution matrix), append the
  assistant message with its tool_calls plus one `role: "tool"` message per call
  containing the REAL result JSON (e.g. `{"ok": true}`, `{"ok": false, "error":
  "..."}`, `{"output": "..."}`, or the vision answer), then request the next
  completion. Stop when a reply has no tool calls (that text is the spoken reply)
  or the round cap is hit (then ask for a final no-tools completion). History from
  `server/session.py`.
- `server/tools.py` — OpenAI-style tool JSON schemas for the six tools in §5, plus
  helpers to convert tool calls into protocol action items.
- `server/vision.py` — `describe_screenshot(jpeg_bytes, query) -> str`: POST to
  Ollama native `/api/chat` with `model: cfg.llm.vision_model`, one user message
  `{content: query-with-instructions, images: [base64 jpeg]}`, `stream: false`,
  no tools. Timeout 120 s; on failure return an error string (the LLM sees it as
  the tool result — never raise).
- `server/storage.py` — `DialogLog.append(entry: dict)` writes one JSON line to
  `data/dialogs/YYYY-MM-DD.jsonl` (entry: ts ISO, client_id, transcript, language,
  reply, actions list, per-stage durations); `Memory.facts() -> list[str]` and
  `Memory.add(fact: str)` over `data/memory.jsonl` (one `{"ts": ..., "fact": ...}`
  per line). Both create `data/` dirs on first use; both live on the server (5090)
  machine by design.
- `server/session.py` — per-connection: system prompt (from `prompts/system.md`,
  with `{devices}` replaced by a list built from the client's `hello`, and
  `{memory}` replaced by the numbered facts from `Memory.facts()`, or "(no saved
  facts yet)"), rolling history of the last `cfg.llm.history_turns` user/assistant
  exchanges.
- `server/tts.py` — Silero via `torch.hub.load('snakers4/silero-models', 'silero_tts',
  language=cfg.tts.language, speaker=cfg.tts.model_id)` on **CPU** (keep VRAM for
  LLM). `synth(text) -> pcm_s16le_bytes` at `cfg.tts.sample_rate`. For
  `language: "ru"` use `model_id: "v4_ru"`, speaker from `cfg.tts.speaker`
  (e.g. `xenia`). Strip characters Silero can't handle; if synthesis fails, log and
  send the `say` text with an empty TTS stream (client still shows/handles text).
  Numbers: use built-in silero handling; latin words may sound off — acceptable v1.

Whisper `large-v3` fp16 ≈ 3 GB VRAM; LLM default `qwen2.5:32b-instruct` (Q4 ≈ 20 GB
via Ollama) — fits in 32 GB together. TTS on CPU.

## 4. WebSocket protocol

JSON text frames for control, binary frames for audio. Constants in
`common/protocol.py` (`MSG_HELLO = "hello"` etc. — one constant per type below).

Client → Server:
1. `{"type": "hello", "client_id": str, "devices": [{"name": str, "type": str, "area": str|null, "description": str|null}]}` — sent once after connect. `devices` built from client config (§6); empty list is normal.
2. `{"type": "utterance_start", "sr": 16000, "format": "pcm_s16le", "channels": 1}`
3. binary frames: raw PCM s16le mono 16 kHz chunks
4. `{"type": "utterance_end"}`
5. `{"type": "action_result", "id": str, "ok": bool, "error": str|null, "output": str|null}` — REQUIRED after executing each action, in order. `output` carries data the LLM needs back (e.g. `run_command` stdout, truncated to 4000 chars); null when there is none.
6. `{"type": "screenshot", "id": str, "format": "jpeg"}` followed by exactly ONE binary frame with the JPEG bytes — reply to `screenshot_request`. On capture failure: `{"type": "screenshot_error", "id": str, "error": str}` and no binary frame.

Server → Client:
1. `{"type": "ready"}` — reply to `hello`.
2. `{"type": "transcript", "text": str, "language": str}` — after STT.
3. `{"type": "actions", "items": [{"id": str, "tool": str, "args": {…}}]}` — client executes in order and sends one `action_result` per item. `id` unique per action within the utterance (`"a1"`, `"a2"`, …). May be sent multiple times per utterance (one per tool round).
4. `{"type": "screenshot_request", "id": str}` — client captures the screen (client/screen.py) and replies per C→S #6.
5. `{"type": "say", "text": str}` — the reply text.
6. `{"type": "tts_start", "sr": <cfg.tts.sample_rate>, "format": "pcm_s16le", "channels": 1}` → binary PCM frames → `{"type": "tts_end"}`. May be an empty stream if TTS failed or text is empty.
7. `{"type": "error", "message": str}` — recoverable; client speaks nothing, plays error beep, returns to wake-word listening.

Binary-frame disambiguation: the client sends binary frames only in two
well-delimited situations — between `utterance_start`/`utterance_end`, and as the
single frame announced by a `screenshot` header. The server keeps per-connection
state to route them; they never overlap (the client records no new utterance while
a response is in flight).

Order per utterance: `transcript` → zero or more rounds of (`actions` and/or
`screenshot_request`, awaiting the matching `action_result`s / `screenshot`) →
`say` → `tts_start…tts_end`. The server waits up to 35 s per action result and
120 s per screenshot; on timeout the tool result becomes
`{"ok": false, "error": "client timeout"}` and the loop continues.
If the transcript is empty/whitespace (false trigger), server sends `transcript`
then `error` with message `"empty transcript"` — no LLM call.

## 5. Tools / actions

Six tools exposed to the LLM. Execution matrix: `set_light`, `set_switch`,
`pc_control`, `run_command` are CLIENT actions (forwarded as protocol action
items, args verbatim; result = the client's `action_result`). `look_at_screen`
and `remember` are SERVER-side (never sent as actions).

1. **`set_light`** — control a light device (LED strip etc.).
   `{"device": str (device name from config), "state": "on"|"off", "brightness": int 1–100 (optional), "color": "#RRGGBB" (optional)}`
2. **`set_switch`** — control a SwitchBot-style physical button pusher.
   `{"device": str, "action": "on"|"off"|"press"|"toggle"}`
   (Bots in press mode treat "toggle"/"on"/"off" as a single press.)
3. **`pc_control`** — control the room PC (the client machine itself).
   `{"command": "volume_set"|"volume_up"|"volume_down"|"mute"|"unmute"|"media_play_pause"|"media_next"|"media_prev"|"display_off"|"display_on"|"sleep"|"open_app"|"close_app"|"type_text"|"hotkey", "value": str|int|null}`
   `value`: `volume_set` int 0–100; `open_app`/`close_app` an app name — resolved
   by the client's installed-app index (§8 apps.py): `cfg.client.apps` overrides
   first, then fuzzy match over ALL installed apps; unknown → error result naming
   the closest candidates. `type_text` types `value` as unicode text into the
   focused window; `hotkey` presses a combo given as `"ctrl+shift+t"`-style string.
4. **`run_command`** — run an arbitrary PowerShell command on the room PC.
   `{"command": str}`. Client executes `powershell -NoProfile -Command <command>`
   with a 30 s timeout, captures stdout+stderr, truncates to 4000 chars, returns it
   in `action_result.output`. Non-zero exit → `ok: false` with output still set.
5. **`look_at_screen`** — see the room PC's screen. `{"query": str}` — what to look
   for/answer. Server-side: request screenshot from client, run `server/vision.py`,
   tool result = the vision model's answer text.
6. **`remember`** — save a fact to persistent memory. `{"fact": str}` — one
   self-contained English sentence. Server-side: `Memory.add`, result `{"ok": true}`.

Tool descriptions (in `server/tools.py`) must tell the model: reply in English;
keep spoken replies to one–two short sentences (they are read aloud — no markdown,
no code); use `look_at_screen` when the user asks about what is on the screen;
use `remember` when the user shares a lasting fact or asks to remember; device
tools only for devices in the system prompt's list (may be empty).

## 6. Config

Single `config.yaml` at repo root (copy of `config.example.yaml`). `common/config.py` API:

```python
from common.config import load_config
cfg = load_config(path)      # path to yaml; returns Config
cfg.server.host              # "0.0.0.0"
cfg.server.port              # 8765
cfg.server.stt.{model, device, compute_type, language}          # language: str | None
cfg.server.llm.{base_url, model, api_key, temperature, max_tokens, history_turns}
cfg.server.llm.{provider, think, vision_model, max_tool_rounds} # v1.1: "ollama_native"|"openai", bool, str, int
cfg.server.tts.{engine, language, model_id, speaker, sample_rate}   # English default: language "en", model_id "v3_en", speaker "en_0"
cfg.client.server_url        # "ws://192.168.x.x:8765/ws"
cfg.client.client_id
cfg.client.wakeword.{word, phrases, vosk_model}                 # phrases: list[str]
cfg.client.audio.{input_device, output_device, sample_rate}     # devices: int|str|None
cfg.client.vad.{aggressiveness, silence_ms, max_utterance_s, pre_roll_ms}
cfg.client.followup_window_s # float, 0 = off
cfg.client.apps              # dict[str, str] friendly name -> exe path/command (OVERRIDES on top of the app index; may be empty)
cfg.client.devices           # list[DeviceConfig]; [] is the current default (no physical devices yet)
```

`DeviceConfig`: `name: str`, `type: str` (`"magichome" | "tuya" | "switchbot_bot"`),
`area: str|None`, `description: str|None`, plus type-specific fields kept in a
`params: dict` (everything else from the yaml mapping). Implementation: pydantic v2
models (`extra="allow"` where needed) or plain dataclasses — pydantic preferred.
Missing optional keys get the defaults shown in `config.example.yaml`; missing
required keys → clear startup error naming the key.

`config.example.yaml` is already written — treat its keys/defaults as normative.

## 7. Client core (W2)

- `client/main.py` — argparse `--config`; loads `cfg.client`; wires everything;
  main loop (single asyncio event loop; audio callbacks push to thread-safe queues):
  1. connect WS (`ws_client.py`, auto-reconnect with 3 s backoff; on reconnect resend `hello`),
  2. wake-word listening → on detection play short ack beep (generated sine, ~880 Hz 120 ms — no wav asset needed),
  3. record utterance via VAD (include `pre_roll_ms` of audio from before trigger end),
  4. stream to server per §4 (chunks of ~30 ms), receive and handle all server messages,
  5. execute `actions` via W3 dispatcher (each in order; send one `action_result`
     per item, including `output` when the dispatcher returns one); answer
     `screenshot_request` via `client/screen.py`,
  6. play TTS stream as it arrives (`audio.py` output stream at server-declared `sr`),
  7. if `followup_window_s > 0`: after playback, run VAD listening for up to that many
     seconds; if speech detected → go to step 3 (skip wake word); else back to step 2.
- `client/audio.py` — `sounddevice`. Input: 16 kHz mono int16 blocks of 480 samples
  (30 ms). Output: playback of raw PCM at given samplerate; also `play_beep(freq, ms)`.
- `client/wakeword.py` — Vosk `KaldiRecognizer(model, 16000, json.dumps([*phrases, "[unk]"]))`
  grammar mode; model dir from `cfg.client.wakeword.vosk_model`. Feed 30 ms blocks;
  detection = any configured phrase appears in a final or partial result; after
  detection reset recognizer. `phrases` defaults to `[word]` if empty.
- `client/vad.py` — `webrtcvad.Vad(cfg.aggressiveness)` on 30 ms frames; utterance
  ends after `silence_ms` of consecutive non-speech or at `max_utterance_s`; returns
  the recorded bytes (or None if no speech at all within a lead-in timeout of 5 s).
- `client/ws_client.py` — `websockets` library wrapper: connect, send json/binary,
  async iterate messages, reconnect loop.
- `client/screen.py` (v1.1) — `capture_jpeg() -> bytes`: full primary-screen
  screenshot via Pillow `ImageGrab.grab()`, downscale to max width 1600 px,
  JPEG quality 80. Runs in `asyncio.to_thread`. Raises with a clear message on
  failure (caller converts to `screenshot_error`).

Deps (client/requirements.txt): `vosk`, `sounddevice`, `webrtcvad-wheels`
(NOT `webrtcvad` — no Windows wheels), `websockets`, `numpy`, `pyyaml`, `pydantic`,
`pillow` (v1.1, screenshots), plus W3's: `pycaw`, `comtypes`, `bleak`, `flux_led`,
`tinytuya`, `keyboard`.

## 8. Actions & devices (W3)

- `client/actions/dispatcher.py` — `Dispatcher(cfg_client, registry)` with
  `async execute(action: dict) -> (ok: bool, error: str|None, output: str|None)`
  (v1.1: third element carries data for the LLM — `run_command` output, app
  resolver hints; None otherwise). Routes `set_light` and `set_switch` to
  `client/devices/registry.py` by device name (unknown device →
  `(False, "unknown device …", None)`), `pc_control` and `run_command` to
  `actions/pc.py`. Never raises — catch everything, return an error string.
- `client/actions/pc.py` — Windows implementations:
  - volume: `pycaw` (`IAudioEndpointVolume`) — set scalar 0..1, step ±5 %, mute/unmute;
  - media keys: ctypes `SendInput` with `VK_MEDIA_*`;
  - `display_off`: `SendMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)`; `display_on`: move mouse 1 px via ctypes;
  - `sleep`: `SetSuspendState` via `powrprof.dll` (ctypes);
  - `type_text` (v1.1): unicode text via `SendInput` with `KEYEVENTF_UNICODE`;
  - `hotkey` (v1.1): parse `"ctrl+shift+t"`-style combos (modifiers: ctrl, alt,
    shift, win; keys: letters, digits, f1–f24, enter, esc, tab, space, arrows,
    del, backspace) and press via `SendInput`; unknown key → error result;
  - `run_command` (v1.1): `powershell -NoProfile -Command <command>` via
    subprocess, 30 s timeout, stdout+stderr merged, truncated to 4000 chars,
    returned as the action's `output`; kill the process tree on timeout;
  - `open_app`/`close_app`: resolve via `client/actions/apps.py` (below);
    `close_app` kills by exe name (`taskkill /IM <exe> /F`) for desktop apps,
    error for UWP apps it cannot map to a process.
- `client/actions/apps.py` (v1.1) — installed-app index + resolver:
  - Index built once at startup (and lazily refreshed if a lookup misses):
    run `powershell -NoProfile -Command "Get-StartApps | ConvertTo-Json"` —
    covers desktop AND Store/UWP apps as `{Name, AppID}`;
    merge `cfg.client.apps` entries on top (name → path, highest priority).
  - `resolve(name) -> AppEntry | None` with candidates: exact case-insensitive,
    then prefix, then fuzzy (`difflib.get_close_matches`, cutoff 0.6).
  - Launch: config-path entries via `os.startfile(path)`; Get-StartApps entries
    via `os.startfile("shell:AppsFolder\\" + app_id)`.
  - On miss, the error result lists up to 3 closest names so the LLM can retry.
- `client/devices/base.py` — `class Device(ABC)`: `name`, `async set_light(state, brightness, color)` (raise `NotImplementedError` where N/A), `async set_switch(action)`.
- `client/devices/magichome.py` — Magic Home / Zengge Wi-Fi LED controllers via
  `flux_led` (`WifiLedBulb(host)`; it's sync — run in `asyncio.to_thread`). on/off,
  brightness, RGB color.
- `client/devices/tuya.py` — Tuya Wi-Fi strips via `tinytuya.BulbDevice(dev_id,
  host, local_key)` with `set_version(params.get("version", 3.3))`; sync → `to_thread`.
- `client/devices/switchbot.py` — SwitchBot Bot over BLE via `bleak` directly
  (works on Windows): connect by MAC (`params["mac"]`), write to characteristic
  `cba20002-224d-11e6-9fb8-0002a5d5c51b`: press `b"\x57\x01\x00"`, on `b"\x57\x01\x01"`,
  off `b"\x57\x01\x02"`. Retry once on failure; disconnect after. If the Bot has a
  password configured — out of scope v1 (document in README).
- `client/devices/registry.py` — builds `{name: Device}` from `cfg.client.devices`
  by `type`; `get(name)`.

## 9. Docs & scripts (W4)

- `common/config.py`, `common/protocol.py` per §4/§6.
- `README.md` — **in Russian**: what this is, hardware shopping list (SwitchBot Bot
  + USB BLE dongle if no Bluetooth; identifying the LED strip type: Magic Home vs
  Tuya vs IR-only), install on both machines (conda env, `scripts/`), Ollama setup
  (`ollama pull qwen2.5:32b-instruct`), config walkthrough, how to find Tuya
  local_key (tinytuya wizard) and SwitchBot MAC (bleak scanner one-liner), running,
  autostart via Task Scheduler, troubleshooting (firewall port 8765, mic selection,
  vosk model path).
- `scripts/install-server.ps1` / `install-client.ps1` — create/update conda env
  `jarvis` (fallback: plain venv), `pip install -r <side>/requirements.txt`.
- `scripts/download-models.ps1` — download + unzip Vosk
  `vosk-model-small-en-us-0.15` into `models/` (wake word "rowan" is English).
- `scripts/run-server.ps1` / `run-client.ps1` — activate env, `python -m server.main` / `python -m client.main` from repo root.
- `.gitignore` — `models/`, `config.yaml`, `__pycache__/`, `*.log`.

## 10. Quality bar

- Every file complete and runnable — no TODO/placeholder/pass-stubs.
- Windows-first: paths, ctypes calls, no POSIX-only APIs.
- Every config key read in code exists in `config.example.yaml` and §6.
- Every message type sent by one side is handled by the other.
- Log with `logging` (INFO default). **English only** — comments, docstrings, log
  messages, script output, docs. (v1.1: any remaining Russian text in code,
  scripts, .bat files, yaml comments or README must be translated.)
- Graceful Ctrl+C on both sides.
