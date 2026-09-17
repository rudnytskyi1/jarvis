# Jarvis — голосовой ассистент для комнаты в общаге

Two-machine voice assistant. **SPEC.md is the contract** — protocol, config keys,
module APIs, file ownership live there. Keep code, `config.example.yaml`, and
SPEC.md §6 in sync whenever config keys change.

## Machines
- **This PC (RTX 5090)** — "brain" server: `server/` (faster-whisper STT, LLM via
  local Ollama at `http://127.0.0.1:11434/v1`, Silero TTS on CPU).
- **Living-room PC (RTX 3060, connected to the TV)** — client: `client/` (mic, Vosk
  wake word from config — currently `rowan`, VAD, playback, device/PC actions:
  Magic Home / Tuya LED strips, SwitchBot Bot button pushers over BLE, pycaw volume,
  media keys).

## Environment & commands
- Python env: conda env **`jarvis`** — `C:\Users\Anton\anaconda3\envs\jarvis\python.exe`.
  Always run/install through it (`conda activate jarvis` or the full python path).
- Run from repo root: `python -m server.main` / `python -m client.main`
  (or `scripts/run-server.ps1`, `scripts/run-client.ps1`).
- Syntax check: `python -m py_compile <files>`; no test suite yet.
- `config.yaml` is machine-local (gitignored); `config.example.yaml` is the template.

## Conventions
- Windows-first (ctypes/pycaw/bleak); no POSIX-only APIs.
- User-facing strings, README, and log messages: Russian is fine; code identifiers
  and docstrings in English.
- Client executes all side effects; server only thinks (STT→LLM→TTS). Don't move
  device control server-side.
- WebSocket message types come from `common/protocol.py` constants — never inline
  string literals for them.
- The global working guide (`~/.claude/CLAUDE.md`, from the my_claude_workspace
  kit) applies here: every implementation batch ends with a fresh-context
  `spec-compliance-reviewer` run before reporting done.

## Project glossary

| The user says | Code / config | Notes |
|---|---|---|
| джарвис / бот | the whole assistant | persona lives in `prompts/system.md` |
| гирлянда / лента / полоска | a `devices:` entry, `type: magichome` or `tuya` | one physical LED strip; user uses the words interchangeably |
| обычный выключатель / свет | `type: switchbot_bot` device | SwitchBot Bot pressing a wall switch over BLE |
| зал / комната | `area:` on a device; the 3060 PC is in this room | |
| сервер | the RTX 5090 PC (`server/` section) | AI only, no device control |

## Known failure modes (check these in every review)

- A change on one side of the WS boundary without the matching change on the
  other (client/server message types, field names, order per SPEC §4).
- A config key read in code that is missing from `config.example.yaml` or
  `common/config.py` (or defaults drifting apart) — SPEC §6 is the contract.
- Tool/arg names drifting between `server/tools.py` (what the LLM emits) and
  `client/actions/dispatcher.py` (what gets executed) — SPEC §5.
- Blocking calls (whisper/LLM/TTS inference, sync device libs) run directly on
  the asyncio event loop instead of `to_thread`.
- Audio frame-size mismatches: webrtcvad and Vosk both assume 30 ms / 480-sample
  int16 mono 16 kHz frames end-to-end.
- `webrtcvad` pinned instead of `webrtcvad-wheels` (no Windows wheels).
