# Jarvis — a voice assistant for a dorm room

**Telegram owner panel:** [permissions, memory, profiles, cameras and alerts](docs/TELEGRAM_ADMIN.md).
Use `/tools` in the owner's DM or configured group. Incoming photos support
SAM3 analysis and Nano Banana editing. Telegram runs entirely on the server.

**Public client package:** [standalone setup and safe export](docs/CLIENT_DISTRIBUTION.md).
Run `python scripts/export_client.py` to build a separate client folder/ZIP
without brain code, API keys, personal settings, recordings or Git history.

**Budgeted OpenAI mode:** see [setup and addressing behavior](docs/OPENAI_SETUP.md).
It keeps speech, cameras and device execution local, uses a text-only API with
persistent local accounting, and requires the wake word for every turn. The
monthly allowance is a setting, not a wall: `0` counts the spending without
ever refusing a request, and an oversized conversation is trimmed instead of
being answered with "conversation is too long" (DECISIONS.md API-01/API-02).
The original local-model configuration below remains available.

**Nano Banana 2:** [image creation, photo editing and Gemini key setup](docs/IMAGE_GENERATION.md).
Requested image edits send the selected photo to Google and share the same allowance.

**Multiple speakers:** [local diarization setup and limitations](docs/MULTI_SPEAKER.md).
Community-1 labels turns locally and selects the addressed speaker where possible.
The active room profile attempts requests despite overlap; strict rejection is
optional. Voice enrollment still requires a clean single-speaker recording.

**Personal room experience:** [guided voice/face registration, personal chat,
tracking, screenshots and interruption confirmation](docs/ROOM_EXPERIENCE.md).

A voice assistant running on two PCs. Speech recognition, camera processing and
voice synthesis run locally. The optional OpenAI profile sends text requests,
the current person's recent history and tool results to the configured OpenAI
model (GPT-5.6 Luna in the active room profile).

The room profile uses local **Kokoro Michael**, an American English male voice.
Rowan's fixed personality is cheeky, uses uncensored profanity in casual chat,
and swears back during invited banter. Casual profanity is aimed at the situation;
personal roasts need an invitation.
Invited comebacks rotate their comic approach and avoid recent wording. On that
path, past requests are quoted references and recent answers are an avoidance
list, not assistant examples to copy. The full personal history is retained for
ordinary questions and tasks.
Past replies and saved tone preferences do not redefine it: history supplies
facts and task context, not a persistent mood. Ordinary requests still get useful
answers, and explicit stop-speaking commands take priority over banter.
An explicit **"Rowan AI, answer as Putin"** or **"Rowan AI, act like Genghis Khan"**
starts a five-minute fictional parody for the recognized speaker. Say
**"Rowan AI, stop roleplay"** or **"Роуан, обычный режим"** to end it early.
The mode survives subsequent questions, keeps the usual Michael voice, and labels
generated answers "Parody:". It changes delivery, not tool permissions or factual
accuracy. It is temporary, is not saved as global memory, and ends on reconnect.
Recognized speakers have separate modes; unidentified guests on one connection
share a temporary guest mode. Racial abuse and threats against people are excluded.
Say **"Rowan AI, update my voice"** to add voice samples after confirming your
identity. Overlapping voices still require a repeat; diarization does not
separate simultaneous speech into clean audio tracks.

Browser choices use actual installed apps/open windows, with personal or admin
global preferences. The current speaker gets 25 recent exchanges plus permanent
memory; earlier messages remain searchable with timestamps. Global preferences
override personal ones. Live camera questions are available to guests too.

You say: **"rowan ai, what's on my screen?"** → the room PC takes a screenshot, a
vision model looks at it, and Jarvis answers out loud in English.

```
  [room PC]                                        [brain PC, RTX 5090]
  mic -> wake word (Vosk) -> VAD                   STT:    faster-whisper large-v3
        |                                          LLM:    Ollama qwen3:30b
        +-- PCM 16 kHz ---- WebSocket :8765 ---->   vision: Ollama qwen3-vl:30b
        <-- actions + reply text + audio -------+   TTS:    Silero v3_en (on CPU)
  speakers, screen, keyboard, PowerShell,          data/:  dialog logs + memory
  LED strip, SwitchBot, volume/media/apps
```

- **Brain PC** (RTX 5090, 32 GB VRAM) — only "thinks": speech → text → decision →
  voice. It also stores the dialog log and the long-term memory in `data/`.
- **Room PC** (RTX 3060, connected to the TV) — the "ears and hands": it listens,
  speaks, controls itself and drives the physical devices.
- Language: the assistant **speaks English** (Silero `v3_en`). You may speak
  English or Russian — Whisper auto-detects — but the reply is always English.
  The wake word is English (`rowan`) so the small English Vosk model can run
  constantly with few false triggers.

## What v1.1 can do

| Tool | What it does |
|---|---|
| `pc_control` | The room PC: volume (set/up/down/mute), play-pause, next/previous track, monitor off/on, sleep, open/close **any installed app**, `type_text` (types unicode text into the focused window), `hotkey` (presses combos like `ctrl+shift+t`) |
| `run_command` | Runs an **arbitrary PowerShell command** on the room PC and gives the output back to the model (30 s timeout, output truncated to 4000 characters) |
| `look_at_screen` | **Screen vision**: the server asks the client for a screenshot and runs it through `qwen3-vl:30b`, so Jarvis can read an error, name a game, or summarize a page |
| `remember` | **Persistent memory**: saves a fact to `data/memory.jsonl` on the brain PC; every saved fact is injected into the system prompt at the start of each session |
| `set_light` | LED strip / garland: on/off, brightness 1–100, color `#RRGGBB` |
| `set_switch` | SwitchBot Bot on a regular wall switch: `on` / `off` / `press` / `toggle` |

Two more things that are not tools but matter:

- **Real action results.** The server waits for the client's `action_result` for
  every action (35 s per action, 120 s per screenshot) and feeds the real result
  back to the model. If something fails, Jarvis knows it failed and says so.
- **Dialog log.** Every exchange is appended to
  `data/dialogs/YYYY-MM-DD.jsonl` on the brain PC: timestamp, transcript,
  detected language, reply, the actions taken, and per-stage durations.

Out of the box `client.devices` is **empty** — there are no physical devices set
up yet, and the assistant knows not to offer them. Everything else (PC control,
PowerShell, screen vision, memory) works with zero hardware purchases.

---

## 1. What to buy (hardware) — when you decide to add devices

The minimum is **nothing**: a voice remote for the room PC (volume, music, apps,
monitor, typing, PowerShell, screen questions) needs only the mic you already
have. Buy the rest only when you actually want physical devices in the room.

| Item | What it is for | What to look for |
|---|---|---|
| **SwitchBot Bot** (button pusher) | Switch a normal wall switch without touching the wiring — ideal for a dorm | Get the original SwitchBot Bot. It sticks next to the rocker with 3M tape. Works over Bluetooth LE, **no hub needed** |
| **USB Bluetooth dongle (BLE, 4.0+ / 5.x)** | If the room PC has no Bluetooth | Check with `Get-PnpDevice -Class Bluetooth` in PowerShell. Empty → you need a dongle. Pick a CSR8510 / Realtek RTL8761B chipset with BLE support and Windows 11 drivers |
| **Desktop microphone** (USB) | The mic built into a webcam or TV is usually noisy and too far away | Any USB mic or lavalier 1–3 m from the couch. Cheap option: a USB conference microphone |
| **Smart LED strip** | Backlight behind the TV | See below — the important part is **not buying an IR-only strip** |
| Speakers / soundbar | Jarvis's voice | The TV speakers over HDMI are fine |

### How to tell the LED strip types apart

This is the main trap. Look at **the controller and the app**, not at the strip
itself.

1. **Magic Home / Zengge (Wi-Fi)** — what you want, works out of the box.
   Signs: the box or controller says **Magic Home**, **MagicHome Pro**,
   **Zengge**, **LEDnet**; the listing mentions "Magic Home Pro app" or
   "WiFi RGB controller". The controller has an antenna or says 2.4G WiFi.
   → in the config `type: magichome`, you only need the controller's IP.
2. **Tuya / Smart Life (Wi-Fi)** — also works, but takes longer to set up.
   Signs: the **Smart Life** or **Tuya Smart** app, labels "Tuya",
   "powered by Tuya", "works with Smart Life".
   → `type: tuya`, you need `dev_id`, the IP and `local_key` (see §6.3).
3. **IR strip (remote only)** — **will not work**. Signs: a small 24/44-button IR
   remote in the box, no Wi-Fi and no Bluetooth, very cheap, no app mentioned
   anywhere in the description. Such a strip can only be driven through a
   separate IR gateway (e.g. Broadlink RM4) — not supported.
   Cheap way out: buy a separate Magic Home Wi-Fi controller (SP105E or similar
   for 12 V RGB) and wire the same strip into it.
4. **Bluetooth strip (no Wi-Fi)** — an app like "HappyLighting" or
   "LotusLantern", no Wi-Fi setup at all. Not supported.

So the "safe" purchase is a strip/controller explicitly labeled **Magic Home**
plus a **SwitchBot Bot** (plus a BLE dongle if the PC has no Bluetooth).

---

## 2. Software prerequisites

**On both PCs:**
- Windows 10/11.
- **Miniconda or Anaconda** (recommended) — the env is called `jarvis`, Python 3.11.
  The install scripts create it themselves; if conda is missing entirely, they
  fall back to a plain `venv` in `.venv` at the repo root.
- Git (or just download the repo archive) — both machines need the same repo
  contents.

**Brain PC only:**
- Up-to-date NVIDIA drivers (for CUDA).
- **Ollama** — https://ollama.com/download

**Room PC only:**
- A microphone and working audio; Bluetooth if you buy a SwitchBot.
- The user must be **logged in** for it to work: the mic, the speakers and
  screenshots all need an active desktop session.

---

## 3. Install on the brain PC (RTX 5090)

```powershell
cd C:\Users\Anton\Desktop\jarvis
powershell -ExecutionPolicy Bypass -File scripts\install-server.ps1
```

The script finds or creates the conda env `jarvis` (Python 3.11), installs
`server\requirements.txt`, creates `config.yaml` from `config.example.yaml`, and
checks torch/CUDA and the presence of Ollama.

Then the models:

```powershell
ollama pull qwen3:30b        # chat + tool calling (Qwen3-30B-A3B)
ollama pull qwen3-vl:30b     # screen vision, used by look_at_screen
ollama list                  # confirm both are there
```

Ollama starts on its own on Windows and listens on `http://127.0.0.1:11434`.
Check the API:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 3
```

The server talks to Ollama through its **native** API (`/api/chat`) with
`think: false`, which switches Qwen3's reasoning off — otherwise every reply
would start with a long chain of thought and the voice answer would take
forever. `llm.base_url` still keeps the `/v1` suffix in the config; the server
strips it for the native endpoint.

If the client connects over the LAN (not ngrok), open port 8765 once, from an
**elevated** PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Jarvis 8765" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765
```

Memory: Whisper `large-v3` in fp16 is about 3 GB of VRAM and `qwen3:30b` in Q4 is
about 19 GB — together they fit into 32 GB. Silero TTS is deliberately kept on
the CPU so it does not eat VRAM. See the VRAM note in §11 about the vision model.

---

## 4. Install on the room PC (RTX 3060)

```powershell
cd C:\path\to\jarvis
powershell -ExecutionPolicy Bypass -File scripts\install-client.ps1
```

The script finds or creates the `jarvis` env, installs
`client\requirements.txt`, downloads the Vosk model into
`models\vosk-model-small-en-us-0.15` (that part is `scripts\download-models.ps1`),
creates `config.yaml`, prints the audio device list and checks Bluetooth.

The Vosk model can also be (re)downloaded on its own:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1          # ~40 MB
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Force   # re-download
```

---

## 5. How the client finds the server: ngrok or the LAN

### Option A (the default): through ngrok

Dorm Wi-Fi often isolates clients from each other (AP isolation) — the PCs then
simply cannot see each other over `192.168.x.x`. So by default the client reaches
the server through an ngrok tunnel:

```yaml
client:
  server_url: wss://dorm-smart-un-iversity-of-nebr-omaha.ngrok.app/ws
```

On the brain PC the tunnel is brought up by `start-jarvis-server.bat` (it starts
both the server and `ngrok http --url=dorm-smart-un-iversity-of-nebr-omaha.ngrok.app 8765`).
Nothing to configure: ngrok is already installed
(`C:\Users\Anton\Desktop\ngrok.exe`) and authenticated. No firewall rule is
needed for this option — the connection is outbound. The cost is roughly
50–150 ms of extra latency per phrase.

### Option B: directly over the LAN (faster, when the PCs can see each other)

On the brain PC:

```powershell
ipconfig | Select-String IPv4
```

Take the `192.168.x.x` address and put it into `client.server_url`
(`ws://192.168.x.x:8765/ws` — scheme `ws://`, not `wss://`).
To keep the address stable, make a DHCP reservation on the router (or set a
static IP). The Windows network must be marked **private**:

```powershell
Get-NetConnectionProfile                                     # check NetworkCategory
Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private   # elevated
```

To verify the client can reach it: `Test-NetConnection 192.168.x.x -Port 8765`.

---

## 6. Configuring `config.yaml`

One file for both machines: the server reads the `server:` section, the client
reads `client:`. Copy the template and edit it (the install scripts do this for
you):

```powershell
Copy-Item config.example.yaml config.yaml
notepad config.yaml
```

`config.yaml` is not committed to git (`.gitignore`). If you miss a required key,
the process refuses to start and names the missing key in plain text.

### 6.1 The `server` section

```yaml
server:
  host: 0.0.0.0            # listen on all interfaces (otherwise the client cannot reach it)
  port: 8765
  stt:
    model: large-v3        # faster-whisper: large-v3 | medium | small | distil-large-v3
    device: cuda           # cpu if there is no GPU
    compute_type: float16  # use int8 on cpu
    language: null         # null = auto-detect; "en" pins English (a bit faster)
  llm:
    provider: ollama_native # ollama_native (recommended: can disable thinking) | openai
    base_url: http://127.0.0.1:11434/v1   # /v1 is stripped automatically for ollama_native
    model: qwen3:30b                      # Qwen3-30B-A3B; tool calling verified
    api_key: ollama                       # Ollama ignores it, the openai client demands non-empty
    think: false                          # disable Qwen3 reasoning for fast voice replies
    vision_model: qwen3-vl:30b            # used by look_at_screen
    temperature: 0.6
    max_tokens: 1024
    max_tool_rounds: 4                    # tool-call rounds per utterance
    history_turns: 12                     # how many recent exchanges to keep
  tts:
    engine: silero
    language: en
    model_id: v3_en
    speaker: en_0          # en_0 .. en_117 (v3_en voices)
    sample_rate: 48000
```

On the first run Silero downloads the voice model into the torch.hub cache
(`C:\Users\<you>\.cache\torch\hub`) — that needs internet and about 100 MB.

### 6.2 The `client` section

```yaml
client:
  server_url: wss://dorm-smart-un-iversity-of-nebr-omaha.ngrok.app/ws   # or ws://192.168.1.100:8765/ws
  client_id: livingroom

  wakeword:
    word: rowan ai
    phrases: [rowan ai, rowan a i, roan ai, roan a i, rowen ai, rowen a i, rowanai]
    vosk_model: models/vosk-model-small-en-us-0.15   # path relative to the repo root

  audio:
    input_device: null       # null = default mic; or an index (3) or a name fragment ("Yeti")
    output_device: null
    sample_rate: 16000       # do not change: Whisper/VAD/Vosk all run at 16 kHz

  vad:
    aggressiveness: 2        # 0..3, higher = cuts silence/noise harder
    silence_ms: 800          # the pause that ends an utterance
    max_utterance_s: 15
    pre_roll_ms: 300         # audio from BEFORE the trigger included in the recording

  followup_window_s: 6       # after a reply, listen N more seconds without the wake word (0 = off)

  apps: {}                   # optional overrides: spoken name -> exe path

  devices: []                # no physical devices yet
```

About `apps`: in v1.1 you normally leave it **empty**. "Open Spotify", "close
Chrome", "launch Photoshop" work on their own — the client builds an index of
every installed app (`Get-StartApps`, which covers both desktop and Microsoft
Store apps) and matches the spoken name against it: exact match first, then
prefix, then fuzzy. If the name is not found, the error lists the three closest
candidates and the model retries with one of them. Add an `apps` entry only when
the automatic match keeps picking the wrong thing:

```yaml
  apps:
    browser: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
```

Backslashes in YAML are doubled (`\\`) or the path goes in single quotes.
`close app` kills the process by exe name (`taskkill /IM chrome.exe /F`).

List the audio devices (indexes and names):

```powershell
& "C:\Users\Anton\anaconda3\envs\jarvis\python.exe" -c "import sounddevice; print(sounddevice.query_devices())"
```

### 6.3 Devices (`client.devices`)

Currently `devices: []` — there is no hardware yet, and with an empty list Jarvis
will say that no smart devices are set up instead of pretending to switch them.
When the hardware arrives, each device gets the common fields `name` (what you
call it out loud — this name is what the model sees), `type`, `area`,
`description`, plus the type-specific fields.

**Magic Home / Zengge:**

```yaml
  devices:
    - name: led strip
      type: magichome
      area: room
      description: LED strip behind the TV
      host: 192.168.1.50      # controller IP
```

To find the controller IP: look in the Magic Home Pro app (device properties),
in the router's client list, or scan for it:

```powershell
& "C:\Users\Anton\anaconda3\envs\jarvis\python.exe" -c "from flux_led import BulbScanner; s=BulbScanner(); print(s.scan(timeout=5))"
```

Give the controller a DHCP reservation, otherwise its IP drifts after a router
reboot.

**Tuya / Smart Life:**

```yaml
    - name: garland
      type: tuya
      area: room
      description: garland on the window
      dev_id: "xxxxxxxxxxxxxxxx"
      host: 192.168.1.51
      local_key: "yyyyyyyyyyyyyyyy"
      version: 3.3            # if it does not work, try 3.4 or 3.1
```

How to get the `local_key` (the tinytuya wizard, done once):

1. Register at https://iot.tuya.com → **Cloud** → **Create Cloud Project**
   (pick the same region as the app; Development Method: Smart Home).
2. In the project: **Devices → Link App Account** → scan the QR code with the
   Smart Life / Tuya Smart app (Profile → scanner icon). Your devices appear in
   the list.
3. Copy the project's **Access ID** and **Access Secret** (the Overview tab).
4. On the room PC:

   ```powershell
   cd C:\path\to\jarvis
   & "C:\Users\Anton\anaconda3\envs\jarvis\python.exe" -m tinytuya wizard
   ```

   The wizard asks for the Access ID, the Access Secret, the region
   (`eu`/`us`/`cn`) and any device ID from the app, then queries the cloud and
   the local network itself.
5. It writes a `devices.json` next to you: for each device it has `id` (that is
   `dev_id`), `key` (that is `local_key`) and `ip` (that is `host`). Copy them
   into `config.yaml`.

Note: the `local_key` changes if you remove and re-add the device in the app —
then the wizard has to be run again.

**SwitchBot Bot:**

```yaml
    - name: main light
      type: switchbot_bot
      area: room
      description: main room light (button pusher on the wall switch)
      mac: "AA:BB:CC:DD:EE:FF"
      mode: press             # press | lever
```

How to find the MAC (a Bot advertises itself as **WoHand**):

```powershell
& "C:\Users\Anton\anaconda3\envs\jarvis\python.exe" -c "import asyncio; from bleak import BleakScanner; print('\n'.join(f'{d.address}  {d.name}' for d in asyncio.run(BleakScanner.discover(timeout=8))))"
```

The MAC is also shown in the SwitchBot app: device → gear icon → Device Info.
While scanning, **close the SwitchBot app on your phone** (BLE allows one
connection per device) and keep the PC within a couple of meters of the Bot.

Modes:
- `mode: press` — the Bot just presses the rocker (a normal switch). The commands
  `on`, `off`, `press` and `toggle` all produce a single press.
- `mode: lever` — the Bot has the pull-up lever attached and can do separate "on"
  and "off" (Switch mode in the SwitchBot app).

Limitation: **a password on the SwitchBot is not supported** — in the SwitchBot
app the device password must be off (Device Info → Password → Off).

---

## 7. Running

The easiest way is the two launchers at the repo root (double-click):

- **Brain PC:** `start-jarvis-server.bat` — brings up the ngrok tunnel and the
  server. It creates `config.yaml` from the template if it is missing, starts
  ngrok in its own window, then runs `scripts\run-server.ps1`.
- **Room PC:** `start-jarvis-client.bat` — starts the client.

Or by hand. On the brain PC:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-server.ps1
```

The log should show: Whisper loading, the connection to Ollama, Silero loading
and `Uvicorn running on http://0.0.0.0:8765`.

On the room PC:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-client.ps1
```

The log shows the audio device list, the Vosk model loading, the connection to
the server and the `ready` reply. Then just talk:

- "**rowan**, set the volume to thirty"
- "**rowan**, next track"
- "**rowan**, open Spotify"
- "**rowan**, what's on my screen?"
- "**rowan**, read me that error message"
- "**rowan**, how much free space is on drive C?"
- "**rowan**, type my email address"
- "**rowan**, press ctrl shift t"
- "**rowan**, remember that my lecture starts at nine on Tuesdays"
- "**rowan**, turn the monitor off"

A short beep means recording started; it ends by itself after a pause
(`vad.silence_ms`). If `followup_window_s > 0`, you can say the next phrase right
after the reply, without the wake word.

Stop either side with **Ctrl+C**.

Both processes can also be started by hand from the repo root:

```powershell
& "C:\Users\Anton\anaconda3\envs\jarvis\python.exe" -m server.main --config config.yaml
& "C:\Users\Anton\anaconda3\envs\jarvis\python.exe" -m client.main --config config.yaml
```

(Always **from the repo root** — otherwise the `common` package and the relative
Vosk model path are not found.)

---

## 8. What Jarvis remembers: `data/` on the brain PC

Both files live on the **brain PC** (the machine that runs `server/`) and are
gitignored.

```
data/
  dialogs/2026-09-17.jsonl   # one JSON line per exchange
  memory.jsonl               # one JSON line per remembered fact
```

- **Dialog log** — one line per utterance: timestamp, client id, transcript,
  detected language, the reply, the list of actions taken, and per-stage
  durations (STT / LLM / TTS). This is the first place to look when an answer was
  odd or slow.
- **Memory** — every `remember` call appends `{"ts": ..., "fact": "..."}`. All
  facts are injected into the system prompt when a session starts, so Jarvis
  knows them in every later conversation.

To read the last few exchanges:

```powershell
Get-Content data\dialogs\$(Get-Date -Format yyyy-MM-dd).jsonl -Tail 5
```

To forget something, edit `data\memory.jsonl` by hand (delete the line) and
restart the server — facts are loaded at session start. To wipe the memory
completely, delete the file.

---

## 9. Autostart (Windows Task Scheduler)

The microphone, the speakers and screenshots are only available inside a user
session, so the tasks must trigger **at logon**, not at boot.

Room PC (elevated PowerShell, adjust the path):

```powershell
schtasks /Create /TN "Jarvis Client" /SC ONLOGON /RL LIMITED /F /TR "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\jarvis\scripts\run-client.ps1"
```

Brain PC:

```powershell
schtasks /Create /TN "Jarvis Server" /SC ONLOGON /RL LIMITED /F /TR "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\Users\Anton\Desktop\jarvis\scripts\run-server.ps1"
```

Note that the task above starts the server **without** the ngrok tunnel. If you
rely on ngrok, point the task at the launcher instead
(`C:\Users\Anton\Desktop\jarvis\start-jarvis-server.bat`), or keep a separate
task that runs ngrok.

Through the GUI (`taskschd.msc` → Create Task):
- **General**: "Run only when user is logged on"; "Run with highest privileges"
  is not needed.
- **Triggers**: "At log on", delayed 30 seconds (so the network and Ollama are up).
- **Actions**: program `powershell.exe`, arguments
  `-ExecutionPolicy Bypass -WindowStyle Hidden -File C:\...\scripts\run-client.ps1`,
  "Start in" = the repo root.
- **Conditions**: uncheck "Start the task only if the computer is on AC power"
  (for a laptop).
- **Settings**: "Restart the task if it fails" every 1 minute, up to 3 times.

Check or remove:

```powershell
schtasks /Run /TN "Jarvis Client"
schtasks /Query /TN "Jarvis Client" /V /FO LIST
schtasks /Delete /TN "Jarvis Client" /F
```

---

## 10. Repository layout

```
jarvis/
  SPEC.md                 # the contract: protocol, config, module APIs
  config.example.yaml     # settings template (config.yaml is yours, not committed)
  common/                 # shared: config loading, protocol constants
  server/                 # brain PC: FastAPI + WebSocket, Whisper, LLM, vision, storage, Silero
  prompts/system.md       # Jarvis's persona (edit to taste)
  client/                 # room PC: mic, wake word, VAD, playback, screenshots
    actions/              # volume, media, monitor, typing, hotkeys, PowerShell, app index
    devices/              # magichome, tuya, switchbot
  scripts/                # install-*.ps1, download-models.ps1, run-*.ps1
  models/                 # the Vosk model (downloaded, not committed)
  data/                   # dialog logs + memory, on the brain PC (not committed)
```

---

## 11. Troubleshooting

**The client does not connect to the server (through ngrok)**
- Is the tunnel up on the brain PC? Open
  https://dorm-smart-un-iversity-of-nebr-omaha.ngrok.app/health in a browser — it
  should return a JSON status. If not, run `start-jarvis-server.bat`.
- `ERR_NGROK_334` in the ngrok window means the domain is already claimed by
  another agent — close the older ngrok window with that domain.
- In `client.server_url` the scheme must be **`wss://`** (not `ws://`) and the
  path must end with `/ws`.

**The client does not connect to the server (directly over the LAN)**
- Is the port open on the brain PC:
  `New-NetFirewallRule -DisplayName "Jarvis 8765" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765` (elevated).
- From the room PC: `Test-NetConnection 192.168.1.100 -Port 8765` →
  `TcpTestSucceeded: True`.
- `server.host` must be `0.0.0.0` (not `127.0.0.1`).
- `client.server_url` must keep the `/ws` path and use the `ws://` scheme.
- The network must be marked **private** (see §5) and both PCs must be on the
  same subnet. Guest Wi-Fi or client isolation is exactly the case where a direct
  connection cannot work — use the ngrok option.
- The client reconnects every 3 seconds on its own, so the server can be
  restarted freely.

**VRAM: the models do not fit / the GPU is thrashing**
- Whisper `large-v3` in fp16 is about 3 GB and `qwen3:30b` in Q4 about 19 GB —
  both fit comfortably in the 5090's 32 GB, and Silero TTS deliberately runs on
  the CPU.
- `qwen3-vl:30b` is **loaded on demand**: it is another ~19 GB, so Ollama unloads
  the chat model, answers the screen question, then loads the chat model back for
  the next reply. That adds roughly **10–20 seconds** to the first question about
  the screen (and to the reply right after it). This is expected, not a bug.
- If you ask about the screen often and the wait annoys you, either use a smaller
  vision model (e.g. `llm.vision_model: qwen2.5vl:7b`) or a smaller chat model, so
  both stay resident.
- Watch it live: `nvidia-smi` and `ollama ps`.
- "CUDA out of memory" → smaller LLM, or `stt.model: medium`, or
  `compute_type: int8_float16`.

**Screen vision (`look_at_screen`) fails**
- Screenshots need an **active desktop session**. On the lock screen, with the
  session disconnected, or over a closed RDP session the capture fails and Jarvis
  reports it — log in on the room PC and try again.
- The model must be pulled: `ollama pull qwen3-vl:30b` and check `ollama list`.
- The first screen question after a restart is slow: the vision model has to load
  (see the VRAM note above). The server waits up to 120 s for the screenshot and
  the vision answer.
- Multiple monitors: only the **primary** screen is captured, downscaled to
  1600 px wide, JPEG quality 80.

**`run_command` returns nothing or times out**
- The timeout is **30 seconds** — anything longer is killed and comes back as a
  failure. Do not ask for commands that wait for input or run forever; anything
  interactive will hang until the timeout.
- Output is merged stdout+stderr and truncated to 4000 characters.
- The command runs as the logged-in user on the **room PC**, without elevation.
  Anything that needs admin rights fails with an access-denied message.
- It is a real shell: the model can do real damage with it. Say what you want
  precisely, and check `data\dialogs\<today>.jsonl` to see what actually ran.

**`open_app` opens the wrong thing / cannot find the app**
- The index comes from `Get-StartApps`. Check what Windows itself reports:
  `powershell -NoProfile -Command "Get-StartApps | Where-Object { $_.Name -like '*spot*' }"`.
- Use the name as it appears there. If the fuzzy match keeps choosing wrong, pin
  it with an `apps:` override in `config.yaml` (§6.2).
- `close_app` works by exe name; Store/UWP apps that cannot be mapped to a
  process return an error instead.

**Microphone: silence or the wrong input**
- List the devices:
  `python -c "import sounddevice; print(sounddevice.query_devices())"`.
- `client.audio.input_device` accepts an index (`3`) or part of a name (`"Yeti"`).
- Windows Settings → Privacy → **Microphone** → access for desktop apps must be on.
- Check the recording level in Sound → Recording → Properties → Levels.

**The wake word does not trigger**
- `client.wakeword.vosk_model` is relative **to the repo root** — start through
  `scripts\run-client.ps1`.
- The model directory must contain the `am`, `conf` and `graph` subfolders. If it
  contains a single nested folder, the archive unpacked one level deeper — move
  it or re-download: `scripts\download-models.ps1 -Force`.
- Add your own pronunciations to `phrases` (e.g. `[rowan ai, rowan a i, roan a i]`).
  All phrases must be written in **Latin letters** — the model is English.
- Triggers too often → drop the shortest variants, raise `vad.aggressiveness`.

**The phrase gets cut off, or never ends**
- Cut off → raise `vad.silence_ms` (e.g. 1200) and `pre_roll_ms`.
- Never ends (noisy room) → raise `vad.aggressiveness` to 3, lower `silence_ms`.

**CUDA / torch**
- Check: `python -c "import torch; print(torch.cuda.is_available())"`. `False` →
  reinstall torch with CUDA, or temporarily set `stt.device: cpu` and
  `compute_type: int8`.

**Ollama**
- `ollama list` — are both models pulled? `ollama pull qwen3:30b`,
  `ollama pull qwen3-vl:30b`.
- Is the API alive? `Invoke-RestMethod http://127.0.0.1:11434/api/tags`.
- The server uses the native API with `think: false`. If replies start arriving
  with visible reasoning text, check that `llm.provider` is `ollama_native` and
  `llm.think` is `false` — the `openai` provider cannot switch thinking off.
- If Ollama runs on another PC, put its IP in `llm.base_url` and set
  `OLLAMA_HOST=0.0.0.0` in that machine's environment variables.

**TTS is silent (there is text, but no voice)**
- The first run downloads the Silero model — internet required.
- Check `output_device` (see the device list) and that the audio goes to the TV
  or the speakers.
- If synthesis fails, the server still sends the text and an empty audio stream —
  look for the error line in the server log.

**Jarvis says something was done when it was not (or the other way round)**
- In v1.1 the server waits for a real `action_result` per action, so it should
  not happen. If it does, check the client log: an action that times out (35 s)
  comes back as `client timeout` and the model is told about it.
- Screenshots have their own 120 s budget.

**The assistant talks about devices that do not exist**
- With `devices: []` the system prompt tells the model there are none. Check that
  `client.devices` is really empty and that the client sent its `hello` (the
  server log prints the device list it received).

**Dependency installation errors**
- `webrtcvad` fails to build → `client/requirements.txt` must list
  **`webrtcvad-wheels`** (prebuilt Windows wheels), not `webrtcvad`.
- Install into the `jarvis` env, not the system Python:
  `& "C:\Users\Anton\anaconda3\envs\jarvis\python.exe" -m pip install -r client\requirements.txt`.
- "running scripts is disabled on this system" → run the scripts as
  `powershell -ExecutionPolicy Bypass -File scripts\...ps1`.

**A config error at startup**
- The message names the key directly, e.g.
  `client.server_url: required key is missing` or
  `client.vad.slience_ms: unknown key (typo? ...)`. Compare with
  `config.example.yaml`.

**It thinks for too long**
- Measure from the log or from `data\dialogs\<today>.jsonl`: STT → LLM → TTS.
  Usually the LLM dominates.
- Faster: a smaller model, `history_turns: 6`, `max_tokens: 512`,
  `stt.language: "en"` (no auto-detect), `stt.model: medium`,
  `max_tool_rounds: 2`.
- A screen question is a special case — see the VRAM note above.

---

## 12. Limitations of v1.1

- Speech is recognized after the phrase ends (not streaming), so the reply comes
  a couple of seconds later.
- One client = one session; several rooms at once have not been tested.
- `run_command` runs unelevated and is capped at 30 s; it is a full shell with no
  allow-list, so the persona in `prompts/system.md` is the only guardrail.
- Screen vision only sees the primary monitor, only on demand, and only when
  somebody is logged in on the room PC.
- Memory is a flat list of facts with no editing tool — you delete lines from
  `data\memory.jsonl` by hand.
- SwitchBot: no password support and no hub (BLE only, near the PC).
- IR strips and Bluetooth-only strips are not supported (a Wi-Fi controller is
  required).

---

## 13. Mass audit: what it is and how to run it

The mass audit (23.09.2026) sends every kind of request a person actually makes
through the same chain a live room uses — Jev reads the utterance once and
narrows the tool set, then the real model answers and the real tools run — and
writes one JSONL line per scenario. It is the check to run after touching the
prompt, the understanding step or a tool. Full results and the repair list are
in `docs/AUDIT_MASS.md`; the "before/after" family table is
`docs/AUDIT_MASS_AFTER.md`; latency is `docs/AUDIT_LATENCY.md`; the decisions
behind them are `DECISIONS.md` (AUDIT-01…AUDIT-19).

```powershell
$py = 'C:\Users\Anton\anaconda3\envs\jarvis\python.exe'   # the hub's own interpreter

# 1. the corpus: every family x subject x phrasing (1112 utterances today)
& $py scripts\gen-audit-scenarios.py            # -> data/audit/scenarios.jsonl

# 2. the live run: real Jev, real model, real tools (--workers 6, as asked)
& $py scripts\live-eval.py --scenarios data\audit\scenarios.jsonl `
    --workers 6 --jsonl data\audit\runs\last.jsonl --quiet

#    one family only (repeatable), or no Jev at all:
& $py scripts\live-eval.py --family browser --family noisy --workers 6
& $py scripts\live-eval.py --scenarios data\audit\scenarios.jsonl --no-understanding

#    the same corpus asked the way the Telegram chat asks it (AU-19): the real
#    TelegramController, the same Jev reading, the same model and tools
& $py scripts\live-eval.py --scenarios data\audit\scenarios.jsonl `
    --telegram --workers 6 --jsonl data\audit\runs\telegram.jsonl --quiet

# 3. the summary the report is written from
& $py scripts\audit-summary.py --runs data\audit\runs\last.jsonl --write docs\AUDIT_MASS.md

# 4. the offline matrix: the same utterances checked without a model
& $py -m pytest tests\audit -q

# 5. latency: bench reports + the live hub's own turn trace + its log
& $py scripts\audit-latency.py --runs data\audit\runs\last.jsonl --write docs\AUDIT_LATENCY.md

# 6. the real room PCs: deploy the current client, then do 57 real actions there
pwsh -File scripts\update-room-pcs.ps1
pwsh -File scripts\room-audit.ps1
```

Three rules keep the numbers honest: the verdict separates the model's choice
from the family narrowing (`offered` is written to every line, so "the wrong
tool" and "the tool was never shown to the model" are different failures), a
scenario the bench cannot test prints `SKIP` with a reason instead of passing,
and a pair of requests counts as done only when **both** halves were called.
The run of 23.09.2026 closed at **1071 of 1077** judged scenarios (99.4 %)
against **812 of 1106** (73.4 %) in the first pass of the same night. What is
left is listed in `docs/AUDIT_MASS.md` ("Остаток"): two stable scenarios, three
that vary run to run, and one provider outage — none of them hidden.

`--telegram` came out of the same night (AU-19): the owner asked that the
Telegram chat have the same capabilities as the voice assistant, and the chat
route was only reachable by hand. The real `hub.telegram_control.
TelegramController` now runs against the bench room, reads the request with the
same Jev call as a spoken turn and hands the model the same narrowed tool set;
only the incoming message object is built by the bench. The first full run of
the 1112-scenario corpus through that route is in `docs/AUDIT_MASS.md`
(`data/audit/runs/au-19-telegram.jsonl`).
