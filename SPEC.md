# Jarvis — Dorm Voice Assistant. Technical Specification v1

This document is the **contract**. All modules must match it exactly: message types,
config keys, module APIs, file ownership. If code and spec disagree, the spec wins.

## Persistent appearance and image references amendment (2026-09-20)

This amendment replaces the old 24-crop/day gallery retention and indefinite
body-track identity assumptions.

- `server.face.appearance_enabled` and `server.face.adaptive_recognition` default
  to `true`. `server/appearance.py` stores an indexed archive under
  `data/appearance/` using SQLite and UUID-named JPEG files. Accepted face and
  unambiguous body photographs are retained without automatic deletion. Active
  selection limits never remove archived files.
- Automatic admission requires a manual face-anchor match >=0.60, runner-up
  margin >=0.12, detector confidence >=0.80, minimum face side 64 pixels, and
  sufficient sharpness. Three consistent observations spanning >=1 second are
  required; captures are spaced >=60 seconds and redundant views are skipped.
  Explicitly confirmed face enrollment may save a quality-checked sample
  immediately. Ambiguous body ownership suppresses body capture. Neither track
  continuity nor learned vectors alone authorize new identity samples.
- Matching may use at most 12 diverse archived vectors per enrolled person in
  addition to manual samples. Every adaptive vector is revalidated against the
  current manual anchors; the recognition network itself is not retrained.
  Voice embeddings, roles and personal conversation histories are unchanged.
- A continuously observed body may retain its last confirmed identity for at
  most six seconds through a weak/turned face. This is a tracking hint, not a
  fresh recognition or enrollment. Contradictory clear faces, duplicate identity
  matches, ambiguous body binding, track replacement/jumps and expiry clear the
  carried identity. Camera receipt timestamps prevent delayed inference from
  reviving an older occupant. Clothing alone never assigns permanent identity.
- `generate_image.reference_people` is an optional array of at most two
  explicitly requested current face-profile names. Each supplies up to two
  selected references, at most four named images total, alongside the optional
  primary scene image. References are labeled by person and face/body kind.
  `source:none` permits a new composition with named identity references.
- `AppearanceGallery.references(name, limit=2, profiles=None)` optionally
  revalidates sample embeddings against current manual anchors. Image generation
  and `list_people` supply current anchors; an empty mapping accepts no people.
  At least one readable face reference is mandatory. Voice-only enrollment,
  missing images or conflicting/changed face anchors cause a clear failure before
  the paid provider call. A renamed/reused label never bypasses this validation.
  `list_people` reports availability of usable image references.
- Only explicitly selected images and the artwork prompt are uploaded for a
  generation request. Archived appearance does not prove current presence,
  position or clothing. Generated images never enter the face gallery. Renames
  and explicitly confirmed merges preserve photographs and historical labels.
  Existing provider handling, shared spending cap and one-attempt limit remain.

## Room reliability amendment (2026-09-20)

- Image generation separates `prompt` (artwork only) from `target: display|
  wallpaper` (default display). Wallpaper target transfers original pixels to
  the room PC with internal `set_wallpaper_file {image_base64}`, stores them in
  LocalAppData/Jarvis/wallpapers, calls Unicode Windows APIs and verifies the
  configured path. Public `set_wallpaper {source:generated|camera|screen,
  fresh?:false}` installs an existing image without regeneration. It needs the
  same trusted role as other PC/photo actions when role checks are enabled;
  the deployed open-access profile remains open to everyone.
- Generation results expose an opaque `image_id`, `storage:brain`, `shown`, and
  `saved_on_client` instead of a brain path. Wallpaper success requires both
  `applied:true` and `verified:true`. The completion check rejects unverified
  image/save/open/wallpaper claims, allows one corrective tool round and then
  supplies a factual failure reply. Repair cannot start another paid generation.
  Failed generation cannot silently apply/show/save an older result that turn.
- With role checks disabled, a newly displayed anonymous image remains usable
  on that room connection for five minutes when the next voice is recognized;
  it takes precedence over an older personal image. Named users' image archives
  remain isolated. Enabling permissions disables this anonymous fallback.
- `utterance_start.verify_wake` is an optional boolean. The room client sets it
  for a normal wake-word recording, but not continuation or interruption turns.
  After STT the server checks for a supported Rowan spelling in the first six
  spoken words (contractions count as one), or after a bounded prefix made only
  of short interjections. Quoted mentions do not confirm a wake. Repeated directly
  addressed silence commands tolerate interjections, but not negation or reported
  commands. Enrollment/name/face selection and silence commands keep their existing
  handling. An unconfirmed wake sends `transcript {ignored:true,...}` followed by
  `tts_end`, with no LLM, spoken reply or personal-history entry. Request audio
  and diagnostic logs remain archived. The overlay removes the live transcript.
  Older clients omitting this field keep their existing behavior.
- Replacing an unknown person's camera track ID cannot bypass the unknown
  greeting cooldown. An increase in the number of unknown people can still
  trigger a greeting; existing quiet-room and conversation gates apply.
- Generic browser focus/maximize/minimize uses visible browser processes. One
  candidate is selected automatically; multiple candidates require a choice.
  These operations do not resolve the literal word "browser" via Start-menu
  entries or launch a new application.
- The deployed room settings use VAD aggressiveness 2, 1100 ms silence,
  25 s maximum request and 1500 ms pre-roll. YOLO uses `camera.fps: 0` (no
  inference rate cap), with actual throughput reported in camera diagnostics.
- While microphone DSP is active, the client logs raw and processed RMS/peak
  levels every 1000 frames (30 s). The diagnostic retains counters only, not
  ambient audio, and does not alter wake sensitivity or speaker identity.

## Room requests amendment (2026-09-18)

This amendment supersedes older permission, memory, app-closing and TTS rules.

- Live `look_at_camera` and `find_object(source=camera)` are available to every
  role, including unknown. Screen access, file saving and PC actions still use
  their existing role gate. SAM3 detects requested objects; local Qwen vision
  describes food and the room. Neither is an identity or authorization source.
- ECAPA defaults: recognition 0.40, runner-up margin 0.15, privileged actions
  0.65. These are initial thresholds, not calibrated probabilities or protection
  against replay. Existing voice/face enrollment requires the matching owner at
  the higher threshold; voice samples are checked again on every enrollment turn.
  Recovery command: `Rowan, update my voice`. Low-confidence refusals mention it.
- `remember` defaults to the recognized speaker's personal memory. Only admin
  can write global memory, with the higher voice threshold. Nobody can write to
  a different person's personal memory. Records retain timestamp, author,
  optional stable `key` and `value`. Global keyed preferences override personal
  preferences deterministically; the prompt also gives global prose priority.
  Effective memory is refreshed every turn, including across connections.
- `server.llm.history_turns` defaults to 25 complete exchanges per recognized
  person. Each request is archived before execution and its answer before TTS;
  interrupted requests retain a status marker. Older exchanges remain in SQLite.
  `recall_conversation(query,since?,until?,limit?)` searches the entire current
  person's archive, including Unicode, returning timestamps and author. Unknown
  voices never inherit someone else's history. Dates are local ISO dates/times.
- `pc_control(open_app/close_app)` is resolved server-side against the room
  client's real inventory. New internal action `app_action` accepts
  `{operation:inspect,action:open|close,name}` or `{operation:execute,target_id}`.
  Open lists installed apps, close lists visible windows grouped by executable.
  Multiple apps require a per-person choice expiring in 180 s; a single match
  executes immediately. Global then personal `apps.browser` applies to opening,
  never to closing several browsers. Each completed browser action offers to
  remember the choice. The client checks the actual launched window and uses
  WM_CLOSE with pid revalidation; unsaved-work dialogs are never force-dismissed.
- New public server tool `save_photo(source,fresh?,filename?,open?)` captures or
  reuses a frame. The internal client action `save_photo_file` carries
  `jpeg_base64,filename,open`, saves to the Windows Desktop with a unique name,
  and reports `saved,path,opened,error`. Image data is omitted from logs. Screen
  capture continues to require an overlay-hide acknowledgement.
- Local TTS optionally uses `server.tts.engine=kokoro`, speaker `am_michael`,
  language `en`: American English male, CPU ONNX, 24 kHz resampled to the wire
  rate. Additional §6 keys: `kokoro_model_path` defaults to
  `models/kokoro/kokoro-v1.0.onnx`; `kokoro_voices_path` defaults to
  `models/kokoro/voices-v1.0.bin`. Install `server/requirements-tts.txt` and run
  `scripts/setup_kokoro.py`. Silero remains supported for existing installations.
  The deployed OpenAI profile uses Kokoro; no paid speech API is involved.
- Multi-speaker behavior is unchanged: word timestamps are attributed using
  diarized spans, and only the uniquely addressed turn is used. Overlapping
  speech is not source-separated and prompts clarification without execution.

## Personal room experience amendment (2026-09-17)

This amendment supersedes older enrollment, shared-history, interruption and
camera-cadence descriptions below. Full behavior is specified in
`docs/ROOM_EXPERIENCE.md`.

- §3: `server/conversations.py` persists person-scoped exchanges in SQLite.
  Each turn resets the connection's rolling context and loads only the recognized
  speaker's recent history. Unknown speakers have no persistent shared history.
- §4 client messages: `interrupt_request` asks for cancellation confirmation;
  `utterance_start.interrupt_id` identifies a confirmation recording. It cannot
  cancel another task. Work continues while a confirmation is pending.
- §4 quiet stop: `dismiss {id}` stops the current turn, control prompts, voice/
  face registration and greetings; `dismissed {id}` acknowledges that all old
  producers have stopped. There is no TTS or follow-up recording. The client
  mutes locally before sending, drops stale speech/actions until acknowledgement
  and a fresh wake word, and the server suppresses greetings until a new turn.
  Local finalized English/Russian silence commands are checked only during work
  or playback. Normal wake words retain the cancellation-confirmation behavior.
- §4 server messages: `chat {person, messages:[{id,ts,question,answer}], question,
  selection_active}` replaces the display's personal history. `notice {text,id}`
  is a spoken status/confirmation. `tts_start` and `tts_end` optionally carry
  `purpose: reply|notice|cancelled`; a notice end does not finish the active task.
  `say.enrollment_sentence` carries the reading prompt and enables the client's
  temporary 4-second silence / 45-second maximum registration recorder.
- §4 camera: both camera state and aligned presence-frame headers may contain
  `tracks:[{id,box:[x1,y1,x2,y2]}]` (normalized coordinates, <=24 tracks).
  Track IDs are namespaced to the camera process. State heartbeats are sent at
  most every 0.4 seconds; presence uses one aligned frame, not a delayed burst.
  On-demand enrollment bursts remain supported.
- §4 binary streams: all server audio and image header/binary sequences share a
  lock. Capture waits for an acknowledged overlay hide and suppresses visibility
  changes until the screenshot is finished. Failed hide means no screenshot.
- §5: `recall_conversation {query}` returns only the current recognized speaker's
  archive; unknown speakers cannot read it. Every tool schema accepts optional
  `purpose`, a public status removed before action dispatch.
- §6 defaults: `server.face.greet_after_s = 0.35` (0 still disables greetings),
  `client.camera.face_check_interval_s = 0.5`. The deployed room profile uses
  `client.camera.fps = 8`; the general fps default remains 5. Quiet-room gates
  and per-visit greeting suppression still apply.
- §9: new modules `server/enrollment.py`, `server/face_registration.py`, and
  `server/room_state.py` own deterministic registration and scene tracking;
  `client/room-tracker.yaml` configures local BoT-SORT with appearance matching.
  `client/overlay_web/chat.html` is the dark, centered chat view. Appearance
  snapshots are bounded and never confer roles or identity for tool permissions.

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

v1.3 — **speaker recognition and roles**:
- Every utterance is voice-identified on the server (`server/speaker.py`,
  resemblyzer 256-d embeddings - ECAPA 192-d since v1.7 - cosine vs enrolled profiles in
  `data/voices.json`). The LLM sees the speaker as a transcript prefix:
  `[speaker: Anton | role: admin] <text>` (or `[speaker: unknown]`).
- Roles: `admin` > `trusted` > `user`; unmatched voices are `unknown`.
  Permissions are enforced SERVER-side in the tool executor, not by the prompt:
  `run_command` and `set_role` — admin only; `click_screen`, `look_at_screen`,
  `remember` and the power `pc_control` commands (open/close/minimize/maximize/
  focus_app, type_text, hotkey, sleep) — admin or trusted; volume/media/display
  `pc_control` commands, `set_light`, `set_switch`, `enroll_voice` and plain
  chat — everyone including unknown. A denied call returns
  `{"ok": false, "error": "permission denied: …"}` to the LLM.
- Enrollment: the server tool `enroll_voice {"name": str}` stores the CURRENT
  utterance's embedding; the first person ever enrolled becomes `admin`, later
  ones start as `user`. The connection then collects the next 2 non-empty
  utterances as extra samples (state on the connection; the transcript prefix
  carries `[enrollment: N sample(s) left for X]` so the model can guide the
  speaker). `set_role {"name": str, "role": "admin"|"trusted"|"user"}` (admin
  only) changes a role. Dialog-log entries gain `speaker` and `speaker_score`.

v1.4 — **camera, faces, presence** (the C920 on the room PC):
- **Unified people registry**: `data/voices.json` becomes `data/people.json` —
  `{"people": {name: {"role": str, "voice_embeddings": [[…]], "face_embeddings":
  [[…]]}}}`. `server/speaker.py`'s registry class owns the file and the roles
  (voice matching unchanged, key `embeddings` renamed to `voice_embeddings`);
  `server/face.py` (insightface buffalo_l via onnxruntime, CUDA when available)
  adds face detection + 512-d embeddings and cosine matching against
  `face_embeddings` with its own threshold.
- **Client camera service** (`client/camera.py`): OpenCV capture of the C920 +
  Ultralytics YOLO (yolo11n) at ~5–10 fps on the 3060 Ti. It reports STATE, not
  video: `{"type": "camera_state", "persons": int, "objects": {label: count}}`
  sent when the picture changes (debounced, ≥2 s apart). While at least one
  person is visible it also pushes ONE frame every `face_check_interval_s`
  (default 5 s): `{"type": "camera_frame", "id": "p<N>", "reason": "presence",
  "w": int, "h": int}` + one binary JPEG (largest side ≤1280). The server may
  also pull a frame with `{"type": "camera_request", "id": str}` (client answers
  like a screenshot, `reason: "request"`; `camera_error` mirrors
  `screenshot_error`). Camera failures must never break the voice pipeline.
- **Server presence tracker**: face-matches every presence frame and keeps
  `{name_or_unknown: last_seen}`; entries expire after `presence_ttl_s`. The
  system prompt gains a `{presence}` placeholder — "Present in the room:
  Anton (admin), 1 unknown person" or "(camera sees nobody)".
- **Greeting**: when an unknown face has been present ≥`greet_after_s` and no
  conversation is active, the server generates ONE short greeting through the
  LLM (offering voice enrollment once, per persona rules) and pushes it as an
  unsolicited `say` + `tts_start…tts_end` block. The client therefore reads the
  socket BETWEEN utterances too and plays such proactive audio only when idle.
- **Voice recognition model (v1.7)**: voices are embedded with SpeechBrain's
  ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 192-d, on the CPU) instead of
  resemblyzer, which could not separate two people on the room's webcam mic
  (same person 0.666, different people 0.659). `people.json` gains a top-level
  `voice_model`; voice vectors made by any other model are dropped on load (faces
  and roles are kept) and those people re-enroll their voice. A person is scored
  against the CENTRE of their samples, not their single closest sample; with two
  or more profiles the winner must lead the runner-up by `margin`, else the
  speaker is `unknown`. During enrollment a follow-up sample that does not match
  the person's own samples (below `threshold - 0.05`) is rejected as somebody
  else's voice instead of being filed under their name. Enrollment clips are kept
  as WAV files under `data/voices/<name>/` for future recalibration.
- **Per-person memory (v1.7)**: `data/memory.jsonl` records carry a `person`
  field. `remember {"fact": str, "about": str}` files a fact against one person
  (`about` is their name, or `"me"` for the current speaker) or against the room
  (`"room"`, or omitted); a fact phrased in the first person is attributed to the
  identified speaker even when the model forgets `about`. `Memory.facts()`
  returns ONLY room facts and those go in the system prompt; `Memory.facts(name)`
  returns only that person's, and they ride in the per-turn message prefix as
  `[about <name>: …]` next to `[speaker: …]` and `[room: …]` — so the system
  prompt stays byte-identical between turns and Ollama's prompt cache survives.
  A personal fact is never shown while somebody else is speaking.
- **`list_people` (v1.7)**: a server-side tool returning every enrolled person
  with their role and whether they are known by voice and/or face, plus the list
  of admins. Admin-or-trusted, because naming who holds admin tells a stranger
  whom to imitate. It exists because roles change at runtime and are never in the
  system prompt: asked who the admins were, the model answered from the
  conversation and was routinely wrong.
- **Scripted greetings (v1.7)**: a greeting is a fixed line chosen from
  `SCRIPTED_GREETING_*` and spoken immediately, not generated by the LLM -
  measured 3-5 s (once 61 s) passed between the face being recognised and the
  first word, which is too late to read as a greeting. The unknown-face script
  still introduces Rowan and asks for the name, and names the known people the
  stranger is standing with; variants rotate per connection. The exchange is
  still written into the session history. `server.face.greeting_llm: true`
  restores model-generated greetings.
- **Per-person greeting cooldowns (v1.7)**: cooldowns are tracked per person,
  not per room. The same stranger is greeted again after `greeting_cooldown_s`;
  somebody the server recognises by name is greeted again after
  `greeting_cooldown_known_s`. A stranger outranks a familiar face when both
  are due, and at most one greeting is spoken per `GREETING_MIN_GAP_S` so two
  people arriving together are greeted one after the other rather than at once.
  An utterance from an identified voice restarts that person's cooldown, so a
  greeting never interrupts a conversation with the person being greeted.
- **New server tools**: `look_at_camera {"query": str}` — pull a camera frame,
  answer through the vision model (same pipeline and permissions as
  `look_at_screen`). `enroll_face {"name": str}` — pull a frame, embed the
  LARGEST face, add it to the person (creating them like `enroll_voice`;
  everyone may enroll themselves). After voice enrollment completes, the model
  offers `enroll_face` ("look at the camera for a second").
- Config §6: `server.face.{enabled, threshold, presence_ttl_s, greet_after_s,
  greeting_cooldown_s, greeting_cooldown_known_s}`
  (true, 0.45, 30.0, 10.0, 300.0, 900.0) and
  `client.camera.{enabled, index, fps, model, face_check_interval_s}`
  (true, 0, 5, "yolo11n.pt", 5.0). Camera deps live in
  `client/requirements-camera.txt` (ultralytics, opencv-python; torch with CUDA
  installed separately) so the base client stays light; without them the client
  logs one clear warning and runs voice-only.

v1.6 — **UX polish: detections photo, smarter greeting/presence, longer voice
enrollment, renaming**:
- **Detections photo on the TV**: new protocol message
  `image_show {"type","id","w","h","title","ttl_s"}` (server -> client) followed
  by ONE binary JPEG. The client accepts it in BOTH idle and conversation mode
  (never mistaken for mic audio, TTS or a conversation message) and hands it to
  `client/viewer.py`: a borderless, always-on-top OpenCV window driven from ONE
  dedicated thread (`imshow`/`waitKey` — never from asyncio), auto-closing after
  `ttl_s` (default 60 s); a newer image replaces the current one. Without `cv2`
  it falls back to `data/last_detections.jpg` + `os.startfile`. After a
  successful `find_object` with `count > 0`, the server draws the boxes on the
  pulled frame (PIL, 3px rectangles + score labels, `#FF3355`), pushes it via
  `image_show` (title `"<target> - N found"`, ttl 60) and adds
  `"note": "the annotated photo is now on the room screen - mention it"` to the
  tool result. Camera pulls for `find_object` request FULL resolution:
  `camera_request` gains an optional `"full": true` (client skips the usual
  1280px downscale for that one pull).
- **Greeting asks the name, only for true strangers**: the greeting instruction
  now explicitly asks the person for their NAME. The greeting task additionally
  holds fire when a KNOWN voice spoke within the last 3 minutes on this
  connection, or a KNOWN face is currently present together with exactly one
  YOLO person — that "unknown" face is almost certainly the same
  not-yet-enrolled or badly-angled person, not a second stranger.
- **Presence reconcile**: `PresenceTracker` drops the unknown bucket once YOLO
  reports N persons and there are already >= N fresh NAMED labels — the same
  person at a bad angle was being double-counted, telling the owner "you and an
  unknown person" while they were alone.
- **Voice enrollment needs 10 seconds minimum**: `server/speaker.py` tracks
  per-enrollment total voiced seconds (estimate: pcm bytes / (16000*2) per
  accepted sample). Enrollment completes only once BOTH >= 3 samples AND
  >= 10.0 s total speech are collected (`MIN_ENROLL_SPEECH_S = 10.0`); the
  pending state in `server/app.py` stays open until then, each sample's
  transcript-prefix note reports progress ("about N more seconds of speech
  needed - ask them to keep talking"), and `say.listen_s` stays 12 while
  pending. A too-short sample still counts its seconds, but the note asks for a
  LONGER sentence instead.
- **Rename a person**: new server tool `rename_person {"old_name","new_name"}`
  — allowed when the requesting speaker IS `old_name` (self-rename, any role)
  or is admin (`server/speaker.py: check_permission`'s own tier, since it
  depends on WHO is speaking, not just their role); merges into an existing
  target profile (embeddings concatenated, higher role wins) or renames in
  place, and works mid-enrollment (updates the pending enrollment's name too).
  `enroll_voice` (and `rename_person`'s `new_name`) reject placeholder names
  (Guest/User/Friend) with a clear error; the persona is told to call
  `rename_person` immediately when someone gives their real name, and to never
  enroll anyone under a placeholder.

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
  `transcribe_detailed(...) -> Transcript` additionally returns word timestamps
  for local speaker attribution; the legacy tuple API remains unchanged.
- `server/diarization.py` — optional local Community-1 pipeline, loaded once at
  startup from `server.diarization.model_path`. Regular overlapping speaker
  tracks are preserved. Non-overlapping turns are decoded as separate Whisper
  crops; ECAPA matches clean intervals per cluster. A unique wake-addressed turn
  reaches the existing permission gate; other turns remain in local transcripts.
  A wake-only turn may bridge up to 3 s to the same speaker's next turn, with no
  intervening voice. In configured wake-word-only mode, an empty first ASR fragment
  of at most 1.25 s may be ignored if there is exactly one diarized voice and a
  later transcribed turn; recognized text and missing later fragments are never
  discarded this way. Overlap/uncertain attribution asks for clarification without
  tools or cloud inference; mixed recordings are excluded from voice enrollment.
  Enabled-but-unavailable is an error, never a fallback to mixed voice identity.
  See `docs/MULTI_SPEAKER.md` for setup, measured limitations and validation.
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
- `server/vision.py` — two entry points, both POST to Ollama native `/api/chat`
  with `model: cfg.llm.vision_model`, `stream: false`, `think: false`,
  `keep_alive` matching the chat model, no tools; timeout 120 s; on failure
  return an error value (the LLM sees it as the tool result — never raise):
  - `describe_screenshot(jpeg_bytes, query) -> str` — the prompt demands
    specific detail relevant to the query: window titles, video/list titles,
    button labels, visible text; explicitly forbids one-word summaries.
  - `locate_on_screen(jpeg_bytes, target, img_w, img_h) -> (x_norm, y_norm) | None`
    — asks for the click point of `target` as JSON `{"x": int, "y": int}` in
    image pixels (Qwen-VL grounding); parse JSON first, fall back to the first
    two integers via regex; clamp to the image, normalize by `img_w`/`img_h`.
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
0. `{"type": "room_speech"}` — optional local idle VAD heartbeat (at most once per second while speech is detected). No audio or transcript follows. Defer proactive greetings for five seconds after the latest heartbeat. Shares the camera/audio wire lock; update both peers together.
1. `{"type": "hello", "client_id": str, "devices": [{"name": str, "type": str, "area": str|null, "description": str|null}]}` — sent once after connect. `devices` built from client config (§6); empty list is normal.
2. `{"type": "utterance_start", "sr": 16000, "format": "pcm_s16le", "channels": 1}`
3. binary frames: raw PCM s16le mono 16 kHz chunks
4. `{"type": "utterance_end"}`
5. `{"type": "action_result", "id": str, "ok": bool, "error": str|null, "output": str|null}` — REQUIRED after executing each action, in order. `output` carries data the LLM needs back (e.g. `run_command` stdout, truncated to 4000 chars); null when there is none.
6. `{"type": "screenshot", "id": str, "format": "jpeg", "w": int, "h": int, "screen_w": int, "screen_h": int}` followed by exactly ONE binary frame with the JPEG bytes — reply to `screenshot_request`. `w`/`h` are the (downscaled) image dimensions, `screen_w`/`screen_h` the real desktop resolution — the server needs both to translate vision-model pixel coordinates into normalized screen coordinates. On capture failure: `{"type": "screenshot_error", "id": str, "error": str}` and no binary frame.

Server → Client:
1. `{"type": "ready"}` — reply to `hello`.
2. `{"type": "transcript", "text": str, "language": str}` — after STT.
   Local diarization optionally adds `segments: [{start, end, speaker_id, speaker,
   text, uncertain}]` and `clarification: str`. Times are seconds from the start
   of the utterance; anonymous IDs are scoped to that utterance. `text` contains
   only the selected addressed turn. Overlap events have null `speaker_id`, empty
   text and `uncertain: true`. An ambiguous recording sends a scripted `say` and
   TTS after the transcript, without LLM/tools/enrollment. Segments also appear in
   the local dialog log. Other speakers' text is never injected as commands.
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

Fifteen tools exposed to the LLM (see ``server/tools.py`` for the current list). Execution matrix: `set_light`, `set_switch`,
`pc_control`, `run_command` are CLIENT actions (forwarded as protocol action
items, args verbatim; result = the client's `action_result`). `look_at_screen`,
`click_screen` and `remember` are SERVER-side (never forwarded verbatim;
`click_screen` internally produces a `mouse_click` client action — see below).

1. **`set_light`** — control a light device (LED strip etc.).
   `{"device": str (device name from config), "state": "on"|"off", "brightness": int 1–100 (optional), "color": "#RRGGBB" (optional)}`
2. **`set_switch`** — control a SwitchBot-style physical button pusher.
   `{"device": str, "action": "on"|"off"|"press"|"toggle"}`
   (Bots in press mode treat "toggle"/"on"/"off" as a single press.)
3. **`pc_control`** — control the room PC (the client machine itself).
   `{"command": "volume_set"|"volume_up"|"volume_down"|"mute"|"unmute"|"media_play_pause"|"media_next"|"media_prev"|"display_off"|"display_on"|"sleep"|"open_app"|"close_app"|"minimize_app"|"focus_app"|"type_text"|"hotkey"|"scroll", "value": str|int|null}`
   `minimize_app` minimizes all top-level windows of the named app (resolved
   like `close_app`; works for UWP apps too via window enumeration by process;
   console/terminal aliases minimize the shell windows hosting the client).
   `focus_app` (v1.2.1) restores + foregrounds the app's main window so
   subsequent `type_text`/`hotkey` reach it; refuses the console aliases.
   Safety: closing hotkeys (ctrl+w, alt+f4, …) are refused while the focused
   window is the client's own console (it once closed itself that way).
   `value`: `volume_set` int 0–100; `open_app`/`close_app` an app name — resolved
   by the client's installed-app index (§8 apps.py): `cfg.client.apps` overrides
   first, then fuzzy match over ALL installed apps; unknown → error result naming
   the closest candidates. `type_text` types `value` as unicode text into the
   focused window; `hotkey` presses a combo given as `"ctrl+shift+t"`-style string.
   `scroll` (v1.7) turns the real mouse wheel over the window under the cursor:
   `value` is a direction with an optional amount (`"down"`, `"up"`, `"down 5"`,
   a bare signed number). Empty means down by `DEFAULT_SCROLL_NOTCHES` (3);
   the amount is capped at `MAX_SCROLL_NOTCHES` (30) per call.
4. **`run_command`** — run an arbitrary PowerShell command on the room PC.
   `{"command": str}`. Client executes `powershell -NoProfile -Command <command>`
   with a 30 s timeout, captures stdout+stderr, truncates to 4000 chars, returns it
   in `action_result.output`. Non-zero exit → `ok: false` with output still set.
5. **`look_at_screen`** — see the room PC's screen. `{"query": str}` — what to look
   for/answer. Server-side: request screenshot from client, run `server/vision.py`,
   tool result = the vision model's answer text. The vision prompt demands
   SPECIFIC detail (titles, names, visible text, list items), not a one-word
   summary.
6. **`click_screen`** — click something visible on the screen.
   `{"target": str (visual description, e.g. "the search box at the top" or
   "the GO button"), "button": "left"|"right"|"double" (optional, default left)}`.
   Server-side pipeline: screenshot → `vision.locate_on_screen` asks the vision
   model for the pixel coordinates of the target (JSON `{"x":…,"y":…}`; tolerant
   parsing with a numbers-regex fallback) → normalize by the screenshot's `w`/`h`
   → send the client a **`mouse_click`** action:
   `{"tool": "mouse_click", "args": {"x_norm": float 0–1, "y_norm": float 0–1, "button": "left"|"right"|"double"}}`
   (client multiplies by `screen_w`/`screen_h`, moves the cursor, clicks).
   Tool result: ok, or an error saying the target was not found.
7. **`remember`** — save a fact to persistent memory. `{"fact": str}` — one
   self-contained English sentence. Server-side: `Memory.add`, result `{"ok": true}`.

Tool descriptions (in `server/tools.py`) must tell the model: reply in English;
keep spoken replies to one–two short sentences (they are read aloud — no markdown,
no code); use `look_at_screen` when the user asks about what is on the screen;
use `remember` when the user shares a lasting fact or asks to remember; device
tools only for devices in the system prompt's list (may be empty).

## 6. Config

Permanent training source archive: `cfg.server.training_archive.enabled` defaults
to `false`, `path` to `data/training_archive`, and `min_free_gb` to `5`. When enabled,
all processed requests (including rejected/cancelled turns), accepted enrollment
source samples and processed camera observations are retained by local date and
person, with detected unknown faces under `unknown/face-<permanent-id>` and
observations without faces under plain `unknown`. There is no automatic retention
deletion. `server/face_identity.py` persists dataset clusters locally using existing
buffalo_l embeddings (cosine >=0.62, margin >=0.08, detection score >=0.80).
Frame assignment is one-to-one; ambiguous/weak faces get isolated provisional
IDs and cannot update established templates. IDs survive dates and restarts.
`face_identities/profiles/<id>/profile.json` and `events.jsonl` group profile data
and image references across dates. Backfill links existing face events without
rewriting media or canonical event JSON. No face cluster grants roles, changes
voice identities, personal memory, or automatic greetings. Raw observations do
not automatically become trusted recognition vectors.
Passive face-profile capture admits no more than 50 observations per rolling
60 seconds for a face ID or its camera track. Storage admission time and an
atomic SQLite transaction enforce this across dates, cameras and restarts.
Unreliable untracked faces share a conservative camera fallback allowance.
Active voice/Telegram requests and enrollment bypass this passive cap, while
recognition still runs between saved samples. Original-frame recording remains
separate; microphone streaming temporarily reserves the wire, with local YOLO
recording continuing. No lifetime count limit or automatic deletion is added.

Single `config.yaml` at repo root (copy of `config.example.yaml`). `common/config.py` API:

```python
from common.config import load_config
cfg = load_config(path)      # path to yaml; returns Config
cfg.server.host              # "0.0.0.0"
cfg.server.port              # 8765
cfg.server.stt.{model, device, compute_type, language}          # language: str | None
cfg.server.stt.allowed_languages                                # ["en","ru","es"]: auto-detect whitelist, [] = all
cfg.server.diarization.enabled                                  # false until local model setup
cfg.server.diarization.model_path                               # "models/speaker-diarization-community-1", relative to repo
cfg.server.diarization.device                                   # "cuda" | "cpu", default cuda
cfg.server.diarization.timeout_s                                # 45, range 5..120
cfg.server.diarization.min_identity_s                           # 1.5, range 0.8..10: minimum clean sample for ECAPA
cfg.server.llm.{base_url, model, api_key, temperature, max_tokens, history_turns}
cfg.server.llm.{provider, think, vision_model, max_tool_rounds} # "ollama_native"|"openai"|"openai_responses", bool, str, int
cfg.server.llm.api_key_env           # "OPENAI_API_KEY": environment-only key for openai_responses
cfg.server.llm.monthly_budget_usd    # 18.0, >0 and <=20; local UTC-month SQLite accounting
cfg.server.llm.max_input_bytes      # 64000, 4096..128000; text request limit before sending
cfg.server.llm.vision_base_url      # null: legacy uses base_url; cloud defaults vision to local Ollama
cfg.server.llm.prompt_file          # null: prompts/system.md; cloud profile uses prompts/cloud.md
cfg.server.llm.verify_actions       # true for legacy; false in budgeted cloud profile
cfg.server.llm.vision_keep_alive   # v1.7: "10m" - the VISION model's own keep_alive; it holds
                                   # ~8.4 GB resident and SAM3 needs that memory
cfg.server.llm.{keep_alive, num_ctx}                            # v1.1.1: "4h" (Ollama keep_alive), 16384 (requested context; v1.7 - the
                                                                #   system prompt + tool schemas alone are ~7.6k tokens)
cfg.server.telegram.enabled                                    # false: fixed-group Telegram transport
cfg.server.telegram.chat_id                                    # null: configured negative group ID; no tool-selected recipients
cfg.server.telegram.control_user_id                            # null: strict positive int sender ID; only this account can use tools in group/own DM
cfg.server.telegram.api_key_env                                # "TELEGRAM_BOT_TOKEN": environment-only bot token
cfg.server.telegram.timeout_s                                  # 30, range 5..120
cfg.server.telegram.respond_to_mentions                        # false: answer new explicit mentions in the configured group
cfg.server.telegram.poll_timeout_s                             # 25, range 1..50
cfg.server.tts.{engine, language, model_id, speaker, sample_rate}   # English default: language "en", model_id "v3_en", speaker "en_0"
cfg.server.speaker.{enabled, threshold, min_speech_s}           # true, 0.40 cosine, 0.8 s minimum audio
cfg.server.speaker.{margin, admin_threshold}                    # 0.15 lead over runner-up, 0.65 for privileged operations
cfg.server.face.greeting_llm                                     # v1.7: false - greet from a script, not a model round
cfg.server.face.greetings_enabled                               # true: false silences proactive greetings while tracking stays active
cfg.client.server_url        # "ws://192.168.x.x:8765/ws"
cfg.client.client_id
cfg.client.workplace_name    # friendly Telegram label, max 80 chars; empty -> client_id
cfg.client.camera.name       # friendly camera label, 1..80 chars; one camera per client
cfg.client.camera.model      # local YOLO weights (default yolo11n.pt)
cfg.client.camera.fps        # 0 = no software cap; fresh capture frames only
cfg.client.wakeword.{word, phrases, vosk_model}                 # phrases: list[str]
                                                              # default word: "rowan ai"; room aliases include "rowan a i" and "rowanai"
cfg.client.audio.{input_device, output_device, sample_rate}     # devices: int|str|None
cfg.client.vad.{aggressiveness, silence_ms, max_utterance_s, pre_roll_ms, min_speech_ms}
cfg.client.attention_mode    # "wake_word" (default) | "window" (explicit legacy opt-in)
cfg.client.followup_window_s # float 0..30, default 0; used only in window mode
cfg.client.thinking_sounds   # bool, true: soft blips while a reply takes > ~1.5 s
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

Budgeted profile (`scripts/configure_openai.py`, output `config.openai.yaml`):
`openai_responses` calls the official `/v1/responses` endpoint, text only,
`gpt-5.4-mini`, reasoning `none`, standard service tier, `store=false`, no
automatic HTTP retries. Credentials never come from YAML. The model is restricted
to the reviewed price table; max output is <=2048 tokens (profile uses 600).
Each tool round reserves estimated input + maximum output cost transactionally
in `data/api_usage.sqlite3` before sending. Actual usage settles the reservation;
timeouts, missing usage, and crashes retain it. No new requests if insufficient
allowance or accounting is unavailable. This ledger covers Jarvis only; other
applications using the same API project are outside its accounting. Input-byte
estimates are conservative estimates, not an exact billing guarantee.

Exact local PC shortcuts go through `Connection._execute_tool` and its usual
permission gate, report the real result, and skip LLM generation/self-check.
TTS sends one start/end pair per answer and synthesizes sentence groups between
audio sends. In wake_word mode, neither `say.listen_s`, enrollment nor proactive
greetings may open an unattended follow-up window: every new utterance needs a
wake word. Old clients need updating for this behavior. Window mode remains
available only as an explicit opt-in.

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
  7. if `followup_window_s > 0`: after playback, wait 2 s (`FOLLOWUP_ECHO_GUARD_S`, so the
     tail of our own reply is not heard as speech), then run VAD listening for up to that
     many seconds; if speech detected → go to step 3 (skip wake word); else back to step 2.
- `client/audio.py` — `sounddevice`. Input: 16 kHz mono int16 blocks of 480 samples
  (30 ms). Output: playback of raw PCM at given samplerate; also `play_beep(freq, ms)`.
- `client/wakeword.py` — Vosk `KaldiRecognizer(model, 16000, json.dumps([*phrases, "[unk]"]))`
  grammar mode; model dir from `cfg.client.wakeword.vosk_model`. Feed 30 ms blocks;
  detection = any configured phrase appears in a final or partial result; after
  detection reset recognizer. `phrases` defaults to `[word]` if empty.
  The default address is Rowan AI. Explicit configured phrases replace legacy
  short-name activation; confidence must cover all words in a multiword phrase.
  Only after local confirmation, the server may recover logged Whisper
  substitutions Roman AI/Ruin AI (also A I) at the beginning of the request.
  This does not add local wake aliases or alter the original transcript. The AI
  suffix stays mandatory for Rowan AI configuration; legacy bare Rowan config
  alone permits the recorded Roman substitution. Noise and non-address mentions
  still cannot supply this recovery.
- `client/vad.py` — `webrtcvad.Vad(cfg.aggressiveness)` on 30 ms frames; utterance
  ends once the trailing `silence_ms` window is ≥90% non-speech frames (sporadic
  false positives from a noisy mic must not reset the tail) or at `max_utterance_s`; returns
  the recorded bytes (or None if no speech at all within a lead-in timeout of 5 s).
- `client/ws_client.py` — `websockets` library wrapper: connect, send json/binary,
  async iterate messages, reconnect loop.
- `client/screen.py` (v1.1) — `capture_jpeg() -> Capture`: full primary-screen
  screenshot via Pillow `ImageGrab.grab()`, downscale to max width 1600 px,
  JPEG quality 80. `Capture` carries `jpeg: bytes`, `w`/`h` (image dims after
  downscale) and `screen_w`/`screen_h` (real desktop resolution) for the
  screenshot header (§4). Runs in `asyncio.to_thread`. Raises with a clear
  message on failure (caller converts to `screenshot_error`).

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
    error for UWP apps it cannot map to a process;
  - `minimize_app` (v1.2): resolve the app name, find the process ids (by exe
    basename, or by matching `Get-Process` main-window titles as a fallback),
    enumerate its top-level windows (`EnumWindows` + `GetWindowThreadProcessId`
    via ctypes) and `ShowWindow(hwnd, SW_MINIMIZE)` each visible one; error if
    no window was found;
  - `mouse_click` action (v1.2, produced by the server's `click_screen`):
    `x_norm`/`y_norm` floats 0–1 → `SetCursorPos(int(x_norm*screen_w),
    int(y_norm*screen_h))` + `SendInput` button events; `button` left/right,
    `"double"` = two left clicks ~120 ms apart. Routed by the dispatcher as its
    own tool name, not through `pc_control`.
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
# Room interaction update: local audio, browser and network camera

This amendment extends §§5–6 without changing WebSocket framing. The client
still sends 16 kHz, mono, signed 16-bit, 30 ms audio frames.

`client.audio.echo_cancellation` and `noise_suppression` default to `false`;
`noise_suppression_level` defaults to `1` (0–3). Optional
`client/requirements-audio.txt` enables WebRTC AEC3 and noise reduction. A
bounded worker preprocesses the microphone before all consumers (wake, VAD,
speaker enrollment/recognition, STT). WASAPI captures the selected playback
endpoint as the echo reference; frames are matched by capture timestamps,
not TTS queue order. No automatic gain boost and no continuous audio files.
Missing reference disables AEC with a log warning; DSP failure preserves raw
microphone operation. External TV audio not rendered by this PC is not a known
reference. Physical room cancellation and double-talk must be checked in situ.

New client tool `browser_control` requires trusted/admin, like other computer
interaction. Commands: `navigate`, `read`, `click`, `fill`, `press`, `back`,
`scroll`; optional `url`, `ref`, `text`, `key`, `direction`, `submit`, `browser`,
`window_ref`, and common `purpose`. The production client controls an existing
ordinary Chrome/Edge window via Windows UI Automation and checked window-local
keyboard actions. It never starts a separate profile or debugging browser.
`browser` selects a reported app name; `window_ref` selects a reported opaque
window choice. Ambiguous applications require selection; stale refs fail before
mutation. `fill` with `submit=true` fills and submits the same input; `press`
without a ref acts on the focused control only in the verified browser window.
Successful explicit browser/window selection returns the shared optional
`remember_offer`; a subsequent user request saves `apps.browser` through the
existing personal/global memory permissions, without automatic saving.
It returns bounded visible page text and element references, excluding password
values and the Rowan HUD. `purpose` updates chat progress; page content remains
untrusted. Closing/cancelling the controller releases only its own automation
work, preserving the user's browser. The legacy Playwright implementation in
`client/actions/browser.py` is retained for explicit isolated test fixtures and
is not instantiated by the production dispatcher.

Exact stale/missing-reference and changed-element failures get one recovery
instruction per user turn and at most two extra tool rounds beyond the normal
cap. A separate page read must precede a retry; reading alone does not confirm
the failed action succeeded. Timeouts and access errors do not trigger this
reserve, and the existing cloud budget still governs every completion.

`client.camera.stream_url` defaults to `null`; an RTSP(S) URL overrides the USB
`index`. USB `width`/`height` default to 1920/1080 and may request up to 7680/4320;
RTSP retains the camera's stream resolution. Network open/read timeouts are
4/2.5 seconds; connection failures retry every 2 seconds, clearing stale frames
after disconnect. One configured camera is supported. The URL stays in local
configuration and is not included in protocol messages or application errors.

## Standalone client distribution

The public client is exported with `scripts/export_client.py` from an explicit
runtime allowlist and templates under `distribution/client/`. No server code,
provider configuration, active YAML, credentials, recordings, profiles or Git
history are included. The build checks credential patterns, local Python
dependency closure and the ZIP's actual contents, and writes a SHA-256 manifest.

`common/client_config.py` owns the client models and shared `RecordingConfig`;
`common/config.py` reexports them to preserve the server API. Client startup
uses `load_client_config`, accepts legacy combined YAML, and discards its
server section without importing brain/provider modules. Configuration errors
do not echo submitted values.

The standalone Windows installer uses Python 3.11/3.12 and a local `.venv`.
`client.setup` collects the server address, microphone and optional camera,
generates a unique installation ID, preserves existing settings, and downloads
only the Vosk wake model. Cloud API keys remain server-side. Public defaults
disable local frame recording; the existing room configuration is unchanged.
This change does not introduce public server exposure or invitation/auth tokens.

## Camera image-edit targeting

`look_at_camera` includes `faces_in_frame`, `face_positions_available`,
`frame_id` and the normalized coordinate convention. Face names and boxes come
from matching the exact requested photo. Recent presence labels cannot assign
positions in that photo. Duplicate matches to the same profile are marked
ambiguous. For one uniquely matched requested target, unknown bystanders alone
do not require clarification. Full-resolution camera pulls update the cached
photo, allowing `generate_image source=camera fresh=false` to edit exactly the
inspected image. Provider restrictions and the one-attempt limit are unchanged.

## Literal image prompt and clarification contract

The current accepted final STT transcript is authoritative for the image's visual
wording. `server/image_prompt.py` only removes narrow leading wake/capture phrases
and unambiguous trailing PC/delivery instructions. Wallpaper installation,
saving/opening and Telegram delivery remain separate actions. It preserves
negations, captions, actual wallpaper/Telegram artwork and later visual details;
ambiguous wording is retained. The assistant must not invent style, emoji,
objects or substitutions, including changing recognized "head" to "hat".

The outbound image prompt is logged as actually submitted. Provider canvas
geometry follows the primary scene's aspect ratio, never face/body
reference crops. Explicit numeric or worded output-format requests take
precedence without rewriting the prompt. The desktop viewer fits images with
uniform scaling and padding, raises the newest image in its own window, and
releases its topmost window before opening a saved file in Windows Photos.
Additional provider
text is limited to validated technical identity metadata: matched scene name,
normalized face box/requester flag and requested reference-image name/kind.
Metadata must not introduce creative prose. A pending literal image request is
eligible for a short clarification only after an assistant clarification
question, for the same recognized person, within 180 seconds. It must not be
inherited by an anonymous or different speaker or replayed after unrelated work.
Showing/installing/sending an existing result never implies a new generation.

Wallpaper installation additionally requires `wallpaper_change_requested` to
confirm a positive instruction in the current accepted user transcript. Earlier
requests, pending creative wording, saved preferences and LLM tool arguments
cannot supply this authorization. Negations, quoted/discussed commands and edits
to an image's own background do not authorize a Windows change. A generated
image defaults to display-only without that instruction, and the common native
application path enforces the same condition for direct installation tools.

## Fixed-group Telegram transport

`server.telegram` contains `enabled:false`, `chat_id:null`, `control_user_id:null`,
`api_key_env:TELEGRAM_BOT_TOKEN`, `timeout_s:30`,
`respond_to_mentions:false` and `poll_timeout_s:25`. `chat_id` accepts one negative
numeric group ID. The optional positive strict integer `control_user_id` is an
account ID, independently checked by router and tool executor. Send methods accept
only a guarded `private_reply_to_user_id` matching that configured account;
model arguments cannot choose arbitrary destinations. The encrypted
brain-side token is loaded through `set-telegram-key.bat` and the normal server
launcher. Credentials never enter YAML or the room client.

`server.telegram.TelegramProvider` exposes async `send_text`, `send_image`,
`check_connection`, `get_me`, `get_updates`, `get_webhook_info`, `download_photo`
and `close`; `ready` describes configuration readiness only. `check_connection`
uses only `getMe` and `getChat`. Send results contain `ok`, `chat_id`,
`message_id` and `kind`; an acknowledged matching group/message is required for
success. `TelegramError` exposes a sanitized message, optional numeric `code`
and `retry_after`, and an `uncertain` flag. Network/API URLs containing tokens
are not logged, redirected, or returned. The HTTPS transport performs no send
retries, including after rate limits or a failed photo upload.

Text/captions use no parse mode and are bounded to 4096/1024 UTF-16 code units.
Image bytes are validated as static PNG/JPEG/WebP. Suitable PNG/JPEGs use a photo
upload (<=10 MB, width+height<=10000, aspect ratio<=20); other supported pictures
use a document upload up to 50 MB. This selection precedes network activity and
preserves supplied bytes. Oversized text is rejected rather than split. Optional
`reply_to_message_id` uses a reply within the selected authorized group/DM route.

Mention mode answers mentions and replies to a message whose sender ID matches
this bot, only in the configured group. It uses one persistent group conversation
with attributed authors/timestamps and the latest 25 exchanges. Older per-sender
Telegram rows are included chronologically without rewriting or deletion. Up to
25 delivered background group messages can supply context without triggering a
reply; all context is bounded by the input budget. Telegram stays separate from
room identities. Other participants get text/image chat without room tools;
only the configured controller gets `TelegramController`'s isolated Connection
facade and existing tool executor. Controller private messages use a separate
durable history and media owner. Full sender ID and private chat ID must both
match; forwarded messages and bot senders cannot authorize actions. Room commands
use unique action IDs and existing receive futures, without replacing room voice
identity or history. An explicit room reservation prevents competing voice
actions; unavailable room transport does not disable brain-only chat/images.
Within that authenticated Telegram route, a direct request to take a room photo
or send/show an image can select the current conversation implicitly. It need
not include the word Telegram. This allowance is bound to the current literal
message and does not authorize quoted, revoked, historical, or third-party
requests; room voice sends still require their existing explicit destination.
Current-room identity questions capture a new camera frame before replying.
Names come only from face matching on that frame; fresh YOLO tracks supply
visible bodies, including people whose faces cannot be identified. Neither old
presence entries nor conversation history may supply current identities.
Requested frames briefly wait for the existing YOLO worker and attach the tracks
of the exact image being encoded, independently for each burst frame. If no new
processed frame arrives within 0.5 seconds, capture falls back to a fresh image
with `tracks:null`; a face-only count is then explicitly a lower bound.
The direct spoken identity question uses this local result without a language
or vision-model call; Telegram supplies it as a current tool observation.
Camera failure is reported as unavailable, never as an empty room.
Startup backlog is acknowledged
without replies; claimed work is never
automatically replayed. Existing webhooks are detected but not deleted. Photo
downloads use provider-issued paths on the fixed Telegram endpoint with an 8 MB
default bound and no token-bearing URL in model context. The transport's update
methods are never invoked by a connection health check.

`server.face.greetings_enabled` defaults to `true`; setting it to `false`
suppresses proactive greetings while face matching, room tracking and appearance
collection continue normally.

## Telegram owner panel and multi-workplace release (2026-09-20)

This amendment extends the earlier controller-only contract: `/tools` belongs
only to `server.telegram.control_user_id` in its DM or configured group. The
owner can grant explicit Telegram users `chat/images/camera/pc/memory/profiles`
capabilities. These checks remain active even when room voice permissions are
disabled. Other group members default to chat/images; DM requires explicit
access. Owner cannot be removed or demoted. Inline callbacks bind actor, chat,
message and panel generation; input is a reply to a distinct bot prompt.

`TelegramAdminState` persists permissions/settings/audit in SQLite; runtime
authorization and route reads use snapshots published after committed writes.
`AdminBackend` validates non-secret settings against `Config`; marked live keys
update engines, other keys apply at the next server start. Client settings are
read-only here and remain local YAML. Personal memory opened from the group is
shown in owner DM. Active profile deletion/reset preserves recording archives.

HELLO adds optional `workplace_name`, `camera_name`, and capability `camera_clip`.
No camera URL/index or cloud key enters this metadata. Each unique `client_id`
is a workplace with one camera. The owner selects it per Telegram chat. If
several are online without a selection, no automatic camera is chosen. An
offline selected workplace never falls back to a different room.

`camera_clip_request` server->client JSON carries `id`, `seconds` (3..10),
`fps` (5..10). Client replies with `camera_clip` JSON (`id`, `format:"mp4"`,
`bytes`, `w`, `h`, `seconds`, `fps`) immediately followed by one binary MP4
under the common send lock. Maximum 20,000,000 bytes; server WS limit is 24MB.
Failure uses `camera_clip_error` JSON. Even stale clip headers consume the next
binary frame as a clip, never audio. Recording runs off-loop using fresh existing
capture frames, at most 100 frames and 960px long side; no second YOLO inference.

Presence rules default disabled. Explicit owner enable allows entry alerts for
any YOLO person, unknown face or named fresh face match, scoped to a workplace
or all workplaces. Quiet hours, stability, absence and cooldown are enforced.
Destination is owner DM or configured group. Durable claims prevent replay;
ambiguous delivery failures are not automatically retried. Video is silent and
starts after the event (no pre-roll); photo/clip always uses the observed room.

`inspect_photo(query,target?)` is server-only, scoped to a current/replied
Telegram attachment. `target` invokes SAM3, otherwise saved face matching and
vision description. Annotation bytes do not enter LLM context. An uploaded
photo never establishes current room presence. Nano Banana uses the exact
attachment as reference. Captionless photo follow-ups retain a durable,
chat/sender/question-message-bound association.

The public client is generated by `scripts/export_client.py` from one explicit
source allowlist. Runtime bytes match the private development client and the
room deployment; SHA256 hashes are in `release-manifest.json`. No brain code,
keys, recordings, profiles, local configs or private Git history are published.
`scripts/publish_client.py --push` updates the approved GitHub repository with a
regular fast-forward. End-user `update-client.bat` preserves local data/config.
See `docs/TELEGRAM_ADMIN.md` and `docs/CLIENT_DISTRIBUTION.md` for setup and limits.

The owner panel uses English built-in text and readable status/profile/permission/
notification cards rather than serialized JSON. Stored names and memory content
are not translated. `/tools` shows connected computer names and a count; Computers
and cameras lists all known clients, online first, seven per page. Refresh reads
current connections; offline rows cannot take photos. Status reports all connected
workplaces correctly even when no single room can be selected automatically.
