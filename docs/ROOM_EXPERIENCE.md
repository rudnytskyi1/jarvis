# Personal room experience

The room client uses local YOLO + BoT-SORT appearance association. Person track
IDs live only for the camera process, and never grant permissions. The brain
associates a face with a body only when exactly one body box contains that face.
A named body track can retain a recently confirmed identity for up to six seconds
through a turned head or sitting down. This continuity expires and never becomes
a new face-learning sample. Contradictory faces, ambiguous bodies and track jumps
clear the identity; a lost track needs fresh face confirmation. Clothing alone
never establishes a permanent identity.
Current capture still uses one camera. Cross-camera identity is not implemented.

Every 0.5 seconds the client sends one native-resolution JPEG of the **same
frame** used for tracking, with its normalized person boxes. Face matching warms
at startup. Unknown greeting stabilization is 0.35 seconds; the greeting loop
checks every 0.15 seconds. This removes the former 5-second sampling and
10-second waiting delays. Actual speech latency still includes inference, TTS,
playback and room-quiet gates; no latency guarantee is claimed.

`server.face.appearance_enabled: true` retains validated face photographs and,
when the face/body association is unambiguous, body crops in `data/appearance/`.
The SQLite index and UUID-named JPEG files survive restarts. Accepted photographs
have **no automatic deletion or age limit**. Older photographs remain archived
even when they are no longer selected for recognition or image generation.

Automatic capture requires a match against manually enrolled face samples of at
least 0.60, a lead of at least 0.12 over other profiles, detector confidence of at
least 0.80, and a sharp face at least 64 pixels across. Three consistent
observations spanning at least one second are required. Captures are spaced by
at least a minute and redundant views are skipped; a confirmed face enrollment
can save its quality-checked photograph immediately. Overlapping people do not
produce a body reference. A single uncertain recognition cannot teach a profile.

`server.face.adaptive_recognition: true` augments matching with up to **12 diverse
active appearance vectors per person**. Both settings default to true. Every
adaptive sample remains checked against the current manual enrollment; it cannot
authorize a chain of progressively mistaken identities. This is an expanding
gallery for the existing face model, not live retraining of its neural network.
It changes neither voice profiles nor roles. Re-enrolling, renaming or explicitly
merging a person preserves archived photographs; a name reused for another face
does not make the old photographs valid references for the new person.

For **“Rowan, make John stand next to me in this photo”**, the image tool can add
John's explicitly selected archived appearance to the current photo. John needs
an enrolled face and a usable saved face photograph; a voice-only profile is
insufficient. At least one valid face reference is required, with an optional
body reference. Historical clothing is not evidence of what someone wears now.
See [image-generation reference behavior](IMAGE_GENERATION.md).

## Registration

- Say **Rowan, register me. My name is Theodric.**
- Read the six prompted sentences, starting each with Rowan. Recording permits
  4-second pauses and up to 45 seconds for a sample. During registration only,
  clean segments of the same diarized voice are joined across pauses. Overlapping
  or multiple voices are still rejected. Rejected samples do not advance progress.
- Each accepted recording needs at least 2.5 seconds of voiced material; completion
  needs six accepted recordings totaling at least twenty voiced seconds.
- Say **Rowan, add more voice samples. My name is Anton.** or **Rowan, update my
  voice** to record six more samples for the same profile. The screen shows the
  sentence number. Samples commit only after the full session, with physical
  confirmation if the existing identity cannot be verified. The latest 30 voice
  samples are retained; names, roles, face samples and conversations stay intact.
- A request without a name asks for one. Placeholder names are rejected in voice
  and face storage. Asking “who am I” uses the recognized speaker and needs no
  permission to list other users.
- **Rowan, remember my face** starts face registration. If several faces are
  visible, the screen displays numbered boxes with position descriptions.
  **Rowan, number two** selects the reference face. A fresh frame must match that
  face before the first write; every later sample must match the same fixed
  reference with a clear margin. A larger face cannot take over registration.
- Say **Rowan, cancel registration** to end an unfinished voice/selection flow.
  Registration expires after three minutes without progress; face selection after
  90 seconds. The completed profile persists on the brain PC.

## History and chat

`server.permissions_enabled: false` temporarily opens all tools to everyone,
including unknown voices, without changing stored roles or recognition results.
It bypasses role/confidence checks for PC actions, shared memory and profile
updates. Speech attribution, recording quality and profile-merge confirmations
still apply. Personal history stays separated; an unknown speaker can name a
profile explicitly for recall or saving a personal note. Unnamed personal notes
are never silently stored as shared memory. Set the option to `true` and restart
the brain server to restore permissions.

`data/conversations.sqlite3` archives recognized speakers' exchanges separately.
Existing non-ambiguous known-speaker dialog logs are imported once. A turn loads
only that speaker's recent exchanges. Unknown voices start with an empty context;
another person's messages are never a fallback. No time-based expiration is used
for conversation history, so another person's five-hour conversation does not
evict an exchange from twelve hours earlier. The display loads the latest 30
exchanges; model context uses the configured recent count. The person-scoped
`recall_conversation {query}` tool searches older exchanges without loading the
whole archive into the API. An empty query reads recent exchanges.

The dark QtWebEngine overlay displays the current question, personal history,
answer, registration sentence and tool progress. Optional tool `purpose` is a
short public status (the display permits 180 characters), removed before dispatching action arguments. It is not
private reasoning. The window stays click-through and does not take app focus.
Every wake clears the previous person's display before voice recognition.
Registration reading prompts and numbered face selection remain visible during
that reset and until their reply windows expire; private chat history is cleared.

Before a screenshot, the client waits for the UI thread to acknowledge hiding,
keeps display updates suppressed, waits for the compositor, captures, then
restores the overlay in a `finally` block. No acknowledgement means no screenshot.
Windows capture exclusion is an additional best-effort protection.

## Interrupting a task

Saying Rowan during work does not cancel the task. It requests a spoken status
and cancellation confirmation while work continues. Say **Rowan, cancel the
task** or **Rowan, continue**. Confirmation is tied to the offered task and expires
after 45 seconds; a late answer cannot cancel a new task. For a recognized task
owner, cancellation requires that same recognized voice. Remaining server steps
are cancelled; a client action already executing may still finish.

Speech notices use their own stream purpose, so their end does not end the main
conversation. All outgoing binary image/audio streams share a lock to prevent
interleaving. No additional cloud session is used for registration, tracking,
greetings or interruption confirmation.

## Quiet stop

While Rowan works or speaks, say **shut up**, **stop talking**, **stop speaking**,
**be quiet**, **that's enough**, **замолчи**, **помолчи** or **хватит говорить**.
No wake word is required. This mutes output immediately after local recognition,
stops remaining task/registration steps, and returns silently to wake-word wait.
There is no spoken acknowledgement, follow-up question, or unsolicited greeting
until a new request. An action already running in native code may still finish.
This does not power off either PC or unload the assistant.

`client/voice_controls.py` reuses the English wake model with a separate, full
vocabulary Vosk recognizer; optional Russian recognition uses
`models/vosk-model-small-ru-0.22`. The room PC has both installed. The model is
available from the [official Vosk catalog](https://alphacephei.com/vosk/models).
Only complete explicit commands count, not partial hypotheses, negations or
sentences mentioning the command. Models can still mishear room audio; the
speech recognizer does not provide acoustic echo cancellation for the webcam.
# Audio, browser and camera update (2026-09-18)

- Optional WebRTC AEC3 uses Windows playback loopback, including browser sound
  on Rowan's output device. Noise reduction is mild, AGC is off. Processing is
  local and adds no API charges. It cannot use sound from a separate TV app as
  a reference. A hardware speakerphone should handle Rowan's input **and** output;
  disable software AEC if two cascaded cancellers degrade speech.
- `browser_control` uses the ordinary browser window and current profile via
  Windows UI Automation. It keeps the user's existing login and tabs, and does
  not create a separate Chrome profile or debugging window. Searches can use
  `fill` with `submit=true`; Enter without an element ref uses the focused
  browser input. Long actions retain chat status; page reads exclude the overlay.
- IP camera: enable RTSP in its app, create a camera account, put its local RTSP
  address in `client.camera.stream_url`. Reolink commonly uses
  `rtsp://USER:PASSWORD@IP:554/h264Preview_01_main` (check model/stream codec).
  Keep the camera and room PC on a network that permits direct local access.
  Leave automatic PTZ tracking off for a stable whole-room view. Network reads
  reconnect after failures. USB remains selected when `stream_url` is null.
- Hardware checkout scripts: `scripts/verify_audio_runtime.py` and
  `scripts/verify_browser_runtime.py`. The first only reports capture statistics,
  saves no audio and plays no test sounds. Real room echo performance still needs
  a spoken test with the actual speaker/microphone placement.
