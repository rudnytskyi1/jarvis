# Speech quality update, 2026-09-19

Server changes:

- Task interruption asks only what is running and “Cancel?”. Unclear replies
  leave the task running without incorrectly blaming a different speaker.
- Single-voice requests are transcribed with the whole sentence's audio context.
  Multiple voices still get separate crops; their boundaries allow up to 300 ms
  of silence without crossing another detected voice.
- Voice identification retains quiet edges and short pauses. Padding is never
  counted as speech and never crosses a detected speaker/overlap boundary.
  Enrollment audio, saved profiles, thresholds and permissions are unchanged.
- Whisper receives the spelling hint `Rowan`, without conversation history or
  suggested commands. Input duration, RMS, peak and clipping percentage are
  included in dialog logs. These levels do not measure SNR or room acoustics.
- A brief overlapping wake word can retain the identity of the immediately
  following clean voice only when that voice spans the wake fragment too.
  Mixed audio is never used to compute the voice embedding or for enrollment.
- The extra noise filter keeps confidently decoded sentences of at least four
  words even when Whisper's no-speech score is high. Short noise and weak
  segments still reject the whole transcript, preserving uncertain negations.
  This fixes an observed Theodric fragment (4.62 s, logprob -.35/no-speech .79).

Validation reports (local audio, no cloud calls):

- `data/voice-quality-audit.json`: 16 enrollment recordings, each held out from
  its temporary reference centroid; full and shorter crops matched their labels.
  These are enrollment-session results, not an accuracy estimate for the room.
- `data/stt-hints-audit.json`: six recordings compared with/without spelling
  hints, plus silence. “grow and” became “Rowan” on one Anton recording.

Room PC update was installed and the client restarted on 2026-09-20. For another
installation, copy
`scripts/update_room_speech_config.py` to the same path on that PC, then run it
with its `miniconda3/envs/jarvis/python.exe`. It validates the config and saves a
backup before changing only VAD settings: aggressiveness at most 2, pause at
least 1100 ms, maximum utterance at least 25 s, pre-roll at least 1500 ms.
Longer existing settings are preserved. `--dry-run` performs no writes.
Restart `JarvisRoomClient` after applying it. The enrollment recorder keeps its
separate 4-second pause window. Microphone/output selections and AEC settings
must be verified on the actual room PC before changing them.

## Recording and camera throughput

The user requested indefinite storage; both retention limits are **0** and no
recordings are auto-deleted. Archive recording is opt-in for other installations.

- Brain: `data/request_audio/` contains WAVs before recognition, including failed
  recognition, interruption confirmations and incomplete disconnected requests.
  `index.sqlite3` stores UTC time, room ID and recognized speaker/text when known.
  Dialog JSONL also links the audio recording. Unknown identity stays unknown.
- Brain: `data/request_camera/` saves all frames explicitly requested from the
  room camera during conversation, even when no person/face is detected. The
  index links them to the original speaker, transcript, conversation and audio
  ID when available, with a server-receive timestamp. Received frames survive
  task cancellation or a later vision error. Presence pushes and PC screenshots
  do not enter this conversation-camera archive. Like audio, no auto-deletion.
- Room PC: `data/camera_frames/` contains native-size JPEGs of every processed
  YOLO frame with at least one `person`, independently of face recognition.
  The index stores UTC capture time, person count and track boxes. No frames
  are sent to cloud storage. Capture may outrun inference: unprocessed frames
  are not archived or retroactively classified.
- `camera.fps: 0` removes the inference rate limit. CUDA FP16 is enabled and each
  fresh frame is processed once. Capture no longer sleeps an extra 5 ms. Presence
  JPEG encoding and recording run outside the YOLO thread. A bounded disk queue
  applies backpressure instead of silently dropping detected-person frames.
- `data/camera-performance.json` and the client log report actual capture/YOLO
  FPS and saved/failed/pending frames. The room 3060 Ti measured 28–29 YOLO FPS
  with a 30 FPS camera on 2026-09-20. No people were present during that check,
  so throughput while archiving people is not yet measured. Disk/encoding speed
  can limit sustainable FPS.
- A 5 GiB free-space reserve stops new recording with explicit logged failures;
  it never deletes old recordings. Server `/health` reports audio save failures.

Run `scripts/configure_recording.py --target server` on the brain, then restart
the brain. The room deployment package includes both speech and camera changes:
`scripts/package_speech_recording_update.py` builds `data/speech-recording-update.zip`.
Copy it to the same room-PC path, copy `scripts/deploy-speech-recording-update.ps1`
to the room's `scripts` directory, then run that PowerShell script there. It
checks hashes/config, backs up files, preserves device selections, enables
recording with no auto-deletion, and restarts `JarvisRoomClient`.

## Room reliability, 2026-09-20

- The client logs raw/processed microphone RMS and peaks every 30 seconds from
  its actual DSP path, without retaining ambient audio. This distinguishes low
  capture level from processing attenuation on the next missed wake. The
  optional `scripts/inspect_wake_audio.py` also compares both wake paths locally;
  its 15-second diagnostic saves only levels and detection counts.
- Normal wake-word requests require Rowan in the final STT transcript. False
  background triggers finish silently without an LLM call or personal-history
  entry; the chat removes provisional captions. Audio remains archived.
  Enrollment, follow-ups and interruption handling preserve their own flow.
- Unknown-person track ID changes no longer bypass the greeting cooldown. A
  newly visible additional unknown person may still receive a prompt greeting.
- Generic browser focus/maximize/minimize resolves actual visible browsers:
  one runs immediately, multiple browsers require clarification.
- Live backend verification replayed a saved false trigger and confirmed no
  reply, then synthesized a normal request and received spoken "ready". This
  tests the server path, not physical microphone recognition. Interactive room
  diagnostics confirmed the selected onn USB microphone supplies frames and
  is unmuted; quiet-room levels do not establish speech recognition accuracy.
