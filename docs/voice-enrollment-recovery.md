# Voice enrollment and profile recovery

Say `Rowan, remember my voice`, then `Rowan, I'm Anton` when asked for a name.
Or use one request: `Rowan, my name is Anton. Update my voice.` Read each of
the three sentences shown on screen; pauses are allowed during recording.

Registration does not require the voice to be recognized beforehand. A single
voice's request split by pauses can start registration even when speech-to-text
omits the wake word. Mixed voices and overlapping speech still cannot supply
enrollment audio. All recordings stay local.

Samples are staged in memory until three usable recordings totaling at least
10 voiced seconds are collected. Cancellation, expiry or disconnect discards
the unfinished batch without changing the registry. Each sample is checked
against the other recordings in this attempt, not just the old voice profile.

For an existing name, every new sample must strongly match that profile and
lead competing profiles before automatic saving. Otherwise the room PC shows
an explicit Save/Cancel dialog. Choose Save only if that person just read all
sentences. Cancel, closing the dialog, a 45-second timeout or a missing display
leaves the old profile unchanged. Speaking a name or saying yes cannot approve
the update. The model has no confirmation tool, and the client finishes previous
automated input before opening the dialog. Screen captures are blocked while
the dialog is open. Roles, faces and conversation history are retained.

Validation: `python -m pytest tests`; the actual Qt dialog can be checked without
a visible window or profile writes using `python tests/live_voice_enrollment_smoke.py`
on a machine with the client's PySide6 installed.

## Correcting a name

Say `Rowan, change my name to Theodric` or, if the voice is not recognized,
`Rowan, rename Theodrik to Theodric`. Personal memories, conversation history,
face and voice samples are kept. Low-confidence changes and combining two
existing profiles require confirmation on the room PC. During registration,
the same command corrects the pending name without discarding the recordings.
The last explicit name correction wins; Rowan cannot be used as a new person's name.

## Text while speaking

The chat shows a provisional transcript and speaker name during recording.
The local diarization and Whisper models refresh it roughly every 1.2 seconds,
using a 12-second moving window. Actual delay also includes inference time.
Earlier words remain on screen; the final transcript replaces the draft.
Mixed/unclear speech clears the provisional identity. Previews never retrieve
personal history, save memories, enroll voices or execute commands.

`server.stt.live_transcript` disables/enables this optional preview;
`live_interval_s` and `live_window_s` control its cadence and window size.
No OpenAI requests are used for captions. Validate against the running server
with `python tests/live_caption_smoke.py` (local synthetic speech, no enrollment samples).
