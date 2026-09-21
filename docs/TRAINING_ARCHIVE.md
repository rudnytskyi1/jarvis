# Permanent person training archive

The brain stores a separate, dated dataset under `data/training_archive/`.
Existing request logs, recordings, registry files and appearance-gallery files
stay in place; this archive does not move, clean or replace their source data.
No age limit, lifetime sample-count limit or automatic deletion is used.
Passive face-profile capture admits at most 50 observations per rolling minute
per face ID; a shared YOLO track also limits uncertain provisional IDs. Active
requests and enrollment bypass that passive limit. Face recognition still runs
between saved observations. Every admitted observation can contain its original,
face crop and unambiguous body crop; 50 refers to source observations, not files.

```text
data/training_archive/
  index.sqlite3
  face_identities/
    identity.sqlite3
    profiles/
      face-<permanent-id>/
        profile.json
        events.jsonl
  2026-09-20/
    Anton--9bd3a17c20/
      profile.json
      events.jsonl
      events/
        210305-123456-8a1.../
          event.json
          request.wav
        210307-234567-9b2.../
          event.json
          original.jpg
          face.png
          body.png
    unknown/
      profile.json
      events.jsonl
      events/...
      face-<permanent-id>/
        profile.json
        events.jsonl
        events/<time-event-id>/
          original.jpg
          face.png
          body.png
```

Dates use the archive's selected timezone, or the brain PC's local timezone when
none is supplied. Every event also records an unambiguous UTC timestamp, local
timestamp and time of archival. Names are made safe for Windows paths and a
stable identity suffix prevents collisions. Unidentified observations use the
literal `unknown` folder. Detected faces now have separate `unknown/face-<id>`
subfolders. Frames or bodies without a detected face stay in plain `unknown`;
that shared folder is not treated as one real person.

## Persistent face IDs

Every detected face receives a local persistent ID. Reliable repeat sightings
reuse it across dates, cameras and server restarts. Matching uses the existing
`buffalo_l` face embeddings, with cosine threshold 0.62, competing-identity margin
0.08 and detector score 0.80. All faces in one frame are assigned together: two
simultaneous faces cannot receive one ID. Templates keep a fixed first anchor
and at most five validated views to limit gradual mixing between people.

Weak or ambiguous detections retain their own provisional IDs and candidate
scores, without changing established templates. Their images are still saved.
Clustering is an estimated dataset label, not proof of identity. It does not
enroll voices, grant roles, change personal chat history or trigger greetings.
Known names remain human-readable labels alongside the independent face ID.

`face_identities/profiles/<id>/profile.json` contains the ID, model, first/last
observation times, count, optional confirmed name, matching status and templates.
Its `events.jsonl` lists every linked event and the paths of its original images
and crops across all dates, including older files that remain in `unknown`.
Named person folders keep their existing layout and also include `face_id`.

To group pre-existing appearance/enrollment observations, run:

```powershell
python scripts/backfill_face_identities.py --dry-run
python scripts/backfill_face_identities.py
```

Backfill uses saved embeddings and full-frame grouping, without model inference
or API calls. It updates identity indexes and manifests; old event JSON and
images are never moved, rewritten or deleted. Repeated runs skip linked events.

## Saved data

- **Conversations:** the submitted request audio as mono signed 16-bit WAV at
  its received sample rate, recognized transcript, text reply, tool/action
  metadata, and any supplied recognition scores, segments, timing, source IDs,
  status or rejection/interruption reason. Empty or rejected transcripts can
  still have their corresponding audio archived. This is request audio, not a
  continuous microphone recording. WAV input can also be retained byte for byte.
- **Voice enrollment:** an original WAV for each supplied enrollment recording,
  with its sentence, acceptance/session information and profile snapshot when
  the caller supplies them. Keeping enrollment audio makes later manual training
  possible without trying to reconstruct a voice from its embedding.
- **Face enrollment:** the original supplied camera image plus a face crop,
  optional unambiguous body crop, face embedding/score/bounds and enrollment
  metadata. The original image is copied byte for byte.
- **Appearance observations:** each frame submitted by the processing pipeline
  can retain its original image and native-resolution face/body crops, including
  their normalized and pixel coordinates, tracking information, recognition
  source and confidence. Unknown body tracks can be recorded without a face.
  An ambiguous body association suppresses its body crop. Crops use lossless PNG
  without resizing, derived from the decoded original frame.
- **Profiles:** `profile.json` contains the latest supplied profile snapshot for
  that day's person folder, including manual face/voice embeddings, roles,
  model identifiers and other registry metadata supplied by the application.
  Each immutable `event.json` retains its own snapshot so later changes do not
  overwrite the historical record. Explicit credential fields are redacted.

Saving this raw dataset has no sharpness or diversity filter. Passive face
observations use the 50-per-minute limit described above, persisted by storage
admission time across cameras and restarts. Unreliable faces without a body track
share a conservative camera-level fallback allowance until they can be matched;
different reliable IDs retain independent allowances. Skipped samples never
delete earlier files. The separate original-frame archive is unaffected.
Those filters still belong to the separate curated appearance gallery used for
recognition and image references. The training archive saves the frames actually
passed to it by the server, not unseen frames between server camera samples.
Full-rate YOLO recording on the room client remains a separate archive.
During microphone streaming, that local archive continues recording detected
people; the protocol reserves its audio connection until the utterance ends.

Recognition labels are recorded observations, not manually verified training
labels. Keeping scores, identity source, raw frames and unknown examples allows
later correction without overwriting the original event. Tracking continuity
alone must not be reported as a fresh face confirmation by the caller.

## Files, identity and durability

`events.jsonl` is a readable sequence of complete event records. `event.json` is
the canonical individual record; its `files` map includes each asset's relative
path, byte count and SHA-256 hash. Each record has a unique ID. An explicit
`event_id` is idempotent within its event kind, preventing duplicate copies when
the same logical write is retried. A later annotation or corrected transcript
should be a new event referencing the old ID, not reuse an ID to replace history.

If the registry supplies a stable `profile_id`, the archive uses it. Otherwise
it creates and persists its own identity/alias mapping. `rename(old, new)` changes
the alias for future writes while old names, folders and event files remain
untouched. It never moves old directories. Recreated profiles can supply a new
stable ID even when reusing a display name. Unknown observations cannot be bulk
renamed into a named identity.

The named Rowan registry uses names and archive rename aliases. Face clustering
has a separate persistent ID that does not depend on the displayed name. Deleting
and recreating the same named registry entry still preserves its archive alias;
use a distinct name for a different person.

Every camera frame received by the brain also gets an unlabeled original event
under `unknown`, including frames skipped by a busy face matcher. A named
appearance event is an additional face/body label for that frame, not a move of
the original. The image SHA-256 and frame ID link these copies. Completed,
cancelled, partial and busy-interruption audio are retained; unfinished audio can
have an empty transcript. Ordinary room speech outside a request is not recorded
as a conversation.

SQLite transactions serialize writers across threads/processes. Event/media
files are written through atomic replacement after a complete temporary write;
profile snapshots are replaced atomically. Completed event files can recover an
interrupted index commit without duplicating the JSONL entry. An incomplete
JSONL tail is retained on its own line rather than mixed into the next record;
individual event files remain available for recovery.

The configured free-space reserve stops new writes and reports a failure. It
never deletes older data. There are no API uploads, model-training jobs or secret
store reads in this module. Metadata credential fields and recognizable API/bot
token strings are redacted; arbitrary binary values are not embedded in JSON.
Media is explicitly supplied as bytes, not copied from arbitrary path arguments.

## Integration API

```python
archive = TrainingArchive("data/training_archive", min_free_gb=5)
archive.conversation(name, pcm=pcm, sample_rate=16000,
                     transcript=text, reply=reply, actions=actions,
                     metadata=details, profile=profile, event_id=request_id)
archive.enrollment(name, "voice", pcm=sample, metadata=enrollment_details)
archive.enrollment(name, "face", jpeg=original, face=selected_face)
archive.appearance(original, name, face=face, row=body_track,
                   metadata=recognition_details)
```

Calls are blocking file operations and should run through `asyncio.to_thread`.
Keep a shared archive on the brain. Archive a conversation in the request's final
cleanup path, including rejection/cancellation; archive accepted enrollment
samples before their original bytes are discarded. Archive per-frame observations
before the curated `AppearanceGallery.observe` filter. Pass `person=None` for
unidentified people and `row['body_unambiguous']=False` for overlapping bodies.
For a live frame, call `assign_face_ids` once with all faces, then pass each result
as `appearance(..., face_identity=assignment)`. Frame identity includes the image
SHA-256 because client frame counters restart. `index_face_event` links old or new
immutable media to the cross-date face manifest and refuses conflicting links.

The generic `record(kind, person, metadata=..., assets={filename: bytes}, ...)`
supports additional event types and controlled legacy imports. It accepts safe
media/JSON/text filenames and never scans the rest of `data/` automatically.
`close()` needs no persistent connection cleanup; `saved` and `failures` count
write outcomes for operational health checks.
