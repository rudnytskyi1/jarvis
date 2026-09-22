# Nano Banana 2 in Rowan

Desktop wallpaper is a computer action, not image content. For "turn me into
Spider-Man and set it as the background", call `generate_image` with
`target: wallpaper` and an artwork-only prompt. This transfers original pixels
to the room PC and verifies the wallpaper path through Windows. Use
`set_wallpaper(source=generated)` to install an existing result without paying
for another generation. Failed installation retains the image for retry.
Results distinguish brain storage, room storage, display, opening and verified
wallpaper installation; a brain filesystem path must never be used on the room
PC. A completion guard rejects unsupported success claims. Failed generation
cannot use an older image as the requested new result in the same turn.

Changing the Windows wallpaper requires an explicit positive instruction in the
**current request**. A previous wallpaper request, saved preference, image-edit
history or a model-selected `target: wallpaper` cannot authorize it. Creating an
image or changing the background inside a photograph does not change the desktop.
Negated, quoted and discussed wallpaper commands are not installation requests.
Without current authorization, generation stays display-only; the shared
wallpaper application path must also reject an unauthorized installation.

## Literal request text

The accepted final speech transcript supplies the visual request. The chat
model cannot replace it with its own expanded prompt. Rowan removes only narrow
wake/capture prefixes and clear trailing workflow clauses such as setting the
desktop wallpaper, saving/opening the result or sending it to Telegram/the group
chat. These operations use separate tools or fields; they are not artwork.
Negations, visual constraints, quoted captions and ambiguous clauses remain.
An instruction followed by another visual detail is kept rather than silently
dropping that detail.

No emoji, sticker, cartoon style, composition advice or extra object is appended.
Speech-recognition errors are not silently rewritten: recognized "head" stays
"head" rather than being guessed to mean "hat". The actual submitted visual text
is recorded for diagnosis. The provider receives only technical metadata when
needed to bind a matched person/face box to the selected scene or label explicitly
requested face/body references; this metadata contains no creative instructions.

After Rowan asks an image clarification, the original literal request may remain
pending for **180 seconds**, bound to the same recognized person. A short answer
such as "the one on the left" can finish that request; another person's reply,
an expired request or unrelated conversation cannot reuse it. An existing image
can be shown, installed or sent without starting a new paid generation.

Image creation and editing use Google's `gemini-3.1-flash-image` at 1K.
Ordinary conversation still uses the configured chat model; cameras, wake word,
voice recognition and room tracking do not generate paid image requests.

## One-time setup on the brain PC

1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey).
   Google image generation requires a project with billing enabled.
2. Run `set-gemini-key.bat` in the project root. Paste the key into the hidden
   prompt. Never put it in YAML or a chat message. It is saved with Windows DPAPI
   for this Windows account, separately from the OpenAI key.
3. Set `server.image_generation.enabled: true` in the active config (already
   enabled in Anton's `config.openai.yaml`). Restart the brain server with
   `start-jarvis-openai.bat`. Close the old server before starting another.
   The room client reconnects; it needs no Gemini key or package installation.

The normal launcher loads the saved key without prompting. A missing Gemini key
does not interrupt voice chat. `/health` reports `image_generation: true` when
enabled and a key is loaded; this is configuration status, not a paid API test.

## Voice examples

- “Rowan, create a cinematic image of a raccoon astronaut.”
- “Rowan, take a photo of the room and turn it into a comic illustration.”
- “Rowan, add sunglasses to the person on the left in that photo.”
- “Rowan, make the background of the generated picture blue.”
- “Rowan, save the generated image on the Desktop and open it.”
- “Rowan, show my generated picture again.”
- “Rowan, make John stand next to me in this photo.”

`generate_image` takes `source: none | camera | screen | last`, a complete prompt,
and optional `fresh` for camera/screen. `fresh: false` uses the already captured
frame. An ambiguous person-specific edit should first ask who to edit.
`look_at_camera` matches faces on the exact captured full-resolution image and
returns `faces_in_frame` with names, normalized face boxes and image-relative
positions. Recent room presence cannot assign a position in another photo.
One uniquely matched target can be edited even when bystanders are unknown;
duplicate matches remain ambiguous. The inspected full-resolution frame is
cached, so `fresh: false` sends that same photograph to the image provider.
Only the explicitly selected frame, description and requested people's selected
appearance references are sent to Google. No continuous camera upload,
conversation history or voice recording is included.
Provider refusals are reported without safety overrides or automatic rephrasing.

## People remembered from earlier photographs

`generate_image` accepts optional `reference_people`, an array containing at most
**two explicitly requested enrolled names**. For each name, the brain selects up
to two valid local photographs: a face plus an optional unambiguous body crop,
or another face view. At most **four named reference images** accompany the
primary `source` image. Each image is labeled with its person's name and role as
an identity reference; the primary photograph remains the scene to edit.

The selected photographs are the person's **newest usable appearance** in the
gallery, and the body crop is taken from the same moment as the face whenever
that moment has one, so the person is generated in the clothes they were last
seen in. A weak frame (low identity score or blurred) is skipped in favor of the
newest frame that clears the bar; only an archive whose every frame is weak
falls back to the old quality ranking. The archive itself grows automatically
while a recognized person stands in front of a room camera, so a new haircut or
outfit becomes the reference after a few seconds of being visible.

For example, after inspecting the current photo, use `source: camera`,
`fresh: false`, `reference_people: ["John"]` and an artwork-only description to
place John next to the person already in that photo. The tool never uploads the
whole gallery or automatically adds everyone recently seen in the room. It also
supports named references with `source: none` for a new composition.

Names are resolved through current **face** enrollment, not just voice profiles.
`list_people` reports who has usable image references. Each selected sample is
revalidated against current manual face anchors, so a corrected or reused name
cannot silently select an old person's appearance. Missing enrollment or a
missing usable face photograph stops the request before the paid image call and
asks for a brief clear view or face enrollment. A remaining body crop alone is
insufficient. The face/body gallery is stored locally in `data/appearance/`
without automatic deletion. A selection limit never deletes archived images.

These images are dated appearance examples of the person's most recent saved
look; the photograph itself does not prove current room presence or position.
The generator receives only requested
people's references. Generated outputs never become new recognition samples.

Successful images appear on the room display and are stored as PNGs in
`data/generated_images/` on the brain PC. The last generated image belongs to
the recognized profile and survives reconnects and restarts. Unrecognized guests
share a temporary connection scope; they do not inherit a known person's image.
`show_photo which=generated` displays that result without another API call.
`save_photo source=generated` exports a high-quality JPEG to the room PC Desktop
using its existing photo action; the PNG original remains on the brain.
Generated pictures are never inserted into live room observations or face data.

## Cost and failure behavior

[Reviewed Google pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-image)
(2026-09-18): $0.50/M input tokens, $3/M text/thinking output, $60/M image output.
A 1K image uses approximately 1,120 image tokens: **$0.0672 for the image**,
plus input and thinking. Retrieving, showing and saving the same image are local.

Google and OpenAI requests share `data/api_usage.sqlite3` and the same
`server.llm.monthly_budget_usd` allowance, currently $18/month. Images do not get
an additional allowance. Before a call, $0.311296 is reserved conservatively
(the model's maximum input and the configured 4,096-token output cap at the
highest output rate). Verified usage reduces the reservation to actual cost;
missing or incomplete modality data is charged conservatively: output tokens
without a known modality use the higher image rate. Invalid totals retain the
full reservation. Near the limit a request may
be refused even when the likely image cost would fit.

There is one image attempt per spoken request, one active image job per server,
a 120-second timeout and no automatic retries. Cancellations/timeouts retain the
reservation because Google may have charged for the request. This is an app
allowance, not a billing limit for other users of the same Google/OpenAI projects.
