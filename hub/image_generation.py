"""On-demand Nano Banana 2, with bounded I/O and the shared API ledger.

No retries, background uploads, provider safety overrides, or chat-history
uploads. Missing/uncertain billing retains the conservative reservation.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from common.config import ImageGenerationConfig
from hub.api_budget import ApiBudget, CloudUnavailable
from hub.image_prompt import person_reference_requested

log = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_TOKENS = 4096
MAX_INPUT_TOKENS = 131072
MAX_NAMED_REFERENCE_PEOPLE = 2
MAX_NAMED_REFERENCE_IMAGES = 4
MAX_REFERENCE_INPUT_BYTES = 16 * 1024 * 1024
MIMES = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp'}
# Gemini 3.1 Flash Image's documented output ratios. Canvas geometry follows
# the primary scene; the shapes of additional identity crops never set it.
ASPECT_RATIOS = ('1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1',
                 '4:3', '4:5', '5:4', '8:1', '9:16', '16:9', '21:9')
_OUTPUT_FORMAT = re.compile(
    r'\b(?:make|turn|convert|change|crop|resize|reframe|set)\s+'
    r'(?:(?:this|that|the|my)\s+)?(?:image|photo|picture|canvas|output|result|it|this)\s+'
    r'(?:(?:to|into|in|as|a|an)\s+){0,3}'
    r'(?:portrait|landscape|vertical|horizontal|square|widescreen)\b(?!\s+(?:of|style)\b)|'
    r'\b(?:use|choose|switch\s+to|change\s+to)\s+(?:a\s+)?'
    r'(?:portrait|landscape|vertical|horizontal|square|widescreen)\s+(?:format|orientation|canvas)\b|'
    r'\b(?:make|create|generate|render)\s+(?:a\s+)?'
    r'(?:vertical|horizontal|square|widescreen)\s+(?:image|photo|picture|canvas)\b|'
    r'\b(?:сделай|сделать|преврати|измени|обрежь|обрезать)\s+'
    r'(?:(?:это|эту|этот|мою)\s+)?(?:фото|фотографию|изображение|картинку|снимок|его|её|ее|это)\s+'
    r'(?:(?:в|под)\s+)?(?:вертикальн|горизонтальн|квадратн|широкоформатн|портретн|альбомн)\w*\b|'
    r'\b(?:сделай|используй|выбери|создай)\s+'
    r'(?:вертикальн|горизонтальн|квадратн|широкоформатн|портретн|альбомн)\w*\s+'
    r'(?:формат|ориентацию|фото|изображение|картинку)\b', re.I)


def _scene_aspect_ratio(width: int, height: int, prompt: str) -> str | None:
    """Use scene geometry unless the user explicitly supplies a supported ratio."""
    # Reuse the literal-mention guard: quoted captions and negated mentions
    # must not become provider settings any more than they become identities.
    requested = [f'{int(match[1])}:{int(match[2])}' for match in
                 re.finditer(r'(?<![\d:])(\d{1,2})\s*:\s*(\d{1,2})(?![\d:])', prompt)
                 if person_reference_requested(prompt, match[0])]
    requested = [value for value in requested if value in ASPECT_RATIOS]
    if len(set(requested)) == 1:
        return requested[0]
    if requested:
        # Multiple mentioned formats may describe a transformation. Preserve
        # the literal request without imposing a conflicting default setting.
        return None
    if any(person_reference_requested(prompt, match[0]) for match in _OUTPUT_FORMAT.finditer(prompt)):
        # "Make this image portrait/square" is an explicit canvas request but
        # supplies no exact ratio. Let the unchanged prompt choose the format;
        # forcing the source ratio would contradict it. A portrait of a person
        # alone describes the subject and does not override scene geometry.
        return None
    ratio = width / height
    # Log distance treats portrait and landscape reciprocal ratios equally.
    return min(ASPECT_RATIOS,
               key=lambda candidate: abs(math.log(ratio / (int(candidate.split(':')[0]) /
                                                            int(candidate.split(':')[1])))))


@dataclass(frozen=True)
class GeneratedImage:
    png: bytes
    jpeg: bytes
    width: int
    height: int


def decode_image(raw: bytes, mime: str) -> GeneratedImage:
    """Verify before decoding; bound dimensions and strip metadata for display."""
    if not raw or len(raw) > MAX_IMAGE_BYTES or mime not in MIMES.values():
        raise CloudUnavailable('The image has an unsupported format or size.')
    try:
        with Image.open(BytesIO(raw)) as probe:
            if MIMES.get(probe.format) != mime or probe.width * probe.height > 8_000_000:
                raise ValueError('invalid image dimensions or MIME')
            if getattr(probe, 'n_frames', 1) != 1:
                raise ValueError('animated images are unsupported')
            probe.verify()
        with Image.open(BytesIO(raw)) as opened:
            picture = ImageOps.exif_transpose(opened).convert('RGB')
            png, jpeg = BytesIO(), BytesIO()
            picture.save(png, format='PNG')
            picture.save(jpeg, format='JPEG', quality=95)
            if max(png.tell(), jpeg.tell()) > MAX_IMAGE_BYTES:
                raise ValueError('decoded image too large')
            return GeneratedImage(png.getvalue(), jpeg.getvalue(), picture.width, picture.height)
    except CloudUnavailable:
        raise
    except Exception as exc:
        raise CloudUnavailable('The returned image could not be decoded safely.') from exc


def _reference_inputs(reference: bytes | None, mime: str,
                      references: Sequence[Mapping[str, object]] | None) -> list[tuple[str | None, bytes, str]]:
    """Snapshot and bound explicitly selected references before decoding them."""
    inputs: list[tuple[str | None, bytes, str]] = []
    if reference is not None:
        if not isinstance(reference, bytes):
            raise CloudUnavailable('The primary image reference must contain image bytes.')
        inputs.append((None, reference, mime))
    if references is not None:
        if not isinstance(references, Sequence) or isinstance(references, (str, bytes, bytearray)):
            raise CloudUnavailable('Named image references must be a sequence of image records.')
        if len(references) > MAX_NAMED_REFERENCE_IMAGES:
            raise CloudUnavailable('Use no more than four named appearance reference images.')
        people: set[str] = set()
        for item in references:
            if not isinstance(item, Mapping):
                raise CloudUnavailable('Each named image reference must be an image record.')
            name, data, image_mime, kind = (item.get(key) for key in ('name', 'image', 'mime', 'kind'))
            if not isinstance(name, str) or not name.strip() or len(name) > 80 or any(ord(c) < 32 or ord(c) == 127 for c in name):
                raise CloudUnavailable('Each appearance reference needs a short, nonempty person name.')
            name = ' '.join(name.split())
            people.add(name.casefold())
            if len(people) > MAX_NAMED_REFERENCE_PEOPLE:
                raise CloudUnavailable('Use appearance references for no more than two named people.')
            if kind not in ('face', 'body') or not isinstance(data, bytes) or image_mime not in MIMES.values():
                raise CloudUnavailable('An appearance reference needs image bytes, a supported MIME and kind face or body.')
            # Technical identity labels only. Creative instructions come
            # exclusively from the user's prompt, including its exact wording.
            label = json.dumps({'name': name, 'kind': kind}, ensure_ascii=False,
                               separators=(',', ':'))
            inputs.append((label, data, image_mime))
    if any(not data or len(data) > MAX_IMAGE_BYTES for _, data, _ in inputs):
        raise CloudUnavailable('An image reference has an unsupported size.')
    if sum(len(data) for _, data, _ in inputs) > MAX_REFERENCE_INPUT_BYTES:
        raise CloudUnavailable('The combined image references exceed the 16 MB input limit.')
    return inputs


def _scene_subject_label(subject: Mapping[str, object] | None, reference: bytes | None) -> str | None:
    """Validate exact-frame target coordinates; add no creative instructions."""
    if subject is None:
        return None
    if reference is None:
        raise CloudUnavailable('Scene subject metadata needs a primary image.')
    if not isinstance(subject, Mapping) or set(subject) != {'name', 'face_box', 'is_requester'}:
        raise CloudUnavailable('Scene subject metadata must contain only name, face_box and is_requester.')
    name, box, requester = subject['name'], subject['face_box'], subject['is_requester']
    if (not isinstance(name, str) or not name.strip() or len(name) > 80
            or any(ord(character) < 32 or ord(character) == 127 for character in name)):
        raise CloudUnavailable('Scene subject metadata needs a short, nonempty person name.')
    if type(requester) is not bool:
        raise CloudUnavailable('Scene subject is_requester must be a boolean.')
    if (not isinstance(box, Sequence) or isinstance(box, (str, bytes, bytearray)) or len(box) != 4
            or any(type(value) not in (int, float) for value in box)):
        raise CloudUnavailable('Scene subject face_box must contain four normalized coordinates.')
    try:
        normalized = [float(value) for value in box]
    except (ValueError, TypeError, OverflowError) as exc:
        raise CloudUnavailable('Scene subject face_box must contain finite normalized coordinates.') from exc
    if (not all(math.isfinite(value) and 0 <= value <= 1 for value in normalized)
            or normalized[2] <= normalized[0] or normalized[3] <= normalized[1]):
        raise CloudUnavailable('Scene subject face_box must describe a positive box inside the primary image.')
    return json.dumps({'name': name, 'face_box': normalized, 'is_requester': requester},
                      ensure_ascii=False, separators=(',', ':'))


class ImageStore:
    """Keep originals and the latest result per recognized profile across restarts."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.database = self.folder / 'index.sqlite3'
        with sqlite3.connect(self.database) as db:
            db.execute('CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, owner TEXT NOT NULL, '
                       'created REAL NOT NULL, model TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS owner_images ON images(owner, created)')

    def save(self, owner: str, result: GeneratedImage, model: str) -> Path:
        image_id = uuid.uuid4().hex
        path = self.folder / (image_id + '.png')
        with path.open('xb') as out:
            out.write(result.png)
        with sqlite3.connect(self.database) as db:
            db.execute('INSERT INTO images VALUES (?, ?, ?, ?)', (image_id, owner, time.time(), model))
        return path

    def last(self, owner: str) -> tuple[GeneratedImage, float] | None:
        with sqlite3.connect(self.database) as db:
            row = db.execute('SELECT id, created FROM images WHERE owner=? ORDER BY created DESC, rowid DESC LIMIT 1',
                             (owner,)).fetchone()
        if row is None:
            return None
        if not re.fullmatch(r'[0-9a-f]{32}', row[0]):
            raise CloudUnavailable('The saved image index is invalid.')
        path = self.folder / (row[0] + '.png')
        if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
            raise CloudUnavailable('The previous generated image is unavailable.')
        return decode_image(path.read_bytes(), 'image/png'), row[1]

    def rename(self, old_name: str, new_name: str) -> None:
        with sqlite3.connect(self.database) as db:
            db.execute('UPDATE images SET owner=? WHERE owner=?',
                       ('person:' + new_name.casefold(), 'person:' + old_name.casefold()))


class ImageGenerator:
    def __init__(self, cfg: ImageGenerationConfig, *, ledger_path: Path,
                 monthly_usd: float, transport=None):
        self.cfg = cfg
        self.budget = ApiBudget(ledger_path, monthly_usd, model=cfg.model)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_s, connect=10),
                                      transport=transport, follow_redirects=False)
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return bool(self.cfg.enabled and os.environ.get(self.cfg.api_key_env, '').strip())

    def check_ready(self) -> None:
        if not self.cfg.enabled:
            raise CloudUnavailable('Image generation is disabled in the server configuration.')
        if not self.ready:
            raise CloudUnavailable('Nano Banana needs a Gemini API key. Run set-gemini-key.bat on the brain PC, '
                                   'then restart the server. Do not send the key in chat.')
        if self._lock.locked():
            raise CloudUnavailable('Another image is being generated. Please wait for it to finish.')

    def _settle(self, reservation: str, data: dict) -> None:
        try:
            usage = data['usageMetadata']
            incoming, candidates = usage['promptTokenCount'], usage['candidatesTokenCount']
            thoughts = usage.get('thoughtsTokenCount', 0)
            if any(type(n) is not int or n < 0 for n in (incoming, candidates, thoughts)):
                raise ValueError('invalid usage')
            # The candidate detail list does not include thinking tokens.
            details = usage.get('candidatesTokensDetails')
            image_tokens = None
            if details is not None:
                counts = {}
                for detail in details:
                    modality, count = detail['modality'], detail['tokenCount']
                    if modality not in {'IMAGE', 'TEXT'} or type(count) is not int or count < 0:
                        raise ValueError('invalid modality usage')
                    counts[modality] = counts.get(modality, 0) + count
                detailed = sum(counts.values())
                if detailed > candidates:
                    raise ValueError('modality usage exceeds output total')
                # Live Gemini responses can list only IMAGE tokens while the
                # candidate total also includes undocumented output tokens.
                # Settle using the valid total; charge any unclassified tokens
                # at the higher image rate instead of retaining the full cap
                # or assuming the cheaper text rate for unknown output.
                image_tokens = candidates - counts.get('TEXT', 0)
                if detailed < candidates:
                    log.info('Gemini usage: %d unclassified output tokens accounted at image rate',
                             candidates - detailed)
            self.budget.settle(reservation, incoming, candidates + thoughts, image_output_tokens=image_tokens)
        except Exception as exc:
            log.warning('Gemini usage not reconciled (%s); reservation retained', type(exc).__name__)

    async def generate(self, prompt: str, reference: bytes | None = None,
                       mime: str = 'image/jpeg', *,
                       references: Sequence[Mapping[str, object]] | None = None,
                       scene_subject: Mapping[str, object] | None = None) -> GeneratedImage:
        self.check_ready()
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 12000:
            raise CloudUnavailable('Use a nonempty image description shorter than 12 KB.')
        parts = [{'text': prompt}]
        image_config = {'imageSize': self.cfg.image_size}
        scene_label = _scene_subject_label(scene_subject, reference)
        sanitized_bytes = 0
        # Only the explicitly selected scene and named appearance examples are
        # uploaded, never the gallery itself or a live feed. The scene remains
        # the first inline image; each additional image has its own name label.
        for label, data, image_mime in _reference_inputs(reference, mime, references):
            verified = await asyncio.to_thread(decode_image, data, image_mime)
            if label is None:
                aspect_ratio = _scene_aspect_ratio(verified.width, verified.height, prompt)
                if aspect_ratio is not None:
                    image_config['aspectRatio'] = aspect_ratio
            reference_bytes = verified.png if image_mime == 'image/png' else verified.jpeg
            sanitized_bytes += len(reference_bytes)
            if sanitized_bytes > MAX_REFERENCE_INPUT_BYTES:
                raise CloudUnavailable('The decoded image references exceed the 16 MB input limit.')
            if label is not None:
                parts.append({'text': label})
            elif scene_label is not None:
                parts.append({'text': scene_label})
            parts.append({'inlineData': {'mimeType': 'image/png' if image_mime == 'image/png' else 'image/jpeg',
                                        'data': base64.b64encode(reference_bytes).decode('ascii')}})
        # Recheck after asynchronous validation: do not queue paid jobs silently.
        self.check_ready()
        async with self._lock:
            payload = {'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': {
                'candidateCount': 1, 'maxOutputTokens': MAX_OUTPUT_TOKENS,
                'responseModalities': ['IMAGE'], 'imageConfig': image_config,
                'thinkingConfig': {'thinkingLevel': 'MINIMAL', 'includeThoughts': False}}}
            try:
                # Reserve the model's full input limit, and all output at the
                # maximum image rate. Usage settles this down after completion.
                reservation = self.budget.reserve(MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS)
            except CloudUnavailable:
                raise
            except Exception as exc:
                raise CloudUnavailable('API accounting is unavailable; no image request was sent.') from exc
            try:
                async with asyncio.timeout(self.cfg.timeout_s):
                    async with self.http.stream('POST',
                            f'https://generativelanguage.googleapis.com/v1/models/{self.cfg.model}:generateContent',
                            headers={'x-goog-api-key': os.environ[self.cfg.api_key_env]}, json=payload) as response:
                        if response.status_code in {401, 403}:
                            raise CloudUnavailable('Google rejected the Gemini key or model access. Check the key and billing in AI Studio.')
                        if response.status_code == 429:
                            raise CloudUnavailable('Google image quota is exhausted. Check Gemini billing or try later.')
                        if response.status_code != 200:
                            raise CloudUnavailable(f'Google image generation failed (HTTP {response.status_code}). No image was returned.')
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > MAX_RESPONSE_BYTES:
                                raise CloudUnavailable('Google returned an image response that is too large.')
                        data = json.loads(body)
            except CloudUnavailable:
                raise
            except Exception as exc:
                log.warning('Gemini request failed (%s); reservation retained', type(exc).__name__)
                raise CloudUnavailable('Nano Banana could not finish the request. No image is ready; it was not retried.') from exc
            if not isinstance(data, dict):
                raise CloudUnavailable('Google returned an invalid image response.')
            self._settle(reservation, data)
            if (data.get('promptFeedback') or {}).get('blockReason'):
                raise CloudUnavailable('Google declined this image request. No image was created. Do not retry or rephrase it automatically.')
            candidates = data.get('candidates') or []
            if not candidates or candidates[0].get('finishReason') != 'STOP':
                raise CloudUnavailable('Google did not complete this image request. No image is ready; do not retry automatically.')
            for part in (candidates[0].get('content') or {}).get('parts', []):
                if part.get('thought'):
                    continue
                inline = part.get('inlineData')
                if inline:
                    try:
                        raw = base64.b64decode(inline['data'], validate=True)
                    except (ValueError, KeyError, TypeError) as exc:
                        raise CloudUnavailable('Google returned invalid image data.') from exc
                    return await asyncio.to_thread(decode_image, raw, inline.get('mimeType', ''))
            raise CloudUnavailable('Google returned no image. Do not claim the photo was edited or retry automatically.')

    async def close(self) -> None:
        await self.http.aclose()
