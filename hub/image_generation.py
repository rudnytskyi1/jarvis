"""On-demand Nano Banana 2, with bounded I/O and the shared API ledger.

No background uploads, provider safety overrides, or chat-history uploads.
Missing/uncertain billing retains the conservative reservation.

Две дороги к одной модели (владелец, 2026-09-23: «для генерации картинок
теперь используй vertexai api (у меня бесплатные 300$ credits)»):

* ``provider: gemini`` - AI Studio, одна ``GEMINI_API_KEY``;
* ``provider: vertex`` - Google Cloud: проект, регион и короткоживущий
  OAuth2-токен из ``hub/vertex_auth.py``.

Различается ровно то, куда уходит запрос и чем он подписан; тело запроса,
учёт расхода, повторы и разбор ответа у двух дорог общие.
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
from typing import Any

import httpx
from PIL import Image, ImageOps

from common.config import ImageGenerationConfig
from hub.api_budget import ApiBudget, CloudUnavailable
from hub.image_prompt import person_reference_requested
from hub.vertex_auth import VertexAuth, VertexAuthError

log = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_TOKENS = 4096
MAX_INPUT_TOKENS = 131072
#: Gemini finished the request and produced no image. That is not a refusal:
#: measured on the owner's own photo edits, the SAME request produced the image
#: on the next try, and the identical prompt succeeded a minute later. One
#: automatic retry is therefore allowed - and only then is the failure reported.
NO_IMAGE_RETRIES = 1
NO_IMAGE_RETRY_DELAY_S = 1.0
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


async def _google_message(response: httpx.Response) -> str:
    """Google's own words about a failed request, short enough to show a person.

    Vertex answers a wrong project, a disabled API and an unbilled account with
    three different sentences in ``error.message``; repeating it beats inventing
    a diagnosis. Never raises: a body that cannot be read or parsed adds nothing.
    """
    try:
        raw = await response.aread()
    except Exception:  # noqa: BLE001 - a body we cannot read is not an error of ours
        return ''
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return ''
    error = parsed.get('error') if isinstance(parsed, dict) else None
    text = error.get('message') if isinstance(error, dict) else error
    text = ' '.join(str(text or '').split())[:200]
    return f': {text}' if text else ''


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


def _counted_tokens(value: Any, field: str) -> int:
    """One non-negative integer token count from a Gemini usage report."""
    if type(value) is not int or value < 0:
        raise ValueError(f'invalid usage field {field}')
    return value


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
        # Vertex is the owner's paid road (Google Cloud credits); the Gemini key
        # stays as the fallback configuration. Only one of the two is built.
        self.vertex = (VertexAuth(cfg, transport=transport) if cfg.provider == 'vertex' else None)
        self._lock = asyncio.Lock()

    @property
    def provider_label(self) -> str:
        """How the provider is named to the person, in their own sentence."""
        return 'Vertex AI' if self.vertex is not None else 'Google Gemini'

    @property
    def ready(self) -> bool:
        if not self.cfg.enabled:
            return False
        if self.vertex is not None:
            return self.vertex.ready
        return bool(os.environ.get(self.cfg.api_key_env, '').strip())

    def check_ready(self) -> None:
        if not self.cfg.enabled:
            raise CloudUnavailable('Image generation is disabled in the server configuration.')
        if self.vertex is not None:
            reason = self.vertex.missing_reason()
            if reason:
                raise CloudUnavailable(
                    f'Vertex AI image generation is missing {reason}. Nothing was sent. '
                    'See docs/VERTEX_IMAGE_GENERATION.md.')
            if self._lock.locked():
                raise CloudUnavailable('Another image is being generated. Please wait for it to finish.')
            return
        if not self.ready:
            raise CloudUnavailable('Nano Banana needs a Gemini API key. Run set-gemini-key.bat on the brain PC, '
                                   'then restart the server. Do not send the key in chat.')
        if self._lock.locked():
            raise CloudUnavailable('Another image is being generated. Please wait for it to finish.')

    def _settle(self, reservation: str, data: dict) -> None:
        """Charge what Google reported, including a request that made no image.

        A request that ends without an image is still billed, and a missing
        field is not a reason to hold the whole cap: the old code kept the FULL
        reservation whenever ANY field was absent, so 18 such requests held
        $3.92 of the $18 monthly allowance and image generation ran out of
        budget while it was working perfectly.
        """
        usage = data.get('usageMetadata') if isinstance(data, dict) else None
        if not isinstance(usage, dict) or not usage:
            log.warning('Gemini reported no usage metadata; the reservation stays as the upper bound')
            return
        try:
            incoming = _counted_tokens(usage.get('promptTokenCount', 0), 'promptTokenCount')
            thoughts = _counted_tokens(usage.get('thoughtsTokenCount', 0), 'thoughtsTokenCount')
            # The candidate detail list does not include thinking tokens.
            details = usage.get('candidatesTokensDetails')
            counts: dict[str, int] = {}
            if details is not None:
                for detail in details:
                    modality = detail.get('modality')
                    if modality not in {'IMAGE', 'TEXT'}:
                        raise ValueError('invalid modality usage')
                    counts[modality] = counts.get(modality, 0) + _counted_tokens(
                        detail.get('tokenCount', 0), 'tokenCount')
            reported = usage.get('candidatesTokenCount')
            if reported is None:
                # An answer with no picture can still omit the candidate total.
                # The input count IS known, so charge it exactly and the output
                # at the FULL image cap instead of holding a reservation built
                # from 131072 input tokens that were never sent: 18 such rows
                # held $3.92 of the $18 monthly allowance and image generation
                # looked broken while it worked perfectly.
                if 'promptTokenCount' not in usage:
                    raise ValueError('no token counts')
                self.budget.settle(reservation, incoming, MAX_OUTPUT_TOKENS,
                                   image_output_tokens=MAX_OUTPUT_TOKENS)
                log.info('Gemini reported input tokens only (%d); output charged at the image cap',
                         incoming)
                return
            candidates = _counted_tokens(reported, 'candidatesTokenCount')
            image_tokens = None
            if details is not None:
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
            self.budget.settle(reservation, incoming, candidates + thoughts,
                               image_output_tokens=image_tokens)
        except Exception as exc:
            log.warning('Gemini usage not reconciled (%s: %s); reservation retained',
                        type(exc).__name__, exc)

    @staticmethod
    def _reply_text(data: dict) -> str:
        """What Google said instead of making a picture, in its own words."""
        candidates = data.get('candidates') or []
        parts = (candidates[0].get('content') or {}).get('parts', []) if candidates else []
        return ' '.join(str(part['text']).strip() for part in parts
                        if isinstance(part, dict) and part.get('text')).strip()

    @classmethod
    def _retryable_no_image(cls, data: dict) -> str:
        """``'NO_IMAGE'`` when Google finished with no picture and no explanation.

        Measured on the owner's own edits: the identical request that answered
        NO_IMAGE produced the picture on the next try, and the same prompt
        succeeded a minute later. A safety answer (IMAGE_SAFETY, SAFETY,
        PROHIBITED_CONTENT) or an answer that came with words instead of a
        picture is never retried - it is reported exactly as it arrived.
        """
        if (data.get('promptFeedback') or {}).get('blockReason'):
            return ''
        candidates = data.get('candidates') or []
        if not candidates:
            return ''
        if str(candidates[0].get('finishReason') or '') != 'NO_IMAGE':
            return ''
        return '' if cls._reply_text(data) else 'NO_IMAGE'

    async def _send(self, payload: dict) -> dict:
        """One metered POST to the provider; the caller decides about a retry."""
        try:
            reservation = self.budget.reserve(MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS)
        except CloudUnavailable:
            raise
        except Exception as exc:
            raise CloudUnavailable('API accounting is unavailable; no image request was sent.') from exc
        provider = self.provider_label
        try:
            url, headers, params = await self._endpoint()
        except (VertexAuthError, KeyError) as exc:
            # Nothing was sent, so nothing is billed: the reservation stays until
            # the shared ledger is reconciled, exactly like a transport failure.
            log.warning('%s credentials are not usable (%s)', provider, type(exc).__name__)
            raise CloudUnavailable(str(exc) or f'{provider} credentials are not usable.') from exc
        try:
            async with asyncio.timeout(self.cfg.timeout_s):
                async with self.http.stream('POST', url, headers=headers, params=params,
                                            json=payload) as response:
                    if response.status_code in {401, 403}:
                        raise CloudUnavailable(self._credential_refusal(response.status_code))
                    if response.status_code == 402:
                        raise CloudUnavailable(
                            f'{provider} refused to bill this request (HTTP 402): the project or key '
                            'has no balance. Nothing was created. Check billing on the Google side; '
                            'rephrasing the request will not help.')
                    if response.status_code == 429:
                        raise CloudUnavailable(f'{provider} image quota is exhausted. Try later.')
                    if response.status_code != 200:
                        detail = await _google_message(response)
                        raise CloudUnavailable(
                            f'{provider} image generation failed (HTTP {response.status_code}{detail}). '
                            'No image was returned.')
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise CloudUnavailable(f'{provider} returned an image response that is too large.')
                    data = json.loads(body)
        except CloudUnavailable:
            raise
        except Exception as exc:
            log.warning('%s request failed (%s); reservation retained', provider, type(exc).__name__)
            raise CloudUnavailable('Nano Banana could not finish the request. No image is ready; it was not retried.') from exc
        if not isinstance(data, dict):
            raise CloudUnavailable(f'{provider} returned an invalid image response.')
        self._settle(reservation, data)
        return data

    async def _endpoint(self) -> tuple[str, dict[str, str], dict[str, str]]:
        """Where one image request goes and how it proves who it is."""
        if self.vertex is not None:
            return await self.vertex.authorization(self.cfg.model)
        return (f'https://generativelanguage.googleapis.com/v1/models/{self.cfg.model}:generateContent',
                {'x-goog-api-key': os.environ[self.cfg.api_key_env]}, {})

    def _credential_refusal(self, status: int) -> str:
        """What Google's 401/403 means on this road, in the owner's terms."""
        if self.vertex is not None:
            return (f'{self.provider_label} rejected the credential or the model access '
                    f'(HTTP {status}). Check that the service account may use Vertex AI in '
                    'server.image_generation.vertex_project and that the Vertex AI API is enabled '
                    'there. See docs/VERTEX_IMAGE_GENERATION.md.')
        return ('Google rejected the Gemini key or model access. Check the key and billing '
                'in AI Studio.')

    async def generate(self, prompt: str, reference: bytes | None = None,
                       mime: str = 'image/jpeg', *,
                       references: Sequence[Mapping[str, object]] | None = None,
                       scene_subject: Mapping[str, object] | None = None) -> GeneratedImage:
        self.check_ready()
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 12000:
            raise CloudUnavailable('Use a nonempty image description shorter than 12 KB.')
        # Панель владельца (/admin/turns): что именно уходит в модель картинок,
        # какие кадры приложены и с какими подписями. Отказ Google отличается от
        # отказа чат-модели, и по цепочке это видно.
        from hub import turn_trace

        turn_trace.record('prompt', 'image', payload={
            'provider': self.cfg.provider, 'model': self.cfg.model, 'prompt': prompt,
            'reference': bool(reference), 'references': len(list(references or ()))})
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
            # The same request that comes back empty usually works on the next
            # try (the identical edit "Add kiss marks around anton" answered
            # NO_IMAGE at 20:24 and produced the picture here in the live check),
            # so one retry is allowed. A real refusal is never retried: it
            # carries promptFeedback.blockReason and is reported as it is.
            data = await self._send(payload)
            reason = self._retryable_no_image(data)
            for _attempt in range(NO_IMAGE_RETRIES if reason else 0):
                log.info('Gemini finished without a picture (%s); asking once more', reason)
                await asyncio.sleep(NO_IMAGE_RETRY_DELAY_S)
                data = await self._send(payload)
                reason = self._retryable_no_image(data)
                if not reason:
                    break
            reason = reason or str(((data.get('candidates') or [{}])[0]).get('finishReason')
                                   or 'no candidates')
            if (data.get('promptFeedback') or {}).get('blockReason'):
                turn_trace.record('image', 'declined', ok=False, payload={
                    'reason': str((data.get('promptFeedback') or {}).get('blockReason')),
                    'model': self.cfg.model, 'prompt': prompt})
                raise CloudUnavailable(
                    f'{self.provider_label} declined this image request. No image was created. '
                    'Do not retry or rephrase it automatically.')
            candidates = data.get('candidates') or []
            if not candidates or candidates[0].get('finishReason') != 'STOP':
                said = self._reply_text(data)
                turn_trace.record('image', 'declined', ok=False, payload={
                    'reason': str((candidates[0] if candidates else {}).get('finishReason')
                                  or 'no candidates'),
                    'said': said[:300],
                    'model': self.cfg.model, 'prompt': prompt})
                detail = f' {self.provider_label} itself said: "{said[:300]}".' if said else ''
                raise CloudUnavailable(
                    f'{self.provider_label} finished this image request without a picture'
                    f' ({reason or "no image"}).{detail}'
                    ' Nothing was changed. Say what to change in other words, or use another photo.')
            for part in (candidates[0].get('content') or {}).get('parts', []):
                if part.get('thought'):
                    continue
                inline = part.get('inlineData')
                if inline:
                    try:
                        raw = base64.b64decode(inline['data'], validate=True)
                    except (ValueError, KeyError, TypeError) as exc:
                        raise CloudUnavailable(
                            f'{self.provider_label} returned invalid image data.') from exc
                    return await asyncio.to_thread(decode_image, raw, inline.get('mimeType', ''))
            said = self._reply_text(data)
            raise CloudUnavailable(
                f'{self.provider_label} returned no image.'
                + (f' It said: "{said[:300]}".' if said else '')
                + ' Nothing was changed; do not claim the photo was edited.')

    async def close(self) -> None:
        await self.http.aclose()
        if self.vertex is not None:
            await self.vertex.close()
