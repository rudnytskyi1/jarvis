"""Check image completion claims against this turn's structured tool results.

Shell exit codes and server file paths do not prove that an image was saved or
applied on the room PC. This inexpensive check also covers the final round cap
and the optional self-check pass, without another image generation request.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

IMAGE_REPAIR_MARKER = '[image completion check:'
_INTERNAL_PREFIXES = ('[system', IMAGE_REPAIR_MARKER)
_IMAGE_TOOLS = {'generate_image', 'save_photo', 'set_wallpaper', 'show_photo'}
_NEGATIVE = re.compile(
    r"\b(?:can't|cannot|couldn't|didn't|wasn't|isn't|hasn't|unable|failed|failure|"
    r"not (?:yet|been|saved|set|applied|opened|created|generated|confirmed)|"
    r"could not|did not|have not|has not|was not|is not|no image)\b", re.I)
_DONE = re.compile(r'^\s*(?:done|all set|finished|completed)\b', re.I)
_WALLPAPER = r'(?:wallpaper|desktop background|background (?:picture|image))'
_APPLIED_CLAIM = re.compile(
    rf'\b(?:set|applied|changed|installed|updated)\b[^.!?]{{0,100}}\b{_WALLPAPER}\b|'
    rf'\b{_WALLPAPER}\b[^.!?]{{0,70}}\b(?:set|applied|changed|updated)\b|'
    rf'\b(?:is|as) (?:now |your |the |computer |new )*{_WALLPAPER}\b', re.I)
_SAVED_CLAIM = re.compile(r'\b(?:saved|downloaded)\b', re.I)
_OPENED_CLAIM = re.compile(r'\b(?:opened|is open|now open)\b', re.I)
_GENERATED_CLAIM = re.compile(r'\b(?:created|generated|edited|turned you into|transformed)\b', re.I)


@dataclass(frozen=True)
class ImageCompletionIssue:
    operation: str
    repair: str
    fallback: str


def _current_turn(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.get('role') == 'user' and not str(message.get('content', '')).lstrip().startswith(_INTERNAL_PREFIXES):
            return history[index:]
    return history


def image_repair_already_requested(history: list[dict[str, Any]]) -> bool:
    return any(message.get('role') == 'user' and str(message.get('content', '')).startswith(IMAGE_REPAIR_MARKER)
               for message in _current_turn(history))


def image_generation_attempted(history: list[dict[str, Any]]) -> bool:
    return any(item.get('role') == 'tool' and (item.get('name') or item.get('tool_name')) == 'generate_image'
               for item in _current_turn(history))


def _result(message: dict[str, Any]) -> dict[str, Any]:
    raw = message.get('content', {})
    try:
        result = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}
    if not isinstance(result, dict):
        return {}
    # Client actions may wrap a structured result in output. Never interpret a
    # plain shell string such as "True" as an application confirmation.
    if isinstance(result.get('output'), str):
        try:
            output = json.loads(result['output'])
        except ValueError:
            output = None
        if isinstance(output, dict):
            return {**output, 'ok': result.get('ok') is True and output.get('ok', True) is not False}
    return result


def _wallpaper_requested(request: str) -> bool:
    return bool(re.search(
        rf'\b(?:set|put|use|apply)\b[^.!?]{{0,180}}\b{_WALLPAPER}\b|'
        rf'\bmake\b[^.!?]{{0,180}}\b(?:as (?:a |the )?{_WALLPAPER}|{_WALLPAPER} on (?:this |the |my )?(?:computer|pc))\b|'
        r'\b(?:поставь(?:те)?|установи(?:те)?|поменяй(?:те)?|смени(?:те)?|используй(?:те)?|примени(?:те)?)\b'
        r'[^.!?]{0,220}\b(?:обои|обоями|фон|фоном|фона)\b|'
        r'\bсделай(?:те)?\b[^.!?]{0,220}\b(?:фоном|обоями|фон рабочего стола|фоновым изображением|на фон)\b',
        request, re.I))


def check_image_completion(history: list[dict[str, Any]], reply: str) -> ImageCompletionIssue | None:
    """Return a missing concrete image step, never infer success from prose."""
    reply = str(reply).replace('\u2019', "'")
    turn = _current_turn(history)
    request = next((str(item.get('content', '')) for item in turn if item.get('role') == 'user'), '')
    results = [(str(item.get('name') or item.get('tool_name') or ''), _result(item))
               for item in turn if item.get('role') == 'tool']
    image_results = [(name, value) for name, value in results if name in _IMAGE_TOOLS]
    generated = generation_attempted = saved = opened = applied = False
    for name, value in image_results:
        if name == 'generate_image':
            if value.get('completion_repair_blocked') is True:
                continue
            # Every real generation attempts a new image. Earlier save/open/
            # wallpaper results describe the old image, including when the new
            # request fails. A blocked duplicate paid call creates no new image.
            generation_attempted = True
            generated = value.get('generated') is True
            saved = value.get('saved_on_client') is True
            opened = applied = False
            wallpaper = value.get('wallpaper', {})
            if isinstance(wallpaper, dict):
                applied = wallpaper.get('applied') is True and wallpaper.get('verified') is True
                saved = saved or wallpaper.get('saved') is True
        elif name == 'set_wallpaper':
            # A failed replacement cannot inherit an earlier successful apply.
            applied = value.get('applied') is True and value.get('verified') is True
            if 'saved' in value:
                saved = value.get('saved') is True
        elif name == 'save_photo':
            saved = value.get('saved') is True
            opened = value.get('opened') is True

    # A refusal, a real failure or a clarification must stay a reply, not turn
    # into a demand to perform an unauthorized or unavailable operation.
    clauses = re.split(r'[.!?;]+|\bbut\b', str(reply), flags=re.I)
    positive = ' '.join(clause for clause in clauses if not _NEGATIVE.search(clause))
    failure_reported = bool(_NEGATIVE.search(str(reply)))
    question = str(reply).rstrip().endswith('?')
    wallpaper_claim = bool(_APPLIED_CLAIM.search(positive)) and not question
    if generation_attempted and not generated and not question and not failure_reported and (
        _GENERATED_CLAIM.search(positive) or _DONE.search(positive) or wallpaper_claim or not reply.strip()
    ):
        # Do not "finish" a failed edit by applying some older successful image.
        return ImageCompletionIssue('generate',
            IMAGE_REPAIR_MARKER + ' image generation did not return generated=true. '
            'State the actual failure. Do not apply a previous image or retry generate_image; '
            'another paid attempt requires a new user request.]',
            'The image was not confirmed created. I have not started another image request.')
    wallpaper_missing = _wallpaper_requested(request) and (generated or image_results) and not failure_reported and not question
    if not applied and (wallpaper_claim or wallpaper_missing):
        return ImageCompletionIssue('wallpaper',
            IMAGE_REPAIR_MARKER + ' the room wallpaper has NOT been confirmed applied. '
            'If the user asked to set the image as wallpaper and the image exists, call set_wallpaper '
            'with source=generated (or their requested camera/screen source). Use the existing image; '
            'do not call generate_image again or run shell wallpaper commands. Only applied=true AND '
            'verified=true confirm success. Otherwise report exactly what remains unfinished.]',
            ('The image was created, but I could not confirm it was set as the computer wallpaper.'
             if generated else 'I could not confirm that the computer wallpaper was changed.'))

    image_context = bool(image_results) or bool(re.search(r'\b(?:photo|picture|image)\b', request, re.I))
    local_save_requested = bool(re.search(r'\b(?:save|download)\b', request, re.I))
    local_save_claim = bool(_SAVED_CLAIM.search(positive)) and (
        local_save_requested or bool(re.search(r'\b(?:pc|computer|desktop|locally)\b', positive, re.I)))
    # "Saved on the brain/server" is an explicit distinct storage location.
    local_save_claim = local_save_claim and not bool(re.search(r'\b(?:brain|server)\b', positive, re.I))
    if image_context and not saved and local_save_claim and not question:
        return ImageCompletionIssue('save',
            IMAGE_REPAIR_MARKER + ' no tool confirms this image was saved on the room PC. '
            'Generation saves on the brain only. If the user asked to save, call save_photo using '
            'the existing image; otherwise correct the reply. Never regenerate the image.]',
            'The image was created, but I could not confirm it was saved on the room PC.' if generated
            else 'I could not confirm that the image was saved on the room PC.')

    if any(name == 'save_photo' for name, _ in image_results) and not opened and _OPENED_CLAIM.search(positive) and not question:
        return ImageCompletionIssue('open',
            IMAGE_REPAIR_MARKER + ' the save_photo result does not confirm opened=true. '
            'Report the saved file separately from any failed opening; do not claim it opened.]',
            'The image was saved, but I could not confirm it opened.' if saved
            else 'I could not confirm that the image was saved or opened.')

    return None
