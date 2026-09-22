"""Explicit target choice and identity-locked face sample collection.

Two jobs, both of them about the SAME question — whose face is being saved:

* :func:`select_locked` / :func:`burst_samples` only ever hand back faces that
  belong to the person the enrollment started with, so a second person walking
  into the camera's view mid-enrollment cannot end up in the profile;
* :func:`choice_number` / :func:`numbered_preview` run the spoken "which one
  are you?" step when the camera sees more than one face at once.
"""

from __future__ import annotations

import math
import re

import numpy as np

from hub.face import cosine, decode_jpeg

#: Same cosine bar as ``server.face.threshold``: how close a face has to be to
#: the enrolled person's first sample to count as the same person.
IDENTITY_THRESHOLD = 0.45
#: A face that is only this close to a DIFFERENT face in the same frame is not
#: a match — two people looking equally similar to the reference is a coin
#: flip, and a coin flip must not decide whose face gets saved.
IDENTITY_MARGIN = 0.1
#: Frames inside one camera burst are only ~250 ms apart, so they are nearly
#: the same view. A frame whose best face ranks below this fraction of the
#: burst's best face is a blurred mid-turn frame, not a new angle: it would
#: only add a worse vector to the profile.
MIN_SAMPLE_SCORE_FRACTION = 0.5


def select_locked(faces, reference, threshold=IDENTITY_THRESHOLD, margin=IDENTITY_MARGIN):
    scored = sorted(((cosine(np.asarray(f['embedding']), np.asarray(reference)), i, f)
                     for i, f in enumerate(faces)), key=lambda row: row[0], reverse=True)
    if not scored or scored[0][0] < threshold:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][2]


def rank_score(face) -> float:
    """``det_score * sqrt(bbox_area)`` — the burst ranking of ``hub.face``.

    The same figure :func:`hub.face.select_best_face` maximizes, so a face the
    burst helper keeps and a face ``best_face`` would keep are the same face.
    A detection without a usable area (or without a score) ranks 0.0 instead
    of raising: the caller decides what to do with an unrankable face.
    """
    try:
        score = float(face.get("score") or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    try:
        area = float(face.get("area") or 0.0)
    except (AttributeError, TypeError, ValueError):
        area = 0.0
    if not math.isfinite(area) or area < 0.0:
        area = 0.0
    return score * math.sqrt(area)


def ranked_faces(faces) -> list[dict]:
    """Usable detections of one frame, best first (never raises)."""
    usable = [
        face for face in (faces or ())
        if isinstance(face, dict) and face.get("embedding") is not None
    ]
    return sorted(usable, key=rank_score, reverse=True)


def located_frames(engine, frames) -> list[list[dict]]:
    """Every face of every frame of one camera burst (SPEC v1.4).

    ``[engine.located_faces(frame.jpeg) for frame in frames]`` — one call per
    frame instead of only the first one, which is what lets the enrollment see
    that a SECOND person appeared in the middle of a burst. ``engine`` may be
    anything with ``located_faces``; a broken frame becomes an empty list.
    """
    return [
        list(engine.located_faces(getattr(frame, "jpeg", b"")) or [])
        for frame in (frames or ())
    ]


def burst_samples(
    frames,
    reference=None,
    threshold=IDENTITY_THRESHOLD,
    margin=IDENTITY_MARGIN,
    floor_fraction=MIN_SAMPLE_SCORE_FRACTION,
):
    """One best face per frame of a burst, best first: ``[(frame_index, face)]``.

    ТЗ F-210 wants 5–10 shots of the face from different angles, so a burst is
    worth more than the single best frame of it: every frame that actually
    holds the person's face becomes one sample.

    Locking: with ``reference`` given, a frame only contributes a face that
    matches it (:func:`select_locked`). Without one — the FIRST burst, which is
    what defines the reference — the best-ranked face of the burst becomes the
    seed and every other frame has to match the seed, so a person who walks in
    half a second after the enrollment started does not get their face saved
    under somebody else's name.

    Quality: only frames ranking at least ``floor_fraction`` of the burst's
    best are kept (frames without a measured area are all kept — there is
    nothing to rank them by). Ranking is deterministic, the input is never
    modified, and nothing raises.
    """
    picks: list[tuple[int, dict]] = []
    for index, faces in enumerate(frames or ()):
        if reference is None:
            ranked = ranked_faces(faces)
            best = ranked[0] if ranked else None
        else:
            best = select_locked(faces, reference, threshold, margin)
        if best is not None:
            picks.append((index, best))
    if not picks:
        return []
    picks.sort(key=lambda row: rank_score(row[1]), reverse=True)
    if reference is None:
        seed = np.asarray(picks[0][1]["embedding"], dtype=np.float32)
        picks = [
            row for row in picks
            if cosine(np.asarray(row[1]["embedding"], dtype=np.float32), seed) >= threshold
        ]
    top = rank_score(picks[0][1])
    if top <= 0.0:
        return picks
    return [row for row in picks if rank_score(row[1]) >= floor_fraction * top]


def choice_number(text, count):
    match = re.search(r"\b(?:number\s*)?([1-9])\b", text, re.I)
    if match:
        number = int(match.group(1))
        return number - 1 if number <= count else None
    words = (("one", "first", "left", "слева", "первый"), ("two", "second", "второй"),
             ("three", "third", "третий"), ("four", "fourth"))
    for index, aliases in enumerate(words[:count]):
        if any(re.search(r"\b" + word + r"\b", text, re.I) for word in aliases):
            return index
    if re.search(r"\b(?:right|справа)\b", text, re.I):
        return count - 1
    return None


def numbered_preview(jpeg, faces):
    import cv2
    image = decode_jpeg(jpeg)
    if image is None:
        return jpeg, []
    h, w = image.shape[:2]
    descriptions = []
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(face['box'])]
        cv2.rectangle(image, (x1, y1), (x2, y2), (190, 210, 115), 3)
        cv2.putText(image, str(index + 1), (x1, max(35, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (190, 255, 160), 3)
        # Describe only directly measured location; no guessed identity or traits.
        center = (x1 + x2) / (2 * w)
        position = 'on the left' if center < .35 else 'on the right' if center > .65 else 'in the middle'
        descriptions.append(f"number {index + 1}, {position}")
    ok, encoded = cv2.imencode('.jpg', image)
    return encoded.tobytes() if ok else jpeg, descriptions
