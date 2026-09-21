"""Explicit target choice and identity-locked face sample collection."""
from __future__ import annotations

import re

import numpy as np

from hub.face import cosine, decode_jpeg


def select_locked(faces, reference, threshold=.45, margin=.1):
    scored = sorted(((cosine(np.asarray(f['embedding']), np.asarray(reference)), i, f)
                     for i, f in enumerate(faces)), key=lambda row: row[0], reverse=True)
    if not scored or scored[0][0] < threshold:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][2]


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
