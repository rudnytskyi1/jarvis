"""Anti-spoofing: живое ли лицо и тот ли это голос (ТЗ F-214).

ТЗ просит две разные проверки, и у них разная природа.

**Лицо.** На бёрсте кадров хаб ищет признаки экрана или фотографии:

* *муар* — периодическая решётка пикселей экрана даёт в спектре кадра
  узкие пики (``peakiness``), которые не размазаны, как у кожи; шум сенсора
  тоже высокочастотный, но он НЕ периодический, поэтому решает произведение
  «доля высоких частот × пиковость»;
* *микродвижения* — у живого лица между кадрами всегда меняются соотношения
  между ключевыми точками (дыхание, мимика, моргание), у распечатки или
  фотографии на экране они застывают; считается и по точкам, и по пикселям
  (кроп приводится к общему размеру, поэтому дрожание камеры само по себе
  не считается движением лица);
* *плоскость* — движение всех ключевых точек должно объясняться одной
  плоскостью (гомография): проекция ЛЮБОЙ плоскости под любым движением
  камеры — это ровно гомография. У живого 3D-лица так не выходит: нос
  выдвинут из плоскости лица, и остаток подгонки растёт вместе с движением.
  Поэтому признак работает только когда в бёрсте ЧТО-ТО заметно сдвинулось:
  застывший кадр ничего не говорит о плоскости, а говорит о движении.

**Голос.** Для привилегированных действий ТЗ просит challenge-слово: хаб сам
выбирает случайное слово, человек его произносит, и проверяются два
независимых факта — что произнесено именно это слово (транскрипт) и что это
тот же голос (ECAPA-сходство с собственным профилем человека).

Нейросетевая liveness-модель ТЗ (insightface anti-spoof / MiniFASNet)
подключается снаружи: :func:`load_model` — точка входа для неё. В этой сборке
модели нет, и это честно: ``load_model`` бросает :class:`SpoofModelUnavailable`,
а настройка ``require_model`` делает отсутствие модели ОТКАЗОМ, а не
«наверное, живой». Покадровые признаки при этом считаются настоящей
математикой по настоящим кадрам, а не заглушкой.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.server.anti_spoofing")

#: Пороги признаков. Значения — консервативные «умолчания», снятые на
#: синтетических кадрах; на стенде они калибруются (см. ``DECISIONS.md`` P2-24).
MOIRE_THRESHOLD = 0.35
MOTION_MIN = 0.004
#: Остаток подгонки, ниже которого движение считается плоским.
PLANAR_RESIDUAL_MAX = 0.02
#: И насколько заметно всё сдвинулось, чтобы о плоскости вообще судить.
PLANAR_MOTION_MIN = 0.02
#: Сколько кадров бёрста нужно, чтобы вообще судить о живости.
MIN_FRAMES = 5
WINDOW_FRAMES = 8
#: ТЗ F-214: окно challenge-слова и порог «тот же голос».
CHALLENGE_WINDOW_S = 20.0
CHALLENGE_VOICE_THRESHOLD = 0.5


class SpoofModelUnavailable(RuntimeError):
    """Нейросетевой liveness-модели в этой сборке нет (не заглушка, а отказ)."""


def load_model(path: Path | str | None = None) -> Any:
    """Точка входа для liveness-модели ТЗ (insightface anti-spoof / MiniFASNet).

    Модель не входит в сборку: ТЗ называет её, но ни весов, ни лицензии в
    песочнице нет, а подделывать вердикт чужой модели запрещено правилами
    проекта (``AGENTS.md``: «никаких фейков»). Поэтому здесь честное
    исключение, а не «модель», возвращающая «живой»: отсутствие модели
    попадает в ``docs/TZ_STATUS.md`` и в раздел 17 ТЗ.
    """
    raise SpoofModelUnavailable(
        f"the anti-spoofing model is not part of this build ({path or 'no path'}); "
        "the burst cues of hub/anti_spoofing.py work without it, and "
        "server.identity.anti_spoofing.require_model makes its absence a refusal")


# ---------------------------------------------------------------------------
# кадры
# ---------------------------------------------------------------------------


def gray_copy(crop: Any) -> Any:
    """Any image-like value → 2-D ``float32`` grayscale, or ``None``."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dependency of the hub
        return None
    if crop is None:
        return None
    try:
        array = np.asarray(crop, dtype="float32")
    except (TypeError, ValueError):
        return None
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2 or array.size == 0:
        return None
    return array


def cut_face(jpeg: Any, box: Any = None, *, size: int = 96) -> Any:
    """The face crop of one frame, grayscale (``None`` when there is none).

    ``box`` is either normalized (all values within ``0..1.5``, as the client
    sends it) or in pixels; a missing box means the whole frame is the face.
    """
    from hub.face import decode_jpeg

    image = decode_jpeg(jpeg) if isinstance(jpeg, (bytes, bytearray)) else jpeg
    array = gray_copy(image)
    if array is None:
        return None
    height, width = array.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = (float(value) for value in box)
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        array = array[y1:y2, x1:x2]
    return norm_size(array, size=size)


def norm_size(array: Any, *, size: int = 64) -> Any:
    """Resize a grayscale crop to ``size × size`` (cv2 when it is there)."""
    gray = gray_copy(array)
    if gray is None:
        return None
    try:
        import cv2

        return cv2.resize(gray, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001 - nearest-neighbour keeps the module usable
        import numpy as np

        gy = (np.linspace(0, gray.shape[0] - 1, int(size))).astype(int)
        gx = (np.linspace(0, gray.shape[1] - 1, int(size))).astype(int)
        return gray[np.ix_(gy, gx)]


@dataclass(frozen=True)
class Frame:
    """One camera frame of one face, as the liveness cues measure it."""

    gray: Any = None
    landmarks: tuple[tuple[float, float], ...] = ()
    quality: float = 0.0

    @property
    def has_landmarks(self) -> bool:
        return len(self.landmarks) >= 5


def frame_of(jpeg: Any, box: Any = None, landmarks: Any = None,
             *, quality: float = 0.0, size: int = 96) -> Frame | None:
    """Build a :class:`Frame` from the client's JPEG, box and landmarks."""
    crop = cut_face(jpeg, box, size=size)
    if crop is None:
        return None
    points: list[tuple[float, float]] = []
    for point in landmarks or ():
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return Frame(gray=crop, landmarks=tuple(points), quality=float(quality or 0.0))


# ---------------------------------------------------------------------------
# признаки
# ---------------------------------------------------------------------------


def moire_score(crop: Any) -> float:
    """How much the crop looks like a screen's pixel lattice (0..1).

    A lattice puts a large share of its energy into a few very bright bins of
    the high-frequency band. Sensor noise is high-frequency too, but flat: it
    spreads over the whole band. Multiplying "how much energy sits high" by
    "how peaked that band is" separates the two, which is what makes this a
    moiré test rather than a noise test.
    """
    import numpy as np

    gray = gray_copy(crop)
    if gray is None or min(gray.shape) < 24:
        return 0.0
    centred = gray - float(gray.mean())
    if not float(centred.std()):
        return 0.0
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centred))) ** 2
    height, width = spectrum.shape
    cy, cx = height // 2, width // 2
    ys, xs = np.ogrid[:height, :width]
    radius = np.sqrt(((ys - cy) / max(cy, 1)) ** 2 + ((xs - cx) / max(cx, 1)) ** 2)
    band = spectrum[(radius >= 0.25) & (radius <= 0.98)]
    total = float(spectrum.sum())
    band_total = float(band.sum())
    if band.size == 0 or band_total <= 0.0 or total <= 0.0:
        return 0.0
    keep = max(1, band.size // 200)
    top = float(np.sort(band.ravel())[-keep:].sum())
    share = band_total / total
    peakiness = top / band_total
    return max(0.0, min(1.0, (share * peakiness) / 0.2))


def _distances(landmarks: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Scale-free ratios of the five key points (eyes, nose, mouth corners)."""
    left, right, nose, mouth_l, mouth_r = ((float(p[0]), float(p[1])) for p in landmarks[:5])
    eye = math.dist(left, right)
    mouth = math.dist(mouth_l, mouth_r)
    eye_mid = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    mouth_mid = ((mouth_l[0] + mouth_r[0]) / 2, (mouth_l[1] + mouth_r[1]) / 2)
    to_nose = math.dist(eye_mid, nose)
    to_mouth = math.dist(mouth_mid, nose)
    if min(eye, mouth, to_nose, to_mouth) <= 1e-6:
        return ()
    return (eye / to_nose, mouth / to_mouth, to_nose / to_mouth)


def _variation(series: Sequence[float]) -> float:
    if len(series) < 2:
        return 0.0
    import numpy as np

    values = np.asarray(series, dtype="float64")
    mean = float(values.mean())
    if abs(mean) < 1e-9:
        return 0.0
    return float(values.std() / abs(mean))


def landmark_motion(frames: Sequence[Frame]) -> float:
    """How much the face's own geometry changes across the burst (0 = frozen)."""
    ratios = [_distances(frame.landmarks) for frame in frames if frame.has_landmarks]
    ratios = [row for row in ratios if row]
    if len(ratios) < 2:
        return 0.0
    return max(_variation([row[index] for row in ratios]) for index in range(3))


def pixel_motion(frames: Sequence[Frame]) -> float:
    """How much the (box-normalized) crops differ between frames (0 = frozen)."""
    import numpy as np

    crops = [norm_size(frame.gray, size=64) for frame in frames]
    crops = [crop for crop in crops if crop is not None]
    if len(crops) < 2:
        return 0.0
    reference = float(crops[0].std()) or 1.0
    steps = [float(np.abs(crops[index] - crops[index - 1]).mean()) / reference
             for index in range(1, len(crops))]
    return float(sum(steps) / len(steps)) if steps else 0.0


def motion_score(frames: Sequence[Frame]) -> tuple[float, str]:
    """``(motion, source)`` — the larger of the two ways of measuring it."""
    if len(frames) < 2:
        return 0.0, "none"
    if all(frame.has_landmarks for frame in frames):
        return max(landmark_motion(frames), pixel_motion(frames)), "landmarks+pixels"
    return pixel_motion(frames), "pixels"


def _homography(source: Sequence[Sequence[float]],
                target: Sequence[Sequence[float]]) -> Any:
    """Least-squares homography mapping ``source`` points onto ``target``."""
    import numpy as np

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    matrix = np.asarray(rows, dtype="float64")
    if matrix.shape[0] < 8:
        return None
    try:
        _u, _s, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover - only on a broken BLAS
        return None
    return vt[-1].reshape(3, 3)


def _project(matrix: Any, points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    import numpy as np

    out: list[tuple[float, float]] = []
    for x, y in points:
        vector = matrix @ np.asarray([float(x), float(y), 1.0])
        if abs(float(vector[2])) < 1e-9:
            return []
        out.append((float(vector[0]) / float(vector[2]), float(vector[1]) / float(vector[2])))
    return out


def planarity(frames: Sequence[Frame]) -> tuple[float, float]:
    """``(residual, displacement)`` — was the motion one plane, and was there one?

    The residual is the RMS distance between the landmarks of a frame and the
    homography that maps the first frame onto it, divided by the face's own
    scale (so a face further from the camera is judged the same). The
    displacement says how far anything moved, in the same units: what did not
    move cannot be judged, and what moved like a plane was a plane.
    """
    with_points = [frame for frame in frames if frame.has_landmarks]
    if len(with_points) < 2:
        return 0.0, 0.0
    first = with_points[0].landmarks
    centre = (sum(point[0] for point in first) / len(first),
              sum(point[1] for point in first) / len(first))
    scale = max(1e-6, sum(math.dist(point, centre) for point in first) / len(first))
    residual = 0.0
    displacement = 0.0
    for frame in with_points[1:]:
        matrix = _homography(first, frame.landmarks)
        if matrix is not None:
            projected = _project(matrix, first)
            if projected:
                errors = [math.dist(projected[index], frame.landmarks[index])
                          for index in range(min(len(projected), len(frame.landmarks)))]
                if errors:
                    residual = max(residual, sum(errors) / len(errors) / scale)
        moved = sum(math.dist(first[index], frame.landmarks[index])
                    for index in range(min(len(first), len(frame.landmarks))))
        count = max(1, min(len(first), len(frame.landmarks)))
        displacement = max(displacement, moved / count / scale)
    return residual, displacement


@dataclass(frozen=True)
class LivenessVerdict:
    """The answer of F-214 about one burst of one face."""

    ok: bool
    code: str
    detail: str = ""
    cues: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"ok": self.ok, "code": self.code, "detail": self.detail,
                "cues": dict(self.cues)}

    def refusal_note(self) -> str:
        """Why the burst may not teach the profile anything, in one line."""
        return self.detail or self.code


def assess_burst(frames: Iterable[Frame | None], *,
                 model: Any = None, require_model: bool = False,
                 moire_threshold: float = MOIRE_THRESHOLD,
                 motion_min: float = MOTION_MIN,
                 planar_residual_max: float = PLANAR_RESIDUAL_MAX,
                 planar_motion_min: float = PLANAR_MOTION_MIN,
                 min_frames: int = MIN_FRAMES) -> LivenessVerdict:
    """ТЗ F-214: may this burst of frames be treated as a live face?

    The order is deliberate. Not enough frames is not an accusation, it is
    "cannot tell" — but it is still not a pass, because a burst too short to
    judge must not vouch for anybody. A configured liveness model speaks first
    when it is there; when the configuration *requires* it and it is missing,
    the answer is ``no_model`` and never "assume live". Only then do the three
    deterministic cues of the ТЗ decide: a screen's lattice (moiré), a burst
    where nothing moved inside the face, and a burst that moved but whose
    motion fitted one plane exactly.
    """
    samples = [frame for frame in frames or () if isinstance(frame, Frame)]
    if len(samples) < int(min_frames):
        return LivenessVerdict(False, "too_few_frames",
                               f"only {len(samples)} frames of the {int(min_frames)} "
                               "needed to judge whether the face is alive",
                               {"frames": len(samples)})
    cues: dict[str, Any] = {"frames": len(samples)}
    if model is not None:
        try:
            live, note = model.is_live(samples)
        except Exception as exc:  # noqa: BLE001 - a broken model is not a "yes"
            log.warning("The anti-spoofing model failed on a burst: %s", exc)
            live, note = False, f"the liveness model failed ({exc})"
        cues["model"] = note
        if not live:
            return LivenessVerdict(False, "model", f"the liveness model says no: {note}", cues)
    elif require_model:
        return LivenessVerdict(
            False, "no_model",
            "the liveness model is required but this hub has none, so no burst "
            "is trusted", cues)
    moire = max(moire_score(frame.gray) for frame in samples)
    motion, source = motion_score(samples)
    residual, displacement = planarity(samples)
    cues.update({"moire": round(moire, 4), "motion": round(motion, 5),
                 "motion_source": source, "planarity": round(residual, 5),
                 "displacement": round(displacement, 4)})
    if moire >= float(moire_threshold):
        return LivenessVerdict(
            False, "screen",
            f"the frame looks like a screen: its spectrum is too regular "
            f"({moire:.2f} against the threshold {float(moire_threshold):.2f})", cues)
    if motion <= float(motion_min):
        return LivenessVerdict(
            False, "no_micro_movement",
            f"nothing inside the face moved across the burst ({motion:.4f} against "
            f"the floor {float(motion_min):.4f}) - a live face breathes and blinks", cues)
    if displacement >= float(planar_motion_min) and residual <= float(planar_residual_max):
        return LivenessVerdict(
            False, "flat_surface",
            f"the face moved {displacement:.1%} of its own size and every point "
            f"stayed on one plane (residual {residual:.4f} against the limit "
            f"{float(planar_residual_max):.4f}) - a live face is not flat", cues)
    return LivenessVerdict(True, "live", "the burst moved like a face and was not a screen",
                           cues)


def verify_face_burst(frames: Iterable[Frame | None], *, enabled: bool = True,
                      **kwargs: Any) -> LivenessVerdict:
    """``assess_burst`` with the room's switch: off means "do not judge"."""
    if not enabled:
        return LivenessVerdict(True, "off", "the anti-spoofing check is switched off",
                               {"frames": len([f for f in frames or () if f])})
    return assess_burst(frames, **kwargs)


# ---------------------------------------------------------------------------
# challenge-слово (голос)
# ---------------------------------------------------------------------------

#: Слова подобраны так, чтобы whisper не мог их перепутать между собой, и
#: чтобы они были короткими: challenge не должен превращаться в диктант.
WORDS: dict[str, tuple[str, ...]] = {
    "ru": ("яблоко", "север", "лампа", "море", "тигр", "ветер", "камень", "дождь"),
    "en": ("apple", "north", "lamp", "ocean", "tiger", "wind", "stone", "rain"),
    "es": ("manzana", "norte", "lámpara", "mar", "tigre", "viento", "piedra", "lluvia"),
}

ASK: dict[str, str] = {
    "ru": ("Скажи «{word}» в течение {seconds} секунд, и я выполню это действие. "
           "Любой другой ответ отменяет его."),
    "en": ("Say «{word}» within {seconds} seconds and I will run this action. "
           "Anything else cancels it."),
    "es": ("Di «{word}» en {seconds} segundos y haré esta acción. "
           "Cualquier otra respuesta la cancela."),
}

DETAILS: dict[str, str] = {
    "word": "I did not hear the word I asked for, so nothing was done.",
    "voice": "That was the right word, but it was not your voice, so nothing was done.",
    "no_voice": "I could not compare the voice, so nothing was done.",
}


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in WORDS else default


def words(language: Any = "ru") -> tuple[str, ...]:
    return WORDS[language_of(language)]


def choose_word(language: Any = "ru", *, rng: random.Random | None = None) -> str:
    """A random word of the person's language (ТЗ F-214: «случайное слово»)."""
    pick = (rng or random).choice(words(language))
    return str(pick)


@dataclass(frozen=True)
class Challenge:
    """One open challenge question, like F-113's confirmation or F-208's PIN."""

    word: str
    language: str = "ru"
    window_s: float = CHALLENGE_WINDOW_S
    opened_at: float = field(default_factory=time.monotonic)

    def expired(self, *, now: float | None = None) -> bool:
        return (float(now if now is not None else time.monotonic())
                - float(self.opened_at)) > float(self.window_s)

    def summary(self) -> dict[str, Any]:
        return {"word": self.word, "language": self.language,
                "window_s": self.window_s}


def challenge(language: Any = "ru", *, window_s: float = CHALLENGE_WINDOW_S,
              rng: random.Random | None = None) -> Challenge:
    code = language_of(language)
    return Challenge(word=choose_word(code, rng=rng), language=code,
                     window_s=float(window_s))


def ask(request: Challenge, language: Any = "") -> str:
    """The question the hub speaks (never the model, like F-113/F-208)."""
    code = language_of(language or request.language)
    return ASK[code].format(word=request.word,
                            seconds=max(1, int(round(float(request.window_s)))))


def _tokens(text: Any) -> list[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in str(text or "").casefold())
    return [word for word in cleaned.split() if word]


def heard(text: Any, word: str, *, ratio: float = 0.8) -> bool:
    """Whether the transcript really contains the challenge word.

    Whisper writes the same word differently from turn to turn (case, accents,
    the odd ending), so the comparison is on normalized words with a high
    similarity bar — close enough for speech, far too strict for another word.
    """
    wanted = str(word or "").casefold().strip()
    if not wanted:
        return False
    for said in _tokens(text):
        if said == wanted:
            return True
        if SequenceMatcher(None, said, wanted).ratio() >= float(ratio):
            return True
    return False


def speaker_similarity(embedding: Any, vectors: Iterable[Any]) -> float | None:
    """Best ECAPA cosine between this utterance and the person's own vectors."""
    if embedding is None:
        return None
    try:
        import numpy as np

        heard_vector = np.asarray(embedding, dtype="float64").ravel()
        norm = float(np.linalg.norm(heard_vector))
        if not norm:
            return None
        heard_vector = heard_vector / norm
        best: float | None = None
        for vector in vectors or ():
            other = np.asarray(vector, dtype="float64").ravel()
            if other.shape != heard_vector.shape:
                continue
            length = float(np.linalg.norm(other))
            if not length:
                continue
            score = float(np.dot(heard_vector, other / length))
            best = score if best is None else max(best, score)
        return best
    except (TypeError, ValueError):  # pragma: no cover - a broken vector is not a match
        return None


@dataclass(frozen=True)
class ChallengeVerdict:
    """The answer of F-214 about the spoken challenge word."""

    ok: bool
    code: str
    detail: str = ""
    factors: dict[str, Any] = field(default_factory=dict)


def verify(text: Any, word: str, similarity: float | None, *,
           threshold: float = CHALLENGE_VOICE_THRESHOLD) -> ChallengeVerdict:
    """Both halves of the challenge: the right word AND the right voice."""
    factors: dict[str, Any] = {
        "word_match": heard(text, word),
        "voice": None if similarity is None else round(float(similarity), 3),
        "threshold": float(threshold),
    }
    if not factors["word_match"]:
        return ChallengeVerdict(False, "word", DETAILS["word"], factors)
    if similarity is None:
        return ChallengeVerdict(False, "no_voice", DETAILS["no_voice"], factors)
    if float(similarity) < float(threshold):
        return ChallengeVerdict(
            False, "voice",
            f"{DETAILS['voice']} (voice {float(similarity):.2f} against "
            f"{float(threshold):.2f})", factors)
    return ChallengeVerdict(True, "ok", "the challenge word was said in the right voice",
                            factors)


__all__ = [
    "ASK",
    "CHALLENGE_VOICE_THRESHOLD",
    "CHALLENGE_WINDOW_S",
    "Challenge",
    "ChallengeVerdict",
    "Frame",
    "LivenessVerdict",
    "MIN_FRAMES",
    "PLANAR_MOTION_MIN",
    "MOIRE_THRESHOLD",
    "MOTION_MIN",
    "PLANAR_RESIDUAL_MAX",
    "SpoofModelUnavailable",
    "WORDS",
    "WINDOW_FRAMES",
    "ask",
    "assess_burst",
    "challenge",
    "choose_word",
    "cut_face",
    "frame_of",
    "gray_copy",
    "heard",
    "landmark_motion",
    "language_of",
    "load_model",
    "moire_score",
    "motion_score",
    "norm_size",
    "pixel_motion",
    "planarity",
    "speaker_similarity",
    "verify",
    "verify_face_burst",
    "words",
]
