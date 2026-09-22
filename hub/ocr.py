"""Текст на экране: OCR скриншота рядом с vision-моделью (ТЗ F-308).

ТЗ F-308: «Скриншот → OCR (RapidOCR или PaddleOCR) + vision-LLM». Две части
делают разное и потому нужны обе: OCR приносит ТОЧНЫЕ строки («ошибка 0x80070005»,
«Максим: ок», номер строки в таблице), а vision-модель объясняет, что это за
окно и как расположены элементы. Скриншот и так приходит на хаб ради
vision-модели, поэтому OCR живёт здесь же.

Пакет OCR загружается ЛЕНИВО и ровно один раз: нет ``rapidocr``/``paddleocr``
или весов — это ``OcrUnavailable`` с НАЗВАННОЙ причиной, одна строка в лог, а
не молчащая функция. Повторный вызов не переимпортирует сломанное. OCR
работает на CPU (onnxruntime), поэтому не занимает слот GPU-очереди раздела
4.5; вызов уходит в отдельный поток.

Текст с экрана — НЕДОВЕРЕННЫЙ (ТЗ F-411, D-09): он возвращается внутри
результата ``look_at_screen``, который хаб оборачивает разделителями
``<<<UNTRUSTED … UNTRUSTED>>>`` и сканирует тем же D-09, что и остальной
внешний текст. Инструкция «игнорируй правила» на чужой странице не запускает
ни один инструмент.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Что ТЗ называет по имени. ``rapidocr`` — вариант по умолчанию: ему не нужен
#: paddle, он идёт на onnxruntime и ставится одной строкой.
DEFAULT_ENGINE = "rapidocr"
ENGINES = ("rapidocr", "paddleocr")

#: Ниже этой уверенности строку не показываем: OCR любит «угадывать» узоры
#: на иконках, и выдуманный текст хуже, чем честно отсутствующий.
MIN_CONFIDENCE = 0.5
#: Сколько знаков текста вообще отдавать: экран с длинной страницей не должен
#: раздувать промпт.
MAX_CHARS = 4000
MAX_LINES = 200


class OcrUnavailable(RuntimeError):
    """OCR использовать нельзя: нет пакета, весов или картинки."""


@dataclass(frozen=True)
class OcrLine:
    """Одна распознанная строка экрана."""

    text: str
    confidence: float = 1.0
    #: Прямоугольник строки в пикселях картинки: ``(x, y, width, height)``.
    box: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text, "confidence": round(self.confidence, 3)}
        if self.box is not None:
            data["box"] = [round(value, 1) for value in self.box]
        return data


@dataclass(frozen=True)
class ScreenText:
    """Что OCR прочитал на этом скриншоте."""

    engine: str
    lines: tuple[OcrLine, ...] = ()
    #: Текст длиннее лимита обрезан; в отчёте видно, что именно обрезали.
    truncated: bool = False

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def empty(self) -> bool:
        return not self.lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "text": self.text,
            "lines": [line.as_dict() for line in self.lines],
            "truncated": self.truncated,
        }


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    return default if value is None else value


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rectangle(shape: Any) -> tuple[float, float, float, float] | None:
    """Прямоугольник вокруг четырёх точек OCR — или ``None``, если это не они."""
    try:
        points = [(float(point[0]), float(point[1])) for point in shape]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def rapidocr_lines(raw: Any) -> list[OcrLine]:
    """Разобрать ответ RapidOCR: ``[[box, text, score], …]`` (или ``(result, elapse)``)."""
    result = raw
    if isinstance(raw, tuple) and len(raw) == 2:
        # RapidOCR отвечает парой (result, elapse), где result может быть None.
        result = raw[0]
    lines: list[OcrLine] = []
    for item in result or ():
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1] or "").strip()
        if not text:
            continue
        lines.append(OcrLine(text=text, confidence=_number(item[2], 1.0)
                             if len(item) > 2 else 1.0, box=_rectangle(item[0])))
    return lines


def _paddleocr_page(page: Any) -> list[OcrLine]:
    """Разобрать одну страницу PaddleOCR — старого (списки) или нового (dict) вида."""
    if isinstance(page, Mapping) or hasattr(page, "get"):
        texts = list(page.get("rec_texts") or ())
        scores = list(page.get("rec_scores") or ())
        shapes = list(page.get("dt_polys") or page.get("rec_polys") or ())
        lines: list[OcrLine] = []
        for index, text in enumerate(texts):
            text = str(text or "").strip()
            if not text:
                continue
            lines.append(OcrLine(
                text=text,
                confidence=_number(scores[index], 1.0) if index < len(scores) else 1.0,
                box=_rectangle(shapes[index]) if index < len(shapes) else None,
            ))
        return lines
    if not isinstance(page, (list, tuple)):
        # PaddleOCR 3.x умеет отвечать словарём-обёрткой; всё прочее — не страница.
        return []
    lines = []
    for item in page or ():
        try:
            shape, answer = item[0], item[1]
            text = str(answer[0] or "").strip()
            confidence = _number(answer[1], 1.0)
        except (TypeError, IndexError, ValueError, KeyError):
            continue
        if not text:
            continue
        lines.append(OcrLine(text=text, confidence=confidence, box=_rectangle(shape)))
    return lines


def paddleocr_lines(raw: Any) -> list[OcrLine]:
    """Разобрать ответ PaddleOCR: ``[[[box, (text, score)], …]]`` или новые словари."""
    if isinstance(raw, Mapping) or hasattr(raw, "get"):
        return _paddleocr_page(raw)
    pages = raw or ()
    if not isinstance(pages, Sequence):
        return []
    lines: list[OcrLine] = []
    for page in pages:
        lines.extend(_paddleocr_page(page))
    return lines


class OcrEngine:
    """RapidOCR/PaddleOCR за ленивым импортом (ТЗ F-308)."""

    def __init__(self, cfg: Any = None) -> None:
        requested = str(_attr(cfg, "engine", DEFAULT_ENGINE) or DEFAULT_ENGINE).strip().lower()
        self.engine = requested if requested in ENGINES else DEFAULT_ENGINE
        self.min_confidence = float(_attr(cfg, "min_confidence", MIN_CONFIDENCE)
                                    or MIN_CONFIDENCE)
        self.max_chars = int(_attr(cfg, "max_chars", MAX_CHARS) or MAX_CHARS)
        self.max_lines = int(_attr(cfg, "max_lines", MAX_LINES) or MAX_LINES)
        self.languages = [str(item) for item in (_attr(cfg, "languages", ()) or ()) if str(item)]
        self._model: Any = None
        self._error = ""

    # --- сборка --------------------------------------------------------
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._error:
            raise OcrUnavailable(self._error)
        try:
            self._model = (self._build_rapidocr() if self.engine == "rapidocr"
                           else self._build_paddleocr())
        except OcrUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - пакет/веса могут не подняться
            self._error = f"{self.engine} could not start ({exc})"
            raise OcrUnavailable(self._error) from exc
        log.info("Screen OCR is up: %s", self.engine)
        return self._model

    def _build_rapidocr(self) -> Any:
        try:
            from rapidocr import RapidOCR  # type: ignore
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
            except ImportError as exc:
                self._error = ("RapidOCR is not installed (pip install rapidocr), "
                               "and no screen OCR is available")
                raise OcrUnavailable(self._error) from exc
        return RapidOCR()

    def _build_paddleocr(self) -> Any:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            self._error = ("PaddleOCR is not installed (pip install paddleocr), "
                           "and no screen OCR is available")
            raise OcrUnavailable(self._error) from exc
        language = self.languages[0] if self.languages else "en"
        try:
            return PaddleOCR(lang=language, use_angle_cls=True, show_log=False)
        except TypeError:
            # PaddleOCR 3.x изменил набор аргументов: пробуем самый скромный.
            return PaddleOCR(lang=language)

    # --- распознавание -------------------------------------------------
    def _pixels(self, jpeg: bytes) -> Any:
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - зависит от установки
            raise OcrUnavailable(f"numpy/Pillow are not installed ({exc})") from exc
        try:
            with Image.open(io.BytesIO(jpeg)) as image:
                return np.array(image.convert("RGB"))
        except OcrUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - битый кадр это не текст
            raise OcrUnavailable(f"the screenshot could not be decoded ({exc})") from exc

    def _recognize(self, model: Any, pixels: Any) -> list[OcrLine]:
        if self.engine == "rapidocr":
            raw = model(pixels)
            return rapidocr_lines(raw)
        if hasattr(model, "predict"):
            return paddleocr_lines(model.predict(pixels))
        return paddleocr_lines(model.ocr(pixels, cls=True))

    def read(self, jpeg: bytes) -> ScreenText:
        """Прочитать текст скриншота; пустой ответ — это «текста не видно».

        :raises OcrUnavailable: пакета/весов нет или картинку не разобрать —
            вызывающий продолжает с vision-моделью и честно называет причину.
        """
        if not jpeg:
            raise OcrUnavailable("there is no screenshot to read")
        model = self._load()
        lines = [line for line in self._recognize(model, self._pixels(jpeg))
                 if line.confidence >= self.min_confidence and line.text.strip()]
        truncated = len(lines) > self.max_lines
        if truncated:
            lines = lines[:self.max_lines]
        total = 0
        kept: list[OcrLine] = []
        for line in lines:
            if total + len(line.text) > self.max_chars:
                truncated = True
                break
            kept.append(line)
            total += len(line.text) + 1
        return ScreenText(engine=self.engine, lines=tuple(kept), truncated=truncated)


def ocr_enabled(cfg: Any) -> bool:
    """Включил ли владелец экранный OCR (``server.ocr.enabled``)."""
    ocr = _attr(_attr(cfg, "server", None), "ocr", None)
    return bool(_attr(ocr, "enabled", False))


def read_screen_text(jpeg: bytes, cfg: Any = None, engine: OcrEngine | None = None) -> ScreenText:
    """Прочитать текст скриншота настроенным движком (``server.ocr``)."""
    reader = engine if engine is not None else OcrEngine(_attr(_attr(cfg, "server", None), "ocr", None))
    return reader.read(jpeg)


__all__ = [
    "DEFAULT_ENGINE",
    "ENGINES",
    "MAX_CHARS",
    "MAX_LINES",
    "MIN_CONFIDENCE",
    "OcrEngine",
    "OcrLine",
    "OcrUnavailable",
    "ScreenText",
    "ocr_enabled",
    "paddleocr_lines",
    "rapidocr_lines",
    "read_screen_text",
]
