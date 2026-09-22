"""Набор для идентичности и его метрики (ТЗ 15.6).

ТЗ 15.6: «Набор для идентичности: не меньше 20 треков (спереди, сбоку, спина,
смена ракурса) с эталонным ``person_id``; метрики точность и полнота
узнавания». Критерий приёмки фазы 2 добавляет отдельную проверку: «со спины
после одного фронтального кадра — не хуже 80 %».

Здесь живут только ДАННЫЕ набора и АРИФМЕТИКА метрик: разметка треков, вызов
предсказателя по каждому кадру и честный отчёт (сколько названо верно, какая
точность и полнота, разбивка по ракурсам, отдельная доля «спина после
фронта»). Сам прогон по настоящим записям — ``scripts/identity_benchmark.py``.

Метрики считаются так, чтобы их нельзя было улучшить, ничего не делая:
неизвестный ответ уменьшает полноту, но не портит точность; названный не тем
человеком портит и то, и другое; «спина после фронта» проверяется только там,
где фронтальный кадр действительно был раньше спинного.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Ракурсы, которые называет ТЗ 15.6.
View = Literal["frontal", "side", "back", "turn"]
VIEWS: tuple[str, ...] = ("frontal", "side", "back", "turn")

#: Не меньше 20 треков — требование ТЗ 15.6, а не пожелание.
MIN_TRACKS = 20
#: Критерий приёмки фазы 2: «со спины после одного фронтального кадра ≥ 80 %».
MIN_BACK_RATE = 0.80

#: Что возвращает предсказатель: ``person_id`` или пустая строка («не знаю»).
Prediction = str
Predictor = Callable[["BenchmarkTrack", int], Prediction]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class BenchmarkFrame(_Strict):
    """Один кадр трека: файл записи и ракурс, под которым он снят."""

    path: str = Field(min_length=1, max_length=400)
    view: View = "frontal"


class BenchmarkTrack(_Strict):
    """Один трек набора: эталонный человек и его кадры по порядку."""

    track_id: str = Field(min_length=1, max_length=100)
    person_id: str = Field(min_length=1, max_length=100)
    frames: list[BenchmarkFrame] = Field(min_length=1)


class BenchmarkSet(_Strict):
    """Размеченный набор треков одного дома (ТЗ 15.6)."""

    format: int = 1
    home_id: str = Field(default="", max_length=64)
    #: Чем снимали — попадает в отчёт, чтобы числа нельзя было перепутать.
    source: str = Field(default="", max_length=200)
    tracks: list[BenchmarkTrack] = Field(default_factory=list)

    @field_validator("tracks")
    @classmethod
    def _tracks_are_unique(cls, tracks: list[BenchmarkTrack]) -> list[BenchmarkTrack]:
        seen = [track.track_id for track in tracks]
        if len(set(seen)) != len(seen):
            raise ValueError("track_id must be unique inside one set")
        return tracks

    def views(self) -> dict[str, int]:
        counts = {view: 0 for view in VIEWS}
        for track in self.tracks:
            for frame in track.frames:
                counts[frame.view] = counts.get(frame.view, 0) + 1
        return counts


class Report(BaseModel):
    """Что набор показал; ``passed`` — по критериям ТЗ, а не по желанию."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    tracks: int = 0
    named: int = 0
    correct: int = 0
    wrong: int = 0
    unknown: int = 0
    precision: float = 0.0
    recall: float = 0.0
    back_cases: int = 0
    back_correct: int = 0
    back_rate: float | None = None
    by_view: dict[str, dict[str, int]] = Field(default_factory=dict)
    views: dict[str, int] = Field(default_factory=dict)
    passed: bool = False
    reasons: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _back_frames(track: BenchmarkTrack) -> list[int]:
    """Indices of back frames that come AFTER a frontal frame (ТЗ 15.6)."""
    if not any(frame.view == "frontal" for frame in track.frames):
        return []
    frontal_seen = False
    found: list[int] = []
    for index, frame in enumerate(track.frames):
        if frame.view == "frontal":
            frontal_seen = True
        elif frame.view == "back" and frontal_seen:
            found.append(index)
    return found


def evaluate(tracks: Sequence[BenchmarkTrack], predict: Predictor, *,
             min_tracks: int = MIN_TRACKS, min_back_rate: float = MIN_BACK_RATE) -> Report:
    """Run ``predict(track, frame_index)`` over a labelled set and score it.

    The final answer of a track is what the system decided by its LAST frame —
    that is the state the room would act on. The back-view check is separate:
    it asks whether a back frame still belongs to the person identified from
    the frontal frames before it.
    """
    report = Report(tracks=len(tracks))
    by_view: dict[str, dict[str, int]] = {view: {"frames": 0, "correct": 0, "unknown": 0}
                                          for view in VIEWS}
    for track in tracks:
        truth = str(track.person_id)
        for index, frame in enumerate(track.frames):
            answer = str(predict(track, index) or "")
            bucket = by_view.setdefault(frame.view, {"frames": 0, "correct": 0, "unknown": 0})
            bucket["frames"] += 1
            if not answer:
                bucket["unknown"] += 1
            elif answer == truth:
                bucket["correct"] += 1
        final = str(predict(track, len(track.frames) - 1) or "")
        if not final:
            report.unknown += 1
        elif final == truth:
            report.named += 1
            report.correct += 1
        else:
            report.named += 1
            report.wrong += 1
        for index in _back_frames(track):
            report.back_cases += 1
            if str(predict(track, index) or "") == truth:
                report.back_correct += 1
    report.by_view = by_view
    report.precision = (report.correct / report.named) if report.named else 0.0
    report.recall = (report.correct / report.tracks) if report.tracks else 0.0
    report.back_rate = (report.back_correct / report.back_cases) if report.back_cases else None
    report.reasons = _reasons(report, min_tracks=min_tracks, min_back_rate=min_back_rate)
    report.passed = not report.reasons
    return report


def _reasons(report: Report, *, min_tracks: int, min_back_rate: float) -> list[str]:
    reasons: list[str] = []
    if report.tracks < min_tracks:
        reasons.append(f"в наборе {report.tracks} трек(ов), а ТЗ 15.6 требует не меньше {min_tracks}")
    if report.back_cases == 0:
        reasons.append("нет ни одного трека «спина после фронтального кадра» — критерий не проверен")
    elif report.back_rate is not None and report.back_rate < min_back_rate:
        reasons.append(
            f"со спины после одного фронтального кадра {report.back_rate * 100:.1f} %, "
            f"а нужно не хуже {min_back_rate * 100:.0f} %")
    if report.tracks and report.named == 0:
        reasons.append("система не назвала ни одного человека — точность считать не на чем")
    return reasons


__all__ = [
    "MIN_BACK_RATE",
    "MIN_TRACKS",
    "VIEWS",
    "BenchmarkFrame",
    "BenchmarkSet",
    "BenchmarkTrack",
    "Prediction",
    "Predictor",
    "Report",
    "View",
    "evaluate",
]
