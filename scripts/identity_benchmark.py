"""Прогон набора идентичности (ТЗ 15.6).

Набор — это размеченные треки: для каждого известно, кто в нём (эталонный
``person_id``), и сняты кадры спереди, сбоку и со спины. Скрипт прогоняет по
ним НАСТОЯЩИЙ путь распознавания хаба (insightface через ``hub/face.py``,
удержание личности на треке через ``hub/room_state.py``), считает точность и
полноту, отдельно проверяет «со спины после одного фронтального кадра ≥ 80 %»
и честно говорит, чего не хватает для прогона.

Раскладка записей (её и ждёт команда):

    data/identity_benchmark/
        manifest.json
        p-max/frontal/001.jpg  p-max/side/002.jpg  p-max/back/003.jpg
        p-anton/frontal/...

``--scan <папка>`` собирает ``manifest.json`` из такой раскладки (имя папки —
это ``person_id``), ``--check`` только докладывает о машине. Без insightface,
без записей или без профилей людей скрипт выходит с кодом 2 и печатает, чего
именно нет: выдуманных чисел он не выдаёт.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.identity_metrics import (  # noqa: E402 - path is fixed above
    MIN_BACK_RATE,
    MIN_TRACKS,
    VIEWS,
    BenchmarkFrame,
    BenchmarkSet,
    BenchmarkTrack,
    Predictor,
    Report,
    evaluate,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "identity_benchmark" / "manifest.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
#: Сколько «живёт» личность на треке при прогоне (как в комнате: ~5 кадров/с).
FRAME_STEP_S = 0.5
EXIT_OK, EXIT_FAILED, EXIT_BLOCKED = 0, 1, 2


def load_manifest(path: Path) -> BenchmarkSet:
    """Read the labelled set; a broken file is an error, never a silent empty set."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"набора нет: {path}") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"набор не читается ({type(exc).__name__}): {path}") from exc
    try:
        return BenchmarkSet.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a bad set must be reported, not raised
        raise ValueError(f"набор не по формату: {type(exc).__name__}") from exc


def scan_recordings(root: Path, *, source: str = "") -> BenchmarkSet:
    """Build a set from ``<person_id>/<view>/<image>`` folders (ТЗ 15.6).

    The view folders are the ones the ТЗ names; anything else is ignored, so a
    stray file cannot silently become a "track". Frames keep their file order.
    """
    base = Path(root)
    tracks: list[BenchmarkTrack] = []
    if not base.is_dir():
        return BenchmarkSet(source=source)
    for person_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        frames: list[BenchmarkFrame] = []
        for view in VIEWS:
            view_dir = person_dir / view
            if not view_dir.is_dir():
                continue
            for image in sorted(view_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
                    continue
                frames.append(BenchmarkFrame(path=str(image.relative_to(base)), view=view))
        if not frames:
            continue
        tracks.append(BenchmarkTrack(track_id=person_dir.name, person_id=person_dir.name,
                                     frames=frames))
    return BenchmarkSet(source=source or str(base), tracks=tracks)


def checked_manifest(base: Path) -> dict[str, Any]:
    """Everything a run needs, or the honest list of what is missing."""
    blockers: list[str] = []
    try:
        from hub.face import FaceEngine, insightface_installed
    except Exception as exc:  # noqa: BLE001 - the hub module must be importable
        return {"blockers": [f"hub/face.py не импортируется: {type(exc).__name__}"]}
    if not insightface_installed():
        blockers.append("insightface не установлен — лица распознавать нечем")
    profiles = face_profiles()
    if not profiles:
        blockers.append("нет профилей лиц (data/people.json): некого узнавать")
    if not people_by_name():
        blockers.append("нет таблицы persons: имена профилей не с чем сверить")
    engine = FaceEngine()
    if not engine.available:
        blockers.append(f"модель лиц не поднялась (провайдер {engine.provider or 'нет'})")
    if not base.is_dir():
        blockers.append(f"записей нет: {base}")
    return {"blockers": blockers, "engine": engine, "profiles": profiles,
            "people": people_by_name()}


def face_profiles() -> dict[str, list[list[float]]]:
    """Enrolled face samples, exactly as the hub's matcher takes them."""
    from hub.speaker import VoiceRegistry

    try:
        registry = VoiceRegistry(REPO_ROOT / "data", enabled=False)
    except Exception as exc:  # noqa: BLE001 - no registry means nothing to match
        print(f"Профили не читаются ({type(exc).__name__}); продолжаю без них.",
              file=sys.stderr)
        return {}
    return registry.face_profiles()


def people_by_name(db_path: Path | None = None) -> dict[str, str]:
    """``display_name`` -> ``person_id`` from the hub's own table (ТЗ 14)."""
    import sqlite3

    path = Path(db_path) if db_path is not None else REPO_ROOT / "data" / "hub.db"
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        rows = conn.execute("SELECT display_name, person_id FROM persons").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {str(name): str(person_id) for name, person_id in rows if name and person_id}


def face_predictor(base: Path) -> Predictor | None:
    """The real predictor: identify from the frontal frames, keep the track."""
    from hub.face import FaceEngine
    from hub.room_state import RoomState

    engine = FaceEngine()
    if not engine.available:
        return None
    profiles = face_profiles()
    if not profiles:
        return None
    by_name = people_by_name()
    room = RoomState()
    cache: dict[tuple[str, int], str] = {}

    def predict(track: BenchmarkTrack, index: int) -> str:
        key = (track.track_id, index)
        if key in cache:
            return cache[key]
        geometry = [{"id": track.track_id, "box": [0.25, 0.05, 0.75, 0.95]}]
        answer = ""
        for position, frame in enumerate(track.frames[: index + 1]):
            try:
                jpeg = (base / frame.path).read_bytes()
            except OSError:
                answer = ""
                break
            located = engine.located_faces(jpeg)
            resolved = room.resolve_faces(located, geometry, engine.match, profiles,
                                          now=position * FRAME_STEP_S)
            names = [item.get("name") for item in resolved if item.get("name")]
            if not names:
                continue
            # Имя трека — то, что хаб знает о нём ПРЯМО СЕЙЧАС (F-204/F-207).
            row = room.tracks.get(track.track_id) or {}
            answer = by_name.get(str(row.get("name") or names[-1]), "")
        cache[key] = answer
        return answer

    return predict


def run(manifest_path: Path = DEFAULT_MANIFEST, *,
        predictor: Predictor | None = None) -> tuple[Report | None, list[str]]:
    """Score one set; ``None`` plus reasons when a run is impossible."""
    try:
        dataset = load_manifest(manifest_path)
    except ValueError as exc:
        return None, [str(exc)]
    if predictor is None:
        check = checked_manifest(Path(manifest_path).parent)
        if check.get("blockers"):
            return None, list(check["blockers"])
        predictor = face_predictor(Path(manifest_path).parent)
        if predictor is None:
            return None, ["путь распознавания лиц недоступен на этой машине"]
    report = evaluate(dataset.tracks, predictor)
    report.views = dataset.views()
    return report, []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Identity benchmark (ТЗ 15.6)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="размеченный набор (JSON)")
    parser.add_argument("--scan", default="", help="собрать набор из папки записей")
    parser.add_argument("--check", action="store_true", help="только доложить о машине")
    args = parser.parse_args(argv)

    if args.scan:
        root = Path(args.scan)
        dataset = scan_recordings(root)
        out = Path(args.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dataset.model_dump(), ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"Набор собран: {len(dataset.tracks)} трек(ов) в {out}")
        if len(dataset.tracks) < MIN_TRACKS:
            print(f"Внимание: ТЗ 15.6 требует не меньше {MIN_TRACKS} треков.",
                  file=sys.stderr)
        return EXIT_OK if dataset.tracks else EXIT_BLOCKED

    if args.check:
        check = checked_manifest(Path(args.manifest).parent)
        print(json.dumps({"blockers": check.get("blockers", []),
                          "tracks_required": MIN_TRACKS,
                          "back_rate_required": MIN_BACK_RATE}, ensure_ascii=False, indent=2))
        return EXIT_BLOCKED if check.get("blockers") else EXIT_OK

    report, blockers = run(Path(args.manifest))
    if report is None:
        print("Прогон невозможен:", file=sys.stderr)
        for reason in blockers:
            print(f"  - {reason}", file=sys.stderr)
        print("Ничего не выдумано: соберите записи и модели на стенде "
              "(см. ТЗ 15.6 и DECISIONS.md P2-36).", file=sys.stderr)
        return EXIT_BLOCKED
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    if not report.passed:
        for reason in report.reasons:
            print(f"НЕ ПРОЙДЕНО: {reason}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Пройдено: точность {report.precision * 100:.1f} %, "
          f"полнота {report.recall * 100:.1f} %, "
          f"спина после фронта {report.back_rate * 100:.1f} %")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
