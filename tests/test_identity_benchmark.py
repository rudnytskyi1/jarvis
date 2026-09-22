"""Набор идентичности и метрики (ТЗ 15.6).

ТЗ 15.6 требует набор не меньше чем из 20 треков (спереди, сбоку, спина,
смена ракурса) с эталонным ``person_id`` и метрики точности и полноты; критерий
фазы 2 добавляет «со спины после одного фронтального кадра — не хуже 80 %».

Здесь проверяется настоящая арифметика метрик, формат набора, сборка набора из
раскладки записей и честный отказ прогона, когда на машине нет моделей или
записей. Живых записей и GPU в песочнице нет — прогон по настоящему набору
делается на стенде (см. `docs/TZ_STATUS.md`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.identity_metrics import (
    MIN_BACK_RATE,
    MIN_TRACKS,
    BenchmarkFrame,
    BenchmarkSet,
    BenchmarkTrack,
    evaluate,
)
from scripts.identity_benchmark import (
    EXIT_BLOCKED,
    EXIT_FAILED,
    EXIT_OK,
    load_manifest,
    main,
    run,
    scan_recordings,
)


def track(track_id: str, person_id: str, views: str, *, frames: int = 1) -> BenchmarkTrack:
    """A track whose frames carry the views in ``views`` (one frame each)."""
    names = {"f": "frontal", "s": "side", "b": "back", "t": "turn"}
    return BenchmarkTrack(
        track_id=track_id, person_id=person_id,
        frames=[BenchmarkFrame(path=f"{person_id}/{names[view]}/{index}.jpg",
                               view=names[view])
                for index, view in enumerate(views) for _ in range(frames)])


def make_set(tracks: int, *, correct_back: int | None = None) -> list[BenchmarkTrack]:
    """A labelled set of ``tracks`` tracks, each frontal + side + back."""
    rows = [track(f"t-{index:02d}", f"p-{index:02d}", "fsb") for index in range(tracks)]
    return rows


# --- метрики -----------------------------------------------------------------


def test_a_perfect_system_scores_every_metric_full():
    rows = make_set(MIN_TRACKS)
    report = evaluate(rows, lambda case, index: case.person_id)
    assert report.tracks == MIN_TRACKS
    assert report.named == MIN_TRACKS and report.correct == MIN_TRACKS
    assert report.precision == 1.0 and report.recall == 1.0
    assert report.back_cases == MIN_TRACKS and report.back_rate == 1.0
    assert report.passed is True and report.reasons == []
    assert report.by_view["back"]["correct"] == MIN_TRACKS


def test_an_unknown_answer_costs_recall_but_not_precision():
    rows = make_set(MIN_TRACKS)
    report = evaluate(rows, lambda case, index: "" if case.track_id.endswith("0") else case.person_id)
    assert report.unknown == 2 and report.named == MIN_TRACKS - 2
    assert report.precision == 1.0
    assert report.recall == pytest.approx((MIN_TRACKS - 2) / MIN_TRACKS)
    assert report.back_rate == pytest.approx((MIN_TRACKS - 2) / MIN_TRACKS)


def test_a_wrong_name_costs_precision_and_recall():
    rows = make_set(MIN_TRACKS)

    def wrong(case: BenchmarkTrack, index: int) -> str:
        return "p-nobody" if case.track_id == "t-00" else case.person_id

    report = evaluate(rows, wrong)
    assert report.wrong == 1 and report.named == MIN_TRACKS
    assert report.precision == pytest.approx((MIN_TRACKS - 1) / MIN_TRACKS)
    assert report.recall == pytest.approx((MIN_TRACKS - 1) / MIN_TRACKS)
    assert report.passed is True, "одна ошибка не отменяет критерий со спины"


def test_the_back_view_criterion_is_measured_on_its_own():
    rows = make_set(10)
    seen: dict[str, bool] = {}

    def loses_the_track(case: BenchmarkTrack, index: int) -> str:
        # Two of the ten tracks lose the person at the back view.
        if case.frames[index].view == "back" and case.track_id in {"t-08", "t-09"}:
            seen[case.track_id] = True
            return ""
        return case.person_id

    report = evaluate(rows, loses_the_track, min_tracks=10)
    assert report.back_cases == 10 and report.back_correct == 8
    assert report.back_rate == pytest.approx(0.8)
    assert report.passed is True, "ровно 80 % — это ещё «не хуже 80 %»"

    def loses_three(case: BenchmarkTrack, index: int) -> str:
        if case.frames[index].view == "back" and case.track_id >= "t-07":
            return ""
        return case.person_id

    worse = evaluate(rows, loses_three, min_tracks=10)
    assert worse.back_rate == pytest.approx(0.7)
    assert worse.passed is False
    assert any("80 %" in reason for reason in worse.reasons)
    assert seen, "спинные кадры действительно проверялись"


def test_the_set_must_be_big_enough_to_mean_anything():
    rows = make_set(MIN_TRACKS - 1)
    report = evaluate(rows, lambda case, index: case.person_id)
    assert report.passed is False
    assert any(str(MIN_TRACKS) in reason for reason in report.reasons)


def test_a_set_without_a_back_view_cannot_pass():
    rows = [track(f"t-{index:02d}", f"p-{index:02d}", "fs") for index in range(MIN_TRACKS)]
    report = evaluate(rows, lambda case, index: case.person_id)
    assert report.back_cases == 0 and report.back_rate is None
    assert report.passed is False
    assert any("спина" in reason for reason in report.reasons)


def test_a_back_frame_without_a_frontal_one_before_it_does_not_count():
    rows = [track("t-00", "p-00", "bs")] + make_set(MIN_TRACKS - 1)
    report = evaluate(rows, lambda case, index: case.person_id)
    assert report.back_cases == MIN_TRACKS - 1, "спина без фронта ничего не проверяет"
    assert report.tracks == MIN_TRACKS


def test_a_silent_system_is_not_called_accurate():
    rows = make_set(MIN_TRACKS)
    report = evaluate(rows, lambda case, index: "")
    assert report.precision == 0.0 and report.recall == 0.0
    assert report.passed is False
    assert any("не назвала ни одного" in reason for reason in report.reasons)


def test_the_report_carries_the_view_counts_it_was_built_from():
    dataset = BenchmarkSet(tracks=make_set(MIN_TRACKS))
    assert dataset.views() == {"frontal": MIN_TRACKS, "side": MIN_TRACKS,
                              "back": MIN_TRACKS, "turn": 0}
    assert MIN_BACK_RATE == 0.8


# --- формат набора -----------------------------------------------------------


def test_an_unknown_view_is_refused_not_ignored():
    with pytest.raises(Exception):
        BenchmarkTrack(track_id="t", person_id="p",
                       frames=[BenchmarkFrame(path="a.jpg", view="ceiling")])
    with pytest.raises(Exception):
        BenchmarkTrack(track_id="t", person_id="p", frames=[])
    with pytest.raises(Exception):
        BenchmarkSet(tracks=[track("t", "p", "f"), track("t", "p2", "f")])
    with pytest.raises(Exception):
        BenchmarkTrack(track_id="t", person_id="p",
                       frames=[BenchmarkFrame(path="a.jpg", view="frontal", extra=1)])


def test_a_manifest_that_is_missing_or_broken_is_reported(tmp_path):
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "nope.json")
    broken = tmp_path / "manifest.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(broken)
    broken.write_text(json.dumps({"tracks": [{"track_id": "t"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(broken)


def test_a_manifest_round_trips(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"home_id": "livingroom", "source": "телефон",
                                "tracks": [row.model_dump() for row in make_set(3)]},
                               ensure_ascii=False), encoding="utf-8")
    dataset = load_manifest(path)
    assert len(dataset.tracks) == 3 and dataset.home_id == "livingroom"
    assert dataset.tracks[0].frames[2].view == "back"


# --- сборка набора из записей ------------------------------------------------


def make_recordings(root: Path, people: int, *, back: bool = True) -> None:
    for index in range(people):
        person = f"p-{index:02d}"
        for view in ("frontal", "side", "back" if back else "turn"):
            folder = root / person / view
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "001.jpg").write_bytes(b"jpeg")


def test_the_set_is_built_from_the_recording_layout(tmp_path):
    make_recordings(tmp_path, MIN_TRACKS)
    dataset = scan_recordings(tmp_path)
    assert len(dataset.tracks) == MIN_TRACKS
    assert dataset.tracks[0].person_id == "p-00"
    assert [frame.view for frame in dataset.tracks[0].frames] == ["frontal", "side", "back"]
    assert dataset.views()["back"] == MIN_TRACKS


def test_a_folder_without_views_is_not_a_track(tmp_path):
    make_recordings(tmp_path, 1)
    (tmp_path / "empty").mkdir()
    (tmp_path / "p-00" / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "p-00" / "ceiling").mkdir()
    (tmp_path / "p-00" / "ceiling" / "001.jpg").write_bytes(b"jpeg")
    dataset = scan_recordings(tmp_path)
    assert [row.track_id for row in dataset.tracks] == ["p-00"]
    assert dataset.views()["turn"] == 0, "незнакомая папка не становится ракурсом"


def test_scanning_an_empty_folder_is_not_a_crash(tmp_path):
    dataset = scan_recordings(tmp_path / "nothing")
    assert dataset.tracks == []


def test_scan_writes_the_manifest_and_warns_about_a_small_set(tmp_path, capsys):
    recordings = tmp_path / "records"
    make_recordings(recordings, 3)
    manifest = tmp_path / "manifest.json"
    code = main(["--scan", str(recordings), "--manifest", str(manifest)])
    assert code == EXIT_OK
    assert "3 трек" in capsys.readouterr().out
    assert len(load_manifest(manifest).tracks) == 3
    assert main(["--scan", str(tmp_path / "empty-records"), "--manifest", str(tmp_path / "m2.json")]) == EXIT_BLOCKED


# --- прогон ------------------------------------------------------------------


def test_a_run_without_a_manifest_says_what_is_missing(tmp_path, capsys):
    report, blockers = run(tmp_path / "manifest.json")
    assert report is None and blockers and "набора нет" in blockers[0]
    assert main(["--manifest", str(tmp_path / "manifest.json")]) == EXIT_BLOCKED
    assert "набора нет" in capsys.readouterr().err


def test_a_run_without_models_reports_the_blockers(tmp_path, capsys):
    make_recordings(tmp_path, MIN_TRACKS)
    dataset = scan_recordings(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(dataset.model_dump()),
                                            encoding="utf-8")
    report, blockers = run(tmp_path / "manifest.json")
    if report is None:
        assert blockers, "отказ обязан называть причину"
        assert "Ничего не выдумано" in (main(["--manifest", str(tmp_path / "manifest.json")])
                                        and capsys.readouterr().err)
    else:
        # На стенде с моделями и записями прогон доходит до метрик.
        assert report.tracks == MIN_TRACKS


def test_a_supplied_predictor_scores_the_set(tmp_path):
    make_recordings(tmp_path, MIN_TRACKS)
    dataset = scan_recordings(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dataset.model_dump()), encoding="utf-8")
    report, blockers = run(manifest, predictor=lambda case, index: case.person_id)
    assert blockers == []
    assert report is not None and report.passed is True
    assert report.recall == 1.0 and report.back_rate == 1.0


def test_the_cli_passes_and_fails_on_the_real_metric(tmp_path, capsys):
    make_recordings(tmp_path, MIN_TRACKS)
    dataset = scan_recordings(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dataset.model_dump()), encoding="utf-8")
    assert run(manifest, predictor=lambda case, index: "")[0].passed is False
    code = main(["--check", "--manifest", str(manifest)])
    assert code in (EXIT_OK, EXIT_BLOCKED)
    assert capsys.readouterr().out.strip(), "проверка машины что-то печатает"


def test_the_failed_run_exits_with_the_failed_code(tmp_path, monkeypatch, capsys):
    import scripts.identity_benchmark as benchmark

    make_recordings(tmp_path, MIN_TRACKS)
    dataset = scan_recordings(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dataset.model_dump()), encoding="utf-8")
    monkeypatch.setattr(benchmark, "face_predictor", lambda base: (lambda case, index: ""))
    monkeypatch.setattr(benchmark, "checked_manifest", lambda base: {"blockers": []})
    assert main(["--manifest", str(manifest)]) == EXIT_FAILED
    assert "НЕ ПРОЙДЕНО" in capsys.readouterr().err
