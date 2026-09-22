"""ТЗ F-312: profiles of the client detector, chosen by a real measurement."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from client import vision_profile as vision
from client.camera import CameraService
from common.client_config import CameraConfig, VisionProfileConfig, default_vision_profiles
from scripts import export_tensorrt


def test_the_tz_profiles_are_the_ones_the_spec_names():
    entries = default_vision_profiles()
    assert list(entries) == ["tensorrt", "gpu", "weak"]  # сильный → слабый
    assert entries["tensorrt"].model == "yolo11x.engine"
    assert entries["tensorrt"].half is True
    # «профиль для слабых ПК: YOLO11s и треки 10 FPS» — ровно это.
    assert entries["weak"].model == "yolo11s.pt"
    assert entries["weak"].fps == 10
    assert entries["weak"].half is False


def test_plan_keeps_the_config_order_and_skips_a_missing_engine(tmp_path):
    camera = SimpleNamespace(profiles=default_vision_profiles())
    usable, skipped = vision.plan_profiles(camera, base_dir=tmp_path)
    assert [profile.name for profile in usable] == ["gpu", "weak"]
    assert [item.name for item in skipped] == ["tensorrt"]
    assert "yolo11x.engine" in skipped[0].reason
    assert skipped[0].latency_ms is None

    (tmp_path / "yolo11x.engine").write_bytes(b"engine")
    usable, skipped = vision.plan_profiles(camera, base_dir=tmp_path)
    assert [profile.name for profile in usable] == ["tensorrt", "gpu", "weak"]
    assert skipped == ()


def test_plan_skips_an_entry_without_a_model_and_reads_the_budgets(tmp_path):
    camera = SimpleNamespace(profiles={
        "broken": SimpleNamespace(model="", fps=10, half=True, budget_ms=30),
        "custom": SimpleNamespace(model="yolo11m.pt", fps=7, half=False, budget_ms=42),
    })
    usable, skipped = vision.plan_profiles(camera, base_dir=tmp_path)
    assert [profile.name for profile in usable] == ["custom"]
    assert usable[0].fps == 7 and usable[0].half is False and usable[0].budget_ms == 42
    assert skipped[0].name == "broken"


def test_choose_profile_takes_the_first_profile_that_fits():
    entries = [vision.Profile("tensorrt", "yolo11x.engine", 20, True, 30),
               vision.Profile("gpu", "yolo11x.pt", 10, True, 90),
               vision.Profile("weak", "yolo11s.pt", 10, False, 200)]
    latency = {"tensorrt": 18.0, "gpu": 45.0, "weak": 120.0}
    choice = vision.choose_profile(entries, lambda profile: latency[profile.name])
    assert choice is not None
    assert choice.profile.name == "tensorrt"
    assert choice.latency_ms == 18.0
    assert [item.name for item in choice.attempts] == ["tensorrt"]
    assert "30" in choice.reason
    assert "tensorrt" in choice.text() and "18" in choice.text()


def test_choose_profile_steps_down_when_the_stronger_profile_is_slow():
    entries = [vision.Profile("tensorrt", "yolo11x.engine", 20, True, 30),
               vision.Profile("gpu", "yolo11x.pt", 10, True, 90)]
    latency = {"tensorrt": 55.0, "gpu": 60.0}
    choice = vision.choose_profile(entries, lambda profile: latency[profile.name])
    assert choice is not None
    assert choice.profile.name == "gpu"
    assert [(item.name, item.reason) for item in choice.attempts] == [
        ("tensorrt", "бюджет 30 мс превышен"),
        ("gpu", ""),
    ]


def test_choose_profile_keeps_the_weakest_measured_when_nothing_fits():
    entries = [vision.Profile("gpu", "yolo11x.pt", 10, True, 90),
               vision.Profile("weak", "yolo11s.pt", 10, False, 200)]
    choice = vision.choose_profile(entries, lambda profile: 400.0)
    assert choice is not None
    assert choice.profile.name == "weak"
    assert choice.latency_ms == 400.0
    assert "бюджет" in choice.reason


def test_a_profile_whose_measurement_failed_is_skipped_not_scored():
    entries = [vision.Profile("tensorrt", "yolo11x.engine", 20, True, 30),
               vision.Profile("gpu", "yolo11x.pt", 10, True, 90)]

    def measure(profile):
        if profile.name == "tensorrt":
            raise RuntimeError("CUDA out of memory")
        return 40.0

    choice = vision.choose_profile(entries, measure)
    assert choice is not None
    assert choice.profile.name == "gpu"
    assert choice.attempts[0].latency_ms is None
    assert "CUDA out of memory" in choice.attempts[0].reason


def test_nothing_measurable_means_no_choice_at_all():
    entries = [vision.Profile("gpu", "yolo11x.pt", 10, True, 90)]

    def measure(profile):
        raise RuntimeError("no CUDA")

    assert vision.choose_profile(entries, measure) is None
    assert vision.choose_profile([], lambda profile: 1.0) is None


def test_measure_ms_times_real_runs_and_excludes_the_warmup(monkeypatch):
    ticks = [0.0, 0.020, 1.0, 1.060, 2.0, 2.040]
    stream = iter(ticks)
    monkeypatch.setattr(vision.time, "perf_counter", lambda: next(stream))
    calls = []

    result = vision.measure_ms(lambda: calls.append(1), frames=3, warmup=1)

    assert len(calls) == 4  # один прогрев + три измеренных
    assert result == pytest.approx(40.0)  # медиана из 20, 60 и 40 мс


# --- the camera uses the choice ---------------------------------------------


class FakeYolo:
    """Ultralytics stand-in that records which model files were loaded."""

    loaded: list[str] = []

    def __init__(self, name: str) -> None:
        self.name = name
        FakeYolo.loaded.append(name)

    def predict(self, **kwargs):
        return []


def camera_profiles(tmp_path) -> dict[str, VisionProfileConfig]:
    """The three ТЗ profiles, with the engine path explicit (никогда не в CWD)."""
    return {
        "tensorrt": VisionProfileConfig(model=str(tmp_path / "yolo11x.engine"),
                                        fps=20, half=True, budget_ms=30),
        "gpu": VisionProfileConfig(model="yolo11x.pt", fps=10, half=True, budget_ms=90),
        "weak": VisionProfileConfig(model="yolo11s.pt", fps=10, half=False, budget_ms=200),
    }


def camera_service(**camera) -> CameraService:
    cfg = CameraConfig(**camera)
    service = CameraService(cfg)
    service._frame_lock = threading.Lock()
    service._frame = "frame"
    service._frame_ts = 0.0
    return service


def test_the_automatic_choice_is_off_until_the_config_turns_it_on():
    service = camera_service()
    assert service.auto_profile is False
    assert service.profile_name == ""
    # Ровно значения конфига: фаза 2 не меняется, пока флаг выключен.
    assert service.model_name == "yolo11n.pt"
    assert (service.fps, service.half) == (5.0, True)


def test_the_camera_runs_the_profile_that_measured_inside_its_budget(monkeypatch, tmp_path):
    FakeYolo.loaded = []
    service = camera_service(auto_profile=True, profiles=camera_profiles(tmp_path))
    monkeypatch.setattr(vision, "measure_ms", lambda run, **kwargs: 50.0)

    model = service._select_profile(FakeYolo)

    # tensorrt пропущен до замера: движок не собран на этой машине.
    assert FakeYolo.loaded == ["yolo11x.pt"]
    assert model is not None and model.name == "yolo11x.pt"
    assert service.profile_name == "gpu"
    assert service.profile_latency_ms == 50.0
    assert service.model_name == "yolo11x.pt"
    assert service.half is True and service.fps == 10.0
    assert service.profile_attempts[0]["profile"] == "tensorrt"
    assert service.profile_attempts[0]["latency_ms"] is None


def test_a_failed_measurement_steps_down_to_the_next_profile(monkeypatch, tmp_path):
    FakeYolo.loaded = []
    service = camera_service(auto_profile=True, profiles=camera_profiles(tmp_path))
    calls = {"n": 0}

    def measure(run, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # первый измеренный профиль (gpu) не запустился
            raise RuntimeError("cuDNN error")
        return 150.0

    monkeypatch.setattr(vision, "measure_ms", measure)
    model = service._select_profile(FakeYolo)

    assert FakeYolo.loaded == ["yolo11x.pt", "yolo11s.pt"]
    assert model is not None and model.name == "yolo11s.pt"
    assert service.profile_name == "weak"
    assert service.half is False and service.fps == 10.0
    reasons = [item["reason"] for item in service.profile_attempts]
    assert any("cuDNN error" in reason for reason in reasons)


def test_a_camera_whose_profiles_cannot_be_measured_keeps_its_configured_model(monkeypatch, tmp_path):
    FakeYolo.loaded = []
    service = camera_service(auto_profile=True, profiles=camera_profiles(tmp_path))

    def measure(run, **kwargs):
        raise RuntimeError("CUDA is unavailable")

    monkeypatch.setattr(vision, "measure_ms", measure)
    assert service._select_profile(FakeYolo) is None
    assert service.profile_name == ""
    assert service.profile_reason == "замер не удался ни на одном профиле"
    assert service.model_name == "yolo11n.pt"       # конфиг остаётся рабочим
    assert (service.fps, service.half) == (5.0, True)


def test_the_camera_without_a_captured_frame_measures_on_a_blank_one(monkeypatch, tmp_path):
    service = CameraService(CameraConfig(auto_profile=True, profiles=camera_profiles(tmp_path)))
    # Кадров ещё не было: замер идёт по пустому кадру нужного размера, а не
    # выдуманным числам, и выбор всё равно делается честно.
    frame = service._probe_frame()
    assert frame is not None and frame.shape == (service.height, service.width, 3)
    monkeypatch.setattr(vision, "measure_ms", lambda run, **kwargs: 45.0)
    assert service._select_profile(FakeYolo) is not None
    assert service.profile_name == "gpu"


def test_the_performance_report_carries_the_measured_profile(caplog):
    service = camera_service()
    service.profile_name = "gpu"
    service.profile_latency_ms = 41.0
    service.profile_reason = "укладывается в бюджет 90 мс"
    service._metrics_at = 0.0  # время «прошло»: отчёт пишется
    with caplog.at_level("INFO"):
        service._report_performance()
    assert "'profile': 'gpu'" in caplog.text
    assert "'profile_latency_ms': 41.0" in caplog.text


# --- the export script -------------------------------------------------------


def test_the_export_script_names_exactly_what_is_missing():
    assert export_tensorrt.blockers({"ultralytics": True, "cuda": True, "tensorrt": True}) == []
    missing = export_tensorrt.blockers({"ultralytics": False, "cuda": False, "tensorrt": False})
    assert len(missing) == 3
    assert any("ultralytics" in reason for reason in missing)
    assert any("TensorRT" in reason for reason in missing)
    assert any("CUDA" in reason for reason in missing)


def test_the_export_script_measures_the_environment_for_real():
    report = export_tensorrt.environment()
    assert isinstance(report["ultralytics"], bool)
    assert isinstance(report["tensorrt"], bool)
    assert isinstance(report["cuda"], bool)
    assert "torch" in report


def test_check_mode_reports_the_machine_without_building_anything(capsys):
    assert export_tensorrt.main(["--check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"ultralytics", "cuda", "tensorrt", "blockers"}


def test_an_impossible_export_exits_non_zero_and_says_why(monkeypatch, capsys):
    monkeypatch.setattr(export_tensorrt, "environment",
                        lambda: {"ultralytics": False, "cuda": False, "tensorrt": False})
    assert export_tensorrt.main([]) == 2
    error = capsys.readouterr().err
    assert "Экспорт невозможен" in error
    assert "ultralytics" in error


def test_a_failing_export_is_never_reported_as_success(monkeypatch, capsys):
    monkeypatch.setattr(export_tensorrt, "environment",
                        lambda: {"ultralytics": True, "cuda": True, "tensorrt": True,
                                 "device": "RTX 3060 Ti"})

    def boom(*args, **kwargs):
        raise RuntimeError("TensorRT refused the weights")

    monkeypatch.setattr(export_tensorrt, "export", boom)
    assert export_tensorrt.main(["--output", "client/yolo11x.engine"]) == 1
    assert "TensorRT refused the weights" in capsys.readouterr().err


def test_paths_are_resolved_against_the_repository_root():
    assert export_tensorrt.solve("client/yolo11x.engine") == (
        export_tensorrt.REPO_ROOT / "client" / "yolo11x.engine")
    absolute = export_tensorrt.solve(str(export_tensorrt.REPO_ROOT / "x.engine"))
    assert absolute.is_absolute()
