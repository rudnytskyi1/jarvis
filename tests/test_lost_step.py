"""Половина просьбы, которую модель потеряла (D-04, AU-21).

Основание — массовый аудит 2026-09-23. «Save a photo and put it on my wallpaper»
в трёх прогонах из трёх возвращался сделанным наполовину, и терялась каждый раз
РАЗНАЯ половина (`docs/AUDIT_MASS.md`, задача AU-21). Открытие было уже
записано (AUDIT-16d): живой хаб после хода не останавливается на первом плане
модели, он задаёт вопрос D-04 «результат совпадает с просьбой?» и даёт модели
ещё один круг, чтобы доделать потерянное (`hub/app.py::chat.verify`), — а стенд
этот проход не делал и мерил только первый план.

Правило взято то же, что и у шага с сайтом (TG-06): `verify_actions: false`
выключает РУТИННУЮ проверку, а не шаг, который доказанно не сделан. Здесь
проверяются обе половины: сам детектор потерянного шага
(``hub/decision_points.py::named_step_unfinished``) и гейт стенда, который по
нему запускает самопроверку (``scripts/live-eval.py::self_check_needed``).
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hub.decision_points import (  # noqa: E402 - path first, then the hub
    action_result_heuristic,
    any_step_unfinished,
    named_step_unfinished,
)


def _load_bench() -> Any:
    """Сам стенд: его гейт самопроверки — часть проверяемого правила.

    Имя файла с дефисом не импортируется как модуль, поэтому загрузка та же,
    что у ``tests/audit/test_request_matrix.py``.
    """
    path = REPO_ROOT / "scripts" / "live-eval.py"
    spec = importlib.util.spec_from_file_location("live_eval_bench", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bench = _load_bench()


@pytest.mark.parametrize("text", [
    "save a photo and put it on my wallpaper",
    "Rowan, save a photo and put it on my wallpaper please",
    "save this picture to the desktop and open it",
    "save a camera photo to my desktop",
    "сохрани фото и поставь на обои",
])
def test_a_request_that_names_a_picture_step_is_recognised(text: str) -> None:
    assert named_step_unfinished(text, []) is True


@pytest.mark.parametrize("text", [
    "",
    None,
    "what is on the screen",
    "who is in the room",
    "turn the volume up",
    "show the screen on the overlay",
    "open youtube and turn the volume up",
])
def test_ordinary_requests_name_no_lost_step(text: str | None) -> None:
    """Гейт обязан молчать там, где просьба не называет такого шага.

    Лишний круг модели на обычном разговоре — это ровно та цена, которой
    боялся TG-06, поэтому список слов узкий: «покажи экран», «что на экране»
    и «кто в комнате» здесь не проходят.
    """
    assert named_step_unfinished(text, []) is False


def test_the_wallpaper_half_is_lost_when_only_the_photo_was_saved() -> None:
    said = "save a photo and put it on my wallpaper"
    saved = {"tool": "save_photo", "args": {"source": "camera"}}
    papered = {"tool": "set_wallpaper", "args": {"source": "last"}}
    assert named_step_unfinished(said, [saved]) is True
    assert named_step_unfinished(said, [papered]) is True
    assert named_step_unfinished(said, [saved, papered]) is False


def test_the_camera_half_is_lost_when_the_model_only_looked() -> None:
    """AU-0996: «show the camera and save a screenshot» модель ответила
    взглядом (`look_at_camera` + `look_at_screen`) — ни показа, ни файла."""
    said = "Rowan, show the camera and save a screenshot please"
    looked = [{"tool": "look_at_camera", "args": {}},
              {"tool": "look_at_screen", "args": {}}]
    done = [{"tool": "show_photo", "args": {"which": "camera"}},
            {"tool": "save_photo", "args": {"source": "screen"}}]
    assert named_step_unfinished(said, looked) is True
    assert named_step_unfinished(said, done) is False


def test_a_wallpaper_made_by_drawing_it_counts_as_done() -> None:
    """«Сделай обои из кота» — это `generate_image`, а не «шаг потерян»."""
    said = "make a wallpaper of a cat wearing a top hat"
    assert named_step_unfinished(
        said, [{"tool": "generate_image", "args": {"prompt": "cat wallpaper"}}]) is False


def test_the_hub_gate_forces_the_self_check_for_a_lost_half() -> None:
    """`verify_actions: false` — про рутинную проверку, а не про потерянный шаг."""
    said = "save a photo and put it on my wallpaper"
    half = [{"tool": "save_photo", "args": {"source": "camera"}}]
    assert any_step_unfinished(said, half) is True
    assert any_step_unfinished(said, half + [{"tool": "set_wallpaper", "args": {}}]) is False
    assert action_result_heuristic(changed_state=False, imperative_without_tool=False,
                                   unfinished_step=True) is True


def test_the_bench_gate_matches_the_hub_including_the_off_switches() -> None:
    said = "save a photo and put it on my wallpaper"
    half = [{"tool": "save_photo", "args": {}}]
    whole = half + [{"tool": "set_wallpaper", "args": {}}]
    # Потерянный шаг проверяется, даже когда рутинная проверка выключена —
    # ровно как в живом конфиге владельца (`verify_actions: false`).
    assert _bench.self_check_needed(said, half, "Saved it.", changed_state=False,
                                    verify_actions=False) is True
    # Полностью сделанная пара самопроверки не требует.
    assert _bench.self_check_needed(said, whole, "Both done.", changed_state=True,
                                    verify_actions=False) is False
    # Обычный разговор без действий — не повод звать проверку.
    assert _bench.self_check_needed("what is on the screen", [], "It is a desktop.",
                                    changed_state=False, verify_actions=True) is False


class _Verifier:
    """Стенд-двойник ``LlmClient``: считает вызовы самопроверки."""

    def __init__(self, text: str = "Both halves are done now.") -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.text = text

    async def verify(self, history: Any, reply: str, executor: Any) -> Any:
        self.calls.append((history, reply))
        return SimpleNamespace(text=self.text, rounds=1, tool_calls=[], history=history)


def _bench_with(actions: list[dict[str, Any]], *, changed: bool = False,
                verify_actions: bool = True) -> Any:
    from common.config import Config

    bench = object.__new__(_bench.Bench)
    bench.cfg = Config()
    bench.cfg.server.llm.verify_actions = verify_actions
    bench.connection = SimpleNamespace(_utterance_actions=list(actions),
                                       _turn_changed_state=lambda: changed)
    bench._llm = _Verifier()
    bench._turn_started = 0.0
    bench.self_checked = False
    bench.self_check_rounds = 0

    async def execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    bench.execute = execute
    return bench


def test_the_bench_runs_the_verifier_after_a_half_finished_turn() -> None:
    """Стенд обязан звать `verify` там же, где его зовёт живой хаб."""
    bench = _bench_with([{"tool": "save_photo", "args": {}}])
    result = SimpleNamespace(history=[{"role": "user", "content": "hi"}],
                             text="Saved the photo.")
    reply, extra = asyncio.run(bench._after_turn(
        "save a photo and put it on my wallpaper", result, result.text))
    assert bench._llm.calls == [(result.history, result.text)]
    assert reply == "Both halves are done now."
    assert extra == 1 and bench.self_checked is True


def test_the_bench_keeps_the_first_reply_when_nothing_was_lost() -> None:
    """Полностью сделанная пара: рутинная проверка выключена, как у владельца."""
    bench = _bench_with([{"tool": "save_photo", "args": {}},
                         {"tool": "set_wallpaper", "args": {}}], changed=True,
                        verify_actions=False)
    result = SimpleNamespace(history=[], text="Both done.")
    reply, extra = asyncio.run(bench._after_turn(
        "save a photo and put it on my wallpaper", result, result.text))
    assert bench._llm.calls == []
    assert reply == "Both done." and extra == 0 and bench.self_checked is False


def test_the_routine_self_check_still_runs_when_the_owner_switched_it_on() -> None:
    """Флаг владельца не потерялся: с `verify_actions: true` проверка идёт."""
    bench = _bench_with([{"tool": "save_photo", "args": {}},
                         {"tool": "set_wallpaper", "args": {}}], changed=True,
                        verify_actions=True)
    result = SimpleNamespace(history=[], text="Both done.")
    reply, extra = asyncio.run(bench._after_turn(
        "save a photo and put it on my wallpaper", result, result.text))
    assert bench._llm.calls == [([], "Both done.")]
    assert reply == "Both halves are done now." and extra == 1 and bench.self_checked
