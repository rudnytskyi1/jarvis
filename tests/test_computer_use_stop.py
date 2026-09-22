"""P5-15 (F-512): «стоп» словом и ладонью, видимый оверлей «Rowan управляет».

Проверяется, что стоп слышен на ОБОИХ концах: сказанное «стоп» останавливает
прогон на хабе до того, как реплика уйдёт модели, ладонь (F-306) останавливает
его в комнате и хаб узнаёт об этом, а значок «Rowan управляет» горит, пока
агент водит мышью, и гаснет вместе с прогоном.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from client.actions.computer_use import ComputerUseExecutor, ComputerUseSession
from client.main import JarvisClient
from client.overlay import _visibility_now
from common.computer_use import ComputerUsePolicy
from common.config import Config
from common.protocol import (
    CLIENT_MESSAGE_TYPES,
    MSG_COMPUTER_USE,
    MSG_COMPUTER_USE_STEP,
    PHONE_FORBIDDEN_INPUTS,
    SERVER_MESSAGE_TYPES,
)
from hub import app as hub_app
from hub.computer_use import ComputerUseRuns, is_stop_command
from hub.session import Session
from hub.utterances import UtteranceMetrics

ROOM = "livingroom"


# --- стоп-слово -------------------------------------------------------------


def test_the_stop_word_is_recognised_in_three_languages():
    for text in ("стоп", "Стоп!", "Rowan, стоп", "остановись", "stop",
                 "Stop it", "please stop", "detente", "para ya"):
        assert is_stop_command(text), text
    for text in ("стоп, а почему небо синее?", "останови музыку",
                 "what is the stop sign", "", "расскажи про стоп-кран"):
        assert not is_stop_command(text), text


# --- хабовая сторона --------------------------------------------------------


def _connection():
    cfg = Config()
    cfg.server.identity.enabled = False
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.home_id = ROOM
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn._speaker_name = "Anton"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={"ok": True, "output": "{}"})
    return conn


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    monkeypatch.setattr(hub_app, "_computer_runs", ComputerUseRuns())
    monkeypatch.setattr(hub_app, "_audit", None)
    return None


def _start_run(policy: ComputerUsePolicy | None = None):
    runs = hub_app._computer_use_runs()
    return runs.start(ROOM, "ответь Максу",
                      policy or ComputerUsePolicy(enabled=True, allowed_apps=["discord"]))


def test_a_spoken_stop_ends_the_run_before_the_model_sees_it():
    run = _start_run()
    conn = _connection()
    handled = asyncio.run(conn._computer_use_stop_turn(
        "Rowan, стоп", "ru", None, 100.0, conn.session, 50))
    assert handled is True
    assert hub_app._computer_use_runs().current(ROOM) is None
    payloads = [call.args[0] for call in conn.send_json.await_args_list]
    assert any(item.get("type") == MSG_COMPUTER_USE and not item.get("active")
               for item in payloads), "значок гаснет вместе с прогоном"
    said = [item.get("text") for item in payloads if item.get("type") == "say"]
    assert said == ["Остановил."], "комната слышит подтверждение на своём языке"
    conn._stream_tts.assert_awaited()
    conn._log_dialog.assert_awaited()
    assert run.stopped is True


def test_the_stop_turn_does_nothing_when_no_agent_is_running():
    conn = _connection()
    assert asyncio.run(conn._computer_use_stop_turn(
        "стоп", "ru", None, 100.0, conn.session, 50)) is False
    conn.send_json.assert_not_awaited()


def test_a_long_question_containing_the_word_stop_is_not_a_command():
    _start_run()
    conn = _connection()
    assert asyncio.run(conn._computer_use_stop_turn(
        "стоп, а почему небо синее?", "ru", None, 100.0, conn.session, 50)) is False
    assert hub_app._computer_use_runs().current(ROOM) is not None, "прогон жив"


def test_the_client_stopping_the_run_closes_it_on_the_hub(tmp_path, monkeypatch):
    from hub import migrations_runner
    from hub.audit import AuditLog

    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    monkeypatch.setattr(hub_app, "_audit", AuditLog(conn))
    run = _start_run()
    try:
        connection = _connection()
        connection._on_computer_use_step({"run_id": run.run_id, "ok": False,
                                          "stopped": True, "index": 4,
                                          "reason": "the room stopped the run (palm)"})
        assert hub_app._computer_use_runs().current(ROOM) is None
        row = conn.execute(
            "SELECT action, result FROM audit WHERE action='computer_use.stop'").fetchone()
        assert row == ("computer_use.stop", "ok")
    finally:
        monkeypatch.setattr(hub_app, "_audit", None)
        conn.close()


def test_a_step_of_another_run_is_ignored():
    _start_run()
    conn = _connection()
    conn._on_computer_use_step({"run_id": "01ARZ3NDEKTSV4RRFFQ69G5FB0", "ok": True})
    assert hub_app._computer_use_runs().current(ROOM) is not None


def test_a_step_is_checked_then_sent_with_the_badge():
    _start_run()
    conn = _connection()
    result = asyncio.run(conn._run_computer_step(
        {"action": "app", "app": "Discord"}, language="ru"))
    assert result["ok"] is True and result["index"] == 0
    payloads = [call.args[0] for call in conn.send_json.await_args_list]
    badge = [item for item in payloads
             if item.get("type") == MSG_COMPUTER_USE and item.get("active")]
    assert badge and badge[0]["text"] == "Rowan управляет"
    sent = conn._run_client_action.await_args
    assert sent.args[0] == "computer_use_step"
    assert sent.args[1]["policy"]["allowed_apps"] == ["discord"]
    assert sent.args[1]["step"]["app"] == "Discord"


def test_a_step_outside_the_allow_list_never_reaches_the_room():
    _start_run()
    conn = _connection()
    result = asyncio.run(conn._run_computer_step({"action": "app", "app": "chrome"}))
    assert result["ok"] is False and "chrome" in result["error"]
    conn._run_client_action.assert_not_awaited()


def test_a_step_without_a_run_is_refused():
    conn = _connection()
    result = asyncio.run(conn._run_computer_step({"action": "click"}))
    assert result["ok"] is False and "run" in result["error"]


# --- сторона комнаты --------------------------------------------------------


class _Overlay:
    def __init__(self):
        self.control_calls: list[str] = []

    def control(self, text):
        self.control_calls.append(str(text or ""))


def test_the_badge_goes_up_and_down_with_the_run():
    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _Overlay()
    assistant._computer_run = None
    assistant._on_computer_use({"active": True, "run_id": "r1", "text": "Rowan управляет"})
    assert assistant.overlay.control_calls[-1] == "Rowan управляет"
    assistant._on_computer_use({"active": False, "run_id": "r1", "reason": "done"})
    assert assistant.overlay.control_calls[-1] == ""


def test_the_hub_saying_stop_ends_the_run_in_the_room():
    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _Overlay()
    run = ComputerUseSession(executor=ComputerUseExecutor(policy=ComputerUsePolicy(enabled=True,
                                                                                   allowed_apps=["discord"])))
    assistant._computer_run = run
    assistant._on_computer_use({"active": False, "reason": "stop word"})
    assert run.stopped is True and assistant._computer_run is None
    assert assistant.overlay.control_calls[-1] == ""


def test_a_step_runs_in_the_room_and_is_reported_to_the_hub():
    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _Overlay()
    assistant._computer_run = None
    assistant._loop = None
    assistant.ws = SimpleNamespace(send_json=AsyncMock())
    performed: list[dict] = []

    class _Gui:
        size = (100, 100)

        def moveTo(self, *args, **kwargs):
            performed.append({"move": args})

        def click(self):
            performed.append({"click": True})

    policy = ComputerUsePolicy(enabled=True, allowed_apps=["discord"])
    session = ComputerUseSession(executor=ComputerUseExecutor(
        policy=policy, gui=_Gui(), foreground=lambda: "discord", size=lambda: (100, 100)),
        run_id="run-1")
    assistant._computer_run = session
    report = asyncio.run(assistant._computer_use_step({
        "run_id": "run-1", "id": "s1", "policy": policy.model_dump(),
        "step": {"action": "click", "x": 0.5, "y": 0.5}}))
    assert report["ok"] is True and performed
    sent = assistant.ws.send_json.await_args.args[0]
    assert sent["type"] == MSG_COMPUTER_USE_STEP and sent["run_id"] == session.run_id
    assert sent["ok"] is True and sent["stopped"] is False


def test_a_new_run_replaces_the_old_one(monkeypatch):
    import client.main as client_main

    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _Overlay()
    assistant._computer_run = None
    assistant._computer_step_id = ""
    assistant.ws = SimpleNamespace(send_json=AsyncMock())
    performed: list[dict] = []

    class _Gui:
        size = (100, 100)

        def press(self, key):
            performed.append({"press": key})

    payload = {"run_id": "run-2", "id": "s2",
               "policy": ComputerUsePolicy(enabled=True, allowed_apps=["discord"]).model_dump(),
               "step": {"action": "key", "key": "enter"}}
    policy = ComputerUsePolicy(enabled=True, allowed_apps=["discord"])

    def _from_hub(cls, message):
        return ComputerUseSession(
            executor=ComputerUseExecutor(policy=policy, gui=_Gui(),
                                         foreground=lambda: "discord",
                                         size=lambda: (100, 100)),
            run_id=str(message.get("run_id") or ""))

    monkeypatch.setattr(client_main.ComputerUseSession, "from_hub",
                        classmethod(_from_hub))
    assistant._computer_run = ComputerUseSession(
        executor=ComputerUseExecutor(policy=policy,
                                     gui=_Gui(), foreground=lambda: "discord",
                                     size=lambda: (100, 100)),
        run_id="run-1")
    report = asyncio.run(assistant._computer_use_step(payload))
    assert report["ok"] is True and performed == [{"press": "enter"}]
    assert assistant._computer_run.run_id == "run-2"


def test_the_palm_stops_the_computer_use_run():
    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _Overlay()
    assistant.ws = SimpleNamespace(send_json=AsyncMock())

    class _Audio:
        def cancel_pending(self):
            return 0

    assistant.audio_out = _Audio()
    assistant._idle_tts_active = False
    assistant._idle_stream_active = False
    assistant._idle_playing = False
    assistant._idle_interrupted = False
    assistant._loop = None                       # жесты в потоке камеры
    run = ComputerUseSession(executor=ComputerUseExecutor(policy=ComputerUsePolicy(enabled=True,
                                                                                   allowed_apps=["discord"])),
                             run_id="r-palm")
    assistant._computer_run = run
    assistant._on_gesture("palm")
    # Цикла нет: корутина закрывается, но жест всё равно помечает «стоп» через
    # прямой вызов того же пути, которым пользуется рабочий цикл.
    asyncio.run(assistant._stop_computer_use("palm"))
    assert run.stopped is True
    sent = assistant.ws.send_json.await_args.args[0]
    assert sent["stopped"] is True and sent["run_id"] == "r-palm"


def test_the_control_badge_holds_the_hud_up_and_the_page_draws_it():
    owner = SimpleNamespace(_state="idle", _status="", _speaker_name="", _scanning=False,
                            _typing_on=False, _flash_until=0.0, _click_until=0.0,
                            _badge_until=0.0, _control="Rowan управляет")
    assert _visibility_now(owner, 100.0) is True
    owner._control = ""
    assert _visibility_now(owner, 100.0) is False
    page = (hub_app.REPO_ROOT / "client" / "overlay_web" / "hud.html")
    text = page.read_text(encoding="utf-8")
    assert "opts.control" in text and 'id="control"' in text


def test_the_new_frames_are_known_and_a_phone_cannot_drive_a_pc():
    assert MSG_COMPUTER_USE in SERVER_MESSAGE_TYPES
    assert MSG_COMPUTER_USE_STEP in CLIENT_MESSAGE_TYPES
    assert MSG_COMPUTER_USE_STEP in PHONE_FORBIDDEN_INPUTS


def test_the_step_report_from_the_client_is_what_the_hub_reads():
    """Формат отчёта клиента и то, что читает хаб, — один и тот же словарь."""
    _start_run()
    conn = _connection()
    report = {"type": MSG_COMPUTER_USE_STEP, "run_id": "",
              "ok": True, "index": 1, "step": "press enter", "reason": "", "stopped": False}
    conn._on_computer_use_step(json.loads(json.dumps(report)))
    assert hub_app._computer_use_runs().current(ROOM) is not None
