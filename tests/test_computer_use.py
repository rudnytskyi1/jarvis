"""P5-14 (F-512): лимит 15 шагов, allow-list приложений, запрет секретов.

Проверяется и хабовая часть (что вообще разрешено и что случилось с шагом), и
исполнитель на комнате (последняя линия перед мышью): шаг за пределами лимита,
приложение вне allow-list, ввод пароля или платёжных данных и печать в
неизвестное окно обязаны останавливаться ДО действия.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from client.actions.computer_use import (
    ComputerUseExecutor,
    ComputerUseRefused,
    ComputerUseUnavailable,
)
from common.computer_use import (
    MAX_STEPS,
    ComputerUsePolicy,
    ComputerUseStep,
    normalize_app,
    sensitive_reason,
)
from common.config import ComputerUseConfig, Config
from hub.computer_use import ComputerUseRefused as HubRefused
from hub.computer_use import ComputerUseRun, ComputerUseRuns, policy_for

# --- правила ---------------------------------------------------------------


def test_the_step_limit_of_the_spec_cannot_be_raised():
    assert MAX_STEPS == 15
    assert ComputerUseConfig(max_steps=15).max_steps == 15
    for too_many in (16, 100, 0):
        with pytest.raises(ValidationError):
            ComputerUseConfig(max_steps=too_many)


def test_secrets_and_money_are_named_in_three_languages():
    for text in ("my password is hunter2", "введи пароль от почты",
                 "escribe la contraseña", "card number 4111 1111",
                 "код из смс", "seed phrase", "iban DE89",
                 "código pin del banco"):
        assert sensitive_reason(text), text
    for text in ("привет, как дела", "reply ok", "открой Discord",
                 "what time is it"):
        assert sensitive_reason(text) == ""


def test_the_allow_list_is_the_only_thing_the_agent_may_touch():
    policy = ComputerUsePolicy(enabled=True, allowed_apps=["Discord", "C:\\Apps\\Telegram.exe"])
    assert policy.allowed_apps == ["discord", "telegram"]
    assert policy.allows_app("DISCORD") and policy.allows_app("C:/Apps/Telegram.exe")
    assert not policy.allows_app("chrome")
    # Пустой список — безопасное умолчание: ни одного приложения.
    assert ComputerUsePolicy(enabled=True).allows_app("discord") is False
    assert normalize_app('"C:\\Program Files\\Steam\\steam.exe"') == "steam"


def test_a_step_outside_the_allow_list_is_refused_with_a_reason():
    policy = ComputerUsePolicy(enabled=True, allowed_apps=["discord"])
    allowed = ComputerUseStep(action="app", app="Discord")
    refused = ComputerUseStep(action="app", app="chrome")
    assert policy.refuse(allowed, index=0) == ""
    assert "chrome" in policy.refuse(refused, index=0)
    # Выключенный флаг закрывает всё, даже шаг в разрешённом приложении.
    assert "switched off" in ComputerUsePolicy(allowed_apps=["discord"]).refuse(allowed, index=0)


def test_typing_a_secret_is_refused_even_when_typing_is_allowed():
    policy = ComputerUsePolicy(enabled=True, allowed_apps=["discord"], allow_typing=True)
    assert policy.refuse(ComputerUseStep(action="type", text="ок"), index=0) == ""
    secret = policy.refuse(ComputerUseStep(action="type", text="password: hunter2"), index=0)
    assert "password" in secret and "themselves" in secret
    off = ComputerUsePolicy(enabled=True, allowed_apps=["discord"], allow_typing=False)
    assert "switched off" in off.refuse(ComputerUseStep(action="type", text="ок"), index=0)


# --- прогон на хабе ---------------------------------------------------------


def _policy(**kwargs) -> ComputerUsePolicy:
    return ComputerUsePolicy(enabled=True, allowed_apps=["discord"], **kwargs)


def test_the_fifteenth_step_runs_and_the_sixteenth_is_refused():
    run = ComputerUseRun(home_id="livingroom", goal="ответь Максу", policy=_policy())
    for index in range(MAX_STEPS):
        decision = run.accept({"action": "click", "x": 0.5, "y": 0.5})
        assert decision.ok is True and decision.index == index
    assert run.used == MAX_STEPS and run.remaining == 0
    over = run.accept({"action": "click", "x": 0.5, "y": 0.5})
    assert over.ok is False and "15" in over.reason
    assert run.used == MAX_STEPS, "отказ не занимает шаг"
    assert len(run.refusals) == 1


def test_a_refused_step_does_not_eat_the_budget():
    run = ComputerUseRun(home_id="livingroom", goal="открой хром", policy=_policy())
    refused = run.accept({"action": "app", "app": "chrome"})
    assert refused.ok is False and "chrome" in refused.reason
    assert run.used == 0 and run.remaining == MAX_STEPS
    assert run.accept({"action": "app", "app": "Discord"}).ok is True


def test_a_malformed_step_is_a_refusal_not_a_crash():
    run = ComputerUseRun(home_id="livingroom", goal="", policy=_policy())
    for payload in ({"action": "fly"}, {"action": "click", "x": 5.0},
                    {"action": "type", "text": "ok", "surprise": 1}):
        decision = run.accept(payload)
        assert decision.ok is False and decision.reason
    assert run.used == 0
    # Питон-объект шага принимается так же, как словарь.
    assert run.accept(ComputerUseStep(action="scroll", direction="down")).ok is True


def test_stopping_a_run_stops_every_further_step():
    run = ComputerUseRun(home_id="livingroom", goal="", policy=_policy())
    assert run.accept({"action": "click"}).ok is True
    assert run.finish("stop word") == "stop word"
    assert run.stopped is True
    after = run.accept({"action": "click"})
    assert after.ok is False and "finished" in after.reason
    assert run.finish("again") == "stop word", "причина закрытия одна"


def test_the_policy_comes_from_the_config_and_the_home_flag():
    cfg = Config()
    cfg.server.computer_use = ComputerUseConfig(enabled=True, allowed_apps=["Discord"])
    cfg.homes = [SimpleNamespace(home_id="livingroom", settings={}),
                 SimpleNamespace(home_id="office", settings={"computer_use": False})]
    assert policy_for(cfg, "livingroom").enabled is True
    assert policy_for(cfg, "livingroom").allowed_apps == ["discord"]
    assert policy_for(cfg, "office").enabled is False, "флаг дома — последнее слово"
    assert policy_for(cfg, "").enabled is False
    # Сервер выключен — включение флагом дома ничего не открывает.
    cfg.server.computer_use = ComputerUseConfig(enabled=False, allowed_apps=["Discord"])
    assert policy_for(cfg, "livingroom").enabled is False


def test_runs_are_one_per_home_and_finish_once():
    runs = ComputerUseRuns()
    with pytest.raises(HubRefused):
        runs.start("livingroom", "goal", ComputerUsePolicy())
    with pytest.raises(HubRefused):
        runs.start("", "goal", _policy())
    first = runs.start("livingroom", "найди сообщение", _policy())
    second = runs.start("livingroom", "другая задача", _policy())
    assert runs.current("livingroom") is second and first is not second
    summary = runs.finish("livingroom", "done")
    assert summary is not None and summary["finished_reason"] == "done"
    assert runs.finish("livingroom") is None
    stopped = runs.start("office", "x", _policy())
    assert runs.stop("gesture") and stopped.stopped
    assert runs.snapshot()["active"] == {}


def test_the_summary_names_every_step_and_refusal():
    run = ComputerUseRun(home_id="livingroom", goal="ответь Максу", policy=_policy())
    run.accept({"action": "app", "app": "Discord"})
    run.accept({"action": "type", "text": "password: 123"})
    run.accept({"action": "type", "text": "ок"})
    run.finish("done")
    summary = run.summary()
    assert summary["used"] == 2 and summary["steps"][-1] == "type 2 characters"
    assert any("password" in reason for reason in summary["refusals"])
    assert summary["run_id"] and summary["max_steps"] == 15


# --- исполнитель на комнате -------------------------------------------------


class _Gui:
    """Подставной ``pyautogui``: записывает, что у него попросили."""

    def __init__(self, size=(1000, 500)):
        self.size = size
        self.calls: list[tuple] = []

    def moveTo(self, x, y, duration=0.2):
        self.calls.append(("moveTo", x, y))

    def click(self, *args, **kwargs):
        self.calls.append(("click",))

    def rightClick(self, *args, **kwargs):
        self.calls.append(("rightClick",))

    def doubleClick(self, *args, **kwargs):
        self.calls.append(("doubleClick",))

    def write(self, text):
        self.calls.append(("write", text))

    def press(self, key):
        self.calls.append(("press", key))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", *keys))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))


def _executor(gui, *, apps=("discord",), foreground="Discord", allow_typing=True):
    return ComputerUseExecutor(
        policy=ComputerUsePolicy(enabled=True, allowed_apps=list(apps),
                                 allow_typing=allow_typing),
        gui=gui,
        foreground=lambda: foreground,
        size=lambda: gui.size,
    )


def test_the_executor_clicks_in_screen_fractions():
    gui = _Gui()
    result = _executor(gui).execute({"action": "click", "x": 0.5, "y": 0.4})
    assert result["ok"] is True
    assert ("moveTo", 500, 200) in gui.calls and ("click",) in gui.calls


def test_the_executor_refuses_to_type_into_an_unknown_window():
    gui = _Gui()
    executor = _executor(gui, foreground="")
    result = executor.execute({"action": "type", "text": "ок"})
    assert result["ok"] is False and "in front" in result["reason"]
    assert ("write", "ок") not in gui.calls, "в неизвестное окно не печатаем"
    with pytest.raises(ComputerUseRefused):
        executor._acting_app(ComputerUseStep(action="type", text="ок"))


def test_the_executor_refuses_an_application_outside_the_allow_list():
    gui = _Gui()
    result = _executor(gui, foreground="chrome").execute({"action": "key", "key": "enter"})
    assert result["ok"] is False and "chrome" in result["reason"]
    assert gui.calls == []


def test_the_executor_never_types_a_secret_even_if_the_hub_asked():
    gui = _Gui()
    result = _executor(gui).execute({"action": "type", "text": "card number 4111"})
    assert result["ok"] is False and "payment" in result["reason"]
    assert not any(call[0] == "write" for call in gui.calls)


def test_the_executor_counts_real_actions_and_obeys_the_limit():
    gui = _Gui()
    executor = _executor(gui)
    for _ in range(MAX_STEPS):
        assert executor.execute({"action": "key", "key": "enter"})["ok"] is True
    over = executor.execute({"action": "key", "key": "enter"})
    assert over["ok"] is False and "15" in over["reason"]
    assert sum(1 for call in gui.calls if call[0] == "press") == MAX_STEPS


def test_a_step_naming_another_app_is_refused_before_acting():
    gui = _Gui()
    result = _executor(gui).execute({"action": "click", "x": 0.1, "y": 0.1, "app": "chrome"})
    assert result["ok"] is False and gui.calls == []


def test_the_launcher_only_opens_apps_from_the_list():
    opened: list[str] = []
    executor = ComputerUseExecutor(
        policy=ComputerUsePolicy(enabled=True, allowed_apps=["discord"]),
        gui=_Gui(), foreground=lambda: "Discord", launcher=opened.append)
    assert executor.execute({"action": "app", "app": "Discord"})["ok"] is True
    refused = executor.execute({"action": "app", "app": "steam"})
    assert refused["ok"] is False and opened == ["Discord"]


def test_the_executor_names_the_missing_package():
    try:
        import pyautogui  # noqa: F401

        pytest.skip("pyautogui установлен в этом окружении")
    except ImportError:
        pass
    executor = ComputerUseExecutor(policy=ComputerUsePolicy(enabled=True,
                                                            allowed_apps=["discord"]),
                                   foreground=lambda: "Discord")
    with pytest.raises(ComputerUseUnavailable) as error:
        executor.execute({"action": "key", "key": "enter"})
    assert "pyautogui" in str(error.value)


def test_a_failing_step_is_reported_not_swallowed():
    class _Angry(_Gui):
        def press(self, key):
            raise RuntimeError("the desktop is locked")

    result = _executor(_Angry()).execute({"action": "key", "key": "enter"})
    assert result["ok"] is False and "the desktop is locked" in result["reason"]
