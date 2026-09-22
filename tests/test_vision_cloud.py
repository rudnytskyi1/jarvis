"""ТЗ F-404: облачный взгляд на СЛОЖНУЮ картинку — и только с разрешения дома.

Картинка из комнаты уезжает в облако только когда сходятся все четыре
условия: изображение сложное, ``models.routing.cloud_vision`` включён, САМ ДОМ
разрешил это в ``homes[].cloud_vision`` и в бюджете есть место. Любое
неизвестное — «нет»: неизвестный бюджет значит «локально», а не «попробуем».
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from common.config import Config, HomeConfig, ModelLevelConfig, ModelsConfig
from hub.api_budget import CloudUnavailable
from hub.model_router import LEVEL_LOCAL_VISION, ModelRouter
from hub.vision_cloud import CloudVision
from hub.vision_levels import build_cloud_vision, cloud_vision_entry, image_is_hard


def vision_models(*, cloud_vision: bool = True, local: bool = True,
                  cloud_model: str = "gpt-5.4", budget: bool = True):
    levels = {
        "local_vision": ModelLevelConfig(provider="ollama_native",
                                         model="qwen2.5vl:7b" if local else ""),
        "cloud_strong": ModelLevelConfig(provider="openai_responses", model=cloud_model),
    }
    cfg = ModelsConfig(enabled=True, levels=levels,
                       routing={"cloud_vision": cloud_vision, "vision_level": "local_vision",
                                "vision_cloud_level": "cloud_strong"})
    return ModelRouter(cfg, budget_allows=(lambda level: budget))


# --- who decides ------------------------------------------------------------


def test_an_easy_image_stays_local_even_when_the_cloud_is_allowed():
    decision = vision_models().vision_pick(hard_image=False, cloud_allowed=True)
    assert decision.level == LEVEL_LOCAL_VISION
    assert decision.reason == "vision"


def test_a_hard_image_goes_to_the_cloud_only_with_both_permissions():
    router = vision_models()
    assert router.vision_pick(hard_image=True, cloud_allowed=False).level == LEVEL_LOCAL_VISION
    cloud = router.vision_pick(hard_image=True, cloud_allowed=True)
    assert cloud.level == "cloud_strong"
    assert cloud.reason == "vision_cloud"


def test_the_routing_flag_alone_is_not_enough():
    router = vision_models(cloud_vision=False)
    assert router.vision_pick(hard_image=True, cloud_allowed=True).level == LEVEL_LOCAL_VISION


def test_no_money_means_the_picture_stays_here():
    router = vision_models(budget=False)
    decision = router.vision_pick(hard_image=True, cloud_allowed=True)
    assert decision.level == LEVEL_LOCAL_VISION, "бюджет — тоже разрешение"


def test_a_hub_without_a_local_vision_level_can_still_look_with_the_cloud():
    router = vision_models(local=False)
    decision = router.vision_pick(hard_image=False, cloud_allowed=True)
    assert decision.level == "cloud_strong"
    # А без разрешения дома смотреть нечем, и это честный None.
    assert router.vision_pick(hard_image=False, cloud_allowed=False) is None


def test_with_levels_off_the_router_does_not_invent_a_vision_level():
    cfg = ModelsConfig(enabled=False, levels={"cloud_strong": ModelLevelConfig(model="gpt-5.4")},
                       routing={"cloud_vision": True})
    assert ModelRouter(cfg).vision_pick(hard_image=True, cloud_allowed=True) is None


@pytest.mark.parametrize("query,hard", [
    ("что на экране?", False),
    ("прочитай текст в окне", True),
    ("покажи таблицу", True),
    ("what error is shown?", True),
])
def test_the_first_reading_of_a_hard_image(query, hard):
    assert image_is_hard(b"x" * 1000, query) is hard


def test_a_busy_screenshot_counts_as_hard():
    assert image_is_hard(b"x" * 400_000, "") is True


# --- the cloud client itself ------------------------------------------------


class _Transport:
    """A stand-in OpenAI endpoint that records what was asked and what it cost."""

    def __init__(self, *, answer: str = "A browser window is open.",
                 usage: dict | None = None, status: str = "completed"):
        self.answer = answer
        self.usage = usage or {"input_tokens": 1200, "output_tokens": 40}
        self.status = status
        self.requests: list[dict] = []

    def __call__(self, request):
        import httpx

        self.requests.append(json.loads(request.content.decode("utf-8")))
        body = {"status": self.status, "usage": self.usage, "output": [
            {"type": "message", "content": [{"type": "output_text", "text": self.answer}]}]}
        return httpx.Response(200, json=body, request=request)

    def mock(self):
        """The same handler in the shape ``httpx.Client`` accepts."""
        import httpx

        return httpx.MockTransport(self)


def cloud(tmp_path, monkeypatch, transport: _Transport, *, budget: float = 18.0) -> CloudVision:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    entry = ModelLevelConfig(provider="openai_responses", model="gpt-5.4", api_key_env="OPENAI_API_KEY")
    return CloudVision(entry, monthly_budget_usd=budget, ledger_path=tmp_path / "usage.sqlite3",
                       transport=transport.mock())


def test_the_cloud_look_sends_the_image_and_reads_the_answer(tmp_path, monkeypatch):
    transport = _Transport(answer="A table of grades is on the screen.")
    client = cloud(tmp_path, monkeypatch, transport)
    try:
        answer = client.describe(b"jpeg-bytes", "что в таблице?")
    finally:
        client.close()
    assert answer == "A table of grades is on the screen."
    sent = transport.requests[0]
    assert sent["model"] == "gpt-5.4"
    content = sent["input"][0]["content"]
    assert content[0]["type"] == "input_text" and "таблице" in content[0]["text"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_the_look_is_charged_to_the_hub_ledger(tmp_path, monkeypatch):
    import sqlite3

    ledger = tmp_path / "usage.sqlite3"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    entry = ModelLevelConfig(provider="openai_responses", model="gpt-5.4",
                             api_key_env="OPENAI_API_KEY")
    client = CloudVision(entry, monthly_budget_usd=18.0, ledger_path=ledger,
                         transport=_Transport().mock())
    try:
        client.describe(b"jpeg", "что тут?")
    finally:
        client.close()
    rows = sqlite3.connect(ledger).execute("SELECT COUNT(*) FROM requests").fetchone()
    assert rows[0] >= 1, "запрос к облаку не должен быть бесплатным по учёту"


def test_without_a_key_the_picture_is_not_sent(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    transport = _Transport()
    entry = ModelLevelConfig(provider="openai_responses", model="gpt-5.4",
                             api_key_env="OPENAI_API_KEY")
    client = CloudVision(entry, ledger_path=tmp_path / "usage.sqlite3",
                         transport=transport.mock())
    try:
        answer = client.describe(b"jpeg", "что тут?")
    finally:
        client.close()
    assert transport.requests == [], "без ключа никуда не уходит"
    assert "OPENAI_API_KEY" in answer


def test_a_cloud_failure_reads_as_an_answer_not_as_a_crash(tmp_path, monkeypatch):
    import httpx

    class Broken:
        def __call__(self, request):
            raise OSError("the network is gone")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    entry = ModelLevelConfig(provider="openai_responses", model="gpt-5.4",
                             api_key_env="OPENAI_API_KEY")
    client = CloudVision(entry, ledger_path=tmp_path / "usage.sqlite3",
                         transport=httpx.MockTransport(Broken()))
    try:
        answer = client.describe(b"jpeg", "что тут?")
    finally:
        client.close()
    assert answer.startswith("Screen check failed:")


def test_an_unpriced_model_is_refused_outright(tmp_path):
    entry = ModelLevelConfig(provider="openai_responses", model="gpt-9-unpriced")
    with pytest.raises(CloudUnavailable):
        CloudVision(entry, ledger_path=tmp_path / "usage.sqlite3")


# --- the home's own permission ---------------------------------------------


def test_the_home_flag_defaults_to_no():
    home = HomeConfig(home_id="livingroom", name="Living room")
    assert home.cloud_vision is False, "картинка из комнаты не уезжает по умолчанию"


def test_the_cloud_client_is_built_only_when_the_routing_allows_it():
    cfg = Config()
    cfg.models = ModelsConfig(enabled=True, levels={
        "cloud_strong": ModelLevelConfig(provider="openai_responses", model="gpt-5.4")},
        routing={"cloud_vision": False})
    assert cloud_vision_entry(cfg) is None
    assert build_cloud_vision(cfg) is None

    cfg.models = ModelsConfig(enabled=True, levels={
        "cloud_strong": ModelLevelConfig(provider="openai_responses", model="gpt-5.4")},
        routing={"cloud_vision": True})
    assert cloud_vision_entry(cfg) is not None
    client = build_cloud_vision(cfg)
    assert client is not None
    assert client.level_model == "gpt-5.4"
    client.close()


def test_the_app_only_uses_the_cloud_for_a_hard_image_and_an_allowing_home(monkeypatch):
    """Настоящий ход ``_run_look_at_screen``: решает роутер, а не флаг сам по себе."""
    from hub import app as hub_app

    used: list[str] = []

    class Cloud:
        def describe(self, jpeg, query=None):
            used.append("cloud")
            return "cloud answer"

    class Local:
        async def describe_screenshot(self, jpeg, query=None):
            used.append("local")
            return "local answer"

    cfg = Config()
    cfg.server.permissions_enabled = False
    cfg.homes = [HomeConfig(home_id="livingroom", name="Living room", cloud_vision=True)]
    cfg.models = ModelsConfig(enabled=True, levels={
        "local_vision": ModelLevelConfig(provider="ollama_native", model="qwen2.5vl:7b"),
        "cloud_strong": ModelLevelConfig(provider="openai_responses", model="gpt-5.4")},
        routing={"cloud_vision": True, "vision_level": "local_vision",
                 "vision_cloud_level": "cloud_strong"})
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._utterance_actions = []
    connection._screenshot_seq = 1

    async def shot(shot_id):
        return SimpleNamespace(jpeg=b"x" * 400_000)

    connection._request_screenshot = shot
    monkeypatch.setattr(hub_app, "_vision", Local())
    monkeypatch.setattr(hub_app, "_vision_cloud", Cloud())
    monkeypatch.setattr(hub_app, "_model_router", lambda decider=None: ModelRouter(
        cfg.models, budget_allows=lambda level: True))

    result = asyncio.run(connection._run_look_at_screen({"query": "что на экране?"}))
    assert result["ok"] is True and result["answer"] == "cloud answer"
    assert result["vision_level"] == "cloud_strong"
    assert used == ["cloud"], "сложная картинка ушла в облако"

    # Тот же ход в доме, который облако не разрешал, остаётся локальным.
    cfg.homes = [HomeConfig(home_id="livingroom", name="Living room")]
    result = asyncio.run(connection._run_look_at_screen({"query": "что на экране?"}))
    assert result["answer"] == "local answer"
    assert used == ["cloud", "local"]
