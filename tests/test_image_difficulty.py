"""ТЗ F-404/P3-03: сложность картинки — измерение или решение, но не догадка.

Сигналы: несколько лиц в кадре, большой скриншот, МЕЛКИЙ ТЕКСТ, измеренный на
самой картинке, и вопрос, который просит подробностей. Правила остаются
запасным ответом, а Decider (D-10) может их переспорить.
"""
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from common.config import ModelLevelConfig, ModelsConfig
from hub.model_router import LEVEL_LOCAL_VISION, ModelRouter
from hub.vision_levels import TEXT_SCORE_THRESHOLD, image_is_hard, text_score


def jpeg(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def page_of_text() -> bytes:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    for line in range(24):
        draw.text((10, 10 + line * 22),
                  "Grades: Mathematics 95   Physics 88   Code error 0x1F line 42",
                  fill="black")
    return jpeg(image)


def smooth_photo() -> bytes:
    image = Image.new("RGB", (800, 600))
    pixels = image.load()
    for y in range(600):
        for x in range(800):
            pixels[x, y] = (x * 255 // 800, y * 255 // 600, 128)
    return jpeg(image)


def face_in_a_room() -> bytes:
    image = Image.new("RGB", (800, 600), (90, 110, 140))
    ImageDraw.Draw(image).ellipse([200, 100, 600, 500], fill=(220, 190, 170))
    return jpeg(image)


def simple_window() -> bytes:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 750, 550], outline="black", width=3)
    draw.text((80, 80), "YouTube", fill="black")
    return jpeg(image)


# --- the measurement itself -------------------------------------------------


def test_text_is_measured_on_the_picture_not_guessed_from_its_size():
    text = text_score(page_of_text())
    photo = text_score(smooth_photo())
    face = text_score(face_in_a_room())
    window = text_score(simple_window())
    assert text >= TEXT_SCORE_THRESHOLD, f"страница текста дала {text:.3f}"
    assert photo < TEXT_SCORE_THRESHOLD and face < TEXT_SCORE_THRESHOLD
    assert window < TEXT_SCORE_THRESHOLD, f"простое окно дало {window:.3f}"


def test_an_unreadable_picture_is_not_called_hard():
    assert text_score(b"not a jpeg") == 0.0
    assert image_is_hard(b"not a jpeg", "") is False


@pytest.mark.parametrize("query,hard", [
    ("что на экране?", False),
    ("покажи таблицу с оценками", True),
    ("what error is shown?", True),
    ("какой код в окне", True),
])
def test_the_question_itself_can_make_a_picture_hard(query, hard):
    assert image_is_hard(simple_window(), query) is hard


def test_several_faces_are_a_hard_frame():
    frame = face_in_a_room()
    assert image_is_hard(frame, "", people=1) is False
    assert image_is_hard(frame, "", people=2) is True


def test_a_busy_screenshot_is_hard():
    assert image_is_hard(b"x" * 400_000, "", people=0) is True


def test_the_decisive_call_is_the_picture_itself():
    """Большая картинка с текстом — сложная; гладкое фото — нет."""
    assert image_is_hard(page_of_text(), "что тут?") is True
    assert image_is_hard(face_in_a_room(), "что тут?") is False


# --- D-10 -------------------------------------------------------------------


def router_with(*, decider=None, budget: bool = True) -> ModelRouter:
    cfg = ModelsConfig(
        enabled=True,
        levels={"local_vision": ModelLevelConfig(provider="ollama_native", model="qwen2.5vl:7b"),
                "cloud_strong": ModelLevelConfig(provider="openai_responses", model="gpt-5.4")},
        routing={"cloud_vision": True, "vision_level": "local_vision",
                 "vision_cloud_level": "cloud_strong"})
    return ModelRouter(cfg, decider=decider, budget_allows=lambda level: budget)


class _Decider:
    """A decider that answers ``value`` (or blows up) for the vision question."""

    def __init__(self, value: str = "cloud_strong", *, provider: str = "local_llm") -> None:
        self.value = value
        self.provider = provider
        self.asked: list[tuple] = []

    async def choose(self, question, options, context, *, decision_type):
        self.asked.append((question, tuple(options), dict(context), decision_type))
        if self.value == "raise":
            raise RuntimeError("the provider is down")
        return SimpleNamespace(value=self.value, provider=self.provider, confidence=0.66)


def test_the_decider_can_overrule_the_rules_about_the_same_picture():
    import asyncio

    decider = _Decider(LEVEL_LOCAL_VISION)
    decision = asyncio.run(router_with(decider=decider).vision_choose(
        hard_image=True, cloud_allowed=True, query="прочитай текст", people=2))
    assert decision.level == LEVEL_LOCAL_VISION, "провайдер решил, что картинка по силам локальной"
    assert decision.reason == "decider:local_llm"
    question, options, context, kind = decider.asked[0]
    assert kind == "vision_level"
    assert options == (LEVEL_LOCAL_VISION, "cloud_strong")
    assert context["hard_image"] is True and context["people"] == 2
    assert context["heuristic"] == "cloud_strong", "правила назвали своё мнение"


def test_a_broken_decider_leaves_the_rules_in_charge():
    import asyncio

    decision = asyncio.run(router_with(decider=_Decider("raise")).vision_choose(
        hard_image=True, cloud_allowed=True))
    assert decision.level == "cloud_strong" and decision.reason == "vision_cloud"


def test_a_decider_that_names_something_unknown_is_ignored():
    import asyncio

    decision = asyncio.run(router_with(decider=_Decider("local_fast")).vision_choose(
        hard_image=True, cloud_allowed=True))
    assert decision.level == "cloud_strong", "чужой уровень не может смотреть на картинку"


def test_without_permission_the_decider_is_never_asked():
    import asyncio

    decider = _Decider("cloud_strong")
    decision = asyncio.run(router_with(decider=decider).vision_choose(
        hard_image=True, cloud_allowed=False))
    assert decision.level == LEVEL_LOCAL_VISION
    assert decider.asked == [], "дом не разрешал — спрашивать нечего"


def test_with_only_one_level_there_is_nothing_to_decide():
    import asyncio

    decider = _Decider("cloud_strong")
    router = router_with(decider=decider, budget=False)
    decision = asyncio.run(router.vision_choose(hard_image=True, cloud_allowed=True))
    assert decision.level == LEVEL_LOCAL_VISION
    assert decider.asked == []
