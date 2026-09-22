"""ТЗ F-404: зрение — это уровень моделей, а не «ещё одна настройка рядом».

Проверяется: ``local_vision`` живёт в той же схеме уровней, что и текстовые
(конфиг и валидация), роутер умеет сказать, кто смотрит картинку, отсутствие
мультимодального уровня — это честное «смотреть нечем», а не вопрос текстовой
модели про картинку, и классический ``server.llm.vision_model`` продолжает
работать, пока схема уровней выключена (не ломать работающее).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.config import DEFAULT_LEVEL_NAMES, ModelLevelConfig, ModelsConfig
from hub.model_router import LEVEL_LOCAL_VISION, ModelRouter
from hub.vision_levels import build_vision, vision_entry, vision_level_name, vision_source


def models_config(*, enabled: bool = True, vision_model: str = "qwen2.5vl:7b",
                  vision_level: str = "local_vision") -> ModelsConfig:
    return ModelsConfig(
        enabled=enabled,
        levels={"local_vision": ModelLevelConfig(provider="ollama_native",
                                                 base_url="http://127.0.0.1:11434/v1",
                                                 model=vision_model)},
        routing={"vision_level": vision_level},
    )


def test_vision_is_a_level_like_any_other():
    assert "local_vision" in DEFAULT_LEVEL_NAMES
    cfg = models_config()
    assert cfg.levels["local_vision"].ready is True
    assert cfg.levels["local_fast"].ready is False, "незанятый уровень остаётся незанятым"


def test_a_vision_level_that_is_not_a_level_is_refused():
    """Опечатка в vision_level не должна тихо сломать зрение."""
    with pytest.raises(ValueError):
        ModelsConfig(enabled=True, levels={}, routing={"vision_level": "camera"})


def test_an_empty_level_means_not_provisioned():
    cfg = models_config(vision_model="")
    assert cfg.levels["local_vision"].ready is False
    assert vision_entry(cfg) is None, "пустая модель — это «уровень не поднят»"


def test_the_router_names_the_vision_level_and_admits_when_there_is_none():
    router = ModelRouter(models_config())
    assert router.vision_available() is True
    decision = router.vision_pick()
    assert decision is not None
    assert decision.level == LEVEL_LOCAL_VISION
    assert decision.reason == "vision"

    # Схема выключена или уровень не поднят — смотреть нечем, и это видно.
    for models in (models_config(enabled=False), models_config(vision_model="")):
        assert ModelRouter(models).vision_pick() is None


def test_a_text_round_never_gets_the_picture():
    """Картинку не отдают текстовой модели — у неё нет глаз."""
    router = ModelRouter(models_config())
    assert router.pick("что на экране?", has_image=True).level != LEVEL_LOCAL_VISION
    assert router.vision_pick().level == LEVEL_LOCAL_VISION


def test_the_level_supplies_the_model_and_the_endpoint():
    client = build_vision(SimpleNamespace(models=models_config(), server=None))
    assert client is not None
    assert client.model == "qwen2.5vl:7b"
    assert "11434" in client.base_url, "уровень приносит свой endpoint"


def _classic(vision_model: str = "qwen3-vl:8b") -> SimpleNamespace:
    return SimpleNamespace(provider="ollama_native", base_url="http://127.0.0.1:11434/v1",
                           vision_model=vision_model, temperature=0.3)


def test_the_classic_vision_model_still_answers_when_levels_are_off():
    cfg = SimpleNamespace(models=models_config(enabled=False), server=SimpleNamespace(llm=_classic()))
    client = build_vision(cfg)
    assert client is not None
    assert client.model == "qwen3-vl:8b"
    assert vision_source(cfg) == "server.llm.vision_model"


def test_a_half_filled_level_scheme_does_not_take_the_hub_s_sight_away():
    """Уровни включены, но vision-уровень пуст — классический ответ остаётся."""
    cfg = SimpleNamespace(models=models_config(vision_model=""),
                          server=SimpleNamespace(llm=_classic()))
    client = build_vision(cfg)
    assert client is not None and client.model == "qwen3-vl:8b"
    assert vision_source(cfg) == "server.llm.vision_model"


def test_without_any_vision_model_the_hub_says_so_instead_of_guessing():
    cfg = SimpleNamespace(models=models_config(vision_model=""),
                          server=SimpleNamespace(llm=_classic(vision_model="")))
    assert build_vision(cfg) is None
    assert vision_source(cfg) == ""


def test_the_vision_level_is_a_pointer_not_a_hard_coded_name():
    """Зрение можно повесить на другой уровень — если его модель умеет видеть."""
    cfg = ModelsConfig(enabled=True,
                       levels={"local_strong": ModelLevelConfig(
                           provider="ollama_native", model="millard-qwen4:latest")},
                       routing={"vision_level": "local_strong"})
    assert vision_level_name(cfg) == "local_strong"
    assert vision_entry(cfg) is not None


def test_the_shipped_configs_name_a_vision_level():
    """Шаблоны и рабочая конфигурация не должны разъезжаться с кодом."""
    import pathlib

    from common.config import load_config

    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("config.example.yaml", "config.yaml"):
        cfg = load_config(str(root / name))
        assert "local_vision" in cfg.models.levels, name
        assert cfg.models.routing.vision_level in cfg.models.levels, name
