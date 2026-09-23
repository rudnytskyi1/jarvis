"""Both yaml files must load and carry the invariants the code relies on."""
from pathlib import Path

import pytest

from common.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["config.yaml", "config.example.yaml"])
def cfg(request):
    return load_config(REPO_ROOT / request.param)


def test_llm_invariants(cfg):
    llm = cfg.server.llm
    assert llm.provider == "ollama_native"
    assert llm.think is False
    # chat and vision are now SEPARATE models (both kept resident via
    # OLLAMA_MAX_LOADED_MODELS); the vision one must just be set.
    assert llm.vision_model, "a vision model must be configured"
    assert llm.max_tool_rounds >= 4
    assert llm.num_ctx >= 4096
    assert llm.keep_alive


def test_no_config_agrees_to_refuse_cloud_turns_by_default():
    """Владелец 2026-09-23: «monthly api allowance убери нахер, чатбот не работает».

    Умолчание — «предела нет»: конфиг, который просто не упоминает ключ, не
    получает право отклонять запросы. Прежнее умолчание $18 молча включало
    отказ, и ассистент отвечал «Monthly API allowance reached».
    """
    from common.config import Config, LLMConfig

    assert LLMConfig().monthly_budget_usd == 0
    assert Config().server.llm.monthly_budget_usd == 0
    # Оба конфига репозитория тоже без потолка: 0 = считать, но не отказывать.
    for name in ("config.yaml", "config.example.yaml"):
        assert load_config(REPO_ROOT / name).server.llm.monthly_budget_usd == 0, name


def test_stt_language_whitelist(cfg):
    assert cfg.server.stt.language is None, "auto-detect stays on"
    assert set(cfg.server.stt.allowed_languages) == {"en", "ru", "es"}


def test_tts_is_english(cfg):
    assert cfg.server.tts.language == "en"
    assert cfg.server.tts.model_id == "v3_en"


def test_client_invariants(cfg):
    c = cfg.client
    assert c.wakeword.word == "rowan ai"
    assert c.wakeword.phrases, "at least one spelling for Vosk"
    assert c.vad.min_speech_ms > 0
    assert c.vad.aggressiveness == 3
    assert c.thinking_sounds is True
    assert c.devices == [], "no physical devices configured yet"
    assert str(c.server_url).endswith("/ws")
