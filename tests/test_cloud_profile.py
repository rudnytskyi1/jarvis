from pathlib import Path

import pytest

from common.config import load_config
from scripts.configure_openai import create_profile


def test_profile_preserves_machine_settings_and_does_not_overwrite(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config.example.yaml"
    output = tmp_path / "cloud.yaml"
    before = load_config(source)
    create_profile(source, output)
    after = load_config(output)
    assert after.client.server_url == before.client.server_url
    assert after.server.port == before.server.port
    assert after.client.camera == before.client.camera
    assert after.server.llm.provider == "openai_responses"
    assert after.server.llm.vision_base_url == before.server.llm.base_url
    assert after.server.llm.api_key == ""
    assert after.client.attention_mode == "wake_word"
    assert after.server.face.greeting_llm is False
    with pytest.raises(FileExistsError):
        create_profile(source, output)


def test_cloud_prompt_is_rendered_without_unexpanded_placeholders():
    from hub.session import Session
    prompt = Path(__file__).resolve().parents[1] / "prompts" / "cloud.md"
    session = Session(client_id="test", devices=[], history_turns=6, prompt_path=prompt, memory_facts=[])
    for placeholder in ("{devices}", "{memory}", "{presence}"):
        assert placeholder not in session.system_prompt
