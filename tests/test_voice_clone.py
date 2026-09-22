"""P5-17 (F-111): клон голоса за флагом дома и согласием владельца.

Проверяется, что клон не синтезирует НИЧЕГО без согласия человека, что окно
референса 10–20 с измеряется по настоящему WAV, что результат кэшируется и что
отсутствие GPU-провайдера — это названный отказ, а не подмена обычным голосом.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.config import Config, VoiceCloneConfig
from hub.voice_clone import (
    REFERENCE_MAX_S,
    REFERENCE_MIN_S,
    ConsentStore,
    LocalCloneProvider,
    VoiceCloneRefused,
    VoiceCloneService,
    VoiceCloneUnavailable,
    reference_seconds,
    service_from_config,
)

HOME, PERSON = "livingroom", "person-anton"


def wav(seconds: float, rate: int = 16000) -> bytes:
    """Настоящий PCM16 WAV нужной длины."""
    frames = int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x00\x01" * frames)
    return buffer.getvalue()


class _Provider:
    name = "fake"

    def __init__(self, *, audio: bytes = b"RIFF-cloned"):
        self.audio = audio
        self.calls: list[tuple] = []

    def ready(self):
        return True

    def synthesize(self, text, reference, *, language):
        self.calls.append((text, len(reference), language))
        return self.audio


def _service(tmp_path, *, enabled=True, provider=None, consent=None):
    store = consent if consent is not None else ConsentStore(tmp_path / "consent.json")
    return VoiceCloneService(enabled=enabled, provider=provider or _Provider(),
                             consent=store, cache_dir=tmp_path / "cache"), store


# --- окно референса ---------------------------------------------------------


def test_the_reference_window_is_measured_on_a_real_wav():
    assert reference_seconds(wav(12.0)) == pytest.approx(12.0, abs=0.01)
    for broken in (b"", b"not-a-wav"):
        with pytest.raises(VoiceCloneUnavailable):
            reference_seconds(broken)


# --- согласие ---------------------------------------------------------------


def test_no_clone_without_the_persons_consent(tmp_path):
    service, store = _service(tmp_path)
    with pytest.raises(VoiceCloneRefused) as error:
        service.synthesize(HOME, PERSON, "привет", wav(12.0))
    assert "agreed" in str(error.value)
    assert service.synthesized == 0
    # Согласие дали — и та же фраза синтезируется.
    assert store.grant(HOME, PERSON, at=100.0) is True
    assert service.allowed(HOME, PERSON) is True
    assert service.synthesize(HOME, PERSON, "привет", wav(12.0)) == b"RIFF-cloned"
    assert service.synthesized == 1


def test_a_consent_belongs_to_one_person_and_one_home(tmp_path):
    store = ConsentStore(tmp_path / "consent.json")
    store.grant(HOME, PERSON)
    assert store.granted(HOME, PERSON) is True
    assert store.granted(HOME, "person-max") is False
    assert store.granted("office", PERSON) is False
    assert store.grant("", PERSON) is False and store.grant(HOME, "") is False
    assert store.snapshot()["consents"] == 1
    assert store.revoke(HOME, PERSON) is True and store.granted(HOME, PERSON) is False
    store.grant(HOME, PERSON)
    # Согласие переживает перезапуск хаба.
    assert ConsentStore(tmp_path / "consent.json").granted(HOME, PERSON) is True


def test_the_flag_off_closes_the_clone_even_with_consent(tmp_path):
    service, store = _service(tmp_path, enabled=False)
    store.grant(HOME, PERSON)
    with pytest.raises(VoiceCloneRefused) as error:
        service.synthesize(HOME, PERSON, "привет", wav(12.0))
    assert "switched off" in str(error.value)


# --- окно 10–20 секунд ------------------------------------------------------


@pytest.mark.parametrize("seconds,ok", [(9.5, False), (10.0, True), (20.0, True), (21.0, False)])
def test_only_the_window_of_the_spec_is_accepted(tmp_path, seconds, ok):
    service, store = _service(tmp_path)
    store.grant(HOME, PERSON)
    if ok:
        assert service.synthesize(HOME, PERSON, "привет", wav(seconds))
    else:
        with pytest.raises(VoiceCloneRefused) as error:
            service.synthesize(HOME, PERSON, "привет", wav(seconds))
        assert f"{REFERENCE_MIN_S:.0f}-{REFERENCE_MAX_S:.0f}" in str(error.value)


def test_an_empty_phrase_is_not_synthesized(tmp_path):
    service, store = _service(tmp_path)
    store.grant(HOME, PERSON)
    with pytest.raises(VoiceCloneRefused):
        service.synthesize(HOME, PERSON, "   ", wav(12.0))


# --- кэш --------------------------------------------------------------------


def test_the_same_phrase_is_synthesized_once(tmp_path):
    provider = _Provider()
    service, store = _service(tmp_path, provider=provider)
    store.grant(HOME, PERSON)
    first = service.synthesize(HOME, PERSON, "доброе утро", wav(12.0))
    second = service.synthesize(HOME, PERSON, "доброе утро", wav(12.0))
    assert first == second and len(provider.calls) == 1
    assert service.cache_hits == 1 and service.cache_misses == 1
    # Другая фраза — снова синтез, а не чужой кэш.
    service.synthesize(HOME, PERSON, "добрый вечер", wav(12.0))
    assert len(provider.calls) == 2
    # Кэш переживает перезапуск сервиса.
    restarted, _ = _service(tmp_path, provider=provider)
    assert restarted.cached(HOME, PERSON, "доброе утро") == first


def test_the_cache_is_pruned_to_its_limit(tmp_path):
    provider = _Provider()
    service, store = _service(tmp_path, provider=provider)
    service.max_cache_entries = 2
    store.grant(HOME, PERSON)
    for word in ("раз", "два", "три"):
        service.synthesize(HOME, PERSON, word, wav(12.0))
    files = list((tmp_path / "cache").glob("*.wav"))
    assert len(files) == 2


# --- провайдер --------------------------------------------------------------


def test_a_missing_provider_is_named_not_faked():
    try:
        import f5_tts  # noqa: F401

        pytest.skip("f5_tts установлен в этом окружении")
    except ImportError:
        pass
    provider = LocalCloneProvider(provider="f5_tts")
    assert provider.ready() is False
    with pytest.raises(VoiceCloneUnavailable) as error:
        provider._load()
    assert "f5_tts" in str(error.value)
    # Повтор не переимпортирует сломанное.
    assert provider.ready() is False


def test_an_unknown_provider_is_refused():
    provider = LocalCloneProvider(provider="magic")
    with pytest.raises(VoiceCloneUnavailable) as error:
        provider._load()
    assert "not one of" in str(error.value)


def test_the_service_is_off_unless_the_owner_turns_it_on():
    cfg = Config()
    assert cfg.server.voice_clone.enabled is False
    cfg.server.voice_clone = VoiceCloneConfig(enabled=True, provider="chatterbox")
    service = service_from_config(cfg, data_dir=Path("data"))
    assert service.enabled is True
    assert service.snapshot()["provider"] == "chatterbox"
    assert service.snapshot()["provider_ready"] is False, "весов в песочнице нет"
    off = service_from_config(Config(), data_dir=Path("data"))
    assert off.enabled is False and off.snapshot()["provider"] == ""


def test_the_health_snapshot_names_consents_and_cache():
    service = VoiceCloneService(enabled=True, provider=_Provider(),
                                consent=ConsentStore("data/nonexistent-test.json"))
    snapshot = service.snapshot()
    assert snapshot["enabled"] is True and snapshot["consents"]["consents"] == 0
    assert snapshot["cache_hits"] == 0
    assert SimpleNamespace(**snapshot) is not None
