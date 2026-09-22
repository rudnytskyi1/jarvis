"""Клонированный голос владельца (ТЗ F-111).

ТЗ F-111: «Провайдер TTS на GPU (F5-TTS или Chatterbox) с референсом 10–20 с;
включается только для дома, где владелец дал согласие на клон своего голоса;
результат кэшируется».

Три вещи, которые здесь реальнее самого провайдера:

* **согласие.** Клон чужого голоса — это не настройка звука, а разрешение
  человека; без согласия владельца дома сервис не синтезирует НИЧЕГО, даже
  если провайдер установлен;
* **окно референса.** Меньше 10 секунд — мало данных для клона, больше 20 —
  ТЗ столько не просит; отказ называет, сколько секунд пришло;
* **кэш.** Одна и та же фраза не должна гонять GPU дважды; ключ — текст +
  голос + провайдер, а не «сегодняшнее настроение».

Провайдер (``f5_tts``/``chatterbox``) грузится ЛЕНИВО и в песочнице
отсутствует: это ``VoiceCloneUnavailable`` с НАЗВАННОЙ причиной, а не тихая
подмена обычным голосом. Подмена была бы хуже отказа: человек думал бы, что
говорит его клон.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: ТЗ F-111: «референс 10–20 с».
REFERENCE_MIN_S = 10.0
REFERENCE_MAX_S = 20.0

#: ТЗ F-111 называет два провайдера по имени.
PROVIDERS = ("f5_tts", "chatterbox")


class VoiceCloneUnavailable(RuntimeError):
    """Клон голоса недоступен: нет провайдера, весов или референса."""


class VoiceCloneRefused(RuntimeError):
    """Клон запрещён: выключен для дома или владелец не давал согласия."""


def reference_seconds(audio: bytes) -> float:
    """Длительность WAV-референса в секундах (ТЗ F-111: 10–20 с).

    Читается настоящий WAV: длительность — это не то, что можно попросить у
    вызывающего «на глазок», иначе окно 10–20 с превратилось бы в пожелание.
    """
    if not audio:
        raise VoiceCloneUnavailable("the voice reference is empty")
    try:
        with wave.open(BytesIO(audio), "rb") as stream:
            frame_rate = float(stream.getframerate() or 0.0)
            frames = float(stream.getnframes() or 0.0)
    except Exception as exc:  # noqa: BLE001 - не WAV или битый
        raise VoiceCloneUnavailable(f"the voice reference is not a WAV ({exc})") from exc
    if frame_rate <= 0:
        raise VoiceCloneUnavailable("the voice reference has no frame rate")
    return frames / frame_rate


class ConsentStore:
    """Кто разрешил клон СВОЕГО голоса (ТЗ F-111), на диске.

    Согласие — не флаг дома: его даёт человек, и он же может его отозвать.
    Хранится рядом с остальными локальными данными хаба; в git и логи не
    попадает ничего, кроме факта и момента.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001 - битый файл не роняет хаб
            log.warning("Could not read the voice-clone consents (%s)", exc)
            return
        if isinstance(raw, Mapping):
            self._items = {str(key): dict(value) for key, value in raw.items()
                           if isinstance(value, Mapping)}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._items, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - согласие не должно терять ход
            log.warning("Could not save the voice-clone consents (%s)", exc)

    @staticmethod
    def _key(home_id: str, person_id: str) -> str:
        return f"{str(home_id or '')}:{str(person_id or '')}"

    def grant(self, home_id: str, person_id: str, *, at: float | None = None) -> bool:
        home, person = str(home_id or ""), str(person_id or "")
        if not home or not person:
            return False
        self._items[self._key(home, person)] = {
            "home_id": home, "person_id": person,
            "granted_at": time.time() if at is None else float(at)}
        self._save()
        log.info("Voice-clone consent for %s in %s recorded", person, home)
        return True

    def granted(self, home_id: str, person_id: str) -> bool:
        return self._key(home_id, person_id) in self._items

    def revoke(self, home_id: str, person_id: str) -> bool:
        removed = self._items.pop(self._key(home_id, person_id), None)
        if removed is not None:
            self._save()
            log.info("Voice-clone consent for %s in %s revoked", person_id, home_id)
        return removed is not None

    def snapshot(self) -> dict[str, Any]:
        return {"consents": len(self._items),
                "people": sorted(str(item.get("person_id") or "")
                                 for item in self._items.values())}


class CloneProvider(Protocol):
    """Провайдер клона: имя, готовность и синтез по референсу."""

    name: str

    def ready(self) -> bool:  # pragma: no cover - протокол
        ...

    def synthesize(self, text: str, reference: bytes, *, language: str) -> bytes:  # pragma: no cover
        ...


@dataclass
class LocalCloneProvider:
    """F5-TTS / Chatterbox за ленивым импортом (ТЗ F-111).

    Ни один из пакетов в песочнице не установлен, и это НЕ повод притвориться,
    что синтез получился: :meth:`synthesize` называет пакет, которого нет.
    """

    provider: str = "f5_tts"
    language: str = "en"
    model: str = ""
    _engine: Any = None
    _error: str = ""

    @property
    def name(self) -> str:
        """Имя провайдера для отчёта и ключа кэша (ТЗ F-111)."""
        return str(self.provider or "f5_tts").strip().lower()

    def ready(self) -> bool:
        try:
            self._load()
        except VoiceCloneUnavailable:
            return False
        return True

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._error:
            raise VoiceCloneUnavailable(self._error)
        wanted = str(self.provider or "f5_tts").strip().lower()
        if wanted not in PROVIDERS:
            self._error = (f"the provider {wanted!r} is not one of {', '.join(PROVIDERS)}")
            raise VoiceCloneUnavailable(self._error)
        module = "f5_tts" if wanted == "f5_tts" else "chatterbox"
        try:
            __import__(module)
        except ImportError as exc:
            self._error = (f"{module} is not installed on the hub "
                           f"(pip install {module.replace('_', '-')}); the cloned "
                           "voice is unavailable")
            raise VoiceCloneUnavailable(self._error) from exc
        try:
            self._engine = self._build(wanted)
        except Exception as exc:  # noqa: BLE001 - веса могут не подняться
            self._error = f"{wanted} could not start ({exc})"
            raise VoiceCloneUnavailable(self._error) from exc
        log.info("Voice cloning is up: %s", wanted)
        return self._engine

    def _build(self, wanted: str) -> Any:
        if wanted == "f5_tts":
            from f5_tts.api import F5TTS  # type: ignore

            return F5TTS(model=self.model or None)
        from chatterbox.tts import ChatterboxTTS  # type: ignore

        return ChatterboxTTS.from_pretrained(device="cuda")

    def synthesize(self, text: str, reference: bytes, *, language: str) -> bytes:
        raise VoiceCloneUnavailable(self._error or (
            f"{self.provider}: synthesis needs the reference audio and the GPU "
            "provider that is not installed"))


@dataclass
class VoiceCloneService:
    """Клон голоса дома: согласие + окно референса + кэш (ТЗ F-111)."""

    enabled: bool = False
    provider: Any = None
    consent: ConsentStore | None = None
    cache_dir: Path | None = None
    max_cache_entries: int = 200
    cache_hits: int = 0
    cache_misses: int = 0
    synthesized: int = 0

    def allowed(self, home_id: str, person_id: str) -> bool:
        """Можно ли клонировать: флаг дома И согласие этого человека."""
        if not self.enabled:
            return False
        if self.consent is None:
            return False
        return self.consent.granted(home_id, person_id)

    def _key(self, home_id: str, person_id: str, text: str) -> str:
        payload = "\x1f".join((str(getattr(self.provider, 'name', '') or ''),
                               str(home_id or ''), str(person_id or ''), str(text or '')))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cached(self, home_id: str, person_id: str, text: str) -> bytes | None:
        if self.cache_dir is None:
            return None
        path = Path(self.cache_dir) / f"{self._key(home_id, person_id, text)}.wav"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - кэш не источник истины
            log.debug("Could not read the voice-clone cache (%s)", exc)
            return None
        return data or None

    def _remember(self, home_id: str, person_id: str, text: str, audio: bytes) -> None:
        if self.cache_dir is None or not audio:
            return
        path = Path(self.cache_dir) / f"{self._key(home_id, person_id, text)}.wav"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        except Exception as exc:  # noqa: BLE001 - кэш не отменяет синтез
            log.debug("Could not write the voice-clone cache (%s)", exc)
            return
        self._prune()

    def _prune(self) -> None:
        if self.cache_dir is None or self.max_cache_entries <= 0:
            return
        try:
            files = sorted(Path(self.cache_dir).glob("*.wav"),
                           key=lambda item: item.stat().st_mtime, reverse=True)
        except Exception:  # noqa: BLE001 - уборка не стоит синтеза
            return
        for old in files[self.max_cache_entries:]:
            try:
                old.unlink()
            except Exception:  # noqa: BLE001
                continue

    def synthesize(self, home_id: str, person_id: str, text: str,
                   reference: bytes, *, language: str = "") -> bytes:
        """Синтезировать фразу клонированным голосом или назвать причину отказа."""
        if not self.enabled:
            raise VoiceCloneRefused("the cloned voice is switched off for this home")
        if self.consent is None or not self.consent.granted(home_id, person_id):
            raise VoiceCloneRefused(
                f"{person_id or 'this person'} has not agreed to a clone of their voice")
        phrase = " ".join(str(text or "").split())
        if not phrase:
            raise VoiceCloneRefused("there is nothing to say")
        cached = self.cached(home_id, person_id, phrase)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        seconds = reference_seconds(reference)
        if seconds < REFERENCE_MIN_S or seconds > REFERENCE_MAX_S:
            raise VoiceCloneRefused(
                f"the voice reference is {seconds:.1f} s; the ТЗ window is "
                f"{REFERENCE_MIN_S:.0f}-{REFERENCE_MAX_S:.0f} s")
        if self.provider is None:
            raise VoiceCloneUnavailable("no voice-clone provider is configured")
        audio = self.provider.synthesize(phrase, reference, language=language)
        if not audio:
            raise VoiceCloneUnavailable("the provider returned no audio")
        self.synthesized += 1
        self._remember(home_id, person_id, phrase, audio)
        return audio

    def snapshot(self) -> dict[str, Any]:
        ready = False
        if self.provider is not None:
            try:
                ready = bool(self.provider.ready())
            except Exception:  # noqa: BLE001 - отчёт не должен падать
                ready = False
        return {"enabled": self.enabled, "provider": str(getattr(self.provider, "name", "") or ""),
                "provider_ready": ready, "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses, "synthesized": self.synthesized,
                "consents": (self.consent.snapshot() if self.consent is not None
                             else {"consents": 0, "people": []})}


def service_from_config(cfg: Any, *, data_dir: str | Path = "data",
                        provider: Any = None) -> VoiceCloneService:
    """Собрать сервис клона из ``server.voice_clone`` (ТЗ F-111).

    Провайдер собирается лениво: выключенный флаг не должен тянуть GPU-модель,
    а незнакомое имя провайдера — ронять старт хаба.
    """
    settings = getattr(getattr(cfg, "server", None), "voice_clone", None)
    if settings is None:
        return VoiceCloneService()
    enabled = bool(getattr(settings, "enabled", False))
    name = str(getattr(settings, "provider", "f5_tts") or "f5_tts").strip().lower()
    engine = provider
    if engine is None and enabled:
        engine = LocalCloneProvider(provider=name,
                                   language=str(getattr(settings, "language", "en") or "en"),
                                   model=str(getattr(settings, "model", "") or ""))
    cache_dir = getattr(settings, "cache_dir", "") or ""
    return VoiceCloneService(
        enabled=enabled,
        provider=engine,
        consent=ConsentStore(Path(data_dir) / "voice_clone_consent.json"),
        cache_dir=Path(cache_dir) if cache_dir else None,
        max_cache_entries=int(getattr(settings, "max_cache_entries", 200) or 200),
    )


__all__ = [
    "PROVIDERS",
    "REFERENCE_MAX_S",
    "REFERENCE_MIN_S",
    "CloneProvider",
    "ConsentStore",
    "LocalCloneProvider",
    "VoiceCloneRefused",
    "VoiceCloneService",
    "VoiceCloneUnavailable",
    "reference_seconds",
    "service_from_config",
]
