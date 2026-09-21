"""STT batching: utterances that arrive together share one decode (ТЗ 4.4)."""
from __future__ import annotations

import asyncio

import pytest

from common.config import Config
from hub.stt import SttBatcher, SttEngine


class _FakePipeline:
    def __init__(self):
        self.calls: list[dict] = []

    def transcribe(self, audio, language=None, **kwargs):
        self.calls.append({"audio_len": len(audio), "language": language, "batch_size": kwargs.get("batch_size")})

        class _Info:
            language = "en"

        class _Seg:
            text = "hello"
            avg_logprob = -0.1
            no_speech_prob = 0.01
            words = None

        return [_Seg()], _Info()


class _FakeEngine:
    """Stands in for :class:`SttEngine` without loading a real model."""

    def __init__(self, results=None):
        self.calls: list[list[bytes]] = []
        self.batch_sizes: list[int] = []
        self._results = results

    def transcribe_batch(self, clips, sample_rate=16000, language=None, *, batch_size=None):
        self.calls.append(list(clips))
        self.batch_sizes.append(int(batch_size or len(clips)))
        if self._results is not None:
            return self._results
        return [(f"text-{index}", "en") for index, _ in enumerate(clips)]


def test_batcher_groups_concurrent_utterances_into_one_call():
    engine = _FakeEngine()
    batcher = SttBatcher(engine, batch_size=4, window_ms=30)

    async def scenario():
        results = await asyncio.gather(
            *[batcher.transcribe(b"clip" + bytes([index]), 16000, "en") for index in range(3)]
        )
        await batcher.aclose()
        return results

    results = asyncio.run(scenario())
    assert results == [("text-0", "en"), ("text-1", "en"), ("text-2", "en")]
    assert len(engine.calls) == 1, "three utterances must share one decode"
    assert engine.batch_sizes == [3]
    assert batcher.stats() == {"batches": 1, "clips": 3, "max_batch": 3}


def test_batcher_never_exceeds_the_configured_batch_size():
    engine = _FakeEngine()
    batcher = SttBatcher(engine, batch_size=2, window_ms=20)

    async def scenario():
        await asyncio.gather(*[batcher.transcribe(b"clip", 16000) for _ in range(5)])
        await batcher.aclose()

    asyncio.run(scenario())
    assert all(len(call) <= 2 for call in engine.calls)
    assert sum(len(call) for call in engine.calls) == 5
    assert batcher.stats()["max_batch"] <= 2


def test_a_single_utterance_is_not_delayed_by_the_window():
    engine = _FakeEngine()
    batcher = SttBatcher(engine, batch_size=4, window_ms=0)

    async def scenario():
        result = await asyncio.wait_for(batcher.transcribe(b"clip", 16000), timeout=1)
        await batcher.aclose()
        return result

    assert asyncio.run(scenario()) == ("text-0", "en")
    assert engine.batch_sizes == [1]


def test_batch_runs_through_the_supplied_runner():
    engine = _FakeEngine()
    seen: list[str] = []

    async def runner(work):
        seen.append("gpu-slot")
        return await asyncio.to_thread(work)

    batcher = SttBatcher(engine, batch_size=2, window_ms=10, runner=runner)

    async def scenario():
        await batcher.transcribe(b"a", 16000)
        await batcher.aclose()

    asyncio.run(scenario())
    assert seen == ["gpu-slot"]


def test_engine_errors_reach_every_waiter():
    class _Broken:
        def transcribe_batch(self, clips, sample_rate=16000, language=None, *, batch_size=None):
            raise RuntimeError("cuda out of memory")

    batcher = SttBatcher(_Broken(), batch_size=3, window_ms=10)

    async def scenario():
        with pytest.raises(RuntimeError, match="out of memory"):
            await asyncio.gather(batcher.transcribe(b"a", 16000), batcher.transcribe(b"b", 16000))
        await batcher.aclose()

    asyncio.run(scenario())


def test_batch_size_limits_are_configurable():
    with pytest.raises(ValueError, match="batch_size"):
        SttBatcher(_FakeEngine(), batch_size=5)
    with pytest.raises(ValueError, match="window_ms"):
        SttBatcher(_FakeEngine(), window_ms=-1)
    cfg = Config()
    assert 2 <= cfg.server.stt.batch_size <= 4
    assert cfg.server.stt.batch_window_ms >= 0


def test_engine_uses_the_batched_pipeline_for_multi_clip_calls(monkeypatch):
    engine = SttEngine.__new__(SttEngine)
    engine.default_language = None
    engine.allowed_languages = []
    engine.hotwords = ""
    pipeline = _FakePipeline()
    engine._batched_pipeline = pipeline

    results = engine.transcribe_batch([b"\x00\x00" * 16000, b"\x00\x00" * 16000], 16000, "en", batch_size=2)
    assert results == [("hello", "en"), ("hello", "en")]
    assert [call["batch_size"] for call in pipeline.calls] == [2, 2]


def test_engine_falls_back_when_the_batched_pipeline_is_unavailable():
    engine = SttEngine.__new__(SttEngine)
    engine.default_language = None
    engine.allowed_languages = []
    engine.hotwords = ""
    engine._batched_pipeline = False
    assert engine.batched_pipeline() is None
