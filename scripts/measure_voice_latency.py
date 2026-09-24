"""Сколько идёт ЖИВОЙ ход комнаты: конец речи → первый звук (ТЗ 15.1, F-101).

Стенд (``scripts/live-eval.py``) отвечает на реплику текстом и меряет только
модель с Jev. Бюджет F-101 (1.2 с) считается от КОНЦА РЕЧИ, то есть включает
STT, чтение Jev, раунды модели и синтез первого предложения. Этот замер идёт
по настоящему пайплайну хаба (``Connection._handle_utterance``) с настоящими
движками из ``config.openai.yaml``: faster-whisper (STT), Jev (по флагу дома),
``deepseek-flash`` и kokoro (TTS). Речь — настоящие записи из ``data/voices``
(те же WAV, что человек наговаривал при регистрации голоса).

Чего у стенда нет и почему это сказано вслух: микрофона и комнатного ПК.
Поэтому ход идёт по загруженному файлу, а действия на ПК честно отказываются
(``the probe has no room PC``) — иначе ход ждал бы ответа машины, которой тут
нет, и мерил бы таймаут, а не задержку ответа.

Запуск::

    python scripts/measure_voice_latency.py --person Anton --limit 5
    python scripts/measure_voice_latency.py --wav data/voices/Anton/<file>.wav
    python scripts/measure_voice_latency.py --json data/audit/voice-latency.json
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_config  # noqa: E402
from hub import app as hub_app  # noqa: E402

#: Строка, которой хаб сам отчитывается о стадиях хода (``hub/app.py``).
DONE_MARKER = "done in"


def _load_bench_module() -> Any:
    """Стенд как модуль: он уже собирает комнату так же, как живой хаб."""
    path = REPO_ROOT / "scripts" / "live-eval.py"
    spec = importlib.util.spec_from_file_location("live_eval_voice_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_eval_voice_probe"] = module
    sys.modules["live_eval_bench_latency"] = module
    spec.loader.exec_module(module)
    return module


def load_dotenv(path: Path, environ: Any = None) -> list[str]:
    """Ключи из ``.env`` (как их читает ``run-openai-server.ps1``), без print."""
    environ = os.environ if environ is None else environ
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and value and not environ.get(name):
            environ[name] = value
            loaded.append(name)
    return loaded


class _StageCapture(logging.Handler):
    """Ловит строку стадий самого хаба, а не считает их заново."""

    def __init__(self) -> None:
        super().__init__()
        self.stages: dict[str, int] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if DONE_MARKER not in message or "stt" not in message:
            return
        try:
            # "Utterance <id> done in 4500 ms (stt 1500, llm 2700, tts 300), 2 tool call(s)"
            tail = message.split(DONE_MARKER, 1)[1]
            total = int(tail.strip().split(" ", 1)[0])
            inside = tail.split("(", 1)[1].split(")", 1)[0]
            parts = [piece.strip().split() for piece in inside.split(",")]
            stages = {piece[0]: int(piece[1]) for piece in parts if len(piece) == 2}
            stages["total"] = total
        except (IndexError, ValueError):
            return
        self.stages = stages


def read_pcm(path: Path) -> bytes:
    """PCM 16 бит моно, как его шлёт клиент (частота берётся из файла)."""
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError(f"{path.name}: ожидается 16-битный WAV")
        channels = source.getnchannels()
        pcm = source.readframes(source.getnframes())
    if channels > 1:  # держим только первый канал
        step = channels * 2
        pcm = b"".join(pcm[index:index + 2] for index in range(0, len(pcm), step))
    return pcm


def wav_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as source:
        return int(source.getframerate())


def recordings(person: str, limit: int) -> list[Path]:
    folder = (REPO_ROOT / "data" / "voices" / person).resolve()
    voices_root = (REPO_ROOT / "data" / "voices").resolve()
    if folder.parent != voices_root:
        raise ValueError("укажите имя профиля из data/voices, а не путь")
    files = sorted(folder.glob("*.wav"), key=lambda path: path.stat().st_size)
    return files[:limit] if limit else files


async def measure(cfg: Any, files: list[Path], *, with_diarization: bool,
                  verbose: bool = False) -> list[dict[str, Any]]:
    bench_module = _load_bench_module()
    bench = bench_module.Bench(cfg, actions=False, worker=95)
    await bench.start()
    # Стенд держит модель на своём объекте: живой ``_handle_utterance`` читает
    # её из модуля ``hub.app``, поэтому ход получил бы «server not ready».
    hub_app._llm = bench._llm
    # Реестр людей — свой, стендовый: живой ход узнаёт говорящего, но не пишет
    # в профиль владельца (``data/people.json``).
    from hub.speaker import VoiceRegistry

    hub_app._voices = VoiceRegistry(bench.data_dir, enabled=True)

    print("loading faster-whisper ...", flush=True)
    from hub.stt import SttEngine

    # ``SttEngine`` грузит веса в конструкторе, поэтому и в отдельном потоке.
    engine = await asyncio.to_thread(SttEngine, cfg.server.stt)
    hub_app._stt = engine
    if with_diarization:
        from hub.diarization import DiarizationEngine

        diarizer = DiarizationEngine(cfg.server.diarization)
        try:
            print("loading diarization ...", flush=True)
            await asyncio.to_thread(diarizer.load)
            hub_app._diarizer = diarizer
        except Exception as exc:  # noqa: BLE001 - the probe says so and goes on
            print(f"note: diarization did not load ({type(exc).__name__}: {exc})", flush=True)

    connection = bench.connection
    from hub.tools import CLIENT_TOOLS

    hub_execute = connection._execute_tool

    async def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Инструменты, которым нужен комнатный ПК, здесь честно отказывают."""
        if name in CLIENT_TOOLS:
            return {"ok": False, "error": "the probe has no room PC"}
        return await hub_execute(name, args)

    connection._execute_tool = execute_tool  # type: ignore[method-assign]

    capture = _StageCapture()
    hub_log = logging.getLogger("jarvis.server.app")
    hub_log.addHandler(capture)
    # Хаб пишет стадии на уровне INFO: без этого запись не создаётся вовсе и
    # в отчёте стояло бы «—» вместо измеренных стадий.
    previous_level = hub_log.level
    if hub_log.getEffectiveLevel() > logging.INFO:
        hub_log.setLevel(logging.INFO)

    results: list[dict[str, Any]] = []
    try:
        for path in files:
            bench.messages.clear()
            capture.stages = None
            pcm = read_pcm(path)
            rate = wav_rate(path)
            audio_s = len(pcm) / 2.0 / float(rate)
            started = time.perf_counter()
            await connection._handle_utterance(pcm)
            wall_ms = int((time.perf_counter() - started) * 1000)
            if verbose:
                kinds = [str(message.get("type")) for message in bench.messages]
                print(f"  [{path.name}] frames: {kinds}", flush=True)
            reply = next((message.get("text", "") for message in reversed(bench.messages)
                          if str(message.get("type")) == "say"), "")
            results.append({
                "wav": path.name,
                "audio_s": round(audio_s, 2),
                "wall_ms": wall_ms,
                "first_audio_ms": connection.last_first_audio_ms,
                "stages_ms": capture.stages or {},
                "reply": reply[:200],
            })
    finally:
        hub_log.removeHandler(capture)
        hub_log.setLevel(previous_level)
        await bench.close()
    return results


def summarize(results: list[dict[str, Any]]) -> str:
    lines = ["| запись | речи, с | STT, мс | LLM, мс | TTS, мс | весь ход, мс | первый звук, мс |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        stages = row["stages_ms"]
        first = row["first_audio_ms"]
        lines.append(
            f"| {row['wav']} | {row['audio_s']:.1f} | {stages.get('stt', '—')} |"
            f" {stages.get('llm', '—')} | {stages.get('tts', '—')} |"
            f" {stages.get('total', row['wall_ms'])} | {first if first is not None else '—'} |")
    measured = [row["first_audio_ms"] for row in results if row["first_audio_ms"] is not None]
    if measured:
        budget = 1200
        over = sum(1 for value in measured if value >= budget)
        lines.append("")
        lines.append(f"Первый звук: замеров {len(measured)}, медиана "
                     f"{int(statistics.median(measured))} мс, максимум {max(measured)} мс; "
                     f"позже бюджета {budget} мс — {over}/{len(measured)}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.openai.yaml")
    parser.add_argument("--person", default="Anton", help="профиль из data/voices")
    parser.add_argument("--wav", action="append", default=[], help="конкретный файл записи")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--diarization", action="store_true",
                        help="поднять и движок диаризации (дольше, ближе к живому хабу)")
    parser.add_argument("--json", default="", help="куда записать замеры")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="печатать кадры каждого хода и лог хаба")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s")
    loaded = load_dotenv(REPO_ROOT / ".env")
    if loaded:
        print(f"keys from .env: {', '.join(loaded)}", flush=True)
    cfg = load_config(str(REPO_ROOT / args.config))
    hub_app.configure(cfg)
    files = [Path(item) for item in args.wav] or recordings(args.person, args.limit)
    if not files:
        print("no recordings to measure", file=sys.stderr)
        return 2
    print(f"measuring {len(files)} recording(s)", flush=True)
    results = asyncio.run(measure(cfg, files, with_diarization=args.diarization,
                                  verbose=args.verbose))
    print(summarize(results))
    for row in results:
        print(f"- {row['wav']}: {row['reply']}")
    if args.json:
        target = Path(args.json)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
