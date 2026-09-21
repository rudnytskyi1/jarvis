"""How much one room delays another (ТЗ 15.1, критерий приёмки фазы 1).

The acceptance criterion is: a reply in one room must not hold up another room
by more than 1.5 s. Two different mechanisms could do that, so both are measured
separately instead of being mixed into one number:

``pipeline``
    The hub's own scheduling. Two real :class:`hub.app.Connection` objects are
    driven through the real turn pipeline at the same time; one room's model
    round is slow, the other room's is fast, and the report gives how much
    longer the fast room's reply took than the same reply on a quiet hub. An
    ``await`` that blocks the loop, a shared lock held across a whole turn or a
    serialized per-room step would all show up here as seconds.

``queue``
    The shared GPU. Rooms contend for the single GPU queue (``hub.gpu_queue``),
    which is the only place where one room's work can really queue behind
    another's. The queue implementation, its priority classes, its per-home
    fair share and its wait bookkeeping are real; the service time of one job
    (STT + the model round) is a parameter, because the RTX 5090 of the target
    machine is not in this sandbox. Run it with the round time measured on the
    stand to turn this into a calibration of the real machine.

Usage::

    python scripts/measure_cross_room_delay.py
    python scripts/measure_cross_room_delay.py --service-s 1.8 --rate 4 --repeats 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import Config, load_config  # noqa: E402
from common.ids import new_ulid  # noqa: E402
from hub import app as hub_app  # noqa: E402
from hub.gpu_queue import PRIORITY_UTTERANCE, GpuQueue  # noqa: E402
from hub.llm import LlmResult  # noqa: E402
from hub.session import Session  # noqa: E402
from scripts.measure_overflow import measure as measure_queue_load  # noqa: E402
from scripts.measure_overflow import summarize  # noqa: E402

#: The acceptance criterion of phase 1 (ТЗ 16).
CRITERION_S = 1.5
#: ТЗ 15.1 speaks about three simultaneously active rooms.
MEASURED_HOMES = ("livingroom", "dorm-max", "dorm-2")
#: The two rooms of the scheduling measurement: one talks slowly, one briefly.
SLOW_HOME, FAST_HOME = MEASURED_HOMES[0], MEASURED_HOMES[1]
#: One class-0 GPU job: the STT pass plus the model round (see P1-24).
DEFAULT_SERVICE_S = 2.5
#: The slow room's model round in the pipeline measurement.
DEFAULT_SLOW_S = 2.0
#: The fast room's model round: a short answer.
DEFAULT_FAST_S = 0.2
#: A cloud round is a network call: no local queue, one round trip.
DEFAULT_CLOUD_S = 0.8
SLOW_MARKER = "the long story"
FAST_MARKER = "a short word"


class _RoomSocket:
    """Just enough websocket for the pipeline to talk to a room."""

    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)
        self.client_state = hub_app.WebSocketState.CONNECTED
        self.frames: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = hub_app.WebSocketState.DISCONNECTED


class _Stt:
    """One hub-wide STT engine returning the transcript of the room's audio."""

    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts

    def transcribe_pcm(self, pcm, sample_rate, language=None):
        return self.texts[pcm[0]], "en"


class _Brain:
    """A model round whose duration depends on which room asked."""

    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays

    async def generate(self, history, run_tool):
        asked = " ".join(str(m.get("content") or "") for m in history if m.get("role") == "user")
        delay = next((seconds for marker, seconds in self.delays.items() if marker in asked), 0.0)
        await asyncio.sleep(delay)
        return LlmResult(text=f"Answered in {delay:.1f} s.", tool_calls=[], rounds=1,
                         history=list(history))

    async def verify(self, history, answer, run_tool):
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))


def _room(home_id: str, client_id: str) -> hub_app.Connection:
    cfg = Config()
    # The criterion is about scheduling, not about who may speak.
    cfg.server.permissions_enabled = False
    conn = hub_app.Connection(_RoomSocket(), cfg)
    conn.peer = f"{client_id}:5100"
    conn.home_id = home_id
    conn.session = Session(client_id=client_id, devices=[], history_turns=4)
    conn.utterance_id = new_ulid()

    async def no_tts(*args, **kwargs) -> None:
        return None

    conn._stream_tts = no_tts  # type: ignore[method-assign]
    return conn


async def _timed(conn: hub_app.Connection, pcm: bytes) -> float:
    started = time.perf_counter()
    await conn._handle_utterance(pcm)
    return time.perf_counter() - started


async def measure_pipeline(*, slow_s: float = DEFAULT_SLOW_S, fast_s: float = DEFAULT_FAST_S,
                           repeats: int = 3) -> dict:
    """The extra reply latency the fast room suffers while the slow room talks.

    The GPU queue is taken out of the picture (``_gpu_off``) so the number is
    about the hub's own scheduling; the queue is measured separately below.
    """
    hub_app._stt = _Stt({1: f"tell me {SLOW_MARKER}", 2: f"say {FAST_MARKER}"})
    hub_app._llm = _Brain({SLOW_MARKER: slow_s, FAST_MARKER: fast_s})
    hub_app._tts = SimpleNamespace(sample_rate=48000)
    hub_app._voices = None
    hub_app._memory = None
    hub_app._dialogs = None
    hub_app._hub_conn = None
    hub_app._decider = None
    hub_app._decision_log = False
    hub_app._gpu = None
    hub_app._gpu_off = True

    solo: list[float] = []
    paired: list[float] = []
    slow_turns: list[float] = []
    for _ in range(max(1, repeats)):
        solo.append(await _timed(_room(FAST_HOME, "pc-fast"), b"\x02" * 16000))
        slow = _timed(_room(SLOW_HOME, "pc-slow"), b"\x01" * 16000)
        fast = _timed(_room(FAST_HOME, "pc-fast"), b"\x02" * 16000)
        slow_s_turn, fast_s_turn = await asyncio.gather(slow, fast)
        slow_turns.append(slow_s_turn)
        paired.append(fast_s_turn)
    extra = [round(paired[index] - solo[index], 4) for index in range(len(solo))]
    return {
        "slow_turn_s": summarize(slow_turns),
        "solo_fast_s": summarize(solo),
        "paired_fast_s": summarize(paired),
        "extra_s": summarize(extra),
    }


def measure_queue(*, service_s: float, rate_per_min: float, per_home: int,
                  max_concurrent: int, fair_share: float, max_waiting: int,
                  time_scale: float = 10.0) -> dict:
    """The admission wait the real queue adds with three rooms talking."""
    result = asyncio.run(measure_queue_load(
        homes=MEASURED_HOMES, rate_per_min=rate_per_min, service_s=service_s,
        utterances_per_home=per_home, max_concurrent=max_concurrent,
        fair_share=fair_share, max_waiting=max_waiting, time_scale=time_scale,
    ))
    return summarize(result["waits"])


async def _queue_run(*, service_s: float, rate_per_min: float, per_home: int,
                     max_concurrent: int, fair_share: float, max_waiting: int,
                     time_scale: float, overflow_wait_s: float | None,
                     cloud_s: float) -> tuple[list[float], int]:
    """Drive the real queue; return the waits and how many jobs overflowed.

    ``overflow_wait_s`` is the F-403 threshold: a job whose *predicted* wait
    (the queue's own estimate) exceeds it leaves the local GPU for the cloud
    level instead of queueing behind another room. ``None`` measures the queue
    without that rule, which is the worst case.
    """
    queue = GpuQueue(max_concurrent=max_concurrent, fair_share=fair_share,
                     max_waiting=max_waiting)
    waits: list[float] = []
    overflowed = 0
    scale = max(1.0, float(time_scale))
    service = service_s / scale
    rng = random.Random(20260921)
    interval = (60.0 / max(0.001, rate_per_min) - service_s) / scale

    async def job() -> None:
        await asyncio.sleep(service)

    async def room(home: str) -> None:
        nonlocal overflowed
        for index in range(per_home):
            if index:
                await asyncio.sleep(max(0.0, rng.expovariate(1.0 / max(0.01, interval))))
            estimate = queue.wait_estimate(PRIORITY_UTTERANCE) * scale
            if overflow_wait_s is not None and estimate > overflow_wait_s:
                overflowed += 1
                await asyncio.sleep(cloud_s / scale)
                waits.append(0.0)
                continue
            await queue.submit(PRIORITY_UTTERANCE, home, job, label=home)
            waits.append(queue.stats()["last_wait_s"]["utterance"] * scale)

    await asyncio.gather(*(room(home) for home in MEASURED_HOMES))
    await queue.close()
    return waits, overflowed


def measure_queue_with_overflow(*, service_s: float, rate_per_min: float, per_home: int,
                                max_concurrent: int, fair_share: float, max_waiting: int,
                                overflow_wait_s: float, cloud_s: float = DEFAULT_CLOUD_S,
                                time_scale: float = 10.0) -> dict:
    """The same load with the F-403 rule on: long waits go to the cloud level."""
    waits, overflowed = asyncio.run(_queue_run(
        service_s=service_s, rate_per_min=rate_per_min, per_home=per_home,
        max_concurrent=max_concurrent, fair_share=fair_share, max_waiting=max_waiting,
        time_scale=time_scale, overflow_wait_s=overflow_wait_s, cloud_s=cloud_s,
    ))
    summary = summarize(waits)
    summary["overflowed_jobs"] = overflowed
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_cross_room_delay", description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--service-s", type=float, default=DEFAULT_SERVICE_S,
                        help="seconds one utterance occupies the GPU (STT + model round)")
    parser.add_argument("--rate", type=float, default=3.0,
                        help="utterances per minute per active room")
    parser.add_argument("--per-home", type=int, default=12)
    parser.add_argument("--time-scale", type=float, default=10.0)
    parser.add_argument("--slow-s", type=float, default=DEFAULT_SLOW_S)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    queue_cfg = cfg.server.gpu_queue
    threshold = cfg.models.routing.overflow_wait_s
    pipeline = asyncio.run(measure_pipeline(slow_s=args.slow_s, repeats=args.repeats))
    queue = measure_queue(
        service_s=args.service_s, rate_per_min=args.rate, per_home=args.per_home,
        max_concurrent=queue_cfg.max_concurrent, fair_share=queue_cfg.fair_share,
        max_waiting=queue_cfg.max_waiting, time_scale=args.time_scale,
    )
    capped = measure_queue_with_overflow(
        service_s=args.service_s, rate_per_min=args.rate, per_home=args.per_home,
        max_concurrent=queue_cfg.max_concurrent, fair_share=queue_cfg.fair_share,
        max_waiting=queue_cfg.max_waiting, overflow_wait_s=threshold,
        time_scale=args.time_scale,
    )
    worst = max(pipeline["extra_s"]["max_s"], capped["max_s"])
    report = {
        "criterion_s": CRITERION_S,
        "homes": list(MEASURED_HOMES),
        "service_s": args.service_s,
        "queue": {"max_concurrent": queue_cfg.max_concurrent, "fair_share": queue_cfg.fair_share},
        "pipeline": pipeline,
        # Without F-403: what the local queue alone does under the same load.
        "queue_wait_s_without_overflow": queue,
        "overflow_wait_s": threshold,
        # With F-403, which is what the shipped pipeline does when the owner has
        # allowed the cloud level: a job predicted to wait too long leaves the
        # local GPU instead of making its room wait.
        "queue_wait_s_with_overflow": capped,
        "worst_measured_s": round(worst, 3),
        "within_criterion": bool(worst <= CRITERION_S),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["within_criterion"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
