"""Measure the F-403 overflow threshold on the hub's real queue (ТЗ 15.1).

The rule (ТЗ F-403) is: a class-0 (utterance) job predicted to wait longer
than ``models.routing.overflow_wait_s`` is handed to ``cloud_cheap`` instead of
the GPU. The number in the config has to come from a measurement, not from a
guess, so this harness drives the hub's own :class:`hub.gpu_queue.GpuQueue`
with three active rooms and reports the admission wait of a class-0 job
(median / p95 / worst) at each load, the estimate the router saw when the job
was submitted, and the threshold those numbers imply.

What is real here: the queue implementation, its fair-share rule, its priority
ordering and its wait estimate. What is synthetic: the service time of one job,
because the GPU of the target machine (RTX 5090) is not in this sandbox. Pass
``--service-s`` with the round time measured on the stand (STT + the model
round) to turn this into a calibration of the queue rather than of the model.

Usage::

    python scripts/measure_overflow.py --homes 3 --rate 3 --service-s 2.5
    python scripts/measure_overflow.py --write      # stores the threshold
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_config  # noqa: E402
from hub.gpu_queue import PRIORITY_UTTERANCE, GpuQueue  # noqa: E402

#: The three rooms of the acceptance criterion (ТЗ 15.1: three active homes).
MEASURED_HOMES = ("livingroom", "dorm-max", "dorm-2")
#: Utterances per minute one active room produces during a conversation.
DEFAULT_RATE_PER_MIN = 3.0
#: One class-0 job: the STT pass plus the model round, measured on the stand.
DEFAULT_SERVICE_S = 2.5


def percentile(samples: list[float], fraction: float) -> float:
    """The ``fraction``-th percentile of ``samples`` (nearest-rank)."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    index = min(len(ordered) - 1, rank - 1)
    return ordered[index]


def summarize(samples: list[float]) -> dict[str, float | int]:
    """Median, p95 and worst of a list of waits (seconds)."""
    if not samples:
        return {"count": 0, "median_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    return {
        "count": len(samples),
        "median_s": round(statistics.median(samples), 3),
        "p95_s": round(percentile(samples, 0.95), 3),
        "max_s": round(max(samples), 3),
    }


def allowed_extra_wait_s(*, stt_s: float = 0.7, decide_s: float = 0.4,
                         growth: float = 1.5) -> float:
    """The extra waiting ТЗ 15.1 allows when three rooms are active at once.

    The table gives 0.7 s from the end of speech to the transcript and 0.4 s
    from there to the start of generation, and allows those budgets to grow by
    at most 1.5x with three active homes. The waiting the queue is allowed to
    add is therefore the difference between the stretched and the plain budget:
    0.55 s. A class-0 job predicted to wait longer cannot stay inside the
    acceptance budget any more, and that is exactly when F-403 overflows.
    """
    return round((stt_s + decide_s) * (growth - 1.0), 2)


def overflow_rate(samples: list[float], threshold_s: float) -> float:
    """The share of jobs whose wait would send them to the overflow level."""
    if not samples:
        return 0.0
    over = sum(1 for wait in samples if wait > threshold_s)
    return round(over / len(samples), 3)


async def measure(*, homes: tuple[str, ...], rate_per_min: float, service_s: float,
                  utterances_per_home: int, max_concurrent: int, fair_share: float,
                  max_waiting: int, time_scale: float = 10.0) -> dict:
    """Drive the real queue with ``homes`` and collect the class-0 waits.

    ``time_scale`` compresses the simulated clock: the service time and the gap
    between two utterances are both divided by it, so the queue sees exactly the
    same load ratio while the run takes seconds. The reported waits are scaled
    back up to real seconds.
    """
    queue = GpuQueue(max_concurrent=max_concurrent, fair_share=fair_share,
                     max_waiting=max_waiting)
    waits: list[float] = []
    estimates: list[float] = []
    scale = max(1.0, float(time_scale))
    service = service_s / scale
    # One room speaks one utterance every 60/rate seconds; the rooms run
    # side by side, so the total offered load is rate * len(homes). Arrivals
    # are exponential around that mean (an M/M/c pattern) rather than perfectly
    # spaced: a room speaks when somebody speaks, not on a metronome.
    rng = random.Random(20260921)
    interval = (60.0 / max(0.001, rate_per_min) - service_s) / scale

    async def job(home: str) -> float:
        started = asyncio.get_running_loop().time()
        await asyncio.sleep(service)
        return started

    async def room(home: str) -> None:
        for index in range(utterances_per_home):
            if index:
                await asyncio.sleep(max(0.0, rng.expovariate(1.0 / max(0.01, interval))))
            estimate = queue.wait_estimate(PRIORITY_UTTERANCE)
            await queue.submit(PRIORITY_UTTERANCE, home, lambda: job(home), label=home)
            # The queue measures the admission wait itself (queued → started);
            # timing it around ``submit`` would add the service time to it.
            waits.append(queue.stats()["last_wait_s"]["utterance"] * scale)
            estimates.append(estimate * scale)

    await asyncio.gather(*(room(home) for home in homes))
    await queue.close()
    return {"waits": waits, "estimates": estimates, "stats": queue.stats()}


def _write_threshold(path: Path, value: float) -> None:
    """Store the measured threshold in ``models.routing.overflow_wait_s``.

    The edit is line-based on purpose: the configuration is a hand-written file
    full of comments and a re-dump would throw every one of them away.
    """
    import re

    lines = path.read_text(encoding="utf-8").splitlines()
    in_models = False
    in_routing = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_models = stripped.startswith("models:")
            in_routing = False
        elif in_models and indent <= 2:
            in_routing = stripped.startswith("routing:")
        elif in_models and in_routing and stripped.startswith("overflow_wait_s:"):
            comment = ""
            marker = re.search(r"\s{2,}#", line)
            if marker:
                comment = line[marker.start():]
            lines[index] = f"{' ' * indent}overflow_wait_s: {value}{comment}"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    raise SystemExit("models.routing.overflow_wait_s is not in the config; add it first")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_overflow", description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_PER_MIN,
                        help="utterances per minute per active room")
    parser.add_argument("--service-s", type=float, default=DEFAULT_SERVICE_S,
                        help="seconds one class-0 job occupies the GPU")
    parser.add_argument("--per-home", type=int, default=12,
                        help="utterances each room speaks during the run")
    parser.add_argument("--time-scale", type=float, default=10.0,
                        help="compress the simulated clock by this factor (queue load ratio unchanged)")
    parser.add_argument("--write", action="store_true",
                        help="store the recommended threshold in the config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    queue_cfg = cfg.server.gpu_queue
    routing = cfg.models.routing
    design = asyncio.run(measure(
        homes=MEASURED_HOMES, rate_per_min=args.rate, service_s=args.service_s,
        utterances_per_home=args.per_home, max_concurrent=queue_cfg.max_concurrent,
        fair_share=queue_cfg.fair_share, max_waiting=queue_cfg.max_waiting,
        time_scale=args.time_scale,
    ))
    double = asyncio.run(measure(
        homes=MEASURED_HOMES, rate_per_min=args.rate * 2, service_s=args.service_s,
        utterances_per_home=args.per_home * 2, max_concurrent=queue_cfg.max_concurrent,
        fair_share=queue_cfg.fair_share, max_waiting=queue_cfg.max_waiting,
        time_scale=args.time_scale,
    ))

    design_summary = summarize(design["waits"])
    double_summary = summarize(double["waits"])
    threshold = allowed_extra_wait_s()
    report = {
        "homes": list(MEASURED_HOMES),
        "rate_per_min_per_home": args.rate,
        "service_s": args.service_s,
        "time_scale": args.time_scale,
        "queue": {"max_concurrent": queue_cfg.max_concurrent,
                  "fair_share": queue_cfg.fair_share},
        "design_load": design_summary,
        "double_load": double_summary,
        # The ratio the threshold is derived from: a wait of one whole
        # class-0 job is what the design load already produces.
        "design_wait_in_service_times": round(float(design_summary["p95_s"]) / args.service_s, 2),
        "configured_overflow_wait_s": routing.overflow_wait_s,
        "recommended_overflow_wait_s": threshold,
        # The threshold is the budget the spec allows, and the run is the check
        # that the design load mostly stays inside it: a few jobs may overflow
        # (that is the point of F-403), most must not.
        "design_overflow_rate": overflow_rate(design["waits"], threshold),
        "double_load_overflow_rate": overflow_rate(double["waits"], threshold),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write:
        _write_threshold(Path(args.config), threshold)
        print(f"\nStored models.routing.overflow_wait_s = {threshold} in {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
