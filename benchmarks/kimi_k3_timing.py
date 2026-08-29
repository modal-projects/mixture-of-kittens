"""Pure statistics used by the TP8 Kimi K3 decode benchmark."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return an R-7 linearly interpolated quantile of nonempty samples."""
    if not samples:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(sample) for sample in samples)
    if not all(math.isfinite(sample) for sample in ordered):
        raise ValueError("percentile samples must be finite")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def geometric_mean(samples: Sequence[float]) -> float:
    """Return the geometric mean, accumulating products in log space."""
    if not samples:
        raise ValueError("geometric mean requires at least one sample")
    values = [float(sample) for sample in samples]
    if not all(math.isfinite(sample) and sample > 0.0 for sample in values):
        raise ValueError("geometric mean samples must be finite and positive")
    return math.exp(math.fsum(math.log(sample) for sample in values) / len(values))


def rank_max_samples(rank_samples: Sequence[Sequence[float]]) -> list[float]:
    """Take the maximum rank latency independently for every iteration."""
    if not rank_samples:
        raise ValueError("rank maxima require at least one rank")
    sample_count = len(rank_samples[0])
    if sample_count == 0:
        raise ValueError("rank maxima require at least one sample")
    if any(len(samples) != sample_count for samples in rank_samples):
        raise ValueError("every rank must provide the same number of samples")
    return [
        max(float(rank_samples[rank][iteration]) for rank in range(len(rank_samples)))
        for iteration in range(sample_count)
    ]


def summarize_rank_max(
    rank_samples: Sequence[Sequence[float]],
) -> dict[str, int | float]:
    """Summarize per-iteration rank maxima in milliseconds."""
    samples = rank_max_samples(rank_samples)
    return {
        "sample_count": len(samples),
        "median_ms": percentile(samples, 0.5),
        "p90_ms": percentile(samples, 0.9),
        "p99_ms": percentile(samples, 0.99),
        "geomean_ms": geometric_mean(samples),
    }


class TimingEvent(Protocol):
    """The part of ``torch.cuda.Event`` a replay measurement uses."""

    def record(self) -> None: ...

    def elapsed_time(self, other: TimingEvent) -> float: ...


def replay_samples(
    replay: Callable[[int], None],
    *,
    warmup_count: int,
    sample_count: int,
    event_factory: Callable[[], TimingEvent],
    synchronize: Callable[[], None],
) -> list[float]:
    """Time ``sample_count`` replays, after warming the kernel and the instrument.

    The warmups are the kernel's. The two discarded pairs are the instrument's.
    A process's first ``cuda.Event`` record pays a one-time driver
    initialization, and every sample here is enqueued back to back before a
    single synchronization, so that cost lands entirely in whichever replay it
    brackets. The first pair pays it. The replay that pair brackets is
    therefore not a steady-state one either, so a second pair runs and is
    discarded as well, and only then does the persisted series begin.

    The series is exactly ``sample_count`` long: the discarded replays are
    extra work, not samples taken out of the count.

    The iteration index continues across the warmups, both discarded replays,
    and the measured ones, so a caller that rotates a graph pool by index keeps
    rotating it.
    """
    if warmup_count < 1 or sample_count < 1:
        raise ValueError("warmup and sample counts must be positive")
    for iteration in range(warmup_count):
        replay(iteration)
    synchronize()

    settled = warmup_count
    for _ in range(2):
        start = event_factory()
        end = event_factory()
        start.record()
        replay(settled)
        end.record()
        synchronize()
        start.elapsed_time(end)
        settled += 1

    starts = [event_factory() for _ in range(sample_count)]
    ends = [event_factory() for _ in range(sample_count)]
    for offset, (start, end) in enumerate(zip(starts, ends, strict=True)):
        start.record()
        replay(settled + offset)
        end.record()
    synchronize()
    return [
        start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
    ]


def rotating_candidate_orders(
    candidates: Sequence[int],
    repeats: int,
) -> list[tuple[int, ...]]:
    """Rotate candidate order once per repeat to expose temporal drift."""
    if not candidates:
        raise ValueError("candidate order requires at least one grid")
    if repeats < 1:
        raise ValueError("candidate order requires at least one repeat")
    values = tuple(int(candidate) for candidate in candidates)
    return [
        values[offset:] + values[:offset]
        for offset in range(repeats)
    ]


def select_grid_with_effect_band(
    candidates: Sequence[Mapping[str, Any]],
    *,
    production_grid: int,
) -> dict[str, Any]:
    """Select a grid only when its gain clears measured median dispersion."""
    accepted = [
        candidate
        for candidate in candidates
        if candidate.get("status") == "accepted"
    ]
    if not accepted:
        raise ValueError("grid selection requires an accepted candidate")
    defaults = [
        candidate
        for candidate in accepted
        if int(candidate["grid_ctas"]) == production_grid
    ]
    if len(defaults) != 1:
        raise ValueError("production grid must be accepted exactly once")
    default = defaults[0]
    fastest = min(
        accepted,
        key=lambda candidate: float(
            candidate["median_of_repeat_medians_ms"]
        ),
    )
    effect_band = max(
        float(default["median_dispersion_ms"]),
        float(fastest["median_dispersion_ms"]),
    )
    improvement = (
        float(default["median_of_repeat_medians_ms"])
        - float(fastest["median_of_repeat_medians_ms"])
    )
    if int(fastest["grid_ctas"]) == production_grid:
        winner = default
        reason = "production default has the lowest measured median"
        recommended_non_default = False
    elif improvement > effect_band:
        winner = fastest
        reason = "non-default improvement exceeds effect band"
        recommended_non_default = True
    else:
        winner = default
        reason = "non-default improvement is inside effect band"
        recommended_non_default = False
    return {
        "winner_grid_ctas": int(winner["grid_ctas"]),
        "fastest_measured_grid_ctas": int(fastest["grid_ctas"]),
        "production_grid_ctas": production_grid,
        "minimum_effect_band_ms": effect_band,
        "fastest_improvement_over_production_ms": improvement,
        "recommended_non_default": recommended_non_default,
        "reason": reason,
    }


__all__ = [
    "TimingEvent",
    "geometric_mean",
    "percentile",
    "rank_max_samples",
    "replay_samples",
    "rotating_candidate_orders",
    "select_grid_with_effect_band",
    "summarize_rank_max",
]
