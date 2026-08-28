"""Pure statistics used by the TP8 Kimi K3 decode benchmark."""

from __future__ import annotations

import math
from collections.abc import Sequence


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


__all__ = [
    "geometric_mean",
    "percentile",
    "rank_max_samples",
    "summarize_rank_max",
]
