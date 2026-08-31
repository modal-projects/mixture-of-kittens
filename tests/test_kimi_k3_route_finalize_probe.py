"""CPU regression contracts for the routed-down baseline probe."""

from __future__ import annotations

import importlib


def test_timing_extrema_use_the_rank_max_sample_series() -> None:
    timing_utils = importlib.import_module("benchmarks.kimi_k3_timing")
    samples = [float(sample) for sample in range(1000)]
    timing = {
        "geomean_ms": 367.6954247709637,
        "median_ms": 499.5,
        "p90_ms": 899.1,
        "p99_ms": 989.01,
        "rank_max_samples_ms": samples,
        "sample_count": 1000,
    }

    assert timing_utils.timing_extrema_ms(timing) == (0.0, 999.0)
