"""CPU-only contracts for the projection-first scheduling A/B."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _function_body(text: str, signature: str) -> str:
    start = text.index(signature)
    depth = 0
    for offset in range(text.index("{", start), len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise AssertionError(f"{signature} is never closed")


def test_projection_first_is_a_guarded_runtime_order_not_a_second_kernel() -> None:
    persistent = _source("csrc/kimi_k3_decode/persistent_kernel.cuh")
    entrypoint = _source("csrc/kimi_k3_decode/entrypoints.cuh")
    kernel = _function_body(
        persistent,
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "benchmark_projection_first_enabled()" in entrypoint
    assert "const int projection_first" in kernel
    assert kernel.count("claim_unit(") == 1
    assert persistent.count("kimi_k3_decode_persistent_kernel<") == 3
    assert "MOK_KIMI_K3_ENABLE_GRID_TUNING" in persistent


def test_candidate_issues_projection_and_shared_work_before_scores() -> None:
    kernel = _function_body(
        _source("csrc/kimi_k3_decode/persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )
    route = kernel.split("// Phase 1:", 1)[1].split("// Phase 2:", 1)[0]
    gate_up = kernel.split("// Phase 3:", 1)[1].split("// Phase 4:", 1)[0]

    assert "projection_first ? projection_units + shared_units : 0" in route
    assert "projection_first ? shared_units : 0" in route
    assert route.index("const int projection_unit") < route.index(
        "const int score_unit ="
    )
    assert "const int shared_units = projection_first ? 0" in gate_up
    assert route.count("shared_experts::project_tensor(") == 1
    assert gate_up.count("shared_experts::project_tensor(") == 1


def test_profile_splits_every_barrier_and_reports_route_makespan() -> None:
    types = _source("csrc/kimi_k3_decode/types.cuh")
    kernel = _function_body(
        _source("csrc/kimi_k3_decode/persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )

    names = (
        "profile_clear_barrier",
        "clear_barrier",
        "route_latent_barrier",
        "route_latent_makespan",
        "assignment_quantize_barrier",
        "down_barrier",
        "publish_barrier",
    )
    for name in names:
        assert f'"{name}"' in types
    assert "clocks.maximum(kClockRouteLatentMakespan" in kernel
    assert "kClockGridBarrier" not in kernel


def test_ab_orders_alternate_to_balance_temporal_drift() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_projection_first_ab"
    )

    assert probe.interleaved_orders(5) == [
        ("score_first", "projection_first"),
        ("projection_first", "score_first"),
        ("score_first", "projection_first"),
        ("projection_first", "score_first"),
        ("score_first", "projection_first"),
    ]


def test_material_gain_requires_effect_band_and_no_p99_regression() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_projection_first_ab"
    )
    repeats = {
        "score_first": [
            {"median_ms": 1.050, "p99_ms": 1.100},
            {"median_ms": 1.052, "p99_ms": 1.102},
            {"median_ms": 1.051, "p99_ms": 1.101},
        ],
        "projection_first": [
            {"median_ms": 0.995, "p99_ms": 1.040},
            {"median_ms": 0.996, "p99_ms": 1.041},
            {"median_ms": 0.994, "p99_ms": 1.039},
        ],
    }

    verdict = probe.ab_verdict(repeats)

    assert verdict["material_improvement"] is True
    assert verdict["p99_regression"] is False
    assert verdict["integrate"] is True

    repeats["projection_first"][0]["p99_ms"] = 1.200
    assert probe.ab_verdict(repeats)["integrate"] is False


def test_ab_verdict_rejects_incomplete_repeats() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_projection_first_ab"
    )

    with pytest.raises(ValueError, match="same nonzero repeat count"):
        probe.ab_verdict(
            {
                "score_first": [{"median_ms": 1.0, "p99_ms": 1.1}],
                "projection_first": [],
            }
        )
