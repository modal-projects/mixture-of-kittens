"""CPU regression contracts for the routed-down baseline probe."""

from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).parents[1]


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


def test_option_a_is_private_fp32_route_major_storage() -> None:
    types = (ROOT / "csrc/kimi_k3_decode/types.cuh").read_text()
    kernel = (ROOT / "csrc/kimi_k3_decode/persistent_kernel.cuh").read_text()
    bindings = (ROOT / "csrc/bindings.cu").read_text()

    assert "kRouteFinalizeScratchBytes = 33813248" in types
    assert "float *route_down" in types
    assert "int *token_slot_assignments" in types
    assert "int *token_output_group_ready" in types
    assert "_kimi_k3_route_finalize_workspace_bytes" in bindings
    assert "_kimi_k3_route_finalize" in bindings
    assert "launch_persistent<false, false>" in kernel
    assert "launch_persistent<true, false>" in kernel
    assert "launch_route_finalize" in kernel


def test_route_finalize_records_assignments_and_orders_producers_first() -> None:
    router = (ROOT / "csrc/kimi_k3_decode/router.cuh").read_text()
    grouped = (
        ROOT / "csrc/kimi_k3_decode/expert_mxfp4_grouped.cuh"
    ).read_text()
    kernel = (ROOT / "csrc/kimi_k3_decode/persistent_kernel.cuh").read_text()

    assert "token_slot_assignments[token * kTopK + lane] = position" in router
    assert "store_grouped_route_down" in grouped
    assert "publish_route_group" in grouped
    assert "finalize_route_group" in grouped
    assert "const int finalize_begin = shared_units + routed_producer_units" in kernel
    assert "kErrorPersistentRouteFinalizeReadiness" in kernel
    assert "finalize_route_group" in kernel


def test_candidate_probe_contains_every_required_gate() -> None:
    probe = (
        ROOT / "benchmarks/kimi_k3_route_finalize_probe.py"
    ).read_text()
    modal = (ROOT / "modal_app.py").read_text()

    assert "DEFAULT_SAMPLES = 1000" in probe
    assert "DEFAULT_REPEATS = 5" in probe
    assert "PRIMARY_MODES = (\"maximally_disjoint\",)" in probe
    assert "PROFILE_TOKENS = (16, 32, 128)" in probe
    assert "runtime.decode_reference" in probe
    assert "runtime.assert_decode_close" in probe
    assert "profiled_kernel_names" in probe
    assert "recorded_allocator_events" in probe
    assert "graph.replay()" in probe
    assert "minimum_m16_improvement_pct=8.0" in probe
    assert "minimum_m128_improvement_pct=0.0" in probe
    assert "sanitize_kimi_k3_route_finalize" in modal
