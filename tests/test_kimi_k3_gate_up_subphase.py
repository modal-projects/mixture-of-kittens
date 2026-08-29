"""Contracts for measured routed gate/up subphase experiments."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SUBPHASE_NAMES = (
    "weight_global_load",
    "weight_shared_store_swizzle",
    "activation_stage",
    "scale_stage_copy",
    "sync_tma_tmem_wait",
    "queue_claim",
    "unit_setup",
    "units",
)


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_gate_up_subphase_trace_reuses_dead_router_score_scratch() -> None:
    types = _source("csrc/kimi_k3_decode/types.cuh")

    names = types.split("kGateUpSubphaseNames[] = {", 1)[1].split(
        "};", 1
    )[0]
    assert tuple(f'"{name}"' for name in SUBPHASE_NAMES) == tuple(
        line.strip().rstrip(",")
        for line in names.splitlines()
        if line.strip()
    )
    assert "kGateUpSubphaseTraceBytes = kRouterScoreBytes" in types
    assert "kGateUpSubphaseTraceEnd <= SCRATCH_BYTES" in types
    assert "SCRATCH_BYTES == 8111104" in types


def test_gate_up_stage_is_partitioned_without_replacing_parent_clocks() -> None:
    expert = _source("csrc/kimi_k3_decode/expert_mxfp4.cuh")
    persistent = _source("csrc/kimi_k3_decode/persistent_kernel.cuh")

    for clock in (
        "kGateUpWeightGlobalLoad",
        "kGateUpWeightSharedStoreSwizzle",
        "kGateUpActivationStage",
        "kGateUpScaleStageCopy",
        "kGateUpSyncTmaTmemWait",
        "kGateUpUnitSetup",
        "kGateUpUnits",
    ):
        assert f"add_gate_up_subphase({clock}" in expert
    assert "add_gate_up_subphase(kGateUpQueueClaim" in persistent
    assert "clocks.lap(kClockRoutedGateUpStage" in expert
    assert "clocks.lap(kClockRoutedGateUpMma" in expert
    assert persistent.count("routed_gate_up_unit(") == 1


def test_subphase_benchmark_requires_five_independent_m16_measurements() -> None:
    benchmark = importlib.import_module(
        "benchmarks.kimi_k3_gate_up_subphase"
    )

    assert benchmark.TOKENS == 16
    assert benchmark.SAMPLE_COUNT == 1000
    assert benchmark.REPEATS == 5
    assert benchmark.SUBPHASE_NAMES == SUBPHASE_NAMES


def test_subphase_summary_uses_per_sample_critical_cta_not_cta_sums() -> None:
    benchmark = importlib.import_module(
        "benchmarks.kimi_k3_gate_up_subphase"
    )
    samples = [
        {
            "weight_global_load": 100,
            "weight_shared_store_swizzle": 40,
            "activation_stage": 10,
            "scale_stage_copy": 20,
            "sync_tma_tmem_wait": 30,
            "queue_claim": 5,
            "unit_setup": 15,
            "units": 3,
            "critical_path": 220,
        },
        {
            "weight_global_load": 120,
            "weight_shared_store_swizzle": 44,
            "activation_stage": 12,
            "scale_stage_copy": 22,
            "sync_tma_tmem_wait": 32,
            "queue_claim": 6,
            "unit_setup": 16,
            "units": 4,
            "critical_path": 252,
        },
    ]

    summary = benchmark.summarize_subphase_samples(samples)

    assert summary["sample_count"] == 2
    assert summary["aggregation"] == "rank-max of each launch's CTA maxima"
    assert summary["dominant_subphase"] == "weight_global_load"
    assert summary["critical_path"]["p50"] == pytest.approx(236.0)
    assert summary["subphases"]["weight_global_load"]["p50"] == pytest.approx(
        110.0
    )
    assert summary["subphases"]["units"]["p50"] == pytest.approx(3.5)


def test_latency_measurement_is_outside_the_profile_context() -> None:
    source = _source("benchmarks/kimi_k3_gate_up_subphase.py")
    latency = source.split("def measure_latency_repeats(", 1)[1].split(
        "\ndef ", 1
    )[0]
    phases = source.split("def _measure_subphase_repeat(", 1)[1].split(
        "\ndef _init_distributed(", 1
    )[0]

    assert "phase_profiling" not in latency
    assert "_kimi_k3_decode_phase_profile()" in latency
    assert "phase_profiling" in phases
    assert "sample_count=sample_count" in latency
    assert "sample_count=sample_count" in phases
