"""CPU contracts for the benchmark-only Kimi K3 m128xN tail probe."""

from __future__ import annotations

import importlib
from pathlib import Path


def _probe():
    return importlib.import_module("benchmarks.kimi_k3_tail_m128n_probe")


def test_candidate_maps_output_channels_to_m_and_tokens_to_n() -> None:
    probe = _probe()

    assert probe.TOKEN_COUNTS == (16, 32, 128)
    assert probe.candidate_plan(16) == {
        "mma_m": 128,
        "mma_n": 16,
        "mma_k": 64,
        "output_tiles": 7,
        "token_tiles": 1,
        "shard_ctas": 7,
        "role_ctas": 40,
    }
    assert probe.candidate_plan(32)["mma_n"] == 32
    assert probe.candidate_plan(32)["shard_ctas"] == 7
    assert probe.candidate_plan(128)["token_tiles"] == 4
    assert probe.candidate_plan(128)["shard_ctas"] == 28
    assert probe.candidate_plan(128)["role_ctas"] == 61


def test_tail_gate_requires_material_wall_phase_numerical_and_resource_gains(
) -> None:
    probe = _probe()
    numerical = {
        "finite": True,
        "relative_l1": 0.0,
        "cosine_similarity": 1.0,
        "max_abs": 0.0,
    }
    resources = {
        "active_blocks_per_sm": 1,
        "registers_per_thread": 200,
        "local_bytes": 0,
        "stack_bytes": 0,
        "resident_role_ctas": True,
    }

    accepted = probe.evaluate_token(
        baseline_repeat_medians_us=[100.0, 101.0, 99.0],
        candidate_repeat_medians_us=[75.0, 76.0, 74.0],
        baseline_shard_mma_us=64.0,
        candidate_shard_mma_us=40.0,
        numerical=numerical,
        resources=resources,
    )
    small_wall_gain = probe.evaluate_token(
        baseline_repeat_medians_us=[100.0, 101.0, 99.0],
        candidate_repeat_medians_us=[93.0, 94.0, 92.0],
        baseline_shard_mma_us=64.0,
        candidate_shard_mma_us=40.0,
        numerical=numerical,
        resources=resources,
    )

    assert accepted["passed"] is True
    assert accepted["material_tail_wall_gain"] is True
    assert accepted["phase_gate_passed"] is True
    assert accepted["numerical_gate_passed"] is True
    assert accepted["resource_gate_passed"] is True
    assert small_wall_gain["passed"] is False
    assert small_wall_gain["material_tail_wall_gain"] is False


def test_probe_never_integrates_the_candidate_automatically() -> None:
    probe = _probe()

    decision = probe.integration_decision(
        [{"tokens": tokens, "passed": True} for tokens in probe.TOKEN_COUNTS]
    )

    assert decision["eligible_for_integration_review"] is True
    assert decision["integrated"] is False
    assert decision["preserve_single_launch"] is True
    assert decision["split_k"] == "deferred"


def test_probe_source_contains_phase_and_single_launch_gates() -> None:
    root = Path(__file__).parents[1]
    source = (
        root / "benchmarks" / "kimi_k3_tail_m128n_probe.py"
    ).read_text(encoding="utf-8")

    assert "_kimi_k3_tail_m128n_probe" in source
    assert "_kimi_k3_tail_m128n_resource_metadata" in source
    assert "runtime.tail_profiling()" in source
    assert "kernel_count" in source
    assert "candidate_shard_mma_us" in source


def test_isolated_shard_harness_preserves_candidate_contract() -> None:
    root = Path(__file__).parents[1]
    header = (
        root / "csrc" / "kimi_k3_decode" / "tail_m128n_probe.cuh"
    ).read_text(encoding="utf-8")
    benchmark = (
        root / "benchmarks" / "kimi_k3_tail_m128n_probe.py"
    ).read_text(encoding="utf-8")

    assert "_kimi_k3_tail_m128n_shard_probe" in benchmark
    assert "run_isolated_shard" in benchmark
    assert "(tokens, LATENT)" in benchmark
    assert "(HIDDEN, LATENT)" in benchmark
    assert "(tokens, SHARD_COLUMNS)" in benchmark
    assert "torch.bfloat16" in benchmark
    assert "reference.float()" in benchmark

    assert "kimi_k3_tail_m128n_shard_probe_kernel" in header
    assert "shard_tensor_m128n<TOKEN_TILE_N, false>" in header
    assert "store_octet(" in header
    isolated_kernel = header.split(
        "void kimi_k3_tail_m128n_shard_probe_kernel", 1
    )[1].split("inline bool guard_enabled", 1)[0]
    for forbidden in (
        "coordinate_ranks(",
        "drain_ranks(",
        "publish_generation(",
        "wait_for_generation(",
        "multimem_",
    ):
        assert forbidden not in isolated_kernel


def test_production_persistent_kernel_does_not_call_the_candidate() -> None:
    root = Path(__file__).parents[1]
    persistent = (
        root / "csrc" / "kimi_k3_decode" / "persistent_kernel.cuh"
    ).read_text(encoding="utf-8")

    assert "shard_tensor_m128n" not in persistent
    assert "kimi_k3_tail_m128n_probe" not in persistent


def test_modal_exposes_focused_8xb300_benchmark_and_memcheck() -> None:
    source = (Path(__file__).parents[1] / "modal_app.py").read_text(
        encoding="utf-8"
    )

    assert "def bench_kimi_k3_tail_m128n_probe(" in source
    assert "def diagnose_kimi_k3_tail_m128n_probe(" in source
    assert "def tail_m128n_probe(" in source
    assert "def tail_m128n_diagnostic(" in source
    assert "def diagnose_kimi_k3_tail_m128n_shard_probe(" in source
    assert "def tail_m128n_shard_diagnostic(" in source
    isolated = source.split(
        "def diagnose_kimi_k3_tail_m128n_shard_probe(", 1
    )[1].split("@app.", 1)[0]
    assert 'gpu="B300"' in source
    assert "torch.distributed.run" not in isolated
    assert "--isolated-shard-tokens" in isolated
    assert '"compute-sanitizer"' in source
    assert 'gpu="B300:8"' in source
