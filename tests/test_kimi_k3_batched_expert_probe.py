"""CPU contracts for the isolated m128x8x32 routed-expert probe."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_temporary_agent_instrumentation_has_been_removed() -> None:
    root = Path(__file__).parents[1]
    sources = (
        root / "benchmarks" / "kimi_k3_batched_expert_probe.py",
        root / "benchmarks" / "kimi_k3_route_finalize_probe.py",
        root / "benchmarks" / "kimi_k3_decode_runtime.py",
        root / "benchmarks" / "compare_kimi_k3_frameworks.py",
        root / "modal_app.py",
        root / "tests" / "kimi_k3_decode_support.py",
        root / "tests" / "test_kimi_k3_decode.py",
    )

    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "/opt/cursor/logs/debug.log" not in text
        assert "agent log" not in text
        assert "hypothesisId" not in text


def test_decode_launch_gate_retries_empty_rank_trace_collectively() -> None:
    source = (
        Path(__file__).parents[1] / "tests" / "kimi_k3_decode_support.py"
    ).read_text(encoding="utf-8")
    profiled = source.split("def profiled_kernel_names(", 1)[1].split(
        "\ndef ", 1
    )[0]

    assert "for _ in range(2):" in profiled
    assert "dist.all_reduce(missed_trace" in profiled
    assert "if not bool(missed_trace.item()):" in profiled


def test_rejected_candidate_switching_has_been_removed() -> None:
    root = Path(__file__).parents[1]
    assert not (root / "benchmarks" / "kimi_k3_grouped_pipeline.py").exists()
    assert not (root / "benchmarks" / "kimi_k3_gate_up_grouping.py").exists()

    sources = (
        root / "benchmarks" / "kimi_k3_decode_runtime.py",
        root / "modal_app.py",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "MOK_KIMI_K3_ENABLE_GROUPED_PIPELINE" not in text
        assert "_kimi_k3_decode_set_grouped_pipeline" not in text
        assert "bench_kimi_k3_grouped_pipeline" not in text
        assert "MOK_KIMI_K3_ENABLE_GATE_UP_GROUPING" not in text
        assert "MOK_KIMI_K3_ENABLE_GATE_UP_DOWN_PIPELINE" not in text
        assert "_kimi_k3_decode_set_gate_up_group_size" not in text
        assert "_kimi_k3_decode_set_gate_up_down_pipeline" not in text
        assert "bench_kimi_k3_gate_up_grouping" not in text


def test_probe_retains_layout_as_evidence_for_a_compound_pipeline() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_batched_expert_probe"
    )

    decision = probe.integration_decision(layout_validated=True)

    assert decision["layout_validated"] is True
    assert decision["standalone_integration_candidate"] is False
    assert decision["next_design"] == "persistent_multi_unit_staged_pipeline"
    assert decision["preserve_single_launch"] is True


def test_probe_rows_are_exactly_the_native_token_columns() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_batched_expert_probe"
    )

    assert probe.PROBE_ROWS == (1, 2, 4, 8)


def test_probe_exposes_rows_one_single_launch_diagnostics() -> None:
    source = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "kimi_k3_batched_expert_probe.py"
    ).read_text(encoding="utf-8")

    assert "def run_focused(" in source
    assert '"setup", "baseline", "candidate", "both"' in source
    assert '"--focus-rows"' in source
    assert '"--focus-variant"' in source


def test_measurement_requires_a_gain_outside_repeat_dispersion() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_batched_expert_probe"
    )
    numerical = {
        "finite": True,
        "relative_l1": 0.0,
        "cosine_similarity": 1.0,
        "max_abs": 0.0,
    }

    accepted = probe.evaluate_row(
        baseline_repeat_medians=[1.000, 1.002, 0.998],
        candidate_repeat_medians=[0.950, 0.952, 0.948],
        numerical=numerical,
    )
    inside_noise = probe.evaluate_row(
        baseline_repeat_medians=[1.000, 1.003, 0.999],
        candidate_repeat_medians=[0.999, 1.001, 0.998],
        numerical=numerical,
    )

    assert accepted["passed"] is True
    assert accepted["measurably_faster"] is True
    assert accepted["improvement_ms"] > accepted["effect_band_ms"]
    assert inside_noise["passed"] is False
    assert inside_noise["measurably_faster"] is False


def test_measurement_rejects_a_numerically_wrong_candidate() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_batched_expert_probe"
    )

    row = probe.evaluate_row(
        baseline_repeat_medians=[1.0, 1.0],
        candidate_repeat_medians=[0.5, 0.5],
        numerical={
            "finite": True,
            "relative_l1": 0.06,
            "cosine_similarity": 0.998,
            "max_abs": 1.1,
        },
    )

    assert row["numerically_correct"] is False
    assert row["passed"] is False


def test_m16_route_shape_has_no_same_expert_pair_to_pack() -> None:
    routes = importlib.import_module("benchmarks.kimi_k3_decode_inputs")
    probe = importlib.import_module(
        "benchmarks.kimi_k3_batched_expert_probe"
    )
    assignments = routes.route_assignments(16, 0)

    coverage = probe.same_expert_batching_coverage(assignments)

    assert coverage == {
        "assignments": 256,
        "distinct_experts": 256,
        "expert_row_histogram": {"1": 256},
        "assignments_in_multirow_experts": 0,
        "multirow_assignment_fraction": pytest.approx(0.0),
        "maximum_rows_per_expert": 1,
    }


def test_modal_exposes_a_focused_single_b300_probe() -> None:
    source = (Path(__file__).parents[1] / "modal_app.py").read_text(
        encoding="utf-8"
    )

    assert "def bench_kimi_k3_batched_expert_probe(" in source
    assert "def diagnose_kimi_k3_batched_expert_probe(" in source
    assert "def batched_expert_probe(" in source
    assert "def batched_expert_diagnostic(" in source
    assert '"benchmarks.kimi_k3_batched_expert_probe"' in source
    assert '"compute-sanitizer"' in source
    assert 'gpu="B300"' in source
