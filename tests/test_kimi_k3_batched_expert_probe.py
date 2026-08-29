"""CPU contracts for the isolated m128x8x32 routed-expert probe."""

from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_temporary_agent_instrumentation_has_been_removed() -> None:
    root = Path(__file__).parents[1]
    sources = (
        root / "benchmarks" / "kimi_k3_batched_expert_probe.py",
        root / "benchmarks" / "compare_kimi_k3_frameworks.py",
    )

    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "/opt/cursor/logs/debug.log" not in text
        assert "agent log" not in text
        assert "hypothesisId" not in text


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


def test_grouped_down_benchmark_is_a_same_grid_m16_m128_ab() -> None:
    benchmark = importlib.import_module(
        "benchmarks.kimi_k3_grouped_pipeline"
    )
    modal_source = (
        Path(__file__).parents[1] / "modal_app.py"
    ).read_text(encoding="utf-8")

    assert benchmark.TOKENS == (16, 128)
    assert len(benchmark.VARIANTS) == 2
    assert benchmark.BASELINE.name == "baseline_148"
    assert benchmark.CANDIDATE.name == "grouped_down_148"
    assert benchmark.CANDIDATE.grouped_pipeline is True
    assert benchmark.CANDIDATE.grid_ctas == benchmark.BASELINE.grid_ctas == 148
    assert "def bench_kimi_k3_grouped_pipeline(" in modal_source
    assert "def grouped_pipeline(" in modal_source
    assert '"benchmarks.kimi_k3_grouped_pipeline"' in modal_source


def test_grouped_pipeline_checks_errors_only_after_capture_and_replay_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = importlib.import_module(
        "benchmarks.kimi_k3_grouped_pipeline"
    )
    events: list[str] = []
    capture_active = False

    class FakeGraph:
        def replay(self) -> None:
            events.append("replay")

    class FakeCapture:
        def __enter__(self) -> None:
            nonlocal capture_active
            capture_active = True
            events.append("capture-enter")

        def __exit__(self, *args: object) -> None:
            nonlocal capture_active
            capture_active = False
            events.append("capture-exit")

    @contextlib.contextmanager
    def benchmark_variant(**_: object):
        events.append("variant-enter")
        yield
        events.append("variant-exit")

    def decode_step(*_: object) -> None:
        raise AssertionError("wrapped decode_step must not run in capture")

    def decode_device_step(*_: object) -> None:
        events.append("device-capture" if capture_active else "device-eager")

    def check_decode_error(workspace: object) -> None:
        assert events[-1] == "sync"
        events.append("check")
        workspace.error_flag.item()

    def error_item() -> int:
        assert capture_active is False
        events.append("item")
        return 0

    def synchronize(_: object) -> None:
        events.append("sync")

    runtime = SimpleNamespace(
        benchmark_decode_variant=benchmark_variant,
        check_decode_error=check_decode_error,
        decode_device_step=decode_device_step,
        decode_step=decode_step,
    )
    workspace = SimpleNamespace(
        error_flag=SimpleNamespace(item=error_item),
    )
    pool = [
        SimpleNamespace(weights=object(), hidden=object()),
    ]
    monkeypatch.setattr(benchmark.torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "graph",
        lambda _: FakeCapture(),
    )
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", synchronize)

    graphs = benchmark._capture_pool(
        runtime,
        workspace,
        pool,
        benchmark.CANDIDATE,
        object(),
    )

    assert events == [
        "variant-enter",
        "device-eager",
        "sync",
        "check",
        "item",
        "capture-enter",
        "device-capture",
        "capture-exit",
        "sync",
        "check",
        "item",
        "variant-exit",
    ]

    events.clear()

    def replay_samples(replay: object, **kwargs: object) -> list[float]:
        replay(0)
        kwargs["synchronize"]()
        return [0.25]

    monkeypatch.setattr(benchmark, "replay_samples", replay_samples)

    samples = benchmark._measure(
        graphs,
        runtime_module=runtime,
        workspace=workspace,
        warmup_count=1,
        sample_count=1,
        device=object(),
    )

    assert samples == [0.25]
    assert events == ["replay", "sync", "check", "item"]


def test_grouped_pipeline_gate_requires_numerics_and_a_measured_gain() -> None:
    benchmark = importlib.import_module(
        "benchmarks.kimi_k3_grouped_pipeline"
    )
    numerical = [
        {
            "candidate_vs_reference": {
                "finite": True,
                "relative_l1": 0.01,
                "cosine_similarity": 0.9999,
                "max_abs": 0.5,
            }
        }
    ]

    accepted = benchmark.evaluate_candidate(
        baseline_repeat_medians=[1.0, 1.01, 0.99],
        candidate_repeat_medians=[0.8, 0.81, 0.79],
        numerical_rows=numerical,
    )
    wrong = benchmark.evaluate_candidate(
        baseline_repeat_medians=[1.0, 1.01, 0.99],
        candidate_repeat_medians=[0.8, 0.81, 0.79],
        numerical_rows=[
            {
                "candidate_vs_reference": {
                    **numerical[0]["candidate_vs_reference"],
                    "finite": False,
                }
            }
        ],
    )

    assert accepted["passed"] is True
    assert accepted["measurably_faster"] is True
    assert wrong["numerically_correct"] is False
    assert wrong["passed"] is False
