"""CPU contracts for the isolated m128x8x32 routed-expert probe."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_debug_log_creates_missing_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_batched_expert_probe"
    )
    debug_log_path = tmp_path / "missing" / "debug.log"
    monkeypatch.setattr(probe, "DEBUG_LOG_PATH", debug_log_path)

    probe._debug_log("A", "test:debug_log", "probe entry", {"rows": [1]})

    payload = json.loads(debug_log_path.read_text(encoding="utf-8"))
    assert payload["hypothesisId"] == "A"
    assert payload["message"] == "probe entry"
    assert payload["data"] == {"rows": [1]}


def test_probe_rows_are_exactly_the_native_token_columns() -> None:
    probe = importlib.import_module(
        "benchmarks.kimi_k3_batched_expert_probe"
    )

    assert probe.PROBE_ROWS == (1, 2, 4, 8)


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
    assert "def batched_expert_probe(" in source
    assert '"benchmarks.kimi_k3_batched_expert_probe"' in source
    assert 'gpu="B300"' in source
