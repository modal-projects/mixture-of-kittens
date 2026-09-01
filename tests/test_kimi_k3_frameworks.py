"""CPU-only contracts for how the Kimi K3 backend comparison is set up.

What a comparison run is pinned to and what it measures: the framework
manifest's images, models and TP size, the adapter's view of the native TP8
expert layout, the versions a run records, the way per-rank samples merge into
one row, and the performance gates over those rows.

The numerical gates and the combined verdict are in
``test_kimi_k3_frameworks_gates.py``; the captured-graph router, the phase
accounting, and what a run reports are in
``test_kimi_k3_frameworks_capture.py``. None of the three needs a GPU.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]


def _compare():
    return importlib.import_module("benchmarks.compare_kimi_k3_frameworks")


def _rows(medians, p99s):
    rows = []
    for backend, values in medians.items():
        for tokens, median in values.items():
            rows.append(
                {
                    "backend": backend,
                    "mode": "block16",
                    "tokens": tokens,
                    "median_ms": median,
                    "p99_ms": p99s[backend][tokens],
                }
            )
    return rows


def _uniform(value):
    return {tokens: value for tokens in range(16, 129, 16)}


def test_framework_manifest_pins_images_models_and_tp8() -> None:
    compare = _compare()
    manifest = compare.load_framework_manifest()

    assert manifest["vllm"]["image"] == "vllm/vllm-openai:kimi-k3"
    assert manifest["sglang"]["image"] == "lmsysorg/sglang:kimi-k3"
    for framework in ("vllm", "sglang"):
        entry = manifest[framework]
        assert entry["model"] == "moonshotai/Kimi-K3"
        assert entry["tensor_parallel_size"] == 8
        assert entry["image_digest"].startswith("sha256:")
        assert entry["image_amd64_digest"].startswith("sha256:")
    assert manifest["sglang"]["moe_runner_backend"] == "flashinfer_mxfp4"
    assert manifest["dflash"]["model"] == "modal-labs/Kimi-K3-DFlash"
    assert manifest["dflash"]["block_sizes"] == [8, 16]
    assert manifest["gpu"] == "B300:8"
    assert manifest["recorded_distributions"][:2] == ["torch", "triton"]


def test_framework_manifest_file_matches_the_loader() -> None:
    compare = _compare()
    path = REPO_ROOT / "benchmarks" / "framework_manifest.json"

    assert json.loads(path.read_text()) == compare.load_framework_manifest()


def test_framework_manifest_rejects_a_missing_pin(tmp_path: Path) -> None:
    compare = _compare()
    manifest = compare.load_framework_manifest()
    del manifest["vllm"]["image_digest"]
    path = tmp_path / "framework_manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="image_digest"):
        compare.load_framework_manifest(path)


def test_framework_manifest_rejects_a_non_tp8_pin(tmp_path: Path) -> None:
    compare = _compare()
    manifest = compare.load_framework_manifest()
    manifest["sglang"]["tensor_parallel_size"] = 4
    path = tmp_path / "framework_manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        compare.load_framework_manifest(path)


def test_adapter_shape_map_matches_the_native_tp8_expert_layout() -> None:
    compare = _compare()
    shapes = compare.adapter_weight_shapes()

    assert shapes["w13_weight"] == (896, 768, 1792)
    assert shapes["w13_weight_scale"] == (896, 768, 112)
    assert shapes["w2_weight"] == (896, 3584, 192)
    assert shapes["w2_weight_scale"] == (896, 3584, 12)
    assert shapes["gate_weight"] == (896, 7168)
    assert shapes["gate_correction_bias"] == (896,)
    assert shapes["shared_gate_up_proj"] == (1536, 7168)
    assert shapes["shared_down_proj"] == (7168, 768)
    assert shapes["routed_expert_down_proj"] == (3584, 7168)
    assert shapes["routed_expert_up_proj"] == (7168, 3584)
    assert shapes["routed_expert_norm"] == (3584,)


def test_adapter_shape_map_places_gate_before_up_in_the_fused_row_block() -> None:
    compare = _compare()
    plan = compare.fused_gate_up_plan()

    assert plan["w13_row_order"] == ["w1_gate", "w3_up"]
    assert plan["w13_rows_per_half"] == 384
    assert plan["shared_row_order"] == ["gate", "up"]
    assert plan["shared_rows_per_half"] == 768


def test_version_capture_records_every_required_distribution() -> None:
    compare = _compare()
    installed = {"torch": "2.13.0+cu130", "flashinfer-python": "0.6.15"}
    captured = compare.capture_versions(
        ["torch", "flashinfer-python", "absent-package"],
        resolver=installed.get,
    )

    assert captured["torch"] == "2.13.0+cu130"
    assert captured["flashinfer-python"] == "0.6.15"
    assert captured["absent-package"] == "not-installed"


def test_version_capture_refuses_an_unpinned_torch() -> None:
    compare = _compare()

    with pytest.raises(ValueError, match="torch"):
        compare.capture_versions(["torch"], resolver=lambda name: None)


def test_sample_merge_takes_per_iteration_rank_maxima_and_summarizes() -> None:
    compare = _compare()
    merged = compare.merge_backend_samples(
        backend="mok",
        mode="block16",
        tokens=16,
        rank_samples=[
            [1.0, 4.0, 9.0, 16.0],
            [0.5, 5.0, 8.0, 15.0],
        ],
    )

    assert merged["backend"] == "mok"
    assert merged["mode"] == "block16"
    assert merged["tokens"] == 16
    assert merged["requests"] == 1
    assert merged["rank_max_samples_ms"] == [1.0, 5.0, 9.0, 16.0]
    assert merged["median_ms"] == 7.0
    assert merged["p90_ms"] == pytest.approx(13.9)
    assert merged["p99_ms"] == pytest.approx(15.79)
    assert merged["geomean_ms"] == pytest.approx((1.0 * 5.0 * 9.0 * 16.0) ** 0.25)
    assert merged["sample_count"] == 4


def test_sample_merge_rejects_ragged_rank_sample_counts() -> None:
    compare = _compare()

    with pytest.raises(ValueError, match="same number"):
        compare.merge_backend_samples(
            backend="mok",
            mode="block8",
            tokens=8,
            rank_samples=[[1.0, 2.0], [1.0]],
        )




def test_performance_gates_pass_when_the_custom_kernel_leads() -> None:
    compare = _compare()
    rows = _rows(
        {"mok": _uniform(1.0), "vllm": _uniform(1.4), "sglang": _uniform(1.2)},
        {"mok": _uniform(1.1), "vllm": _uniform(1.5), "sglang": _uniform(1.3)},
    )
    gates = compare.evaluate_performance_gates(rows)

    assert gates["passed"] is True
    assert gates["concurrency1_median"]["passed"] is True
    assert gates["concurrency1_median"]["custom_median_ms"] == 1.0
    assert gates["block16_geomean"]["passed"] is True
    assert gates["block16_geomean"]["faster_baseline"] == "sglang"
    assert gates["block16_p99"]["passed"] is True
    assert gates["block16_p99"]["limit_ratio"] == 1.10


def test_performance_gates_fail_the_concurrency_one_median_tie() -> None:
    compare = _compare()
    medians = {"mok": _uniform(1.2), "vllm": _uniform(1.4), "sglang": _uniform(1.2)}
    rows = _rows(
        medians,
        {"mok": _uniform(1.2), "vllm": _uniform(1.5), "sglang": _uniform(1.3)},
    )
    gates = compare.evaluate_performance_gates(rows)

    assert gates["passed"] is False
    assert gates["concurrency1_median"]["passed"] is False
    assert gates["concurrency1_median"]["slower_than"] == ["sglang"]


def test_performance_gates_fail_a_geomean_regression() -> None:
    compare = _compare()
    mok = dict(_uniform(1.0))
    mok[128] = 8.0
    rows = _rows(
        {"mok": mok, "vllm": _uniform(1.4), "sglang": _uniform(1.2)},
        {"mok": _uniform(1.1), "vllm": _uniform(1.5), "sglang": _uniform(1.3)},
    )
    gates = compare.evaluate_performance_gates(rows)

    assert gates["passed"] is False
    assert gates["block16_geomean"]["passed"] is False
    assert gates["block16_geomean"]["custom_geomean_ms"] > 1.2


def test_performance_gates_fail_a_p99_above_the_ten_percent_allowance() -> None:
    compare = _compare()
    p99 = dict(_uniform(1.1))
    p99[64] = 1.45
    rows = _rows(
        {"mok": _uniform(1.0), "vllm": _uniform(1.4), "sglang": _uniform(1.2)},
        {"mok": p99, "vllm": _uniform(1.5), "sglang": _uniform(1.3)},
    )
    gates = compare.evaluate_performance_gates(rows)

    assert gates["passed"] is False
    assert gates["block16_p99"]["passed"] is False
    assert [row["tokens"] for row in gates["block16_p99"]["violations"]] == [64]


def test_performance_gates_require_every_block16_shape_from_all_backends() -> None:
    compare = _compare()
    rows = _rows(
        {"mok": _uniform(1.0), "vllm": _uniform(1.4), "sglang": _uniform(1.2)},
        {"mok": _uniform(1.1), "vllm": _uniform(1.5), "sglang": _uniform(1.3)},
    )
    incomplete = [row for row in rows if not (row["backend"] == "vllm" and row["tokens"] == 96)]

    with pytest.raises(ValueError, match="missing"):
        compare.evaluate_performance_gates(incomplete)


def test_phase_profile_is_archived_for_every_measured_mode() -> None:
    compare = _compare()

    assert "phase_profile.json" in compare.ARTIFACT_FILES
    for modes in (["block8"], ["block16"], ["block8", "block16"]):
        assert "phase_profile.json" in compare.comparison_artifact_files(modes)
