"""CPU-only contracts for the Kimi K3 serving-backend comparison."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]


def _compare():
    return importlib.import_module("benchmarks.compare_kimi_k3_frameworks")


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


# --------------------------------------------------------------------------
# Numerical gates
# --------------------------------------------------------------------------


def _parity_row(**overrides):
    """One archived parity row, at the values the passing runs measured."""
    row = {
        "mode": "block16",
        "tokens": 16,
        "pool_index": 0,
        "router": {
            "expert_ids_match": True,
            "expert_id_mismatch_count": 0,
            "router_weight_max_abs": 7.45e-09,
            "router_weight_mean_abs": 1.0e-10,
            "topk": 16,
            "distinct_experts": 256,
        },
        "custom_vs_native": {
            "relative_l1": 0.02160,
            "cosine_similarity": 0.999766,
            "max_abs": 0.0859,
            "finite": True,
        },
        "custom_vs_reference": {
            "relative_l1": 0.00503,
            "cosine_similarity": 0.999985,
            "max_abs": 0.0215,
            "finite": True,
        },
        "native_vs_reference": {
            "relative_l1": 0.02176,
            "cosine_similarity": 0.999761,
            "max_abs": 0.0859,
            "finite": True,
        },
        "routed_latent_vs_reference": {
            "relative_l1": 0.0,
            "cosine_similarity": 1.0,
            "max_abs": 0.0,
        },
        "shared_output_vs_reference": {
            "relative_l1": 1.0e-05,
            "cosine_similarity": 0.9999999,
            "max_abs": 4.88e-04,
        },
    }
    for path, value in overrides.items():
        head, _, tail = path.partition("__")
        if tail:
            row[head] = {**row[head], tail: value}
        else:
            row[head] = value
    return row


def test_numerical_gates_pass_the_official_reference_row() -> None:
    compare = _compare()
    gates = compare.evaluate_numerical_gates([_parity_row()])

    assert gates["passed"] is True
    assert gates["gated_comparison"] == "custom_vs_reference"
    assert gates["row_count"] == 1
    assert gates["violations"] == []
    checks = gates["rows"][0]["checks"]
    assert set(checks) == {
        "finite",
        "relative_l1",
        "cosine_similarity",
        "max_abs",
        "router_expert_ids_exact",
        "router_weight_max_abs",
    }
    assert all(check["passed"] for check in checks.values())


def test_numerical_gates_fail_a_custom_reference_cosine_below_the_floor() -> None:
    compare = _compare()
    gates = compare.evaluate_numerical_gates(
        [_parity_row(custom_vs_reference__cosine_similarity=0.9985)]
    )

    assert gates["passed"] is False
    assert [violation["check"] for violation in gates["violations"]] == [
        "cosine_similarity"
    ]
    assert gates["violations"][0]["value"] == 0.9985
    assert gates["violations"][0]["limit"] == 0.999


def test_numerical_gates_fail_a_relative_l1_above_the_tolerance() -> None:
    compare = _compare()
    gates = compare.evaluate_numerical_gates(
        [_parity_row(custom_vs_reference__relative_l1=0.051)]
    )

    assert gates["passed"] is False
    assert [violation["check"] for violation in gates["violations"]] == [
        "relative_l1"
    ]


def test_numerical_gates_fail_a_max_abs_above_the_tolerance() -> None:
    compare = _compare()
    gates = compare.evaluate_numerical_gates(
        [_parity_row(custom_vs_reference__max_abs=1.5)]
    )

    assert gates["passed"] is False
    assert [violation["check"] for violation in gates["violations"]] == [
        "max_abs"
    ]


def test_numerical_gates_fail_a_non_finite_custom_output() -> None:
    compare = _compare()
    gates = compare.evaluate_numerical_gates(
        [_parity_row(custom_vs_reference__finite=False)]
    )

    assert gates["passed"] is False
    assert "finite" in {violation["check"] for violation in gates["violations"]}


def test_numerical_gates_fail_an_inexact_router_selection() -> None:
    compare = _compare()
    gates = compare.evaluate_numerical_gates(
        [
            _parity_row(
                router__expert_ids_match=False,
                router__expert_id_mismatch_count=3,
            )
        ]
    )

    assert gates["passed"] is False
    assert [violation["check"] for violation in gates["violations"]] == [
        "router_expert_ids_exact"
    ]


def test_numerical_gates_fail_a_router_weight_outside_one_e_minus_five() -> None:
    compare = _compare()
    gates = compare.evaluate_numerical_gates(
        [_parity_row(router__router_weight_max_abs=2.0e-05)]
    )

    assert gates["passed"] is False
    assert [violation["check"] for violation in gates["violations"]] == [
        "router_weight_max_abs"
    ]
    assert gates["violations"][0]["limit"] == 1e-05


def test_numerical_gates_report_a_native_miss_without_failing_the_custom_gate() -> None:
    """SGLang's own MXFP4 rounding is diagnostic, not a custom-kernel failure.

    The measured SGLang archive puts ``native_vs_reference`` at cosine
    ``0.998888`` and ``custom_vs_native`` at ``0.998890``, both under the
    ``0.999`` floor, while the custom kernel sits at ``0.999985`` against the
    official reference. The gate follows the amended design: the custom kernel
    is graded against the reference and the native layer's distance from it is
    recorded rather than imitated.
    """
    compare = _compare()
    gates = compare.evaluate_numerical_gates(
        [
            _parity_row(
                custom_vs_native__cosine_similarity=0.998890,
                custom_vs_native__relative_l1=0.04697,
                custom_vs_native__max_abs=0.1855,
                native_vs_reference__cosine_similarity=0.998888,
                native_vs_reference__relative_l1=0.04706,
                native_vs_reference__max_abs=0.1855,
            )
        ]
    )

    assert gates["passed"] is True
    assert gates["violations"] == []
    diagnostics = gates["diagnostics"]
    assert diagnostics["native_vs_reference"]["min_cosine_similarity"] == 0.998888
    assert diagnostics["custom_vs_native"]["min_cosine_similarity"] == 0.998890
    assert diagnostics["native_vs_reference"]["rows_outside_gate_tolerances"] == 1
    assert diagnostics["custom_vs_native"]["rows_outside_gate_tolerances"] == 1
    assert diagnostics["custom_vs_reference"]["rows_outside_gate_tolerances"] == 0


def test_numerical_gates_require_the_gated_comparison_on_every_row() -> None:
    compare = _compare()
    row = _parity_row()
    del row["custom_vs_reference"]

    with pytest.raises(ValueError, match="custom_vs_reference"):
        compare.evaluate_numerical_gates([row])


def test_numerical_gates_are_archived_next_to_the_performance_gates() -> None:
    compare = _compare()

    assert "numerical_gates.json" in compare.ARTIFACT_FILES
    for modes in (["block8"], ["block16"], ["block8", "block16"]):
        assert "numerical_gates.json" in compare.comparison_artifact_files(modes)


# --------------------------------------------------------------------------
# Registry digest binding
# --------------------------------------------------------------------------


def test_pinned_image_reference_binds_the_manifest_digest() -> None:
    compare = _compare()
    manifest = compare.load_framework_manifest()

    assert compare.pinned_image_reference("vllm") == (
        "vllm/vllm-openai@" + manifest["vllm"]["image_digest"]
    )
    assert compare.pinned_image_reference("sglang") == (
        "lmsysorg/sglang@" + manifest["sglang"]["image_digest"]
    )
    for framework in ("vllm", "sglang"):
        reference = compare.pinned_image_reference(framework)
        assert "@sha256:" in reference
        assert ":kimi-k3" not in reference


def test_pinned_image_reference_rejects_an_unknown_framework() -> None:
    compare = _compare()

    with pytest.raises(ValueError, match="unknown framework"):
        compare.pinned_image_reference("tensorrt")


def test_effective_image_reference_defaults_to_the_pin(monkeypatch) -> None:
    compare = _compare()
    monkeypatch.delenv("MOK_COMPARISON_IMAGE_REF", raising=False)

    assert compare.effective_image_reference("vllm") == (
        compare.pinned_image_reference("vllm")
    )


def test_effective_image_reference_rejects_a_build_off_the_pin(monkeypatch) -> None:
    """The image the container actually booted has to be the pinned digest."""
    compare = _compare()
    monkeypatch.setenv("MOK_COMPARISON_IMAGE_REF", "vllm/vllm-openai:kimi-k3")

    with pytest.raises(ValueError, match="does not match the pinned digest"):
        compare.effective_image_reference("vllm")


def test_modal_derives_comparison_images_from_the_pinned_digest() -> None:
    source = (REPO_ROOT / "modal_app.py").read_text()

    assert "pinned_image_reference" in source
    # The mutable tags may only appear inside the manifest, which is the single
    # place the digest is resolved from.
    assert '"vllm/vllm-openai:kimi-k3"' not in source
    assert '"lmsysorg/sglang:kimi-k3"' not in source
    assert "MOK_COMPARISON_IMAGE_REF" in source


# --------------------------------------------------------------------------
# Combined verdict
# --------------------------------------------------------------------------


def _write_archive(directory: Path, framework: str, *, medians, p99s, parity):
    compare = _compare()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"framework": framework, "benchmark": compare.BENCHMARK})
    )
    rows = []
    for backend in ("mok", framework):
        for tokens in range(16, 129, 16):
            rows.append(
                {
                    "backend": backend,
                    "mode": "block16",
                    "tokens": tokens,
                    "median_ms": medians[backend],
                    "p90_ms": medians[backend],
                    "p99_ms": p99s[backend],
                    "geomean_ms": medians[backend],
                }
            )
    (directory / "latency_block16.json").write_text(
        json.dumps({"rows": rows})
    )
    (directory / "parity.json").write_text(
        json.dumps({"framework": framework, "rows": parity})
    )
    return directory


def test_combined_archive_persists_both_gate_families(tmp_path: Path) -> None:
    compare = _compare()
    directories = [
        _write_archive(
            tmp_path / "vllm",
            "vllm",
            medians={"mok": 1.0, "vllm": 1.4},
            p99s={"mok": 1.1, "vllm": 1.5},
            parity=[_parity_row()],
        ),
        _write_archive(
            tmp_path / "sglang",
            "sglang",
            medians={"mok": 1.0, "sglang": 1.2},
            p99s={"mok": 1.1, "sglang": 1.3},
            parity=[_parity_row(pool_index=1)],
        ),
    ]
    summary = compare.combine_archives(directories, tmp_path / "combined")

    assert (tmp_path / "combined" / "combined_performance_gates.json").is_file()
    assert (tmp_path / "combined" / "combined_numerical_gates.json").is_file()
    assert (tmp_path / "combined" / "combined_gates.json").is_file()
    assert summary["numerical_gates"]["row_count"] == 2
    assert summary["numerical_gates"]["passed"] is True
    assert summary["performance_gates"]["passed"] is True
    assert summary["passed"] is True


def test_combined_verdict_fails_when_the_performance_gates_fail(
    tmp_path: Path,
) -> None:
    compare = _compare()
    directories = [
        _write_archive(
            tmp_path / "vllm",
            "vllm",
            medians={"mok": 9.0, "vllm": 1.4},
            p99s={"mok": 9.1, "vllm": 1.5},
            parity=[_parity_row()],
        ),
        _write_archive(
            tmp_path / "sglang",
            "sglang",
            medians={"mok": 9.0, "sglang": 1.2},
            p99s={"mok": 9.1, "sglang": 1.3},
            parity=[_parity_row(pool_index=1)],
        ),
    ]
    summary = compare.combine_archives(directories, tmp_path / "combined")

    assert summary["performance_gates"]["passed"] is False
    assert summary["numerical_gates"]["passed"] is True
    assert summary["passed"] is False
    # The artifacts are written before the verdict is signalled, so a failing
    # run still leaves everything a reader needs behind.
    persisted = json.loads(
        (tmp_path / "combined" / "combined_gates.json").read_text()
    )
    assert persisted["passed"] is False


def test_combine_cli_exits_non_zero_on_a_failing_gate(tmp_path: Path) -> None:
    import subprocess
    import sys

    for framework, native in (("vllm", 1.4), ("sglang", 1.2)):
        _write_archive(
            tmp_path / framework,
            framework,
            medians={"mok": 9.0, framework: native},
            p99s={"mok": 9.1, framework: native + 0.1},
            parity=[_parity_row()],
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.compare_kimi_k3_frameworks",
            "--combine",
            str(tmp_path / "vllm"),
            str(tmp_path / "sglang"),
            "--output-dir",
            str(tmp_path / "combined"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout
    assert (tmp_path / "combined" / "combined_gates.json").is_file()


def test_combined_gates_record_an_incomplete_sweep_as_not_passed(
    tmp_path: Path,
) -> None:
    """A shortened probe cannot be read as a passing gate."""
    compare = _compare()
    directory = _write_archive(
        tmp_path / "vllm",
        "vllm",
        medians={"mok": 1.0, "vllm": 1.4},
        p99s={"mok": 1.1, "vllm": 1.5},
        parity=[_parity_row()],
    )
    summary = compare.combine_archives([directory], tmp_path / "combined")

    assert summary["performance_gates"]["status"] == "incomplete"
    assert summary["performance_gates"]["passed"] is False
    assert summary["passed"] is False


# --------------------------------------------------------------------------
# Effective CLI overrides and event priming
# --------------------------------------------------------------------------


def test_dry_run_manifest_records_the_effective_cli_overrides(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.compare_kimi_k3_frameworks",
            "--dry-run",
            "--framework",
            "sglang",
            "--output-dir",
            str(tmp_path),
            "--warmup-count",
            "7",
            "--sample-count",
            "11",
            "--pool-size",
            "2",
            "--modes",
            "block16",
            "--tokens",
            "16,32",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["warmup_count"] == 7
    assert manifest["sample_count"] == 11
    assert manifest["graph_pool_size"] == 2
    assert manifest["shape_groups"] == {"block16": [16, 32]}
    assert manifest["routing"]["pool_entries"] == 2
    assert manifest["image_reference"] == _compare().pinned_image_reference(
        "sglang"
    )


class _FakeEvent:
    """A CUDA-event stand-in whose readings come from an explicit clock."""

    def __init__(self, clock: list[float]) -> None:
        self._clock = clock
        self.stamp: float | None = None

    def record(self) -> None:
        self.stamp = self._clock[0]

    def elapsed_time(self, other: _FakeEvent) -> float:
        assert self.stamp is not None and other.stamp is not None
        return other.stamp - self.stamp


def test_replay_samples_discards_a_primed_event_pair() -> None:
    """The first event pair's one-time cost must not land in a sample.

    Creating and recording the very first ``torch.cuda.Event`` of a process
    pays a driver initialization that shows up entirely in sample zero, which
    is then the p99 of a thousand-sample series. One pair is recorded and
    thrown away before the persisted ones.
    """
    from benchmarks.kimi_k3_timing import replay_samples

    clock = [0.0]
    replays: list[int] = []

    def replay(iteration: int) -> None:
        replays.append(iteration)
        clock[0] += 50.0 if len(replays) == 3 else 1.0

    samples = replay_samples(
        replay,
        warmup_count=2,
        sample_count=4,
        event_factory=lambda: _FakeEvent(clock),
        synchronize=lambda: None,
    )

    assert samples == [1.0, 1.0, 1.0, 1.0]
    assert replays == [0, 1, 2, 3, 4, 5, 6]


def test_replay_samples_rejects_a_non_positive_count() -> None:
    from benchmarks.kimi_k3_timing import replay_samples

    with pytest.raises(ValueError, match="positive"):
        replay_samples(
            lambda iteration: None,
            warmup_count=1,
            sample_count=0,
            event_factory=lambda: _FakeEvent([0.0]),
            synchronize=lambda: None,
        )


def test_phase_cycle_summary_ranks_regions_by_their_accounted_share() -> None:
    """The split counters describe a region, they do not add to the total.

    ``routed_gate_up_stage`` and ``routed_gate_up_mma`` are measured inside
    ``routed_gate_up``, so counting all three would report a total larger than
    the cycles the kernel actually spent and would understate every share.
    """
    compare = _compare()
    summary = compare.summarize_phase_cycles(
        {
            "router_score": 100,
            "routed_gate_up": 600,
            "routed_gate_up_stage": 500,
            "routed_gate_up_mma": 90,
            "routed_down": 300,
            "routed_down_stage": 250,
            "routed_down_mma": 40,
        }
    )

    assert summary["accounted_cycles"] == 1000
    assert summary["share_of_accounted"]["routed_gate_up"] == pytest.approx(0.6)
    assert summary["share_of_accounted"]["routed_down"] == pytest.approx(0.3)
    assert summary["ranked"][0] == ("routed_gate_up", 600)
    assert summary["ranked"][1] == ("routed_down", 300)
    assert summary["dominant_region"] == "routed_gate_up"
    assert summary["dominant_share"] == pytest.approx(0.6)


def test_phase_cycle_summary_tolerates_an_unprofiled_launch() -> None:
    compare = _compare()
    summary = compare.summarize_phase_cycles({"router_score": 0, "tail": 0})

    assert summary["accounted_cycles"] == 0
    assert summary["dominant_region"] is None
    assert summary["share_of_accounted"] == {"router_score": 0.0, "tail": 0.0}


def test_phase_clock_names_match_the_kernel_scratch_band() -> None:
    """The reader names the counters; the kernel decides how many there are."""
    compare = _compare()
    source = (REPO_ROOT / "csrc" / "kimi_k3_decode" / "types.cuh").read_text()
    names = [
        match.group(1)
        for match in __import__("re").finditer(
            r'^\s{4}"([a-z0-9_]+)",$',
            source.split("kPhaseClockNames[] = {", 1)[1].split("};", 1)[0],
            __import__("re").MULTILINE,
        )
    ]

    assert names == list(compare.PHASE_CLOCK_NAMES)
    assert names[0] == "queue_clear"
    assert "routed_gate_up_stage" in names
    assert "routed_gate_up_mma" in names


def test_dry_run_writes_the_complete_comparison_manifest(tmp_path: Path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.compare_kimi_k3_frameworks",
            "--dry-run",
            "--framework",
            "vllm",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert json.loads(result.stdout) == manifest
    assert manifest["benchmark"] == "kimi_k3_framework_comparison"
    assert manifest["dry_run"] is True
    assert manifest["framework"] == "vllm"
    assert manifest["backends"] == ["mok", "vllm"]
    assert manifest["tp_size"] == 8
    assert manifest["warmup_count"] == 500
    assert manifest["sample_count"] == 1000
    assert manifest["shape_groups"] == {
        "block8": list(range(8, 65, 8)),
        "block16": list(range(16, 129, 16)),
    }
    assert manifest["numerical_tolerances"] == {
        "relative_l1": 0.05,
        "cosine_similarity": 0.999,
        "max_abs": 1.0,
    }
    assert manifest["performance_gates"]["p99_limit_ratio"] == 1.10
    assert manifest["artifact_files"] == list(compare_artifact_files())


def compare_artifact_files():
    return _compare().ARTIFACT_FILES


def test_base_package_import_does_not_require_a_serving_framework() -> None:
    compare = _compare()
    source = (
        REPO_ROOT / "benchmarks" / "compare_kimi_k3_frameworks.py"
    ).read_text()
    module_level = source.split("def ", 1)[0]

    for forbidden in ("import vllm", "import sglang", "import flashinfer"):
        assert forbidden not in module_level
    assert compare.ADAPTER_MODULES == {
        "vllm": "benchmarks.frameworks.vllm_kimi_k3",
        "sglang": "benchmarks.frameworks.sglang_kimi_k3",
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.frameworks.vllm_kimi_k3",
        "benchmarks.frameworks.sglang_kimi_k3",
    ],
)
def test_adapter_modules_keep_framework_imports_out_of_module_scope(
    module_name: str,
) -> None:
    relative = module_name.replace(".", "/") + ".py"
    source = (REPO_ROOT / relative).read_text()
    module_level = source.split("\nclass ", 1)[0].split("\ndef ", 1)[0]

    for forbidden in ("import vllm", "import sglang", "import flashinfer"):
        assert forbidden not in module_level


def test_modal_exposes_the_two_framework_comparison_entrypoints() -> None:
    source = (REPO_ROOT / "modal_app.py").read_text()

    assert "def compare_vllm(" in source
    assert "def compare_sglang(" in source
    assert source.count("framework_comparison_image(") >= 2
    # The expected artifact set is sourced from the comparison module rather
    # than restated here, so a new artifact cannot pass the driver's
    # completeness check while the Modal function still rejects it.
    assert "comparison_artifact_files" in source
    assert "COMPARISON_ARTIFACT_FILES" not in source
