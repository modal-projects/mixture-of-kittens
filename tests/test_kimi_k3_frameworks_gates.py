"""CPU-only contracts for the Kimi K3 comparison's numerical gates and verdict.

A comparison is only evidence if it fails closed, so most of what is held here
is failure: a parity row below the cosine floor, above the relative-L1 or
max-abs tolerance, non-finite, or with an inexact router selection; a sweep
that is missing a shape; an archive that is missing a gate family. The passing
row is one case among them.

How a run is pinned, and its performance gates, are in
``test_kimi_k3_frameworks.py``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from . import modal_sources

REPO_ROOT = Path(__file__).parents[1]


def _compare():
    return importlib.import_module("benchmarks.compare_kimi_k3_frameworks")


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


# Numerical gates
# --------------------------------------------------------------------------



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


def test_effective_image_reference_rejects_a_build_off_the_pin(monkeypatch) -> None:
    """The image the container actually booted has to be the pinned digest."""
    compare = _compare()
    monkeypatch.setenv("MOK_COMPARISON_IMAGE_REF", "vllm/vllm-openai:kimi-k3")

    with pytest.raises(ValueError, match="does not match the pinned digest"):
        compare.effective_image_reference("vllm")


def test_modal_derives_comparison_images_from_the_pinned_digest() -> None:
    source = modal_sources.read()

    assert "pinned_image_reference" in source
    # The mutable tags may only appear inside the manifest, which is the single
    # place the digest is resolved from.
    assert '"vllm/vllm-openai:kimi-k3"' not in source
    assert '"lmsysorg/sglang:kimi-k3"' not in source
    assert "MOK_COMPARISON_IMAGE_REF" in source


# --------------------------------------------------------------------------
# Combined verdict
# --------------------------------------------------------------------------



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
