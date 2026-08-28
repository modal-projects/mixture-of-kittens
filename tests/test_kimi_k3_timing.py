"""CPU-only contracts for the Kimi K3 decode benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.kimi_k3_timing import (
    geometric_mean,
    percentile,
    rank_max_samples,
    summarize_rank_max,
)


def test_percentile_uses_linear_interpolation() -> None:
    samples = [10.0, 1.0, 4.0, 7.0, 2.0, 9.0, 3.0, 8.0, 6.0, 5.0]

    assert percentile(samples, 0.0) == 1.0
    assert percentile(samples, 0.5) == 5.5
    assert percentile(samples, 0.9) == pytest.approx(9.1)
    assert percentile(samples, 0.99) == pytest.approx(9.91)
    assert percentile(samples, 1.0) == 10.0


def test_percentile_rejects_empty_samples_and_invalid_quantiles() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        percentile([1.0], -0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        percentile([1.0], 1.1)


def test_geometric_mean_is_computed_in_log_space() -> None:
    assert geometric_mean([1.0, 4.0, 16.0]) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="positive"):
        geometric_mean([1.0, 0.0])


def test_rank_max_samples_preserves_iteration_alignment() -> None:
    rank_samples = [
        [1.0, 9.0, 3.0, 4.0],
        [2.0, 8.0, 5.0, 1.0],
        [0.5, 7.0, 6.0, 10.0],
    ]

    assert rank_max_samples(rank_samples) == [2.0, 9.0, 6.0, 10.0]
    with pytest.raises(ValueError, match="same number"):
        rank_max_samples([[1.0], [1.0, 2.0]])


def test_summary_is_over_per_iteration_rank_maxima() -> None:
    summary = summarize_rank_max(
        [
            [1.0, 4.0, 9.0, 16.0],
            [0.5, 5.0, 8.0, 15.0],
        ]
    )

    assert summary == {
        "sample_count": 4,
        "median_ms": 7.0,
        "p90_ms": pytest.approx(13.9),
        "p99_ms": pytest.approx(15.79),
        "geomean_ms": pytest.approx((1.0 * 5.0 * 9.0 * 16.0) ** 0.25),
    }


def test_dry_run_emits_the_complete_shape_manifest(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.bench_kimi_k3_decode",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    manifest_path = tmp_path / "manifest.json"
    assert json.loads(result.stdout) == json.loads(manifest_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert manifest["benchmark"] == "kimi_k3_decode"
    assert manifest["dry_run"] is True
    assert manifest["tp_size"] == 8
    assert manifest["warmup_count"] == 500
    assert manifest["sample_count"] == 1000
    assert manifest["shape_groups"] == {
        "raw_decode": list(range(1, 9)),
        "block8": list(range(8, 65, 8)),
        "block16": list(range(16, 129, 16)),
    }
    assert manifest["grid_candidates"] == [64, 96, 128, 148]
    assert manifest["primary_tuning_point"] == {
        "mode": "block16",
        "tokens": 16,
    }
    assert manifest["launch_count"] == 1
    assert manifest["pool_policy"]["working_set"] == "strictly_greater_than_l2"


def test_modal_exposes_exact_tp8_b300_decode_entrypoints() -> None:
    source = (Path(__file__).parents[1] / "modal_app.py").read_text()

    assert "def test_kimi_k3_decode(" in source
    assert "def bench_kimi_k3_decode(" in source
    assert source.count('gpu="B300:8"') >= 2
    assert '"--nproc-per-node=8"' in source
    assert "return archive_path.read_bytes()" in source
