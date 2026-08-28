"""CPU-only contracts for the Kimi K3 decode benchmark."""

from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def test_percentile_uses_linear_interpolation() -> None:
    timing = importlib.import_module("benchmarks.kimi_k3_timing")
    samples = [10.0, 1.0, 4.0, 7.0, 2.0, 9.0, 3.0, 8.0, 6.0, 5.0]

    assert timing.percentile(samples, 0.0) == 1.0
    assert timing.percentile(samples, 0.5) == 5.5
    assert timing.percentile(samples, 0.9) == pytest.approx(9.1)
    assert timing.percentile(samples, 0.99) == pytest.approx(9.91)
    assert timing.percentile(samples, 1.0) == 10.0


def test_percentile_rejects_empty_samples_and_invalid_quantiles() -> None:
    timing = importlib.import_module("benchmarks.kimi_k3_timing")
    with pytest.raises(ValueError, match="at least one"):
        timing.percentile([], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        timing.percentile([1.0], -0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        timing.percentile([1.0], 1.1)


def test_geometric_mean_is_computed_in_log_space() -> None:
    timing = importlib.import_module("benchmarks.kimi_k3_timing")

    assert timing.geometric_mean([1.0, 4.0, 16.0]) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="positive"):
        timing.geometric_mean([1.0, 0.0])


def test_rank_max_samples_preserves_iteration_alignment() -> None:
    timing = importlib.import_module("benchmarks.kimi_k3_timing")
    rank_samples = [
        [1.0, 9.0, 3.0, 4.0],
        [2.0, 8.0, 5.0, 1.0],
        [0.5, 7.0, 6.0, 10.0],
    ]

    assert timing.rank_max_samples(rank_samples) == [2.0, 9.0, 6.0, 10.0]
    with pytest.raises(ValueError, match="same number"):
        timing.rank_max_samples([[1.0], [1.0, 2.0]])


def test_summary_is_over_per_iteration_rank_maxima() -> None:
    timing = importlib.import_module("benchmarks.kimi_k3_timing")
    summary = timing.summarize_rank_max(
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
    assert manifest["pool_policy"]["graph_pool_size"] == 4
    assert manifest["pool_policy"]["routing"] == (
        "zero-forcing-bias deterministic disjoint 16-expert blocks"
    )


@pytest.mark.parametrize("tokens", [1, 8, 16, 56, 64, 128])
def test_route_assignments_occupy_the_realistic_expert_bound(
    tokens: int,
) -> None:
    routes = importlib.import_module("benchmarks.kimi_k3_decode_inputs")
    pools = [
        routes.route_assignments(tokens, pool_index)
        for pool_index in range(routes.GRAPH_POOL_SIZE)
    ]

    expected = min(16 * tokens, 896)
    for assignments in pools:
        assert len(assignments) == tokens
        assert all(len(set(token_routes)) == 16 for token_routes in assignments)
        assert len({expert for token in assignments for expert in token}) == expected
    assert pools[0] != pools[1]
    expected_pool_coverage = min(expected * routes.GRAPH_POOL_SIZE, 896)
    assert len(
        {
            expert
            for assignments in pools
            for token in assignments
            for expert in token
        }
    ) == expected_pool_coverage


def test_route_metadata_is_explicit_per_replay_and_pool() -> None:
    routes = importlib.import_module("benchmarks.kimi_k3_decode_inputs")
    metadata = routes.route_metadata(
        tokens=16,
        expert_weight_bytes=2_193_408,
        l2_cache_bytes=132_644_864,
    )

    assert metadata["distinct_experts_per_replay"] == 256
    assert metadata["pool_wide_distinct_experts"] == 896
    assert metadata["routed_queue_units_per_replay"] == {
        "gate_up": 768,
        "down": 7168,
        "total": 7936,
    }
    assert metadata["routed_expert_working_set_bytes_per_replay"] == 561_512_448
    assert metadata["routed_expert_working_set_exceeds_l2_per_replay"] is True
    assert len(metadata["route_assignments_by_pool_entry"]) == 4


def test_tuning_order_rotates_and_default_wins_inside_dispersion_band() -> None:
    timing = importlib.import_module("benchmarks.kimi_k3_timing")

    assert timing.rotating_candidate_orders((64, 96, 128, 148), 3) == [
        (64, 96, 128, 148),
        (96, 128, 148, 64),
        (128, 148, 64, 96),
    ]
    winner = timing.select_grid_with_effect_band(
        [
            {
                "grid_ctas": 128,
                "status": "accepted",
                "median_of_repeat_medians_ms": 1.0000,
                "median_dispersion_ms": 0.0010,
            },
            {
                "grid_ctas": 148,
                "status": "accepted",
                "median_of_repeat_medians_ms": 1.0005,
                "median_dispersion_ms": 0.0015,
            },
        ],
        production_grid=148,
    )

    assert winner["winner_grid_ctas"] == 148
    assert winner["recommended_non_default"] is False
    assert winner["minimum_effect_band_ms"] == pytest.approx(0.0015)
    assert winner["reason"] == "non-default improvement is inside effect band"


def test_tuning_selects_a_non_default_only_outside_dispersion_band() -> None:
    timing = importlib.import_module("benchmarks.kimi_k3_timing")
    winner = timing.select_grid_with_effect_band(
        [
            {
                "grid_ctas": 128,
                "status": "accepted",
                "median_of_repeat_medians_ms": 0.990,
                "median_dispersion_ms": 0.001,
            },
            {
                "grid_ctas": 148,
                "status": "accepted",
                "median_of_repeat_medians_ms": 1.000,
                "median_dispersion_ms": 0.002,
            },
        ],
        production_grid=148,
    )

    assert winner["winner_grid_ctas"] == 128
    assert winner["recommended_non_default"] is True
    assert winner["minimum_effect_band_ms"] == pytest.approx(0.002)


def test_archive_bytes_ignore_source_metadata(tmp_path: Path) -> None:
    artifacts = importlib.import_module("benchmarks.kimi_k3_artifacts")
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text('{"stable": true}\n')
    first = artifacts.reproducible_tar_bytes(source)

    os.chmod(source / "manifest.json", 0o600)
    os.utime(source / "manifest.json", (1_800_000_000, 1_800_000_000))
    second = artifacts.reproducible_tar_bytes(source)

    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r") as archive:
        member = archive.getmember("manifest.json")
    assert member.mtime == 0
    assert member.uid == member.gid == 0
    assert member.uname == member.gname == ""
    assert member.mode == 0o644


def test_modal_exposes_exact_tp8_b300_decode_entrypoints() -> None:
    source = (Path(__file__).parents[1] / "modal_app.py").read_text()
    build_files = source.split("BUILD_FILES =", 1)[1].split(
        "REMOTE_ROOT =", 1
    )[0]
    builder = source.split("def build_image", 1)[1].split("IMAGE =", 1)[0]

    assert "def test_kimi_k3_decode(" in source
    assert "def bench_kimi_k3_decode(" in source
    assert source.count('gpu="B300:8"') >= 2
    assert '"--nproc-per-node=8"' in source
    assert "MOK_GIT_SHA" not in builder
    assert '"modal_app.py"' not in build_files
    assert builder.index(".run_commands(") < builder.index(
        '"modal_app.py"',
        builder.index(".run_commands("),
    )
    assert "return first_archive" in source
