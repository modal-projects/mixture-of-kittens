"""Manifest and artifact writers for the Kimi K3 decode benchmark."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from benchmarks.kimi_k3_decode_inputs import (
    GRAPH_POOL_SIZE,
    GRID_CANDIDATES,
    TOPK,
)

RAW_DECODE_SHAPES = tuple(range(1, 9))
BLOCK8_SHAPES = tuple(range(8, 65, 8))
BLOCK16_SHAPES = tuple(range(16, 129, 16))
SHAPE_GROUPS = {
    "raw_decode": RAW_DECODE_SHAPES,
    "block8": BLOCK8_SHAPES,
    "block16": BLOCK16_SHAPES,
}
TP_SIZE = 8
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
TUNING_REPEATS = 3
P99_TO_MEDIAN_LIMIT = 2.0

ARTIFACT_FILES = (
    "manifest.json",
    "latency_raw_decode.json",
    "latency_raw_decode.csv",
    "latency_block8.json",
    "latency_block8.csv",
    "latency_block16.json",
    "latency_block16.csv",
    "correctness.json",
    "workspace_stats.json",
    "tuning.json",
)


def _git_sha() -> str:
    configured = os.environ.get("MOK_GIT_SHA")
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def build_manifest(*, dry_run: bool) -> dict[str, Any]:
    return {
        "benchmark": "kimi_k3_decode",
        "dry_run": dry_run,
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "mok_version": "unavailable" if dry_run else None,
        "tp_size": TP_SIZE,
        "warmup_count": WARMUP_COUNT,
        "sample_count": SAMPLE_COUNT,
        "tuning_repeats": TUNING_REPEATS,
        "shape_groups": {
            name: list(shapes) for name, shapes in SHAPE_GROUPS.items()
        },
        "grid_candidates": list(GRID_CANDIDATES),
        "primary_tuning_point": {"mode": "block16", "tokens": 16},
        "p99_to_median_limit": P99_TO_MEDIAN_LIMIT,
        "launch_count": 1,
        "timing": {
            "unit": "milliseconds",
            "operation": "CUDA Graph replay only",
            "rank_reduction": "maximum per iteration across all eight ranks",
            "percentile_method": "R-7 linear interpolation",
        },
        "pool_policy": {
            "routing": (
                "zero-forcing-bias deterministic disjoint 16-expert blocks"
            ),
            "working_set": "strictly_greater_than_l2",
            "graph_pool_size": GRAPH_POOL_SIZE,
            "pool_entries": "rotated and permuted expert blocks",
            "per_replay_occupancy": "min(16 * tokens, 896)",
            "copies_timed": False,
            "graph_capture_timed": False,
        },
        "cluster_candidates": [1, 2],
        "model": {
            "hidden_size": 7168,
            "latent_size": 3584,
            "routed_intermediate_size": 3072,
            "shared_intermediate_size": 6144,
            "num_experts": 896,
            "topk": TOPK,
            "max_tokens": 128,
            "activation_dtype": "bfloat16",
            "routed_weight_format": (
                "MXFP4 E2M1 with E8M0 group-32 scales"
            ),
        },
        "input_construction": (
            "one-hot token directions and sparse deterministic router weights"
        ),
        "correction_bias": (
            "natural additive K3 semantics with fixed values in [-0.015, 0.015]"
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_dry_run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(dry_run=True)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def write_latency_table(
    output_dir: Path,
    filename_stem: str,
    rows: list[dict[str, Any]],
) -> None:
    write_json(
        output_dir / f"{filename_stem}.json",
        {
            "benchmark": "kimi_k3_decode",
            "unit": "milliseconds",
            "rows": rows,
        },
    )
    columns = [
        "mode",
        "tokens",
        "batch_size",
        "grid_ctas",
        "cluster_size",
        "launch_count",
        "graph_pool_size",
        "distinct_experts_per_replay",
        "route_assignments_by_pool_entry",
        "routed_queue_units_per_replay",
        "pool_wide_distinct_experts",
        "routed_expert_working_set_bytes_per_replay",
        "routed_expert_working_set_exceeds_l2_per_replay",
        "pool_wide_routed_expert_working_set_bytes",
        "pool_wide_routed_expert_working_set_exceeds_l2",
        "l2_cache_bytes",
        "warmup_count",
        "sample_count",
        "median_ms",
        "p90_ms",
        "p99_ms",
        "geomean_ms",
    ]
    with (output_dir / f"{filename_stem}.csv").open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            values = {column: row[column] for column in columns}
            for column in (
                "route_assignments_by_pool_entry",
                "routed_queue_units_per_replay",
            ):
                values[column] = json.dumps(
                    values[column],
                    separators=(",", ":"),
                )
            writer.writerow(values)


def gpu_clocks() -> list[dict[str, str]] | dict[str, str]:
    command = [
        "nvidia-smi",
        (
            "--query-gpu=index,name,clocks.current.sm,clocks.current.memory,"
            "clocks.max.sm,clocks.max.memory"
        ),
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"status": "unavailable", "reason": str(error)}
    fields = (
        "index",
        "name",
        "current_sm_mhz",
        "current_memory_mhz",
        "max_sm_mhz",
        "max_memory_mhz",
    )
    return [
        dict(
            zip(
                fields,
                (value.strip() for value in line.split(",")),
                strict=True,
            )
        )
        for line in lines
    ]


__all__ = [
    "ARTIFACT_FILES",
    "P99_TO_MEDIAN_LIMIT",
    "SAMPLE_COUNT",
    "SHAPE_GROUPS",
    "TP_SIZE",
    "TUNING_REPEATS",
    "WARMUP_COUNT",
    "build_manifest",
    "gpu_clocks",
    "write_dry_run",
    "write_json",
    "write_latency_table",
]
