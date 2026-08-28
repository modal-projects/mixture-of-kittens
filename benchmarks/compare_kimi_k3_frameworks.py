"""Compare the one-launch Kimi K3 decode kernel with native serving backends.

The comparison runs the custom kernel and one framework's complete native Kimi
K3 sparse-MoE layer side by side, in the same process, on the same eight B300s,
against the same prepared weights and the same realistic route pool. Nothing in
this module imports vLLM, SGLang, or FlashInfer: the framework code lives behind
the adapters in :mod:`benchmarks.frameworks`, which only the derived comparison
images can import.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_decode_inputs import GRAPH_POOL_SIZE, mode_batch_size
from benchmarks.kimi_k3_timing import (
    geometric_mean,
    percentile,
    rank_max_samples,
)

BENCHMARK = "kimi_k3_framework_comparison"
CUSTOM_BACKEND = "mok"
BASELINE_BACKENDS = ("vllm", "sglang")
ADAPTER_MODULES = {
    "vllm": "benchmarks.frameworks.vllm_kimi_k3",
    "sglang": "benchmarks.frameworks.sglang_kimi_k3",
}

TP_SIZE = 8
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
BLOCK8_SHAPES = tuple(range(8, 65, 8))
BLOCK16_SHAPES = tuple(range(16, 129, 16))
SHAPE_GROUPS = {"block8": BLOCK8_SHAPES, "block16": BLOCK16_SHAPES}
GATE_SHAPES = BLOCK16_SHAPES
CONCURRENCY_ONE_TOKENS = 16
P99_LIMIT_RATIO = 1.10
NUMERICAL_TOLERANCES = {
    "relative_l1": 0.05,
    "cosine_similarity": 0.999,
    "max_abs": 1.0,
}

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
ROUTED_INTERMEDIATE_SIZE = 3072
SHARED_INTERMEDIATE_SIZE = 6144
NUM_EXPERTS = 896
TOPK = 16
MXFP4_GROUP_SIZE = 32

MANIFEST_PATH = Path(__file__).with_name("framework_manifest.json")
REQUIRED_FRAMEWORK_PINS = (
    "image",
    "image_digest",
    "image_amd64_digest",
    "model",
    "tensor_parallel_size",
)

ARTIFACT_FILES = (
    "manifest.json",
    "versions.json",
    "transformations.json",
    "parity.json",
    "route_occupancy.json",
    "latency_block8.json",
    "latency_block8.csv",
    "latency_block16.json",
    "latency_block16.csv",
    "raw_samples.json",
    "launch_traces.json",
    "phase_profile.json",
    "performance_gates.json",
)


def comparison_artifact_files(modes: Sequence[str]) -> tuple[str, ...]:
    """Return the artifacts one run writes for the modes it measures."""
    unmeasured = set(SHAPE_GROUPS) - set(modes)
    skipped = {
        f"latency_{mode}.{suffix}"
        for mode in unmeasured
        for suffix in ("json", "csv")
    }
    return tuple(name for name in ARTIFACT_FILES if name not in skipped)


# The kernel's clock64 accumulators, in `csrc/kimi_k3_decode/types.cuh` order.
# The two `_stage`/`_mma` pairs measure the inside of the routed region above
# them rather than a region of their own.
PHASE_CLOCK_NAMES = (
    "queue_clear",
    "router_score",
    "latent_project",
    "assignments",
    "latent_quantize",
    "routed_gate_up",
    "routed_gate_up_stage",
    "routed_gate_up_mma",
    "routed_down",
    "routed_down_stage",
    "routed_down_mma",
    "shared_experts",
    "grid_barrier",
    "tail",
)
PHASE_CLOCK_BREAKDOWN_SUFFIXES = ("_stage", "_mma")


def summarize_phase_cycles(cycles: Mapping[str, int]) -> dict[str, Any]:
    """Rank the kernel's accumulated regions by their share of the total.

    Only the regions that partition the launch are summed. A region's own
    breakdown counters are reported alongside their share of the same total,
    which is what makes "the staging inside routed gate/up is 83% of the whole
    launch" a statement about the launch rather than about its parent region.
    """
    accounted = sum(
        value
        for name, value in cycles.items()
        if not name.endswith(PHASE_CLOCK_BREAKDOWN_SUFFIXES)
    )
    ranked = sorted(
        (
            (name, value)
            for name, value in cycles.items()
            if not name.endswith(PHASE_CLOCK_BREAKDOWN_SUFFIXES)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "accounted_cycles": accounted,
        "share_of_accounted": {
            name: (value / accounted if accounted else 0.0)
            for name, value in cycles.items()
        },
        "ranked": ranked,
        "dominant_region": ranked[0][0] if accounted else None,
        "dominant_share": (ranked[0][1] / accounted) if accounted else 0.0,
    }


LATENCY_COLUMNS = (
    "backend",
    "mode",
    "tokens",
    "requests",
    "distinct_experts_per_replay",
    "graph_pool_size",
    "warmup_count",
    "sample_count",
    "median_ms",
    "p90_ms",
    "p99_ms",
    "geomean_ms",
)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def load_framework_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the pinned framework manifest."""
    manifest_path = MANIFEST_PATH if path is None else Path(path)
    manifest = json.loads(manifest_path.read_text())

    for framework in BASELINE_BACKENDS:
        entry = manifest.get(framework)
        if not isinstance(entry, dict):
            raise ValueError(f"{framework} manifest entry is missing")
        for key in REQUIRED_FRAMEWORK_PINS:
            if key not in entry:
                raise ValueError(f"{framework} manifest entry is missing {key}")
        for key in ("image_digest", "image_amd64_digest"):
            if not str(entry[key]).startswith("sha256:"):
                raise ValueError(
                    f"{framework} {key} must be a sha256 registry digest"
                )
        if entry["tensor_parallel_size"] != TP_SIZE:
            raise ValueError(
                f"{framework} tensor_parallel_size must be {TP_SIZE}, "
                f"got {entry['tensor_parallel_size']}"
            )
    if manifest.get("gpu") != f"B300:{TP_SIZE}":
        raise ValueError(f"manifest gpu must be B300:{TP_SIZE}")
    dflash = manifest.get("dflash")
    if not isinstance(dflash, dict) or dflash.get("block_sizes") != [8, 16]:
        raise ValueError("dflash manifest entry must pin block sizes 8 and 16")
    distributions = manifest.get("recorded_distributions")
    if not isinstance(distributions, list) or distributions[:2] != [
        "torch",
        "triton",
    ]:
        raise ValueError(
            "recorded_distributions must start with torch and triton"
        )
    return manifest


# --------------------------------------------------------------------------
# Adapter shape contract
# --------------------------------------------------------------------------


def adapter_weight_shapes() -> dict[str, tuple[int, ...]]:
    """Return the per-rank TP8 tensor shapes both native layers expose.

    The routed experts run in latent space, so their contraction extent is the
    ``3584`` latent width rather than the ``7168`` hidden width, and each rank
    owns ``3072 / 8`` routed and ``6144 / 8`` shared intermediate columns.
    """
    routed = ROUTED_INTERMEDIATE_SIZE // TP_SIZE
    shared = SHARED_INTERMEDIATE_SIZE // TP_SIZE
    return {
        "w13_weight": (NUM_EXPERTS, 2 * routed, LATENT_SIZE // 2),
        "w13_weight_scale": (
            NUM_EXPERTS,
            2 * routed,
            LATENT_SIZE // MXFP4_GROUP_SIZE,
        ),
        "w2_weight": (NUM_EXPERTS, LATENT_SIZE, routed // 2),
        "w2_weight_scale": (
            NUM_EXPERTS,
            LATENT_SIZE,
            routed // MXFP4_GROUP_SIZE,
        ),
        "gate_weight": (NUM_EXPERTS, HIDDEN_SIZE),
        "gate_correction_bias": (NUM_EXPERTS,),
        "shared_gate_up_proj": (2 * shared, HIDDEN_SIZE),
        "shared_down_proj": (HIDDEN_SIZE, shared),
        "routed_expert_down_proj": (LATENT_SIZE, HIDDEN_SIZE),
        "routed_expert_up_proj": (HIDDEN_SIZE, LATENT_SIZE),
        "routed_expert_norm": (LATENT_SIZE,),
    }


def fused_gate_up_plan() -> dict[str, Any]:
    """Describe how the separate gate and up matrices fill a fused row block.

    Both frameworks store one fused ``w13``/``gate_up`` matrix whose first half
    of rows is the gate projection and whose second half is the up projection;
    the custom kernel keeps them as two matrices. This is the only reordering
    the adapters perform on expert bytes.
    """
    return {
        "w13_row_order": ["w1_gate", "w3_up"],
        "w13_rows_per_half": ROUTED_INTERMEDIATE_SIZE // TP_SIZE,
        "shared_row_order": ["gate", "up"],
        "shared_rows_per_half": SHARED_INTERMEDIATE_SIZE // TP_SIZE,
    }


# --------------------------------------------------------------------------
# Version capture
# --------------------------------------------------------------------------


def _installed_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def capture_versions(
    names: Sequence[str],
    *,
    resolver: Callable[[str], str | None] | None = None,
) -> dict[str, str]:
    """Resolve installed distribution versions, refusing an unpinned torch."""
    lookup = _installed_version if resolver is None else resolver
    captured: dict[str, str] = {}
    for name in names:
        found = lookup(name)
        if found is None and name == "torch":
            raise ValueError(
                "torch must be installed to record a comparison version pin"
            )
        captured[name] = "not-installed" if found is None else str(found)
    return captured


# --------------------------------------------------------------------------
# Sample merge
# --------------------------------------------------------------------------


def merge_backend_samples(
    *,
    backend: str,
    mode: str,
    tokens: int,
    rank_samples: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Reduce per-rank latency samples to one rank-max series and summarize it."""
    maxima = rank_max_samples(rank_samples)
    return {
        "backend": backend,
        "mode": mode,
        "tokens": tokens,
        "requests": mode_batch_size(mode, tokens),
        "rank_max_samples_ms": maxima,
        "sample_count": len(maxima),
        "median_ms": percentile(maxima, 0.5),
        "p90_ms": percentile(maxima, 0.9),
        "p99_ms": percentile(maxima, 0.99),
        "geomean_ms": geometric_mean(maxima),
    }


def merge_latency_rows(
    row_groups: Iterable[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Combine per-archive latency rows, keeping the worst custom measurement.

    The custom kernel is measured once inside each framework image, so a
    ``(mode, tokens)`` point carries two independent custom measurements. The
    combined row keeps the slower of the two for every statistic, which can
    only make the performance gates harder to pass.
    """
    combined: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            key = (row["backend"], row["mode"], int(row["tokens"]))
            existing = combined.get(key)
            if existing is None:
                combined[key] = dict(row)
                continue
            if row["backend"] != CUSTOM_BACKEND:
                raise ValueError(
                    f"duplicate {row['backend']} measurement for {key}"
                )
            for statistic in ("median_ms", "p90_ms", "p99_ms", "geomean_ms"):
                existing[statistic] = max(
                    float(existing[statistic]),
                    float(row[statistic]),
                )
            existing.setdefault("sources", []).append(row.get("image", "unknown"))
    return [combined[key] for key in sorted(combined)]


# --------------------------------------------------------------------------
# Performance gates
# --------------------------------------------------------------------------


def _gate_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("mode") != "block16":
            continue
        index[(str(row["backend"]), int(row["tokens"]))] = row
    missing = [
        f"{backend}@{tokens}"
        for backend in (CUSTOM_BACKEND, *BASELINE_BACKENDS)
        for tokens in GATE_SHAPES
        if (backend, tokens) not in index
    ]
    if missing:
        raise ValueError(
            f"performance gates are missing block16 measurements: {missing}"
        )
    return index


def evaluate_performance_gates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the three Task 11 latency gates over the block-16 sweep."""
    index = _gate_index(rows)

    custom_at_one = float(index[(CUSTOM_BACKEND, CONCURRENCY_ONE_TOKENS)]["median_ms"])
    baseline_at_one = {
        backend: float(index[(backend, CONCURRENCY_ONE_TOKENS)]["median_ms"])
        for backend in BASELINE_BACKENDS
    }
    slower_than = sorted(
        backend
        for backend, median in baseline_at_one.items()
        if custom_at_one >= median
    )
    concurrency1 = {
        "tokens": CONCURRENCY_ONE_TOKENS,
        "requests": 1,
        "custom_median_ms": custom_at_one,
        "baseline_median_ms": baseline_at_one,
        "slower_than": slower_than,
        "passed": not slower_than,
    }

    geomeans = {
        backend: geometric_mean(
            [float(index[(backend, tokens)]["median_ms"]) for tokens in GATE_SHAPES]
        )
        for backend in (CUSTOM_BACKEND, *BASELINE_BACKENDS)
    }
    faster_baseline = min(BASELINE_BACKENDS, key=lambda name: geomeans[name])
    custom_geomean = geomeans[CUSTOM_BACKEND]
    baseline_geomean = geomeans[faster_baseline]
    block16_geomean = {
        "tokens": list(GATE_SHAPES),
        "custom_geomean_ms": custom_geomean,
        "baseline_geomean_ms": {
            backend: geomeans[backend] for backend in BASELINE_BACKENDS
        },
        "faster_baseline": faster_baseline,
        "ratio_to_faster_baseline": custom_geomean / baseline_geomean,
        "passed": custom_geomean <= baseline_geomean,
    }

    violations = []
    p99_rows = []
    for tokens in GATE_SHAPES:
        custom_p99 = float(index[(CUSTOM_BACKEND, tokens)]["p99_ms"])
        baseline_p99 = min(
            float(index[(backend, tokens)]["p99_ms"])
            for backend in BASELINE_BACKENDS
        )
        limit = P99_LIMIT_RATIO * baseline_p99
        entry = {
            "tokens": tokens,
            "custom_p99_ms": custom_p99,
            "faster_baseline_p99_ms": baseline_p99,
            "limit_ms": limit,
            "ratio": custom_p99 / baseline_p99,
            "passed": custom_p99 <= limit,
        }
        p99_rows.append(entry)
        if not entry["passed"]:
            violations.append(entry)
    block16_p99 = {
        "limit_ratio": P99_LIMIT_RATIO,
        "rows": p99_rows,
        "violations": violations,
        "passed": not violations,
    }

    return {
        "passed": (
            concurrency1["passed"]
            and block16_geomean["passed"]
            and block16_p99["passed"]
        ),
        "concurrency1_median": concurrency1,
        "block16_geomean": block16_geomean,
        "block16_p99": block16_p99,
    }


# --------------------------------------------------------------------------
# Comparison manifest
# --------------------------------------------------------------------------


def _git_sha() -> str:
    return os.environ.get("MOK_GIT_SHA", "unavailable")


def build_comparison_manifest(
    *,
    framework: str,
    dry_run: bool,
) -> dict[str, Any]:
    if framework not in ADAPTER_MODULES:
        raise ValueError(f"unknown framework {framework!r}")
    pins = load_framework_manifest()
    return {
        "benchmark": BENCHMARK,
        "dry_run": dry_run,
        "framework": framework,
        "backends": [CUSTOM_BACKEND, framework],
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu": pins["gpu"],
        "tp_size": TP_SIZE,
        "warmup_count": WARMUP_COUNT,
        "sample_count": SAMPLE_COUNT,
        "graph_pool_size": GRAPH_POOL_SIZE,
        "shape_groups": {
            name: list(shapes) for name, shapes in SHAPE_GROUPS.items()
        },
        "numerical_tolerances": dict(NUMERICAL_TOLERANCES),
        "performance_gates": {
            "concurrency1_tokens": CONCURRENCY_ONE_TOKENS,
            "geomean_tokens": list(GATE_SHAPES),
            "p99_limit_ratio": P99_LIMIT_RATIO,
        },
        "framework_pins": {
            key: pins[key] for key in ("vllm", "sglang", "dflash")
        },
        "recorded_distributions": list(pins["recorded_distributions"]),
        "routing": {
            "source": "benchmarks.kimi_k3_decode_data.build_routed_input",
            "per_replay_occupancy": "min(16 * tokens, 896)",
            "pool_entries": GRAPH_POOL_SIZE,
        },
        "timing": {
            "unit": "milliseconds",
            "operation": "CUDA Graph replay only",
            "rank_reduction": "maximum per iteration across all eight ranks",
            "percentile_method": "R-7 linear interpolation",
        },
        "adapter_weight_shapes": {
            name: list(shape) for name, shape in adapter_weight_shapes().items()
        },
        "fused_gate_up_plan": fused_gate_up_plan(),
        "artifact_files": list(ARTIFACT_FILES),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_latency_table(
    output_dir: Path,
    stem: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    import csv

    write_json(
        output_dir / f"{stem}.json",
        {"benchmark": BENCHMARK, "unit": "milliseconds", "rows": list(rows)},
    )
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LATENCY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in LATENCY_COLUMNS})


def write_dry_run(output_dir: Path, *, framework: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_comparison_manifest(framework=framework, dry_run=True)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------
# Combining two archives
# --------------------------------------------------------------------------


def combine_archives(
    directories: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge per-framework archives and evaluate the gates over both."""
    groups = []
    manifests = {}
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests[manifest["framework"]] = manifest
        rows: list[dict[str, Any]] = []
        for mode in SHAPE_GROUPS:
            payload = json.loads(
                (directory / f"latency_{mode}.json").read_text()
            )
            for row in payload["rows"]:
                rows.append({**row, "image": manifest["framework"]})
        groups.append(rows)
    combined = merge_latency_rows(groups)
    gates = evaluate_performance_gates(combined)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": BENCHMARK,
        "frameworks": sorted(manifests),
        "custom_row_policy": (
            "slower of the two per-image custom measurements at each shape"
        ),
        "rows": combined,
        "performance_gates": gates,
    }
    write_json(output_dir / "combined_performance_gates.json", summary)
    return summary


# --------------------------------------------------------------------------
# GPU driver
# --------------------------------------------------------------------------


def _init_distributed() -> tuple[int, Any]:
    import torch
    import torch.distributed as dist

    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch the comparison with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP_SIZE:
        raise RuntimeError(f"the comparison requires TP{TP_SIZE}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the comparison requires SM103 B300 GPUs")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return rank, device


def _barrier(device: Any) -> None:
    import torch
    import torch.distributed as dist

    dist.barrier(async_op=True, device_ids=[device.index]).block_current_stream()
    torch.cuda.synchronize(device)


def _numerical_stats(actual: Any, expected: Any) -> dict[str, float]:
    import torch

    left = actual.float()
    right = expected.float()
    difference = left - right
    return {
        "relative_l1": float(
            difference.abs().sum() / right.abs().sum().clamp_min(1e-12)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                left.flatten(), right.flatten(), dim=0
            )
        ),
        "max_abs": float(difference.abs().max()),
        "finite": bool(torch.isfinite(left).all()),
    }


def _gathered_rank_samples(local_samples: Sequence[float], device: Any) -> list[list[float]]:
    import torch
    import torch.distributed as dist

    local = torch.tensor(list(local_samples), dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, local)
    return [samples.cpu().tolist() for samples in gathered]


def _replay_samples(
    replay: Callable[[int], None],
    device: Any,
    *,
    warmup_count: int,
    sample_count: int,
) -> list[float]:
    import torch

    for iteration in range(warmup_count):
        replay(iteration)
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    for iteration, (start, end) in enumerate(zip(starts, ends, strict=True)):
        start.record()
        replay(iteration)
        end.record()
    torch.cuda.synchronize(device)
    return [
        start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
    ]


def _kernel_trace(call: Callable[[], Any]) -> list[str]:
    import tempfile

    import torch

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trace.json")
        profiler.export_chrome_trace(path)
        with open(path, encoding="utf-8") as handle:
            trace = json.load(handle)
    return [
        event["name"]
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
    ]


def _phase_profile(
    workspace: Any,
    pool: Sequence[Any],
    runtime: Any,
    device: Any,
) -> dict[str, Any]:
    """Accumulated clock64 cycles per kernel region for one replayed shape.

    Collected outside the timed section and outside the captured graphs: the
    accumulators cost atomics that the measured launches must not pay, and a
    graph would have recorded whichever launch it captured anyway.
    """
    import torch

    with runtime.phase_profiling():
        for entry in pool:
            runtime.decode_step(workspace, entry.weights, entry.hidden)
        torch.cuda.synchronize(device)
        cycles = runtime.phase_clock_cycles(workspace)
    return {
        "replays": len(pool),
        "cycles": cycles,
        **summarize_phase_cycles(cycles),
    }


def _run_gpu(
    framework: str,
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    shape_groups: Mapping[str, Sequence[int]],
    pool_size: int,
) -> None:
    import torch

    from benchmarks import kimi_k3_decode_data as data
    from benchmarks import kimi_k3_decode_runtime as runtime
    from mok import kimi_k3 as kimi

    adapter_module = importlib.import_module(ADAPTER_MODULES[framework])

    rank, device = _init_distributed()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = data.build_weights(device, rank)
    workspace = kimi.get_kimi_k3_decode_workspace(torch.distributed.group.WORLD, device=device)

    adapter = adapter_module.build_adapter(
        device=device,
        tp_rank=rank,
        tp_size=TP_SIZE,
        weights=weights,
    )

    parity: list[dict[str, Any]] = []
    occupancy: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    tables: dict[str, list[dict[str, Any]]] = {}
    phase_profiles: list[dict[str, Any]] = []

    for mode, shapes in shape_groups.items():
        rows: list[dict[str, Any]] = []
        for tokens in shapes:
            pool = [
                data.build_routed_input(weights, device, tokens, index)
                for index in range(pool_size)
            ]
            occupancy.append(
                {
                    "mode": mode,
                    "tokens": tokens,
                    "distinct_experts_per_replay": [
                        len(entry.distinct_experts) for entry in pool
                    ],
                    "expected_distinct_experts": min(TOPK * tokens, NUM_EXPERTS),
                    "pool_wide_distinct_experts": len(
                        {
                            expert
                            for entry in pool
                            for expert in entry.distinct_experts
                        }
                    ),
                    "route_assignments_by_pool_entry": [
                        [list(token) for token in entry.route_assignments]
                        for entry in pool
                    ],
                }
            )

            for entry in pool:
                adapter.load_router(entry.weights.router_weight)
                custom = runtime.decode_step(
                    workspace, entry.weights, entry.hidden
                ).clone()
                native = adapter.forward(entry.hidden)
                torch.cuda.synchronize(device)
                reference = runtime.decode_reference(entry.hidden, entry.weights)
                comparison = adapter.router_comparison(
                    entry.hidden, entry.weights
                )
                parity.append(
                    {
                        "mode": mode,
                        "tokens": tokens,
                        "pool_index": entry_index(pool, entry),
                        "router": comparison,
                        "custom_vs_native": _numerical_stats(custom, native),
                        "custom_vs_reference": _numerical_stats(
                            custom, reference
                        ),
                        "native_vs_reference": _numerical_stats(
                            native, reference
                        ),
                        **adapter.stage_parity(entry.hidden, entry.weights),
                    }
                )
                del custom, native, reference
            _barrier(device)

            measurements = _measure_backends(
                adapter,
                workspace,
                pool,
                device,
                mode=mode,
                tokens=tokens,
                framework=framework,
                warmup_count=warmup_count,
                sample_count=sample_count,
            )
            phase_profiles.append(
                {
                    "mode": mode,
                    "tokens": tokens,
                    **_phase_profile(workspace, pool, runtime, device),
                }
            )
            for measurement in measurements:
                raw_samples.append(measurement)
                rows.append(
                    {
                        key: value
                        for key, value in measurement.items()
                        if key != "rank_max_samples_ms"
                    }
                )
            del pool
            _barrier(device)
            torch.cuda.empty_cache()
        tables[mode] = rows

    traces = _collect_traces(adapter, workspace, weights, data, runtime, device)

    if rank == 0:
        manifest = build_comparison_manifest(framework=framework, dry_run=False)
        properties = torch.cuda.get_device_properties(device)
        manifest.update(
            {
                "warmup_count": warmup_count,
                "sample_count": sample_count,
                "gpu_detail": {
                    "name": properties.name,
                    "compute_capability": "sm_103",
                    "sm_count": properties.multi_processor_count,
                    "total_memory_bytes": properties.total_memory,
                },
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            }
        )
        write_json(output_dir / "manifest.json", manifest)
        write_json(
            output_dir / "versions.json",
            {
                "framework": framework,
                "captured": capture_versions(
                    manifest["recorded_distributions"]
                ),
                "adapter": adapter.versions(),
            },
        )
        write_json(
            output_dir / "transformations.json",
            {"framework": framework, "transformations": adapter.transformations()},
        )
        write_json(
            output_dir / "parity.json",
            {
                "framework": framework,
                "tolerances": dict(NUMERICAL_TOLERANCES),
                "rows": parity,
            },
        )
        write_json(
            output_dir / "route_occupancy.json",
            {"framework": framework, "rows": occupancy},
        )
        write_json(
            output_dir / "raw_samples.json",
            {"framework": framework, "rows": raw_samples},
        )
        write_json(output_dir / "launch_traces.json", traces)
        write_json(
            output_dir / "phase_profile.json",
            {"backend": "mok", "rows": phase_profiles},
        )
        for mode in shape_groups:
            write_latency_table(output_dir, f"latency_{mode}", tables[mode])
        write_json(
            output_dir / "performance_gates.json",
            _partial_gates(framework, tables.get("block16", [])),
        )
        expected = comparison_artifact_files(list(shape_groups))
        missing = [
            name for name in expected if not (output_dir / name).is_file()
        ]
        if missing:
            raise AssertionError(f"missing comparison artifacts: {missing}")
        print(json.dumps({"framework": framework, "artifacts": list(expected)}))
        print(json.dumps(parity_summary(framework, parity), indent=2, sort_keys=True))
        for mode in shape_groups:
            for row in tables[mode]:
                print(
                    "LATENCY "
                    + json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "backend",
                                "mode",
                                "tokens",
                                "median_ms",
                                "p90_ms",
                                "p99_ms",
                            )
                        },
                        sort_keys=True,
                    )
                )

    _barrier(device)
    adapter.close()
    kimi.clear_kimi_k3_decode_workspace_cache()
    torch.distributed.destroy_process_group()


def parity_summary(
    framework: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce every parity row to the worst value the tolerances constrain."""

    def worst(path: Sequence[str], reducer: Callable[[Iterable[float]], float]) -> float:
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return reducer(values)

    comparisons = {}
    for comparison in (
        "custom_vs_native",
        "custom_vs_reference",
        "native_vs_reference",
        "routed_latent_vs_reference",
        "shared_output_vs_reference",
    ):
        comparisons[comparison] = {
            "max_relative_l1": worst((comparison, "relative_l1"), max),
            "min_cosine_similarity": worst(
                (comparison, "cosine_similarity"), min
            ),
            "max_abs": worst((comparison, "max_abs"), max),
        }
    return {
        "framework": framework,
        "row_count": len(rows),
        "router_expert_ids_all_match": all(
            row["router"]["expert_ids_match"] for row in rows
        ),
        "router_weight_max_abs": max(
            float(row["router"]["router_weight_max_abs"]) for row in rows
        ),
        "comparisons": comparisons,
        "tolerances": dict(NUMERICAL_TOLERANCES),
    }


def entry_index(pool: Sequence[Any], entry: Any) -> int:
    for index, candidate in enumerate(pool):
        if candidate is entry:
            return index
    raise ValueError("pool entry not found")


def _measure_backends(
    adapter: Any,
    workspace: Any,
    pool: Sequence[Any],
    device: Any,
    *,
    mode: str,
    tokens: int,
    framework: str,
    warmup_count: int,
    sample_count: int,
) -> list[dict[str, Any]]:
    import torch

    from mok import kimi_k3 as kimi
    from benchmarks import kimi_k3_decode_runtime as runtime

    results: list[dict[str, Any]] = []

    def summarize(backend: str, graphs: Sequence[Any]) -> dict[str, Any]:
        _barrier(device)
        samples = _replay_samples(
            lambda iteration: graphs[iteration % len(graphs)].replay(),
            device,
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
        return merge_backend_samples(
            backend=backend,
            mode=mode,
            tokens=tokens,
            rank_samples=_gathered_rank_samples(samples, device),
        ) | {
            "distinct_experts_per_replay": len(pool[0].distinct_experts),
            "graph_pool_size": len(pool),
            "warmup_count": warmup_count,
            "sample_count": sample_count,
        }

    custom_graphs: list[Any] = []
    for entry in pool:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kimi.kimi_k3_decode(
                runtime.CONFIG, workspace, entry.weights, entry.hidden
            )
        custom_graphs.append(graph)
    results.append(summarize(CUSTOM_BACKEND, custom_graphs))
    custom_graphs.clear()
    _barrier(device)
    torch.cuda.empty_cache()

    results.append(summarize(framework, adapter.capture(pool)))
    adapter.release()
    _barrier(device)
    torch.cuda.empty_cache()
    return results


def _collect_traces(
    adapter: Any,
    workspace: Any,
    weights: Any,
    data: Any,
    runtime: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    from mok import kimi_k3 as kimi

    entry = data.build_routed_input(weights, device, CONCURRENCY_ONE_TOKENS, 0)
    adapter.load_router(entry.weights.router_weight)
    custom_names = _kernel_trace(
        lambda: kimi.kimi_k3_decode(
            runtime.CONFIG, workspace, entry.weights, entry.hidden
        )
    )
    native_names = _kernel_trace(lambda: adapter.forward(entry.hidden))
    torch.cuda.synchronize(device)
    return {
        "tokens": CONCURRENCY_ONE_TOKENS,
        "mok": {"launch_count": len(custom_names), "kernels": custom_names},
        adapter.name: {
            "launch_count": len(native_names),
            "kernels": native_names,
        },
    }


def _partial_gates(
    framework: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record this archive's half of the gate inputs.

    The full gates need both baselines, so a single-framework archive stores
    the pairwise comparison and defers the verdict to :func:`combine_archives`.
    """
    index = {
        (str(row["backend"]), int(row["tokens"])): row
        for row in rows
        if row.get("mode") == "block16"
    }
    pairwise = []
    for tokens in GATE_SHAPES:
        custom = index.get((CUSTOM_BACKEND, tokens))
        native = index.get((framework, tokens))
        if custom is None or native is None:
            continue
        pairwise.append(
            {
                "tokens": tokens,
                "custom_median_ms": custom["median_ms"],
                "native_median_ms": native["median_ms"],
                "custom_p99_ms": custom["p99_ms"],
                "native_p99_ms": native["p99_ms"],
                "median_ratio": custom["median_ms"] / native["median_ms"],
                "p99_ratio": custom["p99_ms"] / native["p99_ms"],
            }
        )
    custom_medians = [row["custom_median_ms"] for row in pairwise]
    native_medians = [row["native_median_ms"] for row in pairwise]
    return {
        "benchmark": BENCHMARK,
        "framework": framework,
        "status": "partial",
        "reason": "the full gates require both native baselines",
        "limit_ratio": P99_LIMIT_RATIO,
        "pairwise_block16": pairwise,
        "custom_geomean_ms": (
            geometric_mean(custom_medians) if custom_medians else None
        ),
        "native_geomean_ms": (
            geometric_mean(native_medians) if native_medians else None
        ),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=sorted(ADAPTER_MODULES), default="vllm")
    parser.add_argument("--output-dir", type=Path, default=Path("kimi_k3_comparison"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--pool-size", type=int, default=GRAPH_POOL_SIZE)
    parser.add_argument("--modes", default=",".join(SHAPE_GROUPS))
    parser.add_argument("--tokens", default="")
    parser.add_argument("--combine", nargs="*", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.combine:
        summary = combine_archives(args.combine, args.output_dir)
        print(json.dumps(summary["performance_gates"], indent=2, sort_keys=True))
        return
    if args.dry_run:
        manifest = write_dry_run(args.output_dir, framework=args.framework)
        print(json.dumps(manifest, sort_keys=True))
        return
    modes = [mode for mode in args.modes.split(",") if mode]
    override = [int(value) for value in args.tokens.split(",") if value]
    shape_groups = {
        mode: (tuple(override) if override else SHAPE_GROUPS[mode])
        for mode in modes
    }
    if args.warmup_count < 1 or args.sample_count < 1:
        raise ValueError("warmup and sample counts must be positive")
    _run_gpu(
        args.framework,
        args.output_dir,
        warmup_count=args.warmup_count,
        sample_count=args.sample_count,
        shape_groups=shape_groups,
        pool_size=args.pool_size,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
