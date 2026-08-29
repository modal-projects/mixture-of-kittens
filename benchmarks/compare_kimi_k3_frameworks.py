"""Compare the one-launch Kimi K3 decode kernel with native serving backends.

The comparison runs the custom kernel and one framework's complete native Kimi
K3 sparse-MoE layer side by side, in the same process, on the same eight B300s,
against the same prepared weights and the same realistic route pool. Nothing in
this module imports vLLM, SGLang, or FlashInfer: the framework code lives behind
the adapters in :mod:`benchmarks.frameworks`, which only the derived comparison
images can import.

The pins and the archive's manifest live in
:mod:`benchmarks.kimi_k3_comparison_manifest`, and both gate families live in
:mod:`benchmarks.kimi_k3_comparison_gates`; both are re-exported here so the
archive, the CLI, and the tests all name one module.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_comparison_gates import (
    DIAGNOSTIC_COMPARISONS,
    GATED_COMPARISON,
    combine_archives,
    evaluate_numerical_gates,
    evaluate_performance_gates,
    expected_parity_rows,
    merge_latency_rows,
    parity_summary,
    partial_gates,
    write_json,
)
from benchmarks.kimi_k3_comparison_manifest import (
    ADAPTER_MODULES,
    ARTIFACT_FILES,
    BASELINE_BACKENDS,
    BENCHMARK,
    BLOCK8_SHAPES,
    BLOCK16_SHAPES,
    CONCURRENCY_ONE_TOKENS,
    CUSTOM_BACKEND,
    GATE_SHAPES,
    HIDDEN_SIZE,
    IMAGE_REFERENCE_ENV,
    LATENCY_COLUMNS,
    LATENT_SIZE,
    MANIFEST_PATH,
    MXFP4_GROUP_SIZE,
    NUM_EXPERTS,
    NUMERICAL_TOLERANCES,
    P99_LIMIT_RATIO,
    REQUIRED_FRAMEWORK_PINS,
    ROUTED_INTERMEDIATE_SIZE,
    ROUTER_WEIGHT_MAX_ABS,
    SAMPLE_COUNT,
    SHAPE_GROUPS,
    SHARED_INTERMEDIATE_SIZE,
    TOPK,
    TP_SIZE,
    WARMUP_COUNT,
    adapter_weight_shapes,
    build_comparison_manifest,
    capture_versions,
    comparison_artifact_files,
    effective_image_reference,
    fused_gate_up_plan,
    load_framework_manifest,
    pinned_image_reference,
)
from benchmarks.kimi_k3_decode_inputs import GRAPH_POOL_SIZE, mode_batch_size
from benchmarks.kimi_k3_timing import (
    geometric_mean,
    percentile,
    rank_max_samples,
    replay_samples,
)

# The pins, the gates, and the driver are three files; the comparison is one
# module. Everything the archive writers, the Modal app, and the tests reach
# for is named here whether it is defined below or re-exported from a sibling.
__all__ = [
    "ADAPTER_MODULES",
    "ARTIFACT_FILES",
    "BASELINE_BACKENDS",
    "BENCHMARK",
    "BLOCK8_SHAPES",
    "BLOCK16_SHAPES",
    "CONCURRENCY_ONE_TOKENS",
    "CUSTOM_BACKEND",
    "DIAGNOSTIC_COMPARISONS",
    "GATED_COMPARISON",
    "GATE_SHAPES",
    "HIDDEN_SIZE",
    "IMAGE_REFERENCE_ENV",
    "LATENCY_COLUMNS",
    "LATENT_SIZE",
    "MANIFEST_PATH",
    "MXFP4_GROUP_SIZE",
    "NUMERICAL_TOLERANCES",
    "NUM_EXPERTS",
    "P99_LIMIT_RATIO",
    "PHASE_CLOCK_NAMES",
    "REQUIRED_FRAMEWORK_PINS",
    "ROUTED_INTERMEDIATE_SIZE",
    "ROUTER_WEIGHT_MAX_ABS",
    "SAMPLE_COUNT",
    "SHAPE_GROUPS",
    "SHARED_INTERMEDIATE_SIZE",
    "TOPK",
    "TP_SIZE",
    "WARMUP_COUNT",
    "adapter_weight_shapes",
    "build_comparison_manifest",
    "capture_versions",
    "combine_archives",
    "comparison_artifact_files",
    "effective_image_reference",
    "evaluate_numerical_gates",
    "expected_parity_rows",
    "evaluate_performance_gates",
    "fused_gate_up_plan",
    "load_framework_manifest",
    "main",
    "merge_backend_samples",
    "merge_latency_rows",
    "parity_summary",
    "partial_gates",
    "pinned_image_reference",
    "summarize_phase_cycles",
    "write_json",
    "write_latency_table",
]


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
_AGENT_DEBUG_LOG = "/opt/cursor/logs/debug.log"


def _agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Mapping[str, Any],
) -> None:
    try:
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": dict(data),
                        "timestamp": time.time_ns() // 1_000_000,
                    }
                )
                + "\n"
            )
    except OSError:
        pass


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


# --------------------------------------------------------------------------
# Archive writers
# --------------------------------------------------------------------------


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


def write_dry_run(
    output_dir: Path,
    *,
    framework: str,
    warmup_count: int = WARMUP_COUNT,
    sample_count: int = SAMPLE_COUNT,
    shape_groups: Mapping[str, Sequence[int]] | None = None,
    pool_size: int = GRAPH_POOL_SIZE,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_comparison_manifest(
        framework=framework,
        dry_run=True,
        warmup_count=warmup_count,
        sample_count=sample_count,
        shape_groups=shape_groups,
        pool_size=pool_size,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


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
    settle_count: int,
) -> list[float]:
    import torch

    return replay_samples(
        replay,
        warmup_count=warmup_count,
        sample_count=sample_count,
        settle_count=settle_count,
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=lambda: torch.cuda.synchronize(device),
    )


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
    """Clock64 cycles per kernel region, for the last launch over the pool.

    Collected outside the timed section and outside the captured graphs: the
    accumulators cost atomics that the measured launches must not pay, and a
    graph would have recorded whichever launch it captured anyway.

    A profiled launch clears the band before it starts timing, so the counters
    read back afterwards belong to the last pool entry alone. The earlier
    entries are there to warm the path, and `replays` counts them rather than
    describing what the cycles cover.
    """
    import torch

    tokens = int(pool[0].hidden.shape[0])
    batch = 1 if tokens <= 16 else 4
    # region agent log
    _agent_debug_log(
        "A",
        "benchmarks/compare_kimi_k3_frameworks.py:_phase_profile",
        "phase profile entry",
        {"tokens": tokens, "pool_size": len(pool), "routed_claim_batch": batch},
    )
    # endregion
    with runtime.phase_profiling():
        for entry in pool:
            runtime.decode_step(workspace, entry.weights, entry.hidden)
        # region agent log
        _agent_debug_log(
            "C",
            "benchmarks/compare_kimi_k3_frameworks.py:_phase_profile",
            "profile replays queued",
            {"tokens": tokens, "replays": len(pool)},
        )
        # endregion
        torch.cuda.synchronize(device)
        cycles = runtime.phase_clock_cycles(workspace)
    # region agent log
    _agent_debug_log(
        "B",
        "benchmarks/compare_kimi_k3_frameworks.py:_phase_profile",
        "phase profile exit",
        {
            "tokens": tokens,
            "grid_barrier": cycles["grid_barrier"],
            "routed_gate_up": cycles["routed_gate_up"],
            "routed_down": cycles["routed_down"],
        },
    )
    # endregion
    return {
        "replays": len(pool),
        "cycles_cover": "the last replay",
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

    # One router, built before anything is captured and never written again.
    # Every shape, every pool entry, both backends, and every graph read it.
    router = data.shared_router(
        weights,
        device,
        sorted({tokens for shapes in shape_groups.values() for tokens in shapes}),
        pool_size=pool_size,
    )
    weights = dataclasses.replace(weights, router_weight=router.weight)

    adapter = adapter_module.build_adapter(
        device=device,
        tp_rank=rank,
        tp_size=TP_SIZE,
        weights=weights,
    )
    router_fingerprint = adapter.bind_router(
        router.weight, router.correction_bias
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
                data.build_routed_input(
                    weights, device, tokens, index, router=router
                )
                for index in range(pool_size)
            ]

            for entry in pool:
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

            measurements, verification = _measure_backends(
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
            occupancy.append(
                _route_occupancy_row(pool, verification, mode=mode, tokens=tokens)
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

    traces = _collect_traces(
        adapter, workspace, weights, router, data, runtime, device
    )

    if rank == 0:
        manifest = build_comparison_manifest(
            framework=framework,
            dry_run=False,
            warmup_count=warmup_count,
            sample_count=sample_count,
            shape_groups=shape_groups,
            pool_size=pool_size,
        )
        properties = torch.cuda.get_device_properties(device)
        manifest.update(
            {
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
        numerical = evaluate_numerical_gates(
            parity,
            expected_rows=expected_parity_rows(shape_groups, pool_size),
        )
        write_json(
            output_dir / "numerical_gates.json",
            {"framework": framework, **numerical},
        )
        write_json(
            output_dir / "route_occupancy.json",
            {
                "framework": framework,
                "router": {
                    "shared_across_every_graph": True,
                    "column_plan": {
                        str(count): column
                        for count, column in sorted(router.column_plan.items())
                    },
                    "bound_fingerprint": router_fingerprint,
                },
                "rows": occupancy,
            },
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
            partial_gates(framework, tables.get("block16", [])),
        )
        expected = comparison_artifact_files(list(shape_groups))
        missing = [
            name for name in expected if not (output_dir / name).is_file()
        ]
        if missing:
            raise AssertionError(f"missing comparison artifacts: {missing}")
        print(json.dumps({"framework": framework, "artifacts": list(expected)}))
        print(json.dumps(parity_summary(framework, parity), indent=2, sort_keys=True))
        print(
            "NUMERICAL_GATES "
            + json.dumps(
                {
                    "framework": framework,
                    "passed": numerical["passed"],
                    "row_count": numerical["row_count"],
                    "violations": numerical["violations"],
                },
                sort_keys=True,
            )
        )
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


def entry_index(pool: Sequence[Any], entry: Any) -> int:
    for index, candidate in enumerate(pool):
        if candidate is entry:
            return index
    raise ValueError("pool entry not found")


def _route_occupancy_row(
    pool: Sequence[Any],
    verification: Mapping[str, Any],
    *,
    mode: str,
    tokens: int,
) -> dict[str, Any]:
    """Archive what the run's replays actually routed to.

    Every count here is derived from an observation: the custom column from
    the official reference run against the entry's own input, and the native
    column from the ids a replayed native graph wrote. The plan the entry was
    built from is kept beside them to be compared, not to stand in for them.
    """
    observed = verification["observed_route_assignments_by_graph"]
    return {
        "mode": mode,
        "tokens": tokens,
        "expected_distinct_experts": min(TOPK * tokens, NUM_EXPERTS),
        "intended_route_assignments_by_pool_entry": [
            [list(token) for token in entry.route_assignments] for entry in pool
        ],
        "observed_reference_distinct_experts_per_replay": [
            len(entry.distinct_experts) for entry in pool
        ],
        "observed_native_route_assignments_by_graph": observed,
        "observed_native_distinct_experts_per_graph": (
            verification["distinct_experts_per_graph"]
        ),
        "observed_native_distinct_route_sets": (
            verification["distinct_route_sets"]
        ),
        "pool_wide_distinct_experts": len(
            {
                expert
                for entry in observed
                for token in entry
                for expert in token
            }
        ),
        "replayed_output_relative_l1_to_each_entry": (
            verification["replayed_output_relative_l1_to_each_entry"]
        ),
        "router_fingerprint": verification["router_fingerprint"],
    }


def _replayed_routes(
    adapter: Any,
    pool: Sequence[Any],
    device: Any,
) -> list[list[list[int]]]:
    """Replay each captured native router graph and read what it selected.

    This is the routing the graph produces, not the routing it was captured
    for: the ids come out of the buffer the replay writes.
    """
    import torch

    from benchmarks.frameworks.kimi_k3_adapter_common import observed_routes

    graphs, id_buffers = adapter.capture_router(list(pool))
    routes = []
    for graph, ids in zip(graphs, id_buffers, strict=True):
        graph.replay()
        torch.cuda.synchronize(device)
        routes.append(observed_routes(ids))
    adapter.release_router()
    return routes


def _replayed_output_distances(
    adapter: Any,
    pool: Sequence[Any],
    graphs: Sequence[Any],
    device: Any,
) -> list[list[float]]:
    """How far each replayed graph's output is from every entry's own forward.

    Pool entries route to disjoint expert blocks, so a graph that replayed the
    wrong entry's routing computes a visibly different layer output. Distances
    to all of them are recorded rather than a single tolerance, so the check is
    that graph ``p`` is nearest entry ``p`` -- a comparison with no threshold
    to pick.
    """
    import torch

    eager = []
    for entry in pool:
        eager.append(adapter.forward(entry.hidden).float().clone())
    torch.cuda.synchronize(device)

    outputs = adapter.graph_outputs()
    distances: list[list[float]] = []
    for index, graph in enumerate(graphs):
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = outputs[index].float()
        distances.append(
            [
                float(
                    (replayed - expected).abs().sum()
                    / expected.abs().sum().clamp_min(1e-12)
                )
                for expected in eager
            ]
        )
    del eager
    return distances


def _verify_native_graphs(
    adapter: Any,
    pool: Sequence[Any],
    graphs: Sequence[Any],
    device: Any,
    *,
    mode: str,
    tokens: int,
) -> dict[str, Any]:
    """Prove every captured native graph replays its own pool entry.

    Two independent readings. The router graphs report the expert ids a replay
    actually selects, which is checked against the entry's intended block. The
    full graphs report a layer output, which is checked to be nearest its own
    entry's eager forward among all of them. A pool captured around a mutated
    router fails both: every graph reports the last entry's routing.
    """
    from benchmarks.kimi_k3_decode_inputs import verify_graph_routes

    before = adapter.router_fingerprint()
    intended = [entry.route_assignments for entry in pool]
    routes = _replayed_routes(adapter, pool, device)
    summary = verify_graph_routes(intended, routes)

    distances = _replayed_output_distances(adapter, pool, graphs, device)
    misrouted = [
        index
        for index, row in enumerate(distances)
        if min(range(len(row)), key=lambda other: row[other]) != index
    ]
    if misrouted:
        raise AssertionError(
            f"replayed graphs are nearest another entry's output at "
            f"{mode}/{tokens}: pool indices {misrouted}, distances {distances}"
        )

    after = adapter.router_fingerprint()
    if before != after:
        raise AssertionError(
            f"the router changed across capture at {mode}/{tokens}: "
            f"{before} then {after}"
        )
    return {
        "mode": mode,
        "tokens": tokens,
        "router_fingerprint": after,
        "observed_route_assignments_by_graph": routes,
        "replayed_output_relative_l1_to_each_entry": distances,
        **summary,
    }


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            settle_count=len(graphs),
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

    native_graphs = adapter.capture(pool)
    verification = _verify_native_graphs(
        adapter, pool, native_graphs, device, mode=mode, tokens=tokens
    )
    _barrier(device)
    results.append(summarize(framework, native_graphs))
    adapter.release()
    _barrier(device)
    torch.cuda.empty_cache()
    return results, verification


def _collect_traces(
    adapter: Any,
    workspace: Any,
    weights: Any,
    router: Any,
    data: Any,
    runtime: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    from mok import kimi_k3 as kimi

    entry = data.build_routed_input(
        weights, device, CONCURRENCY_ONE_TOKENS, 0, router=router
    )
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


def main(argv: list[str] | None = None) -> int:
    """Run one comparison, one dry run, or one combine, and return an exit code.

    A combine returns non-zero when either gate family fails. The artifacts are
    already on disk by then, so the failure is a verdict rather than a lost run.
    """
    args = _parse_args(argv)
    if args.combine:
        summary = combine_archives(args.combine, args.output_dir)
        print(
            json.dumps(
                {
                    "passed": summary["passed"],
                    "numerical_gates": {
                        "passed": summary["numerical_gates"]["passed"],
                        "row_count": summary["numerical_gates"]["row_count"],
                        "violations": summary["numerical_gates"]["violations"],
                    },
                    "performance_gates": summary["performance_gates"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary["passed"] else 1
    modes = [mode for mode in args.modes.split(",") if mode]
    override = [int(value) for value in args.tokens.split(",") if value]
    shape_groups = {
        mode: (tuple(override) if override else SHAPE_GROUPS[mode])
        for mode in modes
    }
    if args.warmup_count < 1 or args.sample_count < 1:
        raise ValueError("warmup and sample counts must be positive")
    if args.dry_run:
        manifest = write_dry_run(
            args.output_dir,
            framework=args.framework,
            warmup_count=args.warmup_count,
            sample_count=args.sample_count,
            shape_groups=shape_groups,
            pool_size=args.pool_size,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    _run_gpu(
        args.framework,
        args.output_dir,
        warmup_count=args.warmup_count,
        sample_count=args.sample_count,
        shape_groups=shape_groups,
        pool_size=args.pool_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
