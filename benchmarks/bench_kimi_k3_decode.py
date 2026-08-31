"""Reproducible TP8 latency benchmark for the one-launch Kimi K3 decode path."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.kimi_k3_decode_inputs import (
    GRAPH_POOL_SIZE,
    GRID_CANDIDATES,
    mode_batch_size,
    route_metadata,
)
from benchmarks.kimi_k3_decode_output import (
    ARTIFACT_FILES,
    P99_TO_MEDIAN_LIMIT,
    SAMPLE_COUNT,
    SHAPE_GROUPS,
    TP_SIZE,
    TUNING_REPEATS,
    WARMUP_COUNT,
    build_manifest,
    gpu_clocks,
    write_dry_run,
    write_json,
    write_latency_table,
)
from benchmarks.kimi_k3_timing import (
    percentile,
    rank_max_samples,
    rotating_candidate_orders,
    select_grid_with_effect_band,
    summarize_rank_max,
)


@dataclass(slots=True)
class GraphInput:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    weights: Any
    pool_index: int
    route_assignments: tuple[tuple[int, ...], ...]
    distinct_experts: tuple[int, ...]

@dataclass(slots=True)
class RuntimeModules:
    mok: ModuleType
    extension: ModuleType
    kimi: ModuleType
    data: ModuleType
    runtime: ModuleType


def _runtime_modules() -> RuntimeModules:
    # These modules import the CUDA extension. Keeping their import behind the
    # non-dry-run boundary lets shape and manifest validation run on CPU hosts.
    mok = importlib.import_module("mok")
    return RuntimeModules(
        mok=mok,
        extension=importlib.import_module("mok._C"),
        kimi=importlib.import_module("mok.kimi_k3"),
        data=importlib.import_module("benchmarks.kimi_k3_decode_data"),
        runtime=importlib.import_module("benchmarks.kimi_k3_decode_runtime"),
    )


def _init_distributed() -> tuple[int, int, torch.device]:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch the benchmark with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP_SIZE:
        raise RuntimeError(f"Kimi K3 decode benchmark requires TP{TP_SIZE}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("Kimi K3 decode benchmark requires an SM103 B300")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return rank, world_size, device


def _barrier(device: torch.device) -> None:
    dist.barrier(
        async_op=True,
        device_ids=[device.index],
    ).block_current_stream()
    torch.cuda.synchronize(device)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _prepared_weight_bytes(weights: Any) -> int:
    return sum(
        _tensor_bytes(value)
        for field in fields(weights)
        if isinstance((value := getattr(weights, field.name)), torch.Tensor)
    )


def _expert_weight_bytes(weights: Any) -> int:
    """Bytes one expert's own prepared matrices occupy.

    The routed gate and up projections are one fused payload, so slice zero of
    it is both of that expert's halves -- exactly the bytes the two separate
    packed matrices and their two scale blobs used to contribute between them,
    since the fuse is a permutation.
    """
    return sum(
        _tensor_bytes(getattr(weights, name)[0])
        for name in (
            "expert_w13_packed",
            "expert_w13_scale",
            "expert_w2_packed",
            "expert_w2_scale",
        )
    )


def _l2_cache_bytes(properties: Any) -> int:
    for field in ("L2_cache_size", "l2_cache_size"):
        value = getattr(properties, field, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("PyTorch did not expose the B300 L2 cache size")


def _set_grid(modules: RuntimeModules, grid_ctas: int) -> None:
    modules.extension._kimi_k3_decode_set_benchmark_grid(grid_ctas)
    selected = modules.extension._kimi_k3_decode_benchmark_grid()
    if selected != grid_ctas:
        raise AssertionError((grid_ctas, selected))


def _correctness_call(
    modules: RuntimeModules,
    workspace: Any,
    weights: Any,
    hidden: torch.Tensor,
) -> dict[str, float]:
    expected = modules.runtime.decode_reference(hidden, weights)
    actual = modules.runtime.decode_step(workspace, weights, hidden)
    torch.cuda.synchronize(hidden.device)
    relative_l1, cosine, maximum = modules.runtime.assert_decode_close(
        actual,
        expected,
    )
    modules.runtime.assert_identical_across_ranks(actual)
    return {
        "relative_l1": relative_l1,
        "cosine_similarity": cosine,
        "max_abs": maximum,
    }


def _correctness_sweep(
    modules: RuntimeModules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    grid_ctas: int,
    expert_weight_bytes: int,
    l2_cache_bytes: int,
) -> list[dict[str, Any]]:
    _set_grid(modules, grid_ctas)
    rows: list[dict[str, Any]] = []
    for mode, shapes in SHAPE_GROUPS.items():
        for tokens in shapes:
            metadata = route_metadata(
                tokens=tokens,
                expert_weight_bytes=expert_weight_bytes,
                l2_cache_bytes=l2_cache_bytes,
            )
            if not metadata[
                "pool_wide_routed_expert_working_set_exceeds_l2"
            ]:
                raise AssertionError((tokens, metadata))
            for pool_index in range(GRAPH_POOL_SIZE):
                routed = modules.data.build_routed_input(
                    base_weights,
                    device,
                    tokens,
                    pool_index,
                )
                metrics = _correctness_call(
                    modules,
                    workspace,
                    routed.weights,
                    routed.hidden,
                )
                rows.append(
                    {
                        "phase": "tuning_candidate_full_correctness",
                        "grid_ctas": grid_ctas,
                        "mode": mode,
                        "tokens": tokens,
                        "pool_index": pool_index,
                        "route_assignments": [
                            list(token)
                            for token in routed.route_assignments
                        ],
                        "distinct_experts_per_replay": len(
                            routed.distinct_experts
                        ),
                        "pool_wide_distinct_experts": metadata[
                            "pool_wide_distinct_experts"
                        ],
                        "routed_queue_units_per_replay": metadata[
                            "routed_queue_units_per_replay"
                        ],
                        "routed_expert_working_set_bytes_per_replay": (
                            metadata[
                                "routed_expert_working_set_bytes_per_replay"
                            ]
                        ),
                        "pool_wide_routed_expert_working_set_bytes": metadata[
                            "pool_wide_routed_expert_working_set_bytes"
                        ],
                        **metrics,
                    }
                )
                del routed
    _barrier(device)
    torch.cuda.empty_cache()
    return rows


def _capture_graph_pool(
    modules: RuntimeModules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    tokens: int,
    pool_size: int,
) -> list[GraphInput]:
    entries: list[GraphInput] = []
    for pool_index in range(pool_size):
        routed = modules.data.build_routed_input(
            base_weights,
            device,
            tokens,
            pool_index,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            modules.kimi.kimi_k3_decode(
                modules.runtime.CONFIG,
                workspace,
                routed.weights,
                routed.hidden,
            )
        entries.append(
            GraphInput(
                graph,
                routed.hidden,
                routed.weights,
                pool_index,
                routed.route_assignments,
                routed.distinct_experts,
            )
        )
    _barrier(device)
    return entries


def _graph_route_metadata(
    entries: list[GraphInput],
    *,
    expert_weight_bytes: int,
    l2_cache_bytes: int,
) -> dict[str, Any]:
    tokens = int(entries[0].hidden.shape[0])
    metadata = route_metadata(
        tokens=tokens,
        expert_weight_bytes=expert_weight_bytes,
        l2_cache_bytes=l2_cache_bytes,
    )
    observed = [
        [list(token) for token in entry.route_assignments]
        for entry in entries
    ]
    observed_pool = {
        expert
        for entry in entries
        for expert in entry.distinct_experts
    }
    if len(observed_pool) != metadata["pool_wide_distinct_experts"]:
        raise AssertionError((len(observed_pool), metadata))
    metadata["route_assignments_by_pool_entry"] = observed
    if not metadata["pool_wide_routed_expert_working_set_exceeds_l2"]:
        raise AssertionError((tokens, metadata))
    return metadata


def _measure_graph_pool(
    entries: list[GraphInput],
    workspace: Any,
    device: torch.device,
    *,
    warmup_count: int,
    sample_count: int,
) -> dict[str, Any]:
    for iteration in range(warmup_count):
        entries[iteration % len(entries)].graph.replay()
    _barrier(device)

    starts = [
        torch.cuda.Event(enable_timing=True) for _ in range(sample_count)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True) for _ in range(sample_count)
    ]
    for iteration, (start, end) in enumerate(zip(starts, ends, strict=True)):
        start.record()
        entries[iteration % len(entries)].graph.replay()
        end.record()
    torch.cuda.synchronize(device)
    local_samples = [
        start.elapsed_time(end)
        for start, end in zip(starts, ends, strict=True)
    ]
    local_tensor = torch.tensor(
        local_samples,
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local_tensor) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, local_tensor)
    rank_samples = [samples.cpu().tolist() for samples in gathered]
    summary = summarize_rank_max(rank_samples)
    maxima = rank_max_samples(rank_samples)
    if int(workspace.error_flag.item()) != 0:
        raise AssertionError(
            f"persistent kernel error flag: {int(workspace.error_flag.item())}"
        )
    return {
        **summary,
        "rank_max_samples_ms": maxima,
    }


def _verify_graph_result(
    modules: RuntimeModules,
    entries: list[GraphInput],
    workspace: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for entry in entries:
        entry.graph.replay()
        torch.cuda.synchronize(entry.hidden.device)
        expected = modules.runtime.decode_reference(
            entry.hidden,
            entry.weights,
        )
        tokens = entry.hidden.shape[0]
        actual = workspace.output_mailbox.view(128, 7168)[:tokens]
        relative_l1, cosine, maximum = modules.runtime.assert_decode_close(
            actual,
            expected,
        )
        modules.runtime.assert_identical_across_ranks(actual)
        results.append(
            {
                "pool_index": entry.pool_index,
                "route_assignments": [
                    list(token) for token in entry.route_assignments
                ],
                "distinct_experts_per_replay": len(
                    entry.distinct_experts
                ),
                "relative_l1": relative_l1,
                "cosine_similarity": cosine,
                "max_abs": maximum,
            }
        )
    return results


def _candidate_occupancy(
    modules: RuntimeModules,
    properties: Any,
    grid_ctas: int,
) -> dict[str, Any]:
    blocks = {
        "core": modules.extension._kimi_k3_decode_resident_blocks_per_sm(
            False
        ),
        "tensor": modules.extension._kimi_k3_decode_resident_blocks_per_sm(
            True
        ),
    }
    passed = (
        grid_ctas <= properties.multi_processor_count
        and all(value == 1 for value in blocks.values())
    )
    return {
        "passed": passed,
        "available_sms": properties.multi_processor_count,
        "resident_blocks_per_sm": blocks,
    }


def _tune_grids(
    modules: RuntimeModules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    properties: Any,
    pool_size: int,
    expert_weight_bytes: int,
    l2_cache_bytes: int,
    production_grid: int,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    candidates_by_grid: dict[int, dict[str, Any]] = {}
    graph_pools: dict[int, list[GraphInput]] = {}
    all_correctness: list[dict[str, Any]] = []
    for grid_ctas in GRID_CANDIDATES:
        occupancy = _candidate_occupancy(
            modules,
            properties,
            grid_ctas,
        )
        candidate: dict[str, Any] = {
            "grid_ctas": grid_ctas,
            "occupancy": occupancy,
        }
        if not occupancy["passed"]:
            candidate.update(
                {
                    "status": "rejected",
                    "reason": "candidate grid does not fully co-reside",
                }
            )
            candidates.append(candidate)
            candidates_by_grid[grid_ctas] = candidate
            continue

        correctness = _correctness_sweep(
            modules,
            workspace,
            base_weights,
            device,
            grid_ctas,
            expert_weight_bytes,
            l2_cache_bytes,
        )
        all_correctness.extend(correctness)
        candidate["full_correctness"] = {
            "passed": True,
            "shape_count": sum(len(shapes) for shapes in SHAPE_GROUPS.values()),
            "pool_entries_per_shape": GRAPH_POOL_SIZE,
            "shape_pool_checks": len(correctness),
        }
        _set_grid(modules, grid_ctas)
        entries = _capture_graph_pool(
            modules,
            workspace,
            base_weights,
            device,
            16,
            pool_size,
        )
        metadata = _graph_route_metadata(
            entries,
            expert_weight_bytes=expert_weight_bytes,
            l2_cache_bytes=l2_cache_bytes,
        )
        if (
            metadata["distinct_experts_per_replay"] != 256
            or not metadata[
                "routed_expert_working_set_exceeds_l2_per_replay"
            ]
        ):
            raise AssertionError(metadata)
        candidate.update(metadata)
        candidate["repeats"] = []
        graph_pools[grid_ctas] = entries
        candidates.append(candidate)
        candidates_by_grid[grid_ctas] = candidate

    candidate_orders = rotating_candidate_orders(GRID_CANDIDATES, repeats)
    for repeat_index, order in enumerate(candidate_orders):
        for order_position, grid_ctas in enumerate(order):
            candidate = candidates_by_grid[grid_ctas]
            if grid_ctas not in graph_pools:
                continue
            _set_grid(modules, grid_ctas)
            repeated = _measure_graph_pool(
                graph_pools[grid_ctas],
                workspace,
                device,
                warmup_count=warmup_count,
                sample_count=sample_count,
            )
            graph_metrics = _verify_graph_result(
                modules,
                graph_pools[grid_ctas],
                workspace,
            )
            candidate["repeats"].append(
                {
                    "repeat": repeat_index + 1,
                    "order_position": order_position,
                    "candidate_order": list(order),
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key
                        in {
                            "distinct_experts_per_replay",
                            "route_assignments_by_pool_entry",
                            "routed_queue_units_per_replay",
                            "pool_wide_distinct_experts",
                            (
                                "routed_expert_working_set_bytes_per_replay"
                            ),
                            (
                                "routed_expert_working_set_exceeds_l2_per_replay"
                            ),
                            (
                                "pool_wide_routed_expert_working_set_bytes"
                            ),
                            (
                                "pool_wide_routed_expert_working_set_exceeds_l2"
                            ),
                        }
                    },
                    **repeated,
                    "post_timing_graph_checks": graph_metrics,
                }
            )

    for candidate in candidates:
        if "repeats" not in candidate:
            continue
        repeated = candidate["repeats"]
        medians = [float(repeat["median_ms"]) for repeat in repeated]
        p99_ratios = [
            float(repeat["p99_ms"]) / float(repeat["median_ms"])
            for repeat in repeated
        ]
        center = percentile(medians, 0.5)
        dispersion = max(medians) - min(medians)
        p99_passed = all(
            math.isfinite(ratio) and ratio <= P99_TO_MEDIAN_LIMIT
            for ratio in p99_ratios
        )
        candidate.update(
            {
                "graph_replay": {
                    "passed": True,
                    "post_timing_checks_per_repeat": [
                        repeat["post_timing_graph_checks"]
                        for repeat in repeated
                    ],
                },
                "median_of_repeat_medians_ms": center,
                "median_dispersion_ms": dispersion,
                "relative_median_dispersion": (
                    dispersion / center if center > 0.0 else math.inf
                ),
                "p99_to_median_ratios": p99_ratios,
                "p99_check": {
                    "passed": p99_passed,
                    "limit": P99_TO_MEDIAN_LIMIT,
                },
            }
        )
        if p99_passed:
            candidate["status"] = "accepted"
        else:
            candidate.update(
                {
                    "status": "rejected",
                    "reason": "p99/median exceeded the stability limit",
                }
            )

    selection = select_grid_with_effect_band(
        candidates,
        production_grid=production_grid,
    )
    winner_grid = int(selection["winner_grid_ctas"])
    tuning = {
        "primary_point": {"mode": "block16", "tokens": 16},
        "selection_metric": (
            "lowest median of repeat medians outside a minimum-effect band "
            "equal to the larger within-candidate median dispersion"
        ),
        "candidate_orders": [list(order) for order in candidate_orders],
        "selection": selection,
        "winner_grid_ctas": winner_grid,
        "candidates": candidates,
        "cluster_candidates": [
            {
                "cluster_size": 1,
                "status": "evaluated",
                "reason": "production tcgen05 CTA-group contract",
            },
            {
                "cluster_size": 2,
                "status": "rejected",
                "reason": (
                    "The routed expert emits handwritten "
                    "tcgen05.mma.cta_group::1 mixed-MXFP4 instructions and "
                    "every stage owns a tensor_allocator<1, 1>. A two-CTA "
                    "expert would require cta_group::2 allocation, MMA, "
                    "commit, multicast TMA, and paired deallocation throughout; "
                    "changing only the launch cluster would not evaluate that "
                    "contract and changing only the allocator would be invalid."
                ),
            },
        ],
    }
    graph_pools.clear()
    _barrier(device)
    torch.cuda.empty_cache()
    return winner_grid, tuning, all_correctness


def _measure_tables(
    modules: RuntimeModules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    grid_ctas: int,
    pool_size: int,
    expert_weight_bytes: int,
    l2_bytes: int,
    *,
    warmup_count: int,
    sample_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    _set_grid(modules, grid_ctas)
    tables: dict[str, list[dict[str, Any]]] = {}
    correctness: list[dict[str, Any]] = []
    for mode, shapes in SHAPE_GROUPS.items():
        rows: list[dict[str, Any]] = []
        for tokens in shapes:
            entries = _capture_graph_pool(
                modules,
                workspace,
                base_weights,
                device,
                tokens,
                pool_size,
            )
            metadata = _graph_route_metadata(
                entries,
                expert_weight_bytes=expert_weight_bytes,
                l2_cache_bytes=l2_bytes,
            )
            for entry in entries:
                metrics = _correctness_call(
                    modules,
                    workspace,
                    entry.weights,
                    entry.hidden,
                )
                correctness.append(
                    {
                        "phase": "final_before_timing",
                        "grid_ctas": grid_ctas,
                        "mode": mode,
                        "tokens": tokens,
                        "pool_index": entry.pool_index,
                        "route_assignments": [
                            list(token)
                            for token in entry.route_assignments
                        ],
                        "distinct_experts_per_replay": len(
                            entry.distinct_experts
                        ),
                        "routed_queue_units_per_replay": metadata[
                            "routed_queue_units_per_replay"
                        ],
                        "routed_expert_working_set_bytes_per_replay": (
                            metadata[
                                "routed_expert_working_set_bytes_per_replay"
                            ]
                        ),
                        **metrics,
                    }
                )
            timing = _measure_graph_pool(
                entries,
                workspace,
                device,
                warmup_count=warmup_count,
                sample_count=sample_count,
            )
            post_timing = _verify_graph_result(
                modules,
                entries,
                workspace,
            )
            rows.append(
                {
                    "mode": mode,
                    "tokens": tokens,
                    "batch_size": mode_batch_size(mode, tokens),
                    "grid_ctas": grid_ctas,
                    "cluster_size": 1,
                    "launch_count": 1,
                    "graph_pool_size": pool_size,
                    "l2_cache_bytes": l2_bytes,
                    "warmup_count": warmup_count,
                    "sample_count": sample_count,
                    **metadata,
                    **timing,
                    "post_timing_graph_checks": post_timing,
                }
            )
            del entries
            _barrier(device)
            torch.cuda.empty_cache()
        tables[mode] = rows
    return tables, correctness


def _workspace_stats(
    modules: RuntimeModules,
    workspace: Any,
    weights: Any,
    device: torch.device,
    grid_ctas: int,
) -> dict[str, Any]:
    routed = modules.data.build_routed_input(
        weights,
        device,
        16,
        0,
    )
    modules.runtime.decode_step(workspace, routed.weights, routed.hidden)
    _barrier(device)
    before = torch.cuda.memory_allocated(device)
    with modules.runtime.recorded_allocator_events(device) as events:
        modules.kimi.kimi_k3_decode(
            modules.runtime.CONFIG,
            workspace,
            routed.weights,
            routed.hidden,
        )
    after = torch.cuda.memory_allocated(device)
    names = modules.runtime.profiled_kernel_names(
        lambda: modules.kimi.kimi_k3_decode(
            modules.runtime.CONFIG,
            workspace,
            routed.weights,
            routed.hidden,
        )
    )
    if len(names) != 1 or "kimi_k3_decode_persistent_kernel" not in names[0]:
        raise AssertionError(names)
    if events or before != after:
        raise AssertionError((events, before, after))
    return {
        "grid_ctas": grid_ctas,
        "workspace_scratch_bytes": _tensor_bytes(workspace.scratch),
        "workspace_collective_bytes": _tensor_bytes(
            workspace.collective_buffer
        ),
        "workspace_output_mailbox_bytes": _tensor_bytes(
            workspace.output_mailbox
        ),
        "workspace_barrier_bytes": _tensor_bytes(workspace.barrier_buffer),
        "workspace_control_bytes": (
            _tensor_bytes(workspace.barrier_target)
            + _tensor_bytes(workspace.error_flag)
        ),
        "prepared_weight_bytes_per_rank": _prepared_weight_bytes(weights),
        "hot_path_memory_allocated_before": before,
        "hot_path_memory_allocated_after": after,
        "hot_path_allocator_events": events,
        "profiled_launch_count": len(names),
        "profiled_kernel_names": names,
    }


def _run_gpu(
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> None:
    if warmup_count < WARMUP_COUNT:
        raise ValueError(f"warmup count must be at least {WARMUP_COUNT}")
    if sample_count < SAMPLE_COUNT:
        raise ValueError(f"sample count must be at least {SAMPLE_COUNT}")
    if repeats < TUNING_REPEATS:
        raise ValueError(f"tuning repeats must be at least {TUNING_REPEATS}")
    previous_guard = os.environ.get("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    modules = _runtime_modules()
    initialized = False
    production_grid = GRID_CANDIDATES[-1]
    try:
        rank, _, device = _init_distributed()
        initialized = True
        properties = torch.cuda.get_device_properties(device)
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime_candidates = tuple(
            modules.extension._kimi_k3_decode_benchmark_grids()
        )
        if runtime_candidates != GRID_CANDIDATES:
            raise AssertionError((runtime_candidates, GRID_CANDIDATES))

        workspace = modules.kimi.get_kimi_k3_decode_workspace(
            dist.group.WORLD,
            device=device,
        )
        weights = modules.data.build_weights(device, rank)
        l2_bytes = _l2_cache_bytes(properties)
        expert_bytes = _expert_weight_bytes(weights)
        for tokens in {
            tokens
            for shapes in SHAPE_GROUPS.values()
            for tokens in shapes
        }:
            metadata = route_metadata(
                tokens=tokens,
                expert_weight_bytes=expert_bytes,
                l2_cache_bytes=l2_bytes,
            )
            if not metadata[
                "pool_wide_routed_expert_working_set_exceeds_l2"
            ]:
                raise AssertionError((tokens, metadata))

        production_grid = int(
            modules.extension._kimi_k3_decode_grid_shape()[0]
        )
        winner_grid, tuning, tuning_correctness = _tune_grids(
            modules,
            workspace,
            weights,
            device,
            properties,
            GRAPH_POOL_SIZE,
            expert_bytes,
            l2_bytes,
            production_grid,
            warmup_count=warmup_count,
            sample_count=sample_count,
            repeats=repeats,
        )
        _set_grid(modules, production_grid)
        tables, final_correctness = _measure_tables(
            modules,
            workspace,
            weights,
            device,
            winner_grid,
            GRAPH_POOL_SIZE,
            expert_bytes,
            l2_bytes,
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
        stats = _workspace_stats(
            modules,
            workspace,
            weights,
            device,
            winner_grid,
        )
        manifest = build_manifest(dry_run=False)
        manifest.update(
            {
                "warmup_count": warmup_count,
                "sample_count": sample_count,
                "tuning_repeats": repeats,
                "mok_version": modules.mok.__version__,
                "gpu": {
                    "name": properties.name,
                    "compute_capability": "sm_103",
                    "sm_count": properties.multi_processor_count,
                    "total_memory_bytes": properties.total_memory,
                    "l2_cache_bytes": l2_bytes,
                },
                "clocks": gpu_clocks() if rank == 0 else [],
                "production_default_grid_ctas": production_grid,
                "selected_grid_ctas": winner_grid,
                "cluster_size": 1,
                "pool_policy": {
                    **manifest["pool_policy"],
                    "expert_weight_bytes_per_rank": expert_bytes,
                    "l2_cache_bytes": l2_bytes,
                },
                "artifact_files": list(ARTIFACT_FILES),
            }
        )

        if rank == 0:
            write_json(output_dir / "manifest.json", manifest)
            write_latency_table(
                output_dir,
                "latency_raw_decode",
                tables["raw_decode"],
            )
            write_latency_table(
                output_dir,
                "latency_block8",
                tables["block8"],
            )
            write_latency_table(
                output_dir,
                "latency_block16",
                tables["block16"],
            )
            write_json(
                output_dir / "correctness.json",
                {
                    "benchmark": "kimi_k3_decode",
                    "tuning_candidate_checks": tuning_correctness,
                    "final_before_timing": final_correctness,
                },
            )
            write_json(output_dir / "workspace_stats.json", stats)
            write_json(output_dir / "tuning.json", tuning)
            missing = [
                name
                for name in ARTIFACT_FILES
                if not (output_dir / name).is_file()
            ]
            if missing:
                raise AssertionError(
                    f"missing benchmark artifacts: {missing}"
                )
            print(json.dumps(manifest, indent=2, sort_keys=True))
        _barrier(device)
    finally:
        try:
            _set_grid(modules, production_grid)
        finally:
            if previous_guard is None:
                os.environ.pop(
                    "MOK_KIMI_K3_ENABLE_GRID_TUNING",
                    None,
                )
            else:
                os.environ[
                    "MOK_KIMI_K3_ENABLE_GRID_TUNING"
                ] = previous_guard
        if initialized:
            modules.kimi.clear_kimi_k3_decode_workspace_cache()
            dist.destroy_process_group()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kimi_k3_decode_artifacts"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--tuning-repeats", type=int, default=TUNING_REPEATS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.dry_run:
        manifest = write_dry_run(args.output_dir)
        print(json.dumps(manifest, sort_keys=True))
        return
    _run_gpu(
        args.output_dir,
        warmup_count=args.warmup_count,
        sample_count=args.sample_count,
        repeats=args.tuning_repeats,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
