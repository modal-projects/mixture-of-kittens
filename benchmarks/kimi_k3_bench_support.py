"""What the decode benchmark measures with, under everything that measures.

The runtime modules it reaches through, the distributed rendezvous, the
workspace and weight byte counts it reports, the correctness sweep every
measured configuration has to pass first, and the graph pool it replays. None
of it decides anything; the drivers in ``bench_kimi_k3_decode.py`` do, and they
are readable only once this is out from in front of them.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, fields
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.kimi_k3_decode_inputs import (
    GRAPH_POOL_SIZE,
    route_metadata,
)

from benchmarks.kimi_k3_decode_output import (
    SHAPE_GROUPS,
    TP_SIZE,
)

from benchmarks.kimi_k3_timing import (
    rank_max_samples,
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
