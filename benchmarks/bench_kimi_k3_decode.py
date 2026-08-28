"""Reproducible TP8 latency benchmark for the one-launch Kimi K3 decode path."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.kimi_k3_timing import (
    percentile,
    rank_max_samples,
    summarize_rank_max,
)

RAW_DECODE_SHAPES = tuple(range(1, 9))
BLOCK8_SHAPES = tuple(range(8, 65, 8))
BLOCK16_SHAPES = tuple(range(16, 129, 16))
SHAPE_GROUPS = {
    "raw_decode": RAW_DECODE_SHAPES,
    "block8": BLOCK8_SHAPES,
    "block16": BLOCK16_SHAPES,
}
GRID_CANDIDATES = (64, 96, 128, 148)
TP_SIZE = 8
TOPK = 16
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
TUNING_REPEATS = 3
P99_TO_MEDIAN_LIMIT = 2.0
MIB = 1024**2

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


@dataclass(slots=True)
class GraphInput:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    weights: Any
    selected_experts: tuple[int, ...]


@dataclass(slots=True)
class RuntimeModules:
    mok: ModuleType
    extension: ModuleType
    kimi: ModuleType
    support: ModuleType


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
    """Build the hardware-independent portion of the benchmark manifest."""
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
            "routing": "disjoint contiguous groups of 16 experts",
            "working_set": "strictly_greater_than_l2",
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
            "routed_weight_format": "MXFP4 E2M1 with E8M0 group-32 scales",
        },
        "input_seed": "77770000 + tokens * 1000 + pool_index",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_dry_run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(dry_run=True)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _runtime_modules() -> RuntimeModules:
    # These modules import the CUDA extension. Keeping their import behind the
    # non-dry-run boundary lets shape and manifest validation run on CPU hosts.
    mok = importlib.import_module("mok")
    return RuntimeModules(
        mok=mok,
        extension=importlib.import_module("mok._C"),
        kimi=importlib.import_module("mok.kimi_k3"),
        support=importlib.import_module("tests.kimi_k3_decode_support"),
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
        for field in dataclasses.fields(weights)
        if isinstance((value := getattr(weights, field.name)), torch.Tensor)
    )


def _expert_weight_bytes(weights: Any) -> int:
    return sum(
        _tensor_bytes(getattr(weights, name)[0])
        for name in (
            "expert_w1_packed",
            "expert_w1_scale",
            "expert_w3_packed",
            "expert_w3_scale",
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


def _pool_size(l2_bytes: int, expert_bytes: int) -> int:
    experts_per_input = TOPK
    pool_size = l2_bytes // (experts_per_input * expert_bytes) + 1
    maximum = 896 // experts_per_input
    if not 1 <= pool_size <= maximum:
        raise RuntimeError(
            f"cannot exceed {l2_bytes} L2 bytes with {maximum} input groups"
        )
    return pool_size


def _pinned_input(
    modules: RuntimeModules,
    base_weights: Any,
    device: torch.device,
    tokens: int,
    pool_index: int,
) -> tuple[torch.Tensor, Any, tuple[int, ...]]:
    begin = pool_index * TOPK
    selected = tuple(range(begin, begin + TOPK))
    if selected[-1] >= 896:
        raise ValueError("input pool selects an expert outside [0, 896)")
    hidden = modules.support.hidden_states(
        device,
        tokens,
        seed=77_770_000 + tokens * 1000 + pool_index,
    )
    correction_bias = torch.zeros(896, dtype=torch.float32, device=device)
    correction_bias[begin:begin + TOPK] = 8.0
    weights = dataclasses.replace(
        base_weights,
        router_correction_bias=correction_bias,
    )
    expert_ids, _ = modules.kimi.kimi_k3_router_reference(
        hidden,
        weights.router_weight,
        correction_bias,
    )
    actual = {int(expert) for expert in torch.unique(expert_ids).tolist()}
    if actual != set(selected):
        raise AssertionError((pool_index, selected, sorted(actual)))
    return hidden, weights, selected


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
    expected = modules.support.decode_reference(hidden, weights)
    actual = modules.support.decode_step(workspace, weights, hidden)
    torch.cuda.synchronize(hidden.device)
    relative_l1, cosine, maximum = modules.support.assert_decode_close(
        actual,
        expected,
    )
    modules.support.assert_identical_across_ranks(actual)
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
) -> list[dict[str, Any]]:
    _set_grid(modules, grid_ctas)
    rows: list[dict[str, Any]] = []
    for mode, shapes in SHAPE_GROUPS.items():
        for tokens in shapes:
            hidden, weights, selected = _pinned_input(
                modules,
                base_weights,
                device,
                tokens,
                0,
            )
            metrics = _correctness_call(
                modules,
                workspace,
                weights,
                hidden,
            )
            rows.append(
                {
                    "grid_ctas": grid_ctas,
                    "mode": mode,
                    "tokens": tokens,
                    "selected_experts": list(selected),
                    **metrics,
                }
            )
            del hidden, weights
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
        hidden, weights, selected = _pinned_input(
            modules,
            base_weights,
            device,
            tokens,
            pool_index,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            modules.kimi.kimi_k3_decode(
                modules.support.CONFIG,
                workspace,
                weights,
                hidden,
            )
        entries.append(GraphInput(graph, hidden, weights, selected))
    _barrier(device)
    return entries


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
    sample_count: int,
) -> dict[str, float]:
    final_entry = entries[(sample_count - 1) % len(entries)]
    expected = modules.support.decode_reference(
        final_entry.hidden,
        final_entry.weights,
    )
    tokens = final_entry.hidden.shape[0]
    actual = workspace.output_mailbox.view(128, 7168)[:tokens]
    relative_l1, cosine, maximum = modules.support.assert_decode_close(
        actual,
        expected,
    )
    modules.support.assert_identical_across_ranks(actual)
    return {
        "relative_l1": relative_l1,
        "cosine_similarity": cosine,
        "max_abs": maximum,
    }


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
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
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
            continue

        correctness = _correctness_sweep(
            modules,
            workspace,
            base_weights,
            device,
            grid_ctas,
        )
        all_correctness.extend(correctness)
        candidate["full_correctness"] = {
            "passed": True,
            "shape_count": len(correctness),
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
        repeated = [
            _measure_graph_pool(
                entries,
                workspace,
                device,
                warmup_count=warmup_count,
                sample_count=sample_count,
            )
            for _ in range(repeats)
        ]
        graph_metrics = _verify_graph_result(
            modules,
            entries,
            workspace,
            sample_count,
        )
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
                    "correctness": graph_metrics,
                },
                "repeats": [
                    {
                        key: value
                        for key, value in repeat.items()
                        if key != "rank_max_samples_ms"
                    }
                    for repeat in repeated
                ],
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
        candidates.append(candidate)
        del entries
        _barrier(device)
        torch.cuda.empty_cache()

    accepted = [
        candidate
        for candidate in candidates
        if candidate["status"] == "accepted"
    ]
    if not accepted:
        raise RuntimeError("no grid candidate passed tuning gates")
    winner = min(
        accepted,
        key=lambda candidate: candidate["median_of_repeat_medians_ms"],
    )
    winner_grid = int(winner["grid_ctas"])
    tuning = {
        "primary_point": {"mode": "block16", "tokens": 16},
        "selection_metric": "lowest median of three repeat medians",
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
    return winner_grid, tuning, all_correctness


def _mode_batch_size(mode: str, tokens: int) -> int:
    if mode == "raw_decode":
        return tokens
    if mode == "block8":
        return tokens // 8
    if mode == "block16":
        return tokens // 16
    raise ValueError(f"unknown benchmark mode {mode!r}")


def _measure_tables(
    modules: RuntimeModules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    grid_ctas: int,
    pool_size: int,
    selected_expert_bytes: int,
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
            metrics = _correctness_call(
                modules,
                workspace,
                entries[0].weights,
                entries[0].hidden,
            )
            correctness.append(
                {
                    "phase": "final_before_timing",
                    "grid_ctas": grid_ctas,
                    "mode": mode,
                    "tokens": tokens,
                    "selected_experts": list(entries[0].selected_experts),
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
            rows.append(
                {
                    "mode": mode,
                    "tokens": tokens,
                    "batch_size": _mode_batch_size(mode, tokens),
                    "grid_ctas": grid_ctas,
                    "cluster_size": 1,
                    "launch_count": 1,
                    "graph_pool_size": pool_size,
                    "selected_expert_count": pool_size * TOPK,
                    "selected_expert_weight_bytes": selected_expert_bytes,
                    "l2_cache_bytes": l2_bytes,
                    "warmup_count": warmup_count,
                    **timing,
                }
            )
            del entries
            _barrier(device)
            torch.cuda.empty_cache()
        tables[mode] = rows
    return tables, correctness


def _write_latency_table(
    output_dir: Path,
    filename_stem: str,
    rows: list[dict[str, Any]],
) -> None:
    _write_json(
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
        "selected_expert_count",
        "selected_expert_weight_bytes",
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
            writer.writerow({column: row[column] for column in columns})


def _gpu_clocks() -> list[dict[str, str]] | dict[str, str]:
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
        dict(zip(fields, (value.strip() for value in line.split(",")), strict=True))
        for line in lines
    ]


def _workspace_stats(
    modules: RuntimeModules,
    workspace: Any,
    weights: Any,
    device: torch.device,
    grid_ctas: int,
) -> dict[str, Any]:
    hidden, pinned_weights, _ = _pinned_input(
        modules,
        weights,
        device,
        16,
        0,
    )
    modules.support.decode_step(workspace, pinned_weights, hidden)
    _barrier(device)
    before = torch.cuda.memory_allocated(device)
    with modules.support.recorded_allocator_events(device) as events:
        modules.kimi.kimi_k3_decode(
            modules.support.CONFIG,
            workspace,
            pinned_weights,
            hidden,
        )
    after = torch.cuda.memory_allocated(device)
    names = modules.support.profiled_kernel_names(
        lambda: modules.kimi.kimi_k3_decode(
            modules.support.CONFIG,
            workspace,
            pinned_weights,
            hidden,
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
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    modules = _runtime_modules()
    rank, _, device = _init_distributed()
    properties = torch.cuda.get_device_properties(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = modules.kimi.get_kimi_k3_decode_workspace(
        dist.group.WORLD,
        device=device,
    )
    weights = modules.support._build_weights(device, rank)
    l2_bytes = _l2_cache_bytes(properties)
    expert_bytes = _expert_weight_bytes(weights)
    pool_size = _pool_size(l2_bytes, expert_bytes)
    selected_expert_bytes = pool_size * TOPK * expert_bytes
    if selected_expert_bytes <= l2_bytes:
        raise AssertionError((selected_expert_bytes, l2_bytes))

    production_grid = int(
        modules.extension._kimi_k3_decode_grid_shape()[0]
    )
    winner_grid, tuning, tuning_correctness = _tune_grids(
        modules,
        workspace,
        weights,
        device,
        properties,
        pool_size,
        warmup_count=warmup_count,
        sample_count=sample_count,
        repeats=repeats,
    )
    tables, final_correctness = _measure_tables(
        modules,
        workspace,
        weights,
        device,
        winner_grid,
        pool_size,
        selected_expert_bytes,
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
            "clocks": _gpu_clocks() if rank == 0 else [],
            "production_default_grid_ctas": production_grid,
            "selected_grid_ctas": winner_grid,
            "cluster_size": 1,
            "pool_policy": {
                **manifest["pool_policy"],
                "pool_size": pool_size,
                "experts_per_input": TOPK,
                "selected_expert_count": pool_size * TOPK,
                "expert_weight_bytes_per_rank": expert_bytes,
                "selected_expert_weight_bytes_per_rank": (
                    selected_expert_bytes
                ),
                "l2_cache_bytes": l2_bytes,
            },
            "artifact_files": list(ARTIFACT_FILES),
        }
    )

    if rank == 0:
        _write_json(output_dir / "manifest.json", manifest)
        _write_latency_table(
            output_dir,
            "latency_raw_decode",
            tables["raw_decode"],
        )
        _write_latency_table(
            output_dir,
            "latency_block8",
            tables["block8"],
        )
        _write_latency_table(
            output_dir,
            "latency_block16",
            tables["block16"],
        )
        _write_json(
            output_dir / "correctness.json",
            {
                "benchmark": "kimi_k3_decode",
                "tuning_candidate_checks": tuning_correctness,
                "final_before_timing": final_correctness,
            },
        )
        _write_json(output_dir / "workspace_stats.json", stats)
        _write_json(output_dir / "tuning.json", tuning)
        missing = [
            name for name in ARTIFACT_FILES
            if not (output_dir / name).is_file()
        ]
        if missing:
            raise AssertionError(f"missing benchmark artifacts: {missing}")
        print(json.dumps(manifest, indent=2, sort_keys=True))

    _barrier(device)
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
        manifest = _write_dry_run(args.output_dir)
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
