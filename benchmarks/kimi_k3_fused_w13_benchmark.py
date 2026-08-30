"""Private TP8 full-step benchmark for transformed fused-W13 tensors."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.frameworks.kimi_k3_adapter_common import native_weights
from benchmarks.kimi_k3_decode_inputs import GRAPH_POOL_SIZE
from benchmarks.kimi_k3_timing import percentile, summarize_rank_max
from mok.kimi_k3 import (
    clear_kimi_k3_decode_workspace_cache,
    get_kimi_k3_decode_workspace,
)

from . import kimi_k3_decode_data as data
from . import kimi_k3_decode_runtime as runtime


TOKENS = (16, 128)
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5
TP_SIZE = 8
BENCHMARK_GUARD = "MOK_KIMI_K3_ENABLE_FUSED_W13_BENCHMARK"


@dataclass(slots=True)
class GraphEntry:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    weights: Any


def _init_distributed() -> tuple[int, torch.device]:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch the fused-W13 benchmark with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP_SIZE:
        raise RuntimeError(f"fused-W13 benchmark requires TP{TP_SIZE}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("fused-W13 benchmark requires an SM103 B300")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return rank, device


def _barrier(device: torch.device) -> None:
    dist.barrier(
        async_op=True,
        device_ids=[device.index],
    ).block_current_stream()
    torch.cuda.synchronize(device)


def _call(
    variant: str,
    workspace: Any,
    weights: Any,
    fused: Any,
    hidden: torch.Tensor,
) -> torch.Tensor:
    if variant == "production":
        return runtime.decode_device_step(workspace, weights, hidden)
    if variant == "fused_w13":
        return runtime.decode_fused_w13_benchmark_device_step(
            workspace,
            weights,
            fused.w13_weight,
            fused.w13_weight_scale,
            hidden,
        )
    raise ValueError(f"unknown fused-W13 benchmark variant {variant!r}")


def _capture(
    variant: str,
    workspace: Any,
    base_weights: Any,
    fused: Any,
    device: torch.device,
    tokens: int,
) -> list[GraphEntry]:
    entries: list[GraphEntry] = []
    for pool_index in range(GRAPH_POOL_SIZE):
        routed = data.build_routed_input(
            base_weights,
            device,
            tokens,
            pool_index,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _call(
                variant,
                workspace,
                routed.weights,
                fused,
                routed.hidden,
            )
        entries.append(GraphEntry(graph, routed.hidden, routed.weights))
    _barrier(device)
    return entries


def _measure(
    entries: list[GraphEntry],
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
    local = torch.tensor(
        [
            start.elapsed_time(end)
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, local)
    rank_samples = [samples.cpu().tolist() for samples in gathered]
    runtime.check_decode_error(workspace)
    return {
        **summarize_rank_max(rank_samples),
        "rank_samples_ms": rank_samples,
    }


def _phase_cycles(
    variant: str,
    workspace: Any,
    weights: Any,
    fused: Any,
    hidden: torch.Tensor,
    device: torch.device,
) -> dict[str, int]:
    with runtime.phase_profiling():
        _call(variant, workspace, weights, fused, hidden)
        torch.cuda.synchronize(device)
    runtime.check_decode_error(workspace)
    return runtime.phase_clock_cycles(workspace)


def _verdict(
    production: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    production_medians = [float(row["median_ms"]) for row in production]
    candidate_medians = [float(row["median_ms"]) for row in candidate]
    production_center = percentile(production_medians, 0.5)
    candidate_center = percentile(candidate_medians, 0.5)
    production_dispersion = max(production_medians) - min(production_medians)
    candidate_dispersion = max(candidate_medians) - min(candidate_medians)
    effect_band = max(production_dispersion, candidate_dispersion)
    improvement = production_center - candidate_center
    return {
        "production_median_of_repeats_ms": production_center,
        "fused_w13_median_of_repeats_ms": candidate_center,
        "production_median_dispersion_ms": production_dispersion,
        "fused_w13_median_dispersion_ms": candidate_dispersion,
        "effect_band_ms": effect_band,
        "improvement_ms": improvement,
        "improvement_fraction": improvement / production_center,
        "measurably_faster": improvement > effect_band,
    }


def run(
    output_path: Path,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup_count < 1 or sample_count < 1 or repeats < 2:
        raise ValueError("warmups and samples must be positive; repeats >= 2")
    os.environ[BENCHMARK_GUARD] = "1"
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    rank, device = _init_distributed()
    workspace = None
    try:
        workspace = get_kimi_k3_decode_workspace(
            dist.group.WORLD,
            device=device,
        )
        base_weights = data.build_weights(device, rank)
        fused = native_weights(base_weights)
        rows: list[dict[str, Any]] = []
        for tokens in TOKENS:
            routed = data.build_routed_input(
                base_weights,
                device,
                tokens,
                0,
            )
            production_output = _call(
                "production",
                workspace,
                routed.weights,
                fused,
                routed.hidden,
            ).clone()
            candidate_output = _call(
                "fused_w13",
                workspace,
                routed.weights,
                fused,
                routed.hidden,
            ).clone()
            torch.cuda.synchronize(device)
            runtime.check_decode_error(workspace)
            difference = (
                candidate_output.float() - production_output.float()
            ).abs()
            parity = {
                "exact": bool(torch.equal(candidate_output, production_output)),
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
            }
            if not parity["exact"]:
                raise AssertionError(
                    f"fused-W13 full step differs at M={tokens}: {parity}"
                )

            profiles = {
                variant: _phase_cycles(
                    variant,
                    workspace,
                    routed.weights,
                    fused,
                    routed.hidden,
                    device,
                )
                for variant in ("production", "fused_w13")
            }
            kernel_names = {
                variant: runtime.profiled_kernel_names(
                    lambda variant=variant: _call(
                        variant,
                        workspace,
                        routed.weights,
                        fused,
                        routed.hidden,
                    )
                )
                for variant in ("production", "fused_w13")
            }
            for variant, names in kernel_names.items():
                if (
                    len(names) != 1
                    or "kimi_k3_decode_persistent_kernel" not in names[0]
                ):
                    raise AssertionError((variant, names))

            graphs = {
                variant: _capture(
                    variant,
                    workspace,
                    base_weights,
                    fused,
                    device,
                    tokens,
                )
                for variant in ("production", "fused_w13")
            }
            measurements: dict[str, list[dict[str, Any]]] = {
                "production": [],
                "fused_w13": [],
            }
            for repeat in range(repeats):
                order = (
                    ("production", "fused_w13")
                    if repeat % 2 == 0
                    else ("fused_w13", "production")
                )
                for variant in order:
                    measurements[variant].append(
                        _measure(
                            graphs[variant],
                            workspace,
                            device,
                            warmup_count=warmup_count,
                            sample_count=sample_count,
                        )
                    )
            verdict = _verdict(
                measurements["production"],
                measurements["fused_w13"],
            )
            rows.append(
                {
                    "tokens": tokens,
                    "parity": parity,
                    "kernel_names": kernel_names,
                    "phase_cycles": profiles,
                    "measurements": measurements,
                    "verdict": verdict,
                }
            )

        result = {
            "benchmark": "kimi_k3_fused_w13_full_step",
            "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": TP_SIZE,
            "guard": BENCHMARK_GUARD,
            "private_operator": "_kimi_k3_decode_fused_w13_benchmark",
            "tokens": list(TOKENS),
            "graph_pool_size": GRAPH_POOL_SIZE,
            "warmup_count": warmup_count,
            "sample_count": sample_count,
            "repeats": repeats,
            "rows": rows,
        }
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        _barrier(device)
        return result
    finally:
        if workspace is not None:
            clear_kimi_k3_decode_workspace_cache()
        data.clear_shared_router_cache()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    arguments = parser.parse_args()
    run(
        arguments.output,
        warmup_count=arguments.warmup_count,
        sample_count=arguments.sample_count,
        repeats=arguments.repeats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
