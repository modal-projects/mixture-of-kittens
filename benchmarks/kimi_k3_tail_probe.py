"""Temporary TP8 structural probe for the Kimi K3 fused tail."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import platform
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks import kimi_k3_decode_data as data
from benchmarks import kimi_k3_decode_runtime as runtime
from benchmarks.kimi_k3_decode_inputs import GRAPH_POOL_SIZE
from benchmarks.kimi_k3_timing import (
    rank_max_samples,
    replay_samples,
    summarize_rank_max,
)
from mok import _C
from mok import kimi_k3 as kimi
from mok.ops import _kimi_k3_tail


TP_SIZE = 8
TOKEN_COUNTS = (16, 32, 128)
TAIL_PHASES = (
    "entry_rank_rendezvous",
    "reduce_entry_wait",
    "routed_multimem_reduce_and_squares",
    "rmsnorm_scale_weight_store",
    "shared_multimem_reduce",
    "reduce_publish",
    "shard_reduce_wait",
    "latent_up_shard_mma",
    "mailbox_multicast_and_beta",
    "shard_publish",
    "coordinator_shard_wait",
    "exit_rank_rendezvous",
    "drain",
)
HANDOFF_PHASES = (
    "reduce_entry_wait",
    "reduce_publish",
    "shard_reduce_wait",
    "shard_publish",
    "coordinator_shard_wait",
)
COMPUTE_PHASES = (
    "routed_multimem_reduce_and_squares",
    "rmsnorm_scale_weight_store",
    "shared_multimem_reduce",
    "latent_up_shard_mma",
    "mailbox_multicast_and_beta",
)
NATIVE_POST_EXPERT_START = "oneshotAllreduceFusionKernel"
DEBUG_LOG_PATH = Path("/opt/cursor/logs/debug.log")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    payload: dict[str, Any],
) -> None:
    # region agent log
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": payload,
                    "timestamp": time.time_ns() // 1_000_000,
                },
                sort_keys=True,
            )
            + "\n"
        )
    # endregion


def _init_distributed() -> tuple[int, torch.device]:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch the tail probe with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP_SIZE:
        raise RuntimeError(f"the tail probe requires TP{TP_SIZE}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the tail probe requires SM103 B300 GPUs")
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


def _gather_objects(payload: Any) -> list[Any]:
    gathered: list[Any] = [None] * TP_SIZE
    dist.all_gather_object(gathered, payload)
    return gathered


def _gathered_rank_samples(
    local_samples: Sequence[float],
    device: torch.device,
) -> list[list[float]]:
    local = torch.tensor(local_samples, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, local)
    return [samples.cpu().tolist() for samples in gathered]


def _measure_graphs(
    graphs: Sequence[torch.cuda.CUDAGraph],
    device: torch.device,
    *,
    warmup_count: int,
    sample_count: int,
) -> tuple[dict[str, int | float], list[float]]:
    _barrier(device)
    local = replay_samples(
        lambda iteration: graphs[iteration % len(graphs)].replay(),
        warmup_count=warmup_count,
        sample_count=sample_count,
        settle_count=len(graphs),
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=lambda: torch.cuda.synchronize(device),
    )
    rank_samples = _gathered_rank_samples(local, device)
    return summarize_rank_max(rank_samples), rank_max_samples(rank_samples)


def _tail_call(
    workspace: kimi.KimiK3DecodeWorkspace,
    weights: kimi.KimiK3DecodeWeights,
    active_tokens: int,
) -> torch.Tensor:
    return _kimi_k3_tail(
        weights.routed_latent_rmsnorm_weight,
        weights.routed_expert_up_proj,
        workspace.collective_buffer,
        workspace.collective_ptrs,
        workspace.collective_multicast_ptr,
        workspace.output_mailbox,
        workspace.output_mailbox_ptrs,
        workspace.output_mailbox_multicast_ptr,
        workspace.barrier_buffer,
        workspace.barrier_ptrs,
        workspace.barrier_multicast_ptr,
        workspace.barrier_target,
        workspace.scratch,
        workspace.error_flag,
        workspace.tp_rank,
        active_tokens,
        workspace.workspace_signature,
    )


def _capture_custom_graphs(
    workspace: kimi.KimiK3DecodeWorkspace,
    pool: Sequence[data.RoutedInput],
) -> list[torch.cuda.CUDAGraph]:
    graphs: list[torch.cuda.CUDAGraph] = []
    for entry in pool:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kimi.kimi_k3_decode(
                runtime.CONFIG,
                workspace,
                entry.weights,
                entry.hidden,
            )
        graphs.append(graph)
    return graphs


def _capture_private_tail_graph(
    workspace: kimi.KimiK3DecodeWorkspace,
    entry: data.RoutedInput,
) -> torch.cuda.CUDAGraph:
    runtime.decode_step(workspace, entry.weights, entry.hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _tail_call(workspace, entry.weights, entry.hidden.shape[0])
    return graph


def _profile_integrated_tail(
    workspace: kimi.KimiK3DecodeWorkspace,
    entry: data.RoutedInput,
    device: torch.device,
) -> list[dict[str, int]]:
    with runtime.phase_profiling():
        runtime.decode_step(workspace, entry.weights, entry.hidden)
        torch.cuda.synchronize(device)
        rows = runtime.tail_clock_rows(workspace, launch_ctas=148)
    return rows


def _profile_private_tail(
    workspace: kimi.KimiK3DecodeWorkspace,
    entry: data.RoutedInput,
    role_ctas: int,
    device: torch.device,
) -> list[dict[str, int]]:
    runtime.decode_step(workspace, entry.weights, entry.hidden)
    torch.cuda.synchronize(device)
    with runtime.tail_profiling():
        _tail_call(workspace, entry.weights, entry.hidden.shape[0])
        torch.cuda.synchronize(device)
        rows = runtime.tail_clock_rows(workspace, launch_ctas=role_ctas)
    return rows


def _summarize_tail_rows(
    rows: Sequence[dict[str, int]],
    *,
    role_ctas: int,
    clock_ghz: float,
) -> dict[str, Any]:
    if len(rows) < role_ctas:
        raise ValueError((len(rows), role_ctas))
    participant_rows = rows[:role_ctas]
    non_tail_rows = rows[role_ctas:]
    phase_cycles = {
        name: sum(int(row[name]) for row in rows)
        for name in TAIL_PHASES
    }
    participant_total = sum(int(row["total"]) for row in participant_rows)
    non_tail_total = sum(int(row["total"]) for row in non_tail_rows)
    component_total = sum(phase_cycles.values())
    cycles_per_microsecond = clock_ghz * 1000.0
    participant_values = [int(row["total"]) for row in participant_rows]
    return {
        "launch_ctas": len(rows),
        "participating_ctas": role_ctas,
        "non_tail_ctas": len(non_tail_rows),
        "summed_participating_cycles": participant_total,
        "summed_non_tail_cycles": non_tail_total,
        "summed_all_cta_cycles": participant_total + non_tail_total,
        "mean_participating_cycles": participant_total / role_ctas,
        "max_participating_cycles": max(participant_values),
        "mean_participating_us_at_clock": (
            participant_total / role_ctas / cycles_per_microsecond
        ),
        "max_participating_us_at_clock": (
            max(participant_values) / cycles_per_microsecond
        ),
        "summed_non_tail_idle_cycles": sum(
            int(row["non_tail_idle"]) for row in non_tail_rows
        ),
        "phase_cycles": phase_cycles,
        "summed_named_component_cycles": component_total,
        "unattributed_participating_cycles": max(
            0,
            participant_total - component_total,
        ),
        "handoff_cycles": sum(phase_cycles[name] for name in HANDOFF_PHASES),
        "compute_cycles": sum(phase_cycles[name] for name in COMPUTE_PHASES),
        "rendezvous_cycles": (
            phase_cycles["entry_rank_rendezvous"]
            + phase_cycles["exit_rank_rendezvous"]
        ),
    }


def _kernel_trace(call: Callable[[], Any]) -> list[dict[str, Any]]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(prefix="kimi-k3-tail-trace-") as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text(encoding="utf-8"))
    kernels = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
        and "ts" in event
        and "dur" in event
    ]
    if not kernels:
        return []
    first = min(float(event["ts"]) for event in kernels)
    return [
        {
            "index": index,
            "name": str(event["name"]),
            "start_us": float(event["ts"]) - first,
            "duration_us": float(event["dur"]),
        }
        for index, event in enumerate(kernels)
    ]


def _native_post_expert_trace(
    kernels: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    starts = [
        index
        for index, event in enumerate(kernels)
        if NATIVE_POST_EXPERT_START in str(event["name"])
    ]
    if not starts:
        return {
            "status": "not_isolated",
            "reason": (
                f"no kernel contained {NATIVE_POST_EXPERT_START!r}"
            ),
            "all_kernels": list(kernels),
        }
    begin = starts[-1]
    selected = list(kernels[begin:])
    first = float(selected[0]["start_us"])
    end = max(
        float(event["start_us"]) + float(event["duration_us"])
        for event in selected
    )
    return {
        "status": "isolated_by_suffix",
        "selection": (
            "last oneshotAllreduceFusionKernel through final layer kernel"
        ),
        "span_us": end - first,
        "summed_kernel_duration_us": sum(
            float(event["duration_us"]) for event in selected
        ),
        "kernels": selected,
        "caveat": (
            "the trace names establish the native reduction/GEMM/add suffix; "
            "they do not split RMSNorm from a fused reduction kernel"
        ),
    }


def _resource_metadata() -> dict[str, dict[str, int]]:
    names = (
        "launch_ctas",
        "threads_per_cta",
        "dynamic_shared_bytes",
        "active_blocks_per_sm",
        "registers_per_thread",
        "static_shared_bytes",
        "max_dynamic_shared_bytes",
    )
    return {
        "integrated_tensor_kernel": dict(
            zip(
                names,
                map(int, _C._kimi_k3_decode_tensor_resource_metadata()),
                strict=True,
            )
        ),
        "private_tail_tensor_kernel": dict(
            zip(
                names,
                map(int, _C._kimi_k3_tail_tensor_resource_metadata()),
                strict=True,
            )
        ),
    }


def run(
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    clock_ghz: float,
) -> None:
    rank, device = _init_distributed()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_weights = data.build_weights(device, rank)
    router = data.shared_router(
        base_weights,
        device,
        TOKEN_COUNTS,
        pool_size=GRAPH_POOL_SIZE,
    )
    weights = dataclasses.replace(
        base_weights,
        router_weight=router.weight,
    )
    workspace = kimi.get_kimi_k3_decode_workspace(
        dist.group.WORLD,
        device=device,
    )
    adapter_module = importlib.import_module(
        "benchmarks.frameworks.vllm_kimi_k3"
    )
    adapter = adapter_module.build_adapter(
        device=device,
        tp_rank=rank,
        tp_size=TP_SIZE,
        weights=weights,
    )
    adapter.bind_router(router.weight, router.correction_bias)
    resources = _resource_metadata()

    results: list[dict[str, Any]] = []
    raw_samples: dict[str, Any] = {}
    rank_zero_cycle_rows: dict[str, Any] = {}
    native_traces: dict[str, Any] = {}

    for tokens in TOKEN_COUNTS:
        pool = [
            data.build_routed_input(
                weights,
                device,
                tokens,
                pool_index,
                router=router,
            )
            for pool_index in range(GRAPH_POOL_SIZE)
        ]
        coordinator, reduce, shard, role_ctas = map(
            int,
            _C._kimi_k3_tail_role_plan(tokens),
        )

        custom_graphs = _capture_custom_graphs(workspace, pool)
        custom_wall, custom_samples = _measure_graphs(
            custom_graphs,
            device,
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
        custom_graphs.clear()
        _barrier(device)
        torch.cuda.empty_cache()

        private_graph = _capture_private_tail_graph(workspace, pool[-1])
        private_wall, private_samples = _measure_graphs(
            [private_graph],
            device,
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
        del private_graph
        _barrier(device)
        torch.cuda.empty_cache()

        native_graphs = adapter.capture(pool)
        native_wall, native_samples = _measure_graphs(
            native_graphs,
            device,
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
        adapter.release()
        _barrier(device)

        integrated_rows = _profile_integrated_tail(
            workspace,
            pool[-1],
            device,
        )
        private_rows = _profile_private_tail(
            workspace,
            pool[-1],
            role_ctas,
            device,
        )
        integrated_summary = _summarize_tail_rows(
            integrated_rows,
            role_ctas=role_ctas,
            clock_ghz=clock_ghz,
        )
        private_summary = _summarize_tail_rows(
            private_rows,
            role_ctas=role_ctas,
            clock_ghz=clock_ghz,
        )
        integrated_by_rank = _gather_objects(integrated_summary)
        private_by_rank = _gather_objects(private_summary)

        kernels = _kernel_trace(lambda: adapter.forward(pool[-1].hidden))
        native_post_expert = _native_post_expert_trace(kernels)
        native_post_expert_by_rank = _gather_objects(native_post_expert)

        if rank == 0:
            results.append(
                {
                    "tokens": tokens,
                    "route_pool_size": len(pool),
                    "distinct_experts_per_replay": len(
                        pool[-1].distinct_experts
                    ),
                    "role_plan": {
                        "coordinator_ctas": coordinator,
                        "reduce_ctas": reduce,
                        "shard_ctas": shard,
                        "participating_ctas": role_ctas,
                    },
                    "wall_time": {
                        "integrated_full_layer": custom_wall,
                        "private_tail": private_wall,
                        "native_vllm_full_layer": native_wall,
                    },
                    "integrated_tail_by_rank": integrated_by_rank,
                    "private_tail_by_rank": private_by_rank,
                    "native_post_expert_by_rank": (
                        native_post_expert_by_rank
                    ),
                }
            )
            raw_samples[str(tokens)] = {
                "integrated_full_layer_rank_max_ms": custom_samples,
                "private_tail_rank_max_ms": private_samples,
                "native_vllm_full_layer_rank_max_ms": native_samples,
            }
            rank_zero_cycle_rows[str(tokens)] = {
                "integrated": integrated_rows,
                "private": private_rows,
            }
            native_traces[str(tokens)] = {
                "all_kernels": kernels,
                "post_expert": native_post_expert,
            }

        del pool
        _barrier(device)
        torch.cuda.empty_cache()

    adapter.close()
    if rank == 0:
        manifest = {
            "benchmark": "kimi_k3_tail_structural_probe",
            "temporary_instrumentation": True,
            "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
            "gpu": "B300:8",
            "tp_size": TP_SIZE,
            "tokens": list(TOKEN_COUNTS),
            "clock_ghz_for_cycle_conversion": clock_ghz,
            "warmup_count": warmup_count,
            "sample_count": sample_count,
            "graph_pool_size": GRAPH_POOL_SIZE,
            "routing_source": (
                "benchmarks.kimi_k3_decode_data.build_routed_input"
            ),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
        }

        # region agent log
        _debug_log(
            "A",
            "benchmarks/kimi_k3_tail_probe.py:run",
            "fabric rendezvous measurements",
            {
                str(row["tokens"]): {
                    "integrated_rendezvous_cycles_by_rank": [
                        profile["rendezvous_cycles"]
                        for profile in row["integrated_tail_by_rank"]
                    ],
                    "private_rendezvous_cycles_by_rank": [
                        profile["rendezvous_cycles"]
                        for profile in row["private_tail_by_rank"]
                    ],
                    "native_post_expert_span_us_by_rank": [
                        profile.get("span_us")
                        for profile in row["native_post_expert_by_rank"]
                    ],
                }
                for row in results
            },
        )
        # endregion
        # region agent log
        _debug_log(
            "B",
            "benchmarks/kimi_k3_tail_probe.py:run",
            "tail participation and wall floor measurements",
            {
                str(row["tokens"]): {
                    "role_plan": row["role_plan"],
                    "private_tail_median_us": (
                        row["wall_time"]["private_tail"]["median_ms"] * 1000.0
                    ),
                    "integrated_mean_us_by_rank": [
                        profile["mean_participating_us_at_clock"]
                        for profile in row["integrated_tail_by_rank"]
                    ],
                    "integrated_non_tail_cycles_by_rank": [
                        profile["summed_non_tail_cycles"]
                        for profile in row["integrated_tail_by_rank"]
                    ],
                }
                for row in results
            },
        )
        # endregion
        # region agent log
        _debug_log(
            "C",
            "benchmarks/kimi_k3_tail_probe.py:run",
            "reduce RMSNorm shard and multicast measurements",
            {
                str(row["tokens"]): {
                    "integrated_compute_cycles_by_rank": [
                        profile["compute_cycles"]
                        for profile in row["integrated_tail_by_rank"]
                    ],
                    "private_compute_cycles_by_rank": [
                        profile["compute_cycles"]
                        for profile in row["private_tail_by_rank"]
                    ],
                }
                for row in results
            },
        )
        # endregion
        # region agent log
        _debug_log(
            "D",
            "benchmarks/kimi_k3_tail_probe.py:run",
            "dynamic shared memory and occupancy metadata",
            resources,
        )
        # endregion
        # region agent log
        _debug_log(
            "E",
            "benchmarks/kimi_k3_tail_probe.py:run",
            "tail phase handoff measurements",
            {
                str(row["tokens"]): {
                    "integrated_handoff_cycles_by_rank": [
                        profile["handoff_cycles"]
                        for profile in row["integrated_tail_by_rank"]
                    ],
                    "private_handoff_cycles_by_rank": [
                        profile["handoff_cycles"]
                        for profile in row["private_tail_by_rank"]
                    ],
                }
                for row in results
            },
        )
        # endregion

        _write_json(output_dir / "manifest.json", manifest)
        _write_json(
            output_dir / "results.json",
            {
                "resources": resources,
                "rows": results,
            },
        )
        _write_json(output_dir / "raw_samples.json", raw_samples)
        _write_json(
            output_dir / "rank0_cta_cycles.json",
            rank_zero_cycle_rows,
        )
        _write_json(output_dir / "native_traces.json", native_traces)
        if DEBUG_LOG_PATH.is_file():
            (output_dir / "debug.ndjson").write_bytes(
                DEBUG_LOG_PATH.read_bytes()
            )
        print(json.dumps({"artifacts": sorted(
            path.name for path in output_dir.iterdir()
        )}))

    _barrier(device)
    dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-count", type=int, default=500)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--clock-ghz", type=float, default=1.96)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.warmup_count < 1 or arguments.sample_count < 1:
        raise ValueError("warmup and sample counts must be positive")
    if arguments.clock_ghz <= 0.0:
        raise ValueError("clock GHz must be positive")
    run(
        arguments.output_dir,
        warmup_count=arguments.warmup_count,
        sample_count=arguments.sample_count,
        clock_ghz=arguments.clock_ghz,
    )


if __name__ == "__main__":
    main()
