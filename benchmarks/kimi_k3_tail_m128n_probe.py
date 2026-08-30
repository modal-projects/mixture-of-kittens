"""Benchmark-only TP8 comparison of token-M and output-channel-M tail MMAs."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.kimi_k3_timing import (
    percentile,
    rank_max_samples,
    replay_samples,
    summarize_rank_max,
)

try:
    from benchmarks import kimi_k3_decode_data as data
    from benchmarks import kimi_k3_decode_runtime as runtime
    from mok import _C
    from mok import kimi_k3 as kimi
except ImportError as error:
    data = None
    runtime = None
    _C = None
    kimi = None
    _GPU_IMPORT_ERROR: ImportError | None = error
else:
    _GPU_IMPORT_ERROR = None


TP_SIZE = 8
TOKEN_COUNTS = (16, 32, 128)
GRAPH_POOL_SIZE = 4
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5
CLOCK_GHZ = 1.96
MATERIAL_TAIL_GAIN = 0.10
OUTPUT_TILES = 7
REDUCE_CTAS = 32
SHARD_BEGIN = 33
RESOURCE_NAMES = (
    "threads_per_cta",
    "dynamic_shared_bytes",
    "active_blocks_per_sm",
    "registers_per_thread",
    "static_shared_bytes",
    "local_bytes",
)


def candidate_plan(active_tokens: int) -> dict[str, int]:
    if active_tokens not in TOKEN_COUNTS:
        raise ValueError("candidate tokens must be 16, 32, or 128")
    mma_n = 16 if active_tokens == 16 else 32
    token_tiles = active_tokens // mma_n
    shard_ctas = OUTPUT_TILES * token_tiles
    return {
        "mma_m": 128,
        "mma_n": mma_n,
        "mma_k": 64,
        "output_tiles": OUTPUT_TILES,
        "token_tiles": token_tiles,
        "shard_ctas": shard_ctas,
        "role_ctas": SHARD_BEGIN + shard_ctas,
    }


def evaluate_token(
    *,
    baseline_repeat_medians_us: Sequence[float],
    candidate_repeat_medians_us: Sequence[float],
    baseline_shard_mma_us: float,
    candidate_shard_mma_us: float,
    numerical: dict[str, float | bool],
    resources: dict[str, int | bool],
    single_launch: bool = True,
) -> dict[str, float | bool]:
    if not baseline_repeat_medians_us or not candidate_repeat_medians_us:
        raise ValueError("tail evaluation requires repeat medians")
    if len(baseline_repeat_medians_us) != len(
        candidate_repeat_medians_us
    ):
        raise ValueError("baseline and candidate repeat counts must match")
    medians = [
        *(float(value) for value in baseline_repeat_medians_us),
        *(float(value) for value in candidate_repeat_medians_us),
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in medians):
        raise ValueError("tail repeat medians must be finite and positive")

    baseline_center = percentile(baseline_repeat_medians_us, 0.5)
    candidate_center = percentile(candidate_repeat_medians_us, 0.5)
    baseline_dispersion = (
        max(baseline_repeat_medians_us)
        - min(baseline_repeat_medians_us)
    )
    candidate_dispersion = (
        max(candidate_repeat_medians_us)
        - min(candidate_repeat_medians_us)
    )
    effect_band = max(baseline_dispersion, candidate_dispersion)
    improvement_us = baseline_center - candidate_center
    improvement_fraction = improvement_us / baseline_center
    material_tail_wall_gain = (
        improvement_us > effect_band
        and improvement_fraction >= MATERIAL_TAIL_GAIN
    )
    phase_gate_passed = (
        math.isfinite(baseline_shard_mma_us)
        and math.isfinite(candidate_shard_mma_us)
        and candidate_shard_mma_us < baseline_shard_mma_us
    )
    numerical_gate_passed = (
        bool(numerical["finite"])
        and float(numerical["relative_l1"]) <= 0.05
        and float(numerical["cosine_similarity"]) >= 0.999
        and float(numerical["max_abs"]) <= 1.0
    )
    resource_gate_passed = (
        int(resources["active_blocks_per_sm"]) >= 1
        and int(resources["registers_per_thread"]) <= 255
        and int(resources["local_bytes"]) == 0
        and int(resources["stack_bytes"]) == 0
        and bool(resources["resident_role_ctas"])
        and bool(resources.get("shared_fits", True))
    )
    single_launch_gate_passed = bool(single_launch)
    passed = (
        material_tail_wall_gain
        and phase_gate_passed
        and numerical_gate_passed
        and resource_gate_passed
        and single_launch_gate_passed
    )
    return {
        "baseline_median_of_repeats_us": baseline_center,
        "candidate_median_of_repeats_us": candidate_center,
        "baseline_median_dispersion_us": baseline_dispersion,
        "candidate_median_dispersion_us": candidate_dispersion,
        "effect_band_us": effect_band,
        "improvement_us": improvement_us,
        "improvement_fraction": improvement_fraction,
        "material_tail_wall_gain": material_tail_wall_gain,
        "phase_gate_passed": phase_gate_passed,
        "numerical_gate_passed": numerical_gate_passed,
        "resource_gate_passed": resource_gate_passed,
        "single_launch_gate_passed": single_launch_gate_passed,
        "passed": passed,
    }


def integration_decision(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    expected = {int(row["tokens"]) for row in rows}
    all_passed = (
        expected == set(TOKEN_COUNTS)
        and all(bool(row["passed"]) for row in rows)
    )
    return {
        "eligible_for_integration_review": all_passed,
        "integrated": False,
        "preserve_single_launch": True,
        "split_k": "deferred",
        "split_k_reason": (
            "orientation must first prove material full-tail wall improvement; "
            "split-K would add FP32 partial storage, a cross-CTA reduction, and "
            "another synchronization edge"
        ),
        "next_step": (
            "review persistent integration only after all gates pass"
            if all_passed
            else "retain benchmark-only evidence and reject integration"
        ),
    }


def _require_gpu_dependencies() -> None:
    if _GPU_IMPORT_ERROR is not None:
        raise RuntimeError(
            "the m128xN tail probe requires the compiled MoK extension"
        ) from _GPU_IMPORT_ERROR


def _init_distributed() -> tuple[int, torch.device]:
    _require_gpu_dependencies()
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch the m128xN tail probe with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP_SIZE:
        raise RuntimeError(f"the m128xN tail probe requires TP{TP_SIZE}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the m128xN tail probe requires SM103 B300 GPUs")
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


def _tail_call(
    workspace: Any,
    weights: Any,
    active_tokens: int,
) -> None:
    _C._kimi_k3_tail(
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


def _candidate_call(
    workspace: Any,
    weights: Any,
    active_tokens: int,
) -> None:
    _C._kimi_k3_tail_m128n_probe(
        weights.routed_latent_rmsnorm_weight,
        weights.routed_expert_up_proj,
        workspace.collective_multicast_ptr,
        workspace.output_mailbox_multicast_ptr,
        workspace.barrier_buffer,
        workspace.barrier_multicast_ptr,
        workspace.barrier_target,
        workspace.scratch,
        workspace.error_flag,
        workspace.tp_rank,
        active_tokens,
    )


def _variant_call(
    variant: str,
    workspace: Any,
    weights: Any,
    active_tokens: int,
) -> None:
    if variant == "baseline":
        _tail_call(workspace, weights, active_tokens)
        return
    if variant == "candidate":
        _candidate_call(workspace, weights, active_tokens)
        return
    raise ValueError(f"unknown tail variant {variant!r}")


def _seed_tail(workspace: Any, entry: Any) -> None:
    runtime.decode_step(workspace, entry.weights, entry.hidden)


def _output(workspace: Any, active_tokens: int) -> torch.Tensor:
    return workspace.output_mailbox[:active_tokens].view(active_tokens, -1)


def _capture_graph(
    variant: str,
    workspace: Any,
    entry: Any,
    device: torch.device,
) -> torch.cuda.CUDAGraph:
    tokens = int(entry.hidden.shape[0])
    _seed_tail(workspace, entry)
    _variant_call(variant, workspace, entry.weights, tokens)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _variant_call(variant, workspace, entry.weights, tokens)
    return graph


def _measure_graph(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    *,
    warmup_count: int,
    sample_count: int,
) -> tuple[dict[str, int | float], list[float]]:
    _barrier(device)
    local = replay_samples(
        lambda _iteration: graph.replay(),
        warmup_count=warmup_count,
        sample_count=sample_count,
        settle_count=1,
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=lambda: torch.cuda.synchronize(device),
    )
    rank_samples = _gathered_rank_samples(local, device)
    return summarize_rank_max(rank_samples), rank_max_samples(rank_samples)


def _numerical_metrics(
    workspace: Any,
    entry: Any,
    device: torch.device,
) -> dict[str, float | bool]:
    tokens = int(entry.hidden.shape[0])
    _seed_tail(workspace, entry)
    _tail_call(workspace, entry.weights, tokens)
    torch.cuda.synchronize(device)
    baseline = _output(workspace, tokens).clone()

    _seed_tail(workspace, entry)
    _candidate_call(workspace, entry.weights, tokens)
    torch.cuda.synchronize(device)
    candidate = _output(workspace, tokens).clone()

    difference = candidate.float() - baseline.float()
    denominator = baseline.float().abs().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        candidate.float().flatten(),
        baseline.float().flatten(),
        dim=0,
    )
    local = {
        "finite": bool(torch.isfinite(candidate.float()).all()),
        "relative_l1": float(difference.abs().sum() / denominator),
        "cosine_similarity": float(cosine),
        "max_abs": float(difference.abs().max()),
        "bitwise_equal": bool(torch.equal(candidate, baseline)),
    }
    gathered = _gather_objects(local)
    return {
        "finite": all(bool(row["finite"]) for row in gathered),
        "relative_l1": max(float(row["relative_l1"]) for row in gathered),
        "cosine_similarity": min(
            float(row["cosine_similarity"]) for row in gathered
        ),
        "max_abs": max(float(row["max_abs"]) for row in gathered),
        "bitwise_equal": all(
            bool(row["bitwise_equal"]) for row in gathered
        ),
    }


def _profile_variant(
    variant: str,
    workspace: Any,
    entry: Any,
    role_ctas: int,
    device: torch.device,
) -> list[dict[str, int]]:
    _seed_tail(workspace, entry)
    torch.cuda.synchronize(device)
    with runtime.tail_profiling():
        _variant_call(
            variant, workspace, entry.weights, int(entry.hidden.shape[0])
        )
        torch.cuda.synchronize(device)
        return runtime.tail_clock_rows(
            workspace,
            launch_ctas=role_ctas,
        )


def _phase_summary(
    rows: Sequence[dict[str, int]],
    *,
    shard_ctas: int,
    clock_ghz: float,
) -> dict[str, int | float]:
    shard_rows = rows[SHARD_BEGIN:SHARD_BEGIN + shard_ctas]
    if len(shard_rows) != shard_ctas:
        raise ValueError("tail phase rows do not cover every shard CTA")
    cycles_per_microsecond = clock_ghz * 1000.0
    mma = [int(row["latent_up_shard_mma"]) for row in shard_rows]
    mailbox = [
        int(row["mailbox_multicast_and_beta"]) for row in shard_rows
    ]
    total = [int(row["total"]) for row in rows]
    return {
        "launch_ctas": len(rows),
        "shard_ctas": shard_ctas,
        "max_tail_cycles": max(total),
        "max_tail_us": max(total) / cycles_per_microsecond,
        "summed_shard_mma_cycles": sum(mma),
        "max_shard_mma_cycles": max(mma),
        "max_shard_mma_us": max(mma) / cycles_per_microsecond,
        "summed_mailbox_cycles": sum(mailbox),
        "max_mailbox_us": max(mailbox) / cycles_per_microsecond,
    }


def _kernel_trace(call: Callable[[], Any]) -> list[dict[str, Any]]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(
        prefix="kimi-k3-tail-m128n-trace-"
    ) as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "name": str(event["name"]),
            "duration_us": float(event["dur"]),
        }
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
        and "dur" in event
        and "name" in event
    ]


def _single_launch_trace(
    workspace: Any,
    entry: Any,
    device: torch.device,
) -> dict[str, Any]:
    tokens = int(entry.hidden.shape[0])
    _seed_tail(workspace, entry)
    _candidate_call(workspace, entry.weights, tokens)
    torch.cuda.synchronize(device)
    _barrier(device)
    kernels = _kernel_trace(
        lambda: _candidate_call(workspace, entry.weights, tokens)
    )
    local = {
        "kernel_count": len(kernels),
        "candidate_kernel_count": sum(
            "kimi_k3_tail_m128n_probe_kernel" in str(row["name"])
            for row in kernels
        ),
        "kernels": kernels,
    }
    gathered = _gather_objects(local)
    return {
        "all_ranks_single_launch": all(
            int(row["kernel_count"]) == 1
            and int(row["candidate_kernel_count"]) == 1
            for row in gathered
        ),
        "by_rank": gathered,
    }


@functools.cache
def _binary_resource_usage(token_tile_n: int) -> dict[str, int]:
    tool = shutil.which("cuobjdump")
    if tool is None:
        raise RuntimeError("the resource gate requires cuobjdump")
    dump = subprocess.run(
        [tool, "--dump-resource-usage", str(Path(_C.__file__))],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    usage = {
        name: {
            key: int(value)
            for key, value in re.findall(r"([A-Z]+):(\d+)", line)
        }
        for name, line in re.findall(
            r"Function (\S+):\s*\n\s*(REG:.*)", dump
        )
        if "kimi_k3_tail_m128n_probe_kernel" in name
        and f"ILi{token_tile_n}EE" in name
    }
    if len(usage) != 1:
        raise RuntimeError(
            f"expected one m128n{token_tile_n} probe symbol, got "
            f"{sorted(usage)}"
        )
    return next(iter(usage.values()))


def _resource_metadata(
    active_tokens: int,
    device: torch.device,
) -> dict[str, int | bool]:
    plan = candidate_plan(active_tokens)
    metadata = dict(
        zip(
            RESOURCE_NAMES,
            map(
                int,
                _C._kimi_k3_tail_m128n_resource_metadata(
                    plan["mma_n"]
                ),
            ),
            strict=True,
        )
    )
    binary = _binary_resource_usage(plan["mma_n"])
    properties = torch.cuda.get_device_properties(device)
    metadata.update(
        {
            "stack_bytes": int(binary.get("STACK", -1)),
            "local_bytes": int(binary.get("LOCAL", -1)),
            "binary_registers_per_thread": int(binary.get("REG", -1)),
            "binary_static_shared_bytes": int(binary.get("SHARED", -1)),
            "resident_role_ctas": (
                int(metadata["active_blocks_per_sm"])
                * int(properties.multi_processor_count)
                >= plan["role_ctas"]
            ),
            "shared_fits": (
                int(metadata["dynamic_shared_bytes"])
                + int(binary.get("SHARED", -1))
                <= int(properties.shared_memory_per_block_optin)
            ),
            "available_sms": int(properties.multi_processor_count),
            "role_ctas": plan["role_ctas"],
        }
    )
    return metadata


def _build_context(
    device: torch.device,
    rank: int,
) -> tuple[Any, Any, Any]:
    base_weights = data.build_weights(device, rank)
    router = data.shared_router(
        base_weights,
        device,
        TOKEN_COUNTS,
        pool_size=GRAPH_POOL_SIZE,
    )
    workspace = kimi.get_kimi_k3_decode_workspace(
        dist.group.WORLD,
        device=device,
    )
    return base_weights, router, workspace


def _build_entry(
    base_weights: Any,
    router: Any,
    device: torch.device,
    tokens: int,
) -> Any:
    entry = data.build_routed_input(
        base_weights,
        device,
        tokens,
        GRAPH_POOL_SIZE - 1,
        router=router,
    )
    return entry


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest(
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
    clock_ghz: float,
) -> dict[str, object]:
    return {
        "benchmark": "kimi_k3_tail_m128n_probe",
        "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
        "gpu": "B300:8",
        "tp_size": TP_SIZE,
        "tokens": list(TOKEN_COUNTS),
        "candidate_plans": {
            str(tokens): candidate_plan(tokens) for tokens in TOKEN_COUNTS
        },
        "baseline": "existing private-tail shard_tensor",
        "candidate": "output-channel-M/token-N BF16 contraction",
        "accumulator": "FP32",
        "input_output_boundaries": "BF16",
        "launches_per_variant_replay": 1,
        "material_tail_gain_fraction": MATERIAL_TAIL_GAIN,
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "repeats": repeats,
        "clock_ghz_for_cycle_conversion": clock_ghz,
        "split_k": "deferred",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def run(
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
    clock_ghz: float,
) -> dict[str, Any]:
    if warmup_count < 1 or sample_count < 1 or repeats < 2:
        raise ValueError("warmups and samples must be positive; repeats >= 2")
    if clock_ghz <= 0.0:
        raise ValueError("clock GHz must be positive")
    rank, device = _init_distributed()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_weights, router, workspace = _build_context(device, rank)

    rows: list[dict[str, Any]] = []
    raw_samples: dict[str, Any] = {}
    phase_rows: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    resources: dict[str, Any] = {}

    for tokens in TOKEN_COUNTS:
        plan = candidate_plan(tokens)
        entry = _build_entry(base_weights, router, device, tokens)
        numerical = _numerical_metrics(workspace, entry, device)
        baseline_rows = _profile_variant(
            "baseline", workspace, entry, SHARD_BEGIN + OUTPUT_TILES, device
        )
        candidate_rows = _profile_variant(
            "candidate", workspace, entry, plan["role_ctas"], device
        )
        baseline_phase = _phase_summary(
            baseline_rows,
            shard_ctas=OUTPUT_TILES,
            clock_ghz=clock_ghz,
        )
        candidate_phase = _phase_summary(
            candidate_rows,
            shard_ctas=plan["shard_ctas"],
            clock_ghz=clock_ghz,
        )
        trace = _single_launch_trace(workspace, entry, device)
        resource = _resource_metadata(tokens, device)

        graphs = {
            variant: _capture_graph(
                variant, workspace, entry, device
            )
            for variant in ("baseline", "candidate")
        }
        summaries: dict[str, list[dict[str, int | float]]] = {
            "baseline": [],
            "candidate": [],
        }
        samples: dict[str, list[list[float]]] = {
            "baseline": [],
            "candidate": [],
        }
        for repeat in range(repeats):
            order = (
                ("baseline", "candidate")
                if repeat % 2 == 0
                else ("candidate", "baseline")
            )
            for variant in order:
                summary, rank_max = _measure_graph(
                    graphs[variant],
                    device,
                    warmup_count=warmup_count,
                    sample_count=sample_count,
                )
                summaries[variant].append(summary)
                samples[variant].append(rank_max)

        verdict = evaluate_token(
            baseline_repeat_medians_us=[
                float(summary["median_ms"]) * 1000.0
                for summary in summaries["baseline"]
            ],
            candidate_repeat_medians_us=[
                float(summary["median_ms"]) * 1000.0
                for summary in summaries["candidate"]
            ],
            baseline_shard_mma_us=float(
                baseline_phase["max_shard_mma_us"]
            ),
            candidate_shard_mma_us=float(
                candidate_phase["max_shard_mma_us"]
            ),
            numerical=numerical,
            resources=resource,
            single_launch=bool(trace["all_ranks_single_launch"]),
        )
        row = {
            "tokens": tokens,
            "plan": plan,
            "numerical": numerical,
            "baseline_phase": baseline_phase,
            "candidate_phase": candidate_phase,
            "baseline_repeats": summaries["baseline"],
            "candidate_repeats": summaries["candidate"],
            "resources": resource,
            "single_launch": trace["all_ranks_single_launch"],
            **verdict,
        }
        if rank == 0:
            rows.append(row)
            raw_samples[str(tokens)] = samples
            phase_rows[str(tokens)] = {
                "baseline": baseline_rows,
                "candidate": candidate_rows,
            }
            traces[str(tokens)] = trace
            resources[str(tokens)] = resource
        graphs.clear()
        del entry
        _barrier(device)
        torch.cuda.empty_cache()

    result: dict[str, Any] = {}
    if rank == 0:
        decision = integration_decision(rows)
        result = {
            "passed": all(bool(row["passed"]) for row in rows),
            "rows": rows,
            "decision": decision,
        }
        _write_json(
            output_dir / "manifest.json",
            _manifest(
                warmup_count=warmup_count,
                sample_count=sample_count,
                repeats=repeats,
                clock_ghz=clock_ghz,
            ),
        )
        _write_json(output_dir / "results.json", result)
        _write_json(output_dir / "raw_samples.json", raw_samples)
        _write_json(output_dir / "phase_cycles.json", phase_rows)
        _write_json(output_dir / "kernel_traces.json", traces)
        _write_json(output_dir / "resources.json", resources)
        print(json.dumps(result, indent=2, sort_keys=True))

    _barrier(device)
    dist.destroy_process_group()
    return result


def run_focused(*, tokens: int, variant: str) -> dict[str, Any]:
    if tokens not in TOKEN_COUNTS:
        raise ValueError("focused tokens must be 16, 32, or 128")
    if variant not in ("setup", "baseline", "candidate", "both"):
        raise ValueError(
            "focused variant must be setup, baseline, candidate, or both"
        )
    rank, device = _init_distributed()
    base_weights, router, workspace = _build_context(device, rank)
    entry = _build_entry(base_weights, router, device, tokens)
    _seed_tail(workspace, entry)
    torch.cuda.synchronize(device)
    result: dict[str, Any] = {
        "tokens": tokens,
        "variant": variant,
        "rank": rank,
        "setup_synchronized": True,
    }
    if variant == "both":
        result["numerical"] = _numerical_metrics(workspace, entry, device)
        result["trace"] = _single_launch_trace(workspace, entry, device)
        result["resources"] = _resource_metadata(tokens, device)
    elif variant != "setup":
        _variant_call(variant, workspace, entry.weights, tokens)
        torch.cuda.synchronize(device)
        result["launch_synchronized"] = True
        result["finite"] = bool(
            torch.isfinite(_output(workspace, tokens).float()).all()
        )
        result["checksum"] = float(
            _output(workspace, tokens).float().sum()
        )
    _barrier(device)
    dist.destroy_process_group()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kimi_k3_tail_m128n_probe"),
    )
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--clock-ghz", type=float, default=CLOCK_GHZ)
    parser.add_argument("--focus-tokens", type=int)
    parser.add_argument(
        "--focus-variant",
        choices=("setup", "baseline", "candidate", "both"),
        default="both",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.focus_tokens is not None:
        result = run_focused(
            tokens=arguments.focus_tokens,
            variant=arguments.focus_variant,
        )
        if int(result["rank"]) == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
        return
    run(
        arguments.output_dir,
        warmup_count=arguments.warmup_count,
        sample_count=arguments.sample_count,
        repeats=arguments.repeats,
        clock_ghz=arguments.clock_ghz,
    )


if __name__ == "__main__":
    main()
