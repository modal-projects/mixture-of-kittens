"""Measure M16 routed gate/up critical-path subphases on TP8 B300."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


TOKENS = 16
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5
POOL_SIZE = 4
SUBPHASE_NAMES = (
    "weight_global_load",
    "weight_shared_store_swizzle",
    "activation_stage",
    "scale_stage_copy",
    "sync_tma_tmem_wait",
    "queue_claim",
    "unit_setup",
    "units",
)
TIMED_SUBPHASE_NAMES = SUBPHASE_NAMES[:-1]
HYPOTHESES = {
    "A": (
        "packed weight global loads dominate because each M16 expert unit "
        "streams row-strided 16-byte sectors"
    ),
    "B": (
        "expanding packed weights into the shared swizzle dominates because "
        "every payload also writes zero padding"
    ),
    "C": (
        "one-row activation staging is serialization-sensitive because only "
        "a small subset of the CTA participates"
    ),
    "D": (
        "weight/activation scale staging and shared-to-TMEM copies dominate "
        "the short M16 unit"
    ),
    "E": (
        "CTA, TMEM, semaphore, queue, or per-unit setup waits dominate the "
        "critical CTA even when summed work looks balanced"
    ),
}


# region agent log
def _agent_log(
    *,
    location: str,
    message: str,
    data: Mapping[str, object],
    hypothesis_id: str,
) -> None:
    with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": dict(data),
                    "timestamp": int(time.time() * 1000),
                },
                sort_keys=True,
            )
            + "\n"
        )
# endregion


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def summarize_subphase_samples(
    samples: Sequence[Mapping[str, int]],
) -> dict[str, object]:
    if not samples:
        raise ValueError("subphase summary requires samples")
    subphases = {
        name: _distribution([float(sample[name]) for sample in samples])
        for name in SUBPHASE_NAMES
    }
    dominant = max(
        TIMED_SUBPHASE_NAMES,
        key=lambda name: float(subphases[name]["p50"]),
    )
    return {
        "sample_count": len(samples),
        "aggregation": "rank-max of each launch's CTA maxima",
        "dominant_subphase": dominant,
        "subphases": subphases,
        "critical_path": _distribution(
            [float(sample["critical_path"]) for sample in samples]
        ),
    }


def _gathered_rank_samples(
    local_samples: Sequence[float],
    device: torch.device,
) -> list[list[float]]:
    local = torch.tensor(
        list(local_samples), dtype=torch.float64, device=device
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [rank.cpu().tolist() for rank in gathered]


def measure_latency_repeats(
    graphs: Sequence[torch.cuda.CUDAGraph],
    *,
    extension: Any,
    timing: Any,
    device: torch.device,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> list[dict[str, object]]:
    if extension._kimi_k3_decode_phase_profile():
        raise RuntimeError("phase instrumentation must be off for latency")
    measured: list[dict[str, object]] = []
    for repeat in range(repeats):
        samples = timing.replay_samples(
            lambda iteration: graphs[iteration % len(graphs)].replay(),
            warmup_count=warmup_count,
            sample_count=sample_count,
            settle_count=len(graphs),
            event_factory=lambda: torch.cuda.Event(enable_timing=True),
            synchronize=lambda: torch.cuda.synchronize(device),
        )
        rank_samples = _gathered_rank_samples(samples, device)
        maxima = timing.rank_max_samples(rank_samples)
        row = {
            "repeat": repeat,
            "sample_count": sample_count,
            "median_ms": timing.percentile(maxima, 0.50),
            "p90_ms": timing.percentile(maxima, 0.90),
            "p99_ms": timing.percentile(maxima, 0.99),
            "rank_max_samples_ms": maxima,
        }
        measured.append(row)
        if dist.get_rank() == 0:
            # region agent log
            _agent_log(
                location=(
                    "benchmarks/kimi_k3_gate_up_subphase.py:"
                    "measure_latency_repeats"
                ),
                message="uninstrumented M16 latency repeat completed",
                data={
                    "repeat": repeat,
                    "sample_count": sample_count,
                    "median_ms": float(row["median_ms"]),
                },
                hypothesis_id="A,B,C,D,E",
            )
            # endregion
    return measured


def _measure_subphase_repeat(
    workspace: Any,
    pool: Sequence[Any],
    *,
    runtime: Any,
    extension: Any,
    device: torch.device,
    warmup_count: int,
    sample_count: int,
) -> list[dict[str, int]]:
    begin, ctas, names = (
        extension._kimi_k3_decode_gate_up_subphase_metadata()
    )
    if tuple(names) != SUBPHASE_NAMES:
        raise RuntimeError(f"subphase metadata drifted: {tuple(names)!r}")
    if begin < 0 or ctas <= 0:
        raise RuntimeError("invalid routed gate/up subphase metadata")

    for iteration in range(warmup_count):
        entry = pool[iteration % len(pool)]
        runtime.decode_device_step(workspace, entry.weights, entry.hidden)
    torch.cuda.synchronize(device)
    runtime.check_decode_error(workspace)

    traces = torch.empty(
        sample_count,
        ctas,
        len(names),
        dtype=torch.int64,
        device=device,
    )
    with runtime.phase_profiling():
        for iteration in range(sample_count):
            entry = pool[iteration % len(pool)]
            runtime.decode_device_step(workspace, entry.weights, entry.hidden)
            traces[iteration].copy_(
                runtime.gate_up_subphase_tensor(workspace)
            )
    torch.cuda.synchronize(device)
    runtime.check_decode_error(workspace)

    component_maxima = traces.amax(dim=1)
    critical_path = traces[:, :, : len(TIMED_SUBPHASE_NAMES)].sum(
        dim=2
    ).amax(dim=1, keepdim=True)
    reduced = torch.cat((component_maxima, critical_path), dim=1)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    host = reduced.cpu().tolist()
    return [
        {
            **{
                name: int(value)
                for name, value in zip(
                    SUBPHASE_NAMES, row[: len(SUBPHASE_NAMES)], strict=True
                )
            },
            "critical_path": int(row[-1]),
        }
        for row in host
    ]


def measure_subphase_repeats(
    workspace: Any,
    pool: Sequence[Any],
    *,
    runtime: Any,
    extension: Any,
    device: torch.device,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> list[dict[str, object]]:
    measured: list[dict[str, object]] = []
    for repeat in range(repeats):
        samples = _measure_subphase_repeat(
            workspace,
            pool,
            runtime=runtime,
            extension=extension,
            device=device,
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
        summary = summarize_subphase_samples(samples)
        measured.append(
            {"repeat": repeat, "summary": summary, "samples": samples}
        )
        if dist.get_rank() == 0:
            # region agent log
            _agent_log(
                location=(
                    "benchmarks/kimi_k3_gate_up_subphase.py:"
                    "measure_subphase_repeats"
                ),
                message="dedicated M16 subphase repeat completed",
                data={
                    "repeat": repeat,
                    "sample_count": sample_count,
                    "dominant_subphase": summary["dominant_subphase"],
                    "critical_path_p50": summary["critical_path"]["p50"],
                },
                hypothesis_id="A,B,C,D,E",
            )
            # endregion
    return measured


def _init_distributed() -> tuple[int, torch.device]:
    required = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if not required <= os.environ.keys():
        raise RuntimeError("launch the gate/up subphase benchmark with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError("the gate/up subphase benchmark requires TP8")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the gate/up subphase benchmark requires SM103 B300")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return rank, device


def _barrier(device: torch.device) -> None:
    dist.barrier(async_op=True, device_ids=[device.index]).block_current_stream()
    torch.cuda.synchronize(device)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> dict[str, object]:
    if warmup_count < 1 or sample_count < 1 or repeats < 2:
        raise ValueError("warmups/samples must be positive and repeats >= 2")
    rank, device = _init_distributed()

    # These extension-backed modules cannot be imported by CPU-only contract
    # tests, so they are resolved only after the B300 process is initialized.
    data = importlib.import_module("benchmarks.kimi_k3_decode_data")
    runtime = importlib.import_module("benchmarks.kimi_k3_decode_runtime")
    timing = importlib.import_module("benchmarks.kimi_k3_timing")
    kimi = importlib.import_module("mok.kimi_k3")
    extension = importlib.import_module("mok._C")

    if rank == 0:
        # region agent log
        _agent_log(
            location="benchmarks/kimi_k3_gate_up_subphase.py:run",
            message="M16 gate/up bottleneck experiment started",
            data={
                "tokens": TOKENS,
                "warmup_count": warmup_count,
                "sample_count": sample_count,
                "repeats": repeats,
            },
            hypothesis_id="A,B,C,D,E",
        )
        # endregion

    weights = data.build_weights(device, rank)
    workspace = kimi.get_kimi_k3_decode_workspace(
        dist.group.WORLD, device=device
    )
    router = data.shared_router(
        weights, device, [TOKENS], pool_size=POOL_SIZE
    )
    weights = dataclasses.replace(weights, router_weight=router.weight)
    pool = [
        data.build_routed_input(
            weights, device, TOKENS, index, router=router
        )
        for index in range(POOL_SIZE)
    ]

    correctness = []
    for entry in pool:
        actual = runtime.decode_step(
            workspace, entry.weights, entry.hidden
        ).clone()
        torch.cuda.synchronize(device)
        expected = runtime.decode_reference(entry.hidden, entry.weights)
        relative_l1, cosine, maximum = runtime.assert_decode_close(
            actual, expected
        )
        correctness.append(
            {
                "relative_l1": relative_l1,
                "cosine_similarity": cosine,
                "max_abs": maximum,
            }
        )
    if rank == 0:
        # region agent log
        _agent_log(
            location="benchmarks/kimi_k3_gate_up_subphase.py:run:correctness",
            message="production arithmetic gate passed before measurement",
            data={
                "pool_entries": len(correctness),
                "maximum_relative_l1": max(
                    float(row["relative_l1"]) for row in correctness
                ),
            },
            hypothesis_id="A,B,C,D,E",
        )
        # endregion

    graphs: list[torch.cuda.CUDAGraph] = []
    for entry in pool:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kimi.kimi_k3_decode(
                runtime.CONFIG, workspace, entry.weights, entry.hidden
            )
        graphs.append(graph)

    latency = measure_latency_repeats(
        graphs,
        extension=extension,
        timing=timing,
        device=device,
        warmup_count=warmup_count,
        sample_count=sample_count,
        repeats=repeats,
    )
    phases = measure_subphase_repeats(
        workspace,
        pool,
        runtime=runtime,
        extension=extension,
        device=device,
        warmup_count=warmup_count,
        sample_count=sample_count,
        repeats=repeats,
    )

    dominant_votes = [
        str(repeat["summary"]["dominant_subphase"]) for repeat in phases
    ]
    dominant = max(
        set(dominant_votes),
        key=lambda name: (dominant_votes.count(name), name),
    )
    if rank == 0:
        # region agent log
        _agent_log(
            location="benchmarks/kimi_k3_gate_up_subphase.py:run:dominance",
            message="stable M16 critical-path dominance selected",
            data={
                "dominant_subphase": dominant,
                "repeat_votes": dominant_votes,
            },
            hypothesis_id="A,B,C,D,E",
        )
        # endregion

    kernel_names = runtime.profiled_kernel_names(
        lambda: runtime.decode_device_step(
            workspace, pool[0].weights, pool[0].hidden
        )
    )
    grid_ctas, threads, dynamic_shared = (
        extension._kimi_k3_decode_grid_shape()
    )
    resources = {
        "grid_ctas": int(grid_ctas),
        "threads": int(threads),
        "dynamic_shared_bytes": int(dynamic_shared),
        "resident_blocks_per_sm": int(
            extension._kimi_k3_decode_resident_blocks_per_sm(True)
        ),
        "kernel_launches": len(kernel_names),
        "kernel_names": kernel_names,
    }
    result = {
        "tokens": TOKENS,
        "hypotheses": HYPOTHESES,
        "correctness": correctness,
        "latency_repeats": [
            {
                key: value
                for key, value in repeat.items()
                if key != "rank_max_samples_ms"
            }
            for repeat in latency
        ],
        "subphase_repeats": [
            {"repeat": repeat["repeat"], **repeat["summary"]}
            for repeat in phases
        ],
        "dominant_subphase": dominant,
        "resources": resources,
    }

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "manifest.json",
            {
                "benchmark": "kimi_k3_gate_up_subphase",
                "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
                "gpu": "B300:8",
                "tokens": TOKENS,
                "warmup_count": warmup_count,
                "sample_count": sample_count,
                "repeats": repeats,
                "latency_instrumentation": "disabled",
                "subphase_instrumentation": "dedicated launches only",
                "aggregation": "rank-max per-CTA critical path",
                "hypotheses": HYPOTHESES,
            },
        )
        _write_json(output_dir / "results.json", result)
        _write_json(
            output_dir / "raw_samples.json",
            {
                "latency_repeats": latency,
                "subphase_repeats": phases,
            },
        )
        # region agent log
        _agent_log(
            location="benchmarks/kimi_k3_gate_up_subphase.py:run:exit",
            message="M16 gate/up bottleneck experiment completed",
            data={
                "dominant_subphase": dominant,
                "kernel_launches": resources["kernel_launches"],
                "resident_blocks_per_sm": resources[
                    "resident_blocks_per_sm"
                ],
            },
            hypothesis_id="A,B,C,D,E",
        )
        # endregion
        print(json.dumps(result, indent=2, sort_keys=True))

    _barrier(device)
    graphs.clear()
    kimi.clear_kimi_k3_decode_workspace_cache()
    dist.destroy_process_group()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kimi_k3_gate_up_subphase"),
    )
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        args.output_dir,
        warmup_count=args.warmup_count,
        sample_count=args.sample_count,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
