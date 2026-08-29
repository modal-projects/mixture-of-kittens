"""Compare grouped and pipelined gate/up candidates with the shipped path."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.compare_kimi_k3_frameworks import (
    derive_phase_cycles,
    summarize_phase_cycles,
)
from benchmarks.kimi_k3_timing import (
    percentile,
    replay_samples,
    summarize_rank_max,
)


TOKENS = (16, 32, 128)
POOL_SIZE = 4
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5


@dataclass(frozen=True)
class Variant:
    name: str
    group_size: int
    pipeline_gate_up_down: bool


VARIANTS = (
    Variant("baseline", 0, False),
    Variant("group_1", 1, False),
    Variant("group_2", 2, False),
    Variant("pipeline_gate_up_down", 0, True),
)
BASELINE = VARIANTS[0]
CANDIDATES = VARIANTS[1:]
DEBUG_LOG = Path("/opt/cursor/logs/debug.log")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _agent_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    """Append one compact benchmark diagnostic from rank zero."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": time.time_ns() // 1_000_000,
    }
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _init_distributed() -> tuple[int, torch.device]:
    required = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if not required <= os.environ.keys():
        raise RuntimeError("launch the gate/up grouping benchmark with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError("the gate/up grouping benchmark requires TP8")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the gate/up grouping benchmark requires SM103")
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


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    left = actual.float()
    right = expected.float()
    difference = left - right
    return {
        "finite": bool(torch.isfinite(left).all()),
        "relative_l1": float(
            difference.abs().sum() / right.abs().sum().clamp_min(1e-12)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                left.flatten(), right.flatten(), dim=0
            )
        ),
        "max_abs": float(difference.abs().max()),
        "bitwise_equal": bool(torch.equal(actual, expected)),
    }


def _capture_pool(
    runtime_module: ModuleType,
    workspace: Any,
    pool: Sequence[Any],
    variant: Variant,
    device: torch.device,
) -> list[torch.cuda.CUDAGraph]:
    graphs: list[torch.cuda.CUDAGraph] = []
    with runtime_module.benchmark_persistent_variant(
        variant.group_size, variant.pipeline_gate_up_down
    ):
        for entry in pool:
            runtime_module.decode_device_step(
                workspace, entry.weights, entry.hidden
            )
            torch.cuda.synchronize(device)
            runtime_module.check_decode_error(workspace)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runtime_module.decode_device_step(
                    workspace, entry.weights, entry.hidden
                )
            torch.cuda.synchronize(device)
            runtime_module.check_decode_error(workspace)
            graphs.append(graph)
    return graphs


def _gathered_rank_samples(
    local_samples: Sequence[float],
    device: torch.device,
) -> list[list[float]]:
    local = torch.tensor(local_samples, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [value.cpu().tolist() for value in gathered]


def _measure(
    graphs: Sequence[torch.cuda.CUDAGraph],
    *,
    runtime_module: ModuleType,
    workspace: Any,
    warmup_count: int,
    sample_count: int,
    device: torch.device,
) -> list[float]:
    samples = replay_samples(
        lambda iteration: graphs[iteration % len(graphs)].replay(),
        warmup_count=warmup_count,
        sample_count=sample_count,
        settle_count=len(graphs),
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=lambda: torch.cuda.synchronize(device),
    )
    runtime_module.check_decode_error(workspace)
    return samples


def _phase_profile(
    runtime_module: ModuleType,
    workspace: Any,
    pool: Sequence[Any],
    variant: Variant,
    device: torch.device,
) -> dict[str, Any]:
    with runtime_module.benchmark_persistent_variant(
        variant.group_size, variant.pipeline_gate_up_down
    ):
        with runtime_module.phase_profiling():
            for entry in pool:
                runtime_module.decode_step(
                    workspace, entry.weights, entry.hidden
                )
            torch.cuda.synchronize(device)
            cycles = derive_phase_cycles(
                runtime_module.phase_clock_cycles(workspace)
            )
    staging = (
        cycles.get("routed_gate_up_stage", 0)
        + cycles.get("routed_down_stage", 0)
    )
    mma = (
        cycles.get("routed_gate_up_mma", 0)
        + cycles.get("routed_down_mma", 0)
    )
    epilogue = (
        cycles.get("routed_gate_up_epilogue", 0)
        + cycles.get("routed_down_epilogue", 0)
    )
    return {
        "cycles": cycles,
        "categories": {
            "staging": staging,
            "mma": mma,
            "epilogue": epilogue,
            "queue": cycles.get("routed_queue", 0),
            "barrier": cycles.get("grid_barrier", 0),
            "readiness_wait": cycles.get("readiness_wait", 0),
        },
        **summarize_phase_cycles(cycles),
    }


def _kernel_names(
    runtime_module: ModuleType,
    workspace: Any,
    entry: Any,
    variant: Variant,
) -> list[str]:
    with runtime_module.benchmark_persistent_variant(
        variant.group_size, variant.pipeline_gate_up_down
    ):
        return runtime_module.profiled_kernel_names(
            lambda: runtime_module.decode_step(
                workspace, entry.weights, entry.hidden
            )
        )


def evaluate_candidate(
    *,
    baseline_repeat_medians: Sequence[float],
    candidate_repeat_medians: Sequence[float],
    numerical_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_center = percentile(baseline_repeat_medians, 0.5)
    candidate_center = percentile(candidate_repeat_medians, 0.5)
    baseline_dispersion = (
        max(baseline_repeat_medians) - min(baseline_repeat_medians)
    )
    candidate_dispersion = (
        max(candidate_repeat_medians) - min(candidate_repeat_medians)
    )
    effect_band = max(baseline_dispersion, candidate_dispersion)
    improvement = baseline_center - candidate_center
    numerically_correct = all(
        bool(row["candidate_vs_reference"]["finite"])
        and float(row["candidate_vs_reference"]["relative_l1"]) <= 0.05
        and float(row["candidate_vs_reference"]["cosine_similarity"]) >= 0.999
        and float(row["candidate_vs_reference"]["max_abs"]) <= 1.0
        for row in numerical_rows
    )
    return {
        "baseline_median_of_repeats_ms": baseline_center,
        "candidate_median_of_repeats_ms": candidate_center,
        "baseline_median_dispersion_ms": baseline_dispersion,
        "candidate_median_dispersion_ms": candidate_dispersion,
        "effect_band_ms": effect_band,
        "improvement_ms": improvement,
        "improvement_fraction": improvement / baseline_center,
        "measurably_faster": improvement > effect_band,
        "numerically_correct": numerically_correct,
        "passed": numerically_correct and improvement > effect_band,
    }


def run(
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup_count < 1 or sample_count < 1 or repeats < 2:
        raise ValueError("warmups and samples must be positive; repeats >= 2")
    # GPU-only modules are loaded after torchrun has established the process
    # environment. This keeps the pure benchmark contract importable in a
    # source checkout where the CUDA extension has not been built.
    data_module = importlib.import_module("benchmarks.kimi_k3_decode_data")
    runtime_module = importlib.import_module(
        "benchmarks.kimi_k3_decode_runtime"
    )
    kimi_module = importlib.import_module("mok.kimi_k3")
    extension = importlib.import_module("mok._C")
    rank, device = _init_distributed()
    # region agent log
    _agent_log(
        "E,F,G,H",
        "benchmarks/kimi_k3_gate_up_grouping.py:run:start",
        "gate/up benchmark started",
        {
            "tokens": list(TOKENS),
            "variants": [dataclasses.asdict(value) for value in VARIANTS],
            "warmup_count": warmup_count,
            "sample_count": sample_count,
            "repeats": repeats,
        },
    )
    # endregion
    weights = data_module.build_weights(device, rank)
    workspace = kimi_module.get_kimi_k3_decode_workspace(
        dist.group.WORLD,
        device=device,
    )
    router = data_module.shared_router(
        weights,
        device,
        TOKENS,
        pool_size=POOL_SIZE,
    )
    weights = dataclasses.replace(weights, router_weight=router.weight)
    numerical_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    raw_samples: dict[str, Any] = {}
    phase_profiles: dict[str, Any] = {}
    launch_names: dict[str, Any] = {}

    for tokens in TOKENS:
        pool = [
            data_module.build_routed_input(
                weights,
                device,
                tokens,
                index,
                router=router,
            )
            for index in range(POOL_SIZE)
        ]
        for pool_index, entry in enumerate(pool):
            outputs = {}
            for variant in VARIANTS:
                with runtime_module.benchmark_persistent_variant(
                    variant.group_size, variant.pipeline_gate_up_down
                ):
                    outputs[variant.name] = runtime_module.decode_step(
                        workspace, entry.weights, entry.hidden
                    ).clone()
            torch.cuda.synchronize(device)
            reference = runtime_module.decode_reference(
                entry.hidden, entry.weights
            )
            baseline = outputs[BASELINE.name]
            baseline_vs_reference = _stats(baseline, reference)
            runtime_module.assert_decode_close(baseline, reference)
            for candidate in CANDIDATES:
                candidate_output = outputs[candidate.name]
                candidate_vs_reference = _stats(candidate_output, reference)
                runtime_module.assert_decode_close(candidate_output, reference)
                runtime_module.assert_identical_across_ranks(candidate_output)
                numerical_rows.append(
                    {
                        "tokens": tokens,
                        "pool_index": pool_index,
                        "variant": candidate.name,
                        "group_size": candidate.group_size,
                        "pipeline_gate_up_down":
                            candidate.pipeline_gate_up_down,
                        "baseline_vs_reference": baseline_vs_reference,
                        "candidate_vs_reference": candidate_vs_reference,
                        "candidate_vs_baseline": _stats(
                            candidate_output, baseline
                        ),
                    }
                )
            del outputs, baseline, reference
        _barrier(device)
        # region agent log
        _agent_log(
            "H",
            "benchmarks/kimi_k3_gate_up_grouping.py:run:numerical",
            "gate/up numerical checks completed",
            {
                "tokens": tokens,
                "rows": len(
                    [row for row in numerical_rows if row["tokens"] == tokens]
                ),
                "variants": [value.name for value in CANDIDATES],
            },
        )
        # endregion

        graphs = {
            variant.name: _capture_pool(
                runtime_module, workspace, pool, variant, device
            )
            for variant in VARIANTS
        }
        samples_by_variant: dict[str, list[list[float]]] = {
            variant.name: [] for variant in VARIANTS
        }
        for repeat in range(repeats):
            shift = repeat % len(VARIANTS)
            order = (*VARIANTS[shift:], *VARIANTS[:shift])
            for variant in order:
                local_samples = _measure(
                    graphs[variant.name],
                    runtime_module=runtime_module,
                    workspace=workspace,
                    warmup_count=warmup_count,
                    sample_count=sample_count,
                    device=device,
                )
                gathered = _gathered_rank_samples(local_samples, device)
                samples_by_variant[variant.name].append(
                    [
                        max(values)
                        for values in zip(*gathered, strict=True)
                    ]
                )

        repeat_summaries = {
            variant.name: [
                summarize_rank_max([samples])
                for samples in samples_by_variant[variant.name]
            ]
            for variant in VARIANTS
        }
        verdicts = {
            candidate.name: evaluate_candidate(
                baseline_repeat_medians=[
                    float(row["median_ms"])
                    for row in repeat_summaries[BASELINE.name]
                ],
                candidate_repeat_medians=[
                    float(row["median_ms"])
                    for row in repeat_summaries[candidate.name]
                ],
                numerical_rows=[
                    row
                    for row in numerical_rows
                    if row["tokens"] == tokens
                    and row["variant"] == candidate.name
                ],
            )
            for candidate in CANDIDATES
        }
        shape_rows.append(
            {
                "tokens": tokens,
                "repeats": repeat_summaries,
                "verdicts": verdicts,
            }
        )
        # region agent log
        _agent_log(
            "E,F,H",
            "benchmarks/kimi_k3_gate_up_grouping.py:run:latency",
            "gate/up repeat medians evaluated",
            {
                "tokens": tokens,
                "baseline_ms": percentile(
                    [
                        float(row["median_ms"])
                        for row in repeat_summaries[BASELINE.name]
                    ],
                    0.5,
                ),
                "candidate_improvement_ms": {
                    name: value["improvement_ms"]
                    for name, value in verdicts.items()
                },
            },
        )
        # endregion
        raw_samples[str(tokens)] = samples_by_variant
        phase_profiles[str(tokens)] = {
            variant.name: _phase_profile(
                runtime_module, workspace, pool, variant, device
            )
            for variant in VARIANTS
        }
        # region agent log
        _agent_log(
            "E,F,G,H",
            "benchmarks/kimi_k3_gate_up_grouping.py:run:phases",
            "gate/up phase categories collected",
            {
                "tokens": tokens,
                "categories": {
                    name: profile["categories"]
                    for name, profile in phase_profiles[str(tokens)].items()
                },
            },
        )
        # endregion
        launch_names[str(tokens)] = {
            variant.name: _kernel_names(
                runtime_module, workspace, pool[-1], variant
            )
            for variant in VARIANTS
        }
        for variant_graphs in graphs.values():
            variant_graphs.clear()
        del pool
        _barrier(device)
        torch.cuda.empty_cache()

    resource = {
        variant.name: {
            "dynamic_shared_bytes": int(
                extension._kimi_k3_decode_gate_up_group_resource(
                    True, variant.group_size, variant.pipeline_gate_up_down
                )[0]
            ),
            "resident_blocks_per_sm": int(
                extension._kimi_k3_decode_gate_up_group_resource(
                    True, variant.group_size, variant.pipeline_gate_up_down
                )[1]
            ),
        }
        for variant in VARIANTS
    }
    m16 = next(row for row in shape_rows if row["tokens"] == 16)
    m16_winners = [
        candidate.name
        for candidate in CANDIDATES
        if m16["verdicts"][candidate.name]["passed"]
    ]
    result = {
        "candidate_status": "benchmark_only",
        "m16_winners": m16_winners,
        "rows": shape_rows,
        "numerical": numerical_rows,
        "phase_profiles": phase_profiles,
        "launch_names": launch_names,
        "resource": resource,
    }
    # region agent log
    _agent_log(
        "E,F,H",
        "benchmarks/kimi_k3_gate_up_grouping.py:run:exit",
        "gate/up benchmark completed",
        {"m16_winners": m16_winners, "resource": resource},
    )
    # endregion

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "manifest.json",
            {
                "benchmark": "kimi_k3_gate_up_grouping",
                "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
                "tokens": list(TOKENS),
                "pool_size": POOL_SIZE,
                "warmup_count": warmup_count,
                "sample_count": sample_count,
                "repeats": repeats,
                "variants": [dataclasses.asdict(value) for value in VARIANTS],
                "candidate_status": "benchmark_only",
            },
        )
        _write_json(output_dir / "results.json", result)
        _write_json(output_dir / "raw_samples.json", raw_samples)
        if DEBUG_LOG.is_file():
            (output_dir / "debug.ndjson").write_bytes(DEBUG_LOG.read_bytes())
    _barrier(device)
    dist.destroy_process_group()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(
        args.output_dir,
        warmup_count=args.warmup_count,
        sample_count=args.sample_count,
        repeats=args.repeats,
    )
    if int(os.environ["RANK"]) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
