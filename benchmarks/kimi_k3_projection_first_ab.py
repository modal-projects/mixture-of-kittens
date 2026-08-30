"""Benchmark score-first versus projection-first Kimi K3 scheduling on TP8 B300."""

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


SHAPES = (16, 32, 128)
VARIANTS = ("score_first", "projection_first")
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5
POOL_SIZE = 4
SCORE_SHARDS = 8
PROJECTION_UNITS = 28
SHARED_GATE_UP_UNITS = 12
GRID_CTAS = 148
HYPOTHESES = {
    "A": (
        "at M16 score-first strands eight long latent projections behind 128 "
        "shorter score shards, inflating route/latent barrier makespan"
    ),
    "B": (
        "projection-first merely moves the critical tail to delayed router "
        "shards, so route/latent and end-to-end makespans do not improve"
    ),
    "C": (
        "issuing shared gate/up early changes cache or execution pressure and "
        "causes a p99 regression despite reducing the route barrier"
    ),
    "D": (
        "moving shared work across barriers changes arithmetic visibility, "
        "launch count, occupancy, or output bits"
    ),
    "E": (
        "the route/latent barrier improves as modelled but another phase masks "
        "the gain in end-to-end latency"
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


def interleaved_orders(repeats: int) -> list[tuple[str, str]]:
    """Alternate A/B issue order so temporal drift is balanced."""
    if repeats < 1:
        raise ValueError("interleaved orders require at least one repeat")
    return [
        VARIANTS if repeat % 2 == 0 else VARIANTS[::-1]
        for repeat in range(repeats)
    ]


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def ab_verdict(
    repeats: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, float | bool]:
    """Require a dispersion-clearing median gain and no paired p99 loss."""
    score = list(repeats.get("score_first", ()))
    projection = list(repeats.get("projection_first", ()))
    if not score or len(score) != len(projection):
        raise ValueError("A/B variants require the same nonzero repeat count")
    score_medians = [float(row["median_ms"]) for row in score]
    projection_medians = [float(row["median_ms"]) for row in projection]
    score_center = _percentile(score_medians, 0.5)
    projection_center = _percentile(projection_medians, 0.5)
    effect_band = max(_spread(score_medians), _spread(projection_medians))
    improvement = score_center - projection_center
    p99_regression = any(
        float(candidate["p99_ms"]) > float(control["p99_ms"])
        for control, candidate in zip(score, projection, strict=True)
    )
    material = improvement > effect_band
    return {
        "score_first_median_of_medians_ms": score_center,
        "projection_first_median_of_medians_ms": projection_center,
        "improvement_ms": improvement,
        "effect_band_ms": effect_band,
        "material_improvement": material,
        "p99_regression": p99_regression,
        "integrate": material and not p99_regression,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
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


def _init_distributed() -> tuple[int, torch.device]:
    required = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if not required <= os.environ.keys():
        raise RuntimeError("launch the scheduling A/B with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError("the scheduling A/B requires TP8")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the scheduling A/B requires SM103 B300")
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


def _gather_rank_max(
    local_samples: Sequence[float],
    device: torch.device,
    timing: Any,
) -> list[float]:
    local = torch.tensor(local_samples, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return timing.rank_max_samples([rank.cpu().tolist() for rank in gathered])


def _capture_graphs(
    workspace: Any,
    pool: Sequence[Any],
    *,
    kimi: Any,
    runtime: Any,
    projection_first: bool,
    profile: bool,
) -> list[torch.cuda.CUDAGraph]:
    graphs: list[torch.cuda.CUDAGraph] = []
    with runtime.projection_first_scheduling(projection_first):
        profile_context = (
            runtime.phase_profiling()
            if profile
            else _null_context()
        )
        with profile_context:
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


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def _measure_latency(
    graphs: Sequence[torch.cuda.CUDAGraph],
    *,
    timing: Any,
    device: torch.device,
    warmup_count: int,
    sample_count: int,
) -> dict[str, object]:
    samples = timing.replay_samples(
        lambda iteration: graphs[iteration % len(graphs)].replay(),
        warmup_count=warmup_count,
        sample_count=sample_count,
        settle_count=len(graphs),
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=lambda: torch.cuda.synchronize(device),
    )
    maxima = _gather_rank_max(samples, device, timing)
    return {
        "sample_count": sample_count,
        "median_ms": timing.percentile(maxima, 0.50),
        "p90_ms": timing.percentile(maxima, 0.90),
        "p99_ms": timing.percentile(maxima, 0.99),
        "rank_max_samples_ms": maxima,
    }


def _phase_clock_tensor(
    workspace: Any,
    extension: Any,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    begin, names = extension._kimi_k3_decode_phase_clock_metadata()
    words = workspace.scratch[
        begin * 4 : (begin + 2 * len(names)) * 4
    ]
    return words.view(torch.int64), tuple(names)


def _measure_phases(
    graphs: Sequence[torch.cuda.CUDAGraph],
    workspace: Any,
    *,
    extension: Any,
    device: torch.device,
    warmup_count: int,
    sample_count: int,
) -> dict[str, object]:
    for iteration in range(warmup_count):
        graphs[iteration % len(graphs)].replay()
    torch.cuda.synchronize(device)
    counters, names = _phase_clock_tensor(workspace, extension)
    samples = torch.empty(
        sample_count,
        len(names),
        dtype=torch.int64,
        device=device,
    )
    for iteration in range(sample_count):
        graphs[iteration % len(graphs)].replay()
        samples[iteration].copy_(counters)
    torch.cuda.synchronize(device)
    dist.all_reduce(samples, op=dist.ReduceOp.MAX)
    host = samples.cpu().tolist()
    return {
        "sample_count": sample_count,
        "aggregation": "rank maximum of each launch's CTA accumulator",
        "clocks": {
            name: _distribution(
                [float(sample[index]) for sample in host]
            )
            for index, name in enumerate(names)
        },
        "samples": [
            dict(zip(names, (int(value) for value in sample), strict=True))
            for sample in host
        ],
    }


def _check_bitwise_parity(
    workspace: Any,
    pool: Sequence[Any],
    *,
    runtime: Any,
    device: torch.device,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    # region agent log
    _agent_log(
        location=(
            "benchmarks/kimi_k3_projection_first_ab.py:"
            "_check_bitwise_parity:entry"
        ),
        message="starting pool parity checks",
        data={"pool_size": len(pool)},
        hypothesis_id="F,G",
    )
    # endregion
    for enumerated_pool_index, entry in enumerate(pool):
        with runtime.projection_first_scheduling(False):
            score = runtime.decode_step(
                workspace, entry.weights, entry.hidden
            ).clone()
        with runtime.projection_first_scheduling(True):
            projection = runtime.decode_step(
                workspace, entry.weights, entry.hidden
            ).clone()
        torch.cuda.synchronize(device)
        if not torch.equal(score, projection):
            raise AssertionError("A/B outputs are not bitwise identical")
        expected = runtime.decode_reference(entry.hidden, entry.weights)
        relative_l1, cosine, maximum = runtime.assert_decode_close(
            projection, expected
        )
        runtime.assert_identical_across_ranks(projection)
        # region agent log
        _agent_log(
            location=(
                "benchmarks/kimi_k3_projection_first_ab.py:"
                "_check_bitwise_parity:metadata"
            ),
            message="parity passed before result metadata serialization",
            data={
                "enumerated_pool_index": enumerated_pool_index,
                "entry_type": type(entry).__name__,
                "entry_fields": [
                    field.name for field in dataclasses.fields(entry)
                ],
                "has_pool_index": hasattr(entry, "pool_index"),
            },
            hypothesis_id="F,G",
        )
        # endregion
        checks.append(
            {
                "pool_index": enumerated_pool_index,
                "bitwise_ab": True,
                "relative_l1": relative_l1,
                "cosine_similarity": cosine,
                "max_abs": maximum,
            }
        )
    return checks


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
    if warmup_count < 1:
        raise ValueError("warmup count must be positive")
    if sample_count != SAMPLE_COUNT:
        raise ValueError(f"sample count must be exactly {SAMPLE_COUNT}")
    if repeats != REPEATS:
        raise ValueError(f"repeats must be exactly {REPEATS}")
    rank, device = _init_distributed()
    data = importlib.import_module("benchmarks.kimi_k3_decode_data")
    runtime = importlib.import_module("benchmarks.kimi_k3_decode_runtime")
    timing = importlib.import_module("benchmarks.kimi_k3_timing")
    kimi = importlib.import_module("mok.kimi_k3")
    extension = importlib.import_module("mok._C")

    if rank == 0:
        # region agent log
        _agent_log(
            location="benchmarks/kimi_k3_projection_first_ab.py:run:entry",
            message="projection-first scheduling A/B started",
            data={
                "shapes": list(SHAPES),
                "grid_ctas": GRID_CTAS,
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
        weights, device, list(SHAPES), pool_size=POOL_SIZE
    )
    weights = dataclasses.replace(weights, router_weight=router.weight)
    pools = {
        tokens: [
            data.build_routed_input(
                weights, device, tokens, pool_index, router=router
            )
            for pool_index in range(POOL_SIZE)
        ]
        for tokens in SHAPES
    }

    correctness: dict[int, list[dict[str, object]]] = {}
    latency_graphs: dict[int, dict[str, list[torch.cuda.CUDAGraph]]] = {}
    phase_graphs: dict[int, dict[str, list[torch.cuda.CUDAGraph]]] = {}
    for tokens in SHAPES:
        correctness[tokens] = _check_bitwise_parity(
            workspace,
            pools[tokens],
            runtime=runtime,
            device=device,
        )
        latency_graphs[tokens] = {}
        phase_graphs[tokens] = {}
        for variant in VARIANTS:
            projection_first = variant == "projection_first"
            latency_graphs[tokens][variant] = _capture_graphs(
                workspace,
                pools[tokens],
                kimi=kimi,
                runtime=runtime,
                projection_first=projection_first,
                profile=False,
            )
            phase_graphs[tokens][variant] = _capture_graphs(
                workspace,
                pools[tokens],
                kimi=kimi,
                runtime=runtime,
                projection_first=projection_first,
                profile=True,
            )
        if rank == 0:
            # region agent log
            _agent_log(
                location=(
                    "benchmarks/kimi_k3_projection_first_ab.py:"
                    "run:correctness"
                ),
                message="shape passed bitwise A/B and strict reference parity",
                data={
                    "tokens": tokens,
                    "pool_entries": len(correctness[tokens]),
                    "maximum_relative_l1": max(
                        float(row["relative_l1"])
                        for row in correctness[tokens]
                    ),
                },
                hypothesis_id="D",
            )
            # endregion

    latency: dict[int, dict[str, list[dict[str, object]]]] = {
        tokens: {variant: [] for variant in VARIANTS}
        for tokens in SHAPES
    }
    phases: dict[int, dict[str, list[dict[str, object]]]] = {
        tokens: {variant: [] for variant in VARIANTS}
        for tokens in SHAPES
    }
    orders = interleaved_orders(repeats)
    for repeat, order in enumerate(orders):
        for tokens in SHAPES:
            for position, variant in enumerate(order):
                latency_row = _measure_latency(
                    latency_graphs[tokens][variant],
                    timing=timing,
                    device=device,
                    warmup_count=warmup_count,
                    sample_count=sample_count,
                )
                latency_row.update(
                    {
                        "repeat": repeat,
                        "order_position": position,
                        "candidate_order": list(order),
                    }
                )
                latency[tokens][variant].append(latency_row)
                if rank == 0:
                    # region agent log
                    _agent_log(
                        location=(
                            "benchmarks/kimi_k3_projection_first_ab.py:"
                            "run:latency"
                        ),
                        message="uninstrumented interleaved repeat completed",
                        data={
                            "tokens": tokens,
                            "variant": variant,
                            "repeat": repeat,
                            "order_position": position,
                            "median_ms": float(latency_row["median_ms"]),
                            "p99_ms": float(latency_row["p99_ms"]),
                        },
                        hypothesis_id="A,B,C,E",
                    )
                    # endregion
                phase_row = _measure_phases(
                    phase_graphs[tokens][variant],
                    workspace,
                    extension=extension,
                    device=device,
                    warmup_count=warmup_count,
                    sample_count=sample_count,
                )
                phase_row.update(
                    {
                        "repeat": repeat,
                        "order_position": position,
                        "candidate_order": list(order),
                    }
                )
                phases[tokens][variant].append(phase_row)
                if rank == 0:
                    # region agent log
                    _agent_log(
                        location=(
                            "benchmarks/kimi_k3_projection_first_ab.py:"
                            "run:phases"
                        ),
                        message="split barrier-clock repeat completed",
                        data={
                            "tokens": tokens,
                            "variant": variant,
                            "repeat": repeat,
                            "route_latent_barrier_p50_cycles": (
                                phase_row["clocks"][
                                    "route_latent_barrier"
                                ]["p50"]
                            ),
                            "route_latent_makespan_p50_cycles": (
                                phase_row["clocks"][
                                    "route_latent_makespan"
                                ]["p50"]
                            ),
                        },
                        hypothesis_id="A,B,E",
                    )
                    # endregion

    kernel_names: dict[str, list[str]] = {}
    for variant in VARIANTS:
        with runtime.projection_first_scheduling(
            variant == "projection_first"
        ):
            kernel_names[variant] = runtime.profiled_kernel_names(
                lambda: runtime.decode_device_step(
                    workspace,
                    pools[16][0].weights,
                    pools[16][0].hidden,
                )
            )
    grid_ctas, threads, dynamic_shared = (
        extension._kimi_k3_decode_grid_shape()
    )
    resources = {
        "grid_ctas": int(grid_ctas),
        "threads": int(threads),
        "dynamic_shared_bytes": int(dynamic_shared),
        "resident_blocks_per_sm": {
            "core": int(
                extension._kimi_k3_decode_resident_blocks_per_sm(False)
            ),
            "tensor": int(
                extension._kimi_k3_decode_resident_blocks_per_sm(True)
            ),
        },
        "kernel_names": kernel_names,
        "one_launch_each": all(
            len(names) == 1
            and "kimi_k3_decode_persistent_kernel" in names[0]
            for names in kernel_names.values()
        ),
    }
    if (
        resources["grid_ctas"] != GRID_CTAS
        or resources["resident_blocks_per_sm"] != {"core": 1, "tensor": 1}
        or not resources["one_launch_each"]
    ):
        raise AssertionError(resources)
    if rank == 0:
        # region agent log
        _agent_log(
            location="benchmarks/kimi_k3_projection_first_ab.py:run:resources",
            message="A/B launch and residency contract checked",
            data=resources,
            hypothesis_id="C,D",
        )
        # endregion

    verdicts = {
        tokens: ab_verdict(latency[tokens]) for tokens in SHAPES
    }
    result = {
        "hypotheses": HYPOTHESES,
        "shapes": list(SHAPES),
        "orders": [list(order) for order in orders],
        "arithmetic": {
            tokens: {
                "score_units": tokens * SCORE_SHARDS,
                "projection_units": PROJECTION_UNITS,
                "shared_gate_up_units": SHARED_GATE_UP_UNITS,
                "identical_between_variants": True,
            }
            for tokens in SHAPES
        },
        "correctness": correctness,
        "latency_repeats": {
            tokens: {
                variant: [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "rank_max_samples_ms"
                    }
                    for row in latency[tokens][variant]
                ]
                for variant in VARIANTS
            }
            for tokens in SHAPES
        },
        "phase_repeats": {
            tokens: {
                variant: [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "samples"
                    }
                    for row in phases[tokens][variant]
                ]
                for variant in VARIANTS
            }
            for tokens in SHAPES
        },
        "verdicts": verdicts,
        "m16_integrate": verdicts[16]["integrate"],
        "resources": resources,
    }
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "manifest.json",
            {
                "benchmark": "kimi_k3_projection_first_ab",
                "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
                "gpu": "B300:8",
                "shapes": list(SHAPES),
                "variants": list(VARIANTS),
                "warmup_count": warmup_count,
                "sample_count": sample_count,
                "repeats": repeats,
                "grid_ctas": GRID_CTAS,
                "launch_count": 1,
                "latency_instrumentation": "disabled",
                "phase_instrumentation": (
                    "dedicated graph replays with split per-barrier clocks"
                ),
                "hypotheses": HYPOTHESES,
            },
        )
        _write_json(output_dir / "results.json", result)
        _write_json(
            output_dir / "raw_samples.json",
            {"latency": latency, "phases": phases},
        )
        # region agent log
        _agent_log(
            location="benchmarks/kimi_k3_projection_first_ab.py:run:verdict",
            message="projection-first scheduling A/B completed",
            data={
                "verdicts": verdicts,
                "m16_integrate": result["m16_integrate"],
            },
            hypothesis_id="A,B,C,D,E",
        )
        # endregion
        print(json.dumps(result, indent=2, sort_keys=True))

    _barrier(device)
    latency_graphs.clear()
    phase_graphs.clear()
    kimi.clear_kimi_k3_decode_workspace_cache()
    dist.destroy_process_group()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kimi_k3_projection_first_ab"),
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
