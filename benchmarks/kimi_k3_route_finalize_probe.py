"""TP8 A/B probe for benchmark-only FP32 route-major deterministic finalize."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks import kimi_k3_decode_data as data
from benchmarks import kimi_k3_decode_runtime as runtime
from benchmarks import kimi_k3_route_finalize_runtime as candidate
from benchmarks.compare_kimi_k3_frameworks import derive_phase_cycles
from benchmarks.kimi_k3_timing import (
    rank_max_samples,
    summarize_rank_max,
    timing_extrema_ms,
)
from mok import kimi_k3
from tests import kimi_k3_decode_support as decode_support


DEBUG_LOG_PATH = "/opt/cursor/logs/debug.log"
DEFAULT_SAMPLES = 1000
DEFAULT_REPEATS = 5
PROFILE_TOKENS = (16, 32, 128)
PRIMARY_MODES = ("maximally_disjoint",)
VARIANTS = ("production_q24_atomic", "fp32_route_major_finalize")
PERSISTENT_KERNEL = "kimi_k3_decode_persistent_kernel"


@dataclass(frozen=True, slots=True)
class ProbeCase:
    name: str
    tokens: int
    hidden: torch.Tensor
    weights: Any
    distinct_experts: int


Step = Callable[[Any, Any, torch.Tensor], torch.Tensor]


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    payload: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as output:
        output.write(
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


def _init_distributed() -> tuple[int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError(f"route-finalize probe requires TP8, got {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("route-finalize probe requires an SM103 B300")
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


def _cases(
    base_weights: Any,
    device: torch.device,
) -> list[ProbeCase]:
    router = data.shared_router(
        base_weights,
        device,
        PROFILE_TOKENS,
        pool_size=1,
    )
    cases: list[ProbeCase] = []
    for tokens in PROFILE_TOKENS:
        routed = data.build_routed_input(
            base_weights,
            device,
            tokens,
            0,
            router=router,
        )
        cases.append(
            ProbeCase(
                "maximally_disjoint",
                tokens,
                routed.hidden,
                routed.weights,
                len(routed.distinct_experts),
            )
        )
        plan = decode_support.routing(
            "concentrated",
            device,
            tokens,
            base_weights,
        )
        cases.append(
            ProbeCase(
                "concentrated",
                tokens,
                plan.hidden,
                decode_support.with_routing(base_weights, plan),
                16,
            )
        )
    return cases


def _capture(
    step: Step,
    workspace: Any,
    case: ProbeCase,
) -> torch.cuda.CUDAGraph:
    step(workspace, case.weights, case.hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(workspace, case.weights, case.hidden)
    return graph


def _measure(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    samples: int,
) -> dict[str, Any]:
    _barrier(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
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
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    rank_samples = [values.cpu().tolist() for values in gathered]
    return {
        **summarize_rank_max(rank_samples),
        "rank_max_samples_ms": rank_max_samples(rank_samples),
    }


def _warm_graph(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
) -> None:
    for _ in range(100):
        graph.replay()
    _barrier(device)


def _profile(
    step: Step,
    workspace: Any,
    case: ProbeCase,
    device: torch.device,
) -> dict[str, int]:
    with runtime.phase_profiling():
        step(workspace, case.weights, case.hidden)
        torch.cuda.synchronize(device)
        runtime.check_decode_error(workspace)
        cycles = derive_phase_cycles(runtime.phase_clock_cycles(workspace))
    return cycles


def _accuracy(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    relative_l1, cosine, maximum = runtime.assert_decode_close(
        actual, expected
    )
    return {
        "relative_l1": relative_l1,
        "cosine_similarity": cosine,
        "maximum_absolute_error": maximum,
    }


def _validate_variant(
    label: str,
    step: Step,
    workspace: Any,
    case: ProbeCase,
    expected: torch.Tensor,
    device: torch.device,
) -> tuple[torch.cuda.CUDAGraph, dict[str, Any]]:
    # region agent log
    _debug_log(
        "A,B,C,D",
        "benchmarks/kimi_k3_route_finalize_probe.py:_validate_variant:entry",
        "route-finalize variant entry",
        {
            "label": label,
            "mode": case.name,
            "tokens": case.tokens,
            "distinct_experts": case.distinct_experts,
        },
    )
    # endregion
    eager = step(workspace, case.weights, case.hidden).clone()
    torch.cuda.synchronize(device)
    runtime.check_decode_error(workspace)
    accuracy = _accuracy(eager, expected)
    runtime.assert_identical_across_ranks(eager)
    # region agent log
    _debug_log(
        "B,C,D",
        "benchmarks/kimi_k3_route_finalize_probe.py:_validate_variant:eager",
        "route-finalize eager launch complete",
        {
            "label": label,
            "mode": case.name,
            "tokens": case.tokens,
            "accuracy": accuracy,
        },
    )
    # endregion

    kernel_names = runtime.profiled_kernel_names(
        lambda: step(workspace, case.weights, case.hidden)
    )
    if len(kernel_names) != 1 or PERSISTENT_KERNEL not in kernel_names[0]:
        raise AssertionError(
            f"{label} launched {kernel_names!r}, expected one persistent kernel"
        )

    graph = _capture(step, workspace, case)
    graph.replay()
    torch.cuda.synchronize(device)
    runtime.check_decode_error(workspace)
    first = workspace.output_mailbox.clone()
    graph.replay()
    torch.cuda.synchronize(device)
    runtime.check_decode_error(workspace)
    if not torch.equal(workspace.output_mailbox, first):
        raise AssertionError(
            f"{label} graph replay changed output for "
            f"{case.name}/M{case.tokens}"
        )

    step(workspace, case.weights, case.hidden)
    _barrier(device)
    before = torch.cuda.memory_allocated(device)
    with runtime.recorded_allocator_events(device) as allocator_events:
        result = step(workspace, case.weights, case.hidden)
    if torch.cuda.memory_allocated(device) != before:
        raise AssertionError(f"{label} changed allocated device memory")
    if allocator_events:
        raise AssertionError(
            f"{label} raised allocator events: {allocator_events!r}"
        )
    if result.data_ptr() != workspace.output_mailbox.data_ptr():
        raise AssertionError(f"{label} did not return the mailbox view")
    runtime.check_decode_error(workspace)

    # region agent log
    _debug_log(
        "A,B,C,D",
        "benchmarks/kimi_k3_route_finalize_probe.py:_validate_variant:exit",
        "route-finalize variant validation complete",
        {
            "label": label,
            "mode": case.name,
            "tokens": case.tokens,
            "launch_count": len(kernel_names),
            "graph_replay_bit_identical": True,
            "allocator_event_count": len(allocator_events),
        },
    )
    # endregion
    return graph, {
        "accuracy": accuracy,
        "graph_replay_bit_identical": True,
        "kernel_names": kernel_names,
        "launch_count": len(kernel_names),
        "per_call_allocator_events": allocator_events,
    }


def _variant_order(repeat: int) -> tuple[str, str]:
    return VARIANTS if repeat % 2 == 0 else tuple(reversed(VARIANTS))


def _aggregate_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        sample
        for repeat in repeats
        for sample in repeat["rank_max_samples_ms"]
    ]
    summary = summarize_rank_max([samples])
    medians = [float(repeat["median_ms"]) for repeat in repeats]
    return {
        **summary,
        "rank_max_samples_ms": samples,
        "repeat_medians_ms": medians,
        "geomean_repeat_median_ms": math.exp(
            sum(math.log(value) for value in medians) / len(medians)
        ),
    }


def _measure_interleaved(
    graphs: dict[str, torch.cuda.CUDAGraph],
    workspaces: dict[str, Any],
    device: torch.device,
    samples: int,
    repeats: int,
) -> tuple[dict[str, Any], float]:
    measured: dict[str, list[dict[str, Any]]] = {
        label: [] for label in VARIANTS
    }
    for graph in graphs.values():
        _warm_graph(graph, device)
    for repeat in range(repeats):
        for label in _variant_order(repeat):
            timing = _measure(graphs[label], device, samples)
            runtime.check_decode_error(workspaces[label])
            measured[label].append(
                {
                    "repeat": repeat,
                    "order": _variant_order(repeat).index(label),
                    **timing,
                }
            )
    aggregates = {
        label: {
            "repeats": measured[label],
            "aggregate": _aggregate_repeats(measured[label]),
        }
        for label in VARIANTS
    }
    baseline = aggregates["production_q24_atomic"]["aggregate"][
        "geomean_repeat_median_ms"
    ]
    route_finalize = aggregates["fp32_route_major_finalize"]["aggregate"][
        "geomean_repeat_median_ms"
    ]
    improvement_pct = 100.0 * (1.0 - route_finalize / baseline)
    return aggregates, improvement_pct


def _performance_gate(
    rows: list[dict[str, Any]],
    *,
    minimum_m16_improvement_pct: float,
    minimum_m128_improvement_pct: float,
) -> dict[str, Any]:
    primary = {
        int(row["tokens"]): float(row["improvement_pct"])
        for row in rows
        if row["mode"] in PRIMARY_MODES
    }
    missing = sorted(set((16, 128)) - set(primary))
    if missing:
        raise AssertionError(f"missing primary performance shapes: {missing}")
    checks = {
        "m16_at_least_8_percent": (
            primary[16] >= minimum_m16_improvement_pct
        ),
        "m128_no_regression": (
            primary[128] >= minimum_m128_improvement_pct
        ),
    }
    return {
        "thresholds": {
            "minimum_m16_improvement_pct": minimum_m16_improvement_pct,
            "minimum_m128_improvement_pct": minimum_m128_improvement_pct,
        },
        "observed": {
            "m16_improvement_pct": primary[16],
            "m128_improvement_pct": primary[128],
        },
        "checks": checks,
        "recommend_integration": all(checks.values()),
    }


def _run(output: Path, samples: int, repeats: int) -> None:
    previous_guard = os.environ.get("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    # region agent log
    _debug_log(
        "A,B,C,D",
        "benchmarks/kimi_k3_route_finalize_probe.py:_run",
        "route-finalize probe entry",
        {"samples": samples, "repeats": repeats},
    )
    # endregion
    rank, device = _init_distributed()
    try:
        properties = torch.cuda.get_device_properties(device)
        production_workspace = kimi_k3.get_kimi_k3_decode_workspace(
            dist.group.WORLD,
            device=device,
        )
        candidate_workspace = candidate.create_candidate_workspace(
            production_workspace
        )
        base_weights = data.build_weights(device, rank)
        cases = _cases(base_weights, device)
        # region agent log
        _debug_log(
            "C,D",
            "benchmarks/kimi_k3_route_finalize_probe.py:_run:cases",
            "route-finalize cases built",
            {
                "rank": rank,
                "cases": [
                    {
                        "mode": case.name,
                        "tokens": case.tokens,
                        "distinct_experts": case.distinct_experts,
                    }
                    for case in cases
                ],
            },
        )
        # endregion
        rows: list[dict[str, Any]] = []
        with candidate.route_finalize_enabled():
            for case in cases:
                # region agent log
                _debug_log(
                    "C,D",
                    "benchmarks/kimi_k3_route_finalize_probe.py:_run:case_entry",
                    "validating route-finalize case",
                    {
                        "rank": rank,
                        "mode": case.name,
                        "tokens": case.tokens,
                        "distinct_experts": case.distinct_experts,
                    },
                )
                # endregion
                expected = runtime.decode_reference(case.hidden, case.weights)
                graphs: dict[str, torch.cuda.CUDAGraph] = {}
                validations: dict[str, Any] = {}
                steps = {
                    "production_q24_atomic": runtime.decode_device_step,
                    "fp32_route_major_finalize": candidate.decode_device_step,
                }
                workspaces = {
                    "production_q24_atomic": production_workspace,
                    "fp32_route_major_finalize": candidate_workspace,
                }
                for label in VARIANTS:
                    graph, validation = _validate_variant(
                        label, steps[label], workspaces[label], case,
                        expected, device,
                    )
                    graphs[label] = graph
                    validations[label] = validation

                case_repeats = repeats
                timings, improvement_pct = _measure_interleaved(
                    graphs, workspaces, device, samples, case_repeats
                )
                cycles = {
                    label: _profile(
                        steps[label], workspaces[label], case, device
                    )
                    for label in VARIANTS
                }
                for label in VARIANTS:
                    timing = timings[label]["aggregate"]
                    minimum_ms, maximum_ms = timing_extrema_ms(timing)
                    timing["minimum_ms"] = minimum_ms
                    timing["maximum_ms"] = maximum_ms
                row = {
                    "mode": case.name,
                    "tokens": case.tokens,
                    "distinct_experts": case.distinct_experts,
                    "samples_per_repeat": samples,
                    "repeat_count": case_repeats,
                    "interleaved_orders": [
                        list(_variant_order(repeat))
                        for repeat in range(case_repeats)
                    ],
                    "variants": {
                        label: {
                            "validation": validations[label],
                            "cycles": cycles[label],
                            "timing": timings[label],
                        }
                        for label in VARIANTS
                    },
                    "improvement_pct": improvement_pct,
                }
                rows.append(row)
                # region agent log
                _debug_log(
                    "A,B,C,D",
                    "benchmarks/kimi_k3_route_finalize_probe.py:_run:case_exit",
                    "route-finalize case measured",
                    {
                        "rank": rank,
                        "mode": case.name,
                        "tokens": case.tokens,
                        "improvement_pct": improvement_pct,
                        "candidate_accuracy": validations[
                            "fp32_route_major_finalize"
                        ]["accuracy"],
                        "candidate_routed_down_cycles": cycles[
                            "fp32_route_major_finalize"
                        ]["routed_down"],
                    },
                )
                # endregion
                del expected, graphs
                _barrier(device)
        gate = _performance_gate(
            rows,
            minimum_m16_improvement_pct=8.0,
            minimum_m128_improvement_pct=0.0,
        )
        # region agent log
        _debug_log(
            "A,B",
            "benchmarks/kimi_k3_route_finalize_probe.py:_run:gate",
            "route-finalize integration gate evaluated",
            {"rank": rank, **gate},
        )
        # endregion
        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "candidate": "fp32_route_major_deterministic_finalize",
                        "production_dispatch": "production_q24_atomic",
                        "gpu": properties.name,
                        "candidate_scratch_bytes": (
                            candidate_workspace.scratch.numel()
                        ),
                        "performance_gate": gate,
                        "rows": rows,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        _barrier(device)
    finally:
        if previous_guard is None:
            os.environ.pop("MOK_KIMI_K3_ENABLE_GRID_TUNING", None)
        else:
            os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = previous_guard
        kimi_k3.clear_kimi_k3_decode_workspace_cache()
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kimi_k3_route_finalize_candidate.json"),
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.samples < DEFAULT_SAMPLES:
        raise ValueError(f"samples must be at least {DEFAULT_SAMPLES}")
    if arguments.repeats < DEFAULT_REPEATS:
        raise ValueError(f"repeats must be at least {DEFAULT_REPEATS}")
    _run(arguments.output, arguments.samples, arguments.repeats)


if __name__ == "__main__":
    main()
