"""Focused routed-down baseline probe for the deterministic-finalize candidate.

This benchmark does not select or implement the candidate.  It records the
production head's routed-down residual, barrier cost, route sensitivity, and
full-step latency before any behavior changes are made.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks import kimi_k3_decode_data as data
from benchmarks import kimi_k3_decode_runtime as runtime
from benchmarks.compare_kimi_k3_frameworks import derive_phase_cycles
from benchmarks.kimi_k3_timing import rank_max_samples, summarize_rank_max
from mok import kimi_k3
from tests import kimi_k3_decode_support as decode_support


DEBUG_LOG_PATH = "/opt/cursor/logs/debug.log"
DEFAULT_SAMPLES = 1000
PROFILE_TOKENS = (16, 32, 128)


@dataclass(frozen=True, slots=True)
class ProbeCase:
    name: str
    tokens: int
    hidden: torch.Tensor
    weights: Any
    distinct_experts: int


def _agent_log(
    hypothesis_id: str,
    location: str,
    message: str,
    payload: dict[str, Any],
) -> None:
    Path(DEBUG_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    # region agent log
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"hypothesisId": hypothesis_id, "location": location, "message": message, "data": payload, "timestamp": time.time_ns() // 1_000_000}, sort_keys=True) + "\n")
    # endregion


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
    workspace: Any,
    case: ProbeCase,
) -> torch.cuda.CUDAGraph:
    runtime.decode_device_step(workspace, case.weights, case.hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runtime.decode_device_step(workspace, case.weights, case.hidden)
    return graph


def _measure(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    samples: int,
) -> dict[str, Any]:
    for _ in range(100):
        graph.replay()
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


def _profile(
    workspace: Any,
    case: ProbeCase,
    device: torch.device,
) -> dict[str, int]:
    with runtime.phase_profiling():
        runtime.decode_step(workspace, case.weights, case.hidden)
        torch.cuda.synchronize(device)
        cycles = derive_phase_cycles(runtime.phase_clock_cycles(workspace))
    return cycles


def _run(output: Path, samples: int) -> None:
    previous_guard = os.environ.get("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    rank, device = _init_distributed()
    try:
        properties = torch.cuda.get_device_properties(device)
        workspace = kimi_k3.get_kimi_k3_decode_workspace(
            dist.group.WORLD,
            device=device,
        )
        base_weights = data.build_weights(device, rank)
        cases = _cases(base_weights, device)
        if rank == 0:
            # region agent log
            _agent_log("A|B|C|D|E", "kimi_k3_route_finalize_probe.py:_run", "baseline probe entered", {"gpu": properties.name, "sm_count": properties.multi_processor_count, "samples": samples, "cases": [{"mode": case.name, "tokens": case.tokens, "distinct_experts": case.distinct_experts} for case in cases]})
            # endregion
        rows: list[dict[str, Any]] = []
        for case in cases:
            if rank == 0:
                # region agent log
                _agent_log("A|C|D", "kimi_k3_route_finalize_probe.py:_run:case", "case capture begins", {"mode": case.name, "tokens": case.tokens, "distinct_experts": case.distinct_experts})
                # endregion
            graph = _capture(workspace, case)
            graph.replay()
            torch.cuda.synchronize(device)
            expected = workspace.output_mailbox.clone()
            graph.replay()
            torch.cuda.synchronize(device)
            if not torch.equal(workspace.output_mailbox, expected):
                raise AssertionError(
                    f"production graph replay changed output for {case.name}/M{case.tokens}"
                )
            runtime.assert_identical_across_ranks(
                workspace.output_mailbox.view(128, 7168)[: case.tokens]
            )
            timing = _measure(graph, device, samples)
            cycles = _profile(workspace, case, device)
            row = {
                "mode": case.name,
                "tokens": case.tokens,
                "distinct_experts": case.distinct_experts,
                "sample_count": samples,
                "median_ms": timing["median_ms"],
                "p99_ms": timing["p99_ms"],
                "minimum_ms": timing["minimum_ms"],
                "maximum_ms": timing["maximum_ms"],
                "cycles": cycles,
            }
            rows.append(row)
            if rank == 0:
                # region agent log
                _agent_log("A|B|C|D|E", "kimi_k3_route_finalize_probe.py:_run:result", "baseline case measured", row)
                # endregion
            del graph, expected
            _barrier(device)
        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "candidate": "production_q24_atomic_baseline",
                        "gpu": properties.name,
                        "rows": rows,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            # region agent log
            _agent_log("A|B|C|D|E", "kimi_k3_route_finalize_probe.py:_run:exit", "baseline probe completed", {"row_count": len(rows), "output": str(output)})
            # endregion
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
        default=Path("kimi_k3_route_finalize_baseline.json"),
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.samples < DEFAULT_SAMPLES:
        raise ValueError(f"samples must be at least {DEFAULT_SAMPLES}")
    _run(arguments.output, arguments.samples)


if __name__ == "__main__":
    main()
