"""Prove a pool of captured native graphs replays a pool of routings.

A CUDA graph records the address of the router weight, not its contents. The
comparison harness used to load a fresh router before each capture, which meant
the four graphs of a pool all pointed at one tensor holding whichever routing
was written last -- so every replay routed to the final entry's expert block and
the pool's route diversity was a property of the plan alone.

This runs both constructions on the device, in the framework image, against the
framework's own router, and reports what each one actually replays:

``mutable``
    Four per-entry routers, reloaded between captures, exactly as the harness
    used to build them. Expected to collapse.
``shared``
    One immutable router carrying every entry's routing in disjoint hidden
    columns, bound once before anything is captured. Expected to hold.

The probe fails if the shared arm does not hold. It also fails if the mutable
arm does *not* collapse, because then it is not reproducing the defect it exists
to demonstrate and its GREEN arm proves nothing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

PROBE_TOKENS = 16
POOL_SIZE = 4


def _init_distributed() -> tuple[int, Any]:
    import torch

    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return rank, device


def _mutable_pool(base: Any, device: Any, tokens: int) -> list[Any]:
    """The pre-fix construction: one private router per pool entry.

    Kept here rather than in the data module because nothing should build a
    pool this way again; it exists so the collapse can be reproduced.
    """
    import dataclasses

    import torch

    from benchmarks.kimi_k3_decode_data import HIDDEN, RoutedInput
    from benchmarks.kimi_k3_decode_inputs import (
        MAX_TOKENS,
        NUM_EXPERTS,
        route_assignments,
    )

    pool = []
    for pool_index in range(POOL_SIZE):
        intended = route_assignments(tokens, pool_index)
        hidden = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16, device=device)
        router_weight = torch.zeros(
            NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device
        )
        for token, experts in enumerate(intended):
            column = pool_index * MAX_TOKENS + token
            hidden[token, column] = 8.0
            for slot, expert in enumerate(experts):
                router_weight[expert, column] = 0.25 - 0.0078125 * slot
        pool.append(
            RoutedInput(
                hidden=hidden,
                weights=dataclasses.replace(base, router_weight=router_weight),
                route_assignments=intended,
                distinct_experts=tuple(
                    sorted({expert for row in intended for expert in row})
                ),
            )
        )
    return pool


def _replayed_routes(adapter: Any, pool: list[Any], device: Any) -> list[list[list[int]]]:
    import torch

    from benchmarks.frameworks.kimi_k3_adapter_common import observed_routes

    graphs, id_buffers = adapter.capture_router(pool)
    routes = []
    for graph, ids in zip(graphs, id_buffers, strict=True):
        graph.replay()
        torch.cuda.synchronize(device)
        routes.append(observed_routes(ids))
    adapter.release_router()
    return routes


def _arm(
    name: str,
    adapter: Any,
    pool: list[Any],
    device: Any,
    *,
    reload_between_captures: bool,
) -> dict[str, Any]:
    """Capture, replay, and report what one construction actually routes to."""
    import torch

    from benchmarks.kimi_k3_decode_inputs import verify_graph_routes

    if reload_between_captures:
        # The pre-fix capture path: every entry's router is written over the
        # same parameter storage the previous graph recorded the address of.
        from benchmarks.frameworks.kimi_k3_adapter_common import copy_into

        for entry in pool:
            copy_into(adapter._layer.gate.weight, entry.weights.router_weight)
            torch.cuda.synchronize(device)

    intended = [entry.route_assignments for entry in pool]
    routes = _replayed_routes(adapter, pool, device)
    record: dict[str, Any] = {
        "arm": name,
        "intended_route_assignments": [
            [list(token) for token in entry] for entry in intended
        ],
        "observed_route_assignments_by_graph": routes,
        "distinct_observed_route_sets": len(
            {
                tuple(tuple(sorted(token)) for token in entry)
                for entry in routes
            }
        ),
    }
    try:
        record["verification"] = verify_graph_routes(intended, routes)
        record["held"] = True
        record["failure"] = None
    except AssertionError as failure:
        record["held"] = False
        record["failure"] = str(failure)
    return record


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--output-dir", type=Path, default=Path("kimi_k3_graph_routes"))
    parser.add_argument("--tokens", type=int, default=PROBE_TOKENS)
    arguments = parser.parse_args(argv)

    from benchmarks import kimi_k3_decode_data as data
    from benchmarks.kimi_k3_comparison_manifest import ADAPTER_MODULES, TP_SIZE

    rank, device = _init_distributed()
    weights = data.build_weights(device, rank)
    adapter_module = importlib.import_module(ADAPTER_MODULES[arguments.framework])
    adapter = adapter_module.build_adapter(
        device=device,
        tp_rank=rank,
        tp_size=TP_SIZE,
        weights=weights,
    )

    mutable = _arm(
        "mutable",
        adapter,
        _mutable_pool(weights, device, arguments.tokens),
        device,
        reload_between_captures=True,
    )

    router = data.shared_router(weights, device, [arguments.tokens])
    adapter.bind_router(router.weight, router.correction_bias)
    fingerprint = adapter.router_fingerprint()
    shared_pool = [
        data.build_routed_input(
            weights, device, arguments.tokens, index, router=router
        )
        for index in range(POOL_SIZE)
    ]
    shared = _arm(
        "shared",
        adapter,
        shared_pool,
        device,
        reload_between_captures=False,
    )
    shared["router_unchanged"] = adapter.router_fingerprint() == fingerprint

    report = {
        "framework": arguments.framework,
        "tokens": arguments.tokens,
        "pool_size": POOL_SIZE,
        "arms": [mutable, shared],
    }
    if rank == 0:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        (arguments.output_dir / "graph_routes.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))

    torch.distributed.barrier()
    adapter.close()
    torch.distributed.destroy_process_group()

    if mutable["held"]:
        raise AssertionError(
            "the mutable-router pool did not collapse, so this probe is not "
            "reproducing the defect its passing arm is measured against"
        )
    if not shared["held"]:
        raise AssertionError(shared["failure"])
    if not shared["router_unchanged"]:
        raise AssertionError("the shared router changed across capture")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
