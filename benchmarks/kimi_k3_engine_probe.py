"""A/B the Kimi K3 routed gate/up engines against the ring production replaced.

The baseline here is `resident`: one unit per occupied expert, that expert's
whole activation gathered once, and 42 `(task, slab)` weight transfers through a
two-stage K = 512 ring. That was production until this integration. Its measured
profile puts 40.0% of the gate/up band in that ring -- `tma_issue` plus
`tma_wait` plus `ring_full` -- and the reason it could not answer with depth is
arithmetic: a third stage wants 67,584 bytes and its launch left 14,336 above
the static shared memory ptxas assigns.

`production` buys that stage out of the activation, two ways, and picks between
them per expert inside the kernel:

* Inside the threshold -- a batch of at most four rows -- the **compact** ring
  keeps all seven slabs and the one-time gather in a quarter of the bytes, by
  packing slab `s`'s operand four rows past slab `s - 1`'s and moving every
  activation scale into tensor memory.
* Outside it, the **slab-buffered** ring stops holding the activation at all.
  Two eight-row slots replace seven sixteen-row tiles, warps 1 to 7 gather the
  next slab while warp 0 contracts the current one, and the slab loop moves
  outside the task loop so six accumulators stay open.

Both rings were separately selectable while they were being measured against
each other; the integration retired both ids, because production exercises both
and a standalone id for either would be a second way to reach code that already
runs. So the arms here are production and the ring it replaced.

The threshold is a bet on the route, so the shapes are chosen to settle it
rather than to flatter it. Three are the realistic decode routes -- M16 and M32
put one row on each expert, M128 puts two or three, so all three take the
compact ring -- and the fourth puts eight rows on every expert, which is the
shape the packing refuses and the slab-buffered ring answers. The fourth is not
a corner case: it is where the whole adaptive claim is either free or is not,
because that launch still asks for the compact ring's 228,352 shared bytes while
running the other ring inside them.

The claim is a latency claim, so this is the measurement that decides it. Every
arm is the same one launch of the same kernel on the same workspace, weights and
routes, and the engine is a template parameter -- so each arm is its own
compiled kernel, none pays another's register pressure, and a captured graph
records whichever was selected when it was captured. The pools are replayed
interleaved, five repeats of a thousand samples each, with the order rotating
per repeat so temporal drift lands on every arm.

Before any of that, production is held to bitwise equality with the baseline:
same routes, same weights, same output mailbox bytes. A faster engine that moved
a bit is not a faster engine.

The pool is deliberately cold, the same four-entry pool `bench_kimi_k3_decode.py`
tunes the production grid on, so no replay reads a resident expert.

Run under ``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.kimi_k3_decode_inputs import GRAPH_POOL_SIZE
from benchmarks.kimi_k3_timing import (
    rank_max_samples,
    replay_samples,
    summarize_rank_max,
)

from benchmarks.kimi_k3_engine_routes import (
    BAND_SUBPHASES,
    BASELINE,
    _build_routes,
    CANDIDATES,
    ENGINE_IDS,
    GATE_LABEL,
    GUARD_LABELS,
    MAXIMUM_GUARD_REGRESSION,
    MINIMUM_GATE_MEDIAN_GAIN,
    Modules,
    Pool,
    RING_SUBPHASES,
    Routes,
    SHAPES,
    VARIANTS,
)
from benchmarks.kimi_k3_engine_verdict import (
    _audit_raw_samples,
    evaluate_point,
    integration_decision,
    subphase_deltas,
    _summarize,
    variant_orders,
)


TP_SIZE = 8
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5


def _modules() -> Modules:
    return Modules(
        extension=importlib.import_module("mok._C"),
        kimi=importlib.import_module("mok.kimi_k3"),
        data=importlib.import_module("benchmarks.kimi_k3_decode_data"),
        runtime=importlib.import_module("benchmarks.kimi_k3_decode_runtime"),
    )


def _init_distributed() -> tuple[int, torch.device]:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch the engine probe with torchrun")
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != TP_SIZE:
        raise RuntimeError(f"the engine probe requires TP{TP_SIZE}")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the engine probe requires an SM103 B300")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=TP_SIZE,
        device_id=device,
    )
    return rank, device


def _barrier(device: torch.device) -> None:
    dist.barrier(
        async_op=True,
        device_ids=[device.index],
    ).block_current_stream()
    torch.cuda.synchronize(device)


def _l2_bytes(device: torch.device) -> int:
    properties = torch.cuda.get_device_properties(device)
    for name in ("L2_cache_size", "l2_cache_size"):
        value = getattr(properties, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("PyTorch did not expose the B300 L2 cache size")


def _expert_weight_bytes(weights: Any) -> int:
    """Bytes one expert's own prepared matrices occupy."""
    return sum(
        getattr(weights, name)[0].numel()
        * getattr(weights, name)[0].element_size()
        for name in (
            "expert_w13_packed",
            "expert_w13_scale",
            "expert_w2_packed",
            "expert_w2_scale",
        )
    )


def _select(modules: Modules, variant: str) -> None:
    """Point the extension at one engine, and confirm it moved."""
    engine = ENGINE_IDS[variant]
    modules.extension._kimi_k3_decode_set_gate_up_engine(engine)
    if modules.extension._kimi_k3_decode_gate_up_engine() != engine:
        raise AssertionError(variant)


def _ledger(modules: Modules) -> dict[str, dict[str, int]]:
    """The shared-memory ledger and ring shape each engine compiled with.

    Read out of the extension rather than restated here, so the artifact
    carries the numbers ptxas was given rather than a copy of them.
    """
    fields = (
        "launch_shared_bytes",
        "staging_shared_bytes",
        "weight_stages",
        "activation_slabs",
        "live_accumulators",
        "activation_gathers_per_pass",
    )
    return {
        variant: dict(
            zip(
                fields,
                (
                    int(value)
                    for value in modules.extension.
                    _kimi_k3_decode_gate_up_engine_ledger(ENGINE_IDS[variant])
                ),
                strict=True,
            )
        )
        for variant in VARIANTS
    }


def _bitwise_parity(
    modules: Modules,
    workspace: Any,
    device: torch.device,
    routes: Routes,
) -> dict[str, Any]:
    """Hold every engine to identical output bytes on identical input.

    Eager launches rather than replays, and the mailbox is copied out between
    them because the next launch overwrites it. Bitwise, not close: the engines
    contract the same prepared bytes in a different order over the K axis only
    within one slab, and a `situ` byte that differs at all means one of them
    read a byte the other did not.
    """
    checks: list[dict[str, Any]] = []
    for pool_index, (hidden, weights) in enumerate(
        zip(routes.hidden, routes.weights, strict=True)
    ):
        outputs: dict[str, torch.Tensor] = {}
        for variant in VARIANTS:
            _select(modules, variant)
            modules.kimi.kimi_k3_decode(
                modules.runtime.CONFIG, workspace, weights, hidden
            )
            torch.cuda.synchronize(device)
            modules.runtime.check_decode_error(workspace)
            outputs[variant] = workspace.output_mailbox.view(128, 7168)[
                : hidden.shape[0]
            ].clone()
        checks.append(
            {
                "pool_index": pool_index,
                "bitwise_equal": {
                    variant: bool(
                        torch.equal(outputs[BASELINE], outputs[variant])
                    )
                    for variant in CANDIDATES
                },
                "differing_elements": {
                    variant: int(
                        (outputs[BASELINE] != outputs[variant]).sum().item()
                    )
                    for variant in CANDIDATES
                },
            }
        )
    _select(modules, BASELINE)
    _barrier(device)
    if not all(
        equal
        for check in checks
        for equal in check["bitwise_equal"].values()
    ):
        raise AssertionError((routes.shape.label, checks))
    return {
        "shape": routes.shape.label,
        "tokens": routes.shape.tokens,
        "maximum_rows_per_expert": routes.maximum_rows_per_expert,
        "checks": checks,
    }


def _capture(
    modules: Modules,
    workspace: Any,
    device: torch.device,
    variant: str,
    routes: Routes,
) -> Pool:
    """Capture one engine's pool, with that engine selected at capture.

    The selection is a host-side branch in ``launch_decode``, so a graph
    records whichever kernel was chosen while it was being captured and the
    switch is irrelevant at replay. Each entry is warmed outside the capture
    first: the residency proof calls ``cudaFuncSetAttribute`` and the occupancy
    query once per compiled function per device, and a capture may not record a
    runtime API call.
    """
    _select(modules, variant)
    graphs: list[torch.cuda.CUDAGraph] = []
    for hidden, weights in zip(routes.hidden, routes.weights, strict=True):
        modules.kimi.kimi_k3_decode(
            modules.runtime.CONFIG, workspace, weights, hidden
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            modules.kimi.kimi_k3_decode(
                modules.runtime.CONFIG, workspace, weights, hidden
            )
        graphs.append(graph)
    _select(modules, BASELINE)
    _barrier(device)
    return Pool(variant, routes.shape, graphs, routes)


def _measure(
    pool: Pool,
    workspace: Any,
    device: torch.device,
    *,
    warmup_count: int,
    sample_count: int,
) -> dict[str, Any]:
    """Time one pool's replays and reduce the eight ranks to per-iteration maxima."""
    local = replay_samples(
        lambda iteration: pool.graphs[iteration % len(pool.graphs)].replay(),
        warmup_count=warmup_count,
        sample_count=sample_count,
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=lambda: torch.cuda.synchronize(device),
        settle_count=len(pool.graphs),
    )
    error = int(workspace.error_flag.item())
    tensor = torch.tensor(local, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(tensor) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, tensor)
    rank_samples = [samples.cpu().tolist() for samples in gathered]
    return {
        **summarize_rank_max(rank_samples),
        "error_flag": error,
        "rank_max_samples_ms": rank_max_samples(rank_samples),
    }


def _verify(
    modules: Modules,
    pool: Pool,
    workspace: Any,
    device: torch.device,
) -> list[dict[str, float]]:
    """Replay every entry once more and hold its output to the oracle.

    Run after each measured repeat rather than only once. The candidate's
    producers write the activation slot warp 0 is about to contract, so what it
    could break is order-dependent, and a single check would not see the arrival
    skew a thousand interleaved replays produce.
    """
    checks: list[dict[str, float]] = []
    for graph, hidden, weights in zip(
        pool.graphs, pool.routes.hidden, pool.routes.weights, strict=True
    ):
        graph.replay()
        torch.cuda.synchronize(device)
        expected = modules.runtime.decode_reference(hidden, weights)
        actual = workspace.output_mailbox.view(128, 7168)[: hidden.shape[0]]
        relative_l1, cosine, maximum = modules.runtime.assert_decode_close(
            actual, expected
        )
        modules.runtime.assert_identical_across_ranks(actual)
        checks.append(
            {
                "relative_l1": relative_l1,
                "cosine_similarity": cosine,
                "max_abs": maximum,
            }
        )
    if int(workspace.error_flag.item()) != 0:
        raise AssertionError(
            f"{pool.variant} error flag: {int(workspace.error_flag.item())}"
        )
    return checks


def _profile(
    modules: Modules,
    workspace: Any,
    device: torch.device,
    variant: str,
    routes: Routes,
) -> dict[str, Any]:
    """Read one engine's own accumulators back out of a profiled launch.

    The six gate/up subphase counters are what decides *why* an arm won or lost:
    they separate the ring's issue, its arrival wait and its full-ring stall
    from the contraction, the activation handoff and the epilogue. Every one of
    them is warp 0's timeline -- ``PhaseClocks::add`` only accumulates from
    thread 0 -- so the candidate's producer warps appear as whatever of their
    gather the ring did not already cover, and not as work added to the band.
    """
    _select(modules, variant)
    with modules.runtime.phase_profiling():
        modules.kimi.kimi_k3_decode(
            modules.runtime.CONFIG,
            workspace,
            routes.weights[0],
            routes.hidden[0],
        )
        torch.cuda.synchronize(device)
        phases = modules.runtime.phase_clock_cycles(workspace)
    _select(modules, BASELINE)
    _barrier(device)
    top_level = modules.runtime.top_level_phase_clocks()
    total = sum(phases[name] for name in top_level)
    band = phases["routed_gate_up"]
    ring = sum(phases[name] for name in RING_SUBPHASES)
    attributed = sum(phases[name] for name in BAND_SUBPHASES)
    return {
        "phase_clock_cycles": phases,
        "phase_clock_top_level": list(top_level),
        "phase_clock_total_cycles": total,
        "gate_up_band_cycles": band,
        "gate_up_band_fraction": band / total if total else 0.0,
        "ring_cycles": ring,
        "ring_share_of_band": ring / band if band else 0.0,
        "attributed_cycles": attributed,
        "attributed_share_of_band": attributed / band if band else 0.0,
        "subphase_share_of_band": {
            name: (phases[name] / band if band else 0.0)
            for name in BAND_SUBPHASES
        },
    }


def _manifest(
    *,
    dry_run: bool,
    warmup_count: int,
    sample_count: int,
    repeats: int,
    pool_size: int,
) -> dict[str, Any]:
    return {
        "benchmark": "kimi_k3_engine_probe",
        "dry_run": dry_run,
        "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
        "variants": list(VARIANTS),
        "engine_ids": dict(ENGINE_IDS),
        "shapes": [dataclasses.asdict(shape) for shape in SHAPES],
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "repeats": repeats,
        "graph_pool_size": pool_size,
        "tp_size": TP_SIZE,
        "latency_statistic": "per-iteration maximum over the eight ranks",
        "interleaving": (
            "the arm order rotates once per repeat, so temporal drift is "
            "shared rather than charged to one arm"
        ),
        "parity": "every arm's output mailbox bytes must equal production's",
        "gate": {
            "gate_shape": GATE_LABEL,
            "minimum_median_gain": MINIMUM_GATE_MEDIAN_GAIN,
            "guard_shapes": list(GUARD_LABELS),
            "maximum_regression": MAXIMUM_GUARD_REGRESSION,
            "p99": "the gate point's median repeat p99 must improve",
            "mechanism": (
                "`routed_gate_up_tma_wait` must fall at the gate point"
            ),
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
    pool_size: int,
) -> dict[str, Any]:
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    rank, device = _init_distributed()
    modules = _modules()

    base_weights = modules.data.build_weights(device, rank)
    workspace = modules.kimi.get_kimi_k3_decode_workspace(
        dist.group.WORLD, device=device
    )
    l2_bytes = _l2_bytes(device)
    expert_bytes = _expert_weight_bytes(base_weights)
    ledger = _ledger(modules)

    # Both engines' compiled functions are proved resident on both paths before
    # any capture, so no graph records the occupancy query behind that proof.
    # The query names the engine, because a different engine is a different
    # compiled function asking for a different number of dynamic shared bytes.
    residency: dict[str, dict[str, int]] = {
        variant: {
            path: int(
                modules.extension.
                _kimi_k3_decode_schedule_resident_blocks_per_sm(
                    path == "tensor", ENGINE_IDS[variant]
                )
            )
            for path in ("core", "tensor")
        }
        for variant in VARIANTS
    }
    if any(
        blocks != 1
        for paths in residency.values()
        for blocks in paths.values()
    ):
        raise AssertionError(residency)

    routes = {
        shape.label: _build_routes(
            modules, base_weights, device, shape, pool_size
        )
        for shape in SHAPES
    }
    parity = {
        label: _bitwise_parity(modules, workspace, device, entry)
        for label, entry in routes.items()
    }
    profiles = {
        variant: {
            label: _profile(modules, workspace, device, variant, entry)
            for label, entry in routes.items()
        }
        for variant in VARIANTS
    }
    deltas = {
        variant: {
            label: subphase_deltas(
                profiles[BASELINE][label], profiles[variant][label]
            )
            for label in routes
        }
        for variant in CANDIDATES
    }

    points: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    per_point: list[dict[str, Any]] = []
    for shape in SHAPES:
        entry = routes[shape.label]
        # The pool has to miss L2 for the weight ring to be measuring the copy
        # engine rather than the cache, and that is a property of the route
        # rather than of the arm -- so it is checked once, of the shared routes.
        working_set = len(entry.distinct_experts) * expert_bytes
        if working_set <= l2_bytes:
            raise AssertionError((shape.label, working_set, l2_bytes))
        pools = {
            variant: _capture(modules, workspace, device, variant, entry)
            for variant in VARIANTS
        }

        measured: dict[str, list[dict[str, Any]]] = {
            variant: [] for variant in VARIANTS
        }
        correctness: dict[str, list[Any]] = {variant: [] for variant in VARIANTS}
        for repeat_index, order in enumerate(variant_orders(repeats)):
            for position, variant in enumerate(order):
                sample = _measure(
                    pools[variant],
                    workspace,
                    device,
                    warmup_count=warmup_count,
                    sample_count=sample_count,
                )
                if sample["error_flag"] != 0:
                    raise AssertionError(
                        (variant, shape.label, sample["error_flag"])
                    )
                sample["repeat"] = repeat_index + 1
                sample["order_position"] = position
                sample["variant_order"] = list(order)
                measured[variant].append(sample)
                correctness[variant].append(
                    _verify(modules, pools[variant], workspace, device)
                )
                _barrier(device)

        summaries = {
            variant: _summarize(measured[variant]) for variant in VARIANTS
        }
        verdicts = [
            evaluate_point(
                shape=shape,
                variant=variant,
                production=summaries[BASELINE],
                candidate=summaries[variant],
            )
            for variant in CANDIDATES
        ]
        points.extend(verdicts)
        per_point.append(
            {
                "shape": shape.label,
                "tokens": shape.tokens,
                "route": shape.route,
                "purpose": shape.purpose,
                "verdicts": verdicts,
                "pool_distinct_experts": len(entry.distinct_experts),
                "pool_routed_working_set_bytes": working_set,
                "expert_row_histogram": {
                    str(rows): experts
                    for rows, experts in entry.row_histogram.items()
                },
                "maximum_rows_per_expert": entry.maximum_rows_per_expert,
                "summaries": summaries,
                "repeats": {
                    variant: [
                        {
                            key: value
                            for key, value in sample.items()
                            if key != "rank_max_samples_ms"
                        }
                        for sample in measured[variant]
                    ]
                    for variant in VARIANTS
                },
                "post_timing_checks": correctness,
                "bitwise_parity": parity[shape.label],
                "profiles": {
                    variant: profiles[variant][shape.label]
                    for variant in VARIANTS
                },
                "subphase_deltas": {
                    variant: deltas[variant][shape.label]
                    for variant in CANDIDATES
                },
            }
        )
        raw[shape.label] = {
            variant: [
                {
                    "repeat": sample["repeat"],
                    "order_position": sample["order_position"],
                    "variant_order": sample["variant_order"],
                    "reported": {
                        key: sample[key]
                        for key in (
                            "sample_count",
                            "median_ms",
                            "p90_ms",
                            "p99_ms",
                            "geomean_ms",
                        )
                    },
                    "rank_max_samples_ms": sample["rank_max_samples_ms"],
                }
                for sample in measured[variant]
            ]
            for variant in VARIANTS
        }
        _audit_raw_samples(raw[shape.label], summaries, shape.label)
        for pool in pools.values():
            pool.graphs.clear()
        _barrier(device)
        torch.cuda.empty_cache()

    decision = integration_decision(points=points, deltas=deltas)
    result = {
        "passed": decision["passed"],
        "decision": decision,
        "engine_ledger": ledger,
        "residency": residency,
        "working_set": {
            "l2_bytes": l2_bytes,
            "expert_weight_bytes": expert_bytes,
        },
        "points": per_point,
    }

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "manifest.json",
            _manifest(
                dry_run=False,
                warmup_count=warmup_count,
                sample_count=sample_count,
                repeats=repeats,
                pool_size=pool_size,
            ),
        )
        _write_json(output_dir / "results.json", result)
        _write_json(output_dir / "raw_samples.json", raw)
    _barrier(device)
    dist.destroy_process_group()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("kimi_k3_engine_probe")
    )
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--pool-size", type=int, default=GRAPH_POOL_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dry_run:
        manifest = _manifest(
            dry_run=True,
            warmup_count=args.warmup_count,
            sample_count=args.sample_count,
            repeats=args.repeats,
            pool_size=args.pool_size,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    result = run(
        args.output_dir,
        warmup_count=args.warmup_count,
        sample_count=args.sample_count,
        repeats=args.repeats,
        pool_size=args.pool_size,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
        print_points(result["points"])


def print_points(points: Sequence[dict[str, Any]]) -> None:
    """One line per arm per shape, then what moved inside the band."""
    for point in points:
        for verdict in point["verdicts"]:
            variant = verdict["variant"]
            print(
                f"{point['shape']} (r<={point['maximum_rows_per_expert']}) "
                f"{variant}: "
                f"production {verdict['production_median_ms']:.4f} ms, "
                f"candidate {verdict['candidate_median_ms']:.4f} ms, "
                f"{verdict['improvement_fraction']:+.2%}, "
                f"p99 {verdict['p99_change_fraction']:+.2%}, "
                f"{'PASS' if verdict['passed'] else 'FAIL'}"
            )
            subphases = point["subphase_deltas"][variant]["subphases"]
            for name, delta in subphases.items():
                print(
                    f"    {name}: "
                    f"{delta['production_cycles']} -> "
                    f"{delta['candidate_cycles']} cycles, "
                    f"{delta['change_fraction']:+.2%}"
                )


if __name__ == "__main__":
    main()
