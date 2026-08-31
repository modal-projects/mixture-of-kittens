"""A/B the dependency-local Kimi K3 decode schedule against production.

The candidate in ``csrc/kimi_k3_decode/persistent_schedule.cuh`` keeps one of
the production kernel's five full-grid barriers and replaces the rest with seven
topologically ordered task queues and ten bounded readiness edges. The claim it
exists to test is a latency claim, so this is the measurement that decides it.

Both schedules are the same one launch of one kernel on the same workspace, the
same weights, and the same routes, so the only difference a sample can carry is
the order of arrival. The two are captured into two graph pools and replayed
interleaved, five repeats of a thousand samples each, with the order rotated per
repeat so temporal drift lands on both variants rather than on one.

The pool is deliberately cold: four entries, each routing every token to its own
sixteen-expert block, so the pool-wide routed weight working set is far past the
B300's L2 and no replay reads a resident expert. That is the same pool
``bench_kimi_k3_decode.py`` tunes the production grid on.

Run under ``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.kimi_k3_decode_inputs import GRAPH_POOL_SIZE
from benchmarks.kimi_k3_timing import (
    geometric_mean,
    percentile,
    rank_max_samples,
    replay_samples,
    summarize_rank_max,
)


TP_SIZE = 8
TOKEN_COUNTS = (16, 32, 128)
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5
VARIANTS = ("production", "candidate")

# The acceptance thresholds the subtask is measured against. The M16 gain is
# the point of the change: block-16 concurrency-one is the primary shape and
# the one whose profile put a fifth of the step in barrier wait. The M128 limit
# is the guard, because the candidate's readiness edges cost atomics that a
# shape already saturating the routed queues cannot amortize.
MINIMUM_M16_MEDIAN_GAIN = 0.08
MAXIMUM_M128_REGRESSION = 0.01
GATE_POINT = 16
GUARD_POINT = 128

# The bar the change was asked to clear to be promoted, which is not the bar the
# experiment was designed around.
#
# Both are reported, and the experiment gate keeps its 8% and keeps failing. The
# distinction is the whole reason there are two numbers rather than one edited
# one: 8% was the estimate of what removing the barrier idle was worth, the
# measurement said 3.25%, and lowering the 8% afterwards would turn a
# quantified over-estimate into no record at all. So `passed` stays the
# experiment's verdict and `promotion_passed` is the separate question of
# whether a smaller, real, repeatable gain is worth integrating.
PROMOTION_M16_MEDIAN_GAIN = 0.02


@dataclass(slots=True)
class Modules:
    extension: ModuleType
    kimi: ModuleType
    data: ModuleType
    runtime: ModuleType


@dataclass(slots=True)
class Pool:
    """One captured graph pool: one variant at one token count."""

    variant: str
    tokens: int
    graphs: list[torch.cuda.CUDAGraph]
    hidden: list[torch.Tensor]
    weights: list[Any]
    distinct_experts: tuple[int, ...]


def _modules() -> Modules:
    # Imported here rather than at module scope so ``--dry-run`` can validate
    # the plan and the manifest on a host with no CUDA extension built.
    return Modules(
        extension=importlib.import_module("mok._C"),
        kimi=importlib.import_module("mok.kimi_k3"),
        data=importlib.import_module("benchmarks.kimi_k3_decode_data"),
        runtime=importlib.import_module("benchmarks.kimi_k3_decode_runtime"),
    )


def _init_distributed() -> tuple[int, torch.device]:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch the schedule probe with torchrun")
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != TP_SIZE:
        raise RuntimeError(f"the schedule probe requires TP{TP_SIZE}")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the schedule probe requires an SM103 B300")
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
    """Point the extension at one schedule, and confirm it moved."""
    candidate = variant == "candidate"
    modules.extension._kimi_k3_decode_set_dependency_schedule(candidate)
    if modules.extension._kimi_k3_decode_dependency_schedule() != candidate:
        raise AssertionError(variant)


def _capture(
    modules: Modules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    variant: str,
    tokens: int,
    pool_size: int,
) -> Pool:
    """Capture one variant's pool, with that variant selected at capture.

    The selection is a host-side branch in ``launch_decode``, so a graph
    records whichever kernel was chosen while it was being captured and the
    switch is irrelevant at replay. Each entry is warmed outside the capture
    first: the residency proof calls ``cudaFuncSetAttribute`` and the occupancy
    query once per compiled function per device, and a capture may not record a
    runtime API call.
    """
    _select(modules, variant)
    graphs: list[torch.cuda.CUDAGraph] = []
    hidden_states: list[torch.Tensor] = []
    weights: list[Any] = []
    experts: set[int] = set()
    for pool_index in range(pool_size):
        routed = modules.data.build_routed_input(
            base_weights, device, tokens, pool_index
        )
        modules.kimi.kimi_k3_decode(
            modules.runtime.CONFIG, workspace, routed.weights, routed.hidden
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            modules.kimi.kimi_k3_decode(
                modules.runtime.CONFIG,
                workspace,
                routed.weights,
                routed.hidden,
            )
        graphs.append(graph)
        hidden_states.append(routed.hidden)
        weights.append(routed.weights)
        experts.update(routed.distinct_experts)
    _select(modules, "production")
    _barrier(device)
    return Pool(
        variant, tokens, graphs, hidden_states, weights, tuple(sorted(experts))
    )


def _measure(
    pool: Pool,
    workspace: Any,
    device: torch.device,
    *,
    warmup_count: int,
    sample_count: int,
) -> dict[str, Any]:
    """Time one pool's replays and reduce the eight ranks to per-iteration maxima.

    A decode step is a collective, so the latency of one iteration is the
    slowest rank's: a fast rank only means it waited inside the tail for a slow
    one. The maxima are taken per iteration rather than per rank so the summary
    is of steps rather than of a rank that happened to lag on average.
    """
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

    Run after each measured repeat rather than only once, because the property
    the candidate could break is order-dependent: a readiness edge that is
    merely usually sufficient would pass a single check and fail under the
    arrival skew a thousand interleaved replays produce.
    """
    checks: list[dict[str, float]] = []
    for graph, hidden, weights in zip(
        pool.graphs, pool.hidden, pool.weights, strict=True
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


def _barriers_taken(
    modules: Modules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    variant: str,
    tokens: int,
) -> int:
    """Count the full-grid rendezvous one unprofiled launch of a variant takes.

    Every full-grid barrier advances ``kGridGeneration`` exactly once, so the
    value a launch leaves behind, from zero, is the number of times its 148
    CTAs rendezvoused. This is the barrier-reduction number the report quotes;
    it is measured rather than read off the sources.
    """
    routed = modules.data.build_routed_input(base_weights, device, tokens, 0)
    _select(modules, variant)
    modules.kimi.kimi_k3_decode(
        modules.runtime.CONFIG, workspace, routed.weights, routed.hidden
    )
    torch.cuda.synchronize(device)
    _barrier(device)

    generation = modules.extension._kimi_k3_decode_timeout_metadata()[1]
    phase = workspace.scratch[: 128 * 4].view(torch.int32)
    phase[generation].zero_()
    _barrier(device)
    modules.kimi.kimi_k3_decode(
        modules.runtime.CONFIG, workspace, routed.weights, routed.hidden
    )
    torch.cuda.synchronize(device)
    taken = int(phase[generation].item())
    _select(modules, "production")
    _barrier(device)
    return taken


def _profile(
    modules: Modules,
    workspace: Any,
    base_weights: Any,
    device: torch.device,
    variant: str,
    tokens: int,
) -> dict[str, Any]:
    """Read one variant's own accumulators back out of a profiled launch.

    The phase clocks say where the cycles went; for the candidate the schedule
    band additionally says which readiness edge they went into and which queue
    set the step's makespan. Both are summed over the 148 resident CTAs, so
    they compare regions of one launch rather than wall time.
    """
    routed = modules.data.build_routed_input(base_weights, device, tokens, 0)
    _select(modules, variant)
    with modules.runtime.phase_profiling():
        modules.kimi.kimi_k3_decode(
            modules.runtime.CONFIG, workspace, routed.weights, routed.hidden
        )
        torch.cuda.synchronize(device)
        phases = modules.runtime.phase_clock_cycles(workspace)
        edges = (
            modules.runtime.schedule_edge_cycles(workspace)
            if variant == "candidate"
            else {}
        )
    _select(modules, "production")
    _barrier(device)
    # Only the top-level bands. Summing every counter adds the diagnostic
    # children to the parents that contain them, which inflates the denominator
    # and understates every share taken against it -- including the barrier
    # share this whole experiment was sized from.
    top_level = modules.runtime.top_level_phase_clocks()
    total = sum(phases[name] for name in top_level)
    # Both variants wait in the same two top-level bands, so this numerator and
    # the denominator above are the same accounting. The candidate's per-edge
    # counters split `readiness_wait` by edge rather than adding to it: adding
    # them here would count the same cycles twice in the numerator only, which
    # is what the first revision of this harness did.
    wait = phases["grid_barrier"] + phases["readiness_wait"]
    edge_wait = sum(edges.get("edge_wait_cycles", {}).values())
    return {
        "phase_clock_cycles": phases,
        "phase_clock_top_level": list(top_level),
        "phase_clock_total_cycles": total,
        "wait_cycles": wait,
        # The band and the ten edge counters lap the same reading, so the band
        # is the sum of the edges up to the one clock read that separates the
        # two accumulators. Reported so the artifact carries the check rather
        # than the claim, and as a ratio rather than a bound so that a band that
        # stopped containing the edges shows up as a number and not only as a
        # flipped boolean.
        "edge_wait_cycles_total": edge_wait,
        "edge_wait_inside_readiness_band": (
            edge_wait <= phases["readiness_wait"]
        ),
        "edge_wait_share_of_readiness_band": (
            edge_wait / phases["readiness_wait"]
            if phases["readiness_wait"]
            else 0.0
        ),
        "grid_barrier_fraction": (
            phases["grid_barrier"] / total if total else 0.0
        ),
        "readiness_wait_fraction": (
            phases["readiness_wait"] / total if total else 0.0
        ),
        "wait_fraction": wait / total if total else 0.0,
        **edges,
    }


def variant_orders(repeats: int) -> list[tuple[str, ...]]:
    """Alternate which variant runs first, so drift is shared not charged.

    Two variants and an odd repeat count cannot be balanced exactly, so the
    order alternates strictly rather than rotating: with five repeats the
    candidate runs first twice and second three times, and the per-repeat
    ``order_position`` is recorded so the asymmetry is visible in the artifact
    rather than hidden in the aggregate.
    """
    if repeats < 1:
        raise ValueError("the A/B needs at least one repeat")
    forward = tuple(VARIANTS)
    return [
        forward if repeat % 2 == 0 else tuple(reversed(forward))
        for repeat in range(repeats)
    ]


def _summarize(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    medians = [float(repeat["median_ms"]) for repeat in repeats]
    p99s = [float(repeat["p99_ms"]) for repeat in repeats]
    center = percentile(medians, 0.5)
    return {
        "repeat_count": len(repeats),
        "repeat_medians_ms": medians,
        "repeat_p99s_ms": p99s,
        "median_of_repeat_medians_ms": center,
        "median_dispersion_ms": max(medians) - min(medians),
        "median_of_repeat_p99s_ms": percentile(p99s, 0.5),
        "geomean_of_repeat_medians_ms": geometric_mean(medians),
    }


def _audit_raw_samples(
    raw: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    tokens: int,
) -> None:
    """Recompute every reported number from the retained samples, and refuse a
    disagreement.

    The verdict is a claim about a median and a p99, and the only thing standing
    behind either is a reduction the harness performed once and then discarded
    its input. Retaining the input is what makes the claim auditable; deriving
    the reported numbers from it again, here, is what makes the audit something
    the run has already done rather than something a reader could do.

    Exact equality, not a tolerance. Both sides are the same R-7 quantile of
    the same float list, so any difference at all means the samples in the
    artifact are not the samples the verdict was read from.
    """
    for variant, entries in raw.items():
        summary = summaries[variant]
        recomputed_medians = []
        recomputed_p99s = []
        for entry in entries:
            samples = entry["rank_max_samples_ms"]
            reported = entry["reported"]
            if len(samples) != reported["sample_count"]:
                raise AssertionError(
                    (tokens, variant, entry["repeat"], "sample count",
                     len(samples), reported["sample_count"])
                )
            for key, quantile in (
                ("median_ms", 0.5), ("p90_ms", 0.9), ("p99_ms", 0.99)
            ):
                again = percentile(samples, quantile)
                if again != reported[key]:
                    raise AssertionError(
                        (tokens, variant, entry["repeat"], key, again,
                         reported[key])
                    )
            recomputed_medians.append(reported["median_ms"])
            recomputed_p99s.append(reported["p99_ms"])
        if recomputed_medians != summary["repeat_medians_ms"]:
            raise AssertionError((tokens, variant, "repeat medians"))
        if recomputed_p99s != summary["repeat_p99s_ms"]:
            raise AssertionError((tokens, variant, "repeat p99s"))
        if percentile(recomputed_medians, 0.5) != summary[
            "median_of_repeat_medians_ms"
        ]:
            raise AssertionError((tokens, variant, "median of medians"))
        if percentile(recomputed_p99s, 0.5) != summary[
            "median_of_repeat_p99s_ms"
        ]:
            raise AssertionError((tokens, variant, "median of p99s"))


def evaluate_point(
    *,
    tokens: int,
    production: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Score one token count, and say which requirement it was scored against.

    Two different questions are being asked at two different shapes. At the
    gate point the candidate has to be faster by a stated margin *and* by more
    than the measured repeat-median dispersion, because a gain inside the
    dispersion is not a gain. At every other shape it only has to not regress
    past the limit. A shape that is neither is reported and gates nothing.
    """
    production_center = float(production["median_of_repeat_medians_ms"])
    candidate_center = float(candidate["median_of_repeat_medians_ms"])
    if not (
        math.isfinite(production_center)
        and math.isfinite(candidate_center)
        and production_center > 0.0
        and candidate_center > 0.0
    ):
        raise ValueError(f"{tokens}: repeat medians must be finite and positive")

    effect_band = max(
        float(production["median_dispersion_ms"]),
        float(candidate["median_dispersion_ms"]),
    )
    improvement = production_center - candidate_center
    fraction = improvement / production_center
    production_p99 = float(production["median_of_repeat_p99s_ms"])
    candidate_p99 = float(candidate["median_of_repeat_p99s_ms"])
    p99_improved = candidate_p99 < production_p99

    verdict: dict[str, Any] = {
        "tokens": tokens,
        "production_median_ms": production_center,
        "candidate_median_ms": candidate_center,
        "improvement_ms": improvement,
        "improvement_fraction": fraction,
        "effect_band_ms": effect_band,
        "outside_effect_band": improvement > effect_band,
        "production_p99_ms": production_p99,
        "candidate_p99_ms": candidate_p99,
        "p99_improved": p99_improved,
        "p99_change_fraction": (candidate_p99 - production_p99) / production_p99,
    }
    if tokens == GATE_POINT:
        verdict["requirement"] = (
            f"median gain >= {MINIMUM_M16_MEDIAN_GAIN:.0%}, outside the "
            "repeat-median dispersion, with an improved p99"
        )
        verdict["passed"] = bool(
            fraction >= MINIMUM_M16_MEDIAN_GAIN
            and improvement > effect_band
            and p99_improved
        )
        verdict["promotion_requirement"] = (
            f"median gain >= {PROMOTION_M16_MEDIAN_GAIN:.0%}, outside the "
            "repeat-median dispersion, with an improved p99"
        )
        verdict["promotion_passed"] = bool(
            fraction >= PROMOTION_M16_MEDIAN_GAIN
            and improvement > effect_band
            and p99_improved
        )
        verdict["gating"] = True
        return verdict
    if tokens == GUARD_POINT:
        verdict["requirement"] = (
            f"regression <= {MAXIMUM_M128_REGRESSION:.0%}"
        )
        verdict["passed"] = bool(-fraction <= MAXIMUM_M128_REGRESSION)
        # The guard is the same either way: a regression the promotion bar
        # tolerated and the experiment gate did not would be a difference in
        # what "do no harm" means, and there is only one meaning of that.
        verdict["promotion_requirement"] = verdict["requirement"]
        verdict["promotion_passed"] = verdict["passed"]
        verdict["gating"] = True
        return verdict
    verdict["requirement"] = "reported, gates nothing"
    verdict["passed"] = True
    verdict["promotion_requirement"] = "reported, gates nothing"
    verdict["promotion_passed"] = True
    verdict["gating"] = False
    return verdict


def failing_edge(
    points: Sequence[dict[str, Any]],
    candidate_profiles: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Name the readiness edge to blame when a shape did not clear its bar.

    A verdict of "slower" is not actionable on its own. The per-edge makespan
    band says which single wait was the longest any CTA of the launch paid, and
    the queue makespans say which queue the step's length was set by, so the
    report can name a next step rather than a number.
    """
    failed = [point for point in points if not point["passed"]]
    if not failed:
        return None
    worst = min(failed, key=lambda point: point["improvement_fraction"])
    profile = candidate_profiles.get(int(worst["tokens"]), {})
    makespans: dict[str, int] = profile.get("edge_makespan_cycles", {})
    waits: dict[str, int] = profile.get("edge_wait_cycles", {})
    queues: dict[str, int] = profile.get("queue_makespan_cycles", {})
    return {
        "tokens": int(worst["tokens"]),
        "requirement": worst["requirement"],
        "improvement_fraction": worst["improvement_fraction"],
        "longest_single_wait_edge": (
            max(makespans, key=lambda name: makespans[name])
            if makespans
            else None
        ),
        "largest_accumulated_wait_edge": (
            max(waits, key=lambda name: waits[name]) if waits else None
        ),
        "binding_queue": (
            max(queues, key=lambda name: queues[name]) if queues else None
        ),
        "edge_makespan_cycles": makespans,
        "edge_wait_cycles": waits,
        "queue_makespan_cycles": queues,
    }


def integration_decision(
    *,
    points: Sequence[dict[str, Any]],
    blame: dict[str, Any] | None,
) -> dict[str, Any]:
    """Two verdicts, because two different bars were asked about.

    ``passed`` is the experiment's own gate and does not move: 8% at M16 was
    the estimate of what removing the barrier idle was worth, and a run that
    came in under it recorded a quantified over-estimate. Editing the number
    afterwards would replace that record with nothing.

    ``promotion_passed`` is the separate question of whether the gain that was
    measured -- smaller, but real, repeatable, and outside the repeat-median
    dispersion -- is worth integrating. Both are reported, and the
    recommendation is read from the second while the first stays visible beside
    it.
    """
    gating = [point for point in points if point["gating"]]
    passed = bool(gating) and all(point["passed"] for point in gating)
    promotion = bool(gating) and all(
        point["promotion_passed"] for point in gating
    )
    return {
        "passed": passed,
        "experiment_gate_passed": passed,
        "promotion_passed": promotion,
        "experiment_gate": (
            f"M{GATE_POINT} median gain >= {MINIMUM_M16_MEDIAN_GAIN:.0%} and "
            f"M{GUARD_POINT} regression <= {MAXIMUM_M128_REGRESSION:.0%}"
        ),
        "promotion_bar": (
            f"M{GATE_POINT} median gain >= {PROMOTION_M16_MEDIAN_GAIN:.0%} and "
            f"M{GUARD_POINT} regression <= {MAXIMUM_M128_REGRESSION:.0%}"
        ),
        "recommendation": (
            "promote the dependency-local schedule to production"
            if promotion
            else "leave production unchanged; the candidate stays behind the "
            "benchmark guard"
        ),
        "gating_points": [int(point["tokens"]) for point in gating],
        "failing_edge": blame,
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
        "benchmark": "kimi_k3_schedule_probe",
        "dry_run": dry_run,
        "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
        "variants": list(VARIANTS),
        "token_counts": list(TOKEN_COUNTS),
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "repeats": repeats,
        "graph_pool_size": pool_size,
        "tp_size": TP_SIZE,
        "latency_statistic": "per-iteration maximum over the eight ranks",
        "interleaving": (
            "the variant order rotates once per repeat, so temporal drift is "
            "shared rather than charged to one variant"
        ),
        "gate": {
            "gate_point_tokens": GATE_POINT,
            "minimum_median_gain": MINIMUM_M16_MEDIAN_GAIN,
            "guard_point_tokens": GUARD_POINT,
            "maximum_regression": MAXIMUM_M128_REGRESSION,
            "p99": "the gate point's median repeat p99 must improve",
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

    # Both compiled functions of both schedules are proved resident before any
    # capture, so no graph records the occupancy query behind that proof.
    residency = {
        variant: {
            path: (
                modules.extension._kimi_k3_decode_schedule_resident_blocks_per_sm(
                    path == "tensor"
                )
                if variant == "candidate"
                else modules.extension._kimi_k3_decode_resident_blocks_per_sm(
                    path == "tensor"
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

    barriers = {
        variant: _barriers_taken(
            modules, workspace, base_weights, device, variant, GATE_POINT
        )
        for variant in VARIANTS
    }
    profiles = {
        variant: {
            tokens: _profile(
                modules, workspace, base_weights, device, variant, tokens
            )
            for tokens in TOKEN_COUNTS
        }
        for variant in VARIANTS
    }

    points: list[dict[str, Any]] = []
    raw: dict[str, dict[str, list[list[float]]]] = {}
    per_point: list[dict[str, Any]] = []
    for tokens in TOKEN_COUNTS:
        pools = {
            variant: _capture(
                modules,
                workspace,
                base_weights,
                device,
                variant,
                tokens,
                pool_size,
            )
            for variant in VARIANTS
        }
        pool_working_set = {
            variant: len(pool.distinct_experts) * expert_bytes
            for variant, pool in pools.items()
        }
        if any(value <= l2_bytes for value in pool_working_set.values()):
            raise AssertionError((tokens, pool_working_set, l2_bytes))

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
                    raise AssertionError((variant, tokens, sample["error_flag"]))
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
        verdict = evaluate_point(
            tokens=tokens,
            production=summaries["production"],
            candidate=summaries["candidate"],
        )
        points.append(verdict)
        per_point.append(
            {
                **verdict,
                "pool_distinct_experts": {
                    variant: len(pool.distinct_experts)
                    for variant, pool in pools.items()
                },
                "pool_routed_working_set_bytes": pool_working_set,
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
                "profiles": {
                    variant: profiles[variant][tokens] for variant in VARIANTS
                },
            }
        )
        raw[str(tokens)] = {
            variant: [
                {
                    # Enough for an auditor to recompute every reported number
                    # from the samples alone: which repeat produced them, where
                    # in that repeat's interleave the variant ran, and what the
                    # harness said they came to. A bare list of lists could be
                    # re-reduced but not checked against anything.
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
        _audit_raw_samples(raw[str(tokens)], summaries, tokens)
        for pool in pools.values():
            pool.graphs.clear()
        _barrier(device)
        torch.cuda.empty_cache()

    blame = failing_edge(
        points, {tokens: profiles["candidate"][tokens] for tokens in TOKEN_COUNTS}
    )
    decision = integration_decision(points=points, blame=blame)
    result = {
        "passed": decision["passed"],
        "decision": decision,
        "barriers_per_launch": barriers,
        "barrier_reduction": barriers["production"] - barriers["candidate"],
        "schedule": {
            "queues": list(modules.extension._kimi_k3_decode_schedule_queues()),
            "edges": [
                list(edge)
                for edge in modules.extension._kimi_k3_decode_schedule_edges()
            ],
            "queue_units": {
                str(tokens): list(
                    modules.extension._kimi_k3_decode_schedule_queue_units(
                        tokens
                    )
                )
                for tokens in TOKEN_COUNTS
            },
            "counter_bounds": list(
                modules.extension._kimi_k3_decode_schedule_counter_bounds()
            ),
        },
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
        "--output-dir", type=Path, default=Path("kimi_k3_schedule_probe")
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
        for point in result["points"]:
            print(
                f"M{point['tokens']}: "
                f"production {point['production_median_ms']:.4f} ms, "
                f"candidate {point['candidate_median_ms']:.4f} ms, "
                f"{point['improvement_fraction']:+.2%}, "
                f"{'PASS' if point['passed'] else 'FAIL'}"
            )


if __name__ == "__main__":
    main()
