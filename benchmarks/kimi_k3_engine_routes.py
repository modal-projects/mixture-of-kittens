"""The arms the fused-W13 engine A/B measures, and the routes it measures them on.

What separates the two rings the adaptive engine selects between is the number
of rows an occupied expert gets, so a route here is chosen for its row count
and then checked against the routing the router actually produced. A shape that
quietly stopped being an r = 8 shape would report the packed path's latency
under the fallback's name, which is the one way this measurement could lie
without failing.

The sweep router gives each token its own sixteen-expert block, which is the
realistic decode case; the concentrated router hands one block to every token,
which is the adversarial one. Both build the same ``Routes``, so the harness in
``kimi_k3_engine_probe.py`` cannot tell them apart after this point.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch


# The engine ids `expert_mxfp4::fused_w13` compiles, spelled here so the harness
# selects by the same number the kernel dispatches on.
#
# Two arms, because two are what is compiled: production's selector, and the
# resident two-stage ring it replaced. The two rings the selector is built from
# were separately selectable while they were being measured against each other,
# and the integration retired both ids -- production exercises both rings, so
# the question the probe answers now is what the selector is worth against the
# ring that shipped before it. Both pools are replayed interleaved on one
# workspace, so nothing that drifts between runs is confounded with the arm.
#
# `production` is the id a process that has set nothing launches, so it is
# selected explicitly here only so the harness's arms are symmetric.
ENGINE_IDS = {
    "resident": 3,
    "production": 2,
}
VARIANTS = tuple(ENGINE_IDS)
BASELINE = "resident"
CANDIDATES = tuple(name for name in VARIANTS if name != BASELINE)


@dataclass(frozen=True, slots=True)
class Shape:
    """One route this probe measures, and what that route is here to settle.

    ``rows_per_expert`` is asserted against the routing the router actually
    produced rather than assumed, because it is the number the adaptive engine
    branches on: a shape that quietly stopped being an r = 8 shape would report
    the packed path's latency under the fallback's name.
    """

    label: str
    tokens: int
    route: str
    rows_per_expert: int
    purpose: str


# The realistic decode routes, and the one that refuses the packing.
#
# The sweep router gives each token its own sixteen-expert block, so M16 and M32
# put one row on every occupied expert and M128 -- 128 tokens over 56 blocks --
# puts two or three. Those are the shapes decode runs in. `m128_r8` puts eight
# tokens on each block instead, so every occupied expert holds a full pass and
# the compact engine takes production's ring for all of them.
SHAPES = (
    Shape("m16", 16, "sweep", 1, "the shape decode runs in"),
    Shape("m32", 32, "sweep", 1, "the same route one occupancy wider"),
    Shape("m128", 128, "sweep", 3, "the throughput guard"),
    Shape(
        "m128_r8",
        128,
        "concentrated",
        8,
        "the adversarial route the packing must refuse",
    ),
)
SHAPES_BY_LABEL = {shape.label: shape for shape in SHAPES}

# What a candidate was asked to clear. The M16 gain is the point of the change:
# block-16 concurrency-one is the shape decode runs in, and it is where the
# ring's 40% was measured. The two guards are M128 and the r = 8 route, and the
# second matters as much as the first for `compact`: it runs production's ring
# there, so a regression at it would be the price of asking for the compact
# engine's shared bytes rather than of anything the packing does.
MINIMUM_GATE_MEDIAN_GAIN = 0.02
MAXIMUM_GUARD_REGRESSION = 0.01
GATE_LABEL = "m16"
GUARD_LABELS = ("m128", "m128_r8")

# The gate/up subphase counters, which are what this candidate exists to move.
RING_SUBPHASES = (
    "routed_gate_up_tma_issue",
    "routed_gate_up_tma_wait",
    "routed_gate_up_ring_full",
)
BAND_SUBPHASES = RING_SUBPHASES + (
    "routed_gate_up_mma_issue",
    "routed_gate_up_activation",
    "routed_gate_up_epilogue",
)


@dataclass(slots=True)
class Modules:
    extension: ModuleType
    kimi: ModuleType
    data: ModuleType
    runtime: ModuleType


@dataclass(slots=True)
class Pool:
    """One captured graph pool: one engine at one shape."""

    variant: str
    shape: Shape
    graphs: list[torch.cuda.CUDAGraph]
    routes: Routes


@dataclass(frozen=True, slots=True)
class Routes:
    """Every pool entry of one shape, built once and shared by all arms.

    Built once rather than per arm because a route is an input, not a treatment.
    Two arms handed separately-built routes would be two experiments; handed the
    same tensors they are one, and a captured graph records the address of these
    rather than of a copy that may have been routed differently.
    """

    shape: Shape
    hidden: list[torch.Tensor]
    weights: list[Any]
    distinct_experts: tuple[int, ...]
    maximum_rows_per_expert: int
    row_histogram: dict[int, int]


def concentrated_assignments(
    *,
    tokens: int,
    pool_index: int,
    rows: int,
    topk: int,
    num_experts: int,
) -> tuple[tuple[int, ...], ...]:
    """Route ``rows`` tokens onto each occupied expert, and no more.

    The sweep router cannot express this: it gives each token a block of its
    own, so an expert's row count is decided by how many tokens outnumber the
    56 blocks. Here the tokens are cut into groups of ``rows``, each group takes
    one contiguous run of ``topk`` experts, and every expert in that run is
    routed to by exactly the group's tokens -- so the row count is the group
    size and nothing else.

    Pool entries slide the occupied window by a whole entry's worth of experts,
    which keeps the four pools reading different weights while leaving each
    entry's own experts distinct: one entry occupies ``tokens / rows * topk`` of
    them, which is fewer than the layer has, so the wrap cannot fold two groups
    onto one expert.
    """
    if rows < 1 or tokens % rows:
        raise ValueError(f"{tokens} tokens do not divide into groups of {rows}")
    groups = tokens // rows
    occupied = groups * topk
    if occupied > num_experts:
        raise ValueError(
            f"{tokens} tokens at {rows} rows need {occupied} experts, and the "
            f"layer has {num_experts}"
        )
    start = (pool_index * occupied) % num_experts
    return tuple(
        tuple(
            (start + (token // rows) * topk + slot) % num_experts
            for slot in range(topk)
        )
        for token in range(tokens)
    )


def _row_histogram(
    assignments: Sequence[Sequence[int]],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in assignments:
        for expert in row:
            counts[int(expert)] = counts.get(int(expert), 0) + 1
    histogram: dict[int, int] = {}
    for count in counts.values():
        histogram[count] = histogram.get(count, 0) + 1
    return dict(sorted(histogram.items()))


def _concentrated_router(
    modules: Modules,
    base: Any,
    device: torch.device,
    *,
    tokens: int,
    rows: int,
    pool_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[tuple[int, ...], ...]]]:
    """One immutable router carrying every concentrated pool entry at once.

    The same one-hot trick the sweep router uses, for the same reason: a graph
    records the router weight's address, so a pool captured around a router
    being rewritten between captures replays whichever routing was written last.
    Each ``(pool entry, token)`` pair owns a hidden column, and that column
    carries only the experts that pair is meant to reach.
    """
    weight = torch.zeros(
        modules.data.NUM_EXPERTS,
        modules.data.HIDDEN,
        dtype=torch.bfloat16,
        device=device,
    )
    hidden_states: list[torch.Tensor] = []
    intended: list[tuple[tuple[int, ...], ...]] = []
    for pool_index in range(pool_size):
        assignments = concentrated_assignments(
            tokens=tokens,
            pool_index=pool_index,
            rows=rows,
            topk=modules.data.TOPK,
            num_experts=modules.data.NUM_EXPERTS,
        )
        hidden = torch.zeros(
            tokens, modules.data.HIDDEN, dtype=torch.bfloat16, device=device
        )
        for token, experts in enumerate(assignments):
            column = pool_index * tokens + token
            for slot, expert in enumerate(experts):
                weight[expert, column] = 0.25 - 0.0078125 * slot
            hidden[token, column] = 8.0
        hidden_states.append(hidden)
        intended.append(assignments)
    return weight, hidden_states, intended


def _build_routes(
    modules: Modules,
    base_weights: Any,
    device: torch.device,
    shape: Shape,
    pool_size: int,
) -> Routes:
    """Build one shape's pool entries, and prove they route where they claim.

    The proof is the router's own top-k, read back from the reference, not the
    assignment table that was written into the weight. Those differ whenever a
    column was overwritten or a tie broke the other way, and the row count is
    what the adaptive engine branches on -- so a shape whose rows drifted would
    silently measure the wrong specialization.
    """
    hidden_states: list[torch.Tensor] = []
    weights: list[Any] = []
    observed: list[tuple[tuple[int, ...], ...]] = []
    if shape.route == "sweep":
        for pool_index in range(pool_size):
            routed = modules.data.build_routed_input(
                base_weights, device, shape.tokens, pool_index
            )
            hidden_states.append(routed.hidden)
            weights.append(routed.weights)
            observed.append(routed.route_assignments)
    elif shape.route == "concentrated":
        weight, hidden_states, intended = _concentrated_router(
            modules,
            base_weights,
            device,
            tokens=shape.tokens,
            rows=shape.rows_per_expert,
            pool_size=pool_size,
        )
        routed_weights = dataclasses.replace(
            base_weights, router_weight=weight
        )
        for pool_index, hidden in enumerate(hidden_states):
            actual_ids, _ = modules.data.kimi_k3_router_reference(
                hidden, weight, base_weights.router_correction_bias
            )
            actual = tuple(
                tuple(int(expert) for expert in row)
                for row in actual_ids.cpu().tolist()
            )
            for token, (expected, got) in enumerate(
                zip(intended[pool_index], actual, strict=True)
            ):
                if set(expected) != set(got):
                    raise AssertionError((shape.label, pool_index, token))
            weights.append(routed_weights)
            observed.append(actual)
    else:
        raise ValueError(f"unknown route {shape.route!r}")

    distinct = sorted(
        {expert for entry in observed for row in entry for expert in row}
    )
    histograms = [_row_histogram(entry) for entry in observed]
    maximum = max(count for histogram in histograms for count in histogram)
    if maximum != shape.rows_per_expert:
        raise AssertionError((shape.label, maximum, shape.rows_per_expert))
    merged: dict[int, int] = {}
    for histogram in histograms:
        for rows, experts in histogram.items():
            merged[rows] = merged.get(rows, 0) + experts
    return Routes(
        shape=shape,
        hidden=hidden_states,
        weights=weights,
        distinct_experts=tuple(distinct),
        maximum_rows_per_expert=maximum,
        row_histogram=dict(sorted(merged.items())),
    )
