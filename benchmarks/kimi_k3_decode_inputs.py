"""Pure routing plans and metadata for the Kimi K3 decode benchmark."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

NUM_EXPERTS = 896
TOPK = 16
EXPERT_BLOCKS = NUM_EXPERTS // TOPK
MAX_TOKENS = 128
GRAPH_POOL_SIZE = 4
GRID_CANDIDATES = (64, 96, 128, 148)

# Every token count any Kimi K3 decode sweep in this repository measures: the
# raw-decode shapes plus the two DFlash block sweeps. One router carries the
# routing for all of them at once, so the set has to be known before the first
# one is built rather than grown a shape at a time.
SWEEP_TOKEN_COUNTS = (
    *range(1, 9),
    *range(16, 65, 8),
    80,
    96,
    112,
    128,
)
GATE_UP_UNITS_PER_EXPERT = 3
DOWN_UNITS_PER_EXPERT = 28


def _permutation_step(size: int, pool_index: int) -> int:
    for offset in range(EXPERT_BLOCKS):
        candidate = (2 * pool_index + 1 + 2 * offset) % EXPERT_BLOCKS
        if candidate and math.gcd(candidate, size) == 1:
            return candidate
    raise AssertionError(f"no permutation step for {size}")


def route_assignments(
    tokens: int,
    pool_index: int,
) -> tuple[tuple[int, ...], ...]:
    """Assign each token one deterministic 16-expert block.

    A replay occupies a fresh block per token until all 56 blocks are in use.
    Pool entries rotate the occupied window and permute token-to-block order.
    Expert order within each block rotates as well; the GPU reference records
    the router's actual top-k order because K3 intentionally requests an
    unsorted top-k.
    """
    if not 1 <= tokens <= MAX_TOKENS:
        raise ValueError(f"tokens must be in [1, {MAX_TOKENS}]")
    if not 0 <= pool_index < GRAPH_POOL_SIZE:
        raise ValueError(
            f"pool_index must be in [0, {GRAPH_POOL_SIZE})"
        )
    occupied_blocks = min(tokens, EXPERT_BLOCKS)
    start = pool_index * occupied_blocks
    step = _permutation_step(occupied_blocks, pool_index)
    assignments: list[tuple[int, ...]] = []
    for token in range(tokens):
        position = (step * (token % occupied_blocks)) % occupied_blocks
        block = (start + position) % EXPERT_BLOCKS
        first = block * TOPK
        rotation = pool_index % TOPK
        assignments.append(
            tuple(first + (slot + rotation) % TOPK for slot in range(TOPK))
        )
    return tuple(assignments)


def router_column_plan(
    token_counts: Iterable[int],
    *,
    pool_size: int = GRAPH_POOL_SIZE,
    hidden_size: int,
) -> dict[int, int]:
    """Give every graph a run captures its own block of hidden coordinates.

    A CUDA graph records the address of the router weight rather than its
    contents, so a pool of graphs that were captured around a router being
    reloaded between them all replay whichever routing was loaded last. The
    only way a pool of graphs can carry a pool of routes is for one immutable
    router to hold all of them at once.

    It can, because a pool entry's hidden state is one-hot. The logits for a
    token are the router column its single non-zero selects, scaled; every
    other column multiplies by zero and contributes nothing. So each ``(token
    count, pool entry, token)`` triple can own a private column, the shared
    router can carry that triple's sixteen intended experts in it, and no
    triple can see any other's. This returns the first column each token count
    owns; :func:`router_column` indexes within it.
    """
    if pool_size < 1:
        raise ValueError("a graph pool needs at least one entry")
    counts = list(token_counts)
    if len(set(counts)) != len(counts):
        raise ValueError(
            f"each token count may appear once in a column plan, got {counts}"
        )
    plan: dict[int, int] = {}
    base = 0
    for tokens in sorted(counts):
        if not 1 <= tokens <= MAX_TOKENS:
            raise ValueError(f"tokens must be in [1, {MAX_TOKENS}]")
        plan[tokens] = base
        base += pool_size * tokens
    if base > hidden_size:
        raise ValueError(
            f"the sweep {sorted(counts)} needs {base} hidden columns for "
            f"{pool_size} pool entries, and the layer has {hidden_size}"
        )
    return plan


def router_column(
    plan: Mapping[int, int],
    tokens: int,
    pool_index: int,
    token: int,
) -> int:
    """The hidden coordinate one pool entry's token routes through."""
    if tokens not in plan:
        raise ValueError(f"{tokens} tokens is not in the column plan")
    if not 0 <= token < tokens:
        raise ValueError(f"token must be in [0, {tokens})")
    if pool_index < 0:
        raise ValueError("pool_index must not be negative")
    return plan[tokens] + pool_index * tokens + token


def verify_graph_routes(
    intended: Sequence[Sequence[Sequence[int]]],
    observed: Sequence[Sequence[Sequence[int]]],
) -> dict[str, Any]:
    """Check that graph ``p`` routed entry ``p``, and say so if it did not.

    ``observed`` is what each captured graph actually produced when it was
    replayed, read back from the native router's own output buffer. The
    signature of a pool captured around a mutated router is that every graph
    reports the last entry's routing, so that case is named in the failure
    rather than left to be inferred from a list of mismatches.
    """
    if len(intended) != len(observed):
        raise AssertionError(
            f"{len(observed)} replayed graphs for {len(intended)} pool entries"
        )
    if not intended:
        raise AssertionError("a graph pool with no entries verifies nothing")

    normalized_intended = [
        tuple(tuple(sorted(token)) for token in entry) for entry in intended
    ]
    normalized_observed = [
        tuple(tuple(sorted(token)) for token in entry) for entry in observed
    ]
    mismatched = [
        index
        for index, (want, got) in enumerate(
            zip(normalized_intended, normalized_observed, strict=True)
        )
        if want != got
    ]
    if mismatched:
        distinct_intended = len(set(normalized_intended))
        collapsed = (
            distinct_intended > 1 and len(set(normalized_observed)) == 1
        )
        detail = ", ".join(f"pool_index={index}" for index in mismatched)
        raise AssertionError(
            (
                "every replayed graph produced one routing, so the pool "
                "collapsed onto a single router: "
                if collapsed
                else "replayed graphs did not produce their own routing: "
            )
            + detail
        )

    return {
        "graph_count": len(normalized_observed),
        "distinct_route_sets": len(set(normalized_observed)),
        "distinct_experts_per_graph": [
            len({expert for token in entry for expert in token})
            for entry in normalized_observed
        ],
    }


def route_metadata(
    *,
    tokens: int,
    expert_weight_bytes: int,
    l2_cache_bytes: int,
) -> dict[str, Any]:
    """Describe replay-local and pool-wide routed-expert traffic."""
    assignments = [
        route_assignments(tokens, pool_index)
        for pool_index in range(GRAPH_POOL_SIZE)
    ]
    replay_counts = [
        len({expert for token in entry for expert in token})
        for entry in assignments
    ]
    expected = min(TOPK * tokens, NUM_EXPERTS)
    if any(count != expected for count in replay_counts):
        raise AssertionError((tokens, replay_counts, expected))
    pool_experts = {
        expert
        for entry in assignments
        for token in entry
        for expert in token
    }
    per_replay_bytes = expected * expert_weight_bytes
    pool_bytes = len(pool_experts) * expert_weight_bytes
    gate_up = expected * GATE_UP_UNITS_PER_EXPERT
    down = expected * DOWN_UNITS_PER_EXPERT
    return {
        "distinct_experts_per_replay": expected,
        "route_assignments_by_pool_entry": [
            [list(token) for token in entry] for entry in assignments
        ],
        "routed_queue_units_per_replay": {
            "gate_up": gate_up,
            "down": down,
            "total": gate_up + down,
        },
        "pool_wide_distinct_experts": len(pool_experts),
        "routed_expert_working_set_bytes_per_replay": per_replay_bytes,
        "routed_expert_working_set_exceeds_l2_per_replay": (
            per_replay_bytes > l2_cache_bytes
        ),
        "pool_wide_routed_expert_working_set_bytes": pool_bytes,
        "pool_wide_routed_expert_working_set_exceeds_l2": (
            pool_bytes > l2_cache_bytes
        ),
    }


def mode_batch_size(mode: str, tokens: int) -> int:
    if mode == "raw_decode":
        return tokens
    if mode == "block8":
        return tokens // 8
    if mode == "block16":
        return tokens // 16
    raise ValueError(f"unknown benchmark mode {mode!r}")


__all__ = [
    "SWEEP_TOKEN_COUNTS",
    "DOWN_UNITS_PER_EXPERT",
    "EXPERT_BLOCKS",
    "GATE_UP_UNITS_PER_EXPERT",
    "GRAPH_POOL_SIZE",
    "GRID_CANDIDATES",
    "MAX_TOKENS",
    "NUM_EXPERTS",
    "TOPK",
    "mode_batch_size",
    "route_assignments",
    "route_metadata",
    "router_column",
    "router_column_plan",
    "verify_graph_routes",
]
