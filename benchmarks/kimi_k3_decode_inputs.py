"""Pure routing plans and metadata for the Kimi K3 decode benchmark."""

from __future__ import annotations

import math
from typing import Any

NUM_EXPERTS = 896
TOPK = 16
EXPERT_BLOCKS = NUM_EXPERTS // TOPK
MAX_TOKENS = 128
GRAPH_POOL_SIZE = 4
GRID_CANDIDATES = (64, 96, 128, 148)
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
]
