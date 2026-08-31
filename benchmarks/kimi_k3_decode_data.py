"""Prepared weights and deterministic realistic routes for Kimi K3 timing."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from benchmarks.kimi_k3_decode_inputs import (
    GRAPH_POOL_SIZE,
    NUM_EXPERTS,
    SWEEP_TOKEN_COUNTS,
    TOPK,
    route_assignments,
    router_column,
    router_column_plan,
)
from mok.kimi_k3 import (
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    KIMI_K3_SHARED_INTERMEDIATE_SIZE,
    KIMI_K3_TP_SIZE,
    KIMI_K3_W13_PACKED_SHAPE,
    KIMI_K3_W13_SCALE_SHAPE,
    KIMI_K3_W1W3_K,
    KimiK3DecodeWeights,
    kimi_k3_router_reference,
    pack_kimi_k3_mxfp4,
)
from mok.kimi_k3_w13 import fuse_w13_half

HIDDEN = KIMI_K3_HIDDEN_SIZE
LATENT = KIMI_K3_LATENT_SIZE
ROUTED_PER_RANK = KIMI_K3_ROUTED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE
SHARED_PER_RANK = KIMI_K3_SHARED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE


@dataclass(frozen=True, slots=True)
class RoutedInput:
    hidden: torch.Tensor
    weights: KimiK3DecodeWeights
    route_assignments: tuple[tuple[int, ...], ...]
    distinct_experts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SharedRouter:
    """One router weight and bias every graph in a run replays against.

    Reloading a router between two graph captures overwrites the storage both
    graphs recorded the address of, so both replay the last routing that was
    loaded. This holds every pool entry's routing at once instead, in the
    disjoint hidden columns :func:`router_column_plan` assigns, and is never
    written again after it is built.
    """

    weight: torch.Tensor
    correction_bias: torch.Tensor
    column_plan: dict[int, int]
    pool_size: int


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _normal(
    shape: tuple[int, ...],
    device: torch.device,
    seed: int,
    deviation: float,
) -> torch.Tensor:
    values = torch.randn(
        shape,
        generator=_generator(device, seed),
        dtype=torch.float32,
        device=device,
    )
    return (values * deviation).bfloat16().contiguous()


def _pack_expert_matrix(
    device: torch.device,
    seed: int,
    rows: int,
    columns: int,
    padded_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    deviation = 1.0 / math.sqrt(columns)
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    chunk = 112
    for start in range(0, NUM_EXPERTS, chunk):
        dense = _normal(
            (chunk, rows, columns),
            device,
            seed + start,
            deviation,
        )
        packed, scale = pack_kimi_k3_mxfp4(dense, padded_k=padded_k)
        del dense
        packed_chunks.append(packed)
        scale_chunks.append(scale)
    return (
        torch.cat(packed_chunks).contiguous(),
        torch.cat(scale_chunks).contiguous(),
    )


def build_weights(
    device: torch.device,
    tp_rank: int,
) -> KimiK3DecodeWeights:
    """Build one deterministic prepared TP8 shard without test fixtures."""
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 16 * 1024**3:
        raise RuntimeError("prepared Kimi K3 decode weights need 16 GiB free")

    shard = 1_000_000 * (tp_rank + 1)
    # Straight into the fused destination, one projection at a time, so the run
    # never holds a canonical half beside the payload it is being folded into --
    # the same shape `prepare_kimi_k3_decode_weights` prepares in.
    w13_packed = torch.empty(
        KIMI_K3_W13_PACKED_SHAPE, dtype=torch.uint8, device=device
    )
    w13_scale = torch.empty(
        KIMI_K3_W13_SCALE_SHAPE, dtype=torch.uint8, device=device
    )
    for half, seed in enumerate((shard + 11, shard + 22)):
        half_packed, half_scale = _pack_expert_matrix(
            device,
            seed,
            ROUTED_PER_RANK,
            LATENT,
            KIMI_K3_W1W3_K,
        )
        fuse_w13_half(w13_packed, w13_scale, half_packed, half_scale, half)
        del half_packed, half_scale
    w2_packed, w2_scale = _pack_expert_matrix(
        device,
        shard + 33,
        LATENT,
        ROUTED_PER_RANK,
        ROUTED_PER_RANK,
    )
    correction_bias = torch.linspace(
        -0.015,
        0.015,
        NUM_EXPERTS,
        dtype=torch.float32,
        device=device,
    )
    return KimiK3DecodeWeights(
        router_weight=_normal(
            (NUM_EXPERTS, HIDDEN),
            device,
            4_001,
            1.0 / math.sqrt(HIDDEN),
        ),
        router_correction_bias=correction_bias,
        routed_expert_down_proj=_normal(
            (LATENT, HIDDEN),
            device,
            4_002,
            1.0 / math.sqrt(HIDDEN),
        ),
        routed_expert_up_proj=_normal(
            (HIDDEN, LATENT),
            device,
            4_003,
            1.0 / math.sqrt(LATENT),
        ),
        routed_latent_rmsnorm_weight=(
            1.0
            + 0.25 * _normal((LATENT,), device, 4_004, 1.0).float()
        ).bfloat16().contiguous(),
        expert_w13_packed=w13_packed,
        expert_w13_scale=w13_scale,
        expert_w2_packed=w2_packed,
        expert_w2_scale=w2_scale,
        shared_gate_proj=_normal(
            (SHARED_PER_RANK, HIDDEN),
            device,
            shard + 44,
            1.0 / math.sqrt(HIDDEN),
        ),
        shared_up_proj=_normal(
            (SHARED_PER_RANK, HIDDEN),
            device,
            shard + 55,
            1.0 / math.sqrt(HIDDEN),
        ),
        shared_down_proj=_normal(
            (HIDDEN, SHARED_PER_RANK),
            device,
            shard + 66,
            1.0 / math.sqrt(SHARED_PER_RANK),
        ),
        tp_rank=tp_rank,
    )


def build_shared_router(
    base: KimiK3DecodeWeights,
    device: torch.device,
    token_counts: Sequence[int],
    *,
    pool_size: int = GRAPH_POOL_SIZE,
) -> SharedRouter:
    """Build the one router every shape and pool entry in a run routes through.

    Each ``(token count, pool entry, token)`` triple owns a hidden column, and
    that column carries only the sixteen experts the triple is meant to route
    to. A pool entry's hidden state is non-zero in its own columns alone, so
    the columns belonging to other entries multiply by zero and the routing a
    replay produces is decided entirely by the entry's own input.
    """
    plan = router_column_plan(
        token_counts, pool_size=pool_size, hidden_size=HIDDEN
    )
    weight = torch.zeros(
        NUM_EXPERTS,
        HIDDEN,
        dtype=torch.bfloat16,
        device=device,
    )
    for tokens in plan:
        for pool_index in range(pool_size):
            intended = route_assignments(tokens, pool_index)
            for token, experts in enumerate(intended):
                column = router_column(plan, tokens, pool_index, token)
                for slot, expert in enumerate(experts):
                    weight[expert, column] = 0.25 - 0.0078125 * slot
    return SharedRouter(
        weight=weight,
        correction_bias=base.router_correction_bias,
        column_plan=dict(plan),
        pool_size=pool_size,
    )


_SHARED_ROUTERS: dict[tuple[int, str, tuple[int, ...], int], SharedRouter] = {}


def shared_router(
    base: KimiK3DecodeWeights,
    device: torch.device,
    token_counts: Sequence[int] = SWEEP_TOKEN_COUNTS,
    *,
    pool_size: int = GRAPH_POOL_SIZE,
) -> SharedRouter:
    """The one router a process routes a given sweep through.

    Memoized on the shard it extends and the sweep it covers, so every graph a
    run captures records the address of the same tensor and no caller can hand
    a different one to a later capture by accident.
    """
    key = (
        base.router_weight.data_ptr(),
        str(device),
        tuple(sorted(token_counts)),
        pool_size,
    )
    router = _SHARED_ROUTERS.get(key)
    if router is None:
        router = build_shared_router(
            base, device, token_counts, pool_size=pool_size
        )
        _SHARED_ROUTERS[key] = router
    return router


def clear_shared_router_cache() -> None:
    _SHARED_ROUTERS.clear()


def build_routed_input(
    base: KimiK3DecodeWeights,
    device: torch.device,
    tokens: int,
    pool_index: int,
    *,
    router: SharedRouter | None = None,
) -> RoutedInput:
    """Build one pool entry and prove its actual K3 top-k assignment.

    Without an explicit router this uses the process-wide one for the full
    sweep, which is what keeps a pool of captured graphs from collapsing onto
    whichever routing was loaded last.
    """
    if router is None:
        router = shared_router(base, device)
    if not 0 <= pool_index < router.pool_size:
        raise ValueError(f"pool_index must be in [0, {router.pool_size})")
    intended = route_assignments(tokens, pool_index)
    hidden = torch.zeros(
        tokens,
        HIDDEN,
        dtype=torch.bfloat16,
        device=device,
    )
    for token in range(tokens):
        hidden[token, router_column(router.column_plan, tokens, pool_index, token)] = 8.0
    weights = dataclasses.replace(base, router_weight=router.weight)

    actual_ids, _ = kimi_k3_router_reference(
        hidden,
        router.weight,
        router.correction_bias,
    )
    actual = tuple(
        tuple(int(expert) for expert in row)
        for row in actual_ids.cpu().tolist()
    )
    for token, (expected, observed) in enumerate(zip(intended, actual, strict=True)):
        if set(expected) != set(observed):
            raise AssertionError((pool_index, token, expected, observed))
    distinct = tuple(sorted({expert for row in actual for expert in row}))
    expected_count = min(TOPK * tokens, NUM_EXPERTS)
    if len(distinct) != expected_count:
        raise AssertionError((pool_index, len(distinct), expected_count))
    return RoutedInput(hidden, weights, actual, distinct)


__all__ = [
    "RoutedInput",
    "SharedRouter",
    "build_routed_input",
    "build_shared_router",
    "build_weights",
    "clear_shared_router_cache",
    "shared_router",
]
