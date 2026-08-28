"""Prepared weights and deterministic realistic routes for Kimi K3 timing."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from benchmarks.kimi_k3_decode_inputs import (
    GRAPH_POOL_SIZE,
    MAX_TOKENS,
    NUM_EXPERTS,
    TOPK,
    route_assignments,
)
from mok.kimi_k3 import (
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    KIMI_K3_SHARED_INTERMEDIATE_SIZE,
    KIMI_K3_TP_SIZE,
    KIMI_K3_W1W3_K,
    KimiK3DecodeWeights,
    kimi_k3_router_reference,
    pack_kimi_k3_mxfp4,
)

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
    w1_packed, w1_scale = _pack_expert_matrix(
        device,
        shard + 11,
        ROUTED_PER_RANK,
        LATENT,
        KIMI_K3_W1W3_K,
    )
    w3_packed, w3_scale = _pack_expert_matrix(
        device,
        shard + 22,
        ROUTED_PER_RANK,
        LATENT,
        KIMI_K3_W1W3_K,
    )
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
        expert_w1_packed=w1_packed,
        expert_w1_scale=w1_scale,
        expert_w3_packed=w3_packed,
        expert_w3_scale=w3_scale,
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


def build_routed_input(
    base: KimiK3DecodeWeights,
    device: torch.device,
    tokens: int,
    pool_index: int,
) -> RoutedInput:
    """Build one pool entry and prove its actual K3 top-k assignment."""
    if not 0 <= pool_index < GRAPH_POOL_SIZE:
        raise ValueError(f"pool_index must be in [0, {GRAPH_POOL_SIZE})")
    intended = route_assignments(tokens, pool_index)
    hidden = torch.zeros(
        tokens,
        HIDDEN,
        dtype=torch.bfloat16,
        device=device,
    )
    router_weight = torch.zeros(
        NUM_EXPERTS,
        HIDDEN,
        dtype=torch.bfloat16,
        device=device,
    )
    for token, experts in enumerate(intended):
        column = pool_index * MAX_TOKENS + token
        hidden[token, column] = 8.0
        for slot, expert in enumerate(experts):
            router_weight[expert, column] = 0.25 - 0.0078125 * slot
    weights = dataclasses.replace(base, router_weight=router_weight)

    actual_ids, _ = kimi_k3_router_reference(
        hidden,
        router_weight,
        base.router_correction_bias,
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
    "build_routed_input",
    "build_weights",
]
