"""The scratch layout, the fixtures, and the oracle the router tests share.

Both router test files rest on the same construction: one module-scoped scratch
buffer whose regions are computed the way the header computes them, a seeded
router input generator whose top-k selection is provably unambiguous, and a
float64 reference for the selection and the latent projection. They live here
so ``test_kimi_k3_router.py`` and ``test_kimi_k3_router_contract.py`` agree on
what they are comparing against.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


KIMI_K3_HIDDEN_SIZE = 7168
KIMI_K3_LATENT_SIZE = 3584
KIMI_K3_NUM_EXPERTS = 896
KIMI_K3_TOPK = 16
KIMI_K3_MAX_TOKENS = 128
KIMI_K3_MAX_ROUTES = KIMI_K3_MAX_TOKENS * KIMI_K3_TOPK
KIMI_K3_CAPACITY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
SCRATCH_ALIGNMENT = 256
NUM_PHASE_COUNTERS = 128
NUM_SCHEDULE_COUNTERS = 128

_ROUTE_AND_PROJECT_ARGUMENTS = (
    "hidden_states",
    "router_weight",
    "router_correction_bias",
    "routed_expert_down_proj",
    "scratch",
    "active_tokens",
)


# The stage reads these tensors with 16-byte vector loads and TMA descriptors and
# indexes scratch through 256-byte aligned regions, so each first element must sit
# on the matching boundary. Each case pairs the reported field with the private
# helper's argument name, the element offset that breaks the boundary, and the
# boundary itself. ``router_correction_bias`` is only ever read as a scalar float,
# so it is deliberately absent.
_ROUTE_AND_PROJECT_ALIGNMENT = (
    ("hidden_states", "hidden_states", 1, 16),
    ("router_weight", "router_weight", 1, 16),
    ("routed_expert_down_proj", "latent_down_proj", 1, 16),
    ("scratch", "scratch", 16, 256),
)


def _aligned(size_bytes: int) -> int:
    return (
        (size_bytes + SCRATCH_ALIGNMENT - 1) // SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT
    )


def _scratch_layout() -> dict[str, tuple[int, int]]:
    """Model the C++ scratch layout independently, as int32 offsets and counts.

    Regions are laid out in declaration order, each starting on a 256-byte
    boundary, exactly as ``csrc/kimi_k3_decode/types.cuh`` specifies.
    """
    regions = (
        ("phase", NUM_PHASE_COUNTERS),
        ("expert_ids", KIMI_K3_MAX_ROUTES),
        ("expert_weights", KIMI_K3_MAX_ROUTES),
        ("counts", KIMI_K3_NUM_EXPERTS),
        ("offsets", KIMI_K3_NUM_EXPERTS + 1),
        ("assignment_tokens", KIMI_K3_MAX_ROUTES),
        ("assignment_slots", KIMI_K3_MAX_ROUTES),
        ("latent_mxfp8", KIMI_K3_MAX_TOKENS * KIMI_K3_LATENT_SIZE // 4),
        ("latent_scale", KIMI_K3_MAX_TOKENS * (KIMI_K3_LATENT_SIZE // 32) // 4),
        ("situ_mxfp8", KIMI_K3_MAX_ROUTES * 384 // 4),
        ("situ_scale", KIMI_K3_MAX_ROUTES * (384 // 32) // 4),
        (
            "routed_accumulator",
            KIMI_K3_MAX_TOKENS * KIMI_K3_LATENT_SIZE * 2,
        ),
        ("shared_gate", KIMI_K3_MAX_TOKENS * 768 // 2),
        ("shared_up", KIMI_K3_MAX_TOKENS * 768 // 2),
        ("shared_activated", KIMI_K3_MAX_TOKENS * 768 // 2),
        ("tail_normalized", KIMI_K3_MAX_TOKENS * KIMI_K3_LATENT_SIZE // 2),
        ("tail_shared_shard", KIMI_K3_MAX_TOKENS * 896 // 2),
        ("latent_x", KIMI_K3_MAX_TOKENS * KIMI_K3_LATENT_SIZE // 2),
        ("unit_expert", KIMI_K3_NUM_EXPERTS),
        ("router_scores", KIMI_K3_MAX_TOKENS * KIMI_K3_NUM_EXPERTS),
        ("schedule", NUM_SCHEDULE_COUNTERS),
    )
    layout: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, count in regions:
        layout[name] = (cursor // 4, count)
        cursor += _aligned(count * 4)
    layout["total_bytes"] = (cursor, 0)
    return layout


SCRATCH_LAYOUT = _scratch_layout()
SCRATCH_BYTES = SCRATCH_LAYOUT["total_bytes"][0]


def _region(scratch: torch.Tensor, name: str) -> torch.Tensor:
    offset, count = SCRATCH_LAYOUT[name]
    return scratch.view(torch.int32)[offset:offset + count]


def _capacity(active_tokens: int) -> int:
    for bucket in KIMI_K3_CAPACITY_BUCKETS:
        if active_tokens <= bucket:
            return bucket
    raise AssertionError("active_tokens exceeds the Kimi K3 decode contract")


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Kimi K3 routing requires CUDA")
    selected = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(selected)
    if torch.cuda.get_device_capability(selected) != (10, 3):
        pytest.skip("Kimi K3 routing requires an SM103 GPU")
    return selected


@pytest.fixture(scope="module")
def peer_device(device: torch.device) -> Iterator[torch.device]:
    """A second CUDA device, with the first one left current for the caller."""
    if torch.cuda.device_count() < 2:
        pytest.skip("cross-device Kimi K3 routing needs two CUDA devices")
    peer = torch.device("cuda", 1 if device.index == 0 else 0)
    if torch.cuda.get_device_capability(peer) != (10, 3):
        pytest.skip("Kimi K3 routing requires an SM103 GPU")
    try:
        yield peer
    finally:
        torch.cuda.set_device(device)


@pytest.fixture(scope="module")
def scratch(device: torch.device) -> torch.Tensor:
    """One reused workspace, so the generation-tagged counters are exercised."""
    from mok import _C

    return torch.zeros(
        _C.kimi_k3_decode_workspace_bytes(), dtype=torch.uint8, device=device
    )


@pytest.fixture(scope="module")
def latent_down_proj(device: torch.device) -> torch.Tensor:
    """The replicated BF16 ``7168 -> 3584`` routed latent-down projection."""
    generator = torch.Generator(device=device).manual_seed(20260527)
    scale = 8.0 / math.sqrt(KIMI_K3_HIDDEN_SIZE)
    weight = torch.randn(
        (KIMI_K3_LATENT_SIZE, KIMI_K3_HIDDEN_SIZE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return (weight * scale).bfloat16().contiguous()


def _route_and_project(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_correction_bias: torch.Tensor,
    latent_down_proj: torch.Tensor,
    scratch: torch.Tensor,
    active_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from mok import ops

    return ops._kimi_k3_route_and_project(
        hidden_states,
        router_weight,
        router_correction_bias,
        latent_down_proj,
        scratch,
        active_tokens,
    )


def _router_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mok.kimi_k3 import kimi_k3_router_reference

    return kimi_k3_router_reference(
        hidden_states, router_weight, router_correction_bias
    )


def _sorted_routes(
    expert_ids: torch.Tensor, expert_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Order each token's routes by expert id so unsorted top-k is comparable."""
    order = torch.argsort(expert_ids.int(), dim=-1)
    return (
        torch.gather(expert_ids.int(), -1, order),
        torch.gather(expert_weights.float(), -1, order),
    )


def _assert_selection_is_unambiguous(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_correction_bias: torch.Tensor,
) -> None:
    """Guard the fixture: FP64 must agree with the reference by a wide margin.

    Exact-id assertions are only meaningful when the 16th and 17th corrected
    scores are far enough apart that no FP32 summation order can reorder them.
    """
    corrected = torch.sigmoid(
        hidden_states.double() @ router_weight.double().T
    ) + router_correction_bias.double()
    boundary = torch.topk(corrected, KIMI_K3_TOPK + 1, dim=-1, sorted=True).values
    margin = (boundary[:, KIMI_K3_TOPK - 1] - boundary[:, KIMI_K3_TOPK]).min()
    assert margin > 1e-3, f"fixture selection margin is only {float(margin)}"

    exact_ids = torch.topk(corrected, KIMI_K3_TOPK, dim=-1, sorted=False).indices
    reference_ids, _ = _router_reference(
        hidden_states, router_weight, router_correction_bias
    )
    assert torch.equal(
        torch.sort(exact_ids.int(), dim=-1).values,
        torch.sort(reference_ids.int(), dim=-1).values,
    )


def _seeded_router_inputs(
    device: torch.device, tokens: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense seeded BF16 inputs whose top-16 boundary is separated by construction.

    Every logit remains a full 7168-term FP32 dot product, so the router's
    accumulation is exercised at full width. A unit diagonal in ``router_weight``
    paired with a per-token ``+-3.0`` pattern across the first 896 hidden columns
    then lifts each token's own 16 experts far above the other 880, which is what
    makes the selected set independent of FP32 summation order and therefore
    comparable exactly against the reference.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(
        (tokens, KIMI_K3_HIDDEN_SIZE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    router_weight = torch.randn(
        (KIMI_K3_NUM_EXPERTS, KIMI_K3_HIDDEN_SIZE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * (0.25 / math.sqrt(KIMI_K3_HIDDEN_SIZE))
    diagonal = torch.arange(KIMI_K3_NUM_EXPERTS, device=device)
    router_weight[diagonal, diagonal] = 1.0
    chosen = torch.rand(
        (tokens, KIMI_K3_NUM_EXPERTS), generator=generator, device=device
    ).topk(KIMI_K3_TOPK, dim=-1).indices
    selected = torch.zeros(
        (tokens, KIMI_K3_NUM_EXPERTS), device=device, dtype=torch.float32
    )
    selected.scatter_(1, chosen, 1.0)
    hidden_states[:, :KIMI_K3_NUM_EXPERTS] = selected * 6.0 - 3.0
    router_correction_bias = (
        torch.randn(
            (KIMI_K3_NUM_EXPERTS,),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    ).contiguous()
    return (
        hidden_states.bfloat16().contiguous(),
        router_weight.bfloat16().contiguous(),
        router_correction_bias,
    )


def _one_hot_router_inputs(
    device: torch.device, expert_groups: dict[int, tuple[int, ...]], tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inputs whose logits are exactly 8.0 for chosen experts and 0 elsewhere.

    ``hidden_states`` is one-hot, so every logit is a single exact BF16 product
    and no summation order can perturb it.
    """
    hidden_states = torch.zeros(
        (tokens, KIMI_K3_HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    router_weight = torch.zeros(
        (KIMI_K3_NUM_EXPERTS, KIMI_K3_HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    for token, experts in expert_groups.items():
        hidden_states[token, token] = 1.0
        for expert in experts:
            router_weight[expert, token] = 8.0
    router_correction_bias = torch.zeros(
        (KIMI_K3_NUM_EXPERTS,), device=device, dtype=torch.float32
    )
    return (
        hidden_states.contiguous(),
        router_weight.contiguous(),
        router_correction_bias,
    )
