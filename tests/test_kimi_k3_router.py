"""GPU tests for the fused Kimi K3 router and routed latent-down projection."""

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
NUM_PHASE_COUNTERS = 32

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
        ("routed_accumulator", KIMI_K3_MAX_TOKENS * KIMI_K3_LATENT_SIZE),
        ("shared_gate", KIMI_K3_MAX_TOKENS * 768 // 2),
        ("shared_up", KIMI_K3_MAX_TOKENS * 768 // 2),
        ("shared_activated", KIMI_K3_MAX_TOKENS * 768 // 2),
        ("tail_normalized", KIMI_K3_MAX_TOKENS * KIMI_K3_LATENT_SIZE // 2),
        ("tail_shared_shard", KIMI_K3_MAX_TOKENS * 896 // 2),
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


def test_workspace_bytes_matches_the_documented_scratch_layout(
    device: torch.device,
) -> None:
    from mok import _C

    assert SCRATCH_BYTES == 4_896_256
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    assert SCRATCH_LAYOUT["phase"][0] == 0
    assert SCRATCH_LAYOUT["expert_ids"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["expert_weights"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["counts"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["offsets"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["assignment_tokens"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["assignment_slots"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["latent_mxfp8"][0] * 4 == 40_448
    assert SCRATCH_LAYOUT["latent_scale"][0] * 4 == 499_200
    assert SCRATCH_LAYOUT["situ_mxfp8"][0] * 4 == 513_536
    assert SCRATCH_LAYOUT["situ_scale"][0] * 4 == 1_299_968
    assert SCRATCH_LAYOUT["routed_accumulator"][0] * 4 == 1_324_544
    assert SCRATCH_LAYOUT["shared_gate"][0] * 4 == 3_159_552
    assert SCRATCH_LAYOUT["shared_up"][0] * 4 == 3_356_160
    assert SCRATCH_LAYOUT["shared_activated"][0] * 4 == 3_552_768
    assert SCRATCH_LAYOUT["tail_normalized"][0] * 4 == 3_749_376
    assert SCRATCH_LAYOUT["tail_shared_shard"][0] * 4 == 4_666_880


@pytest.mark.parametrize("tokens", [1, 8, 16, 128])
def test_router_selects_reference_experts_exactly(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    tokens: int,
) -> None:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=1000 + tokens
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    reference_ids, reference_weights = _router_reference(
        hidden_states, router_weight, bias
    )

    expert_ids, expert_weights, _ = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
    )

    assert expert_ids.shape == (tokens, KIMI_K3_TOPK)
    assert expert_ids.dtype == torch.int32
    assert expert_weights.shape == (tokens, KIMI_K3_TOPK)
    assert expert_weights.dtype == torch.float32

    actual_ids, actual_weights = _sorted_routes(expert_ids, expert_weights)
    expected_ids, expected_weights = _sorted_routes(
        reference_ids, reference_weights
    )
    assert torch.equal(actual_ids, expected_ids)
    assert (actual_weights - expected_weights).abs().max() <= 1e-5


@pytest.mark.parametrize("tokens", [1, 8, 16, 128])
def test_router_weights_sum_to_one_per_token(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    tokens: int,
) -> None:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=2000 + tokens
    )

    _, expert_weights, _ = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
    )

    sums = expert_weights.double().sum(dim=-1)
    assert (sums - 1.0).abs().max() <= 1e-5
    assert bool((expert_weights > 0).all())


def test_router_orders_routes_by_descending_corrected_score(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    tokens = 16
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=3001
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    corrected = torch.sigmoid(
        hidden_states.double() @ router_weight.double().T
    ) + bias.double()

    expert_ids, _, _ = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
    )

    chosen = torch.gather(corrected, -1, expert_ids.long())
    assert bool((chosen[:, :-1] >= chosen[:, 1:]).all())


def test_router_handles_concentrated_routing(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    tokens = 16
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=4001
    )
    concentrated = tuple(range(3, 3 + 16 * 7, 7))
    bias = bias.clone()
    bias[list(concentrated)] = 10.0
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)

    expert_ids, expert_weights, _ = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
    )

    expected = torch.tensor(concentrated, device=device, dtype=torch.int32)
    assert torch.equal(
        torch.sort(expert_ids, dim=-1).values,
        expected.expand(tokens, KIMI_K3_TOPK),
    )

    counts = _region(scratch, "counts")
    offsets = _region(scratch, "offsets")
    expected_counts = torch.zeros(
        KIMI_K3_NUM_EXPERTS, device=device, dtype=torch.int32
    )
    expected_counts[list(concentrated)] = tokens
    assert torch.equal(counts, expected_counts)
    assert torch.equal(
        offsets,
        torch.cat(
            (
                torch.zeros(1, device=device, dtype=torch.int32),
                torch.cumsum(expected_counts, dim=0).int(),
            )
        ),
    )

    assignment_tokens = _region(scratch, "assignment_tokens")
    ascending = torch.arange(tokens, device=device, dtype=torch.int32)
    for position, expert in enumerate(concentrated):
        base = int(offsets[expert])
        assert base == position * tokens
        assert torch.equal(assignment_tokens[base:base + tokens], ascending)


def test_router_handles_disjoint_routing(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    tokens = 8
    groups = {
        token: tuple(range(token * KIMI_K3_TOPK, (token + 1) * KIMI_K3_TOPK))
        for token in range(tokens)
    }
    hidden_states, router_weight, bias = _one_hot_router_inputs(
        device, groups, tokens
    )

    expert_ids, expert_weights, _ = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
    )

    expected_ids = torch.arange(
        tokens * KIMI_K3_TOPK, device=device, dtype=torch.int32
    ).view(tokens, KIMI_K3_TOPK)
    assert torch.equal(expert_ids, expected_ids)
    assert (expert_weights - 1.0 / KIMI_K3_TOPK).abs().max() <= 1e-5

    counts = _region(scratch, "counts")
    assert int(counts[:tokens * KIMI_K3_TOPK].min()) == 1
    assert int(counts[:tokens * KIMI_K3_TOPK].max()) == 1
    assert int(counts[tokens * KIMI_K3_TOPK:].abs().max()) == 0

    offsets = _region(scratch, "offsets")
    assert torch.equal(
        offsets[:tokens * KIMI_K3_TOPK + 1],
        torch.arange(
            tokens * KIMI_K3_TOPK + 1, device=device, dtype=torch.int32
        ),
    )
    assert int(offsets[KIMI_K3_NUM_EXPERTS]) == tokens * KIMI_K3_TOPK

    assignment_tokens = _region(scratch, "assignment_tokens")
    assignment_slots = _region(scratch, "assignment_slots")
    routes = tokens * KIMI_K3_TOPK
    expected_route_tokens = (
        torch.arange(routes, device=device, dtype=torch.int32) // KIMI_K3_TOPK
    )
    expected_route_slots = (
        torch.arange(routes, device=device, dtype=torch.int32) % KIMI_K3_TOPK
    )
    assert torch.equal(assignment_tokens[:routes], expected_route_tokens)
    assert torch.equal(assignment_slots[:routes], expected_route_slots)


def test_router_breaks_score_ties_by_lowest_expert_id(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    tied = tuple(range(20))
    hidden_states, router_weight, bias = _one_hot_router_inputs(
        device, {0: tied}, tokens=1
    )

    expert_ids, expert_weights, _ = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, 1
    )

    assert torch.equal(
        expert_ids[0],
        torch.arange(KIMI_K3_TOPK, device=device, dtype=torch.int32),
    )
    assert (expert_weights - 1.0 / KIMI_K3_TOPK).abs().max() <= 1e-5


def test_router_writes_expert_ids_and_weights_into_scratch(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    tokens = 8
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=5001
    )

    expert_ids, expert_weights, _ = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
    )

    routes = tokens * KIMI_K3_TOPK
    scratch_ids = _region(scratch, "expert_ids")[:routes].view(
        tokens, KIMI_K3_TOPK
    )
    scratch_weights = (
        _region(scratch, "expert_weights")[:routes]
        .view(torch.float32)
        .view(tokens, KIMI_K3_TOPK)
    )
    assert torch.equal(scratch_ids, expert_ids)
    assert torch.equal(scratch_weights, expert_weights)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 64, 100, 128])
def test_latent_projection_matches_torch(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    tokens: int,
) -> None:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=6000 + tokens
    )
    expected = hidden_states @ latent_down_proj.T

    _, _, latent_x = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
    )

    assert latent_x.shape == (tokens, KIMI_K3_LATENT_SIZE)
    assert latent_x.dtype == torch.bfloat16
    assert float(latent_x.float().abs().max()) > 1.0
    torch.testing.assert_close(
        latent_x.float(), expected.float(), atol=0.5, rtol=0.01
    )


@pytest.mark.parametrize("capacity", KIMI_K3_CAPACITY_BUCKETS)
def test_projection_covers_every_capacity_bucket(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    capacity: int,
) -> None:
    assert _capacity(capacity) == capacity
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, capacity, seed=7000 + capacity
    )
    expected = hidden_states @ latent_down_proj.T

    _, _, latent_x = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, capacity
    )

    torch.testing.assert_close(
        latent_x.float(), expected.float(), atol=0.5, rtol=0.01
    )


@pytest.mark.parametrize("active_tokens", [1, 5, 20, 127])
def test_inactive_capacity_rows_stay_zero(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    active_tokens: int,
) -> None:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, KIMI_K3_MAX_TOKENS, seed=8000 + active_tokens
    )
    active_hidden = hidden_states[:active_tokens].contiguous()
    _assert_selection_is_unambiguous(active_hidden, router_weight, bias)
    reference_ids, reference_weights = _router_reference(
        active_hidden, router_weight, bias
    )
    expected_latent = active_hidden @ latent_down_proj.T

    expert_ids, expert_weights, latent_x = _route_and_project(
        hidden_states, router_weight, bias, latent_down_proj, scratch, active_tokens
    )

    assert int(expert_ids[active_tokens:].abs().max().item() or 0) == 0
    assert float(expert_weights[active_tokens:].abs().max()) == 0.0
    assert float(latent_x[active_tokens:].float().abs().max()) == 0.0

    actual_ids, actual_weights = _sorted_routes(
        expert_ids[:active_tokens], expert_weights[:active_tokens]
    )
    expected_ids, expected_weights = _sorted_routes(
        reference_ids, reference_weights
    )
    assert torch.equal(actual_ids, expected_ids)
    assert (actual_weights - expected_weights).abs().max() <= 1e-5
    torch.testing.assert_close(
        latent_x[:active_tokens].float(),
        expected_latent.float(),
        atol=0.5,
        rtol=0.01,
    )

    routes = active_tokens * KIMI_K3_TOPK
    counts = _region(scratch, "counts")
    offsets = _region(scratch, "offsets")
    assert int(counts.sum()) == routes
    assert int(offsets[KIMI_K3_NUM_EXPERTS]) == routes


def test_repeated_calls_reuse_one_scratch_without_reset(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    for tokens in (4, 33, 4, 128):
        hidden_states, router_weight, bias = _seeded_router_inputs(
            device, tokens, seed=9000 + tokens
        )
        _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
        reference_ids, reference_weights = _router_reference(
            hidden_states, router_weight, bias
        )
        expected_latent = hidden_states @ latent_down_proj.T

        expert_ids, expert_weights, latent_x = _route_and_project(
            hidden_states, router_weight, bias, latent_down_proj, scratch, tokens
        )

        actual_ids, actual_weights = _sorted_routes(expert_ids, expert_weights)
        expected_ids, expected_weights = _sorted_routes(
            reference_ids, reference_weights
        )
        assert torch.equal(actual_ids, expected_ids)
        assert (actual_weights - expected_weights).abs().max() <= 1e-5
        torch.testing.assert_close(
            latent_x.float(), expected_latent.float(), atol=0.5, rtol=0.01
        )
        assert int(_region(scratch, "counts").sum()) == tokens * KIMI_K3_TOPK


def test_route_and_project_uses_the_tensor_devices_current_stream(
    device: torch.device, latent_down_proj: torch.Tensor
) -> None:
    from mok import _C

    tokens = 8
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=10007
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    reference_ids, _ = _router_reference(hidden_states, router_weight, bias)
    side_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side_stream):
        staged_hidden = torch.zeros_like(hidden_states)
        staged_scratch = torch.zeros(
            _C.kimi_k3_decode_workspace_bytes(), dtype=torch.uint8, device=device
        )
        torch.cuda._sleep(1 << 28)
        staged_hidden.copy_(hidden_states)
        expert_ids, _, latent_x = _route_and_project(
            hidden_states=staged_hidden,
            router_weight=router_weight,
            router_correction_bias=bias,
            latent_down_proj=latent_down_proj,
            scratch=staged_scratch,
            active_tokens=tokens,
        )
    side_stream.synchronize()

    actual = torch.sort(expert_ids, dim=-1).values
    expected = torch.sort(reference_ids.int(), dim=-1).values
    assert torch.equal(actual, expected)
    assert float(latent_x.float().abs().max()) > 1.0


def test_route_and_project_on_peer_device_ignores_the_current_device(
    device: torch.device, peer_device: torch.device
) -> None:
    from mok import _C

    tokens = 8
    hidden_states, router_weight, bias = _seeded_router_inputs(
        peer_device, tokens, seed=11009
    )
    generator = torch.Generator(device=peer_device).manual_seed(11010)
    peer_latent_down = (
        torch.randn(
            (KIMI_K3_LATENT_SIZE, KIMI_K3_HIDDEN_SIZE),
            generator=generator,
            device=peer_device,
            dtype=torch.float32,
        )
        * (8.0 / math.sqrt(KIMI_K3_HIDDEN_SIZE))
    ).bfloat16().contiguous()
    peer_scratch = torch.zeros(
        _C.kimi_k3_decode_workspace_bytes(), dtype=torch.uint8, device=peer_device
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    reference_ids, _ = _router_reference(hidden_states, router_weight, bias)
    expected_latent = hidden_states @ peer_latent_down.T

    torch.cuda.set_device(device)
    expert_ids, expert_weights, latent_x = _route_and_project(
        hidden_states, router_weight, bias, peer_latent_down, peer_scratch, tokens
    )
    torch.cuda.synchronize(peer_device)

    assert expert_ids.device == peer_device
    assert expert_weights.device == peer_device
    assert latent_x.device == peer_device
    assert torch.cuda.current_device() == device.index
    assert torch.equal(
        torch.sort(expert_ids, dim=-1).values,
        torch.sort(reference_ids.int(), dim=-1).values,
    )
    torch.testing.assert_close(
        latent_x.float(), expected_latent.float(), atol=0.5, rtol=0.01
    )


def test_route_and_project_fake_reports_prepared_metadata(
    device: torch.device,
) -> None:
    from mok import _fake_impls, ops

    schema_names = tuple(
        argument.name
        for argument in torch.ops.mok._kimi_k3_route_and_project.default._schema.arguments
    )
    assert schema_names == _ROUTE_AND_PROJECT_ARGUMENTS
    assert tuple(
        inspect.signature(_fake_impls._kimi_k3_route_and_project_fake).parameters
    ) == schema_names

    with FakeTensorMode():
        hidden_states = torch.empty(
            17, KIMI_K3_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        expert_ids, expert_weights, latent_x = ops._kimi_k3_route_and_project(
            hidden_states,
            torch.empty(
                KIMI_K3_NUM_EXPERTS,
                KIMI_K3_HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.empty(KIMI_K3_NUM_EXPERTS, dtype=torch.float32, device="cuda"),
            torch.empty(
                KIMI_K3_LATENT_SIZE,
                KIMI_K3_HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda"),
            17,
        )

    assert expert_ids.shape == (17, KIMI_K3_TOPK)
    assert expert_ids.dtype == torch.int32
    assert expert_weights.shape == (17, KIMI_K3_TOPK)
    assert expert_weights.dtype == torch.float32
    assert latent_x.shape == (17, KIMI_K3_LATENT_SIZE)
    assert latent_x.dtype == torch.bfloat16


def _valid_call_arguments(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> dict[str, object]:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, 8, seed=12011
    )
    return {
        "hidden_states": hidden_states,
        "router_weight": router_weight,
        "router_correction_bias": bias,
        "latent_down_proj": latent_down_proj,
        "scratch": scratch,
        "active_tokens": 8,
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("active_tokens", 0, "active_tokens"),
        ("active_tokens", 9, "active_tokens"),
        ("active_tokens", 129, "active_tokens"),
    ],
)
def test_route_and_project_rejects_invalid_active_tokens(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    field: str,
    replacement: object,
    message: str,
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments[field] = replacement

    with pytest.raises(RuntimeError, match=message):
        _route_and_project(**arguments)


def test_route_and_project_rejects_undersized_scratch(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments["scratch"] = torch.zeros(
        SCRATCH_BYTES - 1, dtype=torch.uint8, device=device
    )

    with pytest.raises(RuntimeError, match="scratch"):
        _route_and_project(**arguments)


def test_route_and_project_rejects_wrong_latent_down_shape(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments["latent_down_proj"] = latent_down_proj[:, : KIMI_K3_HIDDEN_SIZE - 64
                                                     ].contiguous()

    with pytest.raises(RuntimeError, match="routed_expert_down_proj"):
        _route_and_project(**arguments)


def test_route_and_project_rejects_float32_hidden_states(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments["hidden_states"] = arguments["hidden_states"].float().contiguous()

    with pytest.raises(RuntimeError, match="hidden_states"):
        _route_and_project(**arguments)


def _offset_copy(source: torch.Tensor, element_offset: int) -> torch.Tensor:
    """Copy ``source`` into a contiguous view starting at a nonzero storage offset.

    The caching allocator hands out 256-byte-aligned blocks, so the returned view
    is under-aligned by exactly ``element_offset`` elements while remaining
    contiguous and correctly shaped. That is the shape of the pointer a caller can
    hand the stage without any dtype, shape, or contiguity check noticing.
    """
    flat = torch.empty(
        source.numel() + element_offset, dtype=source.dtype, device=source.device
    )
    assert flat.data_ptr() % 256 == 0
    view = flat[element_offset:].view(source.shape)
    view.copy_(source)
    assert view.is_contiguous()
    assert view.storage_offset() == element_offset
    return view


@pytest.mark.parametrize(
    ("field", "argument", "element_offset", "alignment"),
    _ROUTE_AND_PROJECT_ALIGNMENT,
)
def test_route_and_project_rejects_misaligned_pointers(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    field: str,
    argument: str,
    element_offset: int,
    alignment: int,
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    misaligned = _offset_copy(arguments[argument], element_offset)
    assert misaligned.data_ptr() % alignment != 0
    arguments[argument] = misaligned

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        _route_and_project(**arguments)


@pytest.mark.parametrize(
    ("field", "argument", "element_offset", "alignment"),
    _ROUTE_AND_PROJECT_ALIGNMENT,
)
def test_c_entrypoint_rejects_misaligned_pointers(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    field: str,
    argument: str,
    element_offset: int,
    alignment: int,
) -> None:
    """The extension must guard itself: callers can bypass ``mok.ops`` entirely."""
    from mok import _C

    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments[argument] = _offset_copy(arguments[argument], element_offset)

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        _C._kimi_k3_route_and_project(
            arguments["hidden_states"],
            arguments["router_weight"],
            arguments["router_correction_bias"],
            arguments["latent_down_proj"],
            arguments["scratch"],
            arguments["active_tokens"],
        )


def test_route_and_project_accepts_sufficiently_aligned_offset_views(
    device: torch.device, latent_down_proj: torch.Tensor
) -> None:
    """Nonzero storage offsets are fine as long as they clear the real boundary.

    The correction bias is only read as a scalar float, so a 4-byte-aligned view
    of it must keep working; everything else is offset to the next 16- or 256-byte
    boundary rather than rejected.
    """
    tokens = 8
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=13001
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    reference_ids, _ = _router_reference(hidden_states, router_weight, bias)
    expected_latent = hidden_states @ latent_down_proj.T
    offset_scratch = _offset_copy(
        torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device), 256
    )
    offset_bias = _offset_copy(bias, 1)
    assert offset_bias.data_ptr() % 16 != 0

    expert_ids, _, latent_x = _route_and_project(
        _offset_copy(hidden_states, 8),
        _offset_copy(router_weight, 8),
        offset_bias,
        _offset_copy(latent_down_proj, 8),
        offset_scratch,
        tokens,
    )

    assert torch.equal(
        torch.sort(expert_ids, dim=-1).values,
        torch.sort(reference_ids.int(), dim=-1).values,
    )
    torch.testing.assert_close(
        latent_x.float(), expected_latent.float(), atol=0.5, rtol=0.01
    )


def _profiled_kernel_names(call: Callable[[], object]) -> list[str]:
    """Names of every CUDA kernel the profiler attributes to ``call()``."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        # ``export_chrome_trace`` renames a temporary file into place, so the
        # trace has to be reopened by path once the export has returned.
        trace_path = os.path.join(directory, "trace.json")
        profiler.export_chrome_trace(trace_path)
        with open(trace_path, encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
    return [
        event["name"]
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
    ]


@pytest.mark.parametrize(
    ("tokens", "active_tokens", "expected_kernel"),
    [
        (8, 8, "route_and_project_core_kernel"),
        (64, 5, "route_and_project_core_kernel"),
        (32, 32, "route_and_project_tensor_kernel"),
        (128, 20, "route_and_project_tensor_kernel"),
    ],
)
def test_route_and_project_is_exactly_one_kernel_launch(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    tokens: int,
    active_tokens: int,
    expected_kernel: str,
) -> None:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=14000 + tokens
    )

    def call() -> object:
        return _route_and_project(
            hidden_states,
            router_weight,
            bias,
            latent_down_proj,
            scratch,
            active_tokens,
        )

    call()
    names = _profiled_kernel_names(call)

    assert len(names) == 1, names
    assert expected_kernel in names[0]


def test_launch_counter_sees_a_second_kernel_launch(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    """Keep the one-launch assertions honest by proving the counter can say two."""
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, 8, seed=14999
    )

    def call_twice() -> None:
        for _ in range(2):
            _route_and_project(
                hidden_states, router_weight, bias, latent_down_proj, scratch, 8
            )

    names = _profiled_kernel_names(call_twice)

    assert len(names) == 2, names
