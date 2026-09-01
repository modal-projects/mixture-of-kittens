"""What the fused Kimi K3 router and routed latent-down projection compute.

The selection against a float64 reference over seeded, concentrated, disjoint
and tied routings; the weights that must sum to one per token; the descending
order the routes are written in; and the projection against torch at every
capacity bucket, with the inactive rows of the bucket left at zero.

The host boundary -- the stream, the device, the fake, the rejections, the
alignment cases and the launch count -- is in
``test_kimi_k3_router_contract.py``, and what both rest on is in
``kimi_k3_router_support.py``.
"""

from __future__ import annotations

import pytest
import torch

from .kimi_k3_router_support import (
    _assert_selection_is_unambiguous,
    _capacity,
    device,
    KIMI_K3_CAPACITY_BUCKETS,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_MAX_TOKENS,
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_TOPK,
    latent_down_proj,
    _one_hot_router_inputs,
    _region,
    _route_and_project,
    _router_reference,
    scratch,
    SCRATCH_ALIGNMENT,
    SCRATCH_BYTES,
    SCRATCH_LAYOUT,
    _seeded_router_inputs,
    _sorted_routes,
)


def test_workspace_bytes_matches_the_documented_scratch_layout(
    device: torch.device,
) -> None:
    from mok import _C

    assert SCRATCH_BYTES == 8_111_872
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    assert SCRATCH_LAYOUT["phase"][0] == 0
    assert SCRATCH_LAYOUT["expert_ids"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["expert_weights"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["counts"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["offsets"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["assignment_tokens"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["assignment_slots"][0] * 4 % SCRATCH_ALIGNMENT == 0
    assert SCRATCH_LAYOUT["latent_mxfp8"][0] * 4 == 40_704
    assert SCRATCH_LAYOUT["latent_scale"][0] * 4 == 499_456
    assert SCRATCH_LAYOUT["situ_mxfp8"][0] * 4 == 513_792
    assert SCRATCH_LAYOUT["situ_scale"][0] * 4 == 1_300_224
    assert SCRATCH_LAYOUT["routed_accumulator"][0] * 4 == 1_324_800
    assert SCRATCH_LAYOUT["shared_gate"][0] * 4 == 4_994_816
    assert SCRATCH_LAYOUT["shared_up"][0] * 4 == 5_191_424
    assert SCRATCH_LAYOUT["shared_activated"][0] * 4 == 5_388_032
    assert SCRATCH_LAYOUT["tail_normalized"][0] * 4 == 5_584_640
    assert SCRATCH_LAYOUT["tail_shared_shard"][0] * 4 == 6_502_144


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
