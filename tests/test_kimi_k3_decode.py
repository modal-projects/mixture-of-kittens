"""What one TP8 launch of the production Kimi K3 decode megakernel computes.

``mok.kimi_k3.kimi_k3_decode`` runs a whole decode step -- routing, the routed
latent projection, the mixed W4A8 routed experts, the BF16 shared expert, and
the fused TP8 tail -- in a single launch. This file holds the arithmetic half
of that: the fixtures the measurement rests on, and what the step's output is
against the reference, over both capacity paths and every routing the support
module can build.

The other two halves are siblings. ``test_kimi_k3_decode_launch.py`` holds that
it really is one launch on a grid proven to hold it;
``test_kimi_k3_decode_workspace.py`` holds that a reused workspace carries no
state between steps. The fixtures, the routings, and the oracle are shared
through ``kimi_k3_decode_support.py``, and the host boundary -- the schema, the
alignment contract, the timeout diagnostics, and the rejections -- lives in
``test_kimi_k3_decode_contract.py``. The private stages the production path no
longer calls keep their own suites in ``test_kimi_k3_router.py``,
``test_kimi_k3_expert.py``, ``test_kimi_k3_shared.py``, and
``test_kimi_k3_collectives.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import pytest
import torch
from mok.kimi_k3 import (
    KIMI_K3_TOPK,
    KIMI_K3_TP_SIZE,
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
    kimi_k3_decode,
    kimi_k3_router_reference,
)

from .kimi_k3_decode_support import (
    ACTIVE_EXPERT_UNITS,
    BLOCK8_TOKENS,
    BLOCK16_TOKENS,
    CONFIG,
    CORE_TOKENS,
    DOWN_QUEUE,
    EXPERTS,
    GATE_UP_QUEUE,
    GRID_GENERATION,
    HIDDEN,
    MAX_TOKENS,
    PERSISTENT_CTAS,
    PERSISTENT_THREADS,
    PRIVATE_STAGE_KERNELS,
    RAW_TOKENS,
    ROUTE_LATENT_QUEUE,
    TENSOR_TOKENS,
    UINT32_MAX,
    _as_int32,
    _phase,
    _synchronize_ranks,
    assert_decode_close,
    assert_distinct,
    assert_identical_across_ranks,
    assert_one_production_launch,
    assert_replicated,
    barrier_schedule,
    decode_reference,
    decode_step as _decode,
    dependency_local_schedule,
    hidden_states,
    poison_scratch,
    schedule_queue_tickets,
    profiled_kernel_names,
    published_routes,
    published_shared_partial,
    recorded_allocator_events,
    routing,
    shared_partial_reference,
    weights,  # noqa: F401
    with_routing,
    workspace,  # noqa: F401
)


def _expected_distinct_experts(
    hidden: torch.Tensor, weights: KimiK3DecodeWeights
) -> int:
    expert_ids, _ = kimi_k3_router_reference(
        hidden, weights.router_weight, weights.router_correction_bias
    )
    return int(torch.unique(expert_ids).numel())


# ---------------------------------------------------------------------------
# The fixtures themselves have to hold before anything measured against them.
# ---------------------------------------------------------------------------


def test_the_prepared_weights_are_replicated_and_sharded_as_tp8_requires(
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A shard every rank shares, or a replica one rank differs on, is a bug.

    Every later comparison rests on this: the oracle sums eight *distinct*
    routed and shared partials, so if the shards were accidentally identical a
    rank-local kernel would pass, and if the replicated tensors disagreed the
    eight oracles would not be evaluating the same model.
    """
    rank, _, _ = tp8_context
    assert weights.tp_rank == rank
    for name in (
        "router_weight",
        "router_correction_bias",
        "routed_expert_down_proj",
        "routed_expert_up_proj",
        "routed_latent_rmsnorm_weight",
    ):
        assert_replicated(name, getattr(weights, name))
    for name in (
        "expert_w13_packed",
        "expert_w2_packed",
        "shared_gate_proj",
        "shared_up_proj",
        "shared_down_proj",
    ):
        assert_distinct(name, getattr(weights, name))


# ---------------------------------------------------------------------------
# What one step computes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", RAW_TOKENS)
def test_raw_decode_shapes_match_the_reference(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """Every raw decode count from one token to a full CUDA-core bucket."""
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)
    actual = _decode(workspace, weights, hidden)
    assert actual.shape == (tokens, HIDDEN)
    assert_decode_close(actual, expected)


@pytest.mark.parametrize("tokens", BLOCK8_TOKENS)
def test_block8_request_batches_match_the_reference(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """DFlash block-8 request batches of one through eight requests."""
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)
    actual = _decode(workspace, weights, hidden)
    assert actual.shape == (tokens, HIDDEN)
    assert_decode_close(actual, expected)


@pytest.mark.parametrize("tokens", BLOCK16_TOKENS)
def test_block16_request_batches_match_the_reference(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """DFlash block-16 request batches, up to the full 128-row workspace."""
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)
    actual = _decode(workspace, weights, hidden)
    assert actual.shape == (tokens, HIDDEN)
    assert_decode_close(actual, expected)


@pytest.mark.parametrize(
    ("mode", "tokens", "distinct"),
    [
        ("balanced", 32, None),
        ("concentrated", 32, KIMI_K3_TOPK),
        ("disjoint", 32, 32 * KIMI_K3_TOPK),
    ],
)
def test_pinned_route_distributions_match_the_reference(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    mode: str,
    tokens: int,
    distinct: int | None,
) -> None:
    """Three route shapes with very different expert occupancy.

    Concentrated puts all 512 assignments on sixteen experts, so those experts
    take a full 32-row batch each and the queue is short and deep. Disjoint
    gives every token its own sixteen, so 512 experts take one row apiece and
    the queue is long and shallow. A scheduler that only works when the load is
    even fails one of the two.
    """
    _, _, device = tp8_context
    plan = routing(mode, device, tokens, weights)
    routed = with_routing(weights, plan)
    if distinct is not None:
        assert _expected_distinct_experts(plan.hidden, routed) == distinct
    expected = decode_reference(plan.hidden, routed)
    actual = _decode(workspace, routed, plan.hidden)
    assert_decode_close(actual, expected)


@pytest.mark.parametrize("mode", ["low", "middle", "final"])
def test_expert_placement_extremes_match_the_reference(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    mode: str,
) -> None:
    """Routes pinned to the lowest, middle, and final expert IDs.

    An off-by-one in the compacted expert list, in the expert-major offsets, or
    in a weight's base address is invisible in the middle of the table and
    obvious at either end of it.
    """
    _, _, device = tp8_context
    tokens = 16
    plan = routing(mode, device, tokens, weights)
    routed = with_routing(weights, plan)
    expected_ids, _ = kimi_k3_router_reference(
        plan.hidden, routed.router_weight, routed.router_correction_bias
    )
    boundary = {
        "low": 0,
        "middle": EXPERTS // 2,
        "final": EXPERTS - KIMI_K3_TOPK,
    }[mode]
    assert set(expected_ids.flatten().tolist()) == set(
        range(boundary, boundary + KIMI_K3_TOPK)
    )
    expected = decode_reference(plan.hidden, routed)
    actual = _decode(workspace, routed, plan.hidden)
    assert_decode_close(actual, expected)


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_the_router_publishes_the_exact_ids_and_weights(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """The selection is exact, not merely close.

    Top-16 selection is discrete: one wrong expert changes the output by a
    whole expert's contribution, which an aggregate error metric on a 7168-wide
    row can average away. Comparing the published IDs and normalized weights
    directly is what makes that impossible to miss.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)

    published_ids, published_weights = published_routes(
        workspace.scratch, tokens
    )
    expected_ids, expected_weights = kimi_k3_router_reference(
        hidden, weights.router_weight, weights.router_correction_bias
    )
    # Neither side promises an order within a token's sixteen slots, so the
    # pairing is compared rather than the sequence.
    for token in range(tokens):
        actual = dict(
            zip(
                published_ids[token].tolist(),
                published_weights[token].tolist(),
            )
        )
        reference = dict(
            zip(
                expected_ids[token].tolist(),
                expected_weights[token].tolist(),
            )
        )
        assert actual.keys() == reference.keys()
        for expert, weight in reference.items():
            assert actual[expert] == pytest.approx(weight, rel=2e-3, abs=1e-6)


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_the_shared_partial_matches_the_bf16_rounded_boundary(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """SiTU reads BF16 projections, not the FP32 accumulators behind them.

    ``shared.cuh`` stores each gate and up projection to BF16 scratch and reads
    it back to evaluate the activation, which is the boundary the official
    model defines. The difference is small -- one BF16 rounding into a
    saturating nonlinearity -- but it is systematic, so an oracle that fed SiTU
    the raw FP32 accumulators would sit permanently offset from a *correct*
    kernel and would have to hide behind a wider tolerance.

    The tail only reads the collective buffer, so this rank's own shared
    partial survives the launch and can be compared against both formulations
    directly, unmixed with the routed path or the eight-way reduction.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)

    actual = published_shared_partial(workspace.collective_buffer, tokens)
    rounded = shared_partial_reference(hidden, weights).bfloat16()
    unrounded = shared_partial_reference(
        hidden, weights, round_projections=False
    ).bfloat16()

    # One BF16 rounding of the gate is half an output ULP by the time it
    # reaches the down projection, so the two formulations are compared where
    # the device leaves its answer -- in BF16 -- and over the whole block. On
    # a third of the elements they land on different BF16 values, which is the
    # separation the comparison below needs to be meaningful.
    disagreement = float((rounded != unrounded).float().mean())
    rounded_error = float((actual.float() - rounded.float()).abs().mean())
    unrounded_error = float((actual.float() - unrounded.float()).abs().mean())
    assert disagreement > 0.05, disagreement
    assert rounded_error < 0.25 * unrounded_error, (
        rounded_error,
        unrounded_error,
        disagreement,
    )


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_the_output_is_finite_and_identical_on_every_rank(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """All eight ranks leave the same rows behind, and none of them is NaN."""
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    actual = _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)
    assert torch.isfinite(actual.float()).all()
    assert_identical_across_ranks(actual)


def test_rows_past_the_active_block_are_neither_returned_nor_disturbed(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A short step must not touch the mailbox rows it was not given.

    The mailbox is a fixed 128-row symmetric allocation that every step shares,
    and every rank writes into every other rank's copy of it, so a shard role
    whose loop bound came from the allocation instead of from the active count
    would corrupt rows this step never claimed.
    """
    _, _, device = tp8_context
    tokens = 9
    sentinel = -3.5
    workspace.output_mailbox.fill_(sentinel)
    _synchronize_ranks(workspace)

    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)
    actual = _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)

    assert actual.shape == (tokens, HIDDEN)
    assert_decode_close(actual, expected)
    untouched = workspace.output_mailbox.view(MAX_TOKENS, HIDDEN)[tokens:]
    assert torch.equal(
        untouched,
        torch.full_like(untouched, sentinel),
    )
