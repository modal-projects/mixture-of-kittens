"""TP8 GPU tests for the production one-launch Kimi K3 decode megakernel.

``mok.kimi_k3.kimi_k3_decode`` runs a whole decode step -- routing, the routed
latent projection, the mixed W4A8 routed experts, the BF16 shared expert, and
the fused TP8 tail -- in a single launch of
``kimi_k3_decode_persistent_kernel``. These tests pin what that step computes,
that it really is one launch, that the resident grid it needs is proven rather
than assumed, and that a reused workspace never carries state between steps.

The fixtures, the routings, and the oracle live in
``kimi_k3_decode_support.py``, and the host boundary -- the schema, the
alignment contract, the timeout diagnostics, and the rejections -- lives in
``test_kimi_k3_decode_contract.py``. The private stages the production path no
longer calls keep their own suites in ``test_kimi_k3_router.py``,
``test_kimi_k3_expert.py``, ``test_kimi_k3_shared.py``, and
``test_kimi_k3_collectives.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator

import pytest
import torch

from mok import _C
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
    LATENT,
    MAX_TOKENS,
    PERSISTENT_CTAS,
    PERSISTENT_KERNEL,
    PERSISTENT_THREADS,
    PRIVATE_STAGE_KERNELS,
    RAW_TOKENS,
    ROUTE_LATENT_QUEUE,
    TENSOR_TOKENS,
    UINT32_MAX,
    _as_int32,
    _phase,
    _region,
    _synchronize_ranks,
    assert_decode_close,
    assert_distinct,
    assert_identical_across_ranks,
    assert_replicated,
    decode_reference,
    decode_step as _decode,
    hidden_states,
    poison_scratch,
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
from .kimi_k3_tail_support import _prime_barrier_serial, _rotating_skew


# Fixed stage widths the task plan is built from, mirrored from the headers so
# that a silent retiling is caught here rather than only by a slow test.
CORE_PROJECTION_UNITS = 112       # skinny_gemm::kCoreCtas
TENSOR_PROJECTION_UNITS = 28      # skinny_gemm::kTensorCtas
SCORE_SHARDS = 8                  # router::kScoreShards
GATE_UP_TILES = 3                 # expert_mxfp4::kGateUpTiles
GROUPED_DOWN_UNITS = 7            # grouped_pipeline::kGroupedDownUnits
CORE_SHARED_GATE_UNITS = 24       # shared_experts::kCoreGateCtas
TENSOR_SHARED_GATE_UNITS = 6      # shared_experts::kTensorGateCtas
ACTIVATION_UNITS = 6              # shared_experts::kActivationCtas
CORE_SHARED_DOWN_UNITS = 112      # shared_experts::kCoreDownCtas
TENSOR_SHARED_DOWN_UNITS = 56     # shared_experts::kTensorDownCtas
CORE_TAIL_UNITS = 1 + 32 + 14     # coordinator + reduce + core shard
TENSOR_TAIL_UNITS = 1 + 32 + 7    # coordinator + reduce + tensor shard


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
        "expert_w1_packed",
        "expert_w3_packed",
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


# ---------------------------------------------------------------------------
# One launch, and a grid that is proven to hold it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_the_whole_step_is_exactly_one_persistent_kernel_launch(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """The launch-count gate, on both capacity paths.

    One launch is the whole point of the megakernel, so it is asserted on the
    profiler's own record of what reached the device rather than inferred from
    the absence of a second entrypoint call.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    # Warm up first: the first call on a device reserves shared memory and
    # measures occupancy, neither of which is a kernel, but both of which are
    # noise in a trace.
    _decode(workspace, weights, hidden)
    _synchronize_ranks(workspace)

    names = profiled_kernel_names(
        lambda: _decode(workspace, weights, hidden)
    )
    assert len(names) == 1, names
    assert PERSISTENT_KERNEL in names[0], names
    for private in PRIVATE_STAGE_KERNELS:
        assert all(private not in name for name in names), names


def test_the_persistent_grid_is_proven_resident_before_it_is_launched(
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Every phase barrier counts all 148 CTAs, so a partial grid deadlocks.

    The occupancy query is the measurement that matters and the SM count is
    what turns it into a whole-grid answer, so both are asserted here, and the
    guard is asserted to reject the two ways either one can fail.
    """
    _, _, device = tp8_context
    ctas, threads, shared_bytes = _C._kimi_k3_decode_grid_shape()
    assert (ctas, threads) == (PERSISTENT_CTAS, PERSISTENT_THREADS)
    # More than half of an SM's 227 KiB, which is what forces one CTA per SM
    # independently of any occupancy heuristic.
    assert 2 * shared_bytes > 227 * 1024

    available = torch.cuda.get_device_properties(device).multi_processor_count
    assert available >= PERSISTENT_CTAS
    for tensor_path in (False, True):
        blocks = _C._kimi_k3_decode_resident_blocks_per_sm(tensor_path)
        assert blocks >= 1, (tensor_path, blocks)
        _C._kimi_k3_decode_validate_residency(available, blocks)

    with pytest.raises(RuntimeError, match="at least one CTA per SM"):
        _C._kimi_k3_decode_validate_residency(available, 0)
    with pytest.raises(RuntimeError, match="co-reside one per SM"):
        _C._kimi_k3_decode_validate_residency(PERSISTENT_CTAS - 1, 1)


def test_the_benchmark_grid_override_cannot_leak_into_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private tuning hook is finite, guarded, and fail-closed."""
    production = _C._kimi_k3_decode_grid_shape()[0]
    candidates = tuple(_C._kimi_k3_decode_benchmark_grids())
    assert production == PERSISTENT_CTAS == candidates[-1]

    monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", raising=False)
    assert _C._kimi_k3_decode_benchmark_grid() == production
    with pytest.raises(RuntimeError, match="benchmark-only"):
        _C._kimi_k3_decode_set_benchmark_grid(candidates[0])

    monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
    for invalid in (0, candidates[0] - 1, candidates[0] + 1, production + 1):
        with pytest.raises(RuntimeError, match="benchmark grid must be one of"):
            _C._kimi_k3_decode_set_benchmark_grid(invalid)
    for candidate in candidates:
        _C._kimi_k3_decode_set_benchmark_grid(candidate)
        assert _C._kimi_k3_decode_benchmark_grid() == candidate

    _C._kimi_k3_decode_set_benchmark_grid(candidates[0])
    monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    assert _C._kimi_k3_decode_benchmark_grid() == production


def test_the_shared_memory_reservation_happens_once_per_device(
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Raising the shared-memory cap is a runtime call a graph must not record.

    It is cached per device and per capacity path, so a step captured into a
    CUDA graph carries no runtime API call at all -- which is what lets the
    graph tests below replay a thousand times.
    """
    _, _, device = tp8_context
    for tensor_path in (False, True):
        _C._kimi_k3_decode_resident_blocks_per_sm(tensor_path)
    reservations = _C._kimi_k3_decode_shared_memory_reservations(device.index)
    assert reservations == 2
    for tensor_path in (False, True):
        _C._kimi_k3_decode_resident_blocks_per_sm(tensor_path)
    assert (
        _C._kimi_k3_decode_shared_memory_reservations(device.index)
        == reservations
    )


@pytest.mark.parametrize(
    "tokens", [1, 8, 9, 16, 32, 64, 128]
)
def test_the_task_plan_covers_every_logical_task_of_the_step(
    tokens: int,
) -> None:
    """The plan the scheduler hands out, against an independent count.

    A token's sixteen routes are sixteen *distinct* experts, so the number of
    occupied experts is bounded by ``min(16M, 896)`` and each occupied expert
    is exactly one batch. That bound is what the queue lengths are built from,
    and it is what keeps a 128-row step from needing more than the 896 experts
    that exist.
    """
    tensor_path = tokens > 8
    experts = min(KIMI_K3_TOPK * tokens, EXPERTS)
    route_latent, gate_up, down, tail, grid = _C._kimi_k3_decode_task_plan(
        tokens
    )
    assert grid == PERSISTENT_CTAS
    assert route_latent == tokens * SCORE_SHARDS + (
        TENSOR_PROJECTION_UNITS if tensor_path else CORE_PROJECTION_UNITS
    )
    assert gate_up == (
        2 * TENSOR_SHARED_GATE_UNITS if tensor_path
        else CORE_SHARED_GATE_UNITS
    ) + experts * GATE_UP_TILES
    assert down == (
        (ACTIVATION_UNITS + TENSOR_SHARED_DOWN_UNITS) if tensor_path
        else CORE_SHARED_DOWN_UNITS
    ) + experts * GROUPED_DOWN_UNITS
    assert tail == (TENSOR_TAIL_UNITS if tensor_path else CORE_TAIL_UNITS)
    # Every phase hands out far more tasks than there are CTAs, which is the
    # reason the grid claims work instead of owning it.
    assert max(gate_up, down) > grid


@pytest.mark.parametrize(
    ("mode", "tokens"), [("balanced", 64), ("concentrated", 64)]
)
def test_the_grid_claims_only_occupied_experts(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    mode: str,
    tokens: int,
) -> None:
    """No production CTA sweeps all 896 experts looking for work.

    Phase 2 compacts the occupied experts into a list and publishes its length,
    and phases 3 and 4 size their queues from that length. The counters the
    launch leaves behind report both facts: the published length is exactly the
    number of experts the router actually chose, and each queue was drained by
    many CTAs racing for tickets rather than by one CTA walking the table.
    """
    _, _, device = tp8_context
    plan = routing(mode, device, tokens, weights)
    routed = with_routing(weights, plan)
    distinct = _expected_distinct_experts(plan.hidden, routed)

    _decode(workspace, routed, plan.hidden)
    torch.cuda.synchronize(device)
    counters = _phase(workspace.scratch)

    assert int(counters[ACTIVE_EXPERT_UNITS].item()) == distinct
    assert distinct < EXPERTS

    gate_up_units = (
        2 * TENSOR_SHARED_GATE_UNITS + distinct * GATE_UP_TILES
    )
    down_units = (
        ACTIVATION_UNITS
        + TENSOR_SHARED_DOWN_UNITS
        + distinct * GROUPED_DOWN_UNITS
    )
    # region agent log
    with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as debug_file:
        debug_file.write(json.dumps({
            "hypothesisId": "Q",
            "location": "tests/test_kimi_k3_decode.py:queue_accounting",
            "message": "grouped queue counters after decode",
            "data": {
                "rank": tp8_context[0],
                "mode": mode,
                "distinct": distinct,
                "gate_up_units": gate_up_units,
                "gate_up_counter": int(counters[GATE_UP_QUEUE].item()),
                "down_units": down_units,
                "down_counter": int(counters[DOWN_QUEUE].item()),
            },
            "timestamp": time.time_ns() // 1_000_000,
        }) + "\n")
    # endregion
    # Batched routed queues stop one width-four claim past their last unit for
    # every CTA that was refused; the route/latent queue still claims singly.
    for counter, units, claim_width in (
        (GATE_UP_QUEUE, gate_up_units, 4),
        (DOWN_QUEUE, down_units, 4),
        (
            ROUTE_LATENT_QUEUE,
            tokens * SCORE_SHARDS + TENSOR_PROJECTION_UNITS,
            1,
        ),
    ):
        drained = int(counters[counter].item())
        rounded = (units + claim_width - 1) // claim_width * claim_width
        assert units <= drained <= (
            rounded + claim_width * PERSISTENT_CTAS
        ), (
            counter,
            drained,
        )


# ---------------------------------------------------------------------------
# A workspace that is reused, replayed, and left in unhelpful states.
# ---------------------------------------------------------------------------


def test_one_workspace_serves_changing_shapes_and_both_capacity_paths(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Nothing a step leaves behind may change what the next step computes.

    The order alternates the capacity paths and revisits the first shape last,
    so a stale queue counter, a routed accumulator that was not re-zeroed, or a
    quantized latent left over from a wider step all show up as a wrong answer
    on a shape that already passed.
    """
    _, _, device = tp8_context
    for tokens in (CORE_TOKENS, TENSOR_TOKENS, 1, 128, CORE_TOKENS):
        hidden = hidden_states(device, tokens)
        expected = decode_reference(hidden, weights)
        actual = _decode(workspace, weights, hidden)
        assert actual.shape == (tokens, HIDDEN)
        assert_decode_close(actual, expected)


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_a_poisoned_scratch_does_not_survive_a_launch(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """Every data region a step reads is one the same step wrote.

    The routed accumulator is the sharpest case: 6 272 down units add into it
    atomically, so a launch that trusted whatever was there would return the
    previous step's routed latent plus this one's. Poisoning every region and
    getting the same answer is what proves the phase-0 clear and the
    producer-before-consumer ordering inside the launch.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    poison_scratch(workspace.scratch)
    _synchronize_ranks(workspace)
    actual = _decode(workspace, weights, hidden)
    assert_decode_close(actual, expected)


def test_the_launch_is_correct_across_the_unsigned_serial_wrap(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Both wrap-safe counters are parked just below 2^32 and pushed over it.

    The grid phase generation advances seven times per launch and the cross-rank
    barrier serial once, and both are compared with unsigned difference rather
    than ordering. Starting them three short of the wrap makes this one launch
    cross it, which a naive ``>=`` comparison could not survive.
    """
    _, _, device = tp8_context
    tokens = 12
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    _prime_barrier_serial(workspace, UINT32_MAX - 2)
    _phase(workspace.scratch)[GRID_GENERATION].fill_(_as_int32(UINT32_MAX - 2))
    _synchronize_ranks(workspace)

    actual = _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)
    assert_decode_close(actual, expected)
    # Six barriers from three short of the wrap lands three past it.
    assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 3
    assert_identical_across_ranks(actual)


def test_rotating_rank_skew_leaves_the_step_bit_identical(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Delay each rank in turn; the answer must not depend on who arrived first.

    A rank that published its collective partial without a system-scope release
    -- or a tail role that read a peer's before its coordinator said it was
    there -- is invisible when all eight ranks happen to keep step. Making each
    rank lead once and trail once is what removes that coincidence.
    """
    rank, _, device = tp8_context
    tokens = 20
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    baseline = _decode(workspace, weights, hidden).clone()
    torch.cuda.synchronize(device)
    assert_decode_close(baseline, expected)
    baseline_routed = _region(
        workspace.scratch, "routed_accumulator", torch.int64
    )[: tokens * LATENT].clone()
    # region agent log
    with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as debug_file:
        debug_file.write(json.dumps({
            "hypothesisId": "F2,F4",
            "location": "tests/test_kimi_k3_decode.py:skew_baseline",
            "message": "rank-skew baseline completed",
            "data": {
                "rank": rank,
                "gate_up_counter": int(_phase(workspace.scratch)[GATE_UP_QUEUE].item()),
                "down_counter": int(_phase(workspace.scratch)[DOWN_QUEUE].item()),
                "fixed_min": int(baseline_routed.min().item()),
                "fixed_max": int(baseline_routed.max().item()),
            },
            "timestamp": time.time_ns() // 1_000_000,
        }) + "\n")
    # endregion

    for step in range(KIMI_K3_TP_SIZE):
        _synchronize_ranks(workspace)
        _rotating_skew(rank, step)
        actual = _decode(workspace, weights, hidden)
        torch.cuda.synchronize(device)
        difference = (actual.float() - baseline.float()).abs()
        # region agent log
        with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps({
                "hypothesisId": "F1,F2,F4",
                "location": "tests/test_kimi_k3_decode.py:skew_comparison",
                "message": "rank-skew replay compared with baseline",
                "data": {
                    "rank": rank,
                    "step": step,
                    "equal": bool(torch.equal(actual, baseline)),
                    "different_values": int(torch.count_nonzero(difference).item()),
                    "max_abs": float(difference.max().item()),
                    "gate_up_counter": int(
                        _phase(workspace.scratch)[GATE_UP_QUEUE].item()
                    ),
                    "down_counter": int(
                        _phase(workspace.scratch)[DOWN_QUEUE].item()
                    ),
                },
                "timestamp": time.time_ns() // 1_000_000,
            }) + "\n")
        # endregion
        routed_difference = (
            _region(workspace.scratch, "routed_accumulator", torch.int64)[
                : tokens * LATENT
            ]
            - baseline_routed
        ).abs()
        # region agent log
        with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps({
                "hypothesisId": "F2",
                "location": "tests/test_kimi_k3_decode.py:skew_routed_accumulator",
                "message": "rank-skew routed accumulator compared with baseline",
                "data": {
                    "rank": rank,
                    "step": step,
                    "different_values": int(
                        torch.count_nonzero(routed_difference).item()
                    ),
                    "max_abs": float(routed_difference.max().item()),
                },
                "timestamp": time.time_ns() // 1_000_000,
            }) + "\n")
        # endregion
        assert_identical_across_ranks(actual)
        assert torch.equal(actual, baseline), step


def test_one_thousand_graph_replays_reproduce_the_eager_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """The step is capturable, and replaying it is not a one-shot trick.

    Everything a launch needs to reset it resets itself on the device, so a
    captured graph -- which can neither allocate nor call a runtime API -- has
    to be replayable indefinitely. A thousand replays is also a thousand
    crossings of every generation counter in the step.
    """
    _, _, device = tp8_context
    tokens = 16
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    # Warm up outside capture so no allocation or lazy init lands in the graph.
    _decode(workspace, weights, hidden)
    _synchronize_ranks(workspace)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kimi_k3_decode(CONFIG, workspace, weights, hidden)
    _synchronize_ranks(workspace)

    workspace.output_mailbox.fill_(-11.0)
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize(device)

    actual = workspace.output_mailbox.view(MAX_TOKENS, HIDDEN)[:tokens]
    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(actual, expected)
    assert_identical_across_ranks(actual)


def test_the_step_allocates_nothing_and_returns_a_mailbox_view(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """The returned rows are the mailbox itself, not a copy of it.

    A decode step runs inside a serving loop's steady state, so an allocation
    or a 1.8 MB copy per call is a cost the caller cannot see and cannot avoid.

    The net allocation is checked as well, but on its own it would pass a step
    that allocated a scratch tensor and freed it again, so the allocator's
    event history is what the claim actually rests on: a warmed call must raise
    no allocator event at all.
    """
    _, _, device = tp8_context
    tokens = 24
    hidden = hidden_states(device, tokens)
    _decode(workspace, weights, hidden)
    _synchronize_ranks(workspace)

    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    with recorded_allocator_events(device) as events:
        actual = kimi_k3_decode(CONFIG, workspace, weights, hidden)
    assert int(workspace.error_flag.item()) == 0
    assert torch.cuda.memory_allocated(device) == before
    assert events == [], events

    # An empty history proves nothing unless the recorder was listening, so
    # the same instrumentation is shown catching a transient that
    # `memory_allocated()` alone would report as no allocation at all.
    with recorded_allocator_events(device) as control:
        torch.empty(1024, device=device).sum()
    assert torch.cuda.memory_allocated(device) == before
    assert "alloc" in control, control

    mailbox = workspace.output_mailbox
    assert actual.data_ptr() == mailbox.data_ptr()
    assert (
        actual.untyped_storage().data_ptr()
        == mailbox.untyped_storage().data_ptr()
    )
    assert actual.shape == (tokens, HIDDEN)
    # A view, so a write through the mailbox is visible through the result.
    mailbox.view(MAX_TOKENS, HIDDEN)[0, 0] = 1.25
    assert float(actual[0, 0]) == 1.25


def test_the_step_runs_on_the_current_stream(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A side stream must carry the launch, and its ordering must hold."""
    _, _, device = tp8_context
    tokens = 6
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    _synchronize_ranks(workspace)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        captured = _decode(workspace, weights, hidden).clone()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    assert_decode_close(captured, expected)


@contextlib.contextmanager
def _phase_profiling() -> Iterator[None]:
    """Turn the clock64 accumulators on, the way the benchmark process does."""
    previous = os.environ.get("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    _C._kimi_k3_decode_set_phase_profile(True)
    try:
        yield
    finally:
        _C._kimi_k3_decode_set_phase_profile(False)
        if previous is None:
            os.environ.pop("MOK_KIMI_K3_ENABLE_GRID_TUNING", None)
        else:
            os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = previous


def _phase_clock_band(workspace: KimiK3DecodeWorkspace) -> torch.Tensor:
    begin, names = _C._kimi_k3_decode_phase_clock_metadata()
    return workspace.scratch[begin * 4 : (begin + 2 * len(names)) * 4]


def _phase_clocks(workspace: KimiK3DecodeWorkspace) -> dict[str, int]:
    _, names = _C._kimi_k3_decode_phase_clock_metadata()
    counters = _phase_clock_band(workspace).cpu().view(torch.int64).tolist()
    return dict(zip(names, counters, strict=True))


def test_a_profiled_launch_reports_its_own_cycles_and_costs_one_barrier(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Block 0 zeroes the band, so the grid has to wait for it before timing.

    Without a barrier between the clearing and the first timed region, a CTA
    that finished phase 0 early would have its cycles erased by a store that
    landed after them, and the profile would under-report by however many CTAs
    won that race -- exactly the launches a profile is read to explain. The
    barrier is what makes each profiled launch report only itself. The band is
    poisoned with a sentinel far above any real cycle count before the second
    profiled launch, so a counter that survives it is a counter the launch
    never cleared, and the extra rendezvous shows up directly as one more grid
    generation than an unprofiled launch spends.
    """
    _, _, device = tp8_context
    tokens = TENSOR_TOKENS
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    _decode(workspace, weights, hidden)
    _synchronize_ranks(workspace)

    # An unprofiled launch takes neither the clearing nor its barrier, and
    # writes no counter, so the band it is handed back is the one it was given.
    _phase_clock_band(workspace).zero_()
    _phase(workspace.scratch)[GRID_GENERATION].zero_()
    _synchronize_ranks(workspace)
    _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)
    assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 6
    assert set(_phase_clocks(workspace).values()) == {0}

    with _phase_profiling():
        assert _C._kimi_k3_decode_phase_profile()
        _phase(workspace.scratch)[GRID_GENERATION].zero_()
        _synchronize_ranks(workspace)
        first_result = _decode(workspace, weights, hidden).clone()
        torch.cuda.synchronize(device)
        first = _phase_clocks(workspace)
        assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 7

        # Every byte of the band set to one is 0x0101010101010101 in each
        # counter, 7.2e16 cycles: eight orders of magnitude above anything a
        # decode step spends, and it survives being added to. A counter that
        # comes back small is one this launch cleared.
        poison_floor = 1 << 40
        _phase_clock_band(workspace).fill_(1)
        _synchronize_ranks(workspace)
        second_result = _decode(workspace, weights, hidden).clone()
        torch.cuda.synchronize(device)
        second = _phase_clocks(workspace)

    assert_decode_close(first_result, expected)
    assert_decode_close(second_result, expected)
    assert int(workspace.error_flag.item()) == 0

    # Every region of the step is timed, by every CTA that ran it, and the
    # poison is gone from all of them: the launch reported itself rather than
    # itself plus whatever the band already held.
    assert set(first) == set(_C._kimi_k3_decode_phase_clock_metadata()[1])
    assert min(first.values()) > 0, first
    for name, cycles in second.items():
        assert 0 < cycles < poison_floor, (name, cycles)

    # Profiling is off again, so the band stops moving and the launch is back
    # to the six generations a measured replay spends.
    assert not _C._kimi_k3_decode_phase_profile()
    _phase(workspace.scratch)[GRID_GENERATION].zero_()
    _synchronize_ranks(workspace)
    _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)
    assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 6
    assert _phase_clocks(workspace) == second
