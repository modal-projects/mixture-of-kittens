"""A Kimi K3 decode workspace that is reused, replayed, and left in bad states.

One workspace serves every shape and both capacity paths, so nothing a step
writes into it may survive into the next one -- not a queue ticket, not an
arrival count, not a stale accumulator, and not a phase generation. That is not
visible in one step's output, so it is held here by replaying steps on one
workspace, by changing shape between them, and by poisoning the scratch a step
is supposed to clear.

What one step computes is in ``test_kimi_k3_decode.py``; that it is one launch
on a resident grid is in ``test_kimi_k3_decode_launch.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator

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

from .kimi_k3_tail_support import _prime_barrier_serial, _rotating_skew


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

    The grid phase generation advances five times per launch and the cross-rank
    barrier serial once, and both are compared with unsigned difference rather
    than ordering. Starting them three short of the wrap makes this one launch
    cross it, which a naive ``>=`` comparison could not survive.

    Five advances is the barrier schedule's count, so this asks for it by name.
    The shipped schedule spends one, and
    ``test_the_candidate_is_correct_across_the_unsigned_generation_wrap`` parks
    its generation on the last representable value so that its single barrier
    lands on zero -- the same property at the count this schedule does not have.
    """
    _, _, device = tp8_context
    tokens = 12
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    with barrier_schedule():
        _prime_barrier_serial(workspace, UINT32_MAX - 2)
        _phase(workspace.scratch)[GRID_GENERATION].fill_(
            _as_int32(UINT32_MAX - 2)
        )
        _synchronize_ranks(workspace)

        actual = _decode(workspace, weights, hidden)
        torch.cuda.synchronize(device)
        assert_decode_close(actual, expected)
        # Five barriers from three short of the wrap lands two past it.
        assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 2
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

    for step in range(KIMI_K3_TP_SIZE):
        _synchronize_ranks(workspace)
        _rotating_skew(rank, step)
        actual = _decode(workspace, weights, hidden)
        torch.cuda.synchronize(device)
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


def test_the_gate_up_descriptor_is_encoded_once_per_payload_not_per_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A driver call per decode step would be a driver call inside the step.

    The routed gate/up unit reads its weight slabs through a tensor map, and a
    kernel may only name one as a ``__grid_constant__`` parameter passed by
    value. Building it is ``cuTensorMapEncodeTiled``, a host-side driver call --
    capture-safe, but a decode step is tens of microseconds and the descriptor's
    only input that is not a compile-time constant is the payload's base
    address. So it is encoded on the first launch against a payload and read
    from a cache on every launch after, and this is what says so rather than the
    comment that claims it.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, 16)
    _decode(workspace, weights, hidden)
    _synchronize_ranks(workspace)

    encoded = _C._kimi_k3_fused_w13_packed_maps_encoded()
    # One payload has been launched against, on both capacity paths, so at least
    # one descriptor exists.
    assert encoded >= 1
    for tokens in (16, 32, 16, 64):
        _decode(workspace, weights, hidden_states(device, tokens))
    torch.cuda.synchronize(device)
    assert _C._kimi_k3_fused_w13_packed_maps_encoded() == encoded


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
    begin, names, _ = _C._kimi_k3_decode_phase_clock_metadata()
    return workspace.scratch[begin * 4 : (begin + 2 * len(names)) * 4]


def _phase_clocks(workspace: KimiK3DecodeWorkspace) -> dict[str, int]:
    _, names, _ = _C._kimi_k3_decode_phase_clock_metadata()
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

    Scoped to the barrier schedule, because two of the things asserted here are
    only true of it: the five and six generations, and ``readiness_wait``, which
    is the clock its gate/up-to-down readiness laps and which the shipped
    schedule leaves at zero because its waits are timed per edge instead. The
    shipped schedule's own profile is checked by
    ``test_a_profiled_candidate_reports_where_its_readiness_waits_went``, which
    reads that per-edge band.
    """
    with barrier_schedule():
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
        assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 5
        assert set(_phase_clocks(workspace).values()) == {0}

        with _phase_profiling():
            assert _C._kimi_k3_decode_phase_profile()
            _phase(workspace.scratch)[GRID_GENERATION].zero_()
            _synchronize_ranks(workspace)
            first_result = _decode(workspace, weights, hidden).clone()
            torch.cuda.synchronize(device)
            first = _phase_clocks(workspace)
            assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 6

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

        # Every production region is timed, and the poison is gone from the whole
        # band: the launch reported itself rather than itself plus whatever the
        # band already held. Production readiness replaces the gate/up-to-down
        # grid barrier, so its own wait clock must also report work.
        _, names, parents = _C._kimi_k3_decode_phase_clock_metadata()
        assert set(first) == set(names)
        assert min(first.values()) > 0, first
        for name, cycles in second.items():
            assert 0 < cycles < poison_floor, (name, cycles)

        # A child region's cycles lie inside its parent's, which is the property
        # the report's totals depend on and the only one a run can check. The
        # slack allows for the parent's lap landing a few hundred cycles after the
        # child's on 148 CTAs; a child that were a *sibling* rather than a child
        # would exceed its parent by whole multiples, as summing the band does.
        for index, (name, parent) in enumerate(zip(names, parents, strict=True)):
            del index
            if parent < 0:
                continue
            assert first[name] <= 1.05 * first[names[parent]], (
                name, names[parent], first[name], first[names[parent]]
            )

        # Profiling is off again, so the band stops moving and the launch is back
        # to the six generations a measured replay spends.
        assert not _C._kimi_k3_decode_phase_profile()
        _phase(workspace.scratch)[GRID_GENERATION].zero_()
        _synchronize_ranks(workspace)
        _decode(workspace, weights, hidden)
        torch.cuda.synchronize(device)
        assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 5
        assert _phase_clocks(workspace) == second
