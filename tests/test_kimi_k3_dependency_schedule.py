"""TP8 GPU tests for the dependency-local Kimi K3 decode schedule.

The candidate replaces four of the production kernel's five full-grid barriers
with seven topologically ordered queues and ten bounded readiness edges. What
the source cannot show is that the resulting order of arrival computes the same
step, so everything here is a differential test: the same workspace, the same
weights, the same tokens, run under both schedules, required to agree with the
oracle and with each other.

The structural properties -- the scan order, the edge directions, the barrier
count, the fence scopes, the counter bounds -- are pinned without a device in
``test_kimi_k3_dependency_schedule_source.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import os
import random
from typing import NamedTuple

import pytest
import torch

from mok import _C
from mok.kimi_k3 import (
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
    kimi_k3_decode,
)

from .kimi_k3_decode_support import (
    CONFIG,
    CORE_TOKENS,
    GRID_GENERATION,
    HIDDEN,
    MAX_TOKENS,
    PERSISTENT_CTAS,
    TENSOR_TOKENS,
    _phase,
    _synchronize_ranks,
    assert_decode_close,
    assert_identical_across_ranks,
    assert_one_production_launch,
    barrier_schedule,
    decode_reference,
    decode_step as _decode,
    dependency_local_schedule,
    hidden_states,
    poison_scratch,
    profiled_kernel_names,
    recorded_allocator_events,
    routing,
    weights,  # noqa: F401
    with_routing,
    workspace,  # noqa: F401
)
from .kimi_k3_tail_support import (
    UINT32_MAX,
    _as_int32,
    _prime_barrier_serial,
)


CANDIDATE_KERNEL = "kimi_k3_decode_dependency_local_kernel"
PRODUCTION_KERNEL = "kimi_k3_decode_persistent_kernel"

# Every capacity bucket the brief names, plus the two off-bucket counts that
# straddle the core/tensor boundary.
SCHEDULE_TOKENS = (1, 8, 9, 16, 32, 64, 128)

# The routings that make the routed queues degenerate in every direction an
# expert-local edge could get wrong.
ADVERSARIAL_ROUTES = ("disjoint", "concentrated", "low", "final", "balanced")


class ScheduleEdge(NamedTuple):
    """One row of ``kScheduleEdges`` as the extension carries it."""

    name: str
    consumer: int
    producer: int
    counter: int
    code: int
    space: int
    scope: int
    target_kind: int
    static_target: int
    counter_indexed: bool


# types.cuh: ScheduleCounterSpace, ScheduleEdgeScope, ScheduleEdgeTarget.
IN_REGION, IN_EXPERT_COUNTS = 0, 1
DEVICE_SCOPE, SYSTEM_SCOPE = 0, 1
STATIC_TARGET, DYNAMIC_TARGET = 0, 1
DIAGNOSTIC_BASE = 128  # types.cuh: kScheduleDiagnosticBase


def _schedule_edges() -> list[ScheduleEdge]:
    return [
        ScheduleEdge(*edge)
        for edge in _C._kimi_k3_decode_schedule_edges()
    ]


def _schedule_band(workspace: KimiK3DecodeWorkspace) -> torch.Tensor:
    wait_begin, _, _, edge_names, queue_names = (
        _C._kimi_k3_decode_schedule_clock_metadata()
    )
    words = 2 * (2 * len(edge_names) + len(queue_names))
    return workspace.scratch[wait_begin * 4 : (wait_begin + words) * 4]


def _schedule_clocks(
    workspace: KimiK3DecodeWorkspace,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    (
        wait_begin,
        edge_makespan_begin,
        queue_makespan_begin,
        edge_names,
        queue_names,
    ) = _C._kimi_k3_decode_schedule_clock_metadata()

    def band(begin: int, count: int) -> list[int]:
        words = workspace.scratch[begin * 4 : (begin + 2 * count) * 4].cpu()
        return words.view(torch.int64).tolist()

    return (
        dict(zip(edge_names, band(wait_begin, len(edge_names)), strict=True)),
        dict(
            zip(
                edge_names,
                band(edge_makespan_begin, len(edge_names)),
                strict=True,
            )
        ),
        dict(
            zip(
                queue_names,
                band(queue_makespan_begin, len(queue_names)),
                strict=True,
            )
        ),
    )


# ---------------------------------------------------------------------------
# The candidate is a schedule, not a different computation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", SCHEDULE_TOKENS)
def test_the_candidate_reproduces_the_production_step_bit_for_bit(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """Every stage is the production stage, so only the order of arrival moves.

    Nothing in the candidate recomputes anything: the routed accumulator is a
    Q24 integer sum, so its reduction is order-independent, and every other
    stage writes a disjoint range. That makes exact equality the right
    assertion rather than a tolerance -- a tolerance would hide precisely the
    reordering a missing readiness edge would cause.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    # Both legs select their schedule explicitly. Leaving either to the default
    # is what makes an A/B silently one-sided the moment the default moves --
    # and it has moved: this leg was the default before promotion.
    with barrier_schedule():
        _synchronize_ranks(workspace)
        production = _decode(workspace, weights, hidden).clone()
        torch.cuda.synchronize(device)
    assert_decode_close(production, expected)

    with dependency_local_schedule():
        _synchronize_ranks(workspace)
        candidate = _decode(workspace, weights, hidden).clone()
        torch.cuda.synchronize(device)

    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(candidate, expected)
    assert torch.equal(candidate, production), tokens
    assert_identical_across_ranks(candidate)


@pytest.mark.parametrize("mode", ADVERSARIAL_ROUTES)
@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_adversarial_routes_agree_with_the_production_schedule(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    mode: str,
    tokens: int,
) -> None:
    """The routed queues are what the candidate changed most, so skew them.

    ``disjoint`` gives every token its own sixteen experts, which makes the
    routed queues as long and as thin as they get -- one row per expert, and
    the expert-local readiness edge exercised once per unit. ``concentrated``
    collapses them to sixteen experts of ``tokens`` rows each, so 880 experts
    are empty and the compaction has to keep a worker off them. ``low`` and
    ``final`` put the occupied experts at the two ends of the offset table.
    """
    _, _, device = tp8_context
    plan = routing(mode, device, tokens, weights)
    skewed = with_routing(weights, plan)
    expected = decode_reference(plan.hidden, skewed)

    with barrier_schedule():
        _synchronize_ranks(workspace)
        production = _decode(workspace, skewed, plan.hidden).clone()
        torch.cuda.synchronize(device)

    with dependency_local_schedule():
        _synchronize_ranks(workspace)
        candidate = _decode(workspace, skewed, plan.hidden).clone()
        torch.cuda.synchronize(device)

    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(production, expected)
    assert_decode_close(candidate, expected)
    assert torch.equal(candidate, production), (mode, tokens)


def test_a_full_expert_batch_still_gates_per_expert(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """One expert holding all 128 rows is the widest single unit there is.

    A token's sixteen routes are distinct experts, so an expert collects at
    most ``active_tokens`` assignments and the concentrated routing at the
    maximum token count puts a full 128-row MMA batch behind one gate/up unit
    and its six ``situ`` arrivals. That is the longest any routed-down unit can
    be made to wait on a single edge, and the only shape where the six arrivals
    and the queue's one claim are farthest apart.
    """
    _, _, device = tp8_context
    plan = routing("concentrated", device, MAX_TOKENS, weights)
    skewed = with_routing(weights, plan)
    expected = decode_reference(plan.hidden, skewed)

    with dependency_local_schedule():
        _synchronize_ranks(workspace)
        candidate = _decode(workspace, skewed, plan.hidden).clone()
        torch.cuda.synchronize(device)

    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(candidate, expected)
    assert_identical_across_ranks(candidate)


def test_a_poisoned_workspace_does_not_survive_a_candidate_launch(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """The candidate re-establishes every data region it reads, as production
    does.

    It has one barrier to do it behind rather than five, so this is the test
    that the clearing it kept is the clearing it needed: a region the launch
    trusted would come back as the poison value rather than as an answer.
    """
    _, _, device = tp8_context
    tokens = TENSOR_TOKENS
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    with dependency_local_schedule():
        poison_scratch(workspace.scratch)
        _synchronize_ranks(workspace)
        candidate = _decode(workspace, weights, hidden)
        torch.cuda.synchronize(device)

    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(candidate, expected)


# ---------------------------------------------------------------------------
# What the schedule costs and what it launches.
# ---------------------------------------------------------------------------


def test_the_candidate_is_one_launch_of_one_kernel_that_allocates_nothing(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A candidate that cost a second launch would not be comparable.

    The whole point of the measurement is that the production step is one
    launch, so the candidate has to be one launch too -- of its own kernel,
    with no allocation, on the caller's stream.
    """
    _, _, device = tp8_context
    tokens = TENSOR_TOKENS
    hidden = hidden_states(device, tokens)

    with dependency_local_schedule():
        # Warm up so lazy initialization is not what the profiler sees.
        _decode(workspace, weights, hidden)
        _synchronize_ranks(workspace)
        names = profiled_kernel_names(
            lambda: kimi_k3_decode(CONFIG, workspace, weights, hidden)
        )
        before = torch.cuda.memory_allocated(device)
        with recorded_allocator_events(device) as events:
            kimi_k3_decode(CONFIG, workspace, weights, hidden)

    launches = [name for name in names if CANDIDATE_KERNEL in name]
    assert len(launches) == 1, names
    assert not [name for name in names if PRODUCTION_KERNEL in name], names
    assert torch.cuda.memory_allocated(device) == before
    assert events == [], events


def test_the_candidate_takes_one_grid_barrier_where_production_takes_five(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """The barrier count is the thing being reduced, so it is measured directly.

    Every full-grid barrier advances ``kGridGeneration`` exactly once, so the
    counter a launch leaves behind is the number of times its 148 CTAs
    rendezvoused. Production spends five; the candidate spends the one that
    publishes its cleared counters and nothing else. A profiled launch of
    either spends one more, for the band it zeroes.
    """
    _, _, device = tp8_context
    tokens = TENSOR_TOKENS
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    with barrier_schedule():
        _decode(workspace, weights, hidden)
        _synchronize_ranks(workspace)

        _phase(workspace.scratch)[GRID_GENERATION].zero_()
        _synchronize_ranks(workspace)
        production = _decode(workspace, weights, hidden).clone()
        torch.cuda.synchronize(device)
        assert int(_phase(workspace.scratch)[GRID_GENERATION].item()) == 5

    with dependency_local_schedule():
        _decode(workspace, weights, hidden)
        _synchronize_ranks(workspace)
        _phase(workspace.scratch)[GRID_GENERATION].zero_()
        _synchronize_ranks(workspace)
        candidate = _decode(workspace, weights, hidden).clone()
        torch.cuda.synchronize(device)
        barriers = int(_phase(workspace.scratch)[GRID_GENERATION].item())

    assert barriers == 1
    assert_decode_close(production, expected)
    assert_decode_close(candidate, expected)
    assert torch.equal(candidate, production)


def test_the_candidate_grid_is_proven_resident_before_it_is_launched(
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Its deadlock argument rests on co-residency, so co-residency is measured.

    A CTA that is not resident cannot take a ticket, and a queue whose tickets
    are not all taken cannot bound a consumer's wait -- so a grid that only
    partly fits does not run slowly, it hangs. The candidate is a different
    compiled function from the production kernel and is therefore measured in
    its own right rather than assumed to inherit its occupancy.
    """
    _, _, device = tp8_context
    for tensor_path in (False, True):
        blocks = _C._kimi_k3_decode_schedule_resident_blocks_per_sm(
            tensor_path
        )
        assert blocks == 1, tensor_path
    assert (
        torch.cuda.get_device_properties(device).multi_processor_count
        >= PERSISTENT_CTAS
    )


# ---------------------------------------------------------------------------
# The queues and the edges the harness reads back.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", SCHEDULE_TOKENS)
def test_the_queue_lengths_cover_every_logical_task_of_the_step(
    tokens: int,
) -> None:
    """Seven queues, and between them the same work production's five phases do.

    The two schedules cut the step differently -- the candidate's assignment
    compaction is a queue of its own and its quantization is folded into the
    projection units -- but the union has to be the same tasks, or the candidate
    is not computing the step. The bound on the routed queues is the one that
    matters: a queue longer than the compacted expert table behind it would
    index that table out of bounds.
    """
    units = _C._kimi_k3_decode_schedule_queue_units(tokens)
    plan = _C._kimi_k3_decode_task_plan(tokens)
    (
        source,
        activation,
        assignment,
        gate_up,
        shared_down,
        routed_down,
        publish,
    ) = units
    tensor_path = tokens > 8
    experts = min(16 * tokens, 896)

    assert len(units) == 7
    assert source == plan[0]
    assert activation == (6 if tensor_path else 0)
    assert assignment == 1
    assert gate_up == experts == plan[1]
    assert shared_down == (56 if tensor_path else 112)
    assert routed_down == 7 * experts
    assert activation + shared_down + routed_down == plan[2]
    assert publish == PERSISTENT_CTAS

    longest_ticket, largest_arrival, bound = (
        _C._kimi_k3_decode_schedule_counter_bounds()
    )
    assert max(units) <= largest_arrival
    assert largest_arrival < longest_ticket < bound == 2**31 - 1


def test_every_edge_points_backward_and_owns_a_cleared_counter() -> None:
    """The table the kernel is built from is the table the tests read.

    The header static-asserts both properties, but the assertions only cover
    the table as the compiler saw it. Reading it back through the binding is
    what checks that the table the shipped extension carries is the same one.
    """
    edges = _schedule_edges()
    queues = _C._kimi_k3_decode_schedule_queues()
    tail = len(queues)

    assert len(edges) == 10
    assert len(queues) == 7
    assert len({edge.name for edge in edges}) == 10
    assert len({edge.code for edge in edges}) == 10

    for edge in edges:
        assert 0 <= edge.producer < edge.consumer <= tail, edge
        assert edge.code > 0, edge
        if edge.space == IN_REGION:
            assert 7 <= edge.counter < 21, edge
        else:
            # The one edge whose counter is not in the appended region: the
            # fused engine publishes routed-down readiness into the compacted
            # assignment counts, per expert, so the counter field is the phase
            # slot the production wait on the same counter reports.
            assert edge.space == IN_EXPERT_COUNTS, edge
            assert 0 <= edge.counter < DIAGNOSTIC_BASE, edge

    codes = {name: code for name, code, _, _ in
             (tuple(site) for site in _C._kimi_k3_timeout_sites())}
    for edge in edges:
        assert edge.code in codes.values(), edge


def test_the_shipped_edge_table_carries_the_scope_and_target_waits_read(
) -> None:
    """The three fields a wrong value in would not fail any other test.

    The counter and the code are checked by the trap injection, and the two
    queue fields by the header's own ``static_assert``. The acquire scope, the
    target kind, and the static target are different: a cross-rank edge
    acquired at device scope would still pass every test on a machine that
    happened to order those writes anyway, and a static target one too low
    would let a consumer start on a partly written producer only under a
    routing skew a fixed suite need not hit.

    They are read from the table at compile time, so reading the table back out
    of the shipped extension is what checks that the values the kernel was
    built against are the values the DAG declares.
    """
    edges = _schedule_edges()

    for edge in edges:
        crosses_ranks = edge.consumer == len(
            _C._kimi_k3_decode_schedule_queues()
        )
        expected = SYSTEM_SCOPE if crosses_ranks else DEVICE_SCOPE
        assert edge.scope == expected, edge
        if edge.target_kind == STATIC_TARGET:
            assert edge.static_target > 0, edge
        else:
            assert edge.target_kind == DYNAMIC_TARGET, edge
            assert edge.static_target == -1, edge

    by_name = {edge.name: edge for edge in edges}
    # The two static targets that are counts of real work: the fused engine's
    # six `situ` arrivals per expert, and one publish unit per resident CTA.
    assert by_name["routed_down_gate_up"].static_target == 6
    assert by_name["tail_publish"].static_target == 148
    assert by_name["shared_activation_pair"].static_target == 2
    assert by_name["gate_up_assignment"].static_target == 1

    # Only the two indexed edges are indexed, and their diagnostic slot moves
    # with the unit while nobody else's does.
    indexed = {edge.name for edge in edges if edge.counter_indexed}
    assert indexed == {"shared_activation_pair", "routed_down_gate_up"}

    base = _C._kimi_k3_decode_schedule_edge_diagnostics(0)
    moved = _C._kimi_k3_decode_schedule_edge_diagnostics(3)
    for edge, first, later in zip(edges, base, moved, strict=True):
        if edge.space == IN_REGION:
            assert first == DIAGNOSTIC_BASE + edge.counter, edge
            step = 3 if edge.counter_indexed else 0
        else:
            # Outside the region: the phase slot, and the unit does not enter
            # it because the production wait on the same counter reports it.
            assert first == edge.counter, edge
            step = 0
        assert later == first + step, edge


def test_a_profiled_candidate_reports_where_its_readiness_waits_went(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """The per-edge counters are the instrument the A/B verdict is read from.

    A candidate that were merely faster would say nothing about why. The
    counters give one accumulated wait and one longest wait per edge, plus one
    makespan per queue, so a regression can be attributed to the edge that
    caused it. They are written only by a profiled launch: the band is
    poisoned before the measured one, and a counter that comes back holding the
    poison is one the launch never wrote.
    """
    _, _, device = tp8_context
    tokens = TENSOR_TOKENS
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    with dependency_local_schedule():
        _decode(workspace, weights, hidden)
        _synchronize_ranks(workspace)

        # Unprofiled: the band is handed back exactly as it was given.
        poison = 1 << 40
        _schedule_band(workspace).fill_(1)
        _synchronize_ranks(workspace)
        _decode(workspace, weights, hidden)
        torch.cuda.synchronize(device)
        waits, edge_makespans, queue_makespans = _schedule_clocks(workspace)
        assert min(waits.values()) > poison, waits

        os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
        _C._kimi_k3_decode_set_phase_profile(True)
        try:
            _synchronize_ranks(workspace)
            profiled = _decode(workspace, weights, hidden).clone()
            torch.cuda.synchronize(device)
            waits, edge_makespans, queue_makespans = _schedule_clocks(
                workspace
            )
        finally:
            _C._kimi_k3_decode_set_phase_profile(False)

    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(profiled, expected)

    names = {edge.name for edge in _schedule_edges()}
    assert set(waits) == set(edge_makespans) == names
    assert set(queue_makespans) == set(
        _C._kimi_k3_decode_schedule_queues()
    )
    # The poison is gone from every counter, so the launch reported itself.
    for name, cycles in {**waits, **edge_makespans}.items():
        assert 0 <= cycles < poison, (name, cycles)
    # Every queue is drained by some CTA, so every makespan is a real interval,
    # and they rise with the scan order because a later queue is drained later.
    order = _C._kimi_k3_decode_schedule_queues()
    drained = [queue_makespans[name] for name in order]
    assert min(drained) > 0, queue_makespans
    assert drained == sorted(drained), queue_makespans
    # The longest single wait on an edge cannot exceed the summed wait on it.
    for name in names:
        assert edge_makespans[name] <= waits[name], name


# ---------------------------------------------------------------------------
# A reused workspace, replayed.
# ---------------------------------------------------------------------------


def test_alternating_capacity_paths_replay_a_thousand_times(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Two captured graphs, one workspace, and nothing carried between them.

    The candidate has one barrier to reset its state behind, so this is the
    test that its counters really are self-restoring: a thousand alternating
    replays is a thousand crossings of every queue ticket and every readiness
    arrival, on both capacity paths, out of the same appended region. Capturing
    both graphs also pins that the selection is baked in at capture rather than
    read at replay.
    """
    _, _, device = tp8_context
    core = hidden_states(device, CORE_TOKENS)
    tensor = hidden_states(device, TENSOR_TOKENS)
    core_expected = decode_reference(core, weights)
    tensor_expected = decode_reference(tensor, weights)

    with dependency_local_schedule():
        # Warm up both paths outside capture, so neither graph records a
        # runtime API call or an allocation.
        _decode(workspace, weights, core)
        _decode(workspace, weights, tensor)
        _synchronize_ranks(workspace)

        core_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(core_graph):
            kimi_k3_decode(CONFIG, workspace, weights, core)
        _synchronize_ranks(workspace)
        tensor_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(tensor_graph):
            kimi_k3_decode(CONFIG, workspace, weights, tensor)
        _synchronize_ranks(workspace)

        for _ in range(500):
            core_graph.replay()
            tensor_graph.replay()
        torch.cuda.synchronize(device)
        last_tensor = workspace.output_mailbox.view(MAX_TOKENS, HIDDEN)[
            :TENSOR_TOKENS
        ].clone()

        core_graph.replay()
        torch.cuda.synchronize(device)
        last_core = workspace.output_mailbox.view(MAX_TOKENS, HIDDEN)[
            :CORE_TOKENS
        ].clone()

    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(last_tensor, tensor_expected)
    assert_decode_close(last_core, core_expected)
    assert_identical_across_ranks(last_core)


def test_the_candidate_is_correct_across_the_unsigned_generation_wrap(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """The one barrier it kept still rides a wrap-safe generation.

    The candidate's own counters need no wrap arithmetic -- it zeroes them --
    but the barrier that publishes those zeros does, and so does the cross-rank
    serial. Both are parked short of the wrap and pushed over it by this
    launch, which a naive ordered comparison could not survive. One barrier per
    launch also means the generation lands one past the wrap rather than two.
    """
    _, _, device = tp8_context
    tokens = 12
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    with dependency_local_schedule():
        _prime_barrier_serial(workspace, UINT32_MAX - 2)
        _phase(workspace.scratch)[GRID_GENERATION].fill_(
            _as_int32(UINT32_MAX)
        )
        _synchronize_ranks(workspace)
        candidate = _decode(workspace, weights, hidden)
        torch.cuda.synchronize(device)
        generation = int(_phase(workspace.scratch)[GRID_GENERATION].item())

    assert_decode_close(candidate, expected)
    assert_identical_across_ranks(candidate)
    # One barrier from the last representable value lands on zero.
    assert generation == 0


def test_random_jitter_between_launches_does_not_change_the_answer(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Removing the barriers is exactly what makes arrival order observable.

    Under the production schedule every CTA leaves a phase together, so a
    delayed rank or a delayed CTA cannot change which unit reads which
    intermediate. Under the candidate it can, so the answer has to be pinned
    against deliberately skewed arrival: each rank sleeps a different random
    amount before each launch, and every launch is required to reproduce the
    unskewed one exactly.
    """
    rank, _, device = tp8_context
    tokens = TENSOR_TOKENS
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    with dependency_local_schedule():
        _synchronize_ranks(workspace)
        baseline = _decode(workspace, weights, hidden).clone()
        torch.cuda.synchronize(device)
        assert_decode_close(baseline, expected)

        jitter = random.Random(9_001 + rank)
        for step in range(8):
            _synchronize_ranks(workspace)
            # Sleeping on the device rather than the host keeps the skew inside
            # the window the launches overlap in.
            torch.cuda._sleep(jitter.randrange(0, 2_000_000))
            actual = _decode(workspace, weights, hidden)
            torch.cuda.synchronize(device)
            assert_identical_across_ranks(actual)
            assert torch.equal(actual, baseline), step

    assert int(workspace.error_flag.item()) == 0


def test_the_dependency_local_schedule_is_what_a_decode_step_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decode step runs it, and the barrier schedule is benchmark-only.

    The default is the promoted schedule, and reaching the other one is guarded
    exactly like the grid override, the phase clocks and the gate/up engine. The
    reason is the adaptive integration rather than the promotion: the barrier
    schedule launches `kimi_k3_decode_persistent_kernel`, which is compiled
    against the resident two-stage ring, so an unguarded write to this switch
    routes a public decode through both retired paths at once.

    So the guard is read on every query rather than only on the write. A process
    that selected the barrier schedule and then dropped the variable is back on
    the dependency-local schedule, which makes the stored value irrelevant to
    production rather than merely usually-correct.
    """
    monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", raising=False)
    assert _C._kimi_k3_decode_dependency_schedule(), (
        "a process that has set nothing must be on the promoted schedule"
    )
    with pytest.raises(RuntimeError, match="benchmark-only"):
        _C._kimi_k3_decode_set_dependency_schedule(False)

    monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
    previous = _C._kimi_k3_decode_dependency_schedule()
    try:
        _C._kimi_k3_decode_set_dependency_schedule(False)
        assert not _C._kimi_k3_decode_dependency_schedule()

        # The write stands and the guard is what hides it: dropping the variable
        # while the storage still says "barrier" reads back as the promoted
        # schedule, and setting it again reveals the value that was never lost.
        monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING")
        assert _C._kimi_k3_decode_dependency_schedule()
        monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
        assert not _C._kimi_k3_decode_dependency_schedule()

        _C._kimi_k3_decode_set_dependency_schedule(True)
        assert _C._kimi_k3_decode_dependency_schedule()
    finally:
        _C._kimi_k3_decode_set_dependency_schedule(previous)


def test_an_unguarded_step_cannot_reach_the_barrier_schedule_once_set(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak this closes, checked on the device rather than on the reader.

    A benchmark process selects the barrier schedule and then drops the guard --
    which is what happens the moment a harness returns, or a `with` block exits,
    or a captured graph is replayed by something else. What the next public
    decode must launch is the dependency-local kernel with production's engine,
    and it must launch nothing else. Asking the reader is not enough here: the
    reader is the thing under test, so the profiler is what answers.

    The stored value is deliberately left saying "barrier" for the unguarded
    launch, because a test that put it back first would be checking the manager
    rather than the guard.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, TENSOR_TOKENS)
    expected = decode_reference(hidden, weights)

    monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
    previous = _C._kimi_k3_decode_dependency_schedule()
    try:
        _C._kimi_k3_decode_set_dependency_schedule(False)
        assert not _C._kimi_k3_decode_dependency_schedule()

        monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING")
        # Warm up under the same conditions as the profiled launch, so lazy
        # initialization is not what the profiler sees.
        _decode(workspace, weights, hidden)
        _synchronize_ranks(workspace)
        names = profiled_kernel_names(
            lambda: kimi_k3_decode(CONFIG, workspace, weights, hidden)
        )
        # Exactly one launch, of the dependency-local schedule, compiled with
        # production's gate/up engine -- the barrier kernel's absence is checked
        # by name too, because it is the one this switch could have reached.
        assert_one_production_launch(names)
        assert not [name for name in names if PRODUCTION_KERNEL in name], names
        assert_decode_close(_decode(workspace, weights, hidden), expected)
    finally:
        monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
        _C._kimi_k3_decode_set_dependency_schedule(previous)
