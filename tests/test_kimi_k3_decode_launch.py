"""One launch of the Kimi K3 decode megakernel, on a grid proven to hold it.

The step is one kernel, not a sequence of them, and the grid it launches with
is one CTA per SM because the queues it hands work out through only drain if
every CTA of the launch is running. Neither is checkable from the output, so
both are read off the profiler and off the occupancy query here: which kernels
a step launches, how many, what the task plan decomposes into, and that the
residency the schedule's deadlock-freedom argument rests on is measured rather
than assumed.

What the step computes is in ``test_kimi_k3_decode.py``; what a reused
workspace carries is in ``test_kimi_k3_decode_workspace.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from mok import _C
from mok.kimi_k3 import (
    KIMI_K3_TOPK,
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
    kimi_k3_router_reference,
)

from . import kimi_k3_decode_support as decode_support
from .kimi_k3_decode_support import (
    ACTIVE_EXPERT_UNITS,
    CORE_TOKENS,
    DOWN_QUEUE,
    EXPERTS,
    GATE_UP_QUEUE,
    PERSISTENT_CTAS,
    PERSISTENT_THREADS,
    PRIVATE_STAGE_KERNELS,
    ROUTE_LATENT_QUEUE,
    TENSOR_TOKENS,
    _phase,
    _synchronize_ranks,
    assert_one_production_launch,
    barrier_schedule,
    decode_step as _decode,
    dependency_local_schedule,
    hidden_states,
    profiled_kernel_names,
    routing,
    schedule_queue_tickets,
    weights,  # noqa: F401
    with_routing,
    workspace,  # noqa: F401
)


# Fixed stage widths the task plan is built from, mirrored from the headers so
# that a silent retiling is caught here rather than only by a slow test.
CORE_PROJECTION_UNITS = 112       # skinny_gemm::kCoreCtas
TENSOR_PROJECTION_UNITS = 28      # skinny_gemm::kTensorCtas
SCORE_SHARDS = 8                  # router::kScoreShards
GATE_UP_UNITS = 1                 # persistent::kGateUpUnitsPerExpert
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
    # Named, not counted: the schedule carries the gate/up engine as a template
    # argument, so a substring of the schedule would accept a measured arm's
    # kernel as though it were production's.
    assert_one_production_launch(names)
    for private in PRIVATE_STAGE_KERNELS:
        assert all(private not in name for name in names), names


def test_one_launch_profiler_retries_an_empty_rank_trace(
    monkeypatch: pytest.MonkeyPatch,
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """One lost rank-local trace makes every rank repeat the same observation."""
    rank, _, device = tp8_context
    hidden = hidden_states(device, CORE_TOKENS)
    _decode(workspace, weights, hidden)
    _synchronize_ranks(workspace)

    original = decode_support._profiled_kernel_names_once
    attempts = 0

    def drop_rank_two_first_trace(
        call: Callable[[], object],
    ) -> list[str]:
        nonlocal attempts
        names = original(call)
        attempts += 1
        if rank == 2 and attempts == 1:
            return []
        return names

    monkeypatch.setattr(
        decode_support,
        "_profiled_kernel_names_once",
        drop_rank_two_first_trace,
    )
    names = decode_support.profiled_kernel_names(
        lambda: _decode(workspace, weights, hidden)
    )

    assert attempts == 2
    assert_one_production_launch(names)


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

    # And put the storage back, rather than leaving the guard to hide it. The
    # read above is the claim being made and it holds either way, but the value
    # is process-wide and `decode_step.cuh` reads it when it builds a launch --
    # so any later block that holds the guard for an unrelated reason would
    # launch on the smallest candidate grid. A barrier schedule that did exactly
    # that wedged its grid barrier and took the watchdog's trap.
    monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
    _C._kimi_k3_decode_set_benchmark_grid(production)
    monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING")


def test_selecting_a_schedule_does_not_inherit_a_left_over_grid(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holding the guard exposes every switch, not only the one being selected.

    The schedule selector is guarded now, and the reader consults the guard at
    launch, so a manager that wants the barrier schedule has to hold
    `MOK_KIMI_K3_ENABLE_GRID_TUNING` across the launch rather than only across the
    setter. That un-hides the grid override and the phase profile at the same
    time, and `decode_step.cuh` reads both when it builds the launch.

    So a dirty grid left behind by an earlier test -- which was harmless while
    every production launch ran with the guard unset -- becomes the grid the next
    selected launch runs on. The first one that did wedged its grid barrier and
    took the watchdog's trap, which is a device error rather than a failed
    assertion and takes the rest of the rank's session with it.

    The dirt is set up deliberately here rather than depended on from test order.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, CORE_TOKENS)
    smallest = tuple(_C._kimi_k3_decode_benchmark_grids())[0]
    assert smallest != PERSISTENT_CTAS

    monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
    _C._kimi_k3_decode_set_benchmark_grid(smallest)
    _C._kimi_k3_decode_set_phase_profile(True)
    monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    try:
        for manager in (dependency_local_schedule, barrier_schedule):
            with manager():
                # The pin is what is being checked, and it is checked from inside
                # the block where the guard is held -- outside it every read
                # returns production regardless and would prove nothing.
                assert _C._kimi_k3_decode_benchmark_grid() == PERSISTENT_CTAS
                assert not _C._kimi_k3_decode_phase_profile()
                _decode(workspace, weights, hidden)
                torch.cuda.synchronize(device)
            assert int(workspace.error_flag.item()) == 0

        # And the block put back what it found rather than what it wanted.
        monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
        assert _C._kimi_k3_decode_benchmark_grid() == smallest
        assert _C._kimi_k3_decode_phase_profile()
    finally:
        monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
        _C._kimi_k3_decode_set_benchmark_grid(PERSISTENT_CTAS)
        _C._kimi_k3_decode_set_phase_profile(False)
        monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING")


def test_the_baseline_gate_up_engine_cannot_leak_into_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resident ring is a measurement, not a switch production carries.

    It is reachable only behind the same guard the grid override is behind, and
    the guard is read on every query rather than only on the write -- so a
    process that selected it and then dropped the variable launches production's
    engine, not the one it last wrote.

    Production's engine is the adaptive selector, and its ledger is the union of
    the two rings it can take: the compact ring's bytes, because they are the
    wider and so what the launch grants, and the slab-buffered ring's six live
    accumulators, because that is the tensor-memory band the pool must keep
    clear whichever ring an expert takes. Neither ring has an id of its own, so
    the union is the only thing the selector can be asked for.
    """
    production_engine = 2
    baseline_engine = 3

    monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", raising=False)
    assert _C._kimi_k3_decode_gate_up_engine() == production_engine
    with pytest.raises(RuntimeError, match="benchmark-only"):
        _C._kimi_k3_decode_set_gate_up_engine(baseline_engine)

    monkeypatch.setenv("MOK_KIMI_K3_ENABLE_GRID_TUNING", "1")
    # 4 and 6 were the two rings while they were being measured against each
    # other. The selector runs both, so an id for either is gone -- and asking
    # for one now has to be refused rather than silently answered.
    for unknown in (0, 1, 4, 5, 6, 7):
        with pytest.raises(RuntimeError, match="unknown Kimi K3 gate/up"):
            _C._kimi_k3_decode_set_gate_up_engine(unknown)

    # The ledger is the compiled constants, so the third stage production buys
    # and what it spent to buy it are read off the kernel rather than restated
    # here.
    assert _C._kimi_k3_decode_gate_up_engine_ledger(production_engine) == (
        228352, 227328, 3, 7, 6, 1
    )
    assert _C._kimi_k3_decode_gate_up_engine_ledger(baseline_engine) == (
        216064, 215040, 2, 7, 1, 1
    )

    # Production's own launch is one CTA per SM at the wider of its two rings.
    for tensor_path in (False, True):
        assert _C._kimi_k3_decode_schedule_resident_blocks_per_sm(
            tensor_path
        ) == 1

    # A different engine is a different compiled function asking for a
    # different number of dynamic shared bytes, so its residency is its own
    # claim and the launch's deadlock argument rests on it.
    for tensor_path in (False, True):
        assert _C._kimi_k3_decode_schedule_resident_blocks_per_sm(
            tensor_path, baseline_engine
        ) == 1

    _C._kimi_k3_decode_set_gate_up_engine(baseline_engine)
    assert _C._kimi_k3_decode_gate_up_engine() == baseline_engine

    _C._kimi_k3_decode_set_gate_up_engine(production_engine)
    monkeypatch.delenv("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    assert _C._kimi_k3_decode_gate_up_engine() == production_engine


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
    ) + (
        2 * TENSOR_SHARED_GATE_UNITS
        if tensor_path
        else CORE_SHARED_GATE_UNITS
    )
    assert gate_up == experts * GATE_UP_UNITS
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
    """No CTA of either schedule sweeps all 896 experts looking for work.

    Both compact the occupied experts into a list and publish its length, and
    both size their routed queues from that length. The published length is a
    phase counter either way, so it is checked once; the ticket counters are
    not, because the two schedules keep their queues in different places. The
    barrier schedule's three sit in the phase region, and the dependency-local
    schedule's seven sit in its own appended region, disjoint from them by
    construction. Reading whichever belongs to the schedule that ran is what
    keeps the claim -- that a queue was drained by many CTAs racing for tickets
    rather than by one CTA walking the table -- true of the code that ships.
    """
    _, _, device = tp8_context
    plan = routing(mode, device, tokens, weights)
    routed = with_routing(weights, plan)
    distinct = _expected_distinct_experts(plan.hidden, routed)

    gate_up_units = distinct * GATE_UP_UNITS
    source_units = (
        tokens * SCORE_SHARDS
        + TENSOR_PROJECTION_UNITS
        + 2 * TENSOR_SHARED_GATE_UNITS
    )
    down_units = (
        ACTIVATION_UNITS
        + TENSOR_SHARED_DOWN_UNITS
        + distinct * GROUPED_DOWN_UNITS
    )

    def assert_drained(
        label: object, drained: int, units: int, claim_width: int
    ) -> None:
        # A batched queue stops one claim past its last unit for every CTA that
        # was refused; a singly-claimed one stops on its last unit.
        rounded = (units + claim_width - 1) // claim_width * claim_width
        assert units <= drained <= (
            rounded + claim_width * PERSISTENT_CTAS
        ), (label, drained, units)

    with barrier_schedule():
        _decode(workspace, routed, plan.hidden)
        torch.cuda.synchronize(device)
        counters = _phase(workspace.scratch)

        assert int(counters[ACTIVE_EXPERT_UNITS].item()) == distinct
        assert distinct < EXPERTS
        for counter, units, claim_width in (
            (GATE_UP_QUEUE, gate_up_units, 4),
            (DOWN_QUEUE, down_units, 4),
            (ROUTE_LATENT_QUEUE, source_units, 1),
        ):
            assert_drained(
                counter, int(counters[counter].item()), units, claim_width
            )

    with dependency_local_schedule():
        _decode(workspace, routed, plan.hidden)
        torch.cuda.synchronize(device)
        counters = _phase(workspace.scratch)

        assert int(counters[ACTIVE_EXPERT_UNITS].item()) == distinct
        tickets = schedule_queue_tickets(workspace.scratch)
        # types.cuh: ScheduleQueue, in the order every CTA scans them. Five of
        # the seven have a length that depends only on the shape, so those come
        # from the extension rather than from a second copy of the arithmetic.
        # The two routed queues are sized from the compacted expert table at
        # runtime, and the binding reports the worst case for the shape, so
        # those two are the ones derived from `distinct` here -- which is the
        # claim being made: the routed queues are as long as the experts the
        # router chose and no longer.
        declared = _C._kimi_k3_decode_schedule_queue_units(tokens)
        assert len(tickets) == len(declared) == 7
        for index, (label, units, claim_width) in enumerate((
            ("source", declared[0], 1),
            ("shared_activation", declared[1], 1),
            ("assignment", declared[2], 1),
            ("routed_gate_up", gate_up_units, 4),
            ("shared_down", declared[4], 1),
            ("routed_down", distinct * GROUPED_DOWN_UNITS, 4),
            ("publish", declared[6], 1),
        )):
            assert_drained(label, tickets[index], units, claim_width)
        assert gate_up_units <= declared[3]
        assert distinct * GROUPED_DOWN_UNITS <= declared[5]
