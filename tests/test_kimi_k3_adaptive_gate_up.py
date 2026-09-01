"""TP8 GPU tests for the adaptive routed gate/up path production runs.

Production's gate/up engine is not one ring. It is a selector inside the expert
unit, and which of two rings an expert takes is a property of that expert's row
count:

* at most `COMPACT_ROWS` rows -- the **compact** ring, three K = 512 weight
  stages over an activation that holds all seven slabs at once by packing slab
  `s`'s eight-row operand four rows past slab `s - 1`'s, in a quarter of the
  bytes a resident activation needs;
* more than that -- the **slab-buffered** ring, three stages over two eight-row
  slots re-gathered per slab, with warps 1 to 7 producing while warp 0
  contracts.

Both replace one ring: the resident two-stage engine, which held the whole
expert's activation and had no room left for a third stage. That ring is still
compiled, as engine 3, because it is the numerical baseline every test here
measures against and the A/B arm the integration's numbers were taken with.

Three things need holding, and they are why this file is long.

**The packing.** It makes an inactive N column read the *next slab's* live rows
rather than zero, which is sound only because accumulator column `n` depends on
activation row `n` and nothing else, and the epilogue reads only the columns the
batch fills. That is an argument, and what stands in for it is equality with the
resident ring byte for byte at every row count on both sides of the threshold.
`test_each_row_count_across_the_compact_threshold_is_the_resident_ring` walks
one to eight rows per expert one at a time, so a packing off by a swizzle
phase, a base offset the tensor core reads differently than the gather wrote it,
or a threshold off by one shows as a differing `situ` byte rather than as a small
error somewhere in a mean.

**The switch between rings.** A CTA runs its experts back to back and may take a
different ring at each, so both rings' mbarriers are armed on its first unit and
each ring's parity is carried in its own shared words. A unit that left a stage
issued but not retired, or that recorded a parity the next unit reads as the
other ring's, shows on a route whose experts straddle the threshold -- which
`balanced` at 128 tokens is, and which the repeated-step case makes worse by
alternating occupancies on one workspace.

**The routes.** `ROUTES` is the matrix: every occupancy the schedule can see,
from one token (sixteen occupied experts and 879 empty) through a route that
gives every expert exactly one row, to one that puts sixteen passes on each of
sixteen. Each is checked against the resident ring byte for byte, and the ones
that exercise a distinct ring transition are also checked against the official
reference.

This file is also what the sanitizers are pointed at.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import pytest
import torch

from mok import _C
from mok.kimi_k3 import KIMI_K3_TOPK, KimiK3DecodeWeights, KimiK3DecodeWorkspace

from .kimi_k3_decode_support import (
    assert_decode_close,
    assert_identical_across_ranks,
    decode_reference,
    decode_step as _decode,
    hidden_states,
    routing,
    weights,  # noqa: F401
    with_routing,
    workspace,  # noqa: F401
)


# The ids `expert_mxfp4::fused_w13` compiles. Spelled rather than imported
# because the point of the guard below is that Python cannot reach the C++
# constant except through the binding being tested.
PRODUCTION_ENGINE = 2
RESIDENT_ENGINE = 3

# Rows the compact ring's packing admits, which is also the rows it spaces slabs
# by. Spelled here for the same reason the engine ids are.
COMPACT_ROWS = 4

# Each engine's ledger, as the compiled constants report it: dynamic bytes as
# launched, staging bytes before the allocator's alignment grain, weight stages,
# activation slabs held resident, live accumulators, and activation gathers per
# eight-row pass.
#
# Production's row is the union of its two arms rather than either arm's. The
# bytes are the compact ring's because they are the wider and so what the launch
# has to grant; the accumulator band is the slab-buffered ring's because that is
# what the tensor pool must keep clear whichever ring an expert takes. Neither
# arm has an id of its own any more -- the selector is the only way into either
# -- so the union is the only ledger production can be asked for.
LEDGERS = {
    PRODUCTION_ENGINE: (228352, 227328, 3, 7, 6, 1),
    RESIDENT_ENGINE: (216064, 215040, 2, 7, 1, 1),
}

#: The dynamic shared bytes one B300 SM grants a block that opts in, which is
#: the ceiling the wider of production's two arms has to sit under.
OPT_IN_SHARED_BYTES = 232_448

#: Every occupancy the adaptive selector can be handed, and which arm it takes.
#:
#: ``rows`` is the widest batch any expert holds, which is what the selector
#: reads. ``arms`` names what the route makes the CTA run: ``compact`` when
#: every expert is inside the threshold, ``slab`` when every expert is outside
#: it, and ``both`` when the route straddles it -- which is the case the arming
#: and the parity handoff live or die on.
ROUTES: tuple[tuple[str, int, str], ...] = (
    # One token: sixteen experts hold one row, 879 hold none.
    ("balanced", 1, "compact"),
    ("balanced", 16, "compact"),
    ("balanced", 32, "compact"),
    # At 128 tokens the mean expert holds two or three rows and the busiest
    # holds more than four, so one CTA runs both arms.
    ("balanced", 128, "both"),
    # Every expert exactly one row, so every gather reads a different token.
    ("disjoint", 32, "compact"),
    # Sixteen experts, `tokens` rows each: the row count is the token count.
    ("concentrated", 4, "compact"),
    ("concentrated", 32, "slab"),
    ("concentrated", 64, "slab"),
    ("concentrated", 128, "slab"),
    # The three placement cases: lowest, middle, and final expert ids.
    ("low", 16, "compact"),
    ("middle", 16, "compact"),
    ("final", 16, "compact"),
)

#: Routes also checked against the official reference rather than only against
#: the resident ring. One per distinct arm transition, plus the two placement
#: extremes, because the reference is an order of magnitude more expensive than
#: a second decode step and equality with the resident ring already carries the
#: numerical claim everywhere else.
REFERENCE_ROUTES: tuple[tuple[str, int, str], ...] = (
    ("balanced", 1, "compact"),
    ("balanced", 128, "both"),
    ("disjoint", 32, "compact"),
    ("concentrated", 128, "slab"),
    ("low", 16, "compact"),
    ("final", 16, "compact"),
)


def _route_id(route: tuple[str, int, str]) -> str:
    mode, tokens, arms = route
    return f"{mode}-{tokens}-{arms}"


@contextlib.contextmanager
def selected_engine(engine: int) -> Iterator[None]:
    """Select a measured arm for the duration of the block, and put it back.

    The selector is guarded by the benchmark variable, so the variable is set
    here too: a test that could reach an arm without it would be testing that
    the guard does not hold.
    """
    previous_flag = os.environ.get("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    try:
        _C._kimi_k3_decode_set_gate_up_engine(engine)
        yield
    finally:
        _C._kimi_k3_decode_set_gate_up_engine(PRODUCTION_ENGINE)
        if previous_flag is None:
            os.environ.pop("MOK_KIMI_K3_ENABLE_GRID_TUNING", None)
        else:
            os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = previous_flag


def test_production_reports_the_ledger_it_compiled() -> None:
    """Three stages, and the bytes production bought them with."""
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    try:
        ledgers = {
            engine: tuple(
                int(value)
                for value in _C._kimi_k3_decode_gate_up_engine_ledger(engine)
            )
            for engine in LEDGERS
        }
    finally:
        os.environ.pop("MOK_KIMI_K3_ENABLE_GRID_TUNING", None)

    assert ledgers == LEDGERS

    production = ledgers[PRODUCTION_ENGINE]
    resident = ledgers[RESIDENT_ENGINE]

    # The whole point of the integration: production's ring is one stage deeper
    # than the ring it replaced.
    assert production[2] == resident[2] + 1

    # It asks for the wider of the two arms it can take, at every launch, rather
    # than for whichever arm the first expert of a given step happens to want --
    # and that width is what has to fit under the opt-in ceiling.
    assert production[0] == 228_352
    assert production[0] <= OPT_IN_SHARED_BYTES

    # The wider arm is the compact one, and its staging is where the third stage
    # was paid for: it holds all seven activation slabs in 16,384 bytes instead
    # of the resident ring's 57,344 and moves all 28 activation scale tiles into
    # tensor memory, which buys the 67,584-byte stage with 12,288 to spare.
    assert production[3] == 7
    assert production[1] - resident[1] == 67_584 - (
        (57_344 - 16_384) + 14_336
    )

    # One gather per eight-row pass, because the compact arm gathers the whole
    # expert once; six live accumulators, because the slab-buffered arm keeps
    # one per interleaved task and the pool has to keep that band clear at every
    # unit rather than only at the units that take it.
    assert (production[4], production[5]) == (6, 1)

    # Both engines must still be one CTA per SM on both capacity paths, which is
    # what the launch's whole deadlock argument rests on. Production's own query
    # takes no engine, because a caller outside a benchmark has no engine to
    # name.
    for tensor_path in (False, True):
        assert _C._kimi_k3_decode_schedule_resident_blocks_per_sm(
            tensor_path
        ) == 1
    for engine in LEDGERS:
        for tensor_path in (False, True):
            assert _C._kimi_k3_decode_schedule_resident_blocks_per_sm(
                tensor_path, engine
            ) == 1


@pytest.mark.parametrize("route", ROUTES, ids=_route_id)
def test_production_is_the_resident_ring_byte_for_byte(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    route: tuple[str, int, str],
) -> None:
    """Equality, not closeness, on every occupancy the schedule can see.

    Both rings issue the identical sixteen block-scaled contractions per slab
    against the identical operands; only the order the 42 of them are walked in
    and where the operands sit differ, and every contraction lands in its own
    task's accumulator. So a `situ` byte that differs at all means one ring read
    a byte the other did not -- an operand offset off by a swizzle phase, a stage
    overwritten before its MMA retired, or a producer that refilled a slot the
    ring was still reading.

    Production runs first and without the guard, which is also the check that
    the adaptive path is what a caller that has set nothing gets.
    """
    mode, tokens, _ = route
    _, _, device = tp8_context
    plan = routing(mode, device, tokens, weights)
    routed = with_routing(weights, plan)

    production = _decode(workspace, routed, plan.hidden).clone()
    with selected_engine(RESIDENT_ENGINE):
        resident = _decode(workspace, routed, plan.hidden).clone()

    assert torch.equal(production, resident), (
        mode,
        tokens,
        int((production != resident).sum().item()),
    )
    assert_identical_across_ranks(production)


@pytest.mark.parametrize("route", REFERENCE_ROUTES, ids=_route_id)
def test_production_matches_the_official_reference(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    route: tuple[str, int, str],
) -> None:
    """Production's own numbers, against the oracle rather than against a peer.

    Byte equality with the resident ring would be satisfied by two rings that
    are wrong in the same way, and nothing about the compact packing makes that
    impossible -- both read the same descriptors. So the arms that matter are
    also held to the reference the rest of the suite is held to.
    """
    mode, tokens, _ = route
    _, _, device = tp8_context
    plan = routing(mode, device, tokens, weights)
    routed = with_routing(weights, plan)
    expected = decode_reference(plan.hidden, routed)
    actual = _decode(workspace, routed, plan.hidden)
    assert_decode_close(actual, expected)
    assert_identical_across_ranks(actual)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_each_row_count_across_the_compact_threshold_is_the_resident_ring(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    rows: int,
) -> None:
    """One row count per case, from one row per expert to a full eight.

    `concentrated` at `n` tokens gives each of sixteen experts exactly `n` rows,
    so this is the row count the selector reads, held one value at a time. Rows
    one to four take the packed operand -- where slab `s` starts `s * 4 * 128`
    bytes into the tile, at a swizzle phase that is zero only for even `s` -- and
    rows five to eight take the slab-buffered ring. Both must be the resident
    ring byte for byte, so the case that fails names the row count and the side
    of the threshold it is on.
    """
    _, _, device = tp8_context
    plan = routing("concentrated", device, rows, weights)
    routed = with_routing(weights, plan)
    expected = decode_reference(plan.hidden, routed)

    production = _decode(workspace, routed, plan.hidden).clone()
    with selected_engine(RESIDENT_ENGINE):
        resident = _decode(workspace, routed, plan.hidden).clone()

    assert_decode_close(production, expected)
    assert torch.equal(production, resident), (
        rows,
        "compact" if rows <= COMPACT_ROWS else "slab-buffered",
        int((production != resident).sum().item()),
    )


def test_production_survives_a_repeated_step_on_one_workspace(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Both rings' semaphore parity has to survive unit and step boundaries.

    Forty-two stream indices over three stages is fourteen whole laps, so a
    unit hands the next one a barrier at the parity it found -- and a step hands
    the next step the same. That is only true if every index that was issued was
    also retired, at every shape, so this runs the same workspace through
    alternating occupancies rather than one shape repeatedly.

    The alternation is also what switches arm between units and between steps:
    `balanced` at 16 is all compact, `concentrated` at 32 is all slab-buffered,
    and `balanced` at 128 is both inside one CTA. A parity one ring recorded and
    the other read would show here and nowhere cheaper.

    This case is too wide for racecheck, and deliberately stays that way. Its
    128-token laps are the expensive part and they are also the point of it, so
    the tool gets `test_production_alternates_arms_on_one_workspace` below,
    which makes the same parity claim on shapes racecheck can afford, and this
    one is left to the suite and to the two cheaper tools.
    """
    _, _, device = tp8_context
    plans = [
        routing(mode, device, tokens, weights)
        for mode, tokens in (
            ("balanced", 16),
            ("concentrated", 32),
            ("disjoint", 32),
            ("balanced", 128),
        )
    ]
    references = [
        decode_reference(plan.hidden, with_routing(weights, plan))
        for plan in plans
    ]
    for _ in range(3):
        for plan, expected in zip(plans, references, strict=True):
            actual = _decode(
                workspace, with_routing(weights, plan), plan.hidden
            )
            assert_decode_close(actual, expected)


def test_production_alternates_arms_on_one_workspace(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """The same claim as the case above, on shapes a sanitizer can afford.

    What has to hold is that a unit which ran one ring hands the next unit --
    and the next step -- semaphores at the parity it left them, when the two
    units ran *different* rings out of the same shared allocation. The case
    above makes that claim with 128-token laps, and racecheck cannot pay for
    them: it instruments every shared access in a kernel that is almost
    entirely shared traffic, and the eight-hour gate did not finish those laps.

    `concentrated` at `n` tokens gives each of sixteen experts exactly `n` rows,
    which is the cheapest way to put a whole launch on one arm and name which:
    four rows is the packed operand and eight is the slab-buffered ring. So
    alternating 4 and 8 over two laps on one workspace crosses the threshold in
    both directions three times, and every step is an eight-token step.

    It is a weaker claim than the case above in one specific way -- both arms
    are never live inside one CTA here, because every occupied expert in a
    `concentrated` route holds the same row count -- and that part is carried by
    the suite and by memcheck, whose selection keeps the 128-token laps because
    it completes in twelve minutes with them.
    """
    _, _, device = tp8_context
    plans = [
        routing("concentrated", device, tokens, weights)
        for tokens in (COMPACT_ROWS, 2 * COMPACT_ROWS)
    ]
    references = [
        decode_reference(plan.hidden, with_routing(weights, plan))
        for plan in plans
    ]
    for _ in range(2):
        for plan, expected in zip(plans, references, strict=True):
            actual = _decode(
                workspace, with_routing(weights, plan), plan.hidden
            )
            assert_decode_close(actual, expected)


def test_a_batch_wider_than_one_pass_is_the_resident_ring(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A pass is eight rows; an expert with more takes several of them.

    `concentrated` at 128 tokens puts every token's whole top-k on the same
    sixteen experts, so each of them takes a batch sixteen passes wide. That is
    where the slab-buffered ring is drained and refilled from empty mid-unit,
    and where the compact packing must *not* be used.
    """
    _, _, device = tp8_context
    plan = routing("concentrated", device, 128, weights)
    routed = with_routing(weights, plan)
    expected = decode_reference(plan.hidden, routed)

    production = _decode(workspace, routed, plan.hidden).clone()
    assert_decode_close(production, expected)
    with selected_engine(RESIDENT_ENGINE):
        resident = _decode(workspace, routed, plan.hidden).clone()
    assert torch.equal(production, resident)


def test_a_plain_step_still_launches_production_after_an_arm_ran(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Leaving the guarded block puts production back, not merely the flag."""
    _, _, device = tp8_context
    hidden = hidden_states(device, 16)
    expected = decode_reference(hidden, weights)
    with selected_engine(RESIDENT_ENGINE):
        _decode(workspace, weights, hidden)
    assert _C._kimi_k3_decode_gate_up_engine() == PRODUCTION_ENGINE
    assert_decode_close(_decode(workspace, weights, hidden), expected)


def test_the_selector_reads_the_row_count_and_nothing_else(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Which arm an expert takes may not depend on which experts they are.

    The threshold is `batch_rows`, so two routes with the same row count and
    different expert ids have to take the same arm and agree with the resident
    ring identically. `low`, `middle`, and `final` at 16 tokens put the same
    one-row-per-expert batch on the lowest, middle, and highest expert ids, and
    `KIMI_K3_TOPK` of them are occupied in each case.
    """
    _, _, device = tp8_context
    for mode in ("low", "middle", "final"):
        plan = routing(mode, device, KIMI_K3_TOPK, weights)
        routed = with_routing(weights, plan)
        production = _decode(workspace, routed, plan.hidden).clone()
        with selected_engine(RESIDENT_ENGINE):
            resident = _decode(workspace, routed, plan.hidden).clone()
        assert torch.equal(production, resident), mode
