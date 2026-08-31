"""Every launch of a long candidate run is checked, not just the last one.

``test_alternating_capacity_paths_replay_a_thousand_times`` replays two captured
graphs five hundred times each and then compares the output mailbox. That is the
right test for "the counters are self-restoring across a thousand crossings",
and it is the wrong test for "no replay was ever wrong": replay 438 overwrites
replay 437, so a stale readiness edge that corrupted one launch in a hundred
would leave no trace in the buffer the assertion reads.

This file is the other half. Every launch is synchronized and verified against
the oracle before the next one starts, which is what makes a single bad replay
fatal rather than invisible. That costs a device synchronize per launch, so the
gate is untimed by construction and runs on its own rather than beside the
latency suite.

What rotates between launches is chosen to defeat the failure modes the
candidate introduced and production does not have. Under a barrier schedule the
arrival order of two CTAs cannot be observed; under this one it can, so the
things that can now leak are:

* **State from the previous launch.** Two workspaces are used in turn, and both
  are poisoned before every launch -- including the appended schedule counters,
  which the kernel is required to zero itself behind its one retained barrier.
  The poison is a distinct value each launch -- no two launches of the gate
  share a byte -- so a stale read reports which launch it came from rather than
  only that it happened.
* **State from a previous *shape*.** The token count changes every launch, so a
  queue length or a readiness target computed from the wrong shape shows up as a
  wrong answer rather than as an unused counter. The sequence crosses the
  core/tensor boundary in both directions.
* **A routing the expert-local edges were not exercised on.** The route rotates
  through the degenerate cases: every token on its own experts, every token on
  one expert, the lowest and highest expert IDs, and the balanced case. Three of
  the five leave all but sixteen experts empty, and ``concentrated`` at M128 is
  also the full-row case -- one expert with every token in its batch.
* **Arrival skew between ranks.** Each rank sleeps a different amount before
  each launch, so the launches genuinely overlap differently every time rather
  than settling into one interleaving.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist

from mok.kimi_k3 import (
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
    create_kimi_k3_decode_workspace,
)

from .kimi_k3_decode_support import (
    HIDDEN,
    MAX_TOKENS,
    SCRATCH_LAYOUT,
    TIMEOUT_PHASE,
    _phase,
    _synchronize_ranks,
    assert_decode_close,
    assert_identical_across_ranks,
    barrier_schedule,
    decode_reference,
    decode_step as _decode,
    dependency_local_schedule,
    routing,
    weights,  # noqa: F401
    with_routing,
)


# Every capacity bucket the decode contract has: `capacity_bucket` rounds the
# active token count up to a power of two, and 1, 2, 4 and 8 take the direct-
# register core path while 16 through 128 take the tcgen05 tensor path. The
# order alternates between the two paths, so every launch in a pass crosses the
# boundary rather than walking up one side of it -- only the wrap from the last
# shape of one pass to the first of the next stays on the tensor path.
#
# 2 and 4 were missing while the comment above them claimed every bucket, which
# is the kind of gap that makes a passing gate mean less than it says: they are
# the two shapes where a core-path unit covers a whole capacity in one pass of
# its register tile, so a readiness target computed per capacity rather than per
# unit would only show up there.
STRESS_TOKENS = (16, 1, 128, 2, 9, 4, 32, 8, 64)

# The routings that make the routed queues degenerate in every direction an
# expert-local edge could get wrong. ``disjoint`` needs one distinct top-16 per
# token, so it only exists for the shapes where that fits inside 128 experts.
STRESS_ROUTES = ("balanced", "concentrated", "low", "final", "disjoint")

# Passes over the shape-by-route grid. Forty launches a pass -- five routes on
# each of the nine shapes, less the five shapes above eight tokens where
# ``disjoint`` does not fit -- alternating between two workspaces, so four
# passes are 160 verified launches and 160 crossings of every queue ticket and
# readiness arrival on each workspace's own appended region.
STRESS_PASSES = 4

# The poison byte a launch is handed, and why no two launches of this gate share
# one.
#
# The failure message names the poison, which is what turns "something leaked"
# into "launch 137 leaked". A byte two launches shared narrows a stale read to
# two launches instead of one, and the oracle leg numbered its bytes
# ``0x40 + launch % 0x80`` before: that wraps at 128, so its last 32 launches
# were handed the same bytes as its first 32 -- exactly the launches a late
# failure is most likely to come from.
#
# A uint8 has room for both legs without overlap, 200 bytes of the 255 nonzero
# ones, so the two get disjoint bands and the byte says which leg it came from
# as well as which launch. Zero stays out of both: a counter the kernel failed
# to zero and a counter poisoned with zero read identically.
ORACLE_POISON_BASE = 0x01
EQUALITY_POISON_BASE = 0xB0


def _poison_byte(base: int, launch: int) -> int:
    """One launch's poison, never zero and never wrapped into another's."""
    byte = base + launch
    assert 0 < byte <= 0xFF, (base, launch, byte)
    return byte


@pytest.fixture(scope="module")
def workspace_pair(
    tp8_context: tuple[int, int, torch.device],
) -> Iterator[tuple[KimiK3DecodeWorkspace, KimiK3DecodeWorkspace]]:
    """Two workspaces, so consecutive launches do not share their state.

    The cached workspace the rest of the suite uses would make every launch
    inherit the previous launch's appended region, which is the one thing a
    replay-to-replay check most wants to vary: a counter that a launch failed
    to zero is invisible if the launch before it left the same value there.
    """
    _, _, device = tp8_context
    created = tuple(
        create_kimi_k3_decode_workspace(
            dist.group.WORLD, device=device, max_tokens=MAX_TOKENS
        )
        for _ in range(2)
    )
    try:
        yield created[0], created[1]
    finally:
        for one in created:
            _synchronize_ranks(one)
        dist.barrier()


def _poison(workspace: KimiK3DecodeWorkspace, value: int) -> None:
    """Fill everything a launch must re-establish, including its own counters.

    ``poison_scratch`` deliberately spares the appended schedule region, since
    for the rest of the suite a launch is entitled to trust the state it was
    handed. This gate is the one place that is not true: the candidate clears
    its own queue tickets and readiness arrivals behind its single retained
    barrier, and *that* is the claim under test, so the region goes in.

    The phase counters stay out. They carry the wrap-safe generation the
    barrier and the cross-rank serial ride on, which is handed from one launch
    to the next by design and is covered by its own wrap test.
    """
    byte = value & 0xFF
    for name, (offset, size) in SCRATCH_LAYOUT.items():
        if name in {"phase", "total_bytes"}:
            continue
        workspace.scratch[offset:offset + size].fill_(byte)
    workspace.output_mailbox.view(torch.uint8).fill_(byte)
    workspace.collective_buffer.view(torch.uint8).fill_(byte)


def _plan() -> list[tuple[int, int, str]]:
    """``(pass, tokens, route)`` for every launch, in the order they run.

    Deterministic, because a stress gate that fails on a schedule nobody can
    reproduce is a stress gate nobody fixes. The skew between ranks is the only
    random part, and it is seeded per rank.
    """
    plan = []
    for index in range(STRESS_PASSES):
        for tokens in STRESS_TOKENS:
            for route in STRESS_ROUTES:
                if route == "disjoint" and tokens * 16 > 128:
                    continue
                plan.append((index, tokens, route))
    return plan


# Both bands fit their leg and stay clear of each other, for the plan this gate
# actually runs rather than for the one it ran when the bases were picked. The
# equality leg walks the same grid once, so its length is one pass of the plan.
_ORACLE_LAUNCHES = len(_plan())
_EQUALITY_LAUNCHES = _ORACLE_LAUNCHES // STRESS_PASSES
assert ORACLE_POISON_BASE + _ORACLE_LAUNCHES <= EQUALITY_POISON_BASE, (
    ORACLE_POISON_BASE, _ORACLE_LAUNCHES, EQUALITY_POISON_BASE
)
assert EQUALITY_POISON_BASE + _EQUALITY_LAUNCHES <= 0x100, (
    EQUALITY_POISON_BASE, _EQUALITY_LAUNCHES
)


def test_every_candidate_launch_of_a_long_rotating_run_is_correct(
    workspace_pair: tuple[KimiK3DecodeWorkspace, KimiK3DecodeWorkspace],
    weights: KimiK3DecodeWeights,  # noqa: F811
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """One wrong launch anywhere in the run fails the gate.

    This is the only test that can catch a readiness edge which is *usually*
    satisfied. Everything else either checks one launch per configuration, in
    which case a one-in-fifty interleaving is unlikely to be the one sampled,
    or checks the last of many, in which case the wrong launch has already been
    overwritten. Here the failure names the launch: its index, its shape, its
    route, which of the two workspaces it ran on, and the poison it was handed.

    A launch is compared against the oracle rather than against a previous
    candidate launch, so a systematic error common to every launch is caught
    too, and its output is compared across ranks, so a rank that diverged is
    caught even where the oracle would tolerate both answers.
    """
    rank, _, device = tp8_context
    plan = _plan()
    assert len(plan) == 160, len(plan)

    # One reference per distinct configuration, computed once. The oracle is
    # expensive and the inputs are pinned, so recomputing it per launch would
    # make the gate slower without making it stronger.
    references: dict[tuple[int, str], tuple[torch.Tensor, object]] = {}
    for _, tokens, route in plan:
        key = (tokens, route)
        if key in references:
            continue
        plan_route = routing(route, device, tokens, weights)
        routed = with_routing(weights, plan_route)
        references[key] = (
            plan_route.hidden,
            (routed, decode_reference(plan_route.hidden, routed)),
        )

    skew = random.Random(20_260_831 + rank)

    with dependency_local_schedule():
        for launch, (index, tokens, route) in enumerate(plan):
            workspace = workspace_pair[launch % 2]
            hidden, (routed, expected) = references[(tokens, route)]
            # A distinct poison every launch, so a stale read says which launch
            # it leaked from rather than only that something leaked.
            poison = _poison_byte(ORACLE_POISON_BASE, launch)

            _synchronize_ranks(workspace)
            _poison(workspace, poison)
            _synchronize_ranks(workspace)
            # On the device rather than the host, so the skew lands inside the
            # window the eight launches overlap in.
            torch.cuda._sleep(skew.randrange(0, 1_500_000))

            where = (
                f"launch {launch} pass {index} tokens {tokens} route {route} "
                f"workspace {launch % 2} poison {poison:#04x}"
            )
            try:
                actual = _decode(
                    workspace, weights=routed, hidden=hidden
                ).clone()
                torch.cuda.synchronize(device)
                assert actual.shape == (tokens, HIDDEN)
                assert_decode_close(actual, expected)
                assert_identical_across_ranks(actual)
            except AssertionError as failure:
                slot = int(_phase(workspace.scratch)[TIMEOUT_PHASE].item())
                code = int(workspace.error_flag.item())
                raise AssertionError(
                    f"{where} (timeout code {code} at slot {slot}): {failure}"
                ) from failure


def test_the_candidate_and_production_agree_launch_for_launch_under_rotation(
    workspace_pair: tuple[KimiK3DecodeWorkspace, KimiK3DecodeWorkspace],
    weights: KimiK3DecodeWeights,  # noqa: F811
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """Interleaving the two schedules is what makes the equality exact.

    The oracle comparison above has to allow a tolerance, because the reference
    is computed in a different order and a different precision. Production does
    not: it runs the same stages over the same inputs, so the candidate must
    match it bit for bit, and interleaving them on the same poisoned workspace
    also checks that neither schedule leaves state the other trips over --
    which the A/B harness depends on and no single-schedule run can show.
    """
    rank, _, device = tp8_context
    skew = random.Random(770_113 + rank)
    plan = [
        (tokens, route)
        for tokens in STRESS_TOKENS
        for route in STRESS_ROUTES
        if not (route == "disjoint" and tokens * 16 > 128)
    ]

    for launch, (tokens, route) in enumerate(plan):
        workspace = workspace_pair[launch % 2]
        plan_route = routing(route, device, tokens, weights)
        routed = with_routing(weights, plan_route)
        poison = _poison_byte(EQUALITY_POISON_BASE, launch)

        where = (
            f"launch {launch} tokens {tokens} route {route} "
            f"poison {poison:#04x}"
        )

        # Each leg names its schedule at the launch rather than taking it from a
        # loop variable, which is what makes the two legs different even to a
        # reader -- and what a source contract can check. Relying on the default
        # for either is how this test went one-sided when the default moved.
        _synchronize_ranks(workspace)
        _poison(workspace, poison)
        _synchronize_ranks(workspace)
        torch.cuda._sleep(skew.randrange(0, 1_500_000))
        with barrier_schedule():
            barrier = _decode(
                workspace, weights=routed, hidden=plan_route.hidden
            ).clone()
        torch.cuda.synchronize(device)
        assert int(workspace.error_flag.item()) == 0, ("barrier", where)

        _synchronize_ranks(workspace)
        _poison(workspace, poison)
        _synchronize_ranks(workspace)
        torch.cuda._sleep(skew.randrange(0, 1_500_000))
        with dependency_local_schedule():
            candidate = _decode(
                workspace, weights=routed, hidden=plan_route.hidden
            ).clone()
        torch.cuda.synchronize(device)
        assert int(workspace.error_flag.item()) == 0, ("candidate", where)

        assert torch.equal(candidate, barrier), where
        assert_identical_across_ranks(candidate)
