"""Source contracts for the decode schedule's release scopes and clock band.

Two properties that are neither the safety argument nor a consequence of
promotion. A release that leaves this rank orders nothing for anybody unless it
is system-scope, and a rank reading a stale peer line would be silent -- so the
scope of every cross-rank release is pinned here. And the profile band the
schedule laps into has to be laid out the way the header says, because a
measurement read out of the wrong slot is worse than no measurement.

The safety argument is in ``test_kimi_k3_dependency_schedule_source.py``.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from . import kimi_k3_decode_sources as decode_sources

_CANDIDATE_KERNEL = "void kimi_k3_decode_dependency_local_kernel("
_PRODUCTION_KERNEL = "void kimi_k3_decode_persistent_kernel("


class Edge(NamedTuple):
    """One row of ``kScheduleEdges``, mirrored field for field.

    All ten fields, not a summary: the whole point of the table is that the
    runtime derives its wait from it, so a mirror that dropped the scope or the
    target would leave exactly the fields that cannot fail a device test
    unchecked. ``consumer`` is ``None`` for the tail, which every queue
    precedes.
    """

    name: str
    consumer: str | None
    producer: str
    counter: str
    code: str
    space: str
    scope: str
    target_kind: str
    static_target: str
    counter_indexed: str


_IN_REGION = "kScheduleCounterInRegion"
_IN_COUNTS = "kScheduleCounterInExpertCounts"
_DEVICE = "kScheduleScopeDevice"
_SYSTEM = "kScheduleScopeSystem"
_STATIC = "kScheduleTargetStatic"
_DYNAMIC = "kScheduleTargetDynamic"
_NO_TARGET = "kScheduleTargetSuppliedAtWait"

_EDGES = (
    Edge("shared_activation_pair", "kQueueSharedActivation", "kQueueSource",
         "kScheduleSharedPairBegin", "kErrorScheduleSharedPair",
         _IN_REGION, _DEVICE, _STATIC, "2", "true"),
    Edge("assignment_score_shards", "kQueueAssignment", "kQueueSource",
         "kScheduleScoreArrivals", "kErrorScheduleScoreShards",
         _IN_REGION, _DEVICE, _DYNAMIC, _NO_TARGET, "false"),
    Edge("gate_up_assignment", "kQueueRoutedGateUp", "kQueueAssignment",
         "kScheduleAssignmentArrivals", "kErrorScheduleAssignment",
         _IN_REGION, _DEVICE, _STATIC, "1", "false"),
    Edge("gate_up_latent", "kQueueRoutedGateUp", "kQueueSource",
         "kScheduleLatentArrivals", "kErrorScheduleLatent",
         _IN_REGION, _DEVICE, _DYNAMIC, _NO_TARGET, "false"),
    Edge("shared_down_activation", "kQueueSharedDown",
         "kQueueSharedActivation",
         "kScheduleActivationArrivals", "kErrorScheduleActivation",
         _IN_REGION, _DEVICE, _DYNAMIC, _NO_TARGET, "false"),
    Edge("shared_down_gate_up", "kQueueSharedDown", "kQueueSource",
         "kScheduleSharedGateUpArrivals", "kErrorScheduleSharedGateUp",
         _IN_REGION, _DEVICE, _DYNAMIC, _NO_TARGET, "false"),
    # The one counter that is not in the appended region: routed down's
    # per-expert readiness is published by the fused engine into the compacted
    # assignment counts, and reading it there is what keeps the edge local to
    # one expert.
    Edge("routed_down_gate_up", "kQueueRoutedDown", "kQueueRoutedGateUp",
         "kGateUpArrivals", "kErrorScheduleExpertGateUp",
         _IN_COUNTS, _DEVICE, _STATIC, "kScheduleExpertGateUpArrivals",
         "true"),
    Edge("publish_routed_down", "kQueuePublish", "kQueueRoutedDown",
         "kScheduleRoutedDownArrivals", "kErrorScheduleRoutedDown",
         _IN_REGION, _DEVICE, _DYNAMIC, _NO_TARGET, "false"),
    Edge("tail_publish", None, "kQueuePublish",
         "kSchedulePublishArrivals", "kErrorSchedulePublish",
         _IN_REGION, _SYSTEM, _STATIC, "kSchedulePublishUnitsForTable",
         "false"),
    Edge("tail_shared_down", None, "kQueueSharedDown",
         "kScheduleSharedDownArrivals", "kErrorScheduleSharedDown",
         _IN_REGION, _SYSTEM, _DYNAMIC, _NO_TARGET, "false"),
)

# The enumerators the kernel takes its waits under, in table order.
_EDGE_IDS = tuple(
    "kEdge" + "".join(part.capitalize() for part in edge.name.split("_"))
    for edge in _EDGES
)

# The accumulated regions, as ``(name, containing region or None)``. Mirrored
# from ``PhaseClock`` and ``kPhaseClockParents`` so that a region added without
# a declared parent, or a child promoted to top-level, has to be written twice.
_PHASE_CLOCKS = (
    ("readiness_wait", None),
    ("router_score", None),
    ("latent_project", None),
    ("routed_queue", None),
    ("latent_quantize", None),
    ("assignment", None),
    ("publish", None),
    ("routed_gate_up", None),
    ("routed_gate_up_stage", "routed_gate_up"),
    ("routed_gate_up_mma", "routed_gate_up"),
    ("routed_gate_up_tma_issue", "routed_gate_up_stage"),
    ("routed_gate_up_tma_wait", "routed_gate_up_stage"),
    ("routed_gate_up_ring_full", "routed_gate_up_stage"),
    ("routed_gate_up_mma_issue", "routed_gate_up_mma"),
    ("routed_gate_up_activation", "routed_gate_up"),
    ("routed_gate_up_epilogue", "routed_gate_up"),
    ("routed_down", None),
    ("routed_down_stage", "routed_down"),
    ("routed_down_mma", "routed_down"),
    ("shared_experts", None),
    ("grid_barrier", None),
    ("tail", None),
)


def _source(name: str) -> str:
    return decode_sources.read(name)


def _function_body(text: str, signature: str) -> str:
    start = text.index(signature)
    depth = 0
    for offset in range(text.index("{", start), len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise AssertionError(f"{signature} is never closed")


def _candidate() -> str:
    return _function_body(_source("persistent_schedule.cuh"), _CANDIDATE_KERNEL)


def _production() -> str:
    return _function_body(_source("persistent_kernel.cuh"), _PRODUCTION_KERNEL)


# ---------------------------------------------------------------------------
# Scope.
# ---------------------------------------------------------------------------


def test_the_releases_that_leave_this_rank_are_system_scope() -> None:
    """A device-scope release orders this rank's writes for nobody else.

    The shared-down units and the publish units write the symmetric collective
    buffer, which the peers' tail roles read back through the fabric. Those two
    publish at system scope, and the coordinator -- the only CTA that tells the
    peers this rank is done -- acquires at system scope before it does, so its
    own cross-rank release is transitive over theirs. Everything that stays on
    this rank uses the cheaper device scope.
    """
    schedule = _source("persistent_schedule.cuh")
    body = _candidate()

    device_release = _function_body(schedule, "void publish_schedule_count(")
    system_release = _function_body(
        schedule, "void publish_schedule_count_system("
    )
    assert "__threadfence();" in device_release
    assert "__threadfence_system();" not in device_release
    assert "__threadfence_system();" in system_release

    device_wait = _function_body(schedule, "void wait_for_schedule_count(")
    system_wait = _function_body(
        schedule, "void wait_for_schedule_count_system("
    )
    assert "__threadfence();" in device_wait
    assert "__threadfence_system();" not in device_wait
    assert "__threadfence_system();" in system_wait

    published = re.findall(
        r"publish_schedule_count_system\(\s*(?:\n\s*)?scratch,\s*(k\w+)", body
    )
    assert set(published) == {
        "kScheduleSharedDownArrivals",
        "kSchedulePublishArrivals",
    }, published

    # Both of the coordinator's waits acquire at system scope, and both come
    # before it opens the cross-rank rendezvous. The scope is not written at the
    # wait site: it is the table's, and the table's rows for the two edges whose
    # consumer is the tail are the two that say system.
    coordinator = body[body.index("if (block < tail::kReduceBegin)"):]
    coordinator = coordinator[: coordinator.index("coordinate_ranks(")]
    crossing = [
        _EDGE_IDS[index]
        for index, edge in enumerate(_EDGES)
        if edge.consumer is None
    ]
    assert len(crossing) == 2
    assert re.findall(r"wait_edge<(k\w+)>\(", coordinator) == crossing
    for index, edge in enumerate(_EDGES):
        expected = _SYSTEM if edge.consumer is None else _DEVICE
        assert edge.scope == expected, edge
        del index


def test_the_tail_internals_are_the_production_tail_verbatim() -> None:
    """The candidate reschedules the step, it does not reimplement its end.

    Everything from ``coordinate_ranks`` onward is generation-tagged inside
    ``tail_sync.cuh`` and needs no barrier of its own, so the two barriers
    production takes in front of it are exactly what the publish and shared-down
    edges replace -- and nothing below that point may differ, or the candidate
    would be measuring a different tail.
    """
    body = _candidate()
    production = _production()

    def tail(text: str) -> str:
        return text[text.index("constexpr int shard_ctas ="):]

    candidate_tail = tail(body)
    production_tail = tail(production)
    for call in (
        "tail::coordinate_ranks(",
        "tail::latch_generation(",
        "tail::wait_for_generation(",
        "tail::reduce_rows(",
        "tail::publish_generation(",
        "tail::shard_tensor(",
        "tail::shard_core<kMaxCoreCapacity>(",
        "tail::drain_ranks(",
    ):
        assert candidate_tail.count(call) == production_tail.count(call), call

    # The only thing the candidate adds in front of the coordinator's first
    # cross-rank edge is the two waits that replace the two barriers.
    added = candidate_tail[: candidate_tail.index("tail::coordinate_ranks(")]
    assert added.count("wait_edge<") == 2
    assert "grid_barrier(" not in candidate_tail


# ---------------------------------------------------------------------------
# The profile band.
# ---------------------------------------------------------------------------


def test_no_stage_clock_can_include_the_wait_that_precedes_it() -> None:
    """A stage that charges itself its own idle cannot be compared with one
    that waited at a barrier instead.

    This is the whole comparison the candidate exists to make. Production
    leaves a phase at a full-grid barrier and charges that idle to
    ``grid_barrier``; the candidate waits inside the queue that needs the data
    and would charge the same idle to ``shared_experts``, ``routed_down``, or
    ``tail`` -- making every stage it moved a wait into look slower than the
    production stage it replaced, by exactly the idle the change was supposed
    to remove.

    So the reset is not left to the wait sites remembering it: ``wait_edge``
    takes the mark and resets it, which is why it takes the mark at all.

    The reset alone is not enough, though, and the first fix of this was wrong
    in the other direction: a mark that is only reset *discards* the waited
    cycles, so the twelve top-level bands no longer account for the launch and
    every share taken against their sum is a share of a denominator that is
    missing exactly the waiting the candidate is judged on. The wait is
    therefore lapped into ``readiness_wait`` -- which is top-level, which is
    where production's own readiness waits already go, and which is the band
    the per-edge counters refine rather than replace.

    And the band is lapped from the same reading the per-edge counter uses, not
    from the incoming mark. Lapping from the mark keeps it live across the spin
    beside that reading, and the extra live pair cost the M = 128 step half a
    percent for a few cycles of index arithmetic. So exactly one clock reading
    crosses the wait, and the band is the sum of the ten edge counters.
    """
    schedule = _source("persistent_schedule.cuh")
    wait = _function_body(schedule, "void wait_edge(")
    assert "unsigned long long *const mark" in wait
    assert "*mark = clocks.lap(kClockReadinessWait, started);" in wait
    assert "*mark = clocks.now();" not in wait
    # Nothing but `started` is read after the wait, so nothing but `started` has
    # to survive it. A lap from the mark would be the regression named above.
    assert "clocks.lap(kClockReadinessWait, *mark)" not in wait
    # One reading, taken before the wait, feeding both accumulators after it.
    readings = re.findall(r"= (?:clocks|edges)\.now\(\)", wait)
    assert readings == [], readings
    assert wait.count("clock64()") == 1, wait
    assert wait.index("clock64()") < wait.index("wait_for_schedule_count")
    assert (
        wait.index("edges.lap_edge(EDGE, started);")
        < wait.index("*mark = clocks.lap(")
    )

    # Every wait site hands its own stage mark in, so there is no wait whose
    # reset lands on a mark some other stage is measuring from.
    body = _candidate()
    calls = re.findall(r"wait_edge<k\w+>\((.*?)\);", body, re.DOTALL)
    assert len(calls) == 10
    for call in calls:
        assert "&mark" in call, call


def test_the_top_level_bands_are_disjoint_and_the_rest_are_their_children(
) -> None:
    """Summing the band overstates a launch by more than a third.

    ``routed_gate_up`` and its eight refinements measure the same cycles at
    three depths: ``stage`` contains the three copy clocks, ``mma`` contains
    the issue clock, and the parent contains all of them plus the activation
    gather and the epilogue. At M16 the leaves of that subtree alone came to
    82.0M cycles against a parent of 46.5M. A reader that added them up would
    report shares of a total that does not exist.

    The tree is therefore declared next to the clocks and shipped in the
    metadata, and the top-level regions are exactly the ones whose intervals
    are disjoint: each begins at a mark reset and ends at its own lap.
    """
    types = _source("types.cuh")
    names = re.findall(r'^\s*"(\w+)",$', types[
        types.index("kPhaseClockNames[] = {") : types.index(
            "kPhaseClockParents[] = {"
        )
    ], re.MULTILINE)
    assert names == [name for name, _ in _PHASE_CLOCKS], names

    declared = re.findall(
        r"^\s*(kPhaseClockTopLevel|kClock\w+),\s*//\s*(\w+)$",
        types[types.index("kPhaseClockParents[] = {"):],
        re.MULTILINE,
    )[: len(_PHASE_CLOCKS)]
    assert len(declared) == len(_PHASE_CLOCKS)
    for (parent, name), (expected_name, expected_parent) in zip(
        declared, _PHASE_CLOCKS, strict=True
    ):
        assert name == expected_name, (name, expected_name)
        if expected_parent is None:
            assert parent == "kPhaseClockTopLevel", name
        else:
            camel = "".join(
                part.capitalize() for part in expected_parent.split("_")
            )
            assert parent == f"kClock{camel}", (name, parent)

    assert "static_assert(phase_clock_parents_are_acyclic()" in types
    assert "static_assert(kPhaseClockCount == 22);" in types


def test_every_top_level_band_is_opened_and_closed_exactly_once_per_unit(
) -> None:
    """A band with no lap is unmeasured; a band with two starts overlaps.

    Six of the twelve top-level regions are the candidate's own: the queue
    claim, the two projection halves,     the assignment build, the publish write,
    and the tail. Q6's publish and Q2's assignment had no band at all before --
    their cycles were discarded by the next mark reset -- so a share of the
    total was simply missing from the report rather than attributed wrongly.

    The expected set is derived from the declared tree rather than listed, so a
    thirteenth top-level band cannot be added without being measured, and the
    accounting the report totals over cannot quietly lose a region.
    """
    schedule = _source("persistent_schedule.cuh")
    body = _candidate()
    lapped = set(re.findall(r"clocks\.lap\((kClock\w+)", body))
    # `readiness_wait` is lapped by `wait_edge`, once, for all ten edges.
    wait = _function_body(schedule, "void wait_edge(")
    lapped |= set(re.findall(r"clocks\.lap\((kClock\w+)", wait))

    def enumerator(name: str) -> str:
        return "kClock" + "".join(
            part.capitalize() for part in name.split("_")
        )

    assert lapped == {
        enumerator(name)
        for name, parent in _PHASE_CLOCKS
        if parent is None
    }, lapped
    assert len(lapped) == 12, lapped

    # The tail's band starts when the queues drain rather than wherever the
    # last queue left this CTA, so it does not vary with the CTA's role.
    drained = body.index('edges.mark_queue(kQueuePublish, launched);')
    reset = body.index("mark = clocks.now();", drained)
    assert reset < body.index("wait_edge<kEdgeTailPublish>")
    assert reset < body.index("tail::latch_generation(")
