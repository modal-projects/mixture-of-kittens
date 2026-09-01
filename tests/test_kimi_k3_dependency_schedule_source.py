"""Source contracts for what makes the Kimi K3 decode schedule deadlock free.

The schedule replaces four of the production kernel's five full-grid barriers
with seven topologically ordered task queues and ten bounded readiness edges.
Almost everything that makes that safe is a property of the source rather than
of a run: which order the queues are scanned in, which direction every edge
points, how many barriers are left, and that every wait is bounded and reports
a code no other wait reports. A run can only show that today's shapes happen to
work.

This file holds the safety argument. ``test_kimi_k3_dependency_schedule_scope.py``
holds what the schedule's release scopes and its profile band have to be, and
``test_kimi_k3_dependency_schedule_promotion.py`` holds what promoting it to
the default did to the rest of the tree. The device suite in
``test_kimi_k3_dependency_schedule.py`` covers what the launch computes.

Keeping them out of the device suite is also what lets them run on a machine
with no B300 attached.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from . import kimi_k3_decode_sources as decode_sources

_CANDIDATE_KERNEL = "void kimi_k3_decode_dependency_local_kernel("
_PRODUCTION_KERNEL = "void kimi_k3_decode_persistent_kernel("

# The queues, in the one order every CTA scans them. Mirrored from the
# ``ScheduleQueue`` enum rather than read out of it, so a reordering has to be
# made twice and cannot pass by editing only the enum.
_QUEUES = (
    "kQueueSource",
    "kQueueSharedActivation",
    "kQueueAssignment",
    "kQueueRoutedGateUp",
    "kQueueSharedDown",
    "kQueueRoutedDown",
    "kQueuePublish",
)


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


def _table_rows(text: str, name: str) -> list[list[str]]:
    """Split one brace-initialized constexpr table into its rows."""
    start = text.index(f"{name}[] = {{")
    depth = 0
    rows: list[list[str]] = []
    current = ""
    for offset in range(text.index("{", start), len(text)):
        character = text[offset]
        if character == "{":
            depth += 1
            if depth == 2:
                current = ""
            continue
        if character == "}":
            depth -= 1
            if depth == 1:
                rows.append(
                    [field.strip() for field in current.split(",")]
                )
            if depth == 0:
                return rows
            continue
        if depth == 2:
            current += character
    raise AssertionError(f"{name} is never closed")


# ---------------------------------------------------------------------------
# One forward scan.
# ---------------------------------------------------------------------------


def test_every_cta_scans_the_queues_once_in_the_declared_order() -> None:
    """The scan order is the deadlock argument, so it is not left to reading.

    A CTA leaves a queue only when that queue's ticket counter is exhausted,
    which means every unit of it is held by a co-resident CTA. That is only
    worth anything if the walk is a straight line: a queue visited twice, or
    visited out of the order the edges were checked against, would let a CTA
    block behind work that is itself blocked behind the CTA. So the claims in
    the kernel are required to be exactly the declared queues, once each, in
    declaration order.
    """
    body = _candidate()
    claimed = re.findall(
        r"claim_schedule_(?:unit|batch)\(\s*(?:\n\s*)?scratch,\s*(kQueue\w+)",
        body,
    )
    assert claimed == list(_QUEUES), claimed

    # The enum the claims are read against declares the same order, and the
    # loop that walks it -- the profile band -- indexes it the same way.
    types = _source("types.cuh")
    enum = types[types.index("enum ScheduleQueue"):]
    enum = enum[: enum.index("};")]
    declared = re.findall(r"^\s*(kQueue\w+)", enum, re.MULTILINE)
    assert declared == list(_QUEUES), declared
    assert "kScheduleQueueCount" in enum

    # Nothing loops back: the only backward jump a queue's own drain loop makes
    # is inside itself, and every `mark_queue` closes exactly one queue.
    marked = re.findall(r"mark_queue\(\s*(kQueue\w+)", body)
    assert marked == list(_QUEUES), marked


def test_the_dependency_table_only_points_backward() -> None:
    """Every edge names a strictly earlier queue, and the compiler checks it.

    This is the property that makes a wait bounded rather than circular, so it
    is a ``static_assert`` in the header and not a comment. The table is
    mirrored here so that adding an edge to the kernel without adding it to the
    table -- or adding a row that points sideways -- fails in two places.
    """
    types = _source("types.cuh")
    rows = _table_rows(types, "kScheduleEdges")
    order = {name: index for index, name in enumerate(_QUEUES)}
    tail = len(_QUEUES)

    assert len(rows) == len(_EDGES) == 10
    for row, edge in zip(rows, _EDGES, strict=True):
        expected = Edge(
            f'"{edge.name}"',
            "kScheduleQueueCount" if edge.consumer is None else edge.consumer,
            *edge[2:],
        )
        assert Edge(*row) == expected, row
        assert order[edge.producer] < order.get(row[1], tail), row

    assert "static_assert(schedule_edges_point_backward()" in types
    assert (
        "static_assert(schedule_edges_never_block_inside_a_queue()" in types
    )
    assert "static_assert(schedule_edge_codes_are_tabulated()" in types
    assert (
        "static_assert(schedule_edge_counters_are_cleared_each_launch()"
        in types
    )
    assert (
        "static_assert(schedule_tail_edges_acquire_at_system_scope()" in types
    )
    assert "static_assert(schedule_static_targets_are_positive()" in types


def test_no_queue_blocks_on_a_producer_in_its_own_queue() -> None:
    """Same-queue blocking is the one shape the ticket argument cannot save.

    Exhausted tickets prove a *claimed* unit is running, not a *finished* one.
    A CTA that blocked inside queue ``k`` on another unit of queue ``k`` could
    therefore be waiting on a CTA blocked the same way, and no amount of
    residency helps. The table is required to contain no such row, and the
    kernel is required to take every wait against a counter an earlier queue
    publishes.
    """
    rows = _table_rows(_source("types.cuh"), "kScheduleEdges")
    for row in rows:
        assert row[1] != row[2], row

    body = _candidate()
    # Q0 is the source queue: it publishes and never waits, which is what makes
    # the whole scan bottom out.
    source = body[body.index("kQueueSource") : body.index(
        "kQueueSharedActivation"
    )]
    assert "wait_edge<" not in source, source
    assert "wait_for_schedule_count" not in source, source
    assert "wait_for_count_at(" not in source, source
    # Four publications: the latent groups, the shared column pair, the shared
    # gate/up total, and the token's score shard.
    assert source.count("publish_schedule_count(") == 4


# ---------------------------------------------------------------------------
# Barriers.
# ---------------------------------------------------------------------------


def test_only_the_cleared_counters_still_cost_a_full_grid_barrier() -> None:
    """One barrier, and it is the one no readiness edge could replace.

    "These counters are zero" is a fact about all 148 CTAs at once, and a
    release/acquire pair between two of them cannot establish it: a CTA that
    took a ticket before the zeroing landed would have its claim erased. Every
    other barrier the production kernel takes separates a producer from a
    consumer, and those are exactly what the edges replace. So the candidate
    keeps one, and the second one in the body is the profiled band's, which an
    unprofiled launch does not take.
    """
    body = _candidate()
    production = _production()
    assert production.count("grid_barrier(") == 6
    assert body.count("grid_barrier(") == 2

    profiled = body[body.index("if (clocks.enabled())"):]
    profiled = profiled[: profiled.index("unsigned long long mark")]
    assert profiled.count("grid_barrier(") == 1

    # The retained barrier follows the clearing and precedes every claim.
    clear = body.index("schedule_cleared_counter(thread)")
    barrier = body.index("grid_barrier(", clear)
    first_claim = body.index("claim_schedule_unit(")
    assert clear < barrier < first_claim

    # And it is the last one: nothing after the first claim rendezvouses.
    assert "grid_barrier(" not in body[first_claim:]


def test_the_per_launch_counters_are_zeroed_before_the_retained_barrier(
) -> None:
    """Nothing in the appended region is generation-tagged, so it must start at
    zero.

    A queue ticket runs past its unit count and a readiness arrival is never
    cleared by its own last arriver, so neither is self-restoring the way the
    grid generation is. One CTA owns the clearing, so no other CTA can be
    mid-claim while it happens, and the barrier is what publishes it. The state
    that *is* never cleared -- the grid generation the barrier itself rides on
    -- is latched wrap-safely before any CTA can move it.
    """
    body = _candidate()
    schedule = _source("persistent_schedule.cuh")
    types = _source("types.cuh")

    clearing = body[body.index("// Stage 0") : body.index("// Q0,")]
    assert "if (block == 0 && thread < kScheduleClearedCounters)" in clearing
    assert "atomicExch(" in clearing
    assert "scratch.expert_counts[thread] = 0;" in clearing
    assert "scratch.routed_accumulator_fixed[index] = 0;" in clearing

    # Wrap-safe, never-cleared state: the barrier generation, latched before
    # any CTA of this launch can have advanced it.
    latch = body.index("latch_grid_phase(scratch, &latch_slot)")
    assert latch < body.index("grid_barrier(")
    assert "generation_advanced" not in schedule
    assert "using serial_sync::generation_advanced;" in _source(
        "persistent_sync.cuh"
    )

    # The cleared band starts at the first queue ticket and runs to the last
    # readiness arrival, so one thread per slot covers all of it.
    assert "inline constexpr int kScheduleQueueBegin = 0;" in types
    assert (
        "inline constexpr int kScheduleClearedCounters =\n"
        "    kSchedulePublishArrivals + 1;" in types
    )
    assert "static_assert(kScheduleClearedCounters == 21);" in types


# ---------------------------------------------------------------------------
# Bounded waits and unique diagnostics.
# ---------------------------------------------------------------------------


def test_every_schedule_wait_is_bounded_and_names_its_own_site() -> None:
    """A broken edge has to surface as a named trap, not as a hung device.

    Ten edges, ten codes, one wait site each. The codes matter more here than
    in the production kernel: production has three sites and the candidate has
    ten, several of them on counters that look alike, so the code is the only
    thing that says which edge did not arrive.
    """
    schedule = _source("persistent_schedule.cuh")
    body = _candidate()

    for helper in ("wait_for_schedule_count", "wait_for_schedule_count_system"):
        wait = _function_body(schedule, f"void {helper}(")
        assert "wait_timed_out(started, clock64())" in wait
        assert "record_timeout_and_trap(" in wait
        assert "error_code);" in wait

    # Every code is reached from a bounded wait rather than from a bare trap,
    # and the candidate defines no trap of its own.
    assert "trap;" not in schedule
    sites = re.findall(r"wait_edge<(k\w+)>\(", body)
    assert sites == sorted(sites, key=_EDGE_IDS.index)
    assert set(sites) == set(_EDGE_IDS)
    assert len(sites) == 10, sites

    # And no wait may be taken any other way, which is what makes the table the
    # single description rather than the preferred one.
    for bypass in (
        "wait_for_schedule_count(",
        "wait_for_schedule_count_system(",
        "wait_for_count_at(",
    ):
        assert bypass not in body, bypass


def test_every_wait_derives_its_whole_contract_from_the_edge_table() -> None:
    """A wait and its table row cannot disagree if there is only one of them.

    The counter, the arrival target, the acquire scope, the diagnostic slot,
    and the timeout code are five facts about an edge, and spelling them at the
    wait site as well as in the table gives fifty chances for the executed DAG
    to drift from the declared one. The scope field is the dangerous one: a
    cross-rank edge silently acquired at device scope makes the coordinator's
    own release non-transitive, and no device test on a machine that happens to
    order those writes anyway would fail.

    So ``wait_edge<Edge>`` reads all five out of ``kScheduleEdges[Edge]`` and
    the wait sites pass only what the table says is not in it: the unit, for
    the two indexed edges, and the target, for the six that depend on the
    launch's shape or on which path it took.
    """
    schedule = _source("persistent_schedule.cuh")
    wait = _function_body(schedule, "void wait_edge(")

    for field in (
        "kScheduleEdges[EDGE].counter",
        "kScheduleEdges[EDGE].error_code",
        "kScheduleEdges[EDGE].space",
        "kScheduleEdges[EDGE].scope",
        "kScheduleEdges[EDGE].static_target",
        "kScheduleEdges[EDGE].target_kind",
        "kScheduleEdges[EDGE].counter_indexed",
    ):
        assert field in wait, field
    assert "schedule_edge_diagnostic(EDGE, 0)" in wait
    assert 'static_assert(kScheduleEdges[EDGE].producer_queue' in wait
    # The scope is a compile-time branch, so a device-scope edge costs no
    # system fence and a system-scope edge cannot forget one.
    assert "if constexpr (kScope == kScheduleScopeSystem)" in wait

    # No wait site carries a counter, a code, or a fence of its own.
    body = _candidate()
    assert "kErrorSchedule" not in body
    assert "schedule_diagnostic(" not in body
    assert "__threadfence" not in body

    # Exactly the dynamic targets are supplied at their wait, and the table is
    # what says they must be.
    dynamic = [
        _EDGE_IDS[index]
        for index, edge in enumerate(_EDGES)
        if edge.target_kind == _DYNAMIC
    ]
    supplied = re.findall(r"wait_edge<(k\w+)>\([^;]*?,\s*\d+,\s*[^;]*?\);",
                          body, re.DOTALL)
    assert sorted(supplied) == sorted(dynamic), supplied


def test_every_wait_records_a_diagnostic_slot_no_phase_counter_can_forge(
) -> None:
    """One timeout word carries slots from two disjoint counter regions.

    ``record_timeout_and_trap`` writes the counter index into
    ``kPersistentTimeoutPhase``, and a schedule counter's own index would
    collide numerically with a phase slot there -- slot 7 is both the first
    shared-pair counter and the grid barrier's error family. Offsetting the
    schedule half by the phase region's width is what keeps every recorded
    number unambiguous without a second word.
    """
    types = _source("types.cuh")

    assert (
        "inline constexpr int kScheduleDiagnosticBase = NUM_PHASE_COUNTERS;"
        in types
    )
    # The two counter spaces number differently and one function knows both, so
    # the wait and the test predict the same slot from the same place.
    diagnostic = _function_body(types, "int schedule_edge_diagnostic(")
    assert "kScheduleCounterInExpertCounts" in diagnostic
    assert "schedule_diagnostic(" in diagnostic
    assert "counter_indexed" in diagnostic

    sites = _table_rows(types, "kTimeoutSites")
    schedule_sites = [row for row in sites if row[0].startswith('"schedule_')]
    assert len(schedule_sites) == 10
    assert len({row[1] for row in schedule_sites}) == 10
    assert len({tuple(row[2:]) for row in sites}) < len(sites)


def test_one_waiter_claims_the_record_and_publishes_the_code_last() -> None:
    """A code from one site beside a slot from another names a third site.

    The two words are written by whichever CTAs give up, and they all give up
    against the same fifteen-second budget, so several of them racing is the
    normal case rather than the exotic one. Two independent exchanges would
    then pair the last-arriving code with the last-arriving slot -- which need
    not be the same waiter's -- and send the reader to a wait that never timed
    out.

    Deciding the winner by compare-and-swapping the *code* was the first fix of
    this, and it was not enough: it made the pair one waiter's, but it published
    the code before the slot, so between those two writes a reader saw a live
    code beside whatever the last launch left in the slot -- and a loser was
    free to trap in that window and end the launch there. What the order has to
    be is slot, release, code; and a waiter that lost the claim has to wait for
    the code before it traps, because its trap ends the launch for the winner
    too.
    """
    publication = _source("timeout_publication.cuh")
    record = _function_body(publication, "void publish_and_trap(")

    # One claim word, cleared by the launch, and the claim is not the code.
    assert "atomicCAS_system(claim, kUnclaimed," in record
    assert "kTimeoutClaim" in _function_body(publication, "void clear_claim(")

    # System scope throughout. The probes place all three words in mapped host
    # memory, where a device-scope read-modify-write would not be atomic at all
    # and two waiters could both win the claim.
    assert "atomicCAS(" not in record.replace("atomicCAS_system(", "")
    assert "atomicExch(" not in record.replace("atomicExch_system(", "")

    slot = record.index("&scratch.phase[slot_index]")
    fence = record.index("__threadfence_system();", slot)
    code = record.index("atomicExch_system(code,", fence)
    assert slot < fence < code < record.index("trap;")

    # The loser's wait is bounded and is on the code, which is published last,
    # so it cannot end the launch while the pair is half written.
    loser = record[record.index("} else {"):]
    assert "load_relaxed_system(code) == 0u" in loser
    assert "wait_timed_out(" in loser
    assert loser.index("load_relaxed_system") < loser.index("trap;")

    # Both families publish through it rather than keeping an exchange of their
    # own, so there is one protocol and not three.
    for header, slot_name in (
        ("persistent_sync.cuh", "kPersistentTimeoutPhase"),
        ("tail_sync.cuh", "kTailTimeoutPhase"),
    ):
        wrapper = _function_body(
            _source(header), "void record_timeout_and_trap("
        )
        assert "timeout::publish_and_trap(" in wrapper, header
        assert slot_name in wrapper, header
        assert "atomicExch(" not in wrapper, header
        assert "atomicCAS(" not in wrapper, header
        assert "trap;" not in wrapper, header


def test_every_launch_that_can_trap_clears_its_own_claim_word() -> None:
    """A claim the launch did not clear can only be published once.

    The sentinel means "this launch has a record", so a launch that inherits a
    set claim word from an earlier one cannot publish at all: every waiter loses
    the claim, waits for a code that is already there from the previous launch,
    and traps reporting it. Each of the four kernels whose waits report through
    the protocol therefore clears the word in its first instructions.
    """
    for header, kernels in (
        ("persistent_kernel.cuh", 1),
        ("persistent_schedule.cuh", 1),
        ("collectives.cuh", 2),
    ):
        source = _source(header)
        assert source.count("timeout::clear_claim(scratch);") == kernels, header

    # In the two one-launch kernels it precedes the barrier latch, which is the
    # first thing either of them does that another CTA can observe.
    for header in ("persistent_kernel.cuh", "persistent_schedule.cuh"):
        source = _source(header)
        assert (
            source.index("timeout::clear_claim(scratch);")
            < source.index("latch_grid_phase(scratch, &latch_slot)")
        ), header
