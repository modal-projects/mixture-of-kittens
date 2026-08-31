"""Source contracts for the dependency-local Kimi K3 decode schedule.

The candidate in ``csrc/kimi_k3_decode/persistent_schedule.cuh`` replaces four
of the production kernel's five full-grid barriers with seven topologically
ordered task queues and ten bounded readiness edges. Almost everything that
makes that safe is a property of the source rather than of a run: which order
the queues are scanned in, which direction every edge points, which releases
are system-scope, which waits are bounded, and how many barriers are left. A
run can only show that today's shapes happen to work.

So those properties are pinned here, without the extension, and the device
suite in ``test_kimi_k3_dependency_schedule.py`` covers what the launch
computes. Keeping them apart is also what lets these run on a machine with no
B300 attached.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import pytest


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "csrc" / "kimi_k3_decode"

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
    return (_SOURCE_ROOT / name).read_text(encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Counters.
# ---------------------------------------------------------------------------


def test_the_counters_are_appended_on_their_own_scratch_grain() -> None:
    """Appending is what makes the two schedules' state provably disjoint.

    Carving the candidate's counters out of the phase region's headroom would
    have been cheaper and would have put them next to slots the production
    kernel and the private stages already own. Appending instead means nothing
    below the new region moves, every offset the existing tests pin stays where
    it was, and no slot is shared between the two schedules.
    """
    types = _source("types.cuh")
    assert (
        "inline constexpr int kScheduleBytes =\n"
        "    kRouterScoreBytes + scratch_region_bytes(kMaxTokens * kNumExperts);"
        in types
    )
    assert (
        "static constexpr int SCRATCH_BYTES =\n"
        "    kScheduleBytes + scratch_region_bytes(NUM_SCHEDULE_COUNTERS);"
        in types
    )
    assert "static_assert(kScheduleBytes == 8111360);" in types
    assert "static_assert(SCRATCH_BYTES == 8111872);" in types
    assert "static_assert(kScheduleBytes % SCRATCH_ALIGNMENT == 0," in types
    assert "static constexpr int SCRATCH_ALIGNMENT = 256;" in types

    # The region the candidate owns is the last one, and the view hands it out
    # under its own name.
    view = _function_body(types, "Scratch scratch_view(")
    assert view.index("base + kScheduleBytes") > view.index(
        "base + kRouterScoreBytes"
    )
    assert view.rindex("base + kScheduleBytes") == view.index(
        "base + kScheduleBytes"
    )


def test_no_counter_can_climb_out_of_a_signed_thirty_two_bit_word() -> None:
    """Nothing resets a ticket inside a launch, so its maximum is a contract.

    The counters are read back as ``int`` by the host and incremented as
    ``unsigned`` by the device, so the bound that matters is the signed
    maximum: past it, a host reading the workspace would see a negative
    ticket. Every one of them is asserted strictly under it in the header.
    """
    schedule = _source("persistent_schedule.cuh")
    assert (
        "inline constexpr int kScheduleCounterBound = 0x7fffffff;" in schedule
    )
    for assertion in (
        "static_assert(kScheduleLongestTicket < kScheduleCounterBound,",
        "static_assert(kScheduleLargestArrival < kScheduleCounterBound,",
        "static_assert(kMaxTokens * router::kScoreShards"
        " < kScheduleCounterBound);",
    ):
        assert assertion in schedule, assertion
    assert "static_assert(kScheduleLongestQueueUnits == 6272);" in schedule
    assert "static_assert(kScheduleLongestTicket == 6864);" in schedule
    assert 6_864 < 2**31


# ---------------------------------------------------------------------------
# What promotion did to the tests that compare the two schedules.
# ---------------------------------------------------------------------------

#: The suites whose subject is which schedule ran.
_SCHEDULE_SUITES = (
    "test_kimi_k3_dependency_schedule.py",
    "test_kimi_k3_dependency_schedule_stress.py",
)

#: The two managers, and the launch helpers that must be inside one of them.
_SCHEDULE_MANAGERS = ("barrier_schedule", "dependency_local_schedule")
_LAUNCHERS = ("_decode", "kimi_k3_decode")


def _selected_schedule(
    function: ast.FunctionDef,
) -> list[tuple[int, tuple[str, ...]]]:
    """Every launch in one test, with the schedules selected around it."""
    launches: list[tuple[int, tuple[str, ...]]] = []

    def walk(node: ast.AST, selected: tuple[str, ...]) -> None:
        if isinstance(node, ast.With):
            entered = selected + tuple(
                item.context_expr.func.id
                for item in node.items
                if isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id in _SCHEDULE_MANAGERS
            )
            for child in node.body:
                walk(child, entered)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _LAUNCHERS
        ):
            launches.append((node.lineno, selected))
        for child in ast.iter_child_nodes(node):
            walk(child, selected)

    for statement in function.body:
        walk(statement, ())
    return launches


@pytest.mark.parametrize("suite", _SCHEDULE_SUITES)
def test_no_test_of_the_two_schedules_leaves_the_choice_to_the_default(
    suite: str,
) -> None:
    """A comparison that names neither schedule stopped being a comparison.

    Before promotion the default was the barrier schedule, so a test could
    capture "production" by simply launching and then capture the candidate
    inside ``dependency_local_schedule()``. Promotion inverted the default and
    turned every one of those into a test of the new schedule against itself --
    which passes, reports nothing, and is the worst way for this to go wrong.
    Three tests were in that shape when the default moved.

    So the rule is that every launch in these suites is inside a manager that
    says which schedule it is, and no launch is inside both. It is checked on
    the syntax tree rather than by pattern, because the thing being asserted is
    a nesting relation and a regex cannot see one.
    """
    path = Path(__file__).resolve().parent / suite
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tests = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert tests, suite

    checked = 0
    for test in tests:
        for line, selected in _selected_schedule(test):
            checked += 1
            assert selected, (
                f"{suite}:{line} in {test.name} launches without selecting a "
                f"schedule, so it tests whichever one is currently the default"
            )
            assert len(set(selected)) == 1, (
                f"{suite}:{line} in {test.name} is inside {selected}, so which "
                f"schedule it runs depends on which manager is innermost"
            )
    assert checked, suite


# ---------------------------------------------------------------------------
# What production must not notice.
# ---------------------------------------------------------------------------


def test_the_dependency_local_schedule_is_the_one_a_step_defaults_to() -> None:
    """Promotion is a default, and the default is in the source.

    The storage initializes to 1 and the read consults nothing else, so a
    process that has set nothing runs the dependency-local schedule. In
    particular the read must not consult ``benchmark_grid_tuning_enabled``:
    while the schedule was a measurement that guard was what kept it out of
    production, and now it would keep production out of the schedule it was
    promoted to.

    The grid override keeps its own guard, which is checked separately -- the
    two switches are independent now and only one is still benchmark-only.
    """
    persistent = _source("persistent_kernel.cuh")
    storage = _function_body(
        persistent, "std::atomic<int> &dependency_schedule_storage("
    )
    assert "static std::atomic<int> schedule{1};" in storage

    guard = _function_body(persistent, "bool dependency_schedule_enabled(")
    assert "benchmark_grid_tuning_enabled" not in guard

    setter = _function_body(
        persistent, "void set_dependency_schedule_for_testing("
    )
    assert "TORCH_CHECK" not in setter
    assert "MOK_KIMI_K3_ENABLE_GRID_TUNING" not in setter

    # Both schedules are still built and both are still reachable: the barrier
    # schedule is the other half of the A/B and the bit-for-bit comparison.
    launch = _function_body(persistent, "void launch_decode(")
    assert launch.count("launch_persistent<") == 2
    assert launch.count("schedule::launch_dependency_local<") == 2
    assert launch.index("dependency_schedule_enabled()") < launch.index(
        "schedule::launch_dependency_local<"
    )


def test_the_production_kernel_compiles_as_though_the_candidate_did_not_exist(
) -> None:
    """A runtime branch inside one kernel would make both pay for both.

    Register pressure is the whole reason the candidate is a second
    ``__global__``: the tensor instantiation is already at 254 registers, and a
    branch carrying the candidate's counters and clocks through it would spill.
    So the production kernel body may not name a single schedule symbol, and
    the candidate may not be reachable from it.
    """
    production = _production()
    assert "schedule::" not in production
    assert "kQueue" not in production
    assert "kSchedule" not in production
    assert "wait_for_schedule_count" not in production
    assert "ScheduleClocks" not in production

    # The production kernel keeps every phase and every barrier it had.
    for phase in range(6):
        assert f"// Phase {phase}:" in production
    assert production.count("claim_unit(") == 1
    assert production.count("claim_unit_batch(") == 2


def test_the_candidate_runs_the_production_stages_rather_than_copies_of_them(
) -> None:
    """A candidate that recomputed anything would not be an A/B of scheduling.

    Every stage the candidate runs is the production device function, called
    with the production arguments. The one exception is latent quantization,
    which is *restricted* rather than reimplemented: a projection unit's column
    range is a whole number of MXFP8 groups, so the CTA that produced them can
    quantize exactly those, which is what removes production's second barrier
    instead of replacing it with an edge.
    """
    body = _candidate()
    for stage in (
        "skinny_gemm::latent_down_tcgen05(",
        "skinny_gemm::latent_down_cuda_core<kMaxCoreCapacity>(",
        "shared_experts::project_tensor(",
        "shared_experts::gate_up_core<kMaxCoreCapacity>(",
        "shared_experts::activate_shared_tile(",
        "shared_experts::down_tensor(",
        "shared_experts::down_core<kMaxCoreCapacity>(",
        "router::score_shard(",
        "router::select_after_score_shard(",
        "router::build_assignments(",
        "router::build_expert_units(",
        "expert_mxfp4::fused_w13::routed_gate_up_fused_unit(",
        "expert_mxfp4::grouped_pipeline::grouped_down_unit(",
    ):
        assert body.count(stage) == 1, stage

    # No separate grid-wide quantization pass, and no barrier in front of one.
    assert "quantize_latent_rows(" not in body
    fused = _function_body(
        _source("persistent_schedule.cuh"), "void quantize_latent_group_range("
    )
    reference = _function_body(
        _source("expert_mxfp4_staging.cuh"), "void quantize_latent_rows("
    )
    for arithmetic in (
        "expert_mxfp4::select_e8m0_scale(absolute_max)",
        "(254u - static_cast<unsigned int>(scale)) << 23",
    ):
        assert arithmetic.replace("expert_mxfp4::", "") in reference
        assert arithmetic in fused
    # The tensor path's columns arrived through a bulk store this CTA's L1
    # never saw, so the group range reads past it.
    assert "__ldcg(" in fused
    assert "__threadfence();" in fused


@pytest.mark.parametrize(
    "name",
    [
        path.name
        for path in sorted(_SOURCE_ROOT.glob("*.cuh"))
        if path.name != "persistent_schedule.cuh"
    ],
)
def test_no_other_decode_source_had_to_learn_about_the_candidate(
    name: str,
) -> None:
    """The candidate is additive, which is what keeps the A/B honest.

    ``types.cuh`` owns the queues, the counters, and the edge table, and
    ``persistent_kernel.cuh`` owns the guarded dispatch. Nothing else in the
    include closure may mention the schedule at all, because anything that did
    would be a change the production side of the measurement also paid for.
    """
    text = _source(name)
    if name in {"types.cuh", "persistent_kernel.cuh"}:
        assert "kQueueSource" in text or "schedule::" in text
        return
    assert "kQueue" not in text
    assert "kSchedule" not in text
    assert "dependency_local" not in text
