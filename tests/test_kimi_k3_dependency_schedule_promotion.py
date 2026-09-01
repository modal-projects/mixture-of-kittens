"""Source contracts for what promoting the decode schedule to the default did.

The dependency-local schedule is what a step launches now, and that changed the
meaning of every test whose subject was which of the two schedules ran. Those
suites, the counters the schedule appends, and the guarded switch that still
selects the old barrier schedule are held here: the default is in the source,
the switch is still reachable for an A/B, and production notices neither.

The safety argument is in ``test_kimi_k3_dependency_schedule_source.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from . import kimi_k3_decode_sources as decode_sources

_CANDIDATE_KERNEL = "void kimi_k3_decode_dependency_local_kernel("
_PRODUCTION_KERNEL = "void kimi_k3_decode_persistent_kernel("


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

#: The tests whose subject is a launch that names no schedule.
#:
#: One entry, and it is the guard's own test: it selects the barrier schedule
#: under the benchmark variable, drops the variable, and then launches with
#: nothing selected to prove that what runs is the dependency-local kernel
#: anyway. A manager around that launch would set the variable again and check
#: nothing. Anything added here has to be a test of the unmanaged case itself
#: rather than a test that merely forgot to say which schedule it wanted.
_UNMANAGED_BY_DESIGN = frozenset(
    {"test_an_unguarded_step_cannot_reach_the_barrier_schedule_once_set"}
)


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

    `_UNMANAGED_BY_DESIGN` is the one shape the rule cannot cover, and it is
    named rather than pattern-matched so that adding to it is a decision. Those
    tests are about what an *unmanaged* launch does, so wrapping them in a
    manager would delete the thing they check.
    """
    path = Path(__file__).resolve().parent / suite
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tests = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test_")
        and node.name not in _UNMANAGED_BY_DESIGN
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
    """Promotion is a default, and the default survives the guard.

    Two things have to hold together and it is the conjunction that is the point.
    The storage initializes to 1, so a process that has set nothing lands on the
    dependency-local schedule. And the read short-circuits to that schedule
    unless ``benchmark_grid_tuning_enabled``, so a process that set the *other*
    one and then dropped the variable lands there too.

    The guard went on after the adaptive integration, and the reason is what
    turning this switch off now reaches. The barrier schedule launches
    ``kimi_k3_decode_persistent_kernel``, compiled against the resident two-stage
    ring, so an unguarded write here routes a public decode through both retired
    paths at once -- which is exactly what the integration was for closing off.
    While the promotion was the only thing at stake the guard would have been
    backwards, because it would have kept production out of the schedule it was
    promoted to; a short-circuit *to* the promoted schedule is the opposite, and
    it is what makes the stored value irrelevant to production rather than merely
    usually-correct.
    """
    persistent = _source("persistent_kernel.cuh")
    storage = _function_body(
        persistent, "std::atomic<int> &dependency_schedule_storage("
    )
    assert "static std::atomic<int> schedule{1};" in storage

    # Guarded, and guarded in the direction that fails safe: the unguarded
    # answer is a constant `true` reached before the storage is consulted at all.
    guard = _function_body(persistent, "bool dependency_schedule_enabled(")
    assert "if (!benchmark_grid_tuning_enabled()) return true;" in guard
    assert guard.index("benchmark_grid_tuning_enabled") < guard.index(
        "dependency_schedule_storage()"
    )

    setter = _function_body(
        persistent, "void set_dependency_schedule_for_testing("
    )
    assert "TORCH_CHECK" in setter
    assert "benchmark_grid_tuning_enabled()" in setter
    assert "MOK_KIMI_K3_ENABLE_GRID_TUNING" in setter

    # Both schedules are still built and both are still reachable: the barrier
    # schedule is the other half of the A/B and the bit-for-bit comparison.
    launch = _function_body(persistent, "void launch_decode(")
    assert launch.count("launch_persistent<") == 2
    assert launch.index("dependency_schedule_enabled()") < launch.index(
        "schedule::launch_dependency_local<"
    )

    # One dispatch per capacity path, and production's engine is the arm each
    # falls through to. Counting the template's call sites would say two only
    # while one engine exists; what the promotion actually claims is that a
    # process that has set nothing lands on `kEngineFusedAdaptive`, and that the
    # other arms are reachable only through `benchmark_gate_up_engine()`, whose
    # own guard is checked in `test_kimi_k3_decode.py`.
    assert launch.count("switch (engine)") == 2
    assert launch.count("default:") == 2
    assert launch.count("fused::kEngineFusedAdaptive>(") == 2
    for arm in launch.split("default:")[1:]:
        assert "fused::kEngineFusedAdaptive>(" in arm.split("return;")[0]
    assert launch.index("benchmark_gate_up_engine()") < launch.index(
        "switch (engine)"
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
        # The headers a caller can include. An umbrella reads as itself plus
        # every part it includes, so this covers the directory exactly once.
        for path in decode_sources.includable()
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
