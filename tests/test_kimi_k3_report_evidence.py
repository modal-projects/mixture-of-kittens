"""The report's numbers are the artifacts' numbers, and nothing else's.

The A/B harness has its own tests for the reductions it performs. This file is
about the step after: a report that quotes those reductions in prose is a second
copy of them, and the first revision of this task's report showed what a second
copy does -- it quoted a wait share whose numerator included the per-edge waits
and whose denominator did not, and both numbers looked fine beside each other.

So the report's numeric tables are generated from the artifacts and checked
against them here. The synthetic cases pin the accounting rules the fix
established; the last test runs the check against the real report, so a hand
edit to a generated table fails a gate rather than shipping.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _evidence():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("benchmarks.kimi_k3_report_evidence")


def _report_gates():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("benchmarks.kimi_k3_report_gates")


def _profile(
    *,
    barrier: int,
    readiness: int,
    work: int,
    edge_waits: dict[str, int] | None = None,
) -> dict[str, object]:
    """One variant's profile, with a child band that must not be totalled.

    ``routed_gate_up_mma`` is a refinement of ``routed_gate_up``: it measures
    cycles the parent already measured. Every fixture carries one, because a
    total that included it is the bug the top-level rule exists for.
    """
    bands = {
        "grid_barrier": barrier,
        "readiness_wait": readiness,
        "routed_gate_up": work,
        "routed_down": 0,
        "tail": 0,
        "publish": 0,
        "shared_experts": 0,
        "router_score": 0,
        "latent_project": 0,
        "latent_quantize": 0,
        "assignment": 0,
        "routed_queue": 0,
        "routed_gate_up_mma": work,
    }
    profile: dict[str, object] = {
        "phase_clock_cycles": bands,
        "phase_clock_top_level": [
            name for name in bands if name != "routed_gate_up_mma"
        ],
        "phase_clock_total_cycles": barrier + readiness + work,
        "wait_cycles": barrier + readiness,
    }
    if edge_waits is not None:
        profile["edge_wait_cycles"] = edge_waits
        profile["edge_makespan_cycles"] = {
            name: value // 2 for name, value in edge_waits.items()
        }
        profile["queue_makespan_cycles"] = {"source": 10, "publish": 40}
    return profile


@pytest.fixture()
def artifacts() -> tuple[dict[str, object], dict[str, object]]:
    """One shape, two variants, five repeats, with a known median and p99."""
    results = {
        "barriers_per_launch": {"production": 5, "candidate": 1},
        "residency": {"production": {"core": 1}, "candidate": {"core": 1}},
        "decision": {
            "experiment_gate_passed": False,
            "promotion_passed": True,
        },
        "schedule": {"queues": ["source", "publish"]},
        "points": [
            {
                "tokens": 16,
                "gating": True,
                "passed": False,
                "promotion_passed": True,
                "profiles": {
                    "production": _profile(
                        barrier=400, readiness=100, work=500
                    ),
                    "candidate": _profile(
                        barrier=40,
                        readiness=260,
                        work=500,
                        edge_waits={"gate_up_latent": 200, "tail_publish": 40},
                    ),
                },
            }
        ],
    }
    raw = {
        "16": {
            variant: [
                {
                    "repeat": repeat,
                    "order_position": repeat % 2,
                    "rank_max_samples_ms": [
                        base + repeat * 0.001 + sample * 0.01
                        for sample in range(101)
                    ],
                }
                for repeat in range(5)
            ]
            for variant, base in (("production", 2.0), ("candidate", 1.0))
        }
    }
    return results, raw


def test_a_launch_total_is_the_sum_of_the_bands_that_do_not_nest(
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """A child band added to its parent inflates the denominator silently.

    ``routed_gate_up_mma`` measures cycles ``routed_gate_up`` already measured,
    so a total over every counter reports a launch that is half again as long as
    the launch was -- and understates every share taken against it.
    """
    evidence = _evidence()
    derived = evidence.derive(*artifacts)
    production = derived["cycles"][16]["production"]

    assert production["total_cycles"] == 400 + 100 + 500
    assert production["total_cycles"] == production["reported_total_cycles"]
    assert sum(production["bands"].values()) > production["total_cycles"]


def test_the_waiting_is_inside_the_total_it_is_a_share_of(
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """This is the finding: a numerator the denominator does not contain.

    The candidate's waiting is ``grid_barrier + readiness_wait``, both of which
    are top-level bands, so the share is well defined. The per-edge counters
    split the readiness band by edge -- they are the same cycles at a finer
    grain -- so adding them to the numerator, which is what the first revision
    did, produces a share of a quantity that does not exist.
    """
    evidence = _evidence()
    derived = evidence.derive(*artifacts)
    candidate = derived["cycles"][16]["candidate"]

    assert candidate["wait_cycles"] == 40 + 260
    assert candidate["edge_wait_cycles"] == 240
    # Inside the band, not beside it.
    assert candidate["edge_wait_cycles"] <= candidate["bands"]["readiness_wait"]
    assert candidate["wait_fraction"] == pytest.approx(300 / 800)
    # The share the first revision would have printed, for contrast.
    assert (candidate["wait_cycles"] + candidate["edge_wait_cycles"]) / (
        candidate["total_cycles"] - candidate["edge_wait_cycles"]
    ) > candidate["wait_fraction"]


def test_the_latency_is_recomputed_from_the_retained_samples(
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """A median nobody can recompute is a claim, not a measurement.

    The samples are a straight ramp per repeat, so both quantiles are known
    exactly, and the repeat medians differ from each other only by the
    per-repeat offset -- which is what the dispersion has to come out as.
    """
    evidence = _evidence()
    derived = evidence.derive(*artifacts)
    one = derived["latency"][16]

    assert one["production"]["repeat_medians_ms"] == pytest.approx(
        [2.5 + repeat * 0.001 for repeat in range(5)]
    )
    assert one["production"]["center_ms"] == pytest.approx(2.502)
    assert one["production"]["dispersion_ms"] == pytest.approx(0.004)
    assert one["production"]["p99_ms"] == pytest.approx(2.992)
    assert one["change"]["median_gain_fraction"] == pytest.approx(
        (2.502 - 1.502) / 2.502
    )
    assert one["candidate"]["sample_counts"] == [101] * 5


def test_the_headline_table_carries_the_gate_it_failed(
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """The result the report is most tempted to round up.

    The 8% experiment gate failed and the 2% promotion bar passed on the same
    measurement, so the headline table has to be able to say both at once, and
    it has to say them from the harness's own verdict fields.
    """
    evidence = _evidence()
    derived = evidence.derive(*artifacts)
    rendered = evidence.render("medians", derived)

    assert "8% gate" in rendered and "2% promotion bar" in rendered
    row = [line for line in rendered.splitlines() if line.startswith("| M = ")]
    assert len(row) == 1
    assert row[0].endswith("| **FAIL** | PASS |")


def test_a_hand_edited_table_fails_the_check(
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """The check is only worth having if it fails on a plausible edit.

    So a digit is changed in a generated block -- the kind of edit that keeps a
    stale number alive through a remeasurement -- and the check has to name that
    block. Rewriting then has to restore it, and a rewritten report has to check
    clean, or the generator and the checker disagree and neither means anything.
    """
    evidence = _evidence()
    derived = evidence.derive(*artifacts)
    report = (
        "# report\n\n"
        "<!-- generated: wait_shares -->\n"
        f"{evidence.render('wait_shares', derived)}"
        "<!-- end: wait_shares -->\n"
    )
    assert evidence.check_blocks(report, derived) == []

    edited = report.replace("300", "250")
    assert edited != report
    problems = evidence.check_blocks(edited, derived)
    assert len(problems) == 1
    assert problems[0].startswith("wait_shares:")

    restored = evidence.rewrite(edited, derived)
    assert evidence.check_blocks(restored, derived) == []
    assert restored == report.rstrip("\n") + "\n"


def _run_document(deltas: dict[int, float], production: float) -> dict:
    return {
        "decision": {"promotion_passed": all(
            delta > -0.01 for delta in deltas.values()
        )},
        "points": [
            {
                "tokens": tokens,
                "improvement_fraction": delta,
                "production_median_ms": production,
            }
            for tokens, delta in deltas.items()
        ],
    }


def test_the_between_run_spread_is_derived_per_arm(tmp_path: Path) -> None:
    """A repeat dispersion inside one run is not a spread between nodes.

    The M = 128 verdict is a 1% bar with a fraction of that in hand, so the
    question the artifacts have to answer is how much the delta moves when the
    run lands somewhere else -- and, since the arm is what is being judged,
    whether the control run beside it moved by the same amount.
    """
    evidence = _evidence()
    runs = tmp_path / "runs"
    runs.mkdir()
    for name, delta, production in (
        ("head-primary", -0.008, 1.735),
        ("head-a", -0.006, 1.737),
        ("control-p", -0.002, 1.736),
    ):
        (runs / f"{name}.json").write_text(
            json.dumps(_run_document({128: delta}, production))
        )

    spread = evidence.run_spread(runs)
    head = spread["arms"]["head"]
    assert head["count"] == 2
    assert head["by_shape"][128]["median"] == pytest.approx(-0.007)
    assert head["by_shape"][128]["spread"] == pytest.approx(0.002)
    assert head["all_promoted"] is True
    # The control ran on the same pool, so its own delta is what the head's is
    # compared against rather than a number from another day.
    assert spread["arms"]["control"]["by_shape"][128]["median"] == pytest.approx(
        -0.002
    )
    assert evidence.run_spread(tmp_path / "absent") is None


def test_a_stale_prose_figure_fails_the_check(
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """Rewriting cannot fix a sentence, so a sentence has to be checked.

    The generated blocks are replaced wholesale on a rewrite; the prose around
    them is not, which is exactly why a prose figure is the one that survives a
    remeasurement. So the handful that restate a derived quantity are pinned as
    literal text, and a report missing one of them names it.
    """
    evidence = _evidence()
    derived = evidence.derive(*artifacts)
    required = evidence.phrases(derived)
    assert required

    complete = "\n".join(required.values())
    assert evidence.check_phrases(complete, derived) == []

    name = "headline_waiting"
    stale = complete.replace(required[name], required[name].replace("4", "9"))
    assert stale != complete
    problems = evidence.check_phrases(stale, derived)
    assert [problem.split(":")[0] for problem in problems] == [name]

    # The report is hard-wrapped, so a pinned figure is routinely split across a
    # line break. A check that failed on that would be a check nobody could keep
    # green without reflowing prose around it.
    wrapped = complete.replace(" ", "\n  ")
    assert evidence.check_phrases(wrapped, derived) == []


def test_every_generated_block_of_the_real_report_matches_its_artifacts(
) -> None:
    """The one that guards the report a reader actually reads.

    Skipped rather than failed when the artifacts are not in the tree, because
    the suite runs in checkouts that have the sources without the measurement --
    but never skipped when the report is there and the artifacts are, which is
    the case that matters.
    """
    evidence = _evidence()
    for path in (
        evidence.REPORT_PATH,
        evidence.RESULTS_PATH,
        evidence.RAW_PATH,
    ):
        if not path.exists():
            pytest.skip(f"{path.name} is not in this checkout")

    derived = evidence.load()
    text = evidence.REPORT_PATH.read_text(encoding="utf-8")
    found = evidence.blocks(text)
    # A report with no generated blocks would pass an empty check, so the
    # blocks the report is required to generate are named here.
    expected = {
        "medians", "latency", "p99", "wait_shares", "queues", "runs",
        "artifacts", "trap_results", "trap_race",
    } | {
        f"cycles_m{tokens}" for tokens in (16, 128)
    } | {f"edges_m{tokens}" for tokens in (16,)}
    assert expected <= set(found), sorted(found)
    assert evidence.check(text, derived) == []


# ---------------------------------------------------------------------------
# The gate logs. The A/B tables were generated a round before these were, and
# the artifact table is what that gap cost: it claimed "784 passed, 1 skipped"
# and "180 individually verified launches" after the suite had grown to 793 and
# the stress plan had been re-cut to 240.
# ---------------------------------------------------------------------------


def _gate_logs(directory: Path, *, tests: str, ranks: int = 8) -> None:
    """A gate log dir holding one pytest tail per rank, as torchrun writes it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task-11c-tests.log").write_text(
        "\n".join(
            f"[rank{rank}]:====== {tests} in 208.33s (0:03:28) ======"
            for rank in range(ranks)
        )
    )


def test_a_gate_count_is_read_out_of_the_log_that_states_it(
    tmp_path: Path,
) -> None:
    """The count and the rank count both, because both were wrong in the report.

    A gate log is one pytest tail per rank. The outcome is the tail's, and the
    number of ranks is how many tails there are -- not a constant, because a
    launch that lost a rank is exactly the run whose log should stop agreeing
    with the report.
    """
    evidence = _evidence()
    _gate_logs(tmp_path, tests="793 passed, 2 skipped")

    gate = evidence.gates(tmp_path)
    assert gate["tests"]["counts"] == {"passed": 793, "skipped": 2}
    assert gate["tests"]["ranks"] == 8
    assert gate["tests"]["agreed"] is True
    assert gate["tests"]["passed"] is True

    # The logs that are not there are absent rather than zero, so a renderer
    # drops their rows instead of publishing a clean-looking nothing.
    assert gate["stress"] is None
    assert gate["trap"] is None


def test_a_gate_whose_ranks_disagree_is_not_reduced_to_one_number(
    tmp_path: Path,
) -> None:
    """Eight ranks run the same selection, so eight different tails are a bug.

    Reducing them to the first would let a rank that quietly ran something else
    be reported as if every rank had run the same thing, which is the failure
    mode a per-rank log exists to expose.
    """
    evidence = _evidence()
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "task-11c-tests.log").write_text(
        "[rank0]:=== 793 passed, 2 skipped in 208.33s (0:03:28) ===\n"
        "[rank1]:=== 792 passed, 3 skipped in 208.31s (0:03:28) ===\n"
    )

    gate = evidence.gates(directory)
    assert gate["tests"]["agreed"] is False
    assert len(gate["tests"]["distinct"]) == 2

    # And a disagreement withholds the pinned prose rather than pinning one of
    # the two readings, so the report cannot quote a number nobody measured.
    assert "suite_outcome" not in evidence._gate_phrases(gate)


def test_the_stress_launch_count_is_the_number_its_own_gate_asserts(
) -> None:
    """Derived from the gate's source, not recomputed beside it.

    The gate asserts its plan length on device, so quoting that assertion means
    the report and the gate cannot disagree without the gate failing first. The
    equality leg runs one pass of the same grid and launches both schedules per
    element, which makes its count a function of the same two numbers.
    """
    plan = _report_gates()._stress_plan()
    assert plan is not None

    assert plan["oracle_verified"] % plan["passes"] == 0
    assert plan["equality_pairs"] == plan["oracle_verified"] // plan["passes"]
    assert plan["equality_launches"] == 2 * plan["equality_pairs"]
    assert plan["total"] == plan["oracle_verified"] + plan["equality_launches"]

    # The gap §1.9 closed: the comment above the tuple claimed every capacity
    # bucket while 2 and 4 were missing from the tuple itself.
    assert {2, 4} <= set(plan["tokens"])

    assert _report_gates()._stress_plan(Path("/nonexistent.py")) is None


def test_the_trap_tables_are_the_diagnostics_the_trapped_launch_wrote(
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """Including the case where the recorded pair is not the expected one.

    A generated table that rendered ``recorded_code`` and called it the result
    would report a mismatch as a fact. The row states the pair only when the
    recorded and expected pairs agree, and says so loudly when they do not,
    because that disagreement is the whole point of the gate.
    """
    evidence = _evidence()
    directory = tmp_path / "logs"
    directory.mkdir()

    def injection(edge: str, code: int, slot: int, **overrides: object) -> str:
        body = {
            "edge": edge,
            "expected_code": code,
            "expected_slot": slot,
            "recorded_code": code,
            "recorded_slot": slot,
            "claiming_block": 0,
            "claim": 1,
            "launch_failed": True,
            **overrides,
        }
        return f"injected {edge}: {json.dumps(body)}"

    race = {
        "blocks": 60,
        "edge_count": 10,
        "units_per_edge": 6,
        "claim": 55,
        "claiming_block": 54,
        "recorded_code": 14,
        "recorded_slot": 144,
        "launch_failed": True,
    }
    (directory / "task-11c-trap.log").write_text(
        "\n".join(
            [
                injection("gate_up_assignment", 12, 142),
                injection("tail_publish", 18, 148),
                f"concurrent race: {json.dumps(race)}",
                "==== 4 passed in 71.60s (0:01:11) ====",
            ]
        )
    )

    gate = evidence.gates(directory)
    derived = evidence.derive(*artifacts, gate_data=gate)

    table = evidence.render("trap_results", derived)
    assert "code 12, slot 142, claimed by CTA 0, launch failed" in table
    assert "`wait_for_schedule_count_system`" in table

    # The claim is what the expected pair is derived from, so the block shows
    # the derivation rather than asserting the answer.
    block = evidence.render("trap_race", derived)
    assert "claim 55" in block
    assert "CTA 54" in block
    assert "edge 54 % 10 = 4" in block
    assert "unit 54 / 10 = 5" in block

    # And the artifact row counts the sites the race actually covered.
    row = evidence.render("artifacts", derived)
    assert "2 injected edges" in row
    assert "60 CTAs racing to report one pair" in row


def test_a_recorded_pair_that_is_not_the_expected_one_renders_as_a_failure(
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """The row a torn diagnostic would produce, spelled out rather than hidden."""
    evidence = _evidence()
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "task-11c-trap.log").write_text(
        "injected gate_up_assignment: "
        + json.dumps(
            {
                "edge": "gate_up_assignment",
                "expected_code": 12,
                "expected_slot": 142,
                "recorded_code": 12,
                "recorded_slot": 148,
                "claiming_block": 0,
                "claim": 1,
                "launch_failed": True,
            }
        )
    )

    gate = evidence.gates(directory)
    derived = evidence.derive(*artifacts, gate_data=gate)
    table = evidence.render("trap_results", derived)
    assert "**recorded (12, 148) but expected (12, 142)**" in table


def test_a_suite_size_counts_node_ids_and_not_lines(tmp_path: Path) -> None:
    """Every rank reports every test, and a parametrized test reports per case.

    Counting lines would multiply by eight; counting ``def test_`` in the source
    would undercount every parametrization. Both were wrong in the report at
    some point, so the count is distinct node ids from the log the run wrote.
    """
    evidence = _evidence()
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "task-11c-tests.log").write_text(
        "\n".join(
            f"[rank{rank}]:tests/test_kimi_k3_report_evidence.py::{name} PASSED"
            for rank in range(8)
            for name in ("test_one", "test_two[a]", "test_two[b]")
        )
        + "\n[rank0]:tests/test_kimi_k3_decode.py::test_other PASSED\n"
        + "[rank0]:=== 25 passed in 1.00s ===\n"
    )

    sizes = evidence.gates(directory)["suite_sizes"]
    assert sizes["tests/test_kimi_k3_report_evidence.py"] == 3
    assert sizes["tests/test_kimi_k3_decode.py"] == 1


def test_the_race_narration_comes_from_the_test_that_resolved_it(
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """Which CTA wins is a real race, so the edge cannot be prose.

    Two runs of this gate named different CTAs on different edges. The block
    therefore carries the test's own resolution of claim to edge, and a report
    that narrated the winner would be false after the next run.
    """
    evidence = _evidence()
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "task-11c-trap.log").write_text(
        "concurrent race: "
        + json.dumps(
            {
                "blocks": 60,
                "edge_count": 10,
                "claim": 27,
                "claiming_block": 26,
                "recorded_code": 16,
                "recorded_slot": 35,
                "launch_failed": True,
            }
        )
        + "\nthe claim named CTA 26, which was waiting on routed_down_gate_up "
        "unit 2; it published code 16 beside slot 35\n"
    )

    gate = evidence.gates(directory)
    assert gate["trap_race"]["resolved"]["edge"] == "routed_down_gate_up"

    block = evidence.render(
        "trap_race", evidence.derive(*artifacts, gate_data=gate)
    )
    assert "edge 26 % 10 = 6" in block
    assert "routed_down_gate_up unit 2's" in block
    assert "only that edge carries code 16" in block


def test_the_sanitizer_rows_come_from_the_headers_the_gates_write(
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """memcheck exits 99 on allowed host errors, and its row must not say 99.

    The header separates the reported errors from the ones the gate allows and
    from the device errors, which is the distinction the row is about: memcheck
    passed with sixteen host allocation errors and zero device errors, and a row
    that quoted the exit code would call that a failure.
    """
    evidence = _evidence()
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "task-11c-memcheck.log").write_text(
        "command: compute-sanitizer --tool memcheck\n"
        "establishes: memory safety only\n"
        "verdict: passed\n"
        "exit_code: 99\n"
        "reported_errors: 16\n"
        "host_allowed_errors: 16\n"
        "device_errors: 0\n"
        "hazards: 0\n"
        "rank_summaries: 8\n"
        "========= COMPUTE-SANITIZER\n"
        "========= device_errors: 99999 not a header line\n"
    )

    gate = evidence.gates(directory)
    header = gate["sanitizers"]["memcheck"]
    assert header["verdict"] == "passed"
    assert header["device_errors"] == "0"
    assert header["host_allowed_errors"] == "16"

    derived = evidence.derive(*artifacts, gate_data=gate)
    row = evidence.render("artifacts", derived)
    assert "| `task-11c-memcheck.log` | 0 device errors, 0 hazards, 8/8 ranks |" in row
    assert "99999" not in row
