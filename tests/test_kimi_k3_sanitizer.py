"""What the compute-sanitizer gate accepts, judged against captured runs.

The gate reads two independent verdicts -- what the tool saw and what the target
did -- and the reason this file exists is that they were observed to disagree.
``racecheck_target_segfault.log`` is a real racecheck run of the decode step that
printed ``RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`` while
rank 1 had already died of a segmentation fault. The orchestration of the day
read the hazard count, saw zero, and recorded the run as clean.

So the fixtures under ``tests/data/sanitizer`` are captured artifacts rather than
constructed ones, and every assertion here is about a run that actually
happened. Two are trimmed, each says so in its own header, and the trimming only
ever removed lines the parser does not read: host backtrace frames in the
memcheck capture and blank lines in the racecheck one.

No GPU, no compiled extension, no ``mok`` import. The one test here that runs a
real process tree runs ordinary Python children, because what it checks is the
orchestration's -- that a tool which outlives its budget takes its descendants
with it and still leaves a verdict behind.
"""

from __future__ import annotations

import os
import re
import signal
import sys
import time
from pathlib import Path

import pytest

from . import modal_sources

from benchmarks.kimi_k3_sanitizer import (
    HOST_ALLOWANCE_PER_RANK,
    K3_SANITIZER_GATES,
    PERMITTED_EXIT_CODES,
    SanitizerVerdict,
    sanitizer_verdict,
)

_FIXTURES = Path(__file__).resolve().parent / "data" / "sanitizer"

#: The marker compute-sanitizer prints before anything it has to say. Everything
#: above it in a persisted artifact is the header the orchestration wrote, which
#: is not part of the stream the gate judges.
_STREAM_MARKER = "========= COMPUTE-SANITIZER"


def _captured(name: str) -> tuple[int, str]:
    """One captured artifact, as the exit code and the stream it carried."""
    text = (_FIXTURES / f"{name}.log").read_text()
    header, marker, stream = text.partition(_STREAM_MARKER)
    assert marker, f"{name} carries no sanitizer stream"
    exit_codes = [
        int(line.split(":", 1)[1])
        for line in header.splitlines()
        if line.startswith("exit_code:")
    ]
    assert len(exit_codes) == 1, f"{name} records {exit_codes} exit codes"
    return exit_codes[0], marker + stream


def _verdict(name: str, tool: str) -> SanitizerVerdict:
    exit_code, stream = _captured(name)
    return sanitizer_verdict(tool, exit_code, stream)


# ---------------------------------------------------------------------------
# The runs that must pass.
# ---------------------------------------------------------------------------


def test_a_clean_racecheck_run_passes() -> None:
    """Exit code 0, one hazard summary at zero, eight passing ranks."""
    verdict = _verdict("racecheck_pass", "racecheck")
    assert verdict.passed, verdict.failures
    assert verdict.exit_code == 0
    assert (verdict.hazards, verdict.hazard_errors, verdict.hazard_warnings) == (
        0,
        0,
        0,
    )
    assert len(verdict.rank_summaries) == 8
    assert verdict.device_errors == 0


def test_a_clean_memcheck_run_passes_on_the_host_allowance_alone() -> None:
    """The only thing this run reported is the container's own denial.

    Sixteen reported errors, sixteen covered by the allowance, none left over --
    and the exit code is 99 because ``--error-exitcode`` fires on the allowance.
    That is the one case where a non-zero exit code is evidence of nothing.
    """
    verdict = _verdict("memcheck_pass", "memcheck")
    assert verdict.passed, verdict.failures
    assert verdict.exit_code == 99
    assert verdict.reported_errors == 16
    assert verdict.host_allowed_errors == 16
    assert verdict.device_errors == 0
    assert len(verdict.rank_summaries) == 8


def test_the_host_allowance_is_two_lines_from_each_of_the_eight_ranks() -> None:
    """The allowance is bounded by what it is, not by what it needs to absorb.

    Each rank's denial prints both a ``CUDA API Error`` line and a ``Program
    hit`` line, so eight ranks account for sixteen and nothing more. A gate that
    subtracted an unbounded count could hide a real finding behind the same
    name.
    """
    verdict = _verdict("memcheck_pass", "memcheck")
    assert HOST_ALLOWANCE_PER_RANK == 2
    assert verdict.host_allowed_errors == 8 * HOST_ALLOWANCE_PER_RANK
    assert verdict.host_allowed_errors == verdict.reported_errors


# ---------------------------------------------------------------------------
# The run that must fail, and the rule that let it pass.
# ---------------------------------------------------------------------------


def test_zero_hazards_does_not_pass_a_run_whose_target_died() -> None:
    """The captured disagreement, and the whole reason the gate is conjunctive.

    Every clause the old rule looked at is satisfied by this run: the tool
    reached its summary, the summary is all zeros, and no device-side error was
    reported. It still must fail, because a rank segfaulted and racecheck has
    nothing to say about a rank that stopped running.
    """
    verdict = _verdict("racecheck_target_segfault", "racecheck")

    assert (verdict.hazards, verdict.hazard_errors, verdict.hazard_warnings) == (
        0,
        0,
        0,
    )
    assert verdict.device_errors == 0
    assert not verdict.passed, "a dead rank must not pass on a zero hazard count"


def test_the_rule_this_replaces_would_have_passed_that_run() -> None:
    """The bug, written down so the fix cannot be undone quietly.

    The orchestration used to gate on ``device_errors != 0`` and nothing else.
    Against the captured segfault that predicate is False, so the run was
    recorded as clean. Asserting it here is what makes the clause above a
    regression test rather than a preference.
    """
    verdict = _verdict("racecheck_target_segfault", "racecheck")
    old_rule_passed = verdict.device_errors == 0
    assert old_rule_passed
    assert not verdict.passed


def test_the_failing_run_names_every_reason_it_failed() -> None:
    """A refusal has to say what it saw, or it cannot be acted on."""
    verdict = _verdict("racecheck_target_segfault", "racecheck")
    reasons = " | ".join(verdict.failures)

    assert "exited 1" in reasons
    assert "fatal Python error" in reasons
    assert "segfaulted" in reasons
    assert "child failure" in reasons
    assert "target application returned an error" in reasons
    # The launcher tore the job down on the first rank to die, so not one rank
    # reached a pytest summary -- and the tool still printed all zeros.
    assert "0 of 8 ranks printed a pytest summary" in reasons
    assert verdict.rank_summaries == ()


def test_the_failing_run_carries_its_verdict_into_the_artifact_header() -> None:
    """The persisted artifact has to disagree with the tool too."""
    verdict = _verdict("racecheck_target_segfault", "racecheck")
    header = "\n".join(verdict.summary_lines())

    assert "verdict: FAILED" in header
    assert "exit_code: 1" in header
    assert "hazards: 0" in header
    assert "rank_summaries: 0/8" in header
    assert header.count("failure: ") == len(verdict.failures)


# ---------------------------------------------------------------------------
# The clauses, one at a time, against a run that is otherwise clean.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exit_code", [1, 2, 134, 139, -6, -11, 99 + 1])
def test_only_a_permitted_exit_code_passes(exit_code: int) -> None:
    """Anything but 0 or 99 is the target's own code coming through."""
    _, stream = _captured("racecheck_pass")
    verdict = sanitizer_verdict("racecheck", exit_code, stream)
    assert exit_code not in PERMITTED_EXIT_CODES
    assert not verdict.passed
    assert any("exited" in reason for reason in verdict.failures)


def test_a_missing_tool_summary_fails_because_the_tool_did_not_finish() -> None:
    """A truncated stream is not a clean one.

    A run killed by its own timeout prints everything up to the point it died,
    which can be a full set of passing ranks and no summary at all.
    """
    _, stream = _captured("racecheck_pass")
    truncated = stream.split("========= RACECHECK SUMMARY")[0]
    verdict = sanitizer_verdict("racecheck", 0, truncated)
    assert not verdict.passed
    assert any("RACECHECK SUMMARY" in reason for reason in verdict.failures)

    _, stream = _captured("memcheck_pass")
    truncated = stream.split("========= ERROR SUMMARY")[0]
    verdict = sanitizer_verdict("memcheck", 0, truncated)
    assert not verdict.passed
    assert any("ERROR SUMMARY" in reason for reason in verdict.failures)


def test_a_hazard_fails_even_with_a_permitted_exit_code() -> None:
    """The count the tool reports is not adjustable by the host allowance."""
    _, stream = _captured("racecheck_pass")
    hazardous = stream.replace(
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
        "RACECHECK SUMMARY: 3 hazards displayed (3 errors, 0 warnings)",
    )
    verdict = sanitizer_verdict("racecheck", 99, hazardous)
    assert verdict.hazards == 3
    assert not verdict.passed
    assert any("3 hazards" in reason for reason in verdict.failures)


def test_a_device_error_beyond_the_allowance_fails() -> None:
    """One more reported error than the denials account for is device-side."""
    _, stream = _captured("memcheck_pass")
    worse = stream.replace(
        "========= ERROR SUMMARY: 16 errors",
        "========= ERROR SUMMARY: 17 errors",
    )
    verdict = sanitizer_verdict("memcheck", 99, worse)
    assert (verdict.reported_errors, verdict.host_allowed_errors) == (17, 16)
    assert verdict.device_errors == 1
    assert not verdict.passed


def test_the_allowance_cannot_absorb_more_than_the_ranks_can_explain() -> None:
    """A ninth rank's worth of denials is not a container quirk any more."""
    _, stream = _captured("memcheck_pass")
    line = (
        "========= CUDA API Error: Failed to allocate physical memory\n"
        "========= Program hit CUDA_ERROR_NOT_PERMITTED (error 800) due to "
        '"operation not permitted" on CUDA API call to cuMemCreate.\n'
    )
    inflated = stream.replace(
        "========= ERROR SUMMARY: 16 errors",
        line + "========= ERROR SUMMARY: 18 errors",
    )
    verdict = sanitizer_verdict("memcheck", 99, inflated)
    assert (verdict.reported_errors, verdict.host_allowed_errors) == (18, 18)
    assert verdict.device_errors == 0
    assert not verdict.passed
    assert any("host allowance absorbed" in reason for reason in verdict.failures)


@pytest.mark.parametrize("ranks", [7, 9])
def test_every_expected_rank_has_to_have_printed_a_passing_summary(
    ranks: int,
) -> None:
    """Eight ranks were asked for, so eight summaries are the evidence."""
    _, stream = _captured("racecheck_pass")
    verdict = sanitizer_verdict("racecheck", 0, stream, expected_ranks=ranks)
    assert len(verdict.rank_summaries) == 8
    assert not verdict.passed
    assert any(
        f"of {ranks} ranks printed a pytest summary" in reason
        for reason in verdict.failures
    )


def test_a_rank_that_reported_a_failure_fails_the_run() -> None:
    """A passing count beside a failing one is still a failing rank."""
    _, stream = _captured("racecheck_pass")
    failing = stream.replace(
        "3 passed, 60 deselected in 1241.21s",
        "1 failed, 2 passed, 60 deselected in 1241.21s",
        1,
    )
    verdict = sanitizer_verdict("racecheck", 0, failing)
    assert not verdict.passed
    assert any("1 failed" in reason for reason in verdict.failures)


def test_a_rank_that_passed_nothing_fails_the_run() -> None:
    """An empty selection is not a clean sanitizer run of anything."""
    _, stream = _captured("racecheck_pass")
    empty = stream.replace(
        "3 passed, 60 deselected in 1241.21s",
        "63 deselected in 0.41s",
        1,
    )
    verdict = sanitizer_verdict("racecheck", 0, empty)
    assert not verdict.passed
    assert any("passed no tests" in reason for reason in verdict.failures)


@pytest.mark.parametrize(
    ("injected", "expected"),
    [
        ("Fatal Python error: Bus error", "fatal Python error"),
        ("terminate called; SIGABRT", "aborted"),
        ("INTERNALERROR> pytest crashed", "pytest reported a failure"),
        ("FAILED tests/test_kimi_k3_decode.py::test_thing", "pytest reported a failure"),
        ("========= Target application returned an error", "target application"),
    ],
)
def test_an_application_error_anywhere_in_the_stream_fails_the_run(
    injected: str, expected: str
) -> None:
    """The tool's summary is silent about the target, so the stream is read."""
    _, stream = _captured("racecheck_pass")
    verdict = sanitizer_verdict("racecheck", 0, f"{injected}\n{stream}")
    assert not verdict.passed
    assert any(expected in reason for reason in verdict.failures)


def test_the_clean_captures_trip_no_application_error_pattern() -> None:
    """The patterns above have to be quiet on runs that really were clean.

    A pattern that matched a clean stream would make the gate useless in the
    other direction, and both clean captures are full of the sanitizer's own
    ``=========`` chatter for it to catch on.
    """
    for name, tool in (("racecheck_pass", "racecheck"), ("memcheck_pass", "memcheck")):
        verdict = _verdict(name, tool)
        assert verdict.passed, (name, verdict.failures)


# ---------------------------------------------------------------------------
# The gate table itself.
# ---------------------------------------------------------------------------


def test_every_sanitizer_gate_selects_tests_that_exist_in_its_own_suite() -> None:
    """A gate whose ``-k`` matches nothing proves nothing.

    ``sanitizer_verdict`` does refuse a rank that passed no tests, so an empty
    selection fails rather than passing -- but it fails for the wrong reason,
    and the artifact it leaves says nothing about the code the gate was pointed
    at. That is the worse failure of the two, because it looks like a finding.

    The candidate's suite and production's share no test name, so this is not a
    hypothetical: the candidate gates were first written reusing production's
    expression, which deselected all fifteen of the candidate's tests.

    Read from the source rather than by collecting, so it holds on a CPU with no
    compiled extension -- the whole point of this file. The gate table lives
    beside the verdict for the same reason: importing it from ``modal_app``
    would pull in ``modal``, which the suite's own image has no reason to carry.
    """
    root = Path(__file__).resolve().parents[1]
    # Only the shapes the expressions actually use. `-k` also accepts `not` and
    # parentheses; if one shows up here it should be added deliberately rather
    # than parsed by accident, so anything else fails the split below.
    assert K3_SANITIZER_GATES, "no sanitizer gates are defined"
    for gate, (tool, files, expression) in K3_SANITIZER_GATES.items():
        terms = [term.strip() for term in expression.split(" or ")]
        assert all(
            re.fullmatch(r"[a-z0-9_]+", term) for term in terms
        ), (gate, expression)
        names = set()
        for path in files.split(","):
            source = (root / path).read_text()
            assert source, (gate, path)
            names.update(re.findall(r"^def (test_[a-z0-9_]+)", source, re.M))
        assert names, (gate, files)
        for term in terms:
            matched = [name for name in names if term in name]
            assert matched, (
                f"{gate}: -k term {term!r} matches none of the "
                f"{len(names)} tests in {files}"
            )
        assert tool in ("memcheck", "racecheck", "synccheck", "initcheck")


def test_a_tool_outside_the_default_gates_can_still_be_asked_for() -> None:
    """A gate's name has to be definable even when it is not a default.

    The default set and the definable set are different things, and conflating
    them refuses a by-name request for anything outside the defaults as an
    unknown gate. That mattered while racecheck was outside them and it still
    has to hold, because the next tool that cannot finish will be asked for the
    same way.

    Read from ``modal_app``'s text rather than by importing it, because this
    file's whole point is to hold without ``modal`` installed.
    """
    source = modal_sources.read()
    body = source[source.index("def verify("):source.index("    functions = {")]

    assert "definable = set(K3_GATES) | set(K3_SANITIZER_GATES)" in body
    assert "unknown = sorted(set(requested) - definable)" in body
    # The shape that was wrong: the default set as the only definition.
    assert "set(requested) - set(K3_GATES)" not in body


def test_every_sanitizer_tool_is_a_default_gate_now() -> None:
    """All three tools run by default, and the image is why.

    racecheck was outside the defaults because the tool's slowdown held a
    legitimate rendezvous past the watchdog's fifteen seconds, so the gate failed
    closed on a fact about the watchdog rather than about races -- on every run.
    A gate that is always red is a gate a reader learns to skip.

    `B300_SANITIZER_IMAGE` compiles the bounded spins with a wider budget, which
    removes the reason rather than the gate. So the assertion is now that nothing
    is absent, and that the definition still says why it is safe to have widened
    anything -- written where the definition is, not somewhere a reader has to
    go looking.
    """
    source = modal_sources.read()
    opened = source.index("K3_GATES = (")
    default = source[opened:source.index("\n)\n", opened)]
    named = set(re.findall(r'^    "([a-z-]+)",$', default, re.M))
    assert named, default
    absent = sorted(set(K3_SANITIZER_GATES) - named)
    assert absent == [], absent

    reason = source[source.index("#: The gates a default"):opened]
    assert "B300_SANITIZER_IMAGE" in reason, reason
    assert "compile-time" in reason, reason
    assert "fifteen seconds" in reason, reason


def test_only_the_sanitizer_image_widens_the_bounded_spin() -> None:
    """The widened budget may reach the sanitizer gates and nothing else.

    It is compiled in, so an image that carries it is a different binary from
    the one every other gate measures and ships. Two things keep that from
    spreading: `build_image` defaults the scale to one, so every image that does
    not ask gets production's budget; and only the sanitizer function is
    declared against the widened image.
    """
    source = modal_sources.read()

    assert "def build_image(\n    spec: GPUSpec, wait_timeout_scale: int = 1\n)" in source
    assert 'IMAGE = build_image(SPEC)\n' in source
    assert 'B300_IMAGE = build_image(SPECS["B300"])\n' in source
    assert "B300_SANITIZER_IMAGE = build_image(\n" in source
    assert "wait_timeout_scale=SANITIZER_WAIT_TIMEOUT_SCALE" in source

    # Exactly one function runs in it, and it is the sanitizer one.
    widened = [
        block
        for block in source.split("@app.function(")[1:]
        if "image=B300_SANITIZER_IMAGE" in block.split(")\n", 1)[0]
    ]
    assert len(widened) == 1, len(widened)
    assert widened[0].split("def ", 1)[1].startswith(
        "sanitize_kimi_k3_decode("
    ), widened[0][:200]


def test_the_base_the_scale_is_taken_against_is_the_compiled_one() -> None:
    """The scale multiplies a constant that lives in a header, not here.

    `modal_images` has to state the base to derive a scale from it, and a
    restated constant is a constant that drifts. The one in `serial_sync.cuh` is
    the one the device compiles, so it is the one this copy has to equal.
    """
    from modal_images import WAIT_TIMEOUT_BASE_CLOCKS

    header = (
        Path(__file__).resolve().parents[1] / "csrc" / "serial_sync.cuh"
    ).read_text()
    declared = re.search(
        r"kWaitTimeoutBaseClocks = ([\d']+)ULL;", header
    )
    assert declared, "the header no longer names a base"
    assert int(declared.group(1).replace("'", "")) == WAIT_TIMEOUT_BASE_CLOCKS


def test_the_sanitizer_gate_times_out_before_its_watchdog_can() -> None:
    """Under the tool, the gate is the bound and the watchdog is not.

    A watchdog trap and a real hang are indistinguishable to the tool -- both
    come back as zero hazards for a launch that did not finish -- so the one
    thing that must not happen under a sanitizer is the watchdog firing first.
    Five runs were lost to a scale that was picked rather than derived, so the
    budget is now the gate's whole wall clock counted at a clock ceiling, and
    this is that ordering as an assertion: whatever the part clocks at, the
    budget outlasts the gate.
    """
    from modal_images import (
        B300_CLOCK_CEILING_HZ,
        SANITIZER_GATE_TIMEOUT,
        SANITIZER_WAIT_TIMEOUT_SCALE,
        WAIT_TIMEOUT_BASE_CLOCKS,
    )

    budget = SANITIZER_WAIT_TIMEOUT_SCALE * WAIT_TIMEOUT_BASE_CLOCKS
    assert budget >= SANITIZER_GATE_TIMEOUT * B300_CLOCK_CEILING_HZ

    source = modal_sources.read()
    opened = source.index("def sanitize_kimi_k3_decode(")
    declared = source[source.rindex("@app.function(", 0, opened):opened]
    assert "timeout=SANITIZER_GATE_TIMEOUT," in declared, declared


def test_the_tools_budget_is_what_is_left_of_the_gates() -> None:
    """Not the gate's timeout less a constant, which is a deadline past Modal's.

    Modal counts a function's timeout from the call, so the image's startup and
    the imports are spent before the tool starts. A subprocess handed
    ``GATE - 300`` from four minutes in has a deadline after Modal's, and the
    container is killed with the tool still running -- which is one eight-hour
    racecheck run, reported as nothing at all because the artifact is written
    after the call that never returned.

    So the budget is measured: what remains of the gate at the moment the tool
    starts, less what writing the artifact needs.
    """
    from modal_images import SANITIZER_GATE_TIMEOUT, SANITIZER_TEARDOWN_SECONDS

    assert 0 < SANITIZER_TEARDOWN_SECONDS < SANITIZER_GATE_TIMEOUT

    source = modal_sources.read()
    opened = source.index("def sanitize_kimi_k3_decode(")
    body = source[opened:source.index("\n@app.function(", opened)]
    assert "entered = time.monotonic()" in body
    assert (
        "remaining = SANITIZER_GATE_TIMEOUT - (time.monotonic() - entered)"
        in body
    )
    assert "budget = remaining - SANITIZER_TEARDOWN_SECONDS" in body
    assert "budget=budget" in body
    # And the constant subtraction that caused it is gone rather than moved.
    assert "SANITIZER_GATE_TIMEOUT - 300" not in source


def test_the_sanitizer_runs_as_its_own_session_and_is_never_run_blocking() -> None:
    """`subprocess.run(timeout=...)` is the wrong call for this gate, twice.

    It ends only the process it started, and what it starts is
    `compute-sanitizer`, whose child is a torchrun whose children hold eight
    devices. And it buffers, so a run that will take eight hours is
    indistinguishable from one that hung in its first minute -- which is how the
    last one was watched for eight hours with no output at all.

    So the gate uses the process-group pattern the benchmark gates already use,
    and this requires the parts of it that matter: its own session, a bounded
    stream, and the group ended on the way out. The behaviour is checked against a
    real process tree in
    `test_a_tool_over_its_budget_takes_its_descendants_and_leaves_a_verdict`;
    this is the narrower claim that the blocking call is gone from the module
    rather than moved somewhere else in it.
    """
    source = modal_sources.read()
    opened = source.index("def _run_sanitizer_session(")
    body = source[opened:source.index("\n@app.function(", opened)]

    assert "subprocess.Popen(" in body
    assert "start_new_session=True," in body
    assert "_stream_bounded(" in body
    assert "_end_session(process)" in body
    # No sanitizer path may block on a call that cannot reach the ranks. Read
    # past the docstring, which names the call it replaced. The module's other
    # `subprocess.run` calls are `cuobjdump` and `nvidia-smi`, which have no
    # children and no budget.
    code = body[body.index('"""', body.index('"""') + 3) + 3:]
    assert "subprocess.run(" not in code
    # `process.wait(timeout=...)` is fine and is there: it reaps the leader after
    # its stream closed, and the group is ended if even that runs out.
    assert "process.wait(timeout=left)" in code


def test_a_cut_off_tool_reports_what_it_had_rather_than_raising() -> None:
    """A `TimeoutExpired` that escapes is a gate that reports nothing.

    The artifact is written after the run, so an exception out of it skips the
    write entirely: the volume keeps whatever the *previous* run left, and a
    reader fetching it gets a stale verdict with a plausible timestamp. That is
    worse than a failure. So the cut-off is returned rather than raised, the
    partial output is kept and marked partial, and the verdict is reached on it --
    where the missing summary line makes it fail closed.
    """
    source = modal_sources.read()
    helper = source.index("def _run_sanitizer_session(")
    helper_body = source[helper:source.index("\n@app.function(", helper)]
    assert "the output above is partial" in helper_body
    assert "return 124, output + note, True" in helper_body
    # Returned, not raised: nothing in the helper raises the cut-off onward.
    assert "raise subprocess.TimeoutExpired" not in helper_body

    opened = source.index("def sanitize_kimi_k3_decode(")
    body = source[opened:source.index("\n@app.function(", opened)]
    assert "verdict = sanitizer_verdict(tool, exit_code, output)" in body
    # And the timeout is visible in the result rather than only in the log.
    assert '"timed_out": timed_out,' in body


def test_a_partial_run_fails_closed_even_with_no_hazards() -> None:
    """The verdict on a cut-off run must refuse it, on the tool's own output.

    This is the fail-closed property from finding 1 applied to the new path: the
    output a killed racecheck leaves has no `RACECHECK SUMMARY` line and no
    pytest summary, so the verdict has to refuse it on both counts rather than
    read "no hazards reported" as "no hazards".
    """
    partial = (
        "========= COMPUTE-SANITIZER\n"
        "tests/test_kimi_k3_adaptive_gate_up.py .\n"
        "\n========= MoK: racecheck was still running after 28000s and was "
        "killed; the output above is partial\n"
    )
    verdict = sanitizer_verdict("racecheck", 124, partial)
    assert not verdict.passed
    reasons = "; ".join(verdict.failures)
    # Both counts, not either: an exit code outside the permitted set, and no
    # rank having printed a summary. A run cut off mid-launch fails on both, and
    # the zero hazards it does report are the absence of a summary line rather
    # than a finding about the kernel.
    assert "exited 124" in reasons, reasons
    assert "pytest summary" in reasons, reasons
    assert "RACECHECK SUMMARY" not in partial


def test_a_tool_over_its_budget_takes_its_descendants_and_leaves_a_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole timeout path, on a real process tree.

    `compute-sanitizer`'s child is a torchrun and a torchrun's children are eight
    ranks holding eight B300s, so what a timeout has to end is the group and not
    the process the gate started. `subprocess.run(timeout=...)` ends only the
    latter, which in a container about to be reused leaves devices nobody owns.

    The stand-in is the same shape: a parent that forks a grandchild ignoring
    SIGTERM the way a rank inside a driver call effectively does. What is asserted
    is the three things the gate needs and the old path got wrong -- the
    grandchild does not survive, the partial output comes back rather than being
    raised away, and the artifact and the refusal are both produced from it.
    """
    from modal_k3_gates import _run_sanitizer_session

    marker = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-u', '-c',\n"
        "    'import signal, time\\n'\n"
        "    'signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n'\n"
        "    'time.sleep(600)\\n',\n"
        "])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "print('========= COMPUTE-SANITIZER')\n"
        "print('tests/test_kimi_k3_adaptive_gate_up.py .')\n"
        "time.sleep(600)\n"
    )
    exit_code, output, timed_out = _run_sanitizer_session(
        [sys.executable, "-u", "-c", script],
        tool="racecheck",
        budget=6,
        cwd=str(tmp_path),
    )

    assert timed_out is True
    assert exit_code == 124
    # What it had, kept: both lines the stand-in printed before it went quiet.
    assert "========= COMPUTE-SANITIZER" in output
    assert "tests/test_kimi_k3_adaptive_gate_up.py ." in output
    # And marked, so a reader cannot take a truncated stream for a complete one.
    assert "the output above is partial" in output

    grandchild = int(marker.read_text())
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail(f"grandchild {grandchild} outlived the sanitizer's session")

    # The verdict the gate would reach on this, and the artifact it would write.
    verdict = sanitizer_verdict("racecheck", exit_code, output)
    assert not verdict.passed
    reasons = "; ".join(verdict.failures)
    assert "exited 124" in reasons, reasons

    import modal_k3_gates

    class _Volume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = _Volume()
    monkeypatch.setattr(modal_k3_gates, "K3_ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(modal_k3_gates, "K3_VOLUME", volume)
    modal_k3_gates._persist_k3_artifact(
        "racecheck.log", "\n".join(verdict.summary_lines()) + f"\n{output}"
    )
    written = (tmp_path / "racecheck.log").read_text()
    assert volume.commits == 1
    assert "verdict: FAILED" in written
    assert "the output above is partial" in written
