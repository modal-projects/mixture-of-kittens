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

No GPU, no compiled extension, no ``mok`` import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.kimi_k3_sanitizer import (
    HOST_ALLOWANCE_PER_RANK,
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
