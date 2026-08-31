"""What a compute-sanitizer run has to show before it counts as evidence.

A sanitizer run produces two independent verdicts and reading only one of them
is how a clean report gets published for a run that never finished. The tool
says whether it saw a violation; the target says whether it did the work. Both
were observed to disagree: a racecheck run of the decode step reported
``RACECHECK SUMMARY: 0 hazards displayed`` while one rank had already died of a
segmentation fault in the reference oracle, and the orchestration that read only
the hazard count recorded the run as clean.

So the verdict here is conjunctive, and every clause has to be satisfied:

* the exit code has to be one the run is allowed to end with,
* the tool's own summary has to be present, because a missing summary means the
  tool did not reach the end,
* the tool's adjusted violation counts have to be zero,
* every expected rank has to have printed a passing pytest summary, and
* nothing in the stream may look like a dead or failing target.

The one adjustment is the host allowance in :data:`HOST_ALLOWED_ERRORS`, and it
is deliberately not hidden. The container cannot create fabric-handle physical
allocations, so PyTorch's symmetric-memory probe raises one ``cuMemCreate``
denial per rank on a path the decode step does not use, the sanitizer counts it,
and ``--error-exitcode`` fires on it. Subtracting it is the only way the gate can
be about device-side violations at all -- but the allowance is named, counted,
bounded at :data:`HOST_ALLOWANCE_PER_RANK` per rank, and carried in the verdict
and the artifact, so it can neither grow silently nor absorb a real finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Exit codes a run is allowed to end with.
#:
#: ``0`` is a clean run. ``99`` is ``--error-exitcode`` firing, which is allowed
#: only because the host allowance below can be the whole reason it fired; the
#: adjusted counts still have to be zero, so 99 never passes on its own.
#: Anything else is the target's own exit code coming through the sanitizer, and
#: a target that did not exit 0 has not produced evidence about anything.
PERMITTED_EXIT_CODES = (0, 99)

#: Host-side API errors the container provokes and the decode step never uses.
#:
#: Each denial prints both lines, so a rank contributes two to the tool's count.
HOST_ALLOWED_ERRORS = (
    re.compile(
        r"Program hit CUDA_ERROR_NOT_PERMITTED .* on CUDA API call to "
        r"cuMemCreate\."
    ),
    re.compile(r"CUDA API Error: Failed to allocate physical memory"),
)

#: Reported errors one rank may contribute to the allowance: the two lines above.
HOST_ALLOWANCE_PER_RANK = 2

_ERROR_SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) errors?")
_RACECHECK_SUMMARY = re.compile(
    r"RACECHECK SUMMARY: (\d+) hazards displayed "
    r"\((\d+) errors?, (\d+) warnings?\)"
)

_OUTCOME = (
    r"\d+ (?:passed|failed|errors?|skipped|deselected"
    r"|xfailed|xpassed|warnings?|reruns?)"
)
_PYTEST_SUMMARY = re.compile(
    rf"(?P<body>{_OUTCOME}(?:, {_OUTCOME})*) in \d+(?:\.\d+)?s"
)
_OUTCOME_FIELD = re.compile(r"(\d+) ([a-z]+)")

#: Shapes in the stream that mean the target died or failed, whatever the tool
#: reported. Each is paired with what it says, so a failure names its evidence.
TARGET_FAILURE_PATTERNS = (
    (
        "a target process died on a fatal Python error",
        re.compile(r"Fatal Python error:"),
    ),
    (
        "a target process segfaulted",
        re.compile(r"Segmentation fault|SIGSEGV|exitcode: -11"),
    ),
    (
        "a target process aborted",
        re.compile(r"SIGABRT|exitcode: -6|^Aborted", re.MULTILINE),
    ),
    (
        "torchrun reported a child failure",
        re.compile(r"ChildFailedError|failed \(exitcode: -?\d+\)"),
    ),
    (
        "the sanitizer said the target application returned an error",
        re.compile(r"Target application returned an error"),
    ),
    (
        "pytest reported a failure",
        re.compile(r"^(?:pytest FAILED|INTERNALERROR|FAILED |ERROR )", re.MULTILINE),
    ),
)

#: Tools whose own verdict is a hazard count rather than an error count.
HAZARD_TOOLS = ("racecheck",)


@dataclass(frozen=True)
class SanitizerVerdict:
    """Everything the gate read, and every reason it would refuse the run."""

    tool: str
    exit_code: int
    expected_ranks: int
    reported_errors: int
    host_allowed_errors: int
    device_errors: int
    hazards: int
    hazard_errors: int
    hazard_warnings: int
    rank_summaries: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary_lines(self) -> tuple[str, ...]:
        """The header the persisted artifact carries, verdict first."""
        return (
            f"verdict: {'passed' if self.passed else 'FAILED'}",
            f"exit_code: {self.exit_code}",
            f"reported_errors: {self.reported_errors}",
            f"host_allowed_errors: {self.host_allowed_errors}",
            f"device_errors: {self.device_errors}",
            f"hazards: {self.hazards}",
            f"rank_summaries: {len(self.rank_summaries)}"
            f"/{self.expected_ranks}",
            *(f"failure: {reason}" for reason in self.failures),
        )


def _outcomes(body: str) -> dict[str, int]:
    return {name: int(count) for count, name in _OUTCOME_FIELD.findall(body)}


def sanitizer_verdict(
    tool: str,
    exit_code: int,
    output: str,
    expected_ranks: int = 8,
) -> SanitizerVerdict:
    """Judge one compute-sanitizer run from its exit code and its stream.

    Pure, so the judgement can be exercised against captured logs on a CPU
    rather than only by provoking the condition on eight GPUs.
    """
    host_allowed = sum(
        len(pattern.findall(output)) for pattern in HOST_ALLOWED_ERRORS
    )
    error_summaries = [int(count) for count in _ERROR_SUMMARY.findall(output)]
    hazard_summaries = _RACECHECK_SUMMARY.findall(output)
    reported = sum(error_summaries)
    hazards = sum(int(found[0]) for found in hazard_summaries)
    hazard_errors = sum(int(found[1]) for found in hazard_summaries)
    hazard_warnings = sum(int(found[2]) for found in hazard_summaries)
    device_errors = reported - min(host_allowed, reported)

    summaries = [match.group("body") for match in _PYTEST_SUMMARY.finditer(output)]
    failures: list[str] = []

    if tool in HAZARD_TOOLS:
        if not hazard_summaries:
            failures.append(
                f"{tool} printed no RACECHECK SUMMARY, so it did not reach the "
                f"end of the run"
            )
    elif not error_summaries:
        failures.append(
            f"{tool} printed no ERROR SUMMARY, so it did not reach the end of "
            f"the run"
        )

    if exit_code not in PERMITTED_EXIT_CODES:
        failures.append(
            f"{tool} exited {exit_code}, which is the target's own exit code "
            f"rather than one of {list(PERMITTED_EXIT_CODES)}"
        )

    if device_errors:
        failures.append(
            f"{tool} reported {reported} errors and only {host_allowed} are "
            f"covered by the host allowance, leaving {device_errors} device-side"
        )
    if host_allowed > expected_ranks * HOST_ALLOWANCE_PER_RANK:
        failures.append(
            f"the host allowance absorbed {host_allowed} errors, more than the "
            f"{expected_ranks * HOST_ALLOWANCE_PER_RANK} that "
            f"{expected_ranks} ranks can account for"
        )
    if hazards or hazard_errors or hazard_warnings:
        failures.append(
            f"{tool} reported {hazards} hazards, {hazard_errors} errors and "
            f"{hazard_warnings} warnings"
        )

    if len(summaries) != expected_ranks:
        failures.append(
            f"{len(summaries)} of {expected_ranks} ranks printed a pytest "
            f"summary"
        )
    for index, body in enumerate(summaries):
        outcomes = _outcomes(body)
        bad = {
            name: count
            for name, count in outcomes.items()
            if name in ("failed", "error", "errors") and count
        }
        if bad:
            failures.append(f"rank summary {index} reported {body!r}")
        elif not outcomes.get("passed"):
            failures.append(
                f"rank summary {index} passed no tests: {body!r}"
            )

    for reason, pattern in TARGET_FAILURE_PATTERNS:
        if pattern.search(output):
            failures.append(reason)

    return SanitizerVerdict(
        tool=tool,
        exit_code=exit_code,
        expected_ranks=expected_ranks,
        reported_errors=reported,
        host_allowed_errors=host_allowed,
        device_errors=device_errors,
        hazards=hazards,
        hazard_errors=hazard_errors,
        hazard_warnings=hazard_warnings,
        rank_summaries=tuple(summaries),
        failures=tuple(failures),
    )
