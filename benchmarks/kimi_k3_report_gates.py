"""Reading the B300 gate logs back, so the report quotes them rather than a memory.

Everything the artifact table in the Task 11b report says about a gate -- its
outcome, how many tests each suite ran, what a sanitizer found, how many
launches the stress plan replays -- is parsed back out of the log the gate
wrote. A number that was typed once is a number that stops being true quietly,
and these are the numbers most likely to.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / ".superpowers" / "sdd"

# The gate logs, and the source of the gate whose size is a property of the test
# rather than of its output.
#
# These are here for the same reason the A/B tables are generated: the artifact
# table at the end of the report quoted "784 passed, 1 skipped" and "180
# individually verified launches" for two rounds after the suite grew and the
# stress plan was re-cut, because a count typed beside a filename is checked by
# nobody. The counts a log states are read back out of the log.
GATE_LOGS = {
    "tests": "task-11c-tests.log",
    "stress": "task-11c-stress.log",
    "trap": "task-11c-trap.log",
}

SANITIZERS = ("memcheck", "synccheck", "racecheck")
STRESS_SOURCE = REPO_ROOT / "tests" / "test_kimi_k3_dependency_schedule_stress.py"


# The gate logs. Everything the artifact table says about them is read back out
# of them rather than remembered.
# ---------------------------------------------------------------------------

# A pytest tail, with the wrapper noise and the rank prefix already allowed for:
# ``[rank3]:====== 793 passed, 2 skipped in 208.33s (0:03:28) ======``.
_PYTEST_TAIL = re.compile(r"(?P<counts>\d+ \w+(?:, \d+ \w+)*) in \d+\.\d+s")
_PYTEST_COUNT = re.compile(r"(\d+) (\w+)")

# A verbose pytest node id, so the per-suite sizes the report quotes come from
# the run rather than from counting ``def test_`` in the source -- which
# undercounts every parametrized test and is how "7 report-evidence tests"
# outlived two rounds of that file growing.
_PYTEST_NODE = re.compile(r"(?P<file>tests/test_[A-Za-z0-9_]+\.py)::(?P<id>\S+)")

# The two shapes of diagnostic the trap gate prints uncaptured, so that what a
# trapped launch actually wrote is the artifact rather than a transcription of
# it. Both carry a JSON body, which is why the report's trap tables are
# generated from this log instead of copied out of it.
_TRAP_RACE = re.compile(r"^concurrent race: (?P<body>\{.*\})$", re.MULTILINE)

# The line the concurrent test prints after it has resolved the claim to an
# edge. The test's derivation, not a second copy of it: which edge won is a
# genuine race -- two runs of this gate named CTA 54 and CTA 26 -- so the report
# cannot narrate the winner in prose and stay true across a re-run.
_TRAP_RACE_EDGE = re.compile(
    r"^the claim named CTA (?P<block>\d+), which was waiting on "
    r"(?P<edge>\w+) unit (?P<unit>\d+); it published code (?P<code>\d+) "
    r"beside slot (?P<slot>\d+)$",
    re.MULTILINE,
)
_TRAP_INJECTION = re.compile(
    r"^injected (?P<edge>\w+): (?P<body>\{.*\})$", re.MULTILINE
)

# What each injected edge is chosen to cover. Judgement, not measurement, so it
# lives beside the renderer rather than in the log -- but it is keyed by the edge
# name the log reports, so an injection that is added or renamed without a note
# here shows up as a missing row rather than as a stale one.
TRAP_COVERAGE = {
    "gate_up_assignment": "appended counter, device scope, whole-queue target",
    "tail_publish": (
        "the cross-rank case — runs `wait_for_schedule_count_system`"
    ),
    "routed_down_gate_up": (
        "counter outside the appended region, slot indexed by expert"
    ),
}


def _pytest_outcome(path: Path) -> dict[str, Any] | None:
    """One gate's outcome, and how many ranks reported it.

    Every rank of a ``torchrun`` gate runs the same selection, so the eight
    summaries must agree; a log where they do not is a log where one rank
    silently ran something else, and the disagreement is returned rather than
    reduced away so a caller can refuse it.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    summaries = []
    for match in _PYTEST_TAIL.finditer(text):
        counts = {
            outcome: int(number)
            for number, outcome in _PYTEST_COUNT.findall(match.group("counts"))
        }
        # ``deselected`` is a property of the -k expression, not of the run, and
        # the sanitizer gates report it while the plain ones do not.
        counts.pop("deselected", None)
        summaries.append(counts)
    if not summaries:
        return None
    distinct = {tuple(sorted(one.items())) for one in summaries}
    return {
        "counts": summaries[0],
        "ranks": len(summaries),
        "agreed": len(distinct) == 1,
        "distinct": [dict(one) for one in sorted(distinct)],
        "passed": all(
            one.get("failed", 0) == 0 and one.get("error", 0) == 0
            for one in summaries
        ),
    }


def _suite_sizes(path: Path) -> dict[str, int]:
    """How many tests each file contributed, from the node ids in the log.

    Counted as distinct node ids rather than as lines, because every rank
    reports every test and a parametrized test reports once per case.
    """
    if not path.exists():
        return {}
    seen: dict[str, set[str]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in _PYTEST_NODE.finditer(text):
        seen.setdefault(match.group("file"), set()).add(match.group("id"))
    return {name: len(ids) for name, ids in sorted(seen.items())}


def _sanitizer(path: Path) -> dict[str, Any] | None:
    """The structured header a sanitizer gate writes above its raw output."""
    if not path.exists():
        return None
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("========="):
            break
        name, _, value = line.partition(": ")
        if _:
            header.setdefault(name.strip(), value.strip())
    return header


def _stress_plan(path: Path = STRESS_SOURCE) -> dict[str, Any] | None:
    """How many launches the replay stress gate verifies, from its own source.

    The oracle-checked count is the number the gate asserts its plan length
    against, so the report quotes the figure the gate would fail on rather than
    a second copy of the arithmetic. The equality leg runs one pass of the same
    grid and launches both schedules per element, which makes its launch count a
    function of that same number.
    """
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    passes: int | None = None
    tokens: tuple[int, ...] | None = None
    verified: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "STRESS_PASSES":
                    passes = ast.literal_eval(node.value)
                elif target.id == "STRESS_TOKENS":
                    tokens = ast.literal_eval(node.value)
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Call)
            and isinstance(test.left.func, ast.Name)
            and test.left.func.id == "len"
            and len(test.left.args) == 1
            and isinstance(test.left.args[0], ast.Name)
            and test.left.args[0].id == "plan"
            and len(test.comparators) == 1
        ):
            verified = ast.literal_eval(test.comparators[0])
    if passes is None or tokens is None or verified is None:
        return None
    pairs = verified // passes
    return {
        "tokens": list(tokens),
        "passes": passes,
        "oracle_verified": verified,
        "equality_pairs": pairs,
        "equality_launches": pairs * 2,
        "total": verified + pairs * 2,
    }


def gates(
    evidence_dir: Path = EVIDENCE_DIR,
    stress_source: Path = STRESS_SOURCE,
) -> dict[str, Any]:
    """What each gate log states, for the artifact table and the prose.

    Every field is absent rather than guessed when its log is: a checkout with
    the sources and not the measurement still derives the A/B tables, and the
    renderers drop the rows they have no log for.
    """
    trap_log = evidence_dir / GATE_LOGS["trap"]
    race = None
    injections: list[dict[str, Any]] = []
    if trap_log.exists():
        text = trap_log.read_text(encoding="utf-8", errors="replace")
        match = _TRAP_RACE.search(text)
        injections = [
            json.loads(one.group("body"))
            for one in _TRAP_INJECTION.finditer(text)
        ]
        resolved = _TRAP_RACE_EDGE.search(text)
        race = {
            "injections": injections,
            "claim": json.loads(match.group("body")) if match else None,
            "resolved": resolved.groupdict() if resolved else None,
        }
    return {
        **{
            name: _pytest_outcome(evidence_dir / filename)
            for name, filename in GATE_LOGS.items()
        },
        "trap_race": race,
        "trap_injections": injections,
        "suite_sizes": _suite_sizes(evidence_dir / GATE_LOGS["tests"]),
        "stress_plan": _stress_plan(stress_source),
        "sanitizers": {
            name: _sanitizer(evidence_dir / f"task-11c-{name}.log")
            for name in SANITIZERS
        },
    }
