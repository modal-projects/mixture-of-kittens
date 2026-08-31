"""Derive the task 11c report's numbers from the A/B artifacts, and check them.

Every quantity the report quotes is a reduction of two files: the A/B result
document and the retained per-repeat rank-max samples. Typing those reductions
into markdown by hand is how a corrected accounting ends up half applied -- the
first revision of the report quoted a wait share whose numerator counted the
per-edge waits and whose denominator did not, and no reader could have told,
because the two numbers were adjacent prose.

So the numeric tables are generated. Each one sits between a pair of HTML
comments naming the block, this module renders the block from the artifacts, and
``check`` compares what the report contains against what the artifacts produce.
The prose around them stays hand-written, which is the part that should be.

Latency is recomputed from the raw samples rather than read from the summary
fields, so a median or a p99 in the report is backed by the thousand samples it
came from and not by a number the harness reduced once and discarded its input
for. The cycle accounting is recomputed too, and its rule is the one the fix
established: a launch's total is the sum of its top-level bands, every share is
taken against that sum, and the per-edge waits are a refinement of the
``readiness_wait`` band rather than an addend beside it.

Usage::

    python -m benchmarks.kimi_k3_report_evidence --check
    python -m benchmarks.kimi_k3_report_evidence --write
    python -m benchmarks.kimi_k3_report_evidence --summary
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_timing import percentile

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / ".superpowers" / "sdd"
REPORT_PATH = EVIDENCE_DIR / "task-11c-report.md"
RESULTS_PATH = EVIDENCE_DIR / "task-11c-ab-results.json"
RAW_PATH = EVIDENCE_DIR / "task-11c-ab-raw-samples.json"

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

# One `results.json` per independent A/B run, named by what it was measured
# against. The repeat dispersion a single run reports is the spread between
# repeats that shared a container and a node; it says nothing about the spread
# between nodes, and the M = 128 verdict is taken against a 1% bar with a
# fraction of that in hand. So the run is repeated, and the control -- the same
# tree with this round's two timed-path edits taken back out -- is repeated
# beside it, because a shift measured against a run from a different node pool
# cannot be attributed to code without one.
RUNS_DIR = EVIDENCE_DIR / "task-11c-ab-runs"
HEAD_RUNS = ("head-primary", "head-a", "head-b", "head-c")
CONTROL_RUNS = ("control-p", "control-q")

# The two variants the A/B names them by. "candidate" is the schedule that was
# promoted; the report calls it "promoted" and the artifacts do not, because the
# artifacts were written by the run that decided it.
VARIANTS = ("production", "candidate")

# The bands the report tabulates, in the order it tabulates them. Every one is
# top-level -- the diagnostic children are reported as shares of their parent in
# a different table -- and the two wait bands come first because the whole
# experiment is about them.
REPORTED_BANDS = (
    "grid_barrier",
    "readiness_wait",
    "routed_gate_up",
    "routed_down",
    "tail",
    "publish",
    "shared_experts",
    "router_score",
    "latent_project",
    "latent_quantize",
    "assignment",
    "routed_queue",
)

BLOCK_PATTERN = re.compile(
    r"<!-- generated: (?P<name>[a-z0-9_]+) -->\n"
    r"(?P<body>.*?)"
    r"<!-- end: (?P=name) -->",
    re.DOTALL,
)


def _shapes(results: dict[str, Any]) -> list[int]:
    return sorted(int(point["tokens"]) for point in results["points"])


def _point(results: dict[str, Any], tokens: int) -> dict[str, Any]:
    for point in results["points"]:
        if int(point["tokens"]) == tokens:
            return point
    raise KeyError(f"no A/B point at M{tokens}")


def latency(raw: dict[str, Any], tokens: int) -> dict[str, dict[str, Any]]:
    """Repeat medians, their center and dispersion, and repeat p99s.

    Recomputed from the samples, in the order the repeats ran. The center is the
    median of the repeat medians and the dispersion is their full range, which
    is the effect band the verdict is taken against: a gain smaller than the
    spread between repeats of the same variant is not a gain.
    """
    derived: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        entries = sorted(
            raw[str(tokens)][variant], key=lambda entry: int(entry["repeat"])
        )
        medians = [
            percentile(entry["rank_max_samples_ms"], 0.5) for entry in entries
        ]
        p99s = [
            percentile(entry["rank_max_samples_ms"], 0.99) for entry in entries
        ]
        derived[variant] = {
            "sample_counts": [
                len(entry["rank_max_samples_ms"]) for entry in entries
            ],
            "repeat_medians_ms": medians,
            "repeat_p99s_ms": p99s,
            "center_ms": percentile(medians, 0.5),
            "dispersion_ms": max(medians) - min(medians),
            "p99_ms": percentile(p99s, 0.5),
        }
    production = derived["production"]
    candidate = derived["candidate"]
    derived["change"] = {
        "median_gain_fraction": (
            (production["center_ms"] - candidate["center_ms"])
            / production["center_ms"]
        ),
        "median_gain_ms": production["center_ms"] - candidate["center_ms"],
        "p99_change_fraction": (
            (candidate["p99_ms"] - production["p99_ms"]) / production["p99_ms"]
        ),
        "dispersion_multiple": (
            abs(production["center_ms"] - candidate["center_ms"])
            / max(production["dispersion_ms"], candidate["dispersion_ms"])
            if max(production["dispersion_ms"], candidate["dispersion_ms"])
            else float("inf")
        ),
    }
    return derived


def cycles(results: dict[str, Any], tokens: int) -> dict[str, Any]:
    """One shape's accumulated CTA cycles, and the shares taken against them.

    The total is the sum of the top-level bands and nothing else. The waiting is
    ``grid_barrier + readiness_wait``, which is inside that total for both
    variants, so the share is a share of a denominator that contains its own
    numerator. ``edge_wait`` is reported beside it as the split of the readiness
    band by edge, and is checked against it rather than added to it.
    """
    point = _point(results, tokens)
    derived: dict[str, Any] = {}
    for variant in VARIANTS:
        profile = point["profiles"][variant]
        bands = profile["phase_clock_cycles"]
        top_level = list(profile["phase_clock_top_level"])
        total = sum(int(bands[name]) for name in top_level)
        wait = int(bands["grid_barrier"]) + int(bands["readiness_wait"])
        edge_wait = sum(
            int(value)
            for value in profile.get("edge_wait_cycles", {}).values()
        )
        derived[variant] = {
            "bands": {name: int(value) for name, value in bands.items()},
            "top_level": top_level,
            "total_cycles": total,
            "wait_cycles": wait,
            "edge_wait_cycles": edge_wait,
            "wait_fraction": wait / total if total else 0.0,
            "barrier_fraction": (
                int(bands["grid_barrier"]) / total if total else 0.0
            ),
            "reported_total_cycles": int(profile["phase_clock_total_cycles"]),
            "reported_wait_cycles": int(profile["wait_cycles"]),
        }
    return derived


def edges(results: dict[str, Any], tokens: int) -> list[dict[str, Any]]:
    """Per-edge accumulated wait and longest single wait, largest first."""
    profile = _point(results, tokens)["profiles"]["candidate"]
    waits = {
        name: int(value)
        for name, value in profile["edge_wait_cycles"].items()
    }
    makespans = {
        name: int(value)
        for name, value in profile["edge_makespan_cycles"].items()
    }
    accumulated = sum(waits.values())
    return [
        {
            "edge": name,
            "accumulated_cycles": waits[name],
            "share_of_edge_wait": (
                waits[name] / accumulated if accumulated else 0.0
            ),
            "longest_single_cycles": makespans.get(name, 0),
        }
        for name in sorted(waits, key=lambda name: -waits[name])
    ]


def verdicts(results: dict[str, Any], tokens: int) -> dict[str, Any]:
    """One shape's gate outcomes, as the harness decided them.

    Both bars are per-shape verdicts on the same measurement, and a shape the
    harness did not gate on reports neither. The report's headline table is
    generated from these rather than from prose, because the 8% gate failing
    while the 2% bar passed is the result most likely to be softened by hand.
    """
    point = _point(results, tokens)
    return {
        "gating": bool(point["gating"]),
        "experiment_gate_passed": bool(point["passed"]),
        "promotion_passed": bool(point["promotion_passed"]),
    }


def queues(results: dict[str, Any], tokens: int) -> list[dict[str, Any]]:
    """Cumulative queue makespans, in the order the queues are scanned."""
    profile = _point(results, tokens)["profiles"]["candidate"]
    names = list(results["schedule"]["queues"])
    makespans = profile["queue_makespan_cycles"]
    step = max(int(value) for value in makespans.values())
    return [
        {
            "queue": name,
            "cycles": int(makespans[name]),
            "share_of_step": int(makespans[name]) / step if step else 0.0,
        }
        for name in names
    ]


def _median(values: list[float]) -> float:
    return percentile(sorted(values), 0.5)


def run_spread(runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    """Each independent run's per-shape delta, and the two groups' spreads.

    Returns ``None`` when the per-run directory is absent, so a checkout with
    only the primary artifacts still derives everything else.

    The ``arm`` of a run is ``head`` or ``control``. Both arms ran within the
    same hour on the same pool, which is what makes the difference between their
    medians attributable to the two edits rather than to the pool.
    """
    if not runs_dir.is_dir():
        return None
    arms: dict[str, list[dict[str, Any]]] = {"head": [], "control": []}
    for arm, names in (("head", HEAD_RUNS), ("control", CONTROL_RUNS)):
        for name in names:
            path = runs_dir / f"{name}.json"
            if not path.exists():
                continue
            results = json.loads(path.read_text(encoding="utf-8"))
            arms[arm].append(
                {
                    "name": name,
                    "deltas": {
                        int(point["tokens"]): float(
                            point["improvement_fraction"]
                        )
                        for point in results["points"]
                    },
                    "production_ms": {
                        int(point["tokens"]): float(
                            point["production_median_ms"]
                        )
                        for point in results["points"]
                    },
                    "promotion_passed": bool(
                        results["decision"]["promotion_passed"]
                    ),
                }
            )
    if not arms["head"]:
        return None
    shapes = sorted(arms["head"][0]["deltas"])
    summarised: dict[str, Any] = {"shapes": shapes, "arms": {}}
    for arm, entries in arms.items():
        if not entries:
            continue
        summarised["arms"][arm] = {
            "runs": entries,
            "count": len(entries),
            "by_shape": {
                tokens: {
                    "values": [entry["deltas"][tokens] for entry in entries],
                    "median": _median(
                        [entry["deltas"][tokens] for entry in entries]
                    ),
                    "spread": (
                        max(entry["deltas"][tokens] for entry in entries)
                        - min(entry["deltas"][tokens] for entry in entries)
                    ),
                    "production_median_ms": _median(
                        [entry["production_ms"][tokens] for entry in entries]
                    ),
                }
                for tokens in shapes
            },
            "all_promoted": all(entry["promotion_passed"] for entry in entries),
        }
    return summarised


# ---------------------------------------------------------------------------
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


def derive(
    results: dict[str, Any],
    raw: dict[str, Any],
    gate_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every quantity the report quotes, keyed by shape."""
    shapes = _shapes(results)
    return {
        "shapes": shapes,
        "gates": gates() if gate_data is None else gate_data,
        "barriers_per_launch": dict(results["barriers_per_launch"]),
        "residency": results["residency"],
        "experiment_gate_passed": bool(
            results["decision"]["experiment_gate_passed"]
        ),
        "promotion_passed": bool(results["decision"]["promotion_passed"]),
        "latency": {tokens: latency(raw, tokens) for tokens in shapes},
        "verdicts": {tokens: verdicts(results, tokens) for tokens in shapes},
        "cycles": {tokens: cycles(results, tokens) for tokens in shapes},
        "edges": {tokens: edges(results, tokens) for tokens in shapes},
        "queues": {tokens: queues(results, tokens) for tokens in shapes},
        "run_spread": run_spread(),
    }


# ---------------------------------------------------------------------------
# Rendering. One function per generated block, keyed by the block's name.
# ---------------------------------------------------------------------------


def _milliseconds(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def _percent(fraction: float, digits: int = 2) -> str:
    """Signed, so a table never leaves the direction of a change to the prose.

    With a real minus sign, because the prose around these tables uses one and a
    figure that has to be recognised in both places should look the same in both.
    """
    return f"{fraction:+.{digits}%}".replace("-", "\N{MINUS SIGN}")


def _thousands(value: int) -> str:
    return f"{value:,}"


def _render_latency(derived: dict[str, Any]) -> str:
    lines = [
        "| Shape | variant | repeat medians (ms) | center | dispersion |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tokens in derived["shapes"]:
        for variant in VARIANTS:
            one = derived["latency"][tokens][variant]
            medians = " ".join(
                _milliseconds(value, 4) for value in one["repeat_medians_ms"]
            )
            shape = f"M = {tokens}" if variant == VARIANTS[0] else ""
            label = "production" if variant == "production" else "promoted"
            lines.append(
                f"| {shape} | {label} | {medians} | "
                f"{_milliseconds(one['center_ms'])} | "
                f"{_milliseconds(one['dispersion_ms'])} |"
            )
    return "\n".join(lines) + "\n"


def _render_p99(derived: dict[str, Any]) -> str:
    lines = [
        "| Shape | production | promoted | change |",
        "| --- | --- | --- | --- |",
    ]
    for tokens in derived["shapes"]:
        one = derived["latency"][tokens]
        lines.append(
            f"| M = {tokens} | {_milliseconds(one['production']['p99_ms'])} ms "
            f"| {_milliseconds(one['candidate']['p99_ms'])} ms | "
            f"{_percent(one['change']['p99_change_fraction'])} |"
        )
    return "\n".join(lines) + "\n"


def _verdict(passed: bool, gating: bool) -> str:
    if not gating:
        return "reported"
    return "PASS" if passed else "**FAIL**"


def _render_medians(derived: dict[str, Any]) -> str:
    lines = [
        "| Shape | production median | promoted median | median change "
        "| p99 change | 8% gate | 2% promotion bar |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tokens in derived["shapes"]:
        one = derived["latency"][tokens]
        verdict = derived["verdicts"][tokens]
        lines.append(
            f"| M = {tokens} | "
            f"{_milliseconds(one['production']['center_ms'])} ms | "
            f"{_milliseconds(one['candidate']['center_ms'])} ms | "
            f"{_percent(one['change']['median_gain_fraction'])} | "
            f"{_percent(one['change']['p99_change_fraction'])} | "
            f"{_verdict(verdict['experiment_gate_passed'], verdict['gating'])}"
            f" | "
            f"{_verdict(verdict['promotion_passed'], verdict['gating'])} |"
        )
    return "\n".join(lines) + "\n"


def _render_cycles(derived: dict[str, Any], tokens: int) -> str:
    production = derived["cycles"][tokens]["production"]
    candidate = derived["cycles"][tokens]["candidate"]

    def row(label: str, left: int, right: int) -> str:
        change = (right - left) / left if left else 0.0
        return (
            f"| `{label}` | {_thousands(left)} | {_thousands(right)} | "
            f"{_percent(change)} |"
        )

    lines = [
        "| Region | production | promoted | change |",
        "| --- | --- | --- | --- |",
    ]
    for band in REPORTED_BANDS:
        lines.append(
            row(band, production["bands"][band], candidate["bands"][band])
        )
    lines.append(
        f"| **all waiting** | **{_thousands(production['wait_cycles'])}** | "
        f"**{_thousands(candidate['wait_cycles'])}** | "
        f"**{_percent((candidate['wait_cycles'] - production['wait_cycles']) / production['wait_cycles'])}** |"
    )
    lines.append(
        f"| ten readiness edges *(inside `readiness_wait`)* | — | "
        f"{_thousands(candidate['edge_wait_cycles'])} | — |"
    )
    lines.append(
        f"| **total of the top-level bands** | "
        f"**{_thousands(production['total_cycles'])}** | "
        f"**{_thousands(candidate['total_cycles'])}** | "
        f"**{_percent((candidate['total_cycles'] - production['total_cycles']) / production['total_cycles'])}** |"
    )
    return "\n".join(lines) + "\n"


def _render_wait_shares(derived: dict[str, Any]) -> str:
    lines = [
        "| Shape | production waiting | promoted waiting | change "
        "| barrier share | wait share |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for tokens in derived["shapes"]:
        production = derived["cycles"][tokens]["production"]
        candidate = derived["cycles"][tokens]["candidate"]
        change = (
            candidate["wait_cycles"] - production["wait_cycles"]
        ) / production["wait_cycles"]
        lines.append(
            f"| M = {tokens} | {_thousands(production['wait_cycles'])} | "
            f"{_thousands(candidate['wait_cycles'])} | {_percent(change)} | "
            f"{production['barrier_fraction']:.1%} → "
            f"{candidate['barrier_fraction']:.1%} | "
            f"{production['wait_fraction']:.1%} → "
            f"{candidate['wait_fraction']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def _render_edges(derived: dict[str, Any], tokens: int) -> str:
    lines = [
        "| Edge | accumulated | share of readiness wait | longest single |",
        "| --- | --- | --- | --- |",
    ]
    for entry in derived["edges"][tokens]:
        lines.append(
            f"| `{entry['edge']}` | "
            f"{_thousands(entry['accumulated_cycles'])} | "
            f"{entry['share_of_edge_wait']:.1%} | "
            f"{_thousands(entry['longest_single_cycles'])} |"
        )
    return "\n".join(lines) + "\n"


def _render_queues(derived: dict[str, Any]) -> str:
    shapes = derived["shapes"]
    first, last = shapes[0], shapes[-1]
    lines = [
        f"| Queue | M = {first} | share | M = {last} | share |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_queue = {
        tokens: {entry["queue"]: entry for entry in derived["queues"][tokens]}
        for tokens in (first, last)
    }
    for entry in derived["queues"][first]:
        name = entry["queue"]
        wide = by_queue[last][name]
        lines.append(
            f"| `{name}` | {_thousands(entry['cycles'])} | "
            f"{entry['share_of_step']:.1%} | {_thousands(wide['cycles'])} | "
            f"{wide['share_of_step']:.1%} |"
        )
    return "\n".join(lines) + "\n"


_ARM_LABELS = {
    "head": "head",
    "control": "control *(this round's two timed-path edits removed)*",
}


def _render_runs(derived: dict[str, Any]) -> str:
    spread = derived["run_spread"]
    if spread is None:
        raise KeyError("no per-run artifacts to render a spread from")
    lines = [
        "| Arm | runs | Shape | per-run delta | median | spread "
        "| production median |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ("head", "control"):
        if arm not in spread["arms"]:
            continue
        one = spread["arms"][arm]
        for position, tokens in enumerate(spread["shapes"]):
            shape = one["by_shape"][tokens]
            values = " ".join(_percent(value, 2) for value in shape["values"])
            label = _ARM_LABELS[arm] if position == 0 else ""
            count = str(one["count"]) if position == 0 else ""
            lines.append(
                f"| {label} | {count} | M = {tokens} | {values} | "
                f"{_percent(shape['median'])} | {shape['spread']:.2%} | "
                f"{_milliseconds(shape['production_median_ms'])} ms |"
            )
    return "\n".join(lines) + "\n"


def _render_raw_audit(derived: dict[str, Any]) -> str:
    """Every published quantile beside the samples it was taken from.

    Not a table: the point is that a reader can run the same reduction over
    ``task-11c-ab-raw-samples.json`` and get these digits, so it is printed the
    way that recomputation prints.
    """
    lines = ["```"]
    for tokens in derived["shapes"]:
        for variant in VARIANTS:
            one = derived["latency"][tokens][variant]
            counts = set(one["sample_counts"])
            lines.append(
                f"M{tokens:>3} {variant:11} "
                f"repeats={len(one['repeat_medians_ms'])} "
                f"n={'/'.join(str(count) for count in sorted(counts))}  "
                f"median {one['center_ms']:.6f}  p99 {one['p99_ms']:.6f}"
            )
    lines.append("```")
    return "\n".join(lines) + "\n"


def _sample_total(derived: dict[str, Any]) -> int:
    """How many rank-max samples the raw document actually holds."""
    return sum(
        sum(derived["latency"][tokens][variant]["sample_counts"])
        for tokens in derived["shapes"]
        for variant in VARIANTS
    )


def _render_artifacts(derived: dict[str, Any]) -> str:
    """The artifact table, with every count read from the file it describes.

    The descriptions that are not counts stay written here rather than in the
    report, because this block replaces the whole table and a row that named
    only its file would tell a reader less than the row it replaced.
    """
    gate = derived["gates"]
    stress_plan = gate["stress_plan"]
    lines = ["| Artifact | What |", "| --- | --- |"]

    def row(filename: str, what: str) -> None:
        lines.append(f"| `{filename}` | {what} |")

    row(
        "task-11c-ab-results.json",
        "the A/B: latency, corrected profiles, edges, makespans, both verdicts",
    )
    row("task-11c-ab-manifest.json", "the pre-registered plan and gate")
    row(
        "task-11c-ab-raw-samples.json",
        f"all {_thousands(_sample_total(derived))} rank-max samples, for an "
        "independent p99",
    )
    if gate["tests"]:
        counts = gate["tests"]["counts"]
        stated = ", ".join(
            f"{counts[outcome]} {outcome}"
            for outcome in ("passed", "skipped", "failed", "error")
            if counts.get(outcome)
        )
        row(
            "task-11c-tests.log",
            f"{stated}, on all {gate['tests']['ranks']} ranks",
        )
    if gate["stress"] and stress_plan:
        row(
            "task-11c-stress.log",
            f"{stress_plan['total']} individually verified launches, "
            f"{gate['stress']['ranks']} ranks",
        )
    if gate["trap"] and gate["trap_race"]:
        race = gate["trap_race"]
        injected = len(race["injections"])
        blocks_racing = race["claim"]["blocks"] if race["claim"] else 0
        row(
            "task-11c-trap.log",
            f"{injected} injected edges trapping at their named sites, and "
            f"{blocks_racing} CTAs racing to report one pair",
        )
    for name in ("memcheck", "synccheck"):
        header = gate["sanitizers"].get(name)
        if not header:
            continue
        row(
            f"task-11c-{name}.log",
            f"{header['device_errors']} device errors, "
            f"{header['hazards']} hazards, "
            f"{header['rank_summaries']}/8 ranks",
        )
    racecheck = gate["sanitizers"].get("racecheck")
    if racecheck:
        row(
            "task-11c-racecheck.log",
            f"the refusal, and the {racecheck['rank_summaries']} ranks that "
            "caused it",
        )
    row(
        "task-11c-sass.json",
        "four instantiations, no spill, one CTA per SM",
    )
    return "\n".join(lines) + "\n"


def _render_trap_results(derived: dict[str, Any]) -> str:
    """One row per injected edge, from the diagnostics the trapped launch wrote.

    ``expected`` and ``recorded`` are both in the log, and the row states the
    recorded pair only when the two agree; a disagreement is what the gate
    exists to catch, so it is rendered loudly rather than reduced to one number.
    """
    lines = [
        "| Injected edge | Shape it covers | Result |",
        "| --- | --- | --- |",
    ]
    for one in derived["gates"]["trap_injections"]:
        edge = one["edge"]
        matched = (
            one["recorded_code"] == one["expected_code"]
            and one["recorded_slot"] == one["expected_slot"]
        )
        result = (
            f"code {one['recorded_code']}, slot {one['recorded_slot']}, "
            f"claimed by CTA {one['claiming_block']}, "
            + ("launch failed" if one["launch_failed"] else "**launch returned**")
        )
        if not matched:
            result = (
                f"**recorded ({one['recorded_code']}, {one['recorded_slot']}) "
                f"but expected ({one['expected_code']}, "
                f"{one['expected_slot']})**"
            )
        lines.append(
            f"| `{edge}` | {TRAP_COVERAGE.get(edge, '—')} | {result} |"
        )
    return "\n".join(lines) + "\n"


def _render_trap_race(derived: dict[str, Any]) -> str:
    """The concurrent injection's claim, and the pair it resolves to.

    The derivation from claim to edge and unit is the test's, repeated here from
    the same three words the test read, so the report's arithmetic is visible
    rather than asserted.
    """
    race = derived["gates"]["trap_race"]
    claim = race["claim"] if race else None
    if not claim:
        return "```\nno concurrent injection in the trap log\n```\n"
    block = claim["claiming_block"]
    edges_total = claim["edge_count"]
    lines = [
        "```",
        f"concurrent race: blocks {claim['blocks']}, claim {claim['claim']} "
        f"\N{RIGHTWARDS ARROW} CTA {block}, code {claim['recorded_code']}, "
        f"slot {claim['recorded_slot']}, "
        + ("launch failed" if claim["launch_failed"] else "LAUNCH RETURNED"),
        f"CTA {block} of an edge-major grid is edge {block} % {edges_total} = "
        f"{block % edges_total} at unit {block} / {edges_total} = "
        f"{block // edges_total}",
    ]
    resolved = race.get("resolved")
    if resolved:
        lines.append(
            f"that pair is {resolved['edge']} unit {resolved['unit']}'s, and "
            f"only that edge carries code {resolved['code']}"
        )
    lines.append("```")
    return "\n".join(lines) + "\n"


def render(name: str, derived: dict[str, Any]) -> str:
    """One generated block, by name."""
    shapes = derived["shapes"]
    renderers = {
        "trap_results": lambda: _render_trap_results(derived),
        "trap_race": lambda: _render_trap_race(derived),
        "medians": lambda: _render_medians(derived),
        "latency": lambda: _render_latency(derived),
        "p99": lambda: _render_p99(derived),
        "wait_shares": lambda: _render_wait_shares(derived),
        "queues": lambda: _render_queues(derived),
        "runs": lambda: _render_runs(derived),
        "raw_audit": lambda: _render_raw_audit(derived),
        "artifacts": lambda: _render_artifacts(derived),
    }
    for tokens in shapes:
        renderers[f"cycles_m{tokens}"] = (
            lambda tokens=tokens: _render_cycles(derived, tokens)
        )
        renderers[f"edges_m{tokens}"] = (
            lambda tokens=tokens: _render_edges(derived, tokens)
        )
    if name not in renderers:
        raise KeyError(f"no generated block named {name}")
    return renderers[name]()


# ---------------------------------------------------------------------------
# Checking and rewriting the report.
# ---------------------------------------------------------------------------


def blocks(text: str) -> dict[str, str]:
    """The generated blocks a report contains, by name."""
    found: dict[str, str] = {}
    for match in BLOCK_PATTERN.finditer(text):
        name = match.group("name")
        if name in found:
            raise ValueError(f"the report generates {name} twice")
        found[name] = match.group("body")
    return found


def phrases(derived: dict[str, Any]) -> dict[str, str]:
    """The prose claims that are numbers, as the exact text the artifacts imply.

    Generating the tables does not protect the sentences around them, and the
    sentences are where a stale number survives longest: a reader who checks the
    headline gain against the table has no way to check "38.4% of the step's
    waiting" against anything. So the handful of prose figures that restate a
    derived quantity are pinned as literal substrings the report must contain.

    Deliberately few, and only quantities that follow from the artifacts. A
    claim that needs judgement -- which edge is the failing one, what the growth
    in ``mma_issue`` means -- stays prose and is not pinned here.
    """
    first = derived["shapes"][0]
    last = derived["shapes"][-1]
    narrow = derived["latency"][first]
    wide = derived["latency"][last]
    narrow_cycles = derived["cycles"][first]
    wait_change = (
        narrow_cycles["candidate"]["wait_cycles"]
        - narrow_cycles["production"]["wait_cycles"]
    ) / narrow_cycles["production"]["wait_cycles"]
    barrier_change = (
        narrow_cycles["candidate"]["bands"]["grid_barrier"]
        - narrow_cycles["production"]["bands"]["grid_barrier"]
    ) / narrow_cycles["production"]["bands"]["grid_barrier"]
    return {
        "headline_waiting": f"{abs(wait_change):.1%} of the step's waiting",
        "headline_gain": (
            f"{narrow['change']['median_gain_fraction']:.2%} faster at the "
            f"M = {first} median"
        ),
        "headline_p99": (
            f"{abs(narrow['change']['p99_change_fraction']):.2%} better p99"
        ),
        "narrow_gain_ms": (
            f"M = {first} gains {narrow['change']['median_gain_ms']:.4f} ms, "
            f"{narrow['change']['dispersion_multiple']:.0f}×"
        ),
        "wide_regression_ms": (
            f"regression is "
            f"{abs(wide['change']['median_gain_ms']):.4f} ms"
        ),
        "barrier_share": (
            f"barrier share at M = {first} is "
            f"**{narrow_cycles['production']['barrier_fraction']:.2%}**"
        ),
        "scoreboard": (
            f"**5 → 1 barriers, {_percent(wait_change, 1)} waiting, "
            f"{_percent(barrier_change, 1)} barrier time**"
        ),
        **_gate_phrases(derived["gates"]),
    }


def _gate_phrases(gate: dict[str, Any]) -> dict[str, str]:
    """The prose figures that restate what a gate log or a gate's plan says.

    Same reason as the derived ones above, one step removed: the suite's size
    and the stress plan's size are quoted in four places between §1.2, §4 and
    §7, and they moved twice this round.
    """
    pinned: dict[str, str] = {}
    if gate["tests"] and gate["tests"]["agreed"]:
        counts = gate["tests"]["counts"]
        # Not the rank clause: the report spells eight as a word there, and the
        # rank count is already generated into the artifact table.
        pinned["suite_outcome"] = (
            f"**{counts['passed']} passed, {counts.get('skipped', 0)} skipped"
        )
    sizes = gate.get("suite_sizes") or {}
    named = (
        ("source contracts", "test_kimi_k3_dependency_schedule_source.py"),
        ("device tests", "test_kimi_k3_dependency_schedule.py"),
        ("report-evidence tests", "test_kimi_k3_report_evidence.py"),
    )
    parts = [
        f"{sizes[f'tests/{filename}']} {label}"
        for label, filename in named
        if f"tests/{filename}" in sizes
    ]
    if len(parts) == len(named):
        pinned["suite_composition"] = (
            f"**{', '.join(parts[:-1])} and {parts[-1]}**"
        )
    plan = gate["stress_plan"]
    if plan:
        pinned["stress_total"] = (
            f"**{plan['total']} individually verified launches**"
        )
        pinned["stress_oracle_leg"] = (
            f"{plan['oracle_verified']} launches synchronized and checked "
            "against the oracle"
        )
        pinned["stress_equality_leg"] = (
            f"{plan['equality_launches']} more ({plan['equality_pairs']} pairs)"
        )
        pinned["stress_shapes"] = (
            "Shapes `"
            + ", ".join(str(one) for one in plan["tokens"])
            + "` in that order"
        )
    return pinned


def _collapse(text: str) -> str:
    """One line, single-spaced.

    The report is hard-wrapped, so a pinned figure and the words around it are
    routinely split across a line break. Matching the collapsed text is what
    makes the check about the sentence rather than about where it happened to
    wrap.
    """
    return " ".join(text.split())


def check_blocks(text: str, derived: dict[str, Any]) -> list[str]:
    """Name every generated block the report and the artifacts disagree on."""
    problems = []
    for name, body in blocks(text).items():
        expected = render(name, derived)
        if body != expected:
            problems.append(
                f"{name}: the report does not match the artifacts\n"
                f"--- report\n{body}--- artifacts\n{expected}"
            )
    return problems


def check_phrases(text: str, derived: dict[str, Any]) -> list[str]:
    """Name every pinned prose figure the report no longer contains."""
    collapsed = _collapse(text)
    return [
        f"{name}: the report does not say what the artifacts imply\n"
        f"--- expected the text to contain\n{phrase}\n"
        for name, phrase in phrases(derived).items()
        if _collapse(phrase) not in collapsed
    ]


def check(text: str, derived: dict[str, Any]) -> list[str]:
    """Everything the report and the artifacts disagree on, tables and prose."""
    return check_blocks(text, derived) + check_phrases(text, derived)


def rewrite(text: str, derived: dict[str, Any]) -> str:
    """Replace every generated block with what the artifacts produce."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        return (
            f"<!-- generated: {name} -->\n"
            f"{render(name, derived)}"
            f"<!-- end: {name} -->"
        )

    return BLOCK_PATTERN.sub(replace, text)


def summary(derived: dict[str, Any]) -> str:
    """The one-screen evidence summary, derived rather than transcribed."""
    lines = [
        "task 11c evidence, derived from the A/B artifacts",
        "",
        f"barriers per launch: {derived['barriers_per_launch']}",
        f"experiment gate (>= 8% at M16): "
        f"{'PASS' if derived['experiment_gate_passed'] else 'FAIL'}",
        f"promotion bar (>= 2% at M16, <= 1% at M128): "
        f"{'PASS' if derived['promotion_passed'] else 'FAIL'}",
        "",
    ]
    for tokens in derived["shapes"]:
        one = derived["latency"][tokens]
        cycle = derived["cycles"][tokens]
        lines += [
            f"M = {tokens}",
            f"  median      {one['production']['center_ms']:.5f} ms -> "
            f"{one['candidate']['center_ms']:.5f} ms "
            f"({one['change']['median_gain_fraction']:+.2%})",
            f"  p99         {one['production']['p99_ms']:.5f} ms -> "
            f"{one['candidate']['p99_ms']:.5f} ms "
            f"({one['change']['p99_change_fraction']:+.2%})",
            f"  dispersion  {one['production']['dispersion_ms']:.5f} ms / "
            f"{one['candidate']['dispersion_ms']:.5f} ms "
            f"({one['change']['dispersion_multiple']:.1f}x the gain)",
            f"  CTA cycles  {cycle['production']['total_cycles']:,} -> "
            f"{cycle['candidate']['total_cycles']:,}",
            f"  waiting     {cycle['production']['wait_cycles']:,} "
            f"({cycle['production']['wait_fraction']:.1%}) -> "
            f"{cycle['candidate']['wait_cycles']:,} "
            f"({cycle['candidate']['wait_fraction']:.1%})",
            f"  edge waits  {cycle['candidate']['edge_wait_cycles']:,} "
            f"inside a readiness band of "
            f"{cycle['candidate']['bands']['readiness_wait']:,}",
            "",
        ]
    spread = derived["run_spread"]
    if spread is not None:
        lines.append("between independent runs")
        for arm, one in spread["arms"].items():
            lines.append(
                f"  {arm} ({one['count']} runs, "
                f"all promoted: {one['all_promoted']})"
            )
            for tokens in spread["shapes"]:
                shape = one["by_shape"][tokens]
                lines.append(
                    f"    M{tokens:>4}  median {shape['median']:+.2%}  "
                    f"spread {shape['spread']:.2%}  "
                    f"production {shape['production_median_ms']:.5f} ms"
                )
        lines.append("")
    lines += _gate_summary(derived["gates"])
    return "\n".join(lines)


def _gate_summary(gate: dict[str, Any]) -> list[str]:
    """The gate counts, so the summary artifact carries them beside the timings.

    The report's artifact table is generated from the same readings; this is the
    one-screen form, and it exists so that a reader who has the summary and not
    the report still sees which counts the logs actually state.
    """
    lines = ["gates, as their logs state them"]
    for name in GATE_LOGS:
        one = gate.get(name)
        if one is None:
            lines.append(f"  {name:9} no log in this checkout")
            continue
        stated = ", ".join(
            f"{count} {outcome}" for outcome, count in sorted(one["counts"].items())
        )
        agreement = "" if one["agreed"] else "  RANKS DISAGREE"
        # The trap gate is deliberately one process with no group (§4.3), so its
        # single summary is not an incomplete eight.
        ranks = one["ranks"]
        lines.append(
            f"  {name:9} {stated}, "
            f"{ranks} {'rank' if ranks == 1 else 'ranks'}{agreement}"
        )
    plan = gate["stress_plan"]
    if plan:
        lines.append(
            f"  stress plan {plan['total']} launches "
            f"({plan['oracle_verified']} oracle-checked + "
            f"{plan['equality_launches']} paired), shapes {plan['tokens']}"
        )
    race = gate["trap_race"]
    if race and race["claim"]:
        claim = race["claim"]
        lines.append(
            f"  trap race  {claim['blocks']} CTAs over "
            f"{claim['edge_count']} sites -> CTA {claim['claiming_block']} "
            f"published code {claim['recorded_code']} at slot "
            f"{claim['recorded_slot']}"
        )
    for name, header in gate["sanitizers"].items():
        if header is None:
            continue
        lines.append(
            f"  {name:9} {header['verdict']}, "
            f"{header['device_errors']} device errors, "
            f"{header['hazards']} hazards, "
            f"{header['rank_summaries']} rank summaries"
        )
    lines.append("")
    return lines


def load(
    results_path: Path = RESULTS_PATH,
    raw_path: Path = RAW_PATH,
) -> dict[str, Any]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    return derive(results, raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)

    derived = load(arguments.results, arguments.raw)
    if arguments.summary:
        print(summary(derived))
    if arguments.write:
        text = arguments.report.read_text(encoding="utf-8")
        arguments.report.write_text(rewrite(text, derived), encoding="utf-8")
        print(f"rewrote the generated blocks of {arguments.report}")
    if arguments.check or not (arguments.write or arguments.summary):
        problems = check(
            arguments.report.read_text(encoding="utf-8"), derived
        )
        for problem in problems:
            print(problem)
        if problems:
            return 1
        print(f"{arguments.report.name}: every generated block matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
