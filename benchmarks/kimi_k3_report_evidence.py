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
import json
import re
import sys
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_report_gates import gates
from benchmarks.kimi_k3_report_render import _percent, render
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
