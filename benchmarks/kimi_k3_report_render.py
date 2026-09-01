"""One function per generated block of the Task 11b report, keyed by its name.

The report's tables are generated rather than typed, for the same reason the
gate logs are parsed rather than remembered. Everything here turns the derived
evidence into the markdown one named block is replaced with; deciding what that
evidence is happens in ``kimi_k3_report_evidence.py``, which calls ``render``.
"""

from __future__ import annotations

from typing import Any

from benchmarks.kimi_k3_report_gates import TRAP_COVERAGE

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
