"""Gate arithmetic for the Kimi K3 serving-backend comparison.

Two families of gate decide whether a comparison run is a pass, and neither
needs a GPU to evaluate, so both live here and are covered by the CPU suite.

The numerical gates grade the custom kernel against the official Kimi K3
reference. The native layers are measured against the same reference and
against the custom output, but those distances are diagnostics: a serving
backend's own MXFP4 rounding is something to report, not something the custom
kernel has to reproduce.

The performance gates grade the custom kernel against whichever native layer
is faster. They need both baselines, so a single-framework archive records its
half and :func:`combine_archives` returns the verdict over the pair.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_comparison_manifest import (
    BASELINE_BACKENDS,
    BENCHMARK,
    CONCURRENCY_ONE_TOKENS,
    CUSTOM_BACKEND,
    GATE_SHAPES,
    NUMERICAL_TOLERANCES,
    P99_LIMIT_RATIO,
    ROUTER_WEIGHT_MAX_ABS,
    SHAPE_GROUPS,
)
from benchmarks.kimi_k3_timing import geometric_mean

# The one comparison the custom kernel is held to.
GATED_COMPARISON = "custom_vs_reference"

# Comparisons that are recorded and summarized but never fail a run. The first
# two are the native layer's own accuracy; the last two are stages the native
# layer exposes, compared with the reference for the same reason.
DIAGNOSTIC_COMPARISONS = (
    "custom_vs_native",
    "native_vs_reference",
    "routed_latent_vs_reference",
    "shared_output_vs_reference",
)


# --------------------------------------------------------------------------
# Numerical gates
# --------------------------------------------------------------------------


def _row_label(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": row.get("mode"),
        "tokens": row.get("tokens"),
        "pool_index": row.get("pool_index"),
    }


# Every distance comparison has to report all three, on every row.
COMPARISON_METRICS = ("relative_l1", "cosine_similarity", "max_abs")

# The comparisons whose finiteness finding is a gate input rather than a
# derived statistic. The two stage comparisons are exposed by only one of the
# native layers' internals and carry distances alone.
FINITENESS_COMPARISONS = (
    "custom_vs_reference",
    "custom_vs_native",
    "native_vs_reference",
)

ROUTER_FIELDS = (
    "expert_ids_match",
    "expert_id_mismatch_count",
    "router_weight_max_abs",
)


def _within_gate_tolerances(stats: Mapping[str, Any]) -> bool:
    """Whether one comparison would clear the gated tolerances.

    Applied to the diagnostics too, which is how a native layer's own miss
    gets counted and reported without failing anything.
    """
    return (
        bool(stats["finite"])
        and float(stats["relative_l1"]) <= NUMERICAL_TOLERANCES["relative_l1"]
        and float(stats["cosine_similarity"])
        >= NUMERICAL_TOLERANCES["cosine_similarity"]
        and float(stats["max_abs"]) <= NUMERICAL_TOLERANCES["max_abs"]
    )


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    for field in ("mode", "tokens", "pool_index"):
        if row.get(field) is None:
            raise ValueError(f"parity row {dict(row)!r} has no {field}")
    return (str(row["mode"]), int(row["tokens"]), int(row["pool_index"]))


def _check_row_is_complete(row: Mapping[str, Any]) -> None:
    """Refuse a row that left any gate input out.

    A missing metric read as a default is a gate that passes because nothing
    measured it, which is the failure mode these gates exist to prevent.
    """
    label = _row_label(row)
    for comparison in (GATED_COMPARISON, *DIAGNOSTIC_COMPARISONS):
        if comparison not in row:
            raise ValueError(
                f"parity row {label} has no {comparison} comparison"
            )
        stats = row[comparison]
        for metric in COMPARISON_METRICS:
            if stats.get(metric) is None:
                raise ValueError(
                    f"parity row {label} {comparison} has no {metric}"
                )
    for comparison in FINITENESS_COMPARISONS:
        if row[comparison].get("finite") is None:
            raise ValueError(
                f"parity row {label} {comparison} has no finite finding"
            )
    if "router" not in row:
        raise ValueError(f"parity row {label} has no router comparison")
    for field in ROUTER_FIELDS:
        if row["router"].get(field) is None:
            raise ValueError(f"parity row {label} router has no {field}")


def _check_coverage(
    rows: Sequence[Mapping[str, Any]],
    expected_rows: Iterable[tuple[str, int, int]] | None,
) -> dict[str, Any]:
    """Refuse a set of rows that is not exactly the set the run promised.

    Two archives measure the same shapes against different native layers, so a
    row is unique by framework as well as by shape, while the shapes a run has
    to cover are the same for each. Duplicates are judged on the first and
    coverage on the second.
    """
    seen: set[tuple[Any, str, int, int]] = set()
    for row in rows:
        key = (row.get("framework"), *_row_key(row))
        if key in seen:
            raise ValueError(f"duplicate parity row for {key}")
        seen.add(key)
    shapes = {key[1:] for key in seen}
    if expected_rows is None:
        return {
            "expected_row_count": None,
            "measured_row_count": len(seen),
            "measured_shape_count": len(shapes),
        }
    expected = set(expected_rows)
    missing = sorted(expected - shapes)
    if missing:
        raise ValueError(f"missing parity rows for {missing}")
    unexpected = sorted(shapes - expected)
    if unexpected:
        raise ValueError(f"unexpected parity rows for {unexpected}")
    return {
        "expected_row_count": len(expected),
        "measured_row_count": len(seen),
        "measured_shape_count": len(shapes),
    }


def expected_parity_rows(
    shape_groups: Mapping[str, Sequence[int]],
    pool_size: int,
) -> list[tuple[str, int, int]]:
    """Every ``(mode, tokens, pool entry)`` a run of this sweep must measure."""
    return [
        (mode, int(tokens), pool_index)
        for mode, shapes in shape_groups.items()
        for tokens in shapes
        for pool_index in range(pool_size)
    ]


def evaluate_numerical_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: Iterable[tuple[str, int, int]] | None = None,
) -> dict[str, Any]:
    """Enforce the official-reference tolerances on every parity row.

    Six checks per row: the custom output is finite, its relative L1, cosine,
    and maximum absolute distance from the reference are inside tolerance, the
    top-16 identities are exactly the reference's, and the selected weights are
    within ``1e-5`` of it. The two router checks are exactness rather than
    tolerance because a routing decision is discrete -- a different expert is a
    different computation, not a rounding difference -- so unlike the output
    comparisons they are required of the native selection as well.

    Everything the gates read has to be there. An absent row set, an absent
    metric, a duplicated shape, or a shape the sweep promised and never
    measured are all raised rather than scored, because each of them is a way
    for a run that measured less than it claimed to report a pass.
    """
    if not rows:
        raise ValueError("no parity rows to gate")
    coverage = _check_coverage(rows, expected_rows)

    evaluated: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for row in rows:
        _check_row_is_complete(row)
        gated = row[GATED_COMPARISON]
        router = row["router"]
        checks = {
            "finite": {
                "value": bool(gated["finite"]),
                "limit": True,
                "passed": bool(gated["finite"]),
            },
            "relative_l1": {
                "value": float(gated["relative_l1"]),
                "limit": NUMERICAL_TOLERANCES["relative_l1"],
                "passed": float(gated["relative_l1"])
                <= NUMERICAL_TOLERANCES["relative_l1"],
            },
            "cosine_similarity": {
                "value": float(gated["cosine_similarity"]),
                "limit": NUMERICAL_TOLERANCES["cosine_similarity"],
                "passed": float(gated["cosine_similarity"])
                >= NUMERICAL_TOLERANCES["cosine_similarity"],
            },
            "max_abs": {
                "value": float(gated["max_abs"]),
                "limit": NUMERICAL_TOLERANCES["max_abs"],
                "passed": float(gated["max_abs"])
                <= NUMERICAL_TOLERANCES["max_abs"],
            },
            "router_expert_ids_exact": {
                "value": int(router["expert_id_mismatch_count"]),
                "limit": 0,
                "passed": bool(router["expert_ids_match"])
                and int(router["expert_id_mismatch_count"]) == 0,
            },
            "router_weight_max_abs": {
                "value": float(router["router_weight_max_abs"]),
                "limit": ROUTER_WEIGHT_MAX_ABS,
                "passed": float(router["router_weight_max_abs"])
                <= ROUTER_WEIGHT_MAX_ABS,
            },
        }
        label = _row_label(row)
        failed = [name for name, check in checks.items() if not check["passed"]]
        for name in failed:
            violations.append(
                {
                    **label,
                    "check": name,
                    "value": checks[name]["value"],
                    "limit": checks[name]["limit"],
                }
            )
        evaluated.append({**label, "passed": not failed, "checks": checks})

    diagnostics: dict[str, Any] = {}
    for comparison in (GATED_COMPARISON, *DIAGNOSTIC_COMPARISONS):
        present = [row[comparison] for row in rows]
        finiteness_known = comparison in FINITENESS_COMPARISONS
        diagnostics[comparison] = {
            "row_count": len(present),
            "max_relative_l1": max(
                float(stats["relative_l1"]) for stats in present
            ),
            "min_cosine_similarity": min(
                float(stats["cosine_similarity"]) for stats in present
            ),
            "max_abs": max(float(stats["max_abs"]) for stats in present),
            "rows_outside_gate_tolerances": (
                sum(
                    0 if _within_gate_tolerances(stats) else 1
                    for stats in present
                )
                if finiteness_known
                else None
            ),
            "gated": comparison == GATED_COMPARISON,
        }

    return {
        "benchmark": BENCHMARK,
        "gated_comparison": GATED_COMPARISON,
        "diagnostic_comparisons": list(DIAGNOSTIC_COMPARISONS),
        "tolerances": dict(NUMERICAL_TOLERANCES),
        "router_weight_max_abs": ROUTER_WEIGHT_MAX_ABS,
        "row_count": len(evaluated),
        "coverage": coverage,
        "rows": evaluated,
        "violations": violations,
        "diagnostics": diagnostics,
        "passed": not violations,
    }


def parity_summary(
    framework: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce every parity row to the worst value the tolerances constrain."""

    def worst(path: Sequence[str], reducer: Any) -> float:
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return reducer(values)

    comparisons = {}
    for comparison in (GATED_COMPARISON, *DIAGNOSTIC_COMPARISONS):
        comparisons[comparison] = {
            "max_relative_l1": worst((comparison, "relative_l1"), max),
            "min_cosine_similarity": worst(
                (comparison, "cosine_similarity"), min
            ),
            "max_abs": worst((comparison, "max_abs"), max),
        }
    gates = evaluate_numerical_gates(rows)
    return {
        "framework": framework,
        "row_count": len(rows),
        "router_expert_ids_all_match": all(
            row["router"]["expert_ids_match"] for row in rows
        ),
        "router_weight_max_abs": max(
            float(row["router"]["router_weight_max_abs"]) for row in rows
        ),
        "comparisons": comparisons,
        "tolerances": dict(NUMERICAL_TOLERANCES),
        "numerical_gates_passed": gates["passed"],
        "numerical_gate_violations": gates["violations"],
    }


# --------------------------------------------------------------------------
# Sample merge
# --------------------------------------------------------------------------


def merge_latency_rows(
    row_groups: Iterable[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Combine per-archive latency rows, keeping the worst custom measurement.

    The custom kernel is measured once inside each framework image, so a
    ``(mode, tokens)`` point carries two independent custom measurements. The
    combined row keeps the slower of the two for every statistic, which can
    only make the performance gates harder to pass.
    """
    combined: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            key = (row["backend"], row["mode"], int(row["tokens"]))
            existing = combined.get(key)
            if existing is None:
                combined[key] = dict(row)
                continue
            if row["backend"] != CUSTOM_BACKEND:
                raise ValueError(
                    f"duplicate {row['backend']} measurement for {key}"
                )
            for statistic in ("median_ms", "p90_ms", "p99_ms", "geomean_ms"):
                existing[statistic] = max(
                    float(existing[statistic]),
                    float(row[statistic]),
                )
            existing.setdefault("sources", []).append(row.get("image", "unknown"))
    return [combined[key] for key in sorted(combined)]


# --------------------------------------------------------------------------
# Performance gates
# --------------------------------------------------------------------------


def _gate_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("mode") != "block16":
            continue
        index[(str(row["backend"]), int(row["tokens"]))] = row
    missing = [
        f"{backend}@{tokens}"
        for backend in (CUSTOM_BACKEND, *BASELINE_BACKENDS)
        for tokens in GATE_SHAPES
        if (backend, tokens) not in index
    ]
    if missing:
        raise ValueError(
            f"performance gates are missing block16 measurements: {missing}"
        )
    return index


def evaluate_performance_gates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the three Task 11 latency gates over the block-16 sweep."""
    index = _gate_index(rows)

    custom_at_one = float(index[(CUSTOM_BACKEND, CONCURRENCY_ONE_TOKENS)]["median_ms"])
    baseline_at_one = {
        backend: float(index[(backend, CONCURRENCY_ONE_TOKENS)]["median_ms"])
        for backend in BASELINE_BACKENDS
    }
    slower_than = sorted(
        backend
        for backend, median in baseline_at_one.items()
        if custom_at_one >= median
    )
    concurrency1 = {
        "tokens": CONCURRENCY_ONE_TOKENS,
        "requests": 1,
        "custom_median_ms": custom_at_one,
        "baseline_median_ms": baseline_at_one,
        "slower_than": slower_than,
        "passed": not slower_than,
    }

    geomeans = {
        backend: geometric_mean(
            [float(index[(backend, tokens)]["median_ms"]) for tokens in GATE_SHAPES]
        )
        for backend in (CUSTOM_BACKEND, *BASELINE_BACKENDS)
    }
    faster_baseline = min(BASELINE_BACKENDS, key=lambda name: geomeans[name])
    custom_geomean = geomeans[CUSTOM_BACKEND]
    baseline_geomean = geomeans[faster_baseline]
    block16_geomean = {
        "tokens": list(GATE_SHAPES),
        "custom_geomean_ms": custom_geomean,
        "baseline_geomean_ms": {
            backend: geomeans[backend] for backend in BASELINE_BACKENDS
        },
        "faster_baseline": faster_baseline,
        "ratio_to_faster_baseline": custom_geomean / baseline_geomean,
        "passed": custom_geomean <= baseline_geomean,
    }

    violations = []
    p99_rows = []
    for tokens in GATE_SHAPES:
        custom_p99 = float(index[(CUSTOM_BACKEND, tokens)]["p99_ms"])
        baseline_p99 = min(
            float(index[(backend, tokens)]["p99_ms"])
            for backend in BASELINE_BACKENDS
        )
        limit = P99_LIMIT_RATIO * baseline_p99
        entry = {
            "tokens": tokens,
            "custom_p99_ms": custom_p99,
            "faster_baseline_p99_ms": baseline_p99,
            "limit_ms": limit,
            "ratio": custom_p99 / baseline_p99,
            "passed": custom_p99 <= limit,
        }
        p99_rows.append(entry)
        if not entry["passed"]:
            violations.append(entry)
    block16_p99 = {
        "limit_ratio": P99_LIMIT_RATIO,
        "rows": p99_rows,
        "violations": violations,
        "passed": not violations,
    }

    return {
        "status": "complete",
        "passed": (
            concurrency1["passed"]
            and block16_geomean["passed"]
            and block16_p99["passed"]
        ),
        "concurrency1_median": concurrency1,
        "block16_geomean": block16_geomean,
        "block16_p99": block16_p99,
    }


def partial_gates(
    framework: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record this archive's half of the gate inputs.

    The full gates need both baselines, so a single-framework archive stores
    the pairwise comparison and defers the verdict to :func:`combine_archives`.
    """
    index = {
        (str(row["backend"]), int(row["tokens"])): row
        for row in rows
        if row.get("mode") == "block16"
    }
    pairwise = []
    for tokens in GATE_SHAPES:
        custom = index.get((CUSTOM_BACKEND, tokens))
        native = index.get((framework, tokens))
        if custom is None or native is None:
            continue
        pairwise.append(
            {
                "tokens": tokens,
                "custom_median_ms": custom["median_ms"],
                "native_median_ms": native["median_ms"],
                "custom_p99_ms": custom["p99_ms"],
                "native_p99_ms": native["p99_ms"],
                "median_ratio": custom["median_ms"] / native["median_ms"],
                "p99_ratio": custom["p99_ms"] / native["p99_ms"],
            }
        )
    custom_medians = [row["custom_median_ms"] for row in pairwise]
    native_medians = [row["native_median_ms"] for row in pairwise]
    return {
        "benchmark": BENCHMARK,
        "framework": framework,
        "status": "partial",
        "passed": False,
        "reason": "the full gates require both native baselines",
        "limit_ratio": P99_LIMIT_RATIO,
        "pairwise_block16": pairwise,
        "custom_geomean_ms": (
            geometric_mean(custom_medians) if custom_medians else None
        ),
        "native_geomean_ms": (
            geometric_mean(native_medians) if native_medians else None
        ),
    }


# --------------------------------------------------------------------------
# Combining archives
# --------------------------------------------------------------------------


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def combine_archives(
    directories: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge per-framework archives and return the verdict over both families.

    Everything is written before the verdict is returned, so a caller that
    turns a failing verdict into a failing exit status still leaves a complete
    set of artifacts behind.

    An archive has to carry the parity rows its own manifest says it measured.
    A performance sweep that fell short is recorded as incomplete and fails the
    verdict, but a numerical archive that fell short is refused outright: an
    absent row is an unmeasured shape, and averaging over the shapes that did
    report is how an incomplete run gets read as a clean one.
    """
    if not directories:
        raise ValueError("combining needs at least one archive")

    latency_groups = []
    parity_rows: list[Mapping[str, Any]] = []
    manifests = {}
    expected_rows: list[tuple[str, int, int]] = []
    for directory in directories:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests[manifest["framework"]] = manifest
        rows: list[dict[str, Any]] = []
        for mode in SHAPE_GROUPS:
            path = directory / f"latency_{mode}.json"
            if not path.is_file():
                continue
            for row in json.loads(path.read_text())["rows"]:
                rows.append({**row, "image": manifest["framework"]})
        latency_groups.append(rows)
        parity_path = directory / "parity.json"
        if not parity_path.is_file():
            raise ValueError(
                f"{directory} has no parity.json to gate numerically"
            )
        archived = [
            {"framework": manifest["framework"], **row}
            for row in json.loads(parity_path.read_text())["rows"]
        ]
        parity_rows.extend(archived)
        shape_groups = manifest.get("shape_groups")
        pool_size = manifest.get("graph_pool_size")
        if shape_groups is None or pool_size is None:
            continue
        promised = set(expected_parity_rows(shape_groups, int(pool_size)))
        measured = {_row_key(row) for row in archived}
        if promised - measured:
            raise ValueError(
                f"{directory} is missing parity rows its manifest promised: "
                f"{sorted(promised - measured)}"
            )
        if measured - promised:
            raise ValueError(
                f"{directory} archived parity rows its manifest does not "
                f"cover: {sorted(measured - promised)}"
            )
        expected_rows.extend(sorted(promised))

    combined = merge_latency_rows(latency_groups)
    try:
        performance = evaluate_performance_gates(combined)
    except ValueError as error:
        performance = {
            "status": "incomplete",
            "passed": False,
            "reason": str(error),
        }
    numerical = evaluate_numerical_gates(
        parity_rows, expected_rows=sorted(set(expected_rows)) or None
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    performance_summary = {
        "benchmark": BENCHMARK,
        "frameworks": sorted(manifests),
        "custom_row_policy": (
            "slower of the two per-image custom measurements at each shape"
        ),
        "rows": combined,
        "performance_gates": performance,
    }
    write_json(
        output_dir / "combined_performance_gates.json", performance_summary
    )
    write_json(
        output_dir / "combined_numerical_gates.json",
        {"frameworks": sorted(manifests), **numerical},
    )
    summary = {
        **performance_summary,
        "numerical_gates": numerical,
        "passed": bool(performance["passed"] and numerical["passed"]),
    }
    write_json(
        output_dir / "combined_gates.json",
        {
            key: value
            for key, value in summary.items()
            if key != "rows"
        },
    )
    return summary


__all__ = [
    "DIAGNOSTIC_COMPARISONS",
    "GATED_COMPARISON",
    "combine_archives",
    "evaluate_numerical_gates",
    "expected_parity_rows",
    "evaluate_performance_gates",
    "merge_latency_rows",
    "parity_summary",
    "partial_gates",
    "write_json",
]
