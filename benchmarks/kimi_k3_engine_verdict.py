"""What the fused-W13 engine A/B's samples have to show to be a verdict.

The bars an arm clears, the interleaving that makes the comparison fair, the
subphase deltas that say a gain came from the mechanism it was supposed to come
from, and the audit that refuses a sample set the harness could have shaped.
All of it is arithmetic over recorded numbers and none of it touches a device,
so it is held to CPU tests in ``tests/test_kimi_k3_engine_probe.py`` rather
than re-derived from a run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from benchmarks.kimi_k3_timing import (
    geometric_mean,
    percentile,
)

from benchmarks.kimi_k3_engine_routes import (
    BAND_SUBPHASES,
    CANDIDATES,
    GATE_LABEL,
    GUARD_LABELS,
    MAXIMUM_GUARD_REGRESSION,
    MINIMUM_GATE_MEDIAN_GAIN,
    Shape,
    VARIANTS,
)


def subphase_deltas(
    production: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """What moved inside the band, in cycles and as a share of it.

    Reported per subphase rather than as one number for the band, because the
    candidate is a trade: it spends the third stage's shared memory to shorten
    `tma_wait` and `ring_full`, and pays for it in a per-slab handoff and six
    epilogues with nothing flying underneath them. A band that got shorter
    without `tma_wait` getting shorter would mean the mechanism is not the one
    claimed.
    """
    before = production["phase_clock_cycles"]
    after = candidate["phase_clock_cycles"]
    return {
        "band": {
            "production_cycles": before["routed_gate_up"],
            "candidate_cycles": after["routed_gate_up"],
            "change_fraction": (
                (after["routed_gate_up"] - before["routed_gate_up"])
                / before["routed_gate_up"]
                if before["routed_gate_up"]
                else 0.0
            ),
        },
        "ring": {
            "production_share_of_band": production["ring_share_of_band"],
            "candidate_share_of_band": candidate["ring_share_of_band"],
        },
        "subphases": {
            name: {
                "production_cycles": before[name],
                "candidate_cycles": after[name],
                "change_fraction": (
                    (after[name] - before[name]) / before[name]
                    if before[name]
                    else 0.0
                ),
                "production_share_of_band": production[
                    "subphase_share_of_band"
                ][name],
                "candidate_share_of_band": candidate[
                    "subphase_share_of_band"
                ][name],
            }
            for name in BAND_SUBPHASES
        },
    }


def variant_orders(repeats: int) -> list[tuple[str, ...]]:
    """Rotate which arm runs first, so drift is shared rather than charged.

    A rotation rather than a reversal, so the shape holds for any number of
    arms: with an odd repeat count every arm takes every slot, and reversing
    would give a middle arm the one slot that temporal drift within a repeat
    does not reach the edges of.
    """
    if repeats < 1:
        raise ValueError("the A/B needs at least one repeat")
    forward = tuple(VARIANTS)
    width = len(forward)
    return [
        forward[repeat % width :] + forward[: repeat % width]
        for repeat in range(repeats)
    ]


def _summarize(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    medians = [float(repeat["median_ms"]) for repeat in repeats]
    p99s = [float(repeat["p99_ms"]) for repeat in repeats]
    center = percentile(medians, 0.5)
    return {
        "repeat_count": len(repeats),
        "repeat_medians_ms": medians,
        "repeat_p99s_ms": p99s,
        "median_of_repeat_medians_ms": center,
        "median_dispersion_ms": max(medians) - min(medians),
        "median_of_repeat_p99s_ms": percentile(p99s, 0.5),
        "geomean_of_repeat_medians_ms": geometric_mean(medians),
    }


def _audit_raw_samples(
    raw: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    label: str,
) -> None:
    """Recompute every reported number from the retained samples, exactly."""
    for variant, entries in raw.items():
        summary = summaries[variant]
        recomputed_medians = []
        recomputed_p99s = []
        for entry in entries:
            samples = entry["rank_max_samples_ms"]
            reported = entry["reported"]
            if len(samples) != reported["sample_count"]:
                raise AssertionError(
                    (label, variant, entry["repeat"], "sample count",
                     len(samples), reported["sample_count"])
                )
            for key, quantile in (
                ("median_ms", 0.5), ("p90_ms", 0.9), ("p99_ms", 0.99)
            ):
                again = percentile(samples, quantile)
                if again != reported[key]:
                    raise AssertionError(
                        (label, variant, entry["repeat"], key, again,
                         reported[key])
                    )
            recomputed_medians.append(reported["median_ms"])
            recomputed_p99s.append(reported["p99_ms"])
        if recomputed_medians != summary["repeat_medians_ms"]:
            raise AssertionError((label, variant, "repeat medians"))
        if recomputed_p99s != summary["repeat_p99s_ms"]:
            raise AssertionError((label, variant, "repeat p99s"))
        if percentile(recomputed_medians, 0.5) != summary[
            "median_of_repeat_medians_ms"
        ]:
            raise AssertionError((label, variant, "median of medians"))
        if percentile(recomputed_p99s, 0.5) != summary[
            "median_of_repeat_p99s_ms"
        ]:
            raise AssertionError((label, variant, "median of p99s"))


def evaluate_point(
    *,
    shape: Shape,
    variant: str,
    production: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Score one arm at one shape, against the requirement for that shape."""
    production_center = float(production["median_of_repeat_medians_ms"])
    candidate_center = float(candidate["median_of_repeat_medians_ms"])
    if not (
        math.isfinite(production_center)
        and math.isfinite(candidate_center)
        and production_center > 0.0
        and candidate_center > 0.0
    ):
        raise ValueError(
            f"{shape.label}: repeat medians must be finite and positive"
        )

    effect_band = max(
        float(production["median_dispersion_ms"]),
        float(candidate["median_dispersion_ms"]),
    )
    improvement = production_center - candidate_center
    fraction = improvement / production_center
    production_p99 = float(production["median_of_repeat_p99s_ms"])
    candidate_p99 = float(candidate["median_of_repeat_p99s_ms"])
    p99_improved = candidate_p99 < production_p99

    verdict: dict[str, Any] = {
        "shape": shape.label,
        "tokens": shape.tokens,
        "route": shape.route,
        "rows_per_expert": shape.rows_per_expert,
        "variant": variant,
        "production_median_ms": production_center,
        "candidate_median_ms": candidate_center,
        "improvement_ms": improvement,
        "improvement_fraction": fraction,
        "effect_band_ms": effect_band,
        "outside_effect_band": improvement > effect_band,
        "production_p99_ms": production_p99,
        "candidate_p99_ms": candidate_p99,
        "p99_improved": p99_improved,
        "p99_change_fraction": (candidate_p99 - production_p99) / production_p99,
    }
    if shape.label == GATE_LABEL:
        verdict["requirement"] = (
            f"median gain >= {MINIMUM_GATE_MEDIAN_GAIN:.0%}, outside the "
            "repeat-median dispersion, with an improved p99"
        )
        verdict["passed"] = bool(
            fraction >= MINIMUM_GATE_MEDIAN_GAIN
            and improvement > effect_band
            and p99_improved
        )
        verdict["gating"] = True
        return verdict
    if shape.label in GUARD_LABELS:
        verdict["requirement"] = (
            f"regression <= {MAXIMUM_GUARD_REGRESSION:.0%}"
        )
        verdict["passed"] = bool(-fraction <= MAXIMUM_GUARD_REGRESSION)
        verdict["gating"] = True
        return verdict
    verdict["requirement"] = "reported, gates nothing"
    verdict["passed"] = True
    verdict["gating"] = False
    return verdict


def variant_decision(
    *,
    variant: str,
    points: Sequence[dict[str, Any]],
    deltas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One arm's verdict, and the mechanism it did or did not confirm.

    Two things have to hold together for an arm to earn integration: it has to
    be faster at the shape decode runs in without regressing the shape that
    guards it, and the reason has to be the one claimed. `tma_wait` falling is
    the mechanism the third stage predicts; a win with it flat would be a win
    this design cannot take credit for and could not be reasoned about again.
    """
    gating = [point for point in points if point["gating"]]
    passed = bool(gating) and all(point["passed"] for point in gating)
    gate_deltas = deltas.get(GATE_LABEL, {}).get("subphases", {})
    wait_change = gate_deltas.get("routed_gate_up_tma_wait", {}).get(
        "change_fraction"
    )
    ring_full_change = gate_deltas.get("routed_gate_up_ring_full", {}).get(
        "change_fraction"
    )
    return {
        "variant": variant,
        "passed": passed,
        "mechanism_confirmed": bool(
            wait_change is not None and wait_change < 0.0
        ),
        "gate_point_gain_fraction": next(
            (
                float(point["improvement_fraction"])
                for point in points
                if point["shape"] == GATE_LABEL
            ),
            0.0,
        ),
        "gate_point_tma_wait_change_fraction": wait_change,
        "gate_point_ring_full_change_fraction": ring_full_change,
        "gating_shapes": [str(point["shape"]) for point in gating],
        "failed_shapes": [
            str(point["shape"]) for point in gating if not point["passed"]
        ],
    }


def integration_decision(
    *,
    points: Sequence[dict[str, Any]],
    deltas: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Whether the candidate cleared the gate, and which one did if several ran.

    The shape is a comparison because the probe was written while three arms
    were being measured against each other. One is compiled now -- production's
    selector -- so `runner_up` is `None` and the verdict reduces to whether the
    selector beat the ring it replaced at the gate point without regressing the
    guards. The shape is kept because it costs nothing and it is what a future
    arm would be judged by.
    """
    per_variant = {
        variant: variant_decision(
            variant=variant,
            points=[point for point in points if point["variant"] == variant],
            deltas=deltas.get(variant, {}),
        )
        for variant in CANDIDATES
    }
    eligible = [
        variant
        for variant, decision in per_variant.items()
        if decision["passed"] and decision["mechanism_confirmed"]
    ]
    best = max(
        eligible,
        key=lambda variant: per_variant[variant]["gate_point_gain_fraction"],
        default=None,
    )
    ranked = sorted(
        (variant for variant in CANDIDATES if variant != best),
        key=lambda variant: per_variant[variant]["gate_point_gain_fraction"],
        reverse=True,
    )
    runner_up = ranked[0] if len(CANDIDATES) > 1 and ranked else None
    return {
        "passed": best is not None,
        "winner": best,
        "runner_up": runner_up,
        "gate": (
            f"{GATE_LABEL} median gain >= {MINIMUM_GATE_MEDIAN_GAIN:.0%}, and "
            f"{' and '.join(GUARD_LABELS)} regression <= "
            f"{MAXIMUM_GUARD_REGRESSION:.0%}"
        ),
        "mechanism": (
            "the third K = 512 stage must shorten `routed_gate_up_tma_wait`"
        ),
        "per_variant": per_variant,
        "recommendation": (
            f"integrate the {best} engine"
            if best is not None
            else "the candidate does not clear the gate against the "
            "resident ring"
        ),
    }
