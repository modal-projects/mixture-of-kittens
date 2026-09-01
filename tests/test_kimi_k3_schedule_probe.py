"""What the A/B harness's reductions and verdicts are required to do.

Every function here is pure, so the arithmetic behind the latency claim can be
held to captured numbers on a CPU rather than only by running the measurement
again on eight B300s. That matters more than usual for this harness: it is the
thing that decides whether production changes, and a reduction that silently
picked the wrong quantile or the wrong bar would not fail anything else.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _probe():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("benchmarks.kimi_k3_schedule_probe")


def _repeat(samples: list[float], repeat: int) -> dict[str, object]:
    probe = _probe()
    return {
        "repeat": repeat,
        "order_position": repeat % 2,
        "variant_order": ["production", "candidate"],
        "reported": {
            "sample_count": len(samples),
            "median_ms": probe.percentile(samples, 0.5),
            "p90_ms": probe.percentile(samples, 0.9),
            "p99_ms": probe.percentile(samples, 0.99),
            "geomean_ms": probe.geometric_mean(samples),
        },
        "rank_max_samples_ms": samples,
    }


def _raw(variant_samples: dict[str, list[list[float]]]) -> dict[str, list]:
    return {
        variant: [
            _repeat(samples, index + 1)
            for index, samples in enumerate(repeats)
        ]
        for variant, repeats in variant_samples.items()
    }


def _summaries(raw: dict[str, list]) -> dict[str, dict]:
    probe = _probe()
    return {
        variant: probe._summarize(
            [
                {
                    "median_ms": entry["reported"]["median_ms"],
                    "p99_ms": entry["reported"]["p99_ms"],
                }
                for entry in entries
            ]
        )
        for variant, entries in raw.items()
    }


# ---------------------------------------------------------------------------
# The raw-sample audit.
# ---------------------------------------------------------------------------


def test_the_retained_samples_reproduce_every_number_the_verdict_quotes(
) -> None:
    """Retaining the samples is only worth something if they are the samples.

    The verdict is a claim about a median and a p99, and the only thing behind
    either is a reduction the harness performed once. Keeping its input is what
    makes the claim auditable -- and re-deriving the reported numbers from that
    input, in the run itself, is what makes the audit a thing the run has
    already done rather than a thing a reader could do if they thought to.
    """
    probe = _probe()
    raw = _raw(
        {
            "production": [
                [0.70, 0.66, 0.68, 0.72, 0.90],
                [0.69, 0.67, 0.68, 0.71, 0.88],
            ],
            "candidate": [
                [0.65, 0.63, 0.64, 0.67, 0.85],
                [0.66, 0.63, 0.65, 0.66, 0.84],
            ],
        }
    )
    probe._audit_raw_samples(raw, _summaries(raw), 16)


@pytest.mark.parametrize(
    ("field", "quantile"),
    [("median_ms", 0.5), ("p90_ms", 0.9), ("p99_ms", 0.99)],
)
def test_a_reported_quantile_the_samples_do_not_support_is_refused(
    field: str,
    quantile: float,
) -> None:
    """The audit has to fail closed, or it is a comment.

    Nudged by one part in ten thousand, which is well inside anything a reader
    would notice by eye and well outside exact equality. Both sides are the
    same R-7 quantile of the same float list, so any difference at all means
    the artifact does not hold the samples the verdict was read from.
    """
    probe = _probe()
    raw = _raw({"candidate": [[0.65, 0.63, 0.64, 0.67, 0.85]]})
    summaries = _summaries(raw)
    raw["candidate"][0]["reported"][field] *= 1.0001

    with pytest.raises(AssertionError) as failure:
        probe._audit_raw_samples(raw, summaries, 16)
    assert field in str(failure.value)
    del quantile


def test_a_truncated_sample_list_is_refused() -> None:
    """A short list would silently shift every quantile it is read for."""
    probe = _probe()
    raw = _raw({"candidate": [[0.65, 0.63, 0.64, 0.67, 0.85]]})
    summaries = _summaries(raw)
    raw["candidate"][0]["rank_max_samples_ms"] = [0.65, 0.63]

    with pytest.raises(AssertionError) as failure:
        probe._audit_raw_samples(raw, summaries, 16)
    assert "sample count" in str(failure.value)


def test_a_summary_that_does_not_match_its_repeats_is_refused() -> None:
    """The two halves of the artifact have to agree, not merely coexist."""
    probe = _probe()
    raw = _raw(
        {"candidate": [[0.65, 0.63, 0.64], [0.66, 0.64, 0.65]]}
    )
    summaries = _summaries(raw)
    summaries["candidate"]["median_of_repeat_medians_ms"] = 0.5

    with pytest.raises(AssertionError) as failure:
        probe._audit_raw_samples(raw, summaries, 16)
    assert "median of medians" in str(failure.value)


# ---------------------------------------------------------------------------
# The two bars.
# ---------------------------------------------------------------------------


def _point(
    tokens: int,
    production_medians: list[float],
    candidate_medians: list[float],
    production_p99: float = 0.90,
    candidate_p99: float = 0.85,
) -> dict[str, object]:
    probe = _probe()

    def summary(medians: list[float], p99: float) -> dict[str, object]:
        return {
            "median_of_repeat_medians_ms": probe.percentile(medians, 0.5),
            "median_dispersion_ms": max(medians) - min(medians),
            "median_of_repeat_p99s_ms": p99,
        }

    return probe.evaluate_point(
        tokens=tokens,
        production=summary(production_medians, production_p99),
        candidate=summary(candidate_medians, candidate_p99),
    )


def test_the_experiment_gate_keeps_its_eight_percent_and_keeps_failing() -> None:
    """The measured 3.25% has to stay a recorded miss, not a moved goalpost.

    8% was the estimate of what removing the full-grid barrier idle was worth.
    The measurement came in at 3.25%, which is a quantified over-estimate and
    the most useful thing the experiment produced. Lowering the 8% to match
    would replace that with no record at all, so the gate does not move and the
    smaller bar is asked as its own question beside it.
    """
    probe = _probe()
    assert probe.MINIMUM_M16_MEDIAN_GAIN == 0.08
    assert probe.PROMOTION_M16_MEDIAN_GAIN == 0.02

    # A 3.25% gain: over the promotion bar, under the experiment gate.
    verdict = _point(16, [0.6590, 0.6595, 0.6600], [0.6376, 0.6380, 0.6385])
    assert verdict["improvement_fraction"] == pytest.approx(0.0325, abs=5e-4)
    assert verdict["passed"] is False
    assert verdict["promotion_passed"] is True
    assert "8%" in verdict["requirement"]
    assert "2%" in verdict["promotion_requirement"]

    decision = probe.integration_decision(points=[verdict], blame=None)
    assert decision["experiment_gate_passed"] is False
    assert decision["promotion_passed"] is True
    assert "promote" in decision["recommendation"]


def test_a_gain_inside_the_repeat_dispersion_clears_neither_bar() -> None:
    """A gain smaller than the run-to-run spread is not a gain.

    This is the clause that stops the promotion bar from being a lower bar in
    the sense that matters. 2% of 0.65 ms is 13 microseconds, and a harness
    whose repeat medians span 100 would report that as an improvement every
    other run and a regression the rest. The p99 improves here and the median
    gain clears 2%, so the effect band is the only clause left to fail.
    """
    verdict = _point(16, [0.60, 0.70], [0.58, 0.68])
    assert verdict["improvement_fraction"] > 0.02
    assert verdict["outside_effect_band"] is False
    assert verdict["passed"] is False
    assert verdict["promotion_passed"] is False


def test_a_worse_p99_clears_neither_bar() -> None:
    """Removing barrier idle that only helps the median is not the claim.

    The barriers made every CTA wait for the slowest one, so the tail of the
    distribution is where their cost should be largest. A candidate that
    improved the median and not the p99 would be doing something other than
    what it says.
    """
    verdict = _point(
        16,
        [0.6590, 0.6595, 0.6600],
        [0.6000, 0.6005, 0.6010],
        production_p99=0.85,
        candidate_p99=0.88,
    )
    assert verdict["improvement_fraction"] > 0.08
    assert verdict["p99_improved"] is False
    assert verdict["passed"] is False
    assert verdict["promotion_passed"] is False


def test_the_guard_shape_is_the_same_bar_under_both_verdicts() -> None:
    """There is only one meaning of "do no harm".

    A regression the promotion bar tolerated and the experiment gate did not
    would make the two verdicts disagree about damage rather than about how
    much gain is enough, which is not the distinction they exist to draw.
    """
    probe = _probe()
    inside = _point(128, [0.9000], [0.9045])  # -0.5%
    outside = _point(128, [0.9000], [0.9200])  # -2.2%

    assert inside["passed"] is inside["promotion_passed"] is True
    assert outside["passed"] is outside["promotion_passed"] is False
    assert inside["requirement"] == inside["promotion_requirement"]

    # And one failing guard shape sinks both verdicts, whatever M16 did.
    gate = _point(16, [0.6590, 0.6595, 0.6600], [0.6376, 0.6380, 0.6385])
    decision = probe.integration_decision(points=[gate, outside], blame=None)
    assert decision["experiment_gate_passed"] is False
    assert decision["promotion_passed"] is False
    assert "leave production unchanged" in decision["recommendation"]


def test_the_profile_total_is_taken_over_the_bands_that_do_not_nest() -> None:
    """A total over every counter weighs the nested phases two and three times.

    ``kPhaseClockParents`` makes the band a tree: ``routed_gate_up`` contains
    ``stage`` and ``mma``, which contain four more between them. Summing all of
    them inflates the denominator, and every share taken against it -- most of
    all the barrier share this experiment was sized from -- comes out smaller
    than it is. At M = 16 the two totals differ by 7%, which moved the reported
    barrier share from 14.4% to 9.8%.

    Checked against the source rather than by profiling, because reproducing it
    needs eight B300s and the mistake is visible in one line. It has already
    been made twice: once in the framework comparison and once here, so the
    reduction has to name where its denominator comes from.
    """
    source = (REPO_ROOT / "benchmarks" / "kimi_k3_schedule_probe.py").read_text()
    body = source[source.index("def _profile("):source.index("def variant_orders(")]

    assert "top_level = modules.runtime.top_level_phase_clocks()" in body
    assert "total = sum(phases[name] for name in top_level)" in body
    # The shape that was wrong: a sum over whatever the band happens to hold.
    assert "sum(value for value in phases.values()" not in body

    # Every published share divides by that total and nothing else.
    for fraction in ("grid_barrier_fraction", "readiness_wait_fraction",
                     "wait_fraction"):
        assert f'"{fraction}": (' in body or f'"{fraction}": ' in body
    assert body.count("/ total if total else 0.0") == 3
