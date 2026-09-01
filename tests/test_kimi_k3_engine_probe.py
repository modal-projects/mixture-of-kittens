"""What the engine A/B harness's routes and verdicts are required to do.

Every function here is pure, so the arithmetic and the routing behind the
integration recommendation can be held to captured numbers on a CPU rather than
only by running the measurement again on eight B300s.

That matters more for this harness than for the schedule one, because the
adaptive engine branches on a number this harness is responsible for producing.
Production packs a batch of at most four rows into its compact ring and hands
anything wider to the slab-buffered one, so a shape that claims to be the r = 8
wide route and is quietly an r = 1 route would report the packed ring's latency
under the wide ring's name -- and the run would pass its guard without ever
having measured the thing the guard is for. The row count each shape declares is
therefore asserted against the routing that shape actually produces, here and
again inside the run.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module(f"benchmarks.{name}")


def _routes():
    return _module("kimi_k3_engine_routes")


def _verdict():
    return _module("kimi_k3_engine_verdict")


def _probe():
    return _module("kimi_k3_engine_probe")


def _inputs():
    return _module("kimi_k3_decode_inputs")


# ---------------------------------------------------------------------------
# The routes, and the row counts the adaptive engine branches on.
# ---------------------------------------------------------------------------


def test_every_shape_routes_to_the_row_count_it_claims() -> None:
    """The shape table is a claim about occupancy; this is the check on it.

    Asserted of all four pool entries rather than the first, because the entries
    rotate which experts they occupy and a rotation that folded two groups onto
    one expert would double that entry's rows while leaving entry zero correct.
    """
    routes = _routes()
    inputs = _inputs()
    for shape in routes.SHAPES:
        for pool_index in range(inputs.GRAPH_POOL_SIZE):
            if shape.route == "sweep":
                assignments = inputs.route_assignments(shape.tokens, pool_index)
            else:
                assignments = routes.concentrated_assignments(
                    tokens=shape.tokens,
                    pool_index=pool_index,
                    rows=shape.rows_per_expert,
                    topk=inputs.TOPK,
                    num_experts=inputs.NUM_EXPERTS,
                )
            histogram = routes._row_histogram(assignments)
            assert max(histogram) == shape.rows_per_expert, (
                shape.label,
                pool_index,
                histogram,
            )


def test_the_realistic_shapes_are_the_ones_the_packing_is_bet_on() -> None:
    """One row at M16 and M32, two or three at M128, and eight at the guard.

    Spelled as its own case because it is the premise of the whole adaptive
    design rather than an incidental property of the sweep router. If the
    router ever routes M128 to four rows or more, `compact` stops taking the
    packed path at the shape it was accepted for and this is what says so.
    """
    routes = _routes()
    inputs = _inputs()
    rows = {
        shape.label: max(
            routes._row_histogram(inputs.route_assignments(shape.tokens, 0))
        )
        for shape in routes.SHAPES
        if shape.route == "sweep"
    }
    assert rows == {"m16": 1, "m32": 1, "m128": 3}
    assert routes.SHAPES_BY_LABEL["m128_r8"].rows_per_expert == 8


def test_a_concentrated_route_gives_every_token_its_whole_top_k() -> None:
    """Sixteen distinct experts per token, or the route is not a K3 route."""
    routes = _routes()
    inputs = _inputs()
    assignments = routes.concentrated_assignments(
        tokens=128,
        pool_index=2,
        rows=8,
        topk=inputs.TOPK,
        num_experts=inputs.NUM_EXPERTS,
    )
    assert len(assignments) == 128
    for token, experts in enumerate(assignments):
        assert len(set(experts)) == inputs.TOPK, token
        assert all(0 <= expert < inputs.NUM_EXPERTS for expert in experts)


def test_the_pool_entries_of_a_concentrated_route_occupy_different_experts(
) -> None:
    """Otherwise the four replays would share a working set and hit in L2.

    The pool exists to keep the weight ring reading memory rather than cache,
    which is what the run's own L2 check enforces. That check is on the union,
    so this is the finer claim: no two entries are the same route.
    """
    routes = _routes()
    inputs = _inputs()
    occupied = [
        frozenset(
            expert
            for row in routes.concentrated_assignments(
                tokens=128,
                pool_index=pool_index,
                rows=8,
                topk=inputs.TOPK,
                num_experts=inputs.NUM_EXPERTS,
            )
            for expert in row
        )
        for pool_index in range(inputs.GRAPH_POOL_SIZE)
    ]
    assert all(len(entry) == 256 for entry in occupied)
    assert len(set(occupied)) == inputs.GRAPH_POOL_SIZE


def test_a_route_the_groups_do_not_divide_is_refused() -> None:
    """A partial group would leave one expert short and the rest at `rows`."""
    routes = _routes()
    with pytest.raises(ValueError, match="do not divide"):
        routes.concentrated_assignments(
            tokens=100, pool_index=0, rows=8, topk=16, num_experts=896
        )


def test_a_route_wider_than_the_layer_is_refused() -> None:
    """896 experts is the layer; a route asking for more is a wrapped route."""
    routes = _routes()
    with pytest.raises(ValueError, match="the layer has"):
        routes.concentrated_assignments(
            tokens=128, pool_index=0, rows=1, topk=16, num_experts=896
        )


# ---------------------------------------------------------------------------
# The bars, which are not the same bars at every shape.
# ---------------------------------------------------------------------------


def _point(
    label: str,
    production_medians: list[float],
    candidate_medians: list[float],
    production_p99: float = 0.90,
    candidate_p99: float = 0.85,
    variant: str = "production",
) -> dict[str, object]:
    routes = _routes()
    verdict = _verdict()

    def summary(medians: list[float], p99: float) -> dict[str, object]:
        return {
            "median_of_repeat_medians_ms": verdict.percentile(medians, 0.5),
            "median_dispersion_ms": max(medians) - min(medians),
            "median_of_repeat_p99s_ms": p99,
        }

    return verdict.evaluate_point(
        shape=routes.SHAPES_BY_LABEL[label],
        variant=variant,
        production=summary(production_medians, production_p99),
        candidate=summary(candidate_medians, candidate_p99),
    )


def test_the_gate_shape_wants_two_percent_outside_the_dispersion() -> None:
    """A gain smaller than the spread between repeats is not a gain."""
    verdict = _point("m16", [0.700, 0.702, 0.698], [0.680, 0.681, 0.679])
    assert verdict["passed"]
    assert verdict["gating"]
    assert verdict["improvement_fraction"] > 0.02
    assert verdict["outside_effect_band"]


def test_a_gate_gain_inside_the_repeat_dispersion_fails() -> None:
    """Same 2.9% median gain, and repeats that wander further than it does."""
    verdict = _point("m16", [0.700, 0.760, 0.640], [0.680, 0.740, 0.620])
    assert verdict["improvement_fraction"] > 0.02
    assert not verdict["outside_effect_band"]
    assert not verdict["passed"]


def test_a_gate_gain_with_a_worse_tail_fails() -> None:
    """The median is what a step costs; the p99 is what a step risks."""
    verdict = _point(
        "m16",
        [0.700, 0.702, 0.698],
        [0.680, 0.681, 0.679],
        production_p99=0.90,
        candidate_p99=0.95,
    )
    assert verdict["improvement_fraction"] > 0.02
    assert not verdict["p99_improved"]
    assert not verdict["passed"]


@pytest.mark.parametrize("label", ["m128", "m128_r8"])
def test_a_guard_shape_tolerates_a_percent_and_no_more(label: str) -> None:
    """Both guards, because the wide route is a guard in its own right.

    `m128_r8` is the route every occupied expert holds a full eight rows on, so
    it is the one production answers with the slab-buffered ring rather than the
    packed one. A regression there is a regression in the wide arm, which is the
    one cost the adaptive design cannot answer by branching.
    """
    routes = _routes()
    within = _point(label, [1.000, 1.000, 1.000], [1.005, 1.005, 1.005])
    beyond = _point(label, [1.000, 1.000, 1.000], [1.020, 1.020, 1.020])
    assert within["gating"] and within["passed"]
    assert beyond["gating"] and not beyond["passed"]
    assert label in routes.GUARD_LABELS


def test_a_shape_that_gates_nothing_says_so() -> None:
    """M32 is reported because it is the route, not because it decides."""
    verdict = _point("m32", [1.000, 1.000, 1.000], [1.100, 1.100, 1.100])
    assert not verdict["gating"]
    assert verdict["passed"]
    assert verdict["requirement"] == "reported, gates nothing"


# ---------------------------------------------------------------------------
# The recommendation.
# ---------------------------------------------------------------------------


def _deltas(variant: str, wait_change: float) -> dict[str, object]:
    routes = _routes()
    return {
        variant: {
            routes.GATE_LABEL: {
                "subphases": {
                    "routed_gate_up_tma_wait": {
                        "change_fraction": wait_change
                    },
                    "routed_gate_up_ring_full": {"change_fraction": -0.1},
                }
            }
        }
    }


def _passing_points(variant: str) -> list[dict[str, object]]:
    return [
        _point("m16", [0.700, 0.702, 0.698], [0.680, 0.681, 0.679],
               variant=variant),
        _point("m128", [1.000, 1.000, 1.000], [0.999, 0.999, 0.999],
               variant=variant),
        _point("m128_r8", [1.000, 1.000, 1.000], [1.000, 1.000, 1.000],
               variant=variant),
    ]


def test_an_arm_that_won_without_its_mechanism_is_not_recommended() -> None:
    """A win the design cannot explain is a win that cannot be reasoned about.

    The third stage is supposed to shorten the wait for a weight slab. An arm
    that got faster with that wait unchanged got faster for some other reason,
    and the next change to this kernel would be made against a false model of
    why it is fast.
    """
    verdict = _verdict()
    decision = verdict.integration_decision(
        points=_passing_points("production"),
        deltas=_deltas("production", wait_change=0.05),
    )
    assert not decision["passed"]
    assert decision["winner"] is None
    assert "does not clear the gate" in decision["recommendation"]
    assert decision["per_variant"]["production"]["passed"]
    assert not decision["per_variant"]["production"]["mechanism_confirmed"]


def test_an_arm_that_failed_a_guard_names_the_shape_it_failed() -> None:
    """A verdict that says only "no" is a verdict nobody can act on."""
    verdict = _verdict()
    points = _passing_points("production")
    points[2] = _point(
        "m128_r8", [1.000, 1.000, 1.000], [1.050, 1.050, 1.050],
        variant="production",
    )
    decision = verdict.integration_decision(
        points=points, deltas=_deltas("production", wait_change=-0.3)
    )
    assert not decision["passed"]
    assert decision["per_variant"]["production"]["failed_shapes"] == ["m128_r8"]


def test_an_arm_that_cleared_every_shape_and_its_mechanism_is_recommended(
) -> None:
    """One arm is compiled, so the verdict is about it and not a comparison.

    Both rings the selector is built from were separately selectable while they
    were being measured; the integration retired those ids, so what is left to
    settle is whether the selector beats the ring it replaced at the gate point
    without regressing either guard, for the reason the third stage predicts.
    """
    verdict = _verdict()
    decision = verdict.integration_decision(
        points=_passing_points("production"),
        deltas=_deltas("production", wait_change=-0.4),
    )
    assert decision["passed"]
    assert decision["winner"] == "production"
    assert decision["runner_up"] is None
    assert decision["recommendation"] == "integrate the production engine"


# ---------------------------------------------------------------------------
# The interleaving.
# ---------------------------------------------------------------------------


def test_every_arm_takes_every_position_across_the_repeats() -> None:
    """Two arms and five repeats, rotated, so no arm owns a slot.

    A fixed order would charge whatever drifts within a repeat to whichever arm
    ran last, and a rotation rather than a reversal keeps that true whatever
    number of arms is compiled.
    """
    routes = _routes()
    verdict = _verdict()
    probe = _probe()
    orders = verdict.variant_orders(probe.REPEATS)
    assert len(orders) == probe.REPEATS
    for position in range(len(routes.VARIANTS)):
        assert {order[position] for order in orders} == set(routes.VARIANTS)


def test_a_run_with_no_repeats_is_refused() -> None:
    verdict = _verdict()
    with pytest.raises(ValueError, match="at least one repeat"):
        verdict.variant_orders(0)
