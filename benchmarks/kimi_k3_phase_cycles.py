"""The decode kernel's clock band, as the tree it is rather than a list.

Nine of the twenty-two regions measure the inside of another region at a finer
grain and therefore the same cycles twice, so a total over all of them would
double-count. Which region contains which is declared here next to the clocks
rather than inferred from the names, because inference over a suffix list could
only express the containments somebody had thought to list -- and a region
derived under an unlisted suffix went missing the moment it was added.

`csrc/kimi_k3_decode/types.cuh` is where these counters are written, and
``tests/test_kimi_k3_frameworks_capture.py`` holds this table to that one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# The kernel's clock64 accumulators, as `(region, containing region)` in
# `csrc/kimi_k3_decode/types.cuh` order.
#
# The band is a tree, not a list. Nine of the twenty-two regions measure the
# inside of another region at a finer grain and therefore the same cycles
# again: `routed_gate_up_stage` contains the three copy clocks,
# `routed_gate_up_mma` contains the issue clock, and `routed_gate_up` contains
# both of those plus the activation gather and the epilogue. At M16 the leaves
# of that one subtree came to 82.0M cycles against a parent of 46.5M, so a
# total taken over the whole band is not a total of anything.
#
# The parent is therefore part of the table rather than inferred from the name.
# It used to be inferred, from a list of suffixes, and that could express only
# "not in the total" -- it could not say that `_tma_wait` is a fraction of
# `_stage` rather than of the launch, and it silently depended on no top-level
# region ever being named with a child's suffix.
PHASE_CLOCKS = (
    ("readiness_wait", None),
    ("router_score", None),
    ("latent_project", None),
    ("routed_queue", None),
    ("latent_quantize", None),
    ("assignment", None),
    ("publish", None),
    ("routed_gate_up", None),
    ("routed_gate_up_stage", "routed_gate_up"),
    ("routed_gate_up_mma", "routed_gate_up"),
    ("routed_gate_up_tma_issue", "routed_gate_up_stage"),
    ("routed_gate_up_tma_wait", "routed_gate_up_stage"),
    ("routed_gate_up_ring_full", "routed_gate_up_stage"),
    ("routed_gate_up_mma_issue", "routed_gate_up_mma"),
    ("routed_gate_up_activation", "routed_gate_up"),
    ("routed_gate_up_epilogue", "routed_gate_up"),
    ("routed_down", None),
    ("routed_down_stage", "routed_down"),
    ("routed_down_mma", "routed_down"),
    ("shared_experts", None),
    ("grid_barrier", None),
    ("tail", None),
)

PHASE_CLOCK_NAMES = tuple(name for name, _ in PHASE_CLOCKS)
# Regions the reader derives rather than reads. `routed_down` has no epilogue
# counter of its own, so its epilogue is what is left of its band after staging
# and MMA -- which makes it a child of that band exactly as a measured one
# would be, and the total must not gain it.
PHASE_CLOCK_DERIVED = {"routed_down_epilogue": "routed_down"}
PHASE_CLOCK_PARENTS = {
    **{name: parent for name, parent in PHASE_CLOCKS},
    **PHASE_CLOCK_DERIVED,
}
# The regions whose intervals are disjoint, and therefore the only ones a total
# may be taken over. Each begins where the previous one ended or at a mark
# reset after a wait, and ends at its own lap.
#
# Membership is what admits a region to the total, rather than the absence of a
# child's name shape. A region nobody declared contributes nothing instead of
# being assumed disjoint, which is the direction that fails safely: the old
# suffix rule counted `routed_down_epilogue` as a region of the launch the
# moment it was derived under a name whose suffix nobody had listed.
PHASE_CLOCK_TOP_LEVEL = tuple(
    name for name, parent in PHASE_CLOCKS if parent is None
)


def derive_phase_cycles(cycles: Mapping[str, int]) -> dict[str, int]:
    """Expose each routed epilogue outside staging and MMA clocks.

    The fused gate/up unit times its own epilogue, so only the region that does
    not gets one derived for it. Overwriting a measured counter with a residual
    would hide exactly the difference the residual exists to estimate.
    """
    derived = dict(cycles)
    for phase in ("routed_gate_up", "routed_down"):
        if f"{phase}_epilogue" in derived:
            continue
        total = derived.get(phase)
        stage = derived.get(f"{phase}_stage")
        mma = derived.get(f"{phase}_mma")
        if total is not None and stage is not None and mma is not None:
            derived[f"{phase}_epilogue"] = max(0, total - stage - mma)
    return derived


def summarize_phase_cycles(cycles: Mapping[str, int]) -> dict[str, Any]:
    """Rank the kernel's accumulated regions by their share of the total.

    The total is over the top-level regions only, because those are the only
    ones whose intervals are disjoint. A child is reported with its share of
    the same total *and* its share of its own parent, which is what makes "TMA
    wait is 71% of gate/up staging and 14% of the launch" two statements rather
    than one ambiguous one -- the first is the actionable number and the second
    is the one that says whether acting on it is worth anything.
    """
    accounted = sum(
        value for name, value in cycles.items()
        if name in PHASE_CLOCK_TOP_LEVEL
    )
    ranked = sorted(
        (
            (name, value)
            for name, value in cycles.items()
            if name in PHASE_CLOCK_TOP_LEVEL
        ),
        key=lambda item: (-item[1], item[0]),
    )
    share_of_parent = {}
    for name, value in cycles.items():
        parent = PHASE_CLOCK_PARENTS.get(name)
        if parent is None:
            continue
        total = cycles.get(parent, 0)
        share_of_parent[name] = value / total if total else 0.0
    return {
        "accounted_cycles": accounted,
        "share_of_accounted": {
            name: (value / accounted if accounted else 0.0)
            for name, value in cycles.items()
        },
        "share_of_parent": share_of_parent,
        "top_level": list(PHASE_CLOCK_TOP_LEVEL),
        "ranked": ranked,
        "dominant_region": ranked[0][0] if accounted else None,
        "dominant_share": (ranked[0][1] / accounted) if accounted else 0.0,
    }
