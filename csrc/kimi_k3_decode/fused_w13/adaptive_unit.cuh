#pragma once

/// The production gate/up unit: one selector over the two three-stage rings.
///
/// This is the only unit production's dispatch reaches. It reads `batch_rows`
/// and takes the compact ring inside the threshold and the slab-buffered ring
/// outside it, per expert, on the device.

#include "compact_unit.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

/// Pick the ring this expert's batch admits. This is the production unit.
///
/// The compact ring packs seven slabs four rows apart, so it contracts a batch
/// of at most `kFusedCompactRows` and something else has to contract the rest.
/// The four-shape A/B measured what that something should be: the compact ring
/// is 5.49% off the step at M16, 8.17% at M32 and 6.87% at the realistic M128,
/// and 0.04% -- nothing -- on a route concentrated to eight rows per expert,
/// because there it never runs. The slab-buffered ring is 5.95% on that same
/// concentrated route. So the wide arm is the slab-buffered ring rather than
/// the resident two-stage ring the A/B fell back to, and the hybrid keeps the
/// gain at both ends of the row distribution instead of one.
///
/// **The choice is per expert, on the device.** A step's experts do not agree:
/// at M128 the realistic route puts two or three rows on most experts, and a
/// concentrated one puts eight on all of them. A host-side switch would have to
/// pick for the whole launch off a row count it does not have until the router
/// has run, and would be wrong for every expert on the other side of the
/// threshold. `batch_rows` is already in hand here.
///
/// **Both rings are armed on a CTA's first unit, whichever ring that unit
/// takes.** A ring's mbarriers are static shared memory and their parity is
/// carried across units in shared memory, so a CTA that runs a narrow expert
/// and then a wide one needs both armed before either runs. Arming is a
/// `batch_rows` of zero: the barriers are initialized and the pass loop does
/// not execute.
///
/// **Switching rings between units is safe because a unit owns the ring it
/// leaves.** Both rings put their three weight stages and scale quads at the
/// same shared offsets and differ only after them, and a unit does not return
/// until all 42 of its stream indices are retired and both its epilogue
/// barriers have passed. So the next unit finds every weight stage free, every
/// mbarrier at a parity its own state records, and the bytes past the stages
/// dead. `test_production_survives_a_repeated_step_on_one_workspace`,
/// `test_production_alternates_arms_on_one_workspace` and the r1-to-r8 sweep are
/// what hold that; the second is the one racecheck runs, because it makes the
/// claim on shapes the tool can afford.
static __device__ void routed_gate_up_fused_adaptive_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const CUtensorMap *__restrict__ packed_map,
    const std::uint8_t *__restrict__ fused_scale,
    const Scratch &scratch,
    int *__restrict__ arrival_counter,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const PhaseClocks clocks,
    const bool first_unit
) {
    const auto wide = [&](const int rows, const bool arming) {
        routed_gate_up_fused_v4_unit(
            shared_raw, tensor_pool, packed_map, fused_scale, scratch,
            arrival_counter, expert, assignment_begin, rows, clocks, arming);
    };

    if (first_unit) {
        routed_gate_up_fused_compact_unit(
            shared_raw, tensor_pool, packed_map, fused_scale, scratch,
            arrival_counter, expert, assignment_begin, 0, clocks, true);
        wide(0, true);
    }
    if (batch_rows <= kFusedCompactRows) {
        routed_gate_up_fused_compact_unit(
            shared_raw, tensor_pool, packed_map, fused_scale, scratch,
            arrival_counter, expert, assignment_begin, batch_rows, clocks,
            false);
        return;
    }
    wide(batch_rows, false);
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
