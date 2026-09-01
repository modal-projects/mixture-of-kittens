#pragma once

/// The phase clocks a profiled launch laps into, and what each slot means.
///
/// Durable instrumentation rather than a one-off: the subphase breakdown these
/// slots carry is the only view the remaining performance work has into where
/// a step's time goes.

#include "schedule_counters.cuh"

namespace kimi_k3_decode {

// Phase clocks.
//
// A CTA reads `clock64()` around each region it runs and accumulates the delta
// into one shared counter per region, so a profiled launch reports where its
// own resident CTAs spent their cycles. The counters are aligned pairs of the
// phase region's tail slots, read as one 64-bit accumulator each: a decode step
// is millions of cycles summed over 148 CTAs, which overflows 32 bits.
//
// Nothing here is benchmark-private. A profiled launch is one predicate on a
// null pointer the caller already hands in, no stage compiles differently for
// it, and the routed gate/up band below is deliberately fine enough to name
// which part of the fused engine a future step is spending its time in --
// which is the only instrument the remaining performance work has.
// ---------------------------------------------------------------------------

/// One accumulated region, in the order the kernel runs them.
///
/// **These are not all disjoint, and a total may not be taken over all of
/// them.** The band is a two-level tree: `kPhaseClockParents` below says which
/// regions are top-level and which are refinements of a parent, and a total is
/// only meaningful over the top-level ones. `routed_gate_up` and its eight
/// children measure the same cycles at three depths -- `stage` contains
/// `tma_issue`, `tma_wait`, and `ring_full`; `mma` contains `mma_issue` -- so
/// summing the leaves of that subtree alone overcounts the phase by 76%.
///
/// The refinements are kept because they are the only instrument the remaining
/// gate/up work has: they say whether the weight ring or the contraction is the
/// binding constraint. They are diagnostic children, not addends.
enum PhaseClock : int {
    kClockReadinessWait = 0,
    kClockRouterScore,
    kClockLatentProject,
    kClockRoutedQueue,
    kClockLatentQuantize,
    /// Router selection compaction: the expert-major table and the unit list.
    kClockAssignment,
    /// Writing this rank's routed partial into the collective buffer.
    kClockPublish,
    kClockRoutedGateUp,
    kClockRoutedGateUpStage,
    kClockRoutedGateUpMma,
    /// Building descriptors and issuing a slab's two bulk copies.
    kClockRoutedGateUpTmaIssue,
    /// Waiting for a slab's payload and scales to land.
    kClockRoutedGateUpTmaWait,
    /// Waiting for the tensor core to free the stage about to be refilled --
    /// the ring being full is the only reason this is ever nonzero.
    kClockRoutedGateUpRingFull,
    /// Staging the scale quads into tensor memory and issuing the sixteen
    /// contractions of a slab.
    kClockRoutedGateUpMmaIssue,
    /// Gathering the batch's activation rows and scales for the whole of K.
    ///
    /// The one cost the engine cannot hand to the copy engine: the rows an
    /// expert's batch names are scattered through the assignment order. It is
    /// paid once per expert rather than once per `(task, slab)` pair, which is
    /// the measurement the engine's shape exists for, so it is reported on its
    /// own rather than folded into the staging above.
    kClockRoutedGateUpActivation,
    /// Reading the accumulator, pairing the gate and up row halves, and
    /// quantizing one task's 64 `situ` columns.
    kClockRoutedGateUpEpilogue,
    kClockRoutedDown,
    kClockRoutedDownStage,
    kClockRoutedDownMma,
    kClockSharedExperts,
    kClockGridBarrier,
    kClockTail,
    kPhaseClockCount,
};

/// First slot of the 64-bit accumulator band, which ends the phase region.
inline constexpr int kPhaseClockBegin =
    NUM_PHASE_COUNTERS - 2 * kPhaseClockCount;

static_assert(kPhaseClockCount == 22);
static_assert(kPhaseClockBegin == 84);
static_assert(kPhaseClockBegin % 2 == 0,
              "a 64-bit accumulator must start on an aligned slot pair");
static_assert(kPhaseClockBegin > kGateUpArrivals,
              "the accumulators must not overlap a live counter");

/// Every accumulated region's reported name, in `PhaseClock` order.
inline constexpr const char *kPhaseClockNames[] = {
    "readiness_wait",
    "router_score",
    "latent_project",
    "routed_queue",
    "latent_quantize",
    "assignment",
    "publish",
    "routed_gate_up",
    "routed_gate_up_stage",
    "routed_gate_up_mma",
    "routed_gate_up_tma_issue",
    "routed_gate_up_tma_wait",
    "routed_gate_up_ring_full",
    "routed_gate_up_mma_issue",
    "routed_gate_up_activation",
    "routed_gate_up_epilogue",
    "routed_down",
    "routed_down_stage",
    "routed_down_mma",
    "shared_experts",
    "grid_barrier",
    "tail",
};

static_assert(sizeof(kPhaseClockNames) / sizeof(kPhaseClockNames[0])
                  == kPhaseClockCount);

/// Names no accumulated region: this clock is top-level and enters a total.
inline constexpr int kPhaseClockTopLevel = -1;

/// Each region's containing region, or `kPhaseClockTopLevel`.
///
/// The top-level regions are disjoint by construction -- every stage's mark is
/// reset when the stage begins and laps when it ends, and no stage's interval
/// spans another's -- so their sum is the accumulated CTA time of the launch
/// and a share of that total is a real share. A child's interval lies inside
/// its parent's, so a child may be compared with its parent but never added to
/// it or to a sibling of its parent.
inline constexpr int kPhaseClockParents[] = {
    kPhaseClockTopLevel,  // readiness_wait
    kPhaseClockTopLevel,  // router_score
    kPhaseClockTopLevel,  // latent_project
    kPhaseClockTopLevel,  // routed_queue
    kPhaseClockTopLevel,  // latent_quantize
    kPhaseClockTopLevel,  // assignment
    kPhaseClockTopLevel,  // publish
    kPhaseClockTopLevel,  // routed_gate_up
    kClockRoutedGateUp,   // routed_gate_up_stage
    kClockRoutedGateUp,   // routed_gate_up_mma
    kClockRoutedGateUpStage,  // routed_gate_up_tma_issue
    kClockRoutedGateUpStage,  // routed_gate_up_tma_wait
    kClockRoutedGateUpStage,  // routed_gate_up_ring_full
    kClockRoutedGateUpMma,    // routed_gate_up_mma_issue
    kClockRoutedGateUp,   // routed_gate_up_activation
    kClockRoutedGateUp,   // routed_gate_up_epilogue
    kPhaseClockTopLevel,  // routed_down
    kClockRoutedDown,     // routed_down_stage
    kClockRoutedDown,     // routed_down_mma
    kPhaseClockTopLevel,  // shared_experts
    kPhaseClockTopLevel,  // grid_barrier
    kPhaseClockTopLevel,  // tail
};

static_assert(sizeof(kPhaseClockParents) / sizeof(kPhaseClockParents[0])
                  == kPhaseClockCount);

/// Whether every parent is a strictly earlier region, so the tree terminates.
inline constexpr bool phase_clock_parents_are_acyclic() {
    for (int clock = 0; clock < kPhaseClockCount; ++clock) {
        const int parent = kPhaseClockParents[clock];
        if (parent == kPhaseClockTopLevel) continue;
        if (parent < 0 || parent >= clock) return false;
    }
    return true;
}

static_assert(phase_clock_parents_are_acyclic(),
              "a region's containing region must be declared before it, so a "
              "walk to the top-level ancestor terminates");

// ---------------------------------------------------------------------------

}  // namespace kimi_k3_decode
