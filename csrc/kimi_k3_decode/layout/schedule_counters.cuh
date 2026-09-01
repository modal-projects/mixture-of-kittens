#pragma once

/// The dependency-local schedule's queues and its appended counter region.
///
/// One ticket counter and one arrival counter per queue, in a region the
/// production kernel and every private stage leave untouched, so the candidate
/// costs the production side of an A/B nothing.

#include "shapes.cuh"

namespace kimi_k3_decode {

// The dependency-local schedule: queues, counters, and readiness edges.
//
// The production kernel separates its phases with generation-tagged full-grid
// barriers, so every CTA waits for the slowest CTA of the phase it is leaving
// even when it has no data dependency on that CTA's output. This candidate
// keeps exactly one of those barriers -- the one that publishes this launch's
// cleared counters -- and replaces the rest with topologically ordered queues
// and bounded release/acquire readiness.
//
// Every CTA scans the queues below in this one forward order. A CTA leaves a
// queue only when that queue's ticket counter is exhausted, which means every
// unit of it is already held by a co-resident CTA, so a consumer's wait is
// bounded by work that is guaranteed to be running rather than by work that
// might never be claimed. Combined with the rule that every readiness edge
// points at a strictly earlier queue, that is what makes the scan deadlock
// free without a barrier between two adjacent queues.
// ---------------------------------------------------------------------------

/// The queues, in the order every CTA scans them.
enum ScheduleQueue : int {
    /// Latent project/quantize, shared gate/up, and the router's score shards.
    kQueueSource = 0,
    /// Shared activation, one unit per shared column block.
    kQueueSharedActivation,
    /// Assignment compaction: the expert-major table and the unit list.
    kQueueAssignment,
    /// Fused-W13 routed gate/up, one expert-pure unit per occupied expert.
    kQueueRoutedGateUp,
    /// Shared down, which reads every activation tile.
    kQueueSharedDown,
    /// Grouped routed down, seven units per occupied expert.
    kQueueRoutedDown,
    /// Publishing this rank's routed partial into the collective buffer.
    kQueuePublish,
    kScheduleQueueCount,
};

/// Counter slots inside the appended schedule region.
///
/// The queue tickets come first so one clearing thread per slot covers them,
/// and the readiness arrivals follow. Everything below
/// `kScheduleClearedCounters` is zeroed by block 0 before the one retained
/// full-grid barrier, which is what lets a workspace be reused and replayed
/// inside a CUDA graph with no host reset and no wrap arithmetic.
inline constexpr int kScheduleQueueBegin = 0;
inline constexpr int kScheduleSharedPairs = 6;
inline constexpr int kScheduleSharedPairBegin =
    kScheduleQueueBegin + kScheduleQueueCount;
inline constexpr int kScheduleScoreArrivals =
    kScheduleSharedPairBegin + kScheduleSharedPairs;
inline constexpr int kScheduleAssignmentArrivals = kScheduleScoreArrivals + 1;
inline constexpr int kScheduleLatentArrivals =
    kScheduleAssignmentArrivals + 1;
inline constexpr int kScheduleActivationArrivals =
    kScheduleLatentArrivals + 1;
inline constexpr int kScheduleSharedGateUpArrivals =
    kScheduleActivationArrivals + 1;
inline constexpr int kScheduleSharedDownArrivals =
    kScheduleSharedGateUpArrivals + 1;
inline constexpr int kScheduleRoutedDownArrivals =
    kScheduleSharedDownArrivals + 1;
inline constexpr int kSchedulePublishArrivals =
    kScheduleRoutedDownArrivals + 1;
inline constexpr int kScheduleClearedCounters =
    kSchedulePublishArrivals + 1;

static_assert(kScheduleQueueCount == 7);
static_assert(kScheduleSharedPairBegin == 7);
static_assert(kScheduleClearedCounters == 21);

/// Arrivals the publish queue produces, which is one per resident CTA.
///
/// Spelled here rather than borrowed from `persistent_sync.cuh`'s
/// `kPersistentCtas`, because that header includes this one. The schedule
/// header asserts the two agree.
inline constexpr int kSchedulePublishUnitsForTable = 148;

/// The slot one clearing thread owns.
__host__ __device__ inline constexpr int schedule_cleared_counter(
    const int index
) {
    return kScheduleQueueBegin + index;
}

/// What a schedule wait writes into the timeout slot.
///
/// A schedule counter lives in the appended region, so its own index would
/// collide numerically with a phase slot in the one word both halves record
/// into. Offsetting by the phase region's width keeps every recorded number
/// unambiguous without a second timeout word.
inline constexpr int kScheduleDiagnosticBase = NUM_PHASE_COUNTERS;

__host__ __device__ inline constexpr int schedule_diagnostic(
    const int schedule_counter
) {
    return kScheduleDiagnosticBase + schedule_counter;
}

// ---------------------------------------------------------------------------

}  // namespace kimi_k3_decode
