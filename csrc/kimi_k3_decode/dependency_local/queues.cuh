#pragma once

/// The queues, the readiness edges, and the bounded waits over them.
///
/// A CTA claims units out of a queue's ticket counter, publishes its arrivals
/// into that queue's arrival counter, and waits on an earlier queue's arrivals
/// before it reads that queue's output. Every wait is bounded by the same
/// clock budget the production barriers use and reports its own timeout code,
/// so a broken edge traps by name instead of hanging the device.

#include "clocks.cuh"

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// Queues and readiness, on the appended counters.
// ---------------------------------------------------------------------------

/// Claim this CTA's next unit of one queue, or -1 once the queue is drained.
static __device__ int claim_schedule_unit(
    const Scratch &scratch,
    const int queue,
    const int units,
    int *const slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int ticket = atomicAdd(
            reinterpret_cast<unsigned int *>(
                &scratch.schedule[kScheduleQueueBegin + queue]),
            1u);
        *slot = ticket < static_cast<unsigned int>(units)
            ? static_cast<int>(ticket)
            : -1;
    }
    __syncthreads();
    return *slot;
}

/// Claim up to `batch` adjacent units of one queue with one atomic.
static __device__ int claim_schedule_batch(
    const Scratch &scratch,
    const int queue,
    const int units,
    const int batch,
    int *const begin_slot,
    int *const end_slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int ticket = atomicAdd(
            reinterpret_cast<unsigned int *>(
                &scratch.schedule[kScheduleQueueBegin + queue]),
            static_cast<unsigned int>(batch));
        if (ticket < static_cast<unsigned int>(units)) {
            const int begin = static_cast<int>(ticket);
            *begin_slot = begin;
            *end_slot = begin + batch < units ? begin + batch : units;
        } else {
            *begin_slot = -1;
            *end_slot = -1;
        }
    }
    __syncthreads();
    return *begin_slot;
}

/// Bounded spin on one readiness counter, then acquire at device scope.
///
/// The counter may be anywhere -- nine edges spin on the appended region and
/// one on the compacted assignment counts -- so this takes the pointer and
/// `wait_edge` below is what decides which pointer it is.
///
/// `record_timeout_and_trap` is `persistent_sync.cuh`'s, so a stalled edge
/// records into the same timeout slot the production waits use and reports its
/// own code from `kScheduleEdges`. The recorded counter index is offset by the
/// phase region's width, so a schedule counter and a phase counter are never
/// confused in that one word.
static __device__ void wait_for_schedule_count(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int *__restrict__ const counter,
    const int diagnostic,
    const int target,
    const int error_code
) {
    if (threadIdx.x == 0) {
        const std::uint64_t started = clock64();
        while (load_relaxed_gpu(counter) < static_cast<std::uint32_t>(target)) {
            if (wait_timed_out(started, clock64())) {
                record_timeout_and_trap(
                    scratch, error_flag, diagnostic, error_code);
            }
            __nanosleep(64);
        }
    }
    __syncthreads();
    __threadfence();
}

/// The same wait, acquiring at system scope.
///
/// Used only where the acquiring CTA goes on to tell peer ranks that this
/// rank's collective buffer is complete: the coordinator's own release is only
/// transitive over the producers' releases if it acquires them at the scope it
/// republishes at.
static __device__ void wait_for_schedule_count_system(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int *__restrict__ const counter,
    const int diagnostic,
    const int target,
    const int error_code
) {
    if (threadIdx.x == 0) {
        const std::uint64_t started = clock64();
        while (load_relaxed_gpu(counter) < static_cast<std::uint32_t>(target)) {
            if (wait_timed_out(started, clock64())) {
                record_timeout_and_trap(
                    scratch, error_flag, diagnostic, error_code);
            }
            __nanosleep(64);
        }
    }
    __syncthreads();
    __threadfence_system();
}

/// Take one readiness edge, deriving every part of the wait from its table row.
///
/// This is the only way the kernel waits. The counter it spins on, the arrival
/// count it spins to, the scope it acquires at, the diagnostic slot it records,
/// and the code it traps with are all read out of `kScheduleEdges[EDGE]` at
/// compile time, so the dependency graph the source contracts check is
/// literally the graph the kernel executes. There is no second description of
/// an edge that could drift from the first: adding a row without a wait leaves
/// an unused enumerator, and taking a wait without a row does not compile.
///
/// `unit` names the shared column pair or the expert for the two edges whose
/// counter is indexed, and `supplied_target` is the arrival count for the six
/// whose target depends on the launch's shape or its path. The table says which
/// is which, and asserts that a static row carries a target and a dynamic row
/// does not.
///
/// The wait is charged to `readiness_wait` and the stage mark is left just past
/// it. Both halves of that matter, and neither is cosmetic:
///
///   * A stage clock that silently included its own entry wait would report a
///     candidate stage as slower than the production stage it is compared
///     against for no reason other than that the candidate waits inside the
///     stage where production waited at a barrier.
///   * The cycles have to land somewhere. `readiness_wait` is a top-level band,
///     so a launch's twelve top-level bands still account for the whole of the
///     accumulated CTA time and the wait share is a share of a denominator that
///     contains it. The per-edge counters below measure the same waiting split
///     by edge; they are diagnostic children of this band, not addends to it.
///
/// One clock reading serves both, taken just inside the wait, and that is a
/// measured requirement rather than tidiness. Charging the band from the end of
/// the *previous* region instead would keep the stage mark live across the spin
/// on top of the reading the per-edge counter already keeps there, and the extra
/// live pair costs the tensor instantiation two registers and the M = 128 step
/// half a percent -- for a few cycles of index arithmetic that no stage clock
/// wanted either. So the band and the ten edge counters measure the same
/// interval, and the band is the sum of the edges rather than merely wider than
/// it.
template<int EDGE>
static __device__ __forceinline__ void wait_edge(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const ScheduleClocks &edges,
    const PhaseClocks &clocks,
    unsigned long long *const mark,
    const int unit = 0,
    const int supplied_target = kScheduleTargetSuppliedAtWait
) {
    static_assert(EDGE >= 0 && EDGE < kScheduleEdgeCount);
    // Pulled out as integers rather than as a copy of the row, so nothing in
    // device code refers to the table's host storage outside a constant
    // expression.
    constexpr int kCounter = kScheduleEdges[EDGE].counter;
    constexpr int kCode = kScheduleEdges[EDGE].error_code;
    constexpr int kSpace = kScheduleEdges[EDGE].space;
    constexpr int kScope = kScheduleEdges[EDGE].scope;
    constexpr int kStaticTarget = kScheduleEdges[EDGE].static_target;
    constexpr bool kStaticallyTargeted =
        kScheduleEdges[EDGE].target_kind == kScheduleTargetStatic;
    constexpr bool kIndexed = kScheduleEdges[EDGE].counter_indexed;
    constexpr int kDiagnostic = schedule_edge_diagnostic(EDGE, 0);
    // The diagnostic of an indexed in-region edge is its base plus the unit,
    // proved here rather than assumed, because the wait adds the unit at
    // runtime and the table computes it at compile time.
    static_assert(
        schedule_edge_diagnostic(EDGE, 1)
            == kDiagnostic + (kIndexed && kSpace == kScheduleCounterInRegion
                                  ? 1
                                  : 0),
        "a diagnostic slot must be linear in the unit");
    static_assert(kScheduleEdges[EDGE].producer_queue
                      < kScheduleEdges[EDGE].consumer_queue,
                  "a wait may only be taken on a strictly earlier queue");

    const int target = kStaticallyTargeted ? kStaticTarget : supplied_target;
    const int index = kIndexed ? unit : 0;
    const int *__restrict__ const counter =
        kSpace == kScheduleCounterInExpertCounts
            ? &scratch.expert_counts[index]
            : &scratch.schedule[kCounter + index];
    const int diagnostic =
        kSpace == kScheduleCounterInExpertCounts ? kDiagnostic
                                                : kDiagnostic + index;

    // One reading, consumed by whichever of the two handles is live. Either can
    // be null on its own -- both are switched by the same launch argument, but
    // nothing here depends on that -- and each accumulator checks its own
    // pointer, so a reading taken for a disabled handle is simply dropped.
    const unsigned long long started = clocks.enabled() || edges.enabled()
        ? static_cast<unsigned long long>(clock64())
        : 0ull;
    if constexpr (kScope == kScheduleScopeSystem) {
        wait_for_schedule_count_system(
            scratch, error_flag, counter, diagnostic, target, kCode);
    } else {
        wait_for_schedule_count(
            scratch, error_flag, counter, diagnostic, target, kCode);
    }
    edges.lap_edge(EDGE, started);
    *mark = clocks.lap(kClockReadinessWait, started);
}

/// Release this unit's writes at device scope, then count one arrival.
static __device__ void publish_schedule_count(
    const Scratch &scratch,
    const int counter
) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(
            reinterpret_cast<unsigned int *>(&scratch.schedule[counter]), 1u);
    }
}

/// Release this unit's writes at system scope, then count one arrival.
///
/// Every producer whose output leaves this rank publishes through here. The
/// collective buffer is read back through the fabric by the peers' tail roles,
/// so a device-scope release would order those writes for this rank's CTAs and
/// for nobody else.
static __device__ void publish_schedule_count_system(
    const Scratch &scratch,
    const int counter
) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(
            reinterpret_cast<unsigned int *>(&scratch.schedule[counter]), 1u);
    }
}

// ---------------------------------------------------------------------------

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
