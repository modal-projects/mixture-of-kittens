#pragma once

/// The profile band a measured launch laps into.
///
/// Three pointers into the appended counter region, handed to the phases as a
/// `ScheduleClocks`. An unmeasured launch carries nulls and pays one predicate
/// per lap, which is why this is a struct of pointers rather than a template
/// parameter: the production launch and the profiled one are the same code.

#include "shapes.cuh"

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// The profile band.
//
// One accumulated wait and one makespan maximum per readiness edge, plus one
// makespan maximum per queue, all written only when the launch asked for them.
// A measured launch carries a null pointer here and pays one predicate.
// ---------------------------------------------------------------------------

struct ScheduleClocks {
    unsigned long long *edge_wait;
    unsigned long long *edge_makespan;
    unsigned long long *queue_makespan;

    __device__ __forceinline__ bool enabled() const {
        return edge_wait != nullptr;
    }

    __device__ __forceinline__ unsigned long long now() const {
        return edge_wait == nullptr
            ? 0ull
            : static_cast<unsigned long long>(clock64());
    }

    /// Accumulate one CTA's cycles inside one edge's wait, and keep the
    /// longest such wait any CTA of the launch paid on that edge.
    __device__ __forceinline__ void lap_edge(
        const int edge,
        const unsigned long long started
    ) const {
        if (edge_wait == nullptr || threadIdx.x != 0) return;
        const unsigned long long elapsed =
            static_cast<unsigned long long>(clock64()) - started;
        atomicAdd(&edge_wait[edge], elapsed);
        atomicMax(&edge_makespan[edge], elapsed);
    }

    /// Keep the longest interval from the retained barrier to the point one
    /// CTA finished draining one queue -- the queue's makespan.
    __device__ __forceinline__ void mark_queue(
        const int queue,
        const unsigned long long launched
    ) const {
        if (queue_makespan == nullptr || threadIdx.x != 0) return;
        atomicMax(
            &queue_makespan[queue],
            static_cast<unsigned long long>(clock64()) - launched);
    }
};

__device__ __forceinline__ ScheduleClocks schedule_clocks(
    const Scratch &scratch,
    const bool profiled
) {
    if (!profiled) return ScheduleClocks{nullptr, nullptr, nullptr};
    auto *const band = reinterpret_cast<unsigned long long *>(
        &scratch.schedule[kScheduleClockBegin]);
    constexpr int edge_words = kScheduleEdgeCount;
    return ScheduleClocks{
        band,
        band + edge_words,
        band + 2 * edge_words};
}

/// Words of the appended region a profiled launch clears.
inline constexpr int kScheduleClockWords =
    kScheduleClockEnd - kScheduleClockBegin;

// ---------------------------------------------------------------------------

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
