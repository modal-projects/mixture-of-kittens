#pragma once

#include "../serial_sync.cuh"
#include "types.cuh"

#include <cstdint>

namespace kimi_k3_decode {
namespace persistent {

// Grid-wide scheduling primitives for the one-launch production kernel.
//
// The kernel runs a fixed, fully resident grid and walks the whole decode step
// inside it, so it needs two things the private stages never did: a barrier
// that separates one phase from the next across every CTA, and a queue that
// lets any CTA claim any logical task. Both are generation-tagged and
// wrap-safe, so a reused workspace never needs a host reset between steps.

/// Every CTA of the production grid, one per B300 SM.
///
/// The grid is fixed rather than derived from the shape because every phase
/// barrier below counts arrivals against it. The host proves before launching
/// that this many CTAs really do co-reside; a partially resident grid would
/// deadlock on the first barrier.
inline constexpr int kPersistentCtas = 148;

/// Dynamic shared memory every CTA requests, in bytes.
///
/// It has to cover the widest stage the kernel runs, which is one routed
/// gate/up unit at 143 KiB. Requesting more than half of an SM's 227 KiB is
/// also what makes the grid one CTA per SM, which is what the residency proof
/// and the tensor-memory allocation both depend on. The remaining 67 KiB is
/// headroom for the static shared memory the stages declare.
inline constexpr int kPersistentSharedBytes = 160 * 1024;

using serial_sync::barrier_reached;
using serial_sync::generation_advanced;
using serial_sync::wait_timed_out;

/// Roughly fifteen seconds of B300 clocks, the same bound the tail spins on.
inline constexpr std::uint64_t kWaitTimeoutClocks =
    serial_sync::kWaitTimeoutClocks;

/// The counters one launch must find at zero, as a contiguous scratch band.
///
/// A queue counter runs past its unit count by up to one overshoot per CTA, and
/// an in-phase arrival counter is never cleared by its own last arriver, so
/// neither is self-restoring the way the generation counters are. The kernel
/// clears them itself before its first grid barrier, which is what lets a
/// workspace be reused, and replayed inside a CUDA graph, with no host reset.
///
/// The grid barrier's own arrival counter is deliberately not in this band: its
/// last arriver clears it, and clearing it here would race with the CTAs that
/// reach the first barrier before block 0 has finished phase 0.
inline constexpr int kPersistentClearedBegin = kRouteLatentQueue;
inline constexpr int kPersistentClearedCounters = 4;

static_assert(kGateUpQueue == kPersistentClearedBegin + 1);
static_assert(kDownQueue == kPersistentClearedBegin + 2);
static_assert(kActivationArrivals == kPersistentClearedBegin + 3);
static_assert(kPersistentClearedBegin + kPersistentClearedCounters
                  <= NUM_PHASE_COUNTERS,
              "the cleared counter band must fit the scratch phase region");

/// The phase slot one clearing thread owns.
__host__ __device__ inline constexpr int cleared_counter(const int index) {
    return kPersistentClearedBegin + index;
}

static __device__ __forceinline__ std::uint32_t load_relaxed_gpu(
    const int *const address
) {
    std::uint32_t value;
    asm volatile(
        "{ld.relaxed.gpu.global.u32 %0, [%1];}"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}

/// Record the counter a bounded spin gave up on, then trap this launch.
///
/// `error_flag` is the caller-visible copy: the scratch slot names which
/// counter stalled, and the flag carries the site's own code, so a host that
/// only reads the flag still learns which wait failed without having to know
/// the scratch layout. Both are nonzero, so zero keeps meaning "no timeout".
static __device__ __forceinline__ void record_timeout_and_trap(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int counter_index,
    const int error_code
) {
    atomicExch(
        reinterpret_cast<unsigned int *>(
            &scratch.phase[kPersistentTimeoutPhase]),
        static_cast<unsigned int>(counter_index));
    atomicExch(reinterpret_cast<unsigned int *>(error_flag),
               static_cast<unsigned int>(error_code));
    __threadfence_system();
    asm volatile("trap;");
}

/// One CTA's view of the grid barrier's generation counter.
struct GridPhase {
    std::uint32_t target;
};

/// Latch this CTA's barrier baseline before any CTA can advance it.
///
/// The counter only moves when all `kPersistentCtas` CTAs have arrived, and a
/// CTA arrives only after latching, so the value read here is always the one
/// the previous launch left behind. Every CTA then passes the same number of
/// barriers, so their targets stay in step for the whole launch.
static __device__ GridPhase latch_grid_phase(
    const Scratch &scratch,
    std::uint32_t *const slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *slot = load_relaxed_gpu(&scratch.phase[kGridGeneration]);
    }
    __syncthreads();
    return GridPhase{*slot};
}

/// Hold every CTA until all of them have finished the current phase.
///
/// The fences are system-scope because one phase's writes leave this rank
/// entirely: the routed and shared partials land in the symmetric collective
/// buffer and the tail reads them back through the fabric. A device-scope
/// release would order them for this rank's CTAs and for nobody else.
///
/// The last arriver clears the arrival counter before advancing the
/// generation, and the fence between the two is what makes the clear visible
/// first. Any CTA that observes the new generation therefore takes its next
/// ticket against a counter that has already been reset.
static __device__ void grid_barrier(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    GridPhase &phase
) {
    __threadfence_system();
    __syncthreads();

    phase.target += 1u;
    if (threadIdx.x == 0) {
        auto *const arrivals = reinterpret_cast<unsigned int *>(
            &scratch.phase[kGridArrivals]);
        auto *const generation = reinterpret_cast<unsigned int *>(
            &scratch.phase[kGridGeneration]);
        const unsigned int ticket = atomicAdd(arrivals, 1u);
        if (ticket == static_cast<unsigned int>(kPersistentCtas - 1)) {
            atomicExch(arrivals, 0u);
            __threadfence();
            atomicAdd(generation, 1u);
        } else {
            const std::uint64_t started = clock64();
            while (!barrier_reached(
                load_relaxed_gpu(&scratch.phase[kGridGeneration]),
                phase.target
            )) {
                if (wait_timed_out(started, clock64())) {
                    record_timeout_and_trap(
                        scratch, error_flag, kGridGeneration,
                        kErrorPersistentGridBarrier);
                }
                __nanosleep(64);
            }
        }
    }
    // Thread 0's observation converges the CTA; every thread then acquires
    // before reading anything the previous phase produced.
    __syncthreads();
    __threadfence_system();
}

/// Claim this CTA's next unit of one phase's queue, or -1 once it is drained.
///
/// Thread 0 takes the ticket and broadcasts it, so the whole CTA runs one unit
/// at a time and the stage device functions keep their CTA-wide barriers. A
/// counter is cleared at kernel entry and then rises to `units` plus exactly
/// one refused ticket per CTA, because a CTA leaves the loop on its first
/// refusal. `persistent_kernel.cuh` static-asserts the resulting bound: the
/// longest queue is the tensor path's 25 150 down units, which with the 148-CTA
/// overshoot stops at 25 298 -- nowhere near an unsigned wrap.
static __device__ int claim_unit(
    const Scratch &scratch,
    const int queue_index,
    const int units,
    int *const slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int ticket = atomicAdd(
            reinterpret_cast<unsigned int *>(&scratch.phase[queue_index]), 1u);
        *slot = ticket < static_cast<unsigned int>(units)
            ? static_cast<int>(ticket)
            : -1;
    }
    __syncthreads();
    return *slot;
}

/// Spin until an in-phase arrival counter reaches `target`, then acquire.
///
/// Used once, for the only producer-consumer edge inside a phase: the tensor
/// path's shared-down units read the activated intermediate that the same
/// phase's activation units write. Tickets are handed out in increasing index
/// order, so every activation unit is already claimed by a resident CTA that
/// waits on nothing before any shared-down unit can be claimed.
static __device__ void wait_for_count(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int counter_index,
    const int target,
    const int error_code
) {
    if (threadIdx.x == 0) {
        const std::uint64_t started = clock64();
        while (load_relaxed_gpu(&scratch.phase[counter_index])
               < static_cast<std::uint32_t>(target)) {
            if (wait_timed_out(started, clock64())) {
                record_timeout_and_trap(
                    scratch, error_flag, counter_index, error_code);
            }
            __nanosleep(64);
        }
    }
    __syncthreads();
    __threadfence();
}

/// Release this unit's writes, then count it into an in-phase arrival counter.
static __device__ void publish_count(
    const Scratch &scratch,
    const int counter_index
) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(
            reinterpret_cast<unsigned int *>(&scratch.phase[counter_index]),
            1u);
    }
}

}  // namespace persistent
}  // namespace kimi_k3_decode
