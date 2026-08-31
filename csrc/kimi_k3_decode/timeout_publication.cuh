#pragma once

#include "../serial_sync.cuh"
#include "types.cuh"

#include <cstdint>

namespace kimi_k3_decode {
namespace timeout {

// How a launch that gave up waiting says so, once, in two words.
//
// Every bounded wait in this kernel family reports two things when its clock
// budget runs out: the site's own error code, in the caller-visible
// `error_flag`, and the counter it was spinning on, in one of the scratch
// timeout slots. A host that reads a nonzero code has to be able to trust the
// slot beside it, because the code names the wait and the slot names the
// publication that never arrived, and a reader given one waiter's code next to
// another waiter's slot is sent to a wait that did not time out.
//
// Several waiters giving up at once is the normal case rather than the exotic
// one. They share one fifteen-second budget and, on the dependency-local
// schedule, ten different readiness edges spin against it, so a producer that
// never publishes stalls every consumer queued behind it and they all give up
// within a few hundred cycles of one another. Independent exchanges -- even
// with the code decided by one compare-and-swap -- leave a window in which the
// code of the winner is already published while the slot beside it still holds
// whatever the last launch left there.
//
// So the pair is published under a claim:
//
//   1. Every waiter that gives up races for the claim word, which this launch
//      cleared. Exactly one compare-and-swap succeeds.
//   2. The winner writes the slot, releases it at system scope, and only then
//      publishes the final nonzero code. The order is the guarantee: a nonzero
//      code means the slot that belongs to it is already visible.
//   3. A loser does not trap while the pair is half written. It waits, bounded
//      by the same budget, for the code to appear, and only then traps.
//
// Step 3 matters because a trap does not stay inside the thread that took it:
// it ends the launch, and a loser trapping first would take the winner down
// between its two writes and leave the reader with a stale slot.

using serial_sync::wait_timed_out;

/// The value a launch's claim word holds before any waiter claims it.
inline constexpr unsigned int kUnclaimed = 0u;

/// What the winner leaves behind: the CTA index that owns the record, plus one.
///
/// Plus one because zero has to keep meaning "nobody claimed this". Recording
/// the CTA is worth the arithmetic twice over: a real hang names the CTA whose
/// wait gave up first, and the concurrent injection test can check that the
/// published slot and code are *that* CTA's rather than merely a pair that
/// appears in the table.
__host__ __device__ inline constexpr unsigned int claim_token(const int block) {
    return static_cast<unsigned int>(block) + 1u;
}

/// The CTA a claim word names, or -1 when nothing claimed it.
__host__ __device__ inline constexpr int claiming_block(
    const unsigned int claim
) {
    return static_cast<int>(claim) - 1;
}

static_assert(claiming_block(kUnclaimed) == -1);
static_assert(claiming_block(claim_token(0)) == 0);
static_assert(claiming_block(claim_token(147)) == 147);

static __device__ __forceinline__ std::uint32_t load_relaxed_system(
    const unsigned int *const address
) {
    std::uint32_t value;
    asm volatile(
        "{ld.relaxed.sys.global.u32 %0, [%1];}"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}

/// Clear this launch's claim word.
///
/// A claim can only have been set by a launch that trapped, and a trap takes
/// its context down with it, so this is not recovery from a survivable state.
/// It is what makes the sentinel mean "this launch": without it a diagnostic
/// could only ever be published once per workspace, and the second launch to
/// give up would trap with the first launch's pair still in place.
///
/// No barrier or fence orders this against the waits it protects, and none is
/// needed. A wait cannot give up until fifteen seconds of clocks after its own
/// CTA started, this runs in the first instructions of the launch, and
/// same-address atomics from every CTA of a launch serialize at L2 whatever the
/// surrounding fences say.
static __device__ __forceinline__ void clear_claim(const Scratch &scratch) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicExch_system(
            reinterpret_cast<unsigned int *>(&scratch.phase[kTimeoutClaim]),
            kUnclaimed);
    }
}

/// Publish one waiter's `(slot, code)` pair, then trap this launch.
///
/// `slot_index` is the phase word this family of waits records into --
/// `kPersistentTimeoutPhase` for the one-launch kernels and their schedule,
/// `kTailTimeoutPhase` for the tail's own rendezvous -- and `counter_index`
/// names the counter inside it. `error_code` is the site's, and must be
/// nonzero, because zero is what "no wait timed out" is spelled as.
static __device__ __forceinline__ void publish_and_trap(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int slot_index,
    const int counter_index,
    const int error_code
) {
    auto *const claim = reinterpret_cast<unsigned int *>(
        &scratch.phase[kTimeoutClaim]);
    auto *const code = reinterpret_cast<unsigned int *>(error_flag);

    // Every atomic here is system scope, and not only because a nonzero code
    // has to reach a host that is about to read it out of a dead context: the
    // injection probes put all three words in mapped host memory, where a
    // device-scope read-modify-write is not the operation it is on device
    // memory. Two waiters that both believed they had won the claim would then
    // publish two pairs into two words, which is the failure this protocol
    // exists to make impossible.
    if (atomicCAS_system(claim, kUnclaimed,
                         claim_token(static_cast<int>(blockIdx.x)))
        == kUnclaimed) {
        atomicExch_system(
            reinterpret_cast<unsigned int *>(&scratch.phase[slot_index]),
            static_cast<unsigned int>(counter_index));
        // The release that makes the slot readable before the code that sends
        // a reader to it.
        __threadfence_system();
        atomicExch_system(code, static_cast<unsigned int>(error_code));
        __threadfence_system();
    } else {
        // Bounded, and it is expected to fall through immediately: the winner
        // publishes in two atomics. The bound is here for the case the winner
        // cannot get there -- if it shared a warp with a loser, the loser's
        // branch runs first -- where a stale record is still better than a
        // hung device.
        const std::uint64_t started = clock64();
        while (load_relaxed_system(code) == 0u) {
            if (wait_timed_out(started,
                               static_cast<std::uint64_t>(clock64()))) {
                break;
            }
            __nanosleep(64);
        }
    }
    asm volatile("trap;");
}

}  // namespace timeout
}  // namespace kimi_k3_decode
