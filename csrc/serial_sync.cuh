#pragma once

#include <cstdint>

/// Wrap-safe comparisons for the monotonic serial numbers this repository uses
/// to synchronize ranks and CTAs, plus the one bound every spin is held to.
///
/// Every counter here is a 32-bit serial number that only ever rises, is never
/// reset by the host, and therefore wraps. Comparing two such numbers with `<`
/// is wrong exactly once per 2^32 arrivals, and the failure is silent: the
/// comparison stops being satisfiable and a barrier falls straight through
/// instead of waiting. Comparing unsigned *differences* against half the range
/// is correct across the wrap as long as the two numbers stay within 2^31 of
/// each other, which holds for any barrier whose participants are within one
/// launch of one another.
namespace serial_sync {

/// Roughly fifteen seconds of B300 clocks. This is what production ships.
///
/// Long enough that no legitimate rendezvous can reach it, short enough that a
/// lost peer surfaces as a device trap rather than a wedged GPU.
inline constexpr std::uint64_t kWaitTimeoutBaseClocks = 30'000'000'000ULL;

/// How much of that budget this build was compiled with, and why it is a knob.
///
/// compute-sanitizer instruments every access, which slows a launch by one to
/// two orders of magnitude. A cross-CTA rendezvous that takes microseconds
/// unmeasured can then take longer than fifteen seconds, the bounded spin burns
/// its budget, and the `trap` that ends it takes the launch down as
/// `cudaErrorLaunchFailure`. The tool then reports zero hazards for a run that
/// never finished, which is a statement about the watchdog rather than about
/// races -- and sections 43 and 63 of the Task 11b report are the five runs
/// that were lost to exactly that.
///
/// So a sanitizer-only image compiles with a wider budget. Three things make
/// that safe to have as a knob at all:
///
/// * it is compile-time, so a build that does not raise it is byte-identical to
///   one from before the knob existed -- there is no branch and no read;
/// * the base is asserted below and unconditionally, so nothing can change what
///   production ships by changing the scale;
/// * `_kimi_k3_decode_wait_timeout_budget` reports the base, the scale, and
///   their product as compiled, and
///   `test_the_wait_budget_is_the_one_this_image_declares` requires the
///   compiled scale to equal what the image says it built with. A production
///   image declares nothing and so must compile a scale of one.
///
/// It is never read at runtime and never settable from a process.
#ifndef MOK_WAIT_TIMEOUT_SCALE
#define MOK_WAIT_TIMEOUT_SCALE 1
#endif
inline constexpr std::uint64_t kWaitTimeoutScale = MOK_WAIT_TIMEOUT_SCALE;

static_assert(kWaitTimeoutBaseClocks == 30'000'000'000ULL,
              "the budget production ships is fifteen seconds of B300 clocks; "
              "a sanitizer image scales it and may not restate it");
static_assert(kWaitTimeoutScale >= 1,
              "the scale widens the budget or leaves it alone; a build that "
              "narrowed it would trap on a rendezvous production tolerates");

inline constexpr std::uint64_t kWaitTimeoutClocks =
    kWaitTimeoutBaseClocks * kWaitTimeoutScale;

/// Report whether a published generation moved past a consumed one.
__host__ __device__ inline constexpr bool generation_advanced(
    const std::uint32_t observed,
    const std::uint32_t consumed
) {
    const std::uint32_t difference = observed - consumed;
    return difference != 0u && difference < 0x80000000u;
}

/// Report whether a monotonically rising arrival counter reached its target.
__host__ __device__ inline constexpr bool barrier_reached(
    const std::uint32_t observed,
    const std::uint32_t target
) {
    return (observed - target) < 0x80000000u;
}

/// Report whether a bounded spin has burned through its clock budget.
__host__ __device__ inline constexpr bool wait_timed_out(
    const std::uint64_t started,
    const std::uint64_t current
) {
    return current - started >= kWaitTimeoutClocks;
}

}  // namespace serial_sync
