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

/// Roughly fifteen seconds of B300 clocks.
///
/// Long enough that no legitimate rendezvous can reach it, short enough that a
/// lost peer surfaces as a device trap rather than a wedged GPU.
inline constexpr std::uint64_t kWaitTimeoutClocks = 30'000'000'000ULL;

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
