#pragma once

/// The probe that proves a missed publication traps instead of hanging.
///
/// Takes one readiness edge with nothing to publish into its counter, once
/// alone and once from a full co-resident grid, and lets the bounded wait give
/// up. Test-only, but it lives here because it launches the same waits the
/// kernel does and has to stay next to them.

#include "launch.cuh"

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// The missed-publication probe.
// ---------------------------------------------------------------------------

/// Take one edge through `wait_edge` with nothing to publish into its counter.
///
/// The point of the probe is to exercise the wait the kernel actually takes,
/// so it goes through `wait_edge` rather than reproducing it: the counter, the
/// diagnostic slot, the code, and -- the part a hand-written probe would have
/// got wrong -- the acquire scope all come from the same table row the kernel
/// reads. Two of the ten edges are system-scope, and a probe that spun on them
/// at device scope would have left `wait_for_schedule_count_system` untested
/// while appearing to test it.
template<int EDGE>
static __device__ __forceinline__ void probe_one_edge(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int unit,
    const int target
) {
    const ScheduleClocks edges{nullptr, nullptr, nullptr};
    const PhaseClocks clocks{nullptr};
    unsigned long long mark = 0ull;
    wait_edge<EDGE>(
        scratch, error_flag, edges, clocks, &mark, unit, target);
}

/// Turn the host's runtime edge index into the template argument it names.
///
/// Unrolled over the whole table rather than switched over the two shapes,
/// which is what guarantees every edge -- and therefore both scopes -- has a
/// reachable instantiation.
template<int EDGE = 0>
static __device__ __forceinline__ void probe_edge_dispatch(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int edge,
    const int unit,
    const int target
) {
    if constexpr (EDGE < kScheduleEdgeCount) {
        if (edge == EDGE) {
            probe_one_edge<EDGE>(scratch, error_flag, unit, target);
            return;
        }
        probe_edge_dispatch<EDGE + 1>(
            scratch, error_flag, edge, unit, target);
    }
}

/// Take one of the schedule's bounded waits on a counter nobody publishes.
///
/// This is how "a missed publication traps at its own named site" is checked at
/// runtime rather than only in the sources. It has to be a separate kernel for
/// two reasons. A trap ends the launch as `cudaErrorLaunchFailure` and takes
/// the context with it, so the two diagnostics only survive if the caller
/// placed the pointers they are written through in mapped host memory -- which
/// a real decode workspace never is. And the schedule kernel's codegen must not
/// depend on the probe existing at all.
__global__ void schedule_wait_probe_kernel(
    std::uint8_t *__restrict__ scratch_bytes,
    int *__restrict__ error_flag,
    const int edge,
    const int unit,
    const int target
) {
    const Scratch scratch = scratch_view(scratch_bytes);
    probe_edge_dispatch<>(scratch, error_flag, edge, unit, target);
}

/// Give up on every edge at once, from one CTA per `(edge, unit)` pair.
///
/// The single-edge probe above proves that a stalled edge reports its own site.
/// It cannot prove the part that only exists when several waits give up
/// together: that the published slot and code are *one* waiter's. Every CTA
/// here spins on a counter nobody publishes, against one clock budget they all
/// started within a few microseconds of, so they reach `publish_and_trap`
/// together and race for the claim.
///
/// The unit is the CTA's own, so the ten edges are covered at several units
/// each and the two indexed edges spin on several different counters. A reader
/// of the result can therefore tell a matching pair from a plausible one: the
/// claim word names the CTA that won, and `(edge, unit)` follows from the CTA.
__global__ void schedule_wait_probe_concurrent_kernel(
    std::uint8_t *__restrict__ scratch_bytes,
    int *__restrict__ error_flag,
    const int target
) {
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);
    probe_edge_dispatch<>(
        scratch, error_flag, block % kScheduleEdgeCount,
        block / kScheduleEdgeCount, target);
}

/// Run the probe on host memory the trap's writes can be read back out of.
///
/// Both tensors must be pinned, because that is the whole point: a trapped
/// launch's device writes are unreadable, and the test's subject is exactly
/// what the trap recorded.
inline void schedule_wait_probe_for_testing(
    const at::Tensor &scratch,
    const at::Tensor &error_flag,
    const std::int64_t edge,
    const std::int64_t unit,
    const std::int64_t target
) {
    TORCH_CHECK(scratch.is_pinned() && error_flag.is_pinned(),
                "MoK: the schedule wait probe reads its diagnostics back after "
                "a trap, so both buffers must be mapped host memory");
    TORCH_CHECK(scratch.dtype() == at::kByte
                    && scratch.numel() >= SCRATCH_BYTES,
                "MoK: the schedule wait probe needs a whole byte workspace");
    TORCH_CHECK(error_flag.dtype() == at::kInt && error_flag.numel() == 1,
                "MoK: the schedule wait probe needs one int32 error flag");
    TORCH_CHECK(edge >= 0 && edge < kScheduleEdgeCount,
                "MoK: the schedule wait probe needs a readiness edge index");
    TORCH_CHECK(unit >= 0 && unit < kNumExperts,
                "MoK: the schedule wait probe's unit must name a column pair "
                "or an expert");
    schedule_wait_probe_kernel<<<1, kDecodeCtaThreads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<std::uint8_t *>(scratch.data_ptr()),
        reinterpret_cast<int *>(error_flag.data_ptr()),
        static_cast<int>(edge),
        static_cast<int>(unit),
        static_cast<int>(target));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// Stall every edge at `units_per_edge` units apiece, concurrently.
///
/// The grid is one CTA per `(edge, unit)` pair, which is what makes the claim
/// contested. `units_per_edge` is capped at the shared-pair count so that the
/// edge whose counter is indexed inside the appended region stays on its own
/// band of counters rather than reaching into the readiness arrivals past it --
/// the point of the probe is a stalled wait, not an out-of-band read.
inline void schedule_wait_probe_concurrent_for_testing(
    const at::Tensor &scratch,
    const at::Tensor &error_flag,
    const std::int64_t units_per_edge,
    const std::int64_t target
) {
    TORCH_CHECK(scratch.is_pinned() && error_flag.is_pinned(),
                "MoK: the schedule wait probe reads its diagnostics back after "
                "a trap, so both buffers must be mapped host memory");
    TORCH_CHECK(scratch.dtype() == at::kByte
                    && scratch.numel() >= SCRATCH_BYTES,
                "MoK: the schedule wait probe needs a whole byte workspace");
    TORCH_CHECK(error_flag.dtype() == at::kInt && error_flag.numel() == 1,
                "MoK: the schedule wait probe needs one int32 error flag");
    TORCH_CHECK(units_per_edge >= 1 && units_per_edge <= kScheduleSharedPairs,
                "MoK: the concurrent schedule wait probe takes between one and ",
                kScheduleSharedPairs, " units of every edge");
    TORCH_CHECK(target >= 1,
                "MoK: a probe target of zero is already satisfied");
    const int blocks =
        kScheduleEdgeCount * static_cast<int>(units_per_edge);
    TORCH_CHECK(blocks <= kPersistentCtas,
                "MoK: the concurrent schedule wait probe must fit the resident "
                "grid so that every waiter is running when it gives up");
    schedule_wait_probe_concurrent_kernel<<<
        blocks, kDecodeCtaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<std::uint8_t *>(scratch.data_ptr()),
        reinterpret_cast<int *>(error_flag.data_ptr()),
        static_cast<int>(target));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
