#pragma once

/// The schedule's profile band, and the typed views over the workspace.
///
/// One wait accumulator and one makespan maximum per edge, one makespan per
/// queue, then `Scratch` and `PhaseClocks` -- the two handles every stage takes
/// by value so that none of them re-derives an offset.

#include "schedule_edges.cuh"

namespace kimi_k3_decode {

// The schedule's profile band: one wait accumulator and one makespan maximum
// per edge, plus one makespan maximum per queue. All 64-bit, all inside the
// appended region, and all written only by a profiled launch.
// ---------------------------------------------------------------------------

inline constexpr int kScheduleClockBegin = 24;
inline constexpr int kScheduleEdgeWaitBegin = kScheduleClockBegin;
inline constexpr int kScheduleEdgeMakespanBegin =
    kScheduleEdgeWaitBegin + 2 * kScheduleEdgeCount;
inline constexpr int kScheduleQueueMakespanBegin =
    kScheduleEdgeMakespanBegin + 2 * kScheduleEdgeCount;
inline constexpr int kScheduleClockEnd =
    kScheduleQueueMakespanBegin + 2 * kScheduleQueueCount;

static_assert(kScheduleClockBegin >= kScheduleClearedCounters,
              "the profile band must not overlap a live counter");
static_assert(kScheduleClockBegin % 2 == 0,
              "a 64-bit accumulator must start on an aligned slot pair");
static_assert(kScheduleClockEnd <= NUM_SCHEDULE_COUNTERS,
              "the profile band must fit the appended region");
static_assert(kScheduleEdgeWaitBegin == 24);
static_assert(kScheduleEdgeMakespanBegin == 44);
static_assert(kScheduleQueueMakespanBegin == 64);
static_assert(kScheduleClockEnd == 78);

/// Typed device pointers into one decode workspace.
struct Scratch {
    int *phase;
    int *expert_ids;
    float *expert_weights;
    int *expert_counts;
    int *expert_offsets;
    int *assignment_tokens;
    int *assignment_slots;
    std::uint8_t *latent_mxfp8;
    std::uint8_t *latent_scale;
    std::uint8_t *situ_mxfp8;
    std::uint8_t *situ_scale;
    float *routed_accumulator;
    long long *routed_accumulator_fixed;
    __nv_bfloat16 *shared_gate;
    __nv_bfloat16 *shared_up;
    __nv_bfloat16 *shared_activated;
    __nv_bfloat16 *tail_normalized;
    __nv_bfloat16 *tail_shared_shard;
    __nv_bfloat16 *latent_x;
    int *unit_expert;
    float *router_scores;
    /// The dependency-local schedule's appended counters. The production
    /// kernel and every private stage leave this region untouched.
    int *schedule;
};

__host__ __device__ inline Scratch scratch_view(std::uint8_t *base) {
    return Scratch{
        reinterpret_cast<int *>(base + kPhaseBytes),
        reinterpret_cast<int *>(base + kExpertIdBytes),
        reinterpret_cast<float *>(base + kExpertWeightBytes),
        reinterpret_cast<int *>(base + kExpertCountBytes),
        reinterpret_cast<int *>(base + kExpertOffsetBytes),
        reinterpret_cast<int *>(base + kAssignmentTokenBytes),
        reinterpret_cast<int *>(base + kAssignmentSlotBytes),
        base + kLatentMxfp8Bytes,
        base + kLatentScaleBytes,
        base + kSituMxfp8Bytes,
        base + kSituScaleBytes,
        reinterpret_cast<float *>(base + kRoutedAccumulatorBytes),
        reinterpret_cast<long long *>(base + kRoutedAccumulatorBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kSharedGateBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kSharedUpBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kSharedActivatedBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kTailNormalizedBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kTailSharedShardBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kLatentXBytes),
        reinterpret_cast<int *>(base + kUnitExpertBytes),
        reinterpret_cast<float *>(base + kRouterScoreBytes),
        reinterpret_cast<int *>(base + kScheduleBytes),
    };
}

/// A CTA's handle on the phase clocks, inert unless the launch enabled them.
///
/// It travels by value into every stage, so a stage neither reads the guard
/// from global memory nor branches on a kernel argument it was not given.
struct PhaseClocks {
    unsigned long long *counters;

    __device__ __forceinline__ bool enabled() const {
        return counters != nullptr;
    }

    /// Read this CTA's cycle counter, or zero when the launch is not profiled.
    __device__ __forceinline__ unsigned long long now() const {
        return counters == nullptr
            ? 0ull
            : static_cast<unsigned long long>(clock64());
    }

    __device__ __forceinline__ void add(
        const int index,
        const unsigned long long cycles
    ) const {
        if (counters != nullptr && threadIdx.x == 0) {
            atomicAdd(&counters[index], cycles);
        }
    }

    /// Accumulate the cycles since `started` and return a fresh reading.
    __device__ __forceinline__ unsigned long long lap(
        const int index,
        const unsigned long long started
    ) const {
        if (counters == nullptr) return 0ull;
        const unsigned long long current =
            static_cast<unsigned long long>(clock64());
        add(index, current - started);
        return current;
    }
};

__device__ __forceinline__ PhaseClocks phase_clocks(
    const Scratch &scratch,
    const bool profiled
) {
    return PhaseClocks{
        profiled ? reinterpret_cast<unsigned long long *>(
                       &scratch.phase[kPhaseClockBegin])
                 : nullptr};
}

/// Round an active token count up to the decode contract's capacity bucket.
inline constexpr int capacity_bucket(const int active_tokens) {
    int capacity = 1;
    while (capacity < active_tokens) capacity *= 2;
    return capacity;
}

// Every CTA in the fused route-and-project launch runs two warpgroups: the
// router spreads its 896 expert rows over all eight warps, and the tcgen05
// projection drives its MMA pipeline from the first warpgroup.
inline constexpr int kDecodeCtaThreads = 256;

/// Largest capacity bucket the direct-register CUDA-core projection covers.
inline constexpr int kMaxCoreCapacity = 8;

/// Accumulate the 8 BF16 products held in one 16-byte pair of vectors.
__device__ __forceinline__ float accumulate_bf16_octet(
    const float4 &left,
    const float4 &right,
    float total
) {
    const __nv_bfloat162 *const left_pairs =
        reinterpret_cast<const __nv_bfloat162 *>(&left);
    const __nv_bfloat162 *const right_pairs =
        reinterpret_cast<const __nv_bfloat162 *>(&right);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        const float2 a = __bfloat1622float2(left_pairs[pair]);
        const float2 b = __bfloat1622float2(right_pairs[pair]);
        total = fmaf(a.x, b.x, total);
        total = fmaf(a.y, b.y, total);
    }
    return total;
}

// Mixed W4A8 `kind::mxf8f6f4` block scaling runs at K=32, so both prepared
// expert contractions are stored at their native width, without padding.
inline constexpr int kRoutedIntermediateSizePerRank =
    kRoutedIntermediateSize / kTensorParallelSize;
inline constexpr int kExpertW1W3PackedRows = kRoutedIntermediateSizePerRank;
inline constexpr int kExpertW1W3PackedColumns = kLatentSize / 2;
inline constexpr int kExpertW1W3ScaleColumns = kLatentSize / 32;
inline constexpr int kExpertW2PackedRows = kLatentSize;
inline constexpr int kExpertW2PackedColumns = kRoutedIntermediateSizePerRank / 2;
inline constexpr int kExpertW2ScaleColumns = kRoutedIntermediateSizePerRank / 32;

}  // namespace kimi_k3_decode
