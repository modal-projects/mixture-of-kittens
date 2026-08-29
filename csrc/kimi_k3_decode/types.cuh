#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace kimi_k3_decode {

inline constexpr int kHiddenSize = 7168;
inline constexpr int kLatentSize = 3584;
inline constexpr int kRoutedIntermediateSize = 3072;
inline constexpr int kSharedIntermediateSize = 6144;
inline constexpr int kNumExperts = 896;
inline constexpr int kTopK = 16;
inline constexpr int kTensorParallelSize = 8;
inline constexpr int kMaxTokens = 128;
inline constexpr int kMaxRoutes = kMaxTokens * kTopK;

// The counters occupy one 256-byte scratch region either way, so the persistent
// kernel's own slots come out of headroom that was already reserved.
static constexpr int NUM_PHASE_COUNTERS = 64;
static constexpr int SCRATCH_ALIGNMENT = 256;

// Hidden states and both projection weights are read through 16-byte vector loads
// and, on the tcgen05 path, through TMA descriptors, so their first element must
// sit on a 16-byte boundary. Their row pitches are multiples of 16 bytes already,
// which leaves the base pointer as the only thing a caller can get wrong.
static constexpr int VECTOR_ALIGNMENT = 16;

/// Round one region of 32-bit words up to the scratch alignment.
inline constexpr int scratch_region_bytes(const int words) {
    return ((words * 4 + SCRATCH_ALIGNMENT - 1) / SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT;
}

/// Round an arbitrary byte region up to the scratch alignment.
inline constexpr int scratch_byte_region_bytes(const int bytes) {
    return ((bytes + SCRATCH_ALIGNMENT - 1) / SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT;
}

// Scratch regions, in declaration order, each starting on a 256-byte boundary.
inline constexpr int kPhaseBytes = 0;
inline constexpr int kExpertIdBytes =
    kPhaseBytes + scratch_region_bytes(NUM_PHASE_COUNTERS);
inline constexpr int kExpertWeightBytes =
    kExpertIdBytes + scratch_region_bytes(kMaxRoutes);
inline constexpr int kExpertCountBytes =
    kExpertWeightBytes + scratch_region_bytes(kMaxRoutes);
inline constexpr int kExpertOffsetBytes =
    kExpertCountBytes + scratch_region_bytes(kNumExperts);
inline constexpr int kAssignmentTokenBytes =
    kExpertOffsetBytes + scratch_region_bytes(kNumExperts + 1);
inline constexpr int kAssignmentSlotBytes =
    kAssignmentTokenBytes + scratch_region_bytes(kMaxRoutes);
inline constexpr int kLatentMxfp8Bytes =
    kAssignmentSlotBytes + scratch_region_bytes(kMaxRoutes);
inline constexpr int kLatentScaleBytes =
    kLatentMxfp8Bytes
    + scratch_byte_region_bytes(kMaxTokens * kLatentSize);
inline constexpr int kSituMxfp8Bytes =
    kLatentScaleBytes
    + scratch_byte_region_bytes(kMaxTokens * (kLatentSize / 32));
inline constexpr int kSituScaleBytes =
    kSituMxfp8Bytes
    + scratch_byte_region_bytes(
        kMaxRoutes * (kRoutedIntermediateSize / kTensorParallelSize));
inline constexpr int kRoutedAccumulatorBytes =
    kSituScaleBytes
    + scratch_byte_region_bytes(
        kMaxRoutes
        * (kRoutedIntermediateSize / kTensorParallelSize / 32));
inline constexpr int kSharedGateBytes =
    kRoutedAccumulatorBytes
    + scratch_byte_region_bytes(kMaxTokens * kLatentSize * sizeof(float));
inline constexpr int kSharedUpBytes =
    kSharedGateBytes
    + scratch_byte_region_bytes(
        kMaxTokens * (kSharedIntermediateSize / kTensorParallelSize)
        * sizeof(__nv_bfloat16));
inline constexpr int kSharedActivatedBytes =
    kSharedUpBytes
    + scratch_byte_region_bytes(
        kMaxTokens * (kSharedIntermediateSize / kTensorParallelSize)
        * sizeof(__nv_bfloat16));
// The tail's normalized latent and its own reduced shared shard. Both are read
// through TMA descriptors on the tcgen05 path, so their row pitches are already
// multiples of 16 bytes and the 256-byte region alignment covers the base.
inline constexpr int kTailNormalizedBytes =
    kSharedActivatedBytes
    + scratch_byte_region_bytes(
        kMaxTokens * (kSharedIntermediateSize / kTensorParallelSize)
        * sizeof(__nv_bfloat16));
inline constexpr int kTailSharedShardBytes =
    kTailNormalizedBytes
    + scratch_byte_region_bytes(
        kMaxTokens * kLatentSize * sizeof(__nv_bfloat16));

// The persistent kernel's own regions. They are appended, so every offset the
// private stages and their tests already pin stays exactly where it was.
//
// `latent_x` is the routed latent the private route-and-project stage returns
// as a tensor. The one-launch kernel has no tensor to return it in and cannot
// borrow another region -- the tail's normalized latent is the only one of the
// right shape and it is still live when the routed experts read this one.
inline constexpr int kLatentXBytes =
    kTailSharedShardBytes
    + scratch_byte_region_bytes(
        kMaxTokens * (kHiddenSize / kTensorParallelSize)
        * sizeof(__nv_bfloat16));
// One work unit per expert that any token routed to, compacted so a worker
// never claims an empty range. A token's sixteen routes are distinct experts,
// so an expert collects at most `kMaxTokens` assignments and therefore always
// fits one 128-row MMA batch; the batch's bounds come from `expert_offsets`.
inline constexpr int kUnitExpertBytes =
    kLatentXBytes
    + scratch_byte_region_bytes(
        kMaxTokens * kLatentSize * sizeof(__nv_bfloat16));
// Every token's raw router score for every expert. Scoring a token reads the
// whole 12.8 MB router weight, which is far more than one CTA can stream in
// the time the rest of a decode step takes, so the persistent kernel splits a
// token's experts over many CTAs and lands their scores here. Selection then
// reads this back and picks the top sixteen with nothing left to contract.
inline constexpr int kRouterScoreBytes =
    kUnitExpertBytes + scratch_region_bytes(kNumExperts);

static constexpr int SCRATCH_BYTES =
    kRouterScoreBytes + scratch_region_bytes(kMaxTokens * kNumExperts);

static_assert(kLatentMxfp8Bytes == 40448);
static_assert(kLatentScaleBytes == 499200);
static_assert(kSituMxfp8Bytes == 513536);
static_assert(kSituScaleBytes == 1299968);
static_assert(kRoutedAccumulatorBytes == 1324544);
static_assert(kSharedGateBytes == 3159552);
static_assert(kSharedUpBytes == 3356160);
static_assert(kSharedActivatedBytes == 3552768);
static_assert(kTailNormalizedBytes == 3749376);
static_assert(kTailSharedShardBytes == 4666880);
static_assert(kLatentXBytes == 4896256);
static_assert(kUnitExpertBytes == 5813760);
static_assert(kRouterScoreBytes == 5817344);
static_assert(SCRATCH_BYTES == 6276096);

// Generation-tagged completion counters. Each role's last CTA clears its arrival
// counter and bumps its generation, so a reused workspace never needs a host reset.
inline constexpr int kRouterArrivals = 0;
inline constexpr int kRouterGeneration = 1;
inline constexpr int kProjectionArrivals = 2;
inline constexpr int kProjectionGeneration = 3;
inline constexpr int kExpertQuantizationArrivals = 4;
inline constexpr int kExpertQuantizationGeneration = 5;
inline constexpr int kExpertCompletionArrivals = 7;
inline constexpr int kExpertCompletionGeneration = 8;
inline constexpr int kSharedGateArrivals = 9;
inline constexpr int kSharedGateGeneration = 10;
inline constexpr int kSharedUpArrivals = 11;
inline constexpr int kSharedUpGeneration = 12;
inline constexpr int kSharedActivationArrivals = 13;
inline constexpr int kSharedActivationGeneration = 14;
inline constexpr int kSharedDownArrivals = 15;
inline constexpr int kSharedDownGeneration = 16;
// A zero value means no wait timed out. On timeout, the waiting CTA records
// the published generation slot (gate, up, or activation) before trapping.
inline constexpr int kSharedTimeoutPhase = 17;
static_assert(kSharedTimeoutPhase < NUM_PHASE_COUNTERS);

// Fused TP8 tail phases. The coordinator CTA owns both cross-rank edges, so its
// entry and exit generations have no arrival counter; the reduce, shard, and
// drain roles each count their CTAs into one arrival slot and the last arrival
// clears it before advancing that role's generation.
inline constexpr int kTailEntryGeneration = 18;
inline constexpr int kTailReduceArrivals = 19;
inline constexpr int kTailReduceGeneration = 20;
inline constexpr int kTailShardArrivals = 21;
inline constexpr int kTailShardGeneration = 22;
inline constexpr int kTailExitGeneration = 23;
inline constexpr int kTailDrainArrivals = 24;
inline constexpr int kTailDrainGeneration = 25;
// A zero value means no tail wait timed out. On timeout the waiting CTA records
// the generation slot it was waiting on before trapping.
inline constexpr int kTailTimeoutPhase = 26;
static_assert(kTailTimeoutPhase < NUM_PHASE_COUNTERS);

// The one-launch production kernel's own slots. The private stages never touch
// them, and the kernel clears every queue counter itself before its first grid
// barrier, so a reused workspace still needs no host reset.
inline constexpr int kGridArrivals = 27;
inline constexpr int kGridGeneration = 28;
inline constexpr int kRouteLatentQueue = 29;
inline constexpr int kGateUpQueue = 30;
inline constexpr int kDownQueue = 31;
inline constexpr int kActivationArrivals = 32;
inline constexpr int kActiveExpertUnits = 33;
// A zero value means no persistent wait timed out. On timeout the waiting CTA
// records the counter slot it gave up on before trapping.
inline constexpr int kPersistentTimeoutPhase = 34;
static_assert(kPersistentTimeoutPhase < NUM_PHASE_COUNTERS);

// ---------------------------------------------------------------------------
// Benchmark-only phase clocks.
//
// A CTA reads `clock64()` around each region it runs and accumulates the delta
// into one shared counter per region, so a single guarded launch reports where
// its own resident CTAs spent their cycles. The counters are aligned pairs of
// the phase region's unused tail slots, read as one 64-bit accumulator each:
// a decode step is millions of cycles summed over 148 CTAs, which overflows 32
// bits, and appending a scratch region would move every offset the private
// stages pin.
// ---------------------------------------------------------------------------

/// One accumulated region, in the order the kernel runs them.
enum PhaseClock : int {
    kClockQueueClear = 0,
    kClockRouterScore,
    kClockLatentProject,
    kClockAssignments,
    kClockLatentQuantize,
    kClockRoutedGateUp,
    kClockRoutedGateUpStage,
    kClockRoutedGateUpMma,
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

static_assert(kPhaseClockBegin == 36);
static_assert(kPhaseClockBegin % 2 == 0,
              "a 64-bit accumulator must start on an aligned slot pair");
static_assert(kPhaseClockBegin > kPersistentTimeoutPhase,
              "the accumulators must not overlap a live counter");

/// Every accumulated region's reported name, in `PhaseClock` order.
inline constexpr const char *kPhaseClockNames[] = {
    "queue_clear",
    "router_score",
    "latent_project",
    "assignments",
    "latent_quantize",
    "routed_gate_up",
    "routed_gate_up_stage",
    "routed_gate_up_mma",
    "routed_down",
    "routed_down_stage",
    "routed_down_mma",
    "shared_experts",
    "grid_barrier",
    "tail",
};

static_assert(sizeof(kPhaseClockNames) / sizeof(kPhaseClockNames[0])
                  == kPhaseClockCount);

// ---------------------------------------------------------------------------
// Timeout diagnostics.
//
// A bounded wait that gives up writes two things before it traps: the phase
// slot it was waiting on, into the timeout counter for its half of the step,
// and one of the codes below, into the caller-visible `error_flag`. Both are
// needed. The slot alone cannot name the site, because several sites wait on
// the same counter -- the entry rendezvous and the reduce role both wait on
// `kTailEntryGeneration` -- and the flag alone does not survive a workspace
// whose scratch the caller never reads. Every code is nonzero, so zero keeps
// its meaning: no wait has ever timed out on this workspace.
// ---------------------------------------------------------------------------

inline constexpr int kErrorTailEntryRendezvous = 1;
inline constexpr int kErrorTailExitRendezvous = 2;
inline constexpr int kErrorTailCoordinatorShard = 3;
inline constexpr int kErrorTailReduceEntry = 4;
inline constexpr int kErrorTailShardReduce = 5;
inline constexpr int kErrorTailDrainExit = 6;
inline constexpr int kErrorPersistentGridBarrier = 7;
inline constexpr int kErrorPersistentActivation = 8;
inline constexpr int kErrorPersistentDownOrder = 9;

/// One bounded wait, named by the code it reports and the slots it writes.
struct TimeoutSite {
    const char *name;
    int code;
    int timeout_slot;
    int counter;
};

/// Every bounded wait either tail path can give up on, in code order.
///
/// The tests walk this table against the sources, so a wait added without a
/// code, or a code added without a wait, fails rather than silently trapping
/// with a diagnostic the caller cannot interpret.
inline constexpr TimeoutSite kTimeoutSites[] = {
    {"tail_entry_rendezvous", kErrorTailEntryRendezvous,
     kTailTimeoutPhase, kTailEntryGeneration},
    {"tail_exit_rendezvous", kErrorTailExitRendezvous,
     kTailTimeoutPhase, kTailExitGeneration},
    {"tail_coordinator_shard", kErrorTailCoordinatorShard,
     kTailTimeoutPhase, kTailShardGeneration},
    {"tail_reduce_entry", kErrorTailReduceEntry,
     kTailTimeoutPhase, kTailEntryGeneration},
    {"tail_shard_reduce", kErrorTailShardReduce,
     kTailTimeoutPhase, kTailReduceGeneration},
    {"tail_drain_exit", kErrorTailDrainExit,
     kTailTimeoutPhase, kTailExitGeneration},
    {"persistent_grid_barrier", kErrorPersistentGridBarrier,
     kPersistentTimeoutPhase, kGridGeneration},
    {"persistent_shared_activation", kErrorPersistentActivation,
     kPersistentTimeoutPhase, kActivationArrivals},
    {"persistent_down_order", kErrorPersistentDownOrder,
     kPersistentTimeoutPhase, kDownQueue},
};

inline constexpr int kTimeoutSiteCount =
    static_cast<int>(sizeof(kTimeoutSites) / sizeof(kTimeoutSites[0]));

static_assert(kTimeoutSiteCount == 9);
static_assert(kTimeoutSites[kTimeoutSiteCount - 1].code == kTimeoutSiteCount,
              "the timeout codes must be a dense nonzero range");

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
    __nv_bfloat16 *shared_gate;
    __nv_bfloat16 *shared_up;
    __nv_bfloat16 *shared_activated;
    __nv_bfloat16 *tail_normalized;
    __nv_bfloat16 *tail_shared_shard;
    __nv_bfloat16 *latent_x;
    int *unit_expert;
    float *router_scores;
    int *down_progress;
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
        reinterpret_cast<__nv_bfloat16 *>(base + kSharedGateBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kSharedUpBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kSharedActivatedBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kTailNormalizedBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kTailSharedShardBytes),
        reinterpret_cast<__nv_bfloat16 *>(base + kLatentXBytes),
        reinterpret_cast<int *>(base + kUnitExpertBytes),
        reinterpret_cast<float *>(base + kRouterScoreBytes),
        reinterpret_cast<int *>(base + kRouterScoreBytes),
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
