#pragma once

/// The model's shapes, the workspace's regions, and the phase counters.
///
/// Every offset a decode step reads is derived here from the shapes above it,
/// so a region that moves moves everything below it by construction rather
/// than by anyone remembering to.

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
inline constexpr float kRoutedAccumulatorScale = 0x1p24f;
inline constexpr float kRoutedAccumulatorScaleInverse = 0x1p-24f;

static_assert(
    kRoutedAccumulatorScale * kRoutedAccumulatorScaleInverse == 1.0f);

// Thirty-seven live counters and a 64-bit accumulator band that ends the
// region. The band is what sets the count: twenty-two accumulated regions at
// two slots each is forty-four slots, which does not fit above slot 36 in 64.
// Two 256-byte scratch regions hold 128, and this region is the first, so
// widening it moves every offset below by exactly one alignment grain and
// nothing else.
static constexpr int NUM_PHASE_COUNTERS = 128;
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
// Production routed-down contributions use a Q24 signed 64-bit sum. Integer
// addition is order-independent, so expert CTAs can accumulate concurrently
// without making rank-skewed launches choose different floating-point orders.
// Private expert-stage tests retain the float view of the same aligned region.
inline constexpr int kRoutedAccumulatorBytes =
    kSituScaleBytes
    + scratch_byte_region_bytes(
        kMaxRoutes
        * (kRoutedIntermediateSize / kTensorParallelSize / 32));
inline constexpr int kSharedGateBytes =
    kRoutedAccumulatorBytes
    + scratch_byte_region_bytes(
        kMaxTokens * kLatentSize * sizeof(long long));
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

// The dependency-local scheduler's own counters, appended rather than carved
// out of the phase region's headroom. Appending is what keeps the candidate's
// state provably disjoint from every counter the production kernel and the
// private stages already own: nothing below this line moves, and no slot is
// shared between the two schedules. One 512-byte grain holds the seven queue
// counters, the readiness arrivals, and the profile-only edge accumulators.
inline constexpr int kScheduleBytes =
    kRouterScoreBytes + scratch_region_bytes(kMaxTokens * kNumExperts);
static constexpr int NUM_SCHEDULE_COUNTERS = 128;

static constexpr int SCRATCH_BYTES =
    kScheduleBytes + scratch_region_bytes(NUM_SCHEDULE_COUNTERS);

static_assert(kLatentMxfp8Bytes == 40704);
static_assert(kLatentScaleBytes == 499456);
static_assert(kSituMxfp8Bytes == 513792);
static_assert(kSituScaleBytes == 1300224);
static_assert(kRoutedAccumulatorBytes == 1324800);
static_assert(kSharedGateBytes == 4994816);
static_assert(kSharedUpBytes == 5191424);
static_assert(kSharedActivatedBytes == 5388032);
static_assert(kTailNormalizedBytes == 5584640);
static_assert(kTailSharedShardBytes == 6502144);
static_assert(kLatentXBytes == 6731520);
static_assert(kUnitExpertBytes == 7649024);
static_assert(kRouterScoreBytes == 7652608);
static_assert(kScheduleBytes == 8111360);
static_assert(SCRATCH_BYTES == 8111872);
static_assert(kScheduleBytes % SCRATCH_ALIGNMENT == 0,
              "the appended scheduler counters must start on a scratch grain");

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
// Shared gate/up units publish completion here before shared-down consumers
// read their outputs.
inline constexpr int kGateUpArrivals = 35;
// Which CTA owns this launch's timeout record, plus one; zero means unclaimed.
//
// One word for the whole launch rather than one per timeout slot. The two slots
// above belong to two families of waits that cannot both be running -- the tail
// only starts once the queues have drained -- and the question the word answers
// is the same for both: which single waiter's `(slot, code)` pair is the one
// that got published. `timeout_publication.cuh` is the protocol; every kernel
// whose waits report through it clears this word at entry.
inline constexpr int kTimeoutClaim = 36;
static_assert(kPersistentTimeoutPhase < NUM_PHASE_COUNTERS);
static_assert(kGateUpArrivals < NUM_PHASE_COUNTERS);
static_assert(kTimeoutClaim < NUM_PHASE_COUNTERS);

// ---------------------------------------------------------------------------

}  // namespace kimi_k3_decode
