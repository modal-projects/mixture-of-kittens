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
// The dependency-local schedule: queues, counters, and readiness edges.
//
// The production kernel separates its phases with generation-tagged full-grid
// barriers, so every CTA waits for the slowest CTA of the phase it is leaving
// even when it has no data dependency on that CTA's output. This candidate
// keeps exactly one of those barriers -- the one that publishes this launch's
// cleared counters -- and replaces the rest with topologically ordered queues
// and bounded release/acquire readiness.
//
// Every CTA scans the queues below in this one forward order. A CTA leaves a
// queue only when that queue's ticket counter is exhausted, which means every
// unit of it is already held by a co-resident CTA, so a consumer's wait is
// bounded by work that is guaranteed to be running rather than by work that
// might never be claimed. Combined with the rule that every readiness edge
// points at a strictly earlier queue, that is what makes the scan deadlock
// free without a barrier between two adjacent queues.
// ---------------------------------------------------------------------------

/// The queues, in the order every CTA scans them.
enum ScheduleQueue : int {
    /// Latent project/quantize, shared gate/up, and the router's score shards.
    kQueueSource = 0,
    /// Shared activation, one unit per shared column block.
    kQueueSharedActivation,
    /// Assignment compaction: the expert-major table and the unit list.
    kQueueAssignment,
    /// Fused-W13 routed gate/up, one expert-pure unit per occupied expert.
    kQueueRoutedGateUp,
    /// Shared down, which reads every activation tile.
    kQueueSharedDown,
    /// Grouped routed down, seven units per occupied expert.
    kQueueRoutedDown,
    /// Publishing this rank's routed partial into the collective buffer.
    kQueuePublish,
    kScheduleQueueCount,
};

/// Counter slots inside the appended schedule region.
///
/// The queue tickets come first so one clearing thread per slot covers them,
/// and the readiness arrivals follow. Everything below
/// `kScheduleClearedCounters` is zeroed by block 0 before the one retained
/// full-grid barrier, which is what lets a workspace be reused and replayed
/// inside a CUDA graph with no host reset and no wrap arithmetic.
inline constexpr int kScheduleQueueBegin = 0;
inline constexpr int kScheduleSharedPairs = 6;
inline constexpr int kScheduleSharedPairBegin =
    kScheduleQueueBegin + kScheduleQueueCount;
inline constexpr int kScheduleScoreArrivals =
    kScheduleSharedPairBegin + kScheduleSharedPairs;
inline constexpr int kScheduleAssignmentArrivals = kScheduleScoreArrivals + 1;
inline constexpr int kScheduleLatentArrivals =
    kScheduleAssignmentArrivals + 1;
inline constexpr int kScheduleActivationArrivals =
    kScheduleLatentArrivals + 1;
inline constexpr int kScheduleSharedGateUpArrivals =
    kScheduleActivationArrivals + 1;
inline constexpr int kScheduleSharedDownArrivals =
    kScheduleSharedGateUpArrivals + 1;
inline constexpr int kScheduleRoutedDownArrivals =
    kScheduleSharedDownArrivals + 1;
inline constexpr int kSchedulePublishArrivals =
    kScheduleRoutedDownArrivals + 1;
inline constexpr int kScheduleClearedCounters =
    kSchedulePublishArrivals + 1;

static_assert(kScheduleQueueCount == 7);
static_assert(kScheduleSharedPairBegin == 7);
static_assert(kScheduleClearedCounters == 21);

/// Arrivals the publish queue produces, which is one per resident CTA.
///
/// Spelled here rather than borrowed from `persistent_sync.cuh`'s
/// `kPersistentCtas`, because that header includes this one. The schedule
/// header asserts the two agree.
inline constexpr int kSchedulePublishUnitsForTable = 148;

/// The slot one clearing thread owns.
__host__ __device__ inline constexpr int schedule_cleared_counter(
    const int index
) {
    return kScheduleQueueBegin + index;
}

/// What a schedule wait writes into the timeout slot.
///
/// A schedule counter lives in the appended region, so its own index would
/// collide numerically with a phase slot in the one word both halves record
/// into. Offsetting by the phase region's width keeps every recorded number
/// unambiguous without a second timeout word.
inline constexpr int kScheduleDiagnosticBase = NUM_PHASE_COUNTERS;

__host__ __device__ inline constexpr int schedule_diagnostic(
    const int schedule_counter
) {
    return kScheduleDiagnosticBase + schedule_counter;
}

// ---------------------------------------------------------------------------
// Phase clocks.
//
// A CTA reads `clock64()` around each region it runs and accumulates the delta
// into one shared counter per region, so a profiled launch reports where its
// own resident CTAs spent their cycles. The counters are aligned pairs of the
// phase region's tail slots, read as one 64-bit accumulator each: a decode step
// is millions of cycles summed over 148 CTAs, which overflows 32 bits.
//
// Nothing here is benchmark-private. A profiled launch is one predicate on a
// null pointer the caller already hands in, no stage compiles differently for
// it, and the routed gate/up band below is deliberately fine enough to name
// which part of the fused engine a future step is spending its time in --
// which is the only instrument the remaining performance work has.
// ---------------------------------------------------------------------------

/// One accumulated region, in the order the kernel runs them.
///
/// **These are not all disjoint, and a total may not be taken over all of
/// them.** The band is a two-level tree: `kPhaseClockParents` below says which
/// regions are top-level and which are refinements of a parent, and a total is
/// only meaningful over the top-level ones. `routed_gate_up` and its eight
/// children measure the same cycles at three depths -- `stage` contains
/// `tma_issue`, `tma_wait`, and `ring_full`; `mma` contains `mma_issue` -- so
/// summing the leaves of that subtree alone overcounts the phase by 76%.
///
/// The refinements are kept because they are the only instrument the remaining
/// gate/up work has: they say whether the weight ring or the contraction is the
/// binding constraint. They are diagnostic children, not addends.
enum PhaseClock : int {
    kClockReadinessWait = 0,
    kClockRouterScore,
    kClockLatentProject,
    kClockRoutedQueue,
    kClockLatentQuantize,
    /// Router selection compaction: the expert-major table and the unit list.
    kClockAssignment,
    /// Writing this rank's routed partial into the collective buffer.
    kClockPublish,
    kClockRoutedGateUp,
    kClockRoutedGateUpStage,
    kClockRoutedGateUpMma,
    /// Building descriptors and issuing a slab's two bulk copies.
    kClockRoutedGateUpTmaIssue,
    /// Waiting for a slab's payload and scales to land.
    kClockRoutedGateUpTmaWait,
    /// Waiting for the tensor core to free the stage about to be refilled --
    /// the ring being full is the only reason this is ever nonzero.
    kClockRoutedGateUpRingFull,
    /// Staging the scale quads into tensor memory and issuing the sixteen
    /// contractions of a slab.
    kClockRoutedGateUpMmaIssue,
    /// Gathering the batch's activation rows and scales for the whole of K.
    ///
    /// The one cost the engine cannot hand to the copy engine: the rows an
    /// expert's batch names are scattered through the assignment order. It is
    /// paid once per expert rather than once per `(task, slab)` pair, which is
    /// the measurement the engine's shape exists for, so it is reported on its
    /// own rather than folded into the staging above.
    kClockRoutedGateUpActivation,
    /// Reading the accumulator, pairing the gate and up row halves, and
    /// quantizing one task's 64 `situ` columns.
    kClockRoutedGateUpEpilogue,
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

static_assert(kPhaseClockCount == 22);
static_assert(kPhaseClockBegin == 84);
static_assert(kPhaseClockBegin % 2 == 0,
              "a 64-bit accumulator must start on an aligned slot pair");
static_assert(kPhaseClockBegin > kGateUpArrivals,
              "the accumulators must not overlap a live counter");

/// Every accumulated region's reported name, in `PhaseClock` order.
inline constexpr const char *kPhaseClockNames[] = {
    "readiness_wait",
    "router_score",
    "latent_project",
    "routed_queue",
    "latent_quantize",
    "assignment",
    "publish",
    "routed_gate_up",
    "routed_gate_up_stage",
    "routed_gate_up_mma",
    "routed_gate_up_tma_issue",
    "routed_gate_up_tma_wait",
    "routed_gate_up_ring_full",
    "routed_gate_up_mma_issue",
    "routed_gate_up_activation",
    "routed_gate_up_epilogue",
    "routed_down",
    "routed_down_stage",
    "routed_down_mma",
    "shared_experts",
    "grid_barrier",
    "tail",
};

static_assert(sizeof(kPhaseClockNames) / sizeof(kPhaseClockNames[0])
                  == kPhaseClockCount);

/// Names no accumulated region: this clock is top-level and enters a total.
inline constexpr int kPhaseClockTopLevel = -1;

/// Each region's containing region, or `kPhaseClockTopLevel`.
///
/// The top-level regions are disjoint by construction -- every stage's mark is
/// reset when the stage begins and laps when it ends, and no stage's interval
/// spans another's -- so their sum is the accumulated CTA time of the launch
/// and a share of that total is a real share. A child's interval lies inside
/// its parent's, so a child may be compared with its parent but never added to
/// it or to a sibling of its parent.
inline constexpr int kPhaseClockParents[] = {
    kPhaseClockTopLevel,  // readiness_wait
    kPhaseClockTopLevel,  // router_score
    kPhaseClockTopLevel,  // latent_project
    kPhaseClockTopLevel,  // routed_queue
    kPhaseClockTopLevel,  // latent_quantize
    kPhaseClockTopLevel,  // assignment
    kPhaseClockTopLevel,  // publish
    kPhaseClockTopLevel,  // routed_gate_up
    kClockRoutedGateUp,   // routed_gate_up_stage
    kClockRoutedGateUp,   // routed_gate_up_mma
    kClockRoutedGateUpStage,  // routed_gate_up_tma_issue
    kClockRoutedGateUpStage,  // routed_gate_up_tma_wait
    kClockRoutedGateUpStage,  // routed_gate_up_ring_full
    kClockRoutedGateUpMma,    // routed_gate_up_mma_issue
    kClockRoutedGateUp,   // routed_gate_up_activation
    kClockRoutedGateUp,   // routed_gate_up_epilogue
    kPhaseClockTopLevel,  // routed_down
    kClockRoutedDown,     // routed_down_stage
    kClockRoutedDown,     // routed_down_mma
    kPhaseClockTopLevel,  // shared_experts
    kPhaseClockTopLevel,  // grid_barrier
    kPhaseClockTopLevel,  // tail
};

static_assert(sizeof(kPhaseClockParents) / sizeof(kPhaseClockParents[0])
                  == kPhaseClockCount);

/// Whether every parent is a strictly earlier region, so the tree terminates.
inline constexpr bool phase_clock_parents_are_acyclic() {
    for (int clock = 0; clock < kPhaseClockCount; ++clock) {
        const int parent = kPhaseClockParents[clock];
        if (parent == kPhaseClockTopLevel) continue;
        if (parent < 0 || parent >= clock) return false;
    }
    return true;
}

static_assert(phase_clock_parents_are_acyclic(),
              "a region's containing region must be declared before it, so a "
              "walk to the top-level ancestor terminates");

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
inline constexpr int kErrorPersistentGateUpDownReadiness = 9;
// The dependency-local schedule's readiness edges. One code per wait site, so
// a stalled candidate names the edge that did not arrive rather than only the
// counter, which several sites would otherwise share.
inline constexpr int kErrorScheduleSharedPair = 10;
inline constexpr int kErrorScheduleScoreShards = 11;
inline constexpr int kErrorScheduleAssignment = 12;
inline constexpr int kErrorScheduleLatent = 13;
inline constexpr int kErrorScheduleActivation = 14;
inline constexpr int kErrorScheduleSharedGateUp = 15;
inline constexpr int kErrorScheduleExpertGateUp = 16;
inline constexpr int kErrorScheduleRoutedDown = 17;
inline constexpr int kErrorSchedulePublish = 18;
inline constexpr int kErrorScheduleSharedDown = 19;

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
    {"persistent_gate_up_down_readiness",
     kErrorPersistentGateUpDownReadiness,
     kPersistentTimeoutPhase, kGateUpArrivals},
    {"schedule_shared_activation_pair", kErrorScheduleSharedPair,
     kPersistentTimeoutPhase, schedule_diagnostic(kScheduleSharedPairBegin)},
    {"schedule_assignment_score_shards", kErrorScheduleScoreShards,
     kPersistentTimeoutPhase, schedule_diagnostic(kScheduleScoreArrivals)},
    {"schedule_gate_up_assignment", kErrorScheduleAssignment,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleAssignmentArrivals)},
    {"schedule_gate_up_latent", kErrorScheduleLatent,
     kPersistentTimeoutPhase, schedule_diagnostic(kScheduleLatentArrivals)},
    {"schedule_shared_down_activation", kErrorScheduleActivation,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleActivationArrivals)},
    {"schedule_shared_down_gate_up", kErrorScheduleSharedGateUp,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleSharedGateUpArrivals)},
    {"schedule_routed_down_gate_up", kErrorScheduleExpertGateUp,
     kPersistentTimeoutPhase, kGateUpArrivals},
    {"schedule_publish_routed_down", kErrorScheduleRoutedDown,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleRoutedDownArrivals)},
    {"schedule_tail_publish", kErrorSchedulePublish,
     kPersistentTimeoutPhase, schedule_diagnostic(kSchedulePublishArrivals)},
    {"schedule_tail_shared_down", kErrorScheduleSharedDown,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleSharedDownArrivals)},
};

inline constexpr int kTimeoutSiteCount =
    static_cast<int>(sizeof(kTimeoutSites) / sizeof(kTimeoutSites[0]));

static_assert(kTimeoutSiteCount == 19);
static_assert(kTimeoutSites[kTimeoutSiteCount - 1].code == kTimeoutSiteCount,
              "the timeout codes must be a dense nonzero range");

// ---------------------------------------------------------------------------
// The dependency-local schedule's readiness edges.
//
// Every wait the candidate takes is one row of this table. The table is what
// the source contracts and the runtime binding both read, so an edge cannot be
// added to the kernel without being named here, and it exists chiefly to carry
// one static proof: every edge's producer queue is strictly earlier in the one
// forward scan than its consumer. That is the whole deadlock argument, so it is
// checked by the compiler rather than by review.
// ---------------------------------------------------------------------------

/// One readiness edge, in the order the profile band reports them.
enum ScheduleEdgeId : int {
    kEdgeSharedActivationPair = 0,
    kEdgeAssignmentScoreShards,
    kEdgeGateUpAssignment,
    kEdgeGateUpLatent,
    kEdgeSharedDownActivation,
    kEdgeSharedDownGateUp,
    kEdgeRoutedDownGateUp,
    kEdgePublishRoutedDown,
    kEdgeTailPublish,
    kEdgeTailSharedDown,
    kScheduleEdgeCount,
};

/// Which array an edge's readiness counter lives in.
///
/// Nine of the ten are in the appended schedule region. The tenth is not:
/// routed down's per-expert readiness is already published into the compacted
/// assignment counts by the fused gate/up engine, and reading it there is what
/// keeps that edge expert-local.
enum ScheduleCounterSpace : int {
    kScheduleCounterInRegion = 0,
    kScheduleCounterInExpertCounts,
};

/// The release scope an edge's producer used, and therefore the acquire scope
/// its consumer owes.
enum ScheduleEdgeScope : int {
    kScheduleScopeDevice = 0,
    kScheduleScopeSystem,
};

/// Where an edge's arrival target comes from.
///
/// Six of the ten are not a property of the edge alone: the score shards scale
/// with the token count, the routed totals with the occupied expert count, and
/// the three shared-expert edges with which path the launch took, since the
/// core and tensor instantiations cut the shared columns differently. Those are
/// supplied at the wait, and the table records only that they must be, so a
/// wait that forgot one is a wait on zero arrivals rather than a wait on a
/// plausible-looking constant.
enum ScheduleEdgeTarget : int {
    kScheduleTargetStatic = 0,
    kScheduleTargetDynamic,
};

/// One readiness edge, and everything a wait on it needs.
///
/// This is deliberately the *whole* contract and not a summary of it. The
/// runtime `wait_edge<Edge>` reads its counter, its target, its acquire scope,
/// its diagnostic slot, and its error code out of this row and out of nowhere
/// else, so the static DAG the contracts check and the waits the kernel takes
/// cannot drift apart: there is only one description of an edge.
struct ScheduleEdge {
    const char *name;
    /// The queue that waits. `kScheduleQueueCount` names the TP8 tail, which
    /// every queue precedes.
    int consumer_queue;
    int producer_queue;
    /// The counter the wait spins on: an index into the appended region, or
    /// into `expert_counts` when `space` says so.
    int counter;
    int error_code;
    /// `kScheduleCounterInRegion` or `kScheduleCounterInExpertCounts`.
    int space;
    /// `kScheduleScopeDevice` or `kScheduleScopeSystem`.
    int scope;
    /// `kScheduleTargetStatic` or `kScheduleTargetDynamic`.
    int target_kind;
    /// The arrival count, when `target_kind` is static; -1 otherwise.
    int static_target;
    /// Whether the counter index is `counter + unit`, which the per-column-pair
    /// and per-expert edges are and the whole-queue edges are not.
    bool counter_indexed;
};

inline constexpr int kScheduleEdgeOutsideRegion = -1;
inline constexpr int kScheduleTargetSuppliedAtWait = -1;

/// The arrivals one fused gate/up unit publishes for its expert.
///
/// Spelled here because the wait's target has to come out of this table and
/// `expert_mxfp4::fused_w13` is not included yet; the schedule header asserts
/// it against the engine's own geometry.
inline constexpr int kScheduleExpertGateUpArrivals = 6;

inline constexpr ScheduleEdge kScheduleEdges[] = {
    {"shared_activation_pair", kQueueSharedActivation, kQueueSource,
     kScheduleSharedPairBegin, kErrorScheduleSharedPair,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetStatic, 2, true},
    {"assignment_score_shards", kQueueAssignment, kQueueSource,
     kScheduleScoreArrivals, kErrorScheduleScoreShards,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"gate_up_assignment", kQueueRoutedGateUp, kQueueAssignment,
     kScheduleAssignmentArrivals, kErrorScheduleAssignment,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetStatic, 1, false},
    {"gate_up_latent", kQueueRoutedGateUp, kQueueSource,
     kScheduleLatentArrivals, kErrorScheduleLatent,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"shared_down_activation", kQueueSharedDown, kQueueSharedActivation,
     kScheduleActivationArrivals, kErrorScheduleActivation,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"shared_down_gate_up", kQueueSharedDown, kQueueSource,
     kScheduleSharedGateUpArrivals, kErrorScheduleSharedGateUp,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"routed_down_gate_up", kQueueRoutedDown, kQueueRoutedGateUp,
     kGateUpArrivals, kErrorScheduleExpertGateUp,
     kScheduleCounterInExpertCounts, kScheduleScopeDevice,
     kScheduleTargetStatic, kScheduleExpertGateUpArrivals, true},
    {"publish_routed_down", kQueuePublish, kQueueRoutedDown,
     kScheduleRoutedDownArrivals, kErrorScheduleRoutedDown,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"tail_publish", kScheduleQueueCount, kQueuePublish,
     kSchedulePublishArrivals, kErrorSchedulePublish,
     kScheduleCounterInRegion, kScheduleScopeSystem,
     kScheduleTargetStatic, kSchedulePublishUnitsForTable, false},
    {"tail_shared_down", kScheduleQueueCount, kQueueSharedDown,
     kScheduleSharedDownArrivals, kErrorScheduleSharedDown,
     kScheduleCounterInRegion, kScheduleScopeSystem,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
};

/// The diagnostic slot a wait on one edge records, given the unit it is on.
///
/// The two counter spaces number differently: a schedule counter is offset past
/// the phase region so the two cannot be confused in the one word both write,
/// while `expert_counts` readiness reports the phase slot the production wait
/// on the same counter reports.
/// Host-side `constexpr` on purpose: every device use of it is a constant
/// expression, so nothing needs the table itself in device memory.
inline constexpr int schedule_edge_diagnostic(
    const int edge,
    const int unit
) {
    return kScheduleEdges[edge].space == kScheduleCounterInExpertCounts
        ? kScheduleEdges[edge].counter
        : schedule_diagnostic(
              kScheduleEdges[edge].counter
              + (kScheduleEdges[edge].counter_indexed ? unit : 0));
}

/// Whether every edge's declared scope is at least its producer's.
///
/// The two edges the coordinator takes before it tells the peers this rank is
/// done must acquire at system scope, because the producers they wait on
/// released at system scope and a device-scope acquire would make the
/// coordinator's own release non-transitive over them. Nothing else needs to.
inline constexpr bool schedule_tail_edges_acquire_at_system_scope() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        const bool crosses = kScheduleEdges[edge].consumer_queue
            == kScheduleQueueCount;
        if (crosses
                && kScheduleEdges[edge].scope != kScheduleScopeSystem) {
            return false;
        }
    }
    return true;
}

/// Whether every static target is a positive arrival count.
inline constexpr bool schedule_static_targets_are_positive() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        const bool is_static = kScheduleEdges[edge].target_kind
            == kScheduleTargetStatic;
        if (is_static != (kScheduleEdges[edge].static_target > 0)) {
            return false;
        }
    }
    return true;
}

static_assert(schedule_tail_edges_acquire_at_system_scope(),
              "an edge the coordinator takes before it publishes to the peers "
              "must acquire at system scope");
static_assert(schedule_static_targets_are_positive(),
              "a static target must be a positive arrival count, and a dynamic "
              "one must not carry a target the wait would ignore");

static_assert(
    static_cast<int>(sizeof(kScheduleEdges) / sizeof(kScheduleEdges[0]))
        == kScheduleEdgeCount);

/// Whether every readiness edge points at a strictly earlier queue.
inline constexpr bool schedule_edges_point_backward() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        if (kScheduleEdges[edge].producer_queue
                >= kScheduleEdges[edge].consumer_queue) {
            return false;
        }
    }
    return true;
}

/// Whether no edge waits on a producer inside its own queue.
inline constexpr bool schedule_edges_never_block_inside_a_queue() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        if (kScheduleEdges[edge].producer_queue
                == kScheduleEdges[edge].consumer_queue) {
            return false;
        }
    }
    return true;
}

/// Whether every edge's code is one of the tabulated timeout codes.
inline constexpr bool schedule_edge_codes_are_tabulated() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        bool found = false;
        for (int site = 0; site < kTimeoutSiteCount; ++site) {
            if (kTimeoutSites[site].code == kScheduleEdges[edge].error_code) {
                found = true;
            }
        }
        if (!found) return false;
        for (int other = 0; other < edge; ++other) {
            if (kScheduleEdges[other].error_code
                    == kScheduleEdges[edge].error_code) {
                return false;
            }
        }
    }
    return true;
}

/// Whether every edge that names an appended counter names a cleared one.
///
/// The candidate carries no generation arithmetic of its own, so a readiness
/// counter is only meaningful if this launch found it at zero. Block 0 zeroes
/// everything below `kScheduleClearedCounters` before the one retained
/// barrier, and this is what says the table cannot grow an edge outside that
/// band without the compiler noticing.
inline constexpr bool schedule_edge_counters_are_cleared_each_launch() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        if (kScheduleEdges[edge].space != kScheduleCounterInRegion) continue;
        const int counter = kScheduleEdges[edge].counter;
        if (counter < kScheduleQueueCount
                || counter >= kScheduleClearedCounters) {
            return false;
        }
    }
    return true;
}

static_assert(schedule_edge_counters_are_cleared_each_launch(),
              "a readiness counter the launch did not zero cannot be waited "
              "on without generation arithmetic the candidate does not have");
static_assert(schedule_edges_point_backward(),
              "every readiness edge must point at a strictly earlier queue, "
              "which is what makes the one forward scan deadlock free");
static_assert(schedule_edges_never_block_inside_a_queue(),
              "a queue may not block on a producer in the same queue");
static_assert(schedule_edge_codes_are_tabulated(),
              "every readiness edge needs its own tabulated timeout code");

/// Every edge's reported name, in `ScheduleEdgeId` order.
inline constexpr const char *kScheduleEdgeNames[] = {
    "shared_activation_pair",
    "assignment_score_shards",
    "gate_up_assignment",
    "gate_up_latent",
    "shared_down_activation",
    "shared_down_gate_up",
    "routed_down_gate_up",
    "publish_routed_down",
    "tail_publish",
    "tail_shared_down",
};

/// Every queue's reported name, in `ScheduleQueue` order.
inline constexpr const char *kScheduleQueueNames[] = {
    "source",
    "shared_activation",
    "assignment",
    "routed_gate_up",
    "shared_down",
    "routed_down",
    "publish",
};

static_assert(
    static_cast<int>(sizeof(kScheduleEdgeNames) / sizeof(kScheduleEdgeNames[0]))
        == kScheduleEdgeCount);
static_assert(
    static_cast<int>(
        sizeof(kScheduleQueueNames) / sizeof(kScheduleQueueNames[0]))
        == kScheduleQueueCount);

// ---------------------------------------------------------------------------
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
