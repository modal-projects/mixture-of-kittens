#pragma once

// The dependency-local persistent schedule: one launch, one full-grid barrier.
//
// The production kernel in `persistent_kernel.cuh` separates its phases with
// five generation-tagged full-grid barriers. A barrier is a correct but coarse
// dependency: it makes every CTA wait for the slowest CTA of the phase it is
// leaving even when it has no data dependency on that CTA's output at all. The
// measured B300 profile of the production step puts roughly a fifth of a
// decode step inside those barriers, and route-major finalize and a transposed
// tail shard were both measured against that idle and rejected, so what is
// left to try is removing the barriers themselves.
//
// This header is that candidate. It keeps exactly one full-grid barrier -- the
// one that publishes this launch's cleared counters, which nothing else can
// establish -- and replaces the other four with seven topologically ordered
// task queues and bounded release/acquire readiness edges. The queues, the
// counters, and the dependency table all live in `types.cuh`, next to the
// appended scratch region that holds them.
//
// Why the scan is deadlock free, in full:
//
//   1. Every CTA walks the queues in the one forward order `ScheduleQueue`
//      declares. There is no path that revisits a queue.
//   2. A CTA leaves a queue only when that queue's ticket counter is
//      exhausted, which means every unit of it is already claimed. The host
//      proves all CTAs of the launch co-reside one per SM, so a claimed unit
//      is held by a running CTA and will complete.
//   3. Every readiness edge points at a strictly earlier queue --
//      `schedule_edges_point_backward` is a `static_assert`, not a comment --
//      so a CTA blocked in queue `k` is waiting only on units of queues below
//      `k`, all of which are claimed by (2) and none of which can be waiting
//      on queue `k`.
//   4. No edge names its own queue as its producer, so no CTA can be blocked
//      behind a unit of the queue it is itself blocking in.
//   5. Every wait is bounded by the same fifteen-second clock budget the
//      production waits use and reports its own timeout code, so a broken
//      edge surfaces as a named trap rather than as a hung device.
//
// The stages themselves are the production stages, called unchanged. Nothing
// here recomputes anything, so a candidate launch is bit-for-bit the
// production launch with a different order of arrival.

#include "kittens.cuh"

#include "expert_mxfp4.cuh"
#include "expert_mxfp4_fused_w13.cuh"
#include "expert_mxfp4_grouped.cuh"
#include "persistent_sync.cuh"
#include "router.cuh"
#include "shared.cuh"
#include "skinny_gemm.cuh"
#include "tail_reduce.cuh"
#include "tail_shard.cuh"
#include "tail_sync.cuh"
#include "types.cuh"

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// ---------------------------------------------------------------------------
// Shapes the schedule pins.
// ---------------------------------------------------------------------------

/// CUDA ordinals the candidate's residency cache covers.
///
/// Spelled here rather than borrowed from `persistent_kernel.cuh`, because that
/// header includes this one and the include cannot go both ways.
inline constexpr int kScheduleMaxCudaDevices = 32;

/// Queue claims one occupied expert's routed gate/up work is one unit, and the
/// arrivals that unit publishes as its six `situ` ranges complete.
///
/// Both are the production facts, restated here for the same include reason,
/// and both are asserted against the engine's own geometry below.
inline constexpr int kScheduleGateUpUnitsPerExpert = 1;
inline constexpr int kScheduleGateUpArrivalsPerExpert =
    expert_mxfp4::fused_w13::kFusedTasks;

static_assert(kScheduleGateUpArrivalsPerExpert == 6);
static_assert(kScheduleGateUpArrivalsPerExpert == kScheduleExpertGateUpArrivals,
              "routed down's per-expert wait reads its target from the edge "
              "table, which must name the number of arrivals the fused engine "
              "actually publishes");
static_assert(kScheduleGateUpArrivalsPerExpert
                      * expert_mxfp4::fused_w13::kFusedSituGroups
                  == expert_mxfp4::kSituGroups,
              "the arrivals must cover every situ quantization group, or "
              "grouped down would start on a partly written expert");

/// Logical units the publish queue is cut into.
///
/// One per production CTA, so the write pattern is exactly the grid-strided
/// loop the production kernel's publish phase runs with `block` replaced by a
/// claimed unit index. Every element is still written once by one thread, so
/// the published bits do not depend on which CTA claimed which slice.
inline constexpr int kSchedulePublishUnits = kPersistentCtas;

static_assert(kSchedulePublishUnits == kSchedulePublishUnitsForTable,
              "the tail's wait on the publish queue reads its target from the "
              "edge table, which must name the same number of units the queue "
              "actually has");

/// Latent quantization groups one projection unit owns.
///
/// A projection unit writes a contiguous column range of the routed latent and
/// MXFP8 groups are 32 contiguous columns, so a unit can quantize exactly the
/// groups it just produced and the separate grid-wide quantization pass -- and
/// the barrier in front of it -- disappear. The tensor path's 128-column tile
/// is four groups; the core path's 32-column block is one.
template<bool TENSOR_PATH>
inline constexpr int kLatentGroupsPerProjectionUnit =
    TENSOR_PATH ? skinny_gemm::kTileN / expert_mxfp4::kMmaK
                : skinny_gemm::kCoreColumnsPerCta / expert_mxfp4::kMmaK;

template<bool TENSOR_PATH>
inline constexpr int kProjectionUnits =
    TENSOR_PATH ? skinny_gemm::kTensorCtas : skinny_gemm::kCoreCtas;

template<bool TENSOR_PATH>
inline constexpr int kSharedGateUpUnits =
    TENSOR_PATH ? 2 * shared_experts::kTensorGateCtas
                : shared_experts::kCoreGateCtas;

template<bool TENSOR_PATH>
inline constexpr int kSharedActivationUnits =
    TENSOR_PATH ? shared_experts::kActivationCtas : 0;

template<bool TENSOR_PATH>
inline constexpr int kSharedDownUnits =
    TENSOR_PATH ? shared_experts::kTensorDownCtas
                : shared_experts::kCoreDownCtas;

static_assert(kLatentGroupsPerProjectionUnit<true> == 4);
static_assert(kLatentGroupsPerProjectionUnit<false> == 1);
static_assert(kProjectionUnits<true> * kLatentGroupsPerProjectionUnit<true>
                  == expert_mxfp4::kLatentGroups,
              "the tensor projection units must cover every latent group");
static_assert(kProjectionUnits<false> * kLatentGroupsPerProjectionUnit<false>
                  == expert_mxfp4::kLatentGroups,
              "the core projection units must cover every latent group");
static_assert(kScheduleSharedPairs == shared_experts::kTensorGateCtas,
              "one pair counter per shared column block");
static_assert(kScheduleSharedPairs == shared_experts::kActivationCtas,
              "an activation unit consumes exactly one shared column pair");
static_assert(kSharedGateUpUnits<true> == 2 * kScheduleSharedPairs,
              "the tensor path's shared gate and up units pair up");

// ---------------------------------------------------------------------------
// Counter bounds.
//
// Nothing inside a launch resets a queue ticket or a readiness arrival, so
// every one of them has to hold its own maximum without approaching the
// unsigned wrap. The bounds below are the reason the candidate needs no
// generation arithmetic for its own state: block 0 zeroes the whole band
// before the one retained barrier, and no counter can climb out of `int` from
// there.
// ---------------------------------------------------------------------------

/// The highest ticket any queue counter can reach.
///
/// A CTA leaves a queue on its first refused claim, so each of them can add
/// one maximum-width batch beyond the logical length.
inline constexpr int kScheduleLongestQueueUnits =
    kNumExperts * expert_mxfp4::grouped_pipeline::kGroupedDownUnits;
inline constexpr int kScheduleLongestTicket =
    kScheduleLongestQueueUnits + kRoutedClaimBatch * kPersistentCtas;

/// The highest value any readiness arrival counter can reach.
inline constexpr int kScheduleLargestArrival = kScheduleLongestQueueUnits;

inline constexpr int kScheduleCounterBound = 0x7fffffff;

static_assert(kScheduleLongestQueueUnits == 6272);
static_assert(kScheduleLongestTicket == 6864);
static_assert(kScheduleLongestTicket < kScheduleCounterBound,
              "a queue ticket must stay inside a signed 32-bit counter");
static_assert(kScheduleLargestArrival < kScheduleCounterBound,
              "a readiness arrival must stay inside a signed 32-bit counter");
static_assert(kMaxTokens * router::kScoreShards < kScheduleCounterBound);
static_assert(kSchedulePublishUnits + kPersistentCtas
                  < kScheduleCounterBound);
static_assert(kScheduleCounterBound == 0x7fffffff,
              "the bound is the signed 32-bit maximum, and every counter above "
              "is asserted strictly under it");

// ---------------------------------------------------------------------------
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
// Queues and readiness, on the appended counters.
// ---------------------------------------------------------------------------

/// Claim this CTA's next unit of one queue, or -1 once the queue is drained.
static __device__ int claim_schedule_unit(
    const Scratch &scratch,
    const int queue,
    const int units,
    int *const slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int ticket = atomicAdd(
            reinterpret_cast<unsigned int *>(
                &scratch.schedule[kScheduleQueueBegin + queue]),
            1u);
        *slot = ticket < static_cast<unsigned int>(units)
            ? static_cast<int>(ticket)
            : -1;
    }
    __syncthreads();
    return *slot;
}

/// Claim up to `batch` adjacent units of one queue with one atomic.
static __device__ int claim_schedule_batch(
    const Scratch &scratch,
    const int queue,
    const int units,
    const int batch,
    int *const begin_slot,
    int *const end_slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int ticket = atomicAdd(
            reinterpret_cast<unsigned int *>(
                &scratch.schedule[kScheduleQueueBegin + queue]),
            static_cast<unsigned int>(batch));
        if (ticket < static_cast<unsigned int>(units)) {
            const int begin = static_cast<int>(ticket);
            *begin_slot = begin;
            *end_slot = begin + batch < units ? begin + batch : units;
        } else {
            *begin_slot = -1;
            *end_slot = -1;
        }
    }
    __syncthreads();
    return *begin_slot;
}

/// Bounded spin on one readiness counter, then acquire at device scope.
///
/// The counter may be anywhere -- nine edges spin on the appended region and
/// one on the compacted assignment counts -- so this takes the pointer and
/// `wait_edge` below is what decides which pointer it is.
///
/// `record_timeout_and_trap` is `persistent_sync.cuh`'s, so a stalled edge
/// records into the same timeout slot the production waits use and reports its
/// own code from `kScheduleEdges`. The recorded counter index is offset by the
/// phase region's width, so a schedule counter and a phase counter are never
/// confused in that one word.
static __device__ void wait_for_schedule_count(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int *__restrict__ const counter,
    const int diagnostic,
    const int target,
    const int error_code
) {
    if (threadIdx.x == 0) {
        const std::uint64_t started = clock64();
        while (load_relaxed_gpu(counter) < static_cast<std::uint32_t>(target)) {
            if (wait_timed_out(started, clock64())) {
                record_timeout_and_trap(
                    scratch, error_flag, diagnostic, error_code);
            }
            __nanosleep(64);
        }
    }
    __syncthreads();
    __threadfence();
}

/// The same wait, acquiring at system scope.
///
/// Used only where the acquiring CTA goes on to tell peer ranks that this
/// rank's collective buffer is complete: the coordinator's own release is only
/// transitive over the producers' releases if it acquires them at the scope it
/// republishes at.
static __device__ void wait_for_schedule_count_system(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int *__restrict__ const counter,
    const int diagnostic,
    const int target,
    const int error_code
) {
    if (threadIdx.x == 0) {
        const std::uint64_t started = clock64();
        while (load_relaxed_gpu(counter) < static_cast<std::uint32_t>(target)) {
            if (wait_timed_out(started, clock64())) {
                record_timeout_and_trap(
                    scratch, error_flag, diagnostic, error_code);
            }
            __nanosleep(64);
        }
    }
    __syncthreads();
    __threadfence_system();
}

/// Take one readiness edge, deriving every part of the wait from its table row.
///
/// This is the only way the kernel waits. The counter it spins on, the arrival
/// count it spins to, the scope it acquires at, the diagnostic slot it records,
/// and the code it traps with are all read out of `kScheduleEdges[EDGE]` at
/// compile time, so the dependency graph the source contracts check is
/// literally the graph the kernel executes. There is no second description of
/// an edge that could drift from the first: adding a row without a wait leaves
/// an unused enumerator, and taking a wait without a row does not compile.
///
/// `unit` names the shared column pair or the expert for the two edges whose
/// counter is indexed, and `supplied_target` is the arrival count for the six
/// whose target depends on the launch's shape or its path. The table says which
/// is which, and asserts that a static row carries a target and a dynamic row
/// does not.
///
/// The wait is charged to `readiness_wait` and the stage mark is left just past
/// it. Both halves of that matter, and neither is cosmetic:
///
///   * A stage clock that silently included its own entry wait would report a
///     candidate stage as slower than the production stage it is compared
///     against for no reason other than that the candidate waits inside the
///     stage where production waited at a barrier.
///   * The cycles have to land somewhere. `readiness_wait` is a top-level band,
///     so a launch's twelve top-level bands still account for the whole of the
///     accumulated CTA time and the wait share is a share of a denominator that
///     contains it. The per-edge counters below measure the same waiting split
///     by edge; they are diagnostic children of this band, not addends to it.
///
/// One clock reading serves both, taken just inside the wait, and that is a
/// measured requirement rather than tidiness. Charging the band from the end of
/// the *previous* region instead would keep the stage mark live across the spin
/// on top of the reading the per-edge counter already keeps there, and the extra
/// live pair costs the tensor instantiation two registers and the M = 128 step
/// half a percent -- for a few cycles of index arithmetic that no stage clock
/// wanted either. So the band and the ten edge counters measure the same
/// interval, and the band is the sum of the edges rather than merely wider than
/// it.
template<int EDGE>
static __device__ __forceinline__ void wait_edge(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const ScheduleClocks &edges,
    const PhaseClocks &clocks,
    unsigned long long *const mark,
    const int unit = 0,
    const int supplied_target = kScheduleTargetSuppliedAtWait
) {
    static_assert(EDGE >= 0 && EDGE < kScheduleEdgeCount);
    // Pulled out as integers rather than as a copy of the row, so nothing in
    // device code refers to the table's host storage outside a constant
    // expression.
    constexpr int kCounter = kScheduleEdges[EDGE].counter;
    constexpr int kCode = kScheduleEdges[EDGE].error_code;
    constexpr int kSpace = kScheduleEdges[EDGE].space;
    constexpr int kScope = kScheduleEdges[EDGE].scope;
    constexpr int kStaticTarget = kScheduleEdges[EDGE].static_target;
    constexpr bool kStaticallyTargeted =
        kScheduleEdges[EDGE].target_kind == kScheduleTargetStatic;
    constexpr bool kIndexed = kScheduleEdges[EDGE].counter_indexed;
    constexpr int kDiagnostic = schedule_edge_diagnostic(EDGE, 0);
    // The diagnostic of an indexed in-region edge is its base plus the unit,
    // proved here rather than assumed, because the wait adds the unit at
    // runtime and the table computes it at compile time.
    static_assert(
        schedule_edge_diagnostic(EDGE, 1)
            == kDiagnostic + (kIndexed && kSpace == kScheduleCounterInRegion
                                  ? 1
                                  : 0),
        "a diagnostic slot must be linear in the unit");
    static_assert(kScheduleEdges[EDGE].producer_queue
                      < kScheduleEdges[EDGE].consumer_queue,
                  "a wait may only be taken on a strictly earlier queue");

    const int target = kStaticallyTargeted ? kStaticTarget : supplied_target;
    const int index = kIndexed ? unit : 0;
    const int *__restrict__ const counter =
        kSpace == kScheduleCounterInExpertCounts
            ? &scratch.expert_counts[index]
            : &scratch.schedule[kCounter + index];
    const int diagnostic =
        kSpace == kScheduleCounterInExpertCounts ? kDiagnostic
                                                : kDiagnostic + index;

    // One reading, consumed by whichever of the two handles is live. Either can
    // be null on its own -- both are switched by the same launch argument, but
    // nothing here depends on that -- and each accumulator checks its own
    // pointer, so a reading taken for a disabled handle is simply dropped.
    const unsigned long long started = clocks.enabled() || edges.enabled()
        ? static_cast<unsigned long long>(clock64())
        : 0ull;
    if constexpr (kScope == kScheduleScopeSystem) {
        wait_for_schedule_count_system(
            scratch, error_flag, counter, diagnostic, target, kCode);
    } else {
        wait_for_schedule_count(
            scratch, error_flag, counter, diagnostic, target, kCode);
    }
    edges.lap_edge(EDGE, started);
    *mark = clocks.lap(kClockReadinessWait, started);
}

/// Release this unit's writes at device scope, then count one arrival.
static __device__ void publish_schedule_count(
    const Scratch &scratch,
    const int counter
) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(
            reinterpret_cast<unsigned int *>(&scratch.schedule[counter]), 1u);
    }
}

/// Release this unit's writes at system scope, then count one arrival.
///
/// Every producer whose output leaves this rank publishes through here. The
/// collective buffer is read back through the fabric by the peers' tail roles,
/// so a device-scope release would order those writes for this rank's CTAs and
/// for nobody else.
static __device__ void publish_schedule_count_system(
    const Scratch &scratch,
    const int counter
) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(
            reinterpret_cast<unsigned int *>(&scratch.schedule[counter]), 1u);
    }
}

// ---------------------------------------------------------------------------
// Fused projection and quantization.
// ---------------------------------------------------------------------------

/// Quantize the latent groups one projection unit just produced.
///
/// Reads through `__ldcg` and behind a device-scope fence, because on the
/// tensor path the values arrived in global memory through a bulk tensor store
/// that this CTA's L1 never saw. The arithmetic is
/// `expert_mxfp4::quantize_latent_rows`, restricted to one group range.
static __device__ void quantize_latent_group_range(
    const __nv_bfloat16 *__restrict__ latent_x,
    const Scratch &scratch,
    const int active_tokens,
    const int group_begin,
    const int group_count
) {
    using expert_mxfp4::kLatentGroups;
    using expert_mxfp4::kMmaK;

    __threadfence();
    __syncthreads();

    const int thread = static_cast<int>(threadIdx.x);
    const int total = active_tokens * group_count;
    for (int index = thread; index < total; index += kDecodeCtaThreads) {
        const int token = index / group_count;
        const int group = group_begin + index % group_count;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const float value = __bfloat162float(
                __ldcg(&latent_x[static_cast<long long>(token) * kLatentSize
                                 + group * kMmaK + k]));
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale =
            expert_mxfp4::select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        scratch.latent_scale[
            static_cast<long long>(token) * kLatentGroups + group] = scale;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            scratch.latent_mxfp8[
                static_cast<long long>(token) * kLatentSize
                + group * kMmaK + k] =
                    expert_mxfp4::quantize_e4m3(values[k], reciprocal);
        }
    }
}

// ---------------------------------------------------------------------------
// The candidate kernel.
//
// Templated on the layouts type rather than including `persistent_kernel.cuh`,
// which includes this header: the tensor descriptors are that header's, and a
// dependent name is all this one needs of them.
// ---------------------------------------------------------------------------

template<bool TENSOR_PATH, class Layouts>
__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void kimi_k3_decode_dependency_local_kernel(
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ router_weight,
    const float *__restrict__ router_correction_bias,
    const __nv_bfloat16 *__restrict__ routed_expert_down_proj,
    const __nv_bfloat16 *__restrict__ routed_expert_up_proj,
    const __nv_bfloat16 *__restrict__ routed_latent_rmsnorm_weight,
    const __grid_constant__ CUtensorMap expert_w13_packed,
    const std::uint8_t *__restrict__ expert_w13_scale,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const __nv_bfloat16 *__restrict__ shared_gate_proj,
    const __nv_bfloat16 *__restrict__ shared_up_proj,
    const __nv_bfloat16 *__restrict__ shared_down_proj,
    const __grid_constant__ Layouts layouts,
    __nv_bfloat16 *__restrict__ collective_buffer,
    __nv_bfloat16 *__restrict__ collective_multicast,
    __nv_bfloat16 *__restrict__ mailbox_multicast,
    std::uint32_t *__restrict__ barrier_multicast,
    const std::uint32_t *__restrict__ barrier_local,
    unsigned int *__restrict__ barrier_target,
    std::uint8_t *__restrict__ scratch_bytes,
    int *__restrict__ error_flag,
    const int tp_rank,
    const int active_tokens,
    const int profile_phases
) {
    extern __shared__ __align__(16) int shared_raw[];
    std::uint8_t *const shared = reinterpret_cast<std::uint8_t *>(shared_raw);
    __shared__ std::uint32_t latch_slot;
    __shared__ int claim_slot;
    __shared__ int claim_end_slot;

    const Scratch scratch = scratch_view(scratch_bytes);
    const PhaseClocks clocks = phase_clocks(scratch, profile_phases != 0);
    const ScheduleClocks edges =
        schedule_clocks(scratch, profile_phases != 0);
    const int block = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int grid_ctas = static_cast<int>(gridDim.x);

    kittens::tensor_allocator<1, 1> tensor_pool{};

    // Cleared before anything can wait. This matters more here than it does on
    // the barrier schedule: ten readiness edges spin against one clock budget,
    // so a producer that fails to publish makes several consumers give up at
    // once and the claim is what picks the one whose pair gets reported.
    timeout::clear_claim(scratch);

    // Latched before the one barrier, which is the only point at which no CTA
    // of this launch can have advanced the counter yet.
    GridPhase grid = latch_grid_phase(scratch, &latch_slot);

    // -----------------------------------------------------------------------
    // Profiled launches only: zero both accumulator bands, grid-wide.
    //
    // Separated from the first timed region by a barrier for the same reason
    // the production kernel separates its own: a CTA that finished a region
    // before the zeroing landed would have its cycles erased by it.
    // `profile_phases` is launch-wide, so either every CTA takes this barrier
    // or none does, and their barrier targets stay in step either way.
    // -----------------------------------------------------------------------
    if (clocks.enabled()) {
        if (block == 0 && thread < kPhaseClockCount) {
            clocks.counters[thread] = 0ull;
        }
        if (block == 0 && thread < kScheduleClockWords) {
            scratch.schedule[kScheduleClockBegin + thread] = 0;
        }
        grid_barrier(scratch, error_flag, grid, grid_ctas);
    }

    unsigned long long mark = clocks.now();

    // -----------------------------------------------------------------------
    // Stage 0: clear this launch's per-launch state.
    //
    // The only full-grid barrier the schedule keeps. One CTA owns the clearing
    // so no other CTA can be mid-claim while it happens, and the barrier is
    // what makes every zero visible before the first claim. Nothing after this
    // point is a barrier, so this is also the only place a launch can
    // establish a fact about all 148 CTAs at once -- which is exactly what
    // "these counters are zero" is.
    // -----------------------------------------------------------------------
    if (block == 0 && thread < kScheduleClearedCounters) {
        atomicExch(
            reinterpret_cast<unsigned int *>(
                &scratch.schedule[schedule_cleared_counter(thread)]),
            0u);
    }
    // The router's per-token score-shard tickets share the assignment counts,
    // which the assignment queue overwrites before any consumer reads them.
    if (block == 0 && thread < active_tokens) {
        scratch.expert_counts[thread] = 0;
    }
    const int routed_values = active_tokens * kLatentSize;
    for (int index = block * kDecodeCtaThreads + thread;
         index < routed_values;
         index += grid_ctas * kDecodeCtaThreads) {
        scratch.routed_accumulator_fixed[index] = 0;
    }
    mark = clocks.now();
    grid_barrier(scratch, error_flag, grid, grid_ctas);
    mark = clocks.lap(kClockGridBarrier, mark);
    const unsigned long long launched = edges.now();

    // -----------------------------------------------------------------------
    // Q0, the source queue: latent project/quantize, shared gate/up, and the
    // router's score shards. Nothing in it waits on anything.
    //
    // The long independent units come first, exactly as production's
    // projection-first queue orders them: a score shard reads the whole 12.8 MB
    // router weight and there are eight of them per token, so issuing the 28
    // projection units and the 12 shared units behind that tail would strand
    // them.
    //
    // Quantization is fused into the projection unit rather than run as a
    // separate grid-wide pass, which is what removes production's second
    // barrier: a unit's 128 latent columns are exactly four MXFP8 groups, so
    // the CTA that produced them is also the CTA that can quantize them.
    // -----------------------------------------------------------------------
    {
        constexpr int projection_units = kProjectionUnits<TENSOR_PATH>;
        constexpr int shared_units = kSharedGateUpUnits<TENSOR_PATH>;
        constexpr int groups = kLatentGroupsPerProjectionUnit<TENSOR_PATH>;
        const int score_units = active_tokens * router::kScoreShards;
        constexpr int shared_begin = projection_units;
        constexpr int score_begin = projection_units + shared_units;
        const int units = projection_units + shared_units + score_units;
        int unit;
        while ((unit = claim_schedule_unit(
                    scratch, kQueueSource, units, &claim_slot)) >= 0) {
            const int projection_unit = unit < projection_units ? unit : -1;
            const int shared_unit =
                unit >= shared_begin && unit < score_begin
                    ? unit - shared_begin
                    : -1;
            const int score_unit = unit - score_begin;
            if (projection_unit >= 0) {
                if constexpr (TENSOR_PATH) {
                    skinny_gemm::latent_down_tcgen05(
                        shared_raw, tensor_pool, layouts.hidden,
                        layouts.latent_down, layouts.latent, projection_unit);
                } else {
                    skinny_gemm::latent_down_cuda_core<kMaxCoreCapacity>(
                        shared, hidden_states, routed_expert_down_proj,
                        scratch.latent_x, projection_unit, active_tokens);
                }
                __syncthreads();
                mark = clocks.lap(kClockLatentProject, mark);
                quantize_latent_group_range(
                    scratch.latent_x, scratch, active_tokens,
                    projection_unit * groups, groups);
                __syncthreads();
                mark = clocks.lap(kClockLatentQuantize, mark);
                publish_schedule_count(scratch, kScheduleLatentArrivals);
                continue;
            }
            if (shared_unit >= 0) {
                if constexpr (TENSOR_PATH) {
                    const bool gate =
                        shared_unit < shared_experts::kTensorGateCtas;
                    shared_experts::project_tensor(
                        shared_raw, tensor_pool, layouts.hidden,
                        gate ? layouts.shared_gate : layouts.shared_up,
                        gate ? layouts.gate : layouts.up,
                        gate
                            ? shared_unit
                            : shared_unit - shared_experts::kTensorGateCtas,
                        shared_experts::kTensorGateKIterations);
                } else {
                    shared_experts::gate_up_core<kMaxCoreCapacity>(
                        shared, hidden_states, shared_gate_proj,
                        shared_up_proj, scratch, shared_unit, active_tokens);
                }
                __syncthreads();
                // The tensor path's activation consumer reads one column pair,
                // so it is counted per column block as well as in total. The
                // core path computes activation inside its own gate/up unit
                // and has no pair to count.
                if constexpr (TENSOR_PATH) {
                    publish_schedule_count(
                        scratch,
                        kScheduleSharedPairBegin
                            + shared_unit % shared_experts::kTensorGateCtas);
                }
                publish_schedule_count(
                    scratch, kScheduleSharedGateUpArrivals);
                mark = clocks.lap(kClockSharedExperts, mark);
                continue;
            }
            if (score_unit >= 0) {
                const int token = score_unit / router::kScoreShards;
                router::score_shard(
                    shared, hidden_states, router_weight, scratch,
                    token,
                    score_unit % router::kScoreShards);
                // Selection happens inside the token's last shard, before that
                // shard's arrival is counted, so the assignment queue's wait on
                // all eight shards of every token is also a wait on every
                // token's selection.
                router::select_after_score_shard(
                    shared, router_correction_bias, scratch, token);
                __syncthreads();
                publish_schedule_count(scratch, kScheduleScoreArrivals);
                mark = clocks.lap(kClockRouterScore, mark);
                continue;
            }
        }
    }
    edges.mark_queue(kQueueSource, launched);

    // -----------------------------------------------------------------------
    // Q1, shared activation, gated per shared column pair.
    //
    // Column block `u` reads exactly what gate unit `u` and up unit `u` wrote,
    // so it waits on those two arrivals rather than on all twelve.
    // -----------------------------------------------------------------------
    if constexpr (TENSOR_PATH) {
        constexpr int units = kSharedActivationUnits<TENSOR_PATH>;
        int unit;
        while ((unit = claim_schedule_unit(
                    scratch, kQueueSharedActivation, units, &claim_slot))
               >= 0) {
            wait_edge<kEdgeSharedActivationPair>(
                scratch, error_flag, edges, clocks, &mark, unit);
            shared_experts::activate_shared_tile(
                scratch, unit, active_tokens);
            __syncthreads();
            publish_schedule_count(scratch, kScheduleActivationArrivals);
            mark = clocks.lap(kClockSharedExperts, mark);
        }
    }
    edges.mark_queue(kQueueSharedActivation, launched);

    // -----------------------------------------------------------------------
    // Q2, assignment compaction, gated on every token's eight score shards.
    //
    // One unit. The CTA that claims it builds the expert-major table and the
    // compacted unit list; every other CTA refuses the claim immediately and
    // goes on to wait for the publication at Q3's entry, which is where the
    // routed queue length comes from.
    // -----------------------------------------------------------------------
    {
        const int score_units = active_tokens * router::kScoreShards;
        if (claim_schedule_unit(scratch, kQueueAssignment, 1, &claim_slot)
                >= 0) {
            wait_edge<kEdgeAssignmentScoreShards>(
                scratch, error_flag, edges, clocks, &mark, 0, score_units);
            router::build_assignments(shared, scratch, active_tokens);
            __syncthreads();
            router::build_expert_units(shared, scratch);
            __syncthreads();
            publish_schedule_count(
                scratch, kScheduleAssignmentArrivals);
            mark = clocks.lap(kClockAssignment, mark);
        }
    }
    edges.mark_queue(kQueueAssignment, launched);

    // -----------------------------------------------------------------------
    // Q3, the fused-W13 routed gate/up units.
    //
    // Two edges, both backward: the assignment publication, which is what the
    // queue's own length is read from, and the latent, which every expert's
    // batch contracts over the whole of. One expert-pure unit writes six
    // 64-column `situ` ranges and publishes six arrivals, so the routed down
    // queue can gate per expert.
    // -----------------------------------------------------------------------
    wait_edge<kEdgeGateUpAssignment>(
        scratch, error_flag, edges, clocks, &mark);
    wait_edge<kEdgeGateUpLatent>(
        scratch, error_flag, edges, clocks, &mark, 0,
        kProjectionUnits<TENSOR_PATH>);

    // Read past L1 and clamp: the count steers three queue lengths, and a
    // queue longer than the table behind it would index that table out of
    // bounds.
    const std::uint32_t published =
        load_relaxed_gpu(&scratch.phase[kActiveExpertUnits]);
    const int expert_units = static_cast<int>(
        min(published, static_cast<std::uint32_t>(kNumExperts)));
    const int routed_batch = routed_claim_batch(active_tokens);

    {
        const int units = expert_units * kScheduleGateUpUnitsPerExpert;
        // The unit's ring barriers are armed by the first unit this CTA runs
        // and then reused, so the flag is a CTA-lifetime fact. Q3 is the only
        // queue that runs the fused engine, and it runs its units back to
        // back, so the arming survives across them exactly as it does in
        // production's gate/up phase.
        bool first_unit = true;
        while (true) {
            const unsigned long long queue_mark = clocks.now();
            const int batch_begin = claim_schedule_batch(
                scratch, kQueueRoutedGateUp, units, routed_batch, &claim_slot,
                &claim_end_slot);
            mark = clocks.lap(kClockRoutedQueue, queue_mark);
            if (batch_begin < 0) break;
            for (int unit = batch_begin; unit < claim_end_slot; ++unit) {
                const int expert = scratch.unit_expert[unit];
                const int begin = scratch.expert_offsets[expert];
                const int rows =
                    scratch.expert_offsets[expert + 1] - begin;
                expert_mxfp4::fused_w13::routed_gate_up_fused_unit(
                    shared_raw, tensor_pool, &expert_w13_packed,
                    expert_w13_scale, scratch,
                    &scratch.expert_counts[expert], expert, begin, rows,
                    clocks, first_unit);
                first_unit = false;
                mark = clocks.lap(kClockRoutedGateUp, mark);
            }
        }
    }
    edges.mark_queue(kQueueRoutedGateUp, launched);

    // -----------------------------------------------------------------------
    // Q4, shared down, gated on all six activation tiles.
    //
    // A shared-down unit contracts the whole 768-column activation, so it
    // needs every tile no matter how few columns it writes. On the core path
    // there is no activation queue -- the gate/up unit computes activation
    // itself -- so the edge is the gate/up total instead.
    //
    // Its output lands in the symmetric collective buffer, so it releases at
    // system scope: the peers' tail roles read those columns through the
    // fabric.
    // -----------------------------------------------------------------------
    {
        constexpr int units = kSharedDownUnits<TENSOR_PATH>;
        int unit;
        while ((unit = claim_schedule_unit(
                    scratch, kQueueSharedDown, units, &claim_slot)) >= 0) {
            if constexpr (TENSOR_PATH) {
                wait_edge<kEdgeSharedDownActivation>(
                    scratch, error_flag, edges, clocks, &mark, 0,
                    kSharedActivationUnits<TENSOR_PATH>);
                shared_experts::down_tensor(
                    shared_raw, tensor_pool, layouts.activated,
                    layouts.shared_down, collective_buffer, unit,
                    active_tokens, active_tokens);
            } else {
                wait_edge<kEdgeSharedDownGateUp>(
                    scratch, error_flag, edges, clocks, &mark, 0,
                    kSharedGateUpUnits<TENSOR_PATH>);
                shared_experts::down_core<kMaxCoreCapacity>(
                    shared, scratch, shared_down_proj, collective_buffer,
                    unit, active_tokens, active_tokens);
            }
            __syncthreads();
            publish_schedule_count_system(
                scratch, kScheduleSharedDownArrivals);
            mark = clocks.lap(kClockSharedExperts, mark);
        }
    }
    edges.mark_queue(kQueueSharedDown, launched);

    // -----------------------------------------------------------------------
    // Q5, grouped routed down, gated per expert on six W13 arrivals.
    //
    // The one edge production already has. It stays expert-local: a unit waits
    // only for the expert whose `situ` it reads, so an expert whose gate/up
    // finished early is contracted while a slower expert's is still running.
    // -----------------------------------------------------------------------
    {
        constexpr int units_per_expert =
            expert_mxfp4::grouped_pipeline::kGroupedDownUnits;
        const int units = expert_units * units_per_expert;
        while (true) {
            const unsigned long long queue_mark = clocks.now();
            const int batch_begin = claim_schedule_batch(
                scratch, kQueueRoutedDown, units, routed_batch, &claim_slot,
                &claim_end_slot);
            mark = clocks.lap(kClockRoutedQueue, queue_mark);
            if (batch_begin < 0) break;
            for (int unit = batch_begin; unit < claim_end_slot; ++unit) {
                const int expert =
                    scratch.unit_expert[unit / units_per_expert];
                const int begin = scratch.expert_offsets[expert];
                wait_edge<kEdgeRoutedDownGateUp>(
                    scratch, error_flag, edges, clocks, &mark, expert);
                expert_mxfp4::grouped_pipeline::grouped_down_unit(
                    shared_raw, tensor_pool, expert_w2_packed,
                    expert_w2_scale, scratch, expert, begin,
                    scratch.expert_offsets[expert + 1] - begin,
                    unit % units_per_expert, clocks);
                __syncthreads();
                publish_schedule_count(
                    scratch, kScheduleRoutedDownArrivals);
                mark = clocks.lap(kClockRoutedDown, mark);
            }
        }
    }
    edges.mark_queue(kQueueRoutedDown, launched);

    // -----------------------------------------------------------------------
    // Q6, publish: this rank's routed partial next to its shared partial.
    //
    // Every routed down unit accumulates into the same latent rows, so a
    // publish unit waits on the routed total rather than on a token or a
    // group. That is the "preserve the existing final buffer" branch of the
    // design: the grouped down path already reduces into one Q24 accumulator
    // with integer atomics, and asking it for per-token readiness would mean
    // giving up that order-independent reduction. One counter and one
    // acquire is what it costs instead.
    //
    // The published columns leave this rank, so this releases at system scope
    // too.
    // -----------------------------------------------------------------------
    {
        constexpr int units_per_expert =
            expert_mxfp4::grouped_pipeline::kGroupedDownUnits;
        const int routed_units = expert_units * units_per_expert;
        int unit;
        while ((unit = claim_schedule_unit(
                    scratch, kQueuePublish, kSchedulePublishUnits,
                    &claim_slot)) >= 0) {
            wait_edge<kEdgePublishRoutedDown>(
                scratch, error_flag, edges, clocks, &mark, 0, routed_units);
            for (int index = unit * kDecodeCtaThreads + thread;
                 index < routed_values;
                 index += kSchedulePublishUnits * kDecodeCtaThreads) {
                const int row = index / kLatentSize;
                collective_buffer[
                    static_cast<long long>(row)
                        * shared_experts::kCollectiveColumns
                    + index - row * kLatentSize] =
                        __float2bfloat16(
                            __ll2float_rn(
                                scratch.routed_accumulator_fixed[index])
                            * kRoutedAccumulatorScaleInverse);
            }
            publish_schedule_count_system(
                scratch, kSchedulePublishArrivals);
            mark = clocks.lap(kClockPublish, mark);
        }
    }
    edges.mark_queue(kQueuePublish, launched);

    // Every queue is drained. The tail's band starts here, so that what it
    // reports is the tail and not the residue of whichever queue this CTA left
    // last -- the three tail roles leave at three different points, and
    // charging that difference to the tail would make the tail look like it
    // varies by role when only the entry time does.
    mark = clocks.now();

    // -----------------------------------------------------------------------
    // The fused TP8 tail, unchanged, on the CTAs that carry its three roles.
    //
    // The two full-grid barriers production takes before this point are
    // replaced by two waits on one CTA. Only the coordinator needs to know
    // that this rank's collective buffer is complete, because it is the only
    // CTA that tells the peers so; the reduce and shard roles are already
    // gated behind its entry generation. So the whole-rank fact is established
    // by a single waiter rather than by 148 CTAs arriving at two barriers.
    // -----------------------------------------------------------------------
    constexpr int shard_ctas =
        TENSOR_PATH ? tail::kTensorShardCtas : tail::kCoreShardCtas;
    if (block < tail::kReduceBegin) {
        wait_edge<kEdgeTailPublish>(
            scratch, error_flag, edges, clocks, &mark);
        wait_edge<kEdgeTailSharedDown>(
            scratch, error_flag, edges, clocks, &mark, 0,
            kSharedDownUnits<TENSOR_PATH>);

        tail::coordinate_ranks(scratch, error_flag, barrier_multicast,
                               barrier_local, barrier_target);
        clocks.lap(kClockTail, mark);
        return;
    }
    if (block >= tail::kShardBegin + shard_ctas) return;

    if (block < tail::kShardBegin) {
        const std::uint32_t baseline = tail::latch_generation(
            scratch, kTailReduceGeneration, &latch_slot);
        tail::wait_for_generation(scratch, error_flag, kTailEntryGeneration,
                                  baseline, kErrorTailReduceEntry);
        tail::reduce_rows(
            collective_multicast, routed_latent_rmsnorm_weight, scratch,
            block - tail::kReduceBegin, tp_rank, active_tokens);
        tail::publish_generation(
            scratch, kTailReduceArrivals, kTailReduceGeneration,
            tail::kReduceCtas);
    } else {
        const std::uint32_t baseline = tail::latch_generation(
            scratch, kTailShardGeneration, &latch_slot);
        tail::wait_for_generation(scratch, error_flag, kTailReduceGeneration,
                                  baseline, kErrorTailShardReduce);
        if constexpr (TENSOR_PATH) {
            tail::shard_tensor(
                shared_raw, tensor_pool, layouts.normalized, layouts.latent_up,
                scratch, mailbox_multicast, block - tail::kShardBegin, tp_rank,
                active_tokens);
        } else {
            tail::shard_core<kMaxCoreCapacity>(
                shared, scratch, routed_expert_up_proj, mailbox_multicast,
                block - tail::kShardBegin, tp_rank, active_tokens);
        }
        tail::publish_generation(
            scratch, kTailShardArrivals, kTailShardGeneration, shard_ctas);
    }

    tail::drain_ranks(scratch, error_flag, &latch_slot,
                      tail::kReduceCtas + shard_ctas);
    clocks.lap(kClockTail, mark);
}

// ---------------------------------------------------------------------------
// Host residency proof and launch.
// ---------------------------------------------------------------------------

/// Raise the candidate kernel's shared-memory cap and measure its occupancy.
///
/// Kept separate from the production kernel's query because both are
/// properties of a compiled function rather than of a launch: the candidate is
/// a different function and has to be proved resident in its own right.
template<bool TENSOR_PATH, class Layouts>
static __host__ int resident_blocks_per_sm() {
    static std::array<std::atomic<int>, kScheduleMaxCudaDevices> measured{};
    static std::array<std::once_flag, kScheduleMaxCudaDevices> reserved;
    constexpr int shared_bytes = kPersistentSharedBytes;
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(device >= 0 && device < kScheduleMaxCudaDevices,
                "MoK: kimi_k3_decode saw an unexpected device ordinal ",
                device);
    std::call_once(reserved[static_cast<std::size_t>(device)], [device] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kimi_k3_decode_dependency_local_kernel<TENSOR_PATH, Layouts>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        int blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            kimi_k3_decode_dependency_local_kernel<TENSOR_PATH, Layouts>,
            kDecodeCtaThreads, shared_bytes));
        measured[static_cast<std::size_t>(device)].store(
            blocks, std::memory_order_relaxed);
    });
    return measured[static_cast<std::size_t>(device)].load(
        std::memory_order_relaxed);
}

/// Launch one dependency-local decode step.
///
/// Templated on the argument and layout types so this header never has to
/// include the one that includes it.
template<bool TENSOR_PATH, class Arguments, class Layouts>
static __host__ void launch_dependency_local(
    const Arguments &arguments,
    const Layouts &layouts
) {
    const int blocks_per_sm = resident_blocks_per_sm<TENSOR_PATH, Layouts>();
    TORCH_CHECK(blocks_per_sm >= 1,
                "MoK: the dependency-local Kimi K3 schedule requires one CTA "
                "per SM at ", kDecodeCtaThreads, " threads and ",
                kPersistentSharedBytes,
                " dynamic shared bytes, but the device reports ",
                blocks_per_sm);
    TORCH_CHECK(arguments.available_sms >= arguments.grid_ctas,
                "MoK: the dependency-local Kimi K3 schedule requires all ",
                arguments.grid_ctas,
                " CTAs to co-reside one per SM, but the selected device "
                "exposes ", arguments.available_sms, " SMs");

    const auto bf16 = [](const at::Tensor &tensor) {
        return reinterpret_cast<const __nv_bfloat16 *>(tensor.data_ptr());
    };
    const auto bytes = [](const at::Tensor &tensor) {
        return reinterpret_cast<const std::uint8_t *>(tensor.data_ptr());
    };

    kimi_k3_decode_dependency_local_kernel<TENSOR_PATH, Layouts>
        <<<arguments.grid_ctas, kDecodeCtaThreads, kPersistentSharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            bf16(arguments.hidden_states),
            bf16(arguments.router_weight),
            reinterpret_cast<const float *>(
                arguments.router_correction_bias.data_ptr()),
            bf16(arguments.routed_expert_down_proj),
            bf16(arguments.routed_expert_up_proj),
            bf16(arguments.routed_latent_rmsnorm_weight),
            *expert_mxfp4::fused_w13::fused_w13_packed_map(
                arguments.expert_w13_packed.data_ptr()),
            bytes(arguments.expert_w13_scale),
            bytes(arguments.expert_w2_packed),
            bytes(arguments.expert_w2_scale),
            bf16(arguments.shared_gate_proj),
            bf16(arguments.shared_up_proj),
            bf16(arguments.shared_down_proj),
            layouts,
            reinterpret_cast<__nv_bfloat16 *>(
                arguments.collective_buffer.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(
                arguments.collective_buffer_multicast_ptr),
            reinterpret_cast<__nv_bfloat16 *>(
                arguments.output_mailbox_multicast_ptr),
            reinterpret_cast<std::uint32_t *>(
                arguments.barrier_buffer_multicast_ptr),
            reinterpret_cast<const std::uint32_t *>(
                arguments.barrier_buffer.data_ptr()),
            reinterpret_cast<unsigned int *>(
                arguments.barrier_target.data_ptr()),
            reinterpret_cast<std::uint8_t *>(arguments.scratch.data_ptr()),
            reinterpret_cast<int *>(arguments.error_flag.data_ptr()),
            arguments.tp_rank,
            arguments.active_tokens,
            arguments.profile_phases);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
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
// What the tests and the A/B harness read back.
// ---------------------------------------------------------------------------

/// Every readiness edge, as the whole row the runtime waits from.
///
/// The columns are the table's columns, all of them, so a test can check that
/// what the kernel derives its wait from is what the DAG declares rather than
/// checking a projection of it: name, consumer queue, producer queue, counter,
/// timeout code, counter space, acquire scope, target kind, static target, and
/// whether the counter is indexed by the unit.
inline std::vector<std::tuple<std::string, std::int64_t, std::int64_t,
                              std::int64_t, std::int64_t, std::int64_t,
                              std::int64_t, std::int64_t, std::int64_t, bool>>
schedule_edges_for_testing() {
    std::vector<std::tuple<std::string, std::int64_t, std::int64_t,
                           std::int64_t, std::int64_t, std::int64_t,
                           std::int64_t, std::int64_t, std::int64_t, bool>>
        rows;
    for (const ScheduleEdge &edge : kScheduleEdges) {
        rows.emplace_back(
            std::string(edge.name),
            static_cast<std::int64_t>(edge.consumer_queue),
            static_cast<std::int64_t>(edge.producer_queue),
            static_cast<std::int64_t>(edge.counter),
            static_cast<std::int64_t>(edge.error_code),
            static_cast<std::int64_t>(edge.space),
            static_cast<std::int64_t>(edge.scope),
            static_cast<std::int64_t>(edge.target_kind),
            static_cast<std::int64_t>(edge.static_target),
            edge.counter_indexed);
    }
    return rows;
}

/// The diagnostic slot each edge records, for the unit the caller names.
///
/// Exposed so the trap test can predict the recorded slot from the same
/// function the kernel computes it with, rather than from a second copy of the
/// offset arithmetic in Python.
inline std::vector<std::int64_t> schedule_edge_diagnostics_for_testing(
    const std::int64_t unit
) {
    std::vector<std::int64_t> slots;
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        slots.push_back(static_cast<std::int64_t>(
            schedule_edge_diagnostic(edge, static_cast<int>(unit))));
    }
    return slots;
}

inline std::vector<std::string> schedule_queue_names_for_testing() {
    std::vector<std::string> names;
    for (const char *const name : kScheduleQueueNames) {
        names.emplace_back(name);
    }
    return names;
}

/// The appended profile band's first slot and the names it reports.
inline std::tuple<std::int64_t, std::int64_t, std::int64_t,
                  std::vector<std::string>, std::vector<std::string>>
schedule_clock_metadata_for_testing() {
    std::vector<std::string> edge_names;
    for (const char *const name : kScheduleEdgeNames) {
        edge_names.emplace_back(name);
    }
    return {
        static_cast<std::int64_t>(kScheduleBytes / 4 + kScheduleEdgeWaitBegin),
        static_cast<std::int64_t>(
            kScheduleBytes / 4 + kScheduleEdgeMakespanBegin),
        static_cast<std::int64_t>(
            kScheduleBytes / 4 + kScheduleQueueMakespanBegin),
        edge_names,
        schedule_queue_names_for_testing()};
}

/// The bounds the counters are asserted against.
inline std::tuple<std::int64_t, std::int64_t, std::int64_t>
schedule_counter_bounds_for_testing() {
    return {
        static_cast<std::int64_t>(kScheduleLongestTicket),
        static_cast<std::int64_t>(kScheduleLargestArrival),
        static_cast<std::int64_t>(kScheduleCounterBound)};
}

/// The logical unit counts of every queue, for one shape.
inline std::vector<std::int64_t> schedule_queue_units_for_testing(
    const std::int64_t active_tokens
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: the dependency-local schedule needs active_tokens in "
                "[1, ", kMaxTokens, "]");
    const int tokens = static_cast<int>(active_tokens);
    const bool tensor_path = capacity_bucket(tokens) > kMaxCoreCapacity;
    const int experts =
        kTopK * tokens < kNumExperts ? kTopK * tokens : kNumExperts;
    const int projection = tensor_path ? kProjectionUnits<true>
                                       : kProjectionUnits<false>;
    const int shared_gate_up = tensor_path ? kSharedGateUpUnits<true>
                                           : kSharedGateUpUnits<false>;
    const int activation = tensor_path ? kSharedActivationUnits<true>
                                       : kSharedActivationUnits<false>;
    const int shared_down = tensor_path ? kSharedDownUnits<true>
                                        : kSharedDownUnits<false>;
    const auto wide = [](const int value) {
        return static_cast<std::int64_t>(value);
    };
    return {
        wide(projection + shared_gate_up + tokens * router::kScoreShards),
        wide(activation),
        wide(1),
        wide(experts * kScheduleGateUpUnitsPerExpert),
        wide(shared_down),
        wide(experts * expert_mxfp4::grouped_pipeline::kGroupedDownUnits),
        wide(kSchedulePublishUnits),
    };
}

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
