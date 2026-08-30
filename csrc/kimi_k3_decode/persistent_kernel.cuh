#pragma once

#include "kittens.cuh"

#include "expert_mxfp4.cuh"
#include "persistent_sync.cuh"
#include "expert_mxfp4_grouped.cuh"
#include "router.cuh"
#include "shared.cuh"
#include "skinny_gemm.cuh"
#include "tail_reduce.cuh"
#include "tail_shard.cuh"
#include "tail_sync.cuh"
#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace kimi_k3_decode {
namespace persistent {

// The production Kimi K3 decode path: every stage of one TP8 decode step in a
// single launch of a single kernel.
//
// The public production grid defaults to one CTA per B300 SM. A guarded
// benchmark may launch a smaller candidate, but a launch's CTA count never
// varies with the token count. The logical work -- up to 128 router tasks,
// 28 latent-column tasks, 2 688 routed gate/up tasks, 6 272 routed down
// tasks, the shared expert tasks, and the tail's three roles -- is handed out
// through the device queues in `persistent_sync.cuh` rather than mapped one
// task to one block. Five generation-tagged grid barriers separate the phases:
//
//   0. clear this launch's queue counters and the routed accumulator;
//   1. score and select every token while projecting the routed latent;
//   2. build the expert-major assignment table while quantizing that latent;
//   3. routed gate/up units interleaved with the shared gate/up units, each
//      publishing dependency readiness as it finishes;
//   4. shared activation, routed down units, and shared down units, each
//      waiting only for its own gate/up producer;
//   5. publish this rank's routed partial into the symmetric collective buffer.
//
// The tail then runs the same coordinator, reduce, and shard roles the private
// stage does, on the CTAs that carry those roles; every other CTA retires.

// One `std::once_flag` per possible CUDA ordinal, so the shared-memory
// reservation and the occupancy query happen once per device even when one
// process drives several.
inline constexpr int kMaxCudaDevices = 32;

/// Benchmark grids are rounded above the widest complete tail role set.
inline constexpr int kBenchmarkGridQuantum = 32;
inline constexpr int kLargestTailRoleCtas =
    tail::kCoreRoleCtas > tail::kTensorRoleCtas
        ? tail::kCoreRoleCtas : tail::kTensorRoleCtas;
inline constexpr int kMinimumBenchmarkCtas =
    ((kLargestTailRoleCtas + kBenchmarkGridQuantum - 1)
        / kBenchmarkGridQuantum) * kBenchmarkGridQuantum;
inline constexpr int kMaximumBenchmarkCtas = kPersistentCtas;
inline constexpr int kBenchmarkGridStep = kMinimumBenchmarkCtas / 2;
inline constexpr std::array<int, 4> kBenchmarkGridCtas{
    kMinimumBenchmarkCtas,
    kMinimumBenchmarkCtas + kBenchmarkGridStep,
    kMinimumBenchmarkCtas + 2 * kBenchmarkGridStep,
    kMaximumBenchmarkCtas,
};

static_assert(kMinimumBenchmarkCtas >= tail::kCoordinatorCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kReduceCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kCoreShardCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kTensorShardCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kCoreRoleCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kTensorRoleCtas);
static_assert(kMinimumBenchmarkCtas == 64);
static_assert(kMaximumBenchmarkCtas == kPersistentCtas);
static_assert(kBenchmarkGridCtas[2] < kBenchmarkGridCtas[3]);

/// The widest shared-memory footprint any stage this kernel runs asks for.
inline constexpr int kWidestStageSharedBytes =
    expert_mxfp4::kGateUpUnitSharedBytes
            > expert_mxfp4::kDownUnitSharedBytes
        ? expert_mxfp4::kGateUpUnitSharedBytes
        : expert_mxfp4::kDownUnitSharedBytes;

// Each CTA holds all 512 tensor-memory columns, so the grid is only correct if
// every launched CTA lands alone on an SM. Requesting more than half of an
// SM's shared memory guarantees at most one resident CTA per SM,
// independently of any occupancy heuristic.
static_assert(2 * kPersistentSharedBytes > kittens::MAX_SHARED_MEMORY,
              "the persistent grid must be one CTA per SM");
static_assert(kPersistentSharedBytes <= kittens::MAX_SHARED_MEMORY - 1024,
              "the persistent grid must leave room for static shared memory");
static_assert(kPersistentSharedBytes >= kWidestStageSharedBytes,
              "the persistent grid must fit its widest stage");
static_assert(
    kPersistentSharedBytes
        >= expert_mxfp4::grouped_pipeline::kGroupedDownPersistentSharedBytes,
    "the persistent grid must fit grouped routed down");
static_assert(kPersistentSharedBytes >= router::kSharedBytes,
              "the persistent grid must fit the router's scoring buffer");

// ---------------------------------------------------------------------------
// Logical task counts.
// ---------------------------------------------------------------------------

/// Every logical task of one decode step, by phase.
///
/// The routed counts are reported at their bound, because the exact number of
/// occupied experts is only known on the device: a token's sixteen routes are
/// distinct experts, so at most `min(16 * active_tokens, 896)` experts take an
/// assignment and each of those is exactly one 128-row batch.
struct TaskPlan {
    int route_latent;
    int gate_up;
    int down;
    int tail;
    int grid;
};

inline constexpr TaskPlan task_plan(const int active_tokens) {
    const bool tensor_path = capacity_bucket(active_tokens) > kMaxCoreCapacity;
    const int experts = kTopK * active_tokens < kNumExperts
        ? kTopK * active_tokens : kNumExperts;
    return TaskPlan{
        active_tokens * router::kScoreShards
            + (tensor_path ? skinny_gemm::kTensorCtas : skinny_gemm::kCoreCtas),
        (tensor_path ? 2 * shared_experts::kTensorGateCtas
                     : shared_experts::kCoreGateCtas)
            + experts * expert_mxfp4::kGateUpTiles,
        (tensor_path ? shared_experts::kActivationCtas : 0)
            + experts
                  * expert_mxfp4::grouped_pipeline::kGroupedDownUnits
            + (tensor_path ? shared_experts::kTensorDownCtas
                           : shared_experts::kCoreDownCtas),
        tail::kCoordinatorCtas + tail::kReduceCtas
            + (tensor_path ? tail::kTensorShardCtas : tail::kCoreShardCtas),
        kPersistentCtas,
    };
}

/// The longest queue any phase of any accepted shape hands out.
///
/// Folded over `task_plan` rather than written out, because which phase is
/// widest is not obvious: the core path's shared-down role is 112 units to the
/// tensor path's 62, but the core path only ever runs eight rows, which caps it
/// at 128 occupied experts. The widest is the tensor path's grouped down phase
/// at a full 896 experts.
inline constexpr int longest_queue_units() {
    int longest = 0;
    for (int tokens = 1; tokens <= kMaxTokens; ++tokens) {
        const TaskPlan plan = task_plan(tokens);
        for (const int units :
             {plan.route_latent, plan.gate_up, plan.down, plan.tail}) {
            if (units > longest) longest = units;
        }
    }
    return longest;
}

/// The longest queue, and the highest routed counter value it can reach.
///
/// Both bounds matter. The length is what a unit index is decoded against, and
/// the ticket is what the counter has to hold without wrapping, because nothing
/// resets it inside a launch. The widest shapes use the maximum routed batch,
/// so the bound rounds the logical length up to that width and then lets every
/// CTA add one refused batch.
inline constexpr int kLongestQueueUnits = longest_queue_units();
inline constexpr int kLongestQueueRounded =
    (kLongestQueueUnits + kRoutedClaimBatch - 1)
        / kRoutedClaimBatch * kRoutedClaimBatch;
inline constexpr int kLongestQueueTicket =
    kLongestQueueRounded + kRoutedClaimBatch * kMaximumBenchmarkCtas;

static_assert(kLongestQueueUnits == 6334,
              "the widest phase is 6 activation + 56 shared-down + 896 * 7 "
              "routed down units");
static_assert(kLongestQueueTicket
                  == kLongestQueueRounded
                       + kRoutedClaimBatch * kPersistentCtas,
              "the ticket bound must cover the largest production grid");
static_assert(kLongestQueueTicket == 6928,
              "the rounded widest phase plus one refused batch per CTA");
static_assert(static_cast<unsigned int>(kLongestQueueTicket)
                  < 0xffffffffu / 2u,
              "a queue counter must not approach the unsigned wrap");

inline std::tuple<int, int, int, int, int> task_plan_for_testing(
    const std::int64_t active_tokens
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: kimi_k3_decode task plan requires active_tokens in [1, ",
                kMaxTokens, "]");
    const TaskPlan plan = task_plan(static_cast<int>(active_tokens));
    return {plan.route_latent, plan.gate_up, plan.down, plan.tail, plan.grid};
}

inline std::tuple<int, int, int, int> timeout_metadata_for_testing() {
    return {
        kPersistentTimeoutPhase,
        kGridGeneration,
        kActivationArrivals,
        kActiveExpertUnits};
}

inline std::tuple<int, int> queue_bound_for_testing() {
    return {kLongestQueueUnits, kLongestQueueTicket};
}

/// Select a non-production grid only from an explicitly-enabled benchmark.
///
/// The public decode wrapper has no grid option and unguarded reads always
/// return the validated production constant. The private binding checks an
/// environment guard before changing or exposing stored state, so application
/// code cannot accidentally retain a tuning candidate.
static __host__ std::atomic<int> &benchmark_grid_ctas_storage() {
    static std::atomic<int> grid{kPersistentCtas};
    return grid;
}

inline bool benchmark_grid_tuning_enabled() {
    const char *const enabled =
        std::getenv("MOK_KIMI_K3_ENABLE_GRID_TUNING");
    return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

inline std::int64_t benchmark_grid_ctas_for_testing() {
    if (!benchmark_grid_tuning_enabled()) return kPersistentCtas;
    return benchmark_grid_ctas_storage().load(std::memory_order_relaxed);
}

/// Whether this process collects the kernel's phase clocks.
///
/// Guarded exactly like the grid override, and for the same reason: the
/// accumulators are a benchmark instrument, and a production launch must not
/// be able to turn them on by accident and pay their atomics.
static __host__ std::atomic<int> &benchmark_phase_profile_storage() {
    static std::atomic<int> profile{0};
    return profile;
}

inline bool benchmark_phase_profile_enabled() {
    if (!benchmark_grid_tuning_enabled()) return false;
    return benchmark_phase_profile_storage().load(std::memory_order_relaxed)
        != 0;
}

inline void set_benchmark_phase_profile_for_testing(const bool enabled) {
    TORCH_CHECK(
        benchmark_grid_tuning_enabled(),
        "MoK: Kimi K3 phase profiling is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_GRID_TUNING=1 in a dedicated benchmark process");
    benchmark_phase_profile_storage().store(
        enabled ? 1 : 0, std::memory_order_relaxed);
}

inline bool benchmark_phase_profile_for_testing() {
    return benchmark_phase_profile_enabled();
}

/// Whether a dedicated A/B moves independent long work ahead of router scores.
///
/// Production always returns false. A benchmark process must opt into both the
/// existing guard and this private switch, so graph capture can compare the
/// two orders without exposing a public scheduling option.
static __host__ std::atomic<int> &benchmark_projection_first_storage() {
    static std::atomic<int> projection_first{0};
    return projection_first;
}

inline bool benchmark_projection_first_enabled() {
    if (!benchmark_grid_tuning_enabled()) return false;
    return benchmark_projection_first_storage().load(
               std::memory_order_relaxed)
        != 0;
}

inline void set_benchmark_projection_first_for_testing(const bool enabled) {
    TORCH_CHECK(
        benchmark_grid_tuning_enabled(),
        "MoK: Kimi K3 projection-first scheduling is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_GRID_TUNING=1 in a dedicated benchmark process");
    benchmark_projection_first_storage().store(
        enabled ? 1 : 0, std::memory_order_relaxed);
}

inline bool benchmark_projection_first_for_testing() {
    return benchmark_projection_first_enabled();
}

/// The accumulators' scratch band and their names, for the reader.
inline std::tuple<std::int64_t, std::vector<std::string>>
phase_clock_metadata_for_testing() {
    std::vector<std::string> names;
    for (const char *const name : kPhaseClockNames) names.emplace_back(name);
    return {static_cast<std::int64_t>(kPhaseClockBegin), names};
}

inline void set_benchmark_grid_ctas_for_testing(const std::int64_t grid_ctas) {
    TORCH_CHECK(
        benchmark_grid_tuning_enabled(),
        "MoK: Kimi K3 grid override is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_GRID_TUNING=1 in a dedicated benchmark process");
    bool accepted = false;
    for (const int candidate : kBenchmarkGridCtas) {
        accepted = accepted || grid_ctas == candidate;
    }
    TORCH_CHECK(
        accepted,
        "MoK: Kimi K3 benchmark grid must be one of 64, 96, 128, or 148, got ",
        grid_ctas);
    benchmark_grid_ctas_storage().store(
        static_cast<int>(grid_ctas), std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Tensor-path descriptors.
//
// Every tcgen05 stage this kernel runs reads through one of two tile shapes, so
// the eleven TMA descriptors it needs collapse to two global-layout types. They
// travel in one `__grid_constant__` struct that only the tensor instantiation
// carries: building a descriptor costs a driver call per launch, and the core
// instantiation would never dereference one.
// ---------------------------------------------------------------------------

using tile_layout = skinny_gemm::hidden_layout;
using square_layout = skinny_gemm::latent_layout;

static_assert(std::is_same_v<tile_layout, skinny_gemm::weight_layout>);
static_assert(std::is_same_v<tile_layout,
                             shared_experts::tensor_input_layout>);
static_assert(std::is_same_v<tile_layout,
                             shared_experts::tensor_weight_layout>);
static_assert(std::is_same_v<tile_layout, tail::tensor_input_layout>);
static_assert(std::is_same_v<tile_layout, tail::tensor_weight_layout>);
static_assert(std::is_same_v<square_layout,
                             shared_experts::tensor_output_layout>);

struct TensorLayouts {
    tile_layout hidden;          // [active, 7168]
    tile_layout latent_down;     // [3584, 7168]
    square_layout latent;        // [active, 3584]
    tile_layout shared_gate;     // [768, 7168]
    tile_layout shared_up;       // [768, 7168]
    tile_layout shared_down;     // [7168, 768]
    square_layout gate;          // [active, 768]
    square_layout up;            // [active, 768]
    tile_layout activated;       // [active, 768]
    tile_layout normalized;      // [active, 3584]
    tile_layout latent_up;       // [896, 3584]
};

/// What the core instantiation carries in place of the descriptors.
///
/// It holds one dead byte rather than nothing, because a `__grid_constant__`
/// parameter names a const reference to an object in the kernel's parameter
/// space and an empty type gives that object no bytes to name.
struct NoTensorLayouts {
    char unused;
};

template<bool TENSOR_PATH>
using layouts_t = std::conditional_t<TENSOR_PATH, TensorLayouts,
                                     NoTensorLayouts>;

// ---------------------------------------------------------------------------
// The single production kernel.
// ---------------------------------------------------------------------------

template<bool TENSOR_PATH>
__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void kimi_k3_decode_persistent_kernel(
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ router_weight,
    const float *__restrict__ router_correction_bias,
    const __nv_bfloat16 *__restrict__ routed_expert_down_proj,
    const __nv_bfloat16 *__restrict__ routed_expert_up_proj,
    const __nv_bfloat16 *__restrict__ routed_latent_rmsnorm_weight,
    const std::uint8_t *__restrict__ expert_w1_packed,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_packed,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const __nv_bfloat16 *__restrict__ shared_gate_proj,
    const __nv_bfloat16 *__restrict__ shared_up_proj,
    const __nv_bfloat16 *__restrict__ shared_down_proj,
    const __grid_constant__ layouts_t<TENSOR_PATH> layouts,
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
    const int profile_phases,
    const int projection_first
) {
    extern __shared__ __align__(16) int shared_raw[];
    std::uint8_t *const shared = reinterpret_cast<std::uint8_t *>(shared_raw);
    __shared__ std::uint32_t latch_slot;
    __shared__ int claim_slot;
    __shared__ int claim_end_slot;

    const Scratch scratch = scratch_view(scratch_bytes);
    const PhaseClocks clocks = phase_clocks(scratch, profile_phases != 0);
    const int block = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int grid_ctas = static_cast<int>(gridDim.x);

    // The managed allocator barriers the whole CTA and a CTA may allocate
    // tensor memory only once, so the pool is provisioned here, before any
    // divergence, and handed to every stage this block later runs.
    kittens::tensor_allocator<1, 1> tensor_pool{};

    // Latched before the first barrier, which is the only point at which no
    // CTA of this launch can have advanced the counter yet.
    GridPhase grid = latch_grid_phase(scratch, &latch_slot);

    // -----------------------------------------------------------------------
    // Phase -1, profiled launches only: zero the accumulators, grid-wide.
    //
    // A profiled replay has to report its own cycles rather than every replay
    // the graph ever made, so block 0 clears the band. That clearing has to be
    // separated from the first timed region by a barrier: a CTA that reached
    // the end of phase 0 before the zeroing landed would have its cycles
    // erased by it, and the region would under-report by however many CTAs
    // won that race. `profile_phases` is a launch-wide argument, so either
    // every CTA takes this barrier or none does, and their barrier targets
    // stay in step either way.
    //
    // A measured launch is never profiled: the condition is one predicate on a
    // null pointer the caller already handed in, and nothing inside runs.
    // -----------------------------------------------------------------------
    if (clocks.enabled()) {
        const unsigned long long profile_mark = clocks.now();
        if (block == 0 && thread < kPhaseClockCount) {
            clocks.counters[thread] = 0ull;
        }
        grid_barrier(scratch, error_flag, grid, grid_ctas);
        clocks.lap(kClockProfileClearBarrier, profile_mark);
    }

    unsigned long long mark = clocks.now();

    // -----------------------------------------------------------------------
    // Phase 0: clear this launch's queues and the routed accumulator.
    //
    // One CTA owns the clearing so no other CTA can be mid-claim while it
    // happens; the barrier below is what makes it visible before the first
    // claim. Every routed down unit accumulates into the same latent row, so
    // that accumulator has to start this launch at zero too.
    // -----------------------------------------------------------------------
    if (block == 0 && thread < kPersistentClearedCounters) {
        atomicExch(
            reinterpret_cast<unsigned int *>(
                &scratch.phase[cleared_counter(thread)]),
            0u);
    }
    if (block == 0 && thread == 0) {
        atomicExch(
            reinterpret_cast<unsigned int *>(
                &scratch.phase[kGateUpArrivals]),
            0u);
    }
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
    mark = clocks.lap(kClockClearBarrier, mark);
    const unsigned long long route_phase_started = mark;

    // -----------------------------------------------------------------------
    // Phase 1: project the routed latent, run independent shared gate/up work,
    // and contract every token's expert scores.
    //
    // A token's 896 scores are eight units rather than one. Scoring a token
    // reads the whole 12.8 MB router weight, and a single CTA streams that at
    // tens of GB/s: the measured profile of the one-unit-per-token layout put
    // 546 us of a 1.39 ms step inside one CTA while the other 132 waited at
    // the barrier below. The guarded candidate issues all long independent
    // projection and shared units first; the control retains production's
    // score-first queue and runs shared gate/up in phase 3.
    // -----------------------------------------------------------------------
    {
        constexpr int projection_units =
            TENSOR_PATH ? skinny_gemm::kTensorCtas : skinny_gemm::kCoreCtas;
        constexpr int shared_units = TENSOR_PATH
            ? 2 * shared_experts::kTensorGateCtas
            : shared_experts::kCoreGateCtas;
        const int score_units = active_tokens * router::kScoreShards;
        const int candidate_prefix_units =
            projection_first ? projection_units + shared_units : 0;
        const int units = score_units + projection_units
            + (projection_first ? shared_units : 0);
        int unit;
        while ((unit = claim_unit(
                    scratch, kRouteLatentQueue, units, &claim_slot)) >= 0) {
            const int projection_unit = projection_first
                ? (unit < projection_units ? unit : -1)
                : (unit >= score_units ? unit - score_units : -1);
            const int shared_unit = projection_first
                && unit >= projection_units
                && unit < candidate_prefix_units
                ? unit - projection_units : -1;
            const int score_unit = projection_first
                ? unit - candidate_prefix_units
                : (unit < score_units ? unit : -1);
            if (projection_unit >= 0) {
                if constexpr (TENSOR_PATH) {
                    skinny_gemm::latent_down_tcgen05(
                        shared_raw, tensor_pool, layouts.hidden,
                        layouts.latent_down, layouts.latent, projection_unit);
                } else {
                    // Row guards inside make one capacity cover every core
                    // bucket.
                    skinny_gemm::latent_down_cuda_core<kMaxCoreCapacity>(
                        shared, hidden_states, routed_expert_down_proj,
                        scratch.latent_x, projection_unit, active_tokens);
                }
                __syncthreads();
                mark = clocks.lap(kClockLatentProject, mark);
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
                publish_count(scratch, kGateUpArrivals);
                mark = clocks.lap(kClockSharedExperts, mark);
                continue;
            }
            if (score_unit >= 0) {
                const int token = score_unit / router::kScoreShards;
                router::score_shard(
                    shared, hidden_states, router_weight, scratch,
                    token,
                    score_unit % router::kScoreShards);
                router::select_after_score_shard(
                    shared, router_correction_bias, scratch, token);
                __syncthreads();
                mark = clocks.lap(kClockRouterScore, mark);
                continue;
            }
        }
    }
    grid_barrier(scratch, error_flag, grid, grid_ctas);
    mark = clocks.lap(kClockRouteLatentBarrier, mark);
    clocks.maximum(kClockRouteLatentMakespan, route_phase_started);

    // -----------------------------------------------------------------------
    // Phase 2: build the expert-major assignment table while the grid
    // quantizes the latent.
    //
    // Route selection finished on each token's last score shard before the
    // preceding barrier. Assignment construction and latent quantization are
    // independent consumers, so block 0 builds the table while every other
    // CTA starts quantizing rather than making the whole grid wait through a
    // separate assignment phase.
    // -----------------------------------------------------------------------
    if (block == 0) {
        router::build_assignments(shared, scratch, active_tokens);
        __syncthreads();
        router::build_expert_units(shared, scratch);
        __syncthreads();
        mark = clocks.now();
    }
    expert_mxfp4::quantize_latent_rows(
        scratch.latent_x, scratch, active_tokens, block, grid_ctas);
    __syncthreads();
    mark = clocks.lap(kClockLatentQuantize, mark);
    grid_barrier(scratch, error_flag, grid, grid_ctas);
    mark = clocks.lap(kClockAssignmentQuantizeBarrier, mark);

    // Read past L1 and clamp: the count steers two queue lengths, and a queue
    // longer than the table behind it would index that table out of bounds.
    const std::uint32_t published =
        load_relaxed_gpu(&scratch.phase[kActiveExpertUnits]);
    const int expert_units = static_cast<int>(
        min(published, static_cast<std::uint32_t>(kNumExperts)));
    const int routed_batch = routed_claim_batch(active_tokens);

    // -----------------------------------------------------------------------
    // Phase 3: routed gate/up units interleaved with the shared gate/up units.
    //
    // The shared units come first in the queue so they are claimed while the
    // grid is still empty rather than straggling behind 2 688 routed units.
    // -----------------------------------------------------------------------
    {
        constexpr int available_shared_units = TENSOR_PATH
            ? 2 * shared_experts::kTensorGateCtas
            : shared_experts::kCoreGateCtas;
        const int shared_units = projection_first ? 0
                                                  : available_shared_units;
        constexpr int routed_units_per_expert =
            expert_mxfp4::kGateUpTiles;
        const int units = shared_units + expert_units * routed_units_per_expert;
        while (true) {
            const unsigned long long queue_mark = clocks.now();
            const int batch_begin = claim_unit_batch(
                scratch, kGateUpQueue, units, routed_batch, &claim_slot,
                &claim_end_slot);
            mark = clocks.lap(kClockRoutedQueue, queue_mark);
            if (batch_begin < 0) break;
            for (int unit = batch_begin; unit < claim_end_slot; ++unit) {
                if (unit < shared_units) {
                    if constexpr (TENSOR_PATH) {
                        const bool gate =
                            unit < shared_experts::kTensorGateCtas;
                        shared_experts::project_tensor(
                            shared_raw, tensor_pool, layouts.hidden,
                            gate ? layouts.shared_gate : layouts.shared_up,
                            gate ? layouts.gate : layouts.up,
                            gate
                                ? unit
                                : unit - shared_experts::kTensorGateCtas,
                            shared_experts::kTensorGateKIterations);
                    } else {
                        // The core producer writes the activated intermediate
                        // too, so the core path has no separate activation unit.
                        shared_experts::gate_up_core<kMaxCoreCapacity>(
                            shared, hidden_states, shared_gate_proj,
                            shared_up_proj, scratch, unit, active_tokens);
                    }
                    __syncthreads();
                    publish_count(scratch, kGateUpArrivals);
                    mark = clocks.lap(kClockSharedExperts, mark);
                    continue;
                }
                const int routed = unit - shared_units;
                const int expert = scratch.unit_expert[
                    routed / routed_units_per_expert];
                const int begin = scratch.expert_offsets[expert];
                expert_mxfp4::routed_gate_up_unit(
                    shared_raw, tensor_pool, expert_w1_packed,
                    expert_w1_scale, expert_w3_packed, expert_w3_scale,
                    scratch, expert, begin,
                    scratch.expert_offsets[expert + 1] - begin,
                    routed % routed_units_per_expert, clocks);
                __syncthreads();
                publish_count_at(&scratch.expert_counts[expert]);
                mark = clocks.lap(kClockRoutedGateUp, mark);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: shared activation, the shared down units, and routed down units.
    //
    // The shared units come first again so they overlap the routed work rather
    // than trailing it. Shared down is the one consumer inside a phase: on the
    // tensor path it reads what the activation units write. That wait is
    // bounded because tickets rise, so all six activation units are already
    // held by resident CTAs -- and an activation unit waits on nothing -- before
    // the first shared-down ticket is handed out.
    // -----------------------------------------------------------------------
    {
        constexpr int activation_units =
            TENSOR_PATH ? shared_experts::kActivationCtas : 0;
        constexpr int shared_down_units = TENSOR_PATH
            ? shared_experts::kTensorDownCtas
            : shared_experts::kCoreDownCtas;
        constexpr int shared_units = activation_units + shared_down_units;
        constexpr int shared_gate_up_units = TENSOR_PATH
            ? 2 * shared_experts::kTensorGateCtas
            : shared_experts::kCoreGateCtas;
        constexpr int routed_units_per_expert =
            expert_mxfp4::grouped_pipeline::kGroupedDownUnits;
        const int units = shared_units + expert_units * routed_units_per_expert;
        while (true) {
            const unsigned long long queue_mark = clocks.now();
            const int batch_begin = claim_unit_batch(
                scratch, kDownQueue, units, routed_batch, &claim_slot,
                &claim_end_slot);
            mark = clocks.lap(kClockRoutedQueue, queue_mark);
            if (batch_begin < 0) break;
            for (int unit = batch_begin; unit < claim_end_slot; ++unit) {
                if (unit < shared_units) {
                    const unsigned long long readiness_mark = clocks.now();
                    wait_for_count(
                        scratch, error_flag, kGateUpArrivals,
                        shared_gate_up_units,
                        kErrorPersistentGateUpDownReadiness);
                    mark = clocks.lap(
                        kClockReadinessWait, readiness_mark);
                    if (unit < activation_units) {
                        shared_experts::activate_shared_tile(
                            scratch, unit, active_tokens);
                        publish_count(scratch, kActivationArrivals);
                    } else if constexpr (TENSOR_PATH) {
                        wait_for_count(
                            scratch, error_flag, kActivationArrivals,
                            activation_units, kErrorPersistentActivation);
                        shared_experts::down_tensor(
                            shared_raw, tensor_pool, layouts.activated,
                            layouts.shared_down, collective_buffer,
                            unit - activation_units, active_tokens,
                            active_tokens);
                    } else {
                        shared_experts::down_core<kMaxCoreCapacity>(
                            shared, scratch, shared_down_proj,
                            collective_buffer, unit - activation_units,
                            active_tokens, active_tokens);
                    }
                    __syncthreads();
                    mark = clocks.lap(kClockSharedExperts, mark);
                    continue;
                }
                const int routed = unit - shared_units;
                const int expert =
                    scratch.unit_expert[routed / routed_units_per_expert];
                const int begin = scratch.expert_offsets[expert];
                const unsigned long long readiness_mark = clocks.now();
                wait_for_count_at(
                    scratch, error_flag,
                    &scratch.expert_counts[expert], kGateUpArrivals,
                    expert_mxfp4::kGateUpTiles,
                    kErrorPersistentGateUpDownReadiness);
                mark = clocks.lap(
                    kClockReadinessWait, readiness_mark);
                expert_mxfp4::grouped_pipeline::grouped_down_unit(
                    shared_raw, tensor_pool, expert_w2_packed,
                    expert_w2_scale, scratch, expert, begin,
                    scratch.expert_offsets[expert + 1] - begin,
                    routed % routed_units_per_expert, clocks);
                __syncthreads();
                mark = clocks.lap(kClockRoutedDown, mark);
            }
        }
    }
    grid_barrier(scratch, error_flag, grid, grid_ctas);
    mark = clocks.lap(kClockDownBarrier, mark);

    // -----------------------------------------------------------------------
    // Phase 5: publish this rank's routed partial next to its shared partial.
    //
    // The rows past the active block are never read: every tail role bounds its
    // own loop by the same active token count.
    // -----------------------------------------------------------------------
    for (int index = block * kDecodeCtaThreads + thread;
         index < routed_values;
         index += grid_ctas * kDecodeCtaThreads) {
        const int row = index / kLatentSize;
        collective_buffer[
            static_cast<long long>(row) * shared_experts::kCollectiveColumns
            + index - row * kLatentSize] =
                __float2bfloat16(
                    __ll2float_rn(scratch.routed_accumulator_fixed[index])
                    * kRoutedAccumulatorScaleInverse);
    }
    // The barrier releases at system scope, so this rank's whole collective
    // buffer is visible to its peers before its coordinator opens the entry
    // rendezvous that tells them it is.
    grid_barrier(scratch, error_flag, grid, grid_ctas);
    mark = clocks.lap(kClockPublishBarrier, mark);

    // -----------------------------------------------------------------------
    // Phase 6: the fused TP8 tail, on the CTAs that carry its three roles.
    // -----------------------------------------------------------------------
    constexpr int shard_ctas =
        TENSOR_PATH ? tail::kTensorShardCtas : tail::kCoreShardCtas;
    if (block < tail::kReduceBegin) {
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

/// How many times the reservation below actually ran, per CUDA ordinal.
///
/// A `std::once_flag` cannot be asked whether it has fired, so the count is
/// kept alongside it, for the test that the reservation happens on the device
/// the tensors live on rather than on whichever device happens to be current.
static __host__ std::array<std::atomic<int>, kMaxCudaDevices> &
shared_memory_reservations() {
    static std::array<std::atomic<int>, kMaxCudaDevices> counts{};
    return counts;
}

static __host__ std::int64_t shared_memory_reservations_for_testing(
    const std::int64_t device
) {
    TORCH_CHECK(device >= 0 && device < kMaxCudaDevices,
                "MoK: kimi_k3_decode tracks devices 0 through ",
                kMaxCudaDevices - 1, ", got ", device);
    return shared_memory_reservations()[static_cast<std::size_t>(device)].load(
        std::memory_order_relaxed);
}

/// Raise this kernel's shared-memory cap and measure its occupancy, once.
///
/// Both are properties of the compiled function rather than of a launch, so
/// caching them keeps the launch itself free of any runtime API call a CUDA
/// graph capture would have to record. The measured occupancy is then checked
/// on every call, so a device that cannot host the grid is rejected every time
/// rather than only on the first launch of a process.
template<bool TENSOR_PATH>
static __host__ int resident_blocks_per_sm() {
    static std::array<std::atomic<int>, kMaxCudaDevices> measured{};
    static std::array<std::once_flag, kMaxCudaDevices> reserved;
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(device >= 0 && device < kMaxCudaDevices,
                "MoK: kimi_k3_decode saw an unexpected device ordinal ",
                device);
    std::call_once(reserved[static_cast<std::size_t>(device)], [device] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kimi_k3_decode_persistent_kernel<TENSOR_PATH>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kPersistentSharedBytes));
        int blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            kimi_k3_decode_persistent_kernel<TENSOR_PATH>,
            kDecodeCtaThreads, kPersistentSharedBytes));
        measured[static_cast<std::size_t>(device)].store(
            blocks, std::memory_order_relaxed);
        shared_memory_reservations()[static_cast<std::size_t>(device)]
            .fetch_add(1, std::memory_order_relaxed);
    });
    return measured[static_cast<std::size_t>(device)].load(
        std::memory_order_relaxed);
}

/// Reject a device that cannot hold the whole grid at once.
///
/// Every phase barrier counts all CTAs in the runtime launch grid, and a CTA
/// that is not resident cannot arrive, so a grid that only partly fits does
/// not run slowly -- it deadlocks. The occupancy query is the measurement that
/// matters; the SM count turns it into a whole-grid answer.
inline void validate_grid_residency(
    const std::int64_t available_sms,
    const std::int64_t blocks_per_sm,
    const std::int64_t grid_ctas
) {
    TORCH_CHECK(blocks_per_sm >= 1,
                "MoK: kimi_k3_decode requires the persistent kernel to place "
                "at least one CTA per SM at ", kDecodeCtaThreads,
                " threads and ", kPersistentSharedBytes,
                " dynamic shared bytes, but the device reports ",
                blocks_per_sm);
    TORCH_CHECK(available_sms >= grid_ctas,
                "MoK: kimi_k3_decode requires all ", grid_ctas,
                " CTAs of the persistent grid to co-reside one per SM, but the "
                "selected device exposes ", available_sms, " SMs");
}

inline void validate_residency(
    const std::int64_t available_sms,
    const std::int64_t blocks_per_sm
) {
    validate_grid_residency(
        available_sms, blocks_per_sm, kPersistentCtas);
}

inline std::int64_t resident_blocks_per_sm_for_testing(
    const bool tensor_path
) {
    return tensor_path ? resident_blocks_per_sm<true>()
                       : resident_blocks_per_sm<false>();
}

/// Every pointer, alias, and count one persistent launch needs.
///
/// The kernel takes twenty-odd arguments and the two capacity paths pass the
/// same ones, so they travel together rather than being spelled out three
/// times between the entrypoint and the two launch helpers.
struct LaunchArguments {
    const at::Tensor &hidden_states;
    const at::Tensor &router_weight;
    const at::Tensor &router_correction_bias;
    const at::Tensor &routed_expert_down_proj;
    const at::Tensor &routed_expert_up_proj;
    const at::Tensor &routed_latent_rmsnorm_weight;
    const at::Tensor &expert_w1_packed;
    const at::Tensor &expert_w1_scale;
    const at::Tensor &expert_w3_packed;
    const at::Tensor &expert_w3_scale;
    const at::Tensor &expert_w2_packed;
    const at::Tensor &expert_w2_scale;
    const at::Tensor &shared_gate_proj;
    const at::Tensor &shared_up_proj;
    const at::Tensor &shared_down_proj;
    const at::Tensor &collective_buffer;
    std::int64_t collective_buffer_multicast_ptr;
    std::int64_t output_mailbox_multicast_ptr;
    const at::Tensor &barrier_buffer;
    std::int64_t barrier_buffer_multicast_ptr;
    const at::Tensor &barrier_target;
    const at::Tensor &scratch;
    const at::Tensor &error_flag;
    int tp_rank;
    int active_tokens;
    int available_sms;
    int grid_ctas;
    int profile_phases;
    int projection_first;
};

template<bool TENSOR_PATH>
static __host__ void launch_persistent(
    const LaunchArguments &arguments,
    const layouts_t<TENSOR_PATH> &layouts
) {
    validate_grid_residency(
        arguments.available_sms,
        resident_blocks_per_sm<TENSOR_PATH>(),
        arguments.grid_ctas);

    const auto bf16 = [](const at::Tensor &tensor) {
        return reinterpret_cast<const __nv_bfloat16 *>(tensor.data_ptr());
    };
    const auto bytes = [](const at::Tensor &tensor) {
        return reinterpret_cast<const std::uint8_t *>(tensor.data_ptr());
    };

    kimi_k3_decode_persistent_kernel<TENSOR_PATH>
        <<<arguments.grid_ctas, kDecodeCtaThreads, kPersistentSharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            bf16(arguments.hidden_states),
            bf16(arguments.router_weight),
            reinterpret_cast<const float *>(
                arguments.router_correction_bias.data_ptr()),
            bf16(arguments.routed_expert_down_proj),
            bf16(arguments.routed_expert_up_proj),
            bf16(arguments.routed_latent_rmsnorm_weight),
            bytes(arguments.expert_w1_packed),
            bytes(arguments.expert_w1_scale),
            bytes(arguments.expert_w3_packed),
            bytes(arguments.expert_w3_scale),
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
            arguments.profile_phases,
            arguments.projection_first);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// Build the eleven TMA descriptors the tcgen05 stages read through.
static __host__ TensorLayouts tensor_layouts(
    const LaunchArguments &arguments
) {
    const auto tile = [](const void *pointer,
                         const std::int64_t rows,
                         const std::int64_t columns) {
        return tile_layout{
            const_cast<kittens::bf16 *>(
                reinterpret_cast<const kittens::bf16 *>(pointer)),
            nullptr, nullptr, static_cast<size_t>(rows),
            static_cast<size_t>(columns)};
    };
    const auto square = [](const void *pointer,
                           const std::int64_t rows,
                           const std::int64_t columns) {
        return square_layout{
            const_cast<kittens::bf16 *>(
                reinterpret_cast<const kittens::bf16 *>(pointer)),
            nullptr, nullptr, static_cast<size_t>(rows),
            static_cast<size_t>(columns)};
    };

    const Scratch pointers = scratch_view(
        reinterpret_cast<std::uint8_t *>(arguments.scratch.data_ptr()));
    const int active = arguments.active_tokens;
    constexpr int shared_intermediate = shared_experts::kIntermediate;

    return TensorLayouts{
        tile(arguments.hidden_states.data_ptr(), active, kHiddenSize),
        tile(arguments.routed_expert_down_proj.data_ptr(), kLatentSize,
             kHiddenSize),
        square(pointers.latent_x, active, kLatentSize),
        tile(arguments.shared_gate_proj.data_ptr(), shared_intermediate,
             kHiddenSize),
        tile(arguments.shared_up_proj.data_ptr(), shared_intermediate,
             kHiddenSize),
        tile(arguments.shared_down_proj.data_ptr(), kHiddenSize,
             shared_intermediate),
        square(pointers.shared_gate, active, shared_intermediate),
        square(pointers.shared_up, active, shared_intermediate),
        tile(pointers.shared_activated, active, shared_intermediate),
        tile(pointers.tail_normalized, active, kLatentSize),
        // Only this rank's contiguous 896-row slice of the replicated
        // latent-up weight is ever contracted.
        tile(reinterpret_cast<const __nv_bfloat16 *>(
                 arguments.routed_expert_up_proj.data_ptr())
                 + static_cast<long long>(arguments.tp_rank)
                       * tail::kShardColumns * kLatentSize,
             tail::kShardColumns, kLatentSize),
    };
}

/// Run one whole TP8 Kimi K3 decode step in one persistent launch.
static __host__ void launch_decode(const LaunchArguments &arguments) {
    if (capacity_bucket(arguments.active_tokens) <= kMaxCoreCapacity) {
        launch_persistent<false>(arguments, NoTensorLayouts{});
        return;
    }
    launch_persistent<true>(arguments, tensor_layouts(arguments));
}

}  // namespace persistent
}  // namespace kimi_k3_decode
