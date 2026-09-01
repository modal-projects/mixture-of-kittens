#pragma once

/// The production kernel: one launch, five generation-tagged grid barriers.
///
/// The phases in the order the preamble lists them, separated by the barriers
/// the dependency-local schedule in `persistent_schedule.cuh` exists to
/// replace. Both kernels call the same stage bodies, so an A/B between them is
/// an A/B on arrival order and nothing else.

#include "descriptors.cuh"

namespace kimi_k3_decode {
namespace persistent {

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
    // The routed gate/up payload travels as a tensor map rather than as a
    // pointer: the engine reads a `(task, slab)` tile with one
    // `cp.async.bulk.tensor.5d`, and a kernel may only name a descriptor it was
    // handed by value in a `__grid_constant__` parameter. Its scales stay an
    // ordinary pointer, because a slab's are already contiguous.
    const __grid_constant__ CUtensorMap expert_w13_packed,
    const std::uint8_t *__restrict__ expert_w13_scale,
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
    const int profile_phases
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

    // Cleared before anything can wait, so a timeout diagnostic this launch
    // publishes is claimed by one of this launch's waiters.
    timeout::clear_claim(scratch);

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
        if (block == 0 && thread < kPhaseClockCount) {
            clocks.counters[thread] = 0ull;
        }
        grid_barrier(scratch, error_flag, grid, grid_ctas);
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
    mark = clocks.lap(kClockGridBarrier, mark);

    // -----------------------------------------------------------------------
    // Phase 1: project the routed latent, run independent shared gate/up work,
    // and contract every token's expert scores.
    //
    // A token's 896 scores are eight units rather than one. Scoring a token
    // reads the whole 12.8 MB router weight, and a single CTA streams that at
    // tens of GB/s: the measured profile of the one-unit-per-token layout put
    // 546 us of a 1.39 ms step inside one CTA while the other 132 waited at
    // the barrier below. Projection and shared gate/up form a fixed prefix so
    // their long independent units are issued before the score-shard tail.
    // -----------------------------------------------------------------------
    {
        constexpr int projection_units =
            TENSOR_PATH ? skinny_gemm::kTensorCtas : skinny_gemm::kCoreCtas;
        constexpr int shared_units = TENSOR_PATH
            ? 2 * shared_experts::kTensorGateCtas
            : shared_experts::kCoreGateCtas;
        const int score_units = active_tokens * router::kScoreShards;
        constexpr int shared_begin = projection_units;
        constexpr int score_begin = projection_units + shared_units;
        const int units = projection_units + shared_units + score_units;
        int unit;
        while ((unit = claim_unit(
                    scratch, kRouteLatentQueue, units, &claim_slot)) >= 0) {
            const int projection_unit =
                unit < projection_units ? unit : -1;
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
    mark = clocks.lap(kClockGridBarrier, mark);

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
        mark = clocks.lap(kClockAssignment, mark);
    }
    expert_mxfp4::quantize_latent_rows(
        scratch.latent_x, scratch, active_tokens, block, grid_ctas);
    __syncthreads();
    mark = clocks.lap(kClockLatentQuantize, mark);
    grid_barrier(scratch, error_flag, grid, grid_ctas);
    mark = clocks.lap(kClockGridBarrier, mark);

    // Read past L1 and clamp: the count steers two queue lengths, and a queue
    // longer than the table behind it would index that table out of bounds.
    const std::uint32_t published =
        load_relaxed_gpu(&scratch.phase[kActiveExpertUnits]);
    const int expert_units = static_cast<int>(
        min(published, static_cast<std::uint32_t>(kNumExperts)));
    const int routed_batch = routed_claim_batch(active_tokens);

    // -----------------------------------------------------------------------
    // Phase 3: routed gate/up units. Shared gate/up already completed in the
    // projection-first phase, before route selection and assignment building.
    // -----------------------------------------------------------------------
    {
        const int units = expert_units * kGateUpUnitsPerExpert;
        // The unit's ring barriers are armed by the first unit this CTA runs
        // and then reused, so the flag has to be a CTA-lifetime fact rather
        // than something each unit rediscovers.
        bool first_unit = true;
        while (true) {
            const unsigned long long queue_mark = clocks.now();
            const int batch_begin = claim_unit_batch(
                scratch, kGateUpQueue, units, routed_batch, &claim_slot,
                &claim_end_slot);
            mark = clocks.lap(kClockRoutedQueue, queue_mark);
            if (batch_begin < 0) break;
            for (int unit = batch_begin; unit < claim_end_slot; ++unit) {
                const int expert = scratch.unit_expert[unit];
                const int begin = scratch.expert_offsets[expert];
                const int rows =
                    scratch.expert_offsets[expert + 1] - begin;
                // The unit publishes its own six arrivals, one per completed
                // 64-column range, because the ranges complete inside it
                // rather than at its boundary.
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
                    kGateUpArrivalsPerExpert,
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
    mark = clocks.lap(kClockGridBarrier, mark);

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
    mark = clocks.lap(kClockPublish, mark);
    // The barrier releases at system scope, so this rank's whole collective
    // buffer is visible to its peers before its coordinator opens the entry
    // rendezvous that tells them it is.
    grid_barrier(scratch, error_flag, grid, grid_ctas);
    mark = clocks.lap(kClockGridBarrier, mark);

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

}  // namespace persistent
}  // namespace kimi_k3_decode
