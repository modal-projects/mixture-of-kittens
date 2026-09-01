#pragma once

/// The dependency-local kernel itself, and the shared budget it launches with.
///
/// One `__global__` over seven topologically ordered queues, templated on the
/// capacity path and on the gate/up engine. `schedule_shared_bytes` is the
/// dynamic allocation each engine needs; the adaptive production engine takes
/// the larger of the two rings it selects between, because the selection is a
/// runtime branch inside one launch.

#include "projection.cuh"

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// The candidate kernel.
//
// Templated on the layouts type rather than including `persistent_kernel.cuh`,
// which includes this header: the tensor descriptors are that header's, and a
// dependent name is all this one needs of them.
// ---------------------------------------------------------------------------

/// Dynamic shared bytes one instantiation of the schedule launches with.
///
/// The routed gate/up ring is the widest thing any stage of this kernel holds,
/// so the grid's request is whichever ring the engine it compiled can reach.
/// The resident ring is two K = 512 weight stages and the whole expert's
/// activation; the slab-buffered ring is three stages and two eight-row
/// activation slots; the compact ring is three stages and a packed activation.
///
/// Production reaches the compact ring and the slab-buffered one, so it asks
/// for the wider of the two at every launch, including the launches where every
/// expert takes the narrower. Asking per launch is not an option: which ring an
/// expert takes is not known until the router has run, and the two rings are
/// both live inside one launch. Both figures are above half of an SM's shared
/// memory, which is what keeps every grid one CTA per SM.
template<int ENGINE>
inline constexpr int schedule_shared_bytes =
    expert_mxfp4::fused_w13::engine_is_adaptive(ENGINE)
        ? expert_mxfp4::fused_w13::kFusedCompactSharedBytes
        : kPersistentSharedBytes;

static_assert(schedule_shared_bytes<
                  expert_mxfp4::fused_w13::kEngineFusedAdaptive> == 228352);
static_assert(schedule_shared_bytes<
                  expert_mxfp4::fused_w13::kEngineFusedResident> == 216064);
// Production's launch has to grant every ring it can take, which is the point
// of stating this as an inequality on the two arms rather than as one number.
static_assert(schedule_shared_bytes<
                  expert_mxfp4::fused_w13::kEngineFusedAdaptive>
                  >= expert_mxfp4::fused_w13::kFusedV4SharedBytes
              && schedule_shared_bytes<
                     expert_mxfp4::fused_w13::kEngineFusedAdaptive>
                     >= expert_mxfp4::fused_w13::kFusedCompactSharedBytes,
              "the production selector's launch must grant both of the rings "
              "it can take at any unit");

/// Every engine's instantiation must be one CTA per SM and must still be able
/// to run every other stage of the step.
template<int ENGINE>
inline constexpr bool schedule_instantiation_is_admissible =
    2 * schedule_shared_bytes<ENGINE> > kittens::MAX_SHARED_MEMORY
    && schedule_shared_bytes<ENGINE>
           <= kittens::MAX_SHARED_MEMORY
                  - expert_mxfp4::fused_w13::kFusedStaticSharedReserve
    && schedule_shared_bytes<ENGINE>
           >= expert_mxfp4::grouped_pipeline::kGroupedDownPersistentSharedBytes
    && schedule_shared_bytes<ENGINE> >= router::kSharedBytes
    && schedule_shared_bytes<ENGINE> >= expert_mxfp4::kGateUpUnitSharedBytes
    && schedule_shared_bytes<ENGINE> >= expert_mxfp4::kDownUnitSharedBytes;

static_assert(
    schedule_instantiation_is_admissible<
        expert_mxfp4::fused_w13::kEngineFusedAdaptive>
    && schedule_instantiation_is_admissible<
        expert_mxfp4::fused_w13::kEngineFusedResident>,
    "every gate/up engine's instantiation must stay one CTA per SM and still "
    "run every other stage of the step");

template<bool TENSOR_PATH,
         int ENGINE = expert_mxfp4::fused_w13::kEngineFusedAdaptive,
         class Layouts>
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
                if constexpr (expert_mxfp4::fused_w13::
                                  engine_is_adaptive(ENGINE)) {
                    expert_mxfp4::fused_w13::
                        routed_gate_up_fused_adaptive_unit(
                            shared_raw, tensor_pool, &expert_w13_packed,
                            expert_w13_scale, scratch,
                            &scratch.expert_counts[expert], expert, begin,
                            rows, clocks, first_unit);
                } else {
                    expert_mxfp4::fused_w13::routed_gate_up_fused_unit(
                        shared_raw, tensor_pool, &expert_w13_packed,
                        expert_w13_scale, scratch,
                        &scratch.expert_counts[expert], expert, begin, rows,
                        clocks, first_unit);
                }
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

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
