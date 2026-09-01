#pragma once

/// The resident two-stage ring: what production ran before the adaptive path.
///
/// One expert-pure CTA claim, that expert's whole activation gathered once, six
/// sequential single-accumulator tasks, and 42 `(task, slab)` weight transfers
/// through a two-stage K = 512 ring. It is kept compiled as engine
/// `kEngineFusedResident` for two reasons: it is the numerical baseline the
/// adaptive path is held to byte for byte, and it is the arm the integration's
/// A/B measured against.
///
/// It is not reachable from production's dispatch.

#include "activation.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

// ---------------------------------------------------------------------------
// One expert unit.
// ---------------------------------------------------------------------------

/// Contract all six fused output tasks of one expert batch and stage its SiTU.
///
/// One queue claim, one accumulator, one activation staging, and one 42-long
/// weight stream. `arrival_counter` is the expert's gate/up readiness counter:
/// this unit publishes six arrivals into it, one per completed 64-column range,
/// so the grouped down phase's threshold is the six it has always been and it
/// cannot start before all 384 columns exist.
static __device__ void routed_gate_up_fused_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const CUtensorMap *__restrict__ packed_map,
    const std::uint8_t *__restrict__ fused_scale,
    const Scratch &scratch,
    int *__restrict__ arrival_counter,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const PhaseClocks clocks,
    const bool first_unit
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    fused_weight_tile (&weight)[kFusedStages] =
        staging.allocate<fused_weight_tile, kFusedStages>();
    mixed_scale_tile (&weight_scale)[kFusedStages][kFusedSlabScaleTiles] =
        staging.allocate<
            mixed_scale_tile, kFusedStages, kFusedSlabScaleTiles>();
    fused_activation_tile (&activation)[kFusedActivationSlabs] =
        staging.allocate<fused_activation_tile, kFusedActivationSlabs>();
    mixed_scale_tile
        (&activation_scale)[kFusedActivationSlabs][kFusedSlabScaleTiles] =
            staging.allocate<
                mixed_scale_tile, kFusedActivationSlabs,
                kFusedSlabScaleTiles>();

    // Its own array rather than an overlay: the epilogue runs with the next
    // task's two weight transfers in flight, so there is no dead region of the
    // ring to borrow.
    fused_result_tile (&result) = staging.allocate<fused_result_tile>();

    // Armed once per CTA and carried by parity. Forty-two stream indices over
    // two stages is not a whole number of laps, so a unit hands the next one a
    // barrier that is mid-phase, and re-arming per unit would mean depending on
    // `mbarrier.init` to reset a parity mid-launch -- which PTX defines only
    // for a barrier that has been invalidated first.
    __shared__ semaphore slab_arrived[kFusedStages];
    __shared__ semaphore slab_retired[kFusedStages];
    __shared__ unsigned int stream_parity[2];

    const int thread = static_cast<int>(threadIdx.x);
    if (first_unit) {
        if (thread < kFusedStages) {
            init_semaphore(slab_arrived[thread], 0, 1);
            init_semaphore(slab_retired[thread], 0, 1);
        }
        if (thread == 0) {
            stream_parity[0] = 0u;
            stream_parity[1] = 0u;
        }
    }
    __syncthreads();

    const fused_accumulator_tile accumulator =
        tensor_pool.allocate<fused_accumulator_tile>(0);
    const auto scale_slot = [&](const int set, const int slot) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kFusedScaleColumnBase
            + (set * kFusedScaleSlots + slot) * kRoutedScaleColumns);
    };

    // Warp 0's alone: it is the only warp that consumes these and the only
    // warp that writes them back. The whole-CTA gather below happens to put a
    // `__syncthreads` between this read and that write, so a read by the other
    // seven warps would be ordered rather than racing -- but the parity is warp
    // 0's state, so warp 0 is the only warp that reads it.
    unsigned int arrived_phase = 0u;
    unsigned int retired_phase = 0u;
    if (warpid() == 0) {
        arrived_phase = stream_parity[0];
        retired_phase = stream_parity[1];
    }
    const auto take_phase = [](unsigned int &bits, const int stage) {
        const int phase = static_cast<int>((bits >> stage) & 1u);
        bits ^= 1u << stage;
        return phase;
    };
    const auto issue = [&](const int index) {
        load_fused_slab(
            weight[index % kFusedStages],
            weight_scale[index % kFusedStages], packed_map, fused_scale,
            expert, index, slab_arrived[index % kFusedStages]);
    };

    for (int assignment_offset = 0; assignment_offset < batch_rows;
         assignment_offset += kFusedN) {
        const int batch_begin = assignment_begin + assignment_offset;
        const int rows = min(kFusedN, batch_rows - assignment_offset);
        const bool last_pass = assignment_offset + kFusedN >= batch_rows;

        // Issue the stream's first two transfers before the gather so they fly
        // underneath it. Nothing is in flight here: the previous pass consumed
        // all 42 of its indices.
        unsigned long long fine = clocks.now();
        if (thread == 0) {
            issue(0);
            issue(1);
        }
        fine = clocks.lap(kClockRoutedGateUpTmaIssue, fine);

        stage_fused_unit_activation(
            activation, activation_scale, scratch, batch_begin, rows);
        // Every thread wrote some of the tile above, so every thread owes the
        // asynchronous proxy a fence before the barrier that orders those
        // writes ahead of warp 0's `tcgen05` reads. One fence for the whole
        // unit, where a per-`(task, slab)` gather needed 42 of them.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        clocks.lap(kClockRoutedGateUpActivation, fine);

        // How far the stream has been retired. The stream retires one index
        // behind its issues, except at a task's last slab where the epilogue
        // needs that slab's own completion -- so the two places that retire
        // share one high-water mark rather than each assuming the other did
        // not run.
        int retired_upto = -1;

        for (int task = 0; task < kFusedTasks; ++task) {
            if (warpid() == 0) {
                const int lane = static_cast<int>(laneid());
                unsigned long long mark = clocks.now();
                unsigned long long inner = clocks.now();

                for (int slab = 0; slab < kFusedSlabs; ++slab) {
                    const int index = task * kFusedSlabs + slab;
                    const int stage = index % kFusedStages;
                    const int set = index % kFusedScaleSets;

                    wait(slab_arrived[stage], take_phase(arrived_phase, stage));
                    mark = clocks.lap(kClockRoutedGateUpStage, mark);
                    inner = clocks.lap(kClockRoutedGateUpTmaWait, inner);

                    if (lane == 0) {
                        #pragma unroll
                        for (int quad = 0; quad < kFusedSlabScaleTiles;
                             ++quad) {
                            auto staged_weight_scale = scale_slot(set, quad);
                            auto staged_activation_scale = scale_slot(
                                set, kFusedSlabScaleTiles + quad);
                            load_mxnv_scale_async(
                                staged_weight_scale, weight_scale[stage][quad]);
                            load_mxnv_scale_async(
                                staged_activation_scale,
                                activation_scale[slab][quad]);
                        }
                    }
                    tensor_store_wait();

                    if (lane == 0) {
                        st_descriptor<fused_weight_tile, transpose::N>
                            weight_desc(weight[stage]);
                        st_descriptor<fused_activation_tile, transpose::N>
                            activation_desc(activation[slab]);
                        #pragma unroll
                        for (int group = 0; group < kFusedSlabGroups; ++group) {
                            const int quad = group / kScaleGroupsPerTile;
                            const int factor = group % kScaleGroupsPerTile;
                            fused_mixed_mma(
                                accumulator,
                                weight_desc.chunk_descriptor(group),
                                activation_desc.chunk_descriptor(group),
                                scale_slot(set, quad),
                                scale_slot(
                                    set, kFusedSlabScaleTiles + quad),
                                factor,
                                slab != 0 || group != 0);
                        }
                        detail::tcgen05::commit<1>(slab_retired[stage]);
                    }
                    inner = clocks.lap(kClockRoutedGateUpMmaIssue, inner);

                    // Deferred retire, one index behind, refilling the stage it
                    // frees with the index after the one in flight. This runs
                    // straight through task boundaries: the stream does not
                    // restart per task, so the tensor core is fed across them.
                    if (index - 1 > retired_upto) {
                        const int retiring = (index - 1) % kFusedStages;
                        wait(slab_retired[retiring],
                             take_phase(retired_phase, retiring));
                        retired_upto = index - 1;
                        mark = clocks.lap(kClockRoutedGateUpMma, mark);
                        inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                        if (index + 1 < kFusedStreamLength && lane == 0) {
                            issue(index + 1);
                        }
                        inner = clocks.lap(kClockRoutedGateUpTmaIssue, inner);
                    }

                    // The epilogue reads tensor memory, so this task's last
                    // slab is the one index the deferred retire cannot cover.
                    // Refill past it first, so the next task's two transfers
                    // are in flight underneath the epilogue below.
                    if (slab == kFusedSlabs - 1 && index > retired_upto) {
                        wait(slab_retired[stage],
                             take_phase(retired_phase, stage));
                        retired_upto = index;
                        mark = clocks.lap(kClockRoutedGateUpMma, mark);
                        inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                        if (index + 2 < kFusedStreamLength && lane == 0) {
                            issue(index + 2);
                        }
                        inner = clocks.lap(kClockRoutedGateUpTmaIssue, inner);
                    }
                    mark = clocks.lap(kClockRoutedGateUpStage, mark);
                }
                if (lane == 0) {
                    stream_parity[0] = arrived_phase;
                    stream_parity[1] = retired_phase;
                }
            }
            __syncthreads();

            const unsigned long long epilogue = clocks.now();
            store_fused_accumulator(accumulator, result);
            __syncthreads();
            quantize_fused_situ(result, scratch, batch_begin, rows, task);
            __syncthreads();
            clocks.lap(kClockRoutedGateUpEpilogue, epilogue);

            // One arrival per completed 64-column range, and only on the pass
            // that finished it. A wide batch is several passes over the same
            // columns, so publishing per pass would let the count reach the
            // down phase's threshold while later rows of those columns were
            // still being written.
            if (last_pass) {
                persistent::publish_count_at(arrival_counter);
            }
        }
    }
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
