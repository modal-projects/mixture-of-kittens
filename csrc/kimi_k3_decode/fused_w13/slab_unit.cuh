#pragma once

/// The slab-buffered three-stage ring: production's wide arm.
///
/// The activation stops being resident. Two eight-row slots replace seven
/// sixteen-row tiles, warps 1 to 7 gather the next slab while warp 0 contracts
/// the current one, and the slab loop moves outside the task loop so each slab
/// is gathered once and all six accumulators stay open.
///
/// This is the ring an expert takes when its batch is wider than the compact
/// packing admits, which is what keeps a concentrated route's measured gain.

#include "engines.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

/// Contract all six fused output tasks of one expert batch, a slab at a time.
///
/// Same signature and same contract as the resident unit: one queue claim, six
/// published column ranges, and `situ` bytes that must be equal to it bit for
/// bit. What differs is inside: the slab is the outer loop, so the six tasks
/// that read one slab are adjacent, the slab is gathered seven times per expert
/// instead of 42, six accumulators are open at once, and the epilogues sit
/// after the stream rather than inside it.
///
/// The opposite order -- task outer, slab inner -- was compiled and measured
/// beside this one and lost at every shape. Section 52 of the Task 11b report
/// records it; nothing here is templated on the choice any more.
static __device__ void routed_gate_up_fused_v4_unit(
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
    constexpr int kEpochLength = kFusedV4EpochLength;
    constexpr int kEpochs = kFusedV4Epochs;

    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    fused_weight_tile (&weight)[kFusedV4Stages] =
        staging.allocate<fused_weight_tile, kFusedV4Stages>();
    mixed_scale_tile (&weight_scale)[kFusedV4Stages][kFusedSlabScaleTiles] =
        staging.allocate<
            mixed_scale_tile, kFusedV4Stages, kFusedSlabScaleTiles>();
    // One tile, two operands. `activation_slot` below is what tells the tensor
    // core which half it is reading.
    fused_activation_tile (&activation) =
        staging.allocate<fused_activation_tile>();
    mixed_scale_tile (&activation_scale)[kFusedV4Slots][kFusedSlabScaleTiles] =
        staging.allocate<
            mixed_scale_tile, kFusedV4Slots, kFusedSlabScaleTiles>();
    fused_result_tile (&result) = staging.allocate<fused_result_tile>();

    __shared__ semaphore slab_arrived[kFusedV4Stages];
    __shared__ semaphore slab_retired[kFusedV4Stages];
    __shared__ unsigned int stream_parity[2];

    const int thread = static_cast<int>(threadIdx.x);
    if (first_unit) {
        if (thread < kFusedV4Stages) {
            init_semaphore(slab_arrived[thread], 0, 1);
            init_semaphore(slab_retired[thread], 0, 1);
        }
        if (thread == 0) {
            stream_parity[0] = 0u;
            stream_parity[1] = 0u;
        }
    }
    __syncthreads();

    // One accumulator per open task, below the column the scale buffers start
    // at. Slab-major interleaves the six tasks inside a slab, so all six are
    // open. Named rather than held:
    // `allocate` at an explicit column is an address computation with no
    // allocator state behind it.
    const auto accumulator = [&](const int task) {
        return tensor_pool.allocate<fused_accumulator_tile>(
            task * kFusedPhysicalN);
    };
    const auto scale_slot = [&](const int set, const int slot) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kFusedScaleColumnBase
            + (set * kFusedScaleSlots + slot) * kRoutedScaleColumns);
    };

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
            weight[index % kFusedV4Stages],
            weight_scale[index % kFusedV4Stages], packed_map, fused_scale,
            expert, fused_v4_task_slab_of(index),
            slab_arrived[index % kFusedV4Stages]);
    };

    for (int assignment_offset = 0; assignment_offset < batch_rows;
         assignment_offset += kFusedN) {
        const int batch_begin = assignment_begin + assignment_offset;
        const int rows = min(kFusedN, batch_rows - assignment_offset);
        const bool last_pass = assignment_offset + kFusedN >= batch_rows;

        // Three transfers before the first gather, so all three fly underneath
        // it. Nothing is in flight here: the previous pass consumed all 42.
        unsigned long long fine = clocks.now();
        if (thread == 0) {
            issue(0);
            issue(1);
            issue(2);
        }
        fine = clocks.lap(kClockRoutedGateUpTmaIssue, fine);

        // Slab 0 by the whole CTA, because warp 0 has nothing to contract yet.
        // Every slab after it is gathered by warps 1 to 7 while warp 0 is
        // inside the ring.
        stage_fused_slab_activation(
            activation, activation_scale[0], scratch, batch_begin, rows, 0, 0,
            0);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        clocks.lap(kClockRoutedGateUpActivation, fine);

        // How far the stream has been issued and retired. Both are high-water
        // marks rather than derived from the index, because three places refill
        // and two retire, and each has to see what the others already did.
        int issued_upto = kFusedV4Stages - 1;
        int retired_upto = -1;

        for (int epoch = 0; epoch < kEpochs; ++epoch) {
            const int slab = fused_v4_slab_of(epoch * kEpochLength);
            const int slot = epoch % kFusedV4Slots;
            if (warpid() == 0) {
                const int lane = static_cast<int>(laneid());
                unsigned long long mark = clocks.now();
                unsigned long long inner = clocks.now();
                const std::uint64_t slot_offset = fused_v4_slot_offset(slot);

                // Refill every stage the retires below freed. One index per
                // call in steady state; the bound is what keeps a stage that
                // is still being read from being overwritten.
                const auto refill = [&]() {
                    while (issued_upto - retired_upto < kFusedV4Stages
                           && issued_upto + 1 < kFusedStreamLength) {
                        ++issued_upto;
                        if (lane == 0) issue(issued_upto);
                    }
                };

                for (int step = 0; step < kEpochLength; ++step) {
                    const int index = epoch * kEpochLength + step;
                    const int task = fused_v4_task_of(index);
                    const int stage = index % kFusedV4Stages;
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
                                activation_scale[slot][quad]);
                        }
                    }
                    tensor_store_wait();

                    if (lane == 0) {
                        st_descriptor<fused_weight_tile, transpose::N>
                            weight_desc(weight[stage]);
                        st_descriptor<fused_activation_tile, transpose::N>
                            activation_desc(activation);
                        #pragma unroll
                        for (int group = 0; group < kFusedSlabGroups; ++group) {
                            const int quad = group / kScaleGroupsPerTile;
                            const int factor = group % kScaleGroupsPerTile;
                            fused_mixed_mma(
                                accumulator(task),
                                weight_desc.chunk_descriptor(group),
                                activation_desc.chunk_descriptor(group)
                                    + slot_offset,
                                scale_slot(set, quad),
                                scale_slot(
                                    set, kFusedSlabScaleTiles + quad),
                                factor,
                                slab != 0 || group != 0);
                        }
                        detail::tcgen05::commit<1>(slab_retired[stage]);
                    }
                    inner = clocks.lap(kClockRoutedGateUpMmaIssue, inner);

                    // Deferred retire, one index behind, exactly as the
                    // resident engine does it: index `i` waits on `i - 1`'s
                    // completion, which the tensor core reached while `i`'s
                    // sixteen issues were being made, so the wait is normally
                    // free and the pipe is never drained mid-slab.
                    if (index - 1 > retired_upto) {
                        const int retiring = (index - 1) % kFusedV4Stages;
                        wait(slab_retired[retiring],
                             take_phase(retired_phase, retiring));
                        retired_upto = index - 1;
                        mark = clocks.lap(kClockRoutedGateUpMma, mark);
                        inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                        refill();
                        inner = clocks.lap(kClockRoutedGateUpTmaIssue, inner);
                    }
                    mark = clocks.lap(kClockRoutedGateUpStage, mark);
                }

                // The slot this epoch read is what the producers refill two
                // barriers from now, so its last reader has to be finished
                // before they may. This is the one place the pipe is drained,
                // and how often is exactly the epoch length: once every six
                // indices under slab-major.
                const int last = epoch * kEpochLength + kEpochLength - 1;
                if (last > retired_upto) {
                    const int retiring = last % kFusedV4Stages;
                    wait(slab_retired[retiring],
                         take_phase(retired_phase, retiring));
                    retired_upto = last;
                    mark = clocks.lap(kClockRoutedGateUpMma, mark);
                    inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                    refill();
                    inner = clocks.lap(kClockRoutedGateUpTmaIssue, inner);
                }
                if (lane == 0) {
                    stream_parity[0] = arrived_phase;
                    stream_parity[1] = retired_phase;
                }
                clocks.lap(kClockRoutedGateUpStage, mark);
            } else if (epoch + 1 < kEpochs) {
                const int next_slot = (epoch + 1) % kFusedV4Slots;
                stage_fused_slab_activation(
                    activation, activation_scale[next_slot], scratch,
                    batch_begin, rows,
                    fused_v4_slab_of((epoch + 1) * kEpochLength),
                    next_slot, kFusedV4ProducerBase);
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            }
            // What warp 0 loses to the producers, which is what the resident
            // engine's `activation` counter measures too -- there it is the
            // gather itself, here it is whatever of the gather the ring did
            // not already cover.
            const unsigned long long handoff = clocks.now();
            __syncthreads();
            clocks.lap(kClockRoutedGateUpActivation, handoff);

        }

        // Six epilogues, after the stream rather than inside it. No task is
        // finished until the stream is over, so nothing flies underneath these
        // -- but nothing is left to fly either, and the resident engine's last
        // epilogue has no cover either.
        for (int task = 0; task < kFusedTasks; ++task) {
            const unsigned long long epilogue = clocks.now();
            store_fused_accumulator(accumulator(task), result);
            __syncthreads();
            quantize_fused_situ(result, scratch, batch_begin, rows, task);
            __syncthreads();
            clocks.lap(kClockRoutedGateUpEpilogue, epilogue);

            if (last_pass) {
                persistent::publish_count_at(arrival_counter);
            }
        }
    }
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
