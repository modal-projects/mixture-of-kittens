#pragma once

// The two routed-expert work units and the private kernel that walks them.
//
// A unit is one 128-column output tile of one expert's assignment batch: three
// gate/up units and twenty-eight down units per batch. `routed_gate_up_unit`
// and `routed_down_unit` are what the persistent kernel hands out through its
// device task queue and what the private fallback kernel below calls in a
// loop, so both paths contract identical arithmetic in identical order.
//
// The contraction primitive is in `expert_mxfp4_mma.cuh` and the staging in
// `expert_mxfp4_staging.cuh`; this header includes the latter, which includes
// the former, so every existing include of this file sees the whole namespace
// exactly as it did before.

#include "expert_mxfp4_staging.cuh"

#include <cstdint>

namespace kimi_k3_decode {
namespace expert_mxfp4 {

// ---------------------------------------------------------------------------
// One routed-expert work unit.
//
// A unit is one 128-column output tile of one expert's assignment batch: three
// gate/up units and twenty-eight down units per batch. The private stage below
// walks every unit of every expert from a single CTA; the persistent kernel
// hands the same units out through a device task queue. Both call the two
// functions here, so the fallback and the production path contract identical
// arithmetic in identical order.
//
// Each unit re-derives its shared tiles from the same base and re-initializes
// its own semaphore, which is safe because a unit always waits for its last
// commit before returning: nothing is ever in flight across the boundary.
// ---------------------------------------------------------------------------

/// Contract one expert batch's gate and up tiles and stage the SiTU result.
///
/// `tensor_pool` is owned by the caller because a CTA may allocate tensor
/// memory only once: the persistent kernel provisions one pool at entry and
/// hands it to every unit, and the private kernel provisions one of its own.
static __device__ void routed_gate_up_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const RoutedLayouts &layouts,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const Scratch &scratch,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const int output_tile,
    const PhaseClocks clocks
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    direct_weight_stage (&first_weight_stage)[kWeightPipelineStages] =
        staging.allocate<direct_weight_stage, kWeightPipelineStages>();
    direct_weight_stage (&second_weight_stage)[kWeightPipelineStages] =
        staging.allocate<direct_weight_stage, kWeightPipelineStages>();
    mixed_scale_tile (&activation_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();
    mixed_scale_tile (&first_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();
    mixed_scale_tile (&second_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();

    // The results are laid over the staging, which is dead by the time the
    // accumulators are read out.
    tma_swizzle_allocator results(shared_raw);
    mixed_result_tile (&first_result_shared) =
        results.allocate<mixed_result_tile>();
    mixed_result_tile (&second_result_shared) =
        results.allocate<mixed_result_tile>();

    __shared__ semaphore gate_up_done;
    __shared__ semaphore weight_arrived;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) {
        init_semaphore(gate_up_done, 0, 1);
        init_semaphore(weight_arrived, 0, 1);
    }

    mixed_accumulator_tile first_accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    mixed_accumulator_tile second_accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(kMmaN);
    const auto scale_slot = [&](const int buffer) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kRoutedScaleColumnBase + buffer * kRoutedScaleColumns);
    };

    // Only the activation is short: TMA and the scale stores rewrite every
    // weight row in full on every round.
    #pragma unroll
    for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
        clear_operand_tile(activation_tile[slot]);
    }
    #pragma unroll
    for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
        clear_scale_tile(activation_scale_shared[quad]);
    }
    __syncthreads();

    // One MMA batch is 128 rows and an expert never collects more, because a
    // token's sixteen routes are sixteen distinct experts. The staging indexes
    // the tile directly rather than guarding each row, so the bound is taken
    // here rather than assumed.
    const int rows = min(batch_rows, kMmaM);

    // One thread owns one row of one of the two weight-scale tiles.
    const int weight_half = thread / kMmaN;
    const int weight_row = thread % kMmaN;
    const std::uint8_t *const weight_scales =
        weight_half == 0 ? expert_w1_scale : expert_w3_scale;
    mixed_scale_tile *const weight_scale_shared =
        weight_half == 0 ? first_scale_shared : second_scale_shared;
    const int output_base = output_tile * kMmaN;
    const long long weight_index =
        static_cast<long long>(expert) * kExpertW1W3PackedRows
        + output_base + weight_row;
    const std::uint8_t *const weight_row_scales =
        weight_scales + weight_index * kExpertW1W3ScaleColumns;

    // TMA writes each four-group round directly into the 128-byte-swizzled
    // layout the MMA consumes. The alternate stage receives the next round
    // while the tensor core reads the current one.
    //
    // Scales remain a small register prefetch. Their row pitches are already
    // compact, while the packed payload is the measured strided-load burden
    // this candidate is isolating.
    std::uint32_t scale_words[kGateUpScaleTiles];
    const auto read_scale_round = [&](
        const int group_base,
        std::uint32_t (&scales)[kGateUpScaleTiles]
    ) {
        #pragma unroll
        for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
            scales[quad] = *reinterpret_cast<const std::uint32_t *>(
                weight_row_scales + group_base + quad * kScaleGroupsPerTile);
        }
    };
    read_scale_round(0, scale_words);
    issue_direct_weight_round(
        first_weight_stage, second_weight_stage,
        layouts.w1, layouts.w3, expert, output_tile, 0, 0, weight_arrived);

    int compute_phase = 0;
    unsigned long long mark = clocks.now();
    #pragma unroll 1
    for (int round = 0; round < kGateUpRounds; ++round) {
        const int group_base = round * kGateUpRoundGroups;
        const int stage = round % kWeightPipelineStages;
        wait(weight_arrived, round % 2);

        #pragma unroll
        for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
            stage_scale_quad(
                weight_scale_shared[quad], weight_row, scale_words[quad]);
        }

        // The batch is at most 128 rows and usually one, so the activation is
        // spread over whatever threads its rows and groups need.
        for (int index = thread; index < rows * kGateUpRoundGroups;
             index += kDecodeCtaThreads) {
            const int row = index / kGateUpRoundGroups;
            const int slot = index % kGateUpRoundGroups;
            const int token =
                scratch.assignment_tokens[assignment_begin + row];
            stage_activation_row(
                activation_tile[slot], row,
                scratch.latent_mxfp8
                    + static_cast<long long>(token) * kLatentSize
                    + (group_base + slot) * kMmaK);
        }
        for (int index = thread; index < rows * kGateUpScaleTiles;
             index += kDecodeCtaThreads) {
            const int row = index / kGateUpScaleTiles;
            const int quad = index % kGateUpScaleTiles;
            const int token =
                scratch.assignment_tokens[assignment_begin + row];
            stage_scale_quad(
                activation_scale_shared[quad], row,
                *reinterpret_cast<const std::uint32_t *>(
                    scratch.latent_scale
                    + static_cast<long long>(token) * kLatentGroups
                    + group_base + quad * kScaleGroupsPerTile));
        }

        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if (round + 1 < kGateUpRounds) {
            read_scale_round(group_base + kGateUpRoundGroups, scale_words);
            issue_direct_weight_round(
                first_weight_stage, second_weight_stage,
                layouts.w1, layouts.w3, expert, output_tile,
                round + 1, (round + 1) % kWeightPipelineStages,
                weight_arrived);
        }
        mark = clocks.lap(kClockRoutedGateUpStage, mark);
        // `tcgen05.cp`, `tcgen05.mma`, and `tcgen05.commit` are single-thread
        // issues, but `tcgen05.wait::st` is `.sync.aligned`, so the whole warp
        // has to reach it.
        if (warpid() == 0) {
            if (laneid() == 0) {
                #pragma unroll
                for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
                    auto activation_scale = scale_slot(quad);
                    auto first_scale = scale_slot(kGateUpScaleTiles + quad);
                    auto second_scale =
                        scale_slot(2 * kGateUpScaleTiles + quad);
                    load_mxnv_scale_async(
                        activation_scale, activation_scale_shared[quad]);
                    load_mxnv_scale_async(
                        first_scale, first_scale_shared[quad]);
                    load_mxnv_scale_async(
                        second_scale, second_scale_shared[quad]);
                }
            }
            tensor_store_wait();
            if (laneid() == 0) {
                #pragma unroll
                for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
                    const int quad = slot / kScaleGroupsPerTile;
                    const int scale_factor_id = slot % kScaleGroupsPerTile;
                    const bool accumulate = round != 0 || slot != 0;
                    mixed_mma_direct(
                        first_accumulator, activation_tile[slot],
                        first_weight_stage[stage], slot, scale_slot(quad),
                        scale_slot(kGateUpScaleTiles + quad),
                        scale_factor_id, accumulate);
                    mixed_mma_direct(
                        second_accumulator, activation_tile[slot],
                        second_weight_stage[stage], slot, scale_slot(quad),
                        scale_slot(2 * kGateUpScaleTiles + quad),
                        scale_factor_id, accumulate);
                }
                detail::tcgen05::commit<1>(gate_up_done);
            }
        }
        wait(gate_up_done, compute_phase);
        __syncthreads();
        mark = clocks.lap(kClockRoutedGateUpMma, mark);
        compute_phase ^= 1;
    }

    store_accumulator(first_accumulator, first_result_shared);
    store_accumulator(second_accumulator, second_result_shared);
    quantize_situ_tile(
        first_result_shared, second_result_shared, scratch, assignment_begin,
        rows, output_base);
    __syncthreads();
}

/// Contract one expert batch's down tile and weight it into the accumulator.
static __device__ void routed_down_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const RoutedLayouts &layouts,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const Scratch &scratch,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const int output_tile,
    const int active_tokens,
    const PhaseClocks clocks
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kDownRoundGroups] =
        staging.allocate<mixed_operand_tile, kDownRoundGroups>();
    direct_weight_stage (&weight_stage)[kWeightPipelineStages] =
        staging.allocate<direct_weight_stage, kWeightPipelineStages>();
    mixed_scale_tile (&activation_scale_shared)[kDownScaleTiles] =
        staging.allocate<mixed_scale_tile, kDownScaleTiles>();
    mixed_scale_tile (&weight_scale_shared)[kDownScaleTiles] =
        staging.allocate<mixed_scale_tile, kDownScaleTiles>();

    tma_swizzle_allocator results(shared_raw);
    mixed_result_tile (&result_shared) =
        results.allocate<mixed_result_tile>();

    __shared__ semaphore down_done;
    __shared__ semaphore weight_arrived;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) {
        init_semaphore(down_done, 0, 1);
        init_semaphore(weight_arrived, 0, 1);
    }

    mixed_accumulator_tile accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    const auto scale_slot = [&](const int buffer) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kRoutedScaleColumnBase + buffer * kRoutedScaleColumns);
    };

    #pragma unroll
    for (int slot = 0; slot < kDownRoundGroups; ++slot) {
        clear_operand_tile(activation_tile[slot]);
    }
    #pragma unroll
    for (int quad = 0; quad < kDownScaleTiles; ++quad) {
        clear_scale_tile(activation_scale_shared[quad]);
    }
    __syncthreads();

    const int rows = min(batch_rows, kMmaM);

    const int output_base = output_tile * kMmaN;
    unsigned long long mark = clocks.now();

    issue_direct_weight_round(
        weight_stage, layouts.w2, expert, output_tile, 0, 0,
        weight_arrived);
    wait(weight_arrived, 0);
    int compute_phase = 0;
    #pragma unroll 1
    for (int round = 0; round < kDownRounds; ++round) {
        const int group_base = round * kDownRoundGroups;
        const int stage = round % kWeightPipelineStages;
        if (round != 0) {
            wait(weight_arrived, round % 2);
        }
        for (int row = thread; row < kMmaN; row += kDecodeCtaThreads) {
            const long long scale_index =
                static_cast<long long>(expert) * kExpertW2PackedRows
                + output_base + row;
            stage_scale_quad(
                weight_scale_shared[0], row,
                *reinterpret_cast<const std::uint32_t *>(
                    expert_w2_scale
                    + scale_index * kExpertW2ScaleColumns
                    + group_base));
        }

        for (int index = thread; index < rows * kDownRoundGroups;
             index += kDecodeCtaThreads) {
            const int row = index / kDownRoundGroups;
            const int group = index % kDownRoundGroups;
            const int assignment = assignment_begin + row;
            stage_activation_row(
                activation_tile[group], row,
                scratch.situ_mxfp8
                    + static_cast<long long>(assignment)
                          * kRoutedIntermediateSizePerRank
                    + (group_base + group) * kMmaK);
        }
        for (int row = thread; row < rows; row += kDecodeCtaThreads) {
            const int assignment = assignment_begin + row;
            stage_scale_quad(
                activation_scale_shared[0], row,
                *reinterpret_cast<const std::uint32_t *>(
                    scratch.situ_scale
                    + static_cast<long long>(assignment) * kSituGroups
                    + group_base));
        }

        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if (round + 1 < kDownRounds) {
            issue_direct_weight_round(
                weight_stage, layouts.w2, expert, output_tile, round + 1,
                (round + 1) % kWeightPipelineStages, weight_arrived);
        }
        mark = clocks.lap(kClockRoutedDownStage, mark);
        if (warpid() == 0) {
            if (laneid() == 0) {
                auto activation_scale = scale_slot(0);
                auto weight_scale = scale_slot(kDownScaleTiles);
                load_mxnv_scale_async(
                    activation_scale, activation_scale_shared[0]);
                load_mxnv_scale_async(
                    weight_scale, weight_scale_shared[0]);
            }
            tensor_store_wait();
            if (laneid() == 0) {
                #pragma unroll
                for (int slot = 0; slot < kDownRoundGroups; ++slot) {
                    mixed_mma_direct(
                        accumulator, activation_tile[slot],
                        weight_stage[stage], slot,
                        scale_slot(0), scale_slot(kDownScaleTiles),
                        slot, round != 0 || slot != 0);
                }
                detail::tcgen05::commit<1>(down_done);
            }
        }
        wait(down_done, compute_phase);
        __syncthreads();
        mark = clocks.lap(kClockRoutedDownMma, mark);
        compute_phase ^= 1;
    }

    store_accumulator(accumulator, result_shared);
    accumulate_down_tile(
        result_shared, scratch, assignment_begin, rows, output_base,
        active_tokens);
    __syncthreads();
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void kimi_k3_routed_experts_kernel(
    const __nv_bfloat16 *__restrict__ latent_x,
    const std::uint8_t *__restrict__ expert_w1_packed,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_packed,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const __grid_constant__ RoutedLayouts layouts,
    __nv_bfloat16 *__restrict__ routed_output,
    std::uint8_t *__restrict__ scratch_bytes,
    const int active_tokens,
    const int tokens
) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];

    // The managed allocator barriers the whole CTA, so provisioning it here
    // covers every unit this block later runs.
    tensor_allocator<1, 1> tensor_pool{};

    const Scratch scratch = scratch_view(scratch_bytes);
    const int thread = static_cast<int>(threadIdx.x);
    const int active_values = active_tokens * kLatentSize;
    for (int index = thread; index < active_values;
         index += kDecodeCtaThreads) {
        scratch.routed_accumulator[index] = 0.0f;
    }
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int index = thread; index < tokens * kLatentSize;
         index += kDecodeCtaThreads) {
        routed_output[index] = zero;
    }
    __syncthreads();

    quantize_latent_rows(latent_x, scratch, active_tokens, 0, 1);
    // Every thread flushes its own MXFP8 writes, and the barrier then holds the
    // publishing thread until all of those flushes have landed device-wide, so a
    // consumer that sees the new generation cannot read a stale block.
    __threadfence();
    __syncthreads();
    if (thread == 0) {
        atomicExch(&scratch.phase[kExpertQuantizationArrivals], 0);
        atomicAdd(&scratch.phase[kExpertQuantizationGeneration], 1);
    }
    __syncthreads();

    const int total_assignments =
        min(max(scratch.expert_offsets[kNumExperts], 0), kMaxRoutes);
    for (int expert = 0; expert < kNumExperts; ++expert) {
        const int expert_begin =
            min(max(scratch.expert_offsets[expert], 0), total_assignments);
        const int expert_end =
            min(max(scratch.expert_offsets[expert + 1], expert_begin),
                total_assignments);
        if (expert_begin == expert_end) continue;

        for (int assignment_begin = expert_begin;
             assignment_begin < expert_end;
             assignment_begin += kMmaM) {
            const int batch_rows =
                min(kMmaM, expert_end - assignment_begin);

            for (int output_tile = 0; output_tile < kGateUpTiles;
                 ++output_tile) {
                routed_gate_up_unit(
                    shared_raw, tensor_pool, layouts, expert_w1_scale,
                    expert_w3_scale, scratch, expert, assignment_begin,
                    batch_rows, output_tile,
                    PhaseClocks{nullptr});
            }

            for (int output_tile = 0; output_tile < kDownTiles;
                 ++output_tile) {
                routed_down_unit(
                    shared_raw, tensor_pool, layouts, expert_w2_scale, scratch,
                    expert, assignment_begin, batch_rows, output_tile,
                    active_tokens, PhaseClocks{nullptr});
            }
        }
    }

    for (int index = thread; index < active_values;
         index += kDecodeCtaThreads) {
        routed_output[index] =
            __float2bfloat16(scratch.routed_accumulator[index]);
    }
    // Same ordering for the routed partial: flush per thread, barrier, publish.
    __threadfence();
    __syncthreads();
    if (thread == 0) {
        atomicExch(&scratch.phase[kExpertCompletionArrivals], 0);
        atomicAdd(&scratch.phase[kExpertCompletionGeneration], 1);
    }
    __syncthreads();
}

static __host__ void launch_routed_experts(
    const at::Tensor &latent_x,
    const at::Tensor &expert_w1_packed,
    const at::Tensor &expert_w1_scale,
    const at::Tensor &expert_w3_packed,
    const at::Tensor &expert_w3_scale,
    const at::Tensor &expert_w2_packed,
    const at::Tensor &expert_w2_scale,
    const at::Tensor &routed_output,
    const at::Tensor &scratch,
    const int active_tokens
) {
    const RoutedLayouts layouts = routed_layouts(
        reinterpret_cast<const std::uint8_t *>(expert_w1_packed.data_ptr()),
        reinterpret_cast<const std::uint8_t *>(expert_w3_packed.data_ptr()),
        reinterpret_cast<const std::uint8_t *>(expert_w2_packed.data_ptr()));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kimi_k3_routed_experts_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kProbeSharedBytes));
    kimi_k3_routed_experts_kernel
        <<<1, kDecodeCtaThreads, kProbeSharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(latent_x.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w1_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w1_scale.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w3_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w3_scale.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w2_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w2_scale.data_ptr()),
            layouts,
            reinterpret_cast<__nv_bfloat16 *>(routed_output.data_ptr()),
            reinterpret_cast<std::uint8_t *>(scratch.data_ptr()),
            active_tokens,
            static_cast<int>(latent_x.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
