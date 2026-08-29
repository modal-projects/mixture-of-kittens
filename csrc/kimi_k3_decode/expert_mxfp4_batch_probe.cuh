#pragma once

// Isolated output-channel-M/token-N routed-expert microprototype.
//
// Production keeps assignments on M and output channels on N. This probe
// flips those axes for batches of at most eight rows, so every mixed
// contraction is m128x8x32: one expert's weights occupy A/M, its selected
// token rows occupy B/N, and the accumulator is transposed while SiTU and the
// final scatter return to assignment-major storage. The launch can select the
// current units or the candidate units so both paths share identical setup,
// quantization, weights, output, and timing boundaries.

#include "expert_mxfp4.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace batch_probe {

inline constexpr int kBatchProbeM = 128;
inline constexpr int kBatchProbeN = 8;
// Tensor-memory and register tiles are sixteen-column granular. The MMA writes
// the first eight columns; readout ignores the untouched physical tail.
inline constexpr int kBatchProbePhysicalN = 16;

using batch_accumulator_tile =
    kittens::tt_fl<kBatchProbeM, kBatchProbePhysicalN>;
using batch_result_tile =
    kittens::st_fl<kBatchProbeM, kBatchProbePhysicalN>;

static_assert(kBatchProbeM == kMmaM);
static_assert(kBatchProbeN <= kBatchProbePhysicalN);

// Matrix A is packed E2M1 expert weight, matrix B is E4M3 activation. This is
// the opposite operand format order from the production assignment-M unit.
__host__ __device__ __forceinline__ constexpr std::uint32_t
batch_instruction_descriptor(const int scale_factor_id) {
    return (static_cast<std::uint32_t>(scale_factor_id) << 4)
         | (5u << 7)
         | (0u << 10)
         | (static_cast<std::uint32_t>(kBatchProbeN / 8) << 17)
         | (1u << 23)
         | (static_cast<std::uint32_t>(kBatchProbeM / 128) << 27)
         | (static_cast<std::uint32_t>(scale_factor_id) << 29);
}

static_assert(batch_instruction_descriptor(0) == 0x08820280u);

__device__ __forceinline__ void batch_mixed_mma(
    const batch_accumulator_tile &destination,
    const mixed_operand_tile &weight,
    const mixed_operand_tile &activation,
    const kittens::full_tt_fp8e8m0<16> &weight_scale,
    const kittens::full_tt_fp8e8m0<16> &activation_scale,
    const int scale_factor_id,
    const bool accumulate
) {
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N>
        weight_desc(weight);
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N>
        activation_desc(activation);
    const std::uint32_t instruction =
        batch_instruction_descriptor(scale_factor_id);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :
        : "r"(destination.addr),
          "l"(weight_desc.base_desc),
          "l"(activation_desc.base_desc),
          "r"(instruction),
          "r"(weight_scale.addr),
          "r"(activation_scale.addr),
          "r"(accumulate ? 1u : 0u)
    );
}

__device__ __forceinline__ void clear_token_operand_tile(
    mixed_operand_tile &tile
) {
    constexpr int atoms_per_row = kMmaK / 16;
    constexpr int atoms = kBatchProbeN * atoms_per_row;
    for (int index = static_cast<int>(threadIdx.x); index < atoms;
         index += kDecodeCtaThreads) {
        *atom_of(
            tile, index / atoms_per_row, index % atoms_per_row) =
                make_uint4(0u, 0u, 0u, 0u);
    }
}

__device__ __forceinline__ void clear_token_scale_tile(
    mixed_scale_tile &tile
) {
    for (int row = static_cast<int>(threadIdx.x); row < kBatchProbeN;
         row += kDecodeCtaThreads) {
        stage_scale_quad(tile, row, 0x7f7f7f7fu);
    }
}

__device__ __forceinline__ void store_batch_accumulator(
    const batch_accumulator_tile &accumulator,
    batch_result_tile &destination
) {
    using namespace kittens;
    if (warpgroup::groupid() == 0) {
        rt_fl<kBatchProbeM / 4, kBatchProbePhysicalN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(destination, result);
    }
    __syncthreads();
}

__device__ __forceinline__ void quantize_transposed_situ_tile(
    const batch_result_tile &gate,
    const batch_result_tile &up,
    const Scratch &scratch,
    const int assignment_begin,
    const int batch_rows,
    const int output_base
) {
    const int thread = static_cast<int>(threadIdx.x);
    constexpr int groups_per_tile = kMmaN / kMmaK;
    for (int index = thread; index < batch_rows * groups_per_tile;
         index += kDecodeCtaThreads) {
        const int row = index / groups_per_tile;
        const int local_group = index % groups_per_tile;
        const int assignment = assignment_begin + row;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const int column = local_group * kMmaK + k;
            const float gate_value = gate[{column, row}];
            const float up_value = up[{column, row}];
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value =
                4.0f * tanhf(gate_value * 0.25f) * sigmoid
                * 25.0f * tanhf(up_value / 25.0f);
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale = select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        const int global_group = output_base / kMmaK + local_group;
        scratch.situ_scale[
            static_cast<long long>(assignment) * kSituGroups
            + global_group] = scale;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            scratch.situ_mxfp8[
                static_cast<long long>(assignment)
                    * kRoutedIntermediateSizePerRank
                + output_base + local_group * kMmaK + k] =
                    quantize_e4m3(values[k], reciprocal);
        }
    }
}

__device__ __forceinline__ void accumulate_transposed_down_tile(
    const batch_result_tile &result,
    const Scratch &scratch,
    const int assignment_begin,
    const int batch_rows,
    const int output_base,
    const int active_tokens
) {
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < batch_rows * kMmaN;
         index += kDecodeCtaThreads) {
        const int row = index / kMmaN;
        const int column = index % kMmaN;
        const int assignment = assignment_begin + row;
        const int token = scratch.assignment_tokens[assignment];
        if (token >= 0 && token < active_tokens) {
            atomicAdd(
                &scratch.routed_accumulator[
                    static_cast<long long>(token) * kLatentSize
                    + output_base + column],
                result[{column, row}]
                    * decode_route_weight(scratch, assignment));
        }
    }
}

static __device__ void batched_gate_up_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const std::uint8_t *__restrict__ expert_w1_packed,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_packed,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const Scratch &scratch,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const int output_tile
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_operand_tile (&first_weight_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_operand_tile (&second_weight_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_scale_tile (&activation_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();
    mixed_scale_tile (&first_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();
    mixed_scale_tile (&second_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();

    tma_swizzle_allocator results(shared_raw);
    batch_result_tile (&first_result_shared) =
        results.allocate<batch_result_tile>();
    batch_result_tile (&second_result_shared) =
        results.allocate<batch_result_tile>();

    __shared__ semaphore gate_up_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(gate_up_done, 0, 1);

    batch_accumulator_tile first_accumulator =
        tensor_pool.allocate<batch_accumulator_tile>(0);
    batch_accumulator_tile second_accumulator =
        tensor_pool.allocate<batch_accumulator_tile>(kBatchProbePhysicalN);
    constexpr int scale_column_base = 2 * kBatchProbePhysicalN;
    const auto scale_slot = [&](const int buffer) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            scale_column_base + buffer * kRoutedScaleColumns);
    };

    #pragma unroll
    for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
        clear_token_operand_tile(activation_tile[slot]);
    }
    #pragma unroll
    for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
        clear_token_scale_tile(activation_scale_shared[quad]);
    }
    __syncthreads();

    const int rows = min(batch_rows, kBatchProbeN);
    const int weight_half = thread / kMmaN;
    const int weight_row = thread % kMmaN;
    const std::uint8_t *const weight_packed =
        weight_half == 0 ? expert_w1_packed : expert_w3_packed;
    const std::uint8_t *const weight_scales =
        weight_half == 0 ? expert_w1_scale : expert_w3_scale;
    mixed_operand_tile *const weight_tile =
        weight_half == 0 ? first_weight_tile : second_weight_tile;
    mixed_scale_tile *const weight_scale_shared =
        weight_half == 0 ? first_scale_shared : second_scale_shared;
    const int output_base = output_tile * kMmaN;
    const long long weight_index =
        static_cast<long long>(expert) * kExpertW1W3PackedRows
        + output_base + weight_row;
    const std::uint8_t *const weight_row_bytes =
        weight_packed + weight_index * kExpertW1W3PackedColumns;
    const std::uint8_t *const weight_row_scales =
        weight_scales + weight_index * kExpertW1W3ScaleColumns;

    uint4 payload[kGateUpRoundGroups];
    std::uint32_t scale_words[kGateUpScaleTiles];
    const auto read_weight_round = [&](
        const int group_base,
        uint4 (&into)[kGateUpRoundGroups],
        std::uint32_t (&scales)[kGateUpScaleTiles]
    ) {
        #pragma unroll
        for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
            into[slot] = *reinterpret_cast<const uint4 *>(
                weight_row_bytes + (group_base + slot) * (kMmaK / 2));
        }
        #pragma unroll
        for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
            scales[quad] = *reinterpret_cast<const std::uint32_t *>(
                weight_row_scales
                + group_base + quad * kScaleGroupsPerTile);
        }
    };
    read_weight_round(0, payload, scale_words);

    int compute_phase = 0;
    for (int round = 0; round < kGateUpRounds; ++round) {
        const int group_base = round * kGateUpRoundGroups;
        #pragma unroll
        for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
            stage_weight_row(weight_tile[slot], weight_row, payload[slot]);
        }
        #pragma unroll
        for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
            stage_scale_quad(
                weight_scale_shared[quad], weight_row, scale_words[quad]);
        }
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
        if (round + 1 < kGateUpRounds) {
            read_weight_round(
                group_base + kGateUpRoundGroups, payload, scale_words);
        }

        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if (warpid() == 0) {
            if (laneid() == 0) {
                #pragma unroll
                for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
                    auto first_scale = scale_slot(quad);
                    auto second_scale =
                        scale_slot(kGateUpScaleTiles + quad);
                    auto activation_scale =
                        scale_slot(2 * kGateUpScaleTiles + quad);
                    load_mxnv_scale_async(
                        first_scale, first_scale_shared[quad]);
                    load_mxnv_scale_async(
                        second_scale, second_scale_shared[quad]);
                    load_mxnv_scale_async(
                        activation_scale, activation_scale_shared[quad]);
                }
            }
            tensor_store_wait();
            if (laneid() == 0) {
                #pragma unroll
                for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
                    const int quad = slot / kScaleGroupsPerTile;
                    const int scale_factor_id =
                        slot % kScaleGroupsPerTile;
                    const bool accumulate = round != 0 || slot != 0;
                    batch_mixed_mma(
                        first_accumulator, first_weight_tile[slot],
                        activation_tile[slot], scale_slot(quad),
                        scale_slot(2 * kGateUpScaleTiles + quad),
                        scale_factor_id, accumulate);
                    batch_mixed_mma(
                        second_accumulator, second_weight_tile[slot],
                        activation_tile[slot],
                        scale_slot(kGateUpScaleTiles + quad),
                        scale_slot(2 * kGateUpScaleTiles + quad),
                        scale_factor_id, accumulate);
                }
                detail::tcgen05::commit<1>(gate_up_done);
            }
        }
        wait(gate_up_done, compute_phase);
        __syncthreads();
        compute_phase ^= 1;
    }

    store_batch_accumulator(first_accumulator, first_result_shared);
    store_batch_accumulator(second_accumulator, second_result_shared);
    quantize_transposed_situ_tile(
        first_result_shared, second_result_shared, scratch, assignment_begin,
        rows, output_base);
    __syncthreads();
}

static __device__ void batched_down_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const Scratch &scratch,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const int output_tile,
    const int active_tokens
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kDownRoundGroups] =
        staging.allocate<mixed_operand_tile, kDownRoundGroups>();
    mixed_operand_tile (&weight_tile)[kDownRoundGroups] =
        staging.allocate<mixed_operand_tile, kDownRoundGroups>();
    mixed_scale_tile (&activation_scale_shared)[kDownScaleTiles] =
        staging.allocate<mixed_scale_tile, kDownScaleTiles>();
    mixed_scale_tile (&weight_scale_shared)[kDownScaleTiles] =
        staging.allocate<mixed_scale_tile, kDownScaleTiles>();

    tma_swizzle_allocator results(shared_raw);
    batch_result_tile (&result_shared) =
        results.allocate<batch_result_tile>();

    __shared__ semaphore down_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(down_done, 0, 1);

    batch_accumulator_tile accumulator =
        tensor_pool.allocate<batch_accumulator_tile>(0);
    constexpr int scale_column_base = kBatchProbePhysicalN;
    const auto scale_slot = [&](const int buffer) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            scale_column_base + buffer * kRoutedScaleColumns);
    };

    #pragma unroll
    for (int slot = 0; slot < kDownRoundGroups; ++slot) {
        clear_token_operand_tile(activation_tile[slot]);
    }
    #pragma unroll
    for (int quad = 0; quad < kDownScaleTiles; ++quad) {
        clear_token_scale_tile(activation_scale_shared[quad]);
    }
    __syncthreads();

    const int rows = min(batch_rows, kBatchProbeN);
    constexpr int groups_per_thread = kDownRoundGroups / 2;
    const int weight_row = thread % kMmaN;
    const int weight_half = thread / kMmaN;
    const int weight_group_base = weight_half * groups_per_thread;
    const long long weight_index =
        static_cast<long long>(expert) * kExpertW2PackedRows
        + output_tile * kMmaN + weight_row;
    const std::uint8_t *const weight_row_bytes =
        expert_w2_packed + weight_index * kExpertW2PackedColumns;
    const int output_base = output_tile * kMmaN;

    uint4 payload[groups_per_thread];
    #pragma unroll
    for (int slot = 0; slot < groups_per_thread; ++slot) {
        payload[slot] = *reinterpret_cast<const uint4 *>(
            weight_row_bytes
            + (weight_group_base + slot) * (kMmaK / 2));
    }
    #pragma unroll
    for (int slot = 0; slot < groups_per_thread; ++slot) {
        stage_weight_row(
            weight_tile[weight_group_base + slot], weight_row,
            payload[slot]);
    }
    for (int index = thread; index < kMmaN * kDownScaleTiles;
         index += kDecodeCtaThreads) {
        const int row = index / kDownScaleTiles;
        const int quad = index % kDownScaleTiles;
        const long long scale_index =
            static_cast<long long>(expert) * kExpertW2PackedRows
            + output_tile * kMmaN + row;
        stage_scale_quad(
            weight_scale_shared[quad], row,
            *reinterpret_cast<const std::uint32_t *>(
                expert_w2_scale + scale_index * kExpertW2ScaleColumns
                + quad * kScaleGroupsPerTile));
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
                + group * kMmaK);
    }
    for (int index = thread; index < rows * kDownScaleTiles;
         index += kDecodeCtaThreads) {
        const int row = index / kDownScaleTiles;
        const int quad = index % kDownScaleTiles;
        const int assignment = assignment_begin + row;
        stage_scale_quad(
            activation_scale_shared[quad], row,
            *reinterpret_cast<const std::uint32_t *>(
                scratch.situ_scale
                + static_cast<long long>(assignment) * kSituGroups
                + quad * kScaleGroupsPerTile));
    }

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    if (warpid() == 0) {
        if (laneid() == 0) {
            #pragma unroll
            for (int quad = 0; quad < kDownScaleTiles; ++quad) {
                auto weight_scale = scale_slot(quad);
                auto activation_scale =
                    scale_slot(kDownScaleTiles + quad);
                load_mxnv_scale_async(
                    weight_scale, weight_scale_shared[quad]);
                load_mxnv_scale_async(
                    activation_scale, activation_scale_shared[quad]);
            }
        }
        tensor_store_wait();
        if (laneid() == 0) {
            #pragma unroll
            for (int slot = 0; slot < kDownRoundGroups; ++slot) {
                const int quad = slot / kScaleGroupsPerTile;
                const int scale_factor_id = slot % kScaleGroupsPerTile;
                batch_mixed_mma(
                    accumulator, weight_tile[slot], activation_tile[slot],
                    scale_slot(quad),
                    scale_slot(kDownScaleTiles + quad),
                    scale_factor_id, slot != 0);
            }
            detail::tcgen05::commit<1>(down_done);
        }
    }
    wait(down_done, 0);
    __syncthreads();

    store_batch_accumulator(accumulator, result_shared);
    accumulate_transposed_down_tile(
        result_shared, scratch, assignment_begin, rows, output_base,
        active_tokens);
    __syncthreads();
}

template <bool UseBatchProbe>
static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void expert_probe_kernel(
    const __nv_bfloat16 *__restrict__ latent_x,
    const std::uint8_t *__restrict__ expert_w1_packed,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_packed,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    __nv_bfloat16 *__restrict__ output,
    std::uint8_t *__restrict__ scratch_bytes,
    const int expert,
    const int rows
) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];
    tensor_allocator<1, 1> tensor_pool{};
    const Scratch scratch = scratch_view(scratch_bytes);
    const int thread = static_cast<int>(threadIdx.x);

    for (int row = thread; row < rows; row += kDecodeCtaThreads) {
        scratch.assignment_tokens[row] = row;
        scratch.assignment_slots[row] = 0;
        scratch.expert_weights[row * kTopK] = 1.0f;
    }
    for (int index = thread; index < rows * kLatentSize;
         index += kDecodeCtaThreads) {
        scratch.routed_accumulator[index] = 0.0f;
        output[index] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    quantize_latent_rows(latent_x, scratch, rows, 0, 1);
    __threadfence();
    __syncthreads();

    for (int output_tile = 0; output_tile < kGateUpTiles; ++output_tile) {
        if constexpr (UseBatchProbe) {
            batched_gate_up_unit(
                shared_raw, tensor_pool, expert_w1_packed, expert_w1_scale,
                expert_w3_packed, expert_w3_scale, scratch, expert, 0, rows,
                output_tile);
        } else {
            routed_gate_up_unit(
                shared_raw, tensor_pool, expert_w1_packed, expert_w1_scale,
                expert_w3_packed, expert_w3_scale, scratch, expert, 0, rows,
                output_tile, PhaseClocks{nullptr});
        }
    }
    for (int output_tile = 0; output_tile < kDownTiles; ++output_tile) {
        if constexpr (UseBatchProbe) {
            batched_down_unit(
                shared_raw, tensor_pool, expert_w2_packed, expert_w2_scale,
                scratch, expert, 0, rows, output_tile, rows);
        } else {
            routed_down_unit(
                shared_raw, tensor_pool, expert_w2_packed, expert_w2_scale,
                scratch, expert, 0, rows, output_tile, rows,
                PhaseClocks{nullptr});
        }
    }

    for (int index = thread; index < rows * kLatentSize;
         index += kDecodeCtaThreads) {
        output[index] =
            __float2bfloat16(scratch.routed_accumulator[index]);
    }
}

inline constexpr int kBatchProbeSharedBytes =
    kGateUpStagingBytes > kDownStagingBytes
        ? kGateUpStagingBytes
        : kDownStagingBytes;
static_assert(kBatchProbeSharedBytes == 102400);

static __host__ void batched_expert_probe_entrypoint(
    const at::Tensor &latent_x,
    const at::Tensor &expert_w1_packed,
    const at::Tensor &expert_w1_scale,
    const at::Tensor &expert_w3_packed,
    const at::Tensor &expert_w3_scale,
    const at::Tensor &expert_w2_packed,
    const at::Tensor &expert_w2_scale,
    const at::Tensor &output,
    const at::Tensor &scratch,
    const std::int64_t expert,
    const bool use_batch_probe
) {
    for (const at::Tensor *tensor : {
             &latent_x, &expert_w1_packed, &expert_w1_scale,
             &expert_w3_packed, &expert_w3_scale,
             &expert_w2_packed, &expert_w2_scale, &output, &scratch}) {
        TORCH_CHECK(
            tensor->device().is_cuda(),
            "MoK: _kimi_k3_batched_expert_probe requires CUDA tensors");
        TORCH_CHECK(
            tensor->is_contiguous(),
            "MoK: _kimi_k3_batched_expert_probe requires contiguous tensors");
    }
    const std::int64_t rows = latent_x.size(0);
    TORCH_CHECK(
        latent_x.dim() == 2 && rows >= 1 && rows <= kBatchProbeN
            && latent_x.size(1) == kLatentSize
            && latent_x.scalar_type() == at::kBFloat16,
        "MoK: _kimi_k3_batched_expert_probe requires BF16 latent_x [1..8, ",
        kLatentSize, "]");
    const auto check_weight = [](
        const at::Tensor &tensor,
        const char *name,
        const int expected_rows,
        const int expected_columns
    ) {
        TORCH_CHECK(
            tensor.dim() == 3 && tensor.size(0) >= 1
                && tensor.size(1) == expected_rows
                && tensor.size(2) == expected_columns
                && tensor.scalar_type() == at::kByte,
            "MoK: _kimi_k3_batched_expert_probe requires uint8 ", name,
            " [E, ", expected_rows, ", ", expected_columns, "]");
    };
    check_weight(
        expert_w1_packed, "expert_w1_packed",
        kExpertW1W3PackedRows, kExpertW1W3PackedColumns);
    check_weight(
        expert_w1_scale, "expert_w1_scale",
        kExpertW1W3PackedRows, kExpertW1W3ScaleColumns);
    check_weight(
        expert_w3_packed, "expert_w3_packed",
        kExpertW1W3PackedRows, kExpertW1W3PackedColumns);
    check_weight(
        expert_w3_scale, "expert_w3_scale",
        kExpertW1W3PackedRows, kExpertW1W3ScaleColumns);
    check_weight(
        expert_w2_packed, "expert_w2_packed",
        kExpertW2PackedRows, kExpertW2PackedColumns);
    check_weight(
        expert_w2_scale, "expert_w2_scale",
        kExpertW2PackedRows, kExpertW2ScaleColumns);
    TORCH_CHECK(
        expert >= 0 && expert < expert_w1_packed.size(0)
            && expert < expert_w1_scale.size(0)
            && expert < expert_w3_packed.size(0)
            && expert < expert_w3_scale.size(0)
            && expert < expert_w2_packed.size(0)
            && expert < expert_w2_scale.size(0),
        "MoK: _kimi_k3_batched_expert_probe expert is out of range");
    TORCH_CHECK(
        output.dim() == 2 && output.sizes() == latent_x.sizes()
            && output.scalar_type() == at::kBFloat16,
        "MoK: _kimi_k3_batched_expert_probe output must match latent_x");
    TORCH_CHECK(
        scratch.dim() == 1 && scratch.scalar_type() == at::kByte
            && scratch.size(0) >= SCRATCH_BYTES,
        "MoK: _kimi_k3_batched_expert_probe scratch is too small");
    const at::Device device = latent_x.device();
    for (const at::Tensor *tensor : {
             &expert_w1_packed, &expert_w1_scale,
             &expert_w3_packed, &expert_w3_scale,
             &expert_w2_packed, &expert_w2_scale, &output, &scratch}) {
        TORCH_CHECK(
            tensor->device() == device,
            "MoK: _kimi_k3_batched_expert_probe requires one CUDA device");
    }

    const c10::cuda::CUDAGuard device_guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (use_batch_probe) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            expert_probe_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kBatchProbeSharedBytes));
        expert_probe_kernel<true>
            <<<1, kDecodeCtaThreads, kBatchProbeSharedBytes, stream>>>(
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
                reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
                reinterpret_cast<std::uint8_t *>(scratch.data_ptr()),
                static_cast<int>(expert), static_cast<int>(rows));
    } else {
        constexpr int baseline_shared_bytes = kGateUpUnitSharedBytes;
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            expert_probe_kernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            baseline_shared_bytes));
        expert_probe_kernel<false>
            <<<1, kDecodeCtaThreads, baseline_shared_bytes, stream>>>(
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
                reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
                reinterpret_cast<std::uint8_t *>(scratch.data_ptr()),
                static_cast<int>(expert), static_cast<int>(rows));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace batch_probe
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
