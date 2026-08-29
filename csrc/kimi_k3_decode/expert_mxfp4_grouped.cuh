#pragma once

// Production expert-pure routed-down pipeline and benchmark-only gate/up
// grouping candidates.
//
// One queue unit owns several output tiles of one expert. The expert's at-most
// eight live assignment rows are placed on N while output channels stay on M,
// so every contraction is the validated native m128x8x32 shape. A K chunk's
// activation is staged once and reused by all output tiles in the group.
// Weight shared memory is double buffered: warp 0 feeds the current buffer to
// tcgen05 while the remaining warps fill the next buffer.
//
// Grouped down ships at width four. Gate/up deliberately does not: the
// measured width-three prototype cut staging but made six accumulators'
// tcgen05 work serial enough to regress. The templates below expose widths one
// and two only to the guarded benchmark path, so the next measurement can
// separate the m128x8 mapping from activation reuse without altering the
// production launch.

#include "expert_mxfp4.cuh"

#include <cstdint>

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace grouped_pipeline {

inline constexpr int kGroupedM = 128;
inline constexpr int kGroupedN = 8;
inline constexpr int kGroupedPhysicalN = 16;
inline constexpr int kGroupedDownWidth = 4;
inline constexpr int kGroupedDownUnits =
    (kDownTiles + kGroupedDownWidth - 1) / kGroupedDownWidth;
inline constexpr int kGroupedRoundGroups = 4;
inline constexpr int kGroupedGateUpRounds =
    kLatentGroups / kGroupedRoundGroups;
inline constexpr int kGroupedDownRounds =
    kSituGroups / kGroupedRoundGroups;
inline constexpr int kGroupedScaleTiles =
    kGroupedRoundGroups / kScaleGroupsPerTile;

static_assert(kGroupedM == kMmaM);
static_assert(kGroupedDownUnits == 7);
static_assert(kLatentGroups % kGroupedRoundGroups == 0);
static_assert(kSituGroups % kGroupedRoundGroups == 0);
static_assert(kGroupedScaleTiles == 1);

template<int GROUP_SIZE>
__host__ __device__ constexpr int grouped_gate_up_units() {
    static_assert(GROUP_SIZE >= 1 && GROUP_SIZE <= 2);
    return (kGateUpTiles + GROUP_SIZE - 1) / GROUP_SIZE;
}

static_assert(grouped_gate_up_units<1>() == 3);
static_assert(grouped_gate_up_units<2>() == 2);

using grouped_accumulator_tile =
    kittens::tt_fl<kGroupedM, kGroupedPhysicalN>;
using grouped_result_tile =
    kittens::st_fl<kGroupedM, kGroupedPhysicalN>;

__host__ __device__ __forceinline__ constexpr std::uint32_t
grouped_instruction_descriptor(const int scale_factor_id) {
    return (static_cast<std::uint32_t>(scale_factor_id) << 4)
         | (5u << 7)
         | (0u << 10)
         | (static_cast<std::uint32_t>(kGroupedN / 8) << 17)
         | (1u << 23)
         | (static_cast<std::uint32_t>(kGroupedM / 128) << 27)
         | (static_cast<std::uint32_t>(scale_factor_id) << 29);
}

static_assert(grouped_instruction_descriptor(0) == 0x08820280u);

__device__ __forceinline__ void grouped_batch_mixed_mma(
    const grouped_accumulator_tile &destination,
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
        grouped_instruction_descriptor(scale_factor_id);
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

__device__ __forceinline__ void clear_grouped_activation(
    mixed_operand_tile (&activation_tile)[kGroupedRoundGroups],
    mixed_scale_tile &activation_scale_shared
) {
    constexpr int atoms_per_row = kMmaK / 16;
    constexpr int atoms =
        kGroupedN * kGroupedRoundGroups * atoms_per_row;
    for (int index = static_cast<int>(threadIdx.x); index < atoms;
         index += kDecodeCtaThreads) {
        const int tile = index / (kGroupedN * atoms_per_row);
        const int within = index % (kGroupedN * atoms_per_row);
        *atom_of(
            activation_tile[tile], within / atoms_per_row,
            within % atoms_per_row) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int row = static_cast<int>(threadIdx.x); row < kGroupedN;
         row += kDecodeCtaThreads) {
        stage_scale_quad(
            activation_scale_shared, row, 0x7f7f7f7fu);
    }
}

__device__ __forceinline__ void stage_grouped_gate_up_activation(
    mixed_operand_tile (&activation_tile)[kGroupedRoundGroups],
    mixed_scale_tile &activation_scale_shared,
    const Scratch &scratch,
    const int assignment_begin,
    const int rows,
    const int group_base
) {
    const int thread = static_cast<int>(threadIdx.x);
    if (thread < 32) return;
    const int worker = thread - 32;
    constexpr int workers = kDecodeCtaThreads - 32;
    for (int index = worker; index < rows * kGroupedRoundGroups;
         index += workers) {
        const int row = index / kGroupedRoundGroups;
        const int group = index % kGroupedRoundGroups;
        const int assignment = assignment_begin + row;
        const int token = scratch.assignment_tokens[assignment];
        stage_activation_row(
            activation_tile[group], row,
            scratch.latent_mxfp8
                + static_cast<long long>(token) * kLatentSize
                + (group_base + group) * kMmaK);
    }
    for (int row = worker; row < rows; row += workers) {
        const int assignment = assignment_begin + row;
        const int token = scratch.assignment_tokens[assignment];
        stage_scale_quad(
            activation_scale_shared, row,
            *reinterpret_cast<const std::uint32_t *>(
                scratch.latent_scale
                + static_cast<long long>(token) * kLatentGroups
                + group_base));
    }
}

__device__ __forceinline__ void stage_grouped_down_activation(
    mixed_operand_tile (&activation_tile)[kGroupedRoundGroups],
    mixed_scale_tile &activation_scale_shared,
    const Scratch &scratch,
    const int assignment_begin,
    const int rows,
    const int group_base
) {
    const int thread = static_cast<int>(threadIdx.x);
    if (thread < 32) return;
    const int worker = thread - 32;
    constexpr int workers = kDecodeCtaThreads - 32;
    for (int index = worker; index < rows * kGroupedRoundGroups;
         index += workers) {
        const int row = index / kGroupedRoundGroups;
        const int group = index % kGroupedRoundGroups;
        const int assignment = assignment_begin + row;
        stage_activation_row(
            activation_tile[group], row,
            scratch.situ_mxfp8
                + static_cast<long long>(assignment)
                      * kRoutedIntermediateSizePerRank
                + (group_base + group) * kMmaK);
    }
    for (int row = worker; row < rows; row += workers) {
        const int assignment = assignment_begin + row;
        stage_scale_quad(
            activation_scale_shared, row,
            *reinterpret_cast<const std::uint32_t *>(
                scratch.situ_scale
                + static_cast<long long>(assignment) * kSituGroups
                + group_base));
    }
}

template<int GROUP_SIZE>
__device__ __forceinline__ void stage_grouped_gate_up_weights(
    mixed_operand_tile
        (&weight_tile)[2][GROUP_SIZE][2][kGroupedRoundGroups],
    mixed_scale_tile
        (&weight_scale)[2][GROUP_SIZE][2][kGroupedScaleTiles],
    const int buffer,
    const std::uint8_t *__restrict__ expert_w1_packed,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_packed,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const int expert,
    const int tile_start,
    const int tile_count,
    const int group_base
) {
    static_assert(GROUP_SIZE >= 1 && GROUP_SIZE <= 2);
    const int thread = static_cast<int>(threadIdx.x);
    if (thread < 32) return;
    const int worker = thread - 32;
    constexpr int workers = kDecodeCtaThreads - 32;
    const int rows = tile_count * 2 * kMmaN;
    for (int index = worker; index < rows; index += workers) {
        const int tile = index / (2 * kMmaN);
        const int within = index % (2 * kMmaN);
        const int half = within / kMmaN;
        const int row = within % kMmaN;
        const int output_row = (tile_start + tile) * kMmaN + row;
        const std::uint8_t *const packed =
            half == 0 ? expert_w1_packed : expert_w3_packed;
        const std::uint8_t *const scales =
            half == 0 ? expert_w1_scale : expert_w3_scale;
        const long long weight_index =
            static_cast<long long>(expert) * kExpertW1W3PackedRows
            + output_row;
        const std::uint8_t *const packed_row =
            packed + weight_index * kExpertW1W3PackedColumns;
        #pragma unroll
        for (int group = 0; group < kGroupedRoundGroups; ++group) {
            stage_weight_row(
                weight_tile[buffer][tile][half][group], row,
                *reinterpret_cast<const uint4 *>(
                    packed_row
                    + (group_base + group) * (kMmaK / 2)));
        }
        stage_scale_quad(
            weight_scale[buffer][tile][half][0], row,
            *reinterpret_cast<const std::uint32_t *>(
                scales + weight_index * kExpertW1W3ScaleColumns
                + group_base));
    }
}

__device__ __forceinline__ void stage_grouped_down_weights(
    mixed_operand_tile
        (&weight_tile)[2][kGroupedDownWidth][kGroupedRoundGroups],
    mixed_scale_tile
        (&weight_scale)[2][kGroupedDownWidth][kGroupedScaleTiles],
    const int buffer,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const int expert,
    const int tile_start,
    const int tile_count,
    const int group_base
) {
    const int thread = static_cast<int>(threadIdx.x);
    if (thread < 32) return;
    const int worker = thread - 32;
    constexpr int workers = kDecodeCtaThreads - 32;
    const int rows = tile_count * kMmaN;
    for (int index = worker; index < rows; index += workers) {
        const int tile = index / kMmaN;
        const int row = index % kMmaN;
        const int output_row = (tile_start + tile) * kMmaN + row;
        const long long weight_index =
            static_cast<long long>(expert) * kExpertW2PackedRows
            + output_row;
        const std::uint8_t *const packed_row =
            expert_w2_packed + weight_index * kExpertW2PackedColumns;
        #pragma unroll
        for (int group = 0; group < kGroupedRoundGroups; ++group) {
            stage_weight_row(
                weight_tile[buffer][tile][group], row,
                *reinterpret_cast<const uint4 *>(
                    packed_row
                    + (group_base + group) * (kMmaK / 2)));
        }
        stage_scale_quad(
            weight_scale[buffer][tile][0], row,
            *reinterpret_cast<const std::uint32_t *>(
                expert_w2_scale
                + weight_index * kExpertW2ScaleColumns + group_base));
    }
}

__device__ __forceinline__ void store_grouped_accumulator(
    const grouped_accumulator_tile &accumulator,
    grouped_result_tile &destination
) {
    using namespace kittens;
    if (warpgroup::groupid() == 0) {
        rt_fl<kGroupedM / 4, kGroupedPhysicalN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(destination, result);
    }
}

__device__ __forceinline__ void quantize_grouped_situ(
    const grouped_result_tile &gate,
    const grouped_result_tile &up,
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

__device__ __forceinline__ void accumulate_grouped_down_fixed(
    const grouped_result_tile (&result)[kGroupedDownWidth],
    const Scratch &scratch,
    const int assignment_begin,
    const int batch_rows,
    const int tile_start,
    const int tile_count
) {
    const int thread = static_cast<int>(threadIdx.x);
    const int columns = tile_count * kMmaN;
    for (int index = thread; index < batch_rows * columns;
         index += kDecodeCtaThreads) {
        const int row = index / columns;
        const int within_row = index % columns;
        const int tile = within_row / kMmaN;
        const int column = within_row % kMmaN;
        const int assignment = assignment_begin + row;
        const int token = scratch.assignment_tokens[assignment];
        const int output_base = (tile_start + tile) * kMmaN;
        const float route_weight =
            decode_route_weight(scratch, assignment);
        const long long contribution = __float2ll_rn(
            result[tile][{column, row}]
            * route_weight * kRoutedAccumulatorScale);
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                &scratch.routed_accumulator_fixed[
                    static_cast<long long>(token) * kLatentSize
                    + output_base + column]),
            static_cast<unsigned long long>(contribution));
    }
}

template<int GROUP_SIZE>
static __device__ void grouped_gate_up_unit(
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
    const int output_group,
    const PhaseClocks clocks
) {
    static_assert(GROUP_SIZE >= 1 && GROUP_SIZE <= 2);
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kGroupedRoundGroups] =
        staging.allocate<mixed_operand_tile, kGroupedRoundGroups>();
    mixed_operand_tile
        (&weight_tile)[2][GROUP_SIZE][2][kGroupedRoundGroups] =
            staging.allocate<
                mixed_operand_tile, 2, GROUP_SIZE, 2,
                kGroupedRoundGroups>();
    mixed_scale_tile (&activation_scale_shared) =
        staging.allocate<mixed_scale_tile>();
    mixed_scale_tile
        (&weight_scale)[2][GROUP_SIZE][2][kGroupedScaleTiles] =
            staging.allocate<
                mixed_scale_tile, 2, GROUP_SIZE, 2,
                kGroupedScaleTiles>();

    tma_swizzle_allocator results(shared_raw);
    grouped_result_tile (&result)[GROUP_SIZE][2] =
        results.allocate<grouped_result_tile, GROUP_SIZE, 2>();

    __shared__ semaphore grouped_gate_up_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(grouped_gate_up_done, 0, 1);

    const int tile_start = output_group * GROUP_SIZE;
    const int tile_count = min(GROUP_SIZE, kGateUpTiles - tile_start);
    constexpr int scale_column_base =
        2 * GROUP_SIZE * kGroupedPhysicalN;
    const auto accumulator = [&](const int tile, const int half) {
        return tensor_pool.allocate<grouped_accumulator_tile>(
            (tile * 2 + half) * kGroupedPhysicalN);
    };
    const auto scale_slot = [&](const int slot) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            scale_column_base + slot * kRoutedScaleColumns);
    };

    int compute_phase = 0;
    for (int assignment_offset = 0; assignment_offset < batch_rows;
         assignment_offset += kGroupedN) {
        const int batch_begin = assignment_begin + assignment_offset;
        const int rows = min(kGroupedN, batch_rows - assignment_offset);
        clear_grouped_activation(activation_tile, activation_scale_shared);
        stage_grouped_gate_up_weights<GROUP_SIZE>(
            weight_tile, weight_scale, 0, expert_w1_packed, expert_w1_scale,
            expert_w3_packed, expert_w3_scale, expert, tile_start, tile_count,
            0);
        __syncthreads();

        unsigned long long mark = clocks.now();
        for (int round = 0; round < kGroupedGateUpRounds; ++round) {
            const int group_base = round * kGroupedRoundGroups;
            stage_grouped_gate_up_activation(
                activation_tile, activation_scale_shared, scratch, batch_begin,
                rows, group_base);
            asm volatile(
                "fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();
            mark = clocks.lap(kClockRoutedGateUpStage, mark);

            const int buffer = round & 1;
            if (warpid() == 0) {
                if (laneid() == 0) {
                    for (int tile = 0; tile < tile_count; ++tile) {
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            auto staged_scale =
                                scale_slot(tile * 2 + half);
                            load_mxnv_scale_async(
                                staged_scale,
                                weight_scale[buffer][tile][half][0]);
                        }
                    }
                    auto staged_activation_scale =
                        scale_slot(2 * GROUP_SIZE);
                    load_mxnv_scale_async(
                        staged_activation_scale,
                        activation_scale_shared);
                }
                tensor_store_wait();
                if (laneid() == 0) {
                    for (int tile = 0; tile < tile_count; ++tile) {
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            #pragma unroll
                            for (int group = 0;
                                 group < kGroupedRoundGroups; ++group) {
                                grouped_batch_mixed_mma(
                                    accumulator(tile, half),
                                    weight_tile[buffer][tile][half][group],
                                    activation_tile[group],
                                    scale_slot(tile * 2 + half),
                                    scale_slot(2 * GROUP_SIZE),
                                    group,
                                    round != 0 || group != 0);
                            }
                        }
                    }
                    detail::tcgen05::commit<1>(grouped_gate_up_done);
                }
            }
            const int next_buffer = (round + 1) & 1;
            if (round + 1 < kGroupedGateUpRounds) {
                stage_grouped_gate_up_weights<GROUP_SIZE>(
                    weight_tile, weight_scale, next_buffer,
                    expert_w1_packed, expert_w1_scale,
                    expert_w3_packed, expert_w3_scale, expert, tile_start,
                    tile_count, group_base + kGroupedRoundGroups);
            }
            wait(grouped_gate_up_done, compute_phase);
            __syncthreads();
            mark = clocks.lap(kClockRoutedGateUpMma, mark);
            compute_phase ^= 1;
        }

        for (int tile = 0; tile < tile_count; ++tile) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                store_grouped_accumulator(
                    accumulator(tile, half), result[tile][half]);
            }
        }
        __syncthreads();
        for (int tile = 0; tile < tile_count; ++tile) {
            quantize_grouped_situ(
                result[tile][0], result[tile][1], scratch, batch_begin, rows,
                (tile_start + tile) * kMmaN);
        }
        __syncthreads();
    }
}

static __device__ void grouped_down_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const Scratch &scratch,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const int output_group,
    const PhaseClocks clocks
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kGroupedRoundGroups] =
        staging.allocate<mixed_operand_tile, kGroupedRoundGroups>();
    mixed_operand_tile
        (&weight_tile)[2][kGroupedDownWidth][kGroupedRoundGroups] =
            staging.allocate<
                mixed_operand_tile, 2, kGroupedDownWidth,
                kGroupedRoundGroups>();
    mixed_scale_tile (&activation_scale_shared) =
        staging.allocate<mixed_scale_tile>();
    mixed_scale_tile
        (&weight_scale)[2][kGroupedDownWidth][kGroupedScaleTiles] =
            staging.allocate<
                mixed_scale_tile, 2, kGroupedDownWidth,
                kGroupedScaleTiles>();

    tma_swizzle_allocator results(shared_raw);
    grouped_result_tile (&result)[kGroupedDownWidth] =
        results.allocate<grouped_result_tile, kGroupedDownWidth>();

    __shared__ semaphore grouped_down_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(grouped_down_done, 0, 1);

    const int tile_start = output_group * kGroupedDownWidth;
    const int tile_count = min(kGroupedDownWidth, kDownTiles - tile_start);
    constexpr int scale_column_base =
        kGroupedDownWidth * kGroupedPhysicalN;
    const auto accumulator = [&](const int tile) {
        return tensor_pool.allocate<grouped_accumulator_tile>(
            tile * kGroupedPhysicalN);
    };
    const auto scale_slot = [&](const int slot) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            scale_column_base + slot * kRoutedScaleColumns);
    };

    int compute_phase = 0;
    for (int assignment_offset = 0; assignment_offset < batch_rows;
         assignment_offset += kGroupedN) {
        const int batch_begin = assignment_begin + assignment_offset;
        const int rows = min(kGroupedN, batch_rows - assignment_offset);
        clear_grouped_activation(activation_tile, activation_scale_shared);
        stage_grouped_down_weights(
            weight_tile, weight_scale, 0, expert_w2_packed, expert_w2_scale,
            expert, tile_start, tile_count, 0);
        __syncthreads();

        unsigned long long mark = clocks.now();
        for (int round = 0; round < kGroupedDownRounds; ++round) {
            const int group_base = round * kGroupedRoundGroups;
            stage_grouped_down_activation(
                activation_tile, activation_scale_shared, scratch, batch_begin,
                rows, group_base);
            asm volatile(
                "fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();
            mark = clocks.lap(kClockRoutedDownStage, mark);

            const int buffer = round & 1;
            if (warpid() == 0) {
                if (laneid() == 0) {
                    for (int tile = 0; tile < tile_count; ++tile) {
                        auto staged_scale = scale_slot(tile);
                        load_mxnv_scale_async(
                            staged_scale,
                            weight_scale[buffer][tile][0]);
                    }
                    auto staged_activation_scale =
                        scale_slot(kGroupedDownWidth);
                    load_mxnv_scale_async(
                        staged_activation_scale,
                        activation_scale_shared);
                }
                tensor_store_wait();
                if (laneid() == 0) {
                    for (int tile = 0; tile < tile_count; ++tile) {
                        #pragma unroll
                        for (int group = 0;
                             group < kGroupedRoundGroups; ++group) {
                            grouped_batch_mixed_mma(
                                accumulator(tile),
                                weight_tile[buffer][tile][group],
                                activation_tile[group],
                                scale_slot(tile),
                                scale_slot(kGroupedDownWidth),
                                group,
                                round != 0 || group != 0);
                        }
                    }
                    detail::tcgen05::commit<1>(grouped_down_done);
                }
            }
            const int next_buffer = (round + 1) & 1;
            if (round + 1 < kGroupedDownRounds) {
                stage_grouped_down_weights(
                    weight_tile, weight_scale, next_buffer,
                    expert_w2_packed, expert_w2_scale, expert, tile_start,
                    tile_count, group_base + kGroupedRoundGroups);
            }
            wait(grouped_down_done, compute_phase);
            __syncthreads();
            mark = clocks.lap(kClockRoutedDownMma, mark);
            compute_phase ^= 1;
        }

        for (int tile = 0; tile < tile_count; ++tile) {
            store_grouped_accumulator(accumulator(tile), result[tile]);
        }
        __syncthreads();
        accumulate_grouped_down_fixed(
            result, scratch, batch_begin, rows, tile_start, tile_count);
        __syncthreads();
    }
}

// Grouped down needs 16 KiB of shared activation, 128 KiB of double-buffered
// weights, and 5 KiB of scales.
inline constexpr int kGroupedDownPersistentSharedBytes = 160 * 1024;
// Width two is the larger benchmark gate/up candidate: 16 KiB activation,
// 128 KiB of double-buffered gate/up weights, and 5 KiB of scales.
inline constexpr int kGroupedGateUpCandidateSharedBytes = 160 * 1024;

static_assert(
    kGroupedDownPersistentSharedBytes < kittens::MAX_SHARED_MEMORY);
static_assert(
    kGroupedGateUpCandidateSharedBytes < kittens::MAX_SHARED_MEMORY);

}  // namespace grouped_pipeline
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
