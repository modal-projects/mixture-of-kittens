#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace kimi_k3_decode {
namespace expert_mxfp4 {

inline constexpr int kMmaM = 128;
inline constexpr int kMmaN = 128;
inline constexpr int kMmaK = 32;
inline constexpr int kScaleRows = 32;
inline constexpr int kScaleColumns = 16;

using mixed_operand_tile = kittens::st_fp8e4m3<kMmaM, kMmaK>;
using mixed_scale_tile =
    kittens::st_fp8e8m0<kScaleRows, kScaleColumns, false>;
using mixed_accumulator_tile = kittens::tt_fl<kMmaM, kMmaN>;
using mixed_result_tile = kittens::st_fl<kMmaM, kMmaN>;

// PTX ISA 9.1, "tcgen05.mma instruction descriptor":
//   a_format [7:10) = 0 (E4M3)
//   b_format [10:13) = 5 (E2M1)
//   n_dim [17:23) = N / 8
//   scale_format [23] = 1 (UE8M0)
//   m_dim [24:29) = M / 16
//   a_sf_id [29:31), b_sf_id [4:6)
//   k_size [31] = 0 (dense MXF8F6F4 K=32)
template<int SCALE_FACTOR_ID>
__device__ __forceinline__ constexpr std::uint32_t
mixed_instruction_descriptor() {
    static_assert(SCALE_FACTOR_ID >= 0 && SCALE_FACTOR_ID < 4);
    return (static_cast<std::uint32_t>(SCALE_FACTOR_ID) << 4)
         | (0u << 7)
         | (5u << 10)
         | (static_cast<std::uint32_t>(kMmaN / 8) << 17)
         | (1u << 23)
         | (static_cast<std::uint32_t>(kMmaM / 16) << 24)
         | (static_cast<std::uint32_t>(SCALE_FACTOR_ID) << 29);
}

static_assert(mixed_instruction_descriptor<0>() == 0x08a01400u);

/// CUTLASS's SM103 `Sm103BlockScaledBasicChunk<32>::SfKMajorAtom`:
/// shape ((8,4,4),(32,4)), stride ((16,128,4),(0,1)).
__host__ __device__ __forceinline__ constexpr int
scale_factor_1x_offset(const int row, const int k_group) {
    return (row % 8) * 16
         + ((row / 8) % 4) * 128
         + (row / 32) * 4
         + k_group;
}

template<int SCALE_FACTOR_ID, bool ACCUMULATE>
__device__ __forceinline__ void mixed_mma(
    const mixed_accumulator_tile &destination,
    const mixed_operand_tile &a,
    const mixed_operand_tile &b,
    const kittens::full_tt_fp8e8m0<16> &scale_a,
    const kittens::full_tt_fp8e8m0<16> &scale_b
) {
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N> a_desc(a);
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N> b_desc(b);
    constexpr std::uint32_t instruction =
        mixed_instruction_descriptor<SCALE_FACTOR_ID>();
    // The operands are populated by ordinary shared-memory stores. Publish
    // those writes to the asynchronous tcgen05 proxy before every issue; a CTA
    // barrier alone does not establish this cross-proxy ordering.
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
          "l"(a_desc.base_desc),
          "l"(b_desc.base_desc),
          "r"(instruction),
          "r"(scale_a.addr),
          "r"(scale_b.addr),
          "n"(ACCUMULATE ? 1 : 0)
    );
}

__device__ __forceinline__ std::uint8_t quantize_e4m3(
    const float value,
    const float reciprocal
) {
    std::uint16_t pair;
    const float scaled = value * reciprocal;
    asm volatile(
        "{cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;}"
        : "=h"(pair)
        : "f"(scaled)
    );
    return static_cast<std::uint8_t>(pair);
}

// OCP MX v1.0 and PTX ISA 9.3 both define an E8M0 scale as 2^(byte - 127) with
// byte 255 reserved for NaN, so byte 0 is the exact minimum scale 2^-127 and
// byte 254 the maximum.
inline constexpr unsigned int kMinE8M0ScaleByte = 0u;
inline constexpr unsigned int kUnitE8M0ScaleByte = 0x7fu;
inline constexpr unsigned int kMaxE8M0ScaleByte = 254u;

// E4M3 tops out at 448 = 1.75 * 2^8, so a significand above 1.75 needs one more
// binade of headroom than one at or below it.
inline constexpr unsigned int kOneAndThreeQuartersMantissa = 0x600000u;

/// Return the E8M0 byte whose scale keeps every E4M3 magnitude at most 448.
///
/// The exponent is derived from the input's own bits rather than from
/// `absolute_max * (1/448)`. That product underflows toward zero for small
/// blocks, so it needed a floor, and any floor pins every block beneath it to
/// one coarse scale and flushes those activations to zero instead of using the
/// scales E8M0 actually has. Working on the exponent keeps the full range
/// reachable and also avoids the rounding of the 1/448 multiply near a binade
/// boundary.
__device__ __forceinline__ std::uint8_t select_e8m0_scale(
    const float absolute_max
) {
    if (absolute_max == 0.0f) {
        return static_cast<std::uint8_t>(kUnitE8M0ScaleByte);
    }
    const unsigned int bits = __float_as_uint(absolute_max);
    const unsigned int exponent_field = (bits >> 23) & 0xffu;
    // Subnormal magnitudes are below 2^-126, and 2^-126 / 448 is already below
    // 2^-134, so the minimum scale is the only available answer.
    if (exponent_field == 0u) {
        return static_cast<std::uint8_t>(kMinE8M0ScaleByte);
    }
    const int exponent = static_cast<int>(exponent_field) - 127;
    const unsigned int mantissa = bits & 0x7fffffu;
    const int scale_exponent =
        (mantissa <= kOneAndThreeQuartersMantissa) ? exponent - 8
                                                  : exponent - 7;
    return static_cast<std::uint8_t>(
        min(max(scale_exponent + 127, static_cast<int>(kMinE8M0ScaleByte)),
            static_cast<int>(kMaxE8M0ScaleByte)));
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void mixed_mma_probe_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const std::uint8_t *__restrict__ b_packed,
    const std::uint8_t *__restrict__ b_scale,
    float *__restrict__ output,
    const int rows
) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];
    tma_swizzle_allocator shared_allocator(shared_raw);
    mixed_operand_tile (&a_tile) =
        shared_allocator.allocate<mixed_operand_tile>();
    mixed_operand_tile (&b_tile) =
        shared_allocator.allocate<mixed_operand_tile>();
    mixed_scale_tile (&scale_a_shared) =
        shared_allocator.allocate<mixed_scale_tile>();
    mixed_scale_tile (&scale_b_shared) =
        shared_allocator.allocate<mixed_scale_tile>();
    mixed_result_tile (&result_shared) =
        shared_allocator.allocate<mixed_result_tile>();

    const int thread = static_cast<int>(threadIdx.x);
    for (int row = thread; row < kMmaM; row += kDecodeCtaThreads) {
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const float value =
                row < rows
                    ? __bfloat162float(a[row * kMmaK + k])
                    : 0.0f;
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale = select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            *reinterpret_cast<std::uint8_t *>(&a_tile[{row, k}]) =
                quantize_e4m3(values[k], reciprocal);
        }
    }

    for (int index = thread; index < kMmaN * kMmaK;
         index += kDecodeCtaThreads) {
        const int row = index / kMmaK;
        const int shared_column = index % kMmaK;
        const int column_in_16b_atom = shared_column % 16;
        std::uint8_t value = 0;
        if (column_in_16b_atom < 8) {
            // CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B places sixteen packed U4
            // values (eight bytes) at the front of each 16-byte SMEM atom.
            const int packed_column =
                (shared_column / 16) * 8 + column_in_16b_atom;
            value = b_packed[row * (kMmaK / 2) + packed_column];
        }
        *reinterpret_cast<std::uint8_t *>(&b_tile[{row, shared_column}]) =
            value;
    }

    for (int index = thread; index < kScaleRows * kScaleColumns;
         index += kDecodeCtaThreads) {
        reinterpret_cast<std::uint8_t *>(scale_a_shared.data)[index] = 0x7fu;
        reinterpret_cast<std::uint8_t *>(scale_b_shared.data)[index] = 0x7fu;
    }
    __syncthreads();

    for (int row = thread; row < kMmaM; row += kDecodeCtaThreads) {
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            if (row < rows) {
                absolute_max = fmaxf(
                    absolute_max,
                    fabsf(__bfloat162float(a[row * kMmaK + k])));
            }
        }
        reinterpret_cast<std::uint8_t *>(scale_a_shared.data)
            [scale_factor_1x_offset(row, 0)] =
                select_e8m0_scale(absolute_max);
        reinterpret_cast<std::uint8_t *>(scale_b_shared.data)
            [scale_factor_1x_offset(row, 0)] = b_scale[row];
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    __shared__ semaphore compute_done;
    if (thread == 0) init_semaphore(compute_done, 0, 1);
    __syncthreads();

    tensor_allocator<1, 1> tensor_pool{};
    mixed_accumulator_tile accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    auto scale_a =
        tensor_pool.allocate<full_tt_fp8e8m0<16>>(256);
    auto scale_b =
        tensor_pool.allocate<full_tt_fp8e8m0<16>>(260);

    if (warpid() == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        load_mxnv_scale_async(scale_a, scale_a_shared);
        load_mxnv_scale_async(scale_b, scale_b_shared);
        tensor_store_wait();
    }
    __syncthreads();

    if (thread == 0) {
        mixed_mma<0, false>(accumulator, a_tile, b_tile, scale_a, scale_b);
        detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);
    if (warpgroup::groupid() == 0) {
        rt_fl<kMmaM / 4, kMmaN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(result_shared, result);
    }
    __syncthreads();

    for (int index = thread; index < rows * kMmaN;
         index += kDecodeCtaThreads) {
        output[index] = result_shared[{index / kMmaN, index % kMmaN}];
    }
}

inline constexpr int kGateUpTiles =
    kRoutedIntermediateSizePerRank / kMmaN;
inline constexpr int kDownTiles = kLatentSize / kMmaN;
inline constexpr int kLatentGroups = kLatentSize / kMmaK;
inline constexpr int kSituGroups =
    kRoutedIntermediateSizePerRank / kMmaK;

static_assert(kGateUpTiles == 3);
static_assert(kDownTiles == 28);
static_assert(kLatentGroups == 112);
static_assert(kSituGroups == 12);

__device__ __forceinline__ float decode_route_weight(
    const Scratch &scratch,
    const int assignment
) {
    const int token = scratch.assignment_tokens[assignment];
    const int slot = scratch.assignment_slots[assignment];
    return scratch.expert_weights[token * kTopK + slot];
}

__device__ __forceinline__ void load_quantized_activation_tile(
    mixed_operand_tile &tile,
    mixed_scale_tile &scale_shared,
    const Scratch &scratch,
    const int assignment_begin,
    const int batch_rows,
    const int k_group,
    const bool situ
) {
    const int thread = static_cast<int>(threadIdx.x);
    const int width = situ ? kRoutedIntermediateSizePerRank : kLatentSize;
    const std::uint8_t *const values =
        situ ? scratch.situ_mxfp8 : scratch.latent_mxfp8;
    const std::uint8_t *const scales =
        situ ? scratch.situ_scale : scratch.latent_scale;
    const int groups = width / kMmaK;

    for (int index = thread; index < kMmaM * kMmaK;
         index += kDecodeCtaThreads) {
        const int row = index / kMmaK;
        const int column = index % kMmaK;
        std::uint8_t value = 0;
        if (row < batch_rows) {
            const int assignment = assignment_begin + row;
            const int source_row =
                situ ? assignment : scratch.assignment_tokens[assignment];
            value = values[
                static_cast<long long>(source_row) * width
                + k_group * kMmaK + column];
        }
        *reinterpret_cast<std::uint8_t *>(&tile[{row, column}]) = value;
    }

    for (int index = thread; index < kScaleRows * kScaleColumns;
         index += kDecodeCtaThreads) {
        reinterpret_cast<std::uint8_t *>(scale_shared.data)[index] = 0x7fu;
    }
    __syncthreads();
    for (int row = thread; row < kMmaM; row += kDecodeCtaThreads) {
        std::uint8_t scale = 0x7fu;
        if (row < batch_rows) {
            const int assignment = assignment_begin + row;
            const int source_row =
                situ ? assignment : scratch.assignment_tokens[assignment];
            scale = scales[
                static_cast<long long>(source_row) * groups + k_group];
        }
        reinterpret_cast<std::uint8_t *>(scale_shared.data)
            [scale_factor_1x_offset(row, 0)] = scale;
    }
}

__device__ __forceinline__ void load_mxfp4_weight_tile(
    mixed_operand_tile &tile,
    mixed_scale_tile &scale_shared,
    const std::uint8_t *__restrict__ packed,
    const std::uint8_t *__restrict__ scales,
    const int expert,
    const int rows,
    const int packed_columns,
    const int scale_columns,
    const int output_base,
    const int k_group
) {
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < kMmaN * kMmaK;
         index += kDecodeCtaThreads) {
        const int row = index / kMmaK;
        const int shared_column = index % kMmaK;
        const int column_in_16b_atom = shared_column % 16;
        std::uint8_t value = 0;
        if (column_in_16b_atom < 8) {
            const int packed_column =
                k_group * (kMmaK / 2)
                + (shared_column / 16) * 8 + column_in_16b_atom;
            const long long row_index =
                static_cast<long long>(expert) * rows + output_base + row;
            value = packed[row_index * packed_columns + packed_column];
        }
        *reinterpret_cast<std::uint8_t *>(
            &tile[{row, shared_column}]) = value;
    }

    for (int index = thread; index < kScaleRows * kScaleColumns;
         index += kDecodeCtaThreads) {
        reinterpret_cast<std::uint8_t *>(scale_shared.data)[index] = 0x7fu;
    }
    __syncthreads();
    for (int row = thread; row < kMmaN; row += kDecodeCtaThreads) {
        const long long row_index =
            static_cast<long long>(expert) * rows + output_base + row;
        reinterpret_cast<std::uint8_t *>(scale_shared.data)
            [scale_factor_1x_offset(row, 0)] =
                scales[row_index * scale_columns + k_group];
    }
}

__device__ __forceinline__ void store_accumulator(
    const mixed_accumulator_tile &accumulator,
    mixed_result_tile &destination
) {
    using namespace kittens;
    if (warpgroup::groupid() == 0) {
        rt_fl<kMmaM / 4, kMmaN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(destination, result);
    }
    __syncthreads();
}

/// Quantize this worker's stride of the routed latent to MXFP8.
///
/// The private stage runs one CTA and passes `worker = 0, workers = 1`; the
/// persistent kernel spreads the same groups over its whole resident grid.
__device__ __forceinline__ void quantize_latent_rows(
    const __nv_bfloat16 *__restrict__ latent_x,
    const Scratch &scratch,
    const int active_tokens,
    const int worker,
    const int workers
) {
    const int thread = static_cast<int>(threadIdx.x);
    const int total_groups = active_tokens * kLatentGroups;
    for (int index = worker * kDecodeCtaThreads + thread;
         index < total_groups;
         index += workers * kDecodeCtaThreads) {
        const int token = index / kLatentGroups;
        const int group = index % kLatentGroups;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const float value = __bfloat162float(
                latent_x[static_cast<long long>(token) * kLatentSize
                         + group * kMmaK + k]);
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale = select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        scratch.latent_scale[
            static_cast<long long>(token) * kLatentGroups + group] = scale;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            scratch.latent_mxfp8[
                static_cast<long long>(token) * kLatentSize
                + group * kMmaK + k] =
                    quantize_e4m3(values[k], reciprocal);
        }
    }
}

__device__ __forceinline__ void quantize_situ_tile(
    const mixed_result_tile &gate,
    const mixed_result_tile &up,
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
            const float gate_value = gate[{row, column}];
            const float up_value = up[{row, column}];
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
        const int global_group =
            output_base / kMmaK + local_group;
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

__device__ __forceinline__ void accumulate_down_tile(
    const mixed_result_tile &result,
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
                result[{row, column}]
                    * decode_route_weight(scratch, assignment));
        }
    }
}

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

/// Shared bytes one gate/up unit stages, which is the widest of any K3 stage.
///
/// The allocator aligns each object independently to 1 KiB, so this counts the
/// padded sizes rather than the raw ones.
inline constexpr int kGateUpUnitSharedBytes =
    3 * static_cast<int>(sizeof(mixed_operand_tile))
    + 3 * 1024
    + 2 * static_cast<int>(sizeof(mixed_result_tile));

static_assert(sizeof(mixed_operand_tile) == 4096);
static_assert(sizeof(mixed_scale_tile) == 512);
static_assert(sizeof(mixed_result_tile) == 65536);
static_assert(kGateUpUnitSharedBytes == 146432);

/// Contract one expert batch's gate and up tiles and stage the SiTU result.
///
/// `tensor_pool` is owned by the caller because a CTA may allocate tensor
/// memory only once: the persistent kernel provisions one pool at entry and
/// hands it to every unit, and the private kernel provisions one of its own.
static __device__ void routed_gate_up_unit(
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
    const int output_tile,
    const PhaseClocks clocks
) {
    using namespace kittens;
    tma_swizzle_allocator allocator(shared_raw);
    mixed_operand_tile (&activation_tile) =
        allocator.allocate<mixed_operand_tile>();
    mixed_operand_tile (&first_weight_tile) =
        allocator.allocate<mixed_operand_tile>();
    mixed_operand_tile (&second_weight_tile) =
        allocator.allocate<mixed_operand_tile>();
    mixed_scale_tile (&activation_scale_shared) =
        allocator.allocate<mixed_scale_tile>();
    mixed_scale_tile (&first_scale_shared) =
        allocator.allocate<mixed_scale_tile>();
    mixed_scale_tile (&second_scale_shared) =
        allocator.allocate<mixed_scale_tile>();
    mixed_result_tile (&first_result_shared) =
        allocator.allocate<mixed_result_tile>();
    mixed_result_tile (&second_result_shared) =
        allocator.allocate<mixed_result_tile>();

    __shared__ semaphore gate_up_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(gate_up_done, 0, 1);
    __syncthreads();

    mixed_accumulator_tile first_accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    mixed_accumulator_tile second_accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(128);
    auto activation_scale = tensor_pool.allocate<full_tt_fp8e8m0<16>>(256);
    auto first_scale = tensor_pool.allocate<full_tt_fp8e8m0<16>>(260);
    auto second_scale = tensor_pool.allocate<full_tt_fp8e8m0<16>>(264);

    const int output_base = output_tile * kMmaN;
    int compute_phase = 0;
    unsigned long long mark = clocks.now();
    for (int k_group = 0; k_group < kLatentGroups; ++k_group) {
        load_quantized_activation_tile(
            activation_tile, activation_scale_shared, scratch,
            assignment_begin, batch_rows, k_group, false);
        load_mxfp4_weight_tile(
            first_weight_tile, first_scale_shared, expert_w1_packed,
            expert_w1_scale, expert, kExpertW1W3PackedRows,
            kExpertW1W3PackedColumns, kExpertW1W3ScaleColumns, output_base,
            k_group);
        load_mxfp4_weight_tile(
            second_weight_tile, second_scale_shared, expert_w3_packed,
            expert_w3_scale, expert, kExpertW1W3PackedRows,
            kExpertW1W3PackedColumns, kExpertW1W3ScaleColumns, output_base,
            k_group);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if (warpid() == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            load_mxnv_scale_async(activation_scale, activation_scale_shared);
            load_mxnv_scale_async(first_scale, first_scale_shared);
            load_mxnv_scale_async(second_scale, second_scale_shared);
            tensor_store_wait();
        }
        __syncthreads();
        mark = clocks.lap(kClockRoutedGateUpStage, mark);
        if (thread == 0) {
            if (k_group == 0) {
                mixed_mma<0, false>(
                    first_accumulator, activation_tile, first_weight_tile,
                    activation_scale, first_scale);
                mixed_mma<0, false>(
                    second_accumulator, activation_tile, second_weight_tile,
                    activation_scale, second_scale);
            } else {
                mixed_mma<0, true>(
                    first_accumulator, activation_tile, first_weight_tile,
                    activation_scale, first_scale);
                mixed_mma<0, true>(
                    second_accumulator, activation_tile, second_weight_tile,
                    activation_scale, second_scale);
            }
            detail::tcgen05::commit<1>(gate_up_done);
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
        batch_rows, output_base);
    __syncthreads();
}

/// Contract one expert batch's down tile and weight it into the accumulator.
static __device__ void routed_down_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const std::uint8_t *__restrict__ expert_w2_packed,
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
    tma_swizzle_allocator allocator(shared_raw);
    mixed_operand_tile (&activation_tile) =
        allocator.allocate<mixed_operand_tile>();
    mixed_operand_tile (&weight_tile) =
        allocator.allocate<mixed_operand_tile>();
    mixed_scale_tile (&activation_scale_shared) =
        allocator.allocate<mixed_scale_tile>();
    mixed_scale_tile (&weight_scale_shared) =
        allocator.allocate<mixed_scale_tile>();
    mixed_result_tile (&result_shared) =
        allocator.allocate<mixed_result_tile>();

    __shared__ semaphore down_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(down_done, 0, 1);
    __syncthreads();

    mixed_accumulator_tile accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    auto activation_scale = tensor_pool.allocate<full_tt_fp8e8m0<16>>(256);
    auto weight_scale = tensor_pool.allocate<full_tt_fp8e8m0<16>>(260);

    const int output_base = output_tile * kMmaN;
    int compute_phase = 0;
    unsigned long long mark = clocks.now();
    for (int k_group = 0; k_group < kSituGroups; ++k_group) {
        load_quantized_activation_tile(
            activation_tile, activation_scale_shared, scratch,
            assignment_begin, batch_rows, k_group, true);
        load_mxfp4_weight_tile(
            weight_tile, weight_scale_shared, expert_w2_packed,
            expert_w2_scale, expert, kExpertW2PackedRows,
            kExpertW2PackedColumns, kExpertW2ScaleColumns, output_base,
            k_group);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if (warpid() == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            load_mxnv_scale_async(activation_scale, activation_scale_shared);
            load_mxnv_scale_async(weight_scale, weight_scale_shared);
            tensor_store_wait();
        }
        __syncthreads();
        mark = clocks.lap(kClockRoutedDownStage, mark);
        if (thread == 0) {
            if (k_group == 0) {
                mixed_mma<0, false>(
                    accumulator, activation_tile, weight_tile,
                    activation_scale, weight_scale);
            } else {
                mixed_mma<0, true>(
                    accumulator, activation_tile, weight_tile,
                    activation_scale, weight_scale);
            }
            detail::tcgen05::commit<1>(down_done);
        }
        wait(down_done, compute_phase);
        __syncthreads();
        mark = clocks.lap(kClockRoutedDownMma, mark);
        compute_phase ^= 1;
    }

    store_accumulator(accumulator, result_shared);
    accumulate_down_tile(
        result_shared, scratch, assignment_begin, batch_rows, output_base,
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
                    shared_raw, tensor_pool, expert_w1_packed, expert_w1_scale,
                    expert_w3_packed, expert_w3_scale, scratch, expert,
                    assignment_begin, batch_rows, output_tile,
                    PhaseClocks{nullptr});
            }

            for (int output_tile = 0; output_tile < kDownTiles;
                 ++output_tile) {
                routed_down_unit(
                    shared_raw, tensor_pool, expert_w2_packed, expert_w2_scale,
                    scratch, expert, assignment_begin, batch_rows, output_tile,
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

// The shared allocator aligns every object independently to 1 KiB and the
// register-to-shared mapping can touch the following swizzle atom. Reserve the
// architecture-supported budget instead of under-counting either padding.
inline constexpr int kProbeSharedBytes = kittens::MAX_SHARED_MEMORY - 1024;

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
            reinterpret_cast<__nv_bfloat16 *>(routed_output.data_ptr()),
            reinterpret_cast<std::uint8_t *>(scratch.data_ptr()),
            active_tokens,
            static_cast<int>(latent_x.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __host__ at::Tensor mixed_mma_probe_entrypoint(
    const at::Tensor &a,
    const at::Tensor &b_packed,
    const at::Tensor &b_scale
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b_packed);
    CHECK_INPUT(b_scale);
    TORCH_CHECK(a.dim() == 2 && a.size(0) >= 1 && a.size(0) <= kMmaM
                    && a.size(1) == kMmaK
                    && a.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_mixed_mma_probe requires BF16 A [1..128, 32]");
    TORCH_CHECK(b_packed.dim() == 2 && b_packed.size(0) == kMmaN
                    && b_packed.size(1) == kMmaK / 2
                    && b_packed.scalar_type() == at::kByte,
                "MoK: _kimi_k3_mixed_mma_probe requires uint8 B [128, 16]");
    TORCH_CHECK(b_scale.dim() == 2 && b_scale.size(0) == kMmaN
                    && b_scale.size(1) == 1
                    && b_scale.scalar_type() == at::kByte,
                "MoK: _kimi_k3_mixed_mma_probe requires uint8 B scale [128, 1]");
    TORCH_CHECK(a.device() == b_packed.device()
                    && a.device() == b_scale.device(),
                "MoK: _kimi_k3_mixed_mma_probe requires one CUDA device");

    const c10::cuda::CUDAGuard device_guard(a.device());
    at::Tensor output =
        at::empty({a.size(0), kMmaN}, a.options().dtype(at::kFloat));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mixed_mma_probe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kProbeSharedBytes));
    mixed_mma_probe_kernel
        <<<1, kDecodeCtaThreads, kProbeSharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(a.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(b_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(b_scale.data_ptr()),
            reinterpret_cast<float *>(output.data_ptr()),
            static_cast<int>(a.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
