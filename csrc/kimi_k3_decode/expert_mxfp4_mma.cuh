#pragma once

// The mixed MXFP8-by-MXFP4 contraction the routed experts are built from.
//
// One `tcgen05.mma.kind::mxf8f6f4` issue with `scale_vec::1X` block scaling,
// the instruction descriptor that selects it, the CUTLASS scale-factor atom
// its shared tiles have to be laid out in, and the E4M3/E8M0 quantization the
// activations reach it through. The probe kernel and its entrypoint are here
// too: they exist to test this instruction on its own, against one 128x128x32
// tile, without any of the scheduling the routed units add.

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
__host__ __device__ __forceinline__ constexpr std::uint32_t
mixed_instruction_descriptor(const int scale_factor_id) {
    return (static_cast<std::uint32_t>(scale_factor_id) << 4)
         | (0u << 7)
         | (5u << 10)
         | (static_cast<std::uint32_t>(kMmaN / 8) << 17)
         | (1u << 23)
         | (static_cast<std::uint32_t>(kMmaM / 16) << 24)
         | (static_cast<std::uint32_t>(scale_factor_id) << 29);
}

static_assert(mixed_instruction_descriptor(0) == 0x08a01400u);

/// CUTLASS's SM103 `Sm103BlockScaledBasicChunk<32>::SfKMajorAtom`:
/// shape ((8,4,4),(32,4)), stride ((16,128,4),(0,1)).
///
/// The atom's second mode is `(32, 4)` with stride `(0, 1)`, so one 512-byte
/// scale tile carries K=128 -- four consecutive K=32 groups -- and the four
/// factors of a row sit in one aligned word. `scale_factor_id` in the MMA
/// descriptor is what picks the group out of the tile.
__host__ __device__ __forceinline__ constexpr int
scale_factor_1x_offset(const int row, const int k_group) {
    return (row % 8) * 16
         + ((row / 8) % 4) * 128
         + (row / 32) * 4
         + k_group;
}

/// K groups one scale tile carries, and the word holding all four for a row.
inline constexpr int kScaleGroupsPerTile = 4;

static_assert(kScaleRows * kScaleColumns == kMmaM * kScaleGroupsPerTile,
              "one scale tile is one byte per row per carried K group");

/// Issue one K=32 block-scaled contraction into `destination`.
///
/// `scale_factor_id` and `accumulate` are run-time arguments rather than
/// template parameters because a unit issues a whole round of K groups from
/// one unrolled body: the group picks its own quarter of the shared scale
/// tile, only the first issue of a unit clears the accumulator, and both fold
/// to immediates once the round is unrolled.
__device__ __forceinline__ void mixed_mma(
    const mixed_accumulator_tile &destination,
    const mixed_operand_tile &a,
    const mixed_operand_tile &b,
    const kittens::full_tt_fp8e8m0<16> &scale_a,
    const kittens::full_tt_fp8e8m0<16> &scale_b,
    const int scale_factor_id,
    const bool accumulate
) {
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N> a_desc(a);
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N> b_desc(b);
    const std::uint32_t instruction =
        mixed_instruction_descriptor(scale_factor_id);
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
          "r"(accumulate ? 1u : 0u)
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
        mixed_mma(accumulator, a_tile, b_tile, scale_a, scale_b, 0, false);
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

// The shared allocator aligns every object independently to 1 KiB and the
// register-to-shared mapping can touch the following swizzle atom. Reserve the
// architecture-supported budget instead of under-counting either padding.
inline constexpr int kProbeSharedBytes = kittens::MAX_SHARED_MEMORY - 1024;

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
