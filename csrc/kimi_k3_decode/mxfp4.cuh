#pragma once

#include "pyutils/torchutils.cuh"

#include "types.cuh"

#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace kimi_k3_decode {
namespace mxfp4 {

// OCP-style group-32 MXFP4: one E8M0 power-of-two scale byte per 32 values plus
// packed E2M1 pairs, with the even element of each pair in the low nibble.
inline constexpr int kGroupSize = 32;
inline constexpr int kBytesPerGroup = kGroupSize / 2;
inline constexpr unsigned int kUnitScaleByte = 0x7fu;
inline constexpr unsigned int kMinScaleByte = 1u;
inline constexpr unsigned int kMaxScaleByte = 254u;
inline constexpr int kPackThreadsPerBlock = 256;

// The E2M1 significand cannot exceed 1.5, so 6 = 1.5 * 2^2 is the largest
// magnitude a group can hold and this mantissa marks that significand.
inline constexpr unsigned int kOneAndAHalfMantissa = 0x400000u;

static_assert(kExpertW1W3ScaleColumns * kGroupSize == kExpertW1W3PackedColumns * 2,
              "prepared W1/W3 layout must hold one E8M0 byte per 32 packed values");
static_assert(kExpertW2ScaleColumns * kGroupSize == kExpertW2PackedColumns * 2,
              "prepared W2 layout must hold one E8M0 byte per 32 packed values");

/// Return the E8M0 byte whose scale keeps every E2M1 magnitude at most 6.
static __device__ __forceinline__ unsigned int select_scale_byte(const float absolute_max) {
    if (absolute_max == 0.0f) return kUnitScaleByte;

    const unsigned int bits = __float_as_uint(absolute_max);
    const unsigned int exponent_field = (bits >> 23) & 0xffu;
    if (exponent_field == 0u) return kMinScaleByte; // magnitudes below 2^-126

    const int exponent = static_cast<int>(exponent_field) - 127;
    const unsigned int mantissa = bits & 0x7fffffu;
    const int scale_exponent = (mantissa <= kOneAndAHalfMantissa) ? exponent - 2
                                                                 : exponent - 1;
    return min(max(scale_exponent + 127, static_cast<int>(kMinScaleByte)),
               static_cast<int>(kMaxScaleByte));
}

/// Build 2^-(scale_byte - 127) without a division, as in the MXFP8 quantizer.
static __device__ __forceinline__ float scale_reciprocal(const unsigned int scale_byte) {
    return __uint_as_float((kMaxScaleByte - scale_byte) << 23);
}

static __global__ void pack_kernel(
    const __nv_bfloat16 *__restrict__ weight,
    std::uint8_t *__restrict__ packed,
    std::uint8_t *__restrict__ scale,
    const long long num_rows,
    const int logical_k,
    const int padded_k
) {
    const int groups_per_row = padded_k / kGroupSize;
    const long long num_groups = num_rows * groups_per_row;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < num_groups; index += stride) {
        const long long row = index / groups_per_row;
        const int group = static_cast<int>(index - row * groups_per_row);
        const int k_base = group * kGroupSize;

        const __nv_bfloat16 *source = weight + row * static_cast<long long>(logical_k);
        float values[kGroupSize];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < kGroupSize; i++) {
            const int k = k_base + i;
            values[i] = k < logical_k ? __bfloat162float(source[k]) : 0.0f;
            absolute_max = fmaxf(absolute_max, fabsf(values[i]));
        }

        const unsigned int scale_byte = select_scale_byte(absolute_max);
        scale[row * groups_per_row + group] = static_cast<std::uint8_t>(scale_byte);

        std::uint8_t *destination =
            packed + row * static_cast<long long>(padded_k / 2) + k_base / 2;
        if (absolute_max == 0.0f) {
            // All-zero and fully padded groups store packed zero, never a
            // signed zero code point.
            #pragma unroll
            for (int i = 0; i < kBytesPerGroup; i++) destination[i] = 0;
            continue;
        }

        const float reciprocal = scale_reciprocal(scale_byte);
        #pragma unroll
        for (int i = 0; i < kBytesPerGroup; i++) {
            const float2 pair{values[i * 2] * reciprocal, values[i * 2 + 1] * reciprocal};
            destination[i] = static_cast<std::uint8_t>(
                __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest));
        }
    }
}

static __global__ void dequant_kernel(
    const std::uint8_t *__restrict__ packed,
    const std::uint8_t *__restrict__ scale,
    __nv_bfloat16 *__restrict__ weight,
    const long long num_rows,
    const int logical_k,
    const int padded_k
) {
    const int groups_per_row = padded_k / kGroupSize;
    const int logical_groups = logical_k / kGroupSize;
    const long long num_groups = num_rows * logical_groups;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < num_groups; index += stride) {
        const long long row = index / logical_groups;
        const int group = static_cast<int>(index - row * logical_groups);

        const unsigned int scale_byte = scale[row * groups_per_row + group];
        const float scale_value = __uint_as_float(scale_byte << 23);
        const std::uint8_t *source =
            packed + row * static_cast<long long>(padded_k / 2) + group * kBytesPerGroup;
        __nv_bfloat16 *destination =
            weight + row * static_cast<long long>(logical_k) + group * kGroupSize;

        #pragma unroll
        for (int i = 0; i < kBytesPerGroup; i++) {
            const __half2_raw decoded = __nv_cvt_fp4x2_to_halfraw2(
                static_cast<__nv_fp4x2_storage_t>(source[i]), __NV_E2M1);
            const float2 pair = __half22float2(*reinterpret_cast<const __half2 *>(&decoded));
            destination[i * 2] = __float2bfloat16(pair.x * scale_value);
            destination[i * 2 + 1] = __float2bfloat16(pair.y * scale_value);
        }
    }
}

static __host__ inline unsigned int grid_blocks(const long long num_groups) {
    const long long blocks =
        (num_groups + kPackThreadsPerBlock - 1) / kPackThreadsPerBlock;
    return static_cast<unsigned int>(std::max<long long>(blocks, 1));
}

static __host__ std::tuple<at::Tensor, at::Tensor> pack_entrypoint(
    const at::Tensor &weight,
    const std::int64_t padded_k
) {
    CHECK_INPUT(weight);
    TORCH_CHECK(weight.dim() == 3, "MoK: pack_kimi_k3_mxfp4 requires a [E, N, K] weight");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16,
                "MoK: pack_kimi_k3_mxfp4 requires a BF16 weight");
    const std::int64_t logical_k = weight.size(2);
    TORCH_CHECK(logical_k > 0 && logical_k % kGroupSize == 0,
                "MoK: pack_kimi_k3_mxfp4 requires a logical K that is a positive "
                "multiple of ", kGroupSize);
    TORCH_CHECK(padded_k % kGroupSize == 0 && padded_k >= logical_k,
                "MoK: pack_kimi_k3_mxfp4 requires a padded K that is a multiple of ",
                kGroupSize, " and at least the logical K");

    // Every launch below must target the weight's own device and that device's
    // current stream, whatever device happens to be current on entry.
    const c10::cuda::CUDAGuard device_guard(weight.device());

    at::Tensor packed = at::empty({weight.size(0), weight.size(1), padded_k / 2},
                                  weight.options().dtype(at::kByte));
    at::Tensor scale = at::empty({weight.size(0), weight.size(1), padded_k / kGroupSize},
                                 weight.options().dtype(at::kByte));

    const long long num_rows = weight.size(0) * weight.size(1);
    const long long num_groups = num_rows * (padded_k / kGroupSize);
    pack_kernel<<<grid_blocks(num_groups), kPackThreadsPerBlock, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr()),
        reinterpret_cast<std::uint8_t *>(packed.data_ptr()),
        reinterpret_cast<std::uint8_t *>(scale.data_ptr()),
        num_rows, static_cast<int>(logical_k), static_cast<int>(padded_k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {packed, scale};
}

static __host__ at::Tensor dequant_entrypoint(
    const at::Tensor &packed,
    const at::Tensor &scale,
    const std::int64_t logical_k
) {
    CHECK_INPUT(packed);
    CHECK_INPUT(scale);
    TORCH_CHECK(packed.dim() == 3 && scale.dim() == 3,
                "MoK: dequant_kimi_k3_mxfp4 requires [E, N, *] packed values and scales");
    TORCH_CHECK(packed.scalar_type() == at::kByte && scale.scalar_type() == at::kByte,
                "MoK: dequant_kimi_k3_mxfp4 requires uint8 packed values and scales");
    TORCH_CHECK(packed.device() == scale.device(),
                "MoK: dequant_kimi_k3_mxfp4 requires one device");
    TORCH_CHECK(packed.size(0) == scale.size(0) && packed.size(1) == scale.size(1),
                "MoK: dequant_kimi_k3_mxfp4 requires matching packed and scale rows");
    const std::int64_t padded_k = packed.size(2) * 2;
    TORCH_CHECK(padded_k % kGroupSize == 0
                    && scale.size(2) == padded_k / kGroupSize,
                "MoK: dequant_kimi_k3_mxfp4 requires one scale byte per 32 packed values");
    TORCH_CHECK(logical_k > 0 && logical_k % kGroupSize == 0 && logical_k <= padded_k,
                "MoK: dequant_kimi_k3_mxfp4 requires a logical K that is a multiple of ",
                kGroupSize, " and at most the padded K");

    const c10::cuda::CUDAGuard device_guard(packed.device());

    at::Tensor weight = at::empty({packed.size(0), packed.size(1), logical_k},
                                  packed.options().dtype(at::kBFloat16));

    const long long num_rows = packed.size(0) * packed.size(1);
    const long long num_groups = num_rows * (logical_k / kGroupSize);
    dequant_kernel<<<grid_blocks(num_groups), kPackThreadsPerBlock, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const std::uint8_t *>(packed.data_ptr()),
        reinterpret_cast<const std::uint8_t *>(scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(weight.data_ptr()),
        num_rows, static_cast<int>(logical_k), static_cast<int>(padded_k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return weight;
}

}  // namespace mxfp4
}  // namespace kimi_k3_decode
