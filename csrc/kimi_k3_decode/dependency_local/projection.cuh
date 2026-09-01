#pragma once

/// The projection unit's body: latent groups, then their quantization.
///
/// Split out of the kernel because it is the one phase whose work decomposes
/// differently on the two capacity paths, and because the kernel below reads
/// far better when the loop it runs four times is named.

#include "queues.cuh"

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// Fused projection and quantization.
// ---------------------------------------------------------------------------

/// Quantize the latent groups one projection unit just produced.
///
/// Reads through `__ldcg` and behind a device-scope fence, because on the
/// tensor path the values arrived in global memory through a bulk tensor store
/// that this CTA's L1 never saw. The arithmetic is
/// `expert_mxfp4::quantize_latent_rows`, restricted to one group range.
static __device__ void quantize_latent_group_range(
    const __nv_bfloat16 *__restrict__ latent_x,
    const Scratch &scratch,
    const int active_tokens,
    const int group_begin,
    const int group_count
) {
    using expert_mxfp4::kLatentGroups;
    using expert_mxfp4::kMmaK;

    __threadfence();
    __syncthreads();

    const int thread = static_cast<int>(threadIdx.x);
    const int total = active_tokens * group_count;
    for (int index = thread; index < total; index += kDecodeCtaThreads) {
        const int token = index / group_count;
        const int group = group_begin + index % group_count;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const float value = __bfloat162float(
                __ldcg(&latent_x[static_cast<long long>(token) * kLatentSize
                                 + group * kMmaK + k]));
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale =
            expert_mxfp4::select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        scratch.latent_scale[
            static_cast<long long>(token) * kLatentGroups + group] = scale;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            scratch.latent_mxfp8[
                static_cast<long long>(token) * kLatentSize
                + group * kMmaK + k] =
                    expert_mxfp4::quantize_e4m3(values[k], reciprocal);
        }
    }
}

// ---------------------------------------------------------------------------

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
