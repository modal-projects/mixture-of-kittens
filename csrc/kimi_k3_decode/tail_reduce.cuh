#pragma once

#include "tail_sync.cuh"

#include <cuda_bf16.h>

namespace kimi_k3_decode {
namespace tail {

// ---------------------------------------------------------------------------
// Reduce role: routed all-reduce plus FP32 RMSNorm, and shared reduce-scatter.
// ---------------------------------------------------------------------------

inline constexpr int kLatentOctetsPerThread =
    (kLatentOctets + kDecodeCtaThreads - 1) / kDecodeCtaThreads;
inline constexpr int kShardOctetsPerThread =
    (kShardOctets + kDecodeCtaThreads - 1) / kDecodeCtaThreads;

static_assert(kLatentOctetsPerThread == 2);
static_assert(kShardOctetsPerThread == 1);

/// Reduce, normalize, and scatter every token row this CTA owns.
///
/// Each thread keeps its share of the reduced latent row in registers across the
/// row's sum-of-squares reduction, so the row is read from the fabric exactly
/// once. The reduction order is fixed by the thread-to-octet mapping, which
/// makes the RMS scale identical on every rank for identical inputs.
static __device__ void reduce_rows(
    __nv_bfloat16 *__restrict__ const collective_multicast,
    const __nv_bfloat16 *__restrict__ const routed_latent_rmsnorm_weight,
    const Scratch &scratch,
    const int reduce_index,
    const int tp_rank,
    const int active_tokens,
    const TailClocks &clocks
) {
    __shared__ float warp_totals[kWarps];
    __shared__ float row_scale;

    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread % 32;

    for (int row = reduce_index; row < active_tokens; row += kReduceCtas) {
        const long long row_base =
            static_cast<long long>(row) * kCollectiveColumns;
        // #region agent log
        unsigned long long mark = clocks.now();
        // #endregion

        Octet reduced[kLatentOctetsPerThread];
        float squares = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < kLatentOctetsPerThread; ++slot) {
            const int octet = thread + slot * kDecodeCtaThreads;
            if (octet < kLatentOctets) {
                multimem_reduce_octet(
                    reduced[slot],
                    collective_multicast + row_base + octet * kOctetLanes);
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float low = low_bf16(reduced[slot].pair[pair]);
                    const float high = high_bf16(reduced[slot].pair[pair]);
                    squares = fmaf(low, low, squares);
                    squares = fmaf(high, high, squares);
                }
            }
        }
        // #region agent log
        mark = clocks.lap(kTailClockRoutedMultimemReduce, mark);
        // #endregion

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            squares += __shfl_down_sync(0xffffffffu, squares, offset);
        }
        __syncthreads();
        if (lane == 0) warp_totals[warp] = squares;
        __syncthreads();
        if (thread == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int index = 0; index < kWarps; ++index) {
                total += warp_totals[index];
            }
            row_scale = rsqrtf(
                total / static_cast<float>(kLatentSize) + kRmsEpsilon);
        }
        __syncthreads();
        const float scale = row_scale;

        #pragma unroll
        for (int slot = 0; slot < kLatentOctetsPerThread; ++slot) {
            const int octet = thread + slot * kDecodeCtaThreads;
            if (octet < kLatentOctets) {
                Octet weight;
                load_octet(
                    weight,
                    routed_latent_rmsnorm_weight + octet * kOctetLanes);
                Octet normalized;
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    // Match the reference's BF16 boundary: round the scaled
                    // latent first, then multiply by the BF16 weight.
                    const float low = __bfloat162float(__float2bfloat16(
                        low_bf16(reduced[slot].pair[pair]) * scale));
                    const float high = __bfloat162float(__float2bfloat16(
                        high_bf16(reduced[slot].pair[pair]) * scale));
                    normalized.pair[pair] = pack_bf16(
                        low * low_bf16(weight.pair[pair]),
                        high * high_bf16(weight.pair[pair]));
                }
                store_octet(
                    scratch.tail_normalized
                        + static_cast<long long>(row) * kLatentSize
                        + octet * kOctetLanes,
                    normalized);
            }
        }
        // #region agent log
        mark = clocks.lap(kTailClockRmsNorm, mark);
        // #endregion

        const long long shard_base =
            row_base + kLatentSize
            + static_cast<long long>(tp_rank) * kShardColumns;
        #pragma unroll
        for (int slot = 0; slot < kShardOctetsPerThread; ++slot) {
            const int octet = thread + slot * kDecodeCtaThreads;
            if (octet < kShardOctets) {
                Octet shard;
                multimem_reduce_octet(
                    shard,
                    collective_multicast + shard_base + octet * kOctetLanes);
                store_octet(
                    scratch.tail_shared_shard
                        + static_cast<long long>(row) * kShardColumns
                        + octet * kOctetLanes,
                    shard);
            }
        }
        // #region agent log
        clocks.lap(kTailClockSharedMultimemReduce, mark);
        // #endregion
    }
}

}  // namespace tail
}  // namespace kimi_k3_decode
