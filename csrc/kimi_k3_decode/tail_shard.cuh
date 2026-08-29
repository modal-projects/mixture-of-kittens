#pragma once

#include "kittens.cuh"

#include "tail_sync.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace kimi_k3_decode {
namespace tail {

// ---------------------------------------------------------------------------
// Shard role: this rank's 896 output columns, beta-added and multicast.
// ---------------------------------------------------------------------------

/// Multicast one token's eight adjacent output columns into every rank's slot.
static __device__ __forceinline__ void publish_shard_octet(
    __nv_bfloat16 *__restrict__ const mailbox_multicast,
    const Octet &value,
    const int row,
    const int tp_rank,
    const int column
) {
    multimem_store_octet(
        mailbox_multicast
            + (static_cast<long long>(row) * kTensorParallelSize + tp_rank)
                  * kShardColumns
            + column,
        value);
}

template<int CAPACITY>
static __device__ void shard_core(
    std::uint8_t *__restrict__ const shared,
    const Scratch &scratch,
    const __nv_bfloat16 *__restrict__ const latent_up_proj,
    __nv_bfloat16 *__restrict__ const mailbox_multicast,
    const int column_block,
    const int tp_rank,
    const int active_tokens,
    const TailClocks &clocks
) {
    static_assert(CAPACITY >= 1 && CAPACITY <= kMaxCoreCapacity);
    // #region agent log
    unsigned long long mark = clocks.now();
    // #endregion
    __nv_bfloat16 *const staged = reinterpret_cast<__nv_bfloat16 *>(shared);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread % 32;
    const int column_base =
        column_block * kCoreShardColumnsPerCta
        + warp * kCoreShardColumnsPerWarp;
    constexpr int vectors_per_row = kCoreLatentChunk / kOctetLanes;

    float accumulator[kCoreShardColumnsPerWarp][CAPACITY];
    #pragma unroll
    for (int column = 0; column < kCoreShardColumnsPerWarp; ++column) {
        #pragma unroll
        for (int row = 0; row < CAPACITY; ++row) {
            accumulator[column][row] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < kLatentSize; chunk += kCoreLatentChunk) {
        __syncthreads();
        const int staged_vectors = active_tokens * vectors_per_row;
        for (int index = thread; index < staged_vectors;
             index += kDecodeCtaThreads) {
            const int row = index / vectors_per_row;
            const int vector = index - row * vectors_per_row;
            *reinterpret_cast<float4 *>(
                staged + row * kCoreLatentChunk + vector * kOctetLanes) =
                    *reinterpret_cast<const float4 *>(
                        scratch.tail_normalized
                        + static_cast<long long>(row) * kLatentSize
                        + chunk + vector * kOctetLanes);
        }
        __syncthreads();

        #pragma unroll
        for (int column = 0; column < kCoreShardColumnsPerWarp; ++column) {
            const long long weight_offset =
                static_cast<long long>(
                    tp_rank * kShardColumns + column_base + column)
                    * kLatentSize
                + chunk;
            for (int k = lane * kOctetLanes; k < kCoreLatentChunk;
                 k += 32 * kOctetLanes) {
                const float4 weight = *reinterpret_cast<const float4 *>(
                    latent_up_proj + weight_offset + k);
                #pragma unroll
                for (int row = 0; row < CAPACITY; ++row) {
                    if (row < active_tokens) {
                        accumulator[column][row] = accumulate_bf16_octet(
                            *reinterpret_cast<const float4 *>(
                                staged + row * kCoreLatentChunk + k),
                            weight,
                            accumulator[column][row]);
                    }
                }
            }
        }
    }

    // #region agent log
    mark = clocks.lap(kTailClockLatentUpShardMma, mark);
    // #endregion
    #pragma unroll
    for (int row = 0; row < CAPACITY; ++row) {
        Octet result;
        #pragma unroll
        for (int pair = 0; pair < kCoreShardColumnsPerWarp / 2; ++pair) {
            float low = accumulator[2 * pair][row];
            float high = accumulator[2 * pair + 1][row];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                low += __shfl_down_sync(0xffffffffu, low, offset);
                high += __shfl_down_sync(0xffffffffu, high, offset);
            }
            if (lane == 0) {
                const long long beta =
                    static_cast<long long>(row) * kShardColumns
                    + column_base + 2 * pair;
                result.pair[pair] = pack_bf16(
                    low + __bfloat162float(scratch.tail_shared_shard[beta]),
                    high
                        + __bfloat162float(
                            scratch.tail_shared_shard[beta + 1]));
            }
        }
        if (lane == 0 && row < active_tokens) {
            publish_shard_octet(
                mailbox_multicast, result, row, tp_rank, column_base);
        }
    }
    // #region agent log
    clocks.lap(kTailClockMailboxMulticast, mark);
    // #endregion
}

/// Contract one 128-column output tile of this rank's shard and multicast it.
///
/// `tensor_pool` is owned by the caller because a CTA may allocate tensor
/// memory only once: the persistent kernel provisions it at entry and hands the
/// same pool to every stage, and the private kernel provisions one of its own.
static __device__ void shard_tensor(
    int *__restrict__ const shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const tensor_input_layout &normalized,
    const tensor_weight_layout &latent_up_proj,
    const Scratch &scratch,
    __nv_bfloat16 *__restrict__ const mailbox_multicast,
    const int column_block,
    const int tp_rank,
    const int active_tokens,
    const TailClocks &clocks
) {
    using namespace kittens;
    // #region agent log
    unsigned long long mark = clocks.now();
    // #endregion
    tma_swizzle_allocator allocator(shared_raw);
    tensor_input_tile (&input_tiles)[kStages] =
        allocator.allocate<tensor_input_tile, kStages>();
    tensor_weight_tile (&weight_tiles)[kStages] =
        allocator.allocate<tensor_weight_tile, kStages>();
    tensor_result_tile (&result_shared) =
        allocator.allocate<tensor_result_tile>();

    __shared__ semaphore inputs_arrived[kStages];
    __shared__ semaphore inputs_finished[kStages];
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 1, 0);
        }
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    if (warpgroup::groupid() == 0) {
        tensor_accumulator_tile accumulator =
            tensor_pool.allocate<tensor_accumulator_tile>(0);
        const int warpgroup_lane = warpgroup::laneid();

        for (int iteration = 0; iteration < kTensorKIterations; ++iteration) {
            const int stage = iteration % kStages;
            const int round = iteration / kStages;
            if (warpgroup_lane == 0) {
                wait(inputs_finished[stage], (round + 1) % 2);
                tma::expect_bytes(
                    inputs_arrived[stage],
                    sizeof(tensor_input_tile) + sizeof(tensor_weight_tile));
                tma::load_async(
                    input_tiles[stage], normalized, {0, iteration},
                    inputs_arrived[stage]);
                tma::load_async(
                    weight_tiles[stage], latent_up_proj,
                    {column_block, iteration}, inputs_arrived[stage]);
            }
            wait(inputs_arrived[stage], round % 2);
            if (warpgroup_lane == 0) {
                if (iteration == 0) {
                    mm_ABt(
                        accumulator, input_tiles[stage], weight_tiles[stage],
                        inputs_finished[stage]);
                } else {
                    mma_ABt(
                        accumulator, input_tiles[stage], weight_tiles[stage],
                        inputs_finished[stage]);
                }
            }
        }

        if (warpgroup_lane == 0) detail::tcgen05::commit<1>(compute_done);
        wait(compute_done, 0);

        rt_fl<kTileM / 4, kTileN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(result_shared, result);
        warpgroup::sync(1);
    }
    __syncthreads();
    // #region agent log
    mark = clocks.lap(kTailClockLatentUpShardMma, mark);
    // #endregion

    constexpr int groups = kTileN / kOctetLanes;
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < active_tokens * groups;
         index += kDecodeCtaThreads) {
        const int row = index / groups;
        const int column = column_block * kTileN + (index - row * groups)
            * kOctetLanes;
        const int tile_column = column - column_block * kTileN;
        Octet value;
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const long long beta =
                static_cast<long long>(row) * kShardColumns
                + column + 2 * pair;
            value.pair[pair] = pack_bf16(
                result_shared[{row, tile_column + 2 * pair}]
                    + __bfloat162float(scratch.tail_shared_shard[beta]),
                result_shared[{row, tile_column + 2 * pair + 1}]
                    + __bfloat162float(
                        scratch.tail_shared_shard[beta + 1]));
        }
        publish_shard_octet(mailbox_multicast, value, row, tp_rank, column);
    }
    // #region agent log
    clocks.lap(kTailClockMailboxMulticast, mark);
    // #endregion
}

}  // namespace tail
}  // namespace kimi_k3_decode
