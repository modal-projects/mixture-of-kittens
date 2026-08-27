#pragma once

#include "kittens.cuh"

#include "types.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace kimi_k3_decode {
namespace skinny_gemm {

// ---------------------------------------------------------------------------
// Direct-register CUDA-core path, used for capacities 1, 2, 4, and 8.
//
// The whole latent-down weight streams past a token block that never fills a
// tensor-core tile, so keeping the accumulators in registers and the staged
// hidden rows in shared memory beats paying for a 128-row MMA.
// ---------------------------------------------------------------------------

inline constexpr int kCoreWarps = kDecodeCtaThreads / 32;
inline constexpr int kCoreColumnsPerWarp = 4;
inline constexpr int kCoreColumnsPerCta = kCoreWarps * kCoreColumnsPerWarp;
inline constexpr int kCoreCtas = kLatentSize / kCoreColumnsPerCta;
inline constexpr int kCoreChunk = 1024;
inline constexpr int kCoreChunkVectors = kCoreChunk / 8;

static_assert(kLatentSize % kCoreColumnsPerCta == 0,
              "the CUDA-core path must tile 3584 latent columns exactly");
static_assert(kHiddenSize % kCoreChunk == 0,
              "the CUDA-core path must tile 7168 hidden values exactly");

inline constexpr int kCoreSharedBytes = kMaxCoreCapacity * kCoreChunk * 2;

// ---------------------------------------------------------------------------
// tcgen05 BF16 path, used for capacities 16, 32, 64, and 128.
//
// One 128-row MMA tile covers every one of those capacities: the TMA descriptors
// are built over exactly `active_tokens` rows, so the hardware zero-fills the
// unused rows on load and drops them on store.
// ---------------------------------------------------------------------------

inline constexpr int kTileM = 128;
inline constexpr int kTileN = 128;
inline constexpr int kTileK = 64;
inline constexpr int kStages = 2;
inline constexpr int kTensorCtas = kLatentSize / kTileN;
inline constexpr int kTensorKIterations = kHiddenSize / kTileK;

static_assert(kLatentSize % kTileN == 0,
              "the tcgen05 path must tile 3584 latent columns exactly");
static_assert(kHiddenSize % kTileK == 0,
              "the tcgen05 path must tile 7168 hidden values exactly");

using hidden_tile = kittens::st_bf<kTileM, kTileK>;
using weight_tile = kittens::st_bf<kTileN, kTileK>;
using latent_tile = kittens::st_bf<kTileM, kTileN>;
using accumulator_tile = kittens::tt<float, kTileM, kTileN>;

using hidden_layout = kittens::gl<kittens::bf16, 1, 1, -1, -1, hidden_tile>;
using weight_layout = kittens::gl<kittens::bf16, 1, 1, -1, -1, weight_tile>;
using latent_layout = kittens::gl<kittens::bf16, 1, 1, -1, -1, latent_tile>;

/// Zero the latent columns this CTA owns for every row past the active block,
/// plus a strided share of the unused router outputs.
///
/// The stage allocates its outputs uninitialised so the private operator stays a
/// single kernel launch, which makes masking the kernel's own responsibility.
static __device__ void mask_inactive_rows(
    int *__restrict__ expert_ids_out,
    float *__restrict__ expert_weights_out,
    __nv_bfloat16 *__restrict__ latent_x,
    const int column_base,
    const int columns,
    const int projection_index,
    const int projection_ctas,
    const int active_tokens,
    const int tokens
) {
    const int thread = static_cast<int>(threadIdx.x);
    const int inactive_rows = tokens - active_tokens;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int index = thread; index < inactive_rows * columns;
         index += kDecodeCtaThreads) {
        const int row = active_tokens + index / columns;
        const int column = column_base + index % columns;
        latent_x[static_cast<long long>(row) * kLatentSize + column] = zero;
    }

    const int tail = inactive_rows * kTopK;
    const int stride = projection_ctas * kDecodeCtaThreads;
    for (int index = projection_index * kDecodeCtaThreads + thread; index < tail;
         index += stride) {
        const int route = active_tokens * kTopK + index;
        expert_ids_out[route] = 0;
        expert_weights_out[route] = 0.0f;
    }
}

/// Publish this role's completion with a generation tag, so a reused workspace
/// never needs a host-side reset between decode steps.
static __device__ void publish_projection_completion(
    const Scratch &scratch,
    const int projection_ctas
) {
    // Fence per thread, then barrier, so the ticket-taking thread only counts
    // this CTA's arrival once every latent write it made is visible device-wide.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int ticket = atomicAdd(&scratch.phase[kProjectionArrivals], 1);
        if (ticket == projection_ctas - 1) {
            atomicExch(&scratch.phase[kProjectionArrivals], 0);
            atomicAdd(&scratch.phase[kProjectionGeneration], 1);
        }
    }
}

/// Project one block of latent columns with CUDA cores.
template<int CAPACITY>
static __device__ void latent_down_cuda_core(
    std::uint8_t *__restrict__ shared,
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ latent_down_proj,
    __nv_bfloat16 *__restrict__ latent_x,
    const int column_block,
    const int active_tokens
) {
    static_assert(CAPACITY >= 1 && CAPACITY <= kMaxCoreCapacity,
                  "the CUDA-core path covers capacities 1 through 8");

    __nv_bfloat16 *const staged = reinterpret_cast<__nv_bfloat16 *>(shared);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread % 32;
    const int column_base =
        column_block * kCoreColumnsPerCta + warp * kCoreColumnsPerWarp;

    float accumulator[kCoreColumnsPerWarp][CAPACITY];
    #pragma unroll
    for (int column = 0; column < kCoreColumnsPerWarp; column++) {
        #pragma unroll
        for (int row = 0; row < CAPACITY; row++) accumulator[column][row] = 0.0f;
    }

    for (int chunk = 0; chunk < kHiddenSize; chunk += kCoreChunk) {
        __syncthreads();
        const int staged_vectors = active_tokens * kCoreChunkVectors;
        for (int index = thread; index < staged_vectors; index += kDecodeCtaThreads) {
            const int row = index / kCoreChunkVectors;
            const int vector = index - row * kCoreChunkVectors;
            *reinterpret_cast<float4 *>(staged + row * kCoreChunk + vector * 8) =
                *reinterpret_cast<const float4 *>(
                    hidden_states + static_cast<long long>(row) * kHiddenSize
                    + chunk + vector * 8);
        }
        __syncthreads();

        #pragma unroll
        for (int column = 0; column < kCoreColumnsPerWarp; column++) {
            const __nv_bfloat16 *const weight_row =
                latent_down_proj
                + static_cast<long long>(column_base + column) * kHiddenSize + chunk;
            for (int i = lane * 8; i < kCoreChunk; i += 32 * 8) {
                const float4 weights =
                    *reinterpret_cast<const float4 *>(weight_row + i);
                #pragma unroll
                for (int row = 0; row < CAPACITY; row++) {
                    if (row < active_tokens) {
                        accumulator[column][row] = accumulate_bf16_octet(
                            weights,
                            *reinterpret_cast<const float4 *>(
                                staged + row * kCoreChunk + i),
                            accumulator[column][row]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int column = 0; column < kCoreColumnsPerWarp; column++) {
        #pragma unroll
        for (int row = 0; row < CAPACITY; row++) {
            float value = accumulator[column][row];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffu, value, offset);
            }
            if (lane == 0 && row < active_tokens) {
                latent_x[static_cast<long long>(row) * kLatentSize
                         + column_base + column] = __float2bfloat16(value);
            }
        }
    }
}

/// Project one 128-column latent tile with a two-stage tcgen05 BF16 pipeline.
///
/// All 256 threads must enter so the tensor-memory allocator's CTA barrier is
/// satisfied; the MMA pipeline and its epilogue then run on the first warpgroup.
static __device__ void latent_down_tcgen05(
    int *__restrict__ shared,
    const hidden_layout &hidden,
    const weight_layout &weight,
    const latent_layout &latent,
    const int column_block
) {
    using namespace kittens;

    tma_swizzle_allocator allocator(shared);
    hidden_tile (&hidden_tiles)[kStages] = allocator.allocate<hidden_tile, kStages>();
    weight_tile (&weight_tiles)[kStages] = allocator.allocate<weight_tile, kStages>();
    latent_tile (&latent_staging) = allocator.allocate<latent_tile>();

    __shared__ semaphore inputs_arrived[kStages];
    __shared__ semaphore inputs_finished[kStages];
    __shared__ semaphore compute_done;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < kStages; stage++) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 1, 0);
        }
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    // The managed allocator barriers the whole CTA, so both warpgroups enter.
    tensor_allocator<1, 1> tensor_pool{};
    if (warpgroup::groupid() == 0) {
        accumulator_tile accumulator = tensor_pool.allocate<accumulator_tile>(0);
        const int warpgroup_lane = warpgroup::laneid();

        for (int iteration = 0; iteration < kTensorKIterations; iteration++) {
            const int stage = iteration % kStages;
            const int round = iteration / kStages;
            if (warpgroup_lane == 0) {
                wait(inputs_finished[stage], (round + 1) % 2);
                tma::expect_bytes(inputs_arrived[stage],
                                  sizeof(hidden_tile) + sizeof(weight_tile));
                tma::load_async(hidden_tiles[stage], hidden, {0, iteration},
                                inputs_arrived[stage]);
                tma::load_async(weight_tiles[stage], weight,
                                {column_block, iteration}, inputs_arrived[stage]);
            }
            wait(inputs_arrived[stage], round % 2);
            if (warpgroup_lane == 0) {
                if (iteration == 0) {
                    mm_ABt(accumulator, hidden_tiles[stage], weight_tiles[stage],
                           inputs_finished[stage]);
                } else {
                    mma_ABt(accumulator, hidden_tiles[stage], weight_tiles[stage],
                            inputs_finished[stage]);
                }
            }
        }

        if (warpgroup_lane == 0) detail::tcgen05::commit<1>(compute_done);
        wait(compute_done, 0);

        rt_bf<kTileM / 4, kTileN> projected;
        warpgroup::load_async(projected, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(latent_staging, projected);
        warpgroup::sync(1);

        if (warpgroup_lane == 0) {
            tma::store_async(latent, latent_staging, {0, column_block});
            tma::store_async_wait();
        }
    }
    __syncthreads();
}

}  // namespace skinny_gemm
}  // namespace kimi_k3_decode
