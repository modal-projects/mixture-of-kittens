#pragma once

#include "kittens.cuh"

#include "skinny_gemm.cuh"
#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <tuple>

namespace kimi_k3_decode {
namespace tail {

// One `std::once_flag` per possible CUDA ordinal, so the tcgen05 shared-memory
// reservation happens once per device even when one process drives several.
inline constexpr int kMaxCudaDevices = 32;

// The fused TP8 tail closes one decode step in a single launch per rank. It
// all-reduces the routed latent and reduce-scatters the shared output straight
// out of the symmetric collective buffer with SM103 multimem instructions,
// normalizes the replicated latent in FP32, contracts this rank's 896 rows of
// the replicated latent-up weight, beta-adds the reduced shared shard, and
// multicasts the resulting token-major shard into every rank's mailbox slot.

inline constexpr int kCollectiveColumns = kLatentSize + kHiddenSize;
inline constexpr int kShardColumns = kHiddenSize / kTensorParallelSize;
inline constexpr float kRmsEpsilon = 1e-5f;

// Every multimem access moves one 16-byte `.v4.bf16x2` octet. The collective
// buffer, the mailbox, and the RMSNorm weight are all 16-byte aligned and every
// extent below is a multiple of eight BF16 values, so no access straddles a
// vector boundary.
inline constexpr int kOctetLanes = 8;
inline constexpr int kLatentOctets = kLatentSize / kOctetLanes;
inline constexpr int kShardOctets = kShardColumns / kOctetLanes;

inline constexpr int kWarps = kDecodeCtaThreads / 32;

// Role bands, producers first. Block 0 owns both cross-rank edges, the reduce
// band publishes the normalized latent and the reduced shared shard, and the
// shard band consumes both. The host residency guard keeps progress independent
// of the order the scheduler happens to launch these blocks in.
inline constexpr int kCoordinatorCtas = 1;
inline constexpr int kReduceCtas = 32;
inline constexpr int kCoordinatorBegin = 0;
inline constexpr int kReduceBegin = kCoordinatorBegin + kCoordinatorCtas;
inline constexpr int kShardBegin = kReduceBegin + kReduceCtas;

// Direct-register shard path, used for capacities 1, 2, 4, and 8. One warp owns
// eight adjacent output columns so its lane 0 holds exactly one multimem octet
// after the cross-lane reduction and the mailbox store needs no staging.
inline constexpr int kCoreShardColumnsPerWarp = kOctetLanes;
inline constexpr int kCoreShardColumnsPerCta =
    kWarps * kCoreShardColumnsPerWarp;
inline constexpr int kCoreShardCtas = kShardColumns / kCoreShardColumnsPerCta;
inline constexpr int kCoreLatentChunk = 512;
inline constexpr int kCoreRoleCtas = kShardBegin + kCoreShardCtas;
inline constexpr int kTailCoreDynamicBytes =
    kMaxCoreCapacity * kCoreLatentChunk * sizeof(__nv_bfloat16);

// tcgen05 shard path, used for capacities 16, 32, 64, and 128. One 128-row MMA
// tile covers every one of those capacities because the input descriptor spans
// exactly the active rows and the hardware zero-fills the remainder.
inline constexpr int kTileM = skinny_gemm::kTileM;
inline constexpr int kTileN = skinny_gemm::kTileN;
inline constexpr int kTileK = skinny_gemm::kTileK;
inline constexpr int kStages = skinny_gemm::kStages;
inline constexpr int kTensorShardCtas = kShardColumns / kTileN;
inline constexpr int kTensorKIterations = kLatentSize / kTileK;
inline constexpr int kTensorRoleCtas = kShardBegin + kTensorShardCtas;
inline constexpr int kTailTensorDynamicBytes =
    kittens::MAX_SHARED_MEMORY - 1024;

// Roughly fifteen seconds of B300 clocks. Every cross-rank and cross-CTA spin
// in this file is bounded by it, so a lost peer surfaces as a device trap with
// a recorded phase slot instead of an unkillable GPU.
inline constexpr std::uint64_t kGenerationWaitTimeoutClocks =
    30'000'000'000ULL;

static_assert(kShardColumns == 896);
static_assert(kLatentSize % kOctetLanes == 0);
static_assert(kShardColumns % kOctetLanes == 0);
static_assert(kShardColumns % kCoreShardColumnsPerCta == 0);
static_assert(kLatentSize % kCoreLatentChunk == 0);
static_assert(kCoreLatentChunk % (kOctetLanes * 32) == 0);
static_assert(kShardColumns % kTileN == 0);
static_assert(kLatentSize % kTileK == 0);
static_assert(kCoordinatorBegin < kReduceBegin);
static_assert(kReduceBegin < kShardBegin);
static_assert(kCoreShardCtas == 14);
static_assert(kTensorShardCtas == 7);
static_assert(kCoreRoleCtas == 47);
static_assert(kTensorRoleCtas == 40);

using tensor_input_tile = skinny_gemm::hidden_tile;
using tensor_weight_tile = skinny_gemm::weight_tile;
using tensor_result_tile = kittens::st_fl<kTileM, kTileN>;
using tensor_accumulator_tile = skinny_gemm::accumulator_tile;
using tensor_input_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_input_tile>;
using tensor_weight_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_weight_tile>;

// ---------------------------------------------------------------------------
// Wrap-safe generation, barrier, and timeout arithmetic, shared with the host.
// ---------------------------------------------------------------------------

/// Report whether a published generation moved past a consumed one.
__host__ __device__ inline constexpr bool generation_advanced(
    const std::uint32_t observed,
    const std::uint32_t consumed
) {
    const std::uint32_t difference = observed - consumed;
    return difference != 0u && difference < 0x80000000u;
}

/// Report whether a monotonically rising barrier counter reached its target.
__host__ __device__ inline constexpr bool barrier_reached(
    const std::uint32_t observed,
    const std::uint32_t target
) {
    return (observed - target) < 0x80000000u;
}

__host__ __device__ inline constexpr bool wait_timed_out(
    const std::uint64_t started,
    const std::uint64_t current
) {
    return current - started >= kGenerationWaitTimeoutClocks;
}

// ---------------------------------------------------------------------------
// Register-resident 16-byte BF16 octets.
// ---------------------------------------------------------------------------

struct Octet {
    std::uint32_t pair[4];
};

/// Sum one octet across all eight ranks through the symmetric multicast alias.
static __device__ __forceinline__ void multimem_reduce_octet(
    Octet &out,
    const __nv_bfloat16 *const address
) {
    asm volatile(
        "multimem.ld_reduce.weak.global.add.acc::f32.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(out.pair[0]), "=r"(out.pair[1]),
          "=r"(out.pair[2]), "=r"(out.pair[3])
        : "l"(address)
        : "memory");
}

/// Write one octet into every rank's copy of a symmetric allocation.
static __device__ __forceinline__ void multimem_store_octet(
    __nv_bfloat16 *const address,
    const Octet &value
) {
    asm volatile(
        "multimem.st.weak.global.v4.bf16x2 [%0], {%1, %2, %3, %4};"
        :: "l"(address), "r"(value.pair[0]), "r"(value.pair[1]),
           "r"(value.pair[2]), "r"(value.pair[3])
        : "memory");
}

static __device__ __forceinline__ void load_octet(
    Octet &out,
    const __nv_bfloat16 *const address
) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(address);
    out.pair[0] = raw.x;
    out.pair[1] = raw.y;
    out.pair[2] = raw.z;
    out.pair[3] = raw.w;
}

static __device__ __forceinline__ void store_octet(
    __nv_bfloat16 *const address,
    const Octet &value
) {
    *reinterpret_cast<uint4 *>(address) =
        make_uint4(value.pair[0], value.pair[1], value.pair[2], value.pair[3]);
}

static __device__ __forceinline__ float low_bf16(const std::uint32_t bits) {
    return __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(bits & 0xffffu)));
}

static __device__ __forceinline__ float high_bf16(const std::uint32_t bits) {
    return __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16)));
}

static __device__ __forceinline__ std::uint32_t pack_bf16(
    const float low,
    const float high
) {
    return static_cast<std::uint32_t>(
               __bfloat16_as_ushort(__float2bfloat16(low)))
         | (static_cast<std::uint32_t>(
                __bfloat16_as_ushort(__float2bfloat16(high))) << 16);
}

// ---------------------------------------------------------------------------
// Generation-tagged phases. Nothing here needs a host reset between launches,
// so a captured graph replays without zeroing the workspace.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ std::uint32_t load_relaxed_gpu(
    const int *const address
) {
    std::uint32_t value;
    asm volatile(
        "{ld.relaxed.gpu.global.u32 %0, [%1];}"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}

static __device__ __forceinline__ std::uint32_t load_relaxed_system(
    const std::uint32_t *const address
) {
    std::uint32_t value;
    asm volatile(
        "{ld.relaxed.sys.global.u32 %0, [%1];}"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}

/// Record the slot a bounded wait gave up on, then trap this thread's launch.
static __device__ __forceinline__ void record_timeout_and_trap(
    const Scratch &scratch,
    const int generation_index
) {
    atomicExch(
        reinterpret_cast<unsigned int *>(&scratch.phase[kTailTimeoutPhase]),
        static_cast<unsigned int>(generation_index));
    __threadfence_system();
    asm volatile("trap;");
}

/// Latch this CTA's baseline for one generation before any producer can move it.
///
/// Every role reads the generation it publishes itself, which lags the
/// generation it consumes by exactly one launch. The last arrival of a role only
/// advances that role's generation once every CTA in the role has already
/// latched its baseline, so no CTA can observe a baseline from the wrong launch.
static __device__ std::uint32_t latch_generation(
    const Scratch &scratch,
    const int generation_index,
    std::uint32_t *const slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *slot = load_relaxed_gpu(&scratch.phase[generation_index]);
    }
    __syncthreads();
    return *slot;
}

/// Spin until `generation_index` moves past `baseline`, then acquire.
static __device__ void wait_for_generation(
    const Scratch &scratch,
    const int generation_index,
    const std::uint32_t baseline
) {
    if (threadIdx.x == 0) {
        const std::uint64_t started = clock64();
        while (!generation_advanced(
            load_relaxed_gpu(&scratch.phase[generation_index]), baseline
        )) {
            if (wait_timed_out(started, clock64())) {
                record_timeout_and_trap(scratch, generation_index);
            }
            __nanosleep(64);
        }
    }
    // Thread 0's observation converges the CTA. Peer writes reach this rank
    // through the coordinator's system-scope acquire, so every consumer thread
    // fences at system scope before reading the collective buffer or a scratch
    // intermediate that a peer's data flowed into.
    __syncthreads();
    __threadfence_system();
}

/// Release this role's writes, then take one arrival ticket for this CTA.
static __device__ void publish_generation(
    const Scratch &scratch,
    const int arrivals_index,
    const int generation_index,
    const int role_ctas
) {
    // Fence per thread, then converge, so the ticket-taking thread only counts
    // this CTA once every write it made is visible to peers and to consumers.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        auto *const arrivals = reinterpret_cast<unsigned int *>(
            &scratch.phase[arrivals_index]);
        const unsigned int ticket = atomicAdd(arrivals, 1u);
        if (ticket == static_cast<unsigned int>(role_ctas - 1)) {
            atomicExch(arrivals, 0u);
            atomicAdd(
                reinterpret_cast<unsigned int *>(
                    &scratch.phase[generation_index]),
                1u);
        }
    }
}

/// Rendezvous all eight ranks once, from one thread.
///
/// The symmetric counter is incremented through its multicast alias and polled
/// through its unicast alias, and the private target rises by the rank count on
/// every call, so the pair stays consistent across launches and across the
/// unsigned wrap without a host reset.
static __device__ void barrier_ranks(
    const Scratch &scratch,
    std::uint32_t *const barrier_multicast,
    const std::uint32_t *const barrier_local,
    unsigned int *const barrier_target,
    const int generation_index
) {
    const std::uint32_t target =
        static_cast<std::uint32_t>(
            atomicAdd(barrier_target,
                      static_cast<unsigned int>(kTensorParallelSize)))
        + static_cast<std::uint32_t>(kTensorParallelSize);

    __threadfence_system();
    asm volatile(
        "{multimem.red.release.sys.global.add.u32 [%0], 1;}"
        :: "l"(barrier_multicast)
        : "memory");
    asm volatile("{fence.proxy.alias;}" ::: "memory");

    const std::uint64_t started = clock64();
    while (!barrier_reached(load_relaxed_system(barrier_local), target)) {
        if (wait_timed_out(started, clock64())) {
            record_timeout_and_trap(scratch, generation_index);
        }
        __nanosleep(64);
    }
    asm volatile("{fence.acquire.sys;}" ::: "memory");
}

/// Drive both cross-rank edges of one tail launch from a single thread.
static __device__ void coordinate_ranks(
    const Scratch &scratch,
    std::uint32_t *const barrier_multicast,
    const std::uint32_t *const barrier_local,
    unsigned int *const barrier_target
) {
    if (threadIdx.x != 0) return;

    // Entry: every rank has finished writing its routed and shared partials
    // into its own copy of the collective buffer.
    barrier_ranks(scratch, barrier_multicast, barrier_local, barrier_target,
                  kTailEntryGeneration);
    __threadfence_system();
    atomicAdd(
        reinterpret_cast<unsigned int *>(&scratch.phase[kTailEntryGeneration]),
        1u);

    const std::uint32_t baseline =
        load_relaxed_gpu(&scratch.phase[kTailExitGeneration]);
    const std::uint64_t started = clock64();
    while (!generation_advanced(
        load_relaxed_gpu(&scratch.phase[kTailShardGeneration]), baseline
    )) {
        if (wait_timed_out(started, clock64())) {
            record_timeout_and_trap(scratch, kTailShardGeneration);
        }
        __nanosleep(64);
    }

    // Exit: every rank has multicast its own shard, so after this rendezvous
    // all eight mailbox slots on this rank hold the current launch's data.
    barrier_ranks(scratch, barrier_multicast, barrier_local, barrier_target,
                  kTailExitGeneration);
    __threadfence_system();
    atomicAdd(
        reinterpret_cast<unsigned int *>(&scratch.phase[kTailExitGeneration]),
        1u);
}

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
    const int active_tokens
) {
    __shared__ float warp_totals[kWarps];
    __shared__ float row_scale;

    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread % 32;

    for (int row = reduce_index; row < active_tokens; row += kReduceCtas) {
        const long long row_base =
            static_cast<long long>(row) * kCollectiveColumns;

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
    }
}

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
    const int active_tokens
) {
    static_assert(CAPACITY >= 1 && CAPACITY <= kMaxCoreCapacity);
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
}

static __device__ void shard_tensor(
    int *__restrict__ const shared_raw,
    const tensor_input_layout &normalized,
    const tensor_weight_layout &latent_up_proj,
    const Scratch &scratch,
    __nv_bfloat16 *__restrict__ const mailbox_multicast,
    const int column_block,
    const int tp_rank,
    const int active_tokens
) {
    using namespace kittens;
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

    // The managed allocator barriers the whole CTA, so both warpgroups enter.
    tensor_allocator<1, 1> tensor_pool{};
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
}

// ---------------------------------------------------------------------------
// Single-launch kernels.
// ---------------------------------------------------------------------------

/// Wait for the coordinator's exit edge before any consumer CTA retires.
///
/// Retiring earlier would let the surrounding grid, or Task 9's persistent
/// kernel, read a mailbox slot a peer has not filled yet.
static __device__ void drain_ranks(
    const Scratch &scratch,
    std::uint32_t *const baseline_slot,
    const int consumer_ctas
) {
    const std::uint32_t baseline =
        latch_generation(scratch, kTailDrainGeneration, baseline_slot);
    wait_for_generation(scratch, kTailExitGeneration, baseline);
    publish_generation(
        scratch, kTailDrainArrivals, kTailDrainGeneration, consumer_ctas);
}

template<int CAPACITY>
__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void kimi_k3_tail_core_kernel(
    const __nv_bfloat16 *__restrict__ routed_latent_rmsnorm_weight,
    const __nv_bfloat16 *__restrict__ latent_up_proj,
    __nv_bfloat16 *__restrict__ collective_multicast,
    __nv_bfloat16 *__restrict__ mailbox_multicast,
    std::uint32_t *__restrict__ barrier_multicast,
    const std::uint32_t *__restrict__ barrier_local,
    unsigned int *__restrict__ barrier_target,
    std::uint8_t *__restrict__ scratch_bytes,
    const int tp_rank,
    const int active_tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    __shared__ std::uint32_t baseline_slot;
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);

    if (block < kReduceBegin) {
        coordinate_ranks(
            scratch, barrier_multicast, barrier_local, barrier_target);
        return;
    }

    if (block < kShardBegin) {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailReduceGeneration, &baseline_slot);
        wait_for_generation(scratch, kTailEntryGeneration, baseline);
        reduce_rows(
            collective_multicast, routed_latent_rmsnorm_weight, scratch,
            block - kReduceBegin, tp_rank, active_tokens);
        publish_generation(
            scratch, kTailReduceArrivals, kTailReduceGeneration, kReduceCtas);
    } else {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailShardGeneration, &baseline_slot);
        wait_for_generation(scratch, kTailReduceGeneration, baseline);
        shard_core<CAPACITY>(
            reinterpret_cast<std::uint8_t *>(shared_raw), scratch,
            latent_up_proj, mailbox_multicast, block - kShardBegin, tp_rank,
            active_tokens);
        publish_generation(
            scratch, kTailShardArrivals, kTailShardGeneration, kCoreShardCtas);
    }

    drain_ranks(scratch, &baseline_slot, kReduceCtas + kCoreShardCtas);
}

__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void kimi_k3_tail_tensor_kernel(
    const __nv_bfloat16 *__restrict__ routed_latent_rmsnorm_weight,
    const __grid_constant__ tensor_input_layout normalized,
    const __grid_constant__ tensor_weight_layout latent_up_proj,
    __nv_bfloat16 *__restrict__ collective_multicast,
    __nv_bfloat16 *__restrict__ mailbox_multicast,
    std::uint32_t *__restrict__ barrier_multicast,
    const std::uint32_t *__restrict__ barrier_local,
    unsigned int *__restrict__ barrier_target,
    std::uint8_t *__restrict__ scratch_bytes,
    const int tp_rank,
    const int active_tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    __shared__ std::uint32_t baseline_slot;
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);

    if (block < kReduceBegin) {
        coordinate_ranks(
            scratch, barrier_multicast, barrier_local, barrier_target);
        return;
    }

    if (block < kShardBegin) {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailReduceGeneration, &baseline_slot);
        wait_for_generation(scratch, kTailEntryGeneration, baseline);
        reduce_rows(
            collective_multicast, routed_latent_rmsnorm_weight, scratch,
            block - kReduceBegin, tp_rank, active_tokens);
        publish_generation(
            scratch, kTailReduceArrivals, kTailReduceGeneration, kReduceCtas);
    } else {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailShardGeneration, &baseline_slot);
        wait_for_generation(scratch, kTailReduceGeneration, baseline);
        shard_tensor(
            shared_raw, normalized, latent_up_proj, scratch, mailbox_multicast,
            block - kShardBegin, tp_rank, active_tokens);
        publish_generation(
            scratch, kTailShardArrivals, kTailShardGeneration,
            kTensorShardCtas);
    }

    drain_ranks(scratch, &baseline_slot, kReduceCtas + kTensorShardCtas);
}

// ---------------------------------------------------------------------------
// Host role planning, residency, and launch.
// ---------------------------------------------------------------------------

struct RolePlan {
    int coordinator;
    int reduce;
    int shard;

    constexpr int total() const { return coordinator + reduce + shard; }
};

inline constexpr RolePlan role_plan(const int active_tokens) {
    return capacity_bucket(active_tokens) <= kMaxCoreCapacity
        ? RolePlan{kCoordinatorCtas, kReduceCtas, kCoreShardCtas}
        : RolePlan{kCoordinatorCtas, kReduceCtas, kTensorShardCtas};
}

inline std::tuple<int, int, int, int> role_plan_for_testing(
    const std::int64_t active_tokens
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: tail role plan requires active_tokens in [1, ",
                kMaxTokens, "]");
    const RolePlan plan = role_plan(static_cast<int>(active_tokens));
    return {plan.coordinator, plan.reduce, plan.shard, plan.total()};
}

inline std::tuple<int, int, int, int, int> timeout_metadata_for_testing() {
    return {
        kTailTimeoutPhase,
        kTailEntryGeneration,
        kTailReduceGeneration,
        kTailShardGeneration,
        kTailExitGeneration};
}

/// Reject a role grid whose spin-waiting CTAs cannot all be resident at once.
inline void validate_residency(
    const std::int64_t active_tokens,
    const std::int64_t available_sms
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: tail residency requires active_tokens in [1, ",
                kMaxTokens, "]");
    TORCH_CHECK(available_sms >= 0,
                "MoK: tail residency requires a nonnegative SM count");
    const RolePlan plan = role_plan(static_cast<int>(active_tokens));
    TORCH_CHECK(
        available_sms >= plan.total(),
        "MoK: _kimi_k3_tail requires all ", plan.total(),
        " role CTAs to co-reside, but the selected device exposes ",
        available_sms, " SMs");
}

/// Raise the tcgen05 kernel's dynamic shared-memory cap once per device.
///
/// The cap is a property of the compiled function, not of a launch, so raising
/// it here keeps the launch itself free of any runtime API call that a CUDA
/// graph capture would have to record or reject.
static __host__ void reserve_tensor_shared_memory() {
    static std::array<std::once_flag, kMaxCudaDevices> reserved;
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(device >= 0 && device < kMaxCudaDevices,
                "MoK: _kimi_k3_tail saw an unexpected device ordinal ", device);
    std::call_once(reserved[static_cast<std::size_t>(device)], [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kimi_k3_tail_tensor_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kTailTensorDynamicBytes));
    });
}

template<int CAPACITY>
static __host__ void launch_core(
    const __nv_bfloat16 *routed_latent_rmsnorm_weight,
    const __nv_bfloat16 *latent_up_proj,
    __nv_bfloat16 *collective_multicast,
    __nv_bfloat16 *mailbox_multicast,
    std::uint32_t *barrier_multicast,
    const std::uint32_t *barrier_local,
    unsigned int *barrier_target,
    std::uint8_t *scratch_bytes,
    const int tp_rank,
    const int active_tokens
) {
    kimi_k3_tail_core_kernel<CAPACITY>
        <<<kCoreRoleCtas, kDecodeCtaThreads, kTailCoreDynamicBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            routed_latent_rmsnorm_weight, latent_up_proj, collective_multicast,
            mailbox_multicast, barrier_multicast, barrier_local, barrier_target,
            scratch_bytes, tp_rank, active_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __host__ void launch_tail(
    const at::Tensor &routed_latent_rmsnorm_weight,
    const at::Tensor &latent_up_proj,
    std::int64_t collective_buffer_multicast_ptr,
    std::int64_t output_mailbox_multicast_ptr,
    const at::Tensor &barrier_buffer,
    std::int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target,
    const at::Tensor &scratch,
    const int tp_rank,
    const int active_tokens,
    const int available_sms
) {
    validate_residency(active_tokens, available_sms);

    const auto *const norm_weight = reinterpret_cast<const __nv_bfloat16 *>(
        routed_latent_rmsnorm_weight.data_ptr());
    const auto *const latent_up =
        reinterpret_cast<const __nv_bfloat16 *>(latent_up_proj.data_ptr());
    auto *const collective_multicast = reinterpret_cast<__nv_bfloat16 *>(
        collective_buffer_multicast_ptr);
    auto *const mailbox_multicast = reinterpret_cast<__nv_bfloat16 *>(
        output_mailbox_multicast_ptr);
    auto *const barrier_multicast =
        reinterpret_cast<std::uint32_t *>(barrier_buffer_multicast_ptr);
    const auto *const barrier_local =
        reinterpret_cast<const std::uint32_t *>(barrier_buffer.data_ptr());
    auto *const target =
        reinterpret_cast<unsigned int *>(barrier_target.data_ptr());
    auto *const scratch_bytes =
        reinterpret_cast<std::uint8_t *>(scratch.data_ptr());

    switch (capacity_bucket(active_tokens)) {
        case 1:
            launch_core<1>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, tp_rank, active_tokens);
            return;
        case 2:
            launch_core<2>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, tp_rank, active_tokens);
            return;
        case 4:
            launch_core<4>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, tp_rank, active_tokens);
            return;
        case 8:
            launch_core<8>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, tp_rank, active_tokens);
            return;
        default:
            break;
    }

    const Scratch scratch_pointers = scratch_view(scratch_bytes);
    const tensor_input_layout normalized_view{
        reinterpret_cast<kittens::bf16 *>(scratch_pointers.tail_normalized),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kLatentSize)};
    // Only this rank's contiguous 896-row slice of the replicated latent-up
    // weight is ever contracted, so the descriptor starts at that slice.
    const tensor_weight_layout latent_up_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(
                latent_up
                + static_cast<long long>(tp_rank) * kShardColumns
                      * kLatentSize)),
        nullptr, nullptr, static_cast<size_t>(kShardColumns),
        static_cast<size_t>(kLatentSize)};

    reserve_tensor_shared_memory();
    kimi_k3_tail_tensor_kernel
        <<<kTensorRoleCtas, kDecodeCtaThreads, kTailTensorDynamicBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            norm_weight, normalized_view, latent_up_view, collective_multicast,
            mailbox_multicast, barrier_multicast, barrier_local, target,
            scratch_bytes, tp_rank, active_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tail
}  // namespace kimi_k3_decode
