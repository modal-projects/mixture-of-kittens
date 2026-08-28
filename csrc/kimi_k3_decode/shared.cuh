#pragma once

#include "kittens.cuh"

#include "skinny_gemm.cuh"
#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <tuple>

namespace kimi_k3_decode {
namespace shared_experts {

inline constexpr int kIntermediate =
    kSharedIntermediateSize / kTensorParallelSize;
inline constexpr int kCollectiveColumns = kLatentSize + kHiddenSize;

inline constexpr int kCoreGateColumnsPerWarp =
    skinny_gemm::kCoreColumnsPerWarp;
inline constexpr int kCoreGateColumnsPerCta =
    (kDecodeCtaThreads / 32) * kCoreGateColumnsPerWarp;
inline constexpr int kCoreGateCtas =
    kIntermediate / kCoreGateColumnsPerCta;
inline constexpr int kCoreDownColumnsPerWarp = 8;
inline constexpr int kCoreDownColumnsPerCta =
    (kDecodeCtaThreads / 32) * kCoreDownColumnsPerWarp;
inline constexpr int kCoreDownCtas =
    kHiddenSize / kCoreDownColumnsPerCta;
inline constexpr int kCoreInputChunk = skinny_gemm::kCoreChunk;
inline constexpr int kSharedExpertCoreDynamicBytes =
    kMaxCoreCapacity * kCoreInputChunk * sizeof(__nv_bfloat16);

inline constexpr int kTileM = skinny_gemm::kTileM;
inline constexpr int kTileN = skinny_gemm::kTileN;
inline constexpr int kTileK = skinny_gemm::kTileK;
inline constexpr int kTensorStages = skinny_gemm::kStages;
inline constexpr int kTensorGateCtas = kIntermediate / kTileN;
inline constexpr int kTensorDownCtas = kHiddenSize / kTileN;
inline constexpr int kTensorGateKIterations = kHiddenSize / kTileK;
inline constexpr int kTensorDownKIterations = kIntermediate / kTileK;
inline constexpr int kSharedExpertTensorDynamicBytes =
    kittens::MAX_SHARED_MEMORY - 1024;
inline constexpr int kActivationCtas = kIntermediate / kTileN;

// Every producer role precedes every consumer role. The core producer also
// computes activation, while the tensor path retains separate gate, up, and
// activation producers. The host residency guard keeps progress independent of
// block scheduling order.
inline constexpr int kCoreGateUpBegin = 0;
inline constexpr int kCoreDownBegin =
    kCoreGateUpBegin + kCoreGateCtas;
inline constexpr int kCoreRoleCtas = kCoreDownBegin + kCoreDownCtas;

inline constexpr int kTensorGateBegin = 0;
inline constexpr int kTensorUpBegin = kTensorGateBegin + kTensorGateCtas;
inline constexpr int kTensorActivationBegin =
    kTensorUpBegin + kTensorGateCtas;
inline constexpr int kTensorDownBegin =
    kTensorActivationBegin + kActivationCtas;
inline constexpr int kTensorRoleCtas = kTensorDownBegin + kTensorDownCtas;

inline constexpr std::uint64_t kGenerationWaitTimeoutClocks =
    5'000'000'000ULL;

static_assert(kIntermediate == 768);
static_assert(kIntermediate % kCoreGateColumnsPerCta == 0);
static_assert(kHiddenSize % kCoreDownColumnsPerCta == 0);
static_assert(kHiddenSize % kCoreInputChunk == 0);
static_assert(kIntermediate % kTileN == 0);
static_assert(kHiddenSize % kTileN == 0);
static_assert(kHiddenSize % kTileK == 0);
static_assert(kIntermediate % kTileK == 0);
static_assert(kActivationCtas == 6);
static_assert(kCoreGateUpBegin < kCoreDownBegin);
static_assert(kCoreRoleCtas == 136);
static_assert(kTensorGateBegin < kTensorUpBegin);
static_assert(kTensorUpBegin < kTensorActivationBegin);
static_assert(kTensorActivationBegin < kTensorDownBegin);
static_assert(kTensorRoleCtas == 74);

using tensor_input_tile = skinny_gemm::hidden_tile;
using tensor_weight_tile = skinny_gemm::weight_tile;
using tensor_output_tile = kittens::st_bf<kTileM, kTileN>;
using tensor_result_tile = kittens::st_fl<kTileM, kTileN>;
using tensor_accumulator_tile = skinny_gemm::accumulator_tile;
using tensor_input_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_input_tile>;
using tensor_weight_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_weight_tile>;
using tensor_output_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_output_tile>;

__device__ __forceinline__ float situ(
    const float gate,
    const float up
) {
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    return 4.0f * tanhf(gate * 0.25f) * sigmoid
         * 25.0f * tanhf(up / 25.0f);
}

struct RolePlan {
    int gate;
    int up;
    int activation;
    int down;

    constexpr int total() const {
        return gate + up + activation + down;
    }
};

inline constexpr RolePlan role_plan(const int active_tokens) {
    return capacity_bucket(active_tokens) <= kMaxCoreCapacity
        ? RolePlan{kCoreGateCtas, 0, 0, kCoreDownCtas}
        : RolePlan{
              kTensorGateCtas, kTensorGateCtas,
              kActivationCtas, kTensorDownCtas};
}

inline std::tuple<int, int, int, int, int> role_plan_for_testing(
    const std::int64_t active_tokens
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: shared expert role plan requires active_tokens in [1, ",
                kMaxTokens, "]");
    const RolePlan plan = role_plan(static_cast<int>(active_tokens));
    return {plan.gate, plan.up, plan.activation, plan.down, plan.total()};
}

inline std::tuple<int, int, int, int> timeout_metadata_for_testing() {
    return {
        kSharedTimeoutPhase,
        kSharedGateGeneration,
        kSharedUpGeneration,
        kSharedActivationGeneration};
}

inline void validate_residency(
    const std::int64_t active_tokens,
    const std::int64_t available_sms
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: shared expert residency requires active_tokens in [1, ",
                kMaxTokens, "]");
    TORCH_CHECK(available_sms >= 0,
                "MoK: shared expert residency requires a nonnegative SM count");
    const RolePlan plan = role_plan(static_cast<int>(active_tokens));
    TORCH_CHECK(
        available_sms >= plan.total(),
        "MoK: _kimi_k3_shared_experts requires all ", plan.total(),
        " role CTAs to co-reside, but the selected device exposes ",
        available_sms, " SMs");
}

__host__ __device__ inline constexpr bool generation_advanced(
    const std::uint32_t observed,
    const std::uint32_t consumed
) {
    const std::uint32_t difference = observed - consumed;
    return difference != 0u && difference < 0x80000000u;
}

__host__ __device__ inline constexpr bool wait_timed_out(
    const std::uint64_t started,
    const std::uint64_t current
) {
    return current - started >= kGenerationWaitTimeoutClocks;
}

static __device__ __forceinline__ std::uint32_t load_relaxed_gpu(
    const int *address
) {
    std::uint32_t value;
    asm volatile(
        "{ld.relaxed.gpu.global.u32 %0, [%1];}"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}

/// Take one role ticket; the caller supplies the release fence and CTA barrier.
static __device__ void arrive_and_publish(
    const Scratch &scratch,
    const int arrivals_index,
    const int generation_index,
    const int role_ctas
) {
    auto *const arrivals = reinterpret_cast<unsigned int *>(
        &scratch.phase[arrivals_index]);
    auto *const generation = reinterpret_cast<unsigned int *>(
        &scratch.phase[generation_index]);
    const unsigned int ticket = atomicAdd(arrivals, 1u);
    if (ticket == static_cast<unsigned int>(role_ctas - 1)) {
        atomicExch(arrivals, 0u);
        atomicAdd(generation, 1u);
    }
}

/// Release this role's global writes before announcing its new generation.
static __device__ void publish_phase(
    const Scratch &scratch,
    const int arrivals_index,
    const int generation_index,
    const int role_ctas
) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        arrive_and_publish(
            scratch, arrivals_index, generation_index, role_ctas);
    }
}

/// The combined core role publishes all three intermediates after one release.
///
/// There is no core activation arrival role. Its generation tag advances with
/// the same last producer so later tensor launches retain a common baseline
/// when capacity paths alternate on a reused workspace.
static __device__ void publish_core_intermediates(
    const Scratch &scratch,
    const int role_ctas
) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        auto *const arrivals = reinterpret_cast<unsigned int *>(
            &scratch.phase[kSharedGateArrivals]);
        const unsigned int ticket = atomicAdd(arrivals, 1u);
        if (ticket == static_cast<unsigned int>(role_ctas - 1)) {
            atomicExch(arrivals, 0u);
            atomicAdd(
                reinterpret_cast<unsigned int *>(
                    &scratch.phase[kSharedGateGeneration]),
                1u);
            atomicAdd(
                reinterpret_cast<unsigned int *>(
                    &scratch.phase[kSharedUpGeneration]),
                1u);
            atomicAdd(
                reinterpret_cast<unsigned int *>(
                    &scratch.phase[kSharedActivationGeneration]),
                1u);
        }
    }
}

/// Wait for this launch's producer generation, then acquire on every thread.
static __device__ void wait_for_phase(
    const Scratch &scratch,
    const int published_generation_index,
    const int consumed_generation_index
) {
    __shared__ std::uint32_t consumed_generation;
    if (threadIdx.x == 0) {
        const int *const published =
            &scratch.phase[published_generation_index];
        const int *const consumed =
            &scratch.phase[consumed_generation_index];
        consumed_generation = load_relaxed_gpu(consumed);
        const std::uint64_t started = clock64();
        while (!generation_advanced(
            load_relaxed_gpu(published), consumed_generation
        )) {
            if (wait_timed_out(started, clock64())) {
                atomicExch(
                    reinterpret_cast<unsigned int *>(
                        &scratch.phase[kSharedTimeoutPhase]),
                    static_cast<unsigned int>(published_generation_index));
                __threadfence_system();
                asm volatile("trap;");
            }
            __nanosleep(64);
        }
    }
    // Thread 0's observation converges the CTA; every consumer then executes a
    // device-scope acquire before any global intermediate read.
    __syncthreads();
    __threadfence();
}

/// Release the rank-local partial and close this scratch generation.
static __device__ void publish_down(
    const Scratch &scratch,
    const int consumer_ctas
) {
    publish_phase(
        scratch, kSharedDownArrivals, kSharedDownGeneration, consumer_ctas);
}

template<int CAPACITY>
static __device__ void gate_up_core(
    std::uint8_t *__restrict__ shared,
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ shared_gate_proj,
    const __nv_bfloat16 *__restrict__ shared_up_proj,
    const Scratch &scratch,
    const int column_block,
    const int active_tokens
) {
    static_assert(CAPACITY >= 1 && CAPACITY <= kMaxCoreCapacity);
    __nv_bfloat16 *const staged =
        reinterpret_cast<__nv_bfloat16 *>(shared);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread % 32;
    const int column_base =
        column_block * kCoreGateColumnsPerCta
        + warp * kCoreGateColumnsPerWarp;

    float gate[kCoreGateColumnsPerWarp][CAPACITY];
    float up[kCoreGateColumnsPerWarp][CAPACITY];
    #pragma unroll
    for (int column = 0; column < kCoreGateColumnsPerWarp; ++column) {
        #pragma unroll
        for (int row = 0; row < CAPACITY; ++row) {
            gate[column][row] = 0.0f;
            up[column][row] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < kHiddenSize; chunk += kCoreInputChunk) {
        __syncthreads();
        const int vectors_per_row = kCoreInputChunk / 8;
        const int staged_vectors = active_tokens * vectors_per_row;
        for (int index = thread; index < staged_vectors;
             index += kDecodeCtaThreads) {
            const int row = index / vectors_per_row;
            const int vector = index % vectors_per_row;
            *reinterpret_cast<float4 *>(
                staged + row * kCoreInputChunk + vector * 8) =
                    *reinterpret_cast<const float4 *>(
                        hidden_states
                        + static_cast<long long>(row) * kHiddenSize
                        + chunk + vector * 8);
        }
        __syncthreads();

        #pragma unroll
        for (int column = 0; column < kCoreGateColumnsPerWarp; ++column) {
            const long long weight_offset =
                static_cast<long long>(column_base + column) * kHiddenSize
                + chunk;
            for (int k = lane * 8; k < kCoreInputChunk; k += 32 * 8) {
                const float4 gate_weight =
                    *reinterpret_cast<const float4 *>(
                        shared_gate_proj + weight_offset + k);
                const float4 up_weight =
                    *reinterpret_cast<const float4 *>(
                        shared_up_proj + weight_offset + k);
                #pragma unroll
                for (int row = 0; row < CAPACITY; ++row) {
                    if (row < active_tokens) {
                        const float4 activation =
                            *reinterpret_cast<const float4 *>(
                                staged + row * kCoreInputChunk + k);
                        gate[column][row] = accumulate_bf16_octet(
                            activation, gate_weight, gate[column][row]);
                        up[column][row] = accumulate_bf16_octet(
                            activation, up_weight, up[column][row]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int column = 0; column < kCoreGateColumnsPerWarp; ++column) {
        #pragma unroll
        for (int row = 0; row < CAPACITY; ++row) {
            float gate_value = gate[column][row];
            float up_value = up[column][row];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                gate_value +=
                    __shfl_down_sync(0xffffffffu, gate_value, offset);
                up_value += __shfl_down_sync(0xffffffffu, up_value, offset);
            }
            if (lane == 0 && row < active_tokens) {
                const long long output =
                    static_cast<long long>(row) * kIntermediate
                    + column_base + column;
                // Match the BF16 linear-module boundary before FP32 SiTU.
                const __nv_bfloat16 gate_bf16 =
                    __float2bfloat16(gate_value);
                const __nv_bfloat16 up_bf16 =
                    __float2bfloat16(up_value);
                scratch.shared_gate[output] = gate_bf16;
                scratch.shared_up[output] = up_bf16;
                scratch.shared_activated[output] = __float2bfloat16(situ(
                    __bfloat162float(gate_bf16),
                    __bfloat162float(up_bf16)));
            }
        }
    }
}

static __device__ void activate_shared_tile(
    const Scratch &scratch,
    const int column_block,
    const int active_tokens
) {
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < active_tokens * kTileN;
         index += kDecodeCtaThreads) {
        const int row = index / kTileN;
        const int column = column_block * kTileN + index % kTileN;
        const long long offset =
            static_cast<long long>(row) * kIntermediate + column;
        const float gate = __bfloat162float(scratch.shared_gate[offset]);
        const float up = __bfloat162float(scratch.shared_up[offset]);
        scratch.shared_activated[offset] = __float2bfloat16(situ(gate, up));
    }
}

static __device__ void mask_inactive_collective_rows(
    __nv_bfloat16 *__restrict__ collective_buffer,
    const int column_base,
    const int columns,
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
        collective_buffer[
            static_cast<long long>(row) * kCollectiveColumns
            + kLatentSize + column] = zero;
    }
}

template<int CAPACITY>
static __device__ void down_core(
    std::uint8_t *__restrict__ shared,
    const Scratch &scratch,
    const __nv_bfloat16 *__restrict__ shared_down_proj,
    __nv_bfloat16 *__restrict__ collective_buffer,
    const int column_block,
    const int active_tokens,
    const int tokens
) {
    static_assert(CAPACITY >= 1 && CAPACITY <= kMaxCoreCapacity);
    __nv_bfloat16 *const staged =
        reinterpret_cast<__nv_bfloat16 *>(shared);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread % 32;
    const int column_base =
        column_block * kCoreDownColumnsPerCta
        + warp * kCoreDownColumnsPerWarp;
    constexpr int vectors_per_row = kIntermediate / 8;

    mask_inactive_collective_rows(
        collective_buffer,
        column_block * kCoreDownColumnsPerCta,
        kCoreDownColumnsPerCta,
        active_tokens,
        tokens);

    for (int index = thread; index < active_tokens * vectors_per_row;
         index += kDecodeCtaThreads) {
        const int row = index / vectors_per_row;
        const int vector = index % vectors_per_row;
        *reinterpret_cast<float4 *>(
            staged + row * kIntermediate + vector * 8) =
                *reinterpret_cast<const float4 *>(
                    scratch.shared_activated
                    + static_cast<long long>(row) * kIntermediate
                    + vector * 8);
    }
    __syncthreads();

    float accumulator[kCoreDownColumnsPerWarp][CAPACITY];
    #pragma unroll
    for (int column = 0; column < kCoreDownColumnsPerWarp; ++column) {
        #pragma unroll
        for (int row = 0; row < CAPACITY; ++row) {
            accumulator[column][row] = 0.0f;
        }
    }
    #pragma unroll
    for (int column = 0; column < kCoreDownColumnsPerWarp; ++column) {
        const __nv_bfloat16 *const weight_row =
            shared_down_proj
            + static_cast<long long>(column_base + column) * kIntermediate;
        for (int k = lane * 8; k < kIntermediate; k += 32 * 8) {
            const float4 weight =
                *reinterpret_cast<const float4 *>(weight_row + k);
            #pragma unroll
            for (int row = 0; row < CAPACITY; ++row) {
                if (row < active_tokens) {
                    accumulator[column][row] = accumulate_bf16_octet(
                        *reinterpret_cast<const float4 *>(
                            staged + row * kIntermediate + k),
                        weight,
                        accumulator[column][row]);
                }
            }
        }
    }

    #pragma unroll
    for (int column = 0; column < kCoreDownColumnsPerWarp; ++column) {
        #pragma unroll
        for (int row = 0; row < CAPACITY; ++row) {
            float value = accumulator[column][row];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffu, value, offset);
            }
            if (lane == 0 && row < active_tokens) {
                collective_buffer[
                    static_cast<long long>(row) * kCollectiveColumns
                    + kLatentSize + column_base + column] =
                        __float2bfloat16(value);
            }
        }
    }
}

static __device__ void project_tensor(
    int *__restrict__ shared_raw,
    const tensor_input_layout &input,
    const tensor_weight_layout &weight,
    const tensor_output_layout &output,
    const int column_block,
    const int k_iterations
) {
    using namespace kittens;
    tma_swizzle_allocator allocator(shared_raw);
    tensor_input_tile (&input_tiles)[kTensorStages] =
        allocator.allocate<tensor_input_tile, kTensorStages>();
    tensor_weight_tile (&weight_tiles)[kTensorStages] =
        allocator.allocate<tensor_weight_tile, kTensorStages>();
    tensor_output_tile (&output_staging) =
        allocator.allocate<tensor_output_tile>();

    __shared__ semaphore inputs_arrived[kTensorStages];
    __shared__ semaphore inputs_finished[kTensorStages];
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < kTensorStages; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 1, 0);
        }
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tensor_pool{};
    if (warpgroup::groupid() == 0) {
        tensor_accumulator_tile accumulator =
            tensor_pool.allocate<tensor_accumulator_tile>(0);
        const int warpgroup_lane = warpgroup::laneid();

        for (int iteration = 0; iteration < k_iterations; ++iteration) {
            const int stage = iteration % kTensorStages;
            const int round = iteration / kTensorStages;
            if (warpgroup_lane == 0) {
                wait(inputs_finished[stage], (round + 1) % 2);
                tma::expect_bytes(
                    inputs_arrived[stage],
                    sizeof(tensor_input_tile) + sizeof(tensor_weight_tile));
                tma::load_async(
                    input_tiles[stage], input, {0, iteration},
                    inputs_arrived[stage]);
                tma::load_async(
                    weight_tiles[stage], weight, {column_block, iteration},
                    inputs_arrived[stage]);
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

        if (warpgroup_lane == 0) {
            detail::tcgen05::commit<1>(compute_done);
        }
        wait(compute_done, 0);

        rt_bf<kTileM / 4, kTileN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(output_staging, result);
        warpgroup::sync(1);
        if (warpgroup_lane == 0) {
            tma::store_async(output, output_staging, {0, column_block});
            tma::store_async_wait();
        }
    }
    __syncthreads();
}

static __device__ void down_tensor(
    int *__restrict__ shared_raw,
    const tensor_input_layout &activated,
    const tensor_weight_layout &down_weight,
    __nv_bfloat16 *__restrict__ collective_buffer,
    const int column_block,
    const int active_tokens,
    const int tokens
) {
    using namespace kittens;
    tma_swizzle_allocator allocator(shared_raw);
    tensor_input_tile (&activation_tiles)[kTensorStages] =
        allocator.allocate<tensor_input_tile, kTensorStages>();
    tensor_weight_tile (&weight_tiles)[kTensorStages] =
        allocator.allocate<tensor_weight_tile, kTensorStages>();
    tensor_result_tile (&result_shared) =
        allocator.allocate<tensor_result_tile>();

    __shared__ semaphore inputs_arrived[kTensorStages];
    __shared__ semaphore inputs_finished[kTensorStages];
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < kTensorStages; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 1, 0);
        }
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tensor_pool{};
    if (warpgroup::groupid() == 0) {
        tensor_accumulator_tile accumulator =
            tensor_pool.allocate<tensor_accumulator_tile>(0);
        const int warpgroup_lane = warpgroup::laneid();

        for (int iteration = 0; iteration < kTensorDownKIterations;
             ++iteration) {
            const int stage = iteration % kTensorStages;
            const int round = iteration / kTensorStages;
            if (warpgroup_lane == 0) {
                wait(inputs_finished[stage], (round + 1) % 2);
                tma::expect_bytes(
                    inputs_arrived[stage],
                    sizeof(tensor_input_tile) + sizeof(tensor_weight_tile));
                tma::load_async(
                    activation_tiles[stage],
                    activated,
                    {0, iteration},
                    inputs_arrived[stage]);
                tma::load_async(
                    weight_tiles[stage],
                    down_weight,
                    {column_block, iteration},
                    inputs_arrived[stage]);
            }
            wait(inputs_arrived[stage], round % 2);
            if (warpgroup_lane == 0) {
                if (iteration == 0) {
                    mm_ABt(
                        accumulator,
                        activation_tiles[stage],
                        weight_tiles[stage],
                        inputs_finished[stage]);
                } else {
                    mma_ABt(
                        accumulator,
                        activation_tiles[stage],
                        weight_tiles[stage],
                        inputs_finished[stage]);
                }
            }
        }

        if (warpgroup_lane == 0) {
            detail::tcgen05::commit<1>(compute_done);
        }
        wait(compute_done, 0);

        rt_fl<kTileM / 4, kTileN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(result_shared, result);
        warpgroup::sync(1);
    }
    __syncthreads();

    mask_inactive_collective_rows(
        collective_buffer,
        column_block * kTileN,
        kTileN,
        active_tokens,
        tokens);
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < active_tokens * kTileN;
         index += kDecodeCtaThreads) {
        const int row = index / kTileN;
        const int column = index % kTileN;
        const long long output =
            static_cast<long long>(row) * kCollectiveColumns
            + kLatentSize + column_block * kTileN + column;
        collective_buffer[output] =
            __float2bfloat16(result_shared[{row, column}]);
    }
}

template<int CAPACITY>
__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void shared_experts_core_kernel(
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ shared_gate_proj,
    const __nv_bfloat16 *__restrict__ shared_up_proj,
    const __nv_bfloat16 *__restrict__ shared_down_proj,
    std::uint8_t *__restrict__ scratch_bytes,
    __nv_bfloat16 *__restrict__ collective_buffer,
    const int active_tokens,
    const int tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);
    if (block < kCoreDownBegin) {
        gate_up_core<CAPACITY>(
            reinterpret_cast<std::uint8_t *>(shared_raw), hidden_states,
            shared_gate_proj, shared_up_proj, scratch, block,
            active_tokens);
        publish_core_intermediates(scratch, kCoreGateCtas);
        return;
    }

    wait_for_phase(
        scratch, kSharedGateGeneration, kSharedDownGeneration);
    wait_for_phase(
        scratch, kSharedUpGeneration, kSharedDownGeneration);
    down_core<CAPACITY>(
        reinterpret_cast<std::uint8_t *>(shared_raw), scratch,
        shared_down_proj, collective_buffer, block - kCoreDownBegin,
        active_tokens, tokens);
    publish_down(scratch, kCoreDownCtas);
}

__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void shared_experts_tensor_kernel(
    const __grid_constant__ tensor_input_layout hidden,
    const __grid_constant__ tensor_weight_layout shared_gate_proj,
    const __grid_constant__ tensor_weight_layout shared_up_proj,
    const __grid_constant__ tensor_weight_layout shared_down_proj,
    const __grid_constant__ tensor_output_layout gate_output,
    const __grid_constant__ tensor_output_layout up_output,
    const __grid_constant__ tensor_input_layout activated,
    std::uint8_t *__restrict__ scratch_bytes,
    __nv_bfloat16 *__restrict__ collective_buffer,
    const int active_tokens,
    const int tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);
    if (block < kTensorUpBegin) {
        project_tensor(
            shared_raw, hidden, shared_gate_proj, gate_output,
            block - kTensorGateBegin, kTensorGateKIterations);
        publish_phase(
            scratch,
            kSharedGateArrivals,
            kSharedGateGeneration,
            kTensorGateCtas);
        return;
    }
    if (block < kTensorActivationBegin) {
        project_tensor(
            shared_raw, hidden, shared_up_proj, up_output,
            block - kTensorUpBegin, kTensorGateKIterations);
        publish_phase(
            scratch,
            kSharedUpArrivals,
            kSharedUpGeneration,
            kTensorGateCtas);
        return;
    }
    if (block < kTensorDownBegin) {
        wait_for_phase(
            scratch, kSharedGateGeneration,
            kSharedActivationGeneration);
        wait_for_phase(
            scratch, kSharedUpGeneration,
            kSharedActivationGeneration);
        activate_shared_tile(
            scratch, block - kTensorActivationBegin, active_tokens);
        publish_phase(
            scratch,
            kSharedActivationArrivals,
            kSharedActivationGeneration,
            kActivationCtas);
        return;
    }

    wait_for_phase(
        scratch, kSharedActivationGeneration, kSharedDownGeneration);
    down_tensor(
        shared_raw, activated, shared_down_proj, collective_buffer,
        block - kTensorDownBegin, active_tokens, tokens);
    publish_down(scratch, kTensorDownCtas);
}

template<int CAPACITY>
static __host__ void launch_core(
    const __nv_bfloat16 *hidden_states,
    const __nv_bfloat16 *shared_gate_proj,
    const __nv_bfloat16 *shared_up_proj,
    const __nv_bfloat16 *shared_down_proj,
    std::uint8_t *scratch,
    __nv_bfloat16 *collective_buffer,
    const int active_tokens,
    const int tokens
) {
    shared_experts_core_kernel<CAPACITY>
        <<<kCoreRoleCtas, kDecodeCtaThreads,
           kSharedExpertCoreDynamicBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            hidden_states, shared_gate_proj, shared_up_proj, shared_down_proj,
            scratch, collective_buffer, active_tokens, tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __host__ void launch_shared_experts(
    const at::Tensor &hidden_states,
    const at::Tensor &shared_gate_proj,
    const at::Tensor &shared_up_proj,
    const at::Tensor &shared_down_proj,
    const at::Tensor &scratch,
    const at::Tensor &collective_buffer,
    const int active_tokens,
    const int available_sms
) {
    const auto *const hidden =
        reinterpret_cast<const __nv_bfloat16 *>(hidden_states.data_ptr());
    const auto *const gate =
        reinterpret_cast<const __nv_bfloat16 *>(shared_gate_proj.data_ptr());
    const auto *const up =
        reinterpret_cast<const __nv_bfloat16 *>(shared_up_proj.data_ptr());
    const auto *const down =
        reinterpret_cast<const __nv_bfloat16 *>(shared_down_proj.data_ptr());
    auto *const workspace =
        reinterpret_cast<std::uint8_t *>(scratch.data_ptr());
    auto *const collective =
        reinterpret_cast<__nv_bfloat16 *>(collective_buffer.data_ptr());
    const int tokens = static_cast<int>(hidden_states.size(0));
    validate_residency(active_tokens, available_sms);

    switch (capacity_bucket(active_tokens)) {
        case 1:
            launch_core<1>(hidden, gate, up, down, workspace, collective,
                           active_tokens, tokens);
            return;
        case 2:
            launch_core<2>(hidden, gate, up, down, workspace, collective,
                           active_tokens, tokens);
            return;
        case 4:
            launch_core<4>(hidden, gate, up, down, workspace, collective,
                           active_tokens, tokens);
            return;
        case 8:
            launch_core<8>(hidden, gate, up, down, workspace, collective,
                           active_tokens, tokens);
            return;
        default:
            break;
    }

    const tensor_input_layout hidden_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(hidden)),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kHiddenSize)};
    const tensor_weight_layout gate_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(gate)),
        nullptr, nullptr, static_cast<size_t>(kIntermediate),
        static_cast<size_t>(kHiddenSize)};
    const tensor_weight_layout up_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(up)),
        nullptr, nullptr, static_cast<size_t>(kIntermediate),
        static_cast<size_t>(kHiddenSize)};
    const tensor_weight_layout down_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(down)),
        nullptr, nullptr, static_cast<size_t>(kHiddenSize),
        static_cast<size_t>(kIntermediate)};
    const Scratch scratch_view_value = scratch_view(workspace);
    const tensor_output_layout gate_output_view{
        reinterpret_cast<kittens::bf16 *>(scratch_view_value.shared_gate),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kIntermediate)};
    const tensor_output_layout up_output_view{
        reinterpret_cast<kittens::bf16 *>(scratch_view_value.shared_up),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kIntermediate)};
    const tensor_input_layout activated_view{
        reinterpret_cast<kittens::bf16 *>(
            scratch_view_value.shared_activated),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kIntermediate)};

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        shared_experts_tensor_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedExpertTensorDynamicBytes));
    shared_experts_tensor_kernel
                <<<kTensorRoleCtas, kDecodeCtaThreads,
                   kSharedExpertTensorDynamicBytes,
                   at::cuda::getCurrentCUDAStream()>>>(
                    hidden_view, gate_view, up_view, down_view,
                    gate_output_view, up_output_view, activated_view,
                    workspace, collective,
                    active_tokens, tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace shared_experts
}  // namespace kimi_k3_decode
