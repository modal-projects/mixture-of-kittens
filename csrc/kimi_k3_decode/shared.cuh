#pragma once

#include "kittens.cuh"

#include "skinny_gemm.cuh"
#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <cstdint>

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
inline constexpr int kCoreSharedBytes =
    kMaxCoreCapacity * kCoreInputChunk * sizeof(__nv_bfloat16);

inline constexpr int kTileM = skinny_gemm::kTileM;
inline constexpr int kTileN = skinny_gemm::kTileN;
inline constexpr int kTileK = skinny_gemm::kTileK;
inline constexpr int kTensorGateCtas = kIntermediate / kTileN;
inline constexpr int kTensorDownCtas = kHiddenSize / kTileN;
inline constexpr int kTensorGateKIterations = kHiddenSize / kTileK;
inline constexpr int kTensorDownKIterations = kIntermediate / kTileK;
inline constexpr int kTensorSharedBytes = kittens::MAX_SHARED_MEMORY - 1024;

static_assert(kIntermediate == 768);
static_assert(kIntermediate % kCoreGateColumnsPerCta == 0);
static_assert(kHiddenSize % kCoreDownColumnsPerCta == 0);
static_assert(kHiddenSize % kCoreInputChunk == 0);
static_assert(kIntermediate % kTileN == 0);
static_assert(kHiddenSize % kTileN == 0);
static_assert(kHiddenSize % kTileK == 0);
static_assert(kIntermediate % kTileK == 0);
// A B300 has enough SMs to make every direct-path role resident together, so
// consumers cannot occupy the machine while the producer CTAs are unscheduled.
static_assert(kCoreGateCtas + kCoreDownCtas == 136);
static_assert(kTensorGateCtas + kTensorDownCtas == 62);

using tensor_input_tile = skinny_gemm::hidden_tile;
using tensor_weight_tile = skinny_gemm::weight_tile;
using tensor_result_tile = kittens::st_fl<kTileM, kTileN>;
using tensor_accumulator_tile = skinny_gemm::accumulator_tile;
using tensor_input_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_input_tile>;
using tensor_weight_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_weight_tile>;

__device__ __forceinline__ float situ(
    const float gate,
    const float up
) {
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    return 4.0f * tanhf(gate * 0.25f) * sigmoid
         * 25.0f * tanhf(up * 0.04f);
}

/// Release all BF16 activation writes before announcing the new generation.
static __device__ void publish_gate_up(
    const Scratch &scratch,
    const int producer_ctas
) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int ticket =
            atomicAdd(&scratch.phase[kSharedGateUpArrivals], 1);
        if (ticket == producer_ctas - 1) {
            atomicExch(&scratch.phase[kSharedGateUpArrivals], 0);
            atomicAdd(&scratch.phase[kSharedGateUpGeneration], 1);
        }
    }
}

/// Acquire the generation produced by this launch before reading activation.
static __device__ void wait_for_gate_up(const Scratch &scratch) {
    __shared__ int consumed_generation;
    if (threadIdx.x == 0) {
        consumed_generation =
            atomicAdd(&scratch.phase[kSharedDownGeneration], 0);
        while (
            atomicAdd(&scratch.phase[kSharedGateUpGeneration], 0)
            <= consumed_generation
        ) {
            __nanosleep(64);
        }
    }
    __syncthreads();
}

/// Release the rank-local partial and close this scratch generation.
static __device__ void publish_down(
    const Scratch &scratch,
    const int consumer_ctas
) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int ticket = atomicAdd(&scratch.phase[kSharedDownArrivals], 1);
        if (ticket == consumer_ctas - 1) {
            atomicExch(&scratch.phase[kSharedDownArrivals], 0);
            atomicAdd(&scratch.phase[kSharedDownGeneration], 1);
        }
    }
}

template<int CAPACITY>
static __device__ void gate_up_core(
    std::uint8_t *__restrict__ shared,
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ shared_gate_proj,
    const __nv_bfloat16 *__restrict__ shared_up_proj,
    const Scratch &scratch,
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

    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
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
                scratch.shared_gate[output] =
                    __float2bfloat16(gate_value);
                scratch.shared_up[output] =
                    __float2bfloat16(up_value);
                scratch.shared_activated[output] =
                    __float2bfloat16(situ(gate_value, up_value));
            }
        }
        if (lane == 0) {
            for (int row = active_tokens; row < tokens; ++row) {
                const long long output =
                    static_cast<long long>(row) * kIntermediate
                    + column_base + column;
                scratch.shared_gate[output] = zero;
                scratch.shared_up[output] = zero;
                scratch.shared_activated[output] = zero;
            }
        }
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

    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
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
        if (lane == 0) {
            for (int row = active_tokens; row < tokens; ++row) {
                collective_buffer[
                    static_cast<long long>(row) * kCollectiveColumns
                    + kLatentSize + column_base + column] = zero;
            }
        }
    }
}

static __device__ void gate_up_tensor(
    int *__restrict__ shared_raw,
    const tensor_input_layout &hidden,
    const tensor_weight_layout &gate_weight,
    const tensor_weight_layout &up_weight,
    const Scratch &scratch,
    const int column_block,
    const int active_tokens,
    const int tokens
) {
    using namespace kittens;
    tma_swizzle_allocator allocator(shared_raw);
    tensor_input_tile (&hidden_tile) =
        allocator.allocate<tensor_input_tile>();
    tensor_weight_tile (&gate_tile) =
        allocator.allocate<tensor_weight_tile>();
    tensor_weight_tile (&up_tile) =
        allocator.allocate<tensor_weight_tile>();
    tensor_result_tile (&gate_result) =
        allocator.allocate<tensor_result_tile>();
    tensor_result_tile (&up_result) =
        allocator.allocate<tensor_result_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tensor_pool{};
    tensor_accumulator_tile gate_accumulator =
        tensor_pool.allocate<tensor_accumulator_tile>(0);
    tensor_accumulator_tile up_accumulator =
        tensor_pool.allocate<tensor_accumulator_tile>(128);

    for (int iteration = 0; iteration < kTensorGateKIterations; ++iteration) {
        if (threadIdx.x == 0) {
            tma::expect_bytes(
                inputs_arrived,
                sizeof(tensor_input_tile) + 2 * sizeof(tensor_weight_tile));
            tma::load_async(
                hidden_tile, hidden, {0, iteration}, inputs_arrived);
            tma::load_async(
                gate_tile, gate_weight, {column_block, iteration},
                inputs_arrived);
            tma::load_async(
                up_tile, up_weight, {column_block, iteration},
                inputs_arrived);
        }
        wait(inputs_arrived, iteration % 2);
        if (threadIdx.x == 0) {
            if (iteration == 0) {
                mm_ABt(gate_accumulator, hidden_tile, gate_tile);
                mm_ABt(up_accumulator, hidden_tile, up_tile);
            } else {
                mma_ABt(gate_accumulator, hidden_tile, gate_tile);
                mma_ABt(up_accumulator, hidden_tile, up_tile);
            }
            detail::tcgen05::commit<1>(compute_done);
        }
        wait(compute_done, iteration % 2);
        __syncthreads();
    }

    if (warpgroup::groupid() == 0) {
        rt_fl<kTileM / 4, kTileN> result;
        warpgroup::load_async(result, gate_accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(gate_result, result);
        warpgroup::sync(1);
        warpgroup::load_async(result, up_accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(up_result, result);
        warpgroup::sync(1);
    }
    __syncthreads();

    const int thread = static_cast<int>(threadIdx.x);
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int index = thread; index < tokens * kTileN;
         index += kDecodeCtaThreads) {
        const int row = index / kTileN;
        const int column = index % kTileN;
        const long long output =
            static_cast<long long>(row) * kIntermediate
            + column_block * kTileN + column;
        if (row < active_tokens) {
            const float gate_value = gate_result[{row, column}];
            const float up_value = up_result[{row, column}];
            scratch.shared_gate[output] = __float2bfloat16(gate_value);
            scratch.shared_up[output] = __float2bfloat16(up_value);
            scratch.shared_activated[output] =
                __float2bfloat16(situ(gate_value, up_value));
        } else {
            scratch.shared_gate[output] = zero;
            scratch.shared_up[output] = zero;
            scratch.shared_activated[output] = zero;
        }
    }
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
    tensor_input_tile (&activation_tile) =
        allocator.allocate<tensor_input_tile>();
    tensor_weight_tile (&weight_tile) =
        allocator.allocate<tensor_weight_tile>();
    tensor_result_tile (&result_shared) =
        allocator.allocate<tensor_result_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tensor_pool{};
    tensor_accumulator_tile accumulator =
        tensor_pool.allocate<tensor_accumulator_tile>(0);
    for (int iteration = 0; iteration < kTensorDownKIterations; ++iteration) {
        if (threadIdx.x == 0) {
            tma::expect_bytes(
                inputs_arrived,
                sizeof(tensor_input_tile) + sizeof(tensor_weight_tile));
            tma::load_async(
                activation_tile, activated, {0, iteration}, inputs_arrived);
            tma::load_async(
                weight_tile, down_weight, {column_block, iteration},
                inputs_arrived);
        }
        wait(inputs_arrived, iteration % 2);
        if (threadIdx.x == 0) {
            if (iteration == 0) {
                mm_ABt(accumulator, activation_tile, weight_tile);
            } else {
                mma_ABt(accumulator, activation_tile, weight_tile);
            }
            detail::tcgen05::commit<1>(compute_done);
        }
        wait(compute_done, iteration % 2);
        __syncthreads();
    }

    if (warpgroup::groupid() == 0) {
        rt_fl<kTileM / 4, kTileN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(result_shared, result);
        warpgroup::sync(1);
    }
    __syncthreads();

    const int thread = static_cast<int>(threadIdx.x);
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int index = thread; index < tokens * kTileN;
         index += kDecodeCtaThreads) {
        const int row = index / kTileN;
        const int column = index % kTileN;
        const long long output =
            static_cast<long long>(row) * kCollectiveColumns
            + kLatentSize + column_block * kTileN + column;
        collective_buffer[output] =
            row < active_tokens
                ? __float2bfloat16(result_shared[{row, column}])
                : zero;
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
    if (block < kCoreGateCtas) {
        gate_up_core<CAPACITY>(
            reinterpret_cast<std::uint8_t *>(shared_raw), hidden_states,
            shared_gate_proj, shared_up_proj, scratch, block,
            active_tokens, tokens);
        publish_gate_up(scratch, kCoreGateCtas);
        return;
    }

    wait_for_gate_up(scratch);
    down_core<CAPACITY>(
        reinterpret_cast<std::uint8_t *>(shared_raw), scratch,
        shared_down_proj, collective_buffer, block - kCoreGateCtas,
        active_tokens, tokens);
    publish_down(scratch, kCoreDownCtas);
}

__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void shared_experts_tensor_kernel(
    const __grid_constant__ tensor_input_layout hidden,
    const __grid_constant__ tensor_weight_layout shared_gate_proj,
    const __grid_constant__ tensor_weight_layout shared_up_proj,
    const __grid_constant__ tensor_weight_layout shared_down_proj,
    const __grid_constant__ tensor_input_layout activated,
    std::uint8_t *__restrict__ scratch_bytes,
    __nv_bfloat16 *__restrict__ collective_buffer,
    const int active_tokens,
    const int tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);
    if (block < kTensorGateCtas) {
        gate_up_tensor(
            shared_raw, hidden, shared_gate_proj, shared_up_proj, scratch,
            block, active_tokens, tokens);
        publish_gate_up(scratch, kTensorGateCtas);
        return;
    }

    wait_for_gate_up(scratch);
    down_tensor(
        shared_raw, activated, shared_down_proj, collective_buffer,
        block - kTensorGateCtas, active_tokens, tokens);
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
        <<<kCoreGateCtas + kCoreDownCtas, kDecodeCtaThreads,
           kCoreSharedBytes, at::cuda::getCurrentCUDAStream()>>>(
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
    const int active_tokens
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
    const tensor_input_layout activated_view{
        reinterpret_cast<kittens::bf16 *>(
            scratch_view_value.shared_activated),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kIntermediate)};

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        shared_experts_tensor_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kTensorSharedBytes));
    shared_experts_tensor_kernel
                <<<kTensorGateCtas + kTensorDownCtas, kDecodeCtaThreads,
                   kTensorSharedBytes,
                   at::cuda::getCurrentCUDAStream()>>>(
                    hidden_view, gate_view, up_view, down_view, activated_view,
                    workspace, collective,
                    active_tokens, tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace shared_experts
}  // namespace kimi_k3_decode
