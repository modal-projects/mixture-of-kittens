#pragma once

// Benchmark-only output-channel-M/token-N tail candidate.
//
// Production's `shard_tensor` computes token-M by output-N in one 128x128
// accumulator per output tile. This probe preserves the full TP8 tail launch
// and its BF16 boundaries, but flips the rank-local contraction to weight-M by
// token-N. M16 uses m128n16k64, M32 uses m128n32k64, and M128 uses four N32
// token tiles, exposing otherwise-idle resident CTAs without introducing a
// split-K partial reduction.

#include "collectives.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <tuple>

namespace kimi_k3_decode {
namespace tail {
namespace m128n_probe {

inline constexpr int kMmaM = 128;
inline constexpr int kMmaK = 64;
inline constexpr int kOutputTiles = kShardColumns / kMmaM;
inline constexpr int kProbeDynamicBytes = kTailTensorDynamicBytes;

static_assert(kMmaM == kTileM);
static_assert(kMmaK == kTileK);
static_assert(kOutputTiles == kTensorShardCtas);

template<int TOKEN_TILE_N>
using input_tile = kittens::st_bf<TOKEN_TILE_N, kMmaK>;

using weight_tile = kittens::st_bf<kMmaM, kMmaK>;

template<int TOKEN_TILE_N>
using result_tile = kittens::st_fl<kMmaM, TOKEN_TILE_N>;

template<int TOKEN_TILE_N>
using accumulator_tile = kittens::tt_fl<kMmaM, TOKEN_TILE_N>;

template<int TOKEN_TILE_N>
using input_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, input_tile<TOKEN_TILE_N>>;

using weight_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, weight_tile>;

template<int TOKEN_TILE_N>
inline constexpr int token_tiles(const int active_tokens) {
    static_assert(TOKEN_TILE_N == 16 || TOKEN_TILE_N == 32);
    return active_tokens / TOKEN_TILE_N;
}

template<int TOKEN_TILE_N>
inline constexpr int shard_ctas(const int active_tokens) {
    return kOutputTiles * token_tiles<TOKEN_TILE_N>(active_tokens);
}

template<int TOKEN_TILE_N>
static __device__ void shard_tensor_m128n(
    int *__restrict__ const shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const input_layout<TOKEN_TILE_N> &normalized,
    const weight_layout &latent_up_proj,
    const Scratch &scratch,
    __nv_bfloat16 *__restrict__ const mailbox_multicast,
    const int shard_unit,
    const int tp_rank,
    const int active_tokens,
    const TailClocks &clocks
) {
    using namespace kittens;
    static_assert(TOKEN_TILE_N == 16 || TOKEN_TILE_N == 32);

    unsigned long long mark = clocks.now();
    const int token_tile_count = token_tiles<TOKEN_TILE_N>(active_tokens);
    const int column_block = shard_unit / token_tile_count;
    const int token_block = shard_unit % token_tile_count;

    tma_swizzle_allocator allocator(shared_raw);
    input_tile<TOKEN_TILE_N> (&input_tiles)[kStages] =
        allocator.allocate<input_tile<TOKEN_TILE_N>, kStages>();
    weight_tile (&weight_tiles)[kStages] =
        allocator.allocate<weight_tile, kStages>();
    result_tile<TOKEN_TILE_N> (&result_shared) =
        allocator.allocate<result_tile<TOKEN_TILE_N>>();

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
        accumulator_tile<TOKEN_TILE_N> accumulator =
            tensor_pool.allocate<accumulator_tile<TOKEN_TILE_N>>(0);
        const int warpgroup_lane = warpgroup::laneid();

        for (int iteration = 0; iteration < kTensorKIterations; ++iteration) {
            const int stage = iteration % kStages;
            const int round = iteration / kStages;
            if (warpgroup_lane == 0) {
                wait(inputs_finished[stage], (round + 1) % 2);
                tma::expect_bytes(
                    inputs_arrived[stage],
                    sizeof(input_tile<TOKEN_TILE_N>) + sizeof(weight_tile));
                tma::load_async(
                    weight_tiles[stage], latent_up_proj,
                    {column_block, iteration}, inputs_arrived[stage]);
                tma::load_async(
                    input_tiles[stage], normalized,
                    {token_block, iteration}, inputs_arrived[stage]);
            }
            wait(inputs_arrived[stage], round % 2);
            if (warpgroup_lane == 0) {
                if (iteration == 0) {
                    mm_ABt(
                        accumulator, weight_tiles[stage], input_tiles[stage],
                        inputs_finished[stage]);
                } else {
                    mma_ABt(
                        accumulator, weight_tiles[stage], input_tiles[stage],
                        inputs_finished[stage]);
                }
            }
        }

        if (warpgroup_lane == 0) detail::tcgen05::commit<1>(compute_done);
        wait(compute_done, 0);

        rt_fl<kMmaM / 4, TOKEN_TILE_N> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(result_shared, result);
        warpgroup::sync(1);
    }
    __syncthreads();
    mark = clocks.lap(kTailClockLatentUpShardMma, mark);

    constexpr int groups_per_output_tile = kMmaM / kOctetLanes;
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread;
         index < TOKEN_TILE_N * groups_per_output_tile;
         index += kDecodeCtaThreads) {
        const int local_row = index / groups_per_output_tile;
        const int row = token_block * TOKEN_TILE_N + local_row;
        const int tile_group = index % groups_per_output_tile;
        const int column = column_block * kMmaM
            + tile_group * kOctetLanes;
        Octet value;
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int tile_column = tile_group * kOctetLanes + 2 * pair;
            const long long beta =
                static_cast<long long>(row) * kShardColumns
                + column + 2 * pair;
            value.pair[pair] = pack_bf16(
                result_shared[{tile_column, local_row}]
                    + __bfloat162float(scratch.tail_shared_shard[beta]),
                result_shared[{tile_column + 1, local_row}]
                    + __bfloat162float(
                        scratch.tail_shared_shard[beta + 1]));
        }
        publish_shard_octet(
            mailbox_multicast, value, row, tp_rank, column);
    }
    clocks.lap(kTailClockMailboxMulticast, mark);
}

template<int TOKEN_TILE_N>
__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void kimi_k3_tail_m128n_probe_kernel(
    const __nv_bfloat16 *__restrict__ routed_latent_rmsnorm_weight,
    const __grid_constant__ input_layout<TOKEN_TILE_N> normalized,
    const __grid_constant__ weight_layout latent_up_proj,
    __nv_bfloat16 *__restrict__ collective_multicast,
    __nv_bfloat16 *__restrict__ mailbox_multicast,
    std::uint32_t *__restrict__ barrier_multicast,
    const std::uint32_t *__restrict__ barrier_local,
    unsigned int *__restrict__ barrier_target,
    std::uint8_t *__restrict__ scratch_bytes,
    int *__restrict__ error_flag,
    const int tp_rank,
    const int active_tokens,
    const int profile_tail
) {
    extern __shared__ __align__(16) int shared_raw[];
    __shared__ std::uint32_t baseline_slot;
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);
    const int candidate_shard_ctas =
        shard_ctas<TOKEN_TILE_N>(active_tokens);
    const TailClocks clocks = tail_clocks(scratch, profile_tail != 0);
    clocks.clear();
    __syncthreads();
    const unsigned long long total_mark = clocks.now();

    kittens::tensor_allocator<1, 1> tensor_pool{};

    if (block < kReduceBegin) {
        coordinate_ranks(
            scratch, error_flag, barrier_multicast, barrier_local,
            barrier_target, clocks);
        clocks.lap(kTailClockTotal, total_mark);
        return;
    }

    if (block < kShardBegin) {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailReduceGeneration, &baseline_slot);
        unsigned long long mark = clocks.now();
        wait_for_generation(
            scratch, error_flag, kTailEntryGeneration, baseline,
            kErrorTailReduceEntry);
        clocks.lap(kTailClockReduceEntryWait, mark);
        reduce_rows(
            collective_multicast, routed_latent_rmsnorm_weight, scratch,
            block - kReduceBegin, tp_rank, active_tokens, clocks);
        mark = clocks.now();
        publish_generation(
            scratch, kTailReduceArrivals, kTailReduceGeneration, kReduceCtas);
        clocks.lap(kTailClockReducePublish, mark);
    } else {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailShardGeneration, &baseline_slot);
        unsigned long long mark = clocks.now();
        wait_for_generation(
            scratch, error_flag, kTailReduceGeneration, baseline,
            kErrorTailShardReduce);
        clocks.lap(kTailClockShardReduceWait, mark);
        shard_tensor_m128n<TOKEN_TILE_N>(
            shared_raw, tensor_pool, normalized, latent_up_proj, scratch,
            mailbox_multicast, block - kShardBegin, tp_rank, active_tokens,
            clocks);
        mark = clocks.now();
        publish_generation(
            scratch, kTailShardArrivals, kTailShardGeneration,
            candidate_shard_ctas);
        clocks.lap(kTailClockShardPublish, mark);
    }

    drain_ranks(
        scratch, error_flag, &baseline_slot,
        kReduceCtas + candidate_shard_ctas, clocks);
    clocks.lap(kTailClockTotal, total_mark);
}

inline bool guard_enabled() {
    const char *const enabled =
        std::getenv("MOK_KIMI_K3_ENABLE_TAIL_M128N_PROBE");
    return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

inline std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                  std::int64_t, std::int64_t, std::int64_t>
plan_for_testing(const std::int64_t active_tokens) {
    TORCH_CHECK(
        active_tokens == 16 || active_tokens == 32 || active_tokens == 128,
        "MoK: m128xN tail probe requires active_tokens in {16, 32, 128}");
    const int n = active_tokens == 16 ? 16 : 32;
    const int token_tile_count = static_cast<int>(active_tokens) / n;
    const int candidate_shard_ctas = kOutputTiles * token_tile_count;
    return {
        kMmaM, n, kMmaK, kOutputTiles, token_tile_count,
        candidate_shard_ctas, kShardBegin + candidate_shard_ctas};
}

template<int TOKEN_TILE_N>
static __host__ void reserve_shared_memory() {
    static std::array<std::once_flag, 32> reserved;
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(
        device >= 0 && device < static_cast<int>(reserved.size()),
        "MoK: m128xN tail probe saw an unexpected device ordinal ", device);
    std::call_once(reserved[static_cast<std::size_t>(device)], [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kimi_k3_tail_m128n_probe_kernel<TOKEN_TILE_N>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kProbeDynamicBytes));
    });
}

template<int TOKEN_TILE_N>
static __host__ std::tuple<std::int64_t, std::int64_t, std::int64_t,
                           std::int64_t, std::int64_t, std::int64_t>
resource_metadata() {
    reserve_shared_memory<TOKEN_TILE_N>();
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, kimi_k3_tail_m128n_probe_kernel<TOKEN_TILE_N>));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kimi_k3_tail_m128n_probe_kernel<TOKEN_TILE_N>,
        kDecodeCtaThreads, kProbeDynamicBytes));
    return {
        static_cast<std::int64_t>(kDecodeCtaThreads),
        static_cast<std::int64_t>(kProbeDynamicBytes),
        static_cast<std::int64_t>(blocks),
        static_cast<std::int64_t>(attributes.numRegs),
        static_cast<std::int64_t>(attributes.sharedSizeBytes),
        static_cast<std::int64_t>(attributes.localSizeBytes)};
}

inline std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                  std::int64_t, std::int64_t>
resource_metadata_for_testing(const std::int64_t token_tile_n) {
    TORCH_CHECK(
        guard_enabled(),
        "MoK: m128xN tail resource metadata is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_TAIL_M128N_PROBE=1");
    if (token_tile_n == 16) return resource_metadata<16>();
    TORCH_CHECK(
        token_tile_n == 32,
        "MoK: m128xN tail resource metadata requires N=16 or N=32");
    return resource_metadata<32>();
}

template<int TOKEN_TILE_N>
static __host__ void launch_variant(
    const at::Tensor &routed_latent_rmsnorm_weight,
    const at::Tensor &latent_up_proj,
    const std::int64_t collective_buffer_multicast_ptr,
    const std::int64_t output_mailbox_multicast_ptr,
    const at::Tensor &barrier_buffer,
    const std::int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target,
    const at::Tensor &scratch,
    const at::Tensor &error_flag,
    const int tp_rank,
    const int active_tokens,
    const int available_sms
) {
    const int candidate_shard_ctas =
        shard_ctas<TOKEN_TILE_N>(active_tokens);
    const int role_ctas = kShardBegin + candidate_shard_ctas;
    TORCH_CHECK(
        available_sms >= role_ctas,
        "MoK: m128xN tail probe requires all ", role_ctas,
        " role CTAs to co-reside, but the selected device exposes ",
        available_sms, " SMs");

    auto *const scratch_bytes =
        reinterpret_cast<std::uint8_t *>(scratch.data_ptr());
    const Scratch scratch_pointers = scratch_view(scratch_bytes);
    const input_layout<TOKEN_TILE_N> normalized_view{
        reinterpret_cast<kittens::bf16 *>(
            scratch_pointers.tail_normalized),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kLatentSize)};
    const weight_layout latent_up_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(
                latent_up_proj.data_ptr())
            + static_cast<long long>(tp_rank) * kShardColumns * kLatentSize),
        nullptr, nullptr, static_cast<size_t>(kShardColumns),
        static_cast<size_t>(kLatentSize)};
    const int profile_tail =
        benchmark_tail_profile_for_testing() ? 1 : 0;

    reserve_shared_memory<TOKEN_TILE_N>();
    kimi_k3_tail_m128n_probe_kernel<TOKEN_TILE_N>
        <<<role_ctas, kDecodeCtaThreads, kProbeDynamicBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(
                routed_latent_rmsnorm_weight.data_ptr()),
            normalized_view, latent_up_view,
            reinterpret_cast<__nv_bfloat16 *>(
                collective_buffer_multicast_ptr),
            reinterpret_cast<__nv_bfloat16 *>(
                output_mailbox_multicast_ptr),
            reinterpret_cast<std::uint32_t *>(
                barrier_buffer_multicast_ptr),
            reinterpret_cast<const std::uint32_t *>(
                barrier_buffer.data_ptr()),
            reinterpret_cast<unsigned int *>(barrier_target.data_ptr()),
            scratch_bytes,
            reinterpret_cast<int *>(error_flag.data_ptr()),
            tp_rank, active_tokens, profile_tail);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __host__ void launch(
    const at::Tensor &routed_latent_rmsnorm_weight,
    const at::Tensor &latent_up_proj,
    const std::int64_t collective_buffer_multicast_ptr,
    const std::int64_t output_mailbox_multicast_ptr,
    const at::Tensor &barrier_buffer,
    const std::int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target,
    const at::Tensor &scratch,
    const at::Tensor &error_flag,
    const std::int64_t tp_rank,
    const std::int64_t active_tokens
) {
    TORCH_CHECK(
        guard_enabled(),
        "MoK: _kimi_k3_tail_m128n_probe is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_TAIL_M128N_PROBE=1");
    TORCH_CHECK(
        active_tokens == 16 || active_tokens == 32 || active_tokens == 128,
        "MoK: _kimi_k3_tail_m128n_probe requires active_tokens in "
        "{16, 32, 128}");
    TORCH_CHECK(
        tp_rank >= 0 && tp_rank < kTensorParallelSize,
        "MoK: _kimi_k3_tail_m128n_probe requires tp_rank in [0, 7]");
    TORCH_CHECK(
        routed_latent_rmsnorm_weight.is_cuda()
            && routed_latent_rmsnorm_weight.is_contiguous()
            && routed_latent_rmsnorm_weight.scalar_type() == at::kBFloat16
            && routed_latent_rmsnorm_weight.numel() == kLatentSize,
        "MoK: _kimi_k3_tail_m128n_probe requires contiguous CUDA BF16 "
        "RMSNorm weight [3584]");
    TORCH_CHECK(
        latent_up_proj.is_cuda() && latent_up_proj.is_contiguous()
            && latent_up_proj.scalar_type() == at::kBFloat16
            && latent_up_proj.dim() == 2
            && latent_up_proj.size(0) == kHiddenSize
            && latent_up_proj.size(1) == kLatentSize,
        "MoK: _kimi_k3_tail_m128n_probe requires contiguous CUDA BF16 "
        "latent-up weight [7168, 3584]");
    TORCH_CHECK(
        barrier_buffer.is_cuda() && barrier_buffer.is_contiguous()
            && barrier_buffer.scalar_type() == at::kInt
            && barrier_buffer.numel() == 1
            && barrier_target.is_cuda() && barrier_target.is_contiguous()
            && barrier_target.scalar_type() == at::kInt
            && barrier_target.numel() == 1
            && error_flag.is_cuda() && error_flag.is_contiguous()
            && error_flag.scalar_type() == at::kInt
            && error_flag.numel() == 1,
        "MoK: _kimi_k3_tail_m128n_probe requires CUDA int32 control words");
    TORCH_CHECK(
        scratch.is_cuda() && scratch.is_contiguous()
            && scratch.scalar_type() == at::kByte
            && scratch.numel() >= SCRATCH_BYTES,
        "MoK: _kimi_k3_tail_m128n_probe requires the decode uint8 scratch");
    TORCH_CHECK(
        collective_buffer_multicast_ptr > 0
            && output_mailbox_multicast_ptr > 0
            && barrier_buffer_multicast_ptr > 0,
        "MoK: _kimi_k3_tail_m128n_probe requires positive multicast pointers");

    const at::Device device = scratch.device();
    for (const at::Tensor *tensor : {
             &routed_latent_rmsnorm_weight, &latent_up_proj,
             &barrier_buffer, &barrier_target, &error_flag}) {
        TORCH_CHECK(
            tensor->device() == device,
            "MoK: _kimi_k3_tail_m128n_probe requires one CUDA device");
    }

    const c10::cuda::CUDAGuard device_guard(device);
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(
        &properties, device.index()));
    TORCH_CHECK(
        properties.major == 10 && properties.minor == 3,
        "MoK: _kimi_k3_tail_m128n_probe requires an SM103 B300");

    if (active_tokens == 16) {
        launch_variant<16>(
            routed_latent_rmsnorm_weight, latent_up_proj,
            collective_buffer_multicast_ptr, output_mailbox_multicast_ptr,
            barrier_buffer, barrier_buffer_multicast_ptr, barrier_target,
            scratch, error_flag, static_cast<int>(tp_rank),
            static_cast<int>(active_tokens), properties.multiProcessorCount);
        return;
    }
    launch_variant<32>(
        routed_latent_rmsnorm_weight, latent_up_proj,
        collective_buffer_multicast_ptr, output_mailbox_multicast_ptr,
        barrier_buffer, barrier_buffer_multicast_ptr, barrier_target,
        scratch, error_flag, static_cast<int>(tp_rank),
        static_cast<int>(active_tokens), properties.multiProcessorCount);
}

}  // namespace m128n_probe
}  // namespace tail
}  // namespace kimi_k3_decode
