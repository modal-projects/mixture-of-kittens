#pragma once

#include "kittens.cuh"

#include "tail_reduce.cuh"
#include "tail_shard.cuh"
#include "tail_sync.cuh"
#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <tuple>

namespace kimi_k3_decode {
namespace tail {

// Orchestration for the fused TP8 tail: the two single-launch kernels that lay
// the coordinator, reduce, and shard roles out across one grid, and the host
// side that plans that grid, checks it can co-reside, and launches it.
//
// The roles themselves live in `tail_sync.cuh` (constants, multimem, and
// generation-tagged rendezvous), `tail_reduce.cuh`, and `tail_shard.cuh`.

// One `std::once_flag` per possible CUDA ordinal, so the tcgen05 shared-memory
// reservation happens once per device even when one process drives several.
inline constexpr int kMaxCudaDevices = 32;

// ---------------------------------------------------------------------------
// Single-launch kernels.
// ---------------------------------------------------------------------------

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
    int *__restrict__ error_flag,
    const int tp_rank,
    const int active_tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    __shared__ std::uint32_t baseline_slot;
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);

    // The private path's tail is its own launch, so it clears the claim word
    // the way the one-launch kernels clear theirs.
    timeout::clear_claim(scratch);

    if (block < kReduceBegin) {
        coordinate_ranks(scratch, error_flag, barrier_multicast, barrier_local,
                         barrier_target);
        return;
    }

    if (block < kShardBegin) {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailReduceGeneration, &baseline_slot);
        wait_for_generation(scratch, error_flag, kTailEntryGeneration,
                            baseline, kErrorTailReduceEntry);
        reduce_rows(
            collective_multicast, routed_latent_rmsnorm_weight, scratch,
            block - kReduceBegin, tp_rank, active_tokens);
        publish_generation(
            scratch, kTailReduceArrivals, kTailReduceGeneration, kReduceCtas);
    } else {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailShardGeneration, &baseline_slot);
        wait_for_generation(scratch, error_flag, kTailReduceGeneration,
                            baseline, kErrorTailShardReduce);
        shard_core<CAPACITY>(
            reinterpret_cast<std::uint8_t *>(shared_raw), scratch,
            latent_up_proj, mailbox_multicast, block - kShardBegin, tp_rank,
            active_tokens);
        publish_generation(
            scratch, kTailShardArrivals, kTailShardGeneration, kCoreShardCtas);
    }

    drain_ranks(scratch, error_flag, &baseline_slot,
                kReduceCtas + kCoreShardCtas);
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
    int *__restrict__ error_flag,
    const int tp_rank,
    const int active_tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    __shared__ std::uint32_t baseline_slot;
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);

    // The private path's tail is its own launch, so it clears the claim word
    // the way the one-launch kernels clear theirs.
    timeout::clear_claim(scratch);

    // The managed allocator barriers the whole CTA, so every block provisions
    // its tensor memory before the roles diverge.
    kittens::tensor_allocator<1, 1> tensor_pool{};

    if (block < kReduceBegin) {
        coordinate_ranks(scratch, error_flag, barrier_multicast, barrier_local,
                         barrier_target);
        return;
    }

    if (block < kShardBegin) {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailReduceGeneration, &baseline_slot);
        wait_for_generation(scratch, error_flag, kTailEntryGeneration,
                            baseline, kErrorTailReduceEntry);
        reduce_rows(
            collective_multicast, routed_latent_rmsnorm_weight, scratch,
            block - kReduceBegin, tp_rank, active_tokens);
        publish_generation(
            scratch, kTailReduceArrivals, kTailReduceGeneration, kReduceCtas);
    } else {
        const std::uint32_t baseline = latch_generation(
            scratch, kTailShardGeneration, &baseline_slot);
        wait_for_generation(scratch, error_flag, kTailReduceGeneration,
                            baseline, kErrorTailShardReduce);
        shard_tensor(
            shared_raw, tensor_pool, normalized, latent_up_proj, scratch,
            mailbox_multicast, block - kShardBegin, tp_rank, active_tokens);
        publish_generation(
            scratch, kTailShardArrivals, kTailShardGeneration,
            kTensorShardCtas);
    }

    drain_ranks(scratch, error_flag, &baseline_slot,
                kReduceCtas + kTensorShardCtas);
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

/// How many times the reservation below actually ran, per CUDA ordinal.
///
/// A `std::once_flag` cannot be asked whether it has fired, so the count is
/// kept alongside it. It exists for one test: that the reservation happens on
/// the device the tensors live on, cold, even when a different device is
/// current -- which is only observable if the test can tell a first launch from
/// a later one.
static __host__ std::array<std::atomic<int>, kMaxCudaDevices> &
shared_memory_reservations() {
    static std::array<std::atomic<int>, kMaxCudaDevices> counts{};
    return counts;
}

static __host__ std::int64_t shared_memory_reservations_for_testing(
    const std::int64_t device
) {
    TORCH_CHECK(device >= 0 && device < kMaxCudaDevices,
                "MoK: _kimi_k3_tail tracks devices 0 through ",
                kMaxCudaDevices - 1, ", got ", device);
    return shared_memory_reservations()[static_cast<std::size_t>(device)].load(
        std::memory_order_relaxed);
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
    std::call_once(reserved[static_cast<std::size_t>(device)], [device] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kimi_k3_tail_tensor_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kTailTensorDynamicBytes));
        shared_memory_reservations()[static_cast<std::size_t>(device)]
            .fetch_add(1, std::memory_order_relaxed);
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
    int *error_flag,
    const int tp_rank,
    const int active_tokens
) {
    kimi_k3_tail_core_kernel<CAPACITY>
        <<<kCoreRoleCtas, kDecodeCtaThreads, kTailCoreDynamicBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            routed_latent_rmsnorm_weight, latent_up_proj, collective_multicast,
            mailbox_multicast, barrier_multicast, barrier_local, barrier_target,
            scratch_bytes, error_flag, tp_rank, active_tokens);
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
    const at::Tensor &error_flag,
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
    auto *const error = reinterpret_cast<int *>(error_flag.data_ptr());

    switch (capacity_bucket(active_tokens)) {
        case 1:
            launch_core<1>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, error, tp_rank,
                           active_tokens);
            return;
        case 2:
            launch_core<2>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, error, tp_rank,
                           active_tokens);
            return;
        case 4:
            launch_core<4>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, error, tp_rank,
                           active_tokens);
            return;
        case 8:
            launch_core<8>(norm_weight, latent_up, collective_multicast,
                           mailbox_multicast, barrier_multicast, barrier_local,
                           target, scratch_bytes, error, tp_rank,
                           active_tokens);
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
            scratch_bytes, error, tp_rank, active_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tail
}  // namespace kimi_k3_decode
