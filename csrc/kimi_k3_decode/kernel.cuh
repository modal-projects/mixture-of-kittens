#pragma once

#include "kittens.cuh"

#include "expert_mxfp4.cuh"
#include "router.cuh"
#include "skinny_gemm.cuh"
#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace kimi_k3_decode {

// The fused stage runs the router and the routed latent-down projection as two
// CTA roles inside one launch. Task 9's persistent kernel calls the same two
// device functions directly, so nothing here returns an intermediate tensor.

inline constexpr int kCoreSharedBytes =
    router::kSharedBytes > skinny_gemm::kCoreSharedBytes
        ? router::kSharedBytes : skinny_gemm::kCoreSharedBytes;

// The tcgen05 role holds all 512 tensor-memory columns, so its shared-memory
// request deliberately keeps one CTA per SM and the allocation always succeeds.
inline constexpr int kTensorSharedBytes = kittens::MAX_SHARED_MEMORY - 1024;

static_assert(router::kSharedBytes <= kTensorSharedBytes,
              "the router role must fit inside the tcgen05 launch's shared memory");

template<int CAPACITY>
__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void route_and_project_core_kernel(
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ router_weight,
    const float *__restrict__ router_correction_bias,
    const __nv_bfloat16 *__restrict__ routed_expert_down_proj,
    std::uint8_t *__restrict__ scratch_bytes,
    int *__restrict__ expert_ids,
    float *__restrict__ expert_weights,
    __nv_bfloat16 *__restrict__ latent_x,
    const int active_tokens,
    const int tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    std::uint8_t *const shared = reinterpret_cast<std::uint8_t *>(shared_raw);
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);

    if (block < active_tokens) {
        router::route_token(shared, hidden_states, router_weight,
                            router_correction_bias, scratch, expert_ids,
                            expert_weights, block, active_tokens);
        return;
    }

    const int projection_index = block - active_tokens;
    skinny_gemm::mask_inactive_rows(
        expert_ids, expert_weights, latent_x,
        projection_index * skinny_gemm::kCoreColumnsPerCta,
        skinny_gemm::kCoreColumnsPerCta, projection_index,
        skinny_gemm::kCoreCtas, active_tokens, tokens);
    skinny_gemm::latent_down_cuda_core<CAPACITY>(
        shared, hidden_states, routed_expert_down_proj, latent_x,
        projection_index, active_tokens);
    skinny_gemm::publish_projection_completion(scratch, skinny_gemm::kCoreCtas);
}

__global__ __launch_bounds__(kDecodeCtaThreads, 1)
void route_and_project_tensor_kernel(
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ router_weight,
    const float *__restrict__ router_correction_bias,
    const __grid_constant__ skinny_gemm::hidden_layout hidden,
    const __grid_constant__ skinny_gemm::weight_layout weight,
    const __grid_constant__ skinny_gemm::latent_layout latent,
    std::uint8_t *__restrict__ scratch_bytes,
    int *__restrict__ expert_ids,
    float *__restrict__ expert_weights,
    __nv_bfloat16 *__restrict__ latent_x,
    const int active_tokens,
    const int tokens
) {
    extern __shared__ __align__(16) int shared_raw[];
    const Scratch scratch = scratch_view(scratch_bytes);
    const int block = static_cast<int>(blockIdx.x);

    if (block < active_tokens) {
        router::route_token(reinterpret_cast<std::uint8_t *>(shared_raw),
                            hidden_states, router_weight, router_correction_bias,
                            scratch, expert_ids, expert_weights, block,
                            active_tokens);
        return;
    }

    const int projection_index = block - active_tokens;
    skinny_gemm::mask_inactive_rows(
        expert_ids, expert_weights, latent_x,
        projection_index * skinny_gemm::kTileN, skinny_gemm::kTileN,
        projection_index, skinny_gemm::kTensorCtas, active_tokens, tokens);
    skinny_gemm::latent_down_tcgen05(shared_raw, hidden, weight, latent,
                                     projection_index);
    skinny_gemm::publish_projection_completion(scratch,
                                               skinny_gemm::kTensorCtas);
}

template<int CAPACITY>
static __host__ void launch_core_stage(
    const __nv_bfloat16 *hidden_states,
    const __nv_bfloat16 *router_weight,
    const float *router_correction_bias,
    const __nv_bfloat16 *routed_expert_down_proj,
    std::uint8_t *scratch_bytes,
    int *expert_ids,
    float *expert_weights,
    __nv_bfloat16 *latent_x,
    const int active_tokens,
    const int tokens
) {
    route_and_project_core_kernel<CAPACITY>
        <<<active_tokens + skinny_gemm::kCoreCtas, kDecodeCtaThreads,
           kCoreSharedBytes, at::cuda::getCurrentCUDAStream()>>>(
            hidden_states, router_weight, router_correction_bias,
            routed_expert_down_proj, scratch_bytes, expert_ids, expert_weights,
            latent_x, active_tokens, tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// Run the fused router and routed latent-down projection in a single launch.
static __host__ void launch_route_and_project(
    const at::Tensor &hidden_states,
    const at::Tensor &router_weight,
    const at::Tensor &router_correction_bias,
    const at::Tensor &routed_expert_down_proj,
    const at::Tensor &scratch,
    const at::Tensor &expert_ids,
    const at::Tensor &expert_weights,
    const at::Tensor &latent_x,
    const int active_tokens
) {
    const int tokens = static_cast<int>(hidden_states.size(0));
    const auto *const hidden_pointer =
        reinterpret_cast<const __nv_bfloat16 *>(hidden_states.data_ptr());
    const auto *const router_pointer =
        reinterpret_cast<const __nv_bfloat16 *>(router_weight.data_ptr());
    const auto *const bias_pointer =
        reinterpret_cast<const float *>(router_correction_bias.data_ptr());
    const auto *const down_pointer =
        reinterpret_cast<const __nv_bfloat16 *>(routed_expert_down_proj.data_ptr());
    auto *const scratch_pointer =
        reinterpret_cast<std::uint8_t *>(scratch.data_ptr());
    auto *const id_pointer = reinterpret_cast<int *>(expert_ids.data_ptr());
    auto *const weight_pointer =
        reinterpret_cast<float *>(expert_weights.data_ptr());
    auto *const latent_pointer =
        reinterpret_cast<__nv_bfloat16 *>(latent_x.data_ptr());

    switch (capacity_bucket(active_tokens)) {
        case 1:
            launch_core_stage<1>(hidden_pointer, router_pointer, bias_pointer,
                                 down_pointer, scratch_pointer, id_pointer,
                                 weight_pointer, latent_pointer, active_tokens,
                                 tokens);
            return;
        case 2:
            launch_core_stage<2>(hidden_pointer, router_pointer, bias_pointer,
                                 down_pointer, scratch_pointer, id_pointer,
                                 weight_pointer, latent_pointer, active_tokens,
                                 tokens);
            return;
        case 4:
            launch_core_stage<4>(hidden_pointer, router_pointer, bias_pointer,
                                 down_pointer, scratch_pointer, id_pointer,
                                 weight_pointer, latent_pointer, active_tokens,
                                 tokens);
            return;
        case 8:
            launch_core_stage<8>(hidden_pointer, router_pointer, bias_pointer,
                                 down_pointer, scratch_pointer, id_pointer,
                                 weight_pointer, latent_pointer, active_tokens,
                                 tokens);
            return;
        default:
            break;
    }

    // Capacities 16, 32, 64, and 128 all share one 128-row tcgen05 tile: the
    // descriptors span exactly the active rows, so the hardware zero-fills the
    // rest on load and drops them on store.
    const skinny_gemm::hidden_layout hidden_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(hidden_pointer)),
        nullptr, nullptr, static_cast<size_t>(active_tokens),
        static_cast<size_t>(kHiddenSize)};
    const skinny_gemm::weight_layout weight_view{
        const_cast<kittens::bf16 *>(
            reinterpret_cast<const kittens::bf16 *>(down_pointer)),
        nullptr, nullptr, static_cast<size_t>(kLatentSize),
        static_cast<size_t>(kHiddenSize)};
    const skinny_gemm::latent_layout latent_view{
        reinterpret_cast<kittens::bf16 *>(latent_pointer), nullptr, nullptr,
        static_cast<size_t>(active_tokens), static_cast<size_t>(kLatentSize)};

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        route_and_project_tensor_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kTensorSharedBytes));
    route_and_project_tensor_kernel
        <<<active_tokens + skinny_gemm::kTensorCtas, kDecodeCtaThreads,
           kTensorSharedBytes, at::cuda::getCurrentCUDAStream()>>>(
            hidden_pointer, router_pointer, bias_pointer, hidden_view,
            weight_view, latent_view, scratch_pointer, id_pointer,
            weight_pointer, latent_pointer, active_tokens, tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace kimi_k3_decode
