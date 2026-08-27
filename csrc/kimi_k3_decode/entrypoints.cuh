#pragma once

#include "pyutils/torchutils.cuh"

#include "kernel.cuh"
#include "mxfp4.cuh"
#include "types.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace kimi_k3_decode {

inline std::int64_t kimi_k3_decode_workspace_bytes() noexcept {
    return SCRATCH_BYTES;
}

static __host__ void check_sm103(const at::Tensor &hidden_states, const char *name) {
    cudaDeviceProp properties{};
    const int device_index = hidden_states.get_device();
    const cudaError_t status = cudaGetDeviceProperties(&properties, device_index);
    TORCH_CHECK(status == cudaSuccess,
                "MoK: cudaGetDeviceProperties failed for device ", device_index,
                ": ", cudaGetErrorString(status));
    TORCH_CHECK(properties.major == 10 && properties.minor == 3,
                "MoK: ", name, " requires SM103, found sm_",
                properties.major, properties.minor);
}

static __host__ void check_route_and_project_alignment(
    const at::Tensor &tensor,
    const char *field,
    const int alignment
) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(address % static_cast<std::uintptr_t>(alignment) == 0,
                "MoK: _kimi_k3_route_and_project requires ", field,
                " aligned to ", alignment, " bytes, got a pointer ",
                address % static_cast<std::uintptr_t>(alignment),
                " bytes past one");
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor>
route_and_project_entrypoint(
    const at::Tensor &hidden_states,
    const at::Tensor &router_weight,
    const at::Tensor &router_correction_bias,
    const at::Tensor &routed_expert_down_proj,
    const at::Tensor &scratch,
    std::int64_t active_tokens
) {
    CHECK_INPUT(hidden_states);
    CHECK_INPUT(router_weight);
    CHECK_INPUT(router_correction_bias);
    CHECK_INPUT(routed_expert_down_proj);
    CHECK_INPUT(scratch);
    TORCH_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) == kHiddenSize,
                "MoK: _kimi_k3_route_and_project requires hidden_states [M, ",
                kHiddenSize, "]");
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_route_and_project requires BF16 hidden_states");
    const std::int64_t tokens = hidden_states.size(0);
    TORCH_CHECK(tokens >= 1 && tokens <= kMaxTokens,
                "MoK: _kimi_k3_route_and_project requires hidden_states with 1 to ",
                kMaxTokens, " tokens");
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= tokens,
                "MoK: _kimi_k3_route_and_project requires active_tokens in [1, ",
                tokens, "]");
    TORCH_CHECK(router_weight.dim() == 2 && router_weight.size(0) == kNumExperts
                    && router_weight.size(1) == kHiddenSize
                    && router_weight.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_route_and_project requires a BF16 router_weight [",
                kNumExperts, ", ", kHiddenSize, "]");
    TORCH_CHECK(router_correction_bias.dim() == 1
                    && router_correction_bias.size(0) == kNumExperts
                    && router_correction_bias.scalar_type() == at::kFloat,
                "MoK: _kimi_k3_route_and_project requires a float32 "
                "router_correction_bias [", kNumExperts, "]");
    TORCH_CHECK(routed_expert_down_proj.dim() == 2
                    && routed_expert_down_proj.size(0) == kLatentSize
                    && routed_expert_down_proj.size(1) == kHiddenSize
                    && routed_expert_down_proj.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_route_and_project requires a BF16 "
                "routed_expert_down_proj [", kLatentSize, ", ", kHiddenSize, "]");
    TORCH_CHECK(scratch.scalar_type() == at::kByte && scratch.dim() == 1
                    && scratch.size(0) >= SCRATCH_BYTES,
                "MoK: _kimi_k3_route_and_project requires a uint8 scratch of at "
                "least ", SCRATCH_BYTES, " bytes");
    TORCH_CHECK(router_weight.device() == hidden_states.device()
                    && router_correction_bias.device() == hidden_states.device()
                    && routed_expert_down_proj.device() == hidden_states.device()
                    && scratch.device() == hidden_states.device(),
                "MoK: _kimi_k3_route_and_project requires every tensor on ",
                hidden_states.device());
    // A contiguous view at a nonzero storage offset clears every check above and
    // still under-aligns the pointer, which faults the vector loads and TMA
    // descriptors or silently shifts every scratch region.
    check_route_and_project_alignment(hidden_states, "hidden_states",
                                      VECTOR_ALIGNMENT);
    check_route_and_project_alignment(router_weight, "router_weight",
                                      VECTOR_ALIGNMENT);
    check_route_and_project_alignment(routed_expert_down_proj,
                                      "routed_expert_down_proj",
                                      VECTOR_ALIGNMENT);
    check_route_and_project_alignment(scratch, "scratch", SCRATCH_ALIGNMENT);

    check_sm103(hidden_states, "_kimi_k3_route_and_project");

    // The stage must run on the tensors' own device and that device's current
    // stream, whatever device happens to be current on entry.
    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    // The kernel masks the inactive rows itself, so the stage stays one launch.
    at::Tensor expert_ids = at::empty({tokens, kTopK},
                                      hidden_states.options().dtype(at::kInt));
    at::Tensor expert_weights = at::empty({tokens, kTopK},
                                          hidden_states.options().dtype(at::kFloat));
    at::Tensor latent_x = at::empty({tokens, kLatentSize}, hidden_states.options());

    launch_route_and_project(hidden_states, router_weight, router_correction_bias,
                             routed_expert_down_proj, scratch, expert_ids,
                             expert_weights, latent_x,
                             static_cast<int>(active_tokens));
    return {expert_ids, expert_weights, latent_x};
}

static __host__ at::Tensor kimi_k3_decode_entrypoint(
    const at::Tensor &hidden_states,
    const at::Tensor &router_weight,
    const at::Tensor &router_correction_bias,
    const at::Tensor &routed_expert_down_proj,
    const at::Tensor &routed_expert_up_proj,
    const at::Tensor &routed_latent_rmsnorm_weight,
    const at::Tensor &expert_w1_packed,
    const at::Tensor &expert_w1_scale,
    const at::Tensor &expert_w3_packed,
    const at::Tensor &expert_w3_scale,
    const at::Tensor &expert_w2_packed,
    const at::Tensor &expert_w2_scale,
    const at::Tensor &shared_gate_proj,
    const at::Tensor &shared_up_proj,
    const at::Tensor &shared_down_proj,
    const at::Tensor &scratch,
    const at::Tensor &collective_buffer,
    const std::vector<std::int64_t> &collective_buffer_ptrs,
    std::int64_t collective_buffer_multicast_ptr,
    const at::Tensor &output_mailbox,
    const std::vector<std::int64_t> &output_mailbox_ptrs,
    const at::Tensor &barrier_buffer,
    const std::vector<std::int64_t> &barrier_buffer_ptrs,
    std::int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &error_flag,
    int tp_rank,
    int active_tokens
) {
    TORCH_CHECK(hidden_states.is_cuda(),
                "MoK: kimi_k3_decode requires CUDA hidden_states");
    check_sm103(hidden_states, "kimi_k3_decode");

    return at::empty_like(hidden_states);
}

}  // namespace kimi_k3_decode
