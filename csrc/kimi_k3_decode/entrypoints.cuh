#pragma once

#include "pyutils/torchutils.cuh"

#include "types.cuh"

#include <ATen/ops/empty_like.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace kimi_k3_decode {

inline std::int64_t kimi_k3_decode_workspace_bytes() noexcept {
    // Task 2 defines the query boundary; subsequent tasks define scratch offsets.
    return 0;
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
    cudaDeviceProp properties{};
    const int device_index = hidden_states.get_device();
    const cudaError_t status = cudaGetDeviceProperties(&properties, device_index);
    TORCH_CHECK(status == cudaSuccess,
                "MoK: cudaGetDeviceProperties failed for device ", device_index,
                ": ", cudaGetErrorString(status));
    TORCH_CHECK(properties.major == 10 && properties.minor == 3,
                "MoK: kimi_k3_decode requires SM103, found sm_",
                properties.major, properties.minor);

    return at::empty_like(hidden_states);
}

}  // namespace kimi_k3_decode
