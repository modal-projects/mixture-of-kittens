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

static __host__ void check_tensor_alignment(
    const at::Tensor &tensor,
    const char *operation,
    const char *field,
    const int alignment
) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(address % static_cast<std::uintptr_t>(alignment) == 0,
                "MoK: ", operation, " requires ", field,
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
    check_tensor_alignment(hidden_states, "_kimi_k3_route_and_project",
                           "hidden_states", VECTOR_ALIGNMENT);
    check_tensor_alignment(router_weight, "_kimi_k3_route_and_project",
                           "router_weight", VECTOR_ALIGNMENT);
    check_tensor_alignment(routed_expert_down_proj,
                           "_kimi_k3_route_and_project",
                           "routed_expert_down_proj", VECTOR_ALIGNMENT);
    check_tensor_alignment(scratch, "_kimi_k3_route_and_project",
                           "scratch", SCRATCH_ALIGNMENT);

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

static __host__ at::Tensor routed_experts_entrypoint(
    const at::Tensor &latent_x,
    const at::Tensor &expert_w1_packed,
    const at::Tensor &expert_w1_scale,
    const at::Tensor &expert_w3_packed,
    const at::Tensor &expert_w3_scale,
    const at::Tensor &expert_w2_packed,
    const at::Tensor &expert_w2_scale,
    const at::Tensor &routed_output,
    const at::Tensor &scratch,
    std::int64_t active_tokens
) {
    CHECK_INPUT(latent_x);
    CHECK_INPUT(expert_w1_packed);
    CHECK_INPUT(expert_w1_scale);
    CHECK_INPUT(expert_w3_packed);
    CHECK_INPUT(expert_w3_scale);
    CHECK_INPUT(expert_w2_packed);
    CHECK_INPUT(expert_w2_scale);
    CHECK_INPUT(routed_output);
    CHECK_INPUT(scratch);

    TORCH_CHECK(latent_x.dim() == 2 && latent_x.size(1) == kLatentSize
                    && latent_x.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_routed_experts requires BF16 latent_x [M, ",
                kLatentSize, "]");
    const std::int64_t tokens = latent_x.size(0);
    TORCH_CHECK(tokens >= 1 && tokens <= kMaxTokens,
                "MoK: _kimi_k3_routed_experts requires latent_x with 1 to ",
                kMaxTokens, " rows");
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= tokens,
                "MoK: _kimi_k3_routed_experts requires active_tokens in [1, ",
                tokens, "]");

    const auto check_weight = [](const at::Tensor &tensor,
                                 const char *name,
                                 const int rows,
                                 const int columns) {
        TORCH_CHECK(tensor.dim() == 3
                        && tensor.size(0) == kNumExperts
                        && tensor.size(1) == rows
                        && tensor.size(2) == columns
                        && tensor.scalar_type() == at::kByte,
                    "MoK: _kimi_k3_routed_experts requires uint8 ", name,
                    " [", kNumExperts, ", ", rows, ", ", columns, "]");
    };
    check_weight(expert_w1_packed, "expert_w1_packed",
                 kExpertW1W3PackedRows, kExpertW1W3PackedColumns);
    check_weight(expert_w1_scale, "expert_w1_scale",
                 kExpertW1W3PackedRows, kExpertW1W3ScaleColumns);
    check_weight(expert_w3_packed, "expert_w3_packed",
                 kExpertW1W3PackedRows, kExpertW1W3PackedColumns);
    check_weight(expert_w3_scale, "expert_w3_scale",
                 kExpertW1W3PackedRows, kExpertW1W3ScaleColumns);
    check_weight(expert_w2_packed, "expert_w2_packed",
                 kExpertW2PackedRows, kExpertW2PackedColumns);
    check_weight(expert_w2_scale, "expert_w2_scale",
                 kExpertW2PackedRows, kExpertW2ScaleColumns);

    TORCH_CHECK(routed_output.dim() == 2
                    && routed_output.size(0) == tokens
                    && routed_output.size(1) == kLatentSize
                    && routed_output.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_routed_experts requires BF16 routed_output [M, ",
                kLatentSize, "]");
    TORCH_CHECK(scratch.dim() == 1 && scratch.scalar_type() == at::kByte
                    && scratch.size(0) >= SCRATCH_BYTES,
                "MoK: _kimi_k3_routed_experts requires a uint8 scratch of at "
                "least ", SCRATCH_BYTES, " bytes");

    const at::Device device = latent_x.device();
    for (const auto *tensor : {
             &expert_w1_packed, &expert_w1_scale,
             &expert_w3_packed, &expert_w3_scale,
             &expert_w2_packed, &expert_w2_scale,
             &routed_output, &scratch}) {
        TORCH_CHECK(tensor->device() == device,
                    "MoK: _kimi_k3_routed_experts requires every tensor on ",
                    device);
    }

    for (const auto &item : {
             std::pair<const at::Tensor *, const char *>{
                 &latent_x, "latent_x"},
             {&expert_w1_packed, "expert_w1_packed"},
             {&expert_w1_scale, "expert_w1_scale"},
             {&expert_w3_packed, "expert_w3_packed"},
             {&expert_w3_scale, "expert_w3_scale"},
             {&expert_w2_packed, "expert_w2_packed"},
             {&expert_w2_scale, "expert_w2_scale"},
             {&routed_output, "routed_output"}}) {
        check_tensor_alignment(*item.first, "_kimi_k3_routed_experts",
                               item.second, VECTOR_ALIGNMENT);
    }
    check_tensor_alignment(scratch, "_kimi_k3_routed_experts", "scratch",
                           SCRATCH_ALIGNMENT);
    check_sm103(latent_x, "_kimi_k3_routed_experts");

    const c10::cuda::CUDAGuard device_guard(device);
    expert_mxfp4::launch_routed_experts(
        latent_x, expert_w1_packed, expert_w1_scale,
        expert_w3_packed, expert_w3_scale,
        expert_w2_packed, expert_w2_scale,
        routed_output, scratch, static_cast<int>(active_tokens));
    return routed_output.narrow(0, 0, active_tokens);
}

static __host__ at::Tensor shared_experts_entrypoint(
    const at::Tensor &hidden_states,
    const at::Tensor &shared_gate_proj,
    const at::Tensor &shared_up_proj,
    const at::Tensor &shared_down_proj,
    const at::Tensor &scratch,
    const at::Tensor &collective_buffer,
    std::int64_t active_tokens
) {
    CHECK_INPUT(hidden_states);
    CHECK_INPUT(shared_gate_proj);
    CHECK_INPUT(shared_up_proj);
    CHECK_INPUT(shared_down_proj);
    CHECK_INPUT(scratch);
    CHECK_INPUT(collective_buffer);

    TORCH_CHECK(hidden_states.dim() == 2
                    && hidden_states.size(1) == kHiddenSize
                    && hidden_states.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_shared_experts requires BF16 hidden_states [M, ",
                kHiddenSize, "]");
    const std::int64_t tokens = hidden_states.size(0);
    TORCH_CHECK(tokens >= 1 && tokens <= kMaxTokens,
                "MoK: _kimi_k3_shared_experts requires hidden_states with 1 to ",
                kMaxTokens, " rows");
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= tokens,
                "MoK: _kimi_k3_shared_experts requires active_tokens in [1, ",
                tokens, "]");
    constexpr int intermediate =
        kSharedIntermediateSize / kTensorParallelSize;
    TORCH_CHECK(shared_gate_proj.dim() == 2
                    && shared_gate_proj.size(0) == intermediate
                    && shared_gate_proj.size(1) == kHiddenSize
                    && shared_gate_proj.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_shared_experts requires BF16 shared_gate_proj [",
                intermediate, ", ", kHiddenSize, "]");
    TORCH_CHECK(shared_up_proj.dim() == 2
                    && shared_up_proj.size(0) == intermediate
                    && shared_up_proj.size(1) == kHiddenSize
                    && shared_up_proj.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_shared_experts requires BF16 shared_up_proj [",
                intermediate, ", ", kHiddenSize, "]");
    TORCH_CHECK(shared_down_proj.dim() == 2
                    && shared_down_proj.size(0) == kHiddenSize
                    && shared_down_proj.size(1) == intermediate
                    && shared_down_proj.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_shared_experts requires BF16 shared_down_proj [",
                kHiddenSize, ", ", intermediate, "]");
    TORCH_CHECK(scratch.dim() == 1 && scratch.scalar_type() == at::kByte
                    && scratch.size(0) >= SCRATCH_BYTES,
                "MoK: _kimi_k3_shared_experts requires a uint8 scratch of at "
                "least ", SCRATCH_BYTES, " bytes");
    TORCH_CHECK(collective_buffer.dim() == 2
                    && collective_buffer.size(0) == tokens
                    && collective_buffer.size(1) == kLatentSize + kHiddenSize
                    && collective_buffer.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_shared_experts requires BF16 collective_buffer [M, ",
                kLatentSize + kHiddenSize, "]");

    const at::Device device = hidden_states.device();
    for (const auto *tensor : {
             &shared_gate_proj, &shared_up_proj, &shared_down_proj,
             &scratch, &collective_buffer}) {
        TORCH_CHECK(tensor->device() == device,
                    "MoK: _kimi_k3_shared_experts requires every tensor on ",
                    device);
    }
    for (const auto &item : {
             std::pair<const at::Tensor *, const char *>{
                 &hidden_states, "hidden_states"},
             {&shared_gate_proj, "shared_gate_proj"},
             {&shared_up_proj, "shared_up_proj"},
             {&shared_down_proj, "shared_down_proj"},
             {&collective_buffer, "collective_buffer"}}) {
        check_tensor_alignment(*item.first, "_kimi_k3_shared_experts",
                               item.second, VECTOR_ALIGNMENT);
    }
    check_tensor_alignment(scratch, "_kimi_k3_shared_experts", "scratch",
                           SCRATCH_ALIGNMENT);
    check_sm103(hidden_states, "_kimi_k3_shared_experts");

    const c10::cuda::CUDAGuard device_guard(device);
    shared_experts::launch_shared_experts(
        hidden_states, shared_gate_proj, shared_up_proj, shared_down_proj,
        scratch, collective_buffer, static_cast<int>(active_tokens));
    return collective_buffer.narrow(0, 0, active_tokens)
        .narrow(1, kLatentSize, kHiddenSize);
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
