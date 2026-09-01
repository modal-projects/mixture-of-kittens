#pragma once

/// The public decode step: validate the whole argument tuple, then launch.
///
/// Every rule the operator enforces lives here rather than in the entrypoint
/// that calls it, assembling one `LaunchArguments` the launcher only has to
/// read. This is the one surface a caller reaches, and it names no engine, no
/// grid, and no schedule.

#include "tail.cuh"

namespace kimi_k3_decode {

/// Validate one decode step's whole argument tuple and assemble the launch.
///
/// Every rule the operator enforces lives here rather than in the entrypoint
/// that calls it: a hundred-odd checks -- shapes, dtypes, devices, alignments,
/// symmetric pointer lists, multicast disjointness, and the workspace signature
/// -- assembling one `LaunchArguments` that the launcher then only has to read.
static __host__ persistent::LaunchArguments decode_launch_arguments(
    const at::Tensor &hidden_states,
    const at::Tensor &router_weight,
    const at::Tensor &router_correction_bias,
    const at::Tensor &routed_expert_down_proj,
    const at::Tensor &routed_expert_up_proj,
    const at::Tensor &routed_latent_rmsnorm_weight,
    const at::Tensor &expert_w13_packed,
    const at::Tensor &expert_w13_scale,
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
    std::int64_t output_mailbox_multicast_ptr,
    const at::Tensor &barrier_buffer,
    const std::vector<std::int64_t> &barrier_buffer_ptrs,
    std::int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target,
    const at::Tensor &error_flag,
    std::int64_t tp_rank,
    std::int64_t active_tokens,
    std::int64_t workspace_signature_value
) {
    constexpr const char *kOperation = "kimi_k3_decode";
    constexpr int shard_columns = kHiddenSize / kTensorParallelSize;
    constexpr int collective_columns = kLatentSize + kHiddenSize;
    constexpr int shared_intermediate =
        kSharedIntermediateSize / kTensorParallelSize;

    // Every tensor the launch touches, with the boundary the device
    // dereferences it on. A contiguous view at a nonzero storage offset clears
    // every shape and dtype rule below and still under-aligns the pointer,
    // which faults the vector loads and TMA descriptors or silently shifts
    // every scratch region. Zero marks the three int32 control words and the
    // correction bias, which are only ever read one scalar at a time.
    struct DecodeTensor {
        const at::Tensor *tensor;
        const char *name;
        int alignment;
    };
    const DecodeTensor decode_tensors[] = {
        {&hidden_states, "hidden_states", VECTOR_ALIGNMENT},
        {&router_weight, "router_weight", VECTOR_ALIGNMENT},
        {&router_correction_bias, "router_correction_bias", 0},
        {&routed_expert_down_proj, "routed_expert_down_proj", VECTOR_ALIGNMENT},
        {&routed_expert_up_proj, "routed_expert_up_proj", VECTOR_ALIGNMENT},
        {&routed_latent_rmsnorm_weight, "routed_latent_rmsnorm_weight",
         VECTOR_ALIGNMENT},
        // Stricter than the rest: the tensor map the fused gate/up engine reads
        // the payload through pins a 32-byte base. Its scales move by
        // `cp.async.bulk`, which pins the usual sixteen.
        {&expert_w13_packed, "expert_w13_packed",
         expert_mxfp4::fused_w13::kFusedPackedAlignment},
        {&expert_w13_scale, "expert_w13_scale", VECTOR_ALIGNMENT},
        {&expert_w2_packed, "expert_w2_packed", VECTOR_ALIGNMENT},
        {&expert_w2_scale, "expert_w2_scale", VECTOR_ALIGNMENT},
        {&shared_gate_proj, "shared_gate_proj", VECTOR_ALIGNMENT},
        {&shared_up_proj, "shared_up_proj", VECTOR_ALIGNMENT},
        {&shared_down_proj, "shared_down_proj", VECTOR_ALIGNMENT},
        {&scratch, "scratch", SCRATCH_ALIGNMENT},
        {&collective_buffer, "collective_buffer", VECTOR_ALIGNMENT},
        {&output_mailbox, "output_mailbox", VECTOR_ALIGNMENT},
        {&barrier_buffer, "barrier_buffer", 0},
        {&barrier_target, "barrier_target", 0},
        {&error_flag, "error_flag", 0},
    };
    const at::Device device = hidden_states.device();
    for (const DecodeTensor &entry : decode_tensors) {
        TORCH_CHECK(entry.tensor->device().is_cuda(),
                    "MoK: kimi_k3_decode requires a CUDA ", entry.name);
        TORCH_CHECK(entry.tensor->is_contiguous(),
                    "MoK: kimi_k3_decode requires a contiguous ", entry.name);
        TORCH_CHECK(entry.tensor->device() == device,
                    "MoK: kimi_k3_decode requires ", entry.name, " on ",
                    device);
        if (entry.alignment > 0) {
            check_tensor_alignment(*entry.tensor, kOperation, entry.name,
                                   entry.alignment);
        }
    }

    TORCH_CHECK(hidden_states.dim() == 2
                    && hidden_states.size(1) == kHiddenSize
                    && hidden_states.scalar_type() == at::kBFloat16,
                "MoK: kimi_k3_decode requires BF16 hidden_states [M, ",
                kHiddenSize, "]");
    const std::int64_t tokens = hidden_states.size(0);
    TORCH_CHECK(tokens >= 1 && tokens <= kMaxTokens,
                "MoK: kimi_k3_decode requires hidden_states with 1 to ",
                kMaxTokens, " tokens");
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= tokens,
                "MoK: kimi_k3_decode requires active_tokens in [1, ", tokens,
                "]");
    TORCH_CHECK(tp_rank >= 0 && tp_rank < kTensorParallelSize,
                "MoK: kimi_k3_decode requires tp_rank in [0, ",
                kTensorParallelSize - 1, "]");

    const auto check_bf16 = [](const at::Tensor &tensor,
                               const char *name,
                               const std::initializer_list<std::int64_t> shape) {
        TORCH_CHECK(tensor.sizes() == at::IntArrayRef(shape)
                        && tensor.scalar_type() == at::kBFloat16,
                    "MoK: kimi_k3_decode requires a BF16 ", name, " ",
                    at::IntArrayRef(shape));
    };
    check_bf16(router_weight, "router_weight", {kNumExperts, kHiddenSize});
    check_bf16(routed_expert_down_proj, "routed_expert_down_proj",
               {kLatentSize, kHiddenSize});
    check_bf16(routed_expert_up_proj, "routed_expert_up_proj",
               {kHiddenSize, kLatentSize});
    check_bf16(routed_latent_rmsnorm_weight, "routed_latent_rmsnorm_weight",
               {kLatentSize});
    check_bf16(shared_gate_proj, "shared_gate_proj",
               {shared_intermediate, kHiddenSize});
    check_bf16(shared_up_proj, "shared_up_proj",
               {shared_intermediate, kHiddenSize});
    check_bf16(shared_down_proj, "shared_down_proj",
               {kHiddenSize, shared_intermediate});
    TORCH_CHECK(router_correction_bias.dim() == 1
                    && router_correction_bias.size(0) == kNumExperts
                    && router_correction_bias.scalar_type() == at::kFloat,
                "MoK: kimi_k3_decode requires a float32 "
                "router_correction_bias [", kNumExperts, "]");

    const auto check_expert = [](const at::Tensor &tensor,
                                 const char *name,
                                 const int rows,
                                 const int columns) {
        TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == kNumExperts
                        && tensor.size(1) == rows
                        && tensor.size(2) == columns
                        && tensor.scalar_type() == at::kByte,
                    "MoK: kimi_k3_decode requires uint8 ", name, " [",
                    kNumExperts, ", ", rows, ", ", columns, "]");
    };
    // The routed gate and up projections arrive as one payload, in the
    // tile-major order the fused engine's descriptor reads: six output tasks of
    // 128 M rows, gate channels in the low half and their own up rows in the
    // high half, seven K = 512 slabs each. `mok.kimi_k3_w13` builds it and
    // `prepare_kimi_k3_decode_weights` is where the transform runs.
    check_expert(expert_w13_packed, "expert_w13_packed",
                 expert_mxfp4::fused_w13::kFusedPackedRows,
                 expert_mxfp4::fused_w13::kFusedPackedColumns);
    check_expert(expert_w13_scale, "expert_w13_scale",
                 expert_mxfp4::fused_w13::kFusedScaleRows,
                 expert_mxfp4::fused_w13::kFusedScaleColumns);
    check_expert(expert_w2_packed, "expert_w2_packed", kExpertW2PackedRows,
                 kExpertW2PackedColumns);
    check_expert(expert_w2_scale, "expert_w2_scale", kExpertW2PackedRows,
                 kExpertW2ScaleColumns);

    TORCH_CHECK(scratch.dim() == 1 && scratch.scalar_type() == at::kByte
                    && scratch.size(0) >= SCRATCH_BYTES,
                "MoK: kimi_k3_decode requires a uint8 scratch of at least ",
                SCRATCH_BYTES, " bytes");
    // The whole workspace is sized for the contract's 128 tokens, not for this
    // call's token count, because one workspace serves every shape.
    TORCH_CHECK(collective_buffer.dim() == 2
                    && collective_buffer.size(0) == kMaxTokens
                    && collective_buffer.size(1) == collective_columns
                    && collective_buffer.scalar_type() == at::kBFloat16,
                "MoK: kimi_k3_decode requires a BF16 collective_buffer [",
                kMaxTokens, ", ", collective_columns, "]");
    TORCH_CHECK(output_mailbox.dim() == 3
                    && output_mailbox.size(0) == kMaxTokens
                    && output_mailbox.size(1) == kTensorParallelSize
                    && output_mailbox.size(2) == shard_columns
                    && output_mailbox.scalar_type() == at::kBFloat16,
                "MoK: kimi_k3_decode requires a token-major BF16 "
                "output_mailbox [", kMaxTokens, ", ", kTensorParallelSize,
                ", ", shard_columns, "]");
    TORCH_CHECK(barrier_buffer.dim() == 1 && barrier_buffer.size(0) == 1
                    && barrier_buffer.scalar_type() == at::kInt,
                "MoK: kimi_k3_decode requires barrier_buffer to be int32 [1]");
    TORCH_CHECK(barrier_target.dim() == 1 && barrier_target.size(0) == 1
                    && barrier_target.scalar_type() == at::kInt,
                "MoK: kimi_k3_decode requires barrier_target to be int32 [1]");
    TORCH_CHECK(error_flag.dim() == 1 && error_flag.size(0) == 1
                    && error_flag.scalar_type() == at::kInt,
                "MoK: kimi_k3_decode requires error_flag to be int32 [1]");

    check_symmetric_pointers(
        kOperation, collective_buffer_ptrs, collective_buffer_multicast_ptr,
        collective_buffer, tp_rank, "collective_buffer_ptrs",
        "collective_buffer_multicast_ptr", VECTOR_ALIGNMENT);
    check_symmetric_pointers(
        kOperation, output_mailbox_ptrs, output_mailbox_multicast_ptr,
        output_mailbox, tp_rank, "output_mailbox_ptrs",
        "output_mailbox_multicast_ptr", VECTOR_ALIGNMENT);
    check_symmetric_pointers(
        kOperation, barrier_buffer_ptrs, barrier_buffer_multicast_ptr,
        barrier_buffer, tp_rank, "barrier_buffer_ptrs",
        "barrier_buffer_multicast_ptr",
        static_cast<std::int64_t>(sizeof(std::int32_t)));
    check_multicast_pointers_are_disjoint(
        kOperation, collective_buffer_multicast_ptr,
        output_mailbox_multicast_ptr, barrier_buffer_multicast_ptr);

    // Per-allocation rules cannot see a caller that took eight of these
    // pointers from one workspace and the ninth from another; the signature
    // was folded from all of them at once, so recomputing it here does.
    const std::int64_t recomputed = workspace_signature::compute(
        collective_buffer, collective_buffer_ptrs,
        collective_buffer_multicast_ptr, output_mailbox, output_mailbox_ptrs,
        output_mailbox_multicast_ptr, barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, tp_rank);
    TORCH_CHECK(recomputed == workspace_signature_value,
                "MoK: kimi_k3_decode requires workspace_signature to match the "
                "supplied tensors, pointer lists, multicast aliases, and "
                "tp_rank, but they hash to ", recomputed, " while the caller "
                "passed ", workspace_signature_value,
                ". One of these pointers belongs to a different workspace, or "
                "the signature does");

    const cudaDeviceProp &properties = check_sm103(hidden_states, kOperation);

    return persistent::LaunchArguments{
        hidden_states,
        router_weight,
        router_correction_bias,
        routed_expert_down_proj,
        routed_expert_up_proj,
        routed_latent_rmsnorm_weight,
        expert_w13_packed,
        expert_w13_scale,
        expert_w2_packed,
        expert_w2_scale,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        collective_buffer,
        collective_buffer_multicast_ptr,
        output_mailbox_multicast_ptr,
        barrier_buffer,
        barrier_buffer_multicast_ptr,
        barrier_target,
        scratch,
        error_flag,
        static_cast<int>(tp_rank),
        static_cast<int>(active_tokens),
        properties.multiProcessorCount,
        static_cast<int>(persistent::benchmark_grid_ctas_for_testing()),
        persistent::benchmark_phase_profile_enabled() ? 1 : 0,
    };
}

/// Run one whole TP8 Kimi K3 decode step in a single persistent launch.
///
/// The operator mutates the workspace and returns nothing: the assembled output
/// is this rank's own mailbox storage, and a custom operator may not return a
/// view that aliases one of its own mutated inputs. `mok.kimi_k3.kimi_k3_decode`
/// takes that view afterwards.
///
/// Every rank must call this with the same active token count, for the reason
/// spelled out above `kimi_k3_tail_entrypoint`: the cross-rank rendezvous is
/// driven by one coordinator thread whose arrival does not depend on the count,
/// so a divergent count returns mixed-generation rows rather than deadlocking.
static __host__ void kimi_k3_decode_entrypoint(
    const at::Tensor &hidden_states,
    const at::Tensor &router_weight,
    const at::Tensor &router_correction_bias,
    const at::Tensor &routed_expert_down_proj,
    const at::Tensor &routed_expert_up_proj,
    const at::Tensor &routed_latent_rmsnorm_weight,
    const at::Tensor &expert_w13_packed,
    const at::Tensor &expert_w13_scale,
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
    std::int64_t output_mailbox_multicast_ptr,
    const at::Tensor &barrier_buffer,
    const std::vector<std::int64_t> &barrier_buffer_ptrs,
    std::int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target,
    const at::Tensor &error_flag,
    std::int64_t tp_rank,
    std::int64_t active_tokens,
    std::int64_t workspace_signature_value
) {
    const persistent::LaunchArguments arguments = decode_launch_arguments(
        hidden_states, router_weight, router_correction_bias,
        routed_expert_down_proj, routed_expert_up_proj,
        routed_latent_rmsnorm_weight, expert_w13_packed, expert_w13_scale,
        expert_w2_packed, expert_w2_scale,
        shared_gate_proj, shared_up_proj, shared_down_proj, scratch,
        collective_buffer, collective_buffer_ptrs,
        collective_buffer_multicast_ptr, output_mailbox, output_mailbox_ptrs,
        output_mailbox_multicast_ptr, barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, barrier_target, error_flag, tp_rank,
        active_tokens, workspace_signature_value);

    // The step must run on the tensors' own device and that device's current
    // stream, whatever device happens to be current on entry.
    const c10::cuda::CUDAGuard device_guard(hidden_states.device());
    persistent::launch_decode(arguments);
}

}  // namespace kimi_k3_decode
