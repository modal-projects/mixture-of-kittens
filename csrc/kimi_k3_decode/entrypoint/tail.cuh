#pragma once

/// The private tail entrypoint, and the workspace signature it checks.
///
/// The signature is what binds one rank's view of a workspace together, and it
/// is computed here with the same code the tail checks against so that
/// `create_kimi_k3_decode_workspace` can record what it just built.

#include "symmetric_memory.cuh"

namespace kimi_k3_decode {

/// Run the fused TP8 latent-MoE tail for one decode step, in one launch.
///
/// Every rank must call this with the same active token count. Both rendezvous
/// are driven by one coordinator thread per rank whose arrival is independent of
/// the token count, so a divergent count does not deadlock: it silently returns
/// wrong data. A rank that passed a smaller count never multicasts its shard for
/// the rows beyond it, so those rows of that rank's mailbox slot keep whatever
/// the previous launch left there, and the ranks that passed a larger count read
/// the resulting mixed-generation rows as if they were current. The caller owns
/// this agreement; it is not checkable from one rank.
static __host__ void kimi_k3_tail_entrypoint(
    const at::Tensor &routed_latent_rmsnorm_weight,
    const at::Tensor &latent_up_proj,
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
    const at::Tensor &scratch,
    const at::Tensor &error_flag,
    std::int64_t tp_rank,
    std::int64_t active_tokens,
    std::int64_t workspace_signature_value
) {
    CHECK_INPUT(routed_latent_rmsnorm_weight);
    CHECK_INPUT(latent_up_proj);
    CHECK_INPUT(collective_buffer);
    CHECK_INPUT(output_mailbox);
    CHECK_INPUT(barrier_buffer);
    CHECK_INPUT(barrier_target);
    CHECK_INPUT(scratch);
    CHECK_INPUT(error_flag);

    constexpr int shard_columns = kHiddenSize / kTensorParallelSize;
    constexpr int collective_columns = kLatentSize + kHiddenSize;

    TORCH_CHECK(routed_latent_rmsnorm_weight.dim() == 1
                    && routed_latent_rmsnorm_weight.size(0) == kLatentSize
                    && routed_latent_rmsnorm_weight.scalar_type()
                        == at::kBFloat16,
                "MoK: _kimi_k3_tail requires a BF16 "
                "routed_latent_rmsnorm_weight [", kLatentSize, "]");
    TORCH_CHECK(latent_up_proj.dim() == 2
                    && latent_up_proj.size(0) == kHiddenSize
                    && latent_up_proj.size(1) == kLatentSize
                    && latent_up_proj.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_tail requires a BF16 latent_up_proj [",
                kHiddenSize, ", ", kLatentSize, "]");
    TORCH_CHECK(collective_buffer.dim() == 2
                    && collective_buffer.size(0) == kMaxTokens
                    && collective_buffer.size(1) == collective_columns
                    && collective_buffer.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_tail requires a BF16 collective_buffer [",
                kMaxTokens, ", ", collective_columns, "]");
    TORCH_CHECK(output_mailbox.dim() == 3
                    && output_mailbox.size(0) == kMaxTokens
                    && output_mailbox.size(1) == kTensorParallelSize
                    && output_mailbox.size(2) == shard_columns
                    && output_mailbox.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_tail requires a token-major BF16 output_mailbox [",
                kMaxTokens, ", ", kTensorParallelSize, ", ", shard_columns, "]");
    TORCH_CHECK(barrier_buffer.dim() == 1 && barrier_buffer.size(0) == 1
                    && barrier_buffer.scalar_type() == at::kInt,
                "MoK: _kimi_k3_tail requires barrier_buffer to be int32 [1]");
    TORCH_CHECK(barrier_target.dim() == 1 && barrier_target.size(0) == 1
                    && barrier_target.scalar_type() == at::kInt,
                "MoK: _kimi_k3_tail requires barrier_target to be int32 [1]");
    TORCH_CHECK(error_flag.dim() == 1 && error_flag.size(0) == 1
                    && error_flag.scalar_type() == at::kInt,
                "MoK: _kimi_k3_tail requires error_flag to be int32 [1]");
    TORCH_CHECK(scratch.dim() == 1 && scratch.scalar_type() == at::kByte
                    && scratch.size(0) >= SCRATCH_BYTES,
                "MoK: _kimi_k3_tail requires a uint8 scratch of at least ",
                SCRATCH_BYTES, " bytes");
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: _kimi_k3_tail requires active_tokens in [1, ",
                kMaxTokens, "]");
    TORCH_CHECK(tp_rank >= 0 && tp_rank < kTensorParallelSize,
                "MoK: _kimi_k3_tail requires tp_rank in [0, ",
                kTensorParallelSize - 1, "]");

    // The two BF16 allocations are dereferenced with 16-byte multimem octets;
    // the barrier is a single int32 word.
    check_symmetric_pointers(
        "_kimi_k3_tail", collective_buffer_ptrs,
        collective_buffer_multicast_ptr, collective_buffer, tp_rank,
        "collective_buffer_ptrs", "collective_buffer_multicast_ptr",
        VECTOR_ALIGNMENT);
    check_symmetric_pointers(
        "_kimi_k3_tail", output_mailbox_ptrs, output_mailbox_multicast_ptr,
        output_mailbox, tp_rank, "output_mailbox_ptrs",
        "output_mailbox_multicast_ptr", VECTOR_ALIGNMENT);
    check_symmetric_pointers(
        "_kimi_k3_tail", barrier_buffer_ptrs, barrier_buffer_multicast_ptr,
        barrier_buffer, tp_rank, "barrier_buffer_ptrs",
        "barrier_buffer_multicast_ptr",
        static_cast<std::int64_t>(sizeof(std::int32_t)));
    check_multicast_pointers_are_disjoint(
        "_kimi_k3_tail", collective_buffer_multicast_ptr,
        output_mailbox_multicast_ptr, barrier_buffer_multicast_ptr);

    // Every rule above is per allocation, so none of them can see a caller that
    // took eight of these pointers from one workspace and the ninth from
    // another. The signature can: it was folded from all of them at once when
    // the workspace was created, so recomputing it here from the arguments
    // actually supplied is enough to reject any tuple that was not built
    // together. It intentionally accepts a complete second workspace passed
    // with its own signature.
    const std::int64_t recomputed = workspace_signature::compute(
        collective_buffer, collective_buffer_ptrs,
        collective_buffer_multicast_ptr, output_mailbox, output_mailbox_ptrs,
        output_mailbox_multicast_ptr, barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, tp_rank);
    TORCH_CHECK(recomputed == workspace_signature_value,
                "MoK: _kimi_k3_tail requires workspace_signature to match the "
                "supplied tensors, pointer lists, multicast aliases, and "
                "tp_rank, but they hash to ", recomputed, " while the caller "
                "passed ", workspace_signature_value,
                ". One of these pointers belongs to a different workspace, or "
                "the signature does");

    const at::Device device = output_mailbox.device();
    for (const auto &item : {
             std::pair<const at::Tensor *, const char *>{
                 &routed_latent_rmsnorm_weight,
                 "routed_latent_rmsnorm_weight"},
             {&latent_up_proj, "latent_up_proj"},
             {&collective_buffer, "collective_buffer"},
             {&barrier_buffer, "barrier_buffer"},
             {&barrier_target, "barrier_target"},
             {&error_flag, "error_flag"},
             {&scratch, "scratch"}}) {
        TORCH_CHECK(item.first->device() == device,
                    "MoK: _kimi_k3_tail requires ", item.second, " on ",
                    device);
    }

    for (const auto &item : {
             std::pair<const at::Tensor *, const char *>{
                 &routed_latent_rmsnorm_weight,
                 "routed_latent_rmsnorm_weight"},
             {&latent_up_proj, "latent_up_proj"},
             {&collective_buffer, "collective_buffer"},
             {&output_mailbox, "output_mailbox"}}) {
        check_tensor_alignment(*item.first, "_kimi_k3_tail", item.second,
                               VECTOR_ALIGNMENT);
    }
    check_tensor_alignment(scratch, "_kimi_k3_tail", "scratch",
                           SCRATCH_ALIGNMENT);

    const cudaDeviceProp &properties =
        check_sm103(output_mailbox, "_kimi_k3_tail");

    const c10::cuda::CUDAGuard device_guard(device);
    tail::launch_tail(
        routed_latent_rmsnorm_weight, latent_up_proj,
        collective_buffer_multicast_ptr, output_mailbox_multicast_ptr,
        barrier_buffer, barrier_buffer_multicast_ptr, barrier_target, scratch,
        error_flag, static_cast<int>(tp_rank),
        static_cast<int>(active_tokens), properties.multiProcessorCount);
}

}  // namespace kimi_k3_decode
