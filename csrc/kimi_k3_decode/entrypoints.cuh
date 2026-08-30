#pragma once

#include "pyutils/torchutils.cuh"

#include "kernel.cuh"
#include "mxfp4.cuh"
#include "persistent_kernel.cuh"
#include "types.cuh"
#include "workspace_signature.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <vector>

namespace kimi_k3_decode {

inline std::int64_t kimi_k3_decode_workspace_bytes() noexcept {
    return SCRATCH_BYTES;
}

/// Return one device's immutable properties, querying the driver once.
///
/// `cudaGetDeviceProperties` fills a kilobyte-sized record and is far too heavy
/// to sit on a decode hot path that is otherwise free of runtime API calls.
/// The record never changes for a device, so it is read once per ordinal and
/// kept.
static __host__ const cudaDeviceProp &cached_device_properties(
    const int device_index
) {
    static std::array<cudaDeviceProp, persistent::kMaxCudaDevices> properties{};
    static std::array<std::once_flag, persistent::kMaxCudaDevices> queried;
    TORCH_CHECK(device_index >= 0
                    && device_index < persistent::kMaxCudaDevices,
                "MoK: Kimi K3 supports CUDA devices 0 through ",
                persistent::kMaxCudaDevices - 1, ", got ", device_index);
    const auto slot = static_cast<std::size_t>(device_index);
    std::call_once(queried[slot], [slot, device_index] {
        const cudaError_t status =
            cudaGetDeviceProperties(&properties[slot], device_index);
        TORCH_CHECK(status == cudaSuccess,
                    "MoK: cudaGetDeviceProperties failed for device ",
                    device_index, ": ", cudaGetErrorString(status));
    });
    return properties[slot];
}

static __host__ const cudaDeviceProp &check_sm103(
    const at::Tensor &hidden_states,
    const char *name
) {
    const cudaDeviceProp &properties =
        cached_device_properties(hidden_states.get_device());
    TORCH_CHECK(properties.major == 10 && properties.minor == 3,
                "MoK: ", name, " requires SM103, found sm_",
                properties.major, properties.minor);
    return properties;
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
    const cudaDeviceProp &properties =
        check_sm103(hidden_states, "_kimi_k3_shared_experts");

    const c10::cuda::CUDAGuard device_guard(device);
    shared_experts::launch_shared_experts(
        hidden_states, shared_gate_proj, shared_up_proj, shared_down_proj,
        scratch, collective_buffer, static_cast<int>(active_tokens),
        properties.multiProcessorCount);
    return collective_buffer.narrow(0, 0, active_tokens)
        .narrow(1, kLatentSize, kHiddenSize);
}

/// Reject a peer-pointer list that is not exactly one positive pointer per rank.
/// Reject a peer-pointer list that does not describe this rank's own allocation.
///
/// The kernel only ever dereferences the multicast alias, so these lists are the
/// one place a caller can reveal that it mixed up a rank, an allocation, or a
/// whole workspace -- and a mix-up is silent otherwise: the launch simply
/// reduces the wrong columns or fills the wrong mailbox slot. Every rule below
/// is one that PyTorch's symmetric memory already satisfies for a valid handle:
/// exactly one positive, distinct, suitably aligned pointer per rank, with this
/// rank's slot holding the local tensor's own address, plus a multicast alias
/// that aliases none of them.
static __host__ void check_symmetric_pointers(
    const char *operation,
    const std::vector<std::int64_t> &pointers,
    const std::int64_t multicast_pointer,
    const at::Tensor &tensor,
    const std::int64_t tp_rank,
    const char *list_field,
    const char *multicast_field,
    const std::int64_t alignment
) {
    TORCH_CHECK(pointers.size() == static_cast<std::size_t>(kTensorParallelSize),
                "MoK: ", operation, " requires ", list_field, " with exactly ",
                kTensorParallelSize, " pointers, got ", pointers.size());
    for (int rank = 0; rank < kTensorParallelSize; ++rank) {
        TORCH_CHECK(pointers[rank] > 0,
                    "MoK: ", operation, " requires ", list_field,
                    " to hold only positive device pointers, but entry ", rank,
                    " is ", pointers[rank]);
    }

    // Checked before alignment and distinctness so that a substituted rank or a
    // swapped list is always reported as what it is.
    const auto local = reinterpret_cast<std::int64_t>(tensor.data_ptr());
    TORCH_CHECK(
        pointers[static_cast<std::size_t>(tp_rank)] == local,
        "MoK: ", operation, " requires ", list_field,
        "[tp_rank] to be this rank's own device pointer, but entry ", tp_rank,
        " is ", pointers[static_cast<std::size_t>(tp_rank)],
        " while the matching tensor is at ", local,
        ". The pointer list, the tensor, or tp_rank came from a different rank "
        "or a different workspace");

    for (int rank = 0; rank < kTensorParallelSize; ++rank) {
        TORCH_CHECK(pointers[rank] % alignment == 0,
                    "MoK: ", operation, " requires every ", list_field,
                    " entry aligned to ", alignment, " bytes, but entry ", rank,
                    " is ", pointers[rank] % alignment, " bytes past one");
    }
    for (int rank = 0; rank < kTensorParallelSize; ++rank) {
        for (int peer = rank + 1; peer < kTensorParallelSize; ++peer) {
            TORCH_CHECK(pointers[rank] != pointers[peer],
                        "MoK: ", operation, " requires ", list_field,
                        " to hold one distinct pointer per rank, but entries ",
                        rank, " and ", peer, " are both ", pointers[rank]);
        }
    }

    // Ask the driver who owns each address. On a live symmetric handle every
    // entry resolves to device memory on the peer that allocated it, so the
    // eight entries name eight distinct devices and this rank's entry names the
    // device its own tensor lives on. That catches a list stitched together
    // from two workspaces even when its addresses happen to be distinct.
    //
    // A failed lookup is never treated as a rejection: an address the driver
    // declines to describe is left to the launch, so a valid mapping on a setup
    // that does not expose peer attributes still works.
    int owner[kTensorParallelSize] = {};
    bool owner_known[kTensorParallelSize] = {};
    for (int rank = 0; rank < kTensorParallelSize; ++rank) {
        cudaPointerAttributes attributes{};
        const cudaError_t status = cudaPointerGetAttributes(
            &attributes, reinterpret_cast<void *>(pointers[rank]));
        owner_known[rank] = status == cudaSuccess;
        if (!owner_known[rank]) {
            cudaGetLastError();
            continue;
        }
        owner[rank] = attributes.device;
        TORCH_CHECK(attributes.type == cudaMemoryTypeDevice,
                    "MoK: ", operation, " requires every ", list_field,
                    " entry to name device memory, but entry ", rank,
                    " is of CUDA memory type ",
                    static_cast<int>(attributes.type));
    }
    for (int rank = 0; rank < kTensorParallelSize; ++rank) {
        if (!owner_known[rank]) continue;
        for (int peer = rank + 1; peer < kTensorParallelSize; ++peer) {
            if (!owner_known[peer]) continue;
            TORCH_CHECK(owner[rank] != owner[peer],
                        "MoK: ", operation, " requires ", list_field,
                        " to hold one distinct device per rank, but entries ",
                        rank, " and ", peer, " both live on CUDA device ",
                        owner[rank]);
        }
    }
    if (owner_known[static_cast<std::size_t>(tp_rank)]) {
        TORCH_CHECK(owner[static_cast<std::size_t>(tp_rank)]
                        == tensor.get_device(),
                    "MoK: ", operation, " requires ", list_field,
                    "[tp_rank] to live on the same device as its tensor, but "
                    "entry ", tp_rank, " is on CUDA device ",
                    owner[static_cast<std::size_t>(tp_rank)],
                    " while the tensor is on ", tensor.get_device());
    }

    TORCH_CHECK(multicast_pointer > 0,
                "MoK: ", operation, " requires ", multicast_field,
                " to be a positive device pointer, got ", multicast_pointer);
    TORCH_CHECK(multicast_pointer % alignment == 0,
                "MoK: ", operation, " requires ", multicast_field,
                " aligned to ", alignment, " bytes, but it is ",
                multicast_pointer % alignment, " bytes past one");
    for (int rank = 0; rank < kTensorParallelSize; ++rank) {
        TORCH_CHECK(
            multicast_pointer != pointers[rank],
            "MoK: ", operation, " requires one distinct multicast pointer per "
            "symmetric allocation, but ", multicast_field, " equals ",
            list_field, " entry ", rank);
    }
    cudaPointerAttributes multicast_attributes{};
    if (cudaPointerGetAttributes(
            &multicast_attributes,
            reinterpret_cast<void *>(multicast_pointer)) == cudaSuccess) {
        TORCH_CHECK(multicast_attributes.type == cudaMemoryTypeDevice,
                    "MoK: ", operation, " requires ", multicast_field,
                    " to name device memory, but it is of CUDA memory type ",
                    static_cast<int>(multicast_attributes.type));
    } else {
        cudaGetLastError();
    }
}

/// Reject a multicast pointer that belongs to one of the other allocations.
///
/// A per-allocation check cannot see this: each of the three pointers is
/// individually valid, so only comparing them against each other reveals that
/// the caller pointed two allocations at the same fabric address.
static __host__ void check_multicast_pointers_are_disjoint(
    const char *operation,
    const std::int64_t collective_buffer_multicast_ptr,
    const std::int64_t output_mailbox_multicast_ptr,
    const std::int64_t barrier_buffer_multicast_ptr
) {
    const std::int64_t pointers[3] = {
        collective_buffer_multicast_ptr,
        output_mailbox_multicast_ptr,
        barrier_buffer_multicast_ptr};
    const char *const fields[3] = {
        "collective_buffer_multicast_ptr",
        "output_mailbox_multicast_ptr",
        "barrier_buffer_multicast_ptr"};
    for (int first = 0; first < 3; ++first) {
        for (int second = first + 1; second < 3; ++second) {
            TORCH_CHECK(
                pointers[first] != pointers[second],
                "MoK: ", operation, " requires one distinct multicast pointer "
                "per symmetric allocation, but ", fields[first], " and ",
                fields[second], " are both ", pointers[first]);
        }
    }
}

/// Compute the signature that binds one rank's view of a workspace together.
///
/// Exposed so `create_kimi_k3_decode_workspace` can record the signature of the
/// workspace it just built, using the same code the tail checks against.
static __host__ std::int64_t kimi_k3_workspace_signature_entrypoint(
    const at::Tensor &collective_buffer,
    const std::vector<std::int64_t> &collective_buffer_ptrs,
    std::int64_t collective_buffer_multicast_ptr,
    const at::Tensor &output_mailbox,
    const std::vector<std::int64_t> &output_mailbox_ptrs,
    std::int64_t output_mailbox_multicast_ptr,
    const at::Tensor &barrier_buffer,
    const std::vector<std::int64_t> &barrier_buffer_ptrs,
    std::int64_t barrier_buffer_multicast_ptr,
    std::int64_t tp_rank
) {
    CHECK_INPUT(collective_buffer);
    CHECK_INPUT(output_mailbox);
    CHECK_INPUT(barrier_buffer);
    return workspace_signature::compute(
        collective_buffer, collective_buffer_ptrs,
        collective_buffer_multicast_ptr, output_mailbox, output_mailbox_ptrs,
        output_mailbox_multicast_ptr, barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, tp_rank);
}

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
        {&expert_w1_packed, "expert_w1_packed", VECTOR_ALIGNMENT},
        {&expert_w1_scale, "expert_w1_scale", VECTOR_ALIGNMENT},
        {&expert_w3_packed, "expert_w3_packed", VECTOR_ALIGNMENT},
        {&expert_w3_scale, "expert_w3_scale", VECTOR_ALIGNMENT},
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
    check_expert(expert_w1_packed, "expert_w1_packed", kExpertW1W3PackedRows,
                 kExpertW1W3PackedColumns);
    check_expert(expert_w1_scale, "expert_w1_scale", kExpertW1W3PackedRows,
                 kExpertW1W3ScaleColumns);
    check_expert(expert_w3_packed, "expert_w3_packed", kExpertW1W3PackedRows,
                 kExpertW1W3PackedColumns);
    check_expert(expert_w3_scale, "expert_w3_scale", kExpertW1W3PackedRows,
                 kExpertW1W3ScaleColumns);
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

    // The step must run on the tensors' own device and that device's current
    // stream, whatever device happens to be current on entry.
    const c10::cuda::CUDAGuard device_guard(device);
    persistent::launch_decode(persistent::LaunchArguments{
        hidden_states,
        router_weight,
        router_correction_bias,
        routed_expert_down_proj,
        routed_expert_up_proj,
        routed_latent_rmsnorm_weight,
        expert_w1_packed,
        expert_w1_scale,
        expert_w3_packed,
        expert_w3_scale,
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
        persistent::benchmark_projection_first_enabled() ? 1 : 0,
    });
}

}  // namespace kimi_k3_decode
