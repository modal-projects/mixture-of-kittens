#pragma once

/// The rules a symmetric-memory handle has to satisfy to be launched on.
///
/// The kernel only ever dereferences the multicast alias, so a caller that
/// mixed up a rank, an allocation, or a whole workspace is silent otherwise:
/// the launch reduces the wrong columns or fills the wrong mailbox slot. Every
/// rule here is one a valid PyTorch symmetric-memory handle already satisfies.

#include "private_stages.cuh"

namespace kimi_k3_decode {

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

}  // namespace kimi_k3_decode
