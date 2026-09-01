#pragma once

#include "types.cuh"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <vector>

namespace kimi_k3_decode {
namespace workspace_signature {

// ---------------------------------------------------------------------------
// One number that names a whole symmetric workspace.
// ---------------------------------------------------------------------------
//
// Every per-allocation rule the tail's entry point enforces looks at one
// pointer list at a time, so it cannot see a caller that mixes allocations
// *between* two workspaces that are each individually valid. Substituting one
// valid multicast alias from a second workspace is the sharpest case: the
// address is device memory, correctly aligned, distinct from every unicast
// entry, and distinct from the other two multicast aliases, so nothing local
// contradicts it -- and the launch then reduces into one workspace while
// publishing into another.
//
// The fix is to fold the whole tuple into one number when the workspace is
// created and to require the caller to hand that number back. The tail
// recomputes it from the arguments it was actually given and compares. That
// costs one pass over 31 integers on the host, needs no registry, no lock, and
// no allocation, and it deliberately does *not* say which workspace is the
// "right" one: a complete, self-consistent second workspace passed with its own
// signature is accepted, because it is a legitimate thing for a caller to own.
//
// The mixing is written in unsigned 64-bit arithmetic, where overflow is
// defined to wrap, and the result is masked to 63 bits. That keeps the value
// representable everywhere it has to travel: a non-negative `int64_t` in C++, a
// plain `int` in the operator schema, and an ordinary Python integer, with no
// implementation-defined signed overflow and no sign-dependent behavior. Each
// step -- xor, multiply by an odd constant, xor-shift -- is invertible over
// 2^64, so no single changed input can leave the state unchanged.

/// Golden-ratio and splitmix64 constants; both odd, so both multiplications are
/// bijections modulo 2^64.
inline constexpr std::uint64_t kFirstMultiplier = 0x9E3779B97F4A7C15ULL;
inline constexpr std::uint64_t kSecondMultiplier = 0xBF58476D1CE4E5B9ULL;

/// The leading hexadecimal digits of pi, used only as a nothing-up-my-sleeve
/// starting state so a workspace of all-zero pointers is not signature zero.
inline constexpr std::uint64_t kInitialState = 0x243F6A8885A308D3ULL;

/// Low 63 bits: the range shared by `int64_t`, the schema's `int`, and Python.
inline constexpr std::uint64_t kSignatureMask = 0x7FFFFFFFFFFFFFFFULL;

/// The number of integers folded into a signature: the rank, plus a local
/// pointer, eight peer pointers, and a multicast alias for each of the three
/// symmetric allocations.
inline constexpr int kSignatureInputs = 1 + 3 * (1 + kTensorParallelSize + 1);

/// Fold one 64-bit value into the running state.
__host__ inline std::uint64_t mix(
    std::uint64_t state,
    const std::uint64_t value
) {
    state ^= value;
    state *= kFirstMultiplier;
    state ^= state >> 29;
    state *= kSecondMultiplier;
    state ^= state >> 32;
    return state;
}

/// Fold one allocation's local pointer, peer list, and multicast alias.
__host__ inline std::uint64_t mix_allocation(
    std::uint64_t state,
    const at::Tensor &tensor,
    const std::vector<std::int64_t> &pointers,
    const std::int64_t multicast_pointer,
    const char *field
) {
    TORCH_CHECK(pointers.size() == static_cast<std::size_t>(kTensorParallelSize),
                "MoK: a Kimi K3 workspace signature needs exactly ",
                kTensorParallelSize, " pointers in ", field, ", got ",
                pointers.size());
    state = mix(state, reinterpret_cast<std::uint64_t>(tensor.data_ptr()));
    for (int rank = 0; rank < kTensorParallelSize; ++rank) {
        state = mix(state, static_cast<std::uint64_t>(pointers[rank]));
    }
    return mix(state, static_cast<std::uint64_t>(multicast_pointer));
}

/// Compute the signature of one rank's view of a symmetric workspace.
///
/// The argument order is the tail's own argument order, and it is part of the
/// definition: folding the same pointers in a different order gives a different
/// signature, which is what makes two swapped lists detectable.
static __host__ std::int64_t compute(
    const at::Tensor &collective_buffer,
    const std::vector<std::int64_t> &collective_buffer_ptrs,
    const std::int64_t collective_buffer_multicast_ptr,
    const at::Tensor &output_mailbox,
    const std::vector<std::int64_t> &output_mailbox_ptrs,
    const std::int64_t output_mailbox_multicast_ptr,
    const at::Tensor &barrier_buffer,
    const std::vector<std::int64_t> &barrier_buffer_ptrs,
    const std::int64_t barrier_buffer_multicast_ptr,
    const std::int64_t tp_rank
) {
    TORCH_CHECK(tp_rank >= 0 && tp_rank < kTensorParallelSize,
                "MoK: a Kimi K3 workspace signature needs tp_rank in [0, ",
                kTensorParallelSize - 1, "], got ", tp_rank);
    std::uint64_t state = mix(kInitialState, static_cast<std::uint64_t>(tp_rank));
    state = mix_allocation(state, collective_buffer, collective_buffer_ptrs,
                           collective_buffer_multicast_ptr,
                           "collective_buffer_ptrs");
    state = mix_allocation(state, output_mailbox, output_mailbox_ptrs,
                           output_mailbox_multicast_ptr,
                           "output_mailbox_ptrs");
    state = mix_allocation(state, barrier_buffer, barrier_buffer_ptrs,
                           barrier_buffer_multicast_ptr,
                           "barrier_buffer_ptrs");
    return static_cast<std::int64_t>(state & kSignatureMask);
}

}  // namespace workspace_signature
}  // namespace kimi_k3_decode
