"""A staged four-launch decode, as the negative control for "exactly one".

``test_the_whole_step_is_exactly_one_persistent_kernel_launch`` asserts that a
production step reaches the device as a single kernel. On its own that
assertion proves less than it looks like it does: a trace with one kernel in it
is also what a step that computed nothing would produce, so the assertion has
to be shown to *fail* on something that genuinely computes the right answer in
more than one launch.

That is what this file is. The adapter below drives the four private stages --
route and latent projection, mixed W4A8 routed experts, shared experts, and the
fused TP8 tail -- in sequence over the same workspace and the same weights the
production call uses. It lands on the same rows, and the profiler sees four K3
kernels rather than one. Nothing in ``mok`` calls it; it exists so the
one-launch gate has something to be measured against.

Eight ranks, so this file runs under ``torchrun --standalone
--nproc-per-node=8``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from mok import ops
from mok.kimi_k3 import (
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_MAX_TOKENS,
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
)

from .kimi_k3_decode_support import (
    CORE_TOKENS,
    HIDDEN,
    PERSISTENT_KERNEL,
    PRIVATE_STAGE_KERNELS,
    TENSOR_TOKENS,
    _synchronize_ranks,
    assert_decode_close,
    decode_reference,
    decode_step,
    hidden_states,
    profiled_kernel_names,
    weights,  # noqa: F401
    workspace,  # noqa: F401
)


@pytest.fixture(scope="module")
def routed_staging(
    tp8_context: tuple[int, int, torch.device],
) -> Iterator[torch.Tensor]:
    """The contiguous routed buffer the private expert stage insists on.

    The production kernel accumulates the routed partial straight into the
    collective buffer's first 3584 columns, which is a strided view; the
    private stage takes a contiguous ``[M, 3584]`` of its own. Owning one here
    is the only thing the adapter needs that the production path does not.
    """
    _, _, device = tp8_context
    buffer = torch.empty(
        KIMI_K3_MAX_TOKENS,
        KIMI_K3_LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    try:
        yield buffer
    finally:
        del buffer
        torch.cuda.empty_cache()


def staged_decode(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
    routed_staging: torch.Tensor,
) -> torch.Tensor:
    """One decode step as four private launches over one workspace.

    The order and the hand-offs are the production kernel's: the router
    publishes its assignment ranges and the projected latent into scratch, the
    routed experts consume them, both partials land in the symmetric collective
    buffer, and the tail reduces, normalizes, and scatters. The only difference
    is that each edge is a launch boundary instead of a grid phase barrier.
    """
    active = hidden.shape[0]
    _, _, latent = ops._kimi_k3_route_and_project(
        hidden,
        weights.router_weight,
        weights.router_correction_bias,
        weights.routed_expert_down_proj,
        workspace.scratch,
        active,
    )
    routed = ops._kimi_k3_routed_experts(
        latent,
        weights.expert_w1_packed,
        weights.expert_w1_scale,
        weights.expert_w3_packed,
        weights.expert_w3_scale,
        weights.expert_w2_packed,
        weights.expert_w2_scale,
        routed_staging[:active],
        workspace.scratch,
        active,
    )
    # The private stage sizes the collective buffer by the block it is given;
    # the tail below wants the whole 128-row allocation. Both are views of the
    # one symmetric buffer the production kernel writes.
    ops._kimi_k3_shared_experts(
        hidden,
        weights.shared_gate_proj,
        weights.shared_up_proj,
        weights.shared_down_proj,
        workspace.scratch,
        workspace.collective_buffer[:active],
        active,
    )
    workspace.collective_buffer[:active, :KIMI_K3_LATENT_SIZE].copy_(routed)
    return ops._kimi_k3_tail(
        weights.routed_latent_rmsnorm_weight,
        weights.routed_expert_up_proj,
        workspace.collective_buffer,
        workspace.collective_ptrs,
        workspace.collective_multicast_ptr,
        workspace.output_mailbox,
        workspace.output_mailbox_ptrs,
        workspace.output_mailbox_multicast_ptr,
        workspace.barrier_buffer,
        workspace.barrier_ptrs,
        workspace.barrier_multicast_ptr,
        workspace.barrier_target,
        workspace.scratch,
        workspace.error_flag,
        workspace.tp_rank,
        active,
        workspace.workspace_signature,
    )


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_the_staged_adapter_computes_the_same_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    routed_staging: torch.Tensor,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """The control has to be a real decode, or it controls for nothing.

    A staged path that returned the wrong rows would make the launch-count
    comparison below vacuous: it would only be showing that a broken
    computation traces differently.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    _synchronize_ranks(workspace)
    actual = staged_decode(workspace, weights, hidden, routed_staging)
    torch.cuda.synchronize(device)

    assert actual.shape == (tokens, HIDDEN)
    assert int(workspace.error_flag.item()) == 0
    assert_decode_close(actual, expected)


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_one_launch_is_measured_against_a_four_launch_control(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    routed_staging: torch.Tensor,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """The same trace, taken the same way, over two paths to the same rows.

    Both paths are warmed first, so neither trace carries a shared-memory
    reservation or a lazy initialization. What is left is the difference the
    gate is meant to catch: the production step is one kernel and names none of
    the private stages, while the staged control is four kernels and names
    nothing else.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    _synchronize_ranks(workspace)
    staged_decode(workspace, weights, hidden, routed_staging)
    _synchronize_ranks(workspace)
    decode_step(workspace, weights, hidden)
    _synchronize_ranks(workspace)

    staged_names = profiled_kernel_names(
        lambda: staged_decode(workspace, weights, hidden, routed_staging)
    )
    _synchronize_ranks(workspace)
    production_names = profiled_kernel_names(
        lambda: decode_step(workspace, weights, hidden)
    )

    staged_k3 = [
        name
        for name in staged_names
        if any(stage in name for stage in PRIVATE_STAGE_KERNELS)
    ]
    assert len(staged_k3) == 4, staged_names
    assert all(PERSISTENT_KERNEL not in name for name in staged_names), (
        staged_names
    )

    assert len(production_names) == 1, production_names
    assert PERSISTENT_KERNEL in production_names[0], production_names

    # And the control really did land on the same rows, so the difference
    # between the two traces is only how the step was launched.
    torch.cuda.synchronize(device)
    assert_decode_close(
        workspace.output_mailbox.view(KIMI_K3_MAX_TOKENS, HIDDEN)[:tokens],
        expected,
    )
