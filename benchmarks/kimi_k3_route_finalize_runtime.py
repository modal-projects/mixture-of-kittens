"""Private runtime boundary for the FP32 route-major finalize benchmark."""

from __future__ import annotations

import contextlib
import os
from dataclasses import replace
from typing import Iterator

import torch

from mok import _C
from mok.kimi_k3 import KimiK3DecodeWeights, KimiK3DecodeWorkspace


ROUTE_FINALIZE_GUARD = "MOK_KIMI_K3_ENABLE_ROUTE_FINALIZE"


@contextlib.contextmanager
def route_finalize_enabled() -> Iterator[None]:
    """Enable the private entrypoint for one dedicated benchmark process."""
    previous = os.environ.get(ROUTE_FINALIZE_GUARD)
    os.environ[ROUTE_FINALIZE_GUARD] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ROUTE_FINALIZE_GUARD, None)
        else:
            os.environ[ROUTE_FINALIZE_GUARD] = previous


def create_candidate_workspace(
    production: KimiK3DecodeWorkspace,
) -> KimiK3DecodeWorkspace:
    """Share symmetric buffers but own option A's max-sized scratch."""
    scratch = torch.zeros(
        _C._kimi_k3_route_finalize_workspace_bytes(),
        dtype=torch.uint8,
        device=production.device,
    )
    return replace(production, scratch=scratch)


def _arguments(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
) -> dict[str, object]:
    return {
        "hidden_states": hidden,
        "router_weight": weights.router_weight,
        "router_correction_bias": weights.router_correction_bias,
        "routed_expert_down_proj": weights.routed_expert_down_proj,
        "routed_expert_up_proj": weights.routed_expert_up_proj,
        "routed_latent_rmsnorm_weight": weights.routed_latent_rmsnorm_weight,
        "expert_w13_packed": weights.expert_w13_packed,
        "expert_w13_scale": weights.expert_w13_scale,
        "expert_w2_packed": weights.expert_w2_packed,
        "expert_w2_scale": weights.expert_w2_scale,
        "shared_gate_proj": weights.shared_gate_proj,
        "shared_up_proj": weights.shared_up_proj,
        "shared_down_proj": weights.shared_down_proj,
        "scratch": workspace.scratch,
        "collective_buffer": workspace.collective_buffer,
        "collective_buffer_ptrs": workspace.collective_ptrs,
        "collective_buffer_multicast_ptr": workspace.collective_multicast_ptr,
        "output_mailbox": workspace.output_mailbox,
        "output_mailbox_ptrs": workspace.output_mailbox_ptrs,
        "output_mailbox_multicast_ptr": (
            workspace.output_mailbox_multicast_ptr
        ),
        "barrier_buffer": workspace.barrier_buffer,
        "barrier_buffer_ptrs": workspace.barrier_ptrs,
        "barrier_buffer_multicast_ptr": workspace.barrier_multicast_ptr,
        "barrier_target": workspace.barrier_target,
        "error_flag": workspace.error_flag,
        "tp_rank": workspace.tp_rank,
        "active_tokens": hidden.shape[0],
        "workspace_signature": workspace.workspace_signature,
    }


def decode_device_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Launch option A once and return the mailbox view."""
    _C._kimi_k3_route_finalize(**_arguments(workspace, weights, hidden))
    tokens, ranks, shard_columns = workspace.output_mailbox.shape
    return workspace.output_mailbox.view(
        tokens, ranks * shard_columns
    )[: hidden.shape[0]]


def decode_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
) -> torch.Tensor:
    result = decode_device_step(workspace, weights, hidden)
    error = int(workspace.error_flag.item())
    if error:
        raise AssertionError(f"route-finalize kernel error flag: {error}")
    return result


__all__ = [
    "create_candidate_workspace",
    "decode_device_step",
    "decode_step",
    "route_finalize_enabled",
]
