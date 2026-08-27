"""Official numerical contract for the Kimi K3 decode MoE block."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


KIMI_K3_HIDDEN_SIZE = 7168
KIMI_K3_LATENT_SIZE = 3584
KIMI_K3_ROUTED_INTERMEDIATE_SIZE = 3072
KIMI_K3_SHARED_INTERMEDIATE_SIZE = 6144
KIMI_K3_NUM_EXPERTS = 896
KIMI_K3_TOPK = 16
KIMI_K3_TP_SIZE = 8
KIMI_K3_MAX_TOKENS = 128
KIMI_K3_RMS_EPS = 1e-5
KIMI_K3_SITU_BETA = 4.0
KIMI_K3_SITU_LINEAR_BETA = 25.0
KIMI_K3_CAPACITY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True, slots=True)
class KimiK3DecodeConfig:
    max_tokens: int = KIMI_K3_MAX_TOKENS


@dataclass(frozen=True, slots=True)
class KimiK3DecodeWeights:
    """Prepared TP8 weights for Kimi K3 decode.

    ``routed_expert_down_proj`` and ``routed_expert_up_proj`` are the replicated
    latent projections from 7168 to 3584 and from 3584 to 7168, respectively.
    They are distinct from the per-expert gate, up, and down projection tensors
    accepted by ``kimi_k3_moe_reference``.
    """

    router_weight: torch.Tensor
    router_correction_bias: torch.Tensor
    routed_expert_down_proj: torch.Tensor
    routed_expert_up_proj: torch.Tensor
    routed_latent_rmsnorm_weight: torch.Tensor
    expert_w1_packed: torch.Tensor
    expert_w1_scale: torch.Tensor
    expert_w3_packed: torch.Tensor
    expert_w3_scale: torch.Tensor
    expert_w2_packed: torch.Tensor
    expert_w2_scale: torch.Tensor
    shared_gate_proj: torch.Tensor
    shared_up_proj: torch.Tensor
    shared_down_proj: torch.Tensor
    tp_rank: int


@dataclass(slots=True)
class KimiK3DecodeWorkspace:
    group_name: str
    tp_rank: int
    tp_size: int
    device: torch.device
    max_tokens: int
    scratch: torch.Tensor
    collective_buffer: torch.Tensor
    collective_handle: Any
    collective_ptrs: list[int]
    collective_multicast_ptr: int
    output_mailbox: torch.Tensor
    output_mailbox_handle: Any
    output_mailbox_ptrs: list[int]
    barrier_buffer: torch.Tensor
    barrier_handle: Any
    barrier_ptrs: list[int]
    barrier_multicast_ptr: int
    barrier_target: torch.Tensor
    error_flag: torch.Tensor


_KIMI_K3_DECODE_WORKSPACE_CACHE: dict[
    tuple[str, int, int],
    KimiK3DecodeWorkspace,
] = {}


def _validate_kimi_k3_decode_workspace_args(
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    max_tokens: int,
) -> tuple[str, int, int, torch.device]:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    if type(max_tokens) is not int or max_tokens != KIMI_K3_MAX_TOKENS:
        raise ValueError(f"max_tokens must equal {KIMI_K3_MAX_TOKENS}")

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    device = torch.device("cuda", device_index)
    if device_index != torch.cuda.current_device():
        raise ValueError(
            "Kimi K3 decode workspace device must be the current CUDA device"
        )
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise NotImplementedError(
            "Kimi K3 decode workspace requires an SM103 GPU"
        )

    group_name = group.group_name
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError("process group must have a nonempty group_name")
    tp_rank = dist.get_rank(group=group)
    tp_size = dist.get_world_size(group=group)
    if tp_size != KIMI_K3_TP_SIZE:
        raise ValueError(
            f"Kimi K3 decode workspace requires TP{KIMI_K3_TP_SIZE}"
        )
    if not 0 <= tp_rank < tp_size:
        raise RuntimeError("current process is not a member of the TP process group")

    return group_name, tp_rank, tp_size, device


def create_kimi_k3_decode_workspace(
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    max_tokens: int = KIMI_K3_MAX_TOKENS,
) -> KimiK3DecodeWorkspace:
    """Create a new caller-owned TP8 Kimi K3 decode workspace."""
    from . import _C

    group_name, tp_rank, tp_size, device = (
        _validate_kimi_k3_decode_workspace_args(
            group,
            device=device,
            max_tokens=max_tokens,
        )
    )
    symm_mem.enable_symm_mem_for_group(group_name)

    def allocate_symmetric(
        *shape: int,
        dtype: torch.dtype,
        zero: bool = False,
    ) -> tuple[torch.Tensor, Any, list[int]]:
        buffer = symm_mem.empty(*shape, dtype=dtype, device=device)
        if zero:
            buffer.zero_()
        handle = symm_mem.rendezvous(buffer, group_name)
        pointers = [
            int(handle.buffer_ptrs[peer_rank])
            for peer_rank in range(tp_size)
        ]
        return buffer, handle, pointers

    scratch = torch.zeros(
        _C.kimi_k3_decode_workspace_bytes(),
        dtype=torch.uint8,
        device=device,
    )
    (
        collective_buffer,
        collective_handle,
        collective_ptrs,
    ) = allocate_symmetric(
        max_tokens,
        KIMI_K3_LATENT_SIZE + KIMI_K3_HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )
    collective_multicast_ptr = int(collective_handle.multicast_ptr)
    (
        output_mailbox,
        output_mailbox_handle,
        output_mailbox_ptrs,
    ) = allocate_symmetric(
        tp_size,
        max_tokens,
        KIMI_K3_NUM_EXPERTS,
        dtype=torch.bfloat16,
    )
    barrier_buffer, barrier_handle, barrier_ptrs = allocate_symmetric(
        1,
        dtype=torch.int32,
        zero=True,
    )
    barrier_multicast_ptr = int(barrier_handle.multicast_ptr)
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)
    error_flag = torch.zeros(1, dtype=torch.int32, device=device)

    dist.barrier(
        group=group,
        async_op=True,
        device_ids=[device.index],
    ).block_current_stream()

    return KimiK3DecodeWorkspace(
        group_name=group_name,
        tp_rank=tp_rank,
        tp_size=tp_size,
        device=device,
        max_tokens=max_tokens,
        scratch=scratch,
        collective_buffer=collective_buffer,
        collective_handle=collective_handle,
        collective_ptrs=collective_ptrs,
        collective_multicast_ptr=collective_multicast_ptr,
        output_mailbox=output_mailbox,
        output_mailbox_handle=output_mailbox_handle,
        output_mailbox_ptrs=output_mailbox_ptrs,
        barrier_buffer=barrier_buffer,
        barrier_handle=barrier_handle,
        barrier_ptrs=barrier_ptrs,
        barrier_multicast_ptr=barrier_multicast_ptr,
        barrier_target=barrier_target,
        error_flag=error_flag,
    )


def get_kimi_k3_decode_workspace(
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    max_tokens: int = KIMI_K3_MAX_TOKENS,
) -> KimiK3DecodeWorkspace:
    """Return the cached TP8 workspace, creating it when absent."""
    group_name, _, _, normalized_device = (
        _validate_kimi_k3_decode_workspace_args(
            group,
            device=device,
            max_tokens=max_tokens,
        )
    )
    cache_key = (
        group_name,
        normalized_device.index,
        KIMI_K3_MAX_TOKENS,
    )
    workspace = _KIMI_K3_DECODE_WORKSPACE_CACHE.get(cache_key)
    if workspace is None:
        workspace = create_kimi_k3_decode_workspace(
            group,
            device=normalized_device,
            max_tokens=max_tokens,
        )
        _KIMI_K3_DECODE_WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def clear_kimi_k3_decode_workspace_cache() -> None:
    """Synchronize participating ranks and release all cached workspaces."""
    workspaces = tuple(_KIMI_K3_DECODE_WORKSPACE_CACHE.values())
    if not workspaces:
        return

    from .ops import barrier_all

    for workspace in workspaces:
        barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_ptrs,
            workspace.barrier_multicast_ptr,
            workspace.barrier_target,
        )
    torch.cuda.synchronize(workspaces[0].device)
    _KIMI_K3_DECODE_WORKSPACE_CACHE.clear()


def validate_kimi_k3_decode_hidden_states(hidden_states: torch.Tensor) -> None:
    """Validate the shape and storage contract for one decode invocation."""
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != KIMI_K3_HIDDEN_SIZE:
        raise ValueError(
            f"hidden_states must have shape [M, {KIMI_K3_HIDDEN_SIZE}]"
        )
    num_tokens = hidden_states.shape[0]
    if not 1 <= num_tokens <= KIMI_K3_MAX_TOKENS:
        raise ValueError(
            f"hidden_states token count must be between 1 and {KIMI_K3_MAX_TOKENS}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must have dtype torch.bfloat16")
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")


def validate_kimi_k3_decode_inputs(
    hidden_states: torch.Tensor,
    weights: KimiK3DecodeWeights,
) -> None:
    """Validate fixed Kimi K3 dimensions and prepared-weight layouts."""
    validate_kimi_k3_decode_hidden_states(hidden_states)
    if not isinstance(weights, KimiK3DecodeWeights):
        raise TypeError("weights must be a KimiK3DecodeWeights instance")
    if type(weights.tp_rank) is not int or not 0 <= weights.tp_rank < KIMI_K3_TP_SIZE:
        raise ValueError(
            f"weights.tp_rank must be an integer between 0 and {KIMI_K3_TP_SIZE - 1}"
        )

    bf16_layouts = (
        ("router_weight", weights.router_weight,
         (KIMI_K3_NUM_EXPERTS, KIMI_K3_HIDDEN_SIZE)),
        ("routed_expert_down_proj", weights.routed_expert_down_proj,
         (KIMI_K3_LATENT_SIZE, KIMI_K3_HIDDEN_SIZE)),
        ("routed_expert_up_proj", weights.routed_expert_up_proj,
         (KIMI_K3_HIDDEN_SIZE, KIMI_K3_LATENT_SIZE)),
        ("routed_latent_rmsnorm_weight", weights.routed_latent_rmsnorm_weight,
         (KIMI_K3_LATENT_SIZE,)),
        ("shared_gate_proj", weights.shared_gate_proj,
         (KIMI_K3_SHARED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE,
          KIMI_K3_HIDDEN_SIZE)),
        ("shared_up_proj", weights.shared_up_proj,
         (KIMI_K3_SHARED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE,
          KIMI_K3_HIDDEN_SIZE)),
        ("shared_down_proj", weights.shared_down_proj,
         (KIMI_K3_HIDDEN_SIZE,
          KIMI_K3_SHARED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE)),
    )
    uint8_layouts = (
        ("expert_w1_packed", weights.expert_w1_packed, (896, 384, 1824)),
        ("expert_w1_scale", weights.expert_w1_scale, (896, 384, 114)),
        ("expert_w3_packed", weights.expert_w3_packed, (896, 384, 1824)),
        ("expert_w3_scale", weights.expert_w3_scale, (896, 384, 114)),
        ("expert_w2_packed", weights.expert_w2_packed, (896, 3584, 192)),
        ("expert_w2_scale", weights.expert_w2_scale, (896, 3584, 12)),
    )
    layouts = (
        *bf16_layouts,
        (
            "router_correction_bias",
            weights.router_correction_bias,
            (KIMI_K3_NUM_EXPERTS,),
        ),
        *uint8_layouts,
    )
    for name, tensor, expected_shape in layouts:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.device != hidden_states.device:
            raise ValueError(f"{name} must be on {hidden_states.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    for name, tensor, _ in bf16_layouts:
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
    if weights.router_correction_bias.dtype != torch.float32:
        raise TypeError("router_correction_bias must have dtype torch.float32")
    for name, tensor, _ in uint8_layouts:
        if tensor.dtype != torch.uint8:
            raise TypeError(f"{name} must have dtype torch.uint8")


def kimi_k3_router_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int = KIMI_K3_TOPK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using corrected scores and weight their raw scores."""
    scores = torch.sigmoid(hidden_states.float() @ router_weight.float().T)
    expert_ids = torch.topk(
        scores + correction_bias.float(),
        topk,
        dim=-1,
        sorted=False,
    ).indices
    router_weights = torch.gather(scores, dim=-1, index=expert_ids)
    router_weights = router_weights / (
        router_weights.sum(dim=-1, keepdim=True) + 1e-20
    )
    return expert_ids, router_weights


def kimi_k3_situ_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Apply Kimi K3's FP32 SiTU activation and return BF16."""
    gate_fp32 = gate.float()
    up_fp32 = up.float()
    return (
        KIMI_K3_SITU_BETA
        * torch.tanh(gate_fp32 / KIMI_K3_SITU_BETA)
        * torch.sigmoid(gate_fp32)
        * KIMI_K3_SITU_LINEAR_BETA
        * torch.tanh(up_fp32 / KIMI_K3_SITU_LINEAR_BETA)
    ).bfloat16()


def kimi_k3_rmsnorm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply Kimi K3's FP32 RMSNorm and return a BF16 weighted result."""
    hidden_states_fp32 = hidden_states.float()
    normalized = hidden_states_fp32 * torch.rsqrt(
        hidden_states_fp32.square().mean(-1, keepdim=True) + KIMI_K3_RMS_EPS
    )
    return normalized.bfloat16() * weight


def _require_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
        )


def kimi_k3_moe_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_latent_down_proj: torch.Tensor,
    routed_expert_gate_proj: torch.Tensor,
    routed_expert_up_proj: torch.Tensor,
    routed_expert_down_proj: torch.Tensor,
    routed_latent_norm_weight: torch.Tensor,
    routed_latent_up_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    *,
    hidden_size: int = KIMI_K3_HIDDEN_SIZE,
    latent_size: int = KIMI_K3_LATENT_SIZE,
    routed_intermediate_size: int = KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    shared_intermediate_size: int = KIMI_K3_SHARED_INTERMEDIATE_SIZE,
    num_experts: int = KIMI_K3_NUM_EXPERTS,
    topk: int = KIMI_K3_TOPK,
) -> torch.Tensor:
    """Evaluate the full Kimi K3 MoE block with an explicit expert loop."""
    num_tokens = hidden_states.shape[0]
    _require_shape("hidden_states", hidden_states, (num_tokens, hidden_size))
    _require_shape("router_weight", router_weight, (num_experts, hidden_size))
    _require_shape("correction_bias", correction_bias, (num_experts,))
    _require_shape(
        "routed_latent_down_proj",
        routed_latent_down_proj,
        (latent_size, hidden_size),
    )
    _require_shape(
        "routed_expert_gate_proj",
        routed_expert_gate_proj,
        (num_experts, routed_intermediate_size, latent_size),
    )
    _require_shape(
        "routed_expert_up_proj",
        routed_expert_up_proj,
        (num_experts, routed_intermediate_size, latent_size),
    )
    _require_shape(
        "routed_expert_down_proj",
        routed_expert_down_proj,
        (num_experts, latent_size, routed_intermediate_size),
    )
    _require_shape(
        "routed_latent_norm_weight",
        routed_latent_norm_weight,
        (latent_size,),
    )
    _require_shape(
        "routed_latent_up_proj",
        routed_latent_up_proj,
        (hidden_size, latent_size),
    )
    _require_shape(
        "shared_gate_proj",
        shared_gate_proj,
        (shared_intermediate_size, hidden_size),
    )
    _require_shape(
        "shared_up_proj",
        shared_up_proj,
        (shared_intermediate_size, hidden_size),
    )
    _require_shape(
        "shared_down_proj",
        shared_down_proj,
        (hidden_size, shared_intermediate_size),
    )

    expert_ids, router_weights = kimi_k3_router_reference(
        hidden_states,
        router_weight,
        correction_bias,
        topk=topk,
    )
    latent_states = hidden_states @ routed_latent_down_proj.T
    routed_latent = torch.zeros(
        (num_tokens, latent_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    for expert_idx in range(num_experts):
        token_indices, route_indices = torch.where(expert_ids == expert_idx)
        if token_indices.numel() == 0:
            continue
        expert_input = latent_states[token_indices]
        gate = expert_input @ routed_expert_gate_proj[expert_idx].T
        up = expert_input @ routed_expert_up_proj[expert_idx].T
        expert_hidden = kimi_k3_situ_reference(gate, up)
        expert_output = expert_hidden @ routed_expert_down_proj[expert_idx].T
        weighted_output = (
            expert_output.float()
            * router_weights[token_indices, route_indices].unsqueeze(-1)
        )
        routed_latent.index_add_(0, token_indices, weighted_output)

    routed_latent = routed_latent.bfloat16()
    routed_latent = kimi_k3_rmsnorm_reference(
        routed_latent,
        routed_latent_norm_weight,
    )
    routed_output = routed_latent @ routed_latent_up_proj.T

    shared_gate = hidden_states @ shared_gate_proj.T
    shared_up = hidden_states @ shared_up_proj.T
    shared_hidden = kimi_k3_situ_reference(shared_gate, shared_up)
    shared_output = shared_hidden @ shared_down_proj.T
    return routed_output + shared_output


__all__ = [
    "KIMI_K3_CAPACITY_BUCKETS",
    "KIMI_K3_HIDDEN_SIZE",
    "KIMI_K3_LATENT_SIZE",
    "KIMI_K3_MAX_TOKENS",
    "KIMI_K3_NUM_EXPERTS",
    "KIMI_K3_RMS_EPS",
    "KIMI_K3_ROUTED_INTERMEDIATE_SIZE",
    "KIMI_K3_SHARED_INTERMEDIATE_SIZE",
    "KIMI_K3_SITU_BETA",
    "KIMI_K3_SITU_LINEAR_BETA",
    "KIMI_K3_TOPK",
    "KIMI_K3_TP_SIZE",
    "KimiK3DecodeConfig",
    "KimiK3DecodeWorkspace",
    "KimiK3DecodeWeights",
    "clear_kimi_k3_decode_workspace_cache",
    "create_kimi_k3_decode_workspace",
    "get_kimi_k3_decode_workspace",
    "kimi_k3_moe_reference",
    "kimi_k3_rmsnorm_reference",
    "kimi_k3_router_reference",
    "kimi_k3_situ_reference",
    "validate_kimi_k3_decode_hidden_states",
    "validate_kimi_k3_decode_inputs",
]
