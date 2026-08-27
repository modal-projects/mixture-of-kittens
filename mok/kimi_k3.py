"""Official numerical contract for the Kimi K3 decode MoE block."""

from dataclasses import dataclass

import torch


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
    "kimi_k3_moe_reference",
    "kimi_k3_rmsnorm_reference",
    "kimi_k3_router_reference",
    "kimi_k3_situ_reference",
]
