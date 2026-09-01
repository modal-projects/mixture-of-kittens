"""Independent CPU oracle for reduced-shape Kimi K3 reference tests."""

import torch


def kimi_k3_moe_oracle(
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
    topk: int,
) -> torch.Tensor:
    """Evaluate the Kimi K3 MoE equations without importing ``mok``."""
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

    latent_states = hidden_states @ routed_latent_down_proj.T
    routed_latent = torch.zeros_like(latent_states, dtype=torch.float32)
    for token_idx in range(hidden_states.shape[0]):
        token_latent = latent_states[token_idx : token_idx + 1]
        for route_idx in range(topk):
            expert_idx = int(expert_ids[token_idx, route_idx])
            gate = token_latent @ routed_expert_gate_proj[expert_idx].T
            up = token_latent @ routed_expert_up_proj[expert_idx].T
            hidden = (
                4.0 * torch.tanh(gate.float() / 4.0) * torch.sigmoid(gate.float())
                * 25.0 * torch.tanh(up.float() / 25.0)
            ).to(gate.dtype)
            expert_output = hidden @ routed_expert_down_proj[expert_idx].T
            routed_latent[token_idx].add_(
                router_weights[token_idx, route_idx] * expert_output[0].float()
            )

    routed_latent_bf16 = routed_latent.to(hidden_states.dtype)
    normalized = (
        routed_latent_bf16.float()
        * torch.rsqrt(
            routed_latent_bf16.float().square().mean(-1, keepdim=True) + 1e-5
        )
    ).to(routed_latent_bf16.dtype) * routed_latent_norm_weight
    routed_output = normalized @ routed_latent_up_proj.T

    shared_gate = hidden_states @ shared_gate_proj.T
    shared_up = hidden_states @ shared_up_proj.T
    shared_hidden = (
        4.0
        * torch.tanh(shared_gate.float() / 4.0)
        * torch.sigmoid(shared_gate.float())
        * 25.0
        * torch.tanh(shared_up.float() / 25.0)
    ).to(shared_gate.dtype)
    shared_output = shared_hidden @ shared_down_proj.T
    return routed_output + shared_output
