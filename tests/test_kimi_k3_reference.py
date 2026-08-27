"""CPU tests for the official Kimi K3 numerical contract."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

import pytest
import torch

from .kimi_k3_reference import kimi_k3_moe_oracle


_MODULE: ModuleType | None = None


def _load_kimi_k3() -> ModuleType:
    """Load the extension-free module without executing ``mok.__init__``."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    module_path = Path(__file__).parents[1] / "mok" / "kimi_k3.py"
    if not module_path.is_file():
        raise ModuleNotFoundError("No module named 'mok.kimi_k3'")
    spec = importlib.util.spec_from_file_location("_mok_kimi_k3_under_test", module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError("No module named 'mok.kimi_k3'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def test_official_constants_and_config_are_fixed() -> None:
    kimi_k3 = _load_kimi_k3()

    assert kimi_k3.KIMI_K3_HIDDEN_SIZE == 7168
    assert kimi_k3.KIMI_K3_LATENT_SIZE == 3584
    assert kimi_k3.KIMI_K3_ROUTED_INTERMEDIATE_SIZE == 3072
    assert kimi_k3.KIMI_K3_SHARED_INTERMEDIATE_SIZE == 6144
    assert kimi_k3.KIMI_K3_NUM_EXPERTS == 896
    assert kimi_k3.KIMI_K3_TOPK == 16
    assert kimi_k3.KIMI_K3_TP_SIZE == 8
    assert kimi_k3.KIMI_K3_MAX_TOKENS == 128
    assert kimi_k3.KIMI_K3_RMS_EPS == 1e-5
    assert kimi_k3.KIMI_K3_SITU_BETA == 4.0
    assert kimi_k3.KIMI_K3_SITU_LINEAR_BETA == 25.0
    assert kimi_k3.KIMI_K3_CAPACITY_BUCKETS == (1, 2, 4, 8, 16, 32, 64, 128)

    config = kimi_k3.KimiK3DecodeConfig()
    assert config.max_tokens == 128
    with pytest.raises(FrozenInstanceError):
        config.max_tokens = 64


def test_router_bias_changes_selection_not_weight() -> None:
    kimi_k3_router_reference = _load_kimi_k3().kimi_k3_router_reference
    x = torch.tensor([[1.0, 0.0]])
    router = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    bias = torch.tensor([-3.0, 3.0, 0.0])
    ids, weights = kimi_k3_router_reference(x, router, bias, topk=2)
    raw = torch.sigmoid(x.float() @ router.float().T)
    assert ids.tolist() == [[1, 2]]
    expected = raw[:, [1, 2]] / raw[:, [1, 2]].sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected)


def test_situ_uses_fp32_gate_and_up_clamps() -> None:
    kimi_k3_situ_reference = _load_kimi_k3().kimi_k3_situ_reference
    gate = torch.tensor([[8.0, -8.0]], dtype=torch.bfloat16)
    up = torch.tensor([[50.0, -50.0]], dtype=torch.bfloat16)
    actual = kimi_k3_situ_reference(gate, up)
    expected = (
        4.0 * torch.tanh(gate.float() / 4.0) * torch.sigmoid(gate.float())
        * 25.0 * torch.tanh(up.float() / 25.0)
    ).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rmsnorm_uses_epsilon_one_e_minus_five() -> None:
    kimi_k3_rmsnorm_reference = _load_kimi_k3().kimi_k3_rmsnorm_reference
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    gamma = torch.tensor([1.0, 0.5, 2.0], dtype=torch.bfloat16)
    expected = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    ).bfloat16() * gamma
    torch.testing.assert_close(kimi_k3_rmsnorm_reference(x, gamma), expected)


def test_reduced_shape_full_moe_matches_official_equations() -> None:
    kimi_k3_moe_reference = _load_kimi_k3().kimi_k3_moe_reference
    generator = torch.Generator().manual_seed(20260827)
    num_tokens = 3
    hidden_size = 4
    latent_size = 3
    routed_intermediate_size = 2
    shared_intermediate_size = 4
    num_experts = 3
    topk = 2

    def bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.25).bfloat16()

    hidden_states = bf16((num_tokens, hidden_size))
    router_weight = bf16((num_experts, hidden_size))
    correction_bias = torch.tensor([0.125, -0.25, 0.375])
    routed_latent_down_proj = bf16((latent_size, hidden_size))
    routed_expert_gate_proj = bf16(
        (num_experts, routed_intermediate_size, latent_size)
    )
    routed_expert_up_proj = bf16(
        (num_experts, routed_intermediate_size, latent_size)
    )
    routed_expert_down_proj = bf16(
        (num_experts, latent_size, routed_intermediate_size)
    )
    routed_latent_norm_weight = bf16((latent_size,))
    routed_latent_up_proj = bf16((hidden_size, latent_size))
    shared_gate_proj = bf16((shared_intermediate_size, hidden_size))
    shared_up_proj = bf16((shared_intermediate_size, hidden_size))
    shared_down_proj = bf16((hidden_size, shared_intermediate_size))
    args = (
        hidden_states,
        router_weight,
        correction_bias,
        routed_latent_down_proj,
        routed_expert_gate_proj,
        routed_expert_up_proj,
        routed_expert_down_proj,
        routed_latent_norm_weight,
        routed_latent_up_proj,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
    )

    expected = kimi_k3_moe_oracle(*args, topk=topk)
    actual = kimi_k3_moe_reference(
        *args,
        hidden_size=hidden_size,
        latent_size=latent_size,
        routed_intermediate_size=routed_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
        num_experts=num_experts,
        topk=topk,
    )

    assert actual.shape == (num_tokens, hidden_size)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
