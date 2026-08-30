"""Private full-step fused-W13 benchmark contracts and B300 parity checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from benchmarks.frameworks.kimi_k3_adapter_common import native_weights
from benchmarks.kimi_k3_decode_runtime import (
    check_decode_error,
    decode_fused_w13_benchmark_device_step,
    profiled_kernel_names,
)
from mok import _C
from mok.kimi_k3 import KimiK3DecodeWeights, KimiK3DecodeWorkspace

from .kimi_k3_decode_support import (
    CONFIG,
    decode_step,
    hidden_states,
    weights,  # noqa: F401
    workspace,  # noqa: F401
)


ROOT = Path(__file__).parents[1]
BENCHMARK_GUARD = "MOK_KIMI_K3_ENABLE_FUSED_W13_BENCHMARK"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_fused_w13_entrypoint_is_marked_private_and_benchmark_only() -> None:
    schema = _source("mok/ops.py")
    binding = _source("csrc/bindings.cu")
    entrypoint = _source("csrc/kimi_k3_decode/entrypoints.cuh")

    name = "_kimi_k3_decode_fused_w13_benchmark"
    assert f'"{name}("' in schema
    assert f'm.def("{name}"' in binding
    assert "PRIVATE BENCHMARK-ONLY" in schema
    assert "PRIVATE BENCHMARK-ONLY" in binding
    assert "PRIVATE BENCHMARK-ONLY" in entrypoint
    assert BENCHMARK_GUARD in schema
    assert BENCHMARK_GUARD in entrypoint


def test_fused_w13_kernel_uses_the_native_per_expert_row_stride() -> None:
    source = _source("csrc/kimi_k3_decode/expert_mxfp4.cuh")

    assert "template<bool FUSED_W13>" in source
    assert "2 * kExpertW1W3PackedRows" in source
    assert "weight_half * kExpertW1W3PackedRows" in source


@pytest.mark.parametrize("tokens", [16, 128])
def test_fused_w13_full_step_matches_public_decode(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
) -> None:
    _, _, device = tp8_context
    monkeypatch.setenv(BENCHMARK_GUARD, "1")
    hidden = hidden_states(device, tokens)
    transformed = native_weights(weights)

    public = decode_step(workspace, weights, hidden).clone()
    candidate = decode_fused_w13_benchmark_device_step(
        workspace,
        weights,
        transformed.w13_weight,
        transformed.w13_weight_scale,
        hidden,
    ).clone()
    check_decode_error(workspace)

    torch.testing.assert_close(candidate, public, rtol=0, atol=0)


def test_fused_w13_full_step_is_one_persistent_launch(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, device = tp8_context
    monkeypatch.setenv(BENCHMARK_GUARD, "1")
    hidden = hidden_states(device, 16)
    transformed = native_weights(weights)

    names = profiled_kernel_names(
        lambda: decode_fused_w13_benchmark_device_step(
            workspace,
            weights,
            transformed.w13_weight,
            transformed.w13_weight_scale,
            hidden,
        )
    )

    assert len(names) == 1
    assert "kimi_k3_decode_persistent_kernel" in names[0]


def test_compiled_fused_w13_binding_requires_the_benchmark_guard(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, device = tp8_context
    monkeypatch.delenv(BENCHMARK_GUARD, raising=False)
    hidden = hidden_states(device, 16)
    transformed = native_weights(weights)

    with pytest.raises(RuntimeError, match="benchmark-only"):
        _C._kimi_k3_decode_fused_w13_benchmark(
            hidden,
            weights.router_weight,
            weights.router_correction_bias,
            weights.routed_expert_down_proj,
            weights.routed_expert_up_proj,
            weights.routed_latent_rmsnorm_weight,
            transformed.w13_weight,
            transformed.w13_weight_scale,
            weights.expert_w2_packed,
            weights.expert_w2_scale,
            weights.shared_gate_proj,
            weights.shared_up_proj,
            weights.shared_down_proj,
            workspace.scratch,
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
            workspace.error_flag,
            workspace.tp_rank,
            hidden.shape[0],
            workspace.workspace_signature,
        )


def test_public_decode_source_does_not_reference_private_candidate() -> None:
    public = _source("mok/kimi_k3.py")

    assert "_kimi_k3_decode_fused_w13_benchmark" not in public
    assert "decode_fused_w13_benchmark_device_step" not in public
    assert "expert_w13_packed" not in public
    assert "expert_w13_scale" not in public
    assert os.path.basename(__file__) not in public
