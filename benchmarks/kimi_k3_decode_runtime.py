"""Correctness oracle and runtime observations for Kimi K3 decode timing."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist

from mok import _C
from mok.kimi_k3 import (
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    KIMI_K3_TP_SIZE,
    KimiK3DecodeConfig,
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
    dequant_kimi_k3_mxfp4,
    kimi_k3_decode,
    kimi_k3_rmsnorm_reference,
    kimi_k3_router_reference,
    kimi_k3_situ_reference,
)

LATENT = KIMI_K3_LATENT_SIZE
ROUTED_PER_RANK = KIMI_K3_ROUTED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE
GROUP = 32
UNIT_SCALE = 0x7F
DEQUANT_CHUNK = 16
CONFIG = KimiK3DecodeConfig()


def decode_device_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
) -> torch.Tensor:
    return kimi_k3_decode(CONFIG, workspace, weights, hidden)


def check_decode_error(workspace: KimiK3DecodeWorkspace) -> None:
    error = int(workspace.error_flag.item())
    if error:
        raise AssertionError(f"persistent kernel error flag: {error}")


def decode_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
) -> torch.Tensor:
    result = decode_device_step(workspace, weights, hidden)
    check_decode_error(workspace)
    return result


@contextlib.contextmanager
def phase_profiling() -> Iterator[None]:
    """Turn the kernel's clock64 accumulators on for the enclosed calls.

    The extension refuses the switch outside a benchmark process, so the guard
    is set here rather than expected of the caller.
    """
    previous_guard = os.environ.get("MOK_KIMI_K3_ENABLE_GRID_TUNING")
    os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = "1"
    _C._kimi_k3_decode_set_phase_profile(True)
    try:
        yield
    finally:
        _C._kimi_k3_decode_set_phase_profile(False)
        if previous_guard is None:
            os.environ.pop("MOK_KIMI_K3_ENABLE_GRID_TUNING", None)
        else:
            os.environ["MOK_KIMI_K3_ENABLE_GRID_TUNING"] = previous_guard


def phase_clock_cycles(
    workspace: KimiK3DecodeWorkspace,
) -> dict[str, int]:
    """Read the accumulators the last profiled launch left in the workspace.

    Every counter is a summed cycle count over the CTAs that ran that region,
    so the useful comparison is between regions of one launch rather than
    against wall time.
    """
    begin, names = _C._kimi_k3_decode_phase_clock_metadata()
    # `.cpu()` rebases the slice on its own storage, which is what lets the
    # uint8 bytes be reinterpreted as the 64-bit counters they hold.
    words = workspace.scratch[begin * 4 : (begin + 2 * len(names)) * 4].cpu()
    counters = words.view(torch.int64).tolist()
    return dict(zip(names, counters, strict=True))


def _e8m0_scale_bytes(absolute_max: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(absolute_max.float())
    scale_exponent = torch.where(mantissa <= 0.875, exponent - 9, exponent - 8)
    scale_bytes = (scale_exponent + 127).clamp(0, 254).to(torch.uint8)
    return torch.where(
        absolute_max == 0,
        torch.full_like(scale_bytes, UNIT_SCALE),
        scale_bytes,
    )


def _mxfp8_dequantized(values: torch.Tensor) -> torch.Tensor:
    grouped = values.float().reshape(*values.shape[:-1], -1, GROUP)
    scale_bytes = _e8m0_scale_bytes(grouped.abs().amax(dim=-1))
    scale = torch.pow(
        2.0,
        (scale_bytes.int() - 127).float(),
    ).unsqueeze(-1)
    quantized = (grouped / scale).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(values.shape)


def _situ_fp32(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )


def _routed_partial_reference(
    latent: torch.Tensor,
    weights: KimiK3DecodeWeights,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate only selected routes, batched in bounded expert chunks."""
    active, topk = expert_ids.shape
    quantized = _mxfp8_dequantized(latent[:active].float())
    partial = torch.zeros(
        active,
        LATENT,
        dtype=torch.float32,
        device=latent.device,
    )
    flat_experts = expert_ids.flatten()
    flat_tokens = (
        torch.arange(active, device=latent.device)
        .unsqueeze(1)
        .expand(active, topk)
        .flatten()
    )
    flat_slots = (
        torch.arange(topk, device=latent.device)
        .unsqueeze(0)
        .expand(active, topk)
        .flatten()
    )
    unique = torch.unique(flat_experts)
    for start in range(0, unique.numel(), DEQUANT_CHUNK):
        chunk = unique[start:start + DEQUANT_CHUNK]
        mask = torch.isin(flat_experts, chunk)
        selected_experts = flat_experts[mask]
        local_experts = torch.searchsorted(chunk, selected_experts)
        tokens = flat_tokens[mask]
        slots = flat_slots[mask]
        selected = quantized.index_select(0, tokens)

        w1 = dequant_kimi_k3_mxfp4(
            weights.expert_w1_packed[chunk],
            weights.expert_w1_scale[chunk],
            logical_k=LATENT,
        )
        gate = torch.bmm(
            w1.index_select(0, local_experts).float(),
            selected.unsqueeze(-1),
        ).squeeze(-1)
        del w1
        w3 = dequant_kimi_k3_mxfp4(
            weights.expert_w3_packed[chunk],
            weights.expert_w3_scale[chunk],
            logical_k=LATENT,
        )
        up = torch.bmm(
            w3.index_select(0, local_experts).float(),
            selected.unsqueeze(-1),
        ).squeeze(-1)
        del w3
        situ = _mxfp8_dequantized(_situ_fp32(gate, up))
        del gate, up
        w2 = dequant_kimi_k3_mxfp4(
            weights.expert_w2_packed[chunk],
            weights.expert_w2_scale[chunk],
            logical_k=ROUTED_PER_RANK,
        )
        contribution = torch.bmm(
            w2.index_select(0, local_experts).float(),
            situ.unsqueeze(-1),
        ).squeeze(-1)
        del w2, situ
        contribution *= router_weights[tokens, slots].unsqueeze(-1)
        partial.index_add_(0, tokens, contribution)
    return partial


def _shared_partial_reference(
    hidden: torch.Tensor,
    weights: KimiK3DecodeWeights,
) -> torch.Tensor:
    activations = hidden.float()
    gate = (
        activations @ weights.shared_gate_proj.float().T
    ).bfloat16()
    up = (
        activations @ weights.shared_up_proj.float().T
    ).bfloat16()
    activated = kimi_k3_situ_reference(gate, up)
    return activated.float() @ weights.shared_down_proj.float().T


def _all_reduced(partial: torch.Tensor) -> torch.Tensor:
    reduced = partial.bfloat16().contiguous()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def decode_reference(
    hidden: torch.Tensor,
    weights: KimiK3DecodeWeights,
) -> torch.Tensor:
    latent = (
        hidden.float() @ weights.routed_expert_down_proj.float().T
    ).bfloat16()
    expert_ids, router_weights = kimi_k3_router_reference(
        hidden,
        weights.router_weight,
        weights.router_correction_bias,
    )
    routed = _all_reduced(
        _routed_partial_reference(
            latent,
            weights,
            expert_ids,
            router_weights,
        )
    )
    shared = _all_reduced(_shared_partial_reference(hidden, weights))
    normalized = kimi_k3_rmsnorm_reference(
        routed,
        weights.routed_latent_rmsnorm_weight,
    )
    return (
        normalized.float() @ weights.routed_expert_up_proj.float().T
        + shared.float()
    ).bfloat16()


def assert_decode_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float, float]:
    left = actual.float()
    right = expected.float()
    difference = left - right
    relative_l1 = float(
        difference.abs().sum() / right.abs().sum().clamp_min(1e-12)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            left.flatten(),
            right.flatten(),
            dim=0,
        )
    )
    maximum = float(difference.abs().max())
    tolerance = 0.05 * float(right.abs().max()) + 0.125
    if not bool(torch.isfinite(left).all()):
        raise AssertionError("decode output contains a non-finite value")
    if relative_l1 > 0.05 or cosine < 0.999 or maximum > tolerance:
        raise AssertionError((relative_l1, cosine, maximum, tolerance))
    return relative_l1, cosine, maximum


def assert_identical_across_ranks(values: torch.Tensor) -> None:
    local = values.float().contiguous()
    smallest = local.clone()
    largest = local.clone()
    dist.all_reduce(smallest, op=dist.ReduceOp.MIN)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    if not torch.equal(smallest, largest) or not torch.equal(smallest, local):
        raise AssertionError("decode output differs across TP8 ranks")


def profiled_kernel_names(call: Callable[[], object]) -> list[str]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace_path = os.path.join(directory, "trace.json")
        profiler.export_chrome_trace(trace_path)
        with open(trace_path, encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
    return [
        event["name"]
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
    ]


def _device_trace(device: torch.device) -> list[str]:
    traces = torch.cuda.memory._snapshot().get("device_traces", [])
    index = 0 if device.index is None else device.index
    entries = traces[index] if index < len(traces) else []
    return [
        str(entry.get("action"))
        for entry in entries
        if entry.get("action") != "snapshot"
    ]


@contextlib.contextmanager
def recorded_allocator_events(
    device: torch.device,
) -> Iterator[list[str]]:
    torch.cuda.synchronize(device)
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context=None,
        stacks="python",
        max_entries=100_000,
    )
    events: list[str] = []
    try:
        before = len(_device_trace(device))
        yield events
        torch.cuda.synchronize(device)
        events.extend(_device_trace(device)[before:])
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


__all__ = [
    "CONFIG",
    "assert_decode_close",
    "assert_identical_across_ranks",
    "benchmark_decode_variant",
    "check_decode_error",
    "decode_reference",
    "decode_device_step",
    "decode_step",
    "profiled_kernel_names",
    "recorded_allocator_events",
]
