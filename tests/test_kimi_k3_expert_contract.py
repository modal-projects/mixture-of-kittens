"""The host boundary of the private Kimi K3 routed expert stage.

The stream and the device the stage runs on, the fake's agreement with the
schema, and every rejection: an undersized scratch, a wrong weight layout, and
each tensor's own alignment requirement, checked one misaligned offset view at
a time.

What the stage computes is ``test_kimi_k3_expert.py``; which expert it reaches
is ``test_kimi_k3_expert_addressing.py``.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .kimi_k3_expert_support import (
    ALIGNMENT,
    _assert_expert_close,
    _call,
    device,
    _EXPERT_ARGUMENTS,
    EXPERTS,
    ExpertWeights,
    GROUP,
    HIDDEN,
    INTERMEDIATE,
    _random_latent,
    _reference,
    scratch,
    SCRATCH_BYTES,
    weights,
    _write_assignments,
)


def test_expert_fake_matches_schema_and_returns_active_alias_metadata() -> None:
    from mok import _fake_impls, ops

    schema_names = tuple(
        argument.name
        for argument in torch.ops.mok._kimi_k3_routed_experts.default._schema.arguments
    )
    assert schema_names == _EXPERT_ARGUMENTS
    assert tuple(
        inspect.signature(_fake_impls._kimi_k3_routed_experts_fake).parameters
    ) == schema_names

    with FakeTensorMode():
        latent = torch.empty(17, HIDDEN, dtype=torch.bfloat16, device="cuda")
        packed_w1 = torch.empty(
            EXPERTS, INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8, device="cuda"
        )
        scale_w1 = torch.empty(
            EXPERTS, INTERMEDIATE, HIDDEN // GROUP, dtype=torch.uint8, device="cuda"
        )
        packed_w2 = torch.empty(
            EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8, device="cuda"
        )
        scale_w2 = torch.empty(
            EXPERTS, HIDDEN, INTERMEDIATE // GROUP, dtype=torch.uint8, device="cuda"
        )
        routed = torch.empty_like(latent)
        actual = ops._kimi_k3_routed_experts(
            latent,
            packed_w1,
            scale_w1,
            packed_w1,
            scale_w1,
            packed_w2,
            scale_w2,
            routed,
            torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda"),
            11,
        )

    assert actual.shape == (11, HIDDEN)
    assert actual.dtype == torch.bfloat16


def test_expert_stage_rejects_undersized_scratch_and_wrong_weight_layout(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    routed = torch.empty_like(latent)
    _write_assignments(scratch, [(0, 0, 0, 1.0)])

    with pytest.raises(RuntimeError, match="scratch"):
        _call(
            latent,
            weights,
            routed,
            scratch[:SCRATCH_BYTES - ALIGNMENT],
            1,
        )

    invalid = ExpertWeights(
        weights.w1_packed[:, :, :-16],
        weights.w1_scale,
        weights.w3_packed,
        weights.w3_scale,
        weights.w2_packed,
        weights.w2_scale,
    )
    with pytest.raises(RuntimeError, match="expert_w1_packed"):
        _call(latent, invalid, routed, scratch, 1)


def _offset_copy(source: torch.Tensor, element_offset: int) -> torch.Tensor:
    """Copy ``source`` into a contiguous view starting at a nonzero storage offset.

    The caching allocator hands out 256-byte-aligned blocks, so the returned view
    starts exactly ``element_offset`` elements into its storage while staying
    contiguous and correctly shaped.  With an offset that breaks the required
    alignment this is the one pointer no dtype, shape, or contiguity check would
    notice; with an offset that preserves it, the same construction proves the
    alignment check accepts offset views instead of rejecting all of them.
    """
    flat = torch.empty(
        source.numel() + element_offset, dtype=source.dtype, device=source.device
    )
    assert flat.data_ptr() % ALIGNMENT == 0
    view = flat[element_offset:].view(source.shape)
    view.copy_(source)
    assert view.is_contiguous()
    assert view.storage_offset() == element_offset
    return view


def _expert_call_arguments(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    latent: torch.Tensor | None = None,
) -> dict[str, object]:
    if latent is None:
        latent = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
        latent[:, 0] = 0.5
    rows = latent.size(0)
    _write_assignments(scratch, [(0, token, 0, 1.0) for token in range(rows)])
    return {
        "latent_x": latent,
        "expert_w1_packed": weights.w1_packed,
        "expert_w1_scale": weights.w1_scale,
        "expert_w3_packed": weights.w3_packed,
        "expert_w3_scale": weights.w3_scale,
        "expert_w2_packed": weights.w2_packed,
        "expert_w2_scale": weights.w2_scale,
        "routed_output": torch.empty_like(latent),
        "scratch": scratch,
        "active_tokens": rows,
    }


# Every expert-stage tensor with its required alignment, one element offset that
# breaks that alignment, and one nonzero element offset that preserves it.
_EXPERT_TENSOR_CASES = (
    ("latent_x", 16, 1, 8),
    ("expert_w1_packed", 16, 1, 16),
    ("expert_w1_scale", 16, 1, 16),
    ("expert_w3_packed", 16, 1, 16),
    ("expert_w3_scale", 16, 1, 16),
    ("expert_w2_packed", 16, 1, 16),
    ("expert_w2_scale", 16, 1, 16),
    ("routed_output", 16, 1, 8),
    ("scratch", ALIGNMENT, 16, ALIGNMENT),
)

_EXPERT_MISALIGNED_CASES = tuple(
    (field, element_offset, alignment)
    for field, alignment, element_offset, _ in _EXPERT_TENSOR_CASES
)
_EXPERT_ALIGNED_CASES = tuple(
    (field, element_offset, alignment)
    for field, alignment, _, element_offset in _EXPERT_TENSOR_CASES
)


@pytest.mark.parametrize(("field", "element_offset", "alignment"),
                         _EXPERT_MISALIGNED_CASES)
def test_expert_stage_rejects_every_misaligned_offset_view(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    field: str,
    element_offset: int,
    alignment: int,
) -> None:
    from mok import ops

    arguments = _expert_call_arguments(device, weights, scratch)
    misaligned = _offset_copy(arguments[field], element_offset)
    assert misaligned.data_ptr() % alignment != 0
    arguments[field] = misaligned

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        ops._kimi_k3_routed_experts(**arguments)


@pytest.mark.parametrize(("field", "element_offset", "alignment"),
                         _EXPERT_MISALIGNED_CASES)
def test_c_entrypoint_rejects_every_misaligned_offset_view(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    field: str,
    element_offset: int,
    alignment: int,
) -> None:
    """The extension must guard itself: callers can bypass ``mok.ops`` entirely."""
    from mok import _C

    arguments = _expert_call_arguments(device, weights, scratch)
    arguments[field] = _offset_copy(arguments[field], element_offset)

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        _C._kimi_k3_routed_experts(
            *(arguments[name] for name in _EXPERT_ARGUMENTS)
        )


@pytest.mark.parametrize(("field", "element_offset", "alignment"),
                         _EXPERT_ALIGNED_CASES)
def test_expert_stage_accepts_every_aligned_offset_view(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    field: str,
    element_offset: int,
    alignment: int,
) -> None:
    """Validation must reject under-alignment, not every nonzero storage offset.

    Without this control the alignment checks could reject every offset view and
    the rejection tests would still pass.
    """
    from mok import ops

    rows = 4
    latent = _random_latent(device, rows, 7200)
    arguments = _expert_call_arguments(device, weights, scratch, latent)
    aligned = _offset_copy(arguments[field], element_offset)
    assert aligned.storage_offset() != 0
    assert aligned.data_ptr() % alignment == 0
    arguments[field] = aligned

    actual = ops._kimi_k3_routed_experts(**arguments)

    assert actual.data_ptr() == arguments["routed_output"].data_ptr()
    assignments = [(0, token, 0, 1.0) for token in range(rows)]
    _assert_expert_close(actual, _reference(latent, weights, assignments, rows))


def _profiled_kernel_names(call: Callable[[], object]) -> list[str]:
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


def test_private_expert_stage_is_exactly_one_kernel_launch(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:, 0] = 1.0
    _write_assignments(scratch, [(0, 0, 0, 1.0)])
    routed = torch.empty_like(latent)

    def call() -> object:
        return _call(latent, weights, routed, scratch, 1)

    call()
    names = _profiled_kernel_names(call)

    assert len(names) == 1, names
    assert "kimi_k3_routed_experts_kernel" in names[0]
