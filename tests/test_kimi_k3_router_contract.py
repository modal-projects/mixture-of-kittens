"""The host boundary of the private Kimi K3 route-and-project stage.

The stream and the device the stage runs on, the fake's agreement with the
schema, every rejection it owes a caller, each tensor's own alignment
requirement one misaligned offset view at a time, and that the whole stage is
exactly one kernel launch.

What the stage computes is in ``test_kimi_k3_router.py``.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .kimi_k3_router_support import (
    _assert_selection_is_unambiguous,
    device,
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_TOPK,
    latent_down_proj,
    peer_device,
    _route_and_project,
    _ROUTE_AND_PROJECT_ALIGNMENT,
    _ROUTE_AND_PROJECT_ARGUMENTS,
    _router_reference,
    scratch,
    SCRATCH_BYTES,
    _seeded_router_inputs,
)


def test_route_and_project_uses_the_tensor_devices_current_stream(
    device: torch.device, latent_down_proj: torch.Tensor
) -> None:
    from mok import _C

    tokens = 8
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=10007
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    reference_ids, _ = _router_reference(hidden_states, router_weight, bias)
    side_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side_stream):
        staged_hidden = torch.zeros_like(hidden_states)
        staged_scratch = torch.zeros(
            _C.kimi_k3_decode_workspace_bytes(), dtype=torch.uint8, device=device
        )
        torch.cuda._sleep(1 << 28)
        staged_hidden.copy_(hidden_states)
        expert_ids, _, latent_x = _route_and_project(
            hidden_states=staged_hidden,
            router_weight=router_weight,
            router_correction_bias=bias,
            latent_down_proj=latent_down_proj,
            scratch=staged_scratch,
            active_tokens=tokens,
        )
    side_stream.synchronize()

    actual = torch.sort(expert_ids, dim=-1).values
    expected = torch.sort(reference_ids.int(), dim=-1).values
    assert torch.equal(actual, expected)
    assert float(latent_x.float().abs().max()) > 1.0


def test_route_and_project_on_peer_device_ignores_the_current_device(
    device: torch.device, peer_device: torch.device
) -> None:
    from mok import _C

    tokens = 8
    hidden_states, router_weight, bias = _seeded_router_inputs(
        peer_device, tokens, seed=11009
    )
    generator = torch.Generator(device=peer_device).manual_seed(11010)
    peer_latent_down = (
        torch.randn(
            (KIMI_K3_LATENT_SIZE, KIMI_K3_HIDDEN_SIZE),
            generator=generator,
            device=peer_device,
            dtype=torch.float32,
        )
        * (8.0 / math.sqrt(KIMI_K3_HIDDEN_SIZE))
    ).bfloat16().contiguous()
    peer_scratch = torch.zeros(
        _C.kimi_k3_decode_workspace_bytes(), dtype=torch.uint8, device=peer_device
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    reference_ids, _ = _router_reference(hidden_states, router_weight, bias)
    expected_latent = hidden_states @ peer_latent_down.T

    torch.cuda.set_device(device)
    expert_ids, expert_weights, latent_x = _route_and_project(
        hidden_states, router_weight, bias, peer_latent_down, peer_scratch, tokens
    )
    torch.cuda.synchronize(peer_device)

    assert expert_ids.device == peer_device
    assert expert_weights.device == peer_device
    assert latent_x.device == peer_device
    assert torch.cuda.current_device() == device.index
    assert torch.equal(
        torch.sort(expert_ids, dim=-1).values,
        torch.sort(reference_ids.int(), dim=-1).values,
    )
    torch.testing.assert_close(
        latent_x.float(), expected_latent.float(), atol=0.5, rtol=0.01
    )


def test_route_and_project_fake_reports_prepared_metadata(
    device: torch.device,
) -> None:
    from mok import _fake_impls, ops

    schema_names = tuple(
        argument.name
        for argument in torch.ops.mok._kimi_k3_route_and_project.default._schema.arguments
    )
    assert schema_names == _ROUTE_AND_PROJECT_ARGUMENTS
    assert tuple(
        inspect.signature(_fake_impls._kimi_k3_route_and_project_fake).parameters
    ) == schema_names

    with FakeTensorMode():
        hidden_states = torch.empty(
            17, KIMI_K3_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        expert_ids, expert_weights, latent_x = ops._kimi_k3_route_and_project(
            hidden_states,
            torch.empty(
                KIMI_K3_NUM_EXPERTS,
                KIMI_K3_HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.empty(KIMI_K3_NUM_EXPERTS, dtype=torch.float32, device="cuda"),
            torch.empty(
                KIMI_K3_LATENT_SIZE,
                KIMI_K3_HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda"),
            17,
        )

    assert expert_ids.shape == (17, KIMI_K3_TOPK)
    assert expert_ids.dtype == torch.int32
    assert expert_weights.shape == (17, KIMI_K3_TOPK)
    assert expert_weights.dtype == torch.float32
    assert latent_x.shape == (17, KIMI_K3_LATENT_SIZE)
    assert latent_x.dtype == torch.bfloat16


def _valid_call_arguments(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> dict[str, object]:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, 8, seed=12011
    )
    return {
        "hidden_states": hidden_states,
        "router_weight": router_weight,
        "router_correction_bias": bias,
        "latent_down_proj": latent_down_proj,
        "scratch": scratch,
        "active_tokens": 8,
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("active_tokens", 0, "active_tokens"),
        ("active_tokens", 9, "active_tokens"),
        ("active_tokens", 129, "active_tokens"),
    ],
)
def test_route_and_project_rejects_invalid_active_tokens(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    field: str,
    replacement: object,
    message: str,
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments[field] = replacement

    with pytest.raises(RuntimeError, match=message):
        _route_and_project(**arguments)


def test_route_and_project_rejects_undersized_scratch(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments["scratch"] = torch.zeros(
        SCRATCH_BYTES - 1, dtype=torch.uint8, device=device
    )

    with pytest.raises(RuntimeError, match="scratch"):
        _route_and_project(**arguments)


def test_route_and_project_rejects_wrong_latent_down_shape(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments["latent_down_proj"] = latent_down_proj[:, : KIMI_K3_HIDDEN_SIZE - 64
                                                     ].contiguous()

    with pytest.raises(RuntimeError, match="routed_expert_down_proj"):
        _route_and_project(**arguments)


def test_route_and_project_rejects_float32_hidden_states(
    device: torch.device, latent_down_proj: torch.Tensor, scratch: torch.Tensor
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments["hidden_states"] = arguments["hidden_states"].float().contiguous()

    with pytest.raises(RuntimeError, match="hidden_states"):
        _route_and_project(**arguments)


def _offset_copy(source: torch.Tensor, element_offset: int) -> torch.Tensor:
    """Copy ``source`` into a contiguous view starting at a nonzero storage offset.

    The caching allocator hands out 256-byte-aligned blocks, so the returned view
    is under-aligned by exactly ``element_offset`` elements while remaining
    contiguous and correctly shaped. That is the shape of the pointer a caller can
    hand the stage without any dtype, shape, or contiguity check noticing.
    """
    flat = torch.empty(
        source.numel() + element_offset, dtype=source.dtype, device=source.device
    )
    assert flat.data_ptr() % 256 == 0
    view = flat[element_offset:].view(source.shape)
    view.copy_(source)
    assert view.is_contiguous()
    assert view.storage_offset() == element_offset
    return view


@pytest.mark.parametrize(
    ("field", "argument", "element_offset", "alignment"),
    _ROUTE_AND_PROJECT_ALIGNMENT,
)
def test_route_and_project_rejects_misaligned_pointers(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    field: str,
    argument: str,
    element_offset: int,
    alignment: int,
) -> None:
    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    misaligned = _offset_copy(arguments[argument], element_offset)
    assert misaligned.data_ptr() % alignment != 0
    arguments[argument] = misaligned

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        _route_and_project(**arguments)


@pytest.mark.parametrize(
    ("field", "argument", "element_offset", "alignment"),
    _ROUTE_AND_PROJECT_ALIGNMENT,
)
def test_c_entrypoint_rejects_misaligned_pointers(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    field: str,
    argument: str,
    element_offset: int,
    alignment: int,
) -> None:
    """The extension must guard itself: callers can bypass ``mok.ops`` entirely."""
    from mok import _C

    arguments = _valid_call_arguments(device, latent_down_proj, scratch)
    arguments[argument] = _offset_copy(arguments[argument], element_offset)

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        _C._kimi_k3_route_and_project(
            arguments["hidden_states"],
            arguments["router_weight"],
            arguments["router_correction_bias"],
            arguments["latent_down_proj"],
            arguments["scratch"],
            arguments["active_tokens"],
        )


def test_route_and_project_accepts_sufficiently_aligned_offset_views(
    device: torch.device, latent_down_proj: torch.Tensor
) -> None:
    """Nonzero storage offsets are fine as long as they clear the real boundary.

    The correction bias is only read as a scalar float, so a 4-byte-aligned view
    of it must keep working; everything else is offset to the next 16- or 256-byte
    boundary rather than rejected.
    """
    tokens = 8
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=13001
    )
    _assert_selection_is_unambiguous(hidden_states, router_weight, bias)
    reference_ids, _ = _router_reference(hidden_states, router_weight, bias)
    expected_latent = hidden_states @ latent_down_proj.T
    offset_scratch = _offset_copy(
        torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device), 256
    )
    offset_bias = _offset_copy(bias, 1)
    assert offset_bias.data_ptr() % 16 != 0

    expert_ids, _, latent_x = _route_and_project(
        _offset_copy(hidden_states, 8),
        _offset_copy(router_weight, 8),
        offset_bias,
        _offset_copy(latent_down_proj, 8),
        offset_scratch,
        tokens,
    )

    assert torch.equal(
        torch.sort(expert_ids, dim=-1).values,
        torch.sort(reference_ids.int(), dim=-1).values,
    )
    torch.testing.assert_close(
        latent_x.float(), expected_latent.float(), atol=0.5, rtol=0.01
    )


def _profiled_kernel_names(call: Callable[[], object]) -> list[str]:
    """Names of every CUDA kernel the profiler attributes to ``call()``."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        # ``export_chrome_trace`` renames a temporary file into place, so the
        # trace has to be reopened by path once the export has returned.
        trace_path = os.path.join(directory, "trace.json")
        profiler.export_chrome_trace(trace_path)
        with open(trace_path, encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
    return [
        event["name"]
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
    ]


@pytest.mark.parametrize(
    ("tokens", "active_tokens", "expected_kernel"),
    [
        (8, 8, "route_and_project_core_kernel"),
        (64, 5, "route_and_project_core_kernel"),
        (32, 32, "route_and_project_tensor_kernel"),
        (128, 20, "route_and_project_tensor_kernel"),
    ],
)
def test_route_and_project_is_exactly_one_kernel_launch(
    device: torch.device,
    scratch: torch.Tensor,
    latent_down_proj: torch.Tensor,
    tokens: int,
    active_tokens: int,
    expected_kernel: str,
) -> None:
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, tokens, seed=14000 + tokens
    )

    def call() -> object:
        return _route_and_project(
            hidden_states,
            router_weight,
            bias,
            latent_down_proj,
            scratch,
            active_tokens,
        )

    call()
    names = _profiled_kernel_names(call)

    assert len(names) == 1, names
    assert expected_kernel in names[0]


def test_launch_counter_sees_a_second_kernel_launch(
    device: torch.device, scratch: torch.Tensor, latent_down_proj: torch.Tensor
) -> None:
    """Keep the one-launch assertions honest by proving the counter can say two."""
    hidden_states, router_weight, bias = _seeded_router_inputs(
        device, 8, seed=14999
    )

    def call_twice() -> None:
        for _ in range(2):
            _route_and_project(
                hidden_states, router_weight, bias, latent_down_proj, scratch, 8
            )

    names = _profiled_kernel_names(call_twice)

    assert len(names) == 2, names
