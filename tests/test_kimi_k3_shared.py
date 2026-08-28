"""GPU tests for the TP-sharded Kimi K3 shared-expert branch."""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mok import _C, _fake_impls, ops


HIDDEN = 7168
INTERMEDIATE = 768
LATENT = 3584
COLLECTIVE_COLUMNS = LATENT + HIDDEN
MAX_TOKENS = 128
ALIGNMENT = 256
GROUP = 32
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
CAPACITY_AND_DFLASH_ACTIVE_ROWS = (1, 2, 3, 4, 5, 8, 16, 20, 32, 64, 127, 128)

_SHARED_ARGUMENTS = (
    "hidden_states",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
    "scratch",
    "collective_buffer",
    "active_tokens",
)


def _aligned(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _scratch_layout() -> dict[str, tuple[int, int]]:
    """Independent byte model of the C++ source-of-truth workspace."""
    regions = (
        ("phase", 20 * 4),
        ("expert_ids", MAX_TOKENS * 16 * 4),
        ("expert_weights", MAX_TOKENS * 16 * 4),
        ("expert_counts", 896 * 4),
        ("expert_offsets", 897 * 4),
        ("assignment_tokens", MAX_TOKENS * 16 * 4),
        ("assignment_slots", MAX_TOKENS * 16 * 4),
        ("latent_mxfp8", MAX_TOKENS * LATENT),
        ("latent_scale", MAX_TOKENS * (LATENT // GROUP)),
        ("situ_mxfp8", MAX_TOKENS * 16 * 384),
        ("situ_scale", MAX_TOKENS * 16 * (384 // GROUP)),
        ("routed_accumulator", MAX_TOKENS * LATENT * 4),
        ("shared_gate", MAX_TOKENS * INTERMEDIATE * 2),
        ("shared_up", MAX_TOKENS * INTERMEDIATE * 2),
        ("shared_activated", MAX_TOKENS * INTERMEDIATE * 2),
    )
    layout: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, size in regions:
        layout[name] = (cursor, size)
        cursor += _aligned(size)
    layout["total_bytes"] = (cursor, 0)
    return layout


SCRATCH_LAYOUT = _scratch_layout()
SCRATCH_BYTES = SCRATCH_LAYOUT["total_bytes"][0]


@dataclass(frozen=True)
class SharedWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Kimi K3 shared experts require CUDA")
    selected = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(selected)
    if torch.cuda.get_device_capability(selected) != (10, 3):
        pytest.skip("Kimi K3 shared experts require an SM103 GPU")
    return selected


@pytest.fixture(scope="module")
def peer_device(device: torch.device) -> Iterator[torch.device]:
    if torch.cuda.device_count() < 2:
        pytest.skip("cross-device Kimi K3 shared experts need two visible GPUs")
    peer = torch.device("cuda", 1 if device.index == 0 else 0)
    if torch.cuda.get_device_capability(peer) != (10, 3):
        pytest.skip("Kimi K3 shared experts require an SM103 GPU")
    try:
        yield peer
    finally:
        torch.cuda.set_device(device)


def _draw(
    shape: tuple[int, ...],
    selected_device: torch.device,
    seed: int,
    scale: float,
) -> torch.Tensor:
    generator = torch.Generator(device=selected_device).manual_seed(seed)
    return (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
            device=selected_device,
        )
        * scale
    ).bfloat16().contiguous()


def _make_weights(selected_device: torch.device) -> SharedWeights:
    return SharedWeights(
        gate=_draw(
            (INTERMEDIATE, HIDDEN),
            selected_device,
            7301,
            1.75 / math.sqrt(HIDDEN),
        ),
        up=_draw(
            (INTERMEDIATE, HIDDEN),
            selected_device,
            7302,
            1.25 / math.sqrt(HIDDEN),
        ),
        down=_draw(
            (HIDDEN, INTERMEDIATE),
            selected_device,
            7303,
            1.0 / math.sqrt(INTERMEDIATE),
        ),
    )


@pytest.fixture(scope="module")
def weights(device: torch.device) -> SharedWeights:
    return _make_weights(device)


@pytest.fixture
def scratch(device: torch.device) -> torch.Tensor:
    return torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)


def _hidden(selected_device: torch.device, rows: int, seed: int) -> torch.Tensor:
    return _draw((rows, HIDDEN), selected_device, seed, 0.5)


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The exact FP32 SiTU formula from the task brief."""
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )


def _reference(
    hidden_states: torch.Tensor,
    weights: SharedWeights,
    active_tokens: int,
) -> torch.Tensor:
    active = hidden_states[:active_tokens].float()
    gate = (active @ weights.gate.float().T).bfloat16()
    up = (active @ weights.up.float().T).bfloat16()
    activated = _situ(gate.float(), up.float()).bfloat16()
    return activated.float() @ weights.down.float().T


def _collective(
    hidden_states: torch.Tensor, fill: float = float("nan")
) -> torch.Tensor:
    return torch.full(
        (hidden_states.size(0), COLLECTIVE_COLUMNS),
        fill,
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )


def _call(
    hidden_states: torch.Tensor,
    weights: SharedWeights,
    scratch: torch.Tensor,
    collective_buffer: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    return ops._kimi_k3_shared_experts(
        hidden_states,
        weights.gate,
        weights.up,
        weights.down,
        scratch,
        collective_buffer,
        active_tokens,
    )


def _accuracy_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    relative_l1 = difference.abs().sum() / expected_float.abs().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    )
    maximum = difference.abs().max()
    return float(relative_l1), float(cosine), float(maximum)


def _assert_shared_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    metrics = _accuracy_metrics(actual, expected)
    assert torch.isfinite(actual.float()).all()
    assert metrics[0] <= 0.03
    assert metrics[1] >= 0.999
    assert metrics[2] <= 0.25


def _region(
    scratch: torch.Tensor, name: str, dtype: torch.dtype
) -> torch.Tensor:
    offset, size = SCRATCH_LAYOUT[name]
    return scratch[offset:offset + size].view(dtype)


def _active_intermediates(
    scratch: torch.Tensor, active_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        _region(scratch, name, torch.bfloat16)
        .view(MAX_TOKENS, INTERMEDIATE)[:active_tokens]
        for name in ("shared_gate", "shared_up", "shared_activated")
    )


def _assert_active_intermediates(
    hidden_states: torch.Tensor,
    weights: SharedWeights,
    scratch: torch.Tensor,
    active_tokens: int,
) -> None:
    active = hidden_states[:active_tokens].float()
    expected_gate = active @ weights.gate.float().T
    expected_up = active @ weights.up.float().T
    actual_gate, actual_up, actual_activated = _active_intermediates(
        scratch, active_tokens
    )
    expected_activated = _situ(
        actual_gate.float(), actual_up.float()
    ).bfloat16()

    torch.testing.assert_close(
        actual_gate.float(), expected_gate, rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        actual_up.float(), expected_up, rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        actual_activated.float(),
        expected_activated.float(),
        rtol=0.02,
        atol=0.02,
    )
    assert torch.isfinite(actual_gate.float()).all()
    assert torch.isfinite(actual_up.float()).all()
    assert torch.isfinite(actual_activated.float()).all()


def test_workspace_bytes_matches_shared_scratch_source_of_truth(
    device: torch.device,
) -> None:
    assert SCRATCH_BYTES == 3_749_376
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    assert SCRATCH_LAYOUT["shared_gate"] == (3_159_552, 196_608)
    assert SCRATCH_LAYOUT["shared_up"] == (3_356_160, 196_608)
    assert SCRATCH_LAYOUT["shared_activated"] == (3_552_768, 196_608)
    for name, (offset, _) in SCRATCH_LAYOUT.items():
        if name != "total_bytes":
            assert offset % ALIGNMENT == 0, name


@pytest.mark.parametrize(
    ("active_tokens", "expected"),
    [
        (1, (24, 0, 0, 112, 136)),
        (8, (24, 0, 0, 112, 136)),
        (16, (6, 6, 6, 56, 74)),
        (128, (6, 6, 6, 56, 74)),
    ],
)
def test_role_plan_orders_all_producers_before_consumers(
    device: torch.device,
    active_tokens: int,
    expected: tuple[int, int, int, int, int],
) -> None:
    plan = _C._kimi_k3_shared_experts_role_plan(active_tokens)

    assert plan == expected
    gate_roles, up_roles, activation_roles, down_roles, total_roles = plan
    assert gate_roles > 0
    assert down_roles > 0
    assert gate_roles + up_roles + activation_roles + down_roles == total_roles
    if active_tokens <= 8:
        assert up_roles == 0
        assert activation_roles == 0
    else:
        assert up_roles > 0
        assert activation_roles > 0


@pytest.mark.parametrize(
    ("active_tokens", "required_sms"), [(8, 136), (20, 74)]
)
def test_host_residency_guard_uses_the_selected_role_grid(
    device: torch.device,
    active_tokens: int,
    required_sms: int,
) -> None:
    _C._kimi_k3_shared_experts_validate_residency(
        active_tokens, required_sms
    )
    with pytest.raises(
        RuntimeError,
        match=rf"requires all {required_sms} role CTAs.*{required_sms - 1} SMs",
    ):
        _C._kimi_k3_shared_experts_validate_residency(
            active_tokens, required_sms - 1
        )


def test_generation_order_and_timeout_helpers_are_wrap_safe(
    device: torch.device,
) -> None:
    advanced = _C._kimi_k3_shared_experts_generation_advanced
    assert not advanced(7, 7)
    assert advanced(8, 7)
    assert advanced(0, UINT32_MAX)
    assert not advanced(UINT32_MAX, 0)

    timeout = _C._kimi_k3_shared_experts_wait_timeout_clocks()
    timed_out = _C._kimi_k3_shared_experts_wait_timed_out
    assert timeout > 0
    assert not timed_out(100, 100 + timeout - 1)
    assert timed_out(100, 100 + timeout)
    start = UINT64_MAX - timeout // 2
    assert timed_out(start, (start + timeout) & UINT64_MAX)
    assert _C._kimi_k3_shared_experts_timeout_metadata() == (
        17,
        10,
        12,
        14,
    )


def test_gate_up_and_down_patterns_are_nonperiodic_and_distinct(
    weights: SharedWeights,
) -> None:
    """Guard that the fixture catches swaps, tile shifts, and row aliasing."""
    gate = weights.gate.float().flatten()
    up = weights.up.float().flatten()
    down = weights.down.float().flatten()
    assert not torch.equal(gate, up)
    for values in (gate, up, down):
        for shift in (1, 8, 32, 64, 128, 768, 7168):
            assert float((values - values.roll(shift)).abs().mean()) > 1e-4


@pytest.mark.parametrize("active_tokens", CAPACITY_AND_DFLASH_ACTIVE_ROWS)
def test_every_capacity_and_dflash_active_row_count_matches_fp32_reference(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
    active_tokens: int,
) -> None:
    hidden_states = _hidden(device, active_tokens, 7400 + active_tokens)
    collective_buffer = _collective(hidden_states)

    actual = _call(
        hidden_states, weights, scratch, collective_buffer, active_tokens
    )
    expected = _reference(hidden_states, weights, active_tokens)

    assert actual.shape == (active_tokens, HIDDEN)
    assert actual.dtype == torch.bfloat16
    assert actual.data_ptr() == (
        collective_buffer.data_ptr() + LATENT * collective_buffer.element_size()
    )
    _assert_active_intermediates(
        hidden_states, weights, scratch, active_tokens
    )
    _assert_shared_close(actual, expected)


def test_gate_and_up_are_not_interchangeable(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> None:
    hidden_states = _hidden(device, 5, 7501)
    collective_buffer = _collective(hidden_states)
    actual = _call(hidden_states, weights, scratch, collective_buffer, 5)
    expected = _reference(hidden_states, weights, 5)
    swapped = _reference(
        hidden_states, SharedWeights(weights.up, weights.gate, weights.down), 5
    )

    assert float((expected - swapped).abs().max()) > 0.25
    _assert_shared_close(actual, expected)


def test_inactive_rows_are_zeroed_without_touching_latent_prefix(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> None:
    hidden_states = _hidden(device, 8, 7502)
    collective_buffer = _collective(hidden_states, fill=3.0)

    actual = _call(hidden_states, weights, scratch, collective_buffer, 3)

    assert actual.shape == (3, HIDDEN)
    _assert_shared_close(actual, _reference(hidden_states, weights, 3))
    assert torch.equal(
        collective_buffer[:, :LATENT],
        torch.full_like(collective_buffer[:, :LATENT], 3.0),
    )
    assert torch.equal(
        collective_buffer[3:, LATENT:],
        torch.zeros_like(collective_buffer[3:, LATENT:]),
    )


def test_reused_scratch_advances_generations_and_replaces_intermediates(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> None:
    gate_generations: list[int] = []
    up_generations: list[int] = []
    activation_generations: list[int] = []
    down_generations: list[int] = []
    for step, active_tokens in enumerate((8, 3, 16, 5, 32, 2)):
        hidden_states = _hidden(device, 32, 7600 + step)
        collective_buffer = _collective(hidden_states, fill=123.0)
        for name in ("shared_gate", "shared_up", "shared_activated"):
            _region(scratch, name, torch.bfloat16).fill_(123.0)

        actual = _call(
            hidden_states,
            weights,
            scratch,
            collective_buffer,
            active_tokens,
        )

        _assert_shared_close(
            actual, _reference(hidden_states, weights, active_tokens)
        )
        assert torch.equal(
            collective_buffer[active_tokens:, LATENT:],
            torch.zeros_like(collective_buffer[active_tokens:, LATENT:]),
        )
        phase = _region(scratch, "phase", torch.int32)
        assert int(phase[9]) == 0
        assert int(phase[11]) == 0
        assert int(phase[13]) == 0
        assert int(phase[15]) == 0
        gate_generations.append(int(phase[10]))
        up_generations.append(int(phase[12]))
        activation_generations.append(int(phase[14]))
        down_generations.append(int(phase[16]))

    expected_generations = [
        gate_generations[0] + step for step in range(len(gate_generations))
    ]
    assert gate_generations == expected_generations
    assert up_generations == expected_generations
    assert activation_generations == expected_generations
    assert down_generations == expected_generations


def test_tensor_generations_advance_across_uint32_wrap(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> None:
    phase = _region(scratch, "phase", torch.int32)
    for generation_index in (10, 12, 14, 16):
        phase[generation_index] = -1
    hidden_states = _hidden(device, 20, 7650)

    actual = _call(
        hidden_states,
        weights,
        scratch,
        _collective(hidden_states),
        20,
    )

    _assert_shared_close(actual, _reference(hidden_states, weights, 20))
    assert [int(phase[index]) for index in (10, 12, 14, 16)] == [0, 0, 0, 0]


def test_tensor_multi_cta_consumers_observe_fresh_published_generations(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> None:
    """Stress tensor gate/up, activation, and down publication edges.

    Every replay poisons all three global intermediates, changes their values,
    and launches 56 down consumers. This can catch stale publication data when
    reordering manifests, but a passing stress test does not prove fence
    necessity.
    """
    cases: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(12):
        hidden_states = _hidden(device, 16, 7700 + step)
        cases.append((hidden_states, _reference(hidden_states, weights, 16)))

    for step, (hidden_states, expected) in enumerate(cases):
        poison = 64.0 if step % 2 == 0 else -64.0
        for name in ("shared_gate", "shared_up", "shared_activated"):
            _region(scratch, name, torch.bfloat16).fill_(poison)
        collective_buffer = _collective(hidden_states, fill=poison)

        actual = _call(
            hidden_states, weights, scratch, collective_buffer, 16
        )

        _assert_shared_close(actual, expected)
        _assert_active_intermediates(
            hidden_states, weights, scratch, 16
        )
        phase = _region(scratch, "phase", torch.int32)
        assert int(phase[10]) == int(phase[12])
        assert int(phase[12]) == int(phase[14])
        assert int(phase[14]) == int(phase[16])
        assert int(phase[17]) == 0


def test_shared_stage_uses_the_tensor_devices_current_stream(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> None:
    source = _hidden(device, 5, 7801)
    side_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side_stream):
        hidden_states = torch.zeros_like(source)
        collective_buffer = _collective(source)
        torch.cuda._sleep(1 << 28)
        hidden_states.copy_(source)
        actual = _call(
            hidden_states, weights, scratch, collective_buffer, 5
        )
    side_stream.synchronize()

    _assert_shared_close(actual, _reference(source, weights, 5))


def test_shared_stage_on_peer_device_ignores_current_device(
    device: torch.device,
    peer_device: torch.device,
) -> None:
    peer_weights = _make_weights(peer_device)
    peer_scratch = torch.zeros(
        SCRATCH_BYTES, dtype=torch.uint8, device=peer_device
    )
    hidden_states = _hidden(peer_device, 3, 7802)
    collective_buffer = _collective(hidden_states)
    torch.cuda.set_device(device)

    actual = _call(
        hidden_states, peer_weights, peer_scratch, collective_buffer, 3
    )
    torch.cuda.synchronize(peer_device)

    assert torch.cuda.current_device() == device.index
    assert actual.device == peer_device
    _assert_shared_close(actual, _reference(hidden_states, peer_weights, 3))


def test_shared_fake_matches_schema_and_returns_active_alias_metadata() -> None:
    schema = torch.ops.mok._kimi_k3_shared_experts.default._schema
    schema_names = tuple(argument.name for argument in schema.arguments)
    assert schema_names == _SHARED_ARGUMENTS
    assert tuple(
        inspect.signature(_fake_impls._kimi_k3_shared_experts_fake).parameters
    ) == schema_names
    assert schema.arguments[4].alias_info is not None
    assert schema.arguments[4].alias_info.is_write
    assert schema.arguments[5].alias_info is not None
    assert schema.arguments[5].alias_info.is_write
    assert schema.returns[0].alias_info is not None

    with FakeTensorMode():
        hidden_states = torch.empty(
            17, HIDDEN, dtype=torch.bfloat16, device="cuda"
        )
        collective_buffer = torch.empty(
            17, COLLECTIVE_COLUMNS, dtype=torch.bfloat16, device="cuda"
        )
        actual = ops._kimi_k3_shared_experts(
            hidden_states,
            torch.empty(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
            ),
            torch.empty(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
            ),
            torch.empty(
                HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
            ),
            torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda"),
            collective_buffer,
            11,
        )

    assert actual.shape == (11, HIDDEN)
    assert actual.dtype == torch.bfloat16


def _offset_copy(source: torch.Tensor, element_offset: int) -> torch.Tensor:
    flat = torch.empty(
        source.numel() + element_offset,
        dtype=source.dtype,
        device=source.device,
    )
    assert flat.data_ptr() % ALIGNMENT == 0
    view = flat[element_offset:].view(source.shape)
    view.copy_(source)
    assert view.is_contiguous()
    assert view.storage_offset() == element_offset
    return view


def _valid_arguments(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> dict[str, object]:
    hidden_states = _hidden(device, 3, 7901)
    return {
        "hidden_states": hidden_states,
        "shared_gate_proj": weights.gate,
        "shared_up_proj": weights.up,
        "shared_down_proj": weights.down,
        "scratch": scratch,
        "collective_buffer": _collective(hidden_states),
        "active_tokens": 3,
    }


# Every tensor argument, the byte boundary it needs, an element offset that
# breaks it, and a nonzero element offset that preserves it.
_SHARED_TENSOR_CASES = (
    ("hidden_states", 16, 1, 8),
    ("shared_gate_proj", 16, 1, 8),
    ("shared_up_proj", 16, 1, 8),
    ("shared_down_proj", 16, 1, 8),
    ("scratch", ALIGNMENT, 16, ALIGNMENT),
    ("collective_buffer", 16, 1, 8),
)


@pytest.mark.parametrize(
    ("field", "alignment", "element_offset", "_"),
    _SHARED_TENSOR_CASES,
)
def test_shared_stage_rejects_every_misaligned_offset_view(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
    field: str,
    alignment: int,
    element_offset: int,
    _: int,
) -> None:
    arguments = _valid_arguments(device, weights, scratch)
    misaligned = _offset_copy(arguments[field], element_offset)
    assert misaligned.data_ptr() % alignment != 0
    arguments[field] = misaligned

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        ops._kimi_k3_shared_experts(**arguments)


@pytest.mark.parametrize(
    ("field", "alignment", "element_offset", "_"),
    _SHARED_TENSOR_CASES,
)
def test_shared_c_entrypoint_rejects_every_misaligned_offset_view(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
    field: str,
    alignment: int,
    element_offset: int,
    _: int,
) -> None:
    arguments = _valid_arguments(device, weights, scratch)
    arguments[field] = _offset_copy(arguments[field], element_offset)

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        _C._kimi_k3_shared_experts(
            *(arguments[name] for name in _SHARED_ARGUMENTS)
        )


@pytest.mark.parametrize(
    ("field", "alignment", "_", "element_offset"),
    _SHARED_TENSOR_CASES,
)
def test_shared_stage_accepts_every_aligned_offset_view(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
    field: str,
    alignment: int,
    _: int,
    element_offset: int,
) -> None:
    arguments = _valid_arguments(device, weights, scratch)
    aligned = _offset_copy(arguments[field], element_offset)
    assert aligned.data_ptr() % alignment == 0
    arguments[field] = aligned

    actual = ops._kimi_k3_shared_experts(**arguments)

    assert actual.data_ptr() == (
        arguments["collective_buffer"].data_ptr()
        + LATENT * actual.element_size()
    )
    aligned_weights = SharedWeights(
        arguments["shared_gate_proj"],
        arguments["shared_up_proj"],
        arguments["shared_down_proj"],
    )
    _assert_shared_close(
        actual,
        _reference(
            arguments["hidden_states"], aligned_weights, arguments["active_tokens"]
        ),
    )


def test_shared_stage_rejects_invalid_shapes_and_active_count(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
) -> None:
    arguments = _valid_arguments(device, weights, scratch)
    with pytest.raises(
        RuntimeError, match=r"shared_gate_proj \[768, 7168\]"
    ):
        ops._kimi_k3_shared_experts(
            **{
                **arguments,
                "shared_gate_proj": weights.gate[:, :-1].contiguous(),
            }
        )
    with pytest.raises(
        RuntimeError, match=r"collective_buffer \[M, 10752\]"
    ):
        ops._kimi_k3_shared_experts(
            **{
                **arguments,
                "collective_buffer": arguments["collective_buffer"][
                    :, :-1
                ].contiguous(),
            }
        )
    with pytest.raises(
        RuntimeError, match=rf"scratch.*at least {SCRATCH_BYTES} bytes"
    ):
        ops._kimi_k3_shared_experts(
            **{**arguments, "scratch": scratch[:-ALIGNMENT]}
        )
    with pytest.raises(
        RuntimeError, match=r"active_tokens in \[1, 3\]"
    ):
        ops._kimi_k3_shared_experts(**{**arguments, "active_tokens": 0})


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


@pytest.mark.parametrize(
    ("active_tokens", "expected_kernel"),
    [
        (5, "shared_experts_core_kernel"),
        (20, "shared_experts_tensor_kernel"),
    ],
)
def test_private_shared_stage_is_exactly_one_kernel_launch(
    device: torch.device,
    weights: SharedWeights,
    scratch: torch.Tensor,
    active_tokens: int,
    expected_kernel: str,
) -> None:
    hidden_states = _hidden(device, active_tokens, 8000 + active_tokens)
    collective_buffer = _collective(hidden_states)

    def call() -> object:
        return _call(
            hidden_states,
            weights,
            scratch,
            collective_buffer,
            active_tokens,
        )

    call()
    names = _profiled_kernel_names(call)

    assert len(names) == 1, names
    assert expected_kernel in names[0]


def test_accuracy_metrics_have_finite_worst_case(
    device: torch.device,
    weights: SharedWeights,
) -> None:
    """Compute aggregate metrics independently of test order or filtering."""
    metrics: list[tuple[float, float, float]] = []
    scratch = torch.zeros(
        SCRATCH_BYTES, dtype=torch.uint8, device=device
    )
    for active_tokens in CAPACITY_AND_DFLASH_ACTIVE_ROWS:
        hidden_states = _hidden(
            device, active_tokens, 8100 + active_tokens
        )
        actual = _call(
            hidden_states,
            weights,
            scratch,
            _collective(hidden_states),
            active_tokens,
        )
        expected = _reference(hidden_states, weights, active_tokens)
        _assert_shared_close(actual, expected)
        metrics.append(_accuracy_metrics(actual, expected))

    worst_rel_l1 = max(metric[0] for metric in metrics)
    worst_cosine = min(metric[1] for metric in metrics)
    worst_max_abs = max(metric[2] for metric in metrics)
    print(
        "K3 shared worst "
        f"rel-L1={worst_rel_l1:.6f} "
        f"cosine={worst_cosine:.6f} "
        f"max-abs={worst_max_abs:.6f}"
    )
    assert math.isfinite(worst_rel_l1)
    assert math.isfinite(worst_cosine)
    assert math.isfinite(worst_max_abs)
