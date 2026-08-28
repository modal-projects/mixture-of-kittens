"""GPU tests for one-time Kimi K3 group-32 MXFP4 weight preparation."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import fields, replace

import pytest
import torch

from mok.kimi_k3 import KimiK3DecodeWorkspace

# The production decode step needs a rendezvoused TP8 workspace, and the tail
# suite already owns the fixture that builds one.
from .kimi_k3_tail_support import workspace  # noqa: F401


KIMI_K3_GROUP_SIZE = 32
KIMI_K3_UNIT_SCALE_BYTE = 0x7F
# OCP E2M1 code points, indexed by the three magnitude bits of a nibble.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
REQUIRED_PREPARE_MEMORY_BYTES = 96 * 1024**3
PREPARE_TP_RANK = 3

PreparedResult = tuple[object, dict[str, torch.Tensor], tuple[int, ...]]
PreparedWeights = Callable[[], PreparedResult]


def _encode_e2m1(value: float) -> int:
    """Encode one float as an OCP E2M1 nibble, round-to-nearest ties-to-even."""
    sign = 0x8 if math.copysign(1.0, value) < 0.0 else 0x0
    magnitude = abs(value)
    if magnitude >= E2M1_MAGNITUDES[7]:
        return sign | 0x7
    for index in range(7):
        low = E2M1_MAGNITUDES[index]
        high = E2M1_MAGNITUDES[index + 1]
        if magnitude > high:
            continue
        midpoint = (low + high) / 2.0
        if magnitude < midpoint:
            return sign | index
        if magnitude > midpoint:
            return sign | (index + 1)
        return sign | (index if index % 2 == 0 else index + 1)
    return sign | 0x7


def _decode_e2m1(nibble: int) -> float:
    magnitude = E2M1_MAGNITUDES[nibble & 0x7]
    return -magnitude if nibble & 0x8 else magnitude


def _reference_scale_byte(absolute_max: float) -> int:
    """Return the E8M0 byte whose scale keeps every E2M1 magnitude at most 6."""
    if absolute_max == 0.0:
        return KIMI_K3_UNIT_SCALE_BYTE
    scale_exponent = math.floor(math.log2(absolute_max)) - 2
    if absolute_max > 6.0 * 2.0**scale_exponent:
        scale_exponent += 1
    return min(max(scale_exponent + 127, 1), 254)


def _reference_pack_group(values: Sequence[float]) -> tuple[int, list[int]]:
    """Pack exactly 32 float values into an E8M0 byte and 16 E2M1 pair bytes."""
    assert len(values) == KIMI_K3_GROUP_SIZE
    scale_byte = _reference_scale_byte(max(abs(value) for value in values))
    if all(value == 0.0 for value in values):
        return KIMI_K3_UNIT_SCALE_BYTE, [0] * (KIMI_K3_GROUP_SIZE // 2)
    reciprocal = 2.0 ** -(scale_byte - 127)
    nibbles = [_encode_e2m1(value * reciprocal) for value in values]
    return scale_byte, [
        nibbles[2 * index] | (nibbles[2 * index + 1] << 4)
        for index in range(KIMI_K3_GROUP_SIZE // 2)
    ]


def _reference_pack(weight: torch.Tensor, padded_k: int) -> tuple[list[int], list[int]]:
    """Pack a BF16 ``[E, N, K]`` weight into flat packed and scale byte lists."""
    logical_k = weight.shape[-1]
    rows = weight.reshape(-1, logical_k).float().tolist()
    packed: list[int] = []
    scale: list[int] = []
    for row in rows:
        padded_row = row + [0.0] * (padded_k - logical_k)
        for start in range(0, padded_k, KIMI_K3_GROUP_SIZE):
            group = padded_row[start:start + KIMI_K3_GROUP_SIZE]
            scale_byte, group_bytes = _reference_pack_group(group)
            scale.append(scale_byte)
            packed.extend(group_bytes)
    return packed, scale


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Kimi K3 MXFP4 preparation requires CUDA")
    selected = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(selected)
    if torch.cuda.get_device_capability(selected) != (10, 3):
        pytest.skip("Kimi K3 MXFP4 preparation requires an SM103 GPU")
    return selected


@pytest.fixture(scope="module")
def peer_device(device: torch.device) -> Iterator[torch.device]:
    """A second CUDA device, with the first one left current for the caller."""
    if torch.cuda.device_count() < 2:
        pytest.skip("cross-device Kimi K3 MXFP4 preparation needs two CUDA devices")
    peer = torch.device("cuda", 1 if device.index == 0 else 0)
    if torch.cuda.get_device_capability(peer) != (10, 3):
        pytest.skip("Kimi K3 MXFP4 preparation requires an SM103 GPU")
    try:
        yield peer
    finally:
        torch.cuda.set_device(device)


def _code_point_row(exponent: int) -> list[float]:
    """Return 32 exactly representable E2M1 values scaled by ``2 ** exponent``."""
    magnitudes = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
    ]
    return [value * 2.0**exponent for value in magnitudes * 2]


def test_pack_mxfp4_uses_native_k32_layout_for_w1w3(device: torch.device) -> None:
    from mok.kimi_k3 import (
        KIMI_K3_LATENT_SIZE,
        KIMI_K3_W1W3_K,
        dequant_kimi_k3_mxfp4,
        pack_kimi_k3_mxfp4,
    )

    # Mixed W4A8 `kind::mxf8f6f4` runs at K=32, so W1/W3 need no padding.
    assert KIMI_K3_W1W3_K == KIMI_K3_LATENT_SIZE == 3584

    weight = torch.zeros(1, 384, KIMI_K3_W1W3_K, dtype=torch.bfloat16, device=device)
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=KIMI_K3_W1W3_K)

    assert packed.shape == (1, 384, 1792)
    assert scale.shape == (1, 384, 112)
    assert packed.dtype == torch.uint8
    assert scale.dtype == torch.uint8
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=KIMI_K3_W1W3_K)
    torch.testing.assert_close(restored, weight)


def test_pack_mxfp4_uses_w2_layout_for_logical_k_384(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    generator = torch.Generator(device=device).manual_seed(20260827)
    weight = torch.randn(
        2, 3584, 384, generator=generator, dtype=torch.bfloat16, device=device
    )
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=384)

    assert packed.shape == (2, 3584, 192)
    assert scale.shape == (2, 3584, 12)
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=384)
    assert restored.shape == weight.shape
    assert restored.dtype == torch.bfloat16


def test_pack_mxfp4_all_zero_groups_use_unit_scale_and_zero_data(
    device: torch.device,
) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(2, 3, 64, dtype=torch.bfloat16, device=device)
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=64)

    assert torch.equal(packed, torch.zeros_like(packed))
    assert torch.equal(scale, torch.full_like(scale, KIMI_K3_UNIT_SCALE_BYTE))


def test_pack_mxfp4_padded_groups_are_zero_with_unit_scale(
    device: torch.device,
) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    weight = torch.full((1, 2, 64), 3.0, dtype=torch.bfloat16, device=device)
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=128)

    assert packed.shape == (1, 2, 64)
    assert scale.shape == (1, 2, 4)
    assert torch.equal(packed[:, :, 32:], torch.zeros_like(packed[:, :, 32:]))
    assert torch.equal(
        scale[:, :, 2:], torch.full_like(scale[:, :, 2:], KIMI_K3_UNIT_SCALE_BYTE)
    )
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=64)
    torch.testing.assert_close(restored, weight, rtol=0, atol=0)


def test_pack_mxfp4_emits_exact_reference_bytes(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    rows = [
        _code_point_row(1) + _code_point_row(-10),
        _code_point_row(6) + [-value for value in _code_point_row(0)],
    ]
    weight = torch.tensor(
        [rows], dtype=torch.bfloat16, device=device
    )
    assert weight.shape == (1, 2, 64)

    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=96)
    expected_packed, expected_scale = _reference_pack(weight, padded_k=96)

    assert packed.flatten().tolist() == expected_packed
    assert scale.flatten().tolist() == expected_scale
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=64)
    torch.testing.assert_close(restored, weight, rtol=0, atol=0)


def test_pack_mxfp4_scales_each_group_independently(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    weight = torch.tensor(
        [[_code_point_row(4) + _code_point_row(-20)]],
        dtype=torch.bfloat16,
        device=device,
    )
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=64)

    assert scale[0, 0, 0].item() == 127 + 4
    assert scale[0, 0, 1].item() == 127 - 20
    torch.testing.assert_close(
        dequant_kimi_k3_mxfp4(packed, scale, logical_k=64), weight, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("value", "expected_scale_byte"),
    [(2.0**-126, 1), (2.0**-120, 5), (4.0, 127), (2.0**127, 252)],
)
def test_pack_mxfp4_encodes_e8m0_exponent_boundaries(
    device: torch.device, value: float, expected_scale_byte: int
) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 1, 32, dtype=torch.bfloat16, device=device)
    weight[0, 0, 5] = value
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=32)

    assert scale[0, 0, 0].item() == expected_scale_byte
    torch.testing.assert_close(
        dequant_kimi_k3_mxfp4(packed, scale, logical_k=32), weight, rtol=0, atol=0
    )


def test_pack_mxfp4_rounds_ties_to_even_code_point(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    tie_values = [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    weight = torch.zeros(1, 1, 32, dtype=torch.bfloat16, device=device)
    weight[0, 0, : len(tie_values)] = torch.tensor(
        tie_values, dtype=torch.bfloat16, device=device
    )
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=32)

    assert scale[0, 0, 0].item() == KIMI_K3_UNIT_SCALE_BYTE
    expected = torch.zeros(1, 1, 32, dtype=torch.bfloat16, device=device)
    expected[0, 0, : len(tie_values)] = torch.tensor(
        [6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], dtype=torch.bfloat16, device=device
    )
    torch.testing.assert_close(
        dequant_kimi_k3_mxfp4(packed, scale, logical_k=32), expected, rtol=0, atol=0
    )


def test_dequant_mxfp4_decodes_every_e2m1_code_point(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4

    packed = torch.tensor(
        [[[nibble | (nibble << 4) for nibble in range(16)]]],
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.full(
        (1, 1, 1), KIMI_K3_UNIT_SCALE_BYTE, dtype=torch.uint8, device=device
    )
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=32)

    expected = torch.tensor(
        [[[_decode_e2m1(nibble) for nibble in range(16) for _ in range(2)]]],
        dtype=torch.bfloat16,
        device=device,
    )
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_dequant_mxfp4_applies_group_e8m0_scale(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4

    packed = torch.full((1, 1, 32), 0x77, dtype=torch.uint8, device=device)
    scale = torch.tensor([[[130, 124]]], dtype=torch.uint8, device=device)
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=64)

    expected = torch.cat(
        (
            torch.full((32,), 6.0 * 2.0**3, dtype=torch.bfloat16, device=device),
            torch.full((32,), 6.0 * 2.0**-3, dtype=torch.bfloat16, device=device),
        )
    ).reshape(1, 1, 64)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_pack_mxfp4_round_trip_error_is_bounded_per_group(
    device: torch.device,
) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    generator = torch.Generator(device=device).manual_seed(4)
    weight = torch.randn(
        3, 17, 128, generator=generator, dtype=torch.bfloat16, device=device
    )
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=160)
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=128)

    # The widest E2M1 gap is 2 (between 4 and 6), so one rounding step per group
    # is bounded by the group's E8M0 scale.
    group_scale = torch.pow(
        2.0, scale[:, :, :4].int().float() - 127.0
    ).repeat_interleave(KIMI_K3_GROUP_SIZE, dim=-1)
    error = (restored.float() - weight.float()).abs()
    assert torch.all(error <= group_scale)
    assert torch.isfinite(restored.float()).all()


@pytest.mark.parametrize("logical_k", [31, 33, 3583])
def test_pack_mxfp4_rejects_logical_k_that_is_not_a_group_multiple(
    device: torch.device, logical_k: int
) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 2, logical_k, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="multiple of 32"):
        pack_kimi_k3_mxfp4(weight, padded_k=3584)


@pytest.mark.parametrize("padded_k", [33, 3647])
def test_pack_mxfp4_rejects_padded_k_that_is_not_a_group_multiple(
    device: torch.device, padded_k: int
) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 2, 32, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="multiple of 32"):
        pack_kimi_k3_mxfp4(weight, padded_k=padded_k)


def test_pack_mxfp4_rejects_padded_k_below_logical_k(device: torch.device) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 2, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="at least"):
        pack_kimi_k3_mxfp4(weight, padded_k=32)


def test_pack_mxfp4_rejects_wrong_dtype(device: torch.device) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 2, 64, dtype=torch.float32, device=device)
    with pytest.raises(TypeError, match="torch.bfloat16"):
        pack_kimi_k3_mxfp4(weight, padded_k=64)


def test_pack_mxfp4_rejects_cpu_tensor(device: torch.device) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        pack_kimi_k3_mxfp4(weight, padded_k=64)


def test_pack_mxfp4_rejects_noncontiguous_tensor(device: torch.device) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 64, 2, dtype=torch.bfloat16, device=device).transpose(1, 2)
    with pytest.raises(ValueError, match="contiguous"):
        pack_kimi_k3_mxfp4(weight, padded_k=64)


def test_pack_mxfp4_rejects_non_3d_tensor(device: torch.device) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(2, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match=r"\[E, N, K\]"):
        pack_kimi_k3_mxfp4(weight, padded_k=64)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_pack_mxfp4_rejects_nonfinite_values(
    device: torch.device, value: float
) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    weight = torch.zeros(1, 2, 64, dtype=torch.bfloat16, device=device)
    weight[0, 1, 7] = value
    with pytest.raises(ValueError, match="finite"):
        pack_kimi_k3_mxfp4(weight, padded_k=64)


def test_dequant_mxfp4_rejects_inconsistent_packed_and_scale_widths(
    device: torch.device,
) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4

    packed = torch.zeros(1, 2, 32, dtype=torch.uint8, device=device)
    scale = torch.full((1, 2, 3), KIMI_K3_UNIT_SCALE_BYTE, dtype=torch.uint8,
                       device=device)
    with pytest.raises(ValueError, match="scale"):
        dequant_kimi_k3_mxfp4(packed, scale, logical_k=64)


def test_dequant_mxfp4_rejects_logical_k_above_padded_k(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4

    packed = torch.zeros(1, 2, 32, dtype=torch.uint8, device=device)
    scale = torch.full((1, 2, 2), KIMI_K3_UNIT_SCALE_BYTE, dtype=torch.uint8,
                       device=device)
    with pytest.raises(ValueError, match="logical_k"):
        dequant_kimi_k3_mxfp4(packed, scale, logical_k=96)


def test_dequant_mxfp4_rejects_wrong_dtype(device: torch.device) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4

    packed = torch.zeros(1, 2, 32, dtype=torch.int8, device=device)
    scale = torch.full((1, 2, 2), KIMI_K3_UNIT_SCALE_BYTE, dtype=torch.uint8,
                       device=device)
    with pytest.raises(TypeError, match="torch.uint8"):
        dequant_kimi_k3_mxfp4(packed, scale, logical_k=64)


def test_pack_mxfp4_on_peer_device_ignores_the_current_device(
    device: torch.device, peer_device: torch.device
) -> None:
    from mok.kimi_k3 import pack_kimi_k3_mxfp4

    torch.cuda.set_device(device)
    weight = torch.full((1, 4, 64), 3.0, dtype=torch.bfloat16, device=peer_device)
    weight[0, 1, :KIMI_K3_GROUP_SIZE] = 0.0

    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=96)
    torch.cuda.synchronize(peer_device)

    assert torch.cuda.current_device() == device.index
    assert packed.device == peer_device
    assert scale.device == peer_device
    expected_packed, expected_scale = _reference_pack(weight, padded_k=96)
    assert packed.flatten().tolist() == expected_packed
    assert scale.flatten().tolist() == expected_scale


def test_dequant_mxfp4_on_peer_device_ignores_the_current_device(
    device: torch.device, peer_device: torch.device
) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4

    torch.cuda.set_device(device)
    packed = torch.full((1, 4, 48), 0x77, dtype=torch.uint8, device=peer_device)
    scale = torch.full((1, 4, 3), 130, dtype=torch.uint8, device=peer_device)

    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=64)
    torch.cuda.synchronize(peer_device)

    assert torch.cuda.current_device() == device.index
    assert restored.device == peer_device
    expected = torch.full(
        (1, 4, 64), 6.0 * 2.0**3, dtype=torch.bfloat16, device=peer_device
    )
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_pack_mxfp4_on_peer_device_uses_that_devices_current_stream(
    device: torch.device, peer_device: torch.device
) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    torch.cuda.set_device(device)
    weight = torch.zeros(1, 4, 64, dtype=torch.bfloat16, device=peer_device)
    side_stream = torch.cuda.Stream(device=peer_device)
    previous_stream = torch.cuda.current_stream(peer_device)
    # set_stream also makes the stream's device current, which is the opposite of
    # the state under test, so put the first device back afterwards.
    torch.cuda.set_stream(side_stream)
    torch.cuda.set_device(device)
    try:
        with torch.cuda.device(peer_device):
            # Long enough that a launch on any other stream would observe the
            # pre-fill zeros instead of the value the pack has to see.
            torch.cuda._sleep(1 << 28)
        weight.fill_(3.0)
        packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=64)
        restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=64)
        assert torch.cuda.current_device() == device.index
        side_stream.synchronize()
        torch.testing.assert_close(
            restored, torch.full_like(weight, 3.0), rtol=0, atol=0
        )
    finally:
        torch.cuda.set_stream(previous_stream)
        torch.cuda.set_device(device)


def _replicated_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    from mok.kimi_k3 import (
        KIMI_K3_HIDDEN_SIZE,
        KIMI_K3_LATENT_SIZE,
        KIMI_K3_NUM_EXPERTS,
        KIMI_K3_SHARED_INTERMEDIATE_SIZE,
    )

    generator = torch.Generator(device=device).manual_seed(11)

    def bf16(*shape: int) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, device=device) * 0.25
        ).bfloat16()

    return {
        "router_weight": bf16(KIMI_K3_NUM_EXPERTS, KIMI_K3_HIDDEN_SIZE),
        "router_correction_bias": torch.randn(
            KIMI_K3_NUM_EXPERTS, generator=generator, device=device
        ),
        "routed_latent_down_proj": bf16(KIMI_K3_LATENT_SIZE, KIMI_K3_HIDDEN_SIZE),
        "routed_latent_up_proj": bf16(KIMI_K3_HIDDEN_SIZE, KIMI_K3_LATENT_SIZE),
        "routed_latent_norm_weight": bf16(KIMI_K3_LATENT_SIZE),
        "shared_gate_proj": bf16(
            KIMI_K3_SHARED_INTERMEDIATE_SIZE, KIMI_K3_HIDDEN_SIZE
        ),
        "shared_up_proj": bf16(KIMI_K3_SHARED_INTERMEDIATE_SIZE, KIMI_K3_HIDDEN_SIZE),
        "shared_down_proj": bf16(
            KIMI_K3_HIDDEN_SIZE, KIMI_K3_SHARED_INTERMEDIATE_SIZE
        ),
    }


def _sparse_expert_pattern(size: int, device: torch.device) -> torch.Tensor:
    """Return index-identifying values that survive MXFP4 packing exactly.

    The seven nonzero E2M1 magnitudes round-trip without error whenever a group
    contains the largest of them, and the period seven does not divide either TP
    shard width, so every rank slices a distinct sequence.
    """
    table = torch.tensor(
        [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.bfloat16, device=device
    )
    return table[torch.arange(size, device=device) % table.numel()]


@pytest.fixture(scope="module")
def prepared_weights(device: torch.device) -> Iterator[PreparedWeights]:
    """Prepare the full-size TP8 shard once, on first use inside a test body.

    Returns the prepared weights, the replicated inputs they were built from,
    and the storage widths preparation asked the packer for, in call order.
    """
    cache: dict[str, PreparedResult] = {}

    def prepare() -> PreparedResult:
        if "value" in cache:
            return cache["value"]

        from mok import ops
        from mok.kimi_k3 import prepare_kimi_k3_decode_weights

        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < REQUIRED_PREPARE_MEMORY_BYTES:
            pytest.skip("full-size Kimi K3 expert preparation needs 96 GiB free")

        replicated = _replicated_inputs(device)
        expert_w1 = torch.zeros(
            896, 3072, 3584, dtype=torch.bfloat16, device=device
        )
        expert_w1[:, :, 0] = _sparse_expert_pattern(3072, device)
        expert_w3 = torch.zeros_like(expert_w1)
        expert_w3[:, :, 0] = -_sparse_expert_pattern(3072, device)
        expert_w2 = torch.zeros(
            896, 3584, 3072, dtype=torch.bfloat16, device=device
        )
        expert_w2[:, 0, :] = _sparse_expert_pattern(3072, device)

        storage_widths: list[int] = []
        packer = ops.pack_kimi_k3_mxfp4

        def recording_packer(
            weight: torch.Tensor, padded_k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            storage_widths.append(padded_k)
            return packer(weight, padded_k)

        ops.pack_kimi_k3_mxfp4 = recording_packer
        try:
            weights = prepare_kimi_k3_decode_weights(
                **replicated,
                expert_w1=expert_w1,
                expert_w3=expert_w3,
                expert_w2=expert_w2,
                tp_rank=PREPARE_TP_RANK,
            )
        finally:
            ops.pack_kimi_k3_mxfp4 = packer
        del expert_w1, expert_w3, expert_w2
        torch.cuda.empty_cache()
        cache["value"] = (weights, replicated, tuple(storage_widths))
        return cache["value"]

    try:
        yield prepare
    finally:
        cache.clear()
        torch.cuda.empty_cache()


def test_prepare_weights_returns_canonical_prepared_layouts(
    device: torch.device, prepared_weights: PreparedWeights
) -> None:
    from mok.kimi_k3 import (
        KimiK3DecodeWeights,
        validate_kimi_k3_decode_inputs,
    )

    weights, replicated, _ = prepared_weights()
    assert isinstance(weights, KimiK3DecodeWeights)
    assert tuple(field.name for field in fields(KimiK3DecodeWeights)) == (
        "router_weight",
        "router_correction_bias",
        "routed_expert_down_proj",
        "routed_expert_up_proj",
        "routed_latent_rmsnorm_weight",
        "expert_w1_packed",
        "expert_w1_scale",
        "expert_w3_packed",
        "expert_w3_scale",
        "expert_w2_packed",
        "expert_w2_scale",
        "shared_gate_proj",
        "shared_up_proj",
        "shared_down_proj",
        "tp_rank",
    )
    assert weights.tp_rank == PREPARE_TP_RANK
    assert weights.expert_w1_packed.shape == (896, 384, 1792)
    assert weights.expert_w1_scale.shape == (896, 384, 112)
    assert weights.expert_w3_packed.shape == (896, 384, 1792)
    assert weights.expert_w3_scale.shape == (896, 384, 112)
    assert weights.expert_w2_packed.shape == (896, 3584, 192)
    assert weights.expert_w2_scale.shape == (896, 3584, 12)
    assert weights.router_weight.data_ptr() == replicated["router_weight"].data_ptr()

    hidden_states = torch.zeros(4, 7168, dtype=torch.bfloat16, device=device)
    assert validate_kimi_k3_decode_inputs(hidden_states, weights) is None


def test_prepare_weights_pack_at_native_storage_widths(
    device: torch.device, prepared_weights: PreparedWeights
) -> None:
    from mok.kimi_k3 import (
        KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
        KIMI_K3_TP_SIZE,
        KIMI_K3_W1W3_K,
    )

    _, _, storage_widths = prepared_weights()
    routed_width = KIMI_K3_ROUTED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE

    # W1, W3, then W2. None of the three may ask for padding.
    assert storage_widths == (KIMI_K3_W1W3_K, KIMI_K3_W1W3_K, routed_width)
    assert storage_widths == (3584, 3584, 384)


def test_prepare_weights_slices_routed_and_shared_tp_ranges(
    device: torch.device, prepared_weights: PreparedWeights
) -> None:
    from mok.kimi_k3 import dequant_kimi_k3_mxfp4

    weights, replicated, _ = prepared_weights()
    routed_start = PREPARE_TP_RANK * 384
    shared_start = PREPARE_TP_RANK * 768
    expected_rows = _sparse_expert_pattern(3072, device)[
        routed_start:routed_start + 384
    ]

    for expert in (0, 895):
        w1 = dequant_kimi_k3_mxfp4(
            weights.expert_w1_packed[expert:expert + 1],
            weights.expert_w1_scale[expert:expert + 1],
            logical_k=3584,
        )
        w3 = dequant_kimi_k3_mxfp4(
            weights.expert_w3_packed[expert:expert + 1],
            weights.expert_w3_scale[expert:expert + 1],
            logical_k=3584,
        )
        assert w1.shape == (1, 384, 3584)
        torch.testing.assert_close(w1[0, :, 0], expected_rows, rtol=0, atol=0)
        torch.testing.assert_close(w3[0, :, 0], -expected_rows, rtol=0, atol=0)
        assert torch.equal(w1[0, :, 1:], torch.zeros_like(w1[0, :, 1:]))

        w2 = dequant_kimi_k3_mxfp4(
            weights.expert_w2_packed[expert:expert + 1],
            weights.expert_w2_scale[expert:expert + 1],
            logical_k=384,
        )
        assert w2.shape == (1, 3584, 384)
        torch.testing.assert_close(
            w2[0, 0, :],
            _sparse_expert_pattern(3072, device)[routed_start:routed_start + 384],
            rtol=0,
            atol=0,
        )
        assert torch.equal(w2[0, 1:, :], torch.zeros_like(w2[0, 1:, :]))

    torch.testing.assert_close(
        weights.shared_gate_proj,
        replicated["shared_gate_proj"][shared_start:shared_start + 768],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        weights.shared_up_proj,
        replicated["shared_up_proj"][shared_start:shared_start + 768],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        weights.shared_down_proj,
        replicated["shared_down_proj"][:, shared_start:shared_start + 768],
        rtol=0,
        atol=0,
    )


def test_prepare_weights_are_not_repacked_by_decode_calls(
    workspace: KimiK3DecodeWorkspace,
    prepared_weights: PreparedWeights,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation is one-time: the decode step reads the shard it was given.

    The packer is the expensive part of preparation, so a hot path that
    repacked -- or that quietly handed back a repacked copy -- would cost far
    more than the step itself. Driving the real production call twice, with the
    packer instrumented, is what rules that out; the tensors the second call
    reads must still be the very ones preparation produced.
    """
    from mok import ops
    from mok.kimi_k3 import KimiK3DecodeConfig, kimi_k3_decode

    weights, _, _ = prepared_weights()
    packed_fields = (
        weights.expert_w1_packed,
        weights.expert_w1_scale,
        weights.expert_w3_packed,
        weights.expert_w3_scale,
        weights.expert_w2_packed,
        weights.expert_w2_scale,
    )
    pointers = [tensor.data_ptr() for tensor in packed_fields]
    calls: list[int] = []
    original_pack = ops.pack_kimi_k3_mxfp4

    def counting_pack(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original_pack(*args, **kwargs)

    monkeypatch.setattr(ops, "pack_kimi_k3_mxfp4", counting_pack)

    # The shard was prepared for one fixed rank so that its packed layout is
    # comparable on every rank; the step only needs the rank it is told, and
    # nothing here reads the values it computes.
    local = replace(weights, tp_rank=workspace.tp_rank)
    hidden_states = torch.zeros(
        4, 7168, dtype=torch.bfloat16, device=workspace.device
    )
    for _ in range(2):
        output = kimi_k3_decode(
            KimiK3DecodeConfig(), workspace, local, hidden_states
        )
        assert output.shape == (4, 7168)

    assert calls == []
    assert [tensor.data_ptr() for tensor in packed_fields] == pointers


def test_prepare_weights_rejects_non_bf16_replicated_weight(
    device: torch.device,
) -> None:
    from mok.kimi_k3 import prepare_kimi_k3_decode_weights

    with pytest.raises(TypeError, match="torch.bfloat16"):
        prepare_kimi_k3_decode_weights(
            router_weight=torch.zeros(
                896, 7168, dtype=torch.float32, device=device
            ),
            router_correction_bias=torch.zeros(896, device=device),
            routed_latent_down_proj=torch.zeros(
                3584, 7168, dtype=torch.bfloat16, device=device
            ),
            routed_latent_up_proj=torch.zeros(
                7168, 3584, dtype=torch.bfloat16, device=device
            ),
            routed_latent_norm_weight=torch.zeros(
                3584, dtype=torch.bfloat16, device=device
            ),
            expert_w1=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            expert_w3=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            expert_w2=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            shared_gate_proj=torch.zeros(
                6144, 7168, dtype=torch.bfloat16, device=device
            ),
            shared_up_proj=torch.zeros(
                6144, 7168, dtype=torch.bfloat16, device=device
            ),
            shared_down_proj=torch.zeros(
                7168, 6144, dtype=torch.bfloat16, device=device
            ),
            tp_rank=0,
        )


def test_prepare_weights_rejects_wrong_expert_shape(device: torch.device) -> None:
    from mok.kimi_k3 import prepare_kimi_k3_decode_weights

    with pytest.raises(ValueError, match="expert_w1 must have shape"):
        prepare_kimi_k3_decode_weights(
            router_weight=torch.zeros(
                896, 7168, dtype=torch.bfloat16, device=device
            ),
            router_correction_bias=torch.zeros(896, device=device),
            routed_latent_down_proj=torch.zeros(
                3584, 7168, dtype=torch.bfloat16, device=device
            ),
            routed_latent_up_proj=torch.zeros(
                7168, 3584, dtype=torch.bfloat16, device=device
            ),
            routed_latent_norm_weight=torch.zeros(
                3584, dtype=torch.bfloat16, device=device
            ),
            expert_w1=torch.zeros(896, 3072, 3583, dtype=torch.bfloat16,
                                  device=device),
            expert_w3=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            expert_w2=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            shared_gate_proj=torch.zeros(
                6144, 7168, dtype=torch.bfloat16, device=device
            ),
            shared_up_proj=torch.zeros(
                6144, 7168, dtype=torch.bfloat16, device=device
            ),
            shared_down_proj=torch.zeros(
                7168, 6144, dtype=torch.bfloat16, device=device
            ),
            tp_rank=0,
        )


@pytest.mark.parametrize("tp_rank", [-1, 8, 1.0])
def test_prepare_weights_rejects_invalid_tp_rank(
    device: torch.device, tp_rank: object
) -> None:
    from mok.kimi_k3 import prepare_kimi_k3_decode_weights

    with pytest.raises(ValueError, match="tp_rank"):
        prepare_kimi_k3_decode_weights(
            router_weight=torch.zeros(
                896, 7168, dtype=torch.bfloat16, device=device
            ),
            router_correction_bias=torch.zeros(896, device=device),
            routed_latent_down_proj=torch.zeros(
                3584, 7168, dtype=torch.bfloat16, device=device
            ),
            routed_latent_up_proj=torch.zeros(
                7168, 3584, dtype=torch.bfloat16, device=device
            ),
            routed_latent_norm_weight=torch.zeros(
                3584, dtype=torch.bfloat16, device=device
            ),
            expert_w1=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            expert_w3=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            expert_w2=torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=device),
            shared_gate_proj=torch.zeros(
                6144, 7168, dtype=torch.bfloat16, device=device
            ),
            shared_up_proj=torch.zeros(
                6144, 7168, dtype=torch.bfloat16, device=device
            ),
            shared_down_proj=torch.zeros(
                7168, 6144, dtype=torch.bfloat16, device=device
            ),
            tp_rank=tp_rank,
        )


def test_mxfp4_operator_fakes_match_registered_schemas(device: torch.device) -> None:
    import inspect

    from mok import _fake_impls

    for operator, fake in (
        (torch.ops.mok.pack_kimi_k3_mxfp4, _fake_impls._pack_kimi_k3_mxfp4_fake),
        (torch.ops.mok.dequant_kimi_k3_mxfp4,
         _fake_impls._dequant_kimi_k3_mxfp4_fake),
    ):
        schema_names = tuple(
            argument.name for argument in operator.default._schema.arguments
        )
        assert tuple(inspect.signature(fake).parameters) == schema_names


def test_mxfp4_operator_fakes_report_prepared_shapes(device: torch.device) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from mok.kimi_k3 import KIMI_K3_W1W3_K

    with FakeTensorMode():
        weight = torch.empty(
            896, 384, KIMI_K3_W1W3_K, dtype=torch.bfloat16, device=device
        )
        packed, scale = torch.ops.mok.pack_kimi_k3_mxfp4(weight, KIMI_K3_W1W3_K)
        assert packed.shape == (896, 384, 1792)
        assert scale.shape == (896, 384, 112)
        assert packed.dtype == torch.uint8
        assert scale.dtype == torch.uint8

        restored = torch.ops.mok.dequant_kimi_k3_mxfp4(
            packed, scale, KIMI_K3_W1W3_K
        )
        assert restored.shape == (896, 384, 3584)
        assert restored.dtype == torch.bfloat16
