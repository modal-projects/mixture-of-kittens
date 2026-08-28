"""GPU tests for assignment-driven Kimi K3 mixed MXFP8-by-MXFP4 experts."""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


HIDDEN = 3584
INTERMEDIATE = 384
EXPERTS = 896
TOPK = 16
MAX_TOKENS = 128
MAX_ASSIGNMENTS = MAX_TOKENS * TOPK
CAPACITY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
PROBE_COLUMNS = 128
GAIN_BINADES = 5
ADDRESS_EXPERTS = (1, 447, 895)
# Deliberately lopsided so reversing a token's two experts across its two slots
# changes the result by 0.75 of the difference between the two expert outputs.
ADDRESS_WEIGHTS = (0.125, 0.875)
GROUP = 32
ALIGNMENT = 256
UNIT_SCALE = 0x7F

_EXPERT_ARGUMENTS = (
    "latent_x",
    "expert_w1_packed",
    "expert_w1_scale",
    "expert_w3_packed",
    "expert_w3_scale",
    "expert_w2_packed",
    "expert_w2_scale",
    "routed_output",
    "scratch",
    "active_tokens",
)


def _aligned(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _scratch_layout() -> dict[str, tuple[int, int]]:
    """Independent byte-level model of ``kimi_k3_decode::Scratch``."""
    regions = (
        ("phase", 20 * 4),
        ("expert_ids", MAX_ASSIGNMENTS * 4),
        ("expert_weights", MAX_ASSIGNMENTS * 4),
        ("expert_counts", EXPERTS * 4),
        ("expert_offsets", (EXPERTS + 1) * 4),
        ("assignment_tokens", MAX_ASSIGNMENTS * 4),
        ("assignment_slots", MAX_ASSIGNMENTS * 4),
        ("latent_mxfp8", MAX_TOKENS * HIDDEN),
        ("latent_scale", MAX_TOKENS * (HIDDEN // GROUP)),
        ("situ_mxfp8", MAX_ASSIGNMENTS * INTERMEDIATE),
        ("situ_scale", MAX_ASSIGNMENTS * (INTERMEDIATE // GROUP)),
        ("routed_accumulator", MAX_TOKENS * HIDDEN * 4),
        ("shared_gate", MAX_TOKENS * 768 * 2),
        ("shared_up", MAX_TOKENS * 768 * 2),
        ("shared_activated", MAX_TOKENS * 768 * 2),
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


@dataclass
class ExpertWeights:
    w1_packed: torch.Tensor
    w1_scale: torch.Tensor
    w3_packed: torch.Tensor
    w3_scale: torch.Tensor
    w2_packed: torch.Tensor
    w2_scale: torch.Tensor

    def arguments(self) -> tuple[torch.Tensor, ...]:
        return (
            self.w1_packed,
            self.w1_scale,
            self.w3_packed,
            self.w3_scale,
            self.w2_packed,
            self.w2_scale,
        )


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Kimi K3 routed experts require CUDA")
    selected = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(selected)
    if torch.cuda.get_device_capability(selected) != (10, 3):
        pytest.skip("Kimi K3 routed experts require an SM103 GPU")
    return selected


@pytest.fixture(scope="module")
def peer_device(device: torch.device) -> Iterator[torch.device]:
    if torch.cuda.device_count() < 2:
        pytest.skip("cross-device Kimi K3 experts need two visible GPUs")
    peer = torch.device("cuda", 1 if device.index == 0 else 0)
    if torch.cuda.get_device_capability(peer) != (10, 3):
        pytest.skip("Kimi K3 routed experts require an SM103 GPU")
    try:
        yield peer
    finally:
        torch.cuda.set_device(device)


def _latent_column_for_row(row: int) -> int:
    """Spread the 384 gate/up rows across all 112 latent scale groups."""
    return row * HIDDEN // INTERMEDIATE


def _down_row_gain(device: torch.device, phase: int = 0) -> torch.Tensor:
    """Give each of the 3584 down-projection rows a distinguishable gain.

    Each exponent comes from an integer hash of the row index, so the sequence
    is nonperiodic: a tile written to the wrong ``output_base``, a row displaced
    by a whole intermediate, and any other displacement all change the result
    instead of landing on an identical value.  A fixed period would have been
    blind to a displacement of exactly that period.  ``phase`` shifts the hash
    input, which gives one expert a shard tag no other expert repeats.  Every
    gain is a power of two in ``[2^-4, 1]`` and so exactly representable in
    E2M1, which keeps the dequantized reference bit-exact.
    """
    rows = torch.arange(HIDDEN, dtype=torch.int64, device=device)
    mixed = (rows + phase + 1) * 0x9E3779B1
    mixed = mixed ^ (mixed >> 15)
    mixed = mixed ^ (mixed >> 7)
    return torch.pow(2.0, -(mixed % GAIN_BINADES).float()).bfloat16()


def _make_structured_weights(device: torch.device) -> ExpertWeights:
    """Make full native-layout weights with an exactly representable FP4 path.

    Gate row ``r`` selects latent column ``_latent_column_for_row(r)`` with
    ``+1.0`` and the matching up row selects the same column with ``+0.5``, so
    the two projections stay distinguishable through the asymmetric SiTU while
    every SiTU value keeps the sign of ``gate * up`` and therefore stays
    positive.  Every W2 row then sums all 384 SiTU columns at the row's own
    ``_down_row_gain``, so the 3584-wide result varies across output tiles
    instead of repeating one value.  Summing 384
    positive terms is what makes the end-to-end error metric average the
    mandatory MXFP8 activation and SiTU roundings instead of exposing one
    E4M3 rounding, whose worst case alone is 6.25%.

    The selected columns walk the whole 3584-wide latent, so both nibbles of
    the packed FP4 byte, all 112 W1/W3 scale groups, all 12 W2 scale groups,
    and all-zero groups are exercised.  Every value is a power of two that
    survives MXFP4 exactly, so the dequantized prepared weights the reference
    consumes are bit-exact.  The same shard is installed for all 896 experts so
    the exhaustive assignment distribution keeps an analytical reference.
    """
    from mok.ops import pack_kimi_k3_mxfp4

    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 4 * 1024**3:
        pytest.skip("full Kimi K3 prepared expert tensors need 4 GiB free")
    rows = torch.arange(INTERMEDIATE, device=device)
    columns = torch.tensor(
        [_latent_column_for_row(row) for row in range(INTERMEDIATE)],
        device=device,
    )
    gate_dense = torch.zeros(
        1, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device
    )
    gate_dense[0, rows, columns] = 1.0
    up_dense = torch.zeros_like(gate_dense)
    up_dense[0, rows, columns] = 0.5
    down_dense = _down_row_gain(device).view(1, HIDDEN, 1).expand(
        1, HIDDEN, INTERMEDIATE
    ).contiguous()

    def _broadcast(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed, scale = pack_kimi_k3_mxfp4(dense, dense.size(-1))
        return (
            packed.expand(EXPERTS, -1, -1).contiguous(),
            scale.expand(EXPERTS, -1, -1).contiguous(),
        )

    w1_packed, w1_scale = _broadcast(gate_dense)
    w3_packed, w3_scale = _broadcast(up_dense)
    w2_packed, w2_scale = _broadcast(down_dense)
    del gate_dense, up_dense, down_dense
    return ExpertWeights(
        w1_packed, w1_scale, w3_packed, w3_scale, w2_packed, w2_scale
    )


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact FP32 Kimi K3 SiTU contract."""
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )


def _random_latent(
    device: torch.device, rows: int, seed: int
) -> torch.Tensor:
    """Draw a dense latent so every gate/up row and scale group is active."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            rows, HIDDEN, generator=generator, dtype=torch.float32, device=device
        )
        * 0.25
    ).bfloat16()


@pytest.fixture(scope="module")
def weights(device: torch.device) -> Iterator[ExpertWeights]:
    value = _make_structured_weights(device)
    try:
        yield value
    finally:
        del value
        torch.cuda.empty_cache()


@pytest.fixture
def scratch(device: torch.device) -> torch.Tensor:
    return torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)


def _region(scratch: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    offset, size = SCRATCH_LAYOUT[name]
    return scratch[offset:offset + size].view(dtype)


Assignment = tuple[int, int, int, float]  # expert, token, slot, normalized weight


def _write_assignments(
    scratch: torch.Tensor, assignments: Sequence[Assignment]
) -> None:
    """Publish the exact expert-major state produced by Task 5."""
    device = scratch.device
    ordered = sorted(assignments, key=lambda item: (item[0], item[1], item[2]))
    counts = torch.zeros(EXPERTS, dtype=torch.int32, device=device)
    if ordered:
        expert_tensor = torch.tensor(
            [item[0] for item in ordered], dtype=torch.int64, device=device
        )
        counts.scatter_add_(
            0, expert_tensor, torch.ones_like(expert_tensor, dtype=torch.int32)
        )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts.cumsum(0).to(torch.int32),
        )
    )

    _region(scratch, "expert_ids", torch.int32).zero_()
    _region(scratch, "expert_weights", torch.float32).zero_()
    _region(scratch, "expert_counts", torch.int32).copy_(counts)
    _region(scratch, "expert_offsets", torch.int32).copy_(offsets)
    assignment_tokens = _region(scratch, "assignment_tokens", torch.int32)
    assignment_slots = _region(scratch, "assignment_slots", torch.int32)
    assignment_tokens.zero_()
    assignment_slots.zero_()
    for position, (expert, token, slot, weight) in enumerate(ordered):
        route = token * TOPK + slot
        _region(scratch, "expert_ids", torch.int32)[route] = expert
        _region(scratch, "expert_weights", torch.float32)[route] = weight
        assignment_tokens[position] = token
        assignment_slots[position] = slot


def _call(
    latent_x: torch.Tensor,
    weights: ExpertWeights,
    routed_output: torch.Tensor,
    scratch: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    from mok import ops

    return ops._kimi_k3_routed_experts(
        latent_x,
        *weights.arguments(),
        routed_output,
        scratch,
        active_tokens,
    )


def _dequant_one(
    packed: torch.Tensor, scale: torch.Tensor, logical_k: int, expert: int
) -> torch.Tensor:
    from mok.ops import dequant_kimi_k3_mxfp4

    return dequant_kimi_k3_mxfp4(
        packed[expert:expert + 1], scale[expert:expert + 1], logical_k
    )[0].float()


def _reference(
    latent_x: torch.Tensor,
    weights: ExpertWeights,
    assignments: Sequence[Assignment],
    active_tokens: int,
) -> torch.Tensor:
    """Dequantize the exact prepared weights and evaluate the FP32 contract."""
    result = torch.zeros(
        active_tokens, HIDDEN, dtype=torch.float32, device=latent_x.device
    )
    by_expert: dict[int, list[Assignment]] = {}
    for assignment in assignments:
        by_expert.setdefault(assignment[0], []).append(assignment)
    for expert, selected in by_expert.items():
        tokens = torch.tensor(
            [item[1] for item in selected], dtype=torch.int64, device=latent_x.device
        )
        route_weights = torch.tensor(
            [item[3] for item in selected],
            dtype=torch.float32,
            device=latent_x.device,
        )
        w1 = _dequant_one(weights.w1_packed, weights.w1_scale, HIDDEN, expert)
        w3 = _dequant_one(weights.w3_packed, weights.w3_scale, HIDDEN, expert)
        w2 = _dequant_one(weights.w2_packed, weights.w2_scale, INTERMEDIATE, expert)
        selected_x = latent_x.index_select(0, tokens).float()
        situ = _situ(selected_x @ w1.T, selected_x @ w3.T)
        contribution = (situ @ w2.T) * route_weights[:, None]
        result.index_add_(0, tokens, contribution)
    return result.bfloat16().float()


def _assert_expert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    relative_l1 = difference.abs().sum() / expected.abs().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    )
    maximum = difference.abs().max()
    assert torch.isfinite(actual).all()
    assert float(relative_l1) <= 0.05
    assert float(cosine) >= 0.999
    assert float(maximum) <= 1.0


def _e8m0_scale_bytes(absolute_max: torch.Tensor) -> torch.Tensor:
    """Model the kernel's E8M0 scale-byte selection over the whole float range.

    OCP MX v1.0 and PTX ISA 9.3 both define the E8M0 scale as ``2^(byte - 127)``
    with byte 255 reserved for NaN, so byte 0 is the exact minimum scale
    ``2^-127`` and byte 254 the maximum.  The selected scale is the smallest
    power of two that keeps ``absolute_max / scale`` inside E4M3's 448 maximum,
    and ``448 == 1.75 * 2^8`` is what puts the mantissa threshold at 1.75.
    """
    mantissa, exponent = torch.frexp(absolute_max.float())
    # frexp returns a mantissa in [0.5, 1), so 1.75 in [1, 2) becomes 0.875.
    scale_exponent = torch.where(mantissa <= 0.875, exponent - 9, exponent - 8)
    scale_bytes = (scale_exponent + 127).clamp(0, 254).to(torch.uint8)
    return torch.where(
        absolute_max == 0,
        torch.full_like(scale_bytes, UNIT_SCALE),
        scale_bytes,
    )


def _mxfp8_quantize_reference(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the dequantized block-32 MXFP8 values and their E8M0 scale bytes."""
    grouped = values.float().reshape(*values.shape[:-1], -1, GROUP)
    scale_bytes = _e8m0_scale_bytes(grouped.abs().amax(dim=-1))
    scale = torch.pow(2.0, (scale_bytes.int() - 127).float()).unsqueeze(-1)
    quantized = (grouped / scale).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(values.shape), scale_bytes


def _mxfp8_dequant_reference(values: torch.Tensor) -> torch.Tensor:
    """Reference the E4M3/E8M0 quantizer used by mixed MMA.

    Derived from the scale byte rather than from ``log2``, so it stays exact at
    the bottom of the E8M0 range where a float ``absolute_max / 448`` becomes
    subnormal.  Torch keeps subnormals, so this models the quantizer the kernel
    would run if the extension were not built with ``-ftz=true``.
    """
    dequantized, _ = _mxfp8_quantize_reference(values)
    return dequantized


def _published_latent(scratch: torch.Tensor, rows: int) -> torch.Tensor:
    """Dequantize the MXFP8 latent the quantization stage published."""
    codes = _region(scratch, "latent_mxfp8", torch.uint8)[
        : rows * HIDDEN
    ].view(rows, HIDDEN)
    scale_bytes = _region(scratch, "latent_scale", torch.uint8)[
        : rows * (HIDDEN // GROUP)
    ].view(rows, HIDDEN // GROUP)
    scale = torch.pow(2.0, (scale_bytes.int() - 127).float())
    values = codes.view(torch.float8_e4m3fn).float().view(rows, -1, GROUP)
    return (values * scale.unsqueeze(-1)).reshape(rows, HIDDEN)


def test_workspace_bytes_matches_extended_expert_scratch(
    device: torch.device,
) -> None:
    from mok import _C

    assert SCRATCH_BYTES == 3_749_376
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    for name, (offset, _) in SCRATCH_LAYOUT.items():
        if name != "total_bytes":
            assert offset % ALIGNMENT == 0, name
    assert SCRATCH_LAYOUT["latent_mxfp8"] == (40_448, 458_752)
    assert SCRATCH_LAYOUT["latent_scale"] == (499_200, 14_336)
    assert SCRATCH_LAYOUT["situ_mxfp8"] == (513_536, 786_432)
    assert SCRATCH_LAYOUT["situ_scale"] == (1_299_968, 24_576)
    assert SCRATCH_LAYOUT["routed_accumulator"] == (1_324_544, 1_835_008)
    assert SCRATCH_LAYOUT["shared_gate"] == (3_159_552, 196_608)
    assert SCRATCH_LAYOUT["shared_up"] == (3_356_160, 196_608)
    assert SCRATCH_LAYOUT["shared_activated"] == (3_552_768, 196_608)


def test_down_row_gain_tag_is_nonperiodic_and_exactly_representable(
    device: torch.device,
) -> None:
    """The output-tile tag must not repeat under any displacement it must catch.

    A tag with period ``p`` is blind to a displacement of ``p`` rows, so the
    fixture's discriminating power is only as good as the tag's aperiodicity.
    """
    gains = _down_row_gain(device).float()

    exponents = torch.log2(gains)
    assert torch.equal(exponents, exponents.round())
    assert float(gains.max()) == 1.0
    assert float(gains.min()) == 2.0 ** -(GAIN_BINADES - 1)
    assert int(gains.unique().numel()) == GAIN_BINADES
    assert torch.equal(gains, gains.bfloat16().float())

    # Every displacement the kernel could plausibly make, including the 128-wide
    # output tile and the 384-wide intermediate, must move most of the rows.
    for shift in (1, 2, 3, 5, 7, 32, 112, 128, 384, 1792):
        changed = int((torch.roll(gains, shift) != gains).sum())
        assert changed > HIDDEN // 3, (shift, changed)
    # The phase argument must give each expert a tag no other expert repeats.
    for phase in (1, 2, 3):
        changed = int((_down_row_gain(device, phase).float() != gains).sum())
        assert changed > HIDDEN // 3, (phase, changed)


def test_sm103_mixed_mxfp8_by_mxfp4_instruction_probe(
    device: torch.device,
) -> None:
    """Validate E4M3 x E2M1, E8M0, scale_vec::1X and logical K=32 alone."""
    from mok import _C
    from mok.ops import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    # Cross the eight-row boundary in the scale-factor swizzle so the probe
    # validates more than the first row atom.
    rows = 16
    a = torch.zeros(rows, GROUP, dtype=torch.bfloat16, device=device)
    code_points = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.bfloat16,
        device=device,
    )
    for row in range(rows):
        a[row] = code_points.repeat(4).roll(row) * (2.0 ** (row - 3))
    generator = torch.Generator(device=device).manual_seed(6001)
    b = (
        torch.randn(
            1, 128, GROUP, generator=generator, dtype=torch.float32, device=device
        )
        * 0.5
    ).bfloat16()
    b_packed, b_scale = pack_kimi_k3_mxfp4(b, GROUP)
    exact_b = dequant_kimi_k3_mxfp4(b_packed, b_scale, GROUP)[0].float()

    actual = _C._kimi_k3_mixed_mma_probe(a, b_packed[0], b_scale[0])
    expected = _mxfp8_dequant_reference(a) @ exact_b.T

    assert actual.shape == (rows, 128)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "exponent", [-125, -121, -100, -60, -49, -32, 0]
)
def test_mixed_mma_probe_spans_the_full_e8m0_activation_range(
    device: torch.device,
    exponent: int,
) -> None:
    """Activation scales must reach the E8M0 minimum, not an arbitrary floor.

    ``exponent == -121`` makes the selected scale byte exactly 0, the minimum
    ``2^-127`` that OCP MX v1.0 and PTX ISA 9.3 define; ``-125`` asks for a
    smaller scale than E8M0 has and must clamp onto that same boundary.  Row 0
    is left as an all-zero block.  Every A value is a power-of-two multiple of
    an E2M1 code point and B carries the compensating inverse scale, so the
    whole product is exactly representable and a correct kernel returns a
    nonzero result at every exponent.
    """
    from mok import _C
    from mok.ops import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    rows = 16
    code_points = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.bfloat16,
        device=device,
    )
    a = torch.zeros(rows, GROUP, dtype=torch.bfloat16, device=device)
    for row in range(1, rows):
        a[row] = code_points.repeat(4).roll(row) * (2.0**exponent)
    b = torch.zeros(1, PROBE_COLUMNS, GROUP, dtype=torch.bfloat16, device=device)
    for row in range(PROBE_COLUMNS):
        b[0, row] = code_points.repeat(4).roll(row + 3) * (
            2.0 ** (-exponent - 6)
        )
    b_packed, b_scale = pack_kimi_k3_mxfp4(b, GROUP)
    exact_b = dequant_kimi_k3_mxfp4(b_packed, b_scale, GROUP)[0].float()

    actual = _C._kimi_k3_mixed_mma_probe(a, b_packed[0], b_scale[0])
    expected = _mxfp8_dequant_reference(a) @ exact_b.T

    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    assert float(actual[1:].abs().max()) > 0.0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


# E2M1 code points to be scaled by 2^-126, the minimum BF16 normal.  0.5 is
# left out because 0.5 * 2^-126 is subnormal.
_MIN_NORMAL_POINTS = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# BF16 subnormals are exactly the integer multiples of 2^-133, so a subnormal
# probe needs integer code points.
_SUBNORMAL_POINTS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


def _bottom_of_range_probe(
    device: torch.device, unit: float, points: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Issue the mixed MMA with A scaled to ``unit`` and a compensating B.

    B carries ``2^120``, which keeps every product an ordinary normal float even
    though A sits at the bottom of the BF16 range, so nothing but A's own
    magnitude is under test.  Row 0 is left as an all-zero block.
    """
    from mok import _C
    from mok.ops import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    rows = 8
    lanes = torch.tensor(points, dtype=torch.bfloat16, device=device)
    lanes = lanes.repeat(GROUP // len(points) + 1)[:GROUP]
    a = torch.zeros(rows, GROUP, dtype=torch.bfloat16, device=device)
    for row in range(1, rows):
        a[row] = lanes.roll(row) * unit
    b = torch.zeros(1, PROBE_COLUMNS, GROUP, dtype=torch.bfloat16, device=device)
    for column in range(PROBE_COLUMNS):
        b[0, column] = lanes.roll(column + 3) * (2.0**120)
    b_packed, b_scale = pack_kimi_k3_mxfp4(b, GROUP)
    exact_b = dequant_kimi_k3_mxfp4(b_packed, b_scale, GROUP)[0].float()

    actual = _C._kimi_k3_mixed_mma_probe(a, b_packed[0], b_scale[0])
    return actual, _mxfp8_dequant_reference(a) @ exact_b.T


def test_mixed_mma_probe_reaches_the_bottom_of_the_bf16_activation_range(
    device: torch.device,
) -> None:
    """Drive the mixed instruction with the smallest activations it supports.

    ``2^-126``, the minimum BF16 normal, is that smallest magnitude: the
    extension is built with ``--use_fast_math``, which implies ``-ftz=true``, so
    a subnormal is flushed before the quantizer reads it.  At ``2^-126`` the
    instruction path is still exact and the result is far from zero.

    The subnormal case is then asserted to be exactly zero rather than left out.
    A flush is the documented consequence of the build's own flags, so the test
    pins it: a nonzero result there would mean the pipeline had produced
    something the contract cannot explain.  ``unflushed`` shows the same inputs
    are nonzero under the FTZ-free reference, so the zero is the flush and not
    an accidentally empty fixture.
    """
    actual, expected = _bottom_of_range_probe(device, 2.0**-126, _MIN_NORMAL_POINTS)

    assert float(expected.abs().max()) > 1.0
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    assert float(actual[1:].abs().max()) > 1.0
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    flushed, unflushed = _bottom_of_range_probe(
        device, 2.0**-133, _SUBNORMAL_POINTS
    )

    assert float(unflushed.abs().max()) > 0.0
    assert torch.equal(flushed, torch.zeros_like(flushed))


def test_activation_scale_bytes_reach_the_e8m0_minimum(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    """The published latent scale bytes must span down to byte 0, not stop early."""
    latent = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device=device)
    # One row per interesting binade: normal, deep below any practical floor,
    # exactly on the E8M0 minimum byte, and past it so the byte clamps to zero.
    for row, exponent in enumerate((0, -60, -119, -125)):
        latent[row] = 2.0**exponent
    _write_assignments(scratch, [(0, token, 0, 1.0) for token in range(4)])

    _call(latent, weights, torch.empty_like(latent), scratch, 4)

    published = _region(scratch, "latent_scale", torch.uint8)[
        : 4 * (HIDDEN // GROUP)
    ].view(4, HIDDEN // GROUP)
    _, expected = _mxfp8_quantize_reference(latent.float())
    assert torch.equal(published, expected)
    # 2^-119 lands on the minimum byte exactly and 2^-125 clamps onto it, while
    # 2^-60 must still pick a byte strictly below the 2^0 row's.
    assert int(published[2].max()) == 0
    assert int(published[3].max()) == 0
    assert int(published[1].max()) < int(published[0].min())


# Name, the constant fed to every latent lane, its BF16 bit pattern, the E8M0
# byte the kernel must publish, and whether the block survives as nonzero codes.
# For an exact power of two 2^k the byte is k + 119, clamped onto byte 0.
_SCALE_BOUNDARY_CASES = (
    ("zero", 0.0, 0x0000, UNIT_SCALE, False),
    ("min bf16 subnormal 2^-133", 2.0**-133, 0x0001, UNIT_SCALE, False),
    ("max bf16 subnormal 127*2^-133", 127 * 2.0**-133, 0x007F, UNIT_SCALE, False),
    ("min bf16 normal 2^-126", 2.0**-126, 0x0080, 0, True),
    ("2^-120", 2.0**-120, 0x0380, 0, True),
    ("2^-60", 2.0**-60, 0x2180, 59, True),
    ("2^-10", 2.0**-10, 0x3A80, 109, True),
    ("2^0", 1.0, 0x3F80, 119, True),
    ("2^8", 2.0**8, 0x4380, 127, True),
)


def test_published_latent_scale_bytes_cover_the_e8m0_boundaries(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    """Walk E8M0 selection from an all-zero block to the top of the BF16 range.

    An all-zero block must take the contract's unit byte, and every ordinary
    normal the mathematically expected ``k + 119`` for ``2^k``, clamped onto the
    minimum byte 0 at the bottom.  A BF16 subnormal cannot reach its own byte:
    the extension is built with ``--use_fast_math``, which implies
    ``-ftz=true``, so the magnitude is flushed before the quantizer reads it and
    the block becomes indistinguishable from an all-zero block.  That is pinned
    here rather than skipped, because it is what a caller actually gets.

    ``2^8`` selects the same byte as an all-zero block, since a scale of 1.0
    already keeps 256 inside E4M3's 448, so the byte alone cannot separate a
    live block from a dead one and every case also asserts whether the block
    survives as nonzero codes.
    """
    rows = len(_SCALE_BOUNDARY_CASES)
    latent = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    for row, (name, value, bits, _, _) in enumerate(_SCALE_BOUNDARY_CASES):
        latent[row] = torch.tensor(value, dtype=torch.bfloat16, device=device)
        assert int(latent[row, 0].view(torch.uint16)) == bits, name
    _write_assignments(scratch, [(0, token, 0, 1.0) for token in range(rows)])

    _call(latent, weights, torch.empty_like(latent), scratch, rows)

    published = _region(scratch, "latent_scale", torch.uint8)[
        : rows * (HIDDEN // GROUP)
    ].view(rows, HIDDEN // GROUP)
    codes = _region(scratch, "latent_mxfp8", torch.uint8)[
        : rows * HIDDEN
    ].view(rows, HIDDEN)
    for row, (name, _, _, byte, survives) in enumerate(_SCALE_BOUNDARY_CASES):
        assert torch.equal(
            published[row], torch.full_like(published[row], byte)
        ), (name, int(published[row].min()), int(published[row].max()))
        assert bool((codes[row] != 0).all()) == survives, name
        assert bool((codes[row] == 0).all()) != survives, name

    # The two subnormal rows must be indistinguishable from the all-zero row in
    # every published byte, which is the whole content of the flush contract.
    for row in (1, 2):
        assert torch.equal(published[row], published[0])
        assert torch.equal(codes[row], codes[0])

    # Every row that survives must match the reference quantizer bit for bit,
    # so the boundary bytes are not merely plausible but exactly right.
    surviving = torch.tensor(
        [row for row, case in enumerate(_SCALE_BOUNDARY_CASES) if case[4]],
        device=device,
    )
    expected_latent, expected_bytes = _mxfp8_quantize_reference(latent.float())
    assert torch.equal(
        published.index_select(0, surviving),
        expected_bytes.index_select(0, surviving),
    )
    assert torch.equal(
        _published_latent(scratch, rows).index_select(0, surviving),
        expected_latent.index_select(0, surviving),
    )


@pytest.mark.parametrize("rows", [1, 2, 8, 16])
def test_single_expert_matches_exact_prepared_weight_reference(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    rows: int,
) -> None:
    latent = _random_latent(device, rows, 6100 + rows)
    assignments = [(0, token, 0, 1.0) for token in range(rows)]
    _write_assignments(scratch, assignments)
    routed = torch.full_like(latent, float("nan"))

    actual = _call(latent, weights, routed, scratch, rows)
    expected = _reference(latent, weights, assignments, rows)

    assert actual.data_ptr() == routed.data_ptr()
    assert actual.shape == (rows, HIDDEN)
    assert actual.dtype == torch.bfloat16
    _assert_expert_close(actual, expected)


@pytest.mark.parametrize("active", CAPACITY_BUCKETS)
def test_every_active_capacity_bucket_matches_the_reference(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    active: int,
) -> None:
    """Cover every decode capacity bucket, including a full 128-row MMA tile."""
    latent = _random_latent(device, active, 6700 + active)
    assignments = [(0, token, 0, 1.0) for token in range(active)]
    _write_assignments(scratch, assignments)
    routed = torch.full_like(latent, float("nan"))

    actual = _call(latent, weights, routed, scratch, active)
    expected = _reference(latent, weights, assignments, active)

    assert actual.shape == (active, HIDDEN)
    _assert_expert_close(actual, expected)


def test_grouped_experts_apply_exact_situ_and_normalized_router_weights(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = _random_latent(device, 4, 6200)
    # Two experts per token, deliberately interleaved before expert-major sorting.
    assignments = [
        (token % 3, token, 0, 0.25)
        for token in range(4)
    ] + [
        ((token + 1) % 3, token, 1, 0.75)
        for token in range(4)
    ]
    _write_assignments(scratch, assignments)
    routed = torch.empty_like(latent)

    actual = _call(latent, weights, routed, scratch, 4)
    expected = _reference(latent, weights, assignments, 4)

    _assert_expert_close(actual, expected)


def test_2048_assignments_cover_all_896_experts_and_stay_finite(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    # A constant latent keeps an analytical reference for all 896 experts, which
    # a per-expert dequantized reference could not afford at this size.
    latent = torch.full(
        (MAX_TOKENS, HIDDEN), 0.25, dtype=torch.bfloat16, device=device
    )
    assignments = [
        ((token * TOPK + slot) % EXPERTS, token, slot, 1.0 / TOPK)
        for token in range(MAX_TOKENS)
        for slot in range(TOPK)
    ]
    assert len({item[0] for item in assignments}) == EXPERTS
    _write_assignments(scratch, assignments)
    counts = _region(scratch, "expert_counts", torch.int32)
    assert int(counts.sum()) == MAX_ASSIGNMENTS
    assert int((counts > 0).sum()) == EXPERTS
    routed = torch.empty_like(latent)

    actual = _call(latent, weights, routed, scratch, MAX_TOKENS)
    # Every gate row sees 0.25 and every up row 0.125, W2 row j sums all 384
    # SiTU columns at its own gain, and the sixteen normalized route weights
    # sum to one.
    gate = torch.tensor(0.25, dtype=torch.float32, device=device)
    expected_value = INTERMEDIATE * _situ(gate, gate * 0.5)
    expected = (
        expected_value * _down_row_gain(device).float()
    ).expand_as(actual).bfloat16().float()

    _assert_expert_close(actual, expected)


def test_empty_experts_are_skipped_and_cannot_change_output(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = _random_latent(device, 8, 6300)
    assignments = [(0, token, 0, 1.0) for token in range(8)]
    _write_assignments(scratch, assignments)
    first = _call(latent, weights, torch.empty_like(latent), scratch, 8).clone()
    assert float(first.float().abs().sum()) > 0.0

    # Expert 895 has count zero.  Poison every prepared byte without touching
    # the selected expert, then require bitwise-identical output.  The poison is
    # reverted so later tests still see the shared module-scoped shard.
    poisoned = (
        (weights.w1_packed, 0xFF),
        (weights.w1_scale, 0xFE),
        (weights.w3_packed, 0x77),
        (weights.w3_scale, 0x01),
        (weights.w2_packed, 0xAA),
        (weights.w2_scale, 0xFE),
    )
    saved = [tensor[895].clone() for tensor, _ in poisoned]
    try:
        for tensor, value in poisoned:
            tensor[895].fill_(value)
        second = _call(latent, weights, torch.empty_like(latent), scratch, 8)
        assert torch.equal(first, second)
    finally:
        for (tensor, _), original in zip(poisoned, saved):
            tensor[895].copy_(original)


def _install_expert_pattern(
    weights: ExpertWeights, expert: int, phase: int, device: torch.device
) -> None:
    """Give one expert a shard that no other expert can imitate.

    The gate/up rows are displaced by whole latent scale groups and the down
    rows carry a shifted gain phase, so reading the wrong expert changes both
    which latent columns are reduced and how each output tile is weighted.
    """
    from mok.ops import pack_kimi_k3_mxfp4

    rows = torch.arange(INTERMEDIATE, device=device)
    columns = (
        torch.tensor(
            [_latent_column_for_row(row) for row in range(INTERMEDIATE)],
            device=device,
        )
        + GROUP * phase
    ) % HIDDEN
    gate_dense = torch.zeros(
        1, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device
    )
    gate_dense[0, rows, columns] = 1.0
    up_dense = torch.zeros_like(gate_dense)
    up_dense[0, rows, columns] = 0.5
    down_dense = _down_row_gain(device, phase).view(1, HIDDEN, 1).expand(
        1, HIDDEN, INTERMEDIATE
    ).contiguous()

    for dense, packed, scale in (
        (gate_dense, weights.w1_packed, weights.w1_scale),
        (up_dense, weights.w3_packed, weights.w3_scale),
        (down_dense, weights.w2_packed, weights.w2_scale),
    ):
        expert_packed, expert_scale = pack_kimi_k3_mxfp4(dense, dense.size(-1))
        packed[expert].copy_(expert_packed[0])
        scale[expert].copy_(expert_scale[0])


def test_selected_expert_ids_are_addressed_exactly(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    """Route to a low, a middle, and the final expert with distinct shards.

    Each selected expert also differs from the shared shard every other expert
    carries, so reading expert 0, aliasing the expert stride onto a neighbour,
    reversing the two experts a token selected across its two slots, or
    misaddressing the final expert all change the result.  Only the selected
    slices are dequantized.
    """
    tensors = (
        weights.w1_packed,
        weights.w1_scale,
        weights.w3_packed,
        weights.w3_scale,
        weights.w2_packed,
        weights.w2_scale,
    )
    saved = {
        expert: tuple(tensor[expert].clone() for tensor in tensors)
        for expert in ADDRESS_EXPERTS
    }
    try:
        for phase, expert in enumerate(ADDRESS_EXPERTS, start=1):
            _install_expert_pattern(weights, expert, phase, device)

        latent = _random_latent(device, 6, 7100)
        first_weight, second_weight = ADDRESS_WEIGHTS
        slot_experts = [
            (ADDRESS_EXPERTS[token % 3], ADDRESS_EXPERTS[(token + 1) % 3])
            for token in range(6)
        ]
        assignments: list[Assignment] = []
        for token, (first, second) in enumerate(slot_experts):
            assignments.append((first, token, 0, first_weight))
            assignments.append((second, token, 1, second_weight))
        _write_assignments(scratch, assignments)

        actual = _call(latent, weights, torch.empty_like(latent), scratch, 6)
        expected = _reference(latent, weights, assignments, 6)
        _assert_expert_close(actual, expected)

        # Prove the shards discriminate: every addressing bug this test is meant
        # to catch must move the reference well past the max-abs tolerance.
        collapsed = [(0, token, slot, weight)
                     for _, token, slot, weight in assignments]
        rotated = [
            (ADDRESS_EXPERTS[(ADDRESS_EXPERTS.index(expert) + 1) % 3],
             token, slot, weight)
            for expert, token, slot, weight in assignments
        ]
        neighbour = [(expert - 1, token, slot, weight)
                     for expert, token, slot, weight in assignments]
        # A true swap: each token keeps its own two experts, its slot positions,
        # and its route weights, and only the pairing between them is reversed.
        # Nothing but reading the assignment's own expert id gets this right.
        reversed_slots: list[Assignment] = []
        for token, (first, second) in enumerate(slot_experts):
            reversed_slots.append((second, token, 0, first_weight))
            reversed_slots.append((first, token, 1, second_weight))
        for wrong in (collapsed, rotated, neighbour, reversed_slots):
            deviation = _reference(latent, weights, wrong, 6) - expected
            assert float(deviation.abs().max()) > 1.0
    finally:
        for expert, originals in saved.items():
            for tensor, original in zip(tensors, originals):
                tensor[expert].copy_(original)


def test_active_token_mask_zeros_inactive_output_rows(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:3, 0] = 1.0
    assignments = [(0, token, 0, 1.0) for token in range(3)]
    _write_assignments(scratch, assignments)
    routed = torch.full_like(latent, float("nan"))

    active = _call(latent, weights, routed, scratch, 3)

    assert active.shape == (3, HIDDEN)
    assert torch.isfinite(active.float()).all()
    assert torch.equal(routed[3:], torch.zeros_like(routed[3:]))


def test_reused_scratch_resets_accumulator_and_generation_counters(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = _random_latent(device, 8, 6400)
    generations: list[torch.Tensor] = []
    for rows in (8, 2, 8):
        assignments = [(0, token, 0, 1.0) for token in range(rows)]
        _write_assignments(scratch, assignments)
        routed = torch.full_like(latent, 123.0)
        actual = _call(latent, weights, routed, scratch, rows)
        expected = _reference(latent, weights, assignments, rows)
        _assert_expert_close(actual, expected)
        assert torch.equal(routed[rows:], torch.zeros_like(routed[rows:]))
        generations.append(_region(scratch, "phase", torch.int32).clone())

    # Quantization and expert completion generations each advance once per call.
    assert int(generations[1][5] - generations[0][5]) == 1
    assert int(generations[2][5] - generations[1][5]) == 1
    assert int(generations[1][8] - generations[0][8]) == 1
    assert int(generations[2][8] - generations[1][8]) == 1


def test_replayed_generations_publish_fresh_quantized_and_routed_state(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    """Consume both published stages on every replay of one reused workspace.

    Each replay uses a different latent and a different row count, so any stale
    MXFP8 block or stale routed row left over from an earlier generation is a
    value mismatch rather than a silently plausible result.
    """
    replays = ((1, 7000), (8, 7001), (3, 7002), (16, 7003), (2, 7004), (8, 7005))
    quantization: list[int] = []
    completion: list[int] = []

    for rows, seed in replays:
        latent = _random_latent(device, rows, seed)
        assignments = [(0, token, 0, 1.0) for token in range(rows)]
        _write_assignments(scratch, assignments)
        routed = torch.full_like(latent, float("nan"))

        actual = _call(latent, weights, routed, scratch, rows)

        # Stage one: the published MXFP8 latent and its E8M0 scale bytes.
        expected_latent, expected_scales = _mxfp8_quantize_reference(
            latent.float()
        )
        published_scales = _region(scratch, "latent_scale", torch.uint8)[
            : rows * (HIDDEN // GROUP)
        ].view(rows, HIDDEN // GROUP)
        assert torch.equal(published_scales, expected_scales)
        assert torch.equal(_published_latent(scratch, rows), expected_latent)

        # Stage two: the routed output for this generation.
        _assert_expert_close(actual, _reference(latent, weights, assignments, rows))

        phase = _region(scratch, "phase", torch.int32)
        assert int(phase[4]) == 0, "quantization arrivals must be reset"
        assert int(phase[7]) == 0, "completion arrivals must be reset"
        quantization.append(int(phase[5]))
        completion.append(int(phase[8]))

    assert quantization == [quantization[0] + step for step in range(len(replays))]
    assert completion == [completion[0] + step for step in range(len(replays))]


def test_expert_stage_uses_the_tensor_devices_current_stream(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    source = _random_latent(device, 8, 6500)
    assignments = [(0, token, 0, 1.0) for token in range(8)]
    _write_assignments(scratch, assignments)
    side_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side_stream):
        latent = torch.zeros_like(source)
        routed = torch.empty_like(source)
        torch.cuda._sleep(1 << 28)
        latent.copy_(source)
        actual = _call(latent, weights, routed, scratch, 8)
    side_stream.synchronize()

    expected = _reference(source, weights, assignments, 8)
    _assert_expert_close(actual, expected)


def test_expert_stage_on_peer_device_ignores_current_device(
    device: torch.device,
    peer_device: torch.device,
) -> None:
    peer_weights = _make_structured_weights(peer_device)
    peer_scratch = torch.zeros(
        SCRATCH_BYTES, dtype=torch.uint8, device=peer_device
    )
    latent = _random_latent(peer_device, 2, 6600)
    assignments = [(0, token, 0, 1.0) for token in range(2)]
    _write_assignments(peer_scratch, assignments)
    torch.cuda.set_device(device)

    actual = _call(
        latent, peer_weights, torch.empty_like(latent), peer_scratch, 2
    )
    torch.cuda.synchronize(peer_device)

    assert torch.cuda.current_device() == device.index
    assert actual.device == peer_device
    expected = _reference(latent, peer_weights, assignments, 2)
    _assert_expert_close(actual, expected)
    del peer_weights, peer_scratch
    torch.cuda.empty_cache()


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
