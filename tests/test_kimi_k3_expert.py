"""What the Kimi K3 mixed MXFP8-by-MXFP4 expert instruction and stage compute.

Two claims, in the order they have to hold. First that the `kind::mxf8f6f4`
instruction itself does what the layout assumes over the whole representable
range -- both ends of E8M0, the bottom of the BF16 activation range, and the
E2M1 code points. Then that the stage built on it matches an exact prepared
weight reference at every capacity bucket, over every routing shape the router
can produce, with exact SiTU and normalized router weights.

Which expert a routing reaches is ``test_kimi_k3_expert_addressing.py``; the
host boundary is ``test_kimi_k3_expert_contract.py``. Everything all three rest
on is in ``kimi_k3_expert_support.py``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
import pytest
import torch

from .kimi_k3_expert_support import (
    ALIGNMENT,
    _assert_expert_close,
    _call,
    CAPACITY_BUCKETS,
    device,
    _down_row_gain,
    EXPERTS,
    ExpertWeights,
    GAIN_BINADES,
    GROUP,
    HIDDEN,
    INTERMEDIATE,
    MAX_ASSIGNMENTS,
    MAX_TOKENS,
    _mxfp8_dequant_reference,
    _mxfp8_quantize_reference,
    PROBE_COLUMNS,
    _published_latent,
    _random_latent,
    _reference,
    _region,
    scratch,
    SCRATCH_BYTES,
    SCRATCH_LAYOUT,
    _situ,
    TOPK,
    UNIT_SCALE,
    weights,
    _write_assignments,
)


def test_workspace_bytes_matches_extended_expert_scratch(
    device: torch.device,
) -> None:
    from mok import _C

    assert SCRATCH_BYTES == 8_111_872
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    for name, (offset, _) in SCRATCH_LAYOUT.items():
        if name != "total_bytes":
            assert offset % ALIGNMENT == 0, name
    assert SCRATCH_LAYOUT["latent_mxfp8"] == (40_704, 458_752)
    assert SCRATCH_LAYOUT["latent_scale"] == (499_456, 14_336)
    assert SCRATCH_LAYOUT["situ_mxfp8"] == (513_792, 786_432)
    assert SCRATCH_LAYOUT["situ_scale"] == (1_300_224, 24_576)
    assert SCRATCH_LAYOUT["routed_accumulator"] == (1_324_800, 3_670_016)
    assert SCRATCH_LAYOUT["shared_gate"] == (4_994_816, 196_608)
    assert SCRATCH_LAYOUT["shared_up"] == (5_191_424, 196_608)
    assert SCRATCH_LAYOUT["shared_activated"] == (5_388_032, 196_608)
    assert SCRATCH_LAYOUT["tail_normalized"] == (5_584_640, 917_504)
    assert SCRATCH_LAYOUT["tail_shared_shard"] == (6_502_144, 229_376)
    assert SCRATCH_LAYOUT["schedule"] == (8_111_360, 512)


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


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_transposed_m128x8_probe_matches_the_current_expert_unit(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    rows: int,
) -> None:
    extension = importlib.import_module("mok._C")
    latent = _random_latent(device, rows, 7300 + rows)
    baseline = torch.empty_like(latent)
    candidate = torch.empty_like(latent)
    arguments = (
        latent,
        weights.w1_packed,
        weights.w1_scale,
        weights.w3_packed,
        weights.w3_scale,
        weights.w2_packed,
        weights.w2_scale,
    )

    extension._kimi_k3_batched_expert_probe(
        *arguments, baseline, scratch, 0, False
    )
    extension._kimi_k3_batched_expert_probe(
        *arguments, candidate, scratch, 0, True
    )

    assignments = [(0, token, 0, 1.0) for token in range(rows)]
    expected = _reference(latent, weights, assignments, rows)
    _assert_expert_close(baseline, expected)
    _assert_expert_close(candidate, expected)
    torch.testing.assert_close(candidate, baseline, rtol=0.0, atol=0.0)


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
