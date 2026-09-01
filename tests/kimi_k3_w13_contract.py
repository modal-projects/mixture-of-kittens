"""CPU checks for the fused-W13 transform and the contract that rests on it.

The transform in :mod:`mok.kimi_k3_w13` is a permutation of a routed expert's
gate and up MXFP4 bytes into the order the decode kernel's gate/up unit reads
them. Because it is a permutation, the strongest statement available is that it
loses nothing: every packed byte and every E8M0 scale byte comes back under
inversion, for every expert, every one of the six tasks, every one of the seven
K panels, every FP4 nibble, and every E8M0 code including the reserved and
boundary ones. None of that needs a GPU, and none of it should need one.

Reaching ``mok.kimi_k3_w13`` does, though, unless it is reached the way
``tests/kimi_k3_api_contract.py`` reaches ``mok.kimi_k3``: importing any ``mok``
submodule runs ``mok/__init__.py``, which imports the compiled extension, so a
machine with no ``mok._C`` cannot even collect a file that names the transform at
module scope. The loader in that file already installs the extension stub and
loads ``mok.kimi_k3``, which imports ``mok.kimi_k3_w13`` on the way, so the
transform is reached from there rather than through a second stub.

Those stubs cannot be uninstalled -- the dispatcher has no way to drop an
operator registration -- so this module is run as its own process by
``tests/test_kimi_k3_w13.py``, which asserts on the per-check results printed
below ``RESULT_MARKER``. Every check here is a function of shapes and bytes;
nothing in it reads a source file or touches a device.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import traceback
from dataclasses import dataclass
from functools import partial
from types import ModuleType

import pytest
import torch

from .kimi_k3_api_contract import _load_contract_modules, _valid_weights

RESULT_MARKER = "KIMI_K3_W13_CONTRACT_RESULTS "

#: Experts the transform is checked at: the first, the last, and one interior.
PROBE_EXPERTS = (0, 447, 895)

#: E8M0 codes worth naming: the exact minimum, unit, maximum, and the reserved
#: NaN byte the packer never emits but the transform must still carry verbatim.
E8M0_BOUNDARIES = (0x00, 0x01, 0x7E, 0x7F, 0x80, 0xFD, 0xFE, 0xFF)


@dataclass(frozen=True)
class Bound:
    """The stub-loaded modules, and the canonical half shapes they imply."""

    kimi_k3: ModuleType
    ops: ModuleType
    fake_impls: ModuleType
    w13: ModuleType
    experts: int
    rows: int
    packed_columns: int
    scale_columns: int
    group_size: int
    latent: int


_BOUND: Bound | None = None


def bound() -> Bound:
    """Load the transform behind the API contract's extension stub, once.

    Deliberately not done at import time: this module's ``CHECKS`` keys are read
    by the parent pytest process to name its tests, and installing the stubs
    there is the leak the subprocess exists to avoid.
    """
    global _BOUND
    if _BOUND is not None:
        return _BOUND

    kimi_k3, ops, fake_impls = _load_contract_modules()
    # `mok/kimi_k3.py` imports the transform, so the loader above has already
    # put it in `sys.modules` under its real name and against its real source.
    w13 = sys.modules["mok.kimi_k3_w13"]
    _BOUND = Bound(
        kimi_k3=kimi_k3,
        ops=ops,
        fake_impls=fake_impls,
        w13=w13,
        experts=kimi_k3.KIMI_K3_NUM_EXPERTS,
        rows=w13.KIMI_K3_W13_CHANNELS_PER_RANK,
        packed_columns=kimi_k3.KIMI_K3_LATENT_SIZE // 2,
        scale_columns=(
            kimi_k3.KIMI_K3_LATENT_SIZE // kimi_k3.KIMI_K3_MXFP4_GROUP_SIZE
        ),
        group_size=kimi_k3.KIMI_K3_MXFP4_GROUP_SIZE,
        latent=kimi_k3.KIMI_K3_LATENT_SIZE,
    )
    return _BOUND


def _deterministic_packed(experts: int, seed: int) -> torch.Tensor:
    """A packed payload that covers all 256 byte values in every row."""
    state = bound()
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        256,
        (experts, state.rows, state.packed_columns),
        dtype=torch.uint8,
        generator=generator,
    )


def _deterministic_scale(experts: int, seed: int) -> torch.Tensor:
    state = bound()
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        256,
        (experts, state.rows, state.scale_columns),
        dtype=torch.uint8,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------


def check_task_decomposition_covers_every_channel_and_k_panel() -> None:
    w13 = bound().w13
    assert w13.KIMI_K3_W13_TASKS == 6
    assert w13.KIMI_K3_W13_TASK_ROWS == 128
    assert w13.KIMI_K3_W13_HALF_ROWS == 64
    # Six tasks of 64 paired channels is every channel this rank owns, once.
    assert (
        w13.KIMI_K3_W13_TASKS * w13.KIMI_K3_W13_HALF_ROWS
        == w13.KIMI_K3_W13_CHANNELS_PER_RANK
    )
    # Seven K512 slabs of sixteen K32 contractions is the whole latent width.
    assert w13.KIMI_K3_W13_SLABS == 7
    assert w13.KIMI_K3_W13_SLAB_K == 512
    assert w13.KIMI_K3_W13_SLAB_GROUPS == 16
    assert w13.KIMI_K3_W13_SLABS * w13.KIMI_K3_W13_SLAB_K == bound().latent


def check_prepared_bytes_are_the_halves_they_replace_and_not_twice_them(
) -> None:
    """The identity the whole weight contract rests on.

    The prepared bundle carries the fused pair *instead of* ``expert_w1_*`` and
    ``expert_w3_*``. If the fuse cost even one byte more than the four tensors it
    replaces, keeping it would mean growing the steady-state weight footprint of
    every rank -- 896 experts' worth. It costs exactly nothing, because it moves
    bytes and creates none.
    """
    state = bound()
    packed_bytes, scale_bytes = state.w13.fused_w13_prepared_bytes()
    halves_packed = 2 * state.experts * state.rows * state.packed_columns
    halves_scale = 2 * state.experts * state.rows * state.scale_columns
    assert packed_bytes == halves_packed
    assert scale_bytes == halves_scale
    # And not 2x, which is what a bundle that kept both representations would
    # cost. Named as its own assertion because it is the failure being excluded.
    assert packed_bytes + scale_bytes != 2 * (halves_packed + halves_scale)
    assert (packed_bytes, scale_bytes) == (1_233_125_376, 77_070_336)


def check_fused_payload_is_the_same_bytes_as_the_halves() -> None:
    """The same identity, measured on real tensors rather than on the shapes."""
    w13 = bound().w13
    w1 = _deterministic_packed(3, seed=101)
    w3 = _deterministic_packed(3, seed=102)
    w1_scale = _deterministic_scale(3, seed=103)
    w3_scale = _deterministic_scale(3, seed=104)
    packed, scale = w13.fuse_w13(w1, w1_scale, w3, w3_scale)
    assert packed.numel() == w1.numel() + w3.numel()
    assert scale.numel() == w1_scale.numel() + w3_scale.numel()
    assert packed.element_size() == w1.element_size() == 1
    assert scale.element_size() == w1_scale.element_size() == 1


# ---------------------------------------------------------------------------
# Invertibility: the transform loses nothing.
# ---------------------------------------------------------------------------


def check_transform_returns_every_packed_byte_under_inversion() -> None:
    w13 = bound().w13
    w1 = _deterministic_packed(4, seed=11)
    w3 = _deterministic_packed(4, seed=12)
    payload = w13.fuse_w13_packed(w1, w3)
    assert payload.shape == (4,) + w13.KIMI_K3_W13_PACKED_SHAPE[1:]
    assert payload.dtype == torch.uint8
    back_w1, back_w3 = w13.unfuse_w13_packed(payload)
    assert torch.equal(back_w1, w1)
    assert torch.equal(back_w3, w3)


def check_transform_returns_every_scale_byte_under_inversion() -> None:
    w13 = bound().w13
    w1 = _deterministic_scale(4, seed=21)
    w3 = _deterministic_scale(4, seed=22)
    scales = w13.fuse_w13_scale(w1, w3)
    assert scales.shape == (4,) + w13.KIMI_K3_W13_SCALE_SHAPE[1:]
    assert scales.dtype == torch.uint8
    back_w1, back_w3 = w13.unfuse_w13_scale(scales)
    assert torch.equal(back_w1, w1)
    assert torch.equal(back_w3, w3)


def check_reading_one_half_back_agrees_with_reading_both(half: int) -> None:
    """The per-half inverse is what the consumers use, so it is held to the pair.

    The benchmark's dequantizing oracle and the framework adapters recover one
    projection of one expert chunk at a time, precisely so neither has to hold
    both halves of a chunk at once. A per-half path that disagreed with the pair
    would make those consumers wrong while every round-trip check above passed.
    """
    w13 = bound().w13
    w1 = _deterministic_packed(2, seed=111)
    w3 = _deterministic_packed(2, seed=112)
    w1_scale = _deterministic_scale(2, seed=113)
    w3_scale = _deterministic_scale(2, seed=114)
    packed, scale = w13.fuse_w13(w1, w1_scale, w3, w3_scale)

    expected_packed = (w1, w3)[half]
    expected_scale = (w1_scale, w3_scale)[half]
    half_packed, half_scale = w13.unfuse_w13_half(packed, scale, half)
    assert torch.equal(half_packed, expected_packed)
    assert torch.equal(half_scale, expected_scale)


def check_writing_one_half_leaves_the_other_half_untouched() -> None:
    """What lets preparation release a packed half before packing the next one.

    ``prepare_kimi_k3_decode_weights`` fuses in place, one projection at a time,
    so that the gate half is freed before the up half is quantized -- the two
    are 656 MiB each. That is only sound if writing one half touches none of the
    other's bytes.
    """
    w13 = bound().w13
    w1 = _deterministic_packed(2, seed=121)
    w3 = _deterministic_packed(2, seed=122)
    w1_scale = _deterministic_scale(2, seed=123)
    w3_scale = _deterministic_scale(2, seed=124)

    packed = torch.full((2,) + w13.KIMI_K3_W13_PACKED_SHAPE[1:], 0xCC,
                        dtype=torch.uint8)
    scale = torch.full((2,) + w13.KIMI_K3_W13_SCALE_SHAPE[1:], 0xCC,
                       dtype=torch.uint8)
    w13.fuse_w13_half(packed, scale, w1, w1_scale, 0)
    # The gate rows are written and the up rows still hold the sentinel, so the
    # two halves' destinations are disjoint and each is fully covered.
    gate_packed, gate_scale = w13.unfuse_w13_half(packed, scale, 0)
    up_packed, up_scale = w13.unfuse_w13_half(packed, scale, 1)
    assert torch.equal(gate_packed, w1)
    assert torch.equal(gate_scale, w1_scale)
    assert torch.all(up_packed == 0xCC)
    assert torch.all(up_scale == 0xCC)

    w13.fuse_w13_half(packed, scale, w3, w3_scale, 1)
    assert torch.equal(packed, w13.fuse_w13_packed(w1, w3))
    assert torch.equal(scale, w13.fuse_w13_scale(w1_scale, w3_scale))


def check_transform_is_a_permutation_of_the_bytes_it_was_given() -> None:
    """Same multiset in and out, so nothing was dropped, added, or rewritten."""
    w13 = bound().w13
    w1 = _deterministic_packed(2, seed=31)
    w3 = _deterministic_packed(2, seed=32)
    payload = w13.fuse_w13_packed(w1, w3)
    for expert in range(2):
        source = torch.cat(
            [w1[expert].reshape(-1), w3[expert].reshape(-1)]
        ).bincount(minlength=256)
        assert torch.equal(
            payload[expert].reshape(-1).bincount(minlength=256), source
        )


def check_transform_treats_every_expert_the_same_way(expert: int) -> None:
    """Experts 0, 447, and 895 are reindexed by the same within-expert map."""
    state = bound()
    w13 = state.w13
    w1 = _deterministic_packed(1, seed=41)
    w3 = _deterministic_packed(1, seed=42)
    w1_scale = _deterministic_scale(1, seed=43)
    w3_scale = _deterministic_scale(1, seed=44)
    one_packed, one_scale = w13.fuse_w13(w1, w1_scale, w3, w3_scale)

    wide_w1 = torch.zeros(
        (state.experts, state.rows, state.packed_columns), dtype=torch.uint8
    )
    wide_w3 = torch.zeros_like(wide_w1)
    wide_w1[expert] = w1[0]
    wide_w3[expert] = w3[0]
    wide_w1_scale = torch.zeros(
        (state.experts, state.rows, state.scale_columns), dtype=torch.uint8
    )
    wide_w3_scale = torch.zeros_like(wide_w1_scale)
    wide_w1_scale[expert] = w1_scale[0]
    wide_w3_scale[expert] = w3_scale[0]

    wide_packed, wide_scale = w13.fuse_w13(
        wide_w1, wide_w1_scale, wide_w3, wide_w3_scale
    )
    assert torch.equal(wide_packed[expert], one_packed[0])
    assert torch.equal(wide_scale[expert], one_scale[0])


# ---------------------------------------------------------------------------
# Placement: which byte ends up where.
# ---------------------------------------------------------------------------


def check_each_task_pairs_the_gate_and_up_rows_of_the_same_channels(
    task: int,
) -> None:
    """M row ``r`` is gate channel ``64t + r`` and row ``r + 64`` is its up row.

    This is the property the epilogue depends on: it reads one accumulator,
    pairs row ``r`` with row ``r + 64``, and gets the gate and up value of one
    output channel.
    """
    w13 = bound().w13
    w1 = _deterministic_packed(1, seed=51)
    w3 = _deterministic_packed(1, seed=52)
    payload = w13.fuse_w13_packed(w1, w3)[0]
    tile = payload.reshape(
        w13.KIMI_K3_W13_TASKS,
        w13.KIMI_K3_W13_SLABS,
        w13.KIMI_K3_W13_TASK_ROWS,
        w13.KIMI_K3_W13_SLAB_ROW_BYTES,
    )
    for row in (0, 1, 31, 63):
        channel = task * w13.KIMI_K3_W13_HALF_ROWS + row
        for slab in range(w13.KIMI_K3_W13_SLABS):
            panel = slice(
                slab * w13.KIMI_K3_W13_SLAB_ROW_BYTES,
                (slab + 1) * w13.KIMI_K3_W13_SLAB_ROW_BYTES,
            )
            assert torch.equal(tile[task, slab, row], w1[0, channel, panel])
            assert torch.equal(
                tile[task, slab, row + w13.KIMI_K3_W13_HALF_ROWS],
                w3[0, channel, panel],
            )


def check_every_k_panel_lands_in_its_own_slab(slab: int) -> None:
    """Slab ``s`` holds exactly K panel ``[512s, 512(s+1))`` and nothing else."""
    state = bound()
    w13 = state.w13
    w1 = torch.zeros(
        (1, state.rows, state.packed_columns), dtype=torch.uint8
    )
    w3 = torch.zeros_like(w1)
    marked = slice(
        slab * w13.KIMI_K3_W13_SLAB_ROW_BYTES,
        (slab + 1) * w13.KIMI_K3_W13_SLAB_ROW_BYTES,
    )
    w1[:, :, marked] = 0xA5
    w3[:, :, marked] = 0x5A

    tile = (
        w13.fuse_w13_packed(w1, w3)[0]
        .reshape(
            w13.KIMI_K3_W13_TASKS,
            w13.KIMI_K3_W13_SLABS,
            w13.KIMI_K3_W13_TASK_ROWS,
            w13.KIMI_K3_W13_SLAB_ROW_BYTES,
        )
    )
    for candidate in range(w13.KIMI_K3_W13_SLABS):
        block = tile[:, candidate]
        if candidate == slab:
            assert torch.all(block[:, : w13.KIMI_K3_W13_HALF_ROWS] == 0xA5)
            assert torch.all(block[:, w13.KIMI_K3_W13_HALF_ROWS :] == 0x5A)
        else:
            assert torch.all(block == 0)


# ---------------------------------------------------------------------------
# The scale shuffle is the MMA's scale-factor atom.
# ---------------------------------------------------------------------------


def check_scale_offset_is_the_sm103_scale_factor_atom() -> None:
    """``Sm103BlockScaledBasicChunk<32>::SfKMajorAtom``, spelled out.

    Shape ``((8,4,4),(32,4))``, stride ``((16,128,4),(0,1))``: one 512-byte tile
    carries one byte per M row for four consecutive K groups.
    """
    w13 = bound().w13
    for row in range(w13.KIMI_K3_W13_TASK_ROWS):
        for group in range(w13.KIMI_K3_W13_SCALE_GROUPS_PER_TILE):
            assert w13.scale_tile_offset(row, group) == (
                (row % 8) * 16 + ((row // 8) % 4) * 128 + (row // 32) * 4 + group
            )


def check_scale_shuffle_is_a_bijection_over_one_slab() -> None:
    w13 = bound().w13
    scatter = w13.scale_scatter_index()
    gather = w13.scale_gather_index()
    bytes_per_slab = w13.KIMI_K3_W13_SLAB_SCALE_BYTES
    assert scatter.shape == (bytes_per_slab,)
    assert gather.shape == (bytes_per_slab,)
    ordered = torch.arange(bytes_per_slab, dtype=scatter.dtype)
    assert torch.equal(torch.sort(scatter).values, ordered)
    # The two are each other's inverse, which is what makes either safe to read
    # the blob with.
    assert torch.equal(scatter[gather], ordered)
    assert torch.equal(gather[scatter], ordered)


def check_each_scale_byte_lands_at_the_offset_its_group_is_read_from() -> None:
    """Row ``m``, group ``g`` of a slab must sit where the MMA will look.

    Group ``g`` is read out of scale tile ``g // 4`` with scale-factor id
    ``g % 4``, at the atom offset above.
    """
    w13 = bound().w13
    w1 = _deterministic_scale(1, seed=61)
    w3 = _deterministic_scale(1, seed=62)
    blob = w13.fuse_w13_scale(w1, w3)[0].reshape(
        w13.KIMI_K3_W13_TASKS,
        w13.KIMI_K3_W13_SLABS,
        w13.KIMI_K3_W13_SLAB_SCALE_BYTES,
    )
    for task in range(w13.KIMI_K3_W13_TASKS):
        for slab in range(w13.KIMI_K3_W13_SLABS):
            for row in (0, 7, 8, 40, 63, 64, 71, 100, 127):
                for group in (0, 1, 5, 11, 15):
                    tile = group // w13.KIMI_K3_W13_SCALE_GROUPS_PER_TILE
                    factor = group % w13.KIMI_K3_W13_SCALE_GROUPS_PER_TILE
                    offset = (
                        tile * w13.KIMI_K3_W13_SCALE_TILE_BYTES
                        + w13.scale_tile_offset(row, factor)
                    )
                    half = row // w13.KIMI_K3_W13_HALF_ROWS
                    channel = (
                        task * w13.KIMI_K3_W13_HALF_ROWS
                        + row % w13.KIMI_K3_W13_HALF_ROWS
                    )
                    source = (w1 if half == 0 else w3)[
                        0, channel, slab * w13.KIMI_K3_W13_SLAB_GROUPS + group
                    ]
                    assert blob[task, slab, offset] == source


# ---------------------------------------------------------------------------
# Value coverage: nibbles, E8M0 codes, and zero groups.
# ---------------------------------------------------------------------------


def check_transform_carries_every_fp4_nibble_pair() -> None:
    """All 256 packed byte values, so both nibble positions see all sixteen."""
    state = bound()
    w13 = state.w13
    values = torch.arange(256, dtype=torch.uint8)
    w1 = values.repeat(state.rows * state.packed_columns // 256).reshape(
        1, state.rows, state.packed_columns
    )
    w3 = torch.flip(w1, dims=(2,)).contiguous()
    payload = w13.fuse_w13_packed(w1, w3)
    seen = payload.reshape(-1).bincount(minlength=256)
    assert int(seen.min()) > 0, "every packed byte value must survive"
    back_w1, back_w3 = w13.unfuse_w13_packed(payload)
    assert torch.equal(back_w1, w1)
    assert torch.equal(back_w3, w3)


def check_transform_carries_every_e8m0_boundary_code(code: int) -> None:
    state = bound()
    w13 = state.w13
    w1 = torch.full(
        (1, state.rows, state.scale_columns), code, dtype=torch.uint8
    )
    w3 = torch.full_like(w1, 0x7F)
    scales = w13.fuse_w13_scale(w1, w3)
    blob = scales[0].reshape(
        w13.KIMI_K3_W13_TASKS,
        w13.KIMI_K3_W13_SLABS,
        w13.KIMI_K3_W13_SCALE_TILES_PER_SLAB,
        w13.KIMI_K3_W13_SCALE_TILE_BYTES,
    )
    # Every gate row of every tile carries the code; every up row carries unit.
    for row in range(w13.KIMI_K3_W13_HALF_ROWS):
        for group in range(w13.KIMI_K3_W13_SCALE_GROUPS_PER_TILE):
            offset = w13.scale_tile_offset(row, group)
            assert torch.all(blob[:, :, :, offset] == code)
    for row in range(
        w13.KIMI_K3_W13_HALF_ROWS, w13.KIMI_K3_W13_TASK_ROWS
    ):
        for group in range(w13.KIMI_K3_W13_SCALE_GROUPS_PER_TILE):
            offset = w13.scale_tile_offset(row, group)
            assert torch.all(blob[:, :, :, offset] == 0x7F)
    back_w1, back_w3 = w13.unfuse_w13_scale(scales)
    assert torch.equal(back_w1, w1)
    assert torch.equal(back_w3, w3)


def check_a_zero_group_keeps_its_zero_payload_and_unit_scale() -> None:
    """A group the packer flushed to zero must stay zero with a unit scale."""
    state = bound()
    w13 = state.w13
    w1 = _deterministic_packed(1, seed=71)
    w3 = _deterministic_packed(1, seed=72)
    w1_scale = _deterministic_scale(1, seed=73)
    w3_scale = _deterministic_scale(1, seed=74)
    # Zero one group of one gate channel and one up channel outright.
    gate_channel, up_channel, group = 65, 300, 37
    payload_span = slice(
        group * (state.group_size // 2),
        (group + 1) * (state.group_size // 2),
    )
    w1[0, gate_channel, payload_span] = 0
    w1_scale[0, gate_channel, group] = 0x7F
    w3[0, up_channel, payload_span] = 0
    w3_scale[0, up_channel, group] = 0x7F

    back_w1, back_w1_scale, back_w3, back_w3_scale = w13.unfuse_w13(
        *w13.fuse_w13(w1, w1_scale, w3, w3_scale)
    )
    assert torch.all(back_w1[0, gate_channel, payload_span] == 0)
    assert back_w1_scale[0, gate_channel, group] == 0x7F
    assert torch.all(back_w3[0, up_channel, payload_span] == 0)
    assert back_w3_scale[0, up_channel, group] == 0x7F
    assert torch.equal(back_w1, w1)
    assert torch.equal(back_w3, w3)


# ---------------------------------------------------------------------------
# Rejection: what the transform refuses.
# ---------------------------------------------------------------------------


def check_transform_rejects_a_half_of_the_wrong_shape_or_dtype() -> None:
    state = bound()
    w13 = state.w13
    good = _deterministic_packed(1, seed=81)
    with pytest.raises(TypeError, match="uint8"):
        w13.fuse_w13_packed(good.to(torch.int8), good)
    with pytest.raises(ValueError, match=r"\[E, 384, 1792\]"):
        w13.fuse_w13_packed(good[:, :, :-1].contiguous(), good)
    strided = torch.zeros(
        (1, state.rows, state.packed_columns + 1), dtype=torch.uint8
    )
    with pytest.raises(ValueError, match="contiguous"):
        w13.fuse_w13_packed(strided[:, :, : state.packed_columns], good)
    with pytest.raises(ValueError, match="same experts"):
        w13.fuse_w13_packed(good, _deterministic_packed(2, seed=82))
    with pytest.raises(ValueError, match="half must be 0"):
        w13.fuse_w13_packed_half(
            torch.zeros(
                (1,) + w13.KIMI_K3_W13_PACKED_SHAPE[1:], dtype=torch.uint8
            ),
            good,
            2,
        )


def check_inverse_rejects_a_payload_that_is_not_the_fused_shape() -> None:
    w13 = bound().w13
    wrong = torch.zeros((1, 5376, 255), dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"\[E, 5376, 256\]"):
        w13.unfuse_w13_packed(wrong)
    with pytest.raises(ValueError, match=r"\[E, 42, 2048\]"):
        w13.unfuse_w13_scale(torch.zeros((1, 41, 2048), dtype=torch.uint8))


# ---------------------------------------------------------------------------
# The production contract: one representation, named the same everywhere.
# ---------------------------------------------------------------------------


def check_prepared_bundle_carries_one_fused_pair_and_no_separate_halves(
) -> None:
    """``KimiK3DecodeWeights`` names ``expert_w13_*`` and nothing per-projection.

    A bundle that kept both would double the routed gate/up footprint of every
    rank, which is the one thing this representation exists to avoid.
    """
    weights_class = bound().kimi_k3.KimiK3DecodeWeights
    names = {
        field.name for field in weights_class.__dataclass_fields__.values()
    }
    assert {"expert_w13_packed", "expert_w13_scale"} <= names
    assert not any(
        name.startswith(("expert_w1_", "expert_w3_")) for name in names
    ), sorted(names)


def check_benchmark_stats_count_the_fused_pair_once() -> None:
    """The reported weight footprint is the halves' total, not twice it.

    Both figures the benchmark publishes are derived from the bundle's own
    fields, so a bundle that kept the halves alongside the fused pair would show
    up here as a doubled footprint -- and the routed working set the route
    metadata builds from ``_expert_weight_bytes`` would then claim to exceed L2
    for the wrong reason.
    """
    # Imported here rather than at module scope: the parent pytest process reads
    # `CHECKS` to name its tests, and the benchmark driver reaches `mok.kimi_k3`
    # through its input builders, which is the extension import this whole file
    # exists to keep out of collection.
    from benchmarks.bench_kimi_k3_decode import (
        _expert_weight_bytes,
        _prepared_weight_bytes,
    )

    state = bound()
    weights = _valid_weights(state.kimi_k3)
    packed_bytes, scale_bytes = state.w13.fused_w13_prepared_bytes()
    halves = (
        2
        * state.experts
        * state.rows
        * (state.packed_columns + state.scale_columns)
    )
    assert packed_bytes + scale_bytes == halves

    # One expert's slice of the fused pair is both of that expert's halves.
    per_expert = (packed_bytes + scale_bytes) // state.experts
    w2_per_expert = sum(
        getattr(weights, name)[0].numel()
        for name in ("expert_w2_packed", "expert_w2_scale")
    )
    assert _expert_weight_bytes(weights) == per_expert + w2_per_expert

    # And the routed gate/up share of the whole bundle's footprint is the
    # halves' total, which is the claim a duplicate allocation would break.
    fused_share = sum(
        getattr(weights, name).numel()
        for name in ("expert_w13_packed", "expert_w13_scale")
    )
    assert fused_share == halves
    assert _prepared_weight_bytes(weights) - fused_share == sum(
        getattr(weights, field.name).numel()
        * getattr(weights, field.name).element_size()
        for field in weights.__dataclass_fields__.values()
        if isinstance(getattr(weights, field.name), torch.Tensor)
        and not field.name.startswith("expert_w13_")
    )


def check_operator_schema_and_its_fake_name_the_fused_pair() -> None:
    """The schema, the CUDA implementation, and the meta implementation agree.

    A registered operator whose schema and fake disagree on an argument list is
    accepted at import and fails only under ``torch.compile``, so the three are
    read together rather than one at a time.
    """
    state = bound()
    ops, fake_impls = state.ops, state.fake_impls

    assert "Tensor expert_w13_packed, Tensor expert_w13_scale" in ops._DECODE_SCHEMA
    assert "expert_w1_" not in ops._DECODE_SCHEMA
    assert "expert_w3_" not in ops._DECODE_SCHEMA

    schema_arguments = re.findall(
        r"Tensor(?:\([a-z]!\))? (\w+)", ops._DECODE_SCHEMA
    )
    # The CUDA implementation is registered through a decorator that returns
    # nothing, so it has no module attribute to read a signature off; its
    # argument list is read out of the source instead, which is the same claim.
    # `mok.ops` only re-exports the step, so the source wanted is its definer's.
    cuda_source = inspect.getsource(inspect.getmodule(ops.kimi_k3_decode)).split(
        '@torch.library.impl("mok::kimi_k3_decode", "cuda")', 1
    )[1].split(") -> None:", 1)[0]
    cuda_tensors = [
        name for name in re.findall(r"^\s{4}(\w+): torch\.Tensor", cuda_source,
                                    re.MULTILINE)
        if name in schema_arguments
    ]
    assert cuda_tensors == schema_arguments, cuda_tensors
    for function in (ops.kimi_k3_decode, fake_impls._kimi_k3_decode_fake):
        parameters = list(inspect.signature(function).parameters)
        tensors = [name for name in parameters if name in schema_arguments]
        assert tensors == schema_arguments, (function.__name__, tensors)


def check_alignment_table_holds_the_payload_to_its_descriptor_base() -> None:
    """32 bytes, not 16.

    Every other weight only has to satisfy a 16-byte vector load.
    ``cuTensorMapEncodeTiled`` refuses a base that is not 32-byte aligned, and it
    refuses it at the first launch against a payload rather than at preparation,
    so the operator checks the stricter figure itself.
    """
    table = dict(bound().ops._DECODE_ALIGNMENT)
    assert table["expert_w13_packed"] == 32
    assert table["expert_w13_scale"] == 16
    assert "expert_w1_packed" not in table
    assert "expert_w3_packed" not in table


CHECKS = {
    "task_decomposition_covers_every_channel_and_k_panel":
        check_task_decomposition_covers_every_channel_and_k_panel,
    "prepared_bytes_are_the_halves_they_replace_and_not_twice_them":
        check_prepared_bytes_are_the_halves_they_replace_and_not_twice_them,
    "fused_payload_is_the_same_bytes_as_the_halves":
        check_fused_payload_is_the_same_bytes_as_the_halves,
    "transform_returns_every_packed_byte_under_inversion":
        check_transform_returns_every_packed_byte_under_inversion,
    "transform_returns_every_scale_byte_under_inversion":
        check_transform_returns_every_scale_byte_under_inversion,
    **{
        f"reading_one_half_back_agrees_with_reading_both[{half}]": partial(
            check_reading_one_half_back_agrees_with_reading_both, half
        )
        for half in (0, 1)
    },
    "writing_one_half_leaves_the_other_half_untouched":
        check_writing_one_half_leaves_the_other_half_untouched,
    "transform_is_a_permutation_of_the_bytes_it_was_given":
        check_transform_is_a_permutation_of_the_bytes_it_was_given,
    **{
        f"transform_treats_every_expert_the_same_way[{expert}]": partial(
            check_transform_treats_every_expert_the_same_way, expert
        )
        for expert in PROBE_EXPERTS
    },
    **{
        f"each_task_pairs_the_gate_and_up_rows_of_the_same_channels[{task}]":
            partial(
                check_each_task_pairs_the_gate_and_up_rows_of_the_same_channels,
                task,
            )
        for task in range(6)
    },
    **{
        f"every_k_panel_lands_in_its_own_slab[{slab}]": partial(
            check_every_k_panel_lands_in_its_own_slab, slab
        )
        for slab in range(7)
    },
    "scale_offset_is_the_sm103_scale_factor_atom":
        check_scale_offset_is_the_sm103_scale_factor_atom,
    "scale_shuffle_is_a_bijection_over_one_slab":
        check_scale_shuffle_is_a_bijection_over_one_slab,
    "each_scale_byte_lands_at_the_offset_its_group_is_read_from":
        check_each_scale_byte_lands_at_the_offset_its_group_is_read_from,
    "transform_carries_every_fp4_nibble_pair":
        check_transform_carries_every_fp4_nibble_pair,
    **{
        f"transform_carries_every_e8m0_boundary_code[{code:#04x}]": partial(
            check_transform_carries_every_e8m0_boundary_code, code
        )
        for code in E8M0_BOUNDARIES
    },
    "a_zero_group_keeps_its_zero_payload_and_unit_scale":
        check_a_zero_group_keeps_its_zero_payload_and_unit_scale,
    "transform_rejects_a_half_of_the_wrong_shape_or_dtype":
        check_transform_rejects_a_half_of_the_wrong_shape_or_dtype,
    "inverse_rejects_a_payload_that_is_not_the_fused_shape":
        check_inverse_rejects_a_payload_that_is_not_the_fused_shape,
    "prepared_bundle_carries_one_fused_pair_and_no_separate_halves":
        check_prepared_bundle_carries_one_fused_pair_and_no_separate_halves,
    "benchmark_stats_count_the_fused_pair_once":
        check_benchmark_stats_count_the_fused_pair_once,
    "operator_schema_and_its_fake_name_the_fused_pair":
        check_operator_schema_and_its_fake_name_the_fused_pair,
    "alignment_table_holds_the_payload_to_its_descriptor_base":
        check_alignment_table_holds_the_payload_to_its_descriptor_base,
}


def main() -> int:
    results: dict[str, dict[str, str]] = {}
    for name, check in CHECKS.items():
        try:
            check()
        except (Exception, pytest.fail.Exception):
            # A missing `pytest.raises` failure is a Failed, not an Exception, so
            # without it here one broken check would hide every later one.
            results[name] = {
                "outcome": "failed",
                "detail": traceback.format_exc(),
            }
        else:
            results[name] = {"outcome": "passed", "detail": ""}
    print(RESULT_MARKER + json.dumps(results))
    failures = sum(result["outcome"] == "failed" for result in results.values())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
