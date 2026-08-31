"""The fused W13 storage layout for Kimi K3 decode's routed experts.

A routed expert's gate (``w1``) and up (``w3``) projections are contracted
together by one tile of one MMA, so they are stored together: one packed tensor
and one scale tensor, in the order the copy engine and the tensor core read
them. This module is the transform from the canonical per-projection MXFP4 form
into that layout, and its exact inverse.

The transform is a permutation. It moves no value across a quantization group,
requantizes nothing, and the fused payload holds exactly as many bytes as the
two halves it replaces -- which is the whole reason the prepared weights can
carry it *instead* of them rather than beside them.

Geometry
--------
One expert, one rank:

* gate ``w1`` and up ``w3`` are each ``[384, 3584]`` logical, stored packed as
  ``[384, 1792]`` FP4 bytes plus ``[384, 112]`` E8M0 group-32 scale bytes;
* the fused form is six output tasks of ``M = 128`` rows, where task ``t`` holds
  gate rows ``[64t, 64t + 64)`` in M rows ``[0, 64)`` and the *same* up rows in
  M rows ``[64, 128)``, so the epilogue pairs M row ``r`` with M row ``r + 64``
  and reads both halves of one output channel from one tensor-memory tile;
* ``K = 3584`` is seven slabs of 512, and a slab is sixteen ``K = 32``
  block-scaled contractions;
* six tasks times 64 channels covers all 384 ``situ`` columns of the rank's
  scratch layout, 64 contiguous columns per task.

Why the payload needs no shuffle within a row
---------------------------------------------
The engine reads a ``(task, slab)`` tile through one ``16U4_ALIGN16B`` TMA whose
box is ``128 x 128`` U4 values, four boxes wide. That descriptor lays a row's
packed bytes down in their own order -- byte ``i`` of a row carries K values
``2i`` and ``2i + 1``, and the MMA's ``K = 32`` chunk ``g`` reads exactly the 32
shared bytes holding K values ``[32g, 32g + 32)``. So the packed transform is
only which row goes where.

Why the scales do
-----------------
The MMA does not read block scales from shared memory at all:
``tcgen05.cp.32x128b.warpx4`` moves a 512-byte shared tile into tensor memory
and that tile's byte order is CUTLASS's SM103
``Sm103BlockScaledBasicChunk<32>::SfKMajorAtom``. Pre-shuffling into that order
is what lets a whole slab's sixteen groups move as one 2 KiB contiguous bulk
copy instead of 128 strided single-byte reads.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from mok.kimi_k3_shapes import (
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_MXFP4_GROUP_SIZE,
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    KIMI_K3_TP_SIZE,
)

# ---------------------------------------------------------------------------
# Task and slab geometry. Mirrors
# `csrc/kimi_k3_decode/expert_mxfp4_fused_w13.cuh`, which consumes the bytes.
# ---------------------------------------------------------------------------

#: Output channels one rank owns per expert, per projection.
KIMI_K3_W13_CHANNELS_PER_RANK = (
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE
)

#: Rows one fused output task presents to the MMA's M axis.
KIMI_K3_W13_TASK_ROWS = 128

#: Gate rows in the low half of a task; the up rows of the same channels follow.
KIMI_K3_W13_HALF_ROWS = KIMI_K3_W13_TASK_ROWS // 2

#: Output tasks one expert is decomposed into.
KIMI_K3_W13_TASKS = KIMI_K3_W13_CHANNELS_PER_RANK // KIMI_K3_W13_HALF_ROWS

#: Contraction width one slab covers, and the slabs that tile K.
KIMI_K3_W13_SLAB_K = 512
KIMI_K3_W13_SLABS = KIMI_K3_LATENT_SIZE // KIMI_K3_W13_SLAB_K

#: ``K = 32`` contractions one slab issues.
KIMI_K3_W13_SLAB_GROUPS = KIMI_K3_W13_SLAB_K // KIMI_K3_MXFP4_GROUP_SIZE

#: Packed FP4 bytes of one row of one slab, and of a whole ``(task, slab)`` tile.
KIMI_K3_W13_SLAB_ROW_BYTES = KIMI_K3_W13_SLAB_K // 2
KIMI_K3_W13_SLAB_PACKED_BYTES = (
    KIMI_K3_W13_TASK_ROWS * KIMI_K3_W13_SLAB_ROW_BYTES
)

#: E8M0 scale bytes of one ``(task, slab)`` tile.
KIMI_K3_W13_SLAB_SCALE_BYTES = (
    KIMI_K3_W13_TASK_ROWS * KIMI_K3_W13_SLAB_GROUPS
)

#: One shared scale tile: 512 bytes, one byte per M row for four K groups.
KIMI_K3_W13_SCALE_TILE_BYTES = 512
KIMI_K3_W13_SCALE_GROUPS_PER_TILE = 4
KIMI_K3_W13_SCALE_TILES_PER_SLAB = (
    KIMI_K3_W13_SLAB_GROUPS // KIMI_K3_W13_SCALE_GROUPS_PER_TILE
)

#: ``(task, slab)`` pairs one expert holds, which is the payload's outer axis.
KIMI_K3_W13_TASK_SLABS = KIMI_K3_W13_TASKS * KIMI_K3_W13_SLABS

#: Prepared shapes the decode operator accepts, per rank.
KIMI_K3_W13_PACKED_SHAPE = (
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_W13_TASK_SLABS * KIMI_K3_W13_TASK_ROWS,
    KIMI_K3_W13_SLAB_ROW_BYTES,
)
KIMI_K3_W13_SCALE_SHAPE = (
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_W13_TASK_SLABS,
    KIMI_K3_W13_SLAB_SCALE_BYTES,
)

#: The canonical per-projection shapes the transform consumes and returns.
KIMI_K3_W13_HALF_PACKED_SHAPE = (
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_W13_CHANNELS_PER_RANK,
    KIMI_K3_LATENT_SIZE // 2,
)
KIMI_K3_W13_HALF_SCALE_SHAPE = (
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_W13_CHANNELS_PER_RANK,
    KIMI_K3_LATENT_SIZE // KIMI_K3_MXFP4_GROUP_SIZE,
)

_PACKED_COLUMNS = KIMI_K3_W13_HALF_PACKED_SHAPE[2]
_SCALE_COLUMNS = KIMI_K3_W13_HALF_SCALE_SHAPE[2]

assert KIMI_K3_W13_CHANNELS_PER_RANK == 384
assert KIMI_K3_W13_TASKS == 6
assert KIMI_K3_W13_SLABS == 7
assert KIMI_K3_W13_SLAB_GROUPS == 16
assert KIMI_K3_W13_SLAB_PACKED_BYTES == 32768
assert KIMI_K3_W13_SLAB_SCALE_BYTES == 2048
assert KIMI_K3_W13_SCALE_TILES_PER_SLAB == 4
assert KIMI_K3_W13_TASK_SLABS == 42
assert KIMI_K3_W13_PACKED_SHAPE == (896, 5376, 256)
assert KIMI_K3_W13_SCALE_SHAPE == (896, 42, 2048)
assert (
    KIMI_K3_W13_SCALE_TILE_BYTES
    == KIMI_K3_W13_TASK_ROWS * KIMI_K3_W13_SCALE_GROUPS_PER_TILE
)

# The transform is a permutation, so the fused payload is byte-for-byte as
# large as the two halves it replaces. This is the identity that lets the
# prepared weights drop `w1` and `w3` rather than keep them alongside, and
# `test_the_fused_payload_is_the_same_bytes_as_the_halves` measures it on real
# tensors; here it is arithmetic on the shapes themselves.
assert KIMI_K3_W13_PACKED_SHAPE[1] * KIMI_K3_W13_PACKED_SHAPE[2] == (
    2 * KIMI_K3_W13_CHANNELS_PER_RANK * _PACKED_COLUMNS
)
assert KIMI_K3_W13_SCALE_SHAPE[1] * KIMI_K3_W13_SCALE_SHAPE[2] == (
    2 * KIMI_K3_W13_CHANNELS_PER_RANK * _SCALE_COLUMNS
)


def _check_half(name: str, tensor: torch.Tensor, columns: int) -> int:
    """Validate one canonical ``[E, 384, columns]`` half and return ``E``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8, got {tensor.dtype}")
    if tensor.dim() != 3 or tensor.shape[1:] != (
        KIMI_K3_W13_CHANNELS_PER_RANK,
        columns,
    ):
        raise ValueError(
            f"{name} must be [E, {KIMI_K3_W13_CHANNELS_PER_RANK}, {columns}], "
            f"got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return int(tensor.shape[0])


def _check_fused(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, int, int],
) -> int:
    """Validate one fused ``[E, rows, columns]`` payload and return ``E``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8, got {tensor.dtype}")
    if tensor.dim() != 3 or tensor.shape[1:] != shape[1:]:
        raise ValueError(
            f"{name} must be uint8 [E, {shape[1]}, {shape[2]}], "
            f"got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return int(tensor.shape[0])


def _half_view(fused: torch.Tensor, experts: int, width: int) -> torch.Tensor:
    """View a fused payload as ``[E, task, half, 64, slab, width]``.

    Both directions of the packed transform are one ``copy_`` through this
    view, which is why neither allocates a concatenation: the fused tensor's
    own strides express the permutation, so the halves are written into place
    and read back out of place with nothing in between.
    """
    return fused.view(
        experts,
        KIMI_K3_W13_TASKS,
        KIMI_K3_W13_SLABS,
        2,
        KIMI_K3_W13_HALF_ROWS,
        width,
    ).permute(0, 1, 3, 4, 2, 5)


def _check_index(half: int) -> int:
    if type(half) is not int or half not in (0, 1):
        raise ValueError(f"half must be 0 (gate) or 1 (up), got {half!r}")
    return half


def fuse_w13_packed_half(
    fused: torch.Tensor,
    half_packed: torch.Tensor,
    half: int,
) -> torch.Tensor:
    """Write one projection's MXFP4 payload into its rows of a fused payload.

    ``half`` is 0 for the gate rows, which occupy M rows ``[0, 64)`` of every
    ``(task, slab)`` tile, and 1 for the up rows, which occupy ``[64, 128)``.
    The other half's rows are untouched.
    """
    experts = _check_half("half_packed", half_packed, _PACKED_COLUMNS)
    if _check_fused("fused", fused, KIMI_K3_W13_PACKED_SHAPE) != experts:
        raise ValueError(
            f"fused must cover {experts} experts, got {fused.shape[0]}"
        )
    view = _half_view(fused, experts, KIMI_K3_W13_SLAB_ROW_BYTES)
    view[:, :, _check_index(half)].copy_(
        half_packed.view(
            experts,
            KIMI_K3_W13_TASKS,
            KIMI_K3_W13_HALF_ROWS,
            KIMI_K3_W13_SLABS,
            KIMI_K3_W13_SLAB_ROW_BYTES,
        )
    )
    return fused


def fuse_w13_packed(
    w1_packed: torch.Tensor,
    w3_packed: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interleave one rank's gate and up MXFP4 payloads into six output tasks.

    Returns ``[E, tasks * slabs * 128, 256]``: task ``t``, slab ``s``, M row
    ``r < 64`` is gate channel ``64t + r``'s K panel ``[512s, 512s + 512)``, and
    M row ``r + 64`` is the same channel's up panel.
    """
    experts = _check_half("w1_packed", w1_packed, _PACKED_COLUMNS)
    if _check_half("w3_packed", w3_packed, _PACKED_COLUMNS) != experts:
        raise ValueError("the gate and up payloads must cover the same experts")
    fused = _fused_destination(
        out,
        experts,
        KIMI_K3_W13_PACKED_SHAPE,
        w1_packed,
    )
    for half, source in enumerate((w1_packed, w3_packed)):
        fuse_w13_packed_half(fused, source, half)
    return fused


def _fused_destination(
    out: torch.Tensor | None,
    experts: int,
    shape: tuple[int, int, int],
    like: torch.Tensor,
) -> torch.Tensor:
    if out is None:
        return torch.empty(
            (experts, shape[1], shape[2]),
            dtype=torch.uint8,
            device=like.device,
        )
    if _check_fused("out", out, shape) != experts:
        raise ValueError(
            f"out must cover {experts} experts, got {out.shape[0]}"
        )
    if out.device != like.device:
        raise ValueError(f"out must be on {like.device}")
    return out


def unfuse_w13_packed_half(
    payload: torch.Tensor,
    half: int,
) -> torch.Tensor:
    """Read one projection's canonical MXFP4 payload back out of a fused one.

    One half at a time, so a caller that only needs the gate rows -- or that
    needs both but cannot hold both at once -- pays for one.
    """
    experts = _check_fused("payload", payload, KIMI_K3_W13_PACKED_SHAPE)
    view = _half_view(payload, experts, KIMI_K3_W13_SLAB_ROW_BYTES)
    return (
        view[:, :, _check_index(half)]
        .reshape(experts, KIMI_K3_W13_CHANNELS_PER_RANK, _PACKED_COLUMNS)
        .contiguous()
    )


def unfuse_w13_packed(
    payload: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert :func:`fuse_w13_packed`, returning the gate and up payloads."""
    return (
        unfuse_w13_packed_half(payload, 0),
        unfuse_w13_packed_half(payload, 1),
    )


def scale_tile_offset(row: int, k_group: int) -> int:
    """Byte offset of row ``row``'s factor for K group ``k_group`` in one tile.

    This is ``expert_mxfp4::scale_factor_1x_offset`` spelled in Python: the
    atom's shape is ``((8,4,4),(32,4))`` and its stride ``((16,128,4),(0,1))``.
    """
    return (
        (row % 8) * 16
        + ((row // 8) % 4) * 128
        + (row // 32) * 4
        + k_group
    )


@lru_cache(maxsize=1)
def scale_scatter_index() -> torch.Tensor:
    """Where row-major ``(row, group)`` scale byte ``row * 16 + group`` lands."""
    rows = torch.arange(KIMI_K3_W13_TASK_ROWS, dtype=torch.int64).unsqueeze(1)
    groups = torch.arange(
        KIMI_K3_W13_SLAB_GROUPS, dtype=torch.int64
    ).unsqueeze(0)
    tile = groups // KIMI_K3_W13_SCALE_GROUPS_PER_TILE
    factor = groups % KIMI_K3_W13_SCALE_GROUPS_PER_TILE
    offset = (
        (rows % 8) * 16
        + ((rows // 8) % 4) * 128
        + (rows // 32) * 4
        + factor
    )
    return (
        tile * KIMI_K3_W13_SCALE_TILE_BYTES + offset
    ).reshape(-1).contiguous()


@lru_cache(maxsize=1)
def scale_gather_index() -> torch.Tensor:
    """Which row-major ``(row, group)`` byte each shuffled offset holds."""
    return torch.argsort(scale_scatter_index()).contiguous()


#: Shuffled scale bytes one projection's half of a ``(task, slab)`` entry owns.
_HALF_SCALE_BYTES = KIMI_K3_W13_HALF_ROWS * KIMI_K3_W13_SLAB_GROUPS


def fuse_w13_scale_half(
    fused: torch.Tensor,
    half_scale: torch.Tensor,
    half: int,
) -> torch.Tensor:
    """Write one projection's E8M0 scales into its bytes of a fused blob.

    The shuffle into ``SfKMajorAtom`` order is a permutation *within* a
    ``(task, slab)`` entry's 2,048 bytes, and rows ``[64h, 64h + 64)`` are
    exactly the flat range ``[1024h, 1024h + 1024)`` of the row-major
    ``(row, group)`` order the scatter index is built over. So one half's
    destinations are a contiguous slice of that index and this writes them
    without touching the other half's.
    """
    experts = _check_half("half_scale", half_scale, _SCALE_COLUMNS)
    if _check_fused("fused", fused, KIMI_K3_W13_SCALE_SHAPE) != experts:
        raise ValueError(
            f"fused must cover {experts} experts, got {fused.shape[0]}"
        )
    index = _check_index(half)
    natural = (
        half_scale.view(
            experts,
            KIMI_K3_W13_TASKS,
            KIMI_K3_W13_HALF_ROWS,
            KIMI_K3_W13_SLABS,
            KIMI_K3_W13_SLAB_GROUPS,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(experts, KIMI_K3_W13_TASK_SLABS, _HALF_SCALE_BYTES)
    )
    destination = scale_scatter_index()[
        index * _HALF_SCALE_BYTES : (index + 1) * _HALF_SCALE_BYTES
    ].to(fused.device)
    fused.index_copy_(2, destination, natural)
    return fused


def fuse_w13_scale(
    w1_scale: torch.Tensor,
    w3_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interleave and shuffle one rank's gate and up E8M0 scales.

    Returns ``[E, tasks * slabs, 2048]``: a ``(task, slab)`` entry is the four
    512-byte scale tiles the slab's sixteen K groups are read from, in the order
    ``tcgen05.cp`` and the MMA's scale-factor id consume them.
    """
    experts = _check_half("w1_scale", w1_scale, _SCALE_COLUMNS)
    if _check_half("w3_scale", w3_scale, _SCALE_COLUMNS) != experts:
        raise ValueError("the gate and up scales must cover the same experts")
    fused = _fused_destination(
        out,
        experts,
        KIMI_K3_W13_SCALE_SHAPE,
        w1_scale,
    )
    for half, source in enumerate((w1_scale, w3_scale)):
        fuse_w13_scale_half(fused, source, half)
    return fused


def unfuse_w13_scale_half(
    scales: torch.Tensor,
    half: int,
) -> torch.Tensor:
    """Read one projection's E8M0 scales back out of a fused blob.

    Gathering with the same scatter index the forward direction scattered by is
    what undoes the shuffle: the forward writes ``fused[.., destination[i]] =
    natural[.., i]``, so ``fused[.., destination]`` *is* ``natural``.
    """
    experts = _check_fused("scales", scales, KIMI_K3_W13_SCALE_SHAPE)
    index = _check_index(half)
    source = scale_scatter_index()[
        index * _HALF_SCALE_BYTES : (index + 1) * _HALF_SCALE_BYTES
    ].to(scales.device)
    return (
        scales.index_select(2, source)
        .view(
            experts,
            KIMI_K3_W13_TASKS,
            KIMI_K3_W13_SLABS,
            KIMI_K3_W13_HALF_ROWS,
            KIMI_K3_W13_SLAB_GROUPS,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(experts, KIMI_K3_W13_CHANNELS_PER_RANK, _SCALE_COLUMNS)
        .contiguous()
    )


def unfuse_w13_scale(
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert :func:`fuse_w13_scale`, returning the gate and up scales."""
    return (
        unfuse_w13_scale_half(scales, 0),
        unfuse_w13_scale_half(scales, 1),
    )


def fuse_w13(
    w1_packed: torch.Tensor,
    w1_scale: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the transform over one rank's canonical MXFP4 gate/up pair."""
    return (
        fuse_w13_packed(w1_packed, w3_packed),
        fuse_w13_scale(w1_scale, w3_scale),
    )


def fuse_w13_half(
    packed: torch.Tensor,
    scale: torch.Tensor,
    half_packed: torch.Tensor,
    half_scale: torch.Tensor,
    half: int,
) -> None:
    """Write one projection's payload and scales into a fused pair.

    What :func:`mok.kimi_k3.prepare_kimi_k3_decode_weights` calls, once per
    projection, so that neither packed half has to be alive while the other is
    being quantized.
    """
    fuse_w13_packed_half(packed, half_packed, half)
    fuse_w13_scale_half(scale, half_scale, half)


def unfuse_w13_half(
    packed: torch.Tensor,
    scale: torch.Tensor,
    half: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover one projection's canonical payload and scales.

    What a consumer that walks the experts in chunks calls -- the benchmark's
    dequantizing oracle does one projection of one chunk at a time -- so no
    caller has to hold both halves of a chunk to read either.
    """
    return (
        unfuse_w13_packed_half(packed, half),
        unfuse_w13_scale_half(scale, half),
    )


def unfuse_w13(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover the canonical gate/up MXFP4 pair from a fused payload.

    The exact inverse of :func:`fuse_w13`, in the order
    :func:`mok.kimi_k3.prepare_kimi_k3_decode_weights` used to return: gate
    payload, gate scales, up payload, up scales. Every consumer that wants the
    per-projection view -- the benchmark oracle, the framework adapters, the
    transform tests -- goes through here rather than reimplementing the index
    arithmetic, so there is one definition of the layout to be wrong about.
    """
    w1_packed, w3_packed = unfuse_w13_packed(packed)
    w1_scale, w3_scale = unfuse_w13_scale(scale)
    return w1_packed, w1_scale, w3_packed, w3_scale


def fused_w13_prepared_bytes() -> tuple[int, int]:
    """Bytes one rank's fused payload and its scales occupy, for all experts."""
    packed = 1
    for extent in KIMI_K3_W13_PACKED_SHAPE:
        packed *= extent
    scale = 1
    for extent in KIMI_K3_W13_SCALE_SHAPE:
        scale *= extent
    return packed, scale


__all__ = [
    "KIMI_K3_W13_CHANNELS_PER_RANK",
    "KIMI_K3_W13_HALF_PACKED_SHAPE",
    "KIMI_K3_W13_HALF_ROWS",
    "KIMI_K3_W13_HALF_SCALE_SHAPE",
    "KIMI_K3_W13_PACKED_SHAPE",
    "KIMI_K3_W13_SCALE_GROUPS_PER_TILE",
    "KIMI_K3_W13_SCALE_SHAPE",
    "KIMI_K3_W13_SCALE_TILES_PER_SLAB",
    "KIMI_K3_W13_SCALE_TILE_BYTES",
    "KIMI_K3_W13_SLABS",
    "KIMI_K3_W13_SLAB_GROUPS",
    "KIMI_K3_W13_SLAB_K",
    "KIMI_K3_W13_SLAB_PACKED_BYTES",
    "KIMI_K3_W13_SLAB_ROW_BYTES",
    "KIMI_K3_W13_SLAB_SCALE_BYTES",
    "KIMI_K3_W13_TASKS",
    "KIMI_K3_W13_TASK_ROWS",
    "KIMI_K3_W13_TASK_SLABS",
    "fuse_w13",
    "fuse_w13_half",
    "fuse_w13_packed",
    "fuse_w13_packed_half",
    "fuse_w13_scale",
    "fuse_w13_scale_half",
    "fused_w13_prepared_bytes",
    "scale_gather_index",
    "scale_scatter_index",
    "scale_tile_offset",
    "unfuse_w13",
    "unfuse_w13_half",
    "unfuse_w13_packed",
    "unfuse_w13_packed_half",
    "unfuse_w13_scale",
    "unfuse_w13_scale_half",
]
