"""Where the routed gate/up descriptor really puts bytes, measured on SM103.

The transform in :mod:`mok.kimi_k3_w13` decides the order the prepared payload
is written in, and ``tests/test_kimi_k3_w13.py`` proves without a GPU that the
order is a permutation that loses nothing. Neither of those says the hardware
reads it back the way the descriptor claims, and two parts of that are
properties of the device rather than of any source file here:

* what ``cp.async.bulk.tensor.5d`` counts as its transaction, which is the
  32,768 bytes it read out of global memory rather than the 65,536 it wrote
  after ``16U4_ALIGN16B`` container padding, and
* where the five dimensions land, including which shared column a box's row
  bytes occupy under ``CU_TENSOR_MAP_SWIZZLE_128B``.

Getting either wrong is not a wrong number, it is a hang or a silent read of
the wrong weights. So the probe entrypoint runs one real transfer under a
bounded wait -- a wrong transaction count is reported rather than spinning the
device forever -- and reads the tile back out through the same ``(row, column)``
indexing ``chunk_descriptor`` and the MMA address, so agreement here is
agreement about the swizzle and not merely about which bytes arrived somewhere.

Every assertion below is against the production payload shape and the
production descriptor: the same ``create_fused_w13_packed_map`` the decode step
launches with, over all 896 experts and all 42 ``(task, slab)`` pairs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch

from mok import _C
from mok.kimi_k3_w13 import (
    KIMI_K3_W13_PACKED_SHAPE,
    KIMI_K3_W13_SCALE_SHAPE,
    KIMI_K3_W13_SLAB_K,
    KIMI_K3_W13_SLABS,
    KIMI_K3_W13_TASK_ROWS,
    KIMI_K3_W13_TASK_SLABS,
)

from . import kimi_k3_decode_sources as decode_sources

#: Packed source bytes at the front of each sixteen-byte shared atom. The other
#: eight are the format's container padding, which the MMA never reads and no
#: byte of the transfer may land in.
LIVE_BYTES_PER_ATOM = 8

#: U4 values one ``16U4_ALIGN16B`` box row carries, which the format pins at 128,
#: and the boxes a 512-wide slab is therefore made of.
BOX_ELEMENTS = 128
BOXES = KIMI_K3_W13_SLAB_K // BOX_ELEMENTS


@pytest.fixture(scope="module")
def geometry() -> dict[str, int]:
    """The engine's own numbers, read out of the extension."""
    return _C._kimi_k3_fused_w13_geometry()


@pytest.fixture(scope="module")
def device() -> torch.device:
    """One device per rank, without joining a process group to find out which.

    This probe is a single-CTA launch and needs no collective, but the suite runs
    it under an eight-rank torchrun, so taking ``current_device`` would put all
    eight ranks' payloads on device 0 unless some earlier test happened to have
    set the device first. ``LOCAL_RANK`` is the same answer without the ordering
    dependency, and the entrypoint guards on the tensor's own device.
    """
    if not torch.cuda.is_available():
        pytest.skip("the routed gate/up descriptor requires a CUDA device")
    local_rank = int(
        os.environ.get("LOCAL_RANK", str(torch.cuda.current_device()))
    )
    selected = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(selected) != (10, 3):
        pytest.skip("the routed gate/up descriptor requires SM103 B300")
    return selected


@pytest.fixture(scope="module")
def payload(device: torch.device) -> Iterator[torch.Tensor]:
    """A full-width prepared payload whose every byte is distinguishable.

    Full width because the descriptor names all 896 experts and all 42
    ``(task, slab)`` pairs, so a narrower tensor would not be the production
    layout. Drawn directly in prepared shape rather than transformed from a gate
    and up pair, because what is being measured is where the descriptor puts
    bytes, not what the transform does with them -- and random rather than
    patterned, so a neighbouring tile's bytes cannot be mistaken for this one's.
    """
    generator = torch.Generator(device=device).manual_seed(103)
    tensor = torch.randint(
        0,
        256,
        KIMI_K3_W13_PACKED_SHAPE,
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    try:
        yield tensor
    finally:
        del tensor
        torch.cuda.empty_cache()


def _probe(
    payload: torch.Tensor,
    expert: int,
    task_slab: int,
    transaction_bytes: int,
) -> tuple[torch.Tensor, int]:
    """One slab transfer under a bounded wait, and whether the wait released."""
    dump, completed = _C._kimi_k3_fused_w13_tma_probe(
        payload, expert, task_slab, transaction_bytes
    )
    return (
        dump.cpu().reshape(KIMI_K3_W13_TASK_ROWS, KIMI_K3_W13_SLAB_K),
        int(completed),
    )


def _landed(
    payload: torch.Tensor, expert: int, task_slab: int, geometry: dict[str, int]
) -> torch.Tensor:
    """Run one slab transfer at the engine's own count and return the tile."""
    image, completed = _probe(
        payload, expert, task_slab, geometry["weight_transaction_bytes"]
    )
    assert completed == 1, (
        f"the mbarrier never released for the "
        f"{geometry['weight_transaction_bytes']} transaction bytes the engine "
        f"expects per slab"
    )
    return image


def _expected(
    payload: torch.Tensor, expert: int, task_slab: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``[128, 512]`` image one tile must present, and its live mask.

    ``16U4_ALIGN16B`` reads 128 U4 values per box row -- 64 packed bytes -- and
    writes them as eight sixteen-byte atoms carrying eight source bytes each.
    Four boxes make a 512-wide slab and the descriptor's third dimension puts
    box ``b`` at logical columns ``[128b, 128b + 128)``. So logical column
    ``128b + 16i + j`` holds source byte ``64b + 8i + j`` of the row while
    ``j < 8``, and is container padding otherwise.
    """
    rows = KIMI_K3_W13_TASK_ROWS
    source = payload[expert, task_slab * rows : (task_slab + 1) * rows, :].cpu()
    expected = torch.zeros(rows, KIMI_K3_W13_SLAB_K, dtype=torch.uint8)
    live = torch.zeros(rows, KIMI_K3_W13_SLAB_K, dtype=torch.bool)
    for atom in range(KIMI_K3_W13_SLAB_K // 16):
        column = atom * 16
        packed = atom * LIVE_BYTES_PER_ATOM
        expected[:, column : column + LIVE_BYTES_PER_ATOM] = source[
            :, packed : packed + LIVE_BYTES_PER_ATOM
        ]
        live[:, column : column + LIVE_BYTES_PER_ATOM] = True
    return expected, live


# ---------------------------------------------------------------------------
# The transaction.
# ---------------------------------------------------------------------------


def test_the_transaction_count_is_the_payload_read_not_the_footprint_written(
    payload: torch.Tensor, geometry: dict[str, int]
) -> None:
    """How many bytes ``16U4_ALIGN16B`` reports, measured rather than assumed.

    A slab is 32,768 packed bytes in global memory and 65,536 bytes in shared
    memory, because the format writes each eight packed bytes into a sixteen-byte
    container. The transaction counter tracks one of those two numbers and
    nothing in the ISA documentation says which; expecting the wrong one either
    hangs the wait forever or releases it while the copy is still in flight.

    Both directions are asserted, because "32,768 releases the wait" alone would
    also be true if the counter reported 65,536 and the barrier were simply
    released early. The tile image below is what separates those, since all four
    boxes and all 128 rows are present at the smaller count.
    """
    assert geometry["weight_transaction_bytes"] == 32_768
    assert geometry["weight_tile_bytes"] == 65_536
    assert geometry["weight_transaction_bytes"] == (
        geometry["weight_tile_bytes"] // 2
    )

    completed = {
        candidate: _probe(payload, 0, 0, candidate)[1]
        for candidate in (
            geometry["weight_transaction_bytes"],
            geometry["weight_tile_bytes"],
        )
    }
    assert completed == {
        geometry["weight_transaction_bytes"]: 1,
        geometry["weight_tile_bytes"]: 0,
    }, completed


def test_the_slabs_mbarrier_also_counts_the_scales_that_travel_with_it(
    geometry: dict[str, int]
) -> None:
    """The ring waits on one barrier for the weights and their scales together.

    The probe issues only the 5D weight transfer, so its count is the weight
    transaction alone. The engine's per-slab count is that plus the 2 KiB of
    E8M0 scales the same barrier carries, and those 2 KiB are exactly the
    prepared scale tensor's row: 42 ``(task, slab)`` pairs of 2,048 bytes.
    """
    scale_bytes = (
        geometry["slab_transaction_bytes"] - geometry["weight_transaction_bytes"]
    )
    assert scale_bytes == 2_048
    assert geometry["slab_transaction_bytes"] == 34_816
    assert KIMI_K3_W13_SCALE_SHAPE[1:] == (KIMI_K3_W13_TASK_SLABS, scale_bytes)
    assert KIMI_K3_W13_TASK_SLABS == 42
    # And the packed side's own accounting: 42 tiles of 32,768 global bytes is
    # the payload row the descriptor walks.
    assert KIMI_K3_W13_PACKED_SHAPE[1] * KIMI_K3_W13_PACKED_SHAPE[2] == (
        KIMI_K3_W13_TASK_SLABS * geometry["weight_transaction_bytes"]
    )


# ---------------------------------------------------------------------------
# The five dimensions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expert", "task_slab"),
    [
        (0, 0),
        (0, KIMI_K3_W13_SLABS - 1),
        (447, 20),
        (KIMI_K3_W13_PACKED_SHAPE[0] - 1, KIMI_K3_W13_TASK_SLABS - 1),
    ],
)
def test_the_tile_lands_the_bytes_the_transform_placed(
    payload: torch.Tensor,
    geometry: dict[str, int],
    expert: int,
    task_slab: int,
) -> None:
    """One slab is one transfer, and it arrives where the MMA will read it.

    The four cases walk the outer two descriptor dimensions to their ends: the
    first and last expert, the first and last ``(task, slab)`` pair, and an
    interior pair. The last case is the final tile of the payload, so a stride
    that overran the tensor would read past its last byte here.
    """
    image = _landed(payload, expert, task_slab, geometry)
    expected, live = _expected(payload, expert, task_slab)
    assert torch.equal(image[live], expected[live])


def test_all_four_boxes_of_a_slab_arrive_in_the_one_transfer(
    payload: torch.Tensor, geometry: dict[str, int]
) -> None:
    """The third dimension is the descriptor's, not four separate instructions.

    A descriptor with the wrong box dimension would land box 0 and leave the
    rest as the zeros the probe clears the tile with, which is what this
    separates from the whole-tile comparison above.
    """
    assert geometry["boxes"] == BOXES == 4
    assert geometry["box_elements"] == BOX_ELEMENTS
    image = _landed(payload, 5, 11, geometry)
    expected, live = _expected(payload, 5, 11)
    for box in range(BOXES):
        columns = slice(box * BOX_ELEMENTS, (box + 1) * BOX_ELEMENTS)
        mask = live[:, columns]
        assert torch.equal(
            image[:, columns][mask], expected[:, columns][mask]
        ), f"box {box} of the slab did not arrive"


@pytest.mark.parametrize("k_group", [0, 1, 4, 7, 8, 15])
def test_each_K32_chunk_holds_the_values_the_MMA_will_contract(
    payload: torch.Tensor, geometry: dict[str, int], k_group: int
) -> None:
    """Chunk ``g`` of a slab must be K values ``[32g, 32g + 32)``.

    ``chunk_descriptor(g)`` advances 32 bytes inside a 128-byte swizzle atom and
    a whole atom column between atoms, so chunk ``g`` is shared columns
    ``[32g, 32g + 32)``: two atoms, sixteen live packed bytes, thirty-two FP4
    values. This is the mapping the sixteen contractions of a slab rest on, so
    it is asserted against the source row rather than left implied by the
    whole-tile comparison.
    """
    assert geometry["slab_groups"] == 16
    expert, task_slab = 3, 17
    image = _landed(payload, expert, task_slab, geometry)
    rows = KIMI_K3_W13_TASK_ROWS
    source = payload[expert, task_slab * rows : (task_slab + 1) * rows, :].cpu()
    for atom in range(2):
        column = k_group * 32 + atom * 16
        packed = k_group * 16 + atom * LIVE_BYTES_PER_ATOM
        assert torch.equal(
            image[:, column : column + LIVE_BYTES_PER_ATOM],
            source[:, packed : packed + LIVE_BYTES_PER_ATOM],
        )


# ---------------------------------------------------------------------------
# What must not arrive.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expert", "task_slab"),
    [
        (0, 0),
        (KIMI_K3_W13_PACKED_SHAPE[0] - 1, KIMI_K3_W13_TASK_SLABS - 1),
    ],
)
def test_the_container_padding_stays_as_the_probe_cleared_it(
    payload: torch.Tensor,
    geometry: dict[str, int],
    expert: int,
    task_slab: int,
) -> None:
    """Half of every shared atom is padding, and nothing may land in it.

    The probe zeroes the tile before issuing the transfer, and the payload is
    random, so a byte in a padding slot could only have come from somewhere
    else in the payload -- a wrong stride, a wrong box dimension, or a swizzle
    the descriptor and ``chunk_descriptor`` disagree about. Every one of those
    would show up here as a non-zero padding slot rather than as wrong decode
    numbers.
    """
    image = _landed(payload, expert, task_slab, geometry)
    _, live = _expected(payload, expert, task_slab)
    inactive = image[~live]
    assert inactive.numel() == image.numel() // 2
    assert int(inactive.max()) == 0, (
        f"{int((inactive != 0).sum())} of {inactive.numel()} padding bytes of "
        f"expert {expert} tile {task_slab} were written by the transfer"
    )


def test_the_transfer_accounts_for_every_byte_of_the_tile_and_no_others(
    payload: torch.Tensor, geometry: dict[str, int]
) -> None:
    """The live bytes and the padding are the whole tile, and the tile is the
    whole dump.

    Together with the padding staying zero, this is the in-bounds statement the
    probe can make: the dump is exactly one tile's shared footprint, the live
    half is exactly the tile's own source rows, and the other half is
    untouched. Nothing arrived outside the region the transform placed, and
    nothing from outside that region arrived inside it.
    """
    expert, task_slab = (
        KIMI_K3_W13_PACKED_SHAPE[0] - 1,
        KIMI_K3_W13_TASK_SLABS - 1,
    )
    image = _landed(payload, expert, task_slab, geometry)
    expected, live = _expected(payload, expert, task_slab)

    assert image.numel() == geometry["weight_tile_bytes"]
    assert int(live.sum()) == geometry["weight_transaction_bytes"]
    assert int((~live).sum()) == geometry["weight_transaction_bytes"]
    # `_expected` leaves the padding slots at zero, so comparing the whole image
    # asserts both halves at once.
    assert torch.equal(image, expected)


def test_a_probe_outside_the_payload_is_refused_before_it_launches(
    payload: torch.Tensor, geometry: dict[str, int]
) -> None:
    """The entrypoint's own bounds, so an out-of-range probe cannot read wild.

    The descriptor's outer two dimensions are the whole tensor, so an expert or
    a ``(task, slab)`` past the end is an out-of-bounds global read with nothing
    between it and the device. It is rejected on the host instead.
    """
    experts = KIMI_K3_W13_PACKED_SHAPE[0]
    transaction = geometry["weight_transaction_bytes"]
    for expert, task_slab in (
        (experts, 0),
        (-1, 0),
        (0, KIMI_K3_W13_TASK_SLABS),
        (0, -1),
    ):
        with pytest.raises(RuntimeError, match="requires (expert|task_slab)"):
            _C._kimi_k3_fused_w13_tma_probe(
                payload, expert, task_slab, transaction
            )
    for transaction_bytes in (0, geometry["weight_tile_bytes"] + 1):
        with pytest.raises(RuntimeError, match="transaction_bytes"):
            _C._kimi_k3_fused_w13_tma_probe(payload, 0, 0, transaction_bytes)


def test_the_probe_is_the_descriptor_production_launches_with(
    geometry: dict[str, int]
) -> None:
    """The measurement is only evidence if it is the production layout.

    The probe builds its descriptor through ``create_fused_w13_packed_map``,
    which is the one the persistent kernel's cached map is built by, so the
    engine's transaction count and the probe's are one constant and the shapes
    the probe accepts are the shapes the operator validates.
    """
    source = decode_sources.read("expert_mxfp4_fused_w13.cuh")
    probe = source.split("fused_w13_tma_probe_kernel")[1]
    assert "load_fused_slab_async(payload, &packed" in probe
    entrypoint = source.split("kimi_k3_fused_w13_tma_probe_entrypoint")[1]
    assert "create_fused_w13_packed_map(&packed" in entrypoint

    assert geometry["packed_rows"] == KIMI_K3_W13_PACKED_SHAPE[1]
    assert geometry["packed_columns"] == KIMI_K3_W13_PACKED_SHAPE[2]
    assert geometry["scale_rows"] == KIMI_K3_W13_SCALE_SHAPE[1]
    assert geometry["scale_columns"] == KIMI_K3_W13_SCALE_SHAPE[2]
