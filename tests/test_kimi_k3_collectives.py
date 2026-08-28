"""TP8 GPU tests for the fused Kimi K3 latent-MoE tail.

Every test in this file needs all eight ranks, so the file must be launched
through ``torchrun --standalone --nproc-per-node=8``. Each rank writes a
*distinct* routed and shared partial into its own symmetric collective buffer;
the tail is only correct when it reduces across ranks, so a rank-local
implementation cannot pass.
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
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode

from mok import _C, _fake_impls, ops
from mok.kimi_k3 import (
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_MAX_TOKENS,
    KIMI_K3_RMS_EPS,
    KIMI_K3_TP_SIZE,
    KimiK3DecodeWorkspace,
    get_kimi_k3_decode_workspace,
    kimi_k3_rmsnorm_reference,
)


HIDDEN = KIMI_K3_HIDDEN_SIZE
LATENT = KIMI_K3_LATENT_SIZE
COLLECTIVE_COLUMNS = LATENT + HIDDEN
SHARD = HIDDEN // KIMI_K3_TP_SIZE
MAX_TOKENS = KIMI_K3_MAX_TOKENS
ALIGNMENT = 256
NUM_PHASE_COUNTERS = 32
UINT32 = 1 << 32
UINT32_MAX = UINT32 - 1
UINT64_MAX = (1 << 64) - 1

# Rank-and-column coding for the bit-exact reduction probe. Each rank raises one
# contiguous band of columns, so omitting or substituting a rank changes the
# *shape* of the reduced row and not only its scale -- which matters because
# RMSNorm divides any uniform scale change straight back out.
LATENT_BAND = LATENT // KIMI_K3_TP_SIZE
SHARED_BAND = SHARD // KIMI_K3_TP_SIZE

# Every capacity bucket, plus off-bucket counts that exercise partially filled
# reduce CTAs and the top of each bucket.
TAIL_ACTIVE_ROWS = (1, 2, 3, 4, 5, 8, 9, 16, 32, 33, 64, 127, 128)
# The counts the task brief pins for the NCCL comparison.
REFERENCE_ACTIVE_ROWS = (1, 5, 16, 64, 128)

_TAIL_ARGUMENTS = (
    "routed_latent_rmsnorm_weight",
    "latent_up_proj",
    "collective_buffer",
    "collective_buffer_ptrs",
    "collective_buffer_multicast_ptr",
    "output_mailbox",
    "output_mailbox_ptrs",
    "output_mailbox_multicast_ptr",
    "barrier_buffer",
    "barrier_buffer_ptrs",
    "barrier_buffer_multicast_ptr",
    "barrier_target",
    "scratch",
    "tp_rank",
    "active_tokens",
)

# Phase-counter slots the tail owns, mirrored from
# ``csrc/kimi_k3_decode/types.cuh`` so a silent renumbering is caught here.
TAIL_ENTRY_GENERATION = 18
TAIL_REDUCE_ARRIVALS = 19
TAIL_REDUCE_GENERATION = 20
TAIL_SHARD_ARRIVALS = 21
TAIL_SHARD_GENERATION = 22
TAIL_EXIT_GENERATION = 23
TAIL_DRAIN_ARRIVALS = 24
TAIL_DRAIN_GENERATION = 25
TAIL_TIMEOUT_PHASE = 26
TAIL_GENERATIONS = (
    TAIL_ENTRY_GENERATION,
    TAIL_REDUCE_GENERATION,
    TAIL_SHARD_GENERATION,
    TAIL_EXIT_GENERATION,
    TAIL_DRAIN_GENERATION,
)
TAIL_ARRIVALS = (
    TAIL_REDUCE_ARRIVALS,
    TAIL_SHARD_ARRIVALS,
    TAIL_DRAIN_ARRIVALS,
)


def _aligned(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _scratch_layout() -> dict[str, tuple[int, int]]:
    """Independent byte model of the C++ source-of-truth workspace."""
    regions = (
        ("phase", NUM_PHASE_COUNTERS * 4),
        ("expert_ids", MAX_TOKENS * 16 * 4),
        ("expert_weights", MAX_TOKENS * 16 * 4),
        ("expert_counts", 896 * 4),
        ("expert_offsets", 897 * 4),
        ("assignment_tokens", MAX_TOKENS * 16 * 4),
        ("assignment_slots", MAX_TOKENS * 16 * 4),
        ("latent_mxfp8", MAX_TOKENS * LATENT),
        ("latent_scale", MAX_TOKENS * (LATENT // 32)),
        ("situ_mxfp8", MAX_TOKENS * 16 * 384),
        ("situ_scale", MAX_TOKENS * 16 * (384 // 32)),
        ("routed_accumulator", MAX_TOKENS * LATENT * 4),
        ("shared_gate", MAX_TOKENS * 768 * 2),
        ("shared_up", MAX_TOKENS * 768 * 2),
        ("shared_activated", MAX_TOKENS * 768 * 2),
        ("tail_normalized", MAX_TOKENS * LATENT * 2),
        ("tail_shared_shard", MAX_TOKENS * SHARD * 2),
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


def _region(
    scratch: torch.Tensor, name: str, dtype: torch.dtype
) -> torch.Tensor:
    offset, size = SCRATCH_LAYOUT[name]
    return scratch[offset:offset + size].view(dtype)


def _phase(scratch: torch.Tensor) -> torch.Tensor:
    return _region(scratch, "phase", torch.int32)


@pytest.fixture(scope="module")
def workspace(
    tp8_context: tuple[int, int, torch.device],
) -> Iterator[KimiK3DecodeWorkspace]:
    _, _, device = tp8_context
    created = get_kimi_k3_decode_workspace(dist.group.WORLD, device=device)
    try:
        yield created
    finally:
        _synchronize_ranks(created)


def _synchronize_ranks(workspace: KimiK3DecodeWorkspace) -> None:
    """Rendezvous every rank without leaving stale symmetric state behind."""
    dist.barrier(
        group=dist.group.WORLD,
        async_op=True,
        device_ids=[workspace.device.index],
    ).block_current_stream()
    torch.cuda.synchronize(workspace.device)


def _draw(
    shape: tuple[int, ...],
    device: torch.device,
    seed: int,
    scale: float,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            shape, generator=generator, dtype=torch.float32, device=device
        )
        * scale
    ).bfloat16().contiguous()


@pytest.fixture(scope="module")
def norm_weight(tp8_context: tuple[int, int, torch.device]) -> torch.Tensor:
    """Replicated RMSNorm weight; every rank must draw the same values."""
    _, _, device = tp8_context
    return (
        1.0 + 0.25 * _draw((LATENT,), device, 9001, 1.0).float()
    ).bfloat16().contiguous()


@pytest.fixture(scope="module")
def latent_up(tp8_context: tuple[int, int, torch.device]) -> torch.Tensor:
    """Replicated latent-up weight [7168, 3584]; identical on every rank."""
    _, _, device = tp8_context
    return _draw((HIDDEN, LATENT), device, 9002, 1.0 / math.sqrt(LATENT))


def _assert_replicated(name: str, tensor: torch.Tensor) -> None:
    """Fail loudly if a replicated fixture drifted between ranks."""
    gathered = tensor.float().clone()
    dist.all_reduce(gathered, op=dist.ReduceOp.MAX)
    assert torch.equal(gathered, tensor.float()), name


def _partials(
    device: torch.device,
    tp_rank: int,
    rows: int,
    seed: int,
    *,
    routed_scale: float = 0.6,
    shared_scale: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw this rank's distinct routed and shared partial contributions."""
    routed = _draw(
        (rows, LATENT), device, seed * 131 + tp_rank, routed_scale
    )
    shared = _draw(
        (rows, HIDDEN), device, seed * 977 + 17 + tp_rank, shared_scale
    )
    return routed, shared


def _load_partials(
    workspace: KimiK3DecodeWorkspace,
    routed: torch.Tensor,
    shared: torch.Tensor,
) -> None:
    rows = routed.shape[0]
    workspace.collective_buffer[:rows, :LATENT].copy_(routed)
    workspace.collective_buffer[:rows, LATENT:].copy_(shared)


def _all_reduced(partial: torch.Tensor) -> torch.Tensor:
    """NCCL all-reduce of one BF16 partial, kept in BF16 like the device path."""
    reduced = partial.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def _reference(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reference normalized latent and full BF16 [M, 7168] output."""
    routed_full = _all_reduced(routed_partial)
    shared_full = _all_reduced(shared_partial)
    normalized = kimi_k3_rmsnorm_reference(routed_full, norm_weight)
    output = (
        normalized.float() @ latent_up.float().T + shared_full.float()
    ).bfloat16()
    return normalized, output


def _call(
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    return ops._kimi_k3_tail(
        norm_weight,
        latent_up,
        workspace.collective_buffer,
        workspace.collective_ptrs,
        workspace.collective_multicast_ptr,
        workspace.output_mailbox,
        workspace.output_mailbox_ptrs,
        workspace.output_mailbox_multicast_ptr,
        workspace.barrier_buffer,
        workspace.barrier_ptrs,
        workspace.barrier_multicast_ptr,
        workspace.barrier_target,
        workspace.scratch,
        workspace.tp_rank,
        active_tokens,
    )


def _accuracy_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    relative_l1 = (
        difference.abs().sum() / expected_float.abs().sum().clamp_min(1e-12)
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    )
    return (
        float(relative_l1),
        float(cosine),
        float(difference.abs().max()),
    )


def _assert_tail_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    relative_l1, cosine, max_abs = _accuracy_metrics(actual, expected)
    # The device path reduces eight BF16 partials with one FP32-accumulating
    # multimem instruction while the reference reduces them through NCCL, so the
    # two disagree by a few BF16 units in the last place. One ULP grows with the
    # magnitude of the result, which is why the absolute bound is scaled by the
    # largest expected value instead of being pinned to a constant.
    tolerance = 0.03 * float(expected.float().abs().max()) + 0.0625
    assert torch.isfinite(actual.float()).all()
    assert relative_l1 <= 0.03, (relative_l1, cosine, max_abs)
    assert cosine >= 0.999, (relative_l1, cosine, max_abs)
    assert max_abs <= tolerance, (relative_l1, cosine, max_abs, tolerance)


def _assert_identical_across_ranks(mailbox_view: torch.Tensor) -> None:
    """Every rank must leave byte-identical mailbox contents behind."""
    values = mailbox_view.float().contiguous()
    minimum = values.clone()
    maximum = values.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    assert torch.equal(minimum, maximum)
    assert torch.equal(minimum, values)


def _as_int32(value: int) -> int:
    """Reinterpret an unsigned 32-bit serial number as int32 for `fill_`."""
    value &= UINT32_MAX
    return value - UINT32 if value >= (1 << 31) else value


def _as_uint32(value: int) -> int:
    return value & UINT32_MAX


def _serial_reached(observed: int, target: int) -> bool:
    """Mirror the device's wrap-safe monotonic-counter comparison."""
    return (_as_uint32(observed) - _as_uint32(target)) % UINT32 < (1 << 31)


def _barrier_all(workspace: KimiK3DecodeWorkspace) -> None:
    ops.barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_ptrs,
        workspace.barrier_multicast_ptr,
        workspace.barrier_target,
    )


def _prime_barrier_serial(
    workspace: KimiK3DecodeWorkspace, start: int
) -> None:
    """Park the shared barrier pair on `start` on every rank.

    The pair's only invariant is that a rank's private target equals the number
    of arrivals its symmetric counter has seen, so presetting both to the same
    value on every rank is consistent -- and putting that value just below the
    unsigned wrap makes the very next rendezvous cross it.

    Two rendezvous precede the write because a rank may have left an earlier
    barrier before its peers' increments landed in its counter; the write has to
    happen after every one of those increments, or it would be overwritten.
    """
    _synchronize_ranks(workspace)
    _synchronize_ranks(workspace)
    workspace.barrier_buffer.fill_(_as_int32(start))
    workspace.barrier_target.fill_(_as_int32(start))
    _synchronize_ranks(workspace)


def _rotating_skew(rank: int, step: int, clocks: int = 1 << 22) -> None:
    """Delay this rank's stream so every rank leads and trails in turn."""
    lag = ((rank + step) % KIMI_K3_TP_SIZE) * clocks
    if lag:
        torch.cuda._sleep(lag)


def _coded_partials(
    device: torch.device, tp_rank: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-and-column-coded partials whose eight-way sum is exact in BF16.

    Rank ``r`` contributes ``(1 + r)`` to every column and doubles that on the
    one band of columns it owns. Every addend is an integer multiple of a power
    of two and the running sum never exceeds 44, so all eight partial sums are
    exact in BF16 with room to spare.
    """
    latent_band = (
        torch.arange(LATENT, device=device) // LATENT_BAND
    ) == tp_rank
    routed = (
        (1 + tp_rank) * (1 + latent_band.float()) / 64.0
    ).bfloat16().expand(rows, LATENT).contiguous()
    shared_band = (
        (torch.arange(HIDDEN, device=device) // SHARED_BAND)
        % KIMI_K3_TP_SIZE
    ) == tp_rank
    shared = (
        (1 + tp_rank) * (1 + shared_band.float()) / 128.0
    ).bfloat16().expand(rows, HIDDEN).contiguous()
    return routed, shared


def _coded_reduction(
    device: torch.device, columns: int, band: int, scale: float
) -> torch.Tensor:
    """The exact eight-way sum of `_coded_partials` for one column extent."""
    index = (torch.arange(columns, device=device) // band) % KIMI_K3_TP_SIZE
    ranks = torch.arange(KIMI_K3_TP_SIZE, device=device)
    total = float((1 + ranks).sum())
    return ((total + 1 + index.float()) / scale).bfloat16()


def test_tail_scratch_layout_matches_the_compiled_source_of_truth(
    workspace: KimiK3DecodeWorkspace,
) -> None:
    assert SCRATCH_BYTES == 4_896_256
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    assert workspace.scratch.numel() == SCRATCH_BYTES
    assert SCRATCH_LAYOUT["tail_normalized"] == (3_749_376, 917_504)
    assert SCRATCH_LAYOUT["tail_shared_shard"] == (4_666_880, 229_376)
    for name, (offset, _) in SCRATCH_LAYOUT.items():
        if name != "total_bytes":
            assert offset % ALIGNMENT == 0, name


def test_mailbox_is_token_major_with_a_contiguous_flat_view(
    workspace: KimiK3DecodeWorkspace,
) -> None:
    mailbox = workspace.output_mailbox
    assert mailbox.shape == (MAX_TOKENS, KIMI_K3_TP_SIZE, SHARD)
    assert mailbox.dtype == torch.bfloat16
    assert mailbox.is_contiguous()
    flat = mailbox.view(MAX_TOKENS, HIDDEN)
    assert flat.data_ptr() == mailbox.data_ptr()
    assert flat.shape == (MAX_TOKENS, HIDDEN)
    assert workspace.output_mailbox_multicast_ptr > 0
    assert len(workspace.output_mailbox_ptrs) == KIMI_K3_TP_SIZE


@pytest.mark.parametrize("active_tokens", TAIL_ACTIVE_ROWS)
def test_every_capacity_bucket_matches_the_nccl_reference(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    active_tokens: int,
) -> None:
    rank, _, device = tp8_context
    routed, shared = _partials(
        device, rank, active_tokens, 3100 + active_tokens
    )
    _load_partials(workspace, routed, shared)
    _, expected = _reference(routed, shared, norm_weight, latent_up)

    actual = _call(workspace, norm_weight, latent_up, active_tokens)

    assert actual.shape == (active_tokens, HIDDEN)
    assert actual.dtype == torch.bfloat16
    _assert_tail_close(actual, expected)
    _assert_identical_across_ranks(actual)


@pytest.mark.parametrize("active_tokens", REFERENCE_ACTIVE_ROWS)
def test_normalized_latent_matches_the_fp32_rmsnorm_reference(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    active_tokens: int,
) -> None:
    rank, _, device = tp8_context
    routed, shared = _partials(
        device, rank, active_tokens, 3200 + active_tokens
    )
    _load_partials(workspace, routed, shared)
    expected_normalized, expected = _reference(
        routed, shared, norm_weight, latent_up
    )

    actual = _call(workspace, norm_weight, latent_up, active_tokens)

    normalized = _region(
        workspace.scratch, "tail_normalized", torch.bfloat16
    ).view(MAX_TOKENS, LATENT)[:active_tokens]
    torch.testing.assert_close(
        normalized.float(), expected_normalized.float(), rtol=0.02, atol=0.02
    )
    # A missing epsilon or a mean over the wrong extent still lands close to the
    # reference in aggregate, so pin the per-row RMS scale too.
    routed_full = _all_reduced(routed).float()
    scale = torch.rsqrt(
        routed_full.square().mean(-1, keepdim=True) + KIMI_K3_RMS_EPS
    )
    torch.testing.assert_close(
        normalized.float(),
        ((routed_full * scale).bfloat16() * norm_weight).float(),
        rtol=0.02,
        atol=0.02,
    )
    _assert_tail_close(actual, expected)

    shard = _region(
        workspace.scratch, "tail_shared_shard", torch.bfloat16
    ).view(MAX_TOKENS, SHARD)[:active_tokens]
    expected_shard = _all_reduced(shared)[
        :, rank * SHARD:(rank + 1) * SHARD
    ]
    torch.testing.assert_close(
        shard.float(), expected_shard.float(), rtol=0.02, atol=0.02
    )


def test_each_rank_owns_its_shard_columns_in_rank_order(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """The mailbox slot index must be the rank, and its columns its W rows."""
    rank, _, device = tp8_context
    active_tokens = 16
    routed, shared = _partials(device, rank, active_tokens, 3300)
    _load_partials(workspace, routed, shared)
    normalized, expected = _reference(
        routed, shared, norm_weight, latent_up
    )

    actual = _call(workspace, norm_weight, latent_up, active_tokens)

    own_rows = latent_up[rank * SHARD:(rank + 1) * SHARD]
    own_shared = _all_reduced(shared)[:, rank * SHARD:(rank + 1) * SHARD]
    expected_shard = (
        normalized.float() @ own_rows.float().T + own_shared.float()
    ).bfloat16()
    slot = workspace.output_mailbox[:active_tokens, rank, :]
    _assert_tail_close(slot, expected_shard)
    torch.testing.assert_close(
        slot.float(),
        actual[:, rank * SHARD:(rank + 1) * SHARD].float(),
        rtol=0,
        atol=0,
    )
    # A shard written to the wrong slot would still assemble into something
    # finite, so check a neighbouring slot belongs to its own rank's rows.
    peer = (rank + 3) % KIMI_K3_TP_SIZE
    peer_expected = (
        normalized.float()
        @ latent_up[peer * SHARD:(peer + 1) * SHARD].float().T
        + _all_reduced(shared)[:, peer * SHARD:(peer + 1) * SHARD].float()
    ).bfloat16()
    _assert_tail_close(
        workspace.output_mailbox[:active_tokens, peer, :], peer_expected
    )
    assert (
        float(
            (expected_shard.float() - peer_expected.float()).abs().max()
        )
        > 0.25
    )
    _assert_tail_close(actual, expected)


def test_inactive_mailbox_rows_are_left_untouched(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, device = tp8_context
    active_tokens = 5
    sentinel = 7.5
    workspace.output_mailbox.fill_(sentinel)
    routed, shared = _partials(device, rank, MAX_TOKENS, 3400)
    _load_partials(workspace, routed, shared)
    _, expected = _reference(
        routed[:active_tokens],
        shared[:active_tokens],
        norm_weight,
        latent_up,
    )

    actual = _call(workspace, norm_weight, latent_up, active_tokens)

    _assert_tail_close(actual, expected)
    inactive = workspace.output_mailbox[active_tokens:]
    assert torch.equal(inactive, torch.full_like(inactive, sentinel))


@pytest.mark.parametrize(
    ("routed_scale", "shared_scale", "active_tokens"),
    [(0.0, 0.0, 8), (0.0, 0.5, 16), (0.5, 0.0, 16), (48.0, 48.0, 16)],
)
def test_zero_and_large_partials_stay_exact_and_finite(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    routed_scale: float,
    shared_scale: float,
    active_tokens: int,
) -> None:
    rank, _, device = tp8_context
    routed, shared = _partials(
        device,
        rank,
        active_tokens,
        3500 + int(routed_scale) + int(shared_scale),
        routed_scale=routed_scale,
        shared_scale=shared_scale,
    )
    _load_partials(workspace, routed, shared)
    _, expected = _reference(routed, shared, norm_weight, latent_up)

    actual = _call(workspace, norm_weight, latent_up, active_tokens)

    assert torch.isfinite(actual.float()).all()
    if routed_scale == 0.0 and shared_scale == 0.0:
        assert torch.equal(actual, torch.zeros_like(actual))
    else:
        _assert_tail_close(actual, expected)
    _assert_identical_across_ranks(actual)


def _normalized_direction(bands: torch.Tensor) -> torch.Tensor:
    """RMS-normalize one expanded latent row, dropping any uniform scale."""
    row = bands.float()
    return row * torch.rsqrt(row.square().mean() + KIMI_K3_RMS_EPS)


def test_rank_coded_partials_reduce_exactly_and_pin_the_rank_set(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """Pin the collective with values whose BF16 sums are exact.

    A rank-uniform probe cannot detect a missing or duplicated rank downstream
    of RMSNorm, because dropping one rank only rescales the row and RMSNorm
    divides that back out. These partials give every rank its own band of
    columns, so the rank set is recoverable from the *direction* of the reduced
    row. The first half of the test proves the coding really is
    direction-sensitive; the second half then requires the device to reproduce
    that direction, and the shared shard bit for bit.
    """
    rank, _, device = tp8_context
    active_tokens = 32
    ranks = torch.arange(KIMI_K3_TP_SIZE, device=device)
    contribution = (1 + ranks).float()
    band_index = (
        torch.arange(LATENT, device=device) // LATENT_BAND
    )
    true_bands = contribution.sum() + 1.0 + band_index.float()
    truth = _normalized_direction(true_bands)

    for dropped in range(KIMI_K3_TP_SIZE):
        omitted = (
            contribution.sum()
            - contribution[dropped]
            + 1.0
            + band_index.float()
            - torch.where(
                band_index == dropped, contribution[dropped], 0.0
            )
        )
        assert float((truth - _normalized_direction(omitted)).abs().max()) > (
            0.01
        ), dropped
        substituted = (
            contribution.sum()
            - contribution[dropped]
            + contribution[(dropped + 1) % KIMI_K3_TP_SIZE]
            + 1.0
            + band_index.float()
            - torch.where(
                band_index == dropped, contribution[dropped], 0.0
            )
            + torch.where(
                band_index == (dropped + 1) % KIMI_K3_TP_SIZE,
                contribution[(dropped + 1) % KIMI_K3_TP_SIZE],
                0.0,
            )
        )
        assert float(
            (truth - _normalized_direction(substituted)).abs().max()
        ) > 0.01, dropped

    routed, shared = _coded_partials(device, rank, active_tokens)
    _load_partials(workspace, routed, shared)

    _call(workspace, norm_weight, latent_up, active_tokens)

    # The shard is a plain eight-way sum with no normalization, so it must be
    # bit-identical to the exact expected value.
    reduced_shard = _region(
        workspace.scratch, "tail_shared_shard", torch.bfloat16
    ).view(MAX_TOKENS, SHARD)[:active_tokens]
    expected_shard = _coded_reduction(
        device, HIDDEN, SHARED_BAND, 128.0
    )[rank * SHARD:(rank + 1) * SHARD].expand(active_tokens, SHARD)
    assert torch.equal(reduced_shard, expected_shard)
    # Eight distinct plateaus inside this rank's own 896 columns, so a shifted
    # or wrong-shard read cannot coincide with the expected values.
    assert len(set(expected_shard[0].tolist())) == KIMI_K3_TP_SIZE

    normalized = _region(
        workspace.scratch, "tail_normalized", torch.bfloat16
    ).view(MAX_TOKENS, LATENT)[:active_tokens]
    reduced_routed = _coded_reduction(
        device, LATENT, LATENT_BAND, 64.0
    ).expand(active_tokens, LATENT)
    # The reduced latent is exact here, so only the FP32 reciprocal square root
    # can still differ; allow the two BF16 units in the last place that costs.
    torch.testing.assert_close(
        normalized,
        kimi_k3_rmsnorm_reference(reduced_routed, norm_weight),
        rtol=2 / 256,
        atol=0,
    )


def test_reused_workspace_advances_every_tail_generation(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, device = tp8_context
    observed: list[tuple[int, ...]] = []
    for step, active_tokens in enumerate((8, 3, 16, 5, 64, 1, 128)):
        routed, shared = _partials(
            device, rank, active_tokens, 3600 + step
        )
        _load_partials(workspace, routed, shared)
        _region(
            workspace.scratch, "tail_normalized", torch.bfloat16
        ).fill_(123.0)
        _region(
            workspace.scratch, "tail_shared_shard", torch.bfloat16
        ).fill_(-123.0)
        _, expected = _reference(routed, shared, norm_weight, latent_up)

        actual = _call(workspace, norm_weight, latent_up, active_tokens)

        _assert_tail_close(actual, expected)
        _assert_identical_across_ranks(actual)
        phase = _phase(workspace.scratch)
        for arrivals in TAIL_ARRIVALS:
            assert int(phase[arrivals]) == 0
        assert int(phase[TAIL_TIMEOUT_PHASE]) == 0
        observed.append(
            tuple(int(phase[slot]) for slot in TAIL_GENERATIONS)
        )

    for index in range(1, len(observed)):
        for slot in range(len(TAIL_GENERATIONS)):
            assert observed[index][slot] == observed[0][slot] + index


def test_tail_generations_advance_across_uint32_wrap(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, device = tp8_context
    active_tokens = 20
    phase = _phase(workspace.scratch)
    for slot in TAIL_GENERATIONS:
        phase[slot] = -1
    routed, shared = _partials(device, rank, active_tokens, 3700)
    _load_partials(workspace, routed, shared)
    _, expected = _reference(routed, shared, norm_weight, latent_up)

    actual = _call(workspace, norm_weight, latent_up, active_tokens)

    _assert_tail_close(actual, expected)
    assert [int(phase[slot]) for slot in TAIL_GENERATIONS] == [0] * len(
        TAIL_GENERATIONS
    )


def test_stale_generations_and_missing_visibility_are_detected_under_skew(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """Runtime multi-rank stress with deliberate per-rank launch skew.

    Each replay changes every rank's partials, poisons the whole mailbox and
    both tail scratch regions, and delays ranks by different amounts so no rank
    can rely on arriving at the collective at the same time as its peers. A tail
    that read a stale collective generation, skipped the entry barrier, or
    exited before every peer slot landed would surface here as poison, as a
    previous replay's values, or as a cross-rank mismatch.
    """
    rank, _, device = tp8_context
    active_tokens = 24
    for step in range(24):
        poison = 64.0 if step % 2 == 0 else -64.0
        workspace.output_mailbox.fill_(poison)
        _region(
            workspace.scratch, "tail_normalized", torch.bfloat16
        ).fill_(poison)
        _region(
            workspace.scratch, "tail_shared_shard", torch.bfloat16
        ).fill_(poison)
        routed, shared = _partials(
            device, rank, active_tokens, 3800 + step
        )
        _load_partials(workspace, routed, shared)
        _, expected = _reference(routed, shared, norm_weight, latent_up)
        # Rotate which rank lags so every rank leads and trails at least once.
        lag = ((rank + step) % KIMI_K3_TP_SIZE) * (1 << 20)
        if lag:
            torch.cuda._sleep(lag)

        actual = _call(workspace, norm_weight, latent_up, active_tokens)

        _assert_tail_close(actual, expected)
        _assert_identical_across_ranks(actual)
        inactive = workspace.output_mailbox[active_tokens:]
        assert torch.equal(inactive, torch.full_like(inactive, poison))
        assert int(_phase(workspace.scratch)[TAIL_TIMEOUT_PHASE]) == 0


def test_graph_replay_needs_no_host_reset(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, device = tp8_context
    active_tokens = 16
    routed, shared = _partials(device, rank, active_tokens, 3900)
    _load_partials(workspace, routed, shared)
    _, expected = _reference(routed, shared, norm_weight, latent_up)

    # Warm up outside capture so no allocation or lazy init lands in the graph.
    _call(workspace, norm_weight, latent_up, active_tokens)
    _synchronize_ranks(workspace)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call(workspace, norm_weight, latent_up, active_tokens)
    _synchronize_ranks(workspace)

    replays = 1000
    workspace.output_mailbox.fill_(-11.0)
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize(device)

    actual = workspace.output_mailbox.view(MAX_TOKENS, HIDDEN)[:active_tokens]
    _assert_tail_close(actual, expected)
    _assert_identical_across_ranks(actual)
    phase = _phase(workspace.scratch)
    for arrivals in TAIL_ARRIVALS:
        assert int(phase[arrivals]) == 0
    assert int(phase[TAIL_TIMEOUT_PHASE]) == 0
    # Each replay must advance every generation exactly once with no host reset.
    generations = [int(phase[slot]) & 0xFFFFFFFF for slot in TAIL_GENERATIONS]
    assert len(set(generations)) == 1
    _synchronize_ranks(workspace)


# The core path needs no launch-time device state, but the tcgen05 path raises a
# per-device dynamic shared-memory cap on first use. Both device-placement tests
# therefore run at a tensor-path capacity as well, and the evidence run selects
# them into a fresh process so that first use happens here.
_DEVICE_PLACEMENT_ROWS = (5, 20)


@pytest.mark.parametrize("active_tokens", _DEVICE_PLACEMENT_ROWS)
def test_tail_uses_the_tensor_devices_current_stream(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    active_tokens: int,
) -> None:
    rank, _, device = tp8_context
    routed, shared = _partials(device, rank, active_tokens, 4000)
    _, expected = _reference(routed, shared, norm_weight, latent_up)
    side_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side_stream):
        workspace.collective_buffer.fill_(float("nan"))
        torch.cuda._sleep(1 << 26)
        _load_partials(workspace, routed, shared)
        actual = _call(workspace, norm_weight, latent_up, active_tokens)
    side_stream.synchronize()
    _synchronize_ranks(workspace)

    _assert_tail_close(actual, expected)


@pytest.mark.parametrize("active_tokens", _DEVICE_PLACEMENT_ROWS)
def test_tail_runs_on_the_workspace_device_when_another_is_current(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    active_tokens: int,
) -> None:
    """CUDAGuard must follow the tensors, not the ambient current device.

    At a tensor-path capacity this also pins the per-device shared-memory
    reservation: it has to be taken on the workspace's device, not on whichever
    device happened to be current when the operator was called.
    """
    rank, _, device = tp8_context
    if torch.cuda.device_count() < 2:
        pytest.skip("the current-device guard test needs two visible GPUs")
    routed, shared = _partials(device, rank, active_tokens, 4100)
    _load_partials(workspace, routed, shared)
    _, expected = _reference(routed, shared, norm_weight, latent_up)
    other = torch.device(
        "cuda", (device.index + 1) % torch.cuda.device_count()
    )

    torch.cuda.set_device(other)
    try:
        actual = _call(workspace, norm_weight, latent_up, active_tokens)
        torch.cuda.synchronize(device)
        assert torch.cuda.current_device() == other.index
    finally:
        torch.cuda.set_device(device)
    _synchronize_ranks(workspace)

    assert actual.device == device
    _assert_tail_close(actual, expected)


def test_role_plan_orders_producers_before_consumers(
    workspace: KimiK3DecodeWorkspace,
) -> None:
    for active_tokens, expected in (
        (1, (1, 32, 14, 47)),
        (8, (1, 32, 14, 47)),
        (16, (1, 32, 7, 40)),
        (128, (1, 32, 7, 40)),
    ):
        plan = _C._kimi_k3_tail_role_plan(active_tokens)
        assert plan == expected, active_tokens
        coordinator, reduce_ctas, shard_ctas, total = plan
        assert coordinator == 1
        assert reduce_ctas > 0
        assert shard_ctas > 0
        assert coordinator + reduce_ctas + shard_ctas == total


@pytest.mark.parametrize(
    ("active_tokens", "required_sms"), [(8, 47), (20, 40)]
)
def test_host_residency_guard_uses_the_selected_role_grid(
    workspace: KimiK3DecodeWorkspace,
    active_tokens: int,
    required_sms: int,
) -> None:
    _C._kimi_k3_tail_validate_residency(active_tokens, required_sms)
    with pytest.raises(
        RuntimeError,
        match=rf"requires all {required_sms} role CTAs.*{required_sms - 1} SMs",
    ):
        _C._kimi_k3_tail_validate_residency(active_tokens, required_sms - 1)


def test_generation_and_timeout_helpers_are_wrap_safe(
    workspace: KimiK3DecodeWorkspace,
) -> None:
    advanced = _C._kimi_k3_tail_generation_advanced
    assert not advanced(7, 7)
    assert advanced(8, 7)
    assert advanced(0, UINT32_MAX)
    assert not advanced(UINT32_MAX, 0)

    reached = _C._kimi_k3_tail_barrier_reached
    assert reached(8, 8)
    assert reached(9, 8)
    assert not reached(7, 8)
    assert reached(0, UINT32_MAX - 7)
    assert not reached(UINT32_MAX - 7, 0)

    timeout = _C._kimi_k3_tail_wait_timeout_clocks()
    timed_out = _C._kimi_k3_tail_wait_timed_out
    assert timeout > 0
    assert not timed_out(100, 100 + timeout - 1)
    assert timed_out(100, 100 + timeout)
    start = UINT64_MAX - timeout // 2
    assert timed_out(start, (start + timeout) & UINT64_MAX)

    assert _C._kimi_k3_tail_timeout_metadata() == (
        TAIL_TIMEOUT_PHASE,
        TAIL_ENTRY_GENERATION,
        TAIL_REDUCE_GENERATION,
        TAIL_SHARD_GENERATION,
        TAIL_EXIT_GENERATION,
    )
    # `barrier_all` drives the very same counter pair as the tail's two
    # cross-rank edges, so it has to hold its rendezvous to the same bound and
    # read the counter with the same wrap-safe comparison. Sharing the timeout
    # constant is the observable half of sharing the implementation.
    assert _C._barrier_all_wait_timeout_clocks() == timeout


def test_tail_custom_op_returns_none_and_declares_its_mutations() -> None:
    schema = torch.ops.mok._kimi_k3_tail.default._schema
    assert tuple(argument.name for argument in schema.arguments) == (
        _TAIL_ARGUMENTS
    )
    assert len(schema.returns) == 0
    mutated = {
        argument.name
        for argument in schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    }
    assert mutated == {
        "collective_buffer",
        "output_mailbox",
        "barrier_buffer",
        "barrier_target",
        "scratch",
    }
    # No custom-op return may alias a mutated input, so the schema must not
    # expose an aliasing output at all.
    assert all(
        returned.alias_info is None for returned in schema.returns
    )
    assert tuple(
        inspect.signature(_fake_impls._kimi_k3_tail_fake).parameters
    ) == _TAIL_ARGUMENTS
    assert tuple(
        inspect.signature(ops._kimi_k3_tail).parameters
    ) == _TAIL_ARGUMENTS


def test_tail_fake_traces_without_touching_the_device() -> None:
    with FakeTensorMode():
        mailbox = torch.empty(
            MAX_TOKENS,
            KIMI_K3_TP_SIZE,
            SHARD,
            dtype=torch.bfloat16,
            device="cuda",
        )
        actual = ops._kimi_k3_tail(
            torch.empty(LATENT, dtype=torch.bfloat16, device="cuda"),
            torch.empty(HIDDEN, LATENT, dtype=torch.bfloat16, device="cuda"),
            torch.empty(
                MAX_TOKENS,
                COLLECTIVE_COLUMNS,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            [1] * KIMI_K3_TP_SIZE,
            1,
            mailbox,
            [1] * KIMI_K3_TP_SIZE,
            1,
            torch.empty(1, dtype=torch.int32, device="cuda"),
            [1] * KIMI_K3_TP_SIZE,
            1,
            torch.empty(1, dtype=torch.int32, device="cuda"),
            torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda"),
            0,
            11,
        )

    assert actual.shape == (11, HIDDEN)
    assert actual.dtype == torch.bfloat16


def test_tail_helper_aliases_the_mailbox_without_allocating(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, device = tp8_context
    active_tokens = 9
    routed, shared = _partials(device, rank, active_tokens, 4200)
    _load_partials(workspace, routed, shared)
    before = torch.cuda.memory_allocated(device)

    actual = _call(workspace, norm_weight, latent_up, active_tokens)

    assert torch.cuda.memory_allocated(device) == before
    assert actual.data_ptr() == workspace.output_mailbox.data_ptr()
    assert actual.shape == (active_tokens, HIDDEN)
    assert actual.stride() == (HIDDEN, 1)
    assert actual._base is not None
    assert (
        torch.ops.mok._kimi_k3_tail(
            norm_weight,
            latent_up,
            workspace.collective_buffer,
            workspace.collective_ptrs,
            workspace.collective_multicast_ptr,
            workspace.output_mailbox,
            workspace.output_mailbox_ptrs,
            workspace.output_mailbox_multicast_ptr,
            workspace.barrier_buffer,
            workspace.barrier_ptrs,
            workspace.barrier_multicast_ptr,
            workspace.barrier_target,
            workspace.scratch,
            workspace.tp_rank,
            active_tokens,
        )
        is None
    )


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
    return view


# Every tensor argument the tail dereferences with vector or scratch-region
# arithmetic, the byte boundary it needs, an element offset that breaks it, and
# a nonzero element offset that preserves it.
_TAIL_TENSOR_CASES = (
    ("routed_latent_rmsnorm_weight", 16, 1, 8),
    ("latent_up_proj", 16, 1, 8),
    ("scratch", ALIGNMENT, 16, ALIGNMENT),
)


def _valid_arguments(
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    active_tokens: int,
) -> dict[str, object]:
    return {
        "routed_latent_rmsnorm_weight": norm_weight,
        "latent_up_proj": latent_up,
        "collective_buffer": workspace.collective_buffer,
        "collective_buffer_ptrs": workspace.collective_ptrs,
        "collective_buffer_multicast_ptr": workspace.collective_multicast_ptr,
        "output_mailbox": workspace.output_mailbox,
        "output_mailbox_ptrs": workspace.output_mailbox_ptrs,
        "output_mailbox_multicast_ptr": (
            workspace.output_mailbox_multicast_ptr
        ),
        "barrier_buffer": workspace.barrier_buffer,
        "barrier_buffer_ptrs": workspace.barrier_ptrs,
        "barrier_buffer_multicast_ptr": workspace.barrier_multicast_ptr,
        "barrier_target": workspace.barrier_target,
        "scratch": workspace.scratch,
        "tp_rank": workspace.tp_rank,
        "active_tokens": active_tokens,
    }


@pytest.mark.parametrize(
    ("field", "alignment", "element_offset", "_"), _TAIL_TENSOR_CASES
)
def test_tail_rejects_every_misaligned_offset_view(
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    field: str,
    alignment: int,
    element_offset: int,
    _: int,
) -> None:
    arguments = _valid_arguments(workspace, norm_weight, latent_up, 4)
    misaligned = _offset_copy(arguments[field], element_offset)
    assert misaligned.data_ptr() % alignment != 0
    arguments[field] = misaligned

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        ops._kimi_k3_tail(**arguments)


@pytest.mark.parametrize(
    ("field", "alignment", "element_offset", "_"), _TAIL_TENSOR_CASES
)
def test_tail_c_entrypoint_rejects_every_misaligned_offset_view(
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    field: str,
    alignment: int,
    element_offset: int,
    _: int,
) -> None:
    arguments = _valid_arguments(workspace, norm_weight, latent_up, 4)
    arguments[field] = _offset_copy(arguments[field], element_offset)

    with pytest.raises(RuntimeError, match=rf"{field}.*{alignment}"):
        _C._kimi_k3_tail(*(arguments[name] for name in _TAIL_ARGUMENTS))


@pytest.mark.parametrize(
    ("field", "alignment", "_", "element_offset"), _TAIL_TENSOR_CASES
)
def test_tail_accepts_every_positively_aligned_offset_view(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    field: str,
    alignment: int,
    _: int,
    element_offset: int,
) -> None:
    rank, _unused, device = tp8_context
    active_tokens = 4
    routed, shared = _partials(device, rank, active_tokens, 4300)
    _load_partials(workspace, routed, shared)
    arguments = _valid_arguments(
        workspace, norm_weight, latent_up, active_tokens
    )
    aligned = _offset_copy(arguments[field], element_offset)
    assert aligned.data_ptr() % alignment == 0
    arguments[field] = aligned
    if field == "scratch":
        aligned.zero_()
    weight = (
        aligned if field == "routed_latent_rmsnorm_weight" else norm_weight
    )
    up_projection = aligned if field == "latent_up_proj" else latent_up
    _, expected = _reference(routed, shared, weight, up_projection)

    actual = ops._kimi_k3_tail(**arguments)

    assert actual.data_ptr() == workspace.output_mailbox.data_ptr()
    _assert_tail_close(actual, expected)


def test_tail_rejects_invalid_shapes_pointers_and_counts(
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    arguments = _valid_arguments(workspace, norm_weight, latent_up, 4)
    with pytest.raises(
        RuntimeError, match=rf"routed_latent_rmsnorm_weight \[{LATENT}\]"
    ):
        ops._kimi_k3_tail(
            **{
                **arguments,
                "routed_latent_rmsnorm_weight": norm_weight[:-1].contiguous(),
            }
        )
    with pytest.raises(
        RuntimeError, match=rf"latent_up_proj \[{HIDDEN}, {LATENT}\]"
    ):
        ops._kimi_k3_tail(
            **{
                **arguments,
                "latent_up_proj": latent_up[:, :-1].contiguous(),
            }
        )
    with pytest.raises(
        RuntimeError,
        match=rf"collective_buffer \[{MAX_TOKENS}, {COLLECTIVE_COLUMNS}\]",
    ):
        ops._kimi_k3_tail(
            **{
                **arguments,
                "collective_buffer": (
                    workspace.collective_buffer[:-1].contiguous()
                ),
            }
        )
    with pytest.raises(
        RuntimeError,
        match=rf"output_mailbox \[{MAX_TOKENS}, {KIMI_K3_TP_SIZE}, {SHARD}\]",
    ):
        ops._kimi_k3_tail(
            **{
                **arguments,
                "output_mailbox": (
                    workspace.output_mailbox.view(MAX_TOKENS, HIDDEN)
                ),
            }
        )
    with pytest.raises(
        RuntimeError, match=rf"scratch.*at least {SCRATCH_BYTES} bytes"
    ):
        ops._kimi_k3_tail(
            **{**arguments, "scratch": workspace.scratch[:-ALIGNMENT]}
        )
    with pytest.raises(RuntimeError, match=r"active_tokens in \[1, 128\]"):
        ops._kimi_k3_tail(**{**arguments, "active_tokens": 0})
    with pytest.raises(RuntimeError, match=r"active_tokens in \[1, 128\]"):
        ops._kimi_k3_tail(**{**arguments, "active_tokens": 129})
    with pytest.raises(RuntimeError, match=r"tp_rank in \[0, 7\]"):
        ops._kimi_k3_tail(**{**arguments, "tp_rank": 8})
    for field in (
        "collective_buffer_ptrs",
        "output_mailbox_ptrs",
        "barrier_buffer_ptrs",
    ):
        with pytest.raises(
            RuntimeError, match=rf"{field}.*{KIMI_K3_TP_SIZE} pointers"
        ):
            ops._kimi_k3_tail(
                **{**arguments, field: list(arguments[field])[:-1]}
            )
        with pytest.raises(RuntimeError, match=rf"{field}.*positive"):
            ops._kimi_k3_tail(
                **{**arguments, field: [0] + list(arguments[field])[1:]}
            )
    for field in (
        "collective_buffer_multicast_ptr",
        "output_mailbox_multicast_ptr",
        "barrier_buffer_multicast_ptr",
    ):
        with pytest.raises(RuntimeError, match=rf"{field}.*positive"):
            ops._kimi_k3_tail(**{**arguments, field: 0})
    with pytest.raises(RuntimeError, match=r"barrier_buffer.*int32 \[1\]"):
        ops._kimi_k3_tail(
            **{
                **arguments,
                "barrier_buffer": workspace.barrier_buffer.view(torch.uint8),
            }
        )
    with pytest.raises(RuntimeError, match=r"barrier_target.*int32 \[1\]"):
        ops._kimi_k3_tail(
            **{
                **arguments,
                "barrier_target": workspace.barrier_target.view(torch.uint8),
            }
        )


# One row per symmetric allocation: the operator's tensor argument, its
# peer-pointer list, its multicast pointer, the matching workspace attribute
# names, and the byte boundary the device dereferences it on. The two BF16
# allocations are read with 16-byte multimem octets; the barrier is one int32.
_SYMMETRIC_FIELDS = (
    (
        "collective_buffer",
        "collective_buffer_ptrs",
        "collective_buffer_multicast_ptr",
        "collective_buffer",
        "collective_ptrs",
        "collective_multicast_ptr",
        16,
    ),
    (
        "output_mailbox",
        "output_mailbox_ptrs",
        "output_mailbox_multicast_ptr",
        "output_mailbox",
        "output_mailbox_ptrs",
        "output_mailbox_multicast_ptr",
        16,
    ),
    (
        "barrier_buffer",
        "barrier_buffer_ptrs",
        "barrier_buffer_multicast_ptr",
        "barrier_buffer",
        "barrier_ptrs",
        "barrier_multicast_ptr",
        4,
    ),
)


def _symmetric_facts(
    workspace: KimiK3DecodeWorkspace,
) -> list[tuple[str, str, torch.Tensor, list[int], int, int]]:
    """(list argument, multicast argument, tensor, ptrs, multicast, boundary)."""
    facts = []
    for (
        _,
        list_field,
        multicast_field,
        tensor_attribute,
        list_attribute,
        multicast_attribute,
        alignment,
    ) in _SYMMETRIC_FIELDS:
        facts.append((
            list_field,
            multicast_field,
            getattr(workspace, tensor_attribute),
            list(getattr(workspace, list_attribute)),
            int(getattr(workspace, multicast_attribute)),
            alignment,
        ))
    return facts


def test_symmetric_pointer_lists_match_the_live_handles(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """Positive validation of the real symmetric-memory handles.

    Everything the entry point enforces is asserted here against the pointers
    PyTorch actually handed back, so a check can never end up stricter than the
    API it guards.
    """
    rank, _, _ = tp8_context
    assert workspace.tp_rank == rank
    for (
        list_field,
        multicast_field,
        tensor,
        pointers,
        multicast,
        alignment,
    ) in _symmetric_facts(workspace):
        assert len(pointers) == KIMI_K3_TP_SIZE, list_field
        assert all(pointer > 0 for pointer in pointers), list_field
        assert pointers[rank] == tensor.data_ptr(), list_field
        assert len(set(pointers)) == KIMI_K3_TP_SIZE, list_field
        assert all(
            pointer % alignment == 0 for pointer in pointers
        ), list_field
        assert multicast > 0 and multicast % alignment == 0, multicast_field
        assert multicast not in pointers, multicast_field
    # The three allocations are distinct objects, so no pointer is shared
    # between them. That is exactly what makes a swapped list detectable.
    every_pointer = [
        pointer for _, _, _, pointers, _, _ in _symmetric_facts(workspace)
        for pointer in pointers
    ] + [
        multicast
        for _, _, _, _, multicast, _ in _symmetric_facts(workspace)
    ]
    assert len(set(every_pointer)) == len(every_pointer)
    # These pointers are what the entry point is about to be handed, so the
    # launch that follows is the positive half: the checks accept the live
    # handles and the tail still produces its normal result.
    ops._kimi_k3_tail(**_valid_arguments(workspace, norm_weight, latent_up, 1))
    _synchronize_ranks(workspace)


def _symmetric_rejection_cases(
    workspace: KimiK3DecodeWorkspace, rank: int
) -> list[tuple[str, dict[str, object], str]]:
    """Every pointer/topology substitution the entry point must reject."""
    facts = _symmetric_facts(workspace)
    arguments = {
        field: value
        for list_field, multicast_field, _, pointers, multicast, _ in facts
        for field, value in (
            (list_field, pointers), (multicast_field, multicast)
        )
    }
    peer = (rank + 1) % KIMI_K3_TP_SIZE
    cases: list[tuple[str, dict[str, object], str]] = [
        (
            "tp_rank is not this rank",
            {"tp_rank": peer},
            r"collective_buffer_ptrs\[tp_rank\]",
        ),
    ]
    for list_field, multicast_field, _, pointers, _, alignment in facts:
        substituted = list(pointers)
        substituted[rank] = pointers[peer]
        cases.append((
            f"{list_field} local entry replaced by a peer",
            {list_field: substituted},
            rf"{list_field}\[tp_rank\]",
        ))
        misaligned = list(pointers)
        # Perturb a *peer* slot, so the local-ownership check cannot mask the
        # alignment check.
        misaligned[peer] = pointers[peer] + 2
        cases.append((
            f"{list_field} peer entry misaligned",
            {list_field: misaligned},
            rf"{list_field} entry aligned to {alignment} bytes",
        ))
        cases.append((
            f"{multicast_field} misaligned",
            {multicast_field: arguments[multicast_field] + 2},
            rf"{multicast_field} aligned to {alignment} bytes",
        ))
        aliased = list(pointers)
        aliased[peer] = pointers[(peer + 1) % KIMI_K3_TP_SIZE]
        cases.append((
            f"{list_field} duplicates a peer entry",
            {list_field: aliased},
            rf"{list_field} to hold one distinct pointer per rank",
        ))
    # Distinct addresses that nonetheless name the same GPU. Only the driver's
    # pointer attributes can tell these apart, and only the collective buffer is
    # large enough that a 16-byte bump is unambiguously inside it.
    collective = list(arguments["collective_buffer_ptrs"])
    same_device = list(collective)
    same_device[(peer + 1) % KIMI_K3_TP_SIZE] = collective[peer] + 16
    if same_device[rank] == collective[rank]:
        cases.append((
            "collective_buffer_ptrs points two ranks at one device",
            {"collective_buffer_ptrs": same_device},
            r"collective_buffer_ptrs to hold one distinct device per rank",
        ))
    cases.append((
        "collective and mailbox lists swapped",
        {
            "collective_buffer_ptrs": list(arguments["output_mailbox_ptrs"]),
            "output_mailbox_ptrs": list(
                arguments["collective_buffer_ptrs"]
            ),
        },
        r"collective_buffer_ptrs\[tp_rank\]",
    ))
    cases.append((
        "barrier list substituted for the collective list",
        {"collective_buffer_ptrs": list(arguments["barrier_buffer_ptrs"])},
        r"collective_buffer_ptrs\[tp_rank\]",
    ))
    cases.append((
        "collective multicast reused as the mailbox multicast",
        {
            "output_mailbox_multicast_ptr": arguments[
                "collective_buffer_multicast_ptr"
            ]
        },
        r"one distinct multicast pointer per symmetric allocation",
    ))
    cases.append((
        "mailbox multicast reused as a mailbox unicast entry",
        {
            "output_mailbox_multicast_ptr": arguments[
                "output_mailbox_ptrs"
            ][peer]
        },
        r"one distinct multicast pointer per symmetric allocation",
    ))
    return cases


def _expect_rejection(
    label: str, pattern: str, call: Callable[[], object]
) -> None:
    """Require one rejection, naming the substitution that was not caught."""
    try:
        with pytest.raises(RuntimeError, match=pattern):
            call()
    except BaseException as failure:
        raise AssertionError(
            f"{label}: expected a RuntimeError matching /{pattern}/"
        ) from failure


def test_tail_rejects_substituted_symmetric_pointers(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, _ = tp8_context
    valid = _valid_arguments(workspace, norm_weight, latent_up, 4)
    for label, overrides, pattern in _symmetric_rejection_cases(
        workspace, rank
    ):
        arguments = {**valid, **overrides}
        _expect_rejection(
            f"python: {label}",
            pattern,
            lambda arguments=arguments: ops._kimi_k3_tail(**arguments),
        )


def test_tail_c_entrypoint_rejects_substituted_symmetric_pointers(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, _ = tp8_context
    valid = _valid_arguments(workspace, norm_weight, latent_up, 4)
    for label, overrides, pattern in _symmetric_rejection_cases(
        workspace, rank
    ):
        arguments = {**valid, **overrides}
        _expect_rejection(
            f"pybind: {label}",
            pattern,
            lambda arguments=arguments: _C._kimi_k3_tail(
                *(arguments[name] for name in _TAIL_ARGUMENTS)
            ),
        )


def test_barrier_all_stays_ordered_across_the_uint32_wrap(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
) -> None:
    """`barrier_all` must still rendezvous when its serial number wraps.

    The pair is re-parked before every round so that each rendezvous lands its
    target exactly on the wrap. A plain ``value < target`` poll cannot be
    satisfied by any counter at that point, so it falls straight through and the
    barrier silently stops synchronizing. The skew rotates so that every rank
    leads one round: only a rank that arrives first can tell the difference,
    because a rank that arrives last sees a full counter either way.

    The snapshot is enqueued on the same stream immediately after the barrier,
    so a barrier that returned early is caught holding a counter that had not
    yet reached its target.
    """
    rank, _, device = tp8_context
    start = UINT32 - KIMI_K3_TP_SIZE

    for step in range(KIMI_K3_TP_SIZE):
        _prime_barrier_serial(workspace, start)
        _rotating_skew(rank, step)
        _barrier_all(workspace)
        snapshot = workspace.barrier_buffer.clone()
        torch.cuda.synchronize(device)
        # start + 8 is exactly 2**32, so this round's target is zero.
        observed = int(snapshot.item())
        assert _serial_reached(observed, 0), (
            step, rank, _as_uint32(observed)
        )
        assert _as_uint32(int(workspace.barrier_target.item())) == 0
    _synchronize_ranks(workspace)


def test_barrier_all_and_tail_interleave_across_the_uint32_wrap(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """The tail and `barrier_all` share one counter pair across the wrap.

    Each step takes three rendezvous off the shared pair -- one for
    `barrier_all` and two for the tail's entry and exit edges -- so the pair
    crosses the unsigned wrap partway through the loop while both users are
    active. Every step then requires a fresh, correct, cross-rank-identical
    tail result and an exactly-advanced generation for each tail phase.
    """
    rank, _, device = tp8_context
    active_tokens = 20
    steps = 8
    per_step = 3 * KIMI_K3_TP_SIZE
    # Park the pair so the wrap happens inside the loop rather than at its edge.
    start = UINT32 - 2 * per_step - KIMI_K3_TP_SIZE
    _prime_barrier_serial(workspace, start)
    phase = _phase(workspace.scratch)
    previous = [int(phase[slot]) for slot in TAIL_GENERATIONS]

    for step in range(steps):
        poison = 96.0 if step % 2 == 0 else -96.0
        workspace.output_mailbox.fill_(poison)
        routed, shared = _partials(device, rank, active_tokens, 4700 + step)
        _load_partials(workspace, routed, shared)
        _, expected = _reference(routed, shared, norm_weight, latent_up)

        _rotating_skew(rank, step)
        _barrier_all(workspace)
        snapshot = workspace.barrier_buffer.clone()
        actual = _call(workspace, norm_weight, latent_up, active_tokens)
        torch.cuda.synchronize(device)

        target = start + step * per_step + KIMI_K3_TP_SIZE
        assert _serial_reached(int(snapshot.item()), target), (
            step, _as_uint32(int(snapshot.item())), _as_uint32(target)
        )
        _assert_tail_close(actual, expected)
        _assert_identical_across_ranks(actual)
        inactive = workspace.output_mailbox[active_tokens:]
        assert torch.equal(inactive, torch.full_like(inactive, poison))
        assert int(phase[TAIL_TIMEOUT_PHASE]) == 0
        for arrivals in TAIL_ARRIVALS:
            assert int(phase[arrivals]) == 0
        current = [int(phase[slot]) for slot in TAIL_GENERATIONS]
        for slot, (before, after) in enumerate(zip(previous, current)):
            assert _as_uint32(after) == _as_uint32(before + 1), (step, slot)
        previous = current

    assert _as_uint32(int(workspace.barrier_target.item())) == _as_uint32(
        start + steps * per_step
    )
    _synchronize_ranks(workspace)


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
    [(5, "kimi_k3_tail_core_kernel"), (20, "kimi_k3_tail_tensor_kernel")],
)
def test_tail_is_exactly_one_kernel_launch_per_rank(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    active_tokens: int,
    expected_kernel: str,
) -> None:
    rank, _, device = tp8_context
    routed, shared = _partials(
        device, rank, active_tokens, 4400 + active_tokens
    )
    _load_partials(workspace, routed, shared)

    def call() -> object:
        return _call(workspace, norm_weight, latent_up, active_tokens)

    call()
    _synchronize_ranks(workspace)
    names = _profiled_kernel_names(call)
    _synchronize_ranks(workspace)

    assert len(names) == 1, names
    assert expected_kernel in names[0], names
    if rank == 0:
        print(f"K3 tail M={active_tokens} launches={len(names)} {names[0]}")


def test_replicated_fixtures_and_partials_are_distinct_per_rank(
    tp8_context: tuple[int, int, torch.device],
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """Guard the fixture contract the whole file depends on."""
    rank, _, device = tp8_context
    _assert_replicated("routed_latent_rmsnorm_weight", norm_weight)
    _assert_replicated("latent_up_proj", latent_up)

    routed, shared = _partials(device, rank, 8, 4500)
    for name, partial in (("routed", routed), ("shared", shared)):
        summed = partial.float().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        assert not torch.allclose(
            summed, partial.float() * KIMI_K3_TP_SIZE, atol=1e-3
        ), name
    # A degenerate fixture that repeated along a row or a shard boundary would
    # hide a wrong-slot or wrong-stride bug, so probe a small prefix for
    # structure instead of materialising the whole 25M-element weight twice.
    for values in (routed.float().flatten()[:4096],
                   latent_up.float().flatten()[:4096]):
        for shift in (1, 8, 64, 896):
            assert float((values - values.roll(shift)).abs().mean()) > 1e-4


def test_accuracy_metrics_have_finite_worst_case(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, device = tp8_context
    metrics: list[tuple[float, float, float]] = []
    for active_tokens in REFERENCE_ACTIVE_ROWS:
        routed, shared = _partials(
            device, rank, active_tokens, 4600 + active_tokens
        )
        _load_partials(workspace, routed, shared)
        _, expected = _reference(routed, shared, norm_weight, latent_up)
        actual = _call(workspace, norm_weight, latent_up, active_tokens)
        _assert_tail_close(actual, expected)
        metrics.append(_accuracy_metrics(actual, expected))

    worst_rel_l1 = max(metric[0] for metric in metrics)
    worst_cosine = min(metric[1] for metric in metrics)
    worst_max_abs = max(metric[2] for metric in metrics)
    if rank == 0:
        print(
            "K3 tail worst "
            f"rel-L1={worst_rel_l1:.6f} "
            f"cosine={worst_cosine:.6f} "
            f"max-abs={worst_max_abs:.6f}"
        )
    assert math.isfinite(worst_rel_l1)
    assert math.isfinite(worst_cosine)
    assert math.isfinite(worst_max_abs)
