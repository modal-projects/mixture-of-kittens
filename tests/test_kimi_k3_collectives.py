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
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

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


def test_exactly_representable_partials_reduce_without_drift(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """Pin the collective itself with values BF16 sums represent exactly.

    Every partial is ``rank + 1`` scaled by a power of two, so the eight-way sum
    is exact in BF16 and the reduced latent must match bit for bit. This
    separates a reduction-order or missing-rank bug from BF16 rounding noise.
    """
    rank, _, device = tp8_context
    active_tokens = 32
    routed = torch.full(
        (active_tokens, LATENT),
        float(rank + 1) * 0.03125,
        dtype=torch.bfloat16,
        device=device,
    )
    shared = torch.full(
        (active_tokens, HIDDEN),
        float(rank + 1) * 0.0625,
        dtype=torch.bfloat16,
        device=device,
    )
    _load_partials(workspace, routed, shared)

    _call(workspace, norm_weight, latent_up, active_tokens)

    reduced_shard = _region(
        workspace.scratch, "tail_shared_shard", torch.bfloat16
    ).view(MAX_TOKENS, SHARD)[:active_tokens]
    assert torch.equal(
        reduced_shard,
        torch.full_like(reduced_shard, 36.0 * 0.0625),
    )
    normalized = _region(
        workspace.scratch, "tail_normalized", torch.bfloat16
    ).view(MAX_TOKENS, LATENT)[:active_tokens]
    reduced_routed = torch.full(
        (active_tokens, LATENT), 36.0 * 0.03125, dtype=torch.bfloat16,
        device=device,
    )
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


def test_tail_uses_the_tensor_devices_current_stream(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    rank, _, device = tp8_context
    active_tokens = 5
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


def test_tail_runs_on_the_workspace_device_when_another_is_current(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """CUDAGuard must follow the tensors, not the ambient current device."""
    rank, _, device = tp8_context
    if torch.cuda.device_count() < 2:
        pytest.skip("the current-device guard test needs two visible GPUs")
    active_tokens = 8
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
