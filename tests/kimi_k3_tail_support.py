"""Shared fixtures, helpers, and layout constants for the Kimi K3 tail tests.

The tail's tests live in three files: ``test_kimi_k3_collectives.py`` covers
what the launch does on the device, ``test_kimi_k3_tail_contract.py`` covers
what the host boundary accepts and rejects, and
``test_kimi_k3_tail_signature.py`` covers the workspace signature that binds
one rank's three symmetric allocations together. They need the same eight-rank
workspace, the same replicated weights, and the same NCCL reference, so
everything they share is defined here and no file owns another's setup.

All three need all eight ranks, so all three must be launched through
``torchrun --standalone --nproc-per-node=8``. Each rank writes a *distinct*
routed and shared partial into its own symmetric collective buffer; the tail is
only correct when it reduces across ranks, so a rank-local implementation cannot
pass.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import pytest
import torch
import torch.distributed as dist

from mok import ops
from mok.kimi_k3 import (
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_MAX_TOKENS,
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
NUM_PHASE_COUNTERS = 64
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
    "error_flag",
    "tp_rank",
    "active_tokens",
    "workspace_signature",
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
        ("latent_x", MAX_TOKENS * LATENT * 2),
        ("unit_expert", 896 * 4),
        ("router_scores", MAX_TOKENS * 896 * 4),
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
        workspace.error_flag,
        workspace.tp_rank,
        active_tokens,
        workspace.workspace_signature,
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
        "error_flag": workspace.error_flag,
        "tp_rank": workspace.tp_rank,
        "active_tokens": active_tokens,
        "workspace_signature": workspace.workspace_signature,
    }


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
