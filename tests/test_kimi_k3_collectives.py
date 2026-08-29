"""TP8 GPU tests for what the fused Kimi K3 latent-MoE tail computes.

Numerics against an NCCL reference, mailbox ownership, generation safety across
reuse and graph replay, device and stream placement, and the single-launch
claim. The host-side contract -- what the boundary accepts and rejects -- lives
in ``test_kimi_k3_tail_contract.py``; the shared workspace, weights, and
reference reduction live in ``kimi_k3_tail_support.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

from mok import _C
from mok.kimi_k3 import (
    KIMI_K3_RMS_EPS,
    KIMI_K3_TP_SIZE,
    KimiK3DecodeWorkspace,
    kimi_k3_rmsnorm_reference,
)

from .kimi_k3_tail_support import (
    ALIGNMENT,
    HIDDEN,
    LATENT,
    LATENT_BAND,
    MAX_TOKENS,
    REFERENCE_ACTIVE_ROWS,
    SCRATCH_BYTES,
    SCRATCH_LAYOUT,
    SHARD,
    SHARED_BAND,
    TAIL_ACTIVE_ROWS,
    TAIL_ARRIVALS,
    TAIL_GENERATIONS,
    TAIL_TIMEOUT_PHASE,
    _accuracy_metrics,
    _all_reduced,
    _assert_identical_across_ranks,
    _assert_replicated,
    _assert_tail_close,
    _call,
    _coded_partials,
    _coded_reduction,
    _load_partials,
    _partials,
    _phase,
    _reference,
    _region,
    _synchronize_ranks,
    latent_up,
    norm_weight,
    workspace,
)


def test_tail_scratch_layout_matches_the_compiled_source_of_truth(
    workspace: KimiK3DecodeWorkspace,
) -> None:
    assert SCRATCH_BYTES == 8_111_104
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    assert workspace.scratch.numel() == SCRATCH_BYTES
    assert SCRATCH_LAYOUT["tail_normalized"] == (5_584_384, 917_504)
    assert SCRATCH_LAYOUT["tail_shared_shard"] == (6_501_888, 229_376)
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


def test_tail_reserves_tensor_shared_memory_cold_on_the_workspace_device(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """The once-per-device reservation must fire on the tensors' device.

    The tcgen05 path raises its dynamic shared-memory cap once per CUDA ordinal,
    and the ordinal it uses has to be the one the tensors live on rather than
    whichever device is current. That is only observable on the launch that
    actually takes the reservation, so this test insists on being the first
    tensor-path launch in the process and skips otherwise; the evidence run
    selects it alone into a fresh process, where it is defined before both
    placement tests so it is collected first.
    """
    rank, _, device = tp8_context
    if torch.cuda.device_count() < 2:
        pytest.skip("the cold reservation test needs two visible GPUs")
    if _C._kimi_k3_tail_shared_memory_reservations(device.index):
        pytest.skip("an earlier tensor-path launch already warmed this device")
    active_tokens = 20
    routed, shared = _partials(device, rank, active_tokens, 4150)
    _load_partials(workspace, routed, shared)
    _, expected = _reference(routed, shared, norm_weight, latent_up)
    other = torch.device(
        "cuda", (device.index + 1) % torch.cuda.device_count()
    )
    assert _C._kimi_k3_tail_shared_memory_reservations(device.index) == 0
    assert _C._kimi_k3_tail_shared_memory_reservations(other.index) == 0

    torch.cuda.set_device(other)
    try:
        actual = _call(workspace, norm_weight, latent_up, active_tokens)
        torch.cuda.synchronize(device)
        assert torch.cuda.current_device() == other.index
    finally:
        torch.cuda.set_device(device)
    _synchronize_ranks(workspace)

    assert _C._kimi_k3_tail_shared_memory_reservations(device.index) == 1
    assert _C._kimi_k3_tail_shared_memory_reservations(other.index) == 0
    assert actual.device == device
    _assert_tail_close(actual, expected)


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

    # Whether or not this launch was the one that took the reservation, the
    # ambient device must never have been the one reserved.
    assert _C._kimi_k3_tail_shared_memory_reservations(other.index) == 0
    if active_tokens > 8:
        assert _C._kimi_k3_tail_shared_memory_reservations(device.index) == 1
    assert actual.device == device
    _assert_tail_close(actual, expected)


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
