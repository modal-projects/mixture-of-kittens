"""TP8 GPU tests for the Kimi K3 latent-MoE tail's host-side contract.

The role plan and residency guard, the wrap-safe serial-number helpers, the
custom operator's schema and fake trace, alignment and shape rejection, the
symmetric pointer topology every rank must supply, and the shared barrier pair
across the unsigned wrap. What the launch computes is covered in
``test_kimi_k3_collectives.py``; the shared workspace, weights, and reference
reduction live in ``kimi_k3_tail_support.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mok import _C, _fake_impls, ops
from mok.kimi_k3 import KIMI_K3_TP_SIZE, KimiK3DecodeWorkspace

from .kimi_k3_tail_support import (
    ALIGNMENT,
    COLLECTIVE_COLUMNS,
    HIDDEN,
    LATENT,
    MAX_TOKENS,
    SCRATCH_BYTES,
    SHARD,
    TAIL_ARRIVALS,
    TAIL_ENTRY_GENERATION,
    TAIL_EXIT_GENERATION,
    TAIL_GENERATIONS,
    TAIL_REDUCE_GENERATION,
    TAIL_SHARD_GENERATION,
    TAIL_TIMEOUT_PHASE,
    UINT32,
    UINT32_MAX,
    UINT64_MAX,
    _TAIL_ARGUMENTS,
    _as_uint32,
    _assert_identical_across_ranks,
    _assert_tail_close,
    _barrier_all,
    _call,
    _load_partials,
    _partials,
    _phase,
    _prime_barrier_serial,
    _reference,
    _rotating_skew,
    _serial_reached,
    _synchronize_ranks,
    latent_up,
    norm_weight,
    workspace,
)


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
