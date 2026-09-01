"""TP8 GPU tests for the Kimi K3 workspace signature.

Task 8 binds one rank's three symmetric allocations, their peer-pointer lists,
their multicast aliases, and its rank into a single number, and the tail
recomputes that number from the pointers it is actually handed. These tests pin
the mixing against an independent Python mirror, prove that every one of the 31
folded integers can move the result, and prove that borrowing one alias from a
second live workspace -- the one mix-up no per-allocation rule can see -- is
rejected. What the launch computes is covered in
``test_kimi_k3_collectives.py``, the rest of the host boundary in
``test_kimi_k3_tail_contract.py``, and the shared workspace, weights, and
reference reduction live in ``kimi_k3_tail_support.py``.

Every test here needs all eight ranks, so this file must be launched through
``torchrun --standalone --nproc-per-node=8``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode

from mok import _C, ops
from mok.kimi_k3 import (
    KIMI_K3_TP_SIZE,
    KimiK3DecodeWorkspace,
    create_kimi_k3_decode_workspace,
)

from .kimi_k3_tail_support import (
    COLLECTIVE_COLUMNS,
    HIDDEN,
    LATENT,
    MAX_TOKENS,
    SCRATCH_BYTES,
    SHARD,
    UINT64_MAX,
    _TAIL_ARGUMENTS,
    _assert_identical_across_ranks,
    _assert_tail_close,
    _expect_rejection,
    _load_partials,
    _partials,
    _reference,
    _symmetric_facts,
    _synchronize_ranks,
    _valid_arguments,
    latent_up,
    norm_weight,
    workspace,
)


# The signature's mixing, mirrored from
# `csrc/kimi_k3_decode/workspace_signature.cuh`. Spelling it out here pins the
# constants, the order the pointers are folded in, and the 63-bit mask from the
# Python side, so C++ and Python cannot drift into computing different numbers
# for the same workspace.
_SIGNATURE_FIRST_MULTIPLIER = 0x9E3779B97F4A7C15
_SIGNATURE_SECOND_MULTIPLIER = 0xBF58476D1CE4E5B9
_SIGNATURE_INITIAL_STATE = 0x243F6A8885A308D3
_SIGNATURE_MASK = (1 << 63) - 1

# The multicast argument of each symmetric allocation, and the workspace
# attribute holding it, so a second workspace's alias can be borrowed by name.
_MULTICAST_ATTRIBUTES = (
    ("collective_buffer_multicast_ptr", "collective_multicast_ptr"),
    ("output_mailbox_multicast_ptr", "output_mailbox_multicast_ptr"),
    ("barrier_buffer_multicast_ptr", "barrier_multicast_ptr"),
)


def _mix(state: int, value: int) -> int:
    state ^= value & UINT64_MAX
    state = (state * _SIGNATURE_FIRST_MULTIPLIER) & UINT64_MAX
    state ^= state >> 29
    state = (state * _SIGNATURE_SECOND_MULTIPLIER) & UINT64_MAX
    return state ^ (state >> 32)


def _mirrored_signature(workspace: KimiK3DecodeWorkspace) -> int:
    state = _mix(_SIGNATURE_INITIAL_STATE, workspace.tp_rank)
    for _, _, tensor, pointers, multicast, _ in _symmetric_facts(workspace):
        state = _mix(state, tensor.data_ptr())
        for pointer in pointers:
            state = _mix(state, pointer)
        state = _mix(state, multicast)
    return state & _SIGNATURE_MASK


def _signature_arguments(
    workspace: KimiK3DecodeWorkspace,
) -> dict[str, object]:
    """The signature helper's arguments, taken from one live workspace."""
    return {
        "collective_buffer": workspace.collective_buffer,
        "collective_buffer_ptrs": list(workspace.collective_ptrs),
        "collective_buffer_multicast_ptr": (
            workspace.collective_multicast_ptr
        ),
        "output_mailbox": workspace.output_mailbox,
        "output_mailbox_ptrs": list(workspace.output_mailbox_ptrs),
        "output_mailbox_multicast_ptr": (
            workspace.output_mailbox_multicast_ptr
        ),
        "barrier_buffer": workspace.barrier_buffer,
        "barrier_buffer_ptrs": list(workspace.barrier_ptrs),
        "barrier_buffer_multicast_ptr": workspace.barrier_multicast_ptr,
        "tp_rank": workspace.tp_rank,
    }


_SIGNATURE_ARGUMENT_ORDER = (
    "collective_buffer",
    "collective_buffer_ptrs",
    "collective_buffer_multicast_ptr",
    "output_mailbox",
    "output_mailbox_ptrs",
    "output_mailbox_multicast_ptr",
    "barrier_buffer",
    "barrier_buffer_ptrs",
    "barrier_buffer_multicast_ptr",
    "tp_rank",
)


def _signature_of(arguments: dict[str, object]) -> int:
    return _C._kimi_k3_workspace_signature(
        *(arguments[name] for name in _SIGNATURE_ARGUMENT_ORDER)
    )


@pytest.fixture(scope="module")
def second_workspace(
    tp8_context: tuple[int, int, torch.device],
) -> Iterator[KimiK3DecodeWorkspace]:
    """A second live workspace over the same TP group as ``workspace``.

    Symmetric allocation and rendezvous are collective, so every rank has to
    build this in the same order -- which it does, because every rank runs the
    same tests in the same order.
    """
    _, _, device = tp8_context
    created = create_kimi_k3_decode_workspace(dist.group.WORLD, device=device)
    try:
        yield created
    finally:
        _synchronize_ranks(created)


def test_workspace_signature_agrees_with_python_and_moves_for_every_input(
    workspace: KimiK3DecodeWorkspace,
    second_workspace: KimiK3DecodeWorkspace,
) -> None:
    """The signature is a pure function of the tuple, and it sees all of it."""
    recorded = workspace.workspace_signature
    assert type(recorded) is int
    assert 0 <= recorded <= _SIGNATURE_MASK
    # Python's mirror of the mixing, the C++ helper, and the value the workspace
    # recorded at creation all have to be the same number.
    assert recorded == _mirrored_signature(workspace)
    arguments = _signature_arguments(workspace)
    assert _signature_of(arguments) == recorded
    assert _signature_of(arguments) == recorded

    other = _signature_arguments(second_workspace)
    assert _signature_of(other) == second_workspace.workspace_signature
    assert _signature_of(other) != recorded

    # Every one of the 31 folded integers must be able to move the result: each
    # pointer of each list, each multicast alias, each local tensor address, and
    # the rank.
    moved = 0
    for name in _SIGNATURE_ARGUMENT_ORDER:
        value = arguments[name]
        if name == "tp_rank":
            perturbed = [(value + 1) % KIMI_K3_TP_SIZE]
        elif isinstance(value, list):
            perturbed = [
                [*value[:index], value[index] + 16, *value[index + 1:]]
                for index in range(KIMI_K3_TP_SIZE)
            ]
        elif isinstance(value, int):
            perturbed = [value + 16]
        else:
            # A local tensor address only changes by handing over a different
            # allocation, so the second workspace supplies one.
            perturbed = [other[name]]
        for candidate in perturbed:
            assert (
                _signature_of({**arguments, name: candidate}) != recorded
            ), name
            moved += 1
    assert moved == 3 * (1 + KIMI_K3_TP_SIZE + 1) + 1


def test_a_whole_second_workspace_is_valid_with_its_own_signature(
    tp8_context: tuple[int, int, torch.device],
    workspace: KimiK3DecodeWorkspace,
    second_workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """Owning two workspaces is legitimate; only mixing them is not."""
    rank, _, device = tp8_context
    assert (
        second_workspace.workspace_signature
        != workspace.workspace_signature
    )
    active_tokens = 6
    routed, shared = _partials(device, rank, active_tokens, 4400)
    _load_partials(second_workspace, routed, shared)
    _, expected = _reference(routed, shared, norm_weight, latent_up)
    arguments = _valid_arguments(
        second_workspace, norm_weight, latent_up, active_tokens
    )

    actual = ops._kimi_k3_tail(**arguments)
    _synchronize_ranks(second_workspace)

    assert actual.data_ptr() == second_workspace.output_mailbox.data_ptr()
    _assert_tail_close(actual, expected)
    _assert_identical_across_ranks(actual)

    # The same complete tuple with the other workspace's signature is not.
    for label, signature in (
        ("second workspace, first signature", workspace.workspace_signature),
        ("flipped low bit", second_workspace.workspace_signature ^ 1),
    ):
        _expect_rejection(
            label,
            r"workspace_signature to match",
            lambda signature=signature: ops._kimi_k3_tail(
                **{**arguments, "workspace_signature": signature}
            ),
        )
    _synchronize_ranks(second_workspace)


def _borrowed_multicast_case(
    workspace: KimiK3DecodeWorkspace,
    second_workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    field: str,
    attribute: str,
) -> dict[str, object]:
    """One workspace's tuple with a second workspace's multicast alias.

    The borrowed alias satisfies every per-allocation rule the entry point can
    apply to it in isolation -- it is a positive, correctly aligned device
    address, distinct from every unicast entry and from the other two multicast
    aliases -- which is exactly why nothing but the signature can catch it.
    """
    arguments = _valid_arguments(workspace, norm_weight, latent_up, 4)
    borrowed = int(getattr(second_workspace, attribute))
    alignment = next(
        boundary
        for _, multicast_field, _, _, _, boundary in _symmetric_facts(
            workspace
        )
        if multicast_field == field
    )
    assert borrowed > 0 and borrowed % alignment == 0
    local = {
        pointer
        for _, _, _, pointers, multicast, _ in _symmetric_facts(workspace)
        for pointer in [*pointers, multicast]
    }
    assert borrowed not in local
    arguments[field] = borrowed
    return arguments


@pytest.mark.parametrize(("field", "attribute"), _MULTICAST_ATTRIBUTES)
def test_tail_rejects_a_multicast_borrowed_from_a_second_workspace(
    workspace: KimiK3DecodeWorkspace,
    second_workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    field: str,
    attribute: str,
) -> None:
    arguments = _borrowed_multicast_case(
        workspace, second_workspace, norm_weight, latent_up, field, attribute
    )
    _expect_rejection(
        f"python: {field} borrowed from a second workspace",
        r"workspace_signature to match",
        lambda: ops._kimi_k3_tail(**arguments),
    )


@pytest.mark.parametrize(("field", "attribute"), _MULTICAST_ATTRIBUTES)
def test_tail_c_entrypoint_rejects_a_borrowed_multicast(
    workspace: KimiK3DecodeWorkspace,
    second_workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
    field: str,
    attribute: str,
) -> None:
    arguments = _borrowed_multicast_case(
        workspace, second_workspace, norm_weight, latent_up, field, attribute
    )
    _expect_rejection(
        f"pybind: {field} borrowed from a second workspace",
        r"workspace_signature to match",
        lambda: _C._kimi_k3_tail(
            *(arguments[name] for name in _TAIL_ARGUMENTS)
        ),
    )


def test_tail_rejects_a_malformed_workspace_signature(
    workspace: KimiK3DecodeWorkspace,
    norm_weight: torch.Tensor,
    latent_up: torch.Tensor,
) -> None:
    """Type and range are checked in Python, before any pointer is used."""
    valid = _valid_arguments(workspace, norm_weight, latent_up, 4)
    for label, signature in (
        ("negative", -1),
        ("above the 63-bit range", 1 << 63),
        ("boolean", True),
        ("floating point", float(workspace.workspace_signature)),
    ):
        _expect_rejection(
            f"malformed signature: {label}",
            r"workspace_signature to be an integer in",
            lambda signature=signature: ops._kimi_k3_tail(
                **{**valid, "workspace_signature": signature}
            ),
        )
    # A trace has no addresses, so it may only carry the documented placeholder.
    with FakeTensorMode():
        _expect_rejection(
            "traced signature is not the placeholder",
            r"workspace_signature 0 while tracing",
            lambda: ops._kimi_k3_tail(
                torch.empty(LATENT, dtype=torch.bfloat16, device="cuda"),
                torch.empty(
                    HIDDEN, LATENT, dtype=torch.bfloat16, device="cuda"
                ),
                torch.empty(
                    MAX_TOKENS,
                    COLLECTIVE_COLUMNS,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                [1] * KIMI_K3_TP_SIZE,
                1,
                torch.empty(
                    MAX_TOKENS,
                    KIMI_K3_TP_SIZE,
                    SHARD,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                [1] * KIMI_K3_TP_SIZE,
                1,
                torch.empty(1, dtype=torch.int32, device="cuda"),
                [1] * KIMI_K3_TP_SIZE,
                1,
                torch.empty(1, dtype=torch.int32, device="cuda"),
                torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda"),
                torch.empty(1, dtype=torch.int32, device="cuda"),
                0,
                11,
                workspace.workspace_signature,
            ),
        )
