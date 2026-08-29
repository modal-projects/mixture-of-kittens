"""The host boundary of the production Kimi K3 decode megakernel.

``test_kimi_k3_decode.py`` covers what one launch computes and how it schedules
itself. This file covers everything the caller can get wrong before the launch
starts and everything the launch reports back afterwards: the operator schema,
the alignment contract, the timeout diagnostics, the queue bound the scheduler
is built on, and the rejections that keep a mismatched workspace from producing
a plausible wrong answer.

The alignment tests need real device pointers and the diagnostics tests need a
real launch, so this file also runs under ``torchrun --standalone
--nproc-per-node=8``.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import torch

from mok import _C, ops
from mok.kimi_k3 import (
    KIMI_K3_TP_SIZE,
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
    kimi_k3_decode,
)
from mok.ops import _DECODE_ALIGNMENT

from .kimi_k3_decode_support import (
    CONFIG,
    CORE_TOKENS,
    MAX_TOKENS,
    PERSISTENT_CTAS,
    TENSOR_TOKENS,
    TIMEOUT_PHASE,
    _phase,
    _synchronize_ranks,
    assert_decode_close,
    decode_reference,
    decode_step as _decode,
    hidden_states,
    low_level_arguments,
    weights,  # noqa: F401
    workspace,  # noqa: F401
)
from .kimi_k3_tail_support import TAIL_TIMEOUT_PHASE


# Read one scalar at a time by the device, so their natural alignment is all
# they need and the contract deliberately leaves them out. Naming them here is
# what lets the structural test below insist the contract covers everything
# else.
_SCALAR_WORD_TENSORS = frozenset(
    {
        "router_correction_bias",
        "barrier_buffer",
        "barrier_target",
        "error_flag",
    }
)

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "csrc" / "kimi_k3_decode"

# Every source that either defines a bounded wait or names the code one
# reports. The structural tests walk these rather than trusting the table.
_TIMEOUT_DEFINITIONS = ("tail_sync.cuh", "persistent_sync.cuh")
_TIMEOUT_CALLERS = ("collectives.cuh", "persistent_kernel.cuh")


def _source(name: str) -> str:
    return (_SOURCE_ROOT / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The operator schema.
# ---------------------------------------------------------------------------


def test_the_low_level_operator_returns_nothing_and_names_its_mutations() -> (
    None
):
    """The schema is what ``torch.compile`` and functionalization reason from.

    The step writes its result into a mutated input, and a custom operator may
    not also return a view that aliases one, so the operator returns nothing
    and the high-level wrapper takes the view afterwards. Every buffer the
    kernel writes has to be declared, or a graph transform is free to reorder a
    reader ahead of it.
    """
    schema = torch.ops.mok.kimi_k3_decode.default._schema
    assert len(schema.returns) == 0
    mutated = {
        argument.name
        for argument in schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    }
    assert mutated == {
        "scratch",
        "collective_buffer",
        "output_mailbox",
        "barrier_buffer",
        "barrier_target",
        "error_flag",
    }


# ---------------------------------------------------------------------------
# The alignment contract, entry by entry.
# ---------------------------------------------------------------------------


def test_the_alignment_contract_covers_every_tensor_the_kernel_dereferences(
) -> None:
    """A tensor added to the operator must be classified, not forgotten.

    The contract is what the parameterized rejections below are generated
    from, so an argument that is in neither the contract nor the scalar-word
    exemption would silently go untested at both boundaries.
    """
    schema = torch.ops.mok.kimi_k3_decode.default._schema
    tensors = {
        argument.name
        for argument in schema.arguments
        if argument.type.isSubtypeOf(torch._C.TensorType.get())
    }
    contracted = {field for field, _ in _DECODE_ALIGNMENT}
    assert contracted.isdisjoint(_SCALAR_WORD_TENSORS)
    assert contracted | _SCALAR_WORD_TENSORS == tensors
    assert all(alignment in (16, 256) for _, alignment in _DECODE_ALIGNMENT)


def _misaligned(original: torch.Tensor, alignment: int) -> torch.Tensor:
    """A contiguous view of ``original``'s shape at an under-aligned address."""
    padding = alignment // original.element_size()
    wide = torch.empty(
        original.numel() + padding,
        dtype=original.dtype,
        device=original.device,
    )
    assert wide.data_ptr() % alignment == 0
    view = wide[1:1 + original.numel()].view(original.shape)
    assert view.data_ptr() % alignment != 0
    return view


@pytest.mark.parametrize(("field", "alignment"), _DECODE_ALIGNMENT)
def test_the_step_rejects_an_under_aligned_input(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    field: str,
    alignment: int,
) -> None:
    """A contiguous view at an odd storage offset is otherwise silent.

    It satisfies every shape and dtype rule while faulting a 16-byte vector
    load or shifting every 256-byte scratch region, so it is rejected by name
    before any device work is queued.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, 4)
    arguments = low_level_arguments(workspace, weights, hidden, 4)
    arguments[field] = _misaligned(arguments[field], alignment)

    with pytest.raises(
        RuntimeError, match=f"requires {field} aligned to {alignment} bytes"
    ):
        ops.kimi_k3_decode(**arguments)


@pytest.mark.parametrize(("field", "alignment"), _DECODE_ALIGNMENT)
def test_the_c_entrypoint_rejects_an_under_aligned_input(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    field: str,
    alignment: int,
) -> None:
    """The Python check is a better message, not the enforcement.

    ``mok.ops`` names the offending argument before anything is queued, but the
    extension is reachable directly and a caller that reaches it must get the
    same refusal, so the whole contract is asserted on both sides.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, 4)
    arguments = low_level_arguments(workspace, weights, hidden, 4)
    arguments[field] = _misaligned(arguments[field], alignment)

    with pytest.raises(
        RuntimeError, match=f"requires {field} aligned to {alignment} bytes"
    ):
        _C.kimi_k3_decode(**arguments)


# ---------------------------------------------------------------------------
# Timeout diagnostics.
# ---------------------------------------------------------------------------


def _timeout_sites() -> list[tuple[str, int, int, int]]:
    return [tuple(site) for site in _C._kimi_k3_timeout_sites()]


def _declared_codes() -> dict[str, int]:
    pattern = re.compile(
        r"inline constexpr int (kError\w+) = (\d+);"
    )
    return {
        name: int(value)
        for name, value in pattern.findall(_source("types.cuh"))
    }


def test_every_declared_timeout_code_is_dense_nonzero_and_tabulated(
) -> None:
    """Zero has to keep meaning "no wait has ever timed out on this workspace".

    The flag is the only diagnostic a caller sees without knowing the scratch
    layout, so a site that reported zero would be indistinguishable from a
    launch that completed. Requiring the codes to be a dense range from one is
    also what makes "the last code equals the count" a sufficient check inside
    the header.
    """
    declared = _declared_codes()
    sites = _timeout_sites()
    codes = [code for _, code, _, _ in sites]

    assert len(sites) == len(declared)
    assert sorted(codes) == list(range(1, len(sites) + 1))
    assert sorted(declared.values()) == sorted(codes)
    assert len({name for name, _, _, _ in sites}) == len(sites)
    assert 0 not in codes


def test_every_declared_timeout_code_is_reported_by_a_real_wait() -> None:
    """A code nobody writes, or a wait nobody named, is a silent trap.

    The table is only worth having if it is the same set of sites the sources
    actually trap at, so the two are compared directly rather than kept in step
    by hand.
    """
    used = set()
    for name in _TIMEOUT_DEFINITIONS + _TIMEOUT_CALLERS:
        used |= set(re.findall(r"\bkError\w+\b", _source(name)))
    assert used == set(_declared_codes())


@pytest.mark.parametrize("name", _TIMEOUT_DEFINITIONS)
def test_every_bounded_wait_records_a_site_specific_code_before_it_traps(
    name: str,
) -> None:
    """No trap may report a bare flag, and none may spin without a bound.

    ``record_timeout_and_trap`` is the only way out of a stalled wait, so each
    of its call sites has to be reached from a ``wait_timed_out`` test and has
    to carry a named code rather than a literal. Counting the two against each
    other is what catches a wait that was added with a bound but without a
    diagnostic.
    """
    text = _source(name)
    definition = re.search(
        r"void record_timeout_and_trap\((?P<body>[^)]*)\)", text
    )
    assert definition is not None
    assert "error_code" in definition.group("body")
    assert "error_flag" in definition.group("body")

    calls = re.findall(
        r"record_timeout_and_trap\(\s*(?P<arguments>[^;]*?)\);", text
    )
    # The definition's own signature matches the call pattern too.
    invocations = [
        arguments for arguments in calls if "const int" not in arguments
    ]
    assert invocations
    for arguments in invocations:
        assert "error_flag" in arguments, arguments
        assert re.search(r"\bkError\w+|\berror_code\b", arguments), arguments
    assert len(invocations) == len(re.findall(r"wait_timed_out\(", text))


def test_a_shared_counter_is_still_told_apart_by_its_code() -> None:
    """The scratch slot alone cannot name the site, which is why both exist.

    Four of the tail's six waits sit on two counters, so a caller reading only
    the timeout slot could not tell the entry rendezvous from the reduce role.
    The code is what closes that gap, so at least one counter has to be shared
    for the pair to be worth writing at all.
    """
    sites = _timeout_sites()
    counters = [(slot, counter) for _, _, slot, counter in sites]
    assert len(set(counters)) < len(sites)
    assert len({code for _, code, _, _ in sites}) == len(sites)
    assert {slot for _, _, slot, _ in sites} == {
        TAIL_TIMEOUT_PHASE,
        TIMEOUT_PHASE,
    }


def test_every_site_spins_on_the_same_bound() -> None:
    """One budget for every site, on both paths.

    ``test_generation_and_timeout_helpers_are_wrap_safe`` pins what the bound
    means; this pins that the persistent kernel's waits are held to it too, so
    a site's diagnostics do not depend on which path happened to run.
    """
    budget = _C._kimi_k3_tail_wait_timeout_clocks()
    assert budget > 0
    assert _C._kimi_k3_decode_wait_timeout_clocks() == budget


@pytest.mark.parametrize("tokens", [CORE_TOKENS, TENSOR_TOKENS])
def test_a_completed_launch_leaves_both_diagnostics_at_zero(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    tokens: int,
) -> None:
    """Both halves are only readable if a normal step never writes either.

    A launch that set the flag or the slot on its way through would make the
    diagnostics useless: every workspace would look like it had timed out. The
    slots are poisoned first so the check cannot pass by nothing having touched
    them.
    """
    _, _, device = tp8_context
    hidden = hidden_states(device, tokens)
    counters = _phase(workspace.scratch)
    counters[TAIL_TIMEOUT_PHASE].fill_(99)
    counters[TIMEOUT_PHASE].fill_(99)
    workspace.error_flag.fill_(0)
    _synchronize_ranks(workspace)

    _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)

    assert int(workspace.error_flag.item()) == 0
    # The launch has no reason to clear a slot it never writes, so the poison
    # is expected to survive; what must not happen is a code appearing.
    assert int(counters[TAIL_TIMEOUT_PHASE].item()) == 99
    assert int(counters[TIMEOUT_PHASE].item()) == 99

    counters[TAIL_TIMEOUT_PHASE].fill_(0)
    counters[TIMEOUT_PHASE].fill_(0)
    _synchronize_ranks(workspace)
    _decode(workspace, weights, hidden)
    torch.cuda.synchronize(device)
    assert int(counters[TAIL_TIMEOUT_PHASE].item()) == 0
    assert int(counters[TIMEOUT_PHASE].item()) == 0
    assert int(workspace.error_flag.item()) == 0


# ---------------------------------------------------------------------------
# The queue bound the scheduler rests on.
# ---------------------------------------------------------------------------


def test_the_longest_queue_is_the_one_the_header_asserts() -> None:
    """The counter is never reset inside a launch, so its bound is load bearing.

    Routed queues advance by four logical units per claim, and a CTA leaves a
    queue on its first refused batch, so their conservative bound is the
    longest queue plus four units per CTA. The header static-asserts both
    numbers; this recomputes the logical length from the task plan of every
    accepted shape, so a retiling cannot pass by updating only the assertion.
    """
    units, ticket = _C._kimi_k3_decode_queue_bound()
    longest = max(
        max(_C._kimi_k3_decode_task_plan(tokens)[:4])
        for tokens in range(1, MAX_TOKENS + 1)
    )
    assert units == longest == 25_150
    assert ticket == units + 4 * PERSISTENT_CTAS == 25_742
    # Four orders of magnitude of headroom under the unsigned wrap.
    assert ticket < 0xffffffff // 2


def test_routed_queues_claim_four_adjacent_units_per_atomic() -> None:
    """Batch only the long routed queues, leaving mixed work ordering intact."""
    sync = _source("persistent_sync.cuh")
    claim = _function_body(sync, "int claim_unit_batch(")
    kernel = _function_body(
        _source("persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "kRoutedClaimBatch = 4" in sync
    assert "atomicAdd(" in claim
    assert "static_cast<unsigned int>(BATCH)" in claim
    assert kernel.count("claim_unit_batch<kRoutedClaimBatch>(") == 2
    assert kernel.count("claim_unit(") == 1


# ---------------------------------------------------------------------------
# Reads the compiler is free to poison.
# ---------------------------------------------------------------------------


def _function_body(text: str, signature: str) -> str:
    """Return the text of one `static __device__` function by its name."""
    start = text.index(signature)
    depth = 0
    for offset in range(text.index("{", start), len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start:offset + 1]
    raise AssertionError(f"{signature} is never closed")


def test_the_gate_up_rounds_never_read_a_prefetch_they_did_not_make() -> None:
    """The last round has no next round, so its prefetch never happened.

    A second buffer to prefetch into is what creates the hazard: on the round
    that skips the prefetch it holds an indeterminate value, and copying it
    forward reads that value, which is undefined however dead the copy proves
    to be. One buffer removes the hazard rather than guarding it -- the round
    stages its own bytes first, then reloads the same registers -- so the
    contract is that no second buffer exists and the reload is the only write
    the loop makes to the one that does.
    """
    body = _function_body(
        _source("expert_mxfp4.cuh"), "void routed_gate_up_unit("
    )
    loop = body[body.index("for (int round = 0; round < kGateUpRounds; ++round)"):]

    declarations = [
        line
        for line in body.splitlines()
        if re.match(r"\s*(uint4|std::uint32_t) \w+\[kGateUp\w+\];", line)
    ]
    assert len(declarations) == 2, declarations
    assert "uint4 payload[kGateUpRoundGroups];" in declarations[0]
    assert "std::uint32_t scale_words[kGateUpScaleTiles];" in declarations[1]

    # The prefetch is the loop's only reload, it targets the one buffer, and
    # it is the only thing the last-round test guards.
    guarded = re.findall(
        r"if \(round \+ 1 < kGateUpRounds\) \{(?P<block>(?:[^{}]|\{[^{}]*\})*)\}",
        loop,
    )
    assert len(guarded) == 1, loop
    assert "read_weight_round(" in guarded[0]
    assert "payload, scale_words);" in guarded[0]
    assert len(re.findall(r"read_weight_round\(", loop)) == 1

    # No copy forward, because there is nothing to copy from.
    assert not re.search(r"\bnext_(?:payload|scale_words)\b", body), body
    assert "payload[slot] =" not in loop
    assert "scale_words[quad] =" not in loop


# ---------------------------------------------------------------------------
# The profiling band.
# ---------------------------------------------------------------------------


def test_clearing_the_phase_clocks_is_fenced_off_from_the_first_timed_region(
) -> None:
    """One CTA zeroes the band, so the rest may not be timing while it does.

    Only block 0 clears the counters, and a CTA that finished a region before
    that store landed would have its cycles zeroed with it, so the profile
    would silently under-report by however many CTAs won the race. The clearing
    is therefore separated from the first ``clocks.now()`` by a grid barrier,
    and this pins that order in the source: the clear, then the barrier, then
    the first mark.
    """
    text = _source("persistent_kernel.cuh")
    body = _function_body(text, "void kimi_k3_decode_persistent_kernel(")

    clear = body.index("clocks.counters[thread] = 0ull;")
    barrier = body.index("grid_barrier(", clear)
    mark = body.index("clocks.now()", clear)
    assert clear < barrier < mark

    # The clear and its barrier are one launch-wide predicate, so every CTA
    # agrees on whether the extra barrier happens and their targets stay in
    # step. An unprofiled launch takes neither.
    guard = re.search(
        r"if \(clocks\.enabled\(\)\) \{(?P<block>(?:[^{}]|\{[^{}]*\})*)\}",
        body,
    )
    assert guard is not None
    assert "clocks.counters[thread] = 0ull;" in guard.group("block")
    assert "grid_barrier(" in guard.group("block")


# ---------------------------------------------------------------------------
# Rejections.
# ---------------------------------------------------------------------------


def test_the_step_rejects_metadata_borrowed_from_another_rank(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A rank mix-up is silent unless it is checked, so it is checked.

    The kernel reaches its peers through pointers and indexes both the
    replicated latent-up weight and the mailbox by rank, so a workspace, a
    weight shard, and a ``tp_rank`` that do not describe the same rank produce
    a plausible wrong answer rather than a failure.
    """
    rank, _, device = tp8_context
    tokens = 4
    hidden = hidden_states(device, tokens)
    other = (rank + 1) % KIMI_K3_TP_SIZE

    with pytest.raises(ValueError, match="weights.tp_rank"):
        kimi_k3_decode(
            CONFIG,
            workspace,
            dataclasses.replace(weights, tp_rank=other),
            hidden,
        )

    arguments = low_level_arguments(workspace, weights, hidden, tokens)
    with pytest.raises(RuntimeError, match="this rank's own device pointer"):
        ops.kimi_k3_decode(**{**arguments, "tp_rank": other})

    with pytest.raises(RuntimeError, match="workspace_signature"):
        ops.kimi_k3_decode(
            **{
                **arguments,
                "workspace_signature": workspace.workspace_signature ^ 1,
            }
        )

    if torch.cuda.device_count() > 1:
        elsewhere = torch.device(
            "cuda", (device.index + 1) % torch.cuda.device_count()
        )
        with pytest.raises(ValueError, match="must be on"):
            kimi_k3_decode(
                CONFIG, workspace, weights, hidden.to(elsewhere)
            )


@pytest.mark.parametrize("active_tokens", [0, -1, MAX_TOKENS + 1])
def test_the_step_rejects_an_impossible_active_token_count(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    active_tokens: int,
) -> None:
    """The count sizes every queue, so an impossible one is refused up front."""
    _, _, device = tp8_context
    hidden = hidden_states(device, 4)
    arguments = low_level_arguments(workspace, weights, hidden, 4)
    with pytest.raises(RuntimeError, match="active_tokens"):
        ops.kimi_k3_decode(**{**arguments, "active_tokens": active_tokens})


def test_the_public_call_leaves_the_workspace_usable_after_a_rejection(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
) -> None:
    """A refused call queues nothing, so the next real call is unaffected.

    Every rejection above is raised on the host before the launch, which is
    what keeps a rejected call from leaving half a step's worth of generation
    counters behind for the next one to wait on.
    """
    _, _, device = tp8_context
    tokens = CORE_TOKENS
    hidden = hidden_states(device, tokens)
    expected = decode_reference(hidden, weights)

    arguments = low_level_arguments(workspace, weights, hidden, tokens)
    with pytest.raises(RuntimeError):
        ops.kimi_k3_decode(**{**arguments, "active_tokens": 0})
    _synchronize_ranks(workspace)

    actual = _decode(workspace, weights, hidden)
    assert_decode_close(actual, expected)
