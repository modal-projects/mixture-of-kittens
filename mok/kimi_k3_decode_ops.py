"""The Kimi K3 decode step and the private stage operators it is built from."""

import torch
from torch._subclasses.fake_tensor import is_fake

from . import _C


# Every Kimi K3 collective runs at TP8, so each pointer list below has
# exactly eight entries.
_K3_TP_SIZE = 8

# The workspace signature is folded in unsigned 64-bit arithmetic and masked to
# 63 bits by `csrc/kimi_k3_decode/workspace_signature.cuh`, so every legitimate
# value is a non-negative integer that fits the schema's `int`.
_K3_SIGNATURE_MAX = (1 << 63) - 1

# A fake trace has no addresses to fold, so there is no signature it could
# carry. Zero is reserved as its placeholder: it is documented here, accepted
# only while tracing, and never recomputed against a real workspace.
_K3_TRACE_SIGNATURE = 0

# Each symmetric allocation a Kimi K3 collective drives: its local tensor, its
# peer-pointer list, its multicast alias, and the byte boundary the device
# dereferences it on. The two BF16 allocations are read with 16-byte multimem
# octets; the barrier is a single int32 word.
_K3_SYMMETRIC = (
    (
        "collective_buffer",
        "collective_buffer_ptrs",
        "collective_buffer_multicast_ptr",
        16,
    ),
    (
        "output_mailbox",
        "output_mailbox_ptrs",
        "output_mailbox_multicast_ptr",
        16,
    ),
    (
        "barrier_buffer",
        "barrier_buffer_ptrs",
        "barrier_buffer_multicast_ptr",
        4,
    ),
)


def _check_k3_symmetric_pointers(
    operation: str,
    arguments: dict[str, object],
) -> None:
    """Reject a pointer list that does not describe this rank's own allocation.

    The kernels only dereference the multicast alias, so these lists are the
    one place a caller can reveal that it mixed up a rank, an allocation, or a
    whole workspace -- otherwise a mix-up is silent and the launch just reduces
    the wrong columns or fills the wrong mailbox slot. The rules mirror
    ``check_symmetric_pointers`` in ``csrc/kimi_k3_decode/entrypoints.cuh``, and
    each one is already satisfied by any valid PyTorch symmetric-memory handle.
    """
    tp_rank = arguments["tp_rank"]
    if type(tp_rank) is not int or not 0 <= tp_rank < _K3_TP_SIZE:
        raise RuntimeError(
            f"MoK: {operation} requires tp_rank in "
            f"[0, {_K3_TP_SIZE - 1}], got {tp_rank}"
        )
    # A trace carries no addresses, so only the shape of the lists can be
    # checked; a fake trace legitimately passes placeholder pointers.
    tracing = any(
        is_fake(arguments[field]) for field, _, _, _ in _K3_SYMMETRIC
    )
    signature = arguments["workspace_signature"]
    if type(signature) is not int or not 0 <= signature <= _K3_SIGNATURE_MAX:
        raise RuntimeError(
            f"MoK: {operation} requires workspace_signature to be an "
            f"integer in [0, {_K3_SIGNATURE_MAX}], got {signature!r}"
        )
    if tracing and signature != _K3_TRACE_SIGNATURE:
        raise RuntimeError(
            f"MoK: {operation} requires workspace_signature "
            f"{_K3_TRACE_SIGNATURE} while tracing fake tensors, because a "
            f"trace has no addresses to fold, got {signature}"
        )
    for tensor_field, list_field, multicast_field, alignment in (
        _K3_SYMMETRIC
    ):
        tensor = arguments[tensor_field]
        pointers = arguments[list_field]
        multicast = arguments[multicast_field]
        if len(pointers) != _K3_TP_SIZE:
            raise RuntimeError(
                f"MoK: {operation} requires {list_field} with exactly "
                f"{_K3_TP_SIZE} pointers, got {len(pointers)}"
            )
        for rank, pointer in enumerate(pointers):
            if type(pointer) is not int or pointer <= 0:
                raise RuntimeError(
                    f"MoK: {operation} requires {list_field} to hold only "
                    f"positive device pointers, but entry {rank} is {pointer}"
                )
        if tracing:
            continue
        # Checked before alignment and distinctness so that a substituted rank
        # or a swapped list is always reported as what it is.
        local = tensor.data_ptr()
        if pointers[tp_rank] != local:
            raise RuntimeError(
                f"MoK: {operation} requires {list_field}[tp_rank] to be "
                f"this rank's own device pointer, but entry {tp_rank} is "
                f"{pointers[tp_rank]} while the matching tensor is at {local}. "
                f"The pointer list, the tensor, or tp_rank came from a "
                f"different rank or a different workspace"
            )
        for rank, pointer in enumerate(pointers):
            past = pointer % alignment
            if past:
                raise RuntimeError(
                    f"MoK: {operation} requires every {list_field} entry "
                    f"aligned to {alignment} bytes, but entry {rank} is "
                    f"{past} bytes past one"
                )
        if len(set(pointers)) != _K3_TP_SIZE:
            raise RuntimeError(
                f"MoK: {operation} requires {list_field} to hold one "
                f"distinct pointer per rank, but it holds "
                f"{_K3_TP_SIZE - len(set(pointers))} repeated entries"
            )
        if type(multicast) is not int or multicast <= 0:
            raise RuntimeError(
                f"MoK: {operation} requires {multicast_field} to be a "
                f"positive device pointer, got {multicast}"
            )
        past = multicast % alignment
        if past:
            raise RuntimeError(
                f"MoK: {operation} requires {multicast_field} aligned to "
                f"{alignment} bytes, but it is {past} bytes past one"
            )
        if multicast in pointers:
            raise RuntimeError(
                f"MoK: {operation} requires one distinct multicast pointer "
                f"per symmetric allocation, but {multicast_field} equals "
                f"{list_field} entry {pointers.index(multicast)}"
            )
    if tracing:
        return
    # A per-allocation check cannot see this: each pointer is individually
    # valid, so only comparing them against each other reveals that the caller
    # pointed two allocations at the same fabric address.
    multicasts = [
        (field, arguments[field]) for _, _, field, _ in _K3_SYMMETRIC
    ]
    for index, (field, multicast) in enumerate(multicasts):
        for other_field, other in multicasts[index + 1:]:
            if multicast == other:
                raise RuntimeError(
                    f"MoK: {operation} requires one distinct multicast "
                    f"pointer per symmetric allocation, but {field} and "
                    f"{other_field} are both {multicast}"
                )


# Every input the persistent kernel reads through 16-byte vector loads or a TMA
# descriptor, plus the scratch it indexes through 256-byte aligned regions. A
# contiguous view at an under-aligned storage offset satisfies every shape and
# dtype rule and still faults the load or silently shifts every region. The
# extension enforces this too; repeating it here names the offending argument
# before any device work is queued. The correction bias is read as a scalar
# float and the three int32 control tensors as single words, so their natural
# alignment is enough.
_DECODE_ALIGNMENT = (
    ("hidden_states", 16),
    ("router_weight", 16),
    ("routed_expert_down_proj", 16),
    ("routed_expert_up_proj", 16),
    ("routed_latent_rmsnorm_weight", 16),
    # Stricter than the rest: the tensor map the fused routed gate/up engine
    # reads this payload through pins a 32-byte base, where every other weight
    # only has to satisfy a 16-byte vector load.
    ("expert_w13_packed", 32),
    ("expert_w13_scale", 16),
    ("expert_w2_packed", 16),
    ("expert_w2_scale", 16),
    ("shared_gate_proj", 16),
    ("shared_up_proj", 16),
    ("shared_down_proj", 16),
    ("collective_buffer", 16),
    ("output_mailbox", 16),
    ("scratch", 256),
)

# The step writes its result into the caller's mailbox and returns nothing: a
# custom operator may not return a view that aliases one of its own mutated
# inputs, so `mok.kimi_k3.kimi_k3_decode` takes the active-row view after this
# operator returns.
_DECODE_SCHEMA = (
    "kimi_k3_decode("
    "Tensor hidden_states, Tensor router_weight, "
    "Tensor router_correction_bias, Tensor routed_expert_down_proj, "
    "Tensor routed_expert_up_proj, Tensor routed_latent_rmsnorm_weight, "
    "Tensor expert_w13_packed, Tensor expert_w13_scale, "
    "Tensor expert_w2_packed, Tensor expert_w2_scale, "
    "Tensor shared_gate_proj, Tensor shared_up_proj, Tensor shared_down_proj, "
    "Tensor(a!) scratch, "
    "Tensor(b!) collective_buffer, int[] collective_buffer_ptrs, "
    "int collective_buffer_multicast_ptr, "
    "Tensor(c!) output_mailbox, int[] output_mailbox_ptrs, "
    "int output_mailbox_multicast_ptr, "
    "Tensor(d!) barrier_buffer, int[] barrier_buffer_ptrs, "
    "int barrier_buffer_multicast_ptr, Tensor(e!) barrier_target, "
    "Tensor(f!) error_flag, int tp_rank, int active_tokens, "
    "int workspace_signature"
    ") -> ()"
)
_DECODE_LIBRARY = torch.library.Library("mok", "FRAGMENT")
_DECODE_LIBRARY.define(_DECODE_SCHEMA)


@torch.library.impl("mok::kimi_k3_decode", "cuda")
def _kimi_k3_decode_cuda(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_correction_bias: torch.Tensor,
    routed_expert_down_proj: torch.Tensor,
    routed_expert_up_proj: torch.Tensor,
    routed_latent_rmsnorm_weight: torch.Tensor,
    expert_w13_packed: torch.Tensor,
    expert_w13_scale: torch.Tensor,
    expert_w2_packed: torch.Tensor,
    expert_w2_scale: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    scratch: torch.Tensor,
    collective_buffer: torch.Tensor,
    collective_buffer_ptrs: list[int],
    collective_buffer_multicast_ptr: int,
    output_mailbox: torch.Tensor,
    output_mailbox_ptrs: list[int],
    output_mailbox_multicast_ptr: int,
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    error_flag: torch.Tensor,
    tp_rank: int,
    active_tokens: int,
    workspace_signature: int,
) -> None:
    _C.kimi_k3_decode(
        hidden_states,
        router_weight,
        router_correction_bias,
        routed_expert_down_proj,
        routed_expert_up_proj,
        routed_latent_rmsnorm_weight,
        expert_w13_packed,
        expert_w13_scale,
        expert_w2_packed,
        expert_w2_scale,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        scratch,
        collective_buffer,
        collective_buffer_ptrs,
        collective_buffer_multicast_ptr,
        output_mailbox,
        output_mailbox_ptrs,
        output_mailbox_multicast_ptr,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        error_flag,
        tp_rank,
        active_tokens,
        workspace_signature,
    )


def kimi_k3_decode(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_correction_bias: torch.Tensor,
    routed_expert_down_proj: torch.Tensor,
    routed_expert_up_proj: torch.Tensor,
    routed_latent_rmsnorm_weight: torch.Tensor,
    expert_w13_packed: torch.Tensor,
    expert_w13_scale: torch.Tensor,
    expert_w2_packed: torch.Tensor,
    expert_w2_scale: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    scratch: torch.Tensor,
    collective_buffer: torch.Tensor,
    collective_buffer_ptrs: list[int],
    collective_buffer_multicast_ptr: int,
    output_mailbox: torch.Tensor,
    output_mailbox_ptrs: list[int],
    output_mailbox_multicast_ptr: int,
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    error_flag: torch.Tensor,
    tp_rank: int,
    active_tokens: int,
    workspace_signature: int,
) -> None:
    """Run one whole TP8 Kimi K3 decode step in a single persistent launch.

    One launch of `kimi_k3_decode_dependency_local_kernel` routes every token,
    projects and quantizes the routed latent, runs the mixed W4A8 routed
    experts and the BF16 shared expert, publishes both partials into the
    symmetric collective buffer, and closes the step with the fused TP8 tail.
    Nothing is allocated and no other kernel is launched.

    The result lands in ``output_mailbox``; this operator returns nothing,
    because a custom operator may not return a view that aliases one of its own
    mutated inputs. ``mok.kimi_k3.kimi_k3_decode`` takes that view afterwards.

    ``workspace_signature`` is the value ``create_kimi_k3_decode_workspace``
    recorded for this rank's workspace. The operator recomputes it from the
    pointers actually passed and refuses to launch on a mismatch, which is what
    binds all three symmetric allocations to one workspace rather than only to
    each other.
    """
    arguments = locals()
    for field, alignment in _DECODE_ALIGNMENT:
        if is_fake(arguments[field]):
            continue
        past = arguments[field].data_ptr() % alignment
        if past:
            raise RuntimeError(
                f"MoK: kimi_k3_decode requires {field} aligned to "
                f"{alignment} bytes, got a pointer {past} bytes past one"
            )
    _check_k3_symmetric_pointers("kimi_k3_decode", arguments)
    torch.ops.mok.kimi_k3_decode(
        hidden_states,
        router_weight,
        router_correction_bias,
        routed_expert_down_proj,
        routed_expert_up_proj,
        routed_latent_rmsnorm_weight,
        expert_w13_packed,
        expert_w13_scale,
        expert_w2_packed,
        expert_w2_scale,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        scratch,
        collective_buffer,
        collective_buffer_ptrs,
        collective_buffer_multicast_ptr,
        output_mailbox,
        output_mailbox_ptrs,
        output_mailbox_multicast_ptr,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        error_flag,
        tp_rank,
        active_tokens,
        workspace_signature,
    )


# The fused stage reads hidden states and both projection weights through 16-byte
# vector loads and TMA descriptors, and indexes scratch through 256-byte aligned
# regions, so a contiguous view at an under-aligned storage offset would fault or
# silently shift every region. The extension enforces this too, for callers that
# reach past this operator; repeating it here names the offending argument before
# any device work is queued. The correction bias is only read as a scalar float,
# so its natural alignment is enough.
_ROUTE_AND_PROJECT_ALIGNMENT = (
    ("hidden_states", 16),
    ("router_weight", 16),
    ("routed_expert_down_proj", 16),
    ("scratch", 256),
)


def _check_route_and_project_alignment(arguments: dict[str, torch.Tensor]) -> None:
    for field, alignment in _ROUTE_AND_PROJECT_ALIGNMENT:
        past = arguments[field].data_ptr() % alignment
        if past:
            raise RuntimeError(
                f"MoK: _kimi_k3_route_and_project requires {field} aligned to "
                f"{alignment} bytes, got a pointer {past} bytes past one"
            )


@torch.library.custom_op(
    "mok::_kimi_k3_route_and_project", mutates_args=("scratch",)
)
def _kimi_k3_route_and_project(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_correction_bias: torch.Tensor,
    routed_expert_down_proj: torch.Tensor,
    scratch: torch.Tensor,
    active_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the fused Kimi K3 router and routed latent-down projection stage.

    This private operator exposes the one-launch device stage that the final
    persistent decode kernel calls directly; it is not part of the public API.

    Inputs:
        hidden_states:           bfloat16 [M, 7168]
        router_weight:           bfloat16 [896, 7168]
        router_correction_bias:  float32 [896]
        routed_expert_down_proj: bfloat16 [3584, 7168]
        scratch:                 uint8 [kimi_k3_decode_workspace_bytes()]
        active_tokens:           int in [1, M]

    Outputs:
        expert_ids:     int32 [M, 16]
        expert_weights: float32 [M, 16]
        latent_x:       bfloat16 [M, 3584]
    """
    _check_route_and_project_alignment(
        {
            "hidden_states": hidden_states,
            "router_weight": router_weight,
            "routed_expert_down_proj": routed_expert_down_proj,
            "scratch": scratch,
        }
    )
    return _C._kimi_k3_route_and_project(
        hidden_states,
        router_weight,
        router_correction_bias,
        routed_expert_down_proj,
        scratch,
        active_tokens,
    )


_ROUTED_EXPERT_ALIGNMENT = (
    ("latent_x", 16),
    ("expert_w1_packed", 16),
    ("expert_w1_scale", 16),
    ("expert_w3_packed", 16),
    ("expert_w3_scale", 16),
    ("expert_w2_packed", 16),
    ("expert_w2_scale", 16),
    ("routed_output", 16),
    ("scratch", 256),
)

_ROUTED_EXPERT_SCHEMA = (
    "_kimi_k3_routed_experts("
    "Tensor latent_x, Tensor expert_w1_packed, Tensor expert_w1_scale, "
    "Tensor expert_w3_packed, Tensor expert_w3_scale, "
    "Tensor expert_w2_packed, Tensor expert_w2_scale, "
    "Tensor(a!) routed_output, Tensor(b!) scratch, int active_tokens"
    ") -> Tensor(a!)"
)
_ROUTED_EXPERT_LIBRARY = torch.library.Library("mok", "FRAGMENT")
_ROUTED_EXPERT_LIBRARY.define(_ROUTED_EXPERT_SCHEMA)


@torch.library.impl("mok::_kimi_k3_routed_experts", "cuda")
def _kimi_k3_routed_experts_cuda(
    latent_x: torch.Tensor,
    expert_w1_packed: torch.Tensor,
    expert_w1_scale: torch.Tensor,
    expert_w3_packed: torch.Tensor,
    expert_w3_scale: torch.Tensor,
    expert_w2_packed: torch.Tensor,
    expert_w2_scale: torch.Tensor,
    routed_output: torch.Tensor,
    scratch: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    return _C._kimi_k3_routed_experts(
        latent_x,
        expert_w1_packed,
        expert_w1_scale,
        expert_w3_packed,
        expert_w3_scale,
        expert_w2_packed,
        expert_w2_scale,
        routed_output,
        scratch,
        active_tokens,
    )


def _kimi_k3_routed_experts(
    latent_x: torch.Tensor,
    expert_w1_packed: torch.Tensor,
    expert_w1_scale: torch.Tensor,
    expert_w3_packed: torch.Tensor,
    expert_w3_scale: torch.Tensor,
    expert_w2_packed: torch.Tensor,
    expert_w2_scale: torch.Tensor,
    routed_output: torch.Tensor,
    scratch: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    """Run one assignment-driven Kimi K3 routed-expert device stage.

    The private operator consumes Task 5's expert-major assignment ranges,
    evaluates mixed MXFP8-by-MXFP4 gate/up and down projections, and returns
    the active view of the caller-owned BF16 output buffer.
    """
    arguments = locals()
    for field, alignment in _ROUTED_EXPERT_ALIGNMENT:
        if is_fake(arguments[field]):
            continue
        past = arguments[field].data_ptr() % alignment
        if past:
            raise RuntimeError(
                f"MoK: _kimi_k3_routed_experts requires {field} aligned to "
                f"{alignment} bytes, got a pointer {past} bytes past one"
            )
    return torch.ops.mok._kimi_k3_routed_experts(
        latent_x,
        expert_w1_packed,
        expert_w1_scale,
        expert_w3_packed,
        expert_w3_scale,
        expert_w2_packed,
        expert_w2_scale,
        routed_output,
        scratch,
        active_tokens,
    )


_SHARED_EXPERT_ALIGNMENT = (
    ("hidden_states", 16),
    ("shared_gate_proj", 16),
    ("shared_up_proj", 16),
    ("shared_down_proj", 16),
    ("scratch", 256),
    ("collective_buffer", 16),
)

_SHARED_EXPERT_SCHEMA = (
    "_kimi_k3_shared_experts("
    "Tensor hidden_states, Tensor shared_gate_proj, Tensor shared_up_proj, "
    "Tensor shared_down_proj, Tensor(a!) scratch, "
    "Tensor(b!) collective_buffer, int active_tokens"
    ") -> Tensor(b!)"
)
_SHARED_EXPERT_LIBRARY = torch.library.Library("mok", "FRAGMENT")
_SHARED_EXPERT_LIBRARY.define(_SHARED_EXPERT_SCHEMA)


@torch.library.impl("mok::_kimi_k3_shared_experts", "cuda")
def _kimi_k3_shared_experts_cuda(
    hidden_states: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    scratch: torch.Tensor,
    collective_buffer: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    return _C._kimi_k3_shared_experts(
        hidden_states,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        scratch,
        collective_buffer,
        active_tokens,
    )


def _kimi_k3_shared_experts(
    hidden_states: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    scratch: torch.Tensor,
    collective_buffer: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    """Run the rank-local Kimi K3 shared-expert stage in one launch.

    The returned active ``[M, 7168]`` view aliases columns ``3584:10752`` of
    ``collective_buffer``. This is a private testing boundary for the same
    producer/consumer role graph that the production persistent kernel uses.
    """
    arguments = locals()
    for field, alignment in _SHARED_EXPERT_ALIGNMENT:
        if is_fake(arguments[field]):
            continue
        past = arguments[field].data_ptr() % alignment
        if past:
            raise RuntimeError(
                f"MoK: _kimi_k3_shared_experts requires {field} aligned to "
                f"{alignment} bytes, got a pointer {past} bytes past one"
            )
    return torch.ops.mok._kimi_k3_shared_experts(
        hidden_states,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        scratch,
        collective_buffer,
        active_tokens,
    )


_TAIL_ALIGNMENT = (
    ("routed_latent_rmsnorm_weight", 16),
    ("latent_up_proj", 16),
    ("collective_buffer", 16),
    ("output_mailbox", 16),
    ("scratch", 256),
)


# The tail mutates the mailbox in place and returns nothing: a custom operator
# may not return a view that aliases one of its own mutated inputs, so the
# active-row view is taken by the Python helper after the operator returns.
_TAIL_SCHEMA = (
    "_kimi_k3_tail("
    "Tensor routed_latent_rmsnorm_weight, Tensor latent_up_proj, "
    "Tensor(a!) collective_buffer, int[] collective_buffer_ptrs, "
    "int collective_buffer_multicast_ptr, "
    "Tensor(b!) output_mailbox, int[] output_mailbox_ptrs, "
    "int output_mailbox_multicast_ptr, "
    "Tensor(c!) barrier_buffer, int[] barrier_buffer_ptrs, "
    "int barrier_buffer_multicast_ptr, Tensor(d!) barrier_target, "
    "Tensor(e!) scratch, Tensor(f!) error_flag, int tp_rank, "
    "int active_tokens, int workspace_signature"
    ") -> ()"
)
_TAIL_LIBRARY = torch.library.Library("mok", "FRAGMENT")
_TAIL_LIBRARY.define(_TAIL_SCHEMA)


@torch.library.impl("mok::_kimi_k3_tail", "cuda")
def _kimi_k3_tail_cuda(
    routed_latent_rmsnorm_weight: torch.Tensor,
    latent_up_proj: torch.Tensor,
    collective_buffer: torch.Tensor,
    collective_buffer_ptrs: list[int],
    collective_buffer_multicast_ptr: int,
    output_mailbox: torch.Tensor,
    output_mailbox_ptrs: list[int],
    output_mailbox_multicast_ptr: int,
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    scratch: torch.Tensor,
    error_flag: torch.Tensor,
    tp_rank: int,
    active_tokens: int,
    workspace_signature: int,
) -> None:
    _C._kimi_k3_tail(
        routed_latent_rmsnorm_weight,
        latent_up_proj,
        collective_buffer,
        collective_buffer_ptrs,
        collective_buffer_multicast_ptr,
        output_mailbox,
        output_mailbox_ptrs,
        output_mailbox_multicast_ptr,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        scratch,
        error_flag,
        tp_rank,
        active_tokens,
        workspace_signature,
    )


def _kimi_k3_tail(
    routed_latent_rmsnorm_weight: torch.Tensor,
    latent_up_proj: torch.Tensor,
    collective_buffer: torch.Tensor,
    collective_buffer_ptrs: list[int],
    collective_buffer_multicast_ptr: int,
    output_mailbox: torch.Tensor,
    output_mailbox_ptrs: list[int],
    output_mailbox_multicast_ptr: int,
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    scratch: torch.Tensor,
    error_flag: torch.Tensor,
    tp_rank: int,
    active_tokens: int,
    workspace_signature: int,
) -> torch.Tensor:
    """Close one TP8 Kimi K3 decode step and return the assembled output.

    The single launch all-reduces the routed latent partials and
    reduce-scatters the shared partials out of the symmetric collective
    buffer, applies FP32 RMSNorm, contracts this rank's 896 rows of the
    replicated latent-up weight, beta-adds its reduced shared shard, and
    multicasts that shard into every rank's token-major mailbox slot. Every
    rank returns the identical ``[active_tokens, 7168]`` view of its own
    mailbox storage; nothing is allocated.

    ``workspace_signature`` is the value
    ``create_kimi_k3_decode_workspace`` recorded for this rank's workspace. The
    operator recomputes it from the pointers actually passed and refuses to
    launch on a mismatch, which is what binds all three symmetric allocations to
    one workspace rather than to each other.

    ``error_flag`` is the caller-visible half of the tail's timeout
    diagnostics. Every bounded wait in the launch writes a site-specific
    nonzero code into it, and the slot it stalled on into the tail timeout
    counter in ``scratch``, before it traps; a launch that completes leaves
    both untouched.
    """
    arguments = locals()
    for field, alignment in _TAIL_ALIGNMENT:
        if is_fake(arguments[field]):
            continue
        past = arguments[field].data_ptr() % alignment
        if past:
            raise RuntimeError(
                f"MoK: _kimi_k3_tail requires {field} aligned to "
                f"{alignment} bytes, got a pointer {past} bytes past one"
            )
    _check_k3_symmetric_pointers("_kimi_k3_tail", arguments)
    torch.ops.mok._kimi_k3_tail(
        routed_latent_rmsnorm_weight,
        latent_up_proj,
        collective_buffer,
        collective_buffer_ptrs,
        collective_buffer_multicast_ptr,
        output_mailbox,
        output_mailbox_ptrs,
        output_mailbox_multicast_ptr,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        scratch,
        error_flag,
        tp_rank,
        active_tokens,
        workspace_signature,
    )
    tokens, ranks, shard_columns = output_mailbox.shape
    return output_mailbox.view(tokens, ranks * shard_columns)[:active_tokens]
