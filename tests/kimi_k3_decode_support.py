"""Shared fixtures and the oracle for the one-launch Kimi K3 decode kernel.

``test_kimi_k3_decode.py`` drives the production
``mok.kimi_k3.kimi_k3_decode`` on all eight ranks. Everything it needs that is
not itself an assertion lives here, so the test file stays a list of claims:
the prepared TP8 weights, the routing constructions that pin which experts a
token reaches, and the reference the device output is measured against.

All of it requires all eight ranks, so it must be launched through
``torchrun --standalone --nproc-per-node=8``. Each rank owns a *distinct* shard
of every routed and shared expert, so a rank-local implementation cannot pass:
the reference sums the per-rank partials the same way the fused tail does.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from mok import _C
from mok.kimi_k3 import (
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_MAX_TOKENS,
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    KIMI_K3_SHARED_INTERMEDIATE_SIZE,
    KIMI_K3_TOPK,
    KIMI_K3_TP_SIZE,
    KIMI_K3_W1W3_K,
    KimiK3DecodeConfig,
    KimiK3DecodeWeights,
    KimiK3DecodeWorkspace,
    dequant_kimi_k3_mxfp4,
    kimi_k3_decode,
    kimi_k3_rmsnorm_reference,
    kimi_k3_router_reference,
    kimi_k3_situ_reference,
    pack_kimi_k3_mxfp4,
)

# Re-exported so the test module gets the tail suite's workspace and scratch
# model rather than a second, independently drifting copy of either.
from .kimi_k3_tail_support import (  # noqa: F401
    SCRATCH_BYTES,
    SCRATCH_LAYOUT,
    UINT32_MAX,
    _as_int32,
    _phase,
    _region,
    _synchronize_ranks,
    workspace,
)


HIDDEN = KIMI_K3_HIDDEN_SIZE
LATENT = KIMI_K3_LATENT_SIZE
EXPERTS = KIMI_K3_NUM_EXPERTS
TOPK = KIMI_K3_TOPK
MAX_TOKENS = KIMI_K3_MAX_TOKENS
ROUTED_PER_RANK = KIMI_K3_ROUTED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE
SHARED_PER_RANK = KIMI_K3_SHARED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE
GROUP = 32
UNIT_SCALE = 0x7F

# Raw decode counts, then the two DFlash request shapes: one request is a
# fixed-size block of key/value rows, so a batch of `n` requests decodes
# `n * block` rows at once.
RAW_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)
BLOCK8_TOKENS = tuple(8 * batch for batch in range(1, 9))
BLOCK16_TOKENS = tuple(16 * batch for batch in range(1, 9))

# The capacity bucket switches from the CUDA-core stages to the tcgen05 stages
# above eight rows, so both sides of that boundary have to be represented
# wherever a test only affords two shapes.
CORE_TOKENS = 5
TENSOR_TOKENS = 24

# Every K3 kernel the private stages launch. The production call must show none
# of them, because it must not launch them at all.
PRIVATE_STAGE_KERNELS = (
    "route_and_project_core_kernel",
    "route_and_project_tensor_kernel",
    "kimi_k3_routed_experts_kernel",
    "shared_experts_core_kernel",
    "shared_experts_tensor_kernel",
    "kimi_k3_tail_core_kernel",
    "kimi_k3_tail_tensor_kernel",
)
PERSISTENT_KERNEL = "kimi_k3_decode_persistent_kernel"

# Enough experts to keep the oracle's weight traffic bounded without making its
# Python loop the cost of the suite.
_DEQUANT_CHUNK = 64

CONFIG = KimiK3DecodeConfig()

PERSISTENT_CTAS = 148
PERSISTENT_THREADS = 256

# The phase slots the persistent scheduler owns. Only the last four are bound
# by the extension, and ``persistent_sync.cuh`` static-asserts that the three
# queue counters sit directly below the activation counter, so deriving them
# keeps this mirror honest without adding three more bindings.
(
    TIMEOUT_PHASE,
    GRID_GENERATION,
    ACTIVATION_ARRIVALS,
    ACTIVE_EXPERT_UNITS,
) = _C._kimi_k3_decode_timeout_metadata()
ROUTE_LATENT_QUEUE = ACTIVATION_ARRIVALS - 3
GATE_UP_QUEUE = ACTIVATION_ARRIVALS - 2
DOWN_QUEUE = ACTIVATION_ARRIVALS - 1


def decode_step(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """One production step, with the launch's own error flag checked."""
    result = kimi_k3_decode(CONFIG, workspace, weights, hidden)
    assert int(workspace.error_flag.item()) == 0, (
        f"the persistent kernel timed out waiting on phase counter "
        f"{int(_phase(workspace.scratch)[TIMEOUT_PHASE].item())}"
    )
    return result


def low_level_arguments(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    hidden: torch.Tensor,
    active_tokens: int,
) -> dict[str, object]:
    """Exactly what the high-level wrapper forwards, as a mutable mapping."""
    return {
        "hidden_states": hidden,
        "router_weight": weights.router_weight,
        "router_correction_bias": weights.router_correction_bias,
        "routed_expert_down_proj": weights.routed_expert_down_proj,
        "routed_expert_up_proj": weights.routed_expert_up_proj,
        "routed_latent_rmsnorm_weight": weights.routed_latent_rmsnorm_weight,
        "expert_w1_packed": weights.expert_w1_packed,
        "expert_w1_scale": weights.expert_w1_scale,
        "expert_w3_packed": weights.expert_w3_packed,
        "expert_w3_scale": weights.expert_w3_scale,
        "expert_w2_packed": weights.expert_w2_packed,
        "expert_w2_scale": weights.expert_w2_scale,
        "shared_gate_proj": weights.shared_gate_proj,
        "shared_up_proj": weights.shared_up_proj,
        "shared_down_proj": weights.shared_down_proj,
        "scratch": workspace.scratch,
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
        "error_flag": workspace.error_flag,
        "tp_rank": workspace.tp_rank,
        "active_tokens": active_tokens,
        "workspace_signature": workspace.workspace_signature,
    }


# ---------------------------------------------------------------------------
# Prepared TP8 weights.
# ---------------------------------------------------------------------------


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _normal(
    shape: tuple[int, ...],
    device: torch.device,
    seed: int,
    deviation: float,
) -> torch.Tensor:
    values = torch.randn(
        shape,
        generator=_generator(device, seed),
        dtype=torch.float32,
        device=device,
    )
    return (values * deviation).bfloat16().contiguous()


def _pack_expert_matrix(
    device: torch.device,
    seed: int,
    rows: int,
    columns: int,
    padded_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw and pack one rank's 896-expert shard a chunk of experts at a time.

    The dense BF16 form of a whole shard is 2.5 GiB and its FP32 draw twice
    that, so the experts are drawn, packed, and released in chunks. Only the
    packed result -- what the kernel actually reads -- is ever held whole.
    """
    deviation = 1.0 / math.sqrt(columns)
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    chunk = 112
    for start in range(0, EXPERTS, chunk):
        dense = _normal(
            (chunk, rows, columns), device, seed + start, deviation
        )
        packed, scale = pack_kimi_k3_mxfp4(dense, padded_k=padded_k)
        del dense
        packed_chunks.append(packed)
        scale_chunks.append(scale)
    return (
        torch.cat(packed_chunks).contiguous(),
        torch.cat(scale_chunks).contiguous(),
    )


def _build_weights(
    device: torch.device, tp_rank: int
) -> KimiK3DecodeWeights:
    """Build this rank's prepared shard without materializing the whole model.

    ``prepare_kimi_k3_decode_weights`` takes the replicated ``[896, 3072,
    3584]`` expert tensors, which are 20 GiB apiece. Nothing downstream can
    tell a shard sliced out of such a tensor from one drawn directly at shard
    width, because a rank only ever reads its own 384 intermediate rows, so the
    shards are drawn directly and seeded by rank. Concatenating the eight
    shards would recover one perfectly ordinary model.

    The replicated tensors -- the router, both latent projections, and the
    RMSNorm weight -- are seeded without the rank, so every rank draws the same
    values. ``assert_weights_are_consistent`` checks that afterwards.
    """
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 16 * 1024**3:
        pytest.skip("prepared Kimi K3 decode weights need 16 GiB free")

    shard = 1_000_000 * (tp_rank + 1)
    w1_packed, w1_scale = _pack_expert_matrix(
        device, shard + 11, ROUTED_PER_RANK, LATENT, KIMI_K3_W1W3_K
    )
    w3_packed, w3_scale = _pack_expert_matrix(
        device, shard + 22, ROUTED_PER_RANK, LATENT, KIMI_K3_W1W3_K
    )
    w2_packed, w2_scale = _pack_expert_matrix(
        device, shard + 33, LATENT, ROUTED_PER_RANK, ROUTED_PER_RANK
    )
    return KimiK3DecodeWeights(
        router_weight=_normal(
            (EXPERTS, HIDDEN), device, 4_001, 1.0 / math.sqrt(HIDDEN)
        ),
        router_correction_bias=torch.zeros(
            EXPERTS, dtype=torch.float32, device=device
        ),
        routed_expert_down_proj=_normal(
            (LATENT, HIDDEN), device, 4_002, 1.0 / math.sqrt(HIDDEN)
        ),
        routed_expert_up_proj=_normal(
            (HIDDEN, LATENT), device, 4_003, 1.0 / math.sqrt(LATENT)
        ),
        routed_latent_rmsnorm_weight=(
            1.0
            + 0.25 * _normal((LATENT,), device, 4_004, 1.0).float()
        ).bfloat16().contiguous(),
        expert_w1_packed=w1_packed,
        expert_w1_scale=w1_scale,
        expert_w3_packed=w3_packed,
        expert_w3_scale=w3_scale,
        expert_w2_packed=w2_packed,
        expert_w2_scale=w2_scale,
        shared_gate_proj=_normal(
            (SHARED_PER_RANK, HIDDEN), device, shard + 44,
            1.0 / math.sqrt(HIDDEN)
        ),
        shared_up_proj=_normal(
            (SHARED_PER_RANK, HIDDEN), device, shard + 55,
            1.0 / math.sqrt(HIDDEN)
        ),
        shared_down_proj=_normal(
            (HIDDEN, SHARED_PER_RANK), device, shard + 66,
            1.0 / math.sqrt(SHARED_PER_RANK)
        ),
        tp_rank=tp_rank,
    )


@pytest.fixture(scope="module")
def weights(
    tp8_context: tuple[int, int, torch.device],
) -> Iterator[KimiK3DecodeWeights]:
    rank, _, device = tp8_context
    built = _build_weights(device, rank)
    try:
        yield built
    finally:
        del built
        torch.cuda.empty_cache()


def assert_replicated(name: str, tensor: torch.Tensor) -> None:
    """Fail loudly if a tensor every rank must agree on drifted between them."""
    values = tensor.float().contiguous()
    largest = values.clone()
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    assert torch.equal(largest, values), name


def assert_distinct(name: str, tensor: torch.Tensor) -> None:
    """Fail loudly if a rank-owned shard is accidentally the same everywhere.

    A strided sample stands in for the whole tensor because the packed expert
    shards are 616 MiB apiece and eight of them would not fit the comparison,
    let alone the all-gather. A sample of a hundred thousand elements cannot
    agree between two independently drawn shards by accident.
    """
    sample = tensor.flatten()[::4099].to(torch.float64).contiguous()
    gathered = [torch.empty_like(sample) for _ in range(KIMI_K3_TP_SIZE)]
    dist.all_gather(gathered, sample)
    rank = dist.get_rank()
    for peer, other in enumerate(gathered):
        if peer != rank:
            assert not torch.equal(other, sample), (name, peer)


# ---------------------------------------------------------------------------
# Inputs and the routings the tests pin.
# ---------------------------------------------------------------------------


def hidden_states(
    device: torch.device, tokens: int, seed: int = 7_777
) -> torch.Tensor:
    """Replicated activations: every rank decodes the same tokens."""
    return _normal((tokens, HIDDEN), device, seed, 1.0)


@dataclass(frozen=True, slots=True)
class Routing:
    """One pinned routing: the inputs that produce it and what it asserts."""

    hidden: torch.Tensor
    router_weight: torch.Tensor
    correction_bias: torch.Tensor


def _one_hot_hidden(device: torch.device, tokens: int) -> torch.Tensor:
    """Give token ``t`` its own hidden direction so routing can be per token."""
    values = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=device)
    columns = torch.arange(tokens, device=device)
    values[columns, columns] = 8.0
    return values.bfloat16().contiguous()


def routing(
    mode: str,
    device: torch.device,
    tokens: int,
    base: KimiK3DecodeWeights,
) -> Routing:
    """Construct inputs whose top-16 selection is known before the call.

    ``sigmoid`` bounds every raw score to ``(0, 1)``, so a correction bias of
    ``+8`` on a set of experts puts exactly that set at the top of the
    corrected ranking whatever the activations do. That pins the three
    *placement* cases -- the lowest, middle, and final expert IDs -- and the
    concentrated case where every token shares one set.

    ``disjoint`` needs a different set per token, which a per-expert bias
    cannot express, so it comes from the activations instead: token ``t``
    carries a single nonzero hidden column ``t`` and expert ``e`` reads only
    column ``e // 16``, which hands token ``t`` exactly experts
    ``[16t, 16t + 16)``. The small per-expert gain keeps the sixteen raw scores
    apart, so the normalized router weights stay distinct too.
    """
    if mode == "balanced":
        return Routing(
            hidden_states(device, tokens),
            base.router_weight,
            base.router_correction_bias,
        )
    if mode == "disjoint":
        assert tokens * TOPK <= EXPERTS
        weight = torch.zeros(
            EXPERTS, HIDDEN, dtype=torch.float32, device=device
        )
        experts = torch.arange(EXPERTS, device=device)
        weight[experts, experts // TOPK] = (
            0.125 + 0.0078125 * (experts % TOPK).float()
        )
        return Routing(
            _one_hot_hidden(device, tokens),
            weight.bfloat16().contiguous(),
            base.router_correction_bias,
        )
    chosen = {
        "concentrated": range(311, 311 + TOPK),
        "low": range(0, TOPK),
        "middle": range(EXPERTS // 2, EXPERTS // 2 + TOPK),
        "final": range(EXPERTS - TOPK, EXPERTS),
    }[mode]
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    bias[torch.tensor(list(chosen), device=device)] = 8.0
    return Routing(hidden_states(device, tokens), base.router_weight, bias)


def with_routing(
    base: KimiK3DecodeWeights, plan: Routing
) -> KimiK3DecodeWeights:
    """Return ``base`` with only its router replaced."""
    return dataclasses.replace(
        base,
        router_weight=plan.router_weight,
        router_correction_bias=plan.correction_bias,
    )


# ---------------------------------------------------------------------------
# The oracle.
# ---------------------------------------------------------------------------


def _e8m0_scale_bytes(absolute_max: torch.Tensor) -> torch.Tensor:
    """Model ``select_e8m0_scale`` over the whole float range.

    OCP MX v1.0 and PTX ISA 9.3 define the E8M0 scale as ``2^(byte - 127)``
    with byte 255 reserved for NaN, so byte 0 is the exact minimum and byte 254
    the maximum. The chosen scale is the smallest power of two that keeps
    ``absolute_max / scale`` within E4M3's 448, and ``448 == 1.75 * 2^8`` is
    what puts the mantissa threshold at 1.75.
    """
    mantissa, exponent = torch.frexp(absolute_max.float())
    # frexp returns a mantissa in [0.5, 1), so 1.75 in [1, 2) becomes 0.875.
    scale_exponent = torch.where(mantissa <= 0.875, exponent - 9, exponent - 8)
    scale_bytes = (scale_exponent + 127).clamp(0, 254).to(torch.uint8)
    return torch.where(
        absolute_max == 0,
        torch.full_like(scale_bytes, UNIT_SCALE),
        scale_bytes,
    )


def mxfp8_dequantized(values: torch.Tensor) -> torch.Tensor:
    """Round through block-32 MXFP8 exactly as the mixed W4A8 path does.

    Both activation operands of the routed experts are quantized on the device
    before the MMA reads them: the projected latent once per step, and each
    expert's SiTU intermediate once per gate/up unit. Modeling that here is
    what makes the comparison sensitive -- an oracle that skipped it would
    disagree with a *correct* kernel by the several percent E4M3 costs, and
    would then have to accept a tolerance wide enough to hide a real fault.
    """
    grouped = values.float().reshape(*values.shape[:-1], -1, GROUP)
    scale_bytes = _e8m0_scale_bytes(grouped.abs().amax(dim=-1))
    scale = torch.pow(2.0, (scale_bytes.int() - 127).float()).unsqueeze(-1)
    quantized = (grouped / scale).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(values.shape)


def _situ_fp32(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The FP32 SiTU contract, evaluated where the kernel evaluates it."""
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )


def routed_partial_reference(
    latent: torch.Tensor,
    weights: KimiK3DecodeWeights,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    """This rank's routed contribution, from its own dequantized shard.

    The routed expert splits along the intermediate dimension, and SiTU is
    elementwise there, so an expert's output is exactly the sum of the eight
    ranks' partials. That is why this stays rank-local: the sum happens in the
    same all-reduce the fused tail performs.
    """
    device = latent.device
    active = expert_ids.shape[0]
    quantized = mxfp8_dequantized(latent[:active].float())
    partial = torch.zeros(active, LATENT, dtype=torch.float32, device=device)
    unique = torch.unique(expert_ids)
    for start in range(0, unique.numel(), _DEQUANT_CHUNK):
        chunk = unique[start:start + _DEQUANT_CHUNK]
        w1 = dequant_kimi_k3_mxfp4(
            weights.expert_w1_packed[chunk],
            weights.expert_w1_scale[chunk],
            logical_k=LATENT,
        )
        w3 = dequant_kimi_k3_mxfp4(
            weights.expert_w3_packed[chunk],
            weights.expert_w3_scale[chunk],
            logical_k=LATENT,
        )
        w2 = dequant_kimi_k3_mxfp4(
            weights.expert_w2_packed[chunk],
            weights.expert_w2_scale[chunk],
            logical_k=ROUTED_PER_RANK,
        )
        for position, expert in enumerate(chunk.tolist()):
            tokens, slots = torch.where(expert_ids == expert)
            selected = quantized.index_select(0, tokens)
            situ = _situ_fp32(
                selected @ w1[position].float().T,
                selected @ w3[position].float().T,
            )
            contribution = (
                mxfp8_dequantized(situ) @ w2[position].float().T
            ) * router_weights[tokens, slots].unsqueeze(-1)
            partial.index_add_(0, tokens, contribution)
        del w1, w3, w2
    return partial


def shared_partial_reference(
    hidden: torch.Tensor,
    weights: KimiK3DecodeWeights,
    *,
    round_projections: bool = True,
) -> torch.Tensor:
    """This rank's shared-expert contribution, in BF16 where the kernel is.

    The gate and up projections accumulate in FP32 and are then *stored* as
    BF16 -- ``shared.cuh`` writes ``scratch.shared_gate`` and
    ``scratch.shared_up`` and reads them back to evaluate SiTU -- so the
    activation sees rounded inputs, not the accumulator. Rounding here is what
    makes the two sides agree at the same boundary the official model defines.

    ``round_projections=False`` feeds the raw FP32 accumulators to SiTU
    instead. Nothing but
    ``test_the_shared_partial_matches_the_bf16_rounded_boundary`` passes it:
    it exists so that test can show the device picks the rounded boundary
    rather than merely being close to both.
    """
    activations = hidden.float()
    gate = activations @ weights.shared_gate_proj.float().T
    up = activations @ weights.shared_up_proj.float().T
    if round_projections:
        gate = gate.bfloat16()
        up = up.bfloat16()
    activated = kimi_k3_situ_reference(gate, up)
    return activated.float() @ weights.shared_down_proj.float().T


def _all_reduced(partial: torch.Tensor) -> torch.Tensor:
    """Sum eight BF16 partials, the dtype the collective buffer carries."""
    reduced = partial.bfloat16().contiguous()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def decode_reference(
    hidden: torch.Tensor, weights: KimiK3DecodeWeights
) -> torch.Tensor:
    """The whole TP8 decode step, assembled the way the megakernel assembles it.

    Both partials cross the fabric in BF16 and both latent projections and the
    RMSNorm are replicated, so this mirrors the device's dtype boundaries as
    well as its arithmetic. It is collective: every rank must call it.
    """
    latent = (
        hidden.float() @ weights.routed_expert_down_proj.float().T
    ).bfloat16()
    expert_ids, router_weights = kimi_k3_router_reference(
        hidden, weights.router_weight, weights.router_correction_bias
    )
    routed = _all_reduced(
        routed_partial_reference(latent, weights, expert_ids, router_weights)
    )
    shared = _all_reduced(shared_partial_reference(hidden, weights))
    normalized = kimi_k3_rmsnorm_reference(
        routed, weights.routed_latent_rmsnorm_weight
    )
    return (
        normalized.float() @ weights.routed_expert_up_proj.float().T
        + shared.float()
    ).bfloat16()


def accuracy(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, float]:
    """Relative L1, cosine similarity, and the largest absolute deviation."""
    left = actual.float()
    right = expected.float()
    difference = left - right
    return (
        float(difference.abs().sum() / right.abs().sum().clamp_min(1e-12)),
        float(
            torch.nn.functional.cosine_similarity(
                left.flatten(), right.flatten(), dim=0
            )
        ),
        float(difference.abs().max()),
    )


def assert_decode_close(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, float]:
    """Require the device output to match the oracle, and report by how much.

    What separates the two is the order eight BF16 partials are summed in --
    one FP32-accumulating multimem instruction against an NCCL tree -- plus
    tcgen05's accumulation order inside each contraction. Both are last-place
    effects on a value whose scale the tolerance follows, which is why the
    absolute bound is relative to the largest expected magnitude rather than a
    constant.
    """
    relative_l1, cosine, maximum = accuracy(actual, expected)
    tolerance = 0.05 * float(expected.float().abs().max()) + 0.125
    assert torch.isfinite(actual.float()).all()
    assert relative_l1 <= 0.05, (relative_l1, cosine, maximum)
    assert cosine >= 0.999, (relative_l1, cosine, maximum)
    assert maximum <= tolerance, (relative_l1, cosine, maximum, tolerance)
    return relative_l1, cosine, maximum


def assert_identical_across_ranks(values: torch.Tensor) -> None:
    """Every rank must hold byte-identical output for the same step."""
    local = values.float().contiguous()
    smallest = local.clone()
    largest = local.clone()
    dist.all_reduce(smallest, op=dist.ReduceOp.MIN)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    assert torch.equal(smallest, largest)
    assert torch.equal(smallest, local)


# ---------------------------------------------------------------------------
# Device-side observation.
# ---------------------------------------------------------------------------


def profiled_kernel_names(call: Callable[[], object]) -> list[str]:
    """Names of every CUDA kernel the profiler attributes to ``call()``."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        # ``export_chrome_trace`` renames a temporary file into place, so the
        # trace has to be reopened by path once the export has returned.
        trace_path = os.path.join(directory, "trace.json")
        profiler.export_chrome_trace(trace_path)
        with open(trace_path, encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
    return [
        event["name"]
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
    ]


def published_shared_partial(
    collective_buffer: torch.Tensor, active_tokens: int
) -> torch.Tensor:
    """This rank's own shared-expert partial, exactly as the launch left it.

    Both of the tail's reductions are multimem loads, so nothing in the step
    writes the collective buffer after the shared-down units do. The local copy
    therefore still holds this rank's contribution alone, unsummed.
    """
    return collective_buffer[:active_tokens, LATENT:]


def published_routes(
    scratch: torch.Tensor, active_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The expert IDs and normalized weights the router left in scratch."""
    ids = _region(scratch, "expert_ids", torch.int32)[
        : active_tokens * TOPK
    ].view(active_tokens, TOPK)
    weights = _region(scratch, "expert_weights", torch.float32)[
        : active_tokens * TOPK
    ].view(active_tokens, TOPK)
    return ids, weights


def poison_scratch(workspace_scratch: torch.Tensor) -> None:
    """Fill every region a launch is expected to re-establish with garbage.

    The phase counters are left alone: they carry the wrap-safe generation
    state one launch hands the next, and a launch is entitled to trust it. What
    a launch may *not* trust is any data region, so all of them are poisoned
    with a value no correct output could survive.
    """
    for name, (offset, size) in SCRATCH_LAYOUT.items():
        if name in {"phase", "total_bytes"}:
            continue
        workspace_scratch[offset:offset + size].fill_(0x5A)
