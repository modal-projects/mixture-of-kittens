"""The fixtures, the structured weights, and the oracle the expert tests share.

The routed expert stage is checked from three angles in three sibling files,
and all three rest on the same construction: one set of deliberately
structured expert weights whose gate/up rows are displaced by whole latent
scale groups and whose down rows carry a non-periodic gain, so that reading the
wrong expert, the wrong row, or the wrong scale group changes the output in a
way no coincidence reproduces. That construction, the scratch layout it is
written into, the MXFP8 quantization reference it is compared against, and the
fixtures that hold it live here.

The three files are ``test_kimi_k3_expert.py`` for what the instruction and the
stage compute, ``test_kimi_k3_expert_addressing.py`` for which expert and which
rows a routing reaches, and ``test_kimi_k3_expert_contract.py`` for the host
boundary.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


HIDDEN = 3584
INTERMEDIATE = 384
EXPERTS = 896
TOPK = 16
MAX_TOKENS = 128
MAX_ASSIGNMENTS = MAX_TOKENS * TOPK
CAPACITY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
PROBE_COLUMNS = 128
GAIN_BINADES = 5
ADDRESS_EXPERTS = (1, 447, 895)
# Deliberately lopsided so reversing a token's two experts across its two slots
# changes the result by 0.75 of the difference between the two expert outputs.
ADDRESS_WEIGHTS = (0.125, 0.875)
GROUP = 32
ALIGNMENT = 256
UNIT_SCALE = 0x7F

_EXPERT_ARGUMENTS = (
    "latent_x",
    "expert_w1_packed",
    "expert_w1_scale",
    "expert_w3_packed",
    "expert_w3_scale",
    "expert_w2_packed",
    "expert_w2_scale",
    "routed_output",
    "scratch",
    "active_tokens",
)


def _aligned(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _scratch_layout() -> dict[str, tuple[int, int]]:
    """Independent byte-level model of ``kimi_k3_decode::Scratch``."""
    regions = (
        ("phase", 128 * 4),
        ("expert_ids", MAX_ASSIGNMENTS * 4),
        ("expert_weights", MAX_ASSIGNMENTS * 4),
        ("expert_counts", EXPERTS * 4),
        ("expert_offsets", (EXPERTS + 1) * 4),
        ("assignment_tokens", MAX_ASSIGNMENTS * 4),
        ("assignment_slots", MAX_ASSIGNMENTS * 4),
        ("latent_mxfp8", MAX_TOKENS * HIDDEN),
        ("latent_scale", MAX_TOKENS * (HIDDEN // GROUP)),
        ("situ_mxfp8", MAX_ASSIGNMENTS * INTERMEDIATE),
        ("situ_scale", MAX_ASSIGNMENTS * (INTERMEDIATE // GROUP)),
        ("routed_accumulator", MAX_TOKENS * HIDDEN * 8),
        ("shared_gate", MAX_TOKENS * 768 * 2),
        ("shared_up", MAX_TOKENS * 768 * 2),
        ("shared_activated", MAX_TOKENS * 768 * 2),
        ("tail_normalized", MAX_TOKENS * HIDDEN * 2),
        ("tail_shared_shard", MAX_TOKENS * 896 * 2),
        ("latent_x", MAX_TOKENS * HIDDEN * 2),
        ("unit_expert", EXPERTS * 4),
        ("router_scores", MAX_TOKENS * EXPERTS * 4),
        ("schedule", 128 * 4),
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


@dataclass
class ExpertWeights:
    w1_packed: torch.Tensor
    w1_scale: torch.Tensor
    w3_packed: torch.Tensor
    w3_scale: torch.Tensor
    w2_packed: torch.Tensor
    w2_scale: torch.Tensor

    def arguments(self) -> tuple[torch.Tensor, ...]:
        return (
            self.w1_packed,
            self.w1_scale,
            self.w3_packed,
            self.w3_scale,
            self.w2_packed,
            self.w2_scale,
        )


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Kimi K3 routed experts require CUDA")
    selected = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(selected)
    if torch.cuda.get_device_capability(selected) != (10, 3):
        pytest.skip("Kimi K3 routed experts require an SM103 GPU")
    return selected


@pytest.fixture(scope="module")
def peer_device(device: torch.device) -> Iterator[torch.device]:
    if torch.cuda.device_count() < 2:
        pytest.skip("cross-device Kimi K3 experts need two visible GPUs")
    peer = torch.device("cuda", 1 if device.index == 0 else 0)
    if torch.cuda.get_device_capability(peer) != (10, 3):
        pytest.skip("Kimi K3 routed experts require an SM103 GPU")
    try:
        yield peer
    finally:
        torch.cuda.set_device(device)


def _latent_column_for_row(row: int) -> int:
    """Spread the 384 gate/up rows across all 112 latent scale groups."""
    return row * HIDDEN // INTERMEDIATE


def _down_row_gain(device: torch.device, phase: int = 0) -> torch.Tensor:
    """Give each of the 3584 down-projection rows a distinguishable gain.

    Each exponent comes from an integer hash of the row index, so the sequence
    is nonperiodic: a tile written to the wrong ``output_base``, a row displaced
    by a whole intermediate, and any other displacement all change the result
    instead of landing on an identical value.  A fixed period would have been
    blind to a displacement of exactly that period.  ``phase`` shifts the hash
    input, which gives one expert a shard tag no other expert repeats.  Every
    gain is a power of two in ``[2^-4, 1]`` and so exactly representable in
    E2M1, which keeps the dequantized reference bit-exact.
    """
    rows = torch.arange(HIDDEN, dtype=torch.int64, device=device)
    mixed = (rows + phase + 1) * 0x9E3779B1
    mixed = mixed ^ (mixed >> 15)
    mixed = mixed ^ (mixed >> 7)
    return torch.pow(2.0, -(mixed % GAIN_BINADES).float()).bfloat16()


def _make_structured_weights(device: torch.device) -> ExpertWeights:
    """Make full native-layout weights with an exactly representable FP4 path.

    Gate row ``r`` selects latent column ``_latent_column_for_row(r)`` with
    ``+1.0`` and the matching up row selects the same column with ``+0.5``, so
    the two projections stay distinguishable through the asymmetric SiTU while
    every SiTU value keeps the sign of ``gate * up`` and therefore stays
    positive.  Every W2 row then sums all 384 SiTU columns at the row's own
    ``_down_row_gain``, so the 3584-wide result varies across output tiles
    instead of repeating one value.  Summing 384
    positive terms is what makes the end-to-end error metric average the
    mandatory MXFP8 activation and SiTU roundings instead of exposing one
    E4M3 rounding, whose worst case alone is 6.25%.

    The selected columns walk the whole 3584-wide latent, so both nibbles of
    the packed FP4 byte, all 112 W1/W3 scale groups, all 12 W2 scale groups,
    and all-zero groups are exercised.  Every value is a power of two that
    survives MXFP4 exactly, so the dequantized prepared weights the reference
    consumes are bit-exact.  The same shard is installed for all 896 experts so
    the exhaustive assignment distribution keeps an analytical reference.
    """
    from mok.ops import pack_kimi_k3_mxfp4

    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 4 * 1024**3:
        pytest.skip("full Kimi K3 prepared expert tensors need 4 GiB free")
    rows = torch.arange(INTERMEDIATE, device=device)
    columns = torch.tensor(
        [_latent_column_for_row(row) for row in range(INTERMEDIATE)],
        device=device,
    )
    gate_dense = torch.zeros(
        1, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device
    )
    gate_dense[0, rows, columns] = 1.0
    up_dense = torch.zeros_like(gate_dense)
    up_dense[0, rows, columns] = 0.5
    down_dense = _down_row_gain(device).view(1, HIDDEN, 1).expand(
        1, HIDDEN, INTERMEDIATE
    ).contiguous()

    def _broadcast(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed, scale = pack_kimi_k3_mxfp4(dense, dense.size(-1))
        return (
            packed.expand(EXPERTS, -1, -1).contiguous(),
            scale.expand(EXPERTS, -1, -1).contiguous(),
        )

    w1_packed, w1_scale = _broadcast(gate_dense)
    w3_packed, w3_scale = _broadcast(up_dense)
    w2_packed, w2_scale = _broadcast(down_dense)
    del gate_dense, up_dense, down_dense
    return ExpertWeights(
        w1_packed, w1_scale, w3_packed, w3_scale, w2_packed, w2_scale
    )


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact FP32 Kimi K3 SiTU contract."""
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )


def _random_latent(
    device: torch.device, rows: int, seed: int
) -> torch.Tensor:
    """Draw a dense latent so every gate/up row and scale group is active."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            rows, HIDDEN, generator=generator, dtype=torch.float32, device=device
        )
        * 0.25
    ).bfloat16()


@pytest.fixture(scope="module")
def weights(device: torch.device) -> Iterator[ExpertWeights]:
    value = _make_structured_weights(device)
    try:
        yield value
    finally:
        del value
        torch.cuda.empty_cache()


@pytest.fixture
def scratch(device: torch.device) -> torch.Tensor:
    return torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)


def _region(scratch: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    offset, size = SCRATCH_LAYOUT[name]
    return scratch[offset:offset + size].view(dtype)


Assignment = tuple[int, int, int, float]  # expert, token, slot, normalized weight


def _write_assignments(
    scratch: torch.Tensor, assignments: Sequence[Assignment]
) -> None:
    """Publish the exact expert-major state produced by Task 5."""
    device = scratch.device
    ordered = sorted(assignments, key=lambda item: (item[0], item[1], item[2]))
    counts = torch.zeros(EXPERTS, dtype=torch.int32, device=device)
    if ordered:
        expert_tensor = torch.tensor(
            [item[0] for item in ordered], dtype=torch.int64, device=device
        )
        counts.scatter_add_(
            0, expert_tensor, torch.ones_like(expert_tensor, dtype=torch.int32)
        )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts.cumsum(0).to(torch.int32),
        )
    )

    _region(scratch, "expert_ids", torch.int32).zero_()
    _region(scratch, "expert_weights", torch.float32).zero_()
    _region(scratch, "expert_counts", torch.int32).copy_(counts)
    _region(scratch, "expert_offsets", torch.int32).copy_(offsets)
    assignment_tokens = _region(scratch, "assignment_tokens", torch.int32)
    assignment_slots = _region(scratch, "assignment_slots", torch.int32)
    assignment_tokens.zero_()
    assignment_slots.zero_()
    for position, (expert, token, slot, weight) in enumerate(ordered):
        route = token * TOPK + slot
        _region(scratch, "expert_ids", torch.int32)[route] = expert
        _region(scratch, "expert_weights", torch.float32)[route] = weight
        assignment_tokens[position] = token
        assignment_slots[position] = slot


def _call(
    latent_x: torch.Tensor,
    weights: ExpertWeights,
    routed_output: torch.Tensor,
    scratch: torch.Tensor,
    active_tokens: int,
) -> torch.Tensor:
    from mok import ops

    return ops._kimi_k3_routed_experts(
        latent_x,
        *weights.arguments(),
        routed_output,
        scratch,
        active_tokens,
    )


def _dequant_one(
    packed: torch.Tensor, scale: torch.Tensor, logical_k: int, expert: int
) -> torch.Tensor:
    from mok.ops import dequant_kimi_k3_mxfp4

    return dequant_kimi_k3_mxfp4(
        packed[expert:expert + 1], scale[expert:expert + 1], logical_k
    )[0].float()


def _reference(
    latent_x: torch.Tensor,
    weights: ExpertWeights,
    assignments: Sequence[Assignment],
    active_tokens: int,
) -> torch.Tensor:
    """Dequantize the exact prepared weights and evaluate the FP32 contract."""
    result = torch.zeros(
        active_tokens, HIDDEN, dtype=torch.float32, device=latent_x.device
    )
    by_expert: dict[int, list[Assignment]] = {}
    for assignment in assignments:
        by_expert.setdefault(assignment[0], []).append(assignment)
    for expert, selected in by_expert.items():
        tokens = torch.tensor(
            [item[1] for item in selected], dtype=torch.int64, device=latent_x.device
        )
        route_weights = torch.tensor(
            [item[3] for item in selected],
            dtype=torch.float32,
            device=latent_x.device,
        )
        w1 = _dequant_one(weights.w1_packed, weights.w1_scale, HIDDEN, expert)
        w3 = _dequant_one(weights.w3_packed, weights.w3_scale, HIDDEN, expert)
        w2 = _dequant_one(weights.w2_packed, weights.w2_scale, INTERMEDIATE, expert)
        selected_x = latent_x.index_select(0, tokens).float()
        situ = _situ(selected_x @ w1.T, selected_x @ w3.T)
        contribution = (situ @ w2.T) * route_weights[:, None]
        result.index_add_(0, tokens, contribution)
    return result.bfloat16().float()


def _assert_expert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    relative_l1 = difference.abs().sum() / expected.abs().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    )
    maximum = difference.abs().max()
    assert torch.isfinite(actual).all()
    assert float(relative_l1) <= 0.05
    assert float(cosine) >= 0.999
    assert float(maximum) <= 1.0


def _e8m0_scale_bytes(absolute_max: torch.Tensor) -> torch.Tensor:
    """Model the kernel's E8M0 scale-byte selection over the whole float range.

    OCP MX v1.0 and PTX ISA 9.3 both define the E8M0 scale as ``2^(byte - 127)``
    with byte 255 reserved for NaN, so byte 0 is the exact minimum scale
    ``2^-127`` and byte 254 the maximum.  The selected scale is the smallest
    power of two that keeps ``absolute_max / scale`` inside E4M3's 448 maximum,
    and ``448 == 1.75 * 2^8`` is what puts the mantissa threshold at 1.75.
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


def _mxfp8_quantize_reference(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the dequantized block-32 MXFP8 values and their E8M0 scale bytes."""
    grouped = values.float().reshape(*values.shape[:-1], -1, GROUP)
    scale_bytes = _e8m0_scale_bytes(grouped.abs().amax(dim=-1))
    scale = torch.pow(2.0, (scale_bytes.int() - 127).float()).unsqueeze(-1)
    quantized = (grouped / scale).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(values.shape), scale_bytes


def _mxfp8_dequant_reference(values: torch.Tensor) -> torch.Tensor:
    """Reference the E4M3/E8M0 quantizer used by mixed MMA.

    Derived from the scale byte rather than from ``log2``, so it stays exact at
    the bottom of the E8M0 range where a float ``absolute_max / 448`` becomes
    subnormal.  Torch keeps subnormals, so this models the quantizer the kernel
    would run if the extension were not built with ``-ftz=true``.
    """
    dequantized, _ = _mxfp8_quantize_reference(values)
    return dequantized


def _published_latent(scratch: torch.Tensor, rows: int) -> torch.Tensor:
    """Dequantize the MXFP8 latent the quantization stage published."""
    codes = _region(scratch, "latent_mxfp8", torch.uint8)[
        : rows * HIDDEN
    ].view(rows, HIDDEN)
    scale_bytes = _region(scratch, "latent_scale", torch.uint8)[
        : rows * (HIDDEN // GROUP)
    ].view(rows, HIDDEN // GROUP)
    scale = torch.pow(2.0, (scale_bytes.int() - 127).float())
    values = codes.view(torch.float8_e4m3fn).float().view(rows, -1, GROUP)
    return (values * scale.unsqueeze(-1)).reshape(rows, HIDDEN)
