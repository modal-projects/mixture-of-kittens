"""GPU tests for assignment-driven Kimi K3 mixed MXFP8-by-MXFP4 experts."""

from __future__ import annotations

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
        ("phase", 16 * 4),
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
        ("routed_accumulator", MAX_TOKENS * HIDDEN * 4),
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


def _make_structured_weights(device: torch.device) -> ExpertWeights:
    """Make full native-layout weights with a cheap, nonzero exact FP4 path.

    Every W1/W3 row sums latent columns 0 and 1, and every W2 row selects SiTU
    column 0.  The two-input path exercises both nibbles of the packed FP4 byte
    and avoids making the end-to-end error metric hinge on one scalar through
    two required MXFP8 quantizations.  The same path is installed for all 896
    experts so the exhaustive assignment distribution has an analytical
    reference while still exercising every expert range and every output tile.
    """
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 4 * 1024**3:
        pytest.skip("full Kimi K3 prepared expert tensors need 4 GiB free")
    w1_packed = torch.zeros(
        EXPERTS, INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8, device=device
    )
    w1_scale = torch.full(
        (EXPERTS, INTERMEDIATE, HIDDEN // GROUP),
        UNIT_SCALE,
        dtype=torch.uint8,
        device=device,
    )
    w3_packed = torch.zeros_like(w1_packed)
    w3_scale = torch.full_like(w1_scale, UNIT_SCALE)
    w2_packed = torch.zeros(
        EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8, device=device
    )
    w2_scale = torch.full(
        (EXPERTS, HIDDEN, INTERMEDIATE // GROUP),
        UNIT_SCALE,
        dtype=torch.uint8,
        device=device,
    )
    # E2M1 code 0x2 is +1.0. Populate both nibbles to select columns 0 and 1.
    w1_packed[:, :, 0] = 0x22
    w3_packed[:, :, 0] = 0x22
    w2_packed[:, :, 0] = 0x02
    return ExpertWeights(
        w1_packed, w1_scale, w3_packed, w3_scale, w2_packed, w2_scale
    )


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
        gate = selected_x @ w1.T
        up = selected_x @ w3.T
        situ = (
            4.0
            * torch.tanh(gate / 4.0)
            * torch.sigmoid(gate)
            * 25.0
            * torch.tanh(up / 25.0)
        )
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


def _mxfp8_dequant_reference(values: torch.Tensor) -> torch.Tensor:
    """Reference the E4M3/E8M0 round-up quantizer used by mixed MMA."""
    values = values.float()
    absolute_max = values.abs().amax(dim=-1, keepdim=True)
    safe = absolute_max.clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(safe / 448.0)).clamp(-126, 127)
    scale = torch.pow(2.0, exponent)
    scale = torch.where(absolute_max == 0, torch.ones_like(scale), scale)
    quantized = (values / scale).to(torch.float8_e4m3fn).float()
    return quantized * scale


def test_workspace_bytes_matches_extended_expert_scratch(
    device: torch.device,
) -> None:
    from mok import _C

    assert SCRATCH_BYTES == 3_159_552
    assert _C.kimi_k3_decode_workspace_bytes() == SCRATCH_BYTES
    for name, (offset, _) in SCRATCH_LAYOUT.items():
        if name != "total_bytes":
            assert offset % ALIGNMENT == 0, name
    assert SCRATCH_LAYOUT["latent_mxfp8"] == (40_448, 458_752)
    assert SCRATCH_LAYOUT["latent_scale"] == (499_200, 14_336)
    assert SCRATCH_LAYOUT["situ_mxfp8"] == (513_536, 786_432)
    assert SCRATCH_LAYOUT["situ_scale"] == (1_299_968, 24_576)
    assert SCRATCH_LAYOUT["routed_accumulator"] == (1_324_544, 1_835_008)


def test_sm103_mixed_mxfp8_by_mxfp4_instruction_probe(
    device: torch.device,
) -> None:
    """Validate E4M3 x E2M1, E8M0, scale_vec::1X and logical K=32 alone."""
    from mok import _C
    from mok.ops import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    # Cross the eight-row boundary in the scale-factor swizzle so the probe
    # validates more than the first row atom.
    rows = 16
    a = torch.zeros(rows, GROUP, dtype=torch.bfloat16, device=device)
    code_points = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.bfloat16,
        device=device,
    )
    for row in range(rows):
        a[row] = code_points.repeat(4).roll(row) * (2.0 ** (row - 3))
    generator = torch.Generator(device=device).manual_seed(6001)
    b = (
        torch.randn(
            1, 128, GROUP, generator=generator, dtype=torch.float32, device=device
        )
        * 0.5
    ).bfloat16()
    b_packed, b_scale = pack_kimi_k3_mxfp4(b, GROUP)
    exact_b = dequant_kimi_k3_mxfp4(b_packed, b_scale, GROUP)[0].float()

    actual = _C._kimi_k3_mixed_mma_probe(a, b_packed[0], b_scale[0])
    expected = _mxfp8_dequant_reference(a) @ exact_b.T

    assert actual.shape == (rows, 128)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rows", [1, 2, 8, 16])
def test_single_expert_matches_exact_prepared_weight_reference(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
    rows: int,
) -> None:
    generator = torch.Generator(device=device).manual_seed(6100 + rows)
    latent = (
        torch.randn(
            rows, HIDDEN, generator=generator, dtype=torch.float32, device=device
        )
        * 0.25
    ).bfloat16()
    assignments = [(0, token, 0, 1.0) for token in range(rows)]
    _write_assignments(scratch, assignments)
    routed = torch.full_like(latent, float("nan"))

    actual = _call(latent, weights, routed, scratch, rows)
    expected = _reference(latent, weights, assignments, rows)

    assert actual.data_ptr() == routed.data_ptr()
    assert actual.shape == (rows, HIDDEN)
    assert actual.dtype == torch.bfloat16
    _assert_expert_close(actual, expected)


def test_grouped_experts_apply_exact_situ_and_normalized_router_weights(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:, 0] = torch.tensor([0.25, -0.5, 1.0, -2.0], device=device)
    # Two experts per token, deliberately interleaved before expert-major sorting.
    assignments = [
        (token % 3, token, 0, 0.25)
        for token in range(4)
    ] + [
        ((token + 1) % 3, token, 1, 0.75)
        for token in range(4)
    ]
    _write_assignments(scratch, assignments)
    routed = torch.empty_like(latent)

    actual = _call(latent, weights, routed, scratch, 4)
    expected = _reference(latent, weights, assignments, 4)

    _assert_expert_close(actual, expected)


def test_2048_assignments_cover_all_896_experts_and_stay_finite(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:, 0] = 0.5
    assignments = [
        ((token * TOPK + slot) % EXPERTS, token, slot, 1.0 / TOPK)
        for token in range(MAX_TOKENS)
        for slot in range(TOPK)
    ]
    assert len({item[0] for item in assignments}) == EXPERTS
    _write_assignments(scratch, assignments)
    counts = _region(scratch, "expert_counts", torch.int32)
    assert int(counts.sum()) == MAX_ASSIGNMENTS
    assert int((counts > 0).sum()) == EXPERTS
    routed = torch.empty_like(latent)

    actual = _call(latent, weights, routed, scratch, MAX_TOKENS)
    gate = torch.tensor(0.5, dtype=torch.float32, device=device)
    expected_value = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(gate / 25.0)
    )
    expected = expected_value.expand_as(actual).bfloat16().float()

    _assert_expert_close(actual, expected)


def test_empty_experts_are_skipped_and_cannot_change_output(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:, 0] = 0.75
    assignments = [(0, token, 0, 1.0) for token in range(8)]
    _write_assignments(scratch, assignments)
    first = _call(latent, weights, torch.empty_like(latent), scratch, 8).clone()

    # Expert 895 has count zero.  Poison every prepared byte without touching
    # the selected expert, then require bitwise-identical output.
    weights.w1_packed[895].fill_(0xFF)
    weights.w1_scale[895].fill_(0xFE)
    weights.w3_packed[895].fill_(0x77)
    weights.w3_scale[895].fill_(0x01)
    weights.w2_packed[895].fill_(0xAA)
    weights.w2_scale[895].fill_(0xFE)
    second = _call(latent, weights, torch.empty_like(latent), scratch, 8)

    assert torch.equal(first, second)


def test_active_token_mask_zeros_inactive_output_rows(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:3, 0] = 1.0
    assignments = [(0, token, 0, 1.0) for token in range(3)]
    _write_assignments(scratch, assignments)
    routed = torch.full_like(latent, float("nan"))

    active = _call(latent, weights, routed, scratch, 3)

    assert active.shape == (3, HIDDEN)
    assert torch.isfinite(active.float()).all()
    assert torch.equal(routed[3:], torch.zeros_like(routed[3:]))


def test_reused_scratch_resets_accumulator_and_generation_counters(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:, 0] = 0.5
    generations: list[torch.Tensor] = []
    for rows in (8, 2, 8):
        assignments = [(0, token, 0, 1.0) for token in range(rows)]
        _write_assignments(scratch, assignments)
        routed = torch.full_like(latent, 123.0)
        actual = _call(latent, weights, routed, scratch, rows)
        expected = _reference(latent, weights, assignments, rows)
        _assert_expert_close(actual, expected)
        assert torch.equal(routed[rows:], torch.zeros_like(routed[rows:]))
        generations.append(_region(scratch, "phase", torch.int32).clone())

    # Quantization and expert completion generations each advance once per call.
    assert int(generations[1][5] - generations[0][5]) == 1
    assert int(generations[2][5] - generations[1][5]) == 1
    assert int(generations[1][8] - generations[0][8]) == 1
    assert int(generations[2][8] - generations[1][8]) == 1


def test_expert_stage_uses_the_tensor_devices_current_stream(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    source = torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=device)
    source[:, 0] = 1.0
    assignments = [(0, token, 0, 1.0) for token in range(8)]
    _write_assignments(scratch, assignments)
    side_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side_stream):
        latent = torch.zeros_like(source)
        routed = torch.empty_like(source)
        torch.cuda._sleep(1 << 28)
        latent.copy_(source)
        actual = _call(latent, weights, routed, scratch, 8)
    side_stream.synchronize()

    expected = _reference(source, weights, assignments, 8)
    _assert_expert_close(actual, expected)


def test_expert_stage_on_peer_device_ignores_current_device(
    device: torch.device,
    peer_device: torch.device,
) -> None:
    peer_weights = _make_structured_weights(peer_device)
    peer_scratch = torch.zeros(
        SCRATCH_BYTES, dtype=torch.uint8, device=peer_device
    )
    latent = torch.zeros(2, HIDDEN, dtype=torch.bfloat16, device=peer_device)
    latent[:, 0] = 1.0
    assignments = [(0, token, 0, 1.0) for token in range(2)]
    _write_assignments(peer_scratch, assignments)
    torch.cuda.set_device(device)

    actual = _call(
        latent, peer_weights, torch.empty_like(latent), peer_scratch, 2
    )
    torch.cuda.synchronize(peer_device)

    assert torch.cuda.current_device() == device.index
    assert actual.device == peer_device
    expected = _reference(latent, peer_weights, assignments, 2)
    _assert_expert_close(actual, expected)
    del peer_weights, peer_scratch
    torch.cuda.empty_cache()


def test_expert_fake_matches_schema_and_returns_active_alias_metadata() -> None:
    from mok import _fake_impls, ops

    schema_names = tuple(
        argument.name
        for argument in torch.ops.mok._kimi_k3_routed_experts.default._schema.arguments
    )
    assert schema_names == _EXPERT_ARGUMENTS
    assert tuple(
        inspect.signature(_fake_impls._kimi_k3_routed_experts_fake).parameters
    ) == schema_names

    with FakeTensorMode():
        latent = torch.empty(17, HIDDEN, dtype=torch.bfloat16, device="cuda")
        packed_w1 = torch.empty(
            EXPERTS, INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8, device="cuda"
        )
        scale_w1 = torch.empty(
            EXPERTS, INTERMEDIATE, HIDDEN // GROUP, dtype=torch.uint8, device="cuda"
        )
        packed_w2 = torch.empty(
            EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8, device="cuda"
        )
        scale_w2 = torch.empty(
            EXPERTS, HIDDEN, INTERMEDIATE // GROUP, dtype=torch.uint8, device="cuda"
        )
        routed = torch.empty_like(latent)
        actual = ops._kimi_k3_routed_experts(
            latent,
            packed_w1,
            scale_w1,
            packed_w1,
            scale_w1,
            packed_w2,
            scale_w2,
            routed,
            torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda"),
            11,
        )

    assert actual.shape == (11, HIDDEN)
    assert actual.dtype == torch.bfloat16


def test_expert_stage_rejects_undersized_scratch_and_wrong_weight_layout(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    routed = torch.empty_like(latent)
    _write_assignments(scratch, [(0, 0, 0, 1.0)])

    with pytest.raises(RuntimeError, match="scratch"):
        _call(
            latent,
            weights,
            routed,
            scratch[:SCRATCH_BYTES - ALIGNMENT],
            1,
        )

    invalid = ExpertWeights(
        weights.w1_packed[:, :, :-16],
        weights.w1_scale,
        weights.w3_packed,
        weights.w3_scale,
        weights.w2_packed,
        weights.w2_scale,
    )
    with pytest.raises(RuntimeError, match="expert_w1_packed"):
        _call(latent, invalid, routed, scratch, 1)


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


def test_private_expert_stage_is_exactly_one_kernel_launch(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    latent = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    latent[:, 0] = 1.0
    _write_assignments(scratch, [(0, 0, 0, 1.0)])
    routed = torch.empty_like(latent)

    def call() -> object:
        return _call(latent, weights, routed, scratch, 1)

    call()
    names = _profiled_kernel_names(call)

    assert len(names) == 1, names
    assert "kimi_k3_routed_experts_kernel" in names[0]
