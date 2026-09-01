"""Which expert, and which of its rows, one routing actually reaches.

Matching a reference proves the arithmetic; it does not prove the addressing,
because a stage that read the wrong expert would still be compared against a
reference that read the wrong expert. So these tests give one expert a shard no
other expert can imitate and then check that the output changes exactly when
the routing says it should: every selected id addressed exactly, inactive rows
zeroed, empty experts skipped, and a reused scratch carrying nothing.

What the stage computes is ``test_kimi_k3_expert.py``; the host boundary is
``test_kimi_k3_expert_contract.py``.
"""

from __future__ import annotations

import torch

from .kimi_k3_expert_support import (
    ADDRESS_EXPERTS,
    ADDRESS_WEIGHTS,
    _assert_expert_close,
    Assignment,
    _call,
    device,
    _down_row_gain,
    ExpertWeights,
    GROUP,
    HIDDEN,
    INTERMEDIATE,
    _latent_column_for_row,
    _make_structured_weights,
    _mxfp8_quantize_reference,
    peer_device,
    _published_latent,
    _random_latent,
    _reference,
    _region,
    scratch,
    SCRATCH_BYTES,
    weights,
    _write_assignments,
)


def _install_expert_pattern(
    weights: ExpertWeights, expert: int, phase: int, device: torch.device
) -> None:
    """Give one expert a shard that no other expert can imitate.

    The gate/up rows are displaced by whole latent scale groups and the down
    rows carry a shifted gain phase, so reading the wrong expert changes both
    which latent columns are reduced and how each output tile is weighted.
    """
    from mok.ops import pack_kimi_k3_mxfp4

    rows = torch.arange(INTERMEDIATE, device=device)
    columns = (
        torch.tensor(
            [_latent_column_for_row(row) for row in range(INTERMEDIATE)],
            device=device,
        )
        + GROUP * phase
    ) % HIDDEN
    gate_dense = torch.zeros(
        1, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device
    )
    gate_dense[0, rows, columns] = 1.0
    up_dense = torch.zeros_like(gate_dense)
    up_dense[0, rows, columns] = 0.5
    down_dense = _down_row_gain(device, phase).view(1, HIDDEN, 1).expand(
        1, HIDDEN, INTERMEDIATE
    ).contiguous()

    for dense, packed, scale in (
        (gate_dense, weights.w1_packed, weights.w1_scale),
        (up_dense, weights.w3_packed, weights.w3_scale),
        (down_dense, weights.w2_packed, weights.w2_scale),
    ):
        expert_packed, expert_scale = pack_kimi_k3_mxfp4(dense, dense.size(-1))
        packed[expert].copy_(expert_packed[0])
        scale[expert].copy_(expert_scale[0])


def test_selected_expert_ids_are_addressed_exactly(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    """Route to a low, a middle, and the final expert with distinct shards.

    Each selected expert also differs from the shared shard every other expert
    carries, so reading expert 0, aliasing the expert stride onto a neighbour,
    reversing the two experts a token selected across its two slots, or
    misaddressing the final expert all change the result.  Only the selected
    slices are dequantized.
    """
    tensors = (
        weights.w1_packed,
        weights.w1_scale,
        weights.w3_packed,
        weights.w3_scale,
        weights.w2_packed,
        weights.w2_scale,
    )
    saved = {
        expert: tuple(tensor[expert].clone() for tensor in tensors)
        for expert in ADDRESS_EXPERTS
    }
    try:
        for phase, expert in enumerate(ADDRESS_EXPERTS, start=1):
            _install_expert_pattern(weights, expert, phase, device)

        latent = _random_latent(device, 6, 7100)
        first_weight, second_weight = ADDRESS_WEIGHTS
        slot_experts = [
            (ADDRESS_EXPERTS[token % 3], ADDRESS_EXPERTS[(token + 1) % 3])
            for token in range(6)
        ]
        assignments: list[Assignment] = []
        for token, (first, second) in enumerate(slot_experts):
            assignments.append((first, token, 0, first_weight))
            assignments.append((second, token, 1, second_weight))
        _write_assignments(scratch, assignments)

        actual = _call(latent, weights, torch.empty_like(latent), scratch, 6)
        expected = _reference(latent, weights, assignments, 6)
        _assert_expert_close(actual, expected)

        # Prove the shards discriminate: every addressing bug this test is meant
        # to catch must move the reference well past the max-abs tolerance.
        collapsed = [(0, token, slot, weight)
                     for _, token, slot, weight in assignments]
        rotated = [
            (ADDRESS_EXPERTS[(ADDRESS_EXPERTS.index(expert) + 1) % 3],
             token, slot, weight)
            for expert, token, slot, weight in assignments
        ]
        neighbour = [(expert - 1, token, slot, weight)
                     for expert, token, slot, weight in assignments]
        # A true swap: each token keeps its own two experts, its slot positions,
        # and its route weights, and only the pairing between them is reversed.
        # Nothing but reading the assignment's own expert id gets this right.
        reversed_slots: list[Assignment] = []
        for token, (first, second) in enumerate(slot_experts):
            reversed_slots.append((second, token, 0, first_weight))
            reversed_slots.append((first, token, 1, second_weight))
        for wrong in (collapsed, rotated, neighbour, reversed_slots):
            deviation = _reference(latent, weights, wrong, 6) - expected
            assert float(deviation.abs().max()) > 1.0
    finally:
        for expert, originals in saved.items():
            for tensor, original in zip(tensors, originals):
                tensor[expert].copy_(original)


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
    latent = _random_latent(device, 8, 6400)
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


def test_replayed_generations_publish_fresh_quantized_and_routed_state(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    """Consume both published stages on every replay of one reused workspace.

    Each replay uses a different latent and a different row count, so any stale
    MXFP8 block or stale routed row left over from an earlier generation is a
    value mismatch rather than a silently plausible result.
    """
    replays = ((1, 7000), (8, 7001), (3, 7002), (16, 7003), (2, 7004), (8, 7005))
    quantization: list[int] = []
    completion: list[int] = []

    for rows, seed in replays:
        latent = _random_latent(device, rows, seed)
        assignments = [(0, token, 0, 1.0) for token in range(rows)]
        _write_assignments(scratch, assignments)
        routed = torch.full_like(latent, float("nan"))

        actual = _call(latent, weights, routed, scratch, rows)

        # Stage one: the published MXFP8 latent and its E8M0 scale bytes.
        expected_latent, expected_scales = _mxfp8_quantize_reference(
            latent.float()
        )
        published_scales = _region(scratch, "latent_scale", torch.uint8)[
            : rows * (HIDDEN // GROUP)
        ].view(rows, HIDDEN // GROUP)
        assert torch.equal(published_scales, expected_scales)
        assert torch.equal(_published_latent(scratch, rows), expected_latent)

        # Stage two: the routed output for this generation.
        _assert_expert_close(actual, _reference(latent, weights, assignments, rows))

        phase = _region(scratch, "phase", torch.int32)
        assert int(phase[4]) == 0, "quantization arrivals must be reset"
        assert int(phase[7]) == 0, "completion arrivals must be reset"
        quantization.append(int(phase[5]))
        completion.append(int(phase[8]))

    assert quantization == [quantization[0] + step for step in range(len(replays))]
    assert completion == [completion[0] + step for step in range(len(replays))]


def test_expert_stage_uses_the_tensor_devices_current_stream(
    device: torch.device,
    weights: ExpertWeights,
    scratch: torch.Tensor,
) -> None:
    source = _random_latent(device, 8, 6500)
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
    latent = _random_latent(peer_device, 2, 6600)
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
