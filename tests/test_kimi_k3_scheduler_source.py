"""Source contracts for Kimi K3 scheduling experiments."""

from pathlib import Path


_SOURCE_ROOT = Path(__file__).parents[1] / "csrc" / "kimi_k3_decode"


def _source(name: str) -> str:
    return (_SOURCE_ROOT / name).read_text(encoding="utf-8")


def _function_body(text: str, signature: str) -> str:
    start = text.index(signature)
    depth = 0
    for offset in range(text.index("{", start), len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise AssertionError(f"{signature} is never closed")


def test_routed_queue_batching_starts_above_the_primary_m16_regime() -> None:
    """Keep fine-grained scheduling at M16 and batch only wider routed work."""
    sync = _source("persistent_sync.cuh")
    policy = _function_body(sync, "int routed_claim_batch(")
    claim = _function_body(sync, "int claim_unit_batch(")
    kernel = _function_body(
        _source("persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "kRoutedClaimBatchThreshold = 16" in sync
    assert "kRoutedClaimBatch = 4" in sync
    assert "active_tokens <= kRoutedClaimBatchThreshold ? 1" in policy
    assert "atomicAdd(" in claim
    assert "static_cast<unsigned int>(batch)" in claim
    assert kernel.count("routed_claim_batch(active_tokens)") == 1
    assert kernel.count("claim_unit_batch(") == 2
    assert kernel.count("claim_unit(") == 1


def test_route_selection_fuses_with_scoring_before_assignment_quantization() -> None:
    """Select on the last score shard and overlap assignments with quantization."""
    router = _source("router.cuh")
    kernel = _function_body(
        _source("persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "void select_after_score_shard(" in router
    select = _function_body(router, "void select_after_score_shard(")
    assert "atomicAdd(&scratch.expert_counts[token], 1)" in select
    assert "ticket == kScoreShards - 1" in select
    assert "select_token(" in select
    assert "select_after_score_shard(" in kernel
    assert kernel.count("grid_barrier(") == 6


def test_batched_expert_probe_is_a_transposed_m128x8x32_microprototype() -> None:
    """Exercise token columns end to end without touching the production path."""
    probe = _source("expert_mxfp4_batch_probe.cuh")
    production = _source("expert_mxfp4.cuh")
    persistent = _source("persistent_kernel.cuh")

    assert "kBatchProbeM = 128" in probe
    assert "kBatchProbeN = 8" in probe
    assert "kBatchProbePhysicalN = 16" in probe
    assert "kBatchProbeSharedBytes = kProbeSharedBytes" in probe
    assert "baseline_shared_bytes = kBatchProbeSharedBytes" in probe
    assert "(5u << 7)" in probe
    assert "(0u << 10)" in probe
    assert "kBatchProbeN / 8" in probe
    assert "weight_tile[slot], activation_tile[slot]" in probe
    assert "gate[{column, row}]" in probe
    assert "up[{column, row}]" in probe
    assert "result[{column, row}]" in probe
    assert "routed_gate_up_unit(" in probe
    assert "routed_down_unit(" in probe
    assert "batched_gate_up_unit(" in probe
    assert "batched_down_unit(" in probe
    assert "batched_gate_up_unit(" not in production
    assert "batched_down_unit(" not in persistent


def test_grouped_down_reuses_activation_across_expert_output_tiles() -> None:
    """Group down output tiles, not unrelated rows, behind m128x8 math."""
    grouped = _source("expert_mxfp4_grouped.cuh")

    assert "kGroupedDownWidth = 4" in grouped
    assert "kGroupedDownUnits == 7" in grouped
    assert "kGroupedM = 128" in grouped
    assert "kGroupedN = 8" in grouped
    assert "kGroupedPhysicalN = 16" in grouped
    assert "kGroupedDownPersistentSharedBytes = 160 * 1024" in grouped
    assert "(5u << 7)" in grouped
    assert "(0u << 10)" in grouped
    assert "kMmaK = 32" not in grouped

    down = _function_body(grouped, "void grouped_down_unit(")
    assert "assignment_offset += kGroupedN" in down
    assert "stage_grouped_down_activation(" in down
    assert "(&weight_tile)[2]" in down
    assert "next_buffer = (round + 1) & 1" in down
    assert down.index("stage_grouped_down_activation(") < down.index(
        "for (int tile = 0; tile < tile_count; ++tile)"
    )
    assert "grouped_batch_mixed_mma(" in down
    assert "accumulate_grouped_down_fixed(" in down
    fixed = _function_body(grouped, "void accumulate_grouped_down_fixed(")
    assert "kRoutedAccumulatorScale" in fixed
    assert "__float2ll_rn(" in fixed
    assert "scratch.routed_accumulator_fixed" in fixed
    assert "atomicAdd(" in fixed
    assert "down_progress" not in fixed
    assert "wait_timed_out" not in fixed


def test_readiness_pipeline_is_the_only_production_instantiation() -> None:
    """Ship baseline gate/up, grouped down, and no rejected candidate switch."""
    persistent = _source("persistent_kernel.cuh")
    grouped = _source("expert_mxfp4_grouped.cuh")
    kernel = _function_body(
        persistent,
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "MOK_KIMI_K3_ENABLE_GATE_UP_GROUPING" not in persistent
    assert "MOK_KIMI_K3_ENABLE_GATE_UP_DOWN_PIPELINE" not in persistent
    assert "set_benchmark_gate_up_group_size_for_testing(" not in persistent
    assert "set_benchmark_gate_up_down_pipeline_for_testing(" not in persistent
    assert "benchmark_gate_up_group_size()" not in persistent
    assert "benchmark_gate_up_down_pipeline()" not in persistent
    assert "template<bool TENSOR_PATH>" in persistent
    assert "GATE_UP_GROUP_SIZE" not in persistent
    assert "PIPELINE_GATE_UP_DOWN" not in persistent
    # The gate/up phase is the fused-W13 engine and nothing else can be reached
    # from here: the template argument that once chose between candidates is
    # gone, and so is the unit it chose between them and.
    assert kernel.count(
        "expert_mxfp4::fused_w13::routed_gate_up_fused_unit("
    ) == 1
    assert "expert_mxfp4::routed_gate_up_unit(" not in kernel
    assert "ENGINE" not in persistent
    assert kernel.count("grouped_down_unit(") == 1
    assert "grouped_gate_up_unit(" not in kernel
    assert "routed_down_unit(" not in kernel
    assert "scratch.routed_accumulator_fixed[index] = 0;" in kernel
    assert "down_progress" not in kernel
    assert "grouped_gate_up_unit(" not in grouped
    assert "quantize_grouped_situ(" not in grouped
    launch = _function_body(persistent, "void launch_decode(")
    assert launch.count("launch_persistent<") == 2


def test_production_pipeline_uses_dependency_readiness_without_deadlock() -> None:
    """All producer tickets precede consumers; waits are expert-local."""
    persistent = _source("persistent_kernel.cuh")
    sync = _source("persistent_sync.cuh")
    router = _source("router.cuh")
    kernel = _function_body(
        persistent,
        "void kimi_k3_decode_persistent_kernel(",
    )
    compact = _function_body(router, "void build_expert_units(")

    assert "scratch.expert_counts[expert] = 0;" in compact
    assert "kGateUpArrivals" in persistent
    assert "publish_count_at(" in sync
    assert "wait_for_count_at(" in sync
    assert "&scratch.expert_counts[expert]" in kernel
    # One claim per expert, six arrivals per expert. The claim count is what the
    # queue is as long as; the arrival count is what the down phase waits for,
    # and the two are separate constants because they are separate facts.
    assert "kGateUpUnitsPerExpert = 1;" in persistent
    assert "kGateUpArrivalsPerExpert" in persistent
    assert "kErrorPersistentGateUpDownReadiness" in persistent

    gate_up = kernel.split("// Phase 3:", 1)[1].split("// Phase 4:", 1)[0]
    down = kernel.split("// Phase 4:", 1)[1].split("// Phase 5:", 1)[0]
    # The arrivals are published from inside the unit, because the ranges the
    # down phase waits on complete inside it rather than at its boundary.
    assert gate_up.index("claim_unit_batch(") < gate_up.index(
        "&scratch.expert_counts[expert]"
    )
    engine = _function_body(
        _source("expert_mxfp4_fused_w13.cuh"),
        "void routed_gate_up_fused_unit(",
    )
    assert engine.count("persistent::publish_count_at(arrival_counter);") == 1
    assert down.index("wait_for_count_at(") < down.index(
        "grouped_down_unit("
    )
