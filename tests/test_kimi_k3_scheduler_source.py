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
    assert kernel.count("grid_barrier(") == 7


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


def test_grouped_pipeline_reuses_activation_across_expert_output_tiles() -> None:
    """Group output tiles, not unrelated expert rows, behind m128x8 math."""
    grouped = _source("expert_mxfp4_grouped.cuh")

    assert "kGroupedGateUpWidth = 3" in grouped
    assert "kGroupedDownWidth = 4" in grouped
    assert "kGroupedGateUpUnits == 1" in grouped
    assert "kGroupedDownUnits == 7" in grouped
    assert "kGroupedM = 128" in grouped
    assert "kGroupedN = 8" in grouped
    assert "kGroupedPhysicalN = 16" in grouped
    assert "kGroupedDownPersistentSharedBytes = 160 * 1024" in grouped
    assert "(5u << 7)" in grouped
    assert "(0u << 10)" in grouped
    assert "kMmaK = 32" not in grouped

    gate_up = _function_body(grouped, "void grouped_gate_up_unit(")
    down = _function_body(grouped, "void grouped_down_unit(")
    for body in (gate_up, down):
        assert "assignment_offset += kGroupedN" in body
        assert "stage_grouped_activation<" in body
        assert "(&weight_tile)[2]" in body
        assert "next_buffer = (round + 1) & 1" in body
        assert body.index("stage_grouped_activation<") < body.index(
            "for (int tile = 0; tile < tile_count; ++tile)"
        )

    assert "grouped_batch_mixed_mma(" in gate_up
    assert "grouped_batch_mixed_mma(" in down
    assert "quantize_grouped_situ(" in gate_up
    assert "accumulate_grouped_down(" in down


def test_grouped_down_is_a_guarded_separate_persistent_instantiation() -> None:
    """The benchmark switch must change down without changing gate/up."""
    persistent = _source("persistent_kernel.cuh")
    kernel = _function_body(
        persistent,
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "MOK_KIMI_K3_ENABLE_GROUPED_PIPELINE" in persistent
    assert "set_benchmark_grouped_pipeline_for_testing(" in persistent
    assert "benchmark_grouped_pipeline_enabled()" in persistent
    assert "template<bool TENSOR_PATH, bool GROUPED_DOWN>" in persistent
    assert kernel.count("if constexpr (GROUPED_DOWN)") == 1
    assert "grouped_gate_up_unit(" not in kernel
    assert kernel.count("routed_gate_up_unit(") == 1
    assert "grouped_down_unit(" in kernel
    assert "launch_persistent<true, false>(" in persistent
    assert "launch_persistent<false, false>(" in persistent
    assert "launch_persistent<true, true>(" in persistent
    assert "launch_persistent<false, true>(" in persistent
    launch = _function_body(persistent, "void launch_decode(")
    assert "const bool grouped = benchmark_grouped_pipeline_enabled();" in launch
    assert launch.count("launch_grouped_decode<") == 2
    assert launch.count("launch_production_decode<") == 2
