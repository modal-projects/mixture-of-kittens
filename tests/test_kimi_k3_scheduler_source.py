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


def test_native_gate_up_probe_is_an_isolated_three_stage_direct_tma_engine() -> None:
    """Pin the requested benchmark-only producer/consumer pipeline."""
    native = _source("expert_mxfp4_native_gate_up_probe.cuh")
    production = _source("expert_mxfp4.cuh")
    persistent = _source("persistent_kernel.cuh")

    assert "kNativeGateUpM = 128" in native
    assert "kNativeGateUpN = 8" in native
    assert "kNativeGateUpK = 32" in native
    assert "kNativeWeightStages = 3" in native
    assert "kNativePanelGroups = 4" in native
    assert "CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B" in native
    assert "CU_TENSOR_MAP_SWIZZLE_128B" in native
    assert "global_dimensions[5]" in native
    assert "global_strides[4]" in native
    assert "box_dimensions[5]" in native
    assert "load_direct_weight_stage(" in native
    assert "native_gate_up_candidate(" in native
    assert "native_gate_up_candidate(" not in production
    assert "native_gate_up_candidate(" not in persistent


def test_native_gate_up_panel_releases_stages_without_cta_barriers() -> None:
    native = _source("expert_mxfp4_native_gate_up_probe.cuh")
    candidate = _function_body(native, "void native_gate_up_candidate(")
    panel_loop = candidate.split(
        "for (int panel = 0; panel < kNativePanels; ++panel)", 1
    )[1].split("native_situ_epilogue(", 1)[0]

    assert "warpid() == kNativeProducerWarp" in candidate
    assert "warpid() == kNativeConsumerWarp" in candidate
    assert "issue_native_panel(" in candidate
    assert panel_loop.count("batch_mixed_mma_direct(") == 2
    assert "detail::tcgen05::commit<1>(stage_released[stage])" in panel_loop
    assert "detail::tcgen05::commit<1>(compute_done)" in candidate
    assert "__syncthreads()" not in panel_loop
    assert "store_batch_accumulator(" not in candidate
    assert "batch_result_tile" not in candidate


def test_native_gate_up_shared_memory_forces_one_cta_per_sm_under_120_kib() -> None:
    native = _source("expert_mxfp4_native_gate_up_probe.cuh")

    assert "kNativeWeightSharedBytes == 96 * 1024" in native
    assert "kNativeGateUpSharedBytes <= 120 * 1024" in native
    assert "kNativeGateUpSharedReservationBytes = 120 * 1024" in native
    assert (
        "2 * kNativeGateUpSharedReservationBytes"
        " > kittens::MAX_SHARED_MEMORY"
    ) in native
    assert "native_gate_up_probe_resources" in native
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in native


def test_native_gate_up_epilogue_reads_live_values_directly_from_tmem() -> None:
    native = _source("expert_mxfp4_native_gate_up_probe.cuh")
    epilogue = _function_body(native, "void native_situ_epilogue(")

    assert "group<1>::load_async(gate, gate_slice)" in epilogue
    assert "group<1>::load_async(up, up_slice)" in epilogue
    assert "tensor_load_wait()" in epilogue
    assert "quantize_e4m3(" in epilogue
    assert "scratch.situ_mxfp8" in epilogue
    assert "scratch.situ_scale" in epilogue
    assert "batch_result_tile" not in epilogue


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
    assert kernel.count("routed_gate_up_unit(") == 1
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
    assert "publish_count_at(&scratch.expert_counts[expert])" in kernel
    assert "&scratch.expert_counts[expert]" in kernel
    assert "kGateUpTiles" in kernel
    assert "kErrorPersistentGateUpDownReadiness" in persistent

    gate_up = kernel.split("// Phase 3:", 1)[1].split("// Phase 4:", 1)[0]
    down = kernel.split("// Phase 4:", 1)[1].split("// Phase 5:", 1)[0]
    assert gate_up.index("claim_unit_batch(") < gate_up.index(
        "publish_count_at(&scratch.expert_counts[expert])"
    )
    assert down.index("wait_for_count_at(") < down.index(
        "grouped_down_unit("
    )
