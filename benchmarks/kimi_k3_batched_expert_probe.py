"""Benchmark the isolated three-stage native Kimi K3 gate/up engine."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from benchmarks.kimi_k3_decode_inputs import route_assignments
from benchmarks.kimi_k3_timing import (
    percentile,
    replay_samples,
    summarize_rank_max,
)


PROBE_ROWS = (1, 2, 4, 8)
SATURATED_CTAS = 148
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5
NUM_EXPERTS = 896
LATENT = 3584
INTERMEDIATE = 384
TILE_CHANNELS = 128
LATENT_GROUPS = LATENT // 32
SITU_GROUPS = INTERMEDIATE // 32
PROFILE_PHASES = ("tma_arrival", "ring_full", "mma", "epilogue")
GATE_UP_TILE_WEIGHT_BYTES = 2 * (
    TILE_CHANNELS * (LATENT // 2)
    + TILE_CHANNELS * (LATENT // 32)
)


def _debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, object],
) -> None:
    # region agent log
    with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": time.time_ns() // 1_000_000,
                },
                sort_keys=True,
            )
            + "\n"
        )
    # endregion


def same_expert_batching_coverage(
    assignments: Sequence[Sequence[int]],
) -> dict[str, int | float | dict[str, int]]:
    counts = Counter(
        int(expert)
        for token_assignments in assignments
        for expert in token_assignments
    )
    histogram = Counter(counts.values())
    assignment_count = sum(counts.values())
    multirow_assignments = sum(
        count for count in counts.values() if count > 1
    )
    return {
        "assignments": assignment_count,
        "distinct_experts": len(counts),
        "expert_row_histogram": {
            str(rows): experts for rows, experts in sorted(histogram.items())
        },
        "assignments_in_multirow_experts": multirow_assignments,
        "multirow_assignment_fraction": (
            multirow_assignments / assignment_count
            if assignment_count
            else 0.0
        ),
        "maximum_rows_per_expert": max(counts.values(), default=0),
    }


def _checked_repeats(
    baseline: Sequence[float],
    candidate: Sequence[float],
    name: str,
) -> tuple[list[float], list[float]]:
    if not baseline or not candidate:
        raise ValueError(f"{name} evaluation requires repeat values")
    if len(baseline) != len(candidate):
        raise ValueError(f"baseline and candidate {name} counts must match")
    baseline_values = [float(value) for value in baseline]
    candidate_values = [float(value) for value in candidate]
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (*baseline_values, *candidate_values)
    ):
        raise ValueError(f"{name} values must be finite and positive")
    return baseline_values, candidate_values


def evaluate_row(
    *,
    rows: int,
    baseline_repeat_medians: Sequence[float],
    candidate_repeat_medians: Sequence[float],
    baseline_repeat_p99s: Sequence[float],
    candidate_repeat_p99s: Sequence[float],
    numerical: dict[str, float | int | bool],
) -> dict[str, float | bool]:
    if rows not in PROBE_ROWS:
        raise ValueError(f"unsupported probe row count: {rows}")
    baseline_medians, candidate_medians = _checked_repeats(
        baseline_repeat_medians,
        candidate_repeat_medians,
        "median",
    )
    baseline_p99s, candidate_p99s = _checked_repeats(
        baseline_repeat_p99s,
        candidate_repeat_p99s,
        "p99",
    )

    baseline_median = percentile(baseline_medians, 0.5)
    candidate_median = percentile(candidate_medians, 0.5)
    baseline_p99 = percentile(baseline_p99s, 0.5)
    candidate_p99 = percentile(candidate_p99s, 0.5)
    median_dispersion = max(
        max(baseline_medians) - min(baseline_medians),
        max(candidate_medians) - min(candidate_medians),
    )
    p99_dispersion = max(
        max(baseline_p99s) - min(baseline_p99s),
        max(candidate_p99s) - min(candidate_p99s),
    )
    median_improvement = baseline_median - candidate_median
    p99_improvement = baseline_p99 - candidate_p99
    median_fraction = median_improvement / baseline_median
    p99_fraction = p99_improvement / baseline_p99
    median_threshold = 0.10 if rows == 1 else 0.05
    p99_threshold = 0.05
    measurably_faster = (
        median_improvement > median_dispersion
        and p99_improvement > p99_dispersion
    )
    numerically_correct = all(
        bool(numerical[name])
        for name in (
            "gate_bitwise_equal",
            "up_bitwise_equal",
            "situ_bitwise_equal",
            "situ_scale_bitwise_equal",
            "inactive_columns_isolated",
        )
    )
    threshold_passed = (
        median_fraction >= median_threshold
        and p99_fraction >= p99_threshold
    )
    return {
        "baseline_median_of_repeats_ms": baseline_median,
        "candidate_median_of_repeats_ms": candidate_median,
        "baseline_p99_of_repeats_ms": baseline_p99,
        "candidate_p99_of_repeats_ms": candidate_p99,
        "median_dispersion_ms": median_dispersion,
        "p99_dispersion_ms": p99_dispersion,
        "effect_band_ms": median_dispersion,
        "improvement_ms": median_improvement,
        "p99_improvement_ms": p99_improvement,
        "improvement_fraction": median_fraction,
        "p99_improvement_fraction": p99_fraction,
        "median_threshold": median_threshold,
        "p99_threshold": p99_threshold,
        "measurably_faster": measurably_faster,
        "threshold_passed": threshold_passed,
        "numerically_correct": numerically_correct,
        "passed": (
            measurably_faster and threshold_passed and numerically_correct
        ),
    }


def integration_decision(*, layout_validated: bool) -> dict[str, object]:
    """Report eligibility without changing the production dispatch."""
    return {
        "layout_validated": layout_validated,
        "standalone_integration_candidate": False,
        "production_integrated": False,
        "preserve_single_launch": True,
        "next_design": "persistent_multi_unit_staged_pipeline",
        "reason": (
            "the isolated engine passed its measurement gate; production "
            "integration still requires an explicit follow-up"
            if layout_validated
            else "the isolated engine did not pass every measurement gate"
        ),
    }


def _extension() -> ModuleType:
    return importlib.import_module("mok._C")


def _l2_bytes(device: torch.device) -> int:
    properties = torch.cuda.get_device_properties(device)
    for name in ("L2_cache_size", "l2_cache_size"):
        value = getattr(properties, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("PyTorch did not expose the B300 L2 cache size")


def _weight_pool(
    device: torch.device,
    experts: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(7400)

    def packed() -> torch.Tensor:
        return torch.randint(
            0,
            256,
            (experts, INTERMEDIATE, LATENT // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def scales() -> torch.Tensor:
        return torch.randint(
            124,
            131,
            (experts, INTERMEDIATE, LATENT // 32),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    return packed(), scales(), packed(), scales()


def _buffers(
    device: torch.device,
    ctas: int,
) -> dict[str, torch.Tensor]:
    return {
        "situ": torch.zeros(
            ctas,
            PROBE_ROWS[-1],
            INTERMEDIATE,
            dtype=torch.uint8,
            device=device,
        ),
        "situ_scale": torch.zeros(
            ctas,
            PROBE_ROWS[-1],
            SITU_GROUPS,
            dtype=torch.uint8,
            device=device,
        ),
        "gate": torch.empty(
            ctas,
            PROBE_ROWS[-1],
            TILE_CHANNELS,
            dtype=torch.float32,
            device=device,
        ),
        "up": torch.empty(
            ctas,
            PROBE_ROWS[-1],
            TILE_CHANNELS,
            dtype=torch.float32,
            device=device,
        ),
        "profile": torch.zeros(
            ctas,
            len(PROFILE_PHASES),
            dtype=torch.int64,
            device=device,
        ),
    }


def _prepare_activations(
    extension: ModuleType,
    latent: torch.Tensor,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation = torch.empty_like(latent, dtype=torch.uint8)
    activation_scale = torch.empty(
        latent.size(0),
        PROBE_ROWS[-1],
        LATENT_GROUPS,
        dtype=torch.uint8,
        device=latent.device,
    )
    extension._kimi_k3_prepare_native_gate_up_probe(
        latent,
        activation,
        activation_scale,
        rows,
    )
    return activation, activation_scale


def _call(
    extension: ModuleType,
    weights: tuple[torch.Tensor, ...],
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    assignment_tokens: torch.Tensor,
    experts: torch.Tensor,
    output_tiles: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    rows: int,
    *,
    candidate: bool,
    poison_inactive: bool = False,
    capture: bool = False,
    profile_enabled: bool = False,
) -> None:
    extension._kimi_k3_native_gate_up_probe(
        *weights,
        activation,
        activation_scale,
        assignment_tokens,
        experts,
        output_tiles,
        buffers["situ"],
        buffers["situ_scale"],
        buffers["gate"],
        buffers["up"],
        buffers["profile"],
        rows,
        candidate,
        poison_inactive,
        capture,
        profile_enabled,
    )


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype == torch.float32:
        return bool(
            torch.equal(
                left.contiguous().view(torch.int32),
                right.contiguous().view(torch.int32),
            )
        )
    return bool(torch.equal(left, right))


def _mismatch_count(left: torch.Tensor, right: torch.Tensor) -> int:
    if left.dtype == torch.float32:
        left = left.contiguous().view(torch.int32)
        right = right.contiguous().view(torch.int32)
    return int(torch.count_nonzero(left != right))


def _validation_schedule(
    device: torch.device,
    experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7600)
    random_experts = torch.randperm(experts, generator=generator)[:8].tolist()
    selected = [0, experts - 1, *(int(value) for value in random_experts)]
    unique = list(dict.fromkeys(selected))
    pairs = [(expert, tile) for expert in unique for tile in range(3)]
    expert_ids = torch.tensor(
        [pair[0] for pair in pairs],
        dtype=torch.int32,
        device=device,
    )
    output_tiles = torch.tensor(
        [pair[1] for pair in pairs],
        dtype=torch.int32,
        device=device,
    )
    return expert_ids, output_tiles


def _numerical_metrics(
    extension: ModuleType,
    device: torch.device,
    weights: tuple[torch.Tensor, ...],
    rows: int,
) -> dict[str, float | int | bool]:
    expert_ids, output_tiles = _validation_schedule(
        device, int(weights[0].size(0))
    )
    cases = int(expert_ids.numel())
    generator = torch.Generator(device=device).manual_seed(7700 + rows)
    latent = (
        torch.randn(
            cases,
            PROBE_ROWS[-1],
            LATENT,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.25
    ).bfloat16()
    activation, activation_scale = _prepare_activations(
        extension, latent, rows
    )
    assignment_tokens = torch.arange(
        PROBE_ROWS[-1], dtype=torch.int32, device=device
    )
    baseline = _buffers(device, cases)
    candidate = _buffers(device, cases)
    poisoned = _buffers(device, cases)

    _call(
        extension,
        weights,
        activation,
        activation_scale,
        assignment_tokens,
        expert_ids,
        output_tiles,
        baseline,
        rows,
        candidate=False,
        capture=True,
    )
    _call(
        extension,
        weights,
        activation,
        activation_scale,
        assignment_tokens,
        expert_ids,
        output_tiles,
        candidate,
        rows,
        candidate=True,
        capture=True,
    )
    _call(
        extension,
        weights,
        activation,
        activation_scale,
        assignment_tokens,
        expert_ids,
        output_tiles,
        poisoned,
        rows,
        candidate=True,
        poison_inactive=True,
        capture=True,
    )
    torch.cuda.synchronize(device)

    baseline_gate = baseline["gate"][:, :rows]
    baseline_up = baseline["up"][:, :rows]
    candidate_gate = candidate["gate"][:, :rows]
    candidate_up = candidate["up"][:, :rows]
    poisoned_gate = poisoned["gate"][:, :rows]
    poisoned_up = poisoned["up"][:, :rows]
    result: dict[str, float | int | bool] = {
        "cases": cases,
        "covers_expert_zero": bool((expert_ids == 0).any()),
        "covers_expert_895": bool((expert_ids == 895).any())
        if int(weights[0].size(0)) > 895
        else True,
        "covers_all_output_tiles": bool(
            torch.equal(
                torch.unique(output_tiles).cpu(),
                torch.tensor([0, 1, 2], dtype=torch.int32),
            )
        ),
        "gate_bitwise_equal": _bitwise_equal(
            candidate_gate, baseline_gate
        ),
        "up_bitwise_equal": _bitwise_equal(candidate_up, baseline_up),
        "situ_bitwise_equal": _bitwise_equal(
            candidate["situ"], baseline["situ"]
        ),
        "situ_scale_bitwise_equal": _bitwise_equal(
            candidate["situ_scale"], baseline["situ_scale"]
        ),
        "inactive_columns_isolated": all(
            (
                _bitwise_equal(poisoned_gate, candidate_gate),
                _bitwise_equal(poisoned_up, candidate_up),
                _bitwise_equal(poisoned["situ"], candidate["situ"]),
                _bitwise_equal(
                    poisoned["situ_scale"], candidate["situ_scale"]
                ),
            )
        ),
        "gate_mismatches": _mismatch_count(
            candidate_gate, baseline_gate
        ),
        "up_mismatches": _mismatch_count(candidate_up, baseline_up),
        "situ_mismatches": _mismatch_count(
            candidate["situ"], baseline["situ"]
        ),
        "situ_scale_mismatches": _mismatch_count(
            candidate["situ_scale"], baseline["situ_scale"]
        ),
    }
    return result


def _saturated_schedules(
    device: torch.device,
    l2_bytes: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    bytes_per_launch = SATURATED_CTAS * GATE_UP_TILE_WEIGHT_BYTES
    pool_size = l2_bytes // bytes_per_launch + 2
    generator = torch.Generator().manual_seed(7800)
    experts: list[torch.Tensor] = []
    output_tiles: list[torch.Tensor] = []
    for pool in range(pool_size):
        schedule = torch.randperm(
            NUM_EXPERTS, generator=generator, dtype=torch.int64
        )[:SATURATED_CTAS].to(torch.int32)
        schedule[0] = 0
        schedule[1] = NUM_EXPERTS - 1
        tiles = (
            torch.arange(SATURATED_CTAS, dtype=torch.int32) + pool
        ) % 3
        experts.append(schedule.to(device))
        output_tiles.append(tiles.to(device))
    return experts, output_tiles, pool_size


def _capture_pool(
    extension: ModuleType,
    weights: tuple[torch.Tensor, ...],
    activations: Sequence[torch.Tensor],
    activation_scales: Sequence[torch.Tensor],
    assignment_tokens: torch.Tensor,
    expert_schedules: Sequence[torch.Tensor],
    tile_schedules: Sequence[torch.Tensor],
    buffers: dict[str, torch.Tensor],
    rows: int,
    candidate: bool,
) -> list[torch.cuda.CUDAGraph]:
    graphs: list[torch.cuda.CUDAGraph] = []
    for activation, scales, experts, tiles in zip(
        activations,
        activation_scales,
        expert_schedules,
        tile_schedules,
        strict=True,
    ):
        _call(
            extension,
            weights,
            activation,
            scales,
            assignment_tokens,
            experts,
            tiles,
            buffers,
            rows,
            candidate=candidate,
        )
        torch.cuda.synchronize(activation.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _call(
                extension,
                weights,
                activation,
                scales,
                assignment_tokens,
                experts,
                tiles,
                buffers,
                rows,
                candidate=candidate,
            )
        graphs.append(graph)
    return graphs


def _measure(
    graphs: Sequence[torch.cuda.CUDAGraph],
    *,
    warmup_count: int,
    sample_count: int,
) -> list[float]:
    return replay_samples(
        lambda iteration: graphs[iteration % len(graphs)].replay(),
        warmup_count=warmup_count,
        sample_count=sample_count,
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=torch.cuda.synchronize,
        settle_count=len(graphs),
    )


def _phase_profile(
    extension: ModuleType,
    weights: tuple[torch.Tensor, ...],
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    assignment_tokens: torch.Tensor,
    experts: torch.Tensor,
    output_tiles: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    rows: int,
) -> dict[str, dict[str, float | int]]:
    buffers["profile"].zero_()
    _call(
        extension,
        weights,
        activation,
        activation_scale,
        assignment_tokens,
        experts,
        output_tiles,
        buffers,
        rows,
        candidate=True,
        profile_enabled=True,
    )
    torch.cuda.synchronize(activation.device)
    profile = buffers["profile"].cpu()
    return {
        name: {
            "sum_cycles": int(profile[:, index].sum()),
            "median_cycles_per_cta": float(
                profile[:, index].double().median()
            ),
            "max_cycles_per_cta": int(profile[:, index].max()),
        }
        for index, name in enumerate(PROFILE_PHASES)
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest(
    *,
    dry_run: bool,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> dict[str, object]:
    return {
        "benchmark": "kimi_k3_native_gate_up_probe",
        "dry_run": dry_run,
        "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
        "mma_shape": {
            "m": 128,
            "n": 8,
            "k": 32,
            "m_axis": "output_channel",
            "n_axis": "assignment_token",
        },
        "rows": list(PROBE_ROWS),
        "saturated_ctas": SATURATED_CTAS,
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "repeats": repeats,
        "repeat_order": "alternating_baseline_candidate",
        "correctness": "bitwise_gate_up_situ_and_inactive_columns",
        "candidate_shared_memory_limit_bytes": 120 * 1024,
        "required_residency_ctas_per_sm": 1,
        "integration_status": "benchmark_only",
        "thresholds": {
            "r1_median_improvement": 0.10,
            "r1_p99_improvement": 0.05,
            "r2_r4_r8_median_improvement": 0.05,
            "r2_r4_r8_p99_improvement": 0.05,
            "effect_exceeds_repeat_dispersion": True,
        },
    }


def run_focused(
    *,
    rows: int,
    variant: str,
) -> dict[str, object]:
    if rows not in PROBE_ROWS:
        raise ValueError("focused probe rows must be one of 1, 2, 4, 8")
    if variant not in ("setup", "baseline", "candidate", "both"):
        raise ValueError(
            "focused probe variant must be setup, baseline, candidate, or both"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the native gate/up probe requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the native gate/up probe requires an SM103 B300")
    extension = _extension()

    # region agent log
    _debug_log(
        hypothesis_id="A,E",
        location="benchmarks/kimi_k3_batched_expert_probe.py:run_focused",
        message="focused probe entry",
        data={"rows": rows, "variant": variant},
    )
    # endregion

    result: dict[str, object] = {
        "rows": rows,
        "variant": variant,
        "resources": dict(
            zip(
                (
                    "actual_shared_bytes",
                    "reserved_shared_bytes",
                    "blocks_per_sm",
                    "registers_per_thread",
                    "local_bytes",
                ),
                extension._kimi_k3_native_gate_up_probe_resources(),
                strict=True,
            )
        ),
    }
    if variant == "setup":
        return result

    weights = _weight_pool(device, 1)
    generator = torch.Generator(device=device).manual_seed(7900 + rows)
    latent = (
        torch.randn(
            1,
            PROBE_ROWS[-1],
            LATENT,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.25
    ).bfloat16()
    activation, activation_scale = _prepare_activations(
        extension, latent, rows
    )
    assignments = torch.arange(
        PROBE_ROWS[-1], dtype=torch.int32, device=device
    )
    experts = torch.zeros(1, dtype=torch.int32, device=device)
    tiles = torch.zeros(1, dtype=torch.int32, device=device)
    buffers = _buffers(device, 1)

    if variant == "both":
        result["numerical"] = _numerical_metrics(
            extension, device, weights, rows
        )
        return result
    _call(
        extension,
        weights,
        activation,
        activation_scale,
        assignments,
        experts,
        tiles,
        buffers,
        rows,
        candidate=variant == "candidate",
        capture=True,
        profile_enabled=variant == "candidate",
    )
    torch.cuda.synchronize(device)
    result["launch_synchronized"] = True
    result["situ_checksum"] = int(buffers["situ"].sum())
    result["profile_cycles"] = buffers["profile"].cpu().tolist()
    return result


def run(
    output_dir: Path,
    *,
    warmup_count: int,
    sample_count: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup_count < 1 or sample_count < 1 or repeats < 2:
        raise ValueError("warmups and samples must be positive; repeats >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("the native gate/up probe requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the native gate/up probe requires an SM103 B300")
    extension = _extension()

    # region agent log
    _debug_log(
        hypothesis_id="A,E",
        location="benchmarks/kimi_k3_batched_expert_probe.py:run",
        message="benchmark configuration",
        data={
            "ctas": SATURATED_CTAS,
            "rows": list(PROBE_ROWS),
            "warmups": warmup_count,
            "samples": sample_count,
            "repeats": repeats,
        },
    )
    # endregion

    resource_values = extension._kimi_k3_native_gate_up_probe_resources()
    resources = dict(
        zip(
            (
                "actual_shared_bytes",
                "reserved_shared_bytes",
                "blocks_per_sm",
                "registers_per_thread",
                "local_bytes",
            ),
            (int(value) for value in resource_values),
            strict=True,
        )
    )
    resources["passed"] = (
        resources["actual_shared_bytes"] <= 120 * 1024
        and resources["reserved_shared_bytes"] <= 120 * 1024
        and resources["blocks_per_sm"] == 1
        and resources["local_bytes"] == 0
    )

    l2_bytes = _l2_bytes(device)
    expert_schedules, tile_schedules, pool_size = _saturated_schedules(
        device, l2_bytes
    )
    weights = _weight_pool(device, NUM_EXPERTS)
    generator = torch.Generator(device=device).manual_seed(8000)
    latents = [
        (
            torch.randn(
                SATURATED_CTAS,
                PROBE_ROWS[-1],
                LATENT,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.25
        ).bfloat16()
        for _ in range(pool_size)
    ]
    prepared = [
        _prepare_activations(extension, latent, PROBE_ROWS[-1])
        for latent in latents
    ]
    activations = [value[0] for value in prepared]
    activation_scales = [value[1] for value in prepared]
    assignments = torch.arange(
        PROBE_ROWS[-1], dtype=torch.int32, device=device
    )
    baseline_buffers = _buffers(device, SATURATED_CTAS)
    candidate_buffers = _buffers(device, SATURATED_CTAS)

    numerical_by_rows = {
        rows: _numerical_metrics(extension, device, weights, rows)
        for rows in PROBE_ROWS
    }
    # region agent log
    _debug_log(
        hypothesis_id="B,C,D",
        location="benchmarks/kimi_k3_batched_expert_probe.py:run",
        message="bitwise validation completed",
        data={
            str(rows): {
                key: value
                for key, value in metrics.items()
                if isinstance(value, bool)
            }
            for rows, metrics in numerical_by_rows.items()
        },
    )
    # endregion

    rows_output: list[dict[str, Any]] = []
    raw_output: dict[str, object] = {}
    profiles: dict[str, object] = {}
    for rows in PROBE_ROWS:
        baseline_graphs = _capture_pool(
            extension,
            weights,
            activations,
            activation_scales,
            assignments,
            expert_schedules,
            tile_schedules,
            baseline_buffers,
            rows,
            False,
        )
        candidate_graphs = _capture_pool(
            extension,
            weights,
            activations,
            activation_scales,
            assignments,
            expert_schedules,
            tile_schedules,
            candidate_buffers,
            rows,
            True,
        )
        samples_by_variant: dict[str, list[list[float]]] = {
            "baseline": [],
            "candidate": [],
        }
        for repeat in range(repeats):
            order = (
                ("baseline", "candidate")
                if repeat % 2 == 0
                else ("candidate", "baseline")
            )
            for variant in order:
                graphs = (
                    baseline_graphs
                    if variant == "baseline"
                    else candidate_graphs
                )
                samples_by_variant[variant].append(
                    _measure(
                        graphs,
                        warmup_count=warmup_count,
                        sample_count=sample_count,
                    )
                )

        baseline_summaries = [
            summarize_rank_max([samples])
            for samples in samples_by_variant["baseline"]
        ]
        candidate_summaries = [
            summarize_rank_max([samples])
            for samples in samples_by_variant["candidate"]
        ]
        numerical = numerical_by_rows[rows]
        verdict = evaluate_row(
            rows=rows,
            baseline_repeat_medians=[
                float(summary["median_ms"])
                for summary in baseline_summaries
            ],
            candidate_repeat_medians=[
                float(summary["median_ms"])
                for summary in candidate_summaries
            ],
            baseline_repeat_p99s=[
                float(summary["p99_ms"])
                for summary in baseline_summaries
            ],
            candidate_repeat_p99s=[
                float(summary["p99_ms"])
                for summary in candidate_summaries
            ],
            numerical=numerical,
        )
        rows_output.append(
            {
                "rows": rows,
                "numerical": numerical,
                "baseline_repeats": baseline_summaries,
                "candidate_repeats": candidate_summaries,
                **verdict,
            }
        )
        profiles[str(rows)] = _phase_profile(
            extension,
            weights,
            activations[0],
            activation_scales[0],
            assignments,
            expert_schedules[0],
            tile_schedules[0],
            candidate_buffers,
            rows,
        )
        raw_output[str(rows)] = samples_by_variant
        baseline_graphs.clear()
        candidate_graphs.clear()

    # region agent log
    _debug_log(
        hypothesis_id="A,B,D",
        location="benchmarks/kimi_k3_batched_expert_probe.py:run",
        message="candidate phase profiles collected",
        data=profiles,
    )
    # endregion

    passed = bool(resources["passed"]) and all(
        bool(row["passed"]) for row in rows_output
    )
    decision = integration_decision(layout_validated=passed)
    decision["isolated_improvement_fraction"] = {
        str(row["rows"]): float(row["improvement_fraction"])
        for row in rows_output
    }
    working_set_bytes = (
        pool_size * SATURATED_CTAS * GATE_UP_TILE_WEIGHT_BYTES
    )
    result = {
        "passed": passed,
        "rows": rows_output,
        "resources": resources,
        "phase_profiles": profiles,
        "route_coverage": {
            str(tokens): same_expert_batching_coverage(
                route_assignments(tokens, 0)
            )
            for tokens in (16, 32, 128)
        },
        "working_set": {
            "l2_bytes": l2_bytes,
            "saturated_ctas": SATURATED_CTAS,
            "pool_size": pool_size,
            "gate_up_tile_weight_bytes": GATE_UP_TILE_WEIGHT_BYTES,
            "weight_pool_bytes": working_set_bytes,
            "exceeds_l2": working_set_bytes > l2_bytes,
        },
        "decision": decision,
    }
    # region agent log
    _debug_log(
        hypothesis_id="C,E",
        location="benchmarks/kimi_k3_batched_expert_probe.py:run",
        message="integration gate evaluated",
        data={
            "passed": passed,
            "resources_passed": bool(resources["passed"]),
            "rows_passed": {
                str(row["rows"]): bool(row["passed"])
                for row in rows_output
            },
        },
    )
    # endregion

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "manifest.json",
        _manifest(
            dry_run=False,
            warmup_count=warmup_count,
            sample_count=sample_count,
            repeats=repeats,
        ),
    )
    _write_json(output_dir / "results.json", result)
    _write_json(output_dir / "raw_samples.json", raw_output)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kimi_k3_batched_expert_probe"),
    )
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--focus-rows", type=int)
    parser.add_argument(
        "--focus-variant",
        choices=("setup", "baseline", "candidate", "both"),
        default="both",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.focus_rows is not None:
        focused = run_focused(
            rows=args.focus_rows,
            variant=args.focus_variant,
        )
        print(json.dumps(focused, indent=2, sort_keys=True))
        return
    if args.dry_run:
        manifest = _manifest(
            dry_run=True,
            warmup_count=args.warmup_count,
            sample_count=args.sample_count,
            repeats=args.repeats,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    result = run(
        args.output_dir,
        warmup_count=args.warmup_count,
        sample_count=args.sample_count,
        repeats=args.repeats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        print(
            "native gate/up probe failed; the candidate remains isolated "
            "and must not be integrated"
        )


if __name__ == "__main__":
    main()
