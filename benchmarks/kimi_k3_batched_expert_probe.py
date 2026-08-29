"""Isolated m128x8x32 Kimi K3 routed-expert contraction benchmark."""

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
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
REPEATS = 5
LATENT = 3584
INTERMEDIATE = 384
TOPK = 16
SCRATCH_BYTES = 6_276_096
EXPERT_WEIGHT_BYTES = 2_193_408
DEBUG_LOG_PATH = Path("/opt/cursor/logs/debug.log")


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, object],
) -> None:
    payload = {
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": time.time_ns() // 1_000_000,
    }
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


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


def evaluate_row(
    *,
    baseline_repeat_medians: Sequence[float],
    candidate_repeat_medians: Sequence[float],
    numerical: dict[str, float | bool],
) -> dict[str, float | bool]:
    if not baseline_repeat_medians or not candidate_repeat_medians:
        raise ValueError("row evaluation requires repeat medians")
    if len(baseline_repeat_medians) != len(candidate_repeat_medians):
        raise ValueError("baseline and candidate repeat counts must match")
    medians = [
        *(float(value) for value in baseline_repeat_medians),
        *(float(value) for value in candidate_repeat_medians),
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in medians):
        raise ValueError("repeat medians must be finite and positive")

    baseline_center = percentile(baseline_repeat_medians, 0.5)
    candidate_center = percentile(candidate_repeat_medians, 0.5)
    baseline_dispersion = (
        max(baseline_repeat_medians) - min(baseline_repeat_medians)
    )
    candidate_dispersion = (
        max(candidate_repeat_medians) - min(candidate_repeat_medians)
    )
    effect_band = max(baseline_dispersion, candidate_dispersion)
    improvement = baseline_center - candidate_center
    measurably_faster = improvement > effect_band
    numerically_correct = (
        bool(numerical["finite"])
        and float(numerical["relative_l1"]) <= 0.05
        and float(numerical["cosine_similarity"]) >= 0.999
        and float(numerical["max_abs"]) <= 1.0
    )
    return {
        "baseline_median_of_repeats_ms": baseline_center,
        "candidate_median_of_repeats_ms": candidate_center,
        "baseline_median_dispersion_ms": baseline_dispersion,
        "candidate_median_dispersion_ms": candidate_dispersion,
        "effect_band_ms": effect_band,
        "improvement_ms": improvement,
        "improvement_fraction": improvement / baseline_center,
        "measurably_faster": measurably_faster,
        "numerically_correct": numerically_correct,
        "passed": measurably_faster and numerically_correct,
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

    def packed(rows: int, columns: int) -> torch.Tensor:
        return torch.randint(
            0,
            256,
            (experts, rows, columns),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def scales(rows: int, columns: int) -> torch.Tensor:
        return torch.randint(
            124,
            131,
            (experts, rows, columns),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    return (
        packed(INTERMEDIATE, LATENT // 2),
        scales(INTERMEDIATE, LATENT // 32),
        packed(INTERMEDIATE, LATENT // 2),
        scales(INTERMEDIATE, LATENT // 32),
        packed(LATENT, INTERMEDIATE // 2),
        scales(LATENT, INTERMEDIATE // 32),
    )


def _call(
    extension: ModuleType,
    latent: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    scratch: torch.Tensor,
    expert: int,
    candidate: bool,
) -> None:
    extension._kimi_k3_batched_expert_probe(
        latent,
        *weights,
        output,
        scratch,
        expert,
        candidate,
    )


def _capture_pool(
    extension: ModuleType,
    latents: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    scratch: torch.Tensor,
    rows: int,
    candidate: bool,
) -> list[torch.cuda.CUDAGraph]:
    graphs: list[torch.cuda.CUDAGraph] = []
    for expert in range(latents.size(0)):
        latent = latents[expert, :rows]
        _call(
            extension, latent, weights, output[:rows], scratch, expert,
            candidate,
        )
        torch.cuda.synchronize(latents.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _call(
                extension, latent, weights, output[:rows], scratch, expert,
                candidate,
            )
        graphs.append(graph)
    return graphs


def _numerical_metrics(
    extension: ModuleType,
    latent: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    baseline_output: torch.Tensor,
    candidate_output: torch.Tensor,
    scratch: torch.Tensor,
    rows: int,
) -> dict[str, float | bool]:
    _call(
        extension, latent[:rows], weights, baseline_output[:rows], scratch,
        0, False,
    )
    torch.cuda.synchronize(latent.device)
    baseline = baseline_output[:rows].clone()
    _call(
        extension, latent[:rows], weights, candidate_output[:rows], scratch,
        0, True,
    )
    torch.cuda.synchronize(latent.device)
    candidate = candidate_output[:rows].clone()
    difference = candidate.float() - baseline.float()
    denominator = baseline.float().abs().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        candidate.float().flatten(),
        baseline.float().flatten(),
        dim=0,
    )
    return {
        "finite": bool(torch.isfinite(candidate.float()).all()),
        "relative_l1": float(difference.abs().sum() / denominator),
        "cosine_similarity": float(cosine),
        "max_abs": float(difference.abs().max()),
        "bitwise_equal": bool(torch.equal(candidate, baseline)),
    }


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
        "benchmark": "kimi_k3_batched_expert_probe",
        "dry_run": dry_run,
        "git_sha": os.environ.get("MOK_GIT_SHA", "unknown"),
        "mma_shape": {
            "m": 128,
            "n": 8,
            "k": 32,
            "m_axis": "output_channel",
            "n_axis": "token",
        },
        "rows": list(PROBE_ROWS),
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "repeats": repeats,
        "integration_status": "isolated_not_integrated",
        "measurement_gate": (
            "candidate median gain must exceed the larger repeat-median "
            "dispersion and pass numerical tolerances at every row count"
        ),
    }


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
        raise RuntimeError("the batched expert probe requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("the batched expert probe requires an SM103 B300")
    extension = _extension()
    if extension.kimi_k3_decode_workspace_bytes() != SCRATCH_BYTES:
        raise RuntimeError("probe scratch contract is stale")

    # region agent log
    _debug_log(
        "A",
        "benchmarks/kimi_k3_batched_expert_probe.py:run",
        "probe entry",
        {
            "rows": list(PROBE_ROWS),
            "warmup_count": warmup_count,
            "sample_count": sample_count,
            "repeats": repeats,
        },
    )
    # endregion

    route_coverage = {
        str(tokens): same_expert_batching_coverage(
            route_assignments(tokens, 0)
        )
        for tokens in (16, 32, 128)
    }
    # region agent log
    _debug_log(
        "C",
        "benchmarks/kimi_k3_batched_expert_probe.py:route_coverage",
        "same-expert batching coverage",
        route_coverage,
    )
    # endregion

    l2_bytes = _l2_bytes(device)
    pool_size = l2_bytes // EXPERT_WEIGHT_BYTES + 1
    weights = _weight_pool(device, pool_size)
    generator = torch.Generator(device=device).manual_seed(7500)
    latents = (
        torch.randn(
            pool_size,
            max(PROBE_ROWS),
            LATENT,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.25
    ).bfloat16()
    baseline_output = torch.empty(
        max(PROBE_ROWS), LATENT, dtype=torch.bfloat16, device=device
    )
    candidate_output = torch.empty_like(baseline_output)
    scratch = torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device=device)

    rows_output: list[dict[str, Any]] = []
    raw_output: dict[str, object] = {}
    for rows in PROBE_ROWS:
        numerical = _numerical_metrics(
            extension,
            latents[0],
            weights,
            baseline_output,
            candidate_output,
            scratch,
            rows,
        )
        # region agent log
        _debug_log(
            "D",
            "benchmarks/kimi_k3_batched_expert_probe.py:numerical",
            "candidate numerical comparison",
            {"rows": rows, **numerical},
        )
        # endregion

        baseline_graphs = _capture_pool(
            extension, latents, weights, baseline_output, scratch, rows,
            False,
        )
        candidate_graphs = _capture_pool(
            extension, latents, weights, candidate_output, scratch, rows,
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
        # region agent log
        _debug_log(
            "A",
            "benchmarks/kimi_k3_batched_expert_probe.py:baseline_timing",
            "baseline timing complete",
            {
                "rows": rows,
                "repeat_medians_ms": [
                    summary["median_ms"] for summary in baseline_summaries
                ],
            },
        )
        # endregion
        candidate_summaries = [
            summarize_rank_max([samples])
            for samples in samples_by_variant["candidate"]
        ]
        # region agent log
        _debug_log(
            "B",
            "benchmarks/kimi_k3_batched_expert_probe.py:candidate_timing",
            "candidate timing complete",
            {
                "rows": rows,
                "repeat_medians_ms": [
                    summary["median_ms"] for summary in candidate_summaries
                ],
            },
        )
        # endregion
        verdict = evaluate_row(
            baseline_repeat_medians=[
                float(summary["median_ms"])
                for summary in baseline_summaries
            ],
            candidate_repeat_medians=[
                float(summary["median_ms"])
                for summary in candidate_summaries
            ],
            numerical=numerical,
        )
        # region agent log
        _debug_log(
            "A,B,D",
            "benchmarks/kimi_k3_batched_expert_probe.py:row_verdict",
            "row gate evaluated",
            {"rows": rows, **verdict},
        )
        # endregion
        rows_output.append(
            {
                "rows": rows,
                "numerical": numerical,
                "baseline_repeats": baseline_summaries,
                "candidate_repeats": candidate_summaries,
                **verdict,
            }
        )
        raw_output[str(rows)] = samples_by_variant
        baseline_graphs.clear()
        candidate_graphs.clear()

    passed = all(bool(row["passed"]) for row in rows_output)
    row_one = next(row for row in rows_output if row["rows"] == 1)
    if bool(row_one["passed"]):
        decision = {
            "same_expert_integration_candidate": True,
            "reason": (
                "m128x8 handles the one-row M16 distribution by padding only "
                "the seven unused token columns"
            ),
            "alternative": None,
        }
    else:
        decision = {
            "same_expert_integration_candidate": False,
            "reason": (
                "the isolated one-row path did not clear the measurement gate"
            ),
            "alternative": (
                "software-batch independent expert units in one CTA pipeline; "
                "keep one weight descriptor and accumulator per expert and "
                "never place different experts in one MMA"
            ),
        }
    result = {
        "passed": passed,
        "rows": rows_output,
        "route_coverage": route_coverage,
        "working_set": {
            "l2_bytes": l2_bytes,
            "expert_weight_bytes": EXPERT_WEIGHT_BYTES,
            "pool_size": pool_size,
            "weight_pool_bytes": pool_size * EXPERT_WEIGHT_BYTES,
            "exceeds_l2": pool_size * EXPERT_WEIGHT_BYTES > l2_bytes,
        },
        "decision": decision,
    }
    # region agent log
    _debug_log(
        "A,B,C,D",
        "benchmarks/kimi_k3_batched_expert_probe.py:exit",
        "probe exit",
        {"passed": passed, "decision": decision},
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
    if DEBUG_LOG_PATH.exists():
        (output_dir / "debug.log").write_bytes(DEBUG_LOG_PATH.read_bytes())
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
            "batched expert probe failed; the candidate remains isolated "
            "and must not be integrated"
        )


if __name__ == "__main__":
    main()
