"""The Kimi K3 probes: batched expert, schedule, and gate/up engine."""

import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from benchmarks.kimi_k3_artifacts import reproducible_tar_bytes
from benchmarks.kimi_k3_engine_probe import print_points as print_engine_points
from modal_images import app, REMOTE_ROOT, B300_IMAGE
from modal_bench import _run_kimi_k3_torchrun
from modal_k3_gates import K3_VOLUME, K3_ARTIFACTS, _persist_k3_artifact


@app.function(image=B300_IMAGE, gpu="B300", timeout=7_200)
def bench_kimi_k3_batched_expert_probe(
    git_sha: str,
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> bytes:
    """Benchmark the isolated m128x8x32 expert candidate on one B300."""
    if len(git_sha) != 40:
        raise ValueError("git_sha must be the full 40-character commit SHA")
    with tempfile.TemporaryDirectory(
        prefix="kimi-k3-batched-expert-"
    ) as directory:
        output_dir = Path(directory) / "artifacts"
        command = [
            "python",
            "-m",
            "benchmarks.kimi_k3_batched_expert_probe",
            "--output-dir",
            str(output_dir),
            "--warmup-count",
            str(warmup_count),
            "--sample-count",
            str(sample_count),
            "--repeats",
            str(repeats),
        ]
        print(f"Launching: {' '.join(command)} on 1 x B300")
        subprocess.run(
            command,
            cwd=REMOTE_ROOT,
            env={
                **os.environ,
                "MOK_GIT_SHA": git_sha,
                "PYTHONUNBUFFERED": "1",
            },
            check=True,
            timeout=7_100,
        )
        expected = {
            "manifest.json",
            "raw_samples.json",
            "results.json",
        }
        actual = {path.name for path in output_dir.iterdir()}
        if actual != expected:
            raise RuntimeError(
                "batched expert probe artifacts differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        archive = reproducible_tar_bytes(output_dir)
        if archive != reproducible_tar_bytes(output_dir):
            raise RuntimeError("normalized probe archive is not reproducible")
        return archive


@app.function(image=B300_IMAGE, gpu="B300", timeout=7_200)
def diagnose_kimi_k3_batched_expert_probe(
    variant: str = "candidate",
    rows: int = 1,
    sanitizer: bool = True,
) -> dict[str, int | str]:
    """Run one synchronized probe launch, optionally under CUDA memcheck."""
    if variant not in ("setup", "baseline", "candidate", "both"):
        raise ValueError(
            "variant must be setup, baseline, candidate, or both"
        )
    if rows < 1 or rows > 8:
        raise ValueError("rows must be between 1 and 8")
    probe_command = [
        "python",
        "-m",
        "benchmarks.kimi_k3_batched_expert_probe",
        "--focus-rows",
        str(rows),
        "--focus-variant",
        variant,
    ]
    command = (
        [
            "compute-sanitizer",
            "--tool",
            "memcheck",
            "--error-exitcode",
            "99",
            "--show-backtrace",
            "yes",
            *probe_command,
        ]
        if sanitizer
        else probe_command
    )
    print(f"Launching: {' '.join(command)} on 1 x B300")
    completed = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=7_100,
    )
    print(completed.stdout, end="")
    print(f"focused probe exit code: {completed.returncode}")
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "output": completed.stdout,
    }


@app.local_entrypoint()
def batched_expert_probe(
    git_sha: str,
    output_dir: str = "kimi_k3_batched_expert_probe",
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> None:
    """Run and unpack the focused one-B300 contraction microbenchmark."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = bench_kimi_k3_batched_expert_probe.remote(
        git_sha,
        warmup_count=warmup_count,
        sample_count=sample_count,
        repeats=repeats,
    )
    (destination / "artifacts.tar").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter="data")
    print(
        "batched expert probe artifacts: "
        f"{sorted(path.name for path in destination.iterdir())}"
    )


@app.function(
    image=B300_IMAGE,
    gpu="B300:8",
    timeout=28_800,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def bench_kimi_k3_schedule_probe(
    git_sha: str,
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> bytes:
    """A/B the dependency-local decode schedule against production on 8x B300.

    The candidate keeps one of the production kernel's five full-grid barriers
    and replaces the rest with dependency-local queues, so the verdict is a
    latency verdict and has to be measured on the eight devices the step is a
    collective over. Both schedules are captured into their own graph pools and
    replayed interleaved, so the comparison is between two orders of arrival on
    one workspace rather than between two runs.
    """
    if len(git_sha) != 40:
        raise ValueError("git_sha must be the full 40-character commit SHA")
    with tempfile.TemporaryDirectory(prefix="kimi-k3-schedule-") as directory:
        output_dir = Path(directory) / "artifacts"
        _run_kimi_k3_torchrun(
            [
                "-m",
                "benchmarks.kimi_k3_schedule_probe",
                "--output-dir",
                str(output_dir),
                "--warmup-count",
                str(warmup_count),
                "--sample-count",
                str(sample_count),
                "--repeats",
                str(repeats),
            ],
            timeout=28_500,
            environment={"MOK_GIT_SHA": git_sha},
        )
        expected = {"manifest.json", "results.json", "raw_samples.json"}
        actual = {path.name for path in output_dir.iterdir()}
        if actual != expected:
            raise RuntimeError(
                "schedule probe artifacts differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        archive = reproducible_tar_bytes(output_dir)
        if archive != reproducible_tar_bytes(output_dir):
            raise RuntimeError("normalized probe archive is not reproducible")
        # Persisted before the return, so a dropped client connection does not
        # take a measurement that already finished with it.
        _persist_k3_artifact("schedule_probe.tar", archive)
        return archive


@app.local_entrypoint()
def schedule_probe(
    git_sha: str,
    output_dir: str = "kimi_k3_schedule_probe",
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> None:
    """Run and unpack the dependency-local schedule A/B."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = bench_kimi_k3_schedule_probe.remote(
        git_sha,
        warmup_count=warmup_count,
        sample_count=sample_count,
        repeats=repeats,
    )
    (destination / "artifacts.tar").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter="data")
    results = json.loads((destination / "results.json").read_text())
    print(json.dumps(results["decision"], indent=2, sort_keys=True))
    for point in results["points"]:
        print(
            f"M{point['tokens']}: "
            f"production {point['production_median_ms']:.4f} ms, "
            f"candidate {point['candidate_median_ms']:.4f} ms, "
            f"{point['improvement_fraction']:+.2%}, "
            f"{'PASS' if point['passed'] else 'FAIL'}"
        )


@app.function(
    image=B300_IMAGE,
    gpu="B300:8",
    timeout=28_800,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def bench_kimi_k3_engine_probe(
    git_sha: str,
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> bytes:
    """A/B production's adaptive gate/up path against the ring it replaced.

    Production spends a third K = 512 weight stage out of the activation two
    ways and picks between them per expert on the device: a compact ring that
    packs all seven slabs into a quarter of the bytes inside a four-row
    threshold, and a slab-buffered ring that stops holding the activation at all
    outside it. The baseline is the two-stage resident ring that was production
    before the integration. The claim is a latency claim about one phase of a
    collective step, so it is measured the same way the schedule A/B was: each
    engine captured into its own graph pool on one workspace and replayed
    interleaved, with the arms held to identical output bytes first.

    Four shapes rather than two, because the selector branches on occupancy:
    M16, M32 and M128 are the realistic routes and take the compact ring, and
    the fourth puts a full eight rows on every expert, which is where production
    runs the slab-buffered ring inside the compact ring's larger allocation.
    """
    if len(git_sha) != 40:
        raise ValueError("git_sha must be the full 40-character commit SHA")
    with tempfile.TemporaryDirectory(prefix="kimi-k3-engine-") as directory:
        output_dir = Path(directory) / "artifacts"
        _run_kimi_k3_torchrun(
            [
                "-m",
                "benchmarks.kimi_k3_engine_probe",
                "--output-dir",
                str(output_dir),
                "--warmup-count",
                str(warmup_count),
                "--sample-count",
                str(sample_count),
                "--repeats",
                str(repeats),
            ],
            timeout=28_500,
            environment={"MOK_GIT_SHA": git_sha},
        )
        expected = {"manifest.json", "results.json", "raw_samples.json"}
        actual = {path.name for path in output_dir.iterdir()}
        if actual != expected:
            raise RuntimeError(
                "engine probe artifacts differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        archive = reproducible_tar_bytes(output_dir)
        if archive != reproducible_tar_bytes(output_dir):
            raise RuntimeError("normalized probe archive is not reproducible")
        _persist_k3_artifact("engine_probe.tar", archive)
        return archive


@app.local_entrypoint()
def engine_probe(
    git_sha: str,
    output_dir: str = "kimi_k3_engine_probe",
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> None:
    """Run and unpack the candidate gate/up engine A/B."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = bench_kimi_k3_engine_probe.remote(
        git_sha,
        warmup_count=warmup_count,
        sample_count=sample_count,
        repeats=repeats,
    )
    (destination / "artifacts.tar").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter="data")
    results = json.loads((destination / "results.json").read_text())
    print(json.dumps(results["decision"], indent=2, sort_keys=True))
    print(json.dumps(results["engine_ledger"], indent=2, sort_keys=True))
    print_engine_points(results["points"])


@app.local_entrypoint()
def batched_expert_diagnostic(
    variant: str = "candidate",
    rows: int = 1,
    sanitizer: bool = True,
    output_path: str = "kimi_k3_batched_expert_diagnostic.log",
) -> None:
    """Run and persist one focused B300 diagnostic invocation."""
    result = diagnose_kimi_k3_batched_expert_probe.remote(
        variant=variant,
        rows=rows,
        sanitizer=sanitizer,
    )
    rendered = (
        f"command: {result['command']}\n"
        f"exit_code: {result['exit_code']}\n"
        f"{result['output']}"
    )
    Path(output_path).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
