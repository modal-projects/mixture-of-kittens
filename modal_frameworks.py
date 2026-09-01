"""The pinned vLLM and SGLang comparison runs and their graph-route probe."""

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from benchmarks.compare_kimi_k3_frameworks import (
    combine_archives,
    comparison_artifact_files,
)
from benchmarks.kimi_k3_artifacts import reproducible_tar_bytes
from modal_images import (
    app,
    REMOTE_ROOT,
    COMPARISON_IMAGES,
    VLLM_COMPARISON_IMAGE,
    SGLANG_COMPARISON_IMAGE,
)
from modal_bench import _run_kimi_k3_torchrun


#: The gap in a comparison's output that is read as the known SGLang deadlock.
#:
#: One SGLang run on the same commit and the same pinned digest printed nothing
#: at all between torchrun's startup and the orchestration giving up, while
#: repeats of it finished: the hang is transient and it is in the framework's own
#: layer, before the first shape. The driver names every stage it enters, so
#: silence this long is that hang rather than the work -- a slow but talking run
#: is not touched by this bound, which is why it is silence and not duration.
COMPARISON_STALL_TIMEOUT = 2_700

#: Attempts per comparison. Two, because the hang is transient and a second one
#: is a finding rather than something to keep retrying through.
COMPARISON_ATTEMPTS = 2

#: Total allowance per attempt, kept to under half the function's own timeout so
#: both attempts and the archive still fit inside it.
COMPARISON_ATTEMPT_TIMEOUT = 38_000


def _run_framework_comparison(
    framework: str,
    git_sha: str,
    *,
    warmup_count: int,
    sample_count: int,
    modes: str,
    tokens: str,
) -> bytes:
    """Run one framework comparison on 8x B300 and return its artifact archive."""
    if len(git_sha) != 40:
        raise ValueError("git_sha must be the full 40-character commit SHA")
    with tempfile.TemporaryDirectory(prefix=f"kimi-k3-{framework}-") as directory:
        output_dir = Path(directory) / "artifacts"
        reference = COMPARISON_IMAGES[framework]
        print(f"{framework} comparison image: {reference}")
        arguments = [
            "-m",
            "benchmarks.compare_kimi_k3_frameworks",
            "--framework",
            framework,
            "--output-dir",
            str(output_dir),
            "--warmup-count",
            str(warmup_count),
            "--sample-count",
            str(sample_count),
            "--modes",
            modes,
        ]
        if tokens:
            arguments += ["--tokens", tokens]
        for attempt in range(1, COMPARISON_ATTEMPTS + 1):
            # A stalled attempt leaves whatever it had written behind, and the
            # artifact check below is an exact set, so the next attempt starts
            # from nothing rather than from a partial run's leftovers.
            shutil.rmtree(output_dir, ignore_errors=True)
            try:
                _run_kimi_k3_torchrun(
                    arguments,
                    timeout=COMPARISON_ATTEMPT_TIMEOUT,
                    stall_timeout=COMPARISON_STALL_TIMEOUT,
                    environment={
                        "MOK_GIT_SHA": git_sha,
                        # The driver records this in the archive manifest and
                        # refuses it if it is not the digest the manifest pins,
                        # so the archive names the image it was actually
                        # produced by.
                        "MOK_COMPARISON_IMAGE_REF": reference,
                    },
                )
            except subprocess.TimeoutExpired as expired:
                # Only a run that stopped talking is retried. A run that failed
                # raises `CalledProcessError` and is not caught here, because a
                # numerical or gate failure is the answer and repeating it would
                # only bury it.
                print(
                    f"{framework} comparison attempt {attempt} of "
                    f"{COMPARISON_ATTEMPTS} went quiet after "
                    f"{expired.timeout}s and was ended"
                )
                if attempt == COMPARISON_ATTEMPTS:
                    raise
                continue
            break
        measured = [mode for mode in modes.split(",") if mode]
        expected = set(comparison_artifact_files(measured))
        actual = {path.name for path in output_dir.iterdir()}
        if actual != expected:
            raise RuntimeError(
                f"{framework} comparison artifacts differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        first_archive = reproducible_tar_bytes(output_dir)
        if first_archive != reproducible_tar_bytes(output_dir):
            raise RuntimeError("normalized comparison archive is not reproducible")
        return first_archive


def _run_graph_route_probe(framework: str) -> bytes:
    """Replay both router constructions on the device and return the report."""
    output_dir = Path(REMOTE_ROOT) / "kimi_k3_graph_routes"
    _run_kimi_k3_torchrun(
        [
            "-m",
            "benchmarks.kimi_k3_graph_route_probe",
            "--framework",
            framework,
            "--output-dir",
            str(output_dir),
        ],
        timeout=5_400,
    )
    return (output_dir / "graph_routes.json").read_bytes()


@app.function(image=VLLM_COMPARISON_IMAGE, gpu="B300:8", timeout=7_200)
def graph_routes_vllm() -> bytes:
    return _run_graph_route_probe("vllm")


@app.function(image=SGLANG_COMPARISON_IMAGE, gpu="B300:8", timeout=7_200)
def graph_routes_sglang() -> bytes:
    return _run_graph_route_probe("sglang")


@app.function(image=VLLM_COMPARISON_IMAGE, gpu="B300:8", timeout=86_400)
def compare_vllm(
    git_sha: str,
    warmup_count: int = 500,
    sample_count: int = 1000,
    modes: str = "block8,block16",
    tokens: str = "",
) -> bytes:
    """Compare the custom kernel with vLLM's native Kimi K3 MoE layer."""
    return _run_framework_comparison(
        "vllm",
        git_sha,
        warmup_count=warmup_count,
        sample_count=sample_count,
        modes=modes,
        tokens=tokens,
    )


@app.function(image=SGLANG_COMPARISON_IMAGE, gpu="B300:8", timeout=86_400)
def compare_sglang(
    git_sha: str,
    warmup_count: int = 500,
    sample_count: int = 1000,
    modes: str = "block8,block16",
    tokens: str = "",
) -> bytes:
    """Compare the custom kernel with SGLang's native Kimi K3 MoE layer."""
    return _run_framework_comparison(
        "sglang",
        git_sha,
        warmup_count=warmup_count,
        sample_count=sample_count,
        modes=modes,
        tokens=tokens,
    )


@app.local_entrypoint()
def compare(
    git_sha: str,
    output_dir: str = "kimi_k3_comparison",
    warmup_count: int = 500,
    sample_count: int = 1000,
    modes: str = "block8,block16",
    tokens: str = "",
    frameworks: str = "vllm,sglang",
) -> None:
    """Run the framework comparisons, combine them, and enforce both gates.

    The two archives are unpacked, then joined into one combined numerical and
    performance verdict that is written next to them. Every artifact is on disk
    before the verdict is signalled, so a failing gate leaves a complete run
    behind rather than an aborted one.
    """
    entrypoints = {"vllm": compare_vllm, "sglang": compare_sglang}
    root = Path(output_dir)
    handles = {
        name: entrypoints[name].spawn(
            git_sha,
            warmup_count=warmup_count,
            sample_count=sample_count,
            modes=modes,
            tokens=tokens,
        )
        for name in frameworks.split(",")
        if name
    }
    directories = []
    for name, handle in handles.items():
        destination = root / name
        destination.mkdir(parents=True, exist_ok=True)
        archive = handle.get()
        (root / f"{name}.tar").write_bytes(archive)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(destination, filter="data")
        directories.append(destination)
        print(f"{name}: {sorted(path.name for path in destination.iterdir())}")

    summary = combine_archives(directories, root / "combined")
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "numerical_gates": {
                    "passed": summary["numerical_gates"]["passed"],
                    "row_count": summary["numerical_gates"]["row_count"],
                    "violations": summary["numerical_gates"]["violations"],
                },
                "performance_gates": summary["performance_gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not summary["passed"]:
        raise SystemExit(
            "Kimi K3 comparison gates failed; artifacts are in "
            f"{root} and the verdict is in {root / 'combined'}"
        )


@app.local_entrypoint()
def graph_routes(
    output_dir: str = "kimi_k3_graph_routes",
    frameworks: str = "vllm,sglang",
) -> None:
    """Show what each captured native router graph actually replays."""
    entrypoints = {"vllm": graph_routes_vllm, "sglang": graph_routes_sglang}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    handles = {
        name: entrypoints[name].spawn()
        for name in frameworks.split(",")
        if name
    }
    for name, handle in handles.items():
        report = handle.get()
        (root / f"{name}.json").write_bytes(report)
        print(f"{name}: {root / f'{name}.json'}")
