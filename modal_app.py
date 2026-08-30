"""Modal app for building and running Mixture-of-Kittens (MoK) on Blackwell GPUs.

MoK is a MoE training megakernel that requires NVIDIA Blackwell GPUs, CUDA 13.0+, and
PyTorch 2.10+. Those GPUs are not present on a typical dev box, so this app builds and
runs MoK on Modal's Blackwell fleet instead.

Target: 8x B300 (SM103). B300 requires CUDA 13.1+, and MoK's ``setup.py`` requires the
nvcc version to match PyTorch's CUDA version, so B300 is built against CUDA 13.2 with
``torch==2.13.0+cu132``. A B200 (SM100) spec is kept alongside it so the same machinery
covers both architectures; each spec produces its own image that Modal content-addresses
and caches independently, so a given architecture is compiled only once and reused across
runs until its source or config changes.

The CUDA extension is compiled once, inside the image build (a GPU-less CPU builder), by
pointing the linker at the CUDA driver *stub* for ``-lcuda``. At runtime the real driver is
injected by Modal on the GPU container.

Usage (from the repo root, with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set):

    modal run modal_app.py                 # build check on a single B300
    modal run modal_app.py::gpu_info       # same, explicit
    modal run modal_app.py::bench          # 8x B300 benchmark + correctness check

Environment overrides:
    MOK_GPU          (default B300)  which spec below to use (B300 or B200)
    MOK_BENCH_NPROC  (default 8)     GPUs / EP ranks for the benchmark (1, 4, or 8)
"""

import io
import json
import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import modal

from benchmarks.compare_kimi_k3_frameworks import (
    combine_archives,
    comparison_artifact_files,
    pinned_image_reference,
)
from benchmarks.kimi_k3_artifacts import reproducible_tar_bytes


@dataclass(frozen=True)
class GPUSpec:
    gpu: str          # Modal GPU type
    cuda_tag: str     # nvidia/cuda devel image tag (nvcc)
    torch_spec: str   # torch requirement
    torch_index: str  # PyTorch wheel index (must match cuda_tag major.minor)
    mok_arch: str     # MOK_ARCH passed to the build


# One spec per architecture. Each yields a separately-cached Modal image.
SPECS: dict[str, GPUSpec] = {
    # B300 (Blackwell Ultra, SM103): CUDA 13.1+ required -> CUDA 13.2 + torch cu132.
    "B300": GPUSpec(
        gpu="B300",
        cuda_tag="13.2.1-devel-ubuntu24.04",
        torch_spec="torch==2.13.0",
        torch_index="https://download.pytorch.org/whl/cu132",
        mok_arch="SM103",
    ),
    # B200 (Blackwell, SM100): CUDA 13.0 + torch cu130.
    "B200": GPUSpec(
        gpu="B200",
        cuda_tag="13.0.1-devel-ubuntu24.04",
        torch_spec="torch==2.10.0",
        torch_index="https://download.pytorch.org/whl/cu130",
        mok_arch="SM100",
    ),
}

GPU_TYPE = os.environ.get("MOK_GPU", "B300")
SPEC = SPECS[GPU_TYPE]
BENCH_NPROC = int(os.environ.get("MOK_BENCH_NPROC", "8"))  # 8x B300 by default

# The CUDA driver stub satisfies `-lcuda` at build time (no GPU/driver on the builder).
CUDA_STUBS = "/usr/local/cuda/lib64/stubs"

app = modal.App("mixture-of-kittens")


# Only the paths needed to build and run MoK are copied into the image. Using an explicit
# allowlist (instead of the whole repo) keeps the image content-hash stable, so Modal's
# layer cache is reused across runs and unrelated files (logs, .venv, .cursor, editor
# state) never invalidate the cached compile.
BUILD_DIRS = ("csrc", "mok", "third_party/ThunderKittens")
# Added after the compile so editing a benchmark or a test reuses the cached
# CUDA layer instead of recompiling the extension.
RUNTIME_DIRS = ("benchmarks", "tests")
BUILD_FILES = (
    "setup.py",
    "pyproject.toml",
    "Makefile",
    "README.md",
    "LICENSE",
)
REMOTE_ROOT = "/root/mok"
_COPY_IGNORE = ["**/__pycache__", "**/*.so", "**/*.egg-info", "**/.git"]


def build_image(spec: GPUSpec) -> modal.Image:
    """Build a MoK image for one architecture. Modal caches each distinct image."""
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{spec.cuda_tag}", add_python="3.12")
        .apt_install("build-essential", "git")
        # setuptools>=80 is required by the repo build (PEP 639 license metadata);
        # install it explicitly since we build with --no-build-isolation.
        .pip_install("setuptools>=80", "wheel")
        .pip_install(spec.torch_spec, index_url=spec.torch_index)
        .pip_install("pytest>=9,<10", "numpy")
        .env({"MOK_ARCH": spec.mok_arch})
    )
    for directory in BUILD_DIRS:
        image = image.add_local_dir(
            directory,
            remote_path=f"{REMOTE_ROOT}/{directory}",
            copy=True,
            ignore=_COPY_IGNORE,
        )
    for file in BUILD_FILES:
        image = image.add_local_file(file, remote_path=f"{REMOTE_ROOT}/{file}", copy=True)
    compiled = image.run_commands(
        # Build the CUDA extension during image build; stub dir provides libcuda.
        f"cd {REMOTE_ROOT} && LIBRARY_PATH={CUDA_STUBS} pip install -e . --no-build-isolation",
    ).workdir(REMOTE_ROOT)
    for directory in RUNTIME_DIRS:
        compiled = compiled.add_local_dir(
            directory,
            remote_path=f"{REMOTE_ROOT}/{directory}",
            copy=True,
            ignore=_COPY_IGNORE,
        )
    # Runtime-only source contracts read this file. Keep it after compilation
    # so edits to Modal orchestration cannot invalidate the CUDA build layer.
    return compiled.add_local_file(
        "modal_app.py",
        remote_path=f"{REMOTE_ROOT}/modal_app.py",
        copy=True,
    )


IMAGE = build_image(SPEC)
B300_IMAGE = build_image(SPECS["B300"])

# Framework comparison images. Each one derives from an official Kimi K3 serving
# image and compiles this repository's extension against that image's own
# PyTorch and CUDA ABI, so no wheel ever crosses an ABI boundary. The images ship
# pip-installed CUDA wheels rather than a full toolkit, so the compile needs
# CPATH pointed at the nvidia package headers.
#
# The reference is `repository@sha256:<digest>` resolved from
# `benchmarks/framework_manifest.json`, not the `:kimi-k3` tag those digests
# were captured from. A tag is a moving pointer, and an archive that recorded a
# digest while its image was built from whatever the tag resolved to that
# morning would be reporting a pin it never used.
COMPARISON_IMAGES = {
    framework: pinned_image_reference(framework)
    for framework in ("vllm", "sglang")
}
_NVIDIA_INCLUDE_GLOB = "/usr/local/lib/python3.12/dist-packages/nvidia/*/include"
_CPATH_SNIPPET = (
    "import glob;print(':'.join(sorted(glob.glob("
    f"'{_NVIDIA_INCLUDE_GLOB}'))))"
)
_COMPARISON_BUILD_COMMAND = (
    f'cd {REMOTE_ROOT} && CPATH="$(python -c "{_CPATH_SNIPPET}")" '
    f"LIBRARY_PATH={CUDA_STUBS} "
    "pip install -e . --no-build-isolation --no-deps"
)


# Only these paths take part in the CUDA compile. The harness packages are added
# after it so editing a driver or an adapter reuses the cached compile layer.
COMPARISON_BUILD_DIRS = BUILD_DIRS
COMPARISON_RUNTIME_DIRS = RUNTIME_DIRS


def framework_comparison_image(registry_reference: str) -> modal.Image:
    """Derive a comparison image from one pinned Kimi K3 serving digest."""
    if "@sha256:" not in registry_reference:
        raise ValueError(
            "a comparison image must be built from a pinned digest, got "
            f"{registry_reference!r}"
        )
    image = (
        modal.Image.from_registry(
            registry_reference,
            setup_dockerfile_commands=[
                "RUN ln -sf /usr/bin/python3 /usr/local/bin/python "
                "&& python --version",
            ],
        )
        .entrypoint([])
        .apt_install("build-essential", "git")
        .pip_install("setuptools>=80", "wheel")
        .env({"MOK_ARCH": "SM103"})
    )
    for directory in COMPARISON_BUILD_DIRS:
        image = image.add_local_dir(
            directory,
            remote_path=f"{REMOTE_ROOT}/{directory}",
            copy=True,
            ignore=_COPY_IGNORE,
        )
    for file in BUILD_FILES:
        image = image.add_local_file(file, remote_path=f"{REMOTE_ROOT}/{file}", copy=True)
    compiled = image.run_commands(_COMPARISON_BUILD_COMMAND)
    for directory in COMPARISON_RUNTIME_DIRS:
        compiled = compiled.add_local_dir(
            directory,
            remote_path=f"{REMOTE_ROOT}/{directory}",
            copy=True,
            ignore=_COPY_IGNORE,
        )
    return compiled.workdir(REMOTE_ROOT)


VLLM_COMPARISON_IMAGE = framework_comparison_image(COMPARISON_IMAGES["vllm"])
SGLANG_COMPARISON_IMAGE = framework_comparison_image(COMPARISON_IMAGES["sglang"])


@app.function(image=IMAGE, gpu=SPEC.gpu, timeout=1800)
def gpu_info() -> None:
    """Confirm the compiled extension imports and the GPU is the expected Blackwell part."""
    import torch

    import mok

    print("=" * 60)
    print(f"torch             : {torch.__version__}")
    print(f"torch CUDA        : {torch.version.cuda}")
    print(f"mok.__version__   : {mok.__version__}")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU               : {name}")
    print(f"compute capability: sm_{major}{minor}")
    print(f"visible GPUs      : {torch.cuda.device_count()}")

    # Exercise a real compiled kernel (mxfp8 weight quantization) end-to-end on the GPU.
    from mok import ops

    w = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    w_fp8, w_sc, _, _ = ops.mxfp8_quantize(w, True, False)
    torch.cuda.synchronize()
    print(f"mxfp8_quantize    : out={tuple(w_fp8.shape)} dtype={w_fp8.dtype} scale={tuple(w_sc.shape)}")
    print("BUILD + KERNEL OK")
    print("=" * 60)


def _run_bench(nproc: int) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # A modest, cheap-but-representative config. NUM_EXPERTS must be divisible by nproc.
    env.setdefault("NUM_EXPERTS", str(8 * nproc))
    env.setdefault("HIDDEN_DIM", "2048")
    env.setdefault("INTERMEDIATE_DIM", "2048")
    env.setdefault("TOPK", "4")
    env.setdefault("NUM_LOCAL_TOKENS", "2048")
    env.setdefault("MINIBATCH_SIZE", "2048")
    env.setdefault("MACROBATCH_SIZE", "8192")
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        "-m",
        "benchmarks.bench_mok",
    ]
    print(f"Launching: {' '.join(cmd)} on {nproc} x {SPEC.gpu}")
    subprocess.run(cmd, cwd="/root/mok", env=env, check=True)


@app.function(image=IMAGE, gpu=f"{SPEC.gpu}:{BENCH_NPROC}", timeout=3600)
def bench() -> None:
    """Run the MoK benchmark (BF16 + MXFP8 forward/backward, correctness + TFLOP/s)."""
    import torch

    print(f"visible GPUs: {torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})")
    _run_bench(BENCH_NPROC)


def _run_kimi_k3_torchrun(
    arguments: list[str],
    *,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> None:
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        *arguments,
    ]
    print(f"Launching: {' '.join(command)} on 8 x B300")
    subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env={
            **os.environ,
            **(environment or {}),
            "PYTHONUNBUFFERED": "1",
        },
        check=True,
        timeout=timeout,
    )


@app.function(image=B300_IMAGE, gpu="B300:8", timeout=14_400)
def test_kimi_k3_decode() -> None:
    """Run TP8 decode correctness plus SM103 resource and launch checks."""
    test_files = sorted(
        str(path)
        for path in Path("tests").glob("test_kimi_k3*.py")
    )
    _run_kimi_k3_torchrun(
        [
            "-m",
            "pytest",
            "-q",
            *test_files,
        ],
        timeout=14_100,
    )


@app.function(image=B300_IMAGE, gpu="B300:8", timeout=86_400)
def bench_kimi_k3_decode(git_sha: str) -> bytes:
    """Run grid tuning and all decode tables, returning rank-0 artifacts."""
    if len(git_sha) != 40:
        raise ValueError("git_sha must be the full 40-character commit SHA")
    with tempfile.TemporaryDirectory(prefix="kimi-k3-decode-") as directory:
        root = Path(directory)
        output_dir = root / "artifacts"
        _run_kimi_k3_torchrun(
            [
                "-m",
                "benchmarks.bench_kimi_k3_decode",
                "--output-dir",
                str(output_dir),
            ],
            timeout=86_100,
            environment={"MOK_GIT_SHA": git_sha},
        )
        expected = {
            "manifest.json",
            "latency_raw_decode.json",
            "latency_raw_decode.csv",
            "latency_block8.json",
            "latency_block8.csv",
            "latency_block16.json",
            "latency_block16.csv",
            "correctness.json",
            "workspace_stats.json",
            "tuning.json",
        }
        actual = {path.name for path in output_dir.iterdir()}
        if actual != expected:
            raise RuntimeError(
                f"Kimi K3 benchmark artifacts differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        first_archive = reproducible_tar_bytes(output_dir)
        second_archive = reproducible_tar_bytes(output_dir)
        if first_archive != second_archive:
            raise RuntimeError("normalized benchmark archive is not reproducible")
        return first_archive


@app.function(image=B300_IMAGE, gpu="B300:8", timeout=86_400)
def bench_kimi_k3_projection_first_ab(
    git_sha: str,
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> bytes:
    """Run the guarded score-first/projection-first scheduling A/B."""
    if len(git_sha) != 40:
        raise ValueError("git_sha must be the full 40-character commit SHA")
    with tempfile.TemporaryDirectory(
        prefix="kimi-k3-projection-first-"
    ) as directory:
        output_dir = Path(directory) / "artifacts"
        debug_log = Path("/opt") / "cursor" / "logs" / "debug.log"
        debug_log.parent.mkdir(parents=True, exist_ok=True)
        debug_log.unlink(missing_ok=True)
        _run_kimi_k3_torchrun(
            [
                "-m",
                "benchmarks.kimi_k3_projection_first_ab",
                "--output-dir",
                str(output_dir),
                "--warmup-count",
                str(warmup_count),
                "--sample-count",
                str(sample_count),
                "--repeats",
                str(repeats),
            ],
            timeout=86_100,
            environment={"MOK_GIT_SHA": git_sha},
        )
        if not debug_log.is_file():
            raise RuntimeError("projection-first A/B debug log was not written")
        (output_dir / "debug.ndjson").write_bytes(debug_log.read_bytes())
        expected = {
            "debug.ndjson",
            "manifest.json",
            "raw_samples.json",
            "results.json",
        }
        actual = {path.name for path in output_dir.iterdir()}
        if actual != expected:
            raise RuntimeError(
                "projection-first A/B artifacts differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        archive = reproducible_tar_bytes(output_dir)
        if archive != reproducible_tar_bytes(output_dir):
            raise RuntimeError(
                "normalized projection-first A/B archive is not reproducible"
            )
        return archive


@app.local_entrypoint()
def projection_first_ab(
    git_sha: str,
    output_dir: str = "kimi_k3_projection_first_ab",
    warmup_count: int = 500,
    sample_count: int = 1000,
    repeats: int = 5,
) -> None:
    """Run and unpack the TP8 B300 scheduling experiment."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = bench_kimi_k3_projection_first_ab.remote(
        git_sha,
        warmup_count=warmup_count,
        sample_count=sample_count,
        repeats=repeats,
    )
    (destination / "artifacts.tar").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter="data")
    debug_log = Path("/opt") / "cursor" / "logs" / "debug.log"
    debug_log.parent.mkdir(parents=True, exist_ok=True)
    debug_log.write_bytes((destination / "debug.ndjson").read_bytes())
    print(
        "projection-first A/B artifacts: "
        f"{sorted(path.name for path in destination.iterdir())}"
    )


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
        _run_kimi_k3_torchrun(
            arguments,
            timeout=79_000,
            environment={
                "MOK_GIT_SHA": git_sha,
                # The driver records this in the archive manifest and refuses
                # it if it is not the digest the manifest pins, so the archive
                # names the image it was actually produced by.
                "MOK_COMPARISON_IMAGE_REF": reference,
            },
        )
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


@app.local_entrypoint()
def main() -> None:
    gpu_info.remote()
