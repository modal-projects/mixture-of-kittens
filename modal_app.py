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

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import modal

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
BUILD_DIRS = ("csrc", "mok", "benchmarks", "tests", "third_party/ThunderKittens")
BUILD_FILES = (
    "setup.py",
    "pyproject.toml",
    "Makefile",
    "README.md",
    "LICENSE",
)
REMOTE_ROOT = "/root/mok"


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
            ignore=["**/__pycache__", "**/*.so", "**/*.egg-info", "**/.git"],
        )
    for file in BUILD_FILES:
        image = image.add_local_file(file, remote_path=f"{REMOTE_ROOT}/{file}", copy=True)
    compiled = image.run_commands(
        # Build the CUDA extension during image build; stub dir provides libcuda.
        f"cd {REMOTE_ROOT} && LIBRARY_PATH={CUDA_STUBS} pip install -e . --no-build-isolation",
    ).workdir(REMOTE_ROOT)
    # Runtime-only source contracts read this file. Keep it after compilation
    # so edits to Modal orchestration cannot invalidate the CUDA build layer.
    return compiled.add_local_file(
        "modal_app.py",
        remote_path=f"{REMOTE_ROOT}/modal_app.py",
        copy=True,
    )


IMAGE = build_image(SPEC)
B300_IMAGE = build_image(SPECS["B300"])


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


@app.local_entrypoint()
def main() -> None:
    gpu_info.remote()
