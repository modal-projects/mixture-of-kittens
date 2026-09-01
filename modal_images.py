"""The app, the architectures it targets, and the images it builds."""

import os
from dataclasses import dataclass

import modal

from benchmarks.compare_kimi_k3_frameworks import pinned_image_reference


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
# Every module the app is spread across, in the order a reader would meet them.
# `modal_app.py` imports the other five, so pointing `modal run` at it registers
# every function and entrypoint, but all six have to be in the image.
ORCHESTRATION_FILES = (
    "modal_app.py",
    "modal_images.py",
    "modal_bench.py",
    "modal_k3_gates.py",
    "modal_k3_probes.py",
    "modal_frameworks.py",
)
_COPY_IGNORE = ["**/__pycache__", "**/*.so", "**/*.egg-info", "**/.git"]


def build_image(
    spec: GPUSpec, wait_timeout_scale: int = 1
) -> modal.Image:
    """Build a MoK image for one architecture. Modal caches each distinct image.

    ``wait_timeout_scale`` widens every bounded spin's compile-time clock
    budget, and only a compute-sanitizer image passes anything but one. The
    tool's instrumentation slows a launch by one to two orders of magnitude, so
    a cross-CTA rendezvous that takes microseconds unmeasured can take longer
    than the fifteen seconds the watchdog allows -- at which point the `trap`
    that ends the spin takes the launch down and the tool reports zero hazards
    for a run that never finished. Five racecheck runs were lost to that.

    It is passed to the compiler, not to the process: the constant is
    ``constexpr``, there is no branch and no read, and an image that leaves it
    at one is byte-identical to one from before the knob existed. Both the
    compiled scale and the scale the image declares are readable at runtime, and
    `test_the_wait_budget_is_the_one_this_image_declares` requires them to
    agree -- so a production image cannot silently carry a widened budget and a
    sanitizer image cannot silently fail to get one.
    """
    if wait_timeout_scale < 1:
        raise ValueError(
            f"the wait budget widens or stays put, got {wait_timeout_scale}"
        )
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{spec.cuda_tag}", add_python="3.12")
        .apt_install("build-essential", "git")
        # setuptools>=80 is required by the repo build (PEP 639 license metadata);
        # install it explicitly since we build with --no-build-isolation.
        .pip_install("setuptools>=80", "wheel")
        .pip_install(spec.torch_spec, index_url=spec.torch_index)
        .pip_install("pytest>=9,<10", "numpy")
        .env(
            {
                "MOK_ARCH": spec.mok_arch,
                # Read by `setup.py` at compile time and by the contract test at
                # run time, which is what ties the two together.
                "MOK_WAIT_TIMEOUT_SCALE": str(wait_timeout_scale),
            }
        )
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
    # Modal deserializes a remote function against its own module, and the
    # runtime-only source contracts read these files, so every orchestration
    # module travels -- not just the one `modal run` is pointed at. Kept after
    # compilation so editing orchestration cannot invalidate the CUDA layer.
    for file in ORCHESTRATION_FILES:
        compiled = compiled.add_local_file(
            file, remote_path=f"{REMOTE_ROOT}/{file}", copy=True
        )
    return compiled


IMAGE = build_image(SPEC)
B300_IMAGE = build_image(SPECS["B300"])

#: What one sanitizer gate is allowed on the wall before Modal ends it.
#:
#: racecheck instruments every shared access in a kernel that is almost entirely
#: shared traffic, so it is the only tool this bound is for; the two cheap ones
#: finish inside twenty minutes and never approach it. Eight hours is not a
#: measured cost -- it is the point past which a run is not worth more than a
#: narrower run that ends, and a racecheck selection that reaches it is a
#: selection to cut rather than a bound to raise. One did reach it, which is why
#: section 63 of the Task 11b report cuts a selection.
SANITIZER_GATE_TIMEOUT = 28_800

#: How much of that gate is kept back to write the artifact with.
#:
#: The tool's own budget is the rest, and it has to be the *rest* rather than the
#: whole less a constant: Modal counts its timeout from the call, so the image's
#: startup and the imports come out of the same eight hours before the tool
#: starts. Giving the subprocess `GATE - 300` from several minutes in is giving it
#: a deadline past Modal's, and an eight-hour racecheck run was lost exactly
#: there -- the container was killed with the tool still running and nothing
#: written. Ten minutes is far more than `_persist_k3_artifact` needs for a log
#: measured in megabytes, and being generous with it costs a run nothing: a
#: selection that only fits in the last ten minutes does not fit.
SANITIZER_TEARDOWN_SECONDS = 600

#: The fastest SM clock a bounded spin on this part can be counting, in Hz.
#:
#: Only an upper bound is wanted, because it is what turns a wall-clock budget
#: into a clock budget that cannot come out short. A B300 SM boosts to about
#: 1.9 GHz, so 2.5 GHz has margin over any part and any future clock bump.
B300_CLOCK_CEILING_HZ = 2_500_000_000

#: Production's budget, restated here so the scale below can be derived.
#:
#: `test_the_base_the_scale_is_taken_against_is_the_compiled_one` reads it back
#: out of `csrc/serial_sync.cuh`, so a copy that drifted from the constant it is
#: a copy of fails on a CPU rather than by silently mis-scaling an image.
WAIT_TIMEOUT_BASE_CLOCKS = 30_000_000_000

#: How much wider the sanitizer image's bounded spins are than production's.
#:
#: Derived rather than picked, because picking it is what failed twice. The
#: watchdog turns a lost peer into a device trap instead of a wedged GPU, and
#: under a sanitizer the gate's own timeout already does that job -- so the only
#: sound budget here is one the gate always reaches first. Anything shorter is a
#: number racing a slowdown nobody measured: at 1x a rendezvous crossed fifteen
#: seconds and section 43 lost four runs to it, and at 64x one crossed sixteen
#: minutes and lost a fifth, each time reported as zero hazards for a launch
#: that never finished.
#:
#: So the budget is the gate's whole wall clock counted at a clock no B300
#: exceeds. A racecheck launch cannot then end as a watchdog trap: either the
#: work finishes, or the gate runs out of wall clock and says so, which is a
#: failure a reader cannot mistake for a clean one.
#:
#: The run after that one did end the second way, and it is worth being clear
#: about what this constant does and does not fix. It makes the *verdict* honest
#: -- a launch that does not finish is reported as not finishing rather than as
#: zero hazards. It does not make a selection affordable, and it cannot: no
#: budget makes racecheck's instrumentation cheaper. That is a selection problem
#: and `K3_SANITIZER_SELECTION` is where it is solved.
SANITIZER_WAIT_TIMEOUT_SCALE = -(
    -SANITIZER_GATE_TIMEOUT * B300_CLOCK_CEILING_HZ // WAIT_TIMEOUT_BASE_CLOCKS
)

#: The image the sanitizer gates run in, and nothing else runs in.
#:
#: A separate image rather than a flag on the shared one, because the widened
#: budget is compiled in: an image that carried it would be a different binary
#: from the one every other gate measures and ships, and the point of the
#: sanitizer gates is to make claims about *that* binary. Everything else about
#: it is identical, so a finding here is attributable to the tool and the source
#: rather than to the build.
B300_SANITIZER_IMAGE = build_image(
    SPECS["B300"], wait_timeout_scale=SANITIZER_WAIT_TIMEOUT_SCALE
)

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
    # The same reason `build_image` carries these: `compare_vllm` lives in
    # `modal_frameworks`, and Modal deserializes it against that module rather
    # than against the file `modal run` was pointed at. Modal mounts the entry
    # module by itself, which is why a single-file app needed nothing here and
    # why the split broke this image first.
    for file in ORCHESTRATION_FILES:
        compiled = compiled.add_local_file(
            file, remote_path=f"{REMOTE_ROOT}/{file}", copy=True
        )
    return compiled.workdir(REMOTE_ROOT)


VLLM_COMPARISON_IMAGE = framework_comparison_image(COMPARISON_IMAGES["vllm"])
SGLANG_COMPARISON_IMAGE = framework_comparison_image(COMPARISON_IMAGES["sglang"])
