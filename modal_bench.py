"""The build check and the MoK benchmark that run on the shared image."""

import contextlib
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from benchmarks.kimi_k3_artifacts import reproducible_tar_bytes
from modal_images import SPEC, BENCH_NPROC, app, REMOTE_ROOT, IMAGE, B300_IMAGE


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


def _end_session(process: "subprocess.Popen[str]") -> None:
    """End the launcher's whole process group, not only the launcher.

    A torchrun's ranks are its children, and `subprocess`'s own timeout kills
    the launcher alone. A rank blocked inside a framework's collective would
    survive that, keep its GPU, and leave the next attempt to fail on a device
    that is still busy. The child is started in its own session so the group can
    be signalled directly here, and because it leads that session the group id is
    its pid -- read from the handle rather than from `getpgid`, which stops
    answering once the launcher itself is gone.

    The kill is unconditional rather than an escalation the launcher's exit can
    cut short. The launcher forwards SIGTERM and exits; a rank that ignores it,
    or that is inside a driver call that will not return to run a handler, does
    not. Waiting only on the launcher would report success with ranks still on
    the GPUs, which is the case this exists for.
    """
    group = process.pid
    for number in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(group, number)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30)


def _stream_bounded(
    process: "subprocess.Popen[str]",
    *,
    deadline: float,
    stall_timeout: int | None,
) -> tuple[str, str | None]:
    """Echo the child's output, bounded by silence as well as by total time.

    Returns the text and why it stopped: ``None`` when the child closed its
    output normally, ``"silence"`` when nothing arrived for ``stall_timeout``
    seconds, and ``"budget"`` when the whole allowance ran out.

    Silence is the useful bound. A run under a deadlock produces no output and
    no exit, so waiting on the process cannot tell it from a slow one, while a
    total-duration bound has to be set so far above a healthy run that it stops
    being a diagnostic. Every caller that passes a ``stall_timeout`` runs
    something that says where it is as it goes.
    """
    lines: list[str] = []
    inbox: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            inbox.put(line)
        inbox.put(None)

    threading.Thread(target=pump, daemon=True).start()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "".join(lines), "budget"
        wait = remaining if stall_timeout is None else min(stall_timeout, remaining)
        try:
            line = inbox.get(timeout=wait)
        except queue.Empty:
            return "".join(lines), "silence" if wait < remaining else "budget"
        if line is None:
            return "".join(lines), None
        print(line, end="", flush=True)
        lines.append(line)


def _run_kimi_k3_torchrun(
    arguments: list[str],
    *,
    timeout: int,
    environment: dict[str, str] | None = None,
    attribute_ranks: bool = False,
    capture: bool = False,
    stall_timeout: int | None = None,
) -> str:
    """Run one torchrun under the B300s, returning its output when captured.

    Every line is echoed as it arrives, because a gate that stops making
    progress has to stay observable; ``capture`` only additionally keeps the text
    so a caller can persist it as the gate's artifact.

    ``stall_timeout`` bounds the gap between lines rather than the run, and a
    run that exceeds either bound raises `subprocess.TimeoutExpired` after its
    whole process group has been ended.
    """
    # Eight ranks write to one console, so an unlabelled `pytest -q` progress
    # line is eight interleaved streams of dots and a run that stops making
    # progress cannot be attributed to a test. `--tee 3` prefixes every line
    # with its rank, which is what makes `-v` readable here.
    labelling = ["--tee", "3", "--role", "rank"] if attribute_ranks else []
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        *labelling,
        *arguments,
    ]
    print(f"Launching: {' '.join(command)} on 8 x B300")
    process = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
        env={
            **os.environ,
            **(environment or {}),
            "PYTHONUNBUFFERED": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    started = time.monotonic()
    output, expired = _stream_bounded(
        process, deadline=started + timeout, stall_timeout=stall_timeout
    )
    if expired is not None:
        _end_session(process)
        elapsed = int(time.monotonic() - started)
        print(
            f"torchrun ended after {expired} at {elapsed}s "
            f"(budget {timeout}s, stall bound {stall_timeout}s)"
        )
        raise subprocess.TimeoutExpired(command, elapsed, output=output)
    left = max(1, int(started + timeout - time.monotonic()))
    try:
        returncode = process.wait(timeout=left)
    except subprocess.TimeoutExpired:
        _end_session(process)
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=output)
    return output if capture else ""


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
