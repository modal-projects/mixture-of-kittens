"""Every Kimi K3 gate: the suites, the sanitizers, the SASS, and `verify`."""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

from benchmarks.kimi_k3_sanitizer import (
    K3_SANITIZER_CLAIMS,
    K3_SANITIZER_GATES,
    K3_SANITIZER_SELECTION,
    sanitizer_verdict,
)
from modal_images import (
    app,
    REMOTE_ROOT,
    B300_IMAGE,
    B300_SANITIZER_IMAGE,
    SANITIZER_GATE_TIMEOUT,
    SANITIZER_TEARDOWN_SECONDS,
)
from modal_bench import (
    _end_session,
    _run_kimi_k3_torchrun,
    _stream_bounded,
    bench_kimi_k3_decode,
)


# Where a Kimi K3 verification gate leaves its evidence, written from inside the
# container.
#
# A gate's result also comes back through its return value, but that path runs
# over the local client's gRPC connection, and this repository's longest gates
# outlive it: the saturated benchmark is 30,000 graph replays plus captures, and
# racecheck is memcheck's work at a hundred times the cost. A connection that
# drops at minute twenty takes the whole gate with it -- `--detach` keeps the
# container running but there is no longer a client to hand the artifact to.
# Writing the artifact to a volume before returning it decouples the evidence
# from the connection: the gate is fetched afterwards with `modal volume get`,
# and re-fetching is free.
K3_VOLUME = modal.Volume.from_name(
    "mok-kimi-k3-decode", create_if_missing=True
)
K3_ARTIFACTS = "/artifacts"


def _persist_k3_artifact(name: str, payload: bytes | str) -> None:
    """Write one gate's artifact into the volume and commit it."""
    path = Path(K3_ARTIFACTS) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)
    K3_VOLUME.commit()
    print(f"persisted {name}: {path.stat().st_size} bytes")


#: The suite that must not run beside a live process group.
#:
#: Its tests inject a missed publication and let the candidate's bounded wait
#: run out, and the `trap` that ends the wait is not contained by the context
#: that executed it: the other members of an initialized NCCL group see
#: `cudaErrorLaunchFailure` on their own devices, sometimes. Run as its own
#: single-process gate on one GPU instead, where the trap has nothing to take
#: down but the process that asked for it.
K3_TRAP_FILE = "tests/test_kimi_k3_dependency_schedule_trap.py"

#: The suite that verifies every launch instead of the last one.
#:
#: It synchronizes and checks the device after each of 160 rotating launches,
#: and runs 80 more as 40 interleaved pairs, which is what makes one bad replay
#: fatal rather than overwritten -- and which also makes it far slower per launch
#: than anything else here. The counts live in the gate itself, which asserts its
#: own plan length; `kimi_k3_report_evidence` reads them from there. Kept out of the
#: main suite so that its cost is a gate the reader can see rather than a tax on
#: every correctness run.
K3_STRESS_FILE = "tests/test_kimi_k3_dependency_schedule_stress.py"

#: Suites the TP8 correctness gate does not run, and why, in one place.
K3_SEPARATE_FILES = (K3_TRAP_FILE, K3_STRESS_FILE)


def _k3_test_files() -> tuple[str, ...]:
    """Every Kimi K3 suite the TP8 session runs, in collection order."""
    return tuple(
        sorted(
            str(path)
            for path in Path("tests").glob("test_kimi_k3*.py")
            if str(path) not in K3_SEPARATE_FILES
        )
    )


@app.function(
    image=B300_IMAGE,
    gpu="B300:1",
    timeout=3_600,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def trap_kimi_k3_schedule() -> None:
    """Let a missed publication on a readiness edge actually trap, on device.

    One GPU and no ranks, which is the whole reason this is a gate of its own
    rather than a file in the TP8 suite.
    """
    # Unbuffered and uncaptured: the injections print what the trapped launch
    # recorded, and that is the artifact. Captured output would only appear on
    # failure, which is the one case where the numbers are least interesting.
    completed = subprocess.run(
        ["python", "-m", "pytest", "-v", "-s", K3_TRAP_FILE],
        cwd=REMOTE_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=3_300,
    )
    print(completed.stdout, end="")
    _persist_k3_artifact("trap.log", completed.stdout)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, K3_TRAP_FILE, output=completed.stdout
        )


@app.function(
    image=B300_IMAGE,
    gpu="B300:8",
    timeout=14_400,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def stress_kimi_k3_schedule() -> None:
    """Verify every launch of a long rotating candidate run, not just the last.

    Untimed on purpose: it synchronizes the device and compares against the
    oracle after each launch, which is the only way a readiness edge that is
    usually satisfied gets caught. The graph replay test stays where it is and
    keeps doing what it is good at -- a thousand crossings of every counter --
    which is a different claim and a much cheaper one.
    """
    output = _run_kimi_k3_torchrun(
        ["-m", "pytest", "-v", "-x", K3_STRESS_FILE],
        timeout=14_100,
        attribute_ranks=True,
        capture=True,
    )
    _persist_k3_artifact(
        "stress.log",
        f"command: pytest -v -x {K3_STRESS_FILE}\n\n{output}",
    )


@app.function(
    image=B300_IMAGE,
    gpu="B300:8",
    timeout=14_400,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def verify_kimi_k3(
    expression: str = "",
    verbose: bool = True,
    files: str = "",
    exit_first: bool = True,
) -> None:
    """Run the whole Kimi K3 suite on all eight ranks and keep pytest's own log.

    ``exit_first`` stops at the first failure, which is what a gate wants: the
    suite takes eight B300s and a run that keeps going after a real failure
    spends them on output nobody reads. Turning it off is for the other case --
    a change that could plausibly have broken several tests at once, where one
    failure per round trip is the slow way to find out.
    """
    selection = ["-k", expression] if expression else []
    chosen = tuple(files.split(",")) if files else _k3_test_files()
    command = [
        "-m",
        "pytest",
        "-v" if verbose else "-q",
        *(["-x"] if exit_first else []),
        *chosen,
        *selection,
    ]

    # Eight ranks tee into one stream and the interleave breaks lines mid-token,
    # so the `-v` log cannot be read back to answer "did test X run". One
    # single-process collection gives a clean list of node ids to check the
    # inventory against; the count it reports is the count each rank then runs.
    def collect() -> str:
        collected = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q", *chosen, *selection],
            cwd=REMOTE_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=900,
        )
        return collected.stdout

    def persist(output: str, collected: str) -> None:
        # The artifact has to carry pytest's own summary lines, because the
        # report quotes the count and a bare "passed" cannot back that. A
        # failing run persists too, so the failure is recoverable from the
        # volume rather than only from the console.
        _persist_k3_artifact(
            "tests.log",
            f"command: pytest {' '.join(command[2:])}\n"
            f"files: {','.join(chosen)}\n"
            f"expression: {expression!r}\n\n"
            f"--- collected node ids (one process) ---\n{collected}\n"
            f"--- the run, 8 ranks teed into one stream ---\n{output}",
        )

    collected = collect()
    try:
        output = _run_kimi_k3_torchrun(
            command,
            timeout=14_100,
            attribute_ranks=verbose,
            capture=True,
        )
    except subprocess.CalledProcessError as error:
        persist(error.output or "", collected)
        raise
    persist(output, collected)


def _run_sanitizer_session(
    command: list[str],
    *,
    tool: str,
    budget: float,
    cwd: str = REMOTE_ROOT,
) -> tuple[int, str, bool]:
    """Run one sanitizer command as its own session, bounded by ``budget``.

    Returns the exit code, everything the tool wrote, and whether it was cut off.

    `subprocess.run(timeout=...)` is wrong here twice over, and this gate has
    been bitten by both. It kills only the process it started, and what it starts
    is `compute-sanitizer`, whose child is a torchrun whose children are eight
    ranks holding eight B300s -- so a timeout leaves the ranks running, and in a
    container that is about to be reused or torn down that is a fleet of devices
    nobody owns. And it buffers: nothing reaches the console until the call
    returns, so a run that is going to take eight hours looks identical to one
    that has hung in its first minute.

    So this borrows what the benchmark gates already use. The child leads its own
    session, `_stream_bounded` echoes each line as it arrives while holding the
    deadline, and `_end_session` signals the whole group -- TERM, a bounded wait,
    then KILL, unconditionally rather than as an escalation the launcher's own
    exit can cut short, because a rank inside a driver call will not return to run
    a handler.

    A cut-off run comes back with what it had, marked partial. It is never raised
    out of here: the caller has to reach `sanitizer_verdict` on this output for the
    gate to fail closed, and an exception escaping is how one eight-hour run
    persisted nothing at all and left the previous run's log looking current.
    """
    print(f"Launching: {' '.join(command)} on 8 x B300")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env={
            **os.environ,
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
    # No stall bound: a sanitizer legitimately goes quiet for a long time inside
    # one instrumented launch, so silence here is not evidence of anything. The
    # total budget is the only sound bound, and it is the gate's own.
    output, expired = _stream_bounded(
        process, deadline=started + budget, stall_timeout=None
    )
    if expired is None:
        left = max(1, int(started + budget - time.monotonic()))
        try:
            return process.wait(timeout=left), output, False
        except subprocess.TimeoutExpired:
            expired = "budget"
    _end_session(process)
    elapsed = int(time.monotonic() - started)
    note = (
        f"\n========= MoK: {tool} was still running after {elapsed}s "
        f"(budget {int(budget)}s) and its process group was ended; "
        f"the output above is partial\n"
    )
    print(note, end="")
    # 124 is what `timeout(1)` reports, which is the closest thing to a
    # convention for this and is outside every tool's own permitted set.
    return 124, output + note, True


@app.function(
    image=B300_SANITIZER_IMAGE,
    gpu="B300:8",
    timeout=SANITIZER_GATE_TIMEOUT,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def sanitize_kimi_k3_decode(
    tool: str = "memcheck",
    expression: str = "",
    files: str = "tests/test_kimi_k3_decode.py",
    artifact: str = "",
) -> dict[str, int | str]:
    """Run the decode step under one compute-sanitizer tool on all eight ranks.

    ``racecheck`` and ``memcheck`` are separate runs because the two tools
    cannot be combined, and both are slow enough that the selection is narrowed
    to the tests that actually drive the routed gate/up ring.

    ``files`` exists so the same invocation can be pointed at a narrower or
    wider suite: a finding is only attributable to a change if the same tool is
    clean without it, and the answer to that has to be measured on the same
    eight devices under the same tool rather than reasoned about.

    A gate that runs out of wall clock ends here rather than in Modal: the tool
    is given what is left of the function's timeout less the time it takes to
    write the artifact, so the partial output is persisted and the verdict says
    the run was cut off.
    """
    entered = time.monotonic()
    if tool not in ("memcheck", "racecheck", "synccheck", "initcheck"):
        raise ValueError(f"unknown compute-sanitizer tool {tool!r}")
    expression = expression or K3_SANITIZER_SELECTION.get(
        tool, "pinned_route_distributions"
    )
    command = [
        "compute-sanitizer",
        "--tool",
        tool,
        "--error-exitcode",
        "99",
        "--target-processes",
        "all",
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "-m",
        "pytest",
        "-q",
        "-x",
        *files.split(","),
        "-k",
        expression,
    ]
    # The budget is what is *left* of the function's, not the function's less a
    # constant. Modal's timeout is counted from the call, and everything before
    # this line -- the image's own startup, importing torch, resolving the
    # selection -- is already spent out of it. An eight-hour racecheck run was
    # lost to that difference: the subprocess was given 28,500 seconds starting
    # several minutes after the function's 28,800 began, so Modal reached its
    # timeout first and killed the container with nothing written.
    remaining = SANITIZER_GATE_TIMEOUT - (time.monotonic() - entered)
    budget = remaining - SANITIZER_TEARDOWN_SECONDS
    if budget <= 0:
        raise RuntimeError(
            f"{tool} has no time left to run in: {remaining:.0f}s of the "
            f"gate's {SANITIZER_GATE_TIMEOUT}s remain, and persisting the "
            f"artifact needs {SANITIZER_TEARDOWN_SECONDS}s of it"
        )
    exit_code, output, timed_out = _run_sanitizer_session(
        command, tool=tool, budget=budget
    )
    # A sanitizer run has two verdicts and reading one of them is how a clean
    # report gets published for a run that never finished: racecheck once
    # reported zero hazards for a step in which a rank had already segfaulted.
    # `sanitizer_verdict` is the conjunction of both, and it is a pure function
    # so `tests/test_kimi_k3_sanitizer.py` can hold it to captured runs on a CPU
    # rather than by provoking the condition on eight GPUs.
    verdict = sanitizer_verdict(tool, exit_code, output)
    # Not reprinted: `_run_sanitizer_session` echoed every line as it arrived,
    # which is the point of streaming it.
    for line in verdict.summary_lines():
        print(f"{tool} {line}")
    claim = K3_SANITIZER_CLAIMS[tool]
    # Persisted before the raise below, so a refused run is still recoverable
    # from the volume rather than only from this container's console.
    _persist_k3_artifact(
        artifact or f"{tool}.log",
        f"command: {' '.join(command)}\n"
        f"establishes: {claim}\n"
        + "\n".join(verdict.summary_lines())
        + f"\n{output}",
    )
    result = {
        "command": " ".join(command),
        "tool": tool,
        "establishes": claim,
        "exit_code": verdict.exit_code,
        "passed": verdict.passed,
        "reported_errors": verdict.reported_errors,
        "host_allowed_errors": verdict.host_allowed_errors,
        "device_errors": verdict.device_errors,
        "hazards": verdict.hazards,
        "rank_summaries": len(verdict.rank_summaries),
        "failures": list(verdict.failures),
        "timed_out": timed_out,
        "output": output,
    }
    if not verdict.passed:
        raise RuntimeError(
            f"{tool} did not produce evidence: "
            + "; ".join(verdict.failures)
        )
    return result


@app.function(
    image=B300_IMAGE,
    gpu="B300",
    timeout=3_600,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def sass_kimi_k3_decode() -> str:
    """Report the SM103 SASS evidence for the decode kernel, per instantiation.

    Resource usage and instruction-family counts for all six instantiations --
    both capacity paths of the dependency-local schedule under both compiled
    gate/up engines, and both capacity paths of the barrier schedule -- plus each
    engine's compiled ledger and the residency the driver measures for each. What
    the ring's arrival has to show is copy engine transfers where the old unit had
    scalar staging -- `UTMALDG` up, `LDG`/`STS` down -- with nothing spilled to
    local memory in any of them.

    Every label here is a kernel and an engine rather than a role. The roles have
    moved twice in this task and each move made the previous labels wrong in a way
    that read as correct: the barrier schedule was called "production" for a round
    after a decode step stopped launching it, and the reported ring geometry was
    the resident two-stage ring's for a round after the adaptive selector shipped.
    So the shipping engine names itself in `shipping_gate_up_engine`, its ledger is
    reported next to the baseline's, and the two figures that describe the ring
    production replaced say so in their own keys.
    """
    import re

    import torch

    from mok import _C

    extension = _C.__file__
    usage = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", extension],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # Both schedules are dumped, and each is labelled with its own name rather
    # than with a role. Calling the barrier schedule "production" and the
    # dependency-local one "candidate" was true while the promotion was pending
    # and has been false since: a decode step launches the dependency-local
    # kernel, and the barrier schedule is a benchmark fallback compiled against
    # the ring production replaced. A label that says which is which by role has
    # to be re-read every time the roles move, so these say which kernel.
    kernels = {
        "kimi_k3_decode_dependency_local_kernel": "dependency_local",
        "kimi_k3_decode_persistent_kernel": "barrier",
    }
    table = {
        name: line
        for name, line in re.findall(r"Function (\S+):\s*\n\s*(REG:.*)", usage)
        if any(kernel in name for kernel in kernels)
    }
    # `ILb0E`/`ILb1E` is the mangled capacity flag: the core path and the tcgen05
    # path are separate instantiations of one template. Matched without the
    # trailing `E` of the barrier schedule's mangling, because the
    # dependency-local one carries a second template argument and so does not
    # have one there.
    paths = {"ILb0E": "core", "ILb1E": "tensor"}
    # And `Li2E`/`Li3E` is the gate/up engine, which the dependency-local
    # schedule carries as its second template argument. Without it both engines
    # land on one label and overwrite each other, which would report whichever
    # the dump happened to emit last -- and the no-spill claim rests on the
    # instantiation, not on the schedule. Engine 2 is the adaptive selector that
    # ships; engine 3 is the resident ring it replaced, kept compiled as the
    # A/B's baseline and as what the parity tests measure against.
    engines = {
        "Li2E": "adaptive",
        "Li3E": "resident_baseline",
    }
    families = ("UTMALDG", "UTCQMMA", "UTCHMMA", "LDG", "STS", "LDL", "STL", "LDTM")
    # The ledger of each compiled engine, read off the compiled constants. This
    # is the figure the gate exists to report and it was reporting the wrong one:
    # `_kimi_k3_fused_w13_geometry` describes the resident two-stage ring, so the
    # artifact said 216,064 bytes over two stages while the kernel a decode step
    # launches asks for 228,352 over three. Both engines are listed, and the one
    # that ships is named as such rather than left to be inferred from an id.
    ledger_fields = (
        "dynamic_shared_bytes_as_launched",
        "staging_bytes_before_allocator_grain",
        "weight_stages",
        "activation_slabs_held",
        "live_accumulators",
        "activation_gathers_per_pass",
    )
    ledgers = {
        name: dict(
            zip(
                ledger_fields,
                (
                    int(value)
                    for value in _C._kimi_k3_decode_gate_up_engine_ledger(
                        engine
                    )
                ),
                strict=True,
            )
        )
        for engine, name in ((2, "adaptive"), (3, "resident_baseline"))
    }
    report: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0),
        "extension": extension,
        # What ships, first, because it is what every other figure here is about.
        "shipping_gate_up_engine": "adaptive",
        "gate_up_engine_ledger": ledgers,
        # And the geometry the shipping engine's wide arm shares with the ring it
        # replaced -- slab count, K, transaction sizes, the opt-in ceiling. The
        # stage count and byte totals in it are the resident ring's, which is why
        # it is named for that rather than presented as the ring that launches.
        "resident_ring_geometry": dict(_C._kimi_k3_fused_w13_geometry()),
        "resident_ring_shared_footprint": dict(
            zip(
                ("measured_bytes", "dynamic_block_offset", "launch_bytes"),
                _C._kimi_k3_fused_w13_shared_footprint(),
                strict=True,
            )
        ),
        # `_kimi_k3_decode_grid_shape` is the barrier schedule's launch
        # configuration, so its shared figure is that schedule's too. The CTA and
        # thread counts are shared by both schedules; the bytes are not.
        "barrier_schedule_grid_shape": dict(
            zip(
                ("ctas", "threads", "dynamic_shared_bytes"),
                _C._kimi_k3_decode_grid_shape(),
                strict=True,
            )
        ),
        "resident_blocks_per_sm": {
            "barrier_core": _C._kimi_k3_decode_resident_blocks_per_sm(False),
            "barrier_tensor": _C._kimi_k3_decode_resident_blocks_per_sm(True),
            **{
                f"dependency_local_{path_name}_{label}": (
                    _C._kimi_k3_decode_schedule_resident_blocks_per_sm(
                        tensor_path, engine
                    )
                )
                for engine, label in (
                    (2, "adaptive"),
                    (3, "resident_baseline"),
                )
                for tensor_path, path_name in ((False, "core"), (True, "tensor"))
            },
        },
        "instantiations": {},
    }
    for mangled, line in sorted(table.items()):
        schedule = next(
            (name for key, name in kernels.items() if key in mangled), "unknown"
        )
        path = next(
            (name for key, name in paths.items() if key in mangled), "unknown"
        )
        engine = next(
            (name for key, name in engines.items() if key in mangled), ""
        )
        label = "_".join(part for part in (schedule, path, engine) if part)
        dump = subprocess.run(
            ["cuobjdump", "-sass", "-fun", mangled, extension],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        counts: dict[str, int] = {}
        for sass_line in dump.splitlines():
            match = re.match(
                r"\s+/\*[0-9a-f]{4,}\*/\s+(@!?\S+\s+)?([A-Z][A-Z0-9_.]*)",
                sass_line,
            )
            if match:
                family = match.group(2).split(".")[0]
                counts[family] = counts.get(family, 0) + 1
        report["instantiations"][label] = {
            "symbol": mangled,
            "resources": {
                key: int(value)
                for key, value in re.findall(r"([A-Z]+):(\d+)", line)
            },
            "instruction_counts": {
                family: counts.get(family, 0) for family in families
            },
            "total_instructions": sum(counts.values()),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    _persist_k3_artifact("sass.json", rendered)
    return rendered


@app.function(
    image=B300_IMAGE,
    gpu="B300:8",
    timeout=86_400,
    volumes={K3_ARTIFACTS: K3_VOLUME},
)
def bench_kimi_k3_decode_persisted(git_sha: str) -> bytes:
    """Run the decode benchmark and put its archive in the volume as well.

    The same gate `bench_kimi_k3_decode` runs. It exists separately because the
    tables take hours and their only delivery path is the client connection,
    which is the one thing no amount of retrying inside the container can
    protect; this writes the archive to the volume before returning it.
    """
    archive = bench_kimi_k3_decode.local(git_sha)
    _persist_k3_artifact("decode_benchmark.tar", archive)
    return archive


#: The gates a default verification run takes.
#:
#: `racecheck` is one of them again, and what changed is the image rather than
#: the schedule. It had completed cleanly against the barrier schedule -- 3
#: passed in 1241s, 0 hazards -- and against the dependency-local schedule the
#: target trapped: `cudaErrorLaunchFailure` on four of eight ranks, which is
#: what a bounded wait giving up looks like from the host. The tool slows the
#: step by some four hundred times, the budget was a fixed fifteen seconds of
#: device clocks, and this schedule's longest single wait spans a third of the
#: step where a barrier's spans a sixth. So the gate failed closed, correctly,
#: on a fact about the watchdog rather than about races -- and it would have
#: failed that way on every run, which trains a reader to skip it.
#:
#: `B300_SANITIZER_IMAGE` compiles the bounded spins with a budget 64 times
#: wider, which is sixteen minutes of B300 clocks against a slowdown measured in
#: hundreds. Nothing else about that image differs, the widening is compile-time
#: so production's binary is untouched, and
#: `test_the_wait_budget_is_the_one_this_image_declares` requires the compiled
#: scale to equal the one the image declares -- so a production image cannot
#: carry the wider budget and a sanitizer image cannot quietly lose it.
K3_GATES = (
    "tests",
    "stress",
    "trap",
    "sass",
    "benchmark",
    # All three tools against the schedule that ships, which since promotion is
    # the one the decode suite takes by default. The three `schedule-*` gates
    # that pointed the same tools at the guarded candidate are gone: they would
    # now be the same runs under another name.
    "memcheck",
    "synccheck",
    "racecheck",
)


@app.local_entrypoint()
def verify(
    git_sha: str,
    output_dir: str = "kimi_k3_verification_b300",
    gates: str = ",".join(K3_GATES),
    spawn: bool = False,
) -> None:
    """Run the named Kimi K3 B300 gates and persist each one's artifact.

    Correctness, the per-launch replay stress, the trap injection, all three
    sanitizer tools against the schedule that ships, the SASS evidence for both
    schedules' instantiations, and the saturated benchmark. Run one after
    another rather than all at once: four of them want all eight B300s, and a
    run that asked for thirty-two would be queued behind itself rather than
    finishing sooner. A gate that fails is recorded and the rest still run, so
    one failure does not cost the other evidence.

    ``gates`` selects a subset, which is what makes the evidence recoverable:
    the client connection that carries a gate's result is the one thing here
    that no amount of retrying inside the container can protect, so each gate
    can be re-run on its own in a fresh client process without redoing the
    ones that already landed.

    ``spawn`` goes further and does not wait at all. Run it under `--detach`
    and the gate finishes on its own, writing its artifact into the
    `mok-kimi-k3-decode` volume; `modal volume get` collects it afterwards.
    That is the only way to run the saturated benchmark and racecheck, whose
    hours outlast any client connection.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    requested = tuple(name.strip() for name in gates.split(",") if name.strip())
    # Every defined gate, not only the default ones: a tool left out of the
    # default set because it cannot finish is still a gate somebody re-runs to
    # check whether that is still true.
    definable = set(K3_GATES) | set(K3_SANITIZER_GATES)
    unknown = sorted(set(requested) - definable)
    if unknown:
        raise SystemExit(f"unknown Kimi K3 gates: {unknown}")
    functions = {
        "tests": (verify_kimi_k3, (), {}),
        "stress": (stress_kimi_k3_schedule, (), {}),
        "trap": (trap_kimi_k3_schedule, (), {}),
        "sass": (sass_kimi_k3_decode, (), {}),
        "benchmark": (bench_kimi_k3_decode_persisted, (git_sha,), {}),
        **{
            name: (
                sanitize_kimi_k3_decode,
                (),
                {
                    "tool": tool,
                    "files": files,
                    "expression": expression,
                    # One artifact per gate, so the candidate's racecheck does
                    # not overwrite production's on the volume.
                    "artifact": f"{name}.log",
                },
            )
            for name, (tool, files, expression) in K3_SANITIZER_GATES.items()
        },
    }
    if spawn:
        for name in requested:
            function, args, keywords = functions[name]
            handle = function.spawn(*args, **keywords)
            print(f"{name}: spawned {handle.object_id}")
        print(
            "collect with: modal volume get mok-kimi-k3-decode "
            f"'**' {output_dir}"
        )
        return
    runners = {
        name: (lambda function=function, args=args, keywords=keywords: (
            function.remote(*args, **keywords)
        ))
        for name, (function, args, keywords) in functions.items()
    }
    failures = []
    for name in requested:
        gate = runners[name]
        try:
            result = gate()
        except Exception as error:  # noqa: BLE001 - every gate must be attempted
            (root / f"{name}.error.txt").write_text(str(error), encoding="utf-8")
            failures.append(name)
            print(f"{name}: FAILED ({type(error).__name__})")
            continue
        (root / f"{name}.error.txt").unlink(missing_ok=True)
        if name in K3_SANITIZER_GATES:
            (root / f"{name}.log").write_text(
                f"command: {result['command']}\n"
                f"establishes: {result['establishes']}\n"
                f"verdict: {'passed' if result['passed'] else 'FAILED'}\n"
                f"exit_code: {result['exit_code']}\n"
                f"reported_errors: {result['reported_errors']}\n"
                f"host_allowed_errors: {result['host_allowed_errors']}\n"
                f"device_errors: {result['device_errors']}\n"
                f"hazards: {result['hazards']}\n"
                f"rank_summaries: {result['rank_summaries']}\n"
                + "".join(
                    f"failure: {reason}\n" for reason in result["failures"]
                )
                + result["output"],
                encoding="utf-8",
            )
            print(
                f"{name}: {result['device_errors']} device errors, "
                f"{result['hazards']} hazards "
                f"({result['reported_errors']} reported, "
                f"{result['host_allowed_errors']} host-allowed), "
                f"{result['rank_summaries']} rank summaries"
            )
            # `sanitize_kimi_k3_decode` refuses a run that produced no evidence
            # before it returns, so reaching here already means the conjunction
            # held. Gating on it again is what keeps that true if the remote
            # ever starts returning a verdict instead of raising on one.
            if not result["passed"]:
                failures.append(name)
        elif name == "sass":
            (root / "sass.json").write_text(result, encoding="utf-8")
        elif name == "benchmark":
            (root / "decode_benchmark.tar").write_bytes(result)
        else:
            (root / f"{name}.log").write_text("passed\n", encoding="utf-8")
        print(f"{name}: done")
    print(f"artifacts: {sorted(path.name for path in root.iterdir())}")
    if failures:
        raise SystemExit(f"Kimi K3 gates failed: {sorted(failures)}")
