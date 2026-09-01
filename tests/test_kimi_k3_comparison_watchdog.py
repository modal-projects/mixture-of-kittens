"""CPU contracts for the bounded watchdog over the framework comparisons.

One SGLang comparison run produced no output at all between torchrun's startup
and the orchestration being interrupted, on the same commit and the same pinned
image digest that other runs completed on. Nothing was wrong with the tree, and
nothing about the run said so: a comparison writes its artifacts only after every
shape has been measured, so a deadlocked run and a slow one look identical from
outside.

Two things were added for that, and this file is what holds them.

**The instrument is silence, not duration.** The driver now names every stage it
enters, so a gap between lines is a run that has stopped rather than a run that
is working. A total-duration bound would have to be set so far above a healthy
comparison -- which builds a framework's whole MoE layer and then measures 1,500
replays per backend per shape -- that it could not tell the two apart inside a
useful window.

**A stalled attempt has to be ended, not abandoned.** `subprocess`'s own timeout
kills the launcher, and a torchrun's ranks are its children: a rank blocked in a
collective would outlive the launcher, keep its GPU, and make the retry fail on a
busy device. So the child is started in its own session and the group is
signalled.

Everything here runs on a CPU against ordinary processes, because the property
being checked is the orchestration's, not the kernel's.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.compare_kimi_k3_frameworks import comparison_artifact_files
from modal_bench import _end_session, _stream_bounded
from modal_frameworks import (
    COMPARISON_ATTEMPTS,
    COMPARISON_ATTEMPT_TIMEOUT,
    COMPARISON_STALL_TIMEOUT,
    _run_framework_comparison,
)


#: A full-length SHA, because the entrypoint refuses anything else.
GIT_SHA = "0" * 40


def _spawn(script: str) -> "subprocess.Popen[str]":
    """Start one Python child in its own session, as the orchestration does."""
    return subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def test_output_that_arrives_and_then_stops_is_read_as_a_stall() -> None:
    """The deadlock's signature: lines, then nothing, and no exit."""
    process = _spawn(
        "import time\n"
        "print('PROGRESS one')\n"
        "time.sleep(600)\n"
    )
    try:
        text, expired = _stream_bounded(
            process, deadline=time.monotonic() + 600, stall_timeout=3
        )
    finally:
        _end_session(process)

    assert expired == "silence"
    assert text == "PROGRESS one\n"
    # The bound has to fire while the child is still alive, or it is measuring
    # the child's exit rather than its silence.
    assert process.returncode is not None


def test_a_child_that_keeps_talking_is_not_stalled() -> None:
    """A slow run that says where it is must survive its own stall bound."""
    process = _spawn(
        "import time\n"
        "for index in range(6):\n"
        "    print(f'PROGRESS {index}')\n"
        "    time.sleep(0.4)\n"
    )
    try:
        text, expired = _stream_bounded(
            process, deadline=time.monotonic() + 600, stall_timeout=3
        )
    finally:
        _end_session(process)

    assert expired is None
    assert text.splitlines() == [f"PROGRESS {index}" for index in range(6)]


def test_the_total_allowance_still_bounds_a_chatty_run() -> None:
    """Silence is the useful bound, but it is not the only one."""
    process = _spawn(
        "import time\n"
        "while True:\n"
        "    print('PROGRESS')\n"
        "    time.sleep(0.05)\n"
    )
    try:
        _, expired = _stream_bounded(
            process, deadline=time.monotonic() + 1.0, stall_timeout=600
        )
    finally:
        _end_session(process)

    assert expired == "budget"


def test_a_stall_with_no_bound_at_all_waits(tmp_path: Path) -> None:
    """Gates that print nothing for minutes pass no bound and must not be cut."""
    process = _spawn("print('done')\n")
    try:
        text, expired = _stream_bounded(
            process, deadline=time.monotonic() + 600, stall_timeout=None
        )
    finally:
        _end_session(process)

    assert (text, expired) == ("done\n", None)


def test_ending_a_session_takes_the_ranks_with_it(tmp_path: Path) -> None:
    """The launcher's children are the point; killing the launcher is not.

    The child here stands in for torchrun: it forks a grandchild that ignores
    SIGTERM the way a rank inside a collective effectively does, and writes its
    pid where the test can find it.
    """
    marker = tmp_path / "grandchild.pid"
    process = _spawn(
        "import os, signal, subprocess, sys, time\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-u', '-c',\n"
        "    'import signal, time\\n'\n"
        "    'signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n'\n"
        "    'time.sleep(600)\\n',\n"
        "])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "print('PROGRESS spawned')\n"
        "time.sleep(600)\n"
    )
    text, expired = _stream_bounded(
        process, deadline=time.monotonic() + 60, stall_timeout=3
    )
    assert (text, expired) == ("PROGRESS spawned\n", "silence")
    grandchild = int(marker.read_text())

    _end_session(process)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail(f"grandchild {grandchild} outlived the session")


def test_a_comparison_that_goes_quiet_once_is_retried(monkeypatch) -> None:
    """The transient hang costs an attempt, not the run.

    The fake torchrun stalls on its first call and produces the archive on its
    second, which is what the SGLang run did when it was repeated. The retry has
    to start from an empty output directory, so the fake writes only the files
    the artifact check expects and the check is what proves the first attempt's
    leftovers are gone.
    """
    calls: list[int] = []

    def torchrun(arguments: list[str], **keywords: object) -> str:
        calls.append(len(calls) + 1)
        output_dir = Path(arguments[arguments.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            (output_dir / "half_written.json").write_text("{}")
            raise subprocess.TimeoutExpired(["torchrun"], 2_700, output="")
        for name in comparison_artifact_files(["block8"]):
            (output_dir / name).write_text("{}")
        return ""

    monkeypatch.setattr("modal_frameworks._run_kimi_k3_torchrun", torchrun)
    archive = _run_framework_comparison(
        "vllm",
        GIT_SHA,
        warmup_count=1,
        sample_count=1,
        modes="block8",
        tokens="",
    )

    assert calls == [1, 2]
    assert archive


def test_a_comparison_that_never_talks_fails(monkeypatch) -> None:
    """Two hangs are a finding, so the second one is raised rather than retried."""
    calls: list[int] = []

    def torchrun(arguments: list[str], **keywords: object) -> str:
        calls.append(len(calls) + 1)
        raise subprocess.TimeoutExpired(["torchrun"], 2_700, output="")

    monkeypatch.setattr("modal_frameworks._run_kimi_k3_torchrun", torchrun)
    with pytest.raises(subprocess.TimeoutExpired):
        _run_framework_comparison(
            "vllm",
            GIT_SHA,
            warmup_count=1,
            sample_count=1,
            modes="block8",
            tokens="",
        )

    assert calls == [1, 2]


def test_a_comparison_that_fails_is_not_retried(monkeypatch) -> None:
    """A gate failure is the answer. Repeating it would only bury it."""
    calls: list[int] = []

    def torchrun(arguments: list[str], **keywords: object) -> str:
        calls.append(len(calls) + 1)
        raise subprocess.CalledProcessError(1, ["torchrun"], output="boom")

    monkeypatch.setattr("modal_frameworks._run_kimi_k3_torchrun", torchrun)
    with pytest.raises(subprocess.CalledProcessError):
        _run_framework_comparison(
            "vllm",
            GIT_SHA,
            warmup_count=1,
            sample_count=1,
            modes="block8",
            tokens="",
        )

    assert calls == [1]


def test_the_bounds_leave_room_for_the_retry_inside_the_function() -> None:
    """A watchdog that cannot fire twice inside the function is not a watchdog."""
    # `compare_vllm` and `compare_sglang` are both declared with this.
    function_timeout = 86_400
    assert COMPARISON_ATTEMPTS * COMPARISON_ATTEMPT_TIMEOUT < function_timeout
    assert COMPARISON_STALL_TIMEOUT < COMPARISON_ATTEMPT_TIMEOUT


def test_the_driver_names_every_stage_the_watchdog_waits_through() -> None:
    """Silence only means a hang if a healthy run is never silent that long.

    The bound is 45 minutes between lines, and the longest gap a healthy run has
    is one shape: build a route pool, check it against the native layer and the
    oracle, then measure 1,500 replays on each backend. So each of those has to
    be announced, and this is the check that the announcements are still there.
    """
    source = Path("benchmarks/compare_kimi_k3_frameworks.py").read_text()
    for stage in ("distributed", "adapter", "router_bound", "shape", "measuring"):
        assert f'_progress(rank, "{stage}"' in source, stage
    assert "def _progress(" in source

    orchestration = Path("modal_frameworks.py").read_text()
    assert "stall_timeout=COMPARISON_STALL_TIMEOUT" in orchestration
