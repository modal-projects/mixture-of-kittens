"""Inject missed publications on schedule edges and report what trapped.

A trapped launch ends as ``cudaErrorLaunchFailure`` and corrupts its context, so
this cannot run inside the suite that also has to launch a real decode step: it
runs as its own process and prints one JSON line, which
``test_kimi_k3_dependency_schedule_trap.py`` reads.

Both diagnostics ``record_timeout_and_trap`` writes are placed in mapped host
memory, which is what makes them readable after the context is gone. That is
also why the subject is
``kimi_k3_decode::persistent::schedule::schedule_wait_probe_kernel`` rather than
the candidate itself: a real decode workspace is device memory, so a real
launch's trap tells the host only that it failed.

The probe takes the *edge*, not a counter and a code, and dispatches through the
same ``wait_edge<Edge>`` template the kernel uses. That is what makes the
injection an injection on the real wait: the counter, the target, the diagnostic
slot, the code, and the acquire scope all come from the one table row, so
``tail_publish`` is spun on at system scope here exactly as the coordinator
spins on it.

The ``--concurrent`` mode injects on *every* edge at once, one CTA per
``(edge, unit)`` pair, which is the only way to exercise the part of the
publication protocol that does not exist when a single waiter gives up: sixty
waiters race for one claim word, and exactly one of them may end up owning both
published words. This mode reports the raw three words -- the claim, the code,
and the slot -- and leaves every derivation to the test, so that what is checked
is the kernel's arithmetic rather than this script's.

Usage: ``python -m tests.kimi_k3_schedule_trap_probe <edge name>`` or
``python -m tests.kimi_k3_schedule_trap_probe --concurrent <units per edge>``.
Each run burns the fifteen-second wait budget once, by design.
"""

from __future__ import annotations

import json
import os
import sys

import torch

from mok import _C


SCRATCH_BYTES = 8_111_872

# Read from the extension rather than spelled here, so a slot that moves in
# `types.cuh` cannot leave this reading the wrong word and reporting a zero.
(
    TIMEOUT_PHASE,
    _GRID_GENERATION,
    _ACTIVATION_ARRIVALS,
    _ACTIVE_EXPERT_UNITS,
    TIMEOUT_CLAIM,
) = _C._kimi_k3_decode_timeout_metadata()

# The expert whose readiness the one indexed injection stalls on. Nonzero so
# that a diagnostic slot computed without the unit would be a different number.
INJECTED_UNIT = 5


def _edges() -> list[tuple]:
    return [tuple(edge) for edge in _C._kimi_k3_decode_schedule_edges()]


def _diagnostics(scratch: torch.Tensor, error_flag: torch.Tensor) -> dict:
    """The three words a trapped launch leaves in mapped host memory."""
    phase = scratch[:512].view(torch.int32)
    claim = int(phase[TIMEOUT_CLAIM].item())
    return {
        "recorded_code": int(error_flag[0].item()),
        "recorded_slot": int(phase[TIMEOUT_PHASE].item()),
        "claim": claim,
        "claiming_block": int(_C._kimi_k3_decode_timeout_claim(claim)),
    }


def _pinned() -> tuple[torch.Tensor, torch.Tensor]:
    torch.cuda.init()
    # Mapped host memory: the diagnostics have to outlive the context the trap
    # takes down, and nothing in device memory does.
    return (
        torch.zeros(SCRATCH_BYTES, dtype=torch.uint8).pin_memory(),
        torch.zeros(1, dtype=torch.int32).pin_memory(),
    )


def _synchronize() -> str:
    try:
        torch.cuda.synchronize()
    except RuntimeError as error:
        return str(error)
    return ""


def main(name: str) -> int:
    edges = {edge[0]: (index, edge) for index, edge in enumerate(_edges())}
    if name not in edges:
        print(json.dumps({"error": f"unknown edge {name}",
                          "edges": sorted(edges)}))
        return 2
    index, edge = edges[name]
    code = int(edge[4])
    # Predicted from the extension's own function rather than from a second
    # copy of the offset arithmetic here.
    slot = int(
        _C._kimi_k3_decode_schedule_edge_diagnostics(INJECTED_UNIT)[index]
    )

    scratch, error_flag = _pinned()

    # Every counter is left at zero and the supplied target is one, so the
    # publication the edge is waiting for never arrives and the bounded wait
    # runs out. The table's own target wins on the four static edges, which is
    # the point: the probe cannot lower the bar the kernel waits at.
    _C._kimi_k3_decode_schedule_wait_probe(
        scratch, error_flag, index, INJECTED_UNIT, 1
    )
    failure = _synchronize()

    print(
        json.dumps(
            {
                "mode": "single",
                "edge": name,
                "edge_index": index,
                "unit": INJECTED_UNIT,
                "scope": int(edge[6]),
                "expected_code": code,
                "expected_slot": slot,
                **_diagnostics(scratch, error_flag),
                "launch_failed": bool(failure),
                "failure": failure[:200],
            }
        )
    )
    return 0


def concurrent(units_per_edge: int) -> int:
    """Stall every edge at several units apiece and report the raw words.

    The grid is one CTA per ``(edge, unit)`` pair. Every one of them spins on a
    counter nobody publishes, they all start within microseconds of each other,
    and they share the fifteen-second budget, so they arrive at the publication
    together. What the test then has to be able to say is that the pair which
    got published belongs to one of them and to that one only.
    """
    edges = _edges()
    scratch, error_flag = _pinned()
    _C._kimi_k3_decode_schedule_wait_probe_concurrent(
        scratch, error_flag, units_per_edge, 1
    )
    failure = _synchronize()

    print(
        json.dumps(
            {
                "mode": "concurrent",
                "units_per_edge": units_per_edge,
                "edge_count": len(edges),
                "blocks": len(edges) * units_per_edge,
                **_diagnostics(scratch, error_flag),
                "launch_failed": bool(failure),
                "failure": failure[:200],
            }
        )
    )
    return 0


def _main(argv: list[str]) -> int:
    if argv[:1] == ["--concurrent"]:
        return concurrent(int(argv[1]) if len(argv) > 1 else 1)
    return main(argv[0] if argv else "")


if __name__ == "__main__":
    status = _main(sys.argv[1:])
    sys.stdout.flush()
    # A trapped launch leaves the context unusable, and CUDA's teardown at
    # interpreter exit is entitled to abort on it. The report is already on
    # stdout, so the exit status has to come from here rather than from a
    # runtime that has nothing left to shut down cleanly.
    os._exit(status)
