"""A missed publication on a schedule edge traps at that edge's named site.

The candidate has ten bounded readiness edges where the production kernel has
three, and several of them spin on counters that look alike from the outside.
So the diagnostic the trap records is the only thing that can say which
publication never arrived, and it is worth an actual trap rather than only a
source contract.

Two shapes of injection live here. The parametrized one stalls a single edge and
checks that the site it names is its own. The last one stalls every edge at once,
from sixty CTAs, and checks the property that only exists under a race: that the
slot and the code that got published are one waiter's rather than two.

A trap corrupts the context it ran in, so the injection runs as its own process
through ``kimi_k3_schedule_trap_probe.py``. It needs one GPU, it burns the
fifteen-second wait budget on purpose, and it must not share the node with a
live process group: a device-side trap does not stay inside the context that
took it, and the peers of an initialized NCCL group observe it as
``cudaErrorLaunchFailure`` on their own devices. That happens often enough to
kill a TP8 session and rarely enough to look flaky, so this file refuses to run
under ``torchrun`` at all and is driven as its own single-process gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mok import _C


_REPO_ROOT = Path(__file__).resolve().parents[1]

# One edge per shape the injection can take. Three runs and forty-five seconds,
# plus one more of each for the concurrent race below:
#
#   * ``gate_up_assignment`` is the plain case -- an appended counter, device
#     scope, a whole-queue target.
#   * ``tail_publish`` is the cross-rank case. The probe dispatches through the
#     same ``wait_edge`` template the coordinator does, so this run is the one
#     that actually executes ``wait_for_schedule_count_system``; before the
#     table carried the scope, a probe could only have spun on it at whatever
#     scope the probe itself picked, which was device.
#   * ``routed_down_gate_up`` is the case whose counter is not in the appended
#     region at all and whose slot is indexed by the expert, so it is the one
#     that would break if the diagnostic dropped either.
_INJECTED_EDGES = (
    "gate_up_assignment",
    "tail_publish",
    "routed_down_gate_up",
)


@pytest.fixture(scope="module")
def trap_device() -> torch.device:
    """One B300, with no process group anywhere on the node."""
    if "RANK" in os.environ:
        pytest.skip("a trap must not share the node with a live NCCL group")
    if not torch.cuda.is_available():
        pytest.skip("the trap injection needs a CUDA device")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (10, 3):
        pytest.skip("the trap injection needs an SM103 B300")
    return device


def _probe(*arguments: str, device: torch.device) -> dict[str, object]:
    """Run one injection as its own process and parse its one JSON line.

    Fail closed in both directions: a nonzero exit is the subprocess reporting
    that it could not run the injection at all, and it fails the test rather
    than being read as "nothing trapped". The stderr of a trapped launch is
    noisy and irrelevant, so it is only quoted when the status is bad.
    """
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(device.index)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.kimi_k3_schedule_trap_probe",
            *arguments,
        ],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    line = [
        text for text in completed.stdout.splitlines() if text.startswith("{")
    ]
    assert line, completed.stdout[-4000:]
    return json.loads(line[-1])


@pytest.mark.parametrize("edge", _INJECTED_EDGES)
def test_a_missed_publication_traps_at_its_own_named_site(
    trap_device: torch.device,
    edge: str,
) -> None:
    """The counter alone cannot name the site, so both halves are checked.

    The probe leaves the edge's counter at zero and takes the edge through
    ``wait_edge``, which is exactly what a producer that failed to publish
    looks like to its consumer. The wait must run out rather than hang, the
    launch must fail rather than return, and the two diagnostics must name
    *this* edge: its own code in the caller-visible flag, and its own slot in
    the timeout word the phase counters share with it.
    """
    rows = [tuple(row) for row in _C._kimi_k3_decode_schedule_edges()]
    index = [row[0] for row in rows].index(edge)
    row = rows[index]
    counter, code, space, scope, indexed = (
        int(row[3]), int(row[4]), int(row[5]), int(row[6]), bool(row[9])
    )

    report = _probe(edge, device=trap_device)
    # The gate's artifact is what the trapped launch recorded, so it is printed
    # rather than only asserted on.
    print(f"\ninjected {edge}: {json.dumps(report, sort_keys=True)}")
    assert report["launch_failed"], report
    assert report["recorded_code"] == code == report["expected_code"], report
    assert report["recorded_slot"] == report["expected_slot"], report
    assert report["scope"] == scope, report

    unit = int(report["unit"])
    if space == 0:  # types.cuh: kScheduleCounterInRegion
        # Offset past the phase region, so no phase counter's index can be
        # mistaken for a schedule counter's in the one word both write.
        assert report["recorded_slot"] >= 128, report
        assert report["recorded_slot"] == 128 + counter + (
            unit if indexed else 0
        ), report
    else:
        # The one edge outside the region reports the phase slot the production
        # wait on the same counter reports, and does not carry the unit.
        assert report["recorded_slot"] == counter < 128, report

    # Even with one waiter, the record is claimed rather than merely written.
    assert report["claiming_block"] == 0, report


# Units of every edge the concurrent injection stalls. Ten edges, six units
# each: sixty CTAs, all resident, all spinning on a counter nobody publishes,
# all giving up against the same fifteen-second budget. Six is the largest the
# extension accepts, because the edge whose counter is indexed inside the
# appended region has six of them and reaching past that would make the probe an
# out-of-band read instead of a stalled wait.
_CONCURRENT_UNITS_PER_EDGE = 6


def test_a_race_to_report_publishes_one_waiters_slot_and_code(
    trap_device: torch.device,
) -> None:
    """Sixty waiters give up together and one pair comes out.

    This is the case the single-edge injections cannot reach. Ten edges spin
    against one clock budget, so a producer that never publishes does not stall
    one consumer -- it stalls every consumer queued behind it, and they arrive
    at the diagnostic together. Two independent exchanges would then pair one
    waiter's code with another waiter's slot, which is worse than no diagnostic:
    it names a wait that did not time out and sends the reader to a publication
    that did arrive.

    So what is checked here is not "a plausible pair" but "*this* waiter's
    pair". The claim word names the CTA that won the claim, the CTA index gives
    the edge and the unit it was spinning on, and the code and the slot must
    both be that ``(edge, unit)``'s -- not merely some row of the table's. A
    torn pair fails, a stale pair fails, and zero in any of the three words
    fails.
    """
    rows = [tuple(row) for row in _C._kimi_k3_decode_schedule_edges()]
    report = _probe(
        "--concurrent",
        str(_CONCURRENT_UNITS_PER_EDGE),
        device=trap_device,
    )

    print(f"\nconcurrent race: {json.dumps(report, sort_keys=True)}")
    assert report["launch_failed"], report
    assert report["edge_count"] == len(rows), report
    assert report["blocks"] == len(rows) * _CONCURRENT_UNITS_PER_EDGE, report

    # Nothing may be left at zero: an unclaimed word, a zero code, or a slot
    # that no wait writes all mean the record was never published.
    claimed = int(report["claiming_block"])
    assert claimed >= 0, report
    assert claimed < int(report["blocks"]), report
    assert int(report["recorded_code"]) != 0, report

    # The kernel's own mapping from CTA to work: edge-major, so consecutive
    # CTAs take different edges and every edge is stalled at several units.
    edge_index = claimed % len(rows)
    unit = claimed // len(rows)
    expected_code = int(rows[edge_index][4])
    expected_slot = int(
        _C._kimi_k3_decode_schedule_edge_diagnostics(unit)[edge_index]
    )

    assert int(report["recorded_code"]) == expected_code, {
        "claimed_block": claimed,
        "edge": rows[edge_index][0],
        "unit": unit,
        **report,
    }
    assert int(report["recorded_slot"]) == expected_slot, {
        "claimed_block": claimed,
        "edge": rows[edge_index][0],
        "unit": unit,
        **report,
    }

    # A code from one site next to a slot from another would still satisfy
    # "both are in the table", so the pair is checked against the table too:
    # exactly one row carries this code, and it is the row the claim names.
    matching = [index for index, row in enumerate(rows)
                if int(row[4]) == int(report["recorded_code"])]
    assert matching == [edge_index], report
    print(
        f"the claim named CTA {claimed}, which was waiting on "
        f"{rows[edge_index][0]} unit {unit}; it published code "
        f"{report['recorded_code']} beside slot {report['recorded_slot']}"
    )
