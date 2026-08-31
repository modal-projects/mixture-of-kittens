"""The prepared fused-W13 routed gate/up layout, and what production rests on it.

Two things are covered here and they are covered for different reasons.

The transform in :mod:`mok.kimi_k3_w13` is a permutation of a routed expert's
gate and up MXFP4 bytes into the order the decode kernel's gate/up unit reads
them. Because it is a permutation, the strongest statement available is that it
loses nothing: every packed byte and every E8M0 scale byte comes back under
inversion, for every expert, every one of the six tasks, every one of the seven
K panels, every FP4 nibble, and every E8M0 code including the reserved and
boundary ones. That is what ``tests/kimi_k3_w13_contract.py`` asserts, and it
needs no GPU -- so it is run here as its own process, behind the same extension
stub ``tests/test_kimi_k3_api.py`` uses, and nothing in this file imports
``mok`` at collection. On a machine with no compiled ``mok._C`` the whole file
still collects and every transform check still runs.

The rest is the production contract. The prepared weights carry the fused pair
*instead of* the two per-projection pairs, not beside them, and the public
operator, its schema, its fake, and the C++ entrypoint all have to name the same
two tensors. The shape half of that is in the contract module with the
transform; the half below is read straight out of the sources, because a claim
about what the kernel can reach is a claim about the text of the kernel.

``tests/test_kimi_k3_w13_layout.py`` is the other side of this file: what the
device does with the layout, measured on SM103.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from .kimi_k3_w13_contract import CHECKS, RESULT_MARKER

_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "kimi_k3_decode"
_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The transform, run behind the extension stub in a process of its own.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def w13_contract_results() -> dict[str, dict[str, str]]:
    """Run the stub-backed transform checks once, in a process of their own.

    ``tests/kimi_k3_api_contract.py`` explains why it has to be a subprocess:
    the stubs it installs to run without a compiled extension cannot be
    uninstalled, so they must not be installed in the process that also runs the
    GPU test files.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "tests.kimi_k3_w13_contract"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    raise AssertionError(
        "the fused-W13 transform checks reported no results "
        f"(exit code {completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )


@pytest.mark.parametrize("check", list(CHECKS))
def test_w13_contract(
    w13_contract_results: dict[str, dict[str, str]], check: str
) -> None:
    result = w13_contract_results[check]
    assert result["outcome"] == "passed", result["detail"]


def test_the_transform_checks_are_reachable_without_the_cuda_extension() -> None:
    """Collection must not need ``mok._C``, which is the point of the split.

    Naming the transform at module scope here would run ``mok/__init__.py`` and
    import the compiled extension, and a collection error takes the whole file
    with it -- including the source contracts below, which need nothing at all.
    So the check is that this module reaches ``mok`` in exactly one way: through
    the subprocess.
    """
    source = Path(__file__).read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+(mok[\w.]*)", source, re.MULTILINE)
    assert imports == [], imports
    assert "mok" not in sys.modules or getattr(
        sys.modules["mok"], "__file__", None
    ) is not None, "the contract stubs must stay in the child process"


# ---------------------------------------------------------------------------
# The production contract: one representation, named the same everywhere.
# ---------------------------------------------------------------------------


def _body(source: str, signature: str) -> str:
    """Return the brace-balanced body that follows ``signature``."""
    start = source.index(signature)
    open_brace = source.index("{", start)
    depth = 0
    for offset in range(open_brace, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace : offset + 1]
    raise AssertionError(f"unbalanced body for {signature!r}")


def test_the_c_entrypoint_validates_the_fused_shapes_from_the_engine() -> None:
    """The operator's shape rule is the engine's own constants, not a literal.

    A prepared payload the engine's descriptor cannot read has to be rejected at
    the boundary, and the only figures that can do that are the ones the
    descriptor is built from.
    """
    source = (_CSRC / "entrypoints.cuh").read_text()
    body = _body(source, "static __host__ persistent::LaunchArguments decode_launch_arguments(")
    assert "expert_w1_" not in body
    assert "expert_w3_" not in body
    for name, rows, columns in (
        ("expert_w13_packed", "kFusedPackedRows", "kFusedPackedColumns"),
        ("expert_w13_scale", "kFusedScaleRows", "kFusedScaleColumns"),
    ):
        assert re.search(
            rf'check_expert\(\s*{name},\s*"{name}",\s*'
            rf"expert_mxfp4::fused_w13::{rows},\s*"
            rf"expert_mxfp4::fused_w13::{columns}\)",
            body,
        ), name
    assert (
        "expert_mxfp4::fused_w13::kFusedPackedAlignment" in body
    ), "the payload's base must be held to the descriptor's own figure"


def test_the_kernel_reaches_only_the_fused_gate_up_unit() -> None:
    """One gate/up unit, no engine selector, and the old unit is gone from here.

    The engine was a benchmark-time template argument while three candidates
    existed. Production runs one, so the argument is gone and the call is
    unconditional -- there is no longer a value of anything that could reach a
    different gate/up unit.
    """
    source = (_CSRC / "persistent_kernel.cuh").read_text()
    assert "ENGINE" not in source, "the engine selector must be gone"
    assert "expert_mxfp4::routed_gate_up_unit(" not in source, (
        "the superseded gate/up unit must not be reachable from the kernel"
    )
    assert source.count(
        "expert_mxfp4::fused_w13::routed_gate_up_fused_unit("
    ) == 1
    instantiations = re.findall(r"launch_persistent<([^>]*)>", source)
    assert instantiations, "launch_decode must instantiate the persistent kernel"
    for arguments in instantiations:
        parameters = [argument.strip() for argument in arguments.split(",")]
        assert parameters in (["false"], ["true"], ["TENSOR_PATH"]), parameters


def test_the_queue_claims_one_unit_per_expert_and_publishes_six_arrivals(
) -> None:
    """The queue length and the readiness threshold are separate constants.

    The whole gate/up saving comes from claiming one unit per expert instead of
    six, so the queue is a sixth as long. But the grouped down phase must still
    not start until all 384 ``situ`` columns of an expert exist, and its
    threshold is a count of *arrivals*, not of claims. So the unit is claimed
    once and publishes six, and the two numbers are named separately rather than
    one being reused for the other.
    """
    source = (_CSRC / "persistent_kernel.cuh").read_text()
    assert "inline constexpr int kGateUpUnitsPerExpert = 1;" in source
    assert re.search(
        r"inline constexpr int kGateUpArrivalsPerExpert =\s*"
        r"expert_mxfp4::fused_w13::kFusedTasks;",
        source,
    )
    assert "static_assert(kGateUpArrivalsPerExpert == 6);" in source

    # Phase 4 must block on the arrival count, not the claim count. Anchored on
    # the expert's own counter rather than on the error code, which the shared
    # gate/up wait a few lines above raises too.
    down = source.index("&scratch.expert_counts[expert], kGateUpArrivals")
    window = source[down : down + 200]
    assert "kGateUpArrivalsPerExpert" in window, (
        "the grouped down readiness wait must use the arrival threshold"
    )

    # And the unit's own publishing has to be one arrival per finished range, on
    # the pass that finished it.
    engine = (_CSRC / "expert_mxfp4_fused_w13.cuh").read_text()
    body = _body(engine, "static __device__ void routed_gate_up_fused_unit(")
    assert body.count("persistent::publish_count_at(arrival_counter);") == 1, (
        "the unit must publish from exactly one place, inside the task loop"
    )
    assert "if (last_pass) {" in body, (
        "a wide batch is several passes over the same columns, so only the "
        "last one may publish"
    )


def test_the_ring_is_armed_once_per_cta_and_never_re_armed_per_unit() -> None:
    """The stage barriers outlive one unit, and the parity is carried.

    Forty-two stream indices over a two-deep ring is twenty-one laps per stage,
    which is odd -- so a unit hands the next one both barriers mid-phase, and
    re-arming per unit would be asking ``mbarrier.init`` to reset a live
    barrier's parity mid-launch, which PTX only defines for a barrier that was
    invalidated first. A CTA runs several units per step, so that is not a
    corner case but the common one: measured on B300 at a shallower depth it
    deadlocked every unit after a CTA's first.
    """
    engine = (_CSRC / "expert_mxfp4_fused_w13.cuh").read_text()
    signature = "static __device__ void routed_gate_up_fused_unit("
    assert "const bool first_unit" in engine
    body = _body(engine, signature)
    guard = body.index("if (first_unit)")
    for match in re.finditer(r"init_semaphore\(", body):
        assert match.start() > guard, (
            "every ring barrier must be armed under the first_unit guard"
        )
    assert body.count("init_semaphore(") == 2, (
        "the ring has one arrival and one retirement barrier per stage, armed "
        "by one call each"
    )
    assert "arrived_phase = stream_parity[0]" in body
    assert "retired_phase = stream_parity[1]" in body
    assert "stream_parity[0] = arrived_phase" in body
    assert "stream_parity[1] = retired_phase" in body

    kernel = (_CSRC / "persistent_kernel.cuh").read_text()
    assert "bool first_unit = true;" in kernel, (
        "the flag must be a CTA-lifetime local of the persistent kernel"
    )
    lowered = kernel.index("first_unit = false;")
    raised = kernel.index("bool first_unit = true;")
    claim = kernel.index("while (true) {", raised)
    assert raised < claim < lowered, (
        "the flag must be raised outside the claim loop and lowered inside it, "
        "or a CTA that claims two batches re-arms the ring"
    )


def test_only_warp_zero_reads_the_carried_parity() -> None:
    """The carried parity is warp 0's state, so warp 0 is its only reader.

    The unit runs its K loop on warp 0 and hands the parity to the next unit
    from warp 0 lane 0. If all eight warps read the shared parity at the top,
    the seven that never use the values would still have raced that write.
    Measured on B300 by ``compute-sanitizer --tool racecheck`` on an earlier
    engine of the same shape: 16 hazards, two inlined copies on each of eight
    ranks. The reads were dead, so the numbers were never wrong, but an
    unsynchronised shared access is a hazard whether or not it is benign.
    """
    engine = (_CSRC / "expert_mxfp4_fused_w13.cuh").read_text()
    body = _body(engine, "static __device__ void routed_gate_up_fused_unit(")

    # Every read of the carried parity must sit inside a warp-0 guard, and that
    # guard must open before the read and hold across both of them.
    reads = [
        match.start()
        for match in re.finditer(r"= stream_parity\[[01]\];", body)
    ]
    assert len(reads) == 2, ("one read per carried phase", reads)
    guards = [
        match.start()
        for match in re.finditer(re.escape("if (warpid() == 0) {"), body)
        if match.start() < reads[0]
    ]
    assert guards, (
        "stream_parity is read with no warp-0 guard open, so all eight warps "
        "read it and the seven that never use it race the store that hands it on"
    )
    guard = guards[-1]
    closed = body.index("}", body.index("\n", reads[-1]))
    for read in reads:
        assert guard < read < closed

    # And the hand-off store stays lane 0's, so within warp 0 exactly one thread
    # writes what the other thirty-one only read. The one-time arming stores of
    # `0u` are not these: they run under `thread == 0` ahead of the CTA barrier
    # that opens the unit, so nothing is reading yet.
    handoffs = list(
        re.finditer(r"stream_parity\[[01]\] = (?:arrived|retired)_phase;", body)
    )
    assert len(handoffs) == 2, "one hand-off per carried phase"
    for match in handoffs:
        lane = body.rindex("if (lane == 0) {", 0, match.start())
        assert body.index("}", lane) > match.start(), (
            "stream_parity must be stored by lane 0 alone"
        )


def test_the_gather_runs_once_per_pass_ahead_of_the_six_tasks() -> None:
    """The one-time full activation gather, which is where the 5.2% came from.

    Which rows a slab needs does not depend on the task, so the seven distinct
    gathers an expert needs are done once for all six of its tasks. A per-task
    or per-slab gather would put the latent reads back on warp 0 between the MMA
    issues -- 42 gathers where seven exist -- which is exactly the cost this
    shape removed.
    """
    engine = (_CSRC / "expert_mxfp4_fused_w13.cuh").read_text()
    body = _body(engine, "static __device__ void routed_gate_up_fused_unit(")
    assert body.count("stage_fused_unit_activation(") == 1, (
        "the unit must stage its activation from exactly one place"
    )
    staged = body.index("stage_fused_unit_activation(")
    task_loop = body.index("for (int task = 0;")
    assert staged < task_loop, (
        "the gather must run once per pass, ahead of the six tasks that read it"
    )
    for field in ("scratch.latent_mxfp8", "scratch.latent_scale"):
        assert field not in body[task_loop:], (
            f"no {field} read may happen inside the task loop; the gather owns "
            "every one of them"
        )


def test_no_benchmark_only_fused_switch_survives_in_public_surface() -> None:
    """The engine selector, its process guard, and its private launcher are gone.

    While three candidates existed they were reached from a guarded private
    entrypoint. Production runs one of them now, so none of that machinery has a
    caller, and a switch with no caller is a switch that can be flipped.
    """
    forbidden = (
        "kFusedW13BenchmarkGuard",
        "fused_w13_benchmark_enabled",
        "launch_decode_fused_w13_benchmark",
        "kimi_k3_decode_fused_w13_benchmark",
        "kEngineFusedTask",
        "kEngineFusedExpert",
        "kEngineFusedPacked",
        "kEngineProduction",
    )
    for module in (
        "csrc/bindings.cu",
        "csrc/kimi_k3_decode/entrypoints.cuh",
        "csrc/kimi_k3_decode/expert_mxfp4_fused_w13.cuh",
        "csrc/kimi_k3_decode/persistent_kernel.cuh",
        "mok/kimi_k3.py",
        "mok/ops.py",
        "mok/_fake_impls.py",
        "modal_app.py",
    ):
        source = (_REPO / module).read_text()
        for name in forbidden:
            assert name not in source, (module, name)


def test_the_bounded_layout_probe_has_a_caller() -> None:
    """A test-only entrypoint with no test is a compiled thing nobody checks.

    The probe exists because what the descriptor's transaction count is and
    where its five dimensions land are properties of the device. That is only
    worth compiling if something calls it, so the call is asserted here rather
    than left to whoever next reads the header.
    """
    engine = (_CSRC / "expert_mxfp4_fused_w13.cuh").read_text()
    bindings = (_REPO / "csrc" / "bindings.cu").read_text()
    assert "kimi_k3_fused_w13_tma_probe_entrypoint" in engine
    assert "_kimi_k3_fused_w13_tma_probe" in bindings

    caller = (_REPO / "tests" / "test_kimi_k3_w13_layout.py").read_text()
    assert "_C._kimi_k3_fused_w13_tma_probe(" in caller
    # And it runs where the measurement means something.
    assert "get_device_capability(selected) != (10, 3)" in caller


def test_the_phase_clocks_carry_the_gate_up_subphases_durably() -> None:
    """The subphase band survives the integration, because the gate is still open.

    The overall performance gate has not closed, so the instrument that
    attributed 43.9% of the gate/up band to TMA wait has to remain available on
    the production path rather than only in a benchmark build. It rides the same
    ``PhaseClocks`` every other phase uses, so it is off unless a profiled
    launch turned the whole band on.
    """
    types_source = (_CSRC / "types.cuh").read_text()
    for clock in (
        "kClockRoutedGateUpTmaIssue",
        "kClockRoutedGateUpTmaWait",
        "kClockRoutedGateUpRingFull",
        "kClockRoutedGateUpMmaIssue",
        "kClockRoutedGateUpActivation",
        "kClockRoutedGateUpEpilogue",
    ):
        assert clock in types_source, clock
        # And each one is named in the reader's table, or a profile reports an
        # unlabelled column.
        assert f'"{clock[len("kClock"):]}"' in types_source or (
            clock.replace("kClock", "") in types_source
        ), clock

    engine = (_CSRC / "expert_mxfp4_fused_w13.cuh").read_text()
    body = _body(engine, "static __device__ void routed_gate_up_fused_unit(")
    for clock in (
        "kClockRoutedGateUpTmaIssue",
        "kClockRoutedGateUpTmaWait",
        "kClockRoutedGateUpRingFull",
        "kClockRoutedGateUpMmaIssue",
        "kClockRoutedGateUpActivation",
        "kClockRoutedGateUpEpilogue",
    ):
        assert clock in body, clock
    # The coarse pair the earlier measurements were taken against stays, so a
    # profile of this engine is still comparable with the ones before it.
    assert "kClockRoutedGateUpStage" in body
    assert "kClockRoutedGateUpMma" in body
