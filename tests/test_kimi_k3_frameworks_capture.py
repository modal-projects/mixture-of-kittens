"""CPU-only contracts for what a Kimi K3 comparison run captures and reports.

The effective CLI overrides a dry run records, the phase accounting derived
from the kernel's own clock band, one immutable router for every captured
graph, the fail-closed rejections the gates owe an empty or partial run, and
that importing the comparison package needs no serving framework installed.

How a run is pinned is in ``test_kimi_k3_frameworks.py``; its numerical gates
are in ``test_kimi_k3_frameworks_gates.py``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from . import kimi_k3_decode_sources as decode_sources

from . import modal_sources

REPO_ROOT = Path(__file__).parents[1]


def _compare():
    return importlib.import_module("benchmarks.compare_kimi_k3_frameworks")


def _parity_row(**overrides):
    """One archived parity row, at the values the passing runs measured."""
    row = {
        "mode": "block16",
        "tokens": 16,
        "pool_index": 0,
        "router": {
            "expert_ids_match": True,
            "expert_id_mismatch_count": 0,
            "router_weight_max_abs": 7.45e-09,
            "router_weight_mean_abs": 1.0e-10,
            "topk": 16,
            "distinct_experts": 256,
        },
        "custom_vs_native": {
            "relative_l1": 0.02160,
            "cosine_similarity": 0.999766,
            "max_abs": 0.0859,
            "finite": True,
        },
        "custom_vs_reference": {
            "relative_l1": 0.00503,
            "cosine_similarity": 0.999985,
            "max_abs": 0.0215,
            "finite": True,
        },
        "native_vs_reference": {
            "relative_l1": 0.02176,
            "cosine_similarity": 0.999761,
            "max_abs": 0.0859,
            "finite": True,
        },
        "routed_latent_vs_reference": {
            "relative_l1": 0.0,
            "cosine_similarity": 1.0,
            "max_abs": 0.0,
        },
        "shared_output_vs_reference": {
            "relative_l1": 1.0e-05,
            "cosine_similarity": 0.9999999,
            "max_abs": 4.88e-04,
        },
    }
    for path, value in overrides.items():
        head, _, tail = path.partition("__")
        if tail:
            row[head] = {**row[head], tail: value}
        else:
            row[head] = value
    return row


def _write_archive(directory: Path, framework: str, *, medians, p99s, parity):
    compare = _compare()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"framework": framework, "benchmark": compare.BENCHMARK})
    )
    rows = []
    for backend in ("mok", framework):
        for tokens in range(16, 129, 16):
            rows.append(
                {
                    "backend": backend,
                    "mode": "block16",
                    "tokens": tokens,
                    "median_ms": medians[backend],
                    "p90_ms": medians[backend],
                    "p99_ms": p99s[backend],
                    "geomean_ms": medians[backend],
                }
            )
    (directory / "latency_block16.json").write_text(
        json.dumps({"rows": rows})
    )
    (directory / "parity.json").write_text(
        json.dumps({"framework": framework, "rows": parity})
    )
    return directory


class _FakeEvent:
    """A CUDA-event stand-in whose readings come from an explicit clock."""

    def __init__(self, clock: list[float]) -> None:
        self._clock = clock
        self.stamp: float | None = None

    def record(self) -> None:
        self.stamp = self._clock[0]

    def elapsed_time(self, other: _FakeEvent) -> float:
        assert self.stamp is not None and other.stamp is not None
        return other.stamp - self.stamp


def compare_artifact_files():
    return _compare().ARTIFACT_FILES


def _inputs():
    return importlib.import_module("benchmarks.kimi_k3_decode_inputs")


# Effective CLI overrides and event priming
# --------------------------------------------------------------------------


def test_dry_run_manifest_records_the_effective_cli_overrides(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.compare_kimi_k3_frameworks",
            "--dry-run",
            "--framework",
            "sglang",
            "--output-dir",
            str(tmp_path),
            "--warmup-count",
            "7",
            "--sample-count",
            "11",
            "--pool-size",
            "2",
            "--modes",
            "block16",
            "--tokens",
            "16,32",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["warmup_count"] == 7
    assert manifest["sample_count"] == 11
    assert manifest["graph_pool_size"] == 2
    assert manifest["shape_groups"] == {"block16": [16, 32]}
    assert manifest["routing"]["pool_entries"] == 2
    # A dry run builds no image, so it has no observed reference to record and
    # records the pin it would have been held to instead.
    assert manifest["image_reference"] is None
    assert manifest["image_reference_expected"] == (
        _compare().pinned_image_reference("sglang")
    )



def test_replay_samples_rejects_a_non_positive_count() -> None:
    from benchmarks.kimi_k3_timing import replay_samples

    with pytest.raises(ValueError, match="positive"):
        replay_samples(
            lambda iteration: None,
            warmup_count=1,
            sample_count=0,
            event_factory=lambda: _FakeEvent([0.0]),
            synchronize=lambda: None,
        )


def test_phase_cycle_summary_ranks_regions_by_their_accounted_share() -> None:
    """The split counters describe a region, they do not add to the total.

    ``routed_gate_up_stage`` and ``routed_gate_up_mma`` are measured inside
    ``routed_gate_up``, so counting all three would report a total larger than
    the cycles the kernel actually spent and would understate every share.
    """
    compare = _compare()
    summary = compare.summarize_phase_cycles(
        {
            "router_score": 100,
            "routed_gate_up": 600,
            "routed_gate_up_stage": 500,
            "routed_gate_up_mma": 90,
            "routed_down": 300,
            "routed_down_stage": 250,
            "routed_down_mma": 40,
        }
    )

    assert summary["accounted_cycles"] == 1000
    assert summary["share_of_accounted"]["routed_gate_up"] == pytest.approx(0.6)
    assert summary["share_of_accounted"]["routed_down"] == pytest.approx(0.3)
    assert summary["ranked"][0] == ("routed_gate_up", 600)
    assert summary["ranked"][1] == ("routed_down", 300)
    assert summary["dominant_region"] == "routed_gate_up"
    assert summary["dominant_share"] == pytest.approx(0.6)


def test_phase_cycle_derivation_separates_routed_epilogues_and_queue() -> None:
    """The focused run must isolate staging, MMA, epilogue, and queue cycles."""
    compare = _compare()

    cycles = compare.derive_phase_cycles(
        {
            "routed_gate_up": 600,
            "routed_gate_up_stage": 500,
            "routed_gate_up_mma": 90,
            "routed_down": 300,
            "routed_down_stage": 250,
            "routed_down_mma": 40,
            "routed_queue": 20,
        }
    )

    assert cycles["routed_gate_up_epilogue"] == 10
    assert cycles["routed_down_epilogue"] == 10
    assert cycles["routed_queue"] == 20
    summary = compare.summarize_phase_cycles(cycles)
    # 600 + 300 + 20: the three top-level regions, and none of the six children
    # that measure the inside of two of them.
    assert summary["accounted_cycles"] == 920
    assert summary["share_of_accounted"]["routed_down_epilogue"] == pytest.approx(
        1 / 92
    )
    # A child's share of its own parent is the actionable number, and the two
    # are reported separately rather than one standing in for the other.
    assert summary["share_of_parent"]["routed_gate_up_stage"] == pytest.approx(
        500 / 600
    )
    assert summary["share_of_parent"]["routed_down_mma"] == pytest.approx(
        40 / 300
    )
    assert "routed_queue" not in summary["share_of_parent"]
    assert summary["top_level"] == list(compare.PHASE_CLOCK_TOP_LEVEL)


def test_phase_cycle_summary_tolerates_an_unprofiled_launch() -> None:
    compare = _compare()
    summary = compare.summarize_phase_cycles({"router_score": 0, "tail": 0})

    assert summary["accounted_cycles"] == 0
    assert summary["dominant_region"] is None
    assert summary["share_of_accounted"] == {"router_score": 0.0, "tail": 0.0}


def test_phase_clock_names_match_the_kernel_scratch_band() -> None:
    """The reader names the counters; the kernel decides how many there are."""
    compare = _compare()
    source = decode_sources.read("types.cuh")
    names = [
        match.group(1)
        for match in __import__("re").finditer(
            r'^\s{4}"([a-z0-9_]+)",$',
            source.split("kPhaseClockNames[] = {", 1)[1].split("};", 1)[0],
            __import__("re").MULTILINE,
        )
    ]

    assert names == list(compare.PHASE_CLOCK_NAMES)
    assert names[0] == "readiness_wait"
    assert "routed_queue" in names
    assert "readiness_wait" in names
    assert "assignments" not in names
    assert "routed_gate_up_stage" in names
    assert "routed_gate_up_mma" in names

    # And the tree. The parent of every region is declared next to the clocks,
    # so the reader's total is the header's notion of which regions are
    # disjoint rather than a guess made from the names.
    parents = __import__("re").findall(
        r"^\s{4}(kPhaseClockTopLevel|kClock\w+),\s*//\s*(\w+)$",
        source.split("kPhaseClockParents[] = {", 1)[1].split("};", 1)[0],
        __import__("re").MULTILINE,
    )
    assert [name for _, name in parents] == names
    for parent, name in parents:
        expected = compare.PHASE_CLOCK_PARENTS[name]
        if expected is None:
            assert parent == "kPhaseClockTopLevel", name
        else:
            camel = "".join(part.capitalize() for part in expected.split("_"))
            assert parent == f"kClock{camel}", (name, parent)

    # The fused unit's own partition of the gate/up band. Each one is a child of
    # the region it partitions, or the comparison would add it to the launch
    # total it is a fraction of.
    for subphase, parent in (
        ("routed_gate_up_tma_issue", "routed_gate_up_stage"),
        ("routed_gate_up_tma_wait", "routed_gate_up_stage"),
        ("routed_gate_up_ring_full", "routed_gate_up_stage"),
        ("routed_gate_up_mma_issue", "routed_gate_up_mma"),
        ("routed_gate_up_activation", "routed_gate_up"),
        ("routed_gate_up_epilogue", "routed_gate_up"),
    ):
        assert subphase in names, subphase
        assert compare.PHASE_CLOCK_PARENTS[subphase] == parent, subphase
        assert subphase not in compare.PHASE_CLOCK_TOP_LEVEL, subphase


def test_a_measured_gate_up_epilogue_is_not_overwritten_by_the_residual() -> None:
    """The fused unit times its epilogue, so nothing may estimate it instead.

    ``routed_down`` still has no epilogue counter of its own and keeps the
    residual. Reporting a residual over a measured counter would hide the gap
    between the six subphases and the band they partition, which is the only
    signal that a subphase is missing time.
    """
    compare = _compare()

    cycles = compare.derive_phase_cycles(
        {
            "routed_gate_up": 600,
            "routed_gate_up_stage": 500,
            "routed_gate_up_mma": 90,
            "routed_gate_up_epilogue": 7,
            "routed_down": 300,
            "routed_down_stage": 250,
            "routed_down_mma": 40,
        }
    )

    assert cycles["routed_gate_up_epilogue"] == 7
    assert cycles["routed_down_epilogue"] == 10
    # And none of the gate/up breakdown enters the launch total.
    assert compare.summarize_phase_cycles(cycles)["accounted_cycles"] == 900


def test_dry_run_writes_the_complete_comparison_manifest(tmp_path: Path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.compare_kimi_k3_frameworks",
            "--dry-run",
            "--framework",
            "vllm",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert json.loads(result.stdout) == manifest
    assert manifest["benchmark"] == "kimi_k3_framework_comparison"
    assert manifest["dry_run"] is True
    assert manifest["framework"] == "vllm"
    assert manifest["backends"] == ["mok", "vllm"]
    assert manifest["tp_size"] == 8
    assert manifest["warmup_count"] == 500
    assert manifest["sample_count"] == 1000
    assert manifest["shape_groups"] == {
        "block8": list(range(8, 65, 8)),
        "block16": list(range(16, 129, 16)),
    }
    assert manifest["numerical_tolerances"] == {
        "relative_l1": 0.05,
        "cosine_similarity": 0.999,
        "max_abs": 1.0,
    }
    assert manifest["performance_gates"]["p99_limit_ratio"] == 1.10
    assert manifest["artifact_files"] == list(compare_artifact_files())




def test_base_package_import_does_not_require_a_serving_framework() -> None:
    compare = _compare()
    source = (
        REPO_ROOT / "benchmarks" / "compare_kimi_k3_frameworks.py"
    ).read_text()
    module_level = source.split("def ", 1)[0]

    for forbidden in ("import vllm", "import sglang", "import flashinfer"):
        assert forbidden not in module_level
    assert compare.ADAPTER_MODULES == {
        "vllm": "benchmarks.frameworks.vllm_kimi_k3",
        "sglang": "benchmarks.frameworks.sglang_kimi_k3",
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.frameworks.vllm_kimi_k3",
        "benchmarks.frameworks.sglang_kimi_k3",
    ],
)
def test_adapter_modules_keep_framework_imports_out_of_module_scope(
    module_name: str,
) -> None:
    relative = module_name.replace(".", "/") + ".py"
    source = (REPO_ROOT / relative).read_text()
    module_level = source.split("\nclass ", 1)[0].split("\ndef ", 1)[0]

    for forbidden in ("import vllm", "import sglang", "import flashinfer"):
        assert forbidden not in module_level


def test_modal_exposes_the_two_framework_comparison_entrypoints() -> None:
    source = modal_sources.read()

    assert "def compare_vllm(" in source
    assert "def compare_sglang(" in source
    assert source.count("framework_comparison_image(") >= 2
    # The expected artifact set is sourced from the comparison module rather
    # than restated here, so a new artifact cannot pass the driver's
    # completeness check while the Modal function still rejects it.
    assert "comparison_artifact_files" in source
    assert "COMPARISON_ARTIFACT_FILES" not in source


# --------------------------------------------------------------------------
# One immutable router for every captured graph
# --------------------------------------------------------------------------




def test_one_router_column_plan_covers_every_graph_a_run_captures() -> None:
    """Every (token count, pool entry, token) triple owns its own coordinate.

    A CUDA graph records the address of the router weight, not its contents,
    so a pool of graphs can only carry a pool of routes if one immutable
    router already holds all of them. That works because a pool entry's hidden
    state is one-hot: the entry's routes are decided by a single column, and
    the columns other entries use contribute nothing to it. The plan is the
    assignment of those columns, and it has to be injective.
    """
    inputs = _inputs()
    plan = inputs.router_column_plan(
        [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128],
        pool_size=4,
        hidden_size=7168,
    )

    seen: dict[int, tuple[int, int, int]] = {}
    for tokens in (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128):
        for pool_index in range(4):
            for token in range(tokens):
                column = inputs.router_column(plan, tokens, pool_index, token)
                assert 0 <= column < 7168
                assert column not in seen, (column, seen.get(column))
                seen[column] = (tokens, pool_index, token)
    assert len(seen) == 4 * (8 + 16 + 24 + 32 + 40 + 48 + 56 + 64 + 80 + 96 + 112 + 128)


def test_the_sweep_token_counts_cover_every_shape_the_benchmarks_measure(
) -> None:
    """One router covers a run, so the run's shapes have to be known up front."""
    inputs = _inputs()
    compare = _compare()
    output = importlib.import_module("benchmarks.kimi_k3_decode_output")

    measured = {
        tokens
        for groups in (compare.SHAPE_GROUPS, output.SHAPE_GROUPS)
        for shapes in groups.values()
        for tokens in shapes
    }

    assert measured == set(inputs.SWEEP_TOKEN_COUNTS)
    plan = inputs.router_column_plan(
        inputs.SWEEP_TOKEN_COUNTS, pool_size=4, hidden_size=7168
    )
    assert len(plan) == len(inputs.SWEEP_TOKEN_COUNTS)


def test_the_router_column_plan_refuses_a_sweep_it_cannot_encode() -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="hidden columns"):
        inputs.router_column_plan(
            list(range(1, 129)), pool_size=4, hidden_size=7168
        )


def test_the_router_column_plan_rejects_a_repeated_token_count() -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="once"):
        inputs.router_column_plan([16, 16], pool_size=4, hidden_size=7168)


def test_graph_route_verification_rejects_a_pool_that_collapsed(
) -> None:
    """The failure a mutable router produces, stated as data.

    Loading a new router between two captures overwrites the storage both
    graphs point at, so every graph in the pool replays the last entry's
    routing. The verifier's job is to see exactly that.
    """
    inputs = _inputs()
    intended = [
        inputs.route_assignments(2, pool_index) for pool_index in range(4)
    ]
    collapsed = [intended[-1]] * 4

    with pytest.raises(AssertionError) as failure:
        inputs.verify_graph_routes(intended, collapsed)

    message = str(failure.value)
    assert "collapsed" in message
    assert "pool_index=0" in message


def test_graph_route_verification_accepts_each_graph_keeping_its_own(
) -> None:
    inputs = _inputs()
    intended = [
        inputs.route_assignments(2, pool_index) for pool_index in range(4)
    ]

    summary = inputs.verify_graph_routes(intended, intended)

    assert summary["graph_count"] == 4
    assert summary["distinct_route_sets"] == 4
    assert summary["distinct_experts_per_graph"] == [32, 32, 32, 32]


def test_graph_route_verification_rejects_one_graph_off_its_entry() -> None:
    inputs = _inputs()
    intended = [
        inputs.route_assignments(2, pool_index) for pool_index in range(4)
    ]
    observed = list(intended)
    observed[2] = intended[3]

    with pytest.raises(AssertionError, match="pool_index=2"):
        inputs.verify_graph_routes(intended, observed)


@pytest.mark.parametrize(
    "module_name",
    ["benchmarks.frameworks.vllm_kimi_k3", "benchmarks.frameworks.sglang_kimi_k3"],
)
def test_no_adapter_writes_a_router_outside_of_binding_it(
    module_name: str,
) -> None:
    """The gate parameters are written once, by name, and nowhere else.

    Neither framework can be imported on a CPU box, so this reads the source:
    the only ``copy_into`` calls that target ``gate.weight`` or the correction
    bias are in ``_load``, which runs before anything is captured, and in
    ``bind_router``, which is called once for the run.
    """
    relative = module_name.replace(".", "/") + ".py"
    source = (REPO_ROOT / relative).read_text()

    assert "def load_router(" not in source
    assert "def bind_router(" in source

    writers = [
        block.split("(", 1)[0].split()[-1]
        for block in source.split("\n    def ")[1:]
        if "copy_into(self._layer.gate" in block
        or "copy_into(layer.gate" in block
    ]
    assert writers == ["_load", "bind_router"], writers

    capture = source.split("\n    def capture(", 1)[1].split("\n    def ", 1)[0]
    assert "copy_into" not in capture
    assert "load_router" not in capture
    assert "bind_router" not in capture


# --------------------------------------------------------------------------
# Fail-closed numerical gates
# --------------------------------------------------------------------------


def test_numerical_gates_reject_an_empty_row_set() -> None:
    """No rows is not a pass; it is a run that measured nothing."""
    compare = _compare()

    with pytest.raises(ValueError, match="no parity rows"):
        compare.evaluate_numerical_gates([])


def test_numerical_gates_require_an_explicit_finiteness_finding() -> None:
    compare = _compare()
    row = _parity_row()
    del row["custom_vs_reference"]["finite"]

    with pytest.raises(ValueError, match="finite"):
        compare.evaluate_numerical_gates([row])


def test_numerical_gates_require_every_metric_on_every_comparison() -> None:
    compare = _compare()
    row = _parity_row()
    del row["native_vs_reference"]["cosine_similarity"]

    with pytest.raises(ValueError, match="cosine_similarity"):
        compare.evaluate_numerical_gates([row])


def test_numerical_gates_require_every_comparison_to_be_present() -> None:
    compare = _compare()
    row = _parity_row()
    del row["custom_vs_native"]

    with pytest.raises(ValueError, match="custom_vs_native"):
        compare.evaluate_numerical_gates([row])


def test_numerical_gates_reject_a_duplicated_shape_and_pool_row() -> None:
    compare = _compare()

    with pytest.raises(ValueError, match="duplicate"):
        compare.evaluate_numerical_gates([_parity_row(), _parity_row()])


def test_numerical_gates_reject_a_row_the_expected_coverage_omits() -> None:
    compare = _compare()

    with pytest.raises(ValueError, match="unexpected"):
        compare.evaluate_numerical_gates(
            [_parity_row(), _parity_row(tokens=48)],
            expected_rows=[("block16", 16, 0)],
        )


def test_numerical_gates_reject_a_shape_the_run_never_measured() -> None:
    compare = _compare()

    with pytest.raises(ValueError, match="missing"):
        compare.evaluate_numerical_gates(
            [_parity_row()],
            expected_rows=[("block16", 16, 0), ("block16", 16, 1)],
        )


def test_numerical_gates_accept_the_exact_expected_coverage() -> None:
    compare = _compare()

    gates = compare.evaluate_numerical_gates(
        [_parity_row(), _parity_row(pool_index=1)],
        expected_rows=[("block16", 16, 0), ("block16", 16, 1)],
    )

    assert gates["passed"] is True
    assert gates["row_count"] == 2
    assert gates["coverage"]["expected_row_count"] == 2


# --------------------------------------------------------------------------
# Fail-closed archive combination
# --------------------------------------------------------------------------


def test_combine_archives_requires_parity_from_every_archive(
    tmp_path: Path,
) -> None:
    compare = _compare()
    directories = [
        _write_archive(
            tmp_path / "vllm",
            "vllm",
            medians={"mok": 1.0, "vllm": 1.4},
            p99s={"mok": 1.1, "vllm": 1.5},
            parity=[_parity_row()],
        ),
        _write_archive(
            tmp_path / "sglang",
            "sglang",
            medians={"mok": 1.0, "sglang": 1.2},
            p99s={"mok": 1.1, "sglang": 1.3},
            parity=[_parity_row(pool_index=1)],
        ),
    ]
    (tmp_path / "sglang" / "parity.json").unlink()

    with pytest.raises(ValueError, match="parity.json"):
        compare.combine_archives(directories, tmp_path / "combined")


def test_combine_archives_requires_the_manifest_coverage_it_claims(
    tmp_path: Path,
) -> None:
    """An archive that dropped a shape cannot be combined as if it had not."""
    compare = _compare()
    directory = _write_archive(
        tmp_path / "vllm",
        "vllm",
        medians={"mok": 1.0, "vllm": 1.4},
        p99s={"mok": 1.1, "vllm": 1.5},
        parity=[_parity_row()],
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["shape_groups"] = {"block16": [16, 32]}
    manifest["graph_pool_size"] = 1
    (directory / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing"):
        compare.combine_archives([directory], tmp_path / "combined")


# --------------------------------------------------------------------------
# Observed provenance
# --------------------------------------------------------------------------


def test_effective_image_reference_requires_the_builder_to_report_one(
    monkeypatch,
) -> None:
    """An expected pin is not evidence of what the container booted.

    Recording the manifest's digest when the builder said nothing would put a
    value in the archive that no observation supports, which is the one thing
    an image pin exists to prevent.
    """
    compare = _compare()
    monkeypatch.delenv("MOK_COMPARISON_IMAGE_REF", raising=False)

    with pytest.raises(ValueError, match="MOK_COMPARISON_IMAGE_REF"):
        compare.effective_image_reference("vllm")


def test_effective_image_reference_is_optional_only_for_a_dry_run(
    monkeypatch,
) -> None:
    compare = _compare()
    monkeypatch.delenv("MOK_COMPARISON_IMAGE_REF", raising=False)

    assert compare.effective_image_reference("vllm", dry_run=True) is None


def test_effective_image_reference_returns_the_reported_pin(
    monkeypatch,
) -> None:
    compare = _compare()
    pinned = compare.pinned_image_reference("vllm")
    monkeypatch.setenv("MOK_COMPARISON_IMAGE_REF", pinned)

    assert compare.effective_image_reference("vllm") == pinned


# --------------------------------------------------------------------------
# Steady-state samples
# --------------------------------------------------------------------------


def test_replay_samples_discards_a_settling_replay_after_priming() -> None:
    """Priming the events is not the same as reaching a steady state.

    The primed pair costs the driver's first-record initialization, so the
    replay it brackets is itself not a steady-state one. One more replay and
    event pair are run and discarded before the persisted series begins, and
    the series is exactly ``sample_count`` long.
    """
    from benchmarks.kimi_k3_timing import replay_samples

    clock = [0.0]
    replays: list[int] = []

    def replay(iteration: int) -> None:
        replays.append(iteration)
        clock[0] += {1: 50.0, 2: 7.0}.get(len(replays) - 2, 1.0)

    samples = replay_samples(
        replay,
        warmup_count=2,
        sample_count=4,
        event_factory=lambda: _FakeEvent(clock),
        synchronize=lambda: None,
    )

    assert samples == [1.0, 1.0, 1.0, 1.0]
    assert replays == [0, 1, 2, 3, 4, 5, 6, 7]


def test_replay_samples_settles_every_graph_in_a_rotating_pool() -> None:
    """A pool rotates, so one settling replay only settles one of its graphs.

    The measurement rotates a pool of graphs by iteration index, and the
    discarded replays have to cover the whole pool: otherwise the persisted
    series opens with some graph's first timed replay. The fake below charges
    every graph's first timed replay an extra 9, so a series that contains one
    cannot read as steady.
    """
    from benchmarks.kimi_k3_timing import replay_samples

    pool = 4
    warmup_count = 2
    clock = [0.0]
    replays: list[int] = []
    timed_before: set[int] = set()

    def replay(iteration: int) -> None:
        replays.append(iteration)
        if len(replays) <= warmup_count:
            return
        graph = iteration % pool
        first_timed = graph not in timed_before
        timed_before.add(graph)
        clock[0] += 1.0 + (9.0 if first_timed else 0.0)
        if len(replays) == warmup_count + 1:
            clock[0] += 50.0  # the driver's first event record

    samples = replay_samples(
        replay,
        warmup_count=warmup_count,
        sample_count=4,
        settle_count=pool,
        event_factory=lambda: _FakeEvent(clock),
        synchronize=lambda: None,
    )

    assert samples == [1.0, 1.0, 1.0, 1.0]
    assert replays == list(range(warmup_count + 1 + pool + 4))
