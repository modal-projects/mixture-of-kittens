"""CPU-only contracts for production projection-first scheduling."""

from __future__ import annotations

from pathlib import Path

from . import kimi_k3_decode_sources as decode_sources
from . import modal_sources


ROOT = Path(__file__).parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _decode(name: str) -> str:
    """A decode header's whole text, parts it umbrellas inlined."""
    return decode_sources.read(name)


def _function_body(text: str, signature: str) -> str:
    start = text.index(signature)
    depth = 0
    for offset in range(text.index("{", start), len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise AssertionError(f"{signature} is never closed")


def test_production_issues_projection_and_shared_work_before_scores() -> None:
    persistent = _decode("persistent_kernel.cuh")
    kernel = _function_body(
        persistent,
        "void kimi_k3_decode_persistent_kernel(",
    )
    route = kernel.split("// Phase 1:", 1)[1].split("// Phase 2:", 1)[0]
    gate_up = kernel.split("// Phase 3:", 1)[1].split("// Phase 4:", 1)[0]

    assert (
        "const int units = projection_units + shared_units + score_units;"
        in route
    )
    assert route.index("const int projection_unit") < route.index(
        "const int shared_unit"
    )
    assert route.index("const int shared_unit") < route.index(
        "const int score_unit ="
    )
    assert route.count("shared_experts::project_tensor(") == 1
    assert "shared_experts::project_tensor(" not in gate_up
    assert "projection_first" not in persistent
    assert kernel.count("claim_unit(") == 1
    assert persistent.count("kimi_k3_decode_persistent_kernel<") == 3


def test_temporary_projection_ab_surface_is_removed() -> None:
    sources = "\n".join(
        (
            _source("csrc/bindings.cu"),
            _decode("entrypoints.cuh"),
            _source("benchmarks/kimi_k3_decode_runtime.py"),
            modal_sources.read(),
        )
    )

    assert "projection_first_scheduling" not in sources
    assert "_kimi_k3_decode_set_projection_first" not in sources
    assert "bench_kimi_k3_projection_first_ab" not in sources
    assert not (ROOT / "benchmarks/kimi_k3_projection_first_ab.py").exists()


def test_durable_phase_profiler_keeps_aggregate_grid_barrier_clock() -> None:
    types = _decode("types.cuh")
    kernel = _function_body(
        _decode("persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )

    # Two 256-byte scratch grains rather than one. The accumulator band ends the
    # region and it grew: the routed gate/up phase reports six subphases of its
    # own now, and the assignment build and the publish write have bands of
    # their own, so twenty-two accumulated regions at two slots each do not fit
    # above the thirty-six live counters in 64.
    assert "NUM_PHASE_COUNTERS = 128" in types
    assert "kPhaseClockCount == 22" in types
    assert "kPhaseClockBegin == 84" in types
    assert '"grid_barrier"' in types
    assert "kClockGridBarrier" in kernel
    assert "route_latent_makespan" not in types
    assert "clocks.maximum(" not in kernel
    assert "phase_profiling" in _source("benchmarks/kimi_k3_decode_runtime.py")
