"""What the one-launch check is required to recognize, and to refuse.

The one-launch gate reads the profiler's record of what reached the device, so
the whole gate is the predicate that decides whether a name is production's
kernel. That predicate has been wrong twice -- once by naming the barrier
schedule after the dependency-local one was promoted, and once by matching the
schedule as a substring while four engines share that schedule's name -- and
both times the gate passed or failed for a reason unrelated to the launch count.

So it is held here to captured names rather than only exercised on eight B300s:
a predicate whose failures are attributable is worth more than one whose
successes are.

No GPU, no compiled extension, no ``mok`` import.
"""

from __future__ import annotations

import pytest

from benchmarks.kimi_k3_launch_names import (
    BARRIER_SCHEDULE_KERNEL,
    DEPENDENCY_LOCAL_KERNEL,
    PRODUCTION_GATE_UP_ENGINE,
    assert_one_production_launch,
    dependency_local_engine,
    is_production_launch,
)


def _demangled(engine: int, tensor_path: bool) -> str:
    """One kernel name as ``torch.profiler`` reports it.

    The shape is what ``c10::demangle`` produces for the schedule: the return
    type, the fully qualified template name, the capacity flag, the engine, and
    the layouts type.
    """
    flag = "true" if tensor_path else "false"
    layouts = "TensorLayouts" if tensor_path else "NoTensorLayouts"
    return (
        f"void kimi_k3_decode::persistent::schedule::"
        f"{DEPENDENCY_LOCAL_KERNEL}<{flag}, {engine}, "
        f"kimi_k3_decode::persistent::{layouts}>(__nv_bfloat16 const*, "
        f"__nv_bfloat16 const*, float const*, int, int, int)"
    )


def _mangled(engine: int, tensor_path: bool) -> str:
    """One kernel name as ``cuobjdump`` reports it, which is not demangled.

    Taken from the shape the SM103 binary dumps in this repository carry, which
    is where the ``ILb0ELi2E`` spelling comes from.
    """
    flag = 1 if tensor_path else 0
    layouts = "13TensorLayouts" if tensor_path else "15NoTensorLayouts"
    return (
        f"_ZN15kimi_k3_decode10persistent8schedule"
        f"{DEPENDENCY_LOCAL_KERNEL}ILb{flag}ELi{engine}ENS0_{layouts}EEEv"
        f"PK13__nv_bfloat16S6_PKfiii"
    )


@pytest.mark.parametrize("spelling", [_demangled, _mangled])
@pytest.mark.parametrize("tensor_path", [False, True])
def test_production_is_recognized_in_both_spellings_on_both_paths(
    spelling, tensor_path: bool
) -> None:
    """Mangled or demangled, core or tensor, it is the same one launch.

    Which capacity path a shape takes is the shape's business and both are
    production, so the predicate reads the engine argument and not the flag in
    front of it.
    """
    name = spelling(PRODUCTION_GATE_UP_ENGINE, tensor_path)
    assert dependency_local_engine(name) == PRODUCTION_GATE_UP_ENGINE
    assert is_production_launch(name)
    assert_one_production_launch([name])


@pytest.mark.parametrize("engine", [3, 4, 6])
@pytest.mark.parametrize("spelling", [_demangled, _mangled])
def test_a_measured_arm_is_not_a_production_launch(spelling, engine: int) -> None:
    """This is the failure the substring match let through.

    Engines 3, 4 and 6 are the arms the integration's A/B measured. Each is its
    own compiled kernel of the same schedule, so a benchmark process that left
    one selected would launch exactly one kernel whose name contains the
    schedule's -- and a gate that only counted to one and looked for that
    substring would call the arm's latency production's.
    """
    name = spelling(engine, False)
    assert dependency_local_engine(name) == engine
    assert not is_production_launch(name)
    with pytest.raises(AssertionError):
        assert_one_production_launch([name])


def test_the_barrier_schedule_is_not_a_production_launch() -> None:
    """This is the other failure: a name that predates the promotion.

    The barrier schedule is still compiled and still reachable through the
    schedule switch, so a gate written against its name would pass in a process
    that had switched to it and fail in every process that had not.
    """
    name = (
        f"void kimi_k3_decode::persistent::{BARRIER_SCHEDULE_KERNEL}<false, "
        f"kimi_k3_decode::persistent::NoTensorLayouts>(int)"
    )
    assert dependency_local_engine(name) is None
    assert not is_production_launch(name)
    with pytest.raises(AssertionError):
        assert_one_production_launch([name])


@pytest.mark.parametrize(
    "name",
    [
        "",
        "void at::native::elementwise_kernel<128, 2>(int)",
        "Memcpy DtoD",
        "kimi_k3_router_kernel",
        # Close enough to matter: the schedule's name as a bare identifier, with
        # no template arguments at all. A regex anchored on the name alone would
        # take this.
        DEPENDENCY_LOCAL_KERNEL,
    ],
)
def test_an_arbitrary_kernel_is_refused(name: str) -> None:
    """Nothing but the schedule under production's engine may be accepted."""
    if DEPENDENCY_LOCAL_KERNEL in name:
        # A name that is the schedule but carries no engine means the mangling
        # changed. That has to be loud, because reporting it as "not production"
        # would surface as a one-launch failure pointing at the launch rather
        # than at the predicate.
        with pytest.raises(AssertionError, match="engine argument"):
            dependency_local_engine(name)
        return
    assert dependency_local_engine(name) is None
    assert not is_production_launch(name)


def test_more_than_one_launch_is_refused_however_it_is_named() -> None:
    """The count still has to be one, and each of them still has to be it."""
    production = _demangled(PRODUCTION_GATE_UP_ENGINE, False)
    with pytest.raises(AssertionError):
        assert_one_production_launch([])
    with pytest.raises(AssertionError):
        assert_one_production_launch([production, production])
    with pytest.raises(AssertionError):
        assert_one_production_launch([production, "Memcpy DtoD"])
