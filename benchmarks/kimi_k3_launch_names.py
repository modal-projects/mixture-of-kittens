"""Recognizing the one kernel a Kimi K3 decode step is allowed to launch.

The one-launch claim is checked against the profiler's record of what reached
the device, and for that to mean anything the check has to name the kernel
production launches -- not merely count to one, and not merely find a substring
that any instantiation of anything would satisfy.

Three things make the naive check wrong, and each of them has been wrong here:

* **There are two schedules.** ``kimi_k3_decode_persistent_kernel`` is the
  full-grid barrier schedule and ``kimi_k3_decode_dependency_local_kernel`` is
  the per-edge readiness one. The second was promoted, so a check written
  against the first passes only while the promotion has not happened.
* **The schedule is compiled once per gate/up engine.** The engine is a template
  argument, so each is its own ``__global__`` with its own name, and a substring
  match on the schedule accepts the A/B baseline that a benchmark process left
  selected. A run that measured production's latency under the baseline's kernel
  would be attributing the number to the wrong thing.
* **A profiler name may arrive mangled or demangled.** CUPTI hands back the
  symbol and PyTorch demangles it, but the SASS dumps this repository also reads
  do not. Both spellings carry the same two template arguments in the same
  order, so both are read here rather than one being assumed.

So a name is recognized only when it names the dependency-local schedule *and*
its engine argument can be read out *and* that argument is production's. Nothing
else is accepted, which is what makes `assert_one_production_launch` a check
rather than a formality.
"""

from __future__ import annotations

import re


#: The schedule a decode step launches, and the one it replaced.
DEPENDENCY_LOCAL_KERNEL = "kimi_k3_decode_dependency_local_kernel"
BARRIER_SCHEDULE_KERNEL = "kimi_k3_decode_persistent_kernel"

#: The gate/up engine id production compiles that schedule with.
#:
#: `kimi_k3_decode::expert_mxfp4::fused_w13::kEngineFusedAdaptive`. Spelled
#: rather than read through the extension, because this module is pure so that a
#: CPU can hold it to captured names, and because the point of the check is that
#: the launched kernel's own mangling agrees with the constant.
PRODUCTION_GATE_UP_ENGINE = 2

# `<false, 2, ...>` demangled, `ILb0ELi2E` mangled. The capacity flag comes
# first in both and is not what is being read: both capacity paths are
# production, and which one a shape takes is the shape's business.
_DEMANGLED = re.compile(
    re.escape(DEPENDENCY_LOCAL_KERNEL)
    + r"<\s*(?:false|true)\s*,\s*(\d+)\s*[,>]"
)
_MANGLED = re.compile(re.escape(DEPENDENCY_LOCAL_KERNEL) + r"ILb[01]ELi(\d+)E")


def dependency_local_engine(name: str) -> int | None:
    """The gate/up engine id one profiled kernel name compiled with.

    ``None`` when the name is not an instantiation of the dependency-local
    schedule at all. A name that is one but whose engine cannot be read is an
    error rather than a ``None``: it means the mangling changed, and silently
    reporting "not production" would turn that into a one-launch failure whose
    message pointed nowhere near the cause.
    """
    for pattern in (_DEMANGLED, _MANGLED):
        match = pattern.search(name)
        if match is not None:
            return int(match.group(1))
    if DEPENDENCY_LOCAL_KERNEL in name:
        raise AssertionError(
            f"the dependency-local schedule's engine argument could not be "
            f"read out of {name!r}; the check below cannot tell production "
            f"from a measured arm until this is understood"
        )
    return None


def is_production_launch(name: str) -> bool:
    """Whether one profiled kernel name is production's decode launch."""
    return dependency_local_engine(name) == PRODUCTION_GATE_UP_ENGINE


def assert_one_production_launch(names: list[str]) -> None:
    """Refuse anything but a single launch of production's decode kernel."""
    if len(names) != 1 or not is_production_launch(names[0]):
        raise AssertionError(names)
