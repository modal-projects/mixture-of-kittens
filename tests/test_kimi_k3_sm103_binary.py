"""The SM103 binary contract for the persistent Kimi K3 decode kernel.

Everything else about the megakernel is measured by running it. Two of its
load-bearing properties are not observable that way, because a build that lost
them still produces the right numbers -- just slowly, or on a grid that no
longer co-resides:

* No stack frame, no local memory, and no ``LDL``/``STL`` anywhere in either
  instantiation. A spilled kernel is correct and several times slower, and the
  register budget is what decides whether 148 CTAs of 256 threads fit one per
  SM at all.
* The mixed MXFP4-by-MXFP8 contraction really is the native tensor-core
  instruction. Nothing in the numerics distinguishes ``UTCQMMA`` from an
  unrolled CUDA-core emulation of the same arithmetic.

So this file reads the built extension instead: ``cuobjdump
--dump-resource-usage`` for the register, stack, and shared-memory contract,
and symbol-bounded ``cuobjdump -sass`` for the instruction classes each
instantiation has to contain and must not. Classes, not counts -- a retiling is
allowed to change how many ``UTCQMMA`` the kernel issues, and is not allowed to
stop issuing them.

Only the built binary and the local device are needed, so this file runs with
or without ``torchrun``.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from mok import _C


PERSISTENT_SYMBOL = "kimi_k3_decode_persistent_kernel"

# The private per-stage entry point the MXFP4 unit tests drive. It compiles the
# same routed gate/up and down units the persistent kernel inlines, at the same
# register ceiling, so it is the first place register pressure shows up.
ROUTED_EXPERTS_SYMBOL = "kimi_k3_routed_experts_kernel"

# What that entry point's stack frame may be, in bytes. Zero is the goal and is
# what CUDA 13.3's ptxas already produces; the image this repository ships in
# builds with 13.2, whose allocator spills 48 bytes here. It spilled 56 before
# the gate/up prefetch was folded into the buffer it replaces, and the header
# split changed it by nothing -- both measured with 13.2 on the same source.
# Neither persistent instantiation spills under either toolchain, which is what
# the two tests above assert unconditionally; this is a regression bound on the
# pressure, not a claim that 48 is acceptable.
PRIVATE_ROUTED_STACK_CEILING = 48

# Itanium mangling spells all three compile-time selections out: tensor path,
# gate/up group size, and gate/up-to-down pipelining. This lets the shipped
# readiness pair and the benchmark-only baseline pair be checked independently.
PRODUCTION_CORE_MANGLING = "ILb0ELi0ELb1EE"
PRODUCTION_TENSOR_MANGLING = "ILb1ELi0ELb1EE"
BASELINE_CORE_MANGLING = "ILb0ELi0ELb0EE"
BASELINE_TENSOR_MANGLING = "ILb1ELi0ELb0EE"

# Blackwell SASS, from the SM103 build. ``UTCQMMA`` is the native mixed
# MXFP4-by-MXFP8 tcgen05 contraction, ``UTCHMMA`` its BF16 sibling, ``LDGMC``
# the multimem load-reduce the tail's all-reduce is, ``REDG`` the multimem
# reduction it scatters with, ``BPT`` the trap a timed-out wait ends in, and
# ``NANOSLEEP`` the backoff every bounded wait spins on.
REQUIRED_EVERYWHERE = frozenset(
    {"UTCQMMA", "LDGMC", "REDG", "BPT", "NANOSLEEP", "ATOMG", "MEMBAR", "BAR"}
)
REQUIRED_TENSOR_ONLY = frozenset({"UTCHMMA", "UTMALDG", "LDTM"})

# Local memory in either direction. Their absence is the same claim as
# ``STACK:0 LOCAL:0``, made against the instruction stream rather than the
# compiler's own summary.
FORBIDDEN = frozenset({"LDL", "STL"})

DECODE_THREADS = 256


@pytest.fixture(scope="module")
def extension_path() -> Path:
    if shutil.which("cuobjdump") is None:
        pytest.skip("the SM103 binary contract needs cuobjdump")
    if not torch.cuda.is_available():
        pytest.skip("the SM103 binary contract needs a CUDA device")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("the SM103 binary contract needs an SM103 B300")
    return Path(_C.__file__)


@functools.cache
def _resource_usage(path: Path) -> dict[str, dict[str, int]]:
    """Every device function in the fatbin, with its per-thread resources.

    Cached because the extension is a hundred megabytes and every test below
    wants the same table out of it.
    """
    dump = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    usage: dict[str, dict[str, int]] = {}
    for name, line in re.findall(r"Function (\S+):\s*\n\s*(REG:.*)", dump):
        usage[name] = {
            key: int(value)
            for key, value in re.findall(r"([A-Z]+):(\d+)", line)
        }
    return usage


@functools.cache
def _mnemonics(path: Path, mangled: str) -> frozenset[str]:
    """The instruction families one symbol's SASS is built from.

    ``-fun`` bounds the disassembly to the one symbol, which is what keeps this
    from accidentally reading a neighbouring kernel's instructions. The family
    is the mnemonic before its first modifier, so ``LDG.E.128`` and ``LDG.E``
    are the same claim.
    """
    dump = subprocess.run(
        ["cuobjdump", "-sass", "-fun", mangled, str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    families = {
        match.group(2).split(".")[0]
        for line in dump.splitlines()
        if (
            match := re.match(
                r"\s+/\*[0-9a-f]{4,}\*/\s+(@!?\S+\s+)?([A-Z][A-Z0-9_.]*)", line
            )
        )
    }
    assert families, mangled
    return frozenset(families)


@pytest.fixture(scope="module")
def persistent_symbols(extension_path: Path) -> dict[str, str]:
    """The two persistent instantiations, keyed by which capacity path they are."""
    usage = _resource_usage(extension_path)
    found = [name for name in usage if PERSISTENT_SYMBOL in name]
    symbols = {
        "core": next(
            name for name in found if PRODUCTION_CORE_MANGLING in name
        ),
        "tensor": next(
            name for name in found if PRODUCTION_TENSOR_MANGLING in name
        ),
    }
    assert symbols["core"] != symbols["tensor"]
    return symbols


@pytest.fixture(scope="module")
def baseline_symbols(extension_path: Path) -> dict[str, str]:
    """The benchmark baseline pair, keyed by which capacity path they are."""
    usage = _resource_usage(extension_path)
    found = [name for name in usage if PERSISTENT_SYMBOL in name]
    symbols = {
        "core": next(
            name for name in found if BASELINE_CORE_MANGLING in name
        ),
        "tensor": next(
            name for name in found if BASELINE_TENSOR_MANGLING in name
        ),
    }
    assert symbols["core"] != symbols["tensor"]
    return symbols

@pytest.mark.parametrize("path_name", ["core", "tensor"])
def test_neither_instantiation_spills(
    extension_path: Path,
    persistent_symbols: dict[str, str],
    path_name: str,
) -> None:
    """A spill is invisible in the numbers and ruinous in the latency.

    The routed expert stage is the register-hungriest thing the kernel runs and
    every CTA is compiled for it, so a spill would be the expected failure mode
    rather than a surprising one -- which is exactly why it is asserted rather
    than assumed.
    """
    usage = _resource_usage(extension_path)[persistent_symbols[path_name]]
    assert usage["STACK"] == 0, usage
    assert usage["LOCAL"] == 0, usage
    dynamic_shared = _C._kimi_k3_decode_grid_shape()[2]
    properties = torch.cuda.get_device_properties(0)
    assert dynamic_shared + usage["SHARED"] <= (
        properties.shared_memory_per_block_optin
    ), usage
    assert _C._kimi_k3_decode_resident_blocks_per_sm(
        path_name == "tensor"
    ) == 1


def test_the_private_routed_expert_kernel_stays_under_its_measured_spill(
    extension_path: Path,
) -> None:
    """The same units, compiled on their own, with the same register budget.

    Nothing the decode step launches calls this entry point, but it inlines the
    routed gate/up and down units the persistent kernel does at the same 255
    registers, with none of the persistent kernel's surrounding code to give
    the scheduler room. It therefore spills where the persistent
    instantiations do not, and how much it spills is the most sensitive
    reading of those units' register pressure the build produces.

    The ceiling is a measurement, not a target, and it is toolchain-dependent:
    see ``PRIVATE_ROUTED_STACK_CEILING``.
    """
    usage = _resource_usage(extension_path)
    found = [name for name in usage if ROUTED_EXPERTS_SYMBOL in name]
    assert len(found) == 1, found
    assert usage[found[0]]["STACK"] <= PRIVATE_ROUTED_STACK_CEILING, usage[found[0]]


@pytest.mark.parametrize("path_name", ["core", "tensor"])
def test_neither_instantiation_touches_local_memory(
    extension_path: Path,
    persistent_symbols: dict[str, str],
    path_name: str,
) -> None:
    """The same claim as ``STACK:0``, read off the instructions themselves."""
    families = _mnemonics(extension_path, persistent_symbols[path_name])
    assert families.isdisjoint(FORBIDDEN), sorted(families & FORBIDDEN)


@pytest.mark.parametrize("path_name", ["core", "tensor"])
def test_baseline_candidate_has_no_spills_or_local_memory(
    extension_path: Path,
    baseline_symbols: dict[str, str],
    path_name: str,
) -> None:
    """The retained benchmark baseline must stay safe while verifying prod."""
    symbol = baseline_symbols[path_name]
    usage = _resource_usage(extension_path)[symbol]
    assert usage["STACK"] == 0, usage
    assert usage["LOCAL"] == 0, usage
    families = _mnemonics(extension_path, symbol)
    assert families.isdisjoint(FORBIDDEN), sorted(families & FORBIDDEN)
    assert "UTCQMMA" in families
    resource = _C._kimi_k3_decode_gate_up_group_resource(
        path_name == "tensor", 0, False
    )
    assert resource[1] == 1


@pytest.mark.parametrize("path_name", ["core", "tensor"])
def test_each_instantiation_issues_the_instructions_its_stages_need(
    extension_path: Path,
    persistent_symbols: dict[str, str],
    path_name: str,
) -> None:
    """Native mixed MMA, native multimem, and a real bounded wait, per path.

    Both paths run the mixed W4A8 routed experts and the fused TP8 tail, so
    both must carry ``UTCQMMA`` and the two multimem opcodes. Only the tcgen05
    path runs the BF16 contractions and the TMA loads that feed them, so the
    BF16 MMA and ``UTMALDG`` are required of it alone -- requiring them of the
    CUDA-core path would be asserting the opposite of what that path is for.
    """
    families = _mnemonics(extension_path, persistent_symbols[path_name])
    required = REQUIRED_EVERYWHERE
    if path_name == "tensor":
        required |= REQUIRED_TENSOR_ONLY
    assert required <= families, sorted(required - families)
    if path_name == "core":
        # The CUDA-core path is the fallback precisely because it does not use
        # the BF16 tensor cores; if it started to, the two paths would no
        # longer be testing different things.
        assert "UTCHMMA" not in families


def test_the_grid_the_binary_supports_is_the_grid_the_host_launches(
    extension_path: Path,
    persistent_symbols: dict[str, str],
) -> None:
    """148 CTAs of 256 threads, one per SM, has to survive the compiler.

    The host proves residency with an occupancy query before every first
    launch, but the query answers for whatever the compiler produced. These are
    the two budgets that decide the answer, checked against the device's own
    limits: 256 threads at this register count have to fit one SM's register
    file, and the dynamic request plus the static shared the stages declare
    have to fit one SM's opt-in shared memory while leaving no room for a
    second CTA.
    """
    ctas, threads, dynamic_shared = _C._kimi_k3_decode_grid_shape()
    properties = torch.cuda.get_device_properties(0)
    usage = _resource_usage(extension_path)

    assert (ctas, threads) == (148, DECODE_THREADS)
    assert properties.multi_processor_count >= ctas

    for path_name, mangled in persistent_symbols.items():
        entry = usage[mangled]
        registers = entry["REG"] * threads
        assert registers <= properties.regs_per_multiprocessor, (
            path_name,
            entry,
        )
        occupied = dynamic_shared + entry["SHARED"]
        assert occupied <= properties.shared_memory_per_block_optin, (
            path_name,
            occupied,
        )
        # More than half an SM's shared memory, so a second CTA cannot land on
        # the same SM whatever the occupancy heuristic decides.
        assert 2 * occupied > properties.shared_memory_per_multiprocessor, (
            path_name,
            occupied,
        )

    for tensor_path in (False, True):
        assert _C._kimi_k3_decode_resident_blocks_per_sm(tensor_path) == 1
