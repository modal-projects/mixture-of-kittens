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

The routed gate/up phase is the fused-W13 K512 engine, so the copy-engine
transfer that phase exists for is a property of the production instantiations
themselves and is asserted on both capacity paths at the end of the file.

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

# Itanium mangling spells ``TENSOR_PATH`` out, so the two instantiations can be
# told apart. There are exactly two: the template carried a gate/up engine
# selector while three candidates were being measured against each other, and
# the winner is the only gate/up unit the kernel has, so the selector is gone
# along with the six instantiations it multiplied the build by.
PRODUCTION_CORE_MANGLING = "ILb0EE"
PRODUCTION_TENSOR_MANGLING = "ILb1EE"

# Blackwell SASS, from the SM103 build. ``UTCQMMA`` is the native mixed
# MXFP4-by-MXFP8 tcgen05 contraction, ``UTCHMMA`` its BF16 sibling, ``LDGMC``
# the multimem load-reduce the tail's all-reduce is, ``REDG`` the multimem
# reduction it scatters with, ``BPT`` the trap a timed-out wait ends in, and
# ``NANOSLEEP`` the backoff every bounded wait spins on.
#
# ``UTMALDG`` and ``LDTM`` are required of *both* paths now. They used to be the
# tcgen05 projection's alone, so the CUDA-core path carried neither; the routed
# gate/up engine moves every weight slab by ``cp.async.bulk.tensor`` and reads
# its accumulator out of tensor memory, and every CTA is compiled for that phase
# whichever capacity path it took.
REQUIRED_EVERYWHERE = frozenset(
    {
        "UTCQMMA",
        "UTMALDG",
        "LDTM",
        "LDGMC",
        "REDG",
        "BPT",
        "NANOSLEEP",
        "ATOMG",
        "MEMBAR",
        "BAR",
    }
)
REQUIRED_TENSOR_ONLY = frozenset({"UTCHMMA"})

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
def _mnemonic_counts(path: Path, mangled: str) -> dict[str, int]:
    """How many instructions of each family one symbol's SASS holds.

    Counts are static, so they say what the compiler emitted rather than what a
    launch executes. That is the right reading for a structural claim: the
    production and fused instantiations of the persistent template differ in
    exactly one inlined phase, so a family whose count moves between them moved
    because that phase changed.
    """
    dump = subprocess.run(
        ["cuobjdump", "-sass", "-fun", mangled, str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    counts: dict[str, int] = {}
    for line in dump.splitlines():
        match = re.match(
            r"\s+/\*[0-9a-f]{4,}\*/\s+(@!?\S+\s+)?([A-Z][A-Z0-9_.]*)", line
        )
        if match:
            family = match.group(2).split(".")[0]
            counts[family] = counts.get(family, 0) + 1
    assert counts, mangled
    return counts


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
    """The two instantiations, keyed by which capacity path they are.

    Both halves are asserted: the build has exactly two of them, and they are
    the two capacity paths. A build that grew a third is a build that grew a
    switch, which is the thing removing the engine selector was for.
    """
    usage = _resource_usage(extension_path)
    found = [name for name in usage if PERSISTENT_SYMBOL in name]
    assert len(found) == 2, found
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


# ---------------------------------------------------------------------------
# The routed gate/up phase: the fused-W13 K512 engine.
#
# The phase is no longer a candidate compiled beside production, so its
# properties are the production instantiations' properties. Two of them are only
# visible in the binary: that the weight slabs really move by copy engine rather
# than by threads naming bytes, and that the ring's footprint still leaves one
# CTA per SM once the static shared memory ptxas assigned is counted.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fused_geometry() -> dict[str, int]:
    return _C._kimi_k3_fused_w13_geometry()


def test_the_gate_up_phase_is_the_six_task_K512_decomposition(
    fused_geometry: dict[str, int],
) -> None:
    """Six tasks of 128 rows, seven K = 512 slabs, one 42-long weight stream.

    The predecessor kept three output tiles, two accumulators, and staged every
    weight byte by hand. These are the numbers that make this a different phase
    rather than that one rebased, and ``tests/test_kimi_k3_w13.py`` holds the
    prepared bytes to the same set -- which is what makes the payload and the
    engine one geometry rather than two that happen to agree.
    """
    assert fused_geometry["tasks"] == 6
    assert fused_geometry["slabs"] == 7
    assert fused_geometry["slab_k"] == 512
    assert fused_geometry["m"] == 128
    assert fused_geometry["half_rows"] == 64
    # 384 situ columns per rank, so six tasks of 64 paired output channels.
    assert fused_geometry["tasks"] * fused_geometry["half_rows"] == 384
    # K = 3584 exactly, with no partial slab.
    assert fused_geometry["slabs"] * fused_geometry["slab_k"] == 3584
    assert fused_geometry["boxes"] == 4
    assert fused_geometry["box_elements"] == 128
    assert fused_geometry["swizzle_bytes"] == 128

    # Two weight stages, the whole of one expert's activation resident, and one
    # 42-long stream that does not restart at a task boundary.
    assert fused_geometry["stages"] == 2
    assert fused_geometry["activation_slabs"] == 7
    assert fused_geometry["stream_length"] == 42


def test_the_ring_asks_for_exactly_the_bytes_the_grid_launches_with(
    fused_geometry: dict[str, int],
) -> None:
    """The whole grid's request is this phase's ring, to the byte.

    Every other stage fits inside it with slack, so this is the one stage whose
    arithmetic decides the launch configuration -- and the launch configuration
    is what decides whether 148 CTAs land one per SM.
    """
    assert fused_geometry["staging_bytes"] == (
        2 * 65536 + 2 * 2048 + 7 * 8192 + 7 * 2048 + 8192
    )
    assert fused_geometry["staging_bytes"] == 215040
    # Plus one allocator grain, because the dynamic block does not begin on one.
    assert fused_geometry["allocator_padding"] == 1024
    assert fused_geometry["shared_bytes"] == (
        fused_geometry["staging_bytes"] + fused_geometry["allocator_padding"]
    )
    assert fused_geometry["shared_bytes"] == 216064
    assert _C._kimi_k3_decode_grid_shape()[2] == fused_geometry["shared_bytes"]


def test_the_allocator_skip_the_ring_pays_for_is_the_skip_it_measures(
    fused_geometry: dict[str, int],
) -> None:
    """Read off the device, not off the header's reasoning.

    ``tma_swizzle_allocator`` aligns by *absolute* shared address and the dynamic
    block does not start on a 1 KiB boundary, so a ring sized to the byte
    overruns its block by however far the driver's base is past one. The overrun
    is invisible in the arithmetic, so the offset is measured.
    """
    consumed, base_offset, launched = _C._kimi_k3_fused_w13_shared_footprint()
    assert launched == fused_geometry["shared_bytes"]
    assert 0 <= base_offset < fused_geometry["allocator_padding"]
    assert consumed <= launched, (consumed, launched)
    # The ring's own bytes, plus whatever the first `align_ptr` skipped.
    assert consumed >= fused_geometry["staging_bytes"]
    assert consumed - fused_geometry["staging_bytes"] < (
        fused_geometry["allocator_padding"]
    )


@pytest.mark.parametrize("path_name", ["core", "tensor"])
def test_the_gate_up_weights_move_by_copy_engine_on_both_paths(
    extension_path: Path,
    persistent_symbols: dict[str, str],
    fused_geometry: dict[str, int],
    path_name: str,
) -> None:
    """``UTMALDG`` in the CUDA-core instantiation is the phase's own transfer.

    The core path runs no BF16 tcgen05 projection at all, so it used to compile
    no TMA whatsoever. Every ``UTMALDG`` in it now belongs to the gate/up ring:
    one ``cp.async.bulk.tensor.5d`` per 128x512 slab and one ``cp.async.bulk``
    per 2 KiB of scales, with no weight byte ever named in a register.
    """
    counts = _mnemonic_counts(extension_path, persistent_symbols[path_name])
    assert counts.get("UTMALDG", 0) > 0, path_name
    # One slab body is sixteen unrolled K = 32 contractions, and the routed-down
    # pipeline contributes its own on top. A build that quietly retiled to a
    # narrower K would emit fewer.
    groups_per_slab = fused_geometry["slab_k"] // 32
    assert groups_per_slab == 16
    assert counts.get("UTCQMMA", 0) >= groups_per_slab, (
        path_name,
        counts.get("UTCQMMA", 0),
    )
    # And the accumulator is read back out of tensor memory rather than through
    # a shared round trip.
    assert counts.get("LDTM", 0) > 0, path_name


def test_a_third_K512_weight_stage_does_not_fit_the_static_shared_ptxas_assigned(
    extension_path: Path,
    persistent_symbols: dict[str, str],
    fused_geometry: dict[str, int],
) -> None:
    """Why the ring is two deep, argued against the real ceiling.

    A third stage is the obvious way to spend the gate/up phase's remaining TMA
    wait, and it does not fit. The reserve the header argues from is a rounded
    figure; this is the same argument against the static shared memory ptxas
    actually assigned to these instantiations, so the shortfall does not rest
    on a placeholder.

    The floor charged for the activation is the resident whole -- seven slabs and
    their scales -- because moving the gather back inside the task loop is what
    the two-stage shape was measured against and cost 5.2% of the step.
    """
    properties = torch.cuda.get_device_properties(0)
    static = max(
        _resource_usage(extension_path)[symbol]["SHARED"]
        for symbol in persistent_symbols.values()
    )
    # A block's opt-in maximum covers static and dynamic together, so the
    # header's reserve is what the launch figure has to leave unasked-for. It
    # has nothing to do with the allocator's grain, which is a skip inside the
    # dynamic block rather than something below its base.
    assert 0 < static <= fused_geometry["static_shared_reserve"], static
    assert (
        fused_geometry["shared_bytes"] + static
        <= properties.shared_memory_per_block_optin
    )

    ceiling = (
        properties.shared_memory_per_block_optin
        - static
        - fused_geometry["allocator_padding"]
    )
    three_stages = 3 * 65536 + 3 * 2048
    activation = 7 * 8192 + 7 * 2048
    result_tile = 8192
    assert three_stages + activation + result_tile > ceiling, (
        static,
        ceiling,
        three_stages + activation + result_tile,
    )
    # And the two-stage ring clears the same ceiling, so the shortfall is the
    # third stage's and not the shape's.
    assert fused_geometry["staging_bytes"] <= ceiling, (static, ceiling)
