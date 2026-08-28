"""CPU contract tests for the Kimi K3 public and custom-operator APIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from .kimi_k3_api_contract import CHECKS, MXFP4_LAYOUTS, RESULT_MARKER


@pytest.fixture(scope="session")
def api_contract_results() -> dict[str, dict[str, str]]:
    """Run the stub-backed contract checks once, in a process of their own.

    ``tests/kimi_k3_api_contract.py`` explains why: the stubs it installs to run
    without a compiled extension cannot be uninstalled, so they must not be
    installed in the process that also runs the GPU test files.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "tests.kimi_k3_api_contract"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    raise AssertionError(
        "the Kimi K3 API contract checks reported no results "
        f"(exit code {completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )


@pytest.mark.parametrize("check", list(CHECKS))
def test_api_contract(
    api_contract_results: dict[str, dict[str, str]], check: str
) -> None:
    result = api_contract_results[check]
    assert result["outcome"] == "passed", result["detail"]


def test_api_contract_pins_native_k32_expert_layouts() -> None:
    """The contract data itself must stay on native K, not padded K.

    Every rejection case above is derived from this table, so a silent return to
    the K=3648 padding would make those cases test the wrong contract.
    """
    assert dict(MXFP4_LAYOUTS) == {
        "expert_w1_packed": (896, 384, 1792),
        "expert_w1_scale": (896, 384, 112),
        "expert_w3_packed": (896, 384, 1792),
        "expert_w3_scale": (896, 384, 112),
        "expert_w2_packed": (896, 3584, 192),
        "expert_w2_scale": (896, 3584, 12),
    }


def test_api_contract_does_not_stub_the_mok_package() -> None:
    """The stubs must stay in the child: later GPU files import the real ``mok``.

    The synthetic package and extension stub are the only ``mok`` modules the
    contract checks build without a source file behind them, so a missing
    ``__file__`` here is exactly the leak that used to break later GPU files.
    """
    for name in ("mok", "mok._C"):
        module = sys.modules.get(name)
        assert module is None or getattr(module, "__file__", None) is not None


def test_workspace_requires_sm103(
    tp8_context: tuple[int, int, torch.device],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mok.kimi_k3 import create_kimi_k3_decode_workspace

    _, _, device = tp8_context
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda selected_device: (10, 0),
    )

    with pytest.raises(NotImplementedError, match="SM103"):
        create_kimi_k3_decode_workspace(dist.group.WORLD, device=device)


def test_workspace_requires_tp8(
    tp8_context: tuple[int, int, torch.device],
) -> None:
    from mok.kimi_k3 import create_kimi_k3_decode_workspace

    rank, _, device = tp8_context
    subgroup_ranks = (list(range(4)), list(range(4, 8)))
    subgroups = [dist.new_group(ranks=ranks) for ranks in subgroup_ranks]
    subgroup = subgroups[rank // 4]
    try:
        with pytest.raises(ValueError, match="TP8"):
            create_kimi_k3_decode_workspace(subgroup, device=device)
    finally:
        dist.barrier()
        for process_group in subgroups:
            if isinstance(process_group, dist.ProcessGroup):
                dist.destroy_process_group(process_group)


def test_workspace_requires_128_max_tokens(
    tp8_context: tuple[int, int, torch.device],
) -> None:
    from mok.kimi_k3 import create_kimi_k3_decode_workspace

    _, _, device = tp8_context
    with pytest.raises(ValueError, match="max_tokens must equal 128"):
        create_kimi_k3_decode_workspace(
            dist.group.WORLD,
            device=device,
            max_tokens=64,
        )


@pytest.mark.filterwarnings(
    "error:`enable_symm_mem_for_group` is deprecated.*:FutureWarning"
)
def test_workspace_create_is_caller_owned_with_canonical_layout(
    tp8_context: tuple[int, int, torch.device],
) -> None:
    from mok.kimi_k3 import (
        create_kimi_k3_decode_workspace,
        get_kimi_k3_decode_workspace,
    )
    from mok.ops import barrier_all

    rank, world_size, device = tp8_context
    created = create_kimi_k3_decode_workspace(
        dist.group.WORLD,
        device=device,
    )
    cached = get_kimi_k3_decode_workspace(
        dist.group.WORLD,
        device=device,
    )
    try:
        assert created is not cached
        assert created.group_name == dist.group.WORLD.group_name
        assert created.tp_rank == rank
        assert created.tp_size == world_size == 8
        assert created.device == device
        assert created.max_tokens == 128

        assert created.scratch.shape == (3_749_376,)
        assert created.scratch.dtype == torch.uint8
        assert created.collective_buffer.shape == (128, 10752)
        assert created.collective_buffer.dtype == torch.bfloat16
        assert created.output_mailbox.shape == (8, 128, 896)
        assert created.output_mailbox.dtype == torch.bfloat16
        assert created.barrier_buffer.shape == (1,)
        assert created.barrier_buffer.dtype == torch.int32
        assert created.barrier_target.shape == (1,)
        assert created.barrier_target.dtype == torch.int32
        assert created.error_flag.shape == (1,)
        assert created.error_flag.dtype == torch.int32

        assert created.collective_handle is not None
        assert created.output_mailbox_handle is not None
        assert created.barrier_handle is not None
        assert len(created.collective_ptrs) == 8
        assert len(created.output_mailbox_ptrs) == 8
        assert len(created.barrier_ptrs) == 8
        assert all(pointer > 0 for pointer in created.collective_ptrs)
        assert all(pointer > 0 for pointer in created.output_mailbox_ptrs)
        assert all(pointer > 0 for pointer in created.barrier_ptrs)
        assert created.collective_multicast_ptr > 0
        assert created.barrier_multicast_ptr > 0
    finally:
        barrier_all(
            created.barrier_buffer,
            created.barrier_ptrs,
            created.barrier_multicast_ptr,
            created.barrier_target,
        )
        torch.cuda.synchronize(device)


@pytest.mark.filterwarnings(
    "error:`enable_symm_mem_for_group` is deprecated.*:FutureWarning"
)
def test_workspace_cache_reuses_entry_and_clear_replaces_it(
    tp8_context: tuple[int, int, torch.device],
) -> None:
    from mok.kimi_k3 import (
        clear_kimi_k3_decode_workspace_cache,
        get_kimi_k3_decode_workspace,
    )

    _, _, device = tp8_context
    first = get_kimi_k3_decode_workspace(dist.group.WORLD, device=device)
    second = get_kimi_k3_decode_workspace(dist.group.WORLD, device=device)
    assert first is second

    clear_kimi_k3_decode_workspace_cache()
    replacement = get_kimi_k3_decode_workspace(
        dist.group.WORLD,
        device=device,
    )
    assert replacement is not first
