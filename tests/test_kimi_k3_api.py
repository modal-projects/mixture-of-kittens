"""CPU contract tests for the Kimi K3 public and custom-operator APIs."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode


_MODULES: tuple[ModuleType, ModuleType, ModuleType] | None = None
_WEIGHT_FIELDS = (
    "router_weight",
    "router_correction_bias",
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "routed_latent_rmsnorm_weight",
    "expert_w1_packed",
    "expert_w1_scale",
    "expert_w3_packed",
    "expert_w3_scale",
    "expert_w2_packed",
    "expert_w2_scale",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
    "tp_rank",
)
_LOW_LEVEL_ARGUMENTS = (
    "hidden_states",
    *_WEIGHT_FIELDS[:-1],
    "scratch",
    "collective_buffer",
    "collective_buffer_ptrs",
    "collective_buffer_multicast_ptr",
    "output_mailbox",
    "output_mailbox_ptrs",
    "barrier_buffer",
    "barrier_buffer_ptrs",
    "barrier_buffer_multicast_ptr",
    "error_flag",
    "tp_rank",
    "active_tokens",
)


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_contract_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load source modules behind a controlled package and extension stub."""
    global _MODULES
    if _MODULES is not None:
        return _MODULES

    root = Path(__file__).parents[1]
    package_dir = root / "mok"
    package = ModuleType("mok")
    package.__path__ = [str(package_dir)]
    package.__package__ = "mok"
    extension = ModuleType("mok._C")
    extension.kimi_k3_decode = lambda hidden_states, *args: torch.empty_like(
        hidden_states
    )
    extension.kimi_k3_decode_workspace_bytes = lambda: 0
    package._C = extension
    sys.modules["mok"] = package
    sys.modules["mok._C"] = extension

    kimi_k3 = _load_source_module("mok.kimi_k3", package_dir / "kimi_k3.py")
    ops = _load_source_module("mok.ops", package_dir / "ops.py")
    fake_impls = _load_source_module(
        "mok._fake_impls", package_dir / "_fake_impls.py"
    )
    _MODULES = kimi_k3, ops, fake_impls
    return _MODULES


def _valid_weights(kimi_k3: ModuleType):
    meta_bf16 = lambda shape: torch.empty(  # noqa: E731
        shape, dtype=torch.bfloat16, device="meta"
    )
    meta_uint8 = lambda shape: torch.empty(  # noqa: E731
        shape, dtype=torch.uint8, device="meta"
    )
    return kimi_k3.KimiK3DecodeWeights(
        router_weight=meta_bf16((896, 7168)),
        router_correction_bias=torch.empty(896, dtype=torch.float32, device="meta"),
        routed_expert_down_proj=meta_bf16((3584, 7168)),
        routed_expert_up_proj=meta_bf16((7168, 3584)),
        routed_latent_rmsnorm_weight=meta_bf16((3584,)),
        expert_w1_packed=meta_uint8((896, 384, 1824)),
        expert_w1_scale=meta_uint8((896, 384, 114)),
        expert_w3_packed=meta_uint8((896, 384, 1824)),
        expert_w3_scale=meta_uint8((896, 384, 114)),
        expert_w2_packed=meta_uint8((896, 3584, 192)),
        expert_w2_scale=meta_uint8((896, 3584, 12)),
        shared_gate_proj=meta_bf16((768, 7168)),
        shared_up_proj=meta_bf16((768, 7168)),
        shared_down_proj=meta_bf16((7168, 768)),
        tp_rank=0,
    )


def test_weight_contract_has_exact_immutable_fields() -> None:
    kimi_k3, _, _ = _load_contract_modules()
    weight_type = kimi_k3.KimiK3DecodeWeights

    assert tuple(field.name for field in fields(weight_type)) == _WEIGHT_FIELDS
    weights = _valid_weights(kimi_k3)
    with pytest.raises(FrozenInstanceError):
        weights.tp_rank = 1


def test_decode_accepts_canonical_prepared_layouts() -> None:
    kimi_k3, _, _ = _load_contract_modules()
    hidden_states = torch.empty(16, 7168, dtype=torch.bfloat16, device="meta")

    assert (
        kimi_k3.validate_kimi_k3_decode_inputs(
            hidden_states, _valid_weights(kimi_k3)
        )
        is None
    )


def test_decode_requires_tp8_sharded_shared_weights() -> None:
    kimi_k3, _, _ = _load_contract_modules()
    hidden_states = torch.empty(16, 7168, dtype=torch.bfloat16, device="meta")
    weights = _valid_weights(kimi_k3)
    sharded_weights = replace(
        weights,
        shared_gate_proj=torch.empty(
            768, 7168, dtype=torch.bfloat16, device="meta"
        ),
        shared_up_proj=torch.empty(
            768, 7168, dtype=torch.bfloat16, device="meta"
        ),
        shared_down_proj=torch.empty(
            7168, 768, dtype=torch.bfloat16, device="meta"
        ),
    )

    assert (
        kimi_k3.validate_kimi_k3_decode_inputs(hidden_states, sharded_weights)
        is None
    )
    full_width_weights = replace(
        sharded_weights,
        shared_gate_proj=torch.empty(
            6144, 7168, dtype=torch.bfloat16, device="meta"
        ),
    )
    with pytest.raises(
        ValueError, match=r"shared_gate_proj must have shape \(768, 7168\)"
    ):
        kimi_k3.validate_kimi_k3_decode_inputs(hidden_states, full_width_weights)


@pytest.mark.parametrize(
    ("field_name", "expected_shape"),
    [
        ("expert_w1_packed", (896, 384, 1824)),
        ("expert_w1_scale", (896, 384, 114)),
        ("expert_w3_packed", (896, 384, 1824)),
        ("expert_w3_scale", (896, 384, 114)),
        ("expert_w2_packed", (896, 3584, 192)),
        ("expert_w2_scale", (896, 3584, 12)),
    ],
)
def test_decode_rejects_noncanonical_mxfp4_layout(
    field_name: str, expected_shape: tuple[int, ...]
) -> None:
    kimi_k3, _, _ = _load_contract_modules()
    hidden_states = torch.empty(16, 7168, dtype=torch.bfloat16, device="meta")
    weights = _valid_weights(kimi_k3)
    invalid = torch.empty((*expected_shape[:-1], expected_shape[-1] - 1),
                          dtype=torch.uint8, device="meta")

    with pytest.raises(ValueError, match=field_name):
        kimi_k3.validate_kimi_k3_decode_inputs(
            hidden_states, replace(weights, **{field_name: invalid})
        )


@pytest.mark.parametrize("tokens", [1, 128])
def test_decode_accepts_token_count_at_contract_bounds(tokens: int) -> None:
    kimi_k3, _, _ = _load_contract_modules()
    hidden_states = torch.empty(tokens, 7168, dtype=torch.bfloat16, device="meta")

    assert kimi_k3.validate_kimi_k3_decode_hidden_states(hidden_states) is None


@pytest.mark.parametrize("tokens", [0, 129])
def test_decode_rejects_token_count_outside_contract(tokens: int) -> None:
    kimi_k3, _, _ = _load_contract_modules()
    hidden_states = torch.empty(tokens, 7168, dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError, match=r"between 1 and 128"):
        kimi_k3.validate_kimi_k3_decode_hidden_states(hidden_states)


@pytest.mark.parametrize(
    "hidden_states",
    [
        torch.empty(8, 7167, dtype=torch.bfloat16, device="meta"),
        torch.empty(8, 7168, 1, dtype=torch.bfloat16, device="meta"),
    ],
)
def test_decode_rejects_noncontract_hidden_shape(
    hidden_states: torch.Tensor,
) -> None:
    kimi_k3, _, _ = _load_contract_modules()

    with pytest.raises(ValueError, match=r"shape \[M, 7168\]"):
        kimi_k3.validate_kimi_k3_decode_hidden_states(hidden_states)


def test_decode_rejects_non_bf16_hidden_states() -> None:
    kimi_k3, _, _ = _load_contract_modules()
    hidden_states = torch.empty(8, 7168, dtype=torch.float32, device="meta")

    with pytest.raises(TypeError, match="torch.bfloat16"):
        kimi_k3.validate_kimi_k3_decode_hidden_states(hidden_states)


def test_decode_rejects_noncontiguous_hidden_states() -> None:
    kimi_k3, _, _ = _load_contract_modules()
    hidden_states = torch.empty(
        7168, 8, dtype=torch.bfloat16, device="meta"
    ).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous"):
        kimi_k3.validate_kimi_k3_decode_hidden_states(hidden_states)


def _fake_operator_args(hidden_states: torch.Tensor) -> tuple[object, ...]:
    weights = tuple(hidden_states.new_empty((1,)) for _ in range(14))
    scratch = hidden_states.new_empty((1,), dtype=torch.uint8)
    collective_buffer = hidden_states.new_empty((1,))
    output_mailbox = hidden_states.new_empty((1,))
    barrier_buffer = hidden_states.new_empty((1,), dtype=torch.int32)
    error_flag = hidden_states.new_empty((1,), dtype=torch.int32)
    return (
        hidden_states,
        *weights,
        scratch,
        collective_buffer,
        [1] * 8,
        1,
        output_mailbox,
        [1] * 8,
        barrier_buffer,
        [1] * 8,
        1,
        error_flag,
        0,
        16,
    )


def test_fake_decode_preserves_active_shape() -> None:
    _, ops, _ = _load_contract_modules()
    mode = FakeTensorMode()
    with mode:
        hidden_states = torch.empty(
            16, 7168, dtype=torch.bfloat16, device="cuda"
        )
        output = ops.kimi_k3_decode(*_fake_operator_args(hidden_states))

    assert output.shape == (16, 7168)
    assert output.dtype == torch.bfloat16


def test_fake_decode_signature_matches_custom_op() -> None:
    _, _, fake_impls = _load_contract_modules()
    schema_names = tuple(
        argument.name
        for argument in torch.ops.mok.kimi_k3_decode.default._schema.arguments
    )

    assert schema_names == _LOW_LEVEL_ARGUMENTS
    assert tuple(
        inspect.signature(fake_impls._kimi_k3_decode_fake).parameters
    ) == schema_names


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

        assert created.scratch.shape == (256,)
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
