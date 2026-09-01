"""End-to-end cases: expert parallelism, workspace reuse, and shape limits.

Everything here runs the whole layer and compares it against the
reference. The two claims that are not about numbers -- the fake-tensor
metadata and what ``torch.compile`` does with this surface -- are in
``test_misc_fake_tensor`` and ``test_misc_compile``, and the case runner
all three share is in ``misc_support``.
"""

import math

import pytest
import torch
import torch.distributed as dist

from mok import functional, ops

from .misc_support import (
    _run_e2e_case,
)


def test_ep1_on_each_rank(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    singleton_groups = [
        dist.new_group(ranks=[group_rank])
        for group_rank in range(world_size)
    ]
    ep_group = singleton_groups[rank]
    assert isinstance(ep_group, dist.ProcessGroup)

    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=512,
        all_gather_top_experts_chunk_bytes=16,
    )
    functional.clear_workspace_cache()
    try:
        workspace = functional.get_workspace(
            config,
            ep_group,
            device=device,
            num_local_tokens=512,
            hidden_size=256,
            topk=1,
        )

        assert workspace.ep_rank == 0
        assert workspace.ep_size == 1
        for buffer, handle, pointers in (
            (
                workspace.x_buffer,
                workspace.x_buffer_handle,
                workspace.x_buffer_ptrs,
            ),
            (
                workspace.combine_buffer,
                workspace.combine_buffer_handle,
                workspace.combine_buffer_ptrs,
            ),
            (
                workspace.d_y_buffer,
                workspace.d_y_buffer_handle,
                workspace.d_y_buffer_ptrs,
            ),
            (
                workspace.d_x_routed_buffer,
                workspace.d_x_routed_buffer_handle,
                workspace.d_x_routed_buffer_ptrs,
            ),
            (
                workspace.router_weight_buffer,
                workspace.router_weight_buffer_handle,
                workspace.router_weight_buffer_ptrs,
            ),
            (
                workspace.d_router_weight_buffer,
                workspace.d_router_weight_buffer_handle,
                workspace.d_router_weight_buffer_ptrs,
            ),
        ):
            assert handle is None
            assert pointers == [buffer.data_ptr()]

        assert workspace.all_gather_top_experts_buffer_handle is None
        assert (
            workspace.all_gather_top_experts_buffer_multicast_ptr
            == workspace.all_gather_top_experts_buffer.data_ptr()
        )
        assert workspace.barrier_buffer_handle is None
        assert workspace.barrier_buffer_ptrs == [
            workspace.barrier_buffer.data_ptr()
        ]
        assert (
            workspace.barrier_buffer_multicast_ptr
            == workspace.barrier_buffer.data_ptr()
        )

        top_experts = torch.zeros(
            512,
            1,
            dtype=torch.int32,
            device=device,
        )
        ops.all_gather_top_experts(
            top_experts,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            0,
            16,
        )
        assert torch.equal(
            workspace.all_gather_top_experts_buffer[0],
            top_experts,
        )

        ops.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )
        torch.cuda.synchronize(device)
        assert int(workspace.barrier_buffer.item()) == 0
        assert int(workspace.barrier_target.item()) == 0

        _run_e2e_case(
            context,
            name="ep1-minimum",
            hidden_size=256,
            intermediate_size=256,
            num_local_experts=1,
            topk=1,
            config=config,
            precisions=("bf16", "mxfp8"),
            group=ep_group,
        )
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        for group in singleton_groups:
            if isinstance(group, dist.ProcessGroup):
                dist.destroy_process_group(group)


def test_e2e_fixed_first_topk_experts_default_capacity(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    functional.clear_workspace_cache()
    topk = 8
    try:
        _run_e2e_case(
            context,
            name="fixed-first-topk/default-capacity",
            hidden_size=7168,
            intermediate_size=2048,
            num_local_experts=4,
            topk=topk,
            config=functional.MoKConfig(
                minibatch_size=256,
                macrobatch_size=512,
            ),
            precisions=("bf16", "mxfp8"),
            top_experts=torch.arange(
                topk,
                device=device,
                dtype=torch.int64,
            ).expand(512, topk).contiguous(),
        )
    finally:
        functional.clear_workspace_cache()


def test_workspace_cache_reuse_and_isolation(
    context: tuple[int, int, torch.device],
) -> None:
    _, world_size, device = context
    functional.clear_workspace_cache()
    duplicate_world = dist.new_group(ranks=list(range(world_size)))
    kwargs = {
        "device": device,
        "num_local_tokens": 512,
        "hidden_size": 256,
        "topk": 1,
    }
    default_config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    larger_config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
        schedule_capacity_multiplier=1.5,
    )
    try:
        created_workspace = functional.create_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        default_workspace = functional.get_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        reused_workspace = functional.get_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        larger_workspace = functional.get_workspace(
            larger_config,
            dist.group.WORLD,
            **kwargs,
        )
        duplicate_group_workspace = functional.get_workspace(
            default_config,
            duplicate_world,
            **kwargs,
        )

        expected_default_capacity = 512 * max(
            2,
            math.ceil(world_size * default_config.schedule_capacity_multiplier),
        )
        expected_larger_capacity = 512 * max(
            2,
            math.ceil(world_size * larger_config.schedule_capacity_multiplier),
        )
        assert created_workspace is not default_workspace
        assert reused_workspace is default_workspace
        assert larger_workspace is not default_workspace
        assert duplicate_group_workspace is not default_workspace
        assert default_workspace.schedule_capacity == expected_default_capacity
        assert larger_workspace.schedule_capacity == expected_larger_capacity
        assert duplicate_group_workspace.group_name == duplicate_world.group_name
        assert duplicate_group_workspace.group_name != default_workspace.group_name
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        if isinstance(duplicate_world, dist.ProcessGroup):
            dist.destroy_process_group(duplicate_world)


def test_ep_subgroups_do_not_use_world(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    if world_size < 8 or world_size % 4 != 0:
        pytest.skip("requires a world size divisible by four and at least eight")

    functional.clear_workspace_cache()
    subgroup_ranks = [
        list(range(start, start + 4))
        for start in range(0, world_size, 4)
    ]
    subgroups = [dist.new_group(ranks=ranks) for ranks in subgroup_ranks]
    subgroup_index = rank // 4
    subgroup = subgroups[subgroup_index]
    subgroup_rank = rank % 4
    topk = 2 + subgroup_index % 2
    num_local_experts = 4
    config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    try:
        workspace = functional.get_workspace(
            config,
            subgroup,
            device=device,
            num_local_tokens=512,
            hidden_size=1024,
            topk=topk,
        )
        routes = (
            torch.arange(512 * topk, device=device, dtype=torch.int64)
            + subgroup_rank
        ).remainder(4 * num_local_experts).view(512, topk)
        schedule = functional.build_schedule(
            workspace,
            config,
            routes,
            num_local_experts=num_local_experts,
        )
        valid_peer_ranks = (
            (schedule.peer_rank == -1)
            | ((schedule.peer_rank >= 0) & (schedule.peer_rank < 4))
        )

        assert workspace.group_name == subgroup.group_name
        assert workspace.group_name != dist.group.WORLD.group_name
        assert workspace.ep_rank == subgroup_rank
        assert workspace.ep_size == 4
        assert workspace.topk == topk
        assert len(workspace.x_buffer_ptrs) == 4
        assert len(workspace.combine_buffer_ptrs) == 4
        assert len(workspace.barrier_buffer_ptrs) == 4
        assert bool(valid_peer_ranks.all().item())
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        for process_group in subgroups:
            if isinstance(process_group, dist.ProcessGroup):
                dist.destroy_process_group(process_group)


def test_supported_shape_alignment(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    pointers = [1, 2, 3, 4]

    def run_shape(
        hidden_size: int,
        intermediate_size: int,
        *,
        num_comm_sms: int,
    ) -> None:
        num_local_tokens = 512
        topk = 2

        def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, device=device, dtype=dtype)

        ops.dispatch_mlp_swiglu_combine_fwd_bf16(
            tensor((num_local_tokens, hidden_size), torch.bfloat16),
            pointers,
            tensor(
                (num_local_tokens * topk, hidden_size),
                torch.bfloat16,
            ),
            pointers,
            tensor((intermediate_size, hidden_size), torch.bfloat16),
            tensor((1, intermediate_size, hidden_size), torch.bfloat16),
            tensor((intermediate_size, hidden_size), torch.bfloat16),
            tensor((1, intermediate_size, hidden_size), torch.bfloat16),
            tensor((hidden_size, intermediate_size), torch.bfloat16),
            tensor((1, hidden_size, intermediate_size), torch.bfloat16),
            tensor((4096,), torch.int32),
            tensor((4096,), torch.int32),
            tensor((1,), torch.int32),
            tensor((1,), torch.int32),
            topk,
            None,
            num_comm_sms,
            512,
            256,
        )

    for hidden_size, intermediate_size in (
        (7168, 2048),
        (2048, 768),
        (2560, 2560),
        (1024, 1024),
        (1280, 1280),
    ):
        with pytest.raises(ValueError, match="num_comm_sms"):
            run_shape(
                hidden_size,
                intermediate_size,
                num_comm_sms=1,
            )
    for hidden_size, intermediate_size in (
        (640, 640),
        (1664, 1664),
        (1920, 1920),
        (128, 256),
        (0, 256),
        (256, 0),
    ):
        with pytest.raises(
            ValueError,
            match="hidden_size|intermediate_size",
        ):
            run_shape(
                hidden_size,
                intermediate_size,
                num_comm_sms=2,
            )


def test_e2e_requested_shapes_and_local_expert_counts(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for hidden_size, intermediate_size in (
            (7168, 2048),
            (2048, 768),
            (2560, 2560),
        ):
            for num_local_experts in (4, 6, 8, 12, 16, 24, 32):
                _run_e2e_case(
                    context,
                    name=(
                        f"shape-h{hidden_size}-i{intermediate_size}"
                        f"/local-experts-{num_local_experts}"
                    ),
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_local_experts=num_local_experts,
                    topk=8,
                    config=functional.MoKConfig(
                        minibatch_size=256,
                        macrobatch_size=512,
                    ),
                    precisions=("mxfp8",),
                )
    finally:
        functional.clear_workspace_cache()


def test_e2e_different_topk(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for topk in (1, 2, 4, 6, 8):
            _run_e2e_case(
                context,
                name=f"topk-{topk}",
                hidden_size=7168,
                intermediate_size=2048,
                num_local_experts=4,
                topk=topk,
                config=functional.MoKConfig(
                    minibatch_size=256,
                    macrobatch_size=512,
                ),
                precisions=("bf16", "mxfp8"),
            )
    finally:
        functional.clear_workspace_cache()


def test_e2e_finiteness_different_seeds(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for seed in (42, 123, 456, 789):
            _run_e2e_case(
                context,
                name=f"finiteness/seed-{seed}",
                hidden_size=7168,
                intermediate_size=2048,
                num_local_experts=4,
                topk=8,
                config=functional.MoKConfig(
                    minibatch_size=256,
                    macrobatch_size=512,
                ),
                precisions=("mxfp8",),
                seed=seed,
                grad_seed=10_000 + seed,
            )
    finally:
        functional.clear_workspace_cache()


def test_epilogues_support_more_than_max_grid_y_tokens(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    num_local_tokens = 2 * (65_535 + 1)
    hidden_dim = 256
    shared = torch.full(
        (num_local_tokens, hidden_dim),
        1.0,
        device=device,
        dtype=torch.bfloat16,
    )
    routed = torch.full_like(shared, 2.0)
    topk_weights = torch.full(
        (num_local_tokens, 1),
        0.25,
        device=device,
        dtype=torch.float32,
    )

    output = ops.fwd_epilogue(shared, routed, topk_weights)
    assert torch.all(output == 1.5)

    d_x = ops.bwd_epilogue(shared, routed)
    assert torch.all(d_x == 3.0)
