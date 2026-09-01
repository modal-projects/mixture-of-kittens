"""The fused forward dispatch, in both precisions.

``dispatch_mlp_swiglu_combine_fwd`` is one launch that dispatches tokens
to their experts, runs the routed and shared MLPs, and combines the
result. Every intermediate it writes is checked, because the whole point
of fusing them is that nothing else gets to look in between.
"""

import itertools

import torch
import torch.distributed as dist

from mok import functional
from mok.functional import get_workspace
from mok.ops import (
    barrier_all,
    dispatch_mlp_swiglu_combine_fwd_mxfp8,
    dispatch_mlp_swiglu_combine_fwd_bf16,
    mxfp8_quantize,
)

from .utils import (
    check_correctness,
    generate_inputs,
    mok_params,
    swiglu_params,
    run_forward_reference_bf16,
    shapes,
)
BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)

FORWARD_RESULT_NAMES = (
    "combine_buffer",
    "gate_shared",
    "up_shared",
    "hidden_shared",
    "y_shared",
)


def test_dispatch_mlp_swiglu_combine_fwd_mxfp8(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    for shape, mok_param, swiglu_param in itertools.product(shapes(world_size), mok_params(), swiglu_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        mok_param_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = mok_param
        swiglu_param_name, swiglu_limit = swiglu_param
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        config = functional.MoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=1.5,
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        inputs = generate_inputs(
            rank,
            device,
            num_experts,
            num_local_experts,
            topk,
            num_local_tokens,
            hidden_dim,
            intermediate_dim,
        )
        (
            x,
            topk_experts,
            _,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            _,
        ) = inputs
        mok_schedule = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )
        (
            w_routed_gate_fp8,
            w_routed_gate_sc,
            _,
            _,
        ) = mxfp8_quantize(w_routed_gate, True, False)
        (
            w_routed_up_fp8,
            w_routed_up_sc,
            _,
            _,
        ) = mxfp8_quantize(w_routed_up, True, False)
        (
            w_routed_down_fp8,
            w_routed_down_sc,
            _,
            _,
        ) = mxfp8_quantize(w_routed_down, True, False)

        workspace.x_buffer.copy_(x)
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        outputs = dispatch_mlp_swiglu_combine_fwd_mxfp8(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            w_shared_gate,
            w_routed_gate_fp8,
            w_routed_gate_sc,
            w_shared_up,
            w_routed_up_fp8,
            w_routed_up_sc,
            w_shared_down,
            w_routed_down_fp8,
            w_routed_down_sc,
            mok_schedule.peer_rank,
            mok_schedule.peer_token_idx,
            mok_schedule.num_tokens,
            mok_schedule.tokens_per_expert,
            topk,
            swiglu_limit,
            fwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        actual = (
            workspace.combine_buffer,
            outputs[2],
            outputs[5],
            outputs[8],
            outputs[11],
        )
        reference = run_forward_reference_bf16(
            x,
            topk_experts,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            swiglu_limit,
        )
        for name, expected, result in zip(
            FORWARD_RESULT_NAMES, reference, actual, strict=True
        ):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                MXFP8_TOLERANCE,
                print_stats=rank == 0,
            )

    num_experts = world_size
    num_local_experts = 1
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=256,
        schedule_capacity_multiplier=1.5,
    )
    workspace = get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    inputs = generate_inputs(
        rank,
        device,
        num_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
    )
    (
        x,
        topk_experts,
        _,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        _,
    ) = inputs
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    w_routed_gate_fp8, w_routed_gate_sc, _, _ = mxfp8_quantize(
        w_routed_gate, True, False)
    w_routed_up_fp8, w_routed_up_sc, _, _ = mxfp8_quantize(
        w_routed_up, True, False)
    w_routed_down_fp8, w_routed_down_sc, _, _ = mxfp8_quantize(
        w_routed_down, True, False)
    workspace.x_buffer.copy_(x)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_kwargs = {
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "combine_buffer": workspace.combine_buffer,
        "combine_buffer_ptrs": workspace.combine_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate_fp8,
        "w_routed_gate_sc": w_routed_gate_sc,
        "w_shared_up": w_shared_up,
        "w_routed_up": w_routed_up_fp8,
        "w_routed_up_sc": w_routed_up_sc,
        "w_shared_down": w_shared_down,
        "w_routed_down": w_routed_down_fp8,
        "w_routed_down_sc": w_routed_down_sc,
        "schedule_peer_rank": mok_schedule.peer_rank,
        "schedule_peer_token_idx": mok_schedule.peer_token_idx,
        "num_tokens": mok_schedule.num_tokens,
        "tokens_per_expert": mok_schedule.tokens_per_expert,
        "topk": topk,
        "swiglu_limit": None,
        "num_comm_sms": 2,
        "macrobatch_size": 256,
        "minibatch_size": 256,
    }

    outputs = dispatch_mlp_swiglu_combine_fwd_mxfp8(**valid_kwargs)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    actual = (
        workspace.combine_buffer,
        outputs[2],
        outputs[5],
        outputs[8],
        outputs[11],
    )
    reference = run_forward_reference_bf16(
        x,
        topk_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    for name, expected, result in zip(
        FORWARD_RESULT_NAMES, reference, actual, strict=True
    ):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            MXFP8_TOLERANCE,
            print_stats=rank == 0,
        )

    for failure_name, overrides, expected_exception in (
        ("input rank", {"x": workspace.x_buffer.unsqueeze(0)}, ValueError),
        ("top-k", {"topk": 0}, ValueError),
        ("communication SMs", {"num_comm_sms": 1}, ValueError),
        ("minibatch alignment", {"minibatch_size": 128}, ValueError),
        ("macrobatch multiple", {"macrobatch_size": 384}, ValueError),
        ("pointer EP size", {"x_ptrs": workspace.x_buffer_ptrs[:-1]}, ValueError),
    ):
        try:
            dispatch_mlp_swiglu_combine_fwd_mxfp8(
                **(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_dispatch_mlp_swiglu_combine_fwd_bf16(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    for shape, mok_param, swiglu_param in itertools.product(shapes(world_size), mok_params(), swiglu_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        mok_param_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = mok_param
        swiglu_param_name, swiglu_limit = swiglu_param
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        config = functional.MoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=1.5,
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        inputs = generate_inputs(
            rank,
            device,
            num_experts,
            num_local_experts,
            topk,
            num_local_tokens,
            hidden_dim,
            intermediate_dim,
        )
        (
            x,
            topk_experts,
            _,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            _,
        ) = inputs
        mok_schedule = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )

        workspace.x_buffer.copy_(x)
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        outputs = dispatch_mlp_swiglu_combine_fwd_bf16(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            w_shared_gate,
            w_routed_gate,
            w_shared_up,
            w_routed_up,
            w_shared_down,
            w_routed_down,
            mok_schedule.peer_rank,
            mok_schedule.peer_token_idx,
            mok_schedule.num_tokens,
            mok_schedule.tokens_per_expert,
            topk,
            swiglu_limit,
            fwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        actual = (
            workspace.combine_buffer,
            outputs[1],
            outputs[3],
            outputs[5],
            outputs[7],
        )
        reference = run_forward_reference_bf16(
            x,
            topk_experts,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            swiglu_limit,
        )
        for name, expected, result in zip(
            FORWARD_RESULT_NAMES, reference, actual, strict=True
        ):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )

    num_experts = world_size
    num_local_experts = 1
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=256,
        schedule_capacity_multiplier=1.5,
    )
    workspace = get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    inputs = generate_inputs(
        rank,
        device,
        num_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
    )
    (
        x,
        topk_experts,
        _,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        _,
    ) = inputs
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    workspace.x_buffer.copy_(x)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_kwargs = {
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "combine_buffer": workspace.combine_buffer,
        "combine_buffer_ptrs": workspace.combine_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate,
        "w_shared_up": w_shared_up,
        "w_routed_up": w_routed_up,
        "w_shared_down": w_shared_down,
        "w_routed_down": w_routed_down,
        "schedule_peer_rank": mok_schedule.peer_rank,
        "schedule_peer_token_idx": mok_schedule.peer_token_idx,
        "num_tokens": mok_schedule.num_tokens,
        "tokens_per_expert": mok_schedule.tokens_per_expert,
        "topk": topk,
        "swiglu_limit": None,
        "num_comm_sms": 2,
        "macrobatch_size": 256,
        "minibatch_size": 256,
    }

    outputs = dispatch_mlp_swiglu_combine_fwd_bf16(**valid_kwargs)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    actual = (
        workspace.combine_buffer,
        outputs[1],
        outputs[3],
        outputs[5],
        outputs[7],
    )
    reference = run_forward_reference_bf16(
        x,
        topk_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    for name, expected, result in zip(
        FORWARD_RESULT_NAMES, reference, actual, strict=True
    ):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    for failure_name, overrides, expected_exception in (
        ("input rank", {"x": workspace.x_buffer.unsqueeze(0)}, ValueError),
        ("top-k", {"topk": 0}, ValueError),
        ("communication SMs", {"num_comm_sms": 1}, ValueError),
        ("minibatch alignment", {"minibatch_size": 128}, ValueError),
        ("macrobatch multiple", {"macrobatch_size": 384}, ValueError),
        ("pointer EP size", {"x_ptrs": workspace.x_buffer_ptrs[:-1]}, ValueError),
    ):
        try:
            dispatch_mlp_swiglu_combine_fwd_bf16(
                **(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")
