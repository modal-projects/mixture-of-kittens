"""The fused backward dispatch, in both precisions.

Eight gradients out of one launch, all of them checked. The two cases are
the longest in the suite because a backward pass has the most outputs and
the fewest of them are implied by the others.
"""

import itertools

import torch
import torch.distributed as dist

from mok import functional
from mok.functional import get_workspace
from mok.ops import (
    barrier_all,
    dispatch_mlp_swiglu_combine_bwd_mxfp8,
    dispatch_mlp_swiglu_combine_bwd_bf16,
    dispatch_mlp_swiglu_combine_fwd_mxfp8,
    dispatch_mlp_swiglu_combine_fwd_bf16,
    mxfp8_quantize,
)

from .utils import (
    check_correctness,
    generate_inputs,
    mok_params,
    swiglu_params,
    run_bwd_epilogue_reference,
    run_reference_bf16,
    shapes,
)
BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)

BACKWARD_RESULT_NAMES = (
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)


def test_dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            d_output,
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
            w_routed_gate_t_fp8,
            w_routed_gate_t_sc,
        ) = mxfp8_quantize(w_routed_gate, True, True)
        (
            w_routed_up_fp8,
            w_routed_up_sc,
            w_routed_up_t_fp8,
            w_routed_up_t_sc,
        ) = mxfp8_quantize(w_routed_up, True, True)
        (
            w_routed_down_fp8,
            w_routed_down_sc,
            w_routed_down_t_fp8,
            w_routed_down_t_sc,
        ) = mxfp8_quantize(w_routed_down, True, True)

        workspace.x_buffer.copy_(x)
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        (
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            up_shared,
            up_fp8_routed,
            up_sc_routed,
            hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            _,
            _,
        ) = dispatch_mlp_swiglu_combine_fwd_mxfp8(
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

        workspace.d_y_buffer.copy_(d_output)
        workspace.x_buffer.copy_(x)
        workspace.router_weight_buffer.copy_(router_weights)
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        outputs = dispatch_mlp_swiglu_combine_bwd_mxfp8(
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            w_shared_gate,
            w_routed_gate_t_fp8,
            w_routed_gate_t_sc,
            w_shared_up,
            w_routed_up_t_fp8,
            w_routed_up_t_sc,
            w_shared_down,
            w_routed_down_t_fp8,
            w_routed_down_t_sc,
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            up_shared,
            up_fp8_routed,
            up_sc_routed,
            hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            w_routed_gate_fp8,
            w_routed_gate_sc,
            w_routed_up_fp8,
            w_routed_up_sc,
            mok_schedule.peer_rank,
            mok_schedule.peer_token_idx,
            mok_schedule.num_tokens,
            mok_schedule.tokens_per_expert,
            topk,
            swiglu_limit,
            bwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        actual = (
            run_bwd_epilogue_reference(outputs[0], workspace.d_x_routed_buffer),
            workspace.d_router_weight_buffer.clone(),
            outputs[13],
            outputs[15],
            outputs[17],
            outputs[12],
            outputs[14],
            outputs[16],
        )
        reference = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, result in zip(
            BACKWARD_RESULT_NAMES, reference, actual, strict=True
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
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
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
        w_routed_gate_t_fp8,
        w_routed_gate_t_sc,
    ) = mxfp8_quantize(w_routed_gate, True, True)
    (
        w_routed_up_fp8,
        w_routed_up_sc,
        w_routed_up_t_fp8,
        w_routed_up_t_sc,
    ) = mxfp8_quantize(w_routed_up, True, True)
    (
        w_routed_down_fp8,
        w_routed_down_sc,
        w_routed_down_t_fp8,
        w_routed_down_t_sc,
    ) = mxfp8_quantize(w_routed_down, True, True)
    workspace.x_buffer.copy_(x)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    (
        x_fp8_t_routed,
        x_sc_t_routed,
        gate_shared,
        gate_fp8_routed,
        gate_sc_routed,
        up_shared,
        up_fp8_routed,
        up_sc_routed,
        hidden_shared,
        hidden_fp8_t_routed,
        hidden_sc_t_routed,
        _,
        _,
    ) = dispatch_mlp_swiglu_combine_fwd_mxfp8(
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
        None,
        2,
        256,
        256,
    )
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    workspace.d_y_buffer.copy_(d_output)
    workspace.x_buffer.copy_(x)
    workspace.router_weight_buffer.copy_(router_weights)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_kwargs = {
        "d_y_buffer": workspace.d_y_buffer,
        "d_y_buffer_ptrs": workspace.d_y_buffer_ptrs,
        "d_x_routed_buffer": workspace.d_x_routed_buffer,
        "d_x_routed_buffer_ptrs": workspace.d_x_routed_buffer_ptrs,
        "router_weight_buffer": workspace.router_weight_buffer,
        "router_weight_buffer_ptrs": workspace.router_weight_buffer_ptrs,
        "d_router_weight_buffer": workspace.d_router_weight_buffer,
        "d_router_weight_buffer_ptrs":
            workspace.d_router_weight_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate_T": w_routed_gate_t_fp8,
        "w_routed_gate_T_sc": w_routed_gate_t_sc,
        "w_shared_up": w_shared_up,
        "w_routed_up_T": w_routed_up_t_fp8,
        "w_routed_up_T_sc": w_routed_up_t_sc,
        "w_shared_down": w_shared_down,
        "w_routed_down_T": w_routed_down_t_fp8,
        "w_routed_down_T_sc": w_routed_down_t_sc,
        "x_fp8_t_routed": x_fp8_t_routed,
        "x_sc_t_routed": x_sc_t_routed,
        "gate_shared": gate_shared,
        "gate_fp8_routed": gate_fp8_routed,
        "gate_sc_routed": gate_sc_routed,
        "up_shared": up_shared,
        "up_fp8_routed": up_fp8_routed,
        "up_sc_routed": up_sc_routed,
        "hidden_shared": hidden_shared,
        "hidden_fp8_t_routed": hidden_fp8_t_routed,
        "hidden_sc_t_routed": hidden_sc_t_routed,
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "w_routed_gate": w_routed_gate_fp8,
        "w_routed_gate_sc": w_routed_gate_sc,
        "w_routed_up": w_routed_up_fp8,
        "w_routed_up_sc": w_routed_up_sc,
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

    outputs = dispatch_mlp_swiglu_combine_bwd_mxfp8(**valid_kwargs)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    actual = (
        run_bwd_epilogue_reference(outputs[0], workspace.d_x_routed_buffer),
        workspace.d_router_weight_buffer.clone(),
        outputs[13],
        outputs[15],
        outputs[17],
        outputs[12],
        outputs[14],
        outputs[16],
    )
    reference = run_reference_bf16(*inputs)[1:]
    for name, expected, result in zip(
        BACKWARD_RESULT_NAMES, reference, actual, strict=True
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
        (
            "gradient output shape",
            {"d_y_buffer": workspace.d_y_buffer[:, :-1]},
            ValueError,
        ),
    ):
        try:
            dispatch_mlp_swiglu_combine_bwd_mxfp8(
                **(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_dispatch_mlp_swiglu_combine_bwd_bf16(
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
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            d_output,
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
        (
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
            _,
            _,
        ) = dispatch_mlp_swiglu_combine_fwd_bf16(
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

        workspace.d_y_buffer.copy_(d_output)
        workspace.x_buffer.copy_(x)
        workspace.router_weight_buffer.copy_(router_weights)
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        outputs = dispatch_mlp_swiglu_combine_bwd_bf16(
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            w_shared_gate,
            w_routed_gate,
            w_shared_up,
            w_routed_up,
            w_shared_down,
            w_routed_down,
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            mok_schedule.peer_rank,
            mok_schedule.peer_token_idx,
            mok_schedule.num_tokens,
            mok_schedule.tokens_per_expert,
            topk,
            swiglu_limit,
            bwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        actual = (
            run_bwd_epilogue_reference(outputs[0], workspace.d_x_routed_buffer),
            workspace.d_router_weight_buffer.clone(),
            outputs[10],
            outputs[12],
            outputs[14],
            outputs[9],
            outputs[11],
            outputs[13],
        )
        reference = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, result in zip(
            BACKWARD_RESULT_NAMES, reference, actual, strict=True
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
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
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
    (
        x_routed,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        _,
        _,
    ) = dispatch_mlp_swiglu_combine_fwd_bf16(
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
        None,
        2,
        256,
        256,
    )
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    workspace.d_y_buffer.copy_(d_output)
    workspace.x_buffer.copy_(x)
    workspace.router_weight_buffer.copy_(router_weights)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_kwargs = {
        "d_y_buffer": workspace.d_y_buffer,
        "d_y_buffer_ptrs": workspace.d_y_buffer_ptrs,
        "d_x_routed_buffer": workspace.d_x_routed_buffer,
        "d_x_routed_buffer_ptrs": workspace.d_x_routed_buffer_ptrs,
        "router_weight_buffer": workspace.router_weight_buffer,
        "router_weight_buffer_ptrs": workspace.router_weight_buffer_ptrs,
        "d_router_weight_buffer": workspace.d_router_weight_buffer,
        "d_router_weight_buffer_ptrs":
            workspace.d_router_weight_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate,
        "w_shared_up": w_shared_up,
        "w_routed_up": w_routed_up,
        "w_shared_down": w_shared_down,
        "w_routed_down": w_routed_down,
        "x_routed": x_routed,
        "gate_shared": gate_shared,
        "gate_routed": gate_routed,
        "up_shared": up_shared,
        "up_routed": up_routed,
        "hidden_shared": hidden_shared,
        "hidden_routed": hidden_routed,
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
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

    outputs = dispatch_mlp_swiglu_combine_bwd_bf16(**valid_kwargs)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    actual = (
        run_bwd_epilogue_reference(outputs[0], workspace.d_x_routed_buffer),
        workspace.d_router_weight_buffer.clone(),
        outputs[10],
        outputs[12],
        outputs[14],
        outputs[9],
        outputs[11],
        outputs[13],
    )
    reference = run_reference_bf16(*inputs)[1:]
    for name, expected, result in zip(
        BACKWARD_RESULT_NAMES, reference, actual, strict=True
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
        (
            "gradient output shape",
            {"d_y_buffer": workspace.d_y_buffer[:, :-1]},
            ValueError,
        ),
    ):
        try:
            dispatch_mlp_swiglu_combine_bwd_bf16(
                **(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")
