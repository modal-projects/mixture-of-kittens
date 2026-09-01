"""Recomputing the forward context, in both precisions.

The backward pass needs the forward activations and does not keep them,
so it recomputes them from the same inputs. That is only sound if the
recomputation is the forward pass -- which is what these check, output by
output, against the same reference the forward cases use.
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
    recompute_forward_context_mxfp8,
    recompute_forward_context_bf16,
)

from .utils import (
    check_correctness,
    generate_inputs,
    mok_params,
    swiglu_params,
    shapes,
)

EXACT_TOLERANCE = (0.0, 0.0)


def test_recompute_forward_context_mxfp8(
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
        forward_outputs = dispatch_mlp_swiglu_combine_fwd_mxfp8(
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

        workspace.x_buffer.copy_(x)
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        actual = recompute_forward_context_mxfp8(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            w_shared_gate,
            w_routed_gate_fp8,
            w_routed_gate_sc,
            w_shared_up,
            w_routed_up_fp8,
            w_routed_up_sc,
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

        valid_rows = min(int(mok_schedule.num_tokens.item()), macrobatch_size)
        valid_scale_blocks = valid_rows // 128
        for name, expected, result in (
            (
                "x_fp8_t_routed",
                forward_outputs[0][:, :valid_rows],
                actual[0][:, :valid_rows],
            ),
            (
                "x_sc_t_routed",
                forward_outputs[1][:, :valid_scale_blocks],
                actual[1][:, :valid_scale_blocks],
            ),
            ("gate_shared", forward_outputs[2], actual[2]),
            (
                "gate_fp8_routed",
                forward_outputs[3][:valid_rows],
                actual[3][:valid_rows],
            ),
            (
                "gate_sc_routed",
                forward_outputs[4][:valid_scale_blocks],
                actual[4][:valid_scale_blocks],
            ),
            ("up_shared", forward_outputs[5], actual[5]),
            (
                "up_fp8_routed",
                forward_outputs[6][:valid_rows],
                actual[6][:valid_rows],
            ),
            (
                "up_sc_routed",
                forward_outputs[7][:valid_scale_blocks],
                actual[7][:valid_scale_blocks],
            ),
            ("hidden_shared", forward_outputs[8], actual[8]),
            (
                "hidden_fp8_t_routed",
                forward_outputs[9][:, :valid_rows],
                actual[9][:, :valid_rows],
            ),
            (
                "hidden_sc_t_routed",
                forward_outputs[10][:, :valid_scale_blocks],
                actual[10][:, :valid_scale_blocks],
            ),
        ):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                EXACT_TOLERANCE,
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
        _,
        _,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        _,
    ) = inputs
    topk_experts = torch.arange(
        num_local_tokens, device=device).remainder(num_experts).view(-1, 1)
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    assert int(mok_schedule.num_tokens.item()) > config.macrobatch_size
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
    forward_outputs = dispatch_mlp_swiglu_combine_fwd_mxfp8(
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
    workspace.x_buffer.copy_(x)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_kwargs = {
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate_fp8,
        "w_routed_gate_sc": w_routed_gate_sc,
        "w_shared_up": w_shared_up,
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
    actual = recompute_forward_context_mxfp8(**valid_kwargs)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_rows = 256
    valid_scale_blocks = valid_rows // 128
    for name, expected, result in (
        (
            "x_fp8_t_routed",
            forward_outputs[0][:, :valid_rows],
            actual[0][:, :valid_rows],
        ),
        (
            "x_sc_t_routed",
            forward_outputs[1][:, :valid_scale_blocks],
            actual[1][:, :valid_scale_blocks],
        ),
        ("gate_shared", forward_outputs[2], actual[2]),
        ("gate_fp8_routed", forward_outputs[3][:valid_rows], actual[3][:valid_rows]),
        (
            "gate_sc_routed",
            forward_outputs[4][:valid_scale_blocks],
            actual[4][:valid_scale_blocks],
        ),
        ("up_shared", forward_outputs[5], actual[5]),
        ("up_fp8_routed", forward_outputs[6][:valid_rows], actual[6][:valid_rows]),
        (
            "up_sc_routed",
            forward_outputs[7][:valid_scale_blocks],
            actual[7][:valid_scale_blocks],
        ),
        ("hidden_shared", forward_outputs[8], actual[8]),
        (
            "hidden_fp8_t_routed",
            forward_outputs[9][:, :valid_rows],
            actual[9][:, :valid_rows],
        ),
        (
            "hidden_sc_t_routed",
            forward_outputs[10][:, :valid_scale_blocks],
            actual[10][:, :valid_scale_blocks],
        ),
    ):
        check_correctness(
            f"Multiple macrobatches/{name}",
            expected,
            result,
            EXACT_TOLERANCE,
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
            recompute_forward_context_mxfp8(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_recompute_forward_context_bf16(
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
        forward_outputs = dispatch_mlp_swiglu_combine_fwd_bf16(
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

        workspace.x_buffer.copy_(x)
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )
        actual = recompute_forward_context_bf16(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            w_shared_gate,
            w_routed_gate,
            w_shared_up,
            w_routed_up,
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

        valid_rows = min(int(mok_schedule.num_tokens.item()), macrobatch_size)
        for name, expected, result in (
            ("x_routed", forward_outputs[0][:valid_rows], actual[0][:valid_rows]),
            ("gate_shared", forward_outputs[1], actual[1]),
            ("gate_routed", forward_outputs[2][:valid_rows], actual[2][:valid_rows]),
            ("up_shared", forward_outputs[3], actual[3]),
            ("up_routed", forward_outputs[4][:valid_rows], actual[4][:valid_rows]),
            ("hidden_shared", forward_outputs[5], actual[5]),
            ("hidden_routed", forward_outputs[6][:valid_rows], actual[6][:valid_rows]),
        ):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                EXACT_TOLERANCE,
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
        _,
        _,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        _,
    ) = inputs
    topk_experts = torch.arange(
        num_local_tokens, device=device).remainder(num_experts).view(-1, 1)
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    assert int(mok_schedule.num_tokens.item()) > config.macrobatch_size

    workspace.x_buffer.copy_(x)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    forward_outputs = dispatch_mlp_swiglu_combine_fwd_bf16(
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
    workspace.x_buffer.copy_(x)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_kwargs = {
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate,
        "w_shared_up": w_shared_up,
        "w_routed_up": w_routed_up,
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
    actual = recompute_forward_context_bf16(**valid_kwargs)
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    valid_rows = 256
    for name, expected, result in (
        ("x_routed", forward_outputs[0][:valid_rows], actual[0][:valid_rows]),
        ("gate_shared", forward_outputs[1], actual[1]),
        ("gate_routed", forward_outputs[2][:valid_rows], actual[2][:valid_rows]),
        ("up_shared", forward_outputs[3], actual[3]),
        ("up_routed", forward_outputs[4][:valid_rows], actual[4][:valid_rows]),
        ("hidden_shared", forward_outputs[5], actual[5]),
        ("hidden_routed", forward_outputs[6][:valid_rows], actual[6][:valid_rows]),
    ):
        check_correctness(
            f"Multiple macrobatches/{name}",
            expected,
            result,
            EXACT_TOLERANCE,
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
            recompute_forward_context_bf16(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")
