"""What the operators promise a fake tensor, and what they promise a schema.

Both are claims about metadata rather than about numbers, which is why
they are here rather than beside the end-to-end cases: a fake-tensor run
computes nothing, and a mutation schema that lies is a compile-time
failure a numerical test would never reach.
"""

import torch

from mok import ops

from .misc_support import (
    _assert_metadata,
    _make_fake_workspace,
)


def test_fake_tensor_metadata(
    context: tuple[int, int, torch.device],
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    _, _, real_device = context
    num_local_tokens = 512
    hidden_size = 1024
    intermediate_size = 256
    num_local_experts = 4
    ep_size = 4
    topk = 2
    macrobatch_size = 512
    schedule_capacity = 4096
    pointers = list(range(1, ep_size + 1))

    with FakeTensorMode():
        device = torch.device("cuda", real_device.index)

        def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, device=device, dtype=dtype)

        workspace = _make_fake_workspace(
            device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_size,
            topk=topk,
            ep_size=ep_size,
            schedule_capacity=schedule_capacity,
        )
        top_experts = tensor((num_local_tokens, topk), torch.int32)
        assert (
            ops.all_gather_top_experts(
                top_experts,
                workspace.all_gather_top_experts_buffer,
                workspace.all_gather_top_experts_buffer_multicast_ptr,
                0,
                1024,
            )
            is None
        )
        assert (
            ops.barrier_all(
                workspace.barrier_buffer,
                pointers,
                workspace.barrier_buffer_multicast_ptr,
                workspace.barrier_target,
            )
            is None
        )
        schedule = ops.schedule(
            workspace.all_gather_top_experts_buffer,
            num_local_experts,
            schedule_capacity,
            0,
        )
        _assert_metadata(
            schedule,
            (
                ((schedule_capacity,), torch.int32),
                ((schedule_capacity,), torch.int32),
                ((1,), torch.int32),
                ((num_local_experts,), torch.int32),
            ),
        )

        x = tensor((num_local_tokens, hidden_size), torch.bfloat16)
        router_weights = tensor((num_local_tokens, topk), torch.float32)
        shared_gate = tensor(
            (intermediate_size, hidden_size),
            torch.bfloat16,
        )
        shared_up = tensor(
            (intermediate_size, hidden_size),
            torch.bfloat16,
        )
        shared_down = tensor(
            (hidden_size, intermediate_size),
            torch.bfloat16,
        )
        routed_gate_bf16 = tensor(
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        )
        routed_up_bf16 = tensor(
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        )
        routed_down_bf16 = tensor(
            (num_local_experts, hidden_size, intermediate_size),
            torch.bfloat16,
        )
        routed_gate = ops.mxfp8_quantize(routed_gate_bf16, True, True)
        routed_up = ops.mxfp8_quantize(routed_up_bf16, True, True)
        routed_down = ops.mxfp8_quantize(routed_down_bf16, True, True)
        _assert_metadata(
            routed_gate,
            (
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.float8_e4m3fn,
                ),
                (
                    (
                        num_local_experts * intermediate_size // 128,
                        hidden_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                (
                    (num_local_experts, hidden_size, intermediate_size),
                    torch.float8_e4m3fn,
                ),
                (
                    (
                        num_local_experts * hidden_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
            ),
        )
        normal_only = ops.mxfp8_quantize(routed_gate_bf16, True, False)
        transposed_only = ops.mxfp8_quantize(
            routed_gate_bf16,
            False,
            True,
        )
        assert normal_only[0] is not None
        assert normal_only[1] is not None
        assert normal_only[2:] == (None, None)
        assert tuple(normal_only[0].shape) == tuple(routed_gate_bf16.shape)
        assert normal_only[0].dtype == torch.float8_e4m3fn
        assert tuple(normal_only[1].shape) == (
            num_local_experts * intermediate_size // 128,
            hidden_size // 128,
            32,
            16,
        )
        assert normal_only[1].dtype == torch.uint8
        assert transposed_only[:2] == (None, None)
        assert transposed_only[2] is not None
        assert transposed_only[3] is not None
        assert tuple(transposed_only[2].shape) == (
            num_local_experts,
            hidden_size,
            intermediate_size,
        )
        assert transposed_only[2].dtype == torch.float8_e4m3fn
        assert tuple(transposed_only[3].shape) == (
            num_local_experts * hidden_size // 128,
            intermediate_size // 128,
            32,
            16,
        )
        assert transposed_only[3].dtype == torch.uint8

        mxfp8_forward = ops.dispatch_mlp_swiglu_combine_fwd_mxfp8(
            x,
            pointers,
            workspace.combine_buffer,
            pointers,
            shared_gate,
            routed_gate[0],
            routed_gate[1],
            shared_up,
            routed_up[0],
            routed_up[1],
            shared_down,
            routed_down[0],
            routed_down[1],
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            mxfp8_forward,
            (
                ((hidden_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (hidden_size // 128, macrobatch_size // 128, 32, 16),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((intermediate_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (
                        intermediate_size // 128,
                        macrobatch_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
            ),
        )
        mxfp8_recomputed = ops.recompute_forward_context_mxfp8(
            x,
            pointers,
            shared_gate,
            routed_gate[0],
            routed_gate[1],
            shared_up,
            routed_up[0],
            routed_up[1],
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            mxfp8_recomputed,
            (
                ((hidden_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (hidden_size // 128, macrobatch_size // 128, 32, 16),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((intermediate_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (
                        intermediate_size // 128,
                        macrobatch_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
            ),
        )
        output = ops.fwd_epilogue(
            mxfp8_forward[-2],
            workspace.combine_buffer,
            router_weights,
        )
        _assert_metadata(
            (output,),
            (((num_local_tokens, hidden_size), torch.bfloat16),),
        )

        mxfp8_backward = ops.dispatch_mlp_swiglu_combine_bwd_mxfp8(
            workspace.d_y_buffer,
            pointers,
            workspace.d_x_routed_buffer,
            pointers,
            workspace.router_weight_buffer,
            pointers,
            workspace.d_router_weight_buffer,
            pointers,
            shared_gate,
            routed_gate[2],
            routed_gate[3],
            shared_up,
            routed_up[2],
            routed_up[3],
            shared_down,
            routed_down[2],
            routed_down[3],
            *mxfp8_forward[:11],
            x,
            pointers,
            routed_gate[0],
            routed_gate[1],
            routed_up[0],
            routed_up[1],
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            mxfp8_backward,
            (
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        hidden_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((hidden_size, intermediate_size), torch.bfloat16),
                (
                    (num_local_experts, hidden_size, intermediate_size),
                    torch.bfloat16,
                ),
            ),
        )
        d_x = ops.bwd_epilogue(
            mxfp8_backward[0],
            workspace.d_x_routed_buffer,
        )
        _assert_metadata(
            (d_x,),
            (((num_local_tokens, hidden_size), torch.bfloat16),),
        )

        bf16_forward = ops.dispatch_mlp_swiglu_combine_fwd_bf16(
            x,
            pointers,
            workspace.combine_buffer,
            pointers,
            shared_gate,
            routed_gate_bf16,
            shared_up,
            routed_up_bf16,
            shared_down,
            routed_down_bf16,
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            bf16_forward,
            (
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
            ),
        )
        bf16_recomputed = ops.recompute_forward_context_bf16(
            x,
            pointers,
            shared_gate,
            routed_gate_bf16,
            shared_up,
            routed_up_bf16,
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            bf16_recomputed,
            (
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
            ),
        )
        bf16_backward = ops.dispatch_mlp_swiglu_combine_bwd_bf16(
            workspace.d_y_buffer,
            pointers,
            workspace.d_x_routed_buffer,
            pointers,
            workspace.router_weight_buffer,
            pointers,
            workspace.d_router_weight_buffer,
            pointers,
            shared_gate,
            routed_gate_bf16,
            shared_up,
            routed_up_bf16,
            shared_down,
            routed_down_bf16,
            *bf16_forward[:7],
            x,
            pointers,
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            bf16_backward,
            (
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((hidden_size, intermediate_size), torch.bfloat16),
                (
                    (num_local_experts, hidden_size, intermediate_size),
                    torch.bfloat16,
                ),
            ),
        )


def test_custom_op_mutation_schemas() -> None:
    expected = {
        "all_gather_top_experts": {"all_gather_top_experts_buffer"},
        "barrier_all": {"barrier_buffer", "target"},
        "schedule": set(),
        "mxfp8_quantize": set(),
        "dispatch_mlp_swiglu_combine_fwd_mxfp8": {"combine_buffer"},
        "dispatch_mlp_swiglu_combine_fwd_bf16": {"combine_buffer"},
        "recompute_forward_context_mxfp8": set(),
        "recompute_forward_context_bf16": set(),
        "dispatch_mlp_swiglu_combine_bwd_mxfp8": {
            "d_x_routed_buffer",
            "d_router_weight_buffer",
            "x_fp8_t_routed",
            "x_sc_t_routed",
            "gate_fp8_routed",
            "gate_sc_routed",
            "up_fp8_routed",
            "up_sc_routed",
            "hidden_fp8_t_routed",
            "hidden_sc_t_routed",
        },
        "dispatch_mlp_swiglu_combine_bwd_bf16": {
            "d_x_routed_buffer",
            "d_router_weight_buffer",
            "x_routed",
            "gate_routed",
            "up_routed",
            "hidden_routed",
        },
        "fwd_epilogue": set(),
        "bwd_epilogue": set(),
    }
    actual = {}
    for name in expected:
        operation = getattr(torch.ops.mok, name)
        actual[name] = {
            argument.name
            for argument in operation.default._schema.arguments
            if argument.alias_info is not None and argument.alias_info.is_write
        }
    assert actual == expected
