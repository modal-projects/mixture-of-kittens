"""``torch.compile`` over this surface, with no graph breaks allowed.

``fullgraph=True`` is the point: a custom operator that is not opaque to
Dynamo, or a barrier whose state update Dynamo does not see, shows up as a
graph break or as a stale counter rather than as a wrong number.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist

from mok import functional, ops

from .misc_support import (
    _assert_metadata,
    _make_fake_workspace,
)


def test_compile_fullgraph(
    context: tuple[int, int, torch.device],
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    _, _, real_device = context
    num_local_tokens = 512
    hidden_size = 1024
    intermediate_size = 256
    num_local_experts = 4
    topk = 2
    config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    expected_metadata = (
        ((num_local_tokens, hidden_size), torch.bfloat16),
        ((num_local_tokens, hidden_size), torch.bfloat16),
        ((num_local_tokens, topk), torch.float32),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, hidden_size, intermediate_size),
            torch.bfloat16,
        ),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((hidden_size, intermediate_size), torch.bfloat16),
    )

    for precision in ("bf16", "mxfp8"):
        captured_graphs: list[torch.fx.GraphModule] = []

        def capture_backend(
            graph_module: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> Callable[..., tuple[torch.Tensor, ...]]:
            captured_graphs.append(graph_module)
            return graph_module.forward

        with FakeTensorMode():
            device = torch.device("cuda", real_device.index)

            def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
                return torch.empty(shape, device=device, dtype=dtype)

            workspace = _make_fake_workspace(
                device,
                num_local_tokens=num_local_tokens,
                hidden_size=hidden_size,
                topk=topk,
                schedule_capacity=4096,
            )
            x = tensor((num_local_tokens, hidden_size), torch.bfloat16)
            top_experts = tensor((num_local_tokens, topk), torch.int64)
            router_weights = tensor((num_local_tokens, topk), torch.float32)
            grad_output = tensor(
                (num_local_tokens, hidden_size),
                torch.bfloat16,
            )
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
            routed_gate = tensor(
                (num_local_experts, intermediate_size, hidden_size),
                torch.bfloat16,
            )
            routed_up = tensor(
                (num_local_experts, intermediate_size, hidden_size),
                torch.bfloat16,
            )
            routed_down = tensor(
                (num_local_experts, hidden_size, intermediate_size),
                torch.bfloat16,
            )

            def complete_path(
                x: torch.Tensor,
                top_experts: torch.Tensor,
                router_weights: torch.Tensor,
                grad_output: torch.Tensor,
                shared_gate: torch.Tensor,
                shared_up: torch.Tensor,
                shared_down: torch.Tensor,
                routed_gate: torch.Tensor,
                routed_up: torch.Tensor,
                routed_down: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                schedule = functional.build_schedule(
                    workspace,
                    config,
                    top_experts,
                    num_local_experts=num_local_experts,
                )
                if precision == "bf16":
                    output, forward_context = functional.forward(
                        config,
                        workspace,
                        schedule,
                        x,
                        router_weights,
                        shared_gate,
                        shared_up,
                        shared_down,
                        routed_gate,
                        routed_up,
                        routed_down,
                    )
                    gradients = functional.backward(
                        config,
                        workspace,
                        schedule,
                        forward_context,
                        grad_output,
                        x,
                        router_weights,
                        shared_gate,
                        shared_up,
                        shared_down,
                        routed_gate,
                        routed_up,
                        routed_down,
                    )
                    return output, *gradients

                (
                    routed_gate_fp8,
                    routed_gate_sc,
                    routed_gate_t_fp8,
                    routed_gate_t_sc,
                ) = ops.mxfp8_quantize(routed_gate, True, True)
                (
                    routed_up_fp8,
                    routed_up_sc,
                    routed_up_t_fp8,
                    routed_up_t_sc,
                ) = ops.mxfp8_quantize(routed_up, True, True)
                (
                    routed_down_fp8,
                    routed_down_sc,
                    routed_down_t_fp8,
                    routed_down_t_sc,
                ) = ops.mxfp8_quantize(routed_down, True, True)
                output, forward_context = functional.forward(
                    config,
                    workspace,
                    schedule,
                    x,
                    router_weights,
                    shared_gate,
                    shared_up,
                    shared_down,
                    (routed_gate_fp8, routed_gate_sc),
                    (routed_up_fp8, routed_up_sc),
                    (routed_down_fp8, routed_down_sc),
                )
                gradients = functional.backward(
                    config,
                    workspace,
                    schedule,
                    forward_context,
                    grad_output,
                    x,
                    router_weights,
                    shared_gate,
                    shared_up,
                    shared_down,
                    (
                        routed_gate_fp8,
                        routed_gate_sc,
                        routed_gate_t_fp8,
                        routed_gate_t_sc,
                    ),
                    (
                        routed_up_fp8,
                        routed_up_sc,
                        routed_up_t_fp8,
                        routed_up_t_sc,
                    ),
                    (
                        routed_down_t_fp8,
                        routed_down_t_sc,
                    ),
                )
                return output, *gradients

            torch._dynamo.reset()
            compiled = torch.compile(
                complete_path,
                backend=capture_backend,
                fullgraph=True,
            )
            outputs = compiled(
                x,
                top_experts,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
                routed_gate,
                routed_up,
                routed_down,
            )
            repeated_outputs = compiled(
                x,
                top_experts,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
                routed_gate,
                routed_up,
                routed_down,
            )
            _assert_metadata(outputs, expected_metadata)
            _assert_metadata(repeated_outputs, expected_metadata)

        assert len(captured_graphs) == 1
        custom_op_targets = [
            str(node.target)
            for node in captured_graphs[0].graph.nodes
            if node.op == "call_function" and "mok" in str(node.target)
        ]
        expected_targets = [
            "mok.all_gather_top_experts.default",
            "mok.barrier_all.default",
            "mok.schedule.default",
            "mok.barrier_all.default",
            f"mok.dispatch_mlp_swiglu_combine_fwd_{precision}.default",
            "mok.barrier_all.default",
            "mok.fwd_epilogue.default",
            "mok.barrier_all.default",
            f"mok.dispatch_mlp_swiglu_combine_bwd_{precision}.default",
            "mok.barrier_all.default",
            "mok.bwd_epilogue.default",
        ]
        if precision == "mxfp8":
            expected_targets = [
                *expected_targets[:3],
                *(["mok.mxfp8_quantize.default"] * 3),
                *expected_targets[3:],
            ]
        assert custom_op_targets == expected_targets


def test_compile_fullgraph_recomputed_forward_context(
    context: tuple[int, int, torch.device],
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    _, _, real_device = context
    num_local_tokens = 512
    hidden_size = 1024
    intermediate_size = 256
    num_local_experts = 4
    topk = 2
    schedule_capacity = 4096
    config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    expected_metadata = (
        ((num_local_tokens, hidden_size), torch.bfloat16),
        ((num_local_tokens, topk), torch.float32),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, hidden_size, intermediate_size),
            torch.bfloat16,
        ),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((hidden_size, intermediate_size), torch.bfloat16),
    )

    for precision in ("bf16", "mxfp8"):
        captured_graphs: list[torch.fx.GraphModule] = []

        def capture_backend(
            graph_module: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> Callable[..., tuple[torch.Tensor, ...]]:
            captured_graphs.append(graph_module)
            return graph_module.forward

        with FakeTensorMode():
            device = torch.device("cuda", real_device.index)

            def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
                return torch.empty(shape, device=device, dtype=dtype)

            workspace = _make_fake_workspace(
                device,
                num_local_tokens=num_local_tokens,
                hidden_size=hidden_size,
                topk=topk,
                schedule_capacity=schedule_capacity,
            )
            schedule = functional.MoKSchedule(
                peer_rank=tensor((schedule_capacity,), torch.int32),
                peer_token_idx=tensor((schedule_capacity,), torch.int32),
                num_tokens=tensor((1,), torch.int32),
                tokens_per_expert=tensor(
                    (num_local_experts,),
                    torch.int32,
                ),
            )
            x = tensor((num_local_tokens, hidden_size), torch.bfloat16)
            router_weights = tensor(
                (num_local_tokens, topk),
                torch.float32,
            )
            grad_output = tensor(
                (num_local_tokens, hidden_size),
                torch.bfloat16,
            )
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

            if precision == "bf16":
                routed_gate_forward = routed_gate_bf16
                routed_up_forward = routed_up_bf16
                routed_gate_backward = routed_gate_bf16
                routed_up_backward = routed_up_bf16
                routed_down_backward = routed_down_bf16
            else:
                routed_gate_quantized = ops.mxfp8_quantize(
                    routed_gate_bf16,
                    True,
                    True,
                )
                routed_up_quantized = ops.mxfp8_quantize(
                    routed_up_bf16,
                    True,
                    True,
                )
                routed_down_quantized = ops.mxfp8_quantize(
                    routed_down_bf16,
                    True,
                    True,
                )
                routed_gate_forward = routed_gate_quantized[:2]
                routed_up_forward = routed_up_quantized[:2]
                routed_gate_backward = routed_gate_quantized
                routed_up_backward = routed_up_quantized
                routed_down_backward = routed_down_quantized[2:]

            def recomputed_backward_path(
                x: torch.Tensor,
                router_weights: torch.Tensor,
                grad_output: torch.Tensor,
                shared_gate: torch.Tensor,
                shared_up: torch.Tensor,
                shared_down: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                forward_context = functional.recompute_forward_context(
                    config,
                    workspace,
                    schedule,
                    x,
                    shared_gate,
                    shared_up,
                    routed_gate_forward,
                    routed_up_forward,
                )
                return functional.backward(
                    config,
                    workspace,
                    schedule,
                    forward_context,
                    grad_output,
                    x,
                    router_weights,
                    shared_gate,
                    shared_up,
                    shared_down,
                    routed_gate_backward,
                    routed_up_backward,
                    routed_down_backward,
                )

            torch._dynamo.reset()
            compiled = torch.compile(
                recomputed_backward_path,
                backend=capture_backend,
                fullgraph=True,
            )
            outputs = compiled(
                x,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
            )
            repeated_outputs = compiled(
                x,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
            )
            _assert_metadata(outputs, expected_metadata)
            _assert_metadata(repeated_outputs, expected_metadata)

        assert len(captured_graphs) == 1
        custom_op_targets = [
            str(node.target)
            for node in captured_graphs[0].graph.nodes
            if node.op == "call_function" and "mok" in str(node.target)
        ]
        assert custom_op_targets == [
            "mok.barrier_all.default",
            f"mok.recompute_forward_context_{precision}.default",
            "mok.barrier_all.default",
            "mok.barrier_all.default",
            f"mok.dispatch_mlp_swiglu_combine_bwd_{precision}.default",
            "mok.barrier_all.default",
            "mok.bwd_epilogue.default",
        ]


def test_compiled_barrier_updates_state(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    functional.clear_workspace_cache()
    workspace = functional.get_workspace(
        functional.MoKConfig(
            minibatch_size=256,
            macrobatch_size=512,
        ),
        dist.group.WORLD,
        device=device,
        num_local_tokens=512,
        hidden_size=256,
        topk=1,
    )

    def run_barrier() -> None:
        ops.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )

    try:
        assert int(workspace.barrier_buffer.item()) == 0
        assert int(workspace.barrier_target.item()) == 0
        torch._dynamo.reset()
        compiled = torch.compile(
            run_barrier,
            backend="inductor",
            fullgraph=True,
        )
        compiled()
        torch.cuda.synchronize(device)
        compiled()
        torch.cuda.synchronize(device)
        dist.barrier()
        expected = 2 * workspace.ep_size
        assert int(workspace.barrier_buffer.item()) == expected
        assert int(workspace.barrier_target.item()) == expected
    finally:
        functional.clear_workspace_cache()
        torch._dynamo.reset()
