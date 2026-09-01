"""What every miscellaneous end-to-end case is built out of.

One case runner, one metadata assertion, one fake workspace, and the
tolerances the three suites compare against. Here rather than in
``utils`` because these are the shapes and the seeds this corner of the
suite chose, not the repository's shared fixtures.
"""

import torch
import torch.distributed as dist

from mok import functional, ops

from .utils import (
    check_correctness,
    run_reference_bf16,
)


BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
BF16_GRADIENT_MIN_COSINE = 0.9999
MXFP8_GRADIENT_MIN_COSINE = 0.996
RESULT_NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)



def _run_e2e_case(
    context: tuple[int, int, torch.device],
    *,
    name: str,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    config: functional.MoKConfig,
    precisions: tuple[str, ...],
    seed: int = 1234,
    grad_seed: int | None = None,
    top_experts: torch.Tensor | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    rank, _, device = context
    ep_group = dist.group.WORLD if group is None else group
    num_local_tokens = 512
    num_experts = num_local_experts * dist.get_world_size(ep_group)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    router_logits = torch.randn(
        num_local_tokens,
        num_experts,
        generator=generator,
        device=device,
    )
    topk_values, generated_top_experts = torch.topk(router_logits, topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)
    x = torch.randn(
        num_local_tokens,
        hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    w_shared_gate = (
        torch.randn(
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_shared_up = (
        torch.randn(
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_shared_down = (
        torch.randn(
            hidden_size,
            intermediate_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * intermediate_size**-0.5
    )
    w_routed_gate = (
        torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_routed_up = (
        torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_routed_down = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * intermediate_size**-0.5
    )
    if grad_seed is None:
        grad_generator = generator
    else:
        grad_generator = torch.Generator(device=device).manual_seed(
            grad_seed + rank
        )
    d_output = (
        torch.randn(
            num_local_tokens,
            hidden_size,
            generator=grad_generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    input_tuple = (
        x,
        generated_top_experts if top_experts is None else top_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    )
    reference = run_reference_bf16(*input_tuple, group=ep_group)
    workspace = functional.get_workspace(
        config,
        ep_group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        input_tuple[1],
        num_local_experts=num_local_experts,
    )

    for precision in precisions:
        if precision == "bf16":
            output, forward_context = functional.forward(
                config,
                workspace,
                schedule,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
            )
            gradients = functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                d_output,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
            )
            actual = (output, *gradients)
            tolerance = BF16_TOLERANCE
            gradient_min_cosine = BF16_GRADIENT_MIN_COSINE
        elif precision == "mxfp8":
            (
                w_routed_gate_fp8,
                w_routed_gate_sc,
                w_routed_gate_t_fp8,
                w_routed_gate_t_sc,
            ) = ops.mxfp8_quantize(w_routed_gate, True, True)
            (
                w_routed_up_fp8,
                w_routed_up_sc,
                w_routed_up_t_fp8,
                w_routed_up_t_sc,
            ) = ops.mxfp8_quantize(w_routed_up, True, True)
            (
                w_routed_down_fp8,
                w_routed_down_sc,
                w_routed_down_t_fp8,
                w_routed_down_t_sc,
            ) = ops.mxfp8_quantize(w_routed_down, True, True)
            assert all(
                tensor is not None
                for tensor in (
                    w_routed_gate_fp8,
                    w_routed_gate_sc,
                    w_routed_gate_t_fp8,
                    w_routed_gate_t_sc,
                    w_routed_up_fp8,
                    w_routed_up_sc,
                    w_routed_up_t_fp8,
                    w_routed_up_t_sc,
                    w_routed_down_fp8,
                    w_routed_down_sc,
                    w_routed_down_t_fp8,
                    w_routed_down_t_sc,
                )
            )
            output, forward_context = functional.forward(
                config,
                workspace,
                schedule,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                (w_routed_gate_fp8, w_routed_gate_sc),
                (w_routed_up_fp8, w_routed_up_sc),
                (w_routed_down_fp8, w_routed_down_sc),
            )
            gradients = functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                d_output,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                (
                    w_routed_gate_fp8,
                    w_routed_gate_sc,
                    w_routed_gate_t_fp8,
                    w_routed_gate_t_sc,
                ),
                (
                    w_routed_up_fp8,
                    w_routed_up_sc,
                    w_routed_up_t_fp8,
                    w_routed_up_t_sc,
                ),
                (
                    w_routed_down_t_fp8,
                    w_routed_down_t_sc,
                ),
            )
            actual = (output, *gradients)
            tolerance = MXFP8_TOLERANCE
            gradient_min_cosine = MXFP8_GRADIENT_MIN_COSINE
        else:
            raise AssertionError(f"unsupported precision {precision!r}")
        for result_name, expected, result in zip(
            RESULT_NAMES,
            reference,
            actual,
            strict=True,
        ):
            check_correctness(
                f"{name}/{precision}/{result_name}",
                expected,
                result,
                tolerance,
                print_stats=rank == 0,
            )
        for result_name, expected, result in zip(
            RESULT_NAMES[1:],
            reference[1:],
            actual[1:],
            strict=True,
        ):
            expected_flat = expected.float().reshape(-1)
            result_flat = result.float().reshape(-1)
            denominator = expected_flat.norm() * result_flat.norm()
            if float(denominator.item()) <= 1e-12:
                local_cosine = (
                    1.0 if torch.equal(expected_flat, result_flat) else 0.0
                )
            else:
                local_cosine = float(
                    ((expected_flat @ result_flat) / denominator).item()
                )
            finite = bool(torch.isfinite(expected_flat).all().item()) and bool(
                torch.isfinite(result_flat).all().item()
            )
            stats = torch.tensor(
                [local_cosine, int(finite)],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(stats[0:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(stats[1:2], op=dist.ReduceOp.MIN)
            minimum_cosine = float(stats[0].item())
            if rank == 0:
                print(
                    f"{name}/{precision}/{result_name}: "
                    f"minimum cosine={minimum_cosine:.8f}"
                )
            assert bool(stats[1].item())
            assert minimum_cosine >= gradient_min_cosine

def _assert_metadata(
    tensors: tuple[torch.Tensor, ...],
    expected: tuple[tuple[tuple[int, ...], torch.dtype], ...],
) -> None:
    assert len(tensors) == len(expected)
    for tensor, (shape, dtype) in zip(tensors, expected, strict=True):
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == shape
        assert tensor.dtype == dtype
        assert tensor.device.type == "cuda"

def _make_fake_workspace(
    device: torch.device,
    *,
    num_local_tokens: int = 512,
    hidden_size: int = 1024,
    topk: int = 2,
    ep_size: int = 4,
    schedule_capacity: int = 4096,
) -> functional.MoKWorkspace:
    pointers = list(range(1, ep_size + 1))

    def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype)

    return functional.MoKWorkspace(
        group_name="fake-ep-group",
        ep_rank=0,
        ep_size=ep_size,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=tensor((num_local_tokens, hidden_size), torch.bfloat16),
        x_buffer_handle=None,
        x_buffer_ptrs=pointers,
        combine_buffer=tensor(
            (num_local_tokens * topk, hidden_size),
            torch.bfloat16,
        ),
        combine_buffer_handle=None,
        combine_buffer_ptrs=pointers,
        d_y_buffer=tensor((num_local_tokens, hidden_size), torch.bfloat16),
        d_y_buffer_handle=None,
        d_y_buffer_ptrs=pointers,
        d_x_routed_buffer=tensor(
            (num_local_tokens * topk, hidden_size),
            torch.bfloat16,
        ),
        d_x_routed_buffer_handle=None,
        d_x_routed_buffer_ptrs=pointers,
        router_weight_buffer=tensor(
            (num_local_tokens, topk),
            torch.float32,
        ),
        router_weight_buffer_handle=None,
        router_weight_buffer_ptrs=pointers,
        d_router_weight_buffer=tensor(
            (num_local_tokens, topk),
            torch.float32,
        ),
        d_router_weight_buffer_handle=None,
        d_router_weight_buffer_ptrs=pointers,
        all_gather_top_experts_buffer=tensor(
            (ep_size, num_local_tokens, topk),
            torch.int32,
        ),
        all_gather_top_experts_buffer_handle=None,
        all_gather_top_experts_buffer_multicast_ptr=1,
        barrier_buffer=tensor((1,), torch.int32),
        barrier_buffer_handle=None,
        barrier_buffer_ptrs=pointers,
        barrier_buffer_multicast_ptr=1,
        barrier_target=tensor((1,), torch.int32),
    )
