"""The operators that stand alone: routing, scheduling, and quantization.

One kernel and one reference each, and short enough to read beside each
other. The epilogues are here too for the same reason.

The four dispatch/MLP/combine operators are an order of magnitude longer
and live in ``test_ops_dispatch_forward``, ``test_ops_recompute`` and
``test_ops_dispatch_backward``.
"""

import math

import torch
import torch.distributed as dist

from mok import functional
from mok.functional import get_workspace
from mok.ops import (
    all_gather_top_experts,
    barrier_all,
    bwd_epilogue,
    fwd_epilogue,
    mxfp8_quantize,
    schedule,
)

from .utils import (
    check_correctness,
    generate_topk_experts,
    run_all_gather_top_experts_reference,
    run_bwd_epilogue_reference,
    run_fwd_epilogue_reference,
    run_mxfp8_quantize_reference,
    run_schedule_reference,
    shapes,
)

EXACT_TOLERANCE = (0.0, 0.0)
BF16_TOLERANCE = (0.5, 0.01)
MXFP8_RESULT_NAMES = (
    "normal",
    "normal_scales",
    "transposed",
    "transposed_scales",
)


def test_all_gather_top_experts(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, num_experts, hidden_dim, _, topk, num_local_tokens = shape
        config = functional.MoKConfig(schedule_capacity_multiplier=1.5)
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        topk_experts = generate_topk_experts(rank, device, num_experts, topk, num_local_tokens).to(torch.int32)

        all_gather_top_experts(
            topk_experts,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            rank,
            config.all_gather_top_experts_chunk_bytes,
        )
        barrier_all(
            workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
        )

        reference = run_all_gather_top_experts_reference(topk_experts)
        check_correctness(
            shape_name,
            reference,
            workspace.all_gather_top_experts_buffer,
            EXACT_TOLERANCE,
            print_stats=rank == 0,
        )

    num_experts = world_size
    hidden_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.MoKConfig(schedule_capacity_multiplier=1.5)
    workspace = get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    topk_experts = generate_topk_experts(
        rank, device, num_experts, topk, num_local_tokens).to(torch.int32)

    all_gather_top_experts(
        topk_experts,
        workspace.all_gather_top_experts_buffer,
        workspace.all_gather_top_experts_buffer_multicast_ptr,
        rank,
        16,
    )
    barrier_all(
        workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr, workspace.barrier_target
    )
    reference = run_all_gather_top_experts_reference(topk_experts)
    check_correctness(
        "Minimum chunk bytes",
        reference,
        workspace.all_gather_top_experts_buffer,
        EXACT_TOLERANCE,
        print_stats=rank == 0,
    )

    valid_kwargs = {
        "top_experts": topk_experts,
        "all_gather_top_experts_buffer": workspace.all_gather_top_experts_buffer,
        "all_gather_top_experts_buffer_multicast_ptr":
            workspace.all_gather_top_experts_buffer_multicast_ptr,
        "rank": rank,
        "chunk_bytes": 16,
    }
    for failure_name, overrides, expected_exception in (
        ("top_experts dtype", {"top_experts": topk_experts.to(torch.int64)}, ValueError),
        (
            "all-gather buffer dtype",
            {"all_gather_top_experts_buffer":
             workspace.all_gather_top_experts_buffer.to(torch.int64)},
            TypeError,
        ),
        (
            "multicast pointer",
            {"all_gather_top_experts_buffer_multicast_ptr": 0},
            TypeError,
        ),
        ("rank", {"rank": world_size}, ValueError),
        ("chunk alignment", {"chunk_bytes": 15}, ValueError),
        ("chunk divisibility", {"chunk_bytes": 48}, ValueError),
    ):
        try:
            all_gather_top_experts(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_schedule(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, num_experts, _, _, topk, num_local_tokens = shape
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        schedule_capacity = num_local_tokens * topk * max(2, math.ceil(world_size * 1.5))
        topk_experts = generate_topk_experts(rank, device, num_experts, topk, num_local_tokens).to(torch.int32)
        topk_all = run_all_gather_top_experts_reference(topk_experts)

        actual = schedule(topk_all, num_local_experts, schedule_capacity, rank)
        reference = run_schedule_reference(topk_all, num_local_experts, schedule_capacity, rank)
        valid_tokens = reference[0] >= 0
        for name, expected, result in (
            ("peer_rank", reference[0], actual[0]),
            ("peer_token_idx", reference[1][valid_tokens], actual[1][valid_tokens]),
            ("num_tokens", reference[2], actual[2]),
            ("tokens_per_expert", reference[3], actual[3]),
        ):
            check_correctness(
                f"{shape_name}/{name}",
                expected,
                result,
                EXACT_TOLERANCE,
                print_stats=rank == 0,
            )

    num_local_experts = 1
    num_local_tokens = 512
    topk = 1
    schedule_capacity = num_local_tokens * topk
    route_pattern = torch.arange(
        num_local_tokens, dtype=torch.int32, device=device).remainder(world_size)
    topk_all = route_pattern.view(1, num_local_tokens, topk).expand(
        world_size, -1, -1).contiguous()

    actual = schedule(
        topk_all, num_local_experts, schedule_capacity, rank)
    reference = run_schedule_reference(
        topk_all, num_local_experts, schedule_capacity, rank)
    valid_tokens = reference[0] >= 0
    for name, expected, result in (
        ("peer_rank", reference[0], actual[0]),
        ("peer_token_idx", reference[1][valid_tokens], actual[1][valid_tokens]),
        ("num_tokens", reference[2], actual[2]),
        ("tokens_per_expert", reference[3], actual[3]),
    ):
        check_correctness(
            f"Minimum schedule capacity/{name}",
            expected,
            result,
            EXACT_TOLERANCE,
            print_stats=rank == 0,
        )

    valid_kwargs = {
        "topk_all": topk_all,
        "num_local_experts": num_local_experts,
        "schedule_capacity": schedule_capacity,
        "rank": rank,
    }
    for failure_name, overrides, expected_exception in (
        ("topk_all rank", {"topk_all": topk_all.squeeze(-1)}, ValueError),
        ("local experts", {"num_local_experts": 0}, ValueError),
        ("schedule capacity zero", {"schedule_capacity": 0}, ValueError),
        ("schedule capacity alignment", {"schedule_capacity": 128}, ValueError),
        ("schedule capacity size", {"schedule_capacity": 256}, ValueError),
        ("rank", {"rank": world_size}, ValueError),
    ):
        try:
            schedule(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_mxfp8_quantize(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, num_experts, hidden_dim, intermediate_dim, _, num_local_tokens = shape
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        input_shapes = (
            ("2D", (num_local_tokens, hidden_dim)),
            ("3D", (num_local_experts, intermediate_dim, hidden_dim)),
        )
        for input_name, input_shape in input_shapes:
            x_bf16 = torch.randn(
                input_shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            actual = mxfp8_quantize(x_bf16, True, True)
            reference = run_mxfp8_quantize_reference(x_bf16, True, True)
            for name, expected, result in zip(
                MXFP8_RESULT_NAMES, reference, actual, strict=True
            ):
                assert expected is not None
                assert result is not None
                if expected.dtype == torch.float8_e4m3fn:
                    expected = expected.view(torch.uint8)
                    result = result.view(torch.uint8)
                check_correctness(
                    f"{shape_name}/{input_name}/{name}",
                    expected,
                    result,
                    EXACT_TOLERANCE,
                    print_stats=rank == 0,
                )

    generator = torch.Generator(device=device).manual_seed(5678 + rank)
    x_bf16 = torch.randn(
        128, 128, generator=generator, device=device, dtype=torch.bfloat16)
    for layout_name, return_normal, return_transposed in (
        ("Normal only", True, False),
        ("Transposed only", False, True),
    ):
        actual = mxfp8_quantize(
            x_bf16, return_normal, return_transposed)
        reference = run_mxfp8_quantize_reference(
            x_bf16, return_normal, return_transposed)
        for name, expected, result in zip(
            MXFP8_RESULT_NAMES, reference, actual, strict=True
        ):
            if expected is None:
                assert result is None
                continue
            assert result is not None
            if expected.dtype == torch.float8_e4m3fn:
                expected = expected.view(torch.uint8)
                result = result.view(torch.uint8)
            check_correctness(
                f"{layout_name}/{name}",
                expected,
                result,
                EXACT_TOLERANCE,
                print_stats=rank == 0,
            )

    for failure_name, x, return_normal, return_transposed in (
        ("input rank", x_bf16.flatten(), True, True),
        (
            "row divisibility",
            torch.randn(127, 128, device=device, dtype=torch.bfloat16),
            True,
            True,
        ),
        (
            "column divisibility",
            torch.randn(128, 127, device=device, dtype=torch.bfloat16),
            True,
            True,
        ),
        ("missing layout", x_bf16, False, False),
    ):
        try:
            mxfp8_quantize(x, return_normal, return_transposed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_fwd_epilogue(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, _, hidden_dim, _, topk, num_local_tokens = shape
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        y_shared = torch.randn(
            num_local_tokens,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        combine_buffer = torch.randn(
            num_local_tokens * topk,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        topk_weights = torch.softmax(
            torch.randn(
                num_local_tokens,
                topk,
                generator=generator,
                device=device,
            ),
            dim=-1,
        )
        actual = fwd_epilogue(y_shared, combine_buffer, topk_weights)
        reference = run_fwd_epilogue_reference(y_shared, combine_buffer, topk_weights)
        check_correctness(
            shape_name,
            reference,
            actual,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    num_local_tokens = 512
    hidden_dim = 256
    shared_memory_limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    topk = (shared_memory_limit - 5120) // 4104
    generator = torch.Generator(device=device).manual_seed(5678 + rank)
    y_shared = torch.randn(
        num_local_tokens, hidden_dim, generator=generator,
        device=device, dtype=torch.bfloat16)
    combine_buffer = torch.randn(
        num_local_tokens * topk, hidden_dim, generator=generator,
        device=device, dtype=torch.bfloat16)
    topk_weights = torch.softmax(
        torch.randn(
            num_local_tokens, topk, generator=generator, device=device),
        dim=-1,
    )

    actual = fwd_epilogue(y_shared, combine_buffer, topk_weights)
    reference = run_fwd_epilogue_reference(
        y_shared, combine_buffer, topk_weights)
    check_correctness(
        "Maximum executable top-k",
        reference,
        actual,
        BF16_TOLERANCE,
        print_stats=rank == 0,
    )

    valid_kwargs = {
        "y_shared": y_shared,
        "combine_buffer": combine_buffer,
        "topk_weights": topk_weights,
    }
    for failure_name, overrides, expected_exception in (
        ("shared output rank", {"y_shared": y_shared.unsqueeze(0)}, ValueError),
        ("local token minimum", {"y_shared": y_shared[:256]}, ValueError),
        ("hidden alignment", {"y_shared": y_shared[:, :128]}, ValueError),
        (
            "top-k minimum",
            {
                "topk_weights": topk_weights[:, :0],
                "combine_buffer": combine_buffer[:0],
            },
            ValueError,
        ),
        (
            "top-k maximum",
            {"topk_weights": topk_weights.new_empty(
                num_local_tokens, 256)},
            ValueError,
        ),
        (
            "combine shape",
            {"combine_buffer": combine_buffer[:-1]},
            ValueError,
        ),
    ):
        try:
            fwd_epilogue(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_bwd_epilogue(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, _, hidden_dim, _, topk, num_local_tokens = shape
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        d_x_shared = torch.randn(
            num_local_tokens,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        d_x_routed_buffer = torch.randn(
            num_local_tokens * topk,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        actual = bwd_epilogue(d_x_shared, d_x_routed_buffer)
        reference = run_bwd_epilogue_reference(d_x_shared, d_x_routed_buffer)
        check_correctness(
            shape_name,
            reference,
            actual,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    num_local_tokens = 512
    hidden_dim = 256
    topk = 55
    generator = torch.Generator(device=device).manual_seed(5678 + rank)
    d_x_shared = torch.randn(
        num_local_tokens, hidden_dim, generator=generator,
        device=device, dtype=torch.bfloat16)
    d_x_routed_buffer = torch.randn(
        num_local_tokens * topk, hidden_dim, generator=generator,
        device=device, dtype=torch.bfloat16)

    actual = bwd_epilogue(d_x_shared, d_x_routed_buffer)
    reference = run_bwd_epilogue_reference(
        d_x_shared, d_x_routed_buffer)
    check_correctness(
        "Large top-k",
        reference,
        actual,
        BF16_TOLERANCE,
        print_stats=rank == 0,
    )

    valid_kwargs = {
        "d_x_shared": d_x_shared,
        "d_x_routed_buffer": d_x_routed_buffer,
    }
    for failure_name, overrides, expected_exception in (
        (
            "shared gradient rank",
            {"d_x_shared": d_x_shared.unsqueeze(0)},
            ValueError,
        ),
        (
            "local token minimum",
            {"d_x_shared": d_x_shared[:256]},
            ValueError,
        ),
        (
            "hidden alignment",
            {"d_x_shared": d_x_shared[:, :128]},
            ValueError,
        ),
        (
            "routed gradient rank",
            {"d_x_routed_buffer": d_x_routed_buffer.flatten()},
            ValueError,
        ),
        (
            "top-k minimum",
            {"d_x_routed_buffer": d_x_routed_buffer[:0]},
            ValueError,
        ),
        (
            "routed gradient shape",
            {"d_x_routed_buffer": d_x_routed_buffer[:-1]},
            ValueError,
        ),
    ):
        try:
            bwd_epilogue(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")
