"""``functional.recompute_forward_context``, in both precisions.

The backward pass does not keep the forward activations; it recomputes
them. These hold the recomputation to the same reference the forward
cases are held to, which is the only thing that makes the two
interchangeable.
"""

import itertools

import torch
import torch.distributed as dist

from mok import functional
from mok.functional import get_workspace
from mok.ops import mxfp8_quantize

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
            router_weights,
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

        _, expected = functional.forward(
            config,
            workspace,
            mok_schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            (w_routed_gate_fp8, w_routed_gate_sc),
            (w_routed_up_fp8, w_routed_up_sc),
            (w_routed_down_fp8, w_routed_down_sc),
            swiglu_limit,
        )
        actual = functional.recompute_forward_context(
            config,
            workspace,
            mok_schedule,
            x,
            w_shared_gate,
            w_shared_up,
            (w_routed_gate_fp8, w_routed_gate_sc),
            (w_routed_up_fp8, w_routed_up_sc),
            swiglu_limit,
        )

        assert isinstance(expected.x_routed, tuple)
        assert isinstance(expected.gate_routed, tuple)
        assert isinstance(expected.up_routed, tuple)
        assert isinstance(expected.hidden_routed, tuple)
        assert isinstance(actual.x_routed, tuple)
        assert isinstance(actual.gate_routed, tuple)
        assert isinstance(actual.up_routed, tuple)
        assert isinstance(actual.hidden_routed, tuple)
        valid_rows = min(int(mok_schedule.num_tokens.item()), macrobatch_size)
        valid_scale_blocks = valid_rows // 128
        for name, reference, result in (
            (
                "x_fp8_t_routed",
                expected.x_routed[0][:, :valid_rows],
                actual.x_routed[0][:, :valid_rows],
            ),
            (
                "x_sc_t_routed",
                expected.x_routed[1][:, :valid_scale_blocks],
                actual.x_routed[1][:, :valid_scale_blocks],
            ),
            ("gate_shared", expected.gate_shared, actual.gate_shared),
            (
                "gate_fp8_routed",
                expected.gate_routed[0][:valid_rows],
                actual.gate_routed[0][:valid_rows],
            ),
            (
                "gate_sc_routed",
                expected.gate_routed[1][:valid_scale_blocks],
                actual.gate_routed[1][:valid_scale_blocks],
            ),
            ("up_shared", expected.up_shared, actual.up_shared),
            (
                "up_fp8_routed",
                expected.up_routed[0][:valid_rows],
                actual.up_routed[0][:valid_rows],
            ),
            (
                "up_sc_routed",
                expected.up_routed[1][:valid_scale_blocks],
                actual.up_routed[1][:valid_scale_blocks],
            ),
            ("hidden_shared", expected.hidden_shared, actual.hidden_shared),
            (
                "hidden_fp8_t_routed",
                expected.hidden_routed[0][:, :valid_rows],
                actual.hidden_routed[0][:, :valid_rows],
            ),
            (
                "hidden_sc_t_routed",
                expected.hidden_routed[1][:, :valid_scale_blocks],
                actual.hidden_routed[1][:, :valid_scale_blocks],
            ),
        ):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                reference,
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
        all_gather_top_experts_chunk_bytes=16,
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
        router_weights,
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
    _, expected = functional.forward(
        config,
        workspace,
        mok_schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        (w_routed_gate_fp8, w_routed_gate_sc),
        (w_routed_up_fp8, w_routed_up_sc),
        (w_routed_down_fp8, w_routed_down_sc),
    )
    valid_kwargs = {
        "config": config,
        "workspace": workspace,
        "schedule": mok_schedule,
        "x": x,
        "shared_gate_weights": w_shared_gate,
        "shared_up_weights": w_shared_up,
        "routed_gate_weights": (w_routed_gate_fp8, w_routed_gate_sc),
        "routed_up_weights": (w_routed_up_fp8, w_routed_up_sc),
    }
    actual = functional.recompute_forward_context(**valid_kwargs)
    assert isinstance(expected.x_routed, tuple)
    assert isinstance(expected.gate_routed, tuple)
    assert isinstance(expected.up_routed, tuple)
    assert isinstance(expected.hidden_routed, tuple)
    assert isinstance(actual.x_routed, tuple)
    assert isinstance(actual.gate_routed, tuple)
    assert isinstance(actual.up_routed, tuple)
    assert isinstance(actual.hidden_routed, tuple)
    valid_rows = config.macrobatch_size
    valid_scale_blocks = valid_rows // 128
    for name, reference, result in (
        (
            "x_fp8_t_routed",
            expected.x_routed[0][:, :valid_rows],
            actual.x_routed[0][:, :valid_rows],
        ),
        (
            "x_sc_t_routed",
            expected.x_routed[1][:, :valid_scale_blocks],
            actual.x_routed[1][:, :valid_scale_blocks],
        ),
        ("gate_shared", expected.gate_shared, actual.gate_shared),
        (
            "gate_fp8_routed",
            expected.gate_routed[0][:valid_rows],
            actual.gate_routed[0][:valid_rows],
        ),
        (
            "gate_sc_routed",
            expected.gate_routed[1][:valid_scale_blocks],
            actual.gate_routed[1][:valid_scale_blocks],
        ),
        ("up_shared", expected.up_shared, actual.up_shared),
        (
            "up_fp8_routed",
            expected.up_routed[0][:valid_rows],
            actual.up_routed[0][:valid_rows],
        ),
        (
            "up_sc_routed",
            expected.up_routed[1][:valid_scale_blocks],
            actual.up_routed[1][:valid_scale_blocks],
        ),
        ("hidden_shared", expected.hidden_shared, actual.hidden_shared),
        (
            "hidden_fp8_t_routed",
            expected.hidden_routed[0][:, :valid_rows],
            actual.hidden_routed[0][:, :valid_rows],
        ),
        (
            "hidden_sc_t_routed",
            expected.hidden_routed[1][:, :valid_scale_blocks],
            actual.hidden_routed[1][:, :valid_scale_blocks],
        ),
    ):
        check_correctness(
            f"Multiple macrobatches/{name}",
            reference,
            result,
            EXACT_TOLERANCE,
            print_stats=rank == 0,
        )

    invalid_schedule = functional.MoKSchedule(
        peer_rank=mok_schedule.peer_rank[:-256],
        peer_token_idx=mok_schedule.peer_token_idx[:-256],
        num_tokens=mok_schedule.num_tokens,
        tokens_per_expert=mok_schedule.tokens_per_expert,
    )
    for failure_name, overrides, expected_exception in (
        ("config type", {"config": object()}, TypeError),
        ("workspace type", {"workspace": object()}, TypeError),
        ("schedule type", {"schedule": object()}, TypeError),
        ("input dtype", {"x": x.float()}, ValueError),
        ("input shape", {"x": x[:-256]}, ValueError),
        ("schedule capacity", {"schedule": invalid_schedule}, ValueError),
        (
            "routed precision mismatch",
            {"routed_up_weights": w_routed_up},
            TypeError,
        ),
    ):
        try:
            functional.recompute_forward_context(**(valid_kwargs | overrides))
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
            router_weights,
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

        _, expected = functional.forward(
            config,
            workspace,
            mok_schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            swiglu_limit,
        )
        actual = functional.recompute_forward_context(
            config,
            workspace,
            mok_schedule,
            x,
            w_shared_gate,
            w_shared_up,
            w_routed_gate,
            w_routed_up,
            swiglu_limit,
        )

        assert isinstance(expected.x_routed, torch.Tensor)
        assert isinstance(expected.gate_routed, torch.Tensor)
        assert isinstance(expected.up_routed, torch.Tensor)
        assert isinstance(expected.hidden_routed, torch.Tensor)
        assert isinstance(actual.x_routed, torch.Tensor)
        assert isinstance(actual.gate_routed, torch.Tensor)
        assert isinstance(actual.up_routed, torch.Tensor)
        assert isinstance(actual.hidden_routed, torch.Tensor)
        valid_rows = min(int(mok_schedule.num_tokens.item()), macrobatch_size)
        for name, reference, result in (
            ("x_routed", expected.x_routed[:valid_rows], actual.x_routed[:valid_rows]),
            ("gate_shared", expected.gate_shared, actual.gate_shared),
            (
                "gate_routed",
                expected.gate_routed[:valid_rows],
                actual.gate_routed[:valid_rows],
            ),
            ("up_shared", expected.up_shared, actual.up_shared),
            ("up_routed", expected.up_routed[:valid_rows], actual.up_routed[:valid_rows]),
            ("hidden_shared", expected.hidden_shared, actual.hidden_shared),
            (
                "hidden_routed",
                expected.hidden_routed[:valid_rows],
                actual.hidden_routed[:valid_rows],
            ),
        ):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                reference,
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
        all_gather_top_experts_chunk_bytes=16,
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
        router_weights,
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
    _, expected = functional.forward(
        config,
        workspace,
        mok_schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    valid_kwargs = {
        "config": config,
        "workspace": workspace,
        "schedule": mok_schedule,
        "x": x,
        "shared_gate_weights": w_shared_gate,
        "shared_up_weights": w_shared_up,
        "routed_gate_weights": w_routed_gate,
        "routed_up_weights": w_routed_up,
    }
    actual = functional.recompute_forward_context(**valid_kwargs)
    assert isinstance(expected.x_routed, torch.Tensor)
    assert isinstance(expected.gate_routed, torch.Tensor)
    assert isinstance(expected.up_routed, torch.Tensor)
    assert isinstance(expected.hidden_routed, torch.Tensor)
    assert isinstance(actual.x_routed, torch.Tensor)
    assert isinstance(actual.gate_routed, torch.Tensor)
    assert isinstance(actual.up_routed, torch.Tensor)
    assert isinstance(actual.hidden_routed, torch.Tensor)
    valid_rows = config.macrobatch_size
    for name, reference, result in (
        ("x_routed", expected.x_routed[:valid_rows], actual.x_routed[:valid_rows]),
        ("gate_shared", expected.gate_shared, actual.gate_shared),
        (
            "gate_routed",
            expected.gate_routed[:valid_rows],
            actual.gate_routed[:valid_rows],
        ),
        ("up_shared", expected.up_shared, actual.up_shared),
        ("up_routed", expected.up_routed[:valid_rows], actual.up_routed[:valid_rows]),
        ("hidden_shared", expected.hidden_shared, actual.hidden_shared),
        (
            "hidden_routed",
            expected.hidden_routed[:valid_rows],
            actual.hidden_routed[:valid_rows],
        ),
    ):
        check_correctness(
            f"Multiple macrobatches/{name}",
            reference,
            result,
            EXACT_TOLERANCE,
            print_stats=rank == 0,
        )

    invalid_schedule = functional.MoKSchedule(
        peer_rank=mok_schedule.peer_rank[:-256],
        peer_token_idx=mok_schedule.peer_token_idx[:-256],
        num_tokens=mok_schedule.num_tokens,
        tokens_per_expert=mok_schedule.tokens_per_expert,
    )
    for failure_name, overrides, expected_exception in (
        ("config type", {"config": object()}, TypeError),
        ("workspace type", {"workspace": object()}, TypeError),
        ("schedule type", {"schedule": object()}, TypeError),
        ("input dtype", {"x": x.float()}, ValueError),
        ("input shape", {"x": x[:-256]}, ValueError),
        ("schedule capacity", {"schedule": invalid_schedule}, ValueError),
        (
            "routed precision mismatch",
            {"routed_up_weights": (w_routed_up, w_routed_up)},
            TypeError,
        ),
    ):
        try:
            functional.recompute_forward_context(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")
