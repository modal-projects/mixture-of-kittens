"""The backward dispatch, MLP and combine operators, and the two epilogues."""

import torch

from . import _C


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_mxfp8",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer",
                  "x_fp8_t_routed", "x_sc_t_routed",
                  "gate_fp8_routed", "gate_sc_routed", "up_fp8_routed", "up_sc_routed",
                  "hidden_fp8_t_routed", "hidden_sc_t_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_mxfp8(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate_T: torch.Tensor,
    w_routed_gate_T_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up_T: torch.Tensor,
    w_routed_up_T_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down_T: torch.Tensor,
    w_routed_down_T_sc: torch.Tensor,
    x_fp8_t_routed: torch.Tensor,
    x_sc_t_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_fp8_routed: torch.Tensor,
    gate_sc_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_fp8_routed: torch.Tensor,
    up_sc_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_fp8_t_routed: torch.Tensor,
    hidden_sc_t_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused MXFP8 MoE backward pass.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [num_local_tokens * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [num_local_tokens, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [num_local_tokens, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate_T:               float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_gate_T_sc:            uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up_T:                 float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_up_T_sc:              uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down_T:               float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_down_T_sc:            uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        x_fp8_t_routed:                float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:                 uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:               float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:                uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:                 float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:                  uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed:           float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:            uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        w_routed_gate:                 float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:              uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_routed_up:                   float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:                uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int

    Outputs:
        d_x_shared:          bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:          bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:       bfloat16 [num_local_tokens, intermediate_size]
        d_gate_fp8_routed:   float8_e4m3fn [macrobatch_size, intermediate_size]
        d_gate_sc_routed:    uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_up_shared:         bfloat16 [num_local_tokens, intermediate_size]
        d_up_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        d_up_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_hidden_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_y_fp8_routed:      float8_e4m3fn [macrobatch_size, hidden_size]
        d_y_sc_routed:       uint8 [macrobatch_size // 128, hidden_size // 128, 32, 16]
        d_w_shared_gate:     bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate:     bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:       bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:       bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down:     bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down:     bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0
            or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0
            or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    ):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
    ):
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    mb_i_sc = (macrobatch_size // 128, intermediate_size // 128, 32, 16)
    i_mb_sc = (intermediate_size // 128, macrobatch_size // 128, 32, 16)
    h_mb_sc = (hidden_size // 128, macrobatch_size // 128, 32, 16)
    e_i_h_sc = (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)
    e_h_i_sc = (num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16)
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (num_local_tokens * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (num_local_tokens, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (num_local_tokens, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate_T", w_routed_gate_T,
         (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc, e_h_i_sc),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up_T", w_routed_up_T, (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_up_T_sc", w_routed_up_T_sc, e_h_i_sc),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down_T", w_routed_down_T,
         (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_down_T_sc", w_routed_down_T_sc, e_i_h_sc),
        ("x_fp8_t_routed", x_fp8_t_routed, (hidden_size, macrobatch_size)),
        ("x_sc_t_routed", x_sc_t_routed, h_mb_sc),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_fp8_routed", gate_fp8_routed, (macrobatch_size, intermediate_size)),
        ("gate_sc_routed", gate_sc_routed, mb_i_sc),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_fp8_routed", up_fp8_routed, (macrobatch_size, intermediate_size)),
        ("up_sc_routed", up_sc_routed, mb_i_sc),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed, (intermediate_size, macrobatch_size)),
        ("hidden_sc_t_routed", hidden_sc_t_routed, i_mb_sc),
        ("w_routed_gate", w_routed_gate,
         (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc, e_i_h_sc),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc, e_i_h_sc),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("d_y_buffer", d_y_buffer),
        ("d_x_routed_buffer", d_x_routed_buffer),
        ("router_weight_buffer", router_weight_buffer),
        ("d_router_weight_buffer", d_router_weight_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate_T", w_routed_gate_T),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up_T", w_routed_up_T),
        ("w_routed_up_T_sc", w_routed_up_T_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down_T", w_routed_down_T),
        ("w_routed_down_T_sc", w_routed_down_T_sc),
        ("x_fp8_t_routed", x_fp8_t_routed),
        ("x_sc_t_routed", x_sc_t_routed),
        ("gate_shared", gate_shared),
        ("gate_fp8_routed", gate_fp8_routed),
        ("gate_sc_routed", gate_sc_routed),
        ("up_shared", up_shared),
        ("up_fp8_routed", up_fp8_routed),
        ("up_sc_routed", up_sc_routed),
        ("hidden_shared", hidden_shared),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed),
        ("hidden_sc_t_routed", hidden_sc_t_routed),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.dispatch_mlp_swiglu_combine_bwd_mxfp8(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
        w_shared_up, w_routed_up_T, w_routed_up_T_sc,
        w_shared_down, w_routed_down_T, w_routed_down_T_sc,
        x_fp8_t_routed, x_sc_t_routed,
        gate_shared, gate_fp8_routed, gate_sc_routed,
        up_shared, up_fp8_routed, up_sc_routed,
        hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
        x, x_ptrs, w_routed_gate, w_routed_gate_sc, w_routed_up, w_routed_up_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_bf16",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer", "x_routed", "gate_routed", "up_routed", "hidden_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_bf16(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    x_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused BF16 MoE backward pass.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [num_local_tokens * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [num_local_tokens, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [num_local_tokens, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:                 bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up:                   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down:                 bfloat16 [num_local_experts, hidden_size, intermediate_size]
        x_routed:                      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:                   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:                     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed:                 bfloat16 [macrobatch_size, intermediate_size]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int

    Outputs:
        d_x_shared:      bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        d_gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        d_up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        d_y_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_w_shared_gate: bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:   bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down: bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down: bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    pointer_lists = (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    )
    for pointer_name, pointers in pointer_lists:
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    for pointer_name, pointers in pointer_lists[:-1]:
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (num_local_tokens * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (num_local_tokens, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (num_local_tokens, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("x_routed", x_routed, (macrobatch_size, hidden_size)),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_routed", gate_routed, (macrobatch_size, intermediate_size)),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_routed", up_routed, (macrobatch_size, intermediate_size)),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_routed", hidden_routed, (macrobatch_size, intermediate_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor, _ in expected_shapes:
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.device != x.device:
        raise ValueError(f"schedule_peer_rank must be on {x.device}")

    return _C.dispatch_mlp_swiglu_combine_bwd_bf16(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        x_routed, gate_shared, gate_routed, up_shared, up_routed,
        hidden_shared, hidden_routed,
        x, x_ptrs,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::fwd_epilogue", mutates_args=())
def fwd_epilogue(
    y_shared: torch.Tensor,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Combines shared and router-weighted routed expert outputs.

    Inputs:
        y_shared:       bfloat16 [num_local_tokens, hidden_size]
        combine_buffer: bfloat16 [num_local_tokens * topk, hidden_size]
        topk_weights:   float32 [num_local_tokens, topk]

    Outputs:
        output: bfloat16 [num_local_tokens, hidden_size]
    """
    if y_shared.ndim != 2:
        raise ValueError("y_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = y_shared.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if (topk_weights.device != y_shared.device
            or combine_buffer.device != y_shared.device):
        raise ValueError("all tensors must be on the same CUDA device")
    if topk_weights.ndim != 2 or topk_weights.shape[0] != num_local_tokens:
        raise ValueError("topk_weights must have shape (num_local_tokens, topk)")
    topk = topk_weights.shape[1]
    if not 0 < topk <= 255:
        raise ValueError("topk must be in [1, 255]")
    if tuple(combine_buffer.shape) != (num_local_tokens * topk, hidden_size):
        raise ValueError("combine_buffer must have shape (num_local_tokens * topk, hidden_size)")

    return _C.fwd_epilogue(y_shared, combine_buffer, topk_weights)


@torch.library.custom_op("mok::bwd_epilogue", mutates_args=())
def bwd_epilogue(
    d_x_shared: torch.Tensor,
    d_x_routed_buffer: torch.Tensor,
) -> torch.Tensor:
    """Combines shared and routed input gradients.

    Inputs:
        d_x_shared:        bfloat16 [num_local_tokens, hidden_size]
        d_x_routed_buffer: bfloat16 [num_local_tokens * topk, hidden_size]

    Outputs:
        d_x: bfloat16 [num_local_tokens, hidden_size]
    """
    if d_x_shared.ndim != 2:
        raise ValueError("d_x_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = d_x_shared.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if d_x_routed_buffer.device != d_x_shared.device:
        raise ValueError("d_x_routed_buffer must be on the same CUDA device")
    if d_x_routed_buffer.ndim != 2:
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")
    if (d_x_routed_buffer.shape[1] != hidden_size
            or d_x_routed_buffer.shape[0] % num_local_tokens != 0
            or d_x_routed_buffer.shape[0] == 0
            or d_x_routed_buffer.shape[0] > num_local_tokens * 255):
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")

    return _C.bwd_epilogue(d_x_shared, d_x_routed_buffer)
