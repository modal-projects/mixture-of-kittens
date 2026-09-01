"""MXFP8 quantization and the forward dispatch, MLP and combine operators."""

import torch

from . import _C


@torch.library.custom_op(
    "mok::mxfp8_quantize", mutates_args=(),
    schema="(Tensor x_bf16, bool return_normal, bool return_transposed) -> (Tensor?, Tensor?, Tensor?, Tensor?)",
)
def mxfp8_quantize(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Quantizes BF16 matrices to MXFP8 in normal and/or transposed layouts.

    Inputs:
        x_bf16:           bfloat16 [M, N] or [E, M, N]
        return_normal:     bool
        return_transposed: bool

    Outputs:
        x_fp8:   float8_e4m3fn [M, N] or [E, M, N] | None
        x_sc:    uint8 [E * M // 128, N // 128, 32, 16] | None
        x_fp8_t: float8_e4m3fn [N, M] or [E, N, M] | None
        x_sc_t:  uint8 [E * N // 128, M // 128, 32, 16] | None
    """
    if x_bf16.ndim not in (2, 3):
        raise ValueError("x_bf16 must have shape (M, N) or (E, M, N)")
    if any(size <= 0 for size in x_bf16.shape):
        raise ValueError("x_bf16 dimensions must be positive")
    if x_bf16.shape[-2] % 128 != 0 or x_bf16.shape[-1] % 128 != 0:
        raise ValueError("x_bf16 M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("at least one quantized layout must be requested")

    return _C.mxfp8_quantize(x_bf16, return_normal, return_transposed)


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_mxfp8", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_mxfp8(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    w_routed_down_sc: torch.Tensor,
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
]:
    """Runs the fused MXFP8 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [num_local_tokens * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:        uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:          uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_down_sc:        uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_fp8_t_routed:      float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:       uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:         bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:           bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:       float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:        uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:       bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed: float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:  uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        y_shared:            bfloat16 [num_local_tokens, hidden_size]
        y_routed:            bfloat16 [macrobatch_size, hidden_size]
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
    for pointer_name, pointers in (("x_ptrs", x_ptrs),
                                   ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if w_shared_gate.ndim != 2:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    expected_shapes = (
        ("combine_buffer", combine_buffer, (num_local_tokens * topk, hidden_size)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_down_sc", w_routed_down_sc,
         (num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("w_routed_down_sc", w_routed_down_sc),
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

    return _C.dispatch_mlp_swiglu_combine_fwd_mxfp8(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_routed_gate_sc,
        w_shared_up, w_routed_up, w_routed_up_sc,
        w_shared_down, w_routed_down, w_routed_down_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_bf16", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_bf16(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
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
]:
    """Runs the fused BF16 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [num_local_tokens * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           bfloat16 [num_local_experts, hidden_size, intermediate_size]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_routed:      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        y_shared:      bfloat16 [num_local_tokens, hidden_size]
        y_routed:      bfloat16 [macrobatch_size, hidden_size]
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
    for pointer_name, pointers in (("x_ptrs", x_ptrs), ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("combine_buffer", combine_buffer, (num_local_tokens * topk, hidden_size)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")

    return _C.dispatch_mlp_swiglu_combine_fwd_bf16(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::recompute_forward_context_mxfp8", mutates_args=())
def recompute_forward_context_mxfp8(
    x: torch.Tensor,
    x_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
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
]:
    """Recomputes the MXFP8 MoE forward context.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:        uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:          uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_fp8_t_routed:      float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:       uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:         bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:           bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:       float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:        uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:       bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed: float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:  uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
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
    if (type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if not isinstance(x_ptrs, list) or any(type(pointer) is not int or pointer <= 0 for pointer in x_ptrs):
        raise TypeError("x_ptrs must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if w_shared_gate.ndim != 2:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_shared_up", w_shared_up),
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

    return _C.recompute_forward_context_mxfp8(
        x, x_ptrs,
        w_shared_gate, w_routed_gate, w_routed_gate_sc,
        w_shared_up, w_routed_up, w_routed_up_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::recompute_forward_context_bf16", mutates_args=())
def recompute_forward_context_bf16(
    x: torch.Tensor,
    x_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
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
]:
    """Recomputes the BF16 MoE forward context.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             bfloat16 [num_local_experts, intermediate_size, hidden_size]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_routed:      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
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
    if not isinstance(x_ptrs, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in x_ptrs
    ):
        raise TypeError("x_ptrs must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
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

    return _C.recompute_forward_context_bf16(
        x, x_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )
