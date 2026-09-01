"""MXFP4 expert packing and the collectives a Kimi K3 rank schedules around."""

import torch

from . import _C


@torch.library.custom_op("mok::pack_kimi_k3_mxfp4", mutates_args=())
def pack_kimi_k3_mxfp4(
    weight: torch.Tensor,
    padded_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs BF16 expert weights as group-32 MXFP4 with K zero-padding.

    Inputs:
        weight:   bfloat16 [E, N, K] with K divisible by 32
        padded_k: int, a multiple of 32 that is at least K

    Outputs:
        packed: uint8 [E, N, padded_k // 2]
        scale:  uint8 [E, N, padded_k // 32]
    """
    return _C.pack_kimi_k3_mxfp4(weight, padded_k)


@torch.library.custom_op("mok::dequant_kimi_k3_mxfp4", mutates_args=())
def dequant_kimi_k3_mxfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    logical_k: int,
) -> torch.Tensor:
    """Decodes group-32 MXFP4 bytes back to BF16 and truncates padded K.

    Inputs:
        packed:    uint8 [E, N, padded_k // 2]
        scale:     uint8 [E, N, padded_k // 32]
        logical_k: int, a multiple of 32 that is at most padded_k

    Outputs:
        weight: bfloat16 [E, N, logical_k]
    """
    return _C.dequant_kimi_k3_mxfp4(packed, scale, logical_k)


@torch.library.custom_op("mok::all_gather_top_experts", mutates_args=("all_gather_top_experts_buffer",))
def all_gather_top_experts(
    top_experts: torch.Tensor,
    all_gather_top_experts_buffer: torch.Tensor,
    all_gather_top_experts_buffer_multicast_ptr: int,
    rank: int,
    chunk_bytes: int,
) -> None:
    """All-gathers top-expert assignments across expert-parallel ranks.

    Inputs:
        top_experts:                                 int32 [num_local_tokens, topk]
        all_gather_top_experts_buffer:               int32 [ep_size, num_local_tokens, topk]
        all_gather_top_experts_buffer_multicast_ptr: int
        rank:                                        int
        chunk_bytes:                                 int

    Outputs:
        None
    """
    if (not top_experts.is_cuda or top_experts.dtype != torch.int32 or not top_experts.is_contiguous() or top_experts.ndim != 2):
        raise ValueError("top_experts must be contiguous CUDA int32 [num_local_tokens, topk]")
    if not all_gather_top_experts_buffer.is_cuda:
        raise ValueError("all_gather_top_experts_buffer must be a CUDA tensor")
    if all_gather_top_experts_buffer.dtype != torch.int32:
        raise TypeError("all_gather_top_experts_buffer must have dtype torch.int32")
    if not all_gather_top_experts_buffer.is_contiguous():
        raise ValueError("all_gather_top_experts_buffer must be contiguous")
    if all_gather_top_experts_buffer.ndim != 3:
        raise ValueError("all_gather_top_experts_buffer must have shape (ep_size, num_local_tokens, topk)")
    if any(size <= 0 for size in top_experts.shape):
        raise ValueError("top_experts dimensions must be positive")
    if any(size <= 0 for size in all_gather_top_experts_buffer.shape):
        raise ValueError("all_gather_top_experts_buffer dimensions must be positive")
    ep_size = all_gather_top_experts_buffer.shape[0]
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("all_gather_top_experts_buffer ep_size must be one of 1, 4, 8, 16, 32, 64")
    if (all_gather_top_experts_buffer.device != top_experts.device
            or tuple(all_gather_top_experts_buffer.shape[1:]) != tuple(top_experts.shape)):
        raise ValueError("all_gather_top_experts_buffer must match top_experts shape and device")
    if type(all_gather_top_experts_buffer_multicast_ptr) is not int or all_gather_top_experts_buffer_multicast_ptr <= 0:
        raise TypeError("all_gather_top_experts_buffer_multicast_ptr must be a positive integer")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if chunk_bytes % 16 != 0:
        raise ValueError("chunk_bytes must be divisible by 16")
    rank_buffer_bytes = top_experts.numel() * top_experts.element_size()
    if rank_buffer_bytes % chunk_bytes != 0:
        raise ValueError("chunk_bytes must divide one rank's route-buffer bytes")

    if ep_size == 1:
        all_gather_top_experts_buffer[0].copy_(top_experts)
    else:
        _C.all_gather_top_experts(top_experts, all_gather_top_experts_buffer, all_gather_top_experts_buffer_multicast_ptr, rank, chunk_bytes)


@torch.library.custom_op("mok::barrier_all", mutates_args=("barrier_buffer", "target"))
def barrier_all(
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    target: torch.Tensor,
) -> None:
    """Synchronizes all expert-parallel ranks at device-side.

    Inputs:
        barrier_buffer:               int32 [1]
        barrier_buffer_ptrs:          list[int] [ep_size]
        barrier_buffer_multicast_ptr: int
        target:                       int32 [1]

    Outputs:
        None
    """
    if not barrier_buffer.is_cuda:
        raise ValueError("barrier_buffer must be a CUDA tensor")
    if barrier_buffer.dtype != torch.int32:
        raise TypeError("barrier_buffer must have dtype torch.int32")
    if not barrier_buffer.is_contiguous():
        raise ValueError("barrier_buffer must be contiguous")
    if tuple(barrier_buffer.shape) != (1,):
        raise ValueError("barrier_buffer must have shape (1,)")
    if not isinstance(barrier_buffer_ptrs, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in barrier_buffer_ptrs):
        raise TypeError("barrier_buffer_ptrs must be a list of positive integers")
    ep_size = len(barrier_buffer_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("barrier_buffer_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if type(barrier_buffer_multicast_ptr) is not int or barrier_buffer_multicast_ptr <= 0:
        raise TypeError("barrier_buffer_multicast_ptr must be a positive integer")
    if (not target.is_cuda or target.device != barrier_buffer.device
            or target.dtype != torch.int32 or not target.is_contiguous()
            or tuple(target.shape) != (1,)):
        raise ValueError("target must be contiguous int32 [1] on the barrier CUDA device")

    if ep_size == 1:
        return
    else:
        _C.barrier_all(barrier_buffer, barrier_buffer_ptrs, barrier_buffer_multicast_ptr, target)


@torch.library.custom_op("mok::schedule", mutates_args=())
def schedule(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Builds the routed-token schedule for the current expert-parallel rank.

    Inputs:
        topk_all:          int32 [ep_size, num_local_tokens, topk]
        num_local_experts: int
        schedule_capacity: int
        rank:              int

    Outputs:
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
    """
    if topk_all.ndim != 3:
        raise ValueError("topk_all must have shape (ep_size, num_local_tokens, topk)")
    ep_size, num_local_tokens, topk = topk_all.shape
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("topk_all ep_size must be one of 1, 4, 8, 16, 32, 64")
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError(
            "topk_all num_local_tokens must be at least 512 and divisible by 256"
        )
    if not 0 < topk <= 255:
        raise ValueError("topk_all topk must be in [1, 255]")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if (type(schedule_capacity) is not int or schedule_capacity <= 0
            or schedule_capacity % 256 != 0):
        raise ValueError("schedule_capacity must be positive and divisible by 256")
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError("schedule_capacity must hold at least one rank's routed tokens")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")

    return _C.schedule(topk_all, num_local_experts, schedule_capacity, rank)
