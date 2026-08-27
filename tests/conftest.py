import gc
import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="session")
def context() -> Iterator[tuple[int, int, torch.device]]:
    from mok.functional import clear_workspace_cache

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        yield rank, world_size, device
    finally:
        dist.barrier()
        clear_workspace_cache()
        gc.collect()
        dist.destroy_process_group()


@pytest.fixture(scope="session")
def tp8_context(
    context: tuple[int, int, torch.device],
) -> Iterator[tuple[int, int, torch.device]]:
    from mok.kimi_k3 import clear_kimi_k3_decode_workspace_cache

    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("Kimi K3 decode requires TP8")
    if torch.cuda.get_device_capability(device) != (10, 3):
        pytest.skip("Kimi K3 decode requires SM103 B300")

    try:
        yield rank, world_size, device
    finally:
        dist.barrier()
        clear_kimi_k3_decode_workspace_cache()
