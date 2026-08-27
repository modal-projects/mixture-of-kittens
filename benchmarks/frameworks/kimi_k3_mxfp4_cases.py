"""Deterministic BF16 fixtures for the Kimi K3 MXFP4 cross-check.

Shared by the packer (which runs in the MoK image) and the FlashInfer consumer
(which runs in the official Kimi K3 vLLM image). Only the packer imports the
tensor builders; the consumer reads shapes and metadata back out of the payload
that the packer writes, so this module never has to exist in both images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is unavailable on the machine that submits the job
    import torch

# One TP8 rank of Kimi K3's routed experts.
LATENT_SIZE = 3584
ROUTED_INTERMEDIATE_PER_RANK = 384
# Zero-padded W1/W3 contraction dimension of the canonical prepared layout.
W1W3_PADDED_K = 3648
GROUP_SIZE = 32
UNIT_SCALE_BYTE = 0x7F
PAYLOAD_NAME = "mok_mxfp4_bytes.pt"


@dataclass(frozen=True)
class Case:
    """One BF16 matrix to pack, compare and (for ``consumer``) execute."""

    name: str
    rows: int
    columns: int
    seed: int
    corners: bool      # exercise MXFP4 corner cases instead of plain values
    consumer: bool     # feed this matrix through the FlashInfer MoE kernel
    padded_k: int      # canonical prepared contraction width

    @property
    def logical_k(self) -> int:
        return self.columns


CASES: tuple[Case, ...] = (
    Case("w1", ROUTED_INTERMEDIATE_PER_RANK, LATENT_SIZE, 101, True, False,
         W1W3_PADDED_K),
    Case("w3", ROUTED_INTERMEDIATE_PER_RANK, LATENT_SIZE, 202, True, False,
         W1W3_PADDED_K),
    Case("w2", LATENT_SIZE, ROUTED_INTERMEDIATE_PER_RANK, 303, True, False,
         ROUTED_INTERMEDIATE_PER_RANK),
    Case("moe_w1", ROUTED_INTERMEDIATE_PER_RANK, LATENT_SIZE, 401, False, True,
         W1W3_PADDED_K),
    Case("moe_w3", ROUTED_INTERMEDIATE_PER_RANK, LATENT_SIZE, 402, False, True,
         W1W3_PADDED_K),
    Case("moe_w2", LATENT_SIZE, ROUTED_INTERMEDIATE_PER_RANK, 403, False, True,
         ROUTED_INTERMEDIATE_PER_RANK),
)

E2M1_CODE_POINTS = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -0.0,
)
E2M1_MIDPOINT_TIES = (
    0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
    -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0,
)


def build_case(case: Case, device: str = "cuda") -> "torch.Tensor":
    """Return the case's BF16 matrix. Identical on every run and every GPU."""
    import torch

    generator = torch.Generator(device=device).manual_seed(case.seed)
    values = torch.randn(
        case.rows, case.columns, generator=generator, device=device,
        dtype=torch.float32,
    )

    if case.corners:
        # Row 0 exact E2M1 code points, row 1 midpoint ties, row 2 all zero,
        # row 3 across the E8M0 exponent range, row 4 a zero group next to a
        # nonzero one, remaining rows pseudo-random.
        values[0] = torch.tensor(E2M1_CODE_POINTS, device=device).repeat(
            case.columns // len(E2M1_CODE_POINTS)
        )
        values[1] = torch.tensor(E2M1_MIDPOINT_TIES, device=device).repeat(
            case.columns // len(E2M1_MIDPOINT_TIES)
        )
        values[2] = 0.0
        exponents = torch.arange(case.columns, device=device) % 40 - 20
        values[3] = torch.ldexp(torch.ones(case.columns, device=device), exponents)
        values[4] = 0.0
        values[4, GROUP_SIZE:2 * GROUP_SIZE] = 1.5
    else:
        # A well-conditioned matrix: the corner-case rows above span 2^-20 to
        # 2^19, which makes an end-to-end MoE reference overflow long before it
        # says anything about byte compatibility. One group per matrix is still
        # zeroed so the kernel run itself consumes the mandated 0x7f scale byte.
        values *= 0.05
        zero_row = case.seed % case.rows
        zero_group = case.seed % (case.columns // GROUP_SIZE)
        values[zero_row, zero_group * GROUP_SIZE:(zero_group + 1) * GROUP_SIZE] = 0.0

    return values.bfloat16()


def count_zero_groups(weight: "torch.Tensor") -> int:
    """Number of 32-value groups whose source values are all exactly zero."""
    import torch

    groups = weight.float().reshape(weight.shape[0], -1, GROUP_SIZE)
    return int((groups.abs().amax(-1) == 0.0).sum())
