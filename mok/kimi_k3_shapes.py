"""Kimi K3's fixed dimensions and numerical constants.

These are the model's own numbers, with no torch state and no operator
registration behind them, so both the public contract in :mod:`mok.kimi_k3` and
the storage layout in :mod:`mok.kimi_k3_w13` can read them without either
importing the other. :mod:`mok.kimi_k3` re-exports every name here, which is
where callers are expected to find them.
"""

from __future__ import annotations

KIMI_K3_HIDDEN_SIZE = 7168
KIMI_K3_LATENT_SIZE = 3584
KIMI_K3_ROUTED_INTERMEDIATE_SIZE = 3072
KIMI_K3_SHARED_INTERMEDIATE_SIZE = 6144
KIMI_K3_NUM_EXPERTS = 896
KIMI_K3_TOPK = 16
KIMI_K3_TP_SIZE = 8
KIMI_K3_MAX_TOKENS = 128
KIMI_K3_RMS_EPS = 1e-5
KIMI_K3_SITU_BETA = 4.0
KIMI_K3_SITU_LINEAR_BETA = 25.0
KIMI_K3_CAPACITY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
KIMI_K3_MXFP4_GROUP_SIZE = 32
KIMI_K3_MXFP4_UNIT_SCALE_BYTE = 0x7F
# Routed w1/w3 store native K: mixed W4A8 `kind::mxf8f6f4` block scaling runs at
# K=32, so the contraction needs no padding to a wider instruction shape.
KIMI_K3_W1W3_K = KIMI_K3_LATENT_SIZE

__all__ = [
    "KIMI_K3_CAPACITY_BUCKETS",
    "KIMI_K3_HIDDEN_SIZE",
    "KIMI_K3_LATENT_SIZE",
    "KIMI_K3_MAX_TOKENS",
    "KIMI_K3_MXFP4_GROUP_SIZE",
    "KIMI_K3_MXFP4_UNIT_SCALE_BYTE",
    "KIMI_K3_NUM_EXPERTS",
    "KIMI_K3_RMS_EPS",
    "KIMI_K3_ROUTED_INTERMEDIATE_SIZE",
    "KIMI_K3_SHARED_INTERMEDIATE_SIZE",
    "KIMI_K3_SITU_BETA",
    "KIMI_K3_SITU_LINEAR_BETA",
    "KIMI_K3_TOPK",
    "KIMI_K3_TP_SIZE",
    "KIMI_K3_W1W3_K",
]
