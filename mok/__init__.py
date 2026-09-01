import torch  # noqa: F401

from . import ops as ops
from . import _fake_impls as _fake_impls
from . import functional as functional
from . import kimi_k3 as kimi_k3
from .kimi_k3 import (
    KIMI_K3_CAPACITY_BUCKETS,
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_MAX_TOKENS,
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_RMS_EPS,
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    KIMI_K3_SHARED_INTERMEDIATE_SIZE,
    KIMI_K3_SITU_BETA,
    KIMI_K3_SITU_LINEAR_BETA,
    KIMI_K3_TOPK,
    KIMI_K3_TP_SIZE,
    KimiK3DecodeConfig,
    kimi_k3_moe_reference,
    kimi_k3_rmsnorm_reference,
    kimi_k3_router_reference,
    kimi_k3_situ_reference,
)

__version__ = "0.1.0"

__all__ = [
    "KIMI_K3_CAPACITY_BUCKETS",
    "KIMI_K3_HIDDEN_SIZE",
    "KIMI_K3_LATENT_SIZE",
    "KIMI_K3_MAX_TOKENS",
    "KIMI_K3_NUM_EXPERTS",
    "KIMI_K3_RMS_EPS",
    "KIMI_K3_ROUTED_INTERMEDIATE_SIZE",
    "KIMI_K3_SHARED_INTERMEDIATE_SIZE",
    "KIMI_K3_SITU_BETA",
    "KIMI_K3_SITU_LINEAR_BETA",
    "KIMI_K3_TOPK",
    "KIMI_K3_TP_SIZE",
    "KimiK3DecodeConfig",
    "__version__",
    "functional",
    "kimi_k3",
    "kimi_k3_moe_reference",
    "kimi_k3_rmsnorm_reference",
    "kimi_k3_router_reference",
    "kimi_k3_situ_reference",
    "ops",
]
