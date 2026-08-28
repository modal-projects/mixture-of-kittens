"""Shared plumbing for the native Kimi K3 serving-backend adapters.

Only framework-agnostic work lives here: mapping one prepared ``mok`` weight
shard onto the tensor names both native layers expose, capturing a CUDA graph
per route-pool entry, and describing the resulting comparison. Every vLLM and
SGLang import stays inside the two adapter modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from mok.kimi_k3 import (
    KIMI_K3_HIDDEN_SIZE,
    KIMI_K3_LATENT_SIZE,
    KIMI_K3_NUM_EXPERTS,
    KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    KIMI_K3_SHARED_INTERMEDIATE_SIZE,
    KIMI_K3_TOPK,
    KIMI_K3_TP_SIZE,
    kimi_k3_router_reference,
)

# Each loader validates ``architectures`` against its own model registry before
# it will build a config, and the two registries spell the Kimi K3 linear
# decoder differently. Nothing else in the standalone config differs.
ARCHITECTURES = {
    "vllm": ["KimiLinearForCausalLM"],
    "sglang": ["KimiK3LinearForCausalLM"],
}

MODEL_CONFIG = {
    "model_type": "kimi_linear",
    "hidden_size": KIMI_K3_HIDDEN_SIZE,
    "intermediate_size": KIMI_K3_HIDDEN_SIZE,
    "moe_intermediate_size": KIMI_K3_ROUTED_INTERMEDIATE_SIZE,
    "num_experts": KIMI_K3_NUM_EXPERTS,
    "n_routed_experts": KIMI_K3_NUM_EXPERTS,
    "num_experts_per_token": KIMI_K3_TOPK,
    "num_shared_experts": 2,
    "routed_expert_hidden_size": KIMI_K3_LATENT_SIZE,
    "latent_moe_use_norm": True,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "moe_renormalize": True,
    "use_grouped_topk": True,
    "num_expert_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "moe_router_activation_func": "sigmoid",
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-5,
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "head_dim": 224,
    "max_position_embeddings": 4096,
    "vocab_size": 1024,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
}

LAYER_PREFIX = "model.layers.1.mlp"
SHARED_IGNORED_LAYERS = (
    f"{LAYER_PREFIX}.shared_experts.gate_up_proj",
    f"{LAYER_PREFIX}.shared_experts.gate_proj",
    f"{LAYER_PREFIX}.shared_experts.up_proj",
    f"{LAYER_PREFIX}.shared_experts.down_proj",
)

ROUTED_PER_RANK = KIMI_K3_ROUTED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE
SHARED_PER_RANK = KIMI_K3_SHARED_INTERMEDIATE_SIZE // KIMI_K3_TP_SIZE


def write_model_config(directory: str, framework: str) -> str:
    """Write the standalone Kimi K3 ``config.json`` one loader reads."""
    import json
    import os

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "config.json")
    payload = {**MODEL_CONFIG, "architectures": ARCHITECTURES[framework]}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return directory


@dataclass(frozen=True, slots=True)
class NativeWeights:
    """One rank's native-layout view of a prepared ``mok`` weight shard."""

    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    gate_weight: torch.Tensor
    gate_correction_bias: torch.Tensor
    shared_gate_up_proj: torch.Tensor
    shared_down_proj: torch.Tensor
    routed_expert_down_proj: torch.Tensor
    routed_expert_up_proj: torch.Tensor
    routed_expert_norm: torch.Tensor


def native_weights(weights: Any) -> NativeWeights:
    """Rearrange a prepared shard into the fused native tensors.

    The MXFP4 expert payload is passed through byte for byte: both frameworks
    store exactly the ``[E, N, K/2]`` E2M1 pairs and ``[E, N, K/32]`` E8M0
    scale bytes that :func:`mok.kimi_k3.pack_kimi_k3_mxfp4` produces. The only
    rearrangement is concatenating the separate gate and up matrices into the
    fused row block the native layers index.
    """
    return NativeWeights(
        w13_weight=torch.cat(
            (weights.expert_w1_packed, weights.expert_w3_packed), dim=1
        ).contiguous(),
        w13_weight_scale=torch.cat(
            (weights.expert_w1_scale, weights.expert_w3_scale), dim=1
        ).contiguous(),
        w2_weight=weights.expert_w2_packed,
        w2_weight_scale=weights.expert_w2_scale,
        gate_weight=weights.router_weight,
        gate_correction_bias=weights.router_correction_bias,
        shared_gate_up_proj=torch.cat(
            (weights.shared_gate_proj, weights.shared_up_proj), dim=0
        ).contiguous(),
        shared_down_proj=weights.shared_down_proj,
        routed_expert_down_proj=weights.routed_expert_down_proj,
        routed_expert_up_proj=weights.routed_expert_up_proj,
        routed_expert_norm=weights.routed_latent_rmsnorm_weight,
    )


def expected_native_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "w13_weight": (KIMI_K3_NUM_EXPERTS, 2 * ROUTED_PER_RANK, KIMI_K3_LATENT_SIZE // 2),
        "w13_weight_scale": (
            KIMI_K3_NUM_EXPERTS,
            2 * ROUTED_PER_RANK,
            KIMI_K3_LATENT_SIZE // 32,
        ),
        "w2_weight": (KIMI_K3_NUM_EXPERTS, KIMI_K3_LATENT_SIZE, ROUTED_PER_RANK // 2),
        "w2_weight_scale": (
            KIMI_K3_NUM_EXPERTS,
            KIMI_K3_LATENT_SIZE,
            ROUTED_PER_RANK // 32,
        ),
        "gate_weight": (KIMI_K3_NUM_EXPERTS, KIMI_K3_HIDDEN_SIZE),
        "gate_correction_bias": (KIMI_K3_NUM_EXPERTS,),
        "shared_gate_up_proj": (2 * SHARED_PER_RANK, KIMI_K3_HIDDEN_SIZE),
        "shared_down_proj": (KIMI_K3_HIDDEN_SIZE, SHARED_PER_RANK),
        "routed_expert_down_proj": (KIMI_K3_LATENT_SIZE, KIMI_K3_HIDDEN_SIZE),
        "routed_expert_up_proj": (KIMI_K3_HIDDEN_SIZE, KIMI_K3_LATENT_SIZE),
        "routed_expert_norm": (KIMI_K3_LATENT_SIZE,),
    }


def check_native_shapes(mapped: NativeWeights) -> None:
    expected = expected_native_shapes()
    for name, shape in expected.items():
        actual = tuple(getattr(mapped, name).shape)
        if actual != shape:
            raise ValueError(f"{name} must have shape {shape}, got {actual}")


EXPERT_TENSOR_NAMES = (
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_bias",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_bias",
)


def expert_tensor_shapes(experts: Any) -> dict[str, Any]:
    """Snapshot the expert tensors a native runner currently exposes.

    ``process_weights_after_loading`` is free to drop, rename, or re-type these
    attributes when it converts to a kernel format, so the snapshot records
    whatever is present rather than assuming a fixed set.
    """
    snapshot: dict[str, Any] = {}
    for name in EXPERT_TENSOR_NAMES:
        value = getattr(experts, name, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        snapshot[name] = {
            "shape": list(shape) if shape is not None else None,
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
        }
    return snapshot


def copy_into(destination: torch.Tensor, source: torch.Tensor) -> None:
    if tuple(destination.shape) != tuple(source.shape):
        raise ValueError(
            f"cannot load {tuple(source.shape)} into {tuple(destination.shape)}"
        )
    destination.data.copy_(source)


def router_reference(hidden: torch.Tensor, weights: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return kimi_k3_router_reference(
        hidden,
        weights.router_weight,
        weights.router_correction_bias,
    )


def compare_routes(
    native_ids: torch.Tensor,
    native_weights_tensor: torch.Tensor,
    hidden: torch.Tensor,
    weights: Any,
) -> dict[str, Any]:
    """Compare a native router decision with the official Kimi K3 contract."""
    reference_ids, reference_weights = router_reference(hidden, weights)
    native_sorted, native_order = torch.sort(native_ids.long(), dim=-1)
    reference_sorted, reference_order = torch.sort(reference_ids.long(), dim=-1)
    ids_match = bool(torch.equal(native_sorted, reference_sorted))
    aligned_native = torch.gather(
        native_weights_tensor.float(), -1, native_order
    )
    aligned_reference = torch.gather(reference_weights.float(), -1, reference_order)
    difference = (aligned_native - aligned_reference).abs()
    return {
        "expert_ids_match": ids_match,
        "expert_id_mismatch_count": int(
            (native_sorted != reference_sorted).sum()
        ),
        "router_weight_max_abs": float(difference.max()),
        "router_weight_mean_abs": float(difference.mean()),
        "topk": int(native_ids.shape[-1]),
        "distinct_experts": int(torch.unique(native_ids).numel()),
    }


def all_reduced(tensor: torch.Tensor) -> torch.Tensor:
    reduced = tensor.bfloat16().contiguous()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def shared_reference(hidden: torch.Tensor, weights: Any) -> torch.Tensor:
    """The TP8-reduced shared-expert output under the official contract."""
    from mok.kimi_k3 import kimi_k3_situ_reference

    activations = hidden.float()
    gate = (activations @ weights.shared_gate_proj.float().T).bfloat16()
    up = (activations @ weights.shared_up_proj.float().T).bfloat16()
    activated = kimi_k3_situ_reference(gate, up)
    partial = activated.float() @ weights.shared_down_proj.float().T
    return all_reduced(partial)


def latent_reference(hidden: torch.Tensor, weights: Any) -> torch.Tensor:
    return (
        hidden.float() @ weights.routed_expert_down_proj.float().T
    ).bfloat16()


def tensor_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    left = actual.float()
    right = expected.float()
    difference = left - right
    return {
        "relative_l1": float(
            difference.abs().sum() / right.abs().sum().clamp_min(1e-12)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                left.flatten(), right.flatten(), dim=0
            )
        ),
        "max_abs": float(difference.abs().max()),
    }


@dataclass
class GraphPool:
    """CUDA graphs for one route pool, kept alive with their static inputs."""

    graphs: list[torch.cuda.CUDAGraph] = field(default_factory=list)
    outputs: list[torch.Tensor] = field(default_factory=list)
    memory_pool: Any = None

    def clear(self) -> None:
        self.graphs.clear()
        self.outputs.clear()
        self.memory_pool = None


__all__ = [
    "ARCHITECTURES",
    "EXPERT_TENSOR_NAMES",
    "GraphPool",
    "LAYER_PREFIX",
    "MODEL_CONFIG",
    "NativeWeights",
    "ROUTED_PER_RANK",
    "SHARED_IGNORED_LAYERS",
    "SHARED_PER_RANK",
    "all_reduced",
    "check_native_shapes",
    "compare_routes",
    "copy_into",
    "expected_native_shapes",
    "expert_tensor_shapes",
    "latent_reference",
    "native_weights",
    "router_reference",
    "shared_reference",
    "tensor_stats",
    "write_model_config",
]
