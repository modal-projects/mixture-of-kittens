"""Adapter around SGLang's complete native Kimi K3 sparse-MoE layer.

The adapter builds one real ``sglang.srt.models.kimi_k3.KimiK3MoE`` on the
running TP8 group with the ``flashinfer_mxfp4`` MoE runner, then drives its own
``forward``: the FP32 gate, the replicated latent down projection, the
trtllm-gen MXFP4 expert runner with the native SiTU activation, the shared
experts, the latent/shared reduction, the RMSNorm, the replicated latent up
projection, and the final add. No stage is reimplemented and none is skipped.

Every SGLang and FlashInfer import is deferred into the functions below so the
base ``mok`` package and the CPU contract tests never need a serving runtime.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any

import torch

from benchmarks.frameworks.kimi_k3_adapter_common import (
    LAYER_PREFIX,
    MODEL_CONFIG,
    SHARED_IGNORED_LAYERS,
    GraphPool,
    all_reduced,
    check_native_shapes,
    compare_routes,
    copy_into,
    expert_tensor_shapes,
    latent_reference,
    native_weights,
    shared_reference,
    tensor_stats,
    write_model_config,
)

FRAMEWORK = "sglang"
MOE_RUNNER_BACKEND = "flashinfer_mxfp4"
RECORDED_PACKAGES = (
    "sglang",
    "sglang-kernel",
    "torch",
    "triton",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "transformers",
    "nvidia-cublas",
    "nvidia-nccl-cu13",
)


def _distribution_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    captured = {}
    for name in RECORDED_PACKAGES:
        try:
            captured[name] = version(name)
        except PackageNotFoundError:
            captured[name] = "not-installed"
    return captured


class SglangKimiK3Adapter:
    """Own one native SGLang Kimi K3 MoE layer and its captured graph pool."""

    name = FRAMEWORK

    def __init__(self, *, device: torch.device, tp_rank: int, tp_size: int, weights: Any):
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._exit_stack = contextlib.ExitStack()
        self._pool = GraphPool()
        self._transformations: list[dict[str, Any]] = []
        self._config_dir = self._exit_stack.enter_context(
            tempfile.TemporaryDirectory(prefix="kimi-k3-sglang-")
        )
        write_model_config(self._config_dir, FRAMEWORK)
        self._layer = self._build_layer()
        self._load(weights)

    # -- construction ----------------------------------------------------

    def _build_layer(self) -> Any:
        from sglang.srt.configs.kimi_linear import KimiLinearConfig
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
        from sglang.srt.models.kimi_k3 import KimiK3MoE
        from sglang.srt.runtime_context import get_context
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=self._config_dir,
            tokenizer_path=self._config_dir,
            skip_tokenizer_init=True,
            tp_size=self.tp_size,
            dtype="bfloat16",
            moe_runner_backend=MOE_RUNNER_BACKEND,
            trust_remote_code=False,
            load_format="dummy",
            disable_cuda_graph=False,
        )
        get_context().set_server_args(server_args)

        init_distributed_environment(
            world_size=self.tp_size,
            rank=self.tp_rank,
            distributed_init_method="env://",
            local_rank=self.device.index,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)

        config = KimiLinearConfig(
            **{
                key: value
                for key, value in MODEL_CONFIG.items()
                if key != "model_type"
            }
        )
        quant_config = Mxfp4Config(
            ignored_layers=list(SHARED_IGNORED_LAYERS),
            is_checkpoint_mxfp4_serialized=True,
        )
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = KimiK3MoE(
                config=config,
                quant_config=quant_config,
                prefix=LAYER_PREFIX,
                layer_idx=1,
                alt_stream=torch.cuda.Stream(device=self.device),
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        return layer.to(self.device)

    # -- weight loading --------------------------------------------------

    def _load(self, weights: Any) -> None:
        mapped = native_weights(weights)
        check_native_shapes(mapped)
        layer = self._layer
        experts = layer.experts

        copy_into(experts.w13_weight, mapped.w13_weight)
        copy_into(experts.w13_weight_scale, mapped.w13_weight_scale)
        copy_into(experts.w2_weight, mapped.w2_weight)
        copy_into(experts.w2_weight_scale, mapped.w2_weight_scale)
        for name in ("w13_weight_bias", "w2_weight_bias"):
            bias = getattr(experts, name, None)
            if bias is not None:
                bias.data.zero_()
        copy_into(layer.gate.weight, mapped.gate_weight)
        copy_into(layer.gate.e_score_correction_bias, mapped.gate_correction_bias)
        copy_into(layer.shared_experts.gate_up_proj.weight, mapped.shared_gate_up_proj)
        copy_into(layer.shared_experts.down_proj.weight, mapped.shared_down_proj)
        copy_into(layer.routed_expert_down_proj.weight, mapped.routed_expert_down_proj)
        copy_into(layer.routed_expert_up_proj.weight, mapped.routed_expert_up_proj)
        copy_into(layer.routed_expert_norm.weight, mapped.routed_expert_norm)

        before = expert_tensor_shapes(experts)
        experts.quant_method.process_weights_after_loading(experts)
        after = expert_tensor_shapes(experts)
        self._transformations.append(
            {
                "stage": "process_weights_after_loading",
                "callee": (
                    f"{type(experts.quant_method).__module__}."
                    f"{type(experts.quant_method).__name__}"
                ),
                "description": (
                    "one-time trtllm-gen MXFP4 shuffle and block-scale swizzle "
                    "of the canonical E2M1 bytes and E8M0 group-32 scales; "
                    "no requantization"
                ),
                "shapes_before": before,
                "shapes_after": after,
            }
        )
        self._transformations.append(
            {
                "stage": "fused_gate_up_concat",
                "description": (
                    "mok w1 (gate) and w3 (up) packed bytes concatenated along "
                    "the row axis into the native w13 block, gate rows first"
                ),
                "rows_per_half": mapped.w13_weight.shape[1] // 2,
            }
        )

        layer._merge_front_weights()
        self._transformations.append(
            {
                "stage": "merge_front_weights",
                "description": (
                    "native one-time merge of the shared gate_up, router gate, "
                    "and latent down weights into one front GEMM; each module "
                    "weight becomes a view of the merged buffer"
                ),
                "merged": layer._front_w is not None,
                "front_sizes": list(layer._front_sizes or []),
            }
        )
        torch.cuda.synchronize(self.device)

    # -- driving ---------------------------------------------------------

    def load_router(self, router_weight: torch.Tensor) -> None:
        copy_into(self._layer.gate.weight, router_weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._layer(hidden)

    def router_comparison(self, hidden: torch.Tensor, weights: Any) -> dict[str, Any]:
        layer = self._layer
        logits = layer.gate(hidden)
        topk_output = layer.topk(hidden, logits)
        return compare_routes(
            topk_output.topk_ids,
            topk_output.topk_weights,
            hidden,
            weights,
        )

    def stage_parity(self, hidden: torch.Tensor, weights: Any) -> dict[str, Any]:
        layer = self._layer
        latent, _ = layer.routed_expert_down_proj(hidden)
        shared = all_reduced(layer.shared_experts(hidden))
        return {
            "routed_latent_vs_reference": tensor_stats(
                latent, latent_reference(hidden, weights)
            ),
            "shared_output_vs_reference": tensor_stats(
                shared, shared_reference(hidden, weights)
            ),
        }

    # -- graph capture ---------------------------------------------------

    def capture(self, pool: list[Any]) -> list[torch.cuda.CUDAGraph]:
        from sglang.srt.distributed.parallel_state import graph_capture

        self.release()
        for entry in pool:
            self.load_router(entry.weights.router_weight)
            self.forward(entry.hidden)
        torch.cuda.synchronize(self.device)

        with graph_capture() as context:
            stream = getattr(context, "stream", None)
            for entry in pool:
                self.load_router(entry.weights.router_weight)
                self._layer(entry.hidden)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=self._pool.memory_pool,
                    stream=stream,
                ):
                    output = self._layer(entry.hidden)
                self._pool.memory_pool = graph.pool()
                self._pool.graphs.append(graph)
                self._pool.outputs.append(output)
        torch.cuda.synchronize(self.device)
        return list(self._pool.graphs)

    def release(self) -> None:
        self._pool.clear()
        torch.cuda.synchronize(self.device)

    # -- reporting -------------------------------------------------------

    def transformations(self) -> list[dict[str, Any]]:
        return list(self._transformations)

    def versions(self) -> dict[str, Any]:
        import sglang

        return {
            "framework": FRAMEWORK,
            "distributions": _distribution_versions(),
            "sglang_version": sglang.__version__,
            "moe_runner_backend": MOE_RUNNER_BACKEND,
            "flashinfer_version": os.environ.get("FLASHINFER_VERSION", "unset"),
            "torch_cuda": torch.version.cuda,
            "layer_class": type(self._layer).__name__,
            "expert_quant_method": type(self._layer.experts.quant_method).__name__,
            "fused_front_active": bool(self._layer._eligible_for_fused_front),
        }

    def close(self) -> None:
        self.release()
        self._exit_stack.close()


def build_adapter(
    *,
    device: torch.device,
    tp_rank: int,
    tp_size: int,
    weights: Any,
) -> SglangKimiK3Adapter:
    return SglangKimiK3Adapter(
        device=device,
        tp_rank=tp_rank,
        tp_size=tp_size,
        weights=weights,
    )
