"""Adapter around vLLM's complete native Kimi K3 sparse-MoE layer.

The adapter builds one real ``vllm.models.kimi_k3.nvidia.model.KimiMoE`` on
the running TP8 group and drives its own ``forward``: the FP32
router, the replicated latent down projection, the MXFP4 ``FusedMoE`` with the
native SiTU activation, the shared experts, the fused latent/shared reduction,
the RMSNorm, the replicated latent up projection, and the final add. No stage
is reimplemented here and no stage is skipped.

Every vLLM and FlashInfer import is deferred into the functions below so the
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
    GraphPool,
    all_reduced,
    check_native_shapes,
    compare_routes,
    copy_into,
    expert_tensor_shapes,
    latent_reference,
    native_weights,
    router_fingerprint,
    shared_reference,
    tensor_stats,
    write_model_config,
)

FRAMEWORK = "vllm"
RECORDED_PACKAGES = (
    "vllm",
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


class VllmKimiK3Adapter:
    """Own one native vLLM Kimi K3 MoE layer and its captured graph pool."""

    name = FRAMEWORK

    def __init__(self, *, device: torch.device, tp_rank: int, tp_size: int, weights: Any):
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._exit_stack = contextlib.ExitStack()
        self._pool = GraphPool()
        self._router_graphs: list[torch.cuda.CUDAGraph] = []
        self._transformations: list[dict[str, Any]] = []
        self._config_dir = self._exit_stack.enter_context(
            tempfile.TemporaryDirectory(prefix="kimi-k3-vllm-")
        )
        write_model_config(self._config_dir, FRAMEWORK)
        self._layer = self._build_layer()
        self._load(weights)

    # -- construction ----------------------------------------------------

    def _build_layer(self) -> Any:
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
        from vllm.models.kimi_k3.nvidia.model import KimiMoE
        from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

        # EngineArgs wires every sub-config the way a real server would, which
        # is what the layer's own constructor reads (kernel backend selection,
        # parallel config, compilation config).
        vllm_config = EngineArgs(
            model=self._config_dir,
            tokenizer=self._config_dir,
            skip_tokenizer_init=True,
            trust_remote_code=False,
            dtype="bfloat16",
            seed=0,
            max_model_len=MODEL_CONFIG["max_position_embeddings"],
            tensor_parallel_size=self.tp_size,
            load_format="dummy",
            enforce_eager=False,
        ).create_engine_config()

        self._vllm_config = vllm_config
        text_config = KimiLinearConfig(
            **{
                key: value
                for key, value in MODEL_CONFIG.items()
                if key != "model_type"
            }
        )
        quant_config = Mxfp4Config(ignored_layers=list(_ignored_layers()))
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            # Both the group setup and the layer read the ambient config, so
            # they have to share one context.
            with set_current_vllm_config(vllm_config):
                init_distributed_environment(
                    world_size=self.tp_size,
                    rank=self.tp_rank,
                    distributed_init_method="env://",
                    local_rank=self.device.index,
                    backend="nccl",
                )
                ensure_model_parallel_initialized(self.tp_size, 1)
                layer = KimiMoE(
                    config=text_config,
                    vllm_config=vllm_config,
                    quant_config=quant_config,
                    prefix=LAYER_PREFIX,
                    layer_idx=1,
                )
        finally:
            torch.set_default_dtype(previous_dtype)
        return layer.to(self.device)

    # -- weight loading --------------------------------------------------

    def _load(self, weights: Any) -> None:
        mapped = native_weights(weights)
        check_native_shapes(mapped)
        layer = self._layer
        experts = _routed_experts(layer)

        copy_into(experts.w13_weight, mapped.w13_weight)
        copy_into(experts.w13_weight_scale, mapped.w13_weight_scale)
        copy_into(experts.w2_weight, mapped.w2_weight)
        copy_into(experts.w2_weight_scale, mapped.w2_weight_scale)
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
                "backend": str(getattr(experts.quant_method, "mxfp4_backend", "")),
                "description": (
                    "one-time MXFP4 kernel-format conversion of the canonical "
                    "E2M1 bytes and E8M0 group-32 scales; no requantization"
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
        torch.cuda.synchronize(self.device)

    # -- driving ---------------------------------------------------------

    def bind_router(
        self,
        router_weight: torch.Tensor,
        correction_bias: torch.Tensor,
    ) -> dict[str, Any]:
        """Install the run's one router, once, and fingerprint what landed.

        There is no reloading it. A CUDA graph records the address of
        ``gate.weight``, so writing it between two captures makes both graphs
        replay whichever routing was written last; the pool of routings has to
        be inside the one tensor instead.
        """
        copy_into(self._layer.gate.weight, router_weight)
        copy_into(self._layer.gate.e_score_correction_bias, correction_bias)
        torch.cuda.synchronize(self.device)
        return self.router_fingerprint()

    def router_fingerprint(self) -> dict[str, Any]:
        return router_fingerprint(
            self._layer.gate.weight.data,
            self._layer.gate.e_score_correction_bias.data,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with self._forward_context(hidden.shape[0]):
            return self._layer(hidden)

    def _router_topk(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The native top-16 decision, exactly as the layer's forward takes it."""
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            fused_grouped_topk,
        )

        layer = self._layer
        with self._forward_context(hidden.shape[0]):
            logits, _ = layer.gate(hidden)
            topk_weights, topk_ids = fused_grouped_topk(
                hidden_states=hidden,
                gating_output=logits,
                topk=layer.experts.moe_config.experts_per_token,
                renormalize=layer.moe_renormalize,
                e_score_correction_bias=layer.gate.e_score_correction_bias.data,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                scoring_func=layer.moe_router_activation_func,
                routed_scaling_factor=layer.routed_scaling_factor,
            )
        return topk_ids, topk_weights

    @contextlib.contextmanager
    def _forward_context(self, num_tokens: int):
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context

        with set_current_vllm_config(self._vllm_config):
            with set_forward_context(
                None, self._vllm_config, num_tokens=num_tokens
            ):
                yield

    def router_comparison(self, hidden: torch.Tensor, weights: Any) -> dict[str, Any]:
        topk_ids, topk_weights = self._router_topk(hidden)
        return compare_routes(topk_ids, topk_weights, hidden, weights)

    def stage_parity(self, hidden: torch.Tensor, weights: Any) -> dict[str, Any]:
        layer = self._layer
        with self._forward_context(hidden.shape[0]):
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
        """Capture one graph per pool entry, touching no parameter in between.

        Each entry differs only in its hidden state, which is the entry's own
        static input tensor, so graph ``p`` reads entry ``p``'s coordinates out
        of the run's one router and produces entry ``p``'s routing.
        """
        from vllm.distributed.parallel_state import graph_capture

        self.release()
        for entry in pool:
            self.forward(entry.hidden)
        torch.cuda.synchronize(self.device)

        with graph_capture(device=self.device) as context:
            for entry in pool:
                with self._forward_context(entry.hidden.shape[0]):
                    self._layer(entry.hidden)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with self._forward_context(entry.hidden.shape[0]):
                    with torch.cuda.graph(
                        graph,
                        pool=self._pool.memory_pool,
                        stream=context.stream,
                    ):
                        output = self._layer(entry.hidden)
                self._pool.memory_pool = graph.pool()
                self._pool.graphs.append(graph)
                self._pool.outputs.append(output)
        torch.cuda.synchronize(self.device)
        return list(self._pool.graphs)

    def capture_router(
        self, pool: list[Any]
    ) -> tuple[list[torch.cuda.CUDAGraph], list[torch.Tensor]]:
        """Capture the router decision alone, one graph per pool entry.

        The full graphs write only a final hidden state, so what routing they
        replayed can only be inferred from it. These read the same
        ``gate.weight`` through the same top-16 the layer's forward uses and
        keep the expert ids in a buffer a replay overwrites, which is a direct
        reading of what each graph actually selects.
        """
        from vllm.distributed.parallel_state import graph_capture

        for entry in pool:
            self._router_topk(entry.hidden)
        torch.cuda.synchronize(self.device)

        graphs: list[torch.cuda.CUDAGraph] = []
        outputs: list[torch.Tensor] = []
        memory_pool: Any = None
        with graph_capture(device=self.device) as context:
            for entry in pool:
                self._router_topk(entry.hidden)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, pool=memory_pool, stream=context.stream
                ):
                    ids, _ = self._router_topk(entry.hidden)
                memory_pool = graph.pool()
                graphs.append(graph)
                outputs.append(ids)
        torch.cuda.synchronize(self.device)
        self._router_graphs = graphs
        return graphs, outputs

    def graph_outputs(self) -> list[torch.Tensor]:
        return list(self._pool.outputs)

    def release_router(self) -> None:
        self._router_graphs = []
        torch.cuda.synchronize(self.device)

    def release(self) -> None:
        self._pool.clear()
        self._router_graphs = []
        torch.cuda.synchronize(self.device)

    # -- reporting -------------------------------------------------------

    def transformations(self) -> list[dict[str, Any]]:
        return list(self._transformations)

    def versions(self) -> dict[str, Any]:
        import vllm

        return {
            "framework": FRAMEWORK,
            "distributions": _distribution_versions(),
            "vllm_version": vllm.__version__,
            "vllm_commit": getattr(vllm, "__version_tuple__", None) and str(
                vllm.__version_tuple__
            ),
            "flashinfer_private_cubin_dir": os.environ.get(
                "FLASHINFER_PRIVATE_CUBIN_DIR", "unset"
            ),
            "torch_cuda": torch.version.cuda,
            "layer_class": type(self._layer).__name__,
            "expert_quant_method": type(
                _routed_experts(self._layer).quant_method
            ).__name__,
            "runner_class": type(self._layer.experts).__name__,
        }

    def close(self) -> None:
        self.release()
        self._exit_stack.close()


def _routed_experts(layer: Any) -> Any:
    """The module that owns the MXFP4 expert parameters.

    ``FusedMoE`` is a factory that returns a ``MoERunner`` (here a
    ``LatentMoERunner``); the parameters and their quant method live one level
    down on its ``routed_experts``.
    """
    return layer.experts.routed_experts


def _ignored_layers() -> tuple[str, ...]:
    return (
        f"{LAYER_PREFIX}.shared_experts.gate_up_proj",
        f"{LAYER_PREFIX}.shared_experts.down_proj",
        f"{LAYER_PREFIX}.routed_expert_down_proj",
        f"{LAYER_PREFIX}.routed_expert_up_proj",
    )


def build_adapter(
    *,
    device: torch.device,
    tp_rank: int,
    tp_size: int,
    weights: Any,
) -> VllmKimiK3Adapter:
    return VllmKimiK3Adapter(
        device=device,
        tp_rank=tp_rank,
        tp_size=tp_size,
        weights=weights,
    )
