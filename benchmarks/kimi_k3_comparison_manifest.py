"""Pins, shapes, and the manifest the Kimi K3 backend comparison records.

Everything here answers "what was compared, against what, built from what",
and none of it touches a GPU: the pinned registry digests, the per-rank tensor
shapes both native layers expose, the installed-distribution capture, and the
manifest that ends up at the head of every archive. The driver in
:mod:`benchmarks.compare_kimi_k3_frameworks` imports these and re-exports them,
so the archive's shape is decided in one place.
"""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_decode_inputs import GRAPH_POOL_SIZE

BENCHMARK = "kimi_k3_framework_comparison"
CUSTOM_BACKEND = "mok"
BASELINE_BACKENDS = ("vllm", "sglang")
ADAPTER_MODULES = {
    "vllm": "benchmarks.frameworks.vllm_kimi_k3",
    "sglang": "benchmarks.frameworks.sglang_kimi_k3",
}

TP_SIZE = 8
WARMUP_COUNT = 500
SAMPLE_COUNT = 1000
BLOCK8_SHAPES = tuple(range(8, 65, 8))
BLOCK16_SHAPES = tuple(range(16, 129, 16))
SHAPE_GROUPS = {"block8": BLOCK8_SHAPES, "block16": BLOCK16_SHAPES}
GATE_SHAPES = BLOCK16_SHAPES
CONCURRENCY_ONE_TOKENS = 16
P99_LIMIT_RATIO = 1.10

# The official-reference tolerances the amended design gates the custom kernel
# on. A native backend is measured against the same numbers, but its own
# distance from the reference is reported rather than required.
NUMERICAL_TOLERANCES = {
    "relative_l1": 0.05,
    "cosine_similarity": 0.999,
    "max_abs": 1.0,
}

# The top-16 selection is discrete, so both the identities and the normalized
# weights are held to an exactness rather than a tolerance.
ROUTER_WEIGHT_MAX_ABS = 1e-5

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
ROUTED_INTERMEDIATE_SIZE = 3072
SHARED_INTERMEDIATE_SIZE = 6144
NUM_EXPERTS = 896
TOPK = 16
MXFP4_GROUP_SIZE = 32

MANIFEST_PATH = Path(__file__).with_name("framework_manifest.json")
REQUIRED_FRAMEWORK_PINS = (
    "image",
    "image_digest",
    "image_amd64_digest",
    "model",
    "tensor_parallel_size",
)

# The environment variable a Modal comparison container reports the registry
# reference its image was actually derived from through.
IMAGE_REFERENCE_ENV = "MOK_COMPARISON_IMAGE_REF"

ARTIFACT_FILES = (
    "manifest.json",
    "versions.json",
    "transformations.json",
    "parity.json",
    "numerical_gates.json",
    "route_occupancy.json",
    "latency_block8.json",
    "latency_block8.csv",
    "latency_block16.json",
    "latency_block16.csv",
    "raw_samples.json",
    "launch_traces.json",
    "phase_profile.json",
    "performance_gates.json",
)

LATENCY_COLUMNS = (
    "backend",
    "mode",
    "tokens",
    "requests",
    "distinct_experts_per_replay",
    "graph_pool_size",
    "warmup_count",
    "sample_count",
    "median_ms",
    "p90_ms",
    "p99_ms",
    "geomean_ms",
)


def comparison_artifact_files(modes: Sequence[str]) -> tuple[str, ...]:
    """Return the artifacts one run writes for the modes it measures."""
    unmeasured = set(SHAPE_GROUPS) - set(modes)
    skipped = {
        f"latency_{mode}.{suffix}"
        for mode in unmeasured
        for suffix in ("json", "csv")
    }
    return tuple(name for name in ARTIFACT_FILES if name not in skipped)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def load_framework_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the pinned framework manifest."""
    manifest_path = MANIFEST_PATH if path is None else Path(path)
    manifest = json.loads(manifest_path.read_text())

    for framework in BASELINE_BACKENDS:
        entry = manifest.get(framework)
        if not isinstance(entry, dict):
            raise ValueError(f"{framework} manifest entry is missing")
        for key in REQUIRED_FRAMEWORK_PINS:
            if key not in entry:
                raise ValueError(f"{framework} manifest entry is missing {key}")
        for key in ("image_digest", "image_amd64_digest"):
            if not str(entry[key]).startswith("sha256:"):
                raise ValueError(
                    f"{framework} {key} must be a sha256 registry digest"
                )
        if entry["tensor_parallel_size"] != TP_SIZE:
            raise ValueError(
                f"{framework} tensor_parallel_size must be {TP_SIZE}, "
                f"got {entry['tensor_parallel_size']}"
            )
    if manifest.get("gpu") != f"B300:{TP_SIZE}":
        raise ValueError(f"manifest gpu must be B300:{TP_SIZE}")
    dflash = manifest.get("dflash")
    if not isinstance(dflash, dict) or dflash.get("block_sizes") != [8, 16]:
        raise ValueError("dflash manifest entry must pin block sizes 8 and 16")
    distributions = manifest.get("recorded_distributions")
    if not isinstance(distributions, list) or distributions[:2] != [
        "torch",
        "triton",
    ]:
        raise ValueError(
            "recorded_distributions must start with torch and triton"
        )
    return manifest


# --------------------------------------------------------------------------
# Registry digest binding
# --------------------------------------------------------------------------


def pinned_image_reference(
    framework: str,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Return ``repository@sha256:<digest>`` for one pinned serving image.

    A tag moves. ``vllm/vllm-openai:kimi-k3`` resolved to one manifest on the
    day the versions in this archive were captured and may resolve to another
    tomorrow, so the image a comparison is built from is named by the digest
    the manifest pins and the tag survives only as documentation of where that
    digest came from.
    """
    if framework not in ADAPTER_MODULES:
        raise ValueError(f"unknown framework {framework!r}")
    pins = load_framework_manifest() if manifest is None else manifest
    entry = pins[framework]
    repository = str(entry["image"]).split("@", 1)[0].rsplit(":", 1)[0]
    return f"{repository}@{entry['image_digest']}"


def effective_image_reference(
    framework: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the reference this container's image was derived from.

    The builder records what it actually used; this refuses anything that is
    not the pin, so an archive can never claim a digest it was not produced
    from.
    """
    pinned = pinned_image_reference(framework)
    env = os.environ if environment is None else environment
    reported = env.get(IMAGE_REFERENCE_ENV)
    if reported is None:
        return pinned
    if reported != pinned:
        raise ValueError(
            f"{framework} image reference {reported!r} does not match the "
            f"pinned digest {pinned!r}"
        )
    return reported


# --------------------------------------------------------------------------
# Adapter shape contract
# --------------------------------------------------------------------------


def adapter_weight_shapes() -> dict[str, tuple[int, ...]]:
    """Return the per-rank TP8 tensor shapes both native layers expose.

    The routed experts run in latent space, so their contraction extent is the
    ``3584`` latent width rather than the ``7168`` hidden width, and each rank
    owns ``3072 / 8`` routed and ``6144 / 8`` shared intermediate columns.
    """
    routed = ROUTED_INTERMEDIATE_SIZE // TP_SIZE
    shared = SHARED_INTERMEDIATE_SIZE // TP_SIZE
    return {
        "w13_weight": (NUM_EXPERTS, 2 * routed, LATENT_SIZE // 2),
        "w13_weight_scale": (
            NUM_EXPERTS,
            2 * routed,
            LATENT_SIZE // MXFP4_GROUP_SIZE,
        ),
        "w2_weight": (NUM_EXPERTS, LATENT_SIZE, routed // 2),
        "w2_weight_scale": (
            NUM_EXPERTS,
            LATENT_SIZE,
            routed // MXFP4_GROUP_SIZE,
        ),
        "gate_weight": (NUM_EXPERTS, HIDDEN_SIZE),
        "gate_correction_bias": (NUM_EXPERTS,),
        "shared_gate_up_proj": (2 * shared, HIDDEN_SIZE),
        "shared_down_proj": (HIDDEN_SIZE, shared),
        "routed_expert_down_proj": (LATENT_SIZE, HIDDEN_SIZE),
        "routed_expert_up_proj": (HIDDEN_SIZE, LATENT_SIZE),
        "routed_expert_norm": (LATENT_SIZE,),
    }


def fused_gate_up_plan() -> dict[str, Any]:
    """Describe how the separate gate and up matrices fill a fused row block.

    Both frameworks store one fused ``w13``/``gate_up`` matrix whose first half
    of rows is the gate projection and whose second half is the up projection;
    the custom kernel keeps them as two matrices. This is the only reordering
    the adapters perform on expert bytes.
    """
    return {
        "w13_row_order": ["w1_gate", "w3_up"],
        "w13_rows_per_half": ROUTED_INTERMEDIATE_SIZE // TP_SIZE,
        "shared_row_order": ["gate", "up"],
        "shared_rows_per_half": SHARED_INTERMEDIATE_SIZE // TP_SIZE,
    }


# --------------------------------------------------------------------------
# Version capture
# --------------------------------------------------------------------------


def _installed_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def capture_versions(
    names: Sequence[str],
    *,
    resolver: Callable[[str], str | None] | None = None,
) -> dict[str, str]:
    """Resolve installed distribution versions, refusing an unpinned torch."""
    lookup = _installed_version if resolver is None else resolver
    captured: dict[str, str] = {}
    for name in names:
        found = lookup(name)
        if found is None and name == "torch":
            raise ValueError(
                "torch must be installed to record a comparison version pin"
            )
        captured[name] = "not-installed" if found is None else str(found)
    return captured


# --------------------------------------------------------------------------
# Comparison manifest
# --------------------------------------------------------------------------


def _git_sha() -> str:
    return os.environ.get("MOK_GIT_SHA", "unavailable")


def build_comparison_manifest(
    *,
    framework: str,
    dry_run: bool,
    warmup_count: int = WARMUP_COUNT,
    sample_count: int = SAMPLE_COUNT,
    shape_groups: Mapping[str, Sequence[int]] | None = None,
    pool_size: int = GRAPH_POOL_SIZE,
) -> dict[str, Any]:
    """Describe one comparison run, at the settings it actually ran with.

    The counts, the measured shapes, and the route-pool depth are arguments
    rather than module constants because the CLI can override all four, and a
    manifest that reported the defaults while the run used something else would
    mislabel every table in the archive.
    """
    if framework not in ADAPTER_MODULES:
        raise ValueError(f"unknown framework {framework!r}")
    pins = load_framework_manifest()
    measured = SHAPE_GROUPS if shape_groups is None else shape_groups
    return {
        "benchmark": BENCHMARK,
        "dry_run": dry_run,
        "framework": framework,
        "backends": [CUSTOM_BACKEND, framework],
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu": pins["gpu"],
        "tp_size": TP_SIZE,
        "image_reference": effective_image_reference(framework),
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "graph_pool_size": pool_size,
        "shape_groups": {
            name: list(shapes) for name, shapes in measured.items()
        },
        "numerical_tolerances": dict(NUMERICAL_TOLERANCES),
        "numerical_gates": {
            "gated_comparison": "custom_vs_reference",
            "router_weight_max_abs": ROUTER_WEIGHT_MAX_ABS,
            "diagnostic_comparisons": [
                "custom_vs_native",
                "native_vs_reference",
                "routed_latent_vs_reference",
                "shared_output_vs_reference",
            ],
        },
        "performance_gates": {
            "concurrency1_tokens": CONCURRENCY_ONE_TOKENS,
            "geomean_tokens": list(GATE_SHAPES),
            "p99_limit_ratio": P99_LIMIT_RATIO,
        },
        "framework_pins": {
            key: pins[key] for key in ("vllm", "sglang", "dflash")
        },
        "recorded_distributions": list(pins["recorded_distributions"]),
        "routing": {
            "source": "benchmarks.kimi_k3_decode_data.build_routed_input",
            "per_replay_occupancy": "min(16 * tokens, 896)",
            "pool_entries": pool_size,
        },
        "timing": {
            "unit": "milliseconds",
            "operation": "CUDA Graph replay only",
            "rank_reduction": "maximum per iteration across all eight ranks",
            "percentile_method": "R-7 linear interpolation",
            "event_priming": (
                "one event pair recorded and discarded before the persisted "
                "samples"
            ),
        },
        "adapter_weight_shapes": {
            name: list(shape) for name, shape in adapter_weight_shapes().items()
        },
        "fused_gate_up_plan": fused_gate_up_plan(),
        "artifact_files": list(ARTIFACT_FILES),
    }


__all__ = [
    "ADAPTER_MODULES",
    "ARTIFACT_FILES",
    "BASELINE_BACKENDS",
    "BENCHMARK",
    "BLOCK8_SHAPES",
    "BLOCK16_SHAPES",
    "CONCURRENCY_ONE_TOKENS",
    "CUSTOM_BACKEND",
    "GATE_SHAPES",
    "HIDDEN_SIZE",
    "IMAGE_REFERENCE_ENV",
    "LATENCY_COLUMNS",
    "LATENT_SIZE",
    "MANIFEST_PATH",
    "MXFP4_GROUP_SIZE",
    "NUMERICAL_TOLERANCES",
    "NUM_EXPERTS",
    "P99_LIMIT_RATIO",
    "REQUIRED_FRAMEWORK_PINS",
    "ROUTED_INTERMEDIATE_SIZE",
    "ROUTER_WEIGHT_MAX_ABS",
    "SAMPLE_COUNT",
    "SHAPE_GROUPS",
    "SHARED_INTERMEDIATE_SIZE",
    "TOPK",
    "TP_SIZE",
    "WARMUP_COUNT",
    "adapter_weight_shapes",
    "build_comparison_manifest",
    "capture_versions",
    "comparison_artifact_files",
    "effective_image_reference",
    "fused_gate_up_plan",
    "load_framework_manifest",
    "pinned_image_reference",
]
