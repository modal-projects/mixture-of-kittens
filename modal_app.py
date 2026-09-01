"""Modal app for building and running Mixture-of-Kittens (MoK) on Blackwell GPUs.

Every function and entrypoint lives in one of the five modules imported below.
This file is the one `modal run` is pointed at, so it is the one place that has
to import all of them: each `@app.function` and `@app.local_entrypoint` is
registered by the import, and each is re-exported by name because `modal run
modal_app.py::verify` resolves an attribute of *this* module. Importing the
five modules alone would register everything and still leave every documented
command below unresolvable, so `test_every_registered_name_is_reachable_by_name`
holds the two lists together.

Usage (from the repo root, with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set):

    modal run modal_app.py                 # build check on a single B300
    modal run modal_app.py::gpu_info       # same, explicit
    modal run modal_app.py::bench          # 8x B300 benchmark + correctness check
    modal run modal_app.py::verify         # the Kimi K3 gates
    modal run modal_app.py::engine_probe   # the gate/up engine A/B
    modal run modal_app.py::compare        # the pinned framework comparison

Environment overrides:
    MOK_GPU          (default B300)  which spec to use (B300 or B200)
    MOK_BENCH_NPROC  (default 8)     GPUs / EP ranks for the benchmark (1, 4, or 8)
"""

from modal_images import app
from modal_bench import (  # noqa: F401
    bench,
    bench_kimi_k3_decode,
    gpu_info,
    test_kimi_k3_decode,
)
from modal_k3_gates import (  # noqa: F401
    bench_kimi_k3_decode_persisted,
    sanitize_kimi_k3_decode,
    sass_kimi_k3_decode,
    stress_kimi_k3_schedule,
    trap_kimi_k3_schedule,
    verify,
    verify_kimi_k3,
)
from modal_k3_probes import (  # noqa: F401
    batched_expert_diagnostic,
    batched_expert_probe,
    bench_kimi_k3_batched_expert_probe,
    bench_kimi_k3_engine_probe,
    bench_kimi_k3_schedule_probe,
    diagnose_kimi_k3_batched_expert_probe,
    engine_probe,
    schedule_probe,
)
from modal_frameworks import (  # noqa: F401
    compare,
    compare_sglang,
    compare_vllm,
    graph_routes,
    graph_routes_sglang,
    graph_routes_vllm,
)


@app.local_entrypoint()
def main() -> None:
    gpu_info.remote()
