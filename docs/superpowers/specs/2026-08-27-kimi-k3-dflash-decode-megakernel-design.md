# Kimi K3 DFlash Decode Megakernel Design

## Status

This design incorporates the approved standalone TP8 operator approach and the
subsequent scope changes:

- one 8x NVIDIA B300 node;
- latency-first decode;
- DFlash request concurrency 1 through 8;
- DFlash block size 16 as the primary operating point and block size 8 as the
  fallback operating point;
- direct correctness and latency comparisons with vLLM and SGLang.

## Goal

Add a forward-only, TP8 Kimi K3 MoE operator to Mixture-of-Kittens. The hot path
is one persistent CUDA launch per GPU, performs no host synchronization or
allocation, and computes the exact Kimi K3 sparse-MoE block on 1 through 128
flattened verifier tokens.

The operator is standalone but uses weight and activation contracts that can be
adapted to vLLM and SGLang without repacking on every invocation.

## Workload

The primary workload is Kimi K3 target-model verification for
`modal-labs/Kimi-K3-DFlash`:

- request concurrency: 1 through 8;
- primary draft block size: 16;
- fallback draft block size: 8;
- primary flattened target-token counts:
  `16, 32, 48, 64, 80, 96, 112, 128`;
- fallback flattened target-token counts:
  `8, 16, 24, 32, 40, 48, 56, 64`.

The kernel accepts every token count in `[1, 128]`, so it also covers
non-speculative decode and framework padding choices. It specializes execution
with capacity buckets `1, 2, 4, 8, 16, 32, 64, 128`, masks inactive rows, and
returns only the unpadded rows.

## Scope

The operator computes `KimiSparseMoeBlock`, not an entire transformer block.
Its input is the post-attention-normalized hidden state and its output is the
MoE contribution. Attention, DFlash drafting, outer RMSNorms, Attention
Residuals, and residual addition remain the caller's responsibility.

Included:

1. FP32 router logits and sigmoid scoring.
2. Correction-bias top-16 expert selection.
3. Selected-score renormalization.
4. Replicated BF16 routed latent down-projection.
5. TP8-sharded MXFP4 routed expert gate, up, and down projections.
6. Exact FP32 SiTU activation.
7. Router-weighted routed-expert combination.
8. Cross-rank routed reduction and latent RMSNorm.
9. Replicated BF16 routed latent up-projection, evaluated in output shards.
10. TP8-sharded BF16 shared-expert MLP.
11. Fused routed/shared tail communication and addition.

Excluded:

- attention and KV/KDA state;
- DFlash draft-model execution inside the custom operator;
- a production checkpoint loader or serving-engine fork;
- training and backward propagation;
- expert parallel or multi-node execution;
- GPUs other than SM103 B300 in the first implementation.

The verification harness may download checkpoints and launch unmodified vLLM
and SGLang servers. Those are test fixtures, not features of the shipped
operator.

## Exact Model Semantics

Let `x` have shape `[M, 7168]`, where `1 <= M <= 128`.

### Router

The router has 896 experts and selects 16 per token:

```text
logits       = fp32(x) @ fp32(router_weight).T
scores       = sigmoid(logits)
choice_score = scores + correction_bias
expert_ids   = topk(choice_score, 16, sorted=False)
weights      = gather(scores, expert_ids)
weights      = weights / (sum(weights, dim=-1) + 1e-20)
```

The correction bias affects expert selection only. The unmodified sigmoid
scores are gathered and normalized. The routed scaling factor is 1.

### Routed branch

```text
latent_x = x @ routed_expert_down_proj.T          # 7168 -> 3584

gate = latent_x @ expert_w1[e].T                  # 3584 -> 3072
up   = latent_x @ expert_w3[e].T                  # 3584 -> 3072

situ_gate = 4 * tanh(fp32(gate) / 4) * sigmoid(fp32(gate))
situ_up   = 25 * tanh(fp32(up) / 25)
expert_y  = (situ_gate * situ_up) @ expert_w2[e].T # 3072 -> 3584

routed_latent = sum_e(weights[e] * expert_y[e])
routed_latent = RMSNorm(routed_latent, eps=1e-5)
routed_y      = routed_latent @ routed_expert_up_proj.T # 3584 -> 7168
```

SiTU and RMSNorm accumulate in FP32 and cast their outputs back to BF16.

### Shared branch

Kimi K3's two shared experts are represented by one MLP whose intermediate
width is `2 * 3072 = 6144`:

```text
shared_gate = x @ shared_gate_proj.T
shared_up   = x @ shared_up_proj.T
shared_h    = SiTU(shared_gate, shared_up)
shared_y    = shared_h @ shared_down_proj.T
```

The BF16 shared linear modules round `shared_gate` and `shared_up` to BF16
before SiTU converts them to FP32. This matches the official module boundary
and the serving-framework implementations.

The final result is:

```text
y = routed_y + shared_y
```

Every TP rank receives the same BF16 `[M, 7168]` result.

## Precision and Weight Contract

Runtime tensors:

- hidden states and output: BF16;
- router logits, router scores, SiTU internals, and RMSNorm internals: FP32;
- routed-expert activations at MXFP4 GEMM boundaries: MXFP8 E4M3 with
  block-32 E8M0 scales;
- expert assignments accumulate locally in FP32 and cast once to BF16;
- TP8 routed and shared collectives reduce BF16 values, matching the serving
  baselines, before RMSNorm converts the routed latent back to FP32.

Replicated weights on every TP rank:

- router weight: BF16 `[896, 7168]`;
- router correction bias: FP32 `[896]`;
- routed latent down weight: BF16 `[3584, 7168]`;
- routed latent up weight: BF16 `[7168, 3584]`;
- routed latent RMSNorm weight: BF16 `[3584]`.

Rank-local TP8 shards:

- routed `w1/w3`: packed MXFP4 for all 896 experts, each with local output
  width `3072 / 8 = 384` and logical K 3584;
- routed `w2`: packed MXFP4 for all 896 experts, each consuming the matching
  local width 384 and producing a partial width-3584 result;
- shared gate/up: BF16 local output width `6144 / 8 = 768`;
- shared down: BF16 local input width 768 and partial output width 7168.

MXFP4 values use packed E2M1 data and group-size-32 E8M0 scales. The public API
accepts prepared rank-local tensors. A separate preparation helper converts test
BF16 tensors or maps framework-owned checkpoint tensors into this native
group-32 layout without dequantizing checkpoint MXFP4 values; the hot path
never repacks weights.

## TP8 Execution

All ranks receive the same hidden states and independently produce the same
router decisions. Every rank stores a TP shard of all 896 experts, matching the
latency-oriented TP8 deployment used by vLLM and SGLang.

The persistent kernel uses role-specialized CTA clusters and workspace
semaphores:

1. **Router CTAs** compute FP32 logits and top-16 choices while latent-projection
   CTAs compute the replicated `7168 -> 3584` projection.
2. **Routing CTAs** build a device-side expert-major assignment list with a
   fixed maximum of `128 * 16 = 2048` entries. No CPU schedule or device-to-host
   count transfer occurs.
3. **Expert CTAs** steal ready expert groups, dynamically quantize latent
   activations to MXFP8, run rank-local MXFP4 gate/up, apply FP32 SiTU, run the
   rank-local down projection, and accumulate weighted partial latent outputs.
   Expert grouping permits weight reuse when multiple verifier tokens select
   the same expert; no row is padded to the training kernel's 256-token tile.
4. **Shared CTAs** execute the TP-sharded BF16 shared branch concurrently with
   routed experts.
5. **Tail CTAs** perform a device-side TP8 collective, latent RMSNorm, sharded
   latent up-projection, shared-output addition, and final distribution.

### Tail collective

For `M <= 128`, the tail follows the small-batch communication decomposition
used by vLLM's Kimi K3 tail:

1. all-reduce the partial `[M, 3584]` routed latent;
2. reduce-scatter the partial `[M, 7168]` shared output;
3. apply RMSNorm to the replicated routed latent;
4. multiply by the rank's `[896, 3584]` row shard of the replicated latent-up
   weight and beta-add the `[M, 896]` shared shard;
5. use symmetric-memory mailboxes to distribute the eight hidden shards and
   assemble `[M, 7168]` on every rank.

The implementation reuses PyTorch symmetric-memory handles established during
workspace creation. It uses explicit device-side phase counters and generation
numbers so repeated CUDA Graph replays do not require zeroing buffers from the
host.

## Low-Latency Decisions

- Exactly one CUDA launch per rank for the complete MoE block.
- No per-call allocation, Python-side scheduling, host barrier, or `.item()`.
- Fixed-capacity workspace for 128 tokens and 2048 expert assignments.
- Router and latent down-projection overlap within one launch.
- Shared and routed branches overlap within one launch.
- Dedicated skinny BF16 projection paths for `M <= 8`; tcgen05 paths are used
  only when their setup cost wins for the active capacity bucket.
- Expert work is assignment-driven rather than dense over 896 experts.
- The kernel consumes checkpoint-native MXFP4 instead of dequantizing expert
  weights to BF16.
- Routed GEMMs use SM103 `mxf8f6f4` block-scaled MMA with MXFP8 E4M3
  activations, MXFP4 E2M1 weights, E8M0 scales, and K=32. K96
  `mxf4nvf4` is not used because it is an FP4-by-FP4 path.
- All supported shapes are CUDA Graph safe.

## Python API

The high-level API lives in `mok/kimi_k3.py`:

```python
config = KimiK3DecodeConfig()
workspace = create_kimi_k3_decode_workspace(
    tp_group,
    device=device,
    max_tokens=128,
)
weights = KimiK3DecodeWeights(...)
output = kimi_k3_decode(
    config,
    workspace,
    weights,
    hidden_states,
)
```

`KimiK3DecodeWeights` names the replicated and rank-local tensors explicitly.
`KimiK3DecodeWorkspace` owns symmetric mailboxes, assignment storage, temporary
activations, counters, and cached pointer lists. The low-level custom op in
`mok/ops.py` accepts unpacked tensors and pointers for compatibility with
`torch.library` and `torch.compile`.

The operator is inference-only and registered with a fake implementation that
returns BF16 `[M, 7168]`.

## Validation and Error Handling

Workspace creation validates:

- CUDA device capability is exactly SM103;
- TP world size is exactly 8;
- all ranks are in one symmetric-memory-capable process group;
- maximum token count is 128;
- rank-local weight shards have identical metadata across ranks.

Every invocation validates:

- `1 <= M <= 128`;
- exact Kimi K3 dimensions;
- dtype, device, contiguity, alignment, packed MXFP4 layout, and scale layout;
- all tensors belong to the workspace device;
- weight and workspace TP ranks match.

Invalid inputs raise host-side `TypeError`, `ValueError`, or
`NotImplementedError` before launch. Device code writes an error flag for
impossible internal invariant violations; debug tests inspect it after the
normal stream synchronization. The production hot path does not synchronously
read that flag.

## Testing

### Reference correctness

A PyTorch reference implements the official Kimi K3 equations using the same
dequantized MXFP4 values as the kernel. It covers:

- every raw decode size `M = 1..8`;
- DFlash block-8 request batches `1..8`;
- DFlash block-16 request batches `1..8`;
- balanced random routing;
- all tokens selecting the same 16 experts;
- disjoint expert selections;
- repeated CUDA Graph replay;
- workspace reuse across changing token counts.

Required checks:

- router expert IDs are exact for non-tied scores;
- normalized selected weights have max absolute error at most `1e-5`;
- output aggregate relative L1 error is at most `0.05`;
- output cosine similarity is at least `0.999`;
- output max absolute error is at most `1.0`;
- no NaN or infinity is produced.

### vLLM comparison

Run in the K3-enabled vLLM CUDA 13 image pinned by the benchmark manifest:

1. map identical packed expert shards and BF16 dense weights into vLLM's Kimi
   K3 `FusedMoE`/`LatentMoERunner`;
2. compare layer outputs for all DFlash block-8 and block-16 shapes;
3. benchmark the native vLLM layer and the custom operator under identical
   CUDA Graph, stream, input pool, and timing conditions;
4. attempt end-to-end K3 DFlash serving with
   `modal-labs/Kimi-K3-DFlash` through vLLM's generic DFlash configuration and
   record compatibility explicitly.

The layer comparison remains required even if vLLM cannot load that DFlash
checkpoint end to end.

### SGLang comparison

Run in the K3-enabled SGLang image with:

- TP size 8;
- `flashinfer_mxfp4` native MoE runner;
- `modal-labs/Kimi-K3-DFlash`;
- DFlash block sizes 8 and 16;
- request concurrency 1 through 8.

Map identical weights into the SGLang native K3 MoE layer, compare outputs, and
benchmark native versus custom execution. Also run an end-to-end server smoke
test with the published DFlash configuration.

### Benchmark methodology

- one 8x B300 node;
- locked framework/container revisions recorded in the output;
- warm up every shape before measurement;
- at least 1,000 measured layer iterations per shape;
- CUDA-event device latency with a final synchronization outside the measured
  region;
- rotating input/route pool large enough to avoid reporting an L2-resident
  expert subset;
- report median, p90, p99, and geometric mean;
- report launch count and temporary workspace bytes;
- report raw decode, DFlash block-8, and DFlash block-16 tables separately.

The primary optimization metric is median layer latency at DFlash block 16,
request concurrency 1. The secondary metric is the geometric-mean median over
block-16 request concurrency 1 through 8. At the primary point, the custom
operator must have lower median latency than both native framework baselines.
Across request concurrency 1 through 8, its geometric-mean median must not be
slower than the faster framework baseline. P99 must not regress by more than
10% relative to the faster native framework baseline.

## Planned Files

- `csrc/kimi_k3_decode/types.cuh`
- `csrc/kimi_k3_decode/router.cuh`
- `csrc/kimi_k3_decode/skinny_gemm.cuh`
- `csrc/kimi_k3_decode/expert_mxfp4.cuh`
- `csrc/kimi_k3_decode/collectives.cuh`
- `csrc/kimi_k3_decode/kernel.cuh`
- `csrc/kimi_k3_decode/entrypoints.cuh`
- `csrc/bindings.cu`
- `mok/kimi_k3.py`
- `mok/ops.py`
- `mok/_fake_impls.py`
- `tests/kimi_k3_reference.py`
- `tests/test_kimi_k3_decode.py`
- `benchmarks/bench_kimi_k3_decode.py`
- `benchmarks/frameworks/vllm_kimi_k3.py`
- `benchmarks/frameworks/sglang_kimi_k3.py`
- `modal_app.py`

## External References

- Kimi K3 model and architecture:
  <https://huggingface.co/moonshotai/Kimi-K3>
- Official Kimi K3 implementation:
  <https://huggingface.co/moonshotai/Kimi-K3/resolve/main/modeling_kimi_linear.py>
- Modal Kimi K3 DFlash checkpoint and serving recipe:
  <https://huggingface.co/modal-labs/Kimi-K3-DFlash>
- vLLM Kimi K3 serving design:
  <https://vllm.ai/blog/2026-07-27-k3>
- vLLM Kimi K3 latent-MoE tail implementation:
  <https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/latent_moe_runner.py>
- SGLang Kimi K3 serving cookbook:
  <https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3>

