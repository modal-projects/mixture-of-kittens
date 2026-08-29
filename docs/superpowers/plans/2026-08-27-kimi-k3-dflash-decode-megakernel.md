# Kimi K3 DFlash Decode Megakernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a one-launch TP8 Kimi K3 sparse-MoE forward operator for DFlash verification on one 8x B300 node.

**Architecture:** Add a new `csrc/kimi_k3_decode/` persistent SM103 kernel rather than changing the training megakernel. All TP ranks receive the same 1–128 verifier tokens, hold tensor-parallel shards of all 896 routed experts, overlap router/latent/shared work, and finish with device-side NVLink collectives. A separate Python module owns fixed-shape validation, one-time MXFP4 preparation, symmetric workspace reuse, framework adapters, and benchmark orchestration.

**Tech Stack:** CUDA 13.2, C++20, ThunderKittens SM103/tcgen05, PyTorch 2.13 custom operators and symmetric memory, NCCL, pytest, Modal 8x B300, vLLM K3 image, SGLang K3 image, DFlash.

## Global Constraints

- Source design: `docs/superpowers/specs/2026-08-27-kimi-k3-dflash-decode-megakernel-design.md`.
- Hardware is exactly one TP8 B300 SM103 NVLink/NVSwitch domain.
- The operator computes only the Kimi K3 sparse-MoE block, not attention, DFlash drafting, or outer residuals.
- Kimi K3 dimensions are fixed: hidden 7168, latent 3584, routed intermediate 3072, shared intermediate 6144, 896 experts, top-16.
- Runtime token count is any integer in `[1, 128]`; primary DFlash shapes are `16,32,48,64,80,96,112,128`.
- The production hot path performs one CUDA kernel launch per rank, no allocation, no host synchronization, and no device-to-host schedule read.
- Router, SiTU, and RMSNorm semantics must match the official Kimi K3 implementation exactly.
- Routed experts use group-32 MXFP4 E2M1 weights and dynamically quantized MXFP8 E4M3 activations; latent and shared projections remain BF16.
- Training APIs and files under `csrc/megakernel/` retain their existing behavior.
- Every logical task ends with focused verification and a separate commit.
- Red tests import an existing module and access the new symbol inside the test
  body, so pytest records an expected failing test rather than a collection
  error.

## File Structure

| Path | Responsibility |
|---|---|
| `mok/kimi_k3.py` | Public fixed-dimension config, prepared weights, symmetric workspace, validators, and decode call |
| `csrc/kimi_k3_decode/types.cuh` | K3 constants, packed tensor layouts, scratch offsets, and launch globals |
| `csrc/kimi_k3_decode/mxfp4.cuh` | One-time group-32 MXFP4 pack/dequant utilities |
| `csrc/kimi_k3_decode/router.cuh` | FP32 sigmoid router, correction-bias top-16, normalized weights, assignment histogram |
| `csrc/kimi_k3_decode/skinny_gemm.cuh` | BF16 projection kernels specialized for token capacities 1–128 |
| `csrc/kimi_k3_decode/expert_mxfp4.cuh` | MXFP8 activation quantization, MXFP4 routed GEMMs, FP32 SiTU, weighted latent accumulation |
| `csrc/kimi_k3_decode/shared.cuh` | TP-sharded BF16 shared-expert branch |
| `csrc/kimi_k3_decode/collectives.cuh` | Routed all-reduce, shared reduce-scatter, RMSNorm, latent-up projection, mailbox assembly |
| `csrc/kimi_k3_decode/kernel.cuh` | Role-specialized persistent scheduler and single launch |
| `csrc/kimi_k3_decode/entrypoints.cuh` | PyTorch tensor checks, layout conversion, and host launch |
| `tests/kimi_k3_reference.py` | Independent PyTorch reference for router, MXFP4, SiTU, RMSNorm, and full block |
| `tests/test_kimi_k3_reference.py` | CPU/reference semantic tests |
| `tests/test_kimi_k3_decode.py` | TP8 full-shape correctness, graph replay, workspace reuse, and launch-count tests |
| `benchmarks/kimi_k3_timing.py` | Shared fair timing and percentile calculations |
| `benchmarks/bench_kimi_k3_decode.py` | Custom-operator latency runner and artifact writer |
| `benchmarks/frameworks/vllm_kimi_k3.py` | Native vLLM K3 layer adapter |
| `benchmarks/frameworks/sglang_kimi_k3.py` | Native SGLang K3 layer adapter |
| `benchmarks/compare_kimi_k3_frameworks.py` | Identical-input native/custom comparison |
| `benchmarks/framework_manifest.json` | Reproducible framework images, commands, and K3/DFlash model IDs |
| `benchmarks/smoke_dflash_server.py` | Concurrency 1–8 server health and generation smoke |

---

### Task 1: Lock the official numerical contract

**Files:**
- Create: `mok/kimi_k3.py`
- Create: `tests/kimi_k3_reference.py`
- Create: `tests/test_kimi_k3_reference.py`
- Modify: `mok/__init__.py`

**Interfaces:**
- Produces: `KimiK3DecodeConfig`, dimension constants, `kimi_k3_router_reference`, `kimi_k3_situ_reference`, `kimi_k3_rmsnorm_reference`, and `kimi_k3_moe_reference`.
- Consumes: only PyTorch; no compiled K3 operator.

- [ ] **Step 1: Write failing semantic tests**

Add tests that encode the official equations directly:

```python
def test_router_bias_changes_selection_not_weight() -> None:
    x = torch.tensor([[1.0, 0.0]])
    router = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    bias = torch.tensor([-3.0, 3.0, 0.0])
    ids, weights = kimi_k3_router_reference(x, router, bias, topk=2)
    raw = torch.sigmoid(x.float() @ router.float().T)
    assert ids.tolist() == [[1, 2]]
    expected = raw[:, [1, 2]] / raw[:, [1, 2]].sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected)


def test_situ_uses_fp32_gate_and_up_clamps() -> None:
    gate = torch.tensor([[8.0, -8.0]], dtype=torch.bfloat16)
    up = torch.tensor([[50.0, -50.0]], dtype=torch.bfloat16)
    actual = kimi_k3_situ_reference(gate, up)
    expected = (
        4.0 * torch.tanh(gate.float() / 4.0) * torch.sigmoid(gate.float())
        * 25.0 * torch.tanh(up.float() / 25.0)
    ).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rmsnorm_uses_epsilon_one_e_minus_five() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    gamma = torch.tensor([1.0, 0.5, 2.0], dtype=torch.bfloat16)
    expected = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    ).bfloat16() * gamma
    torch.testing.assert_close(kimi_k3_rmsnorm_reference(x, gamma), expected)
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run:

```bash
python -m pytest -q tests/test_kimi_k3_reference.py
```

Expected: tests fail inside their bodies with `ModuleNotFoundError` for the
new reference modules.

- [ ] **Step 3: Add fixed constants and the reference implementation**

Use these exact constants in `mok/kimi_k3.py`:

```python
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


@dataclass(frozen=True, slots=True)
class KimiK3DecodeConfig:
    max_tokens: int = KIMI_K3_MAX_TOKENS
```

Implement the reference router as FP32 linear, sigmoid, selection from
`scores + correction_bias`, gather from unmodified `scores`, normalization by
`sum + 1e-20`, and unsorted top-k. Implement SiTU and RMSNorm with explicit
FP32 intermediates. Implement `kimi_k3_moe_reference` as a clear expert loop
that accepts reduced test dimensions as keyword arguments while the public
K3 wrapper passes the fixed dimensions above.

- [ ] **Step 4: Run reference tests**

Run:

```bash
python -m pytest -q tests/test_kimi_k3_reference.py
```

Expected: all router, SiTU, RMSNorm, and reduced-shape full-block tests pass.

- [ ] **Step 5: Commit**

```bash
git add mok/kimi_k3.py mok/__init__.py tests/kimi_k3_reference.py tests/test_kimi_k3_reference.py
git commit -m "test: define Kimi K3 MoE reference semantics"
```

### Task 2: Establish the public API and compiled operator boundary

**Files:**
- Create: `csrc/kimi_k3_decode/types.cuh`
- Create: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `csrc/bindings.cu`
- Modify: `Makefile`
- Modify: `mok/kimi_k3.py`
- Modify: `mok/ops.py`
- Modify: `mok/_fake_impls.py`
- Create: `tests/test_kimi_k3_api.py`

**Interfaces:**
- Produces: `KimiK3DecodeWeights`, `validate_kimi_k3_decode_inputs`, low-level `mok::kimi_k3_decode`, and its fake implementation.
- Consumes: constants from Task 1.

- [ ] **Step 1: Write failing API and fake-tensor tests**

Cover exact input shape, BF16 requirement, token bounds, and fake output:

```python
def test_fake_decode_preserves_active_shape() -> None:
    mode = FakeTensorMode()
    with mode:
        hidden = torch.empty(16, 7168, dtype=torch.bfloat16, device="cuda")
        output = ops.kimi_k3_decode_fake_contract(hidden)
    assert output.shape == (16, 7168)
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("tokens", [0, 129])
def test_decode_rejects_token_count_outside_contract(tokens: int) -> None:
    hidden = torch.empty(tokens, 7168, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match=r"between 1 and 128"):
        validate_kimi_k3_decode_hidden_states(hidden)
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
python -m pytest -q tests/test_kimi_k3_api.py
```

Expected: tests fail inside their bodies with `AttributeError` because the
weight dataclass and contract helpers are absent.

- [ ] **Step 3: Define the prepared-weight contract**

Add an immutable dataclass with these fields:

```python
@dataclass(frozen=True, slots=True)
class KimiK3DecodeWeights:
    router_weight: torch.Tensor
    router_correction_bias: torch.Tensor
    routed_expert_down_proj: torch.Tensor
    routed_expert_up_proj: torch.Tensor
    routed_latent_rmsnorm_weight: torch.Tensor
    expert_w1_packed: torch.Tensor
    expert_w1_scale: torch.Tensor
    expert_w3_packed: torch.Tensor
    expert_w3_scale: torch.Tensor
    expert_w2_packed: torch.Tensor
    expert_w2_scale: torch.Tensor
    shared_gate_proj: torch.Tensor
    shared_up_proj: torch.Tensor
    shared_down_proj: torch.Tensor
    tp_rank: int
```

The canonical prepared layouts are:

```text
expert_w1_packed, expert_w3_packed: uint8 [896, 384, 1792]
expert_w1_scale,  expert_w3_scale:  uint8 [896, 384, 112]
expert_w2_packed:                    uint8 [896, 3584, 192]
expert_w2_scale:                     uint8 [896, 3584, 12]
```

The W1/W3 layouts preserve the checkpoint's logical K=3584 exactly. Kimi K3's
W4A8 path uses mixed `mxf8f6f4` block-scaled MMA with K=32, so no K96 padding
is required.

- [ ] **Step 4: Add the low-level custom-op schema and fake**

The Python custom op accepts `hidden_states`, the fourteen weight tensors,
five mutable workspace tensors (`scratch`, `collective_buffer`,
`output_mailbox`, `barrier_buffer`, and `error_flag`), pointer lists, TP rank,
and active token count. Register the fake with the exact same positional
signature as the custom op; its body returns
`hidden_states.new_empty(hidden_states.shape)`.

Add a temporary C++ entrypoint that validates SM103 and returns
`at::empty_like(hidden_states)`. This entrypoint exists only to stabilize the
ABI while subsequent tasks replace its body with the persistent launch.
Expose `kimi_k3_decode_workspace_bytes()` from the same header so Python never
duplicates local scratch offsets.

- [ ] **Step 5: Extend the build dependency set**

Change the Makefile header line to:

```makefile
HEADERS := $(wildcard csrc/*.cuh) $(wildcard csrc/megakernel/*.cuh) $(wildcard csrc/kimi_k3_decode/*.cuh)
```

Include `kimi_k3_decode/entrypoints.cuh` from `csrc/bindings.cu` and register
the exact arguments used by `mok/ops.py`.

- [ ] **Step 6: Run API tests and compile on the B300 image**

Run:

```bash
python -m pytest -q tests/test_kimi_k3_api.py
modal run modal_app.py::gpu_info
```

Expected: API tests pass; Modal reports `sm_103` and `BUILD + KERNEL OK`.

- [ ] **Step 7: Commit**

```bash
git add Makefile csrc/bindings.cu csrc/kimi_k3_decode mok tests/test_kimi_k3_api.py
git commit -m "feat: add Kimi K3 decode operator contract"
```

### Task 3: Add reusable TP8 symmetric workspace

**Files:**
- Modify: `mok/kimi_k3.py`
- Modify: `tests/test_kimi_k3_api.py`
- Modify: `tests/conftest.py`
- Modify: `csrc/kimi_k3_decode/types.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`

**Interfaces:**
- Produces: `KimiK3DecodeWorkspace`, `create_kimi_k3_decode_workspace`, `get_kimi_k3_decode_workspace`, and `clear_kimi_k3_decode_workspace_cache`.
- Consumes: `barrier_all` from `mok.ops`.

- [ ] **Step 1: Write failing workspace validation and cache tests**

Add a TP8 fixture and verify capability, group size, identity caching, and
cache clearing:

```python
@pytest.fixture(scope="session")
def tp8_context(context):
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("Kimi K3 decode requires TP8")
    if torch.cuda.get_device_capability(device) != (10, 3):
        pytest.skip("Kimi K3 decode requires SM103 B300")
    return rank, world_size, device


def test_workspace_cache_reuses_entry(tp8_context) -> None:
    _, _, device = tp8_context
    first = get_kimi_k3_decode_workspace(dist.group.WORLD, device=device)
    second = get_kimi_k3_decode_workspace(dist.group.WORLD, device=device)
    assert first is second
```

- [ ] **Step 2: Run the focused TP8 test and observe the missing symbol**

Run on B300:

```bash
torchrun --standalone --nproc-per-node=8 -m pytest -q tests/test_kimi_k3_api.py -k workspace
```

Expected: failure because workspace lifecycle functions are absent.

- [ ] **Step 3: Implement the workspace**

Use a single local `uint8` scratch tensor plus three symmetric allocations:

```python
@dataclass(slots=True)
class KimiK3DecodeWorkspace:
    group_name: str
    tp_rank: int
    tp_size: int
    device: torch.device
    max_tokens: int
    scratch: torch.Tensor
    collective_buffer: torch.Tensor
    collective_handle: Any
    collective_ptrs: list[int]
    collective_multicast_ptr: int
    output_mailbox: torch.Tensor
    output_mailbox_handle: Any
    output_mailbox_ptrs: list[int]
    barrier_buffer: torch.Tensor
    barrier_handle: Any
    barrier_ptrs: list[int]
    barrier_multicast_ptr: int
    barrier_target: torch.Tensor
    error_flag: torch.Tensor
```

Allocate `collective_buffer` as BF16 `[128, 10752]`, where 10752 is
`3584 + 7168`. Allocate `output_mailbox` as BF16 `[128, 8, 896]`, so its
token-major storage is also a contiguous `[128, 7168]` final-output view. Allocate
`barrier_buffer` as int32 `[1]`; generation-tagged phase counters live in the
local scratch tensor. Size scratch with `_C.kimi_k3_decode_workspace_bytes()`.
Obtain pointer lists and multicast pointers from
`torch.distributed._symmetric_memory.rendezvous`, following
`mok/functional.py`.

Reserve 16 int32 phase counters at the start of scratch and round the allocation
up to 256 bytes:

```cpp
static constexpr int NUM_PHASE_COUNTERS = 16;
static constexpr int SCRATCH_ALIGNMENT = 256;
static constexpr int SCRATCH_BYTES =
    ((NUM_PHASE_COUNTERS * sizeof(int) + SCRATCH_ALIGNMENT - 1)
     / SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT;
```

Return `SCRATCH_BYTES` from `_C.kimi_k3_decode_workspace_bytes()`. Subsequent
kernel tasks extend this layout from the same C++ constant.

Cache by `(group_name, device_index, 128)`. Cache clearing performs
`barrier_all`, synchronizes once during teardown, and drops all references.

- [ ] **Step 4: Run TP8 workspace tests**

Run:

```bash
torchrun --standalone --nproc-per-node=8 -m pytest -q tests/test_kimi_k3_api.py -k workspace
modal run --env rahul-dev modal_app.py::gpu_info
```

Expected: all workspace tests pass on 8x B300 and the modified C++ layout
compiles for SM103.

- [ ] **Step 5: Commit**

```bash
git add csrc/kimi_k3_decode/types.cuh csrc/kimi_k3_decode/entrypoints.cuh mok/kimi_k3.py tests/conftest.py tests/test_kimi_k3_api.py
git commit -m "feat: add TP8 Kimi K3 decode workspace"
```

### Task 4: Implement checkpoint-compatible MXFP4 preparation

**Files:**
- Create: `csrc/kimi_k3_decode/mxfp4.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `csrc/bindings.cu`
- Modify: `mok/ops.py`
- Modify: `mok/_fake_impls.py`
- Modify: `mok/kimi_k3.py`
- Create: `tests/test_kimi_k3_mxfp4.py`

**Interfaces:**
- Produces: `pack_kimi_k3_mxfp4`, `dequant_kimi_k3_mxfp4`, and `prepare_kimi_k3_decode_weights`.
- Consumes: prepared layouts fixed in Task 2.

- [ ] **Step 1: Write failing pack/dequant tests**

Test exact zero handling, group scaling, native K preservation, and round-trip
behavior:

```python
def test_pack_mxfp4_preserves_native_k_3584(device: torch.device) -> None:
    weight = torch.zeros(1, 384, 3584, dtype=torch.bfloat16, device=device)
    packed, scale = pack_kimi_k3_mxfp4(weight, padded_k=3584)
    assert packed.shape == (1, 384, 1792)
    assert scale.shape == (1, 384, 112)
    restored = dequant_kimi_k3_mxfp4(packed, scale, logical_k=3584)
    torch.testing.assert_close(restored, weight)
```

- [ ] **Step 2: Verify failure on one B300**

Run:

```bash
torchrun --standalone --nproc-per-node=1 -m pytest -q tests/test_kimi_k3_mxfp4.py
```

Expected: tests fail inside their bodies because pack/dequant operators are
unregistered.

- [ ] **Step 3: Implement raw group-32 packing**

For each output row and 32-value K group:

1. Compute FP32 absolute maximum.
2. Select the E8M0 power-of-two scale that keeps the E2M1 magnitude at most 6.
3. Encode scale 1 as byte `0x7f`; use scale 1 for an all-zero group.
4. Divide by scale, convert each float pair with the
   `__nv_cvt_float2_to_fp4x2` intrinsic in `__NV_E2M1` round-to-nearest mode,
   and store one byte per pair.
5. Write zero data and scale `0x7f` for padded K positions.

The dequant test operator multiplies each decoded E2M1 value by its group's
E8M0 scale and truncates to `logical_k`.

- [ ] **Step 4: Implement one-time prepared weights**

`prepare_kimi_k3_decode_weights` validates replicated BF16 tensors, slices the
rank's routed intermediate range `[tp_rank*384:(tp_rank+1)*384]`, slices the
shared intermediate range `[tp_rank*768:(tp_rank+1)*768]`, packs routed
`w1/w3` with native K=3584, packs routed `w2` with K=384, and returns
`KimiK3DecodeWeights`.

- [ ] **Step 5: Cross-check values against FlashInfer's MXFP4 path**

In the vLLM K3 image, construct the same deterministic BF16 group, load its
packed values through `trtllm_fp4_block_scale_routed_moe`, and compare the
single-expert output with the local dequantized matrix multiplication.

Run:

```bash
python -m pytest -q tests/test_kimi_k3_mxfp4.py
```

Expected: exact packed shape and scale bytes; dequantized max error is bounded
by one E2M1 quantization step per group.

- [ ] **Step 6: Commit**

```bash
git add csrc/kimi_k3_decode/mxfp4.cuh csrc/kimi_k3_decode/entrypoints.cuh csrc/bindings.cu mok tests/test_kimi_k3_mxfp4.py
git commit -m "feat: prepare Kimi K3 MXFP4 expert weights"
```

### Task 5: Fuse router and routed latent-down projection

**Files:**
- Create: `csrc/kimi_k3_decode/router.cuh`
- Create: `csrc/kimi_k3_decode/skinny_gemm.cuh`
- Create: `csrc/kimi_k3_decode/kernel.cuh`
- Modify: `csrc/kimi_k3_decode/types.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `csrc/bindings.cu`
- Modify: `mok/ops.py`
- Modify: `mok/_fake_impls.py`
- Create: `tests/test_kimi_k3_router.py`

**Interfaces:**
- Produces: device arrays `expert_ids[M,16]`, `expert_weights[M,16]`,
  expert-major assignment offsets, and `latent_x[M,3584]`.
- Consumes: hidden states, router weight/bias, and latent-down weight.

- [ ] **Step 1: Write full-dimension router tests**

Use seeded BF16 tensors at `M=1,8,16,128`; assert exact IDs and `1e-5` maximum
weight error against `kimi_k3_router_reference`. Add concentrated and disjoint
router fixtures by constructing correction biases with separated FP32 values.

- [ ] **Step 2: Run and confirm the temporary entrypoint is wrong**

Run on B300:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::gpu_info --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=1 -m pytest -q tests/test_kimi_k3_router.py'" /dev/null
```

Expected: output/debug router buffers do not match the reference.

- [ ] **Step 3: Implement the router CTA**

Assign one 256-thread CTA per active token. Each warp evaluates one expert row
at a time with coalesced vector loads and strides across the 896 experts.
Reduce to a deterministic top-16 in shared memory, using expert ID as the
tie-breaker. Gather weights from raw sigmoid scores and divide by their FP32
sum plus `1e-20`.

Use a 896-bin global histogram and prefix scan to produce expert-major
assignment offsets for at most 2048 assignments. Store token index, top-k slot,
and normalized weight; keep the count entirely on device.

Extend the C++ scratch layout with aligned arrays for 2048 expert IDs, 2048
FP32 weights, 896 histogram counts, 897 offsets, 2048 assignment token IDs,
and 2048 assignment slots. Continue to expose its total byte size only through
`kimi_k3_decode_workspace_bytes()`.

- [ ] **Step 4: Implement token-capacity BF16 projection paths**

Add direct-register CUDA-core GEMM for capacities 1, 2, 4, and 8. Add tcgen05
BF16 paths for capacities 16, 32, 64, and 128. Both compute
`hidden_states @ routed_expert_down_proj.T` and mask rows `>= active_m`.

Router and latent projection receive separate persistent CTA roles and publish
completion through generation-tagged counters in `scratch`.

Add a private test/fallback operator named `mok::_kimi_k3_route_and_project`
that invokes the same one-launch device implementation and returns
`(expert_ids, expert_weights, latent_x)`. Its C++ and fake signatures must
match. The production `kimi_k3_decode` API remains unchanged and Task 9 calls
the same device functions directly inside the final persistent kernel; no
intermediate tensor is returned by the production path.

- [ ] **Step 5: Run router and latent projection tests**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::gpu_info --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=1 -m pytest -q tests/test_kimi_k3_router.py'" /dev/null
```

Expected: IDs exact; normalized weights within `1e-5`; BF16 latent projection
matches PyTorch within `atol=0.5, rtol=0.01`.

- [ ] **Step 6: Commit**

```bash
git add csrc/kimi_k3_decode csrc/bindings.cu mok/ops.py mok/_fake_impls.py tests/test_kimi_k3_router.py
git commit -m "feat: fuse Kimi K3 routing and latent projection"
```

### Task 6: Implement assignment-driven mixed MXFP8-by-MXFP4 routed experts

**Files:**
- Create: `csrc/kimi_k3_decode/expert_mxfp4.cuh`
- Modify: `csrc/kimi_k3_decode/kernel.cuh`
- Modify: `csrc/kimi_k3_decode/types.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `csrc/bindings.cu`
- Modify: `mok/ops.py`
- Modify: `mok/_fake_impls.py`
- Modify: `tests/test_kimi_k3_api.py`
- Create: `tests/test_kimi_k3_expert.py`

**Interfaces:**
- Produces: rank-local partial routed latent `[M,3584]`.
- Consumes: latent rows, expert assignments, prepared `w1/w3/w2`, and router weights.

- [ ] **Step 1: Write single-expert and grouped-expert tests**

Test one selected expert with 1, 2, 8, and 16 rows; then test 2048 assignments
distributed over all 896 experts. Reference computation dequantizes the exact
prepared weights, applies FP32 SiTU, and multiplies by the matching normalized
router weights.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::gpu_info --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=1 -m pytest -q tests/test_kimi_k3_expert.py'" /dev/null
```

Expected: routed partials remain empty or zero.

- [ ] **Step 3: Implement activation quantization and mixed K32 gate/up**

Quantize each 32-value block of the 3584-wide latent row to MXFP8 E4M3 with an
E8M0 scale. Add a first-party SM103 `mxf8f6f4` wrapper whose A type is E4M3,
B type is E2M1, scale type is E8M0, scale vector is `1X`, and K is 32. Use it
for the 384-wide local gate/up output. Do not use ThunderKittens' K96
`mxf4nvf4` helper, which is an FP4-by-FP4 instruction.

Use expert-major assignment ranges so one CTA cluster reuses an expert's
weights across all selected rows. Do not launch work for zero-token experts.

Extend the scratch struct with aligned storage for MXFP8 latent data/scales,
MXFP8 SiTU data/scales, an FP32 routed accumulator, and generation-tagged
quantization/expert completion counters. Update the existing workspace-byte
test from the same C++ source of truth.

Add a private `mok::_kimi_k3_routed_experts` operator across Python, fake,
C++, and pybind. It consumes `latent_x`, the Task 5 scratch assignment state,
the six prepared expert tensors, a mutable BF16 routed-output buffer, scratch,
and active token count. It returns the active `[M,3584]` view for tests. The
production decode ABI remains unchanged; Task 9 invokes the same device
functions from the final persistent grid.

- [ ] **Step 4: Implement FP32 SiTU and MXFP4 down projection**

Compute:

```text
4*tanh(gate/4)*sigmoid(gate) * 25*tanh(up/25)
```

in FP32, quantize the 384 values to MXFP8 block-32, run the same mixed K32
MXFP8-by-MXFP4 down projection to 3584, multiply by the normalized router
weight, and atomically accumulate the rank-local FP32 partial. Cast the
completed per-token partial to BF16 before the TP8 collective.

- [ ] **Step 5: Run expert tests**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::gpu_info --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=1 -m pytest -q tests/test_kimi_k3_expert.py'" /dev/null
```

Expected: relative L1 at most `0.05`, cosine at least `0.999`, maximum absolute
error at most `1.0`, and no non-finite values.

- [ ] **Step 6: Commit**

```bash
git add csrc/kimi_k3_decode csrc/bindings.cu mok/ops.py mok/_fake_impls.py tests/test_kimi_k3_api.py tests/test_kimi_k3_expert.py
git commit -m "feat: add Kimi K3 MXFP4 routed experts"
```

### Task 7: Add the TP-sharded shared-expert branch

**Files:**
- Create: `csrc/kimi_k3_decode/shared.cuh`
- Modify: `csrc/kimi_k3_decode/kernel.cuh`
- Modify: `csrc/kimi_k3_decode/types.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `csrc/bindings.cu`
- Modify: `mok/ops.py`
- Modify: `mok/_fake_impls.py`
- Modify: `tests/test_kimi_k3_api.py`
- Create: `tests/test_kimi_k3_shared.py`

**Interfaces:**
- Produces: rank-local shared partial `[M,7168]`.
- Consumes: BF16 shared gate/up shards `[768,7168]` and down shard `[7168,768]`.

- [ ] **Step 1: Write shared-branch tests**

For every capacity bucket, compare the rank-local partial to:

```python
gate = (hidden.float() @ shared_gate.float().T).bfloat16()
up = (hidden.float() @ shared_up.float().T).bfloat16()
activated = (
    4.0 * torch.tanh(gate.float() / 4.0) * torch.sigmoid(gate.float())
    * 25.0 * torch.tanh(up.float() / 25.0)
).bfloat16()
expected = activated.float() @ shared_down.float().T
```

- [ ] **Step 2: Verify failure**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::gpu_info --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=1 -m pytest -q tests/test_kimi_k3_shared.py'" /dev/null
```

Expected: shared partial does not match.

- [ ] **Step 3: Implement shared gate/up, SiTU, and down**

Reuse capacity-specialized BF16 routines from `skinny_gemm.cuh`. Gate and up
GEMMs accumulate in FP32 and round to BF16 before SiTU converts those values
back to FP32, matching the official BF16 linear-module boundary. Give the
shared branch independent persistent CTA roles so it overlaps expert work.
Write its full-width partial into the shared region of
`collective_buffer[M,3584:10752]`.

Extend aligned scratch with shared gate, up, and activated intermediates plus
generation-tagged shared-phase counters. Update the workspace-byte assertions
from the C++ source of truth.

Add a private `mok::_kimi_k3_shared_experts` operator across Python, fake,
C++, and pybind. It consumes hidden states, the three rank-local shared
weights, scratch, the mutable collective buffer, and active token count. It
returns the active shared partial for tests. The production decode ABI remains
unchanged; Task 9 invokes the same device functions from the final grid.

- [ ] **Step 4: Run shared tests**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::gpu_info --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=1 -m pytest -q tests/test_kimi_k3_shared.py'" /dev/null
```

Expected: BF16 tolerances pass for all capacities and active-row masks.

- [ ] **Step 5: Commit**

```bash
git add csrc/kimi_k3_decode csrc/bindings.cu mok/ops.py mok/_fake_impls.py tests/test_kimi_k3_api.py tests/test_kimi_k3_shared.py
git commit -m "feat: add Kimi K3 shared expert branch"
```

### Task 8: Implement the fused TP8 tail

**Files:**
- Create: `csrc/kimi_k3_decode/collectives.cuh`
- Modify: `csrc/kimi_k3_decode/kernel.cuh`
- Modify: `csrc/kimi_k3_decode/types.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `csrc/bindings.cu`
- Modify: `mok/kimi_k3.py`
- Modify: `mok/ops.py`
- Modify: `mok/_fake_impls.py`
- Modify: `tests/test_kimi_k3_api.py`
- Create: `tests/test_kimi_k3_collectives.py`

**Interfaces:**
- Produces: identical BF16 `[M,7168]` output on all ranks.
- Consumes: routed latent partial, shared output partial, RMSNorm weight,
  latent-up weight, symmetric collective buffer, output mailbox, and barriers.

- [ ] **Step 1: Write TP8 collective tests**

Generate a distinct routed/shared partial on each rank. Compare the custom tail
against NCCL all-reduce plus the PyTorch RMSNorm, sharded up-projection, add,
and all-gather reference for `M=1,5,16,64,128`.

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::bench --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=8 -m pytest -q tests/test_kimi_k3_collectives.py'" /dev/null
```

Expected: the tail output is unavailable.

- [ ] **Step 3: Implement generation-safe device collectives**

Use `multimem.red` over the symmetric collective buffer to all-reduce the
3584 routed columns and reduce-scatter the 7168 shared columns. Use generation
values stored in symmetric int32 barriers so CUDA Graph replay never requires
host zeroing.

Apply FP32 RMSNorm to the replicated routed latent. Each rank multiplies by its
contiguous `[896,3584]` row slice of the replicated latent-up weight, beta-adds
its `[M,896]` shared shard, writes its output shard to every peer's rank slot,
then assembles all eight slots into `[M,7168]`.

Migrate `KimiK3DecodeWorkspace.output_mailbox` to token-major BF16
`[128,8,896]`. Each rank multicasts its `[M,896]` output shard into mailbox
slot `[:,tp_rank,:]`; after the device barrier, the same allocation is viewed
without a copy as contiguous `[128,7168]`. The production path returns an
active-row alias of this workspace storage and does not allocate an output.

Add a private `mok::_kimi_k3_tail` operator across Python, fake, C++, and
pybind. It consumes the routed/shared partials in the symmetric collective
buffer, replicated RMSNorm and latent-up weights, output mailbox pointers,
barrier state, scratch, TP rank, and active token count. It performs the full
tail in exactly one launch, mutates the mailbox, and returns `None`; PyTorch
custom operators must not return a view that aliases a mutated input. The
Python private helper and final public API return
`output_mailbox.view(128,7168)[:M]` after the op. Task 9 invokes the same
device implementation from the final grid.

- [ ] **Step 4: Run collective tests and graph replay**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::bench --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=8 -m pytest -q tests/test_kimi_k3_collectives.py'" /dev/null
```

Expected: every rank matches the NCCL reference; 1,000 graph replays complete
without deadlock or stale-generation output.

- [ ] **Step 5: Commit**

```bash
git add csrc/kimi_k3_decode csrc/bindings.cu mok/kimi_k3.py mok/ops.py mok/_fake_impls.py tests/test_kimi_k3_api.py tests/test_kimi_k3_collectives.py
git commit -m "feat: fuse Kimi K3 TP8 latent MoE tail"
```

### Task 9: Merge all stages into one persistent launch

**Files:**
- Modify: `csrc/kimi_k3_decode/kernel.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `csrc/kimi_k3_decode/router.cuh`
- Modify: `csrc/kimi_k3_decode/skinny_gemm.cuh`
- Modify: `csrc/kimi_k3_decode/expert_mxfp4.cuh`
- Modify: `csrc/kimi_k3_decode/shared.cuh`
- Modify: `csrc/kimi_k3_decode/collectives.cuh`
- Modify: `csrc/bindings.cu`
- Modify: `mok/kimi_k3.py`
- Modify: `mok/ops.py`
- Modify: `mok/_fake_impls.py`
- Modify: `tests/test_kimi_k3_api.py`
- Modify: `tests/test_kimi_k3_tail_contract.py`
- Create: `tests/test_kimi_k3_tail_signature.py`
- Create: `tests/test_kimi_k3_decode.py`

**Interfaces:**
- Produces: `kimi_k3_decode(config, workspace, weights, hidden_states)`.
- Consumes: all device stages and workspace contracts from Tasks 2–8.

- [ ] **Step 1: Write full-block TP8 tests**

Parameterize raw decode `M=1..8`, DFlash block-8 shapes
`8,16,24,32,40,48,56,64`, and block-16 shapes
`16,32,48,64,80,96,112,128`. Cover balanced, concentrated, and disjoint
routing, changing M on one workspace, and repeated CUDA Graph replay.

- [ ] **Step 2: Confirm the multi-stage implementation fails the launch-count gate**

Record one call with `torch.profiler`; require exactly one kernel whose name
contains `kimi_k3_decode_persistent_kernel` and no separately launched K3
router, projection, expert, shared, or tail kernel.

- [ ] **Step 3: Implement role scheduling inside one grid**

Use a fixed 148-CTA SM103 grid that is residency-checked before launch. Every
CTA is resident before any grid-wide phase wait. Device task counters let CTAs
claim work instead of assigning one permanent block to every logical task.
Phases are:

1. router and latent-down work, including assignment histogram/scan;
2. routed-expert work stealing overlapped with shared gate/up work;
3. remaining routed work overlapped with shared SiTU/down work;
4. tail routed/shared reduction, RMSNorm, latent-up, and mailbox assembly.

Every phase uses wrap-safe generation-tagged release/acquire barriers. Expert
workers claim expert IDs atomically and skip empty ranges; no CTA serially
sweeps all 896 experts in the production path. The tail begins only after
routed and shared completion counts reach their bucket-specific totals. The
host entrypoint launches only `kimi_k3_decode_persistent_kernel`.

Change the low-level production custom op to return `None` and add the
mailbox-multicast pointer, barrier target, and workspace signature required by
Task 8. The high-level
`mok.kimi_k3.kimi_k3_decode(config, workspace, weights, hidden_states)` validates
workspace/weight rank and device identity, invokes the mutating op, and returns
`workspace.output_mailbox.view(128,7168)[:M]` without allocating.

Keep all private stage operators as independently testable fallbacks, but the
public call must not invoke them. Split the signature fixtures/tests out of
`tests/test_kimi_k3_tail_contract.py` so each test file remains below 1,000
lines, and make the fake's zero workspace signature semantics explicit.

- [ ] **Step 4: Run full correctness and launch-count tests**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::bench --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=8 -m pytest -q tests/test_kimi_k3_decode.py'" /dev/null
```

Expected: all numerical, finite-value, workspace-reuse, graph-replay, and
single-launch assertions pass.

- [ ] **Step 5: Run existing regression tests**

Run:

```bash
source .venv/bin/activate
script -qec "modal shell --env rahul-dev modal_app.py::bench --cmd 'cd /root/mok && torchrun --standalone --nproc-per-node=8 -m pytest -q tests/test_ops.py tests/test_functional.py'" /dev/null
```

Expected: existing MoK operations and functional validation remain green.

- [ ] **Step 6: Commit**

```bash
git add csrc/kimi_k3_decode csrc/bindings.cu mok tests/test_kimi_k3_api.py tests/test_kimi_k3_tail_contract.py tests/test_kimi_k3_tail_signature.py tests/test_kimi_k3_decode.py
git commit -m "feat: complete persistent Kimi K3 decode megakernel"
```

### Task 10: Add reproducible latency measurement and Modal execution

**Files:**
- Create: `benchmarks/kimi_k3_timing.py`
- Create: `benchmarks/bench_kimi_k3_decode.py`
- Create: `tests/test_kimi_k3_timing.py`
- Modify: `csrc/kimi_k3_decode/persistent_kernel.cuh`
- Modify: `csrc/kimi_k3_decode/entrypoints.cuh`
- Modify: `modal_app.py`

**Interfaces:**
- Produces: JSON/CSV latency artifacts for raw, block-8, and block-16 shapes.
- Consumes: public decode API and prepared deterministic weight/input pools.

- [ ] **Step 1: Write timing-helper tests**

Test percentile calculation with deterministic samples and `--dry-run` shape
enumeration without a GPU.

- [ ] **Step 2: Implement fair measurement**

Warm up each shape 500 times, time at least 1,000 CUDA Graph replays, gather
per-iteration rank maxima, and report median/p90/p99/geometric mean. Rotate
through an input pool whose selected expert weights exceed B300 L2 capacity.
Write:

```text
manifest.json
latency_raw_decode.json
latency_block8.json
latency_block16.json
correctness.json
workspace_stats.json
```

- [ ] **Step 3: Add Modal TP8 functions**

Add `test_kimi_k3_decode()` and `bench_kimi_k3_decode()` with
`gpu="B300:8"`. Both invoke `torch.distributed.run --nproc-per-node=8`; the
benchmark function returns a tar archive of rank-0 JSON/CSV artifacts. The
local invocation writes that returned archive into `/opt/cursor/artifacts`
with Modal's `--write-result`; remote containers do not share the local
artifact filesystem.

- [ ] **Step 4: Run custom benchmark**

Run:

```bash
modal run --env rahul-dev modal_app.py::test_kimi_k3_decode
modal run --env rahul-dev --write-result /opt/cursor/artifacts/kimi_k3_decode_benchmark.tar modal_app.py::bench_kimi_k3_decode
```

Expected: correctness passes and all three latency tables contain every
required shape, 1,000 samples, launch count 1, and workspace bytes.

- [ ] **Step 5: Profile and tune fixed candidates**

Compare persistent-grid sizes 64, 96, 128, and the correctness baseline 148
CTAs. Compare expert cluster sizes 1 and 2 only when the latter preserves the
mixed-MMA/tensor-memory contract. Keep the configuration with the lowest
block-16 M=16 median that passes the complete correctness, graph, occupancy,
and p99 checks. Record every rejected candidate and its reason in the
benchmark artifact, not source comments.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/kimi_k3_timing.py benchmarks/bench_kimi_k3_decode.py tests/test_kimi_k3_timing.py csrc/kimi_k3_decode/persistent_kernel.cuh csrc/kimi_k3_decode/entrypoints.cuh modal_app.py
git commit -m "bench: measure Kimi K3 decode latency on B300"
```

### Task 11: Compare identical layers with vLLM and SGLang

**Files:**
- Create: `benchmarks/framework_manifest.json`
- Modify: `benchmarks/frameworks/__init__.py`
- Create: `benchmarks/frameworks/vllm_kimi_k3.py`
- Create: `benchmarks/frameworks/sglang_kimi_k3.py`
- Create: `benchmarks/compare_kimi_k3_frameworks.py`
- Create: `tests/test_kimi_k3_frameworks.py`
- Modify: `csrc/kimi_k3_decode/persistent_kernel.cuh`
- Modify: `csrc/kimi_k3_decode/expert_mxfp4.cuh`
- Modify: `modal_app.py`

**Interfaces:**
- Produces: layer-output parity and latency tables for `mok`, `vllm`, and `sglang`.
- Consumes: the same deterministic prepared-weight bundle and hidden-state pool.

- [ ] **Step 1: Pin framework manifests**

Record these initial images and require every run to add resolved package
versions and image IDs to its output manifest:

```json
{
  "vllm": {
    "image": "vllm/vllm-openai:kimi-k3",
    "model": "moonshotai/Kimi-K3",
    "tensor_parallel_size": 8
  },
  "sglang": {
    "image": "lmsysorg/sglang:kimi-k3",
    "model": "moonshotai/Kimi-K3",
    "tensor_parallel_size": 8,
    "moe_runner_backend": "flashinfer_mxfp4"
  },
  "dflash": {
    "model": "modal-labs/Kimi-K3-DFlash",
    "block_sizes": [8, 16]
  }
}
```

- [ ] **Step 2: Write adapter parity tests**

Instantiate vLLM's K3 `KimiMoE`/`LatentMoERunner` and SGLang's K3
`flashinfer_mxfp4` layer with the same logical BF16 weights, then map each
framework's packed tensors into the standalone preparation API. Compare
router IDs, selected weights, intermediate routed latent, and final output.
Use Task 10's realistic route construction: each replay occupies
`min(16*M,896)` experts, including 256 at M=16 and all 896 at M>=56.

- [ ] **Step 3: Build derived comparison images**

Derive one Modal image from each framework image and compile this repository's
extension against that image's installed PyTorch and CUDA ABI. Do not copy a
wheel built against a different framework image.

- [ ] **Step 4: Run numerical and latency comparisons**

Run each backend with identical inputs, routing, graph capture, warmup, and
1,000 measured iterations. Execute:

```bash
modal run --env rahul-dev modal_app.py::compare_vllm
modal run --env rahul-dev modal_app.py::compare_sglang
```

Expected: the custom kernel meets the official-reference gates of relative L1
`0.05`, cosine `0.999`, and max absolute `1.0`; router IDs match exactly and
selected weights stay within `1e-5`. Native vLLM/SGLang outputs are compared
to both the custom result and the official reference, but a native backend's
own miss does not require the custom kernel to reproduce that error. Output
includes 500 warmups and 1,000 rank-max samples with median/p90/p99 for every
DFlash shape. Each Modal function returns a deterministic artifact archive
containing resolved image/package revisions, raw samples, numerical
comparisons, and launch traces.

- [ ] **Step 5: Enforce performance gates**

At block-16 request concurrency 1 (`M=16`), require the custom median below
both native baselines. Across block-16 concurrency 1–8, require its
geometric-mean median no slower than the faster baseline and p99 no more than
10% slower. If a gate fails, return to Task 10's profile candidates and retain
the new winner only after rerunning full correctness. If grid tuning alone
cannot close the gap, profile the production kernel by phase, optimize the
measured routed-expert or synchronization bottleneck, and repeat the full
correctness/resource/graph checks before remeasuring all three backends.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/framework_manifest.json benchmarks/frameworks benchmarks/compare_kimi_k3_frameworks.py tests/test_kimi_k3_frameworks.py csrc/kimi_k3_decode/persistent_kernel.cuh csrc/kimi_k3_decode/expert_mxfp4.cuh modal_app.py
git commit -m "bench: compare Kimi K3 kernel with serving backends"
```

### Task 12: Validate DFlash serving assumptions

**Files:**
- Create: `benchmarks/smoke_dflash_server.py`
- Modify: `modal_app.py`

**Interfaces:**
- Produces: `dflash_smoke_vllm.json` and `dflash_smoke_sglang.json`.
- Consumes: unmodified framework servers and `modal-labs/Kimi-K3-DFlash`.

- [ ] **Step 1: Add a persistent checkpoint volume**

Mount a Modal Volume named `kimi-k3-dflash-checkpoint`. Download
`moonshotai/Kimi-K3` and `modal-labs/Kimi-K3-DFlash` at runtime with resume,
never during image build.

- [ ] **Step 2: Add the SGLang smoke command**

Launch the published TP8 B300 configuration with
`--speculative-algorithm DFLASH`,
`--speculative-draft-model-path modal-labs/Kimi-K3-DFlash`,
`--speculative-dflash-block-size 16`, and
`--moe-runner-backend flashinfer_mxfp4`. Send fixed prompts at request
concurrency 1–8 and record health, acceptance length, output tok/s, and errors.

- [ ] **Step 3: Attempt the vLLM generic DFlash path**

Launch K3 TP8 with:

```text
--speculative-config {"model":"modal-labs/Kimi-K3-DFlash","method":"dflash","num_speculative_tokens":16}
```

Record whether the current K3 image accepts the draft architecture. A clear
unsupported-architecture result is acceptable for this server smoke because
the required vLLM layer-level numerical and latency comparison is enforced in
Task 11.

- [ ] **Step 4: Run both server smokes**

Run:

```bash
modal run modal_app.py::dflash_smoke_sglang
modal run modal_app.py::dflash_smoke_vllm
```

Expected: SGLang serves a successful generation at every concurrency. The
vLLM artifact records either successful generations or the exact compatibility
error without masking it.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/smoke_dflash_server.py modal_app.py
git commit -m "test: validate Kimi K3 DFlash serving"
```

### Task 13: Final audit, documentation, and evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-27-kimi-k3-dflash-decode-megakernel-design.md`

**Interfaces:**
- Produces: documented API, reproducible commands, and user-facing benchmark evidence.
- Consumes: all test and benchmark artifacts.

- [ ] **Step 1: Document the inference API**

Add requirements, prepared-weight layout, workspace lifecycle, DFlash shape
mapping, one-launch guarantee, and an executable TP8 example to `README.md`.

- [ ] **Step 2: Run the complete verification matrix**

Run:

```bash
python -m pytest -q tests/test_kimi_k3_reference.py tests/test_kimi_k3_api.py
modal run modal_app.py::test_kimi_k3_decode
modal run modal_app.py::bench_kimi_k3_decode
modal run modal_app.py::compare_vllm
modal run modal_app.py::compare_sglang
modal run modal_app.py::dflash_smoke_sglang
modal run modal_app.py::dflash_smoke_vllm
```

Expected: all mandatory correctness and performance gates pass; vLLM DFlash
server compatibility is reported truthfully even when unsupported.

- [ ] **Step 3: Review artifacts**

Check each JSON/CSV file for all shapes, framework versions, sample counts,
percentiles, correctness thresholds, launch count, and workspace bytes. Copy
the minimal correctness log and latency table to `/opt/cursor/artifacts`.

- [ ] **Step 4: Commit final documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-27-kimi-k3-dflash-decode-megakernel-design.md
git commit -m "docs: document Kimi K3 decode megakernel"
```

- [ ] **Step 5: Push and update the pull request**

```bash
git push -u origin cursor/kimi-k3-decode-kernel-16bc
```

Update the pull request with the final implementation summary, exact test
commands, framework comparison table, and artifact links. Mark it ready for
review only after every mandatory gate above is evidenced.

