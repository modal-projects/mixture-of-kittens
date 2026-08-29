# Native Gate/Up Microprototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan task-by-task with test-first RED/GREEN evidence.

**Goal:** Build and measure an isolated B300 gate/up kernel with output channels on M=128, assignment tokens on N=8, and direct three-stage W1/W3 TMA staging.

**Architecture:** Add a benchmark-only header and bindings beside the existing batched expert probe. The candidate uses persistent 5-D `CUtensorMap` descriptors, three 32-KiB direct-to-128B-swizzled W1/W3 stages, producer/consumer mbarriers, eight mixed MMAs per four-K-group panel, and a register-resident TMEM epilogue. The production routed-expert kernel remains unchanged.

**Tech Stack:** CUDA C++20, SM103 tcgen05/TMA PTX, ThunderKittens tensor/shared/register tiles, PyTorch C++ extension, pytest, Modal B300.

## Global Constraints

- Start at production commit `0290904c49a73b26505cef3badb4b14622d3be90`.
- Keep the candidate benchmark-only until every performance and correctness gate passes.
- Candidate dynamic shared-memory reservation is at most 120 KiB and forces one CTA per SM.
- The K loop contains no CTA-wide barrier.
- Temporary `/opt/cursor/logs/debug.log` instrumentation stays outside production and is removed after post-fix verification.

---

### Task 1: Source contracts and benchmark policy

**Files:**
- Modify: `tests/test_kimi_k3_batched_expert_probe.py`
- Modify: `tests/test_kimi_k3_scheduler_source.py`

**Interfaces:**
- Consumes: the existing source-test helpers and benchmark import path.
- Produces: failing contracts for the direct descriptor, three-stage ring, TMEM-direct epilogue, 148-CTA pool, exact output checks, and performance thresholds.

- [ ] Add tests requiring `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`, a 128-U4 by 128-row box, 128-byte swizzle, three W1/W3 stages, exactly eight MMAs per panel, stage-release commits, and no `__syncthreads()` in the candidate K loop.
- [ ] Add tests requiring candidate shared memory `<= 120 * 1024`, a reservation above half the SM budget, and an occupancy helper reporting one resident block per SM.
- [ ] Add benchmark-policy tests for r=1 median >=10%, r=1 p99 >=5%, r=2/4/8 median and p99 >=5%, bitwise gate/up/SiTU equality, and improvement outside repeat dispersion.
- [ ] Run the focused CPU source tests and confirm failure because the native candidate symbols do not exist.
- [ ] Commit the RED contracts.

### Task 2: Isolated native gate/up CUDA engine

**Files:**
- Create: `csrc/kimi_k3_decode/expert_mxfp4_native_gate_up_probe.cuh`
- Modify: `csrc/bindings.cu`

**Interfaces:**
- Consumes: packed W1/W3 tensors, row-major W1/W3 scales, prequantized E4M3 activations/scales, CTA expert IDs, CTA output-tile IDs, and row count.
- Produces: baseline or candidate gate/up FP32 captures, SiTU E4M3/E8M0 output, four candidate phase counters, and resource metadata.

- [ ] Encode persistent W1/W3 tensor maps with dimensions `[128 U4, 384 rows, 28 panels, 1, experts]`, strides `[1792, 64, 384*1792, 384*1792]` bytes, box `[128, 128, 1, 1, 1]`, and `CU_TENSOR_MAP_SWIZZLE_128B`.
- [ ] Allocate three expanded stages per operand (`3 * 2 * 16 KiB = 96 KiB`), twelve compact 16x32 activation tiles, and nine row-major scale tiles; assert actual and reserved bytes are <=120 KiB.
- [ ] Implement producer warp priming, TMA-arrival waits, stage-release waits before reuse, activation/scale staging, and panel-ready publication with mbarriers.
- [ ] Implement consumer-warp scale copies, four W1 plus four W3 tcgen05 MMAs per panel, asynchronous stage-release commits, and one terminal compute phase.
- [ ] Load four 32-channel accumulator slices directly from TMEM into warp register tiles, compute SiTU, reduce per-token absolute maxima, quantize E4M3/E8M0, and scatter without a full shared accumulator tile.
- [ ] Add a validation-only current-layout reference capture and an inactive-column poison mode.
- [ ] Bind preparation, launch, and resource-query entrypoints without changing production dispatch.

### Task 3: Saturated B300 benchmark and diagnostics

**Files:**
- Modify: `benchmarks/kimi_k3_batched_expert_probe.py`
- Modify: `modal_app.py`

**Interfaces:**
- Consumes: the new extension entrypoints and one B300.
- Produces: `manifest.json`, `results.json`, `raw_samples.json`, correctness evidence, resource evidence, and TMA/ring-full/MMA/epilogue phase profiles.

- [ ] Build deterministic validation cases for r=1/2/4/8, experts 0 and 895 plus randomized experts, and all three output tiles.
- [ ] Compare captured gate, up, SiTU data, and SiTU scales bitwise; rerun the candidate with poisoned inactive N columns and require identical active output.
- [ ] Build graph pools containing 148 CTAs per replay and enough distinct expert/tile weight panels to exceed L2.
- [ ] Run 500 warmups and five alternating baseline/candidate 1000-sample repeats.
- [ ] Apply the exact median, p99, dispersion, correctness, shared-memory, occupancy, sanitizer, and spill gates; never mark the production path integrated.
- [ ] Add separate memcheck and racecheck focused Modal diagnostics.
- [ ] Add exactly 3-8 compact NDJSON debug logs around setup, correctness, profile, and decision boundaries.

### Task 4: Build, runtime evidence, and cleanup

**Files:**
- Modify only files from Tasks 1-3 if compilation or runtime evidence identifies a root cause.

**Interfaces:**
- Consumes: local CUDA compiler output and B300 Modal runs.
- Produces: a reproducible command, before/after debug logs, sanitizer output, and a final benchmark-only commit.

- [ ] Commit and push the candidate before compilation/testing, as required by the cloud task.
- [ ] Compile with verbose ptxas output and require zero spill stores and loads for the native kernel.
- [ ] Run CPU contracts, focused B300 correctness, memcheck, racecheck, resource query, and the full interleaved benchmark.
- [ ] Evaluate each ordering/layout/resource hypothesis against `/opt/cursor/logs/debug.log`.
- [ ] If evidence requires a fix, retain instrumentation, make one evidence-backed correction, commit/push before retesting, and compare post-fix logs.
- [ ] After successful verification and caller confirmation, remove temporary agent logs, rerun contracts/build, commit, and push.
