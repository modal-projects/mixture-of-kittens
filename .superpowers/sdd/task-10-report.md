# Task 10 report: reproducible Kimi K3 decode latency on 8x B300

## Status

Complete at commit `3410d9745fb84220c2613c74c306d9b6ae1cda45`
(`bench: measure Kimi K3 decode latency on B300`). The measured winner is the
existing production grid of 148 CTAs, so the production constant did not need
to change.

## RED / GREEN

### RED 1: timing API absent

Command:

```text
python -m pytest -q tests/test_kimi_k3_timing.py
```

Observed:

```text
ModuleNotFoundError: No module named 'benchmarks.kimi_k3_timing'
1 error
```

### RED 2: Modal TP8 entrypoints absent

After the timing helper went green, the Modal source contract failed on:

```text
assert "def test_kimi_k3_decode()" in source
```

Observed: `1 failed, 6 passed`.

### GREEN

Local focused checks:

```text
ruff check benchmarks/kimi_k3_timing.py benchmarks/bench_kimi_k3_decode.py \
  tests/test_kimi_k3_timing.py modal_app.py
All checks passed!

python -m pytest -q tests/test_kimi_k3_timing.py
7 passed in 0.91s

python -m py_compile benchmarks/kimi_k3_timing.py \
  benchmarks/bench_kimi_k3_decode.py modal_app.py \
  tests/test_kimi_k3_timing.py
exit 0

git diff --check
exit 0
```

The CPU tests cover R-7 percentile interpolation, log-space geometric mean,
iteration-aligned rank maxima, the combined summary, exact dry-run shape and
manifest enumeration, and the exact `B300:8`/torchrun Modal interface.

## Modal runs

All runs used `--env rahul-dev`.

1. Pre-tuning full TP8 correctness, launch, and SM103 binary/resource checks:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-an6uYQtPkpBmKRchOuso6l).
   Every one of the eight ranks reported `66 passed` in about 64.5 seconds.
2. First complete benchmark:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-6zbEI7WGdiZwi99F7ITk0p).
   Remote execution completed and selected 148 CTAs, but local
   `--write-result /opt/cursor/artifacts/kimi_k3_decode_benchmark.tar` failed
   because the local artifact path rejected the subagent write.
3. Complete benchmark rerun with the required workspace fallback:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-AGVNlBsVDhuFtMYFraNzz5).
   Remote execution and local result write both completed.
4. Post-tuning full TP8 correctness, launch, and SM103 binary/resource checks:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-FBXdTsI51c27Tr7NAmsqdc).
   Every one of the eight ranks reported `66 passed` in about 69 seconds.

## Benchmark method

- Hardware: 8x NVIDIA B300 SXM6 AC, SM103, 148 SMs/GPU.
- Software: CUDA 13.2, PyTorch `2.13.0+cu132`, MoK `0.1.0`.
- Clocks during the saved run: 2032 MHz SM and 3996 MHz memory on all GPUs.
- Timing: 500 warmups and 1,000 CUDA Graph replays per shape.
- Each reported sample is that iteration's maximum CUDA-event duration across
  all eight ranks. Median, p90, p99, and geometric mean are computed from those
  1,000 rank-max samples.
- Input copies and graph capture are outside every timed interval.
- Four pre-captured graphs and input buffers rotate during warmup and timing.
  They pin 64 distinct experts. Their per-rank expert-weight working set is
  140,378,112 bytes, strictly larger than B300's measured 132,644,864-byte L2
  by 7,733,248 bytes.
- One prepared 2,113,939,968-byte weight allocation and one workspace are
  reused for tuning, correctness, graph capture, and all latency tables.
- Every candidate grid passed all 24 required correctness shapes before its
  primary timing point. Every final table shape received another correctness
  call immediately before timing.

## Final latency tables

All values are milliseconds at the winning 148-CTA, cluster-1 configuration.

### Raw decode

| M | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|
| 1 | 1.585216 | 1.615904 | 1.626758 | 1.589780 |
| 2 | 1.550400 | 1.581088 | 1.591443 | 1.555606 |
| 3 | 1.561024 | 1.593635 | 1.604996 | 1.567278 |
| 4 | 1.568960 | 1.601571 | 1.610888 | 1.574907 |
| 5 | 1.579040 | 1.611718 | 1.624128 | 1.584762 |
| 6 | 1.586272 | 1.615968 | 1.627083 | 1.591620 |
| 7 | 1.597536 | 1.628134 | 1.638657 | 1.602838 |
| 8 | 1.603680 | 1.634368 | 1.646752 | 1.608675 |

### DFlash block 8

| M | Requests | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 1.603712 | 1.636301 | 1.646738 | 1.609218 |
| 16 | 2 | 1.577024 | 1.604845 | 1.617701 | 1.581484 |
| 24 | 3 | 1.617696 | 1.640422 | 1.658023 | 1.620030 |
| 32 | 4 | 1.658000 | 1.677622 | 1.693378 | 1.660688 |
| 40 | 5 | 1.705920 | 1.725574 | 1.741921 | 1.707629 |
| 48 | 6 | 1.747072 | 1.761536 | 1.777537 | 1.748493 |
| 56 | 7 | 1.787616 | 1.802179 | 1.816704 | 1.787991 |
| 64 | 8 | 1.827968 | 1.843264 | 1.855616 | 1.828944 |

### DFlash block 16

| M | Requests | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1.577536 | 1.605635 | 1.618024 | 1.581951 |
| 32 | 2 | 1.658864 | 1.679341 | 1.691873 | 1.660814 |
| 48 | 3 | 1.748240 | 1.763683 | 1.779842 | 1.749402 |
| 64 | 4 | 1.828928 | 1.843206 | 1.855426 | 1.829076 |
| 80 | 5 | 1.923168 | 1.937408 | 1.945760 | 1.923951 |
| 96 | 6 | 2.012256 | 2.025411 | 2.038080 | 2.012790 |
| 112 | 7 | 2.101376 | 2.111587 | 2.121824 | 2.102206 |
| 128 | 8 | 2.199712 | 2.211971 | 2.224320 | 2.201009 |

## Grid tuning

The primary point was M=16/block16. Each candidate passed occupancy
(one resident CTA/SM for both instantiations), all 24 required correctness
shapes, graph replay correctness, and a p99/median stability limit of 2.0.

| Grid | Repeat | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 1.736704 | 1.770051 | 1.779715 | 1.743036 |
| 64 | 2 | 1.736736 | 1.769350 | 1.779045 | 1.742207 |
| 64 | 3 | 1.736752 | 1.767555 | 1.777760 | 1.742487 |
| 96 | 1 | 1.640448 | 1.671142 | 1.678312 | 1.646388 |
| 96 | 2 | 1.640512 | 1.670314 | 1.679392 | 1.646324 |
| 96 | 3 | 1.640480 | 1.669965 | 1.677281 | 1.646076 |
| 128 | 1 | 1.583216 | 1.612838 | 1.622321 | 1.588210 |
| 128 | 2 | 1.583360 | 1.611872 | 1.622946 | 1.588177 |
| 128 | 3 | 1.584096 | 1.611872 | 1.622112 | 1.588497 |
| 148 | 1 | 1.576928 | 1.605667 | 1.617032 | 1.581463 |
| 148 | 2 | 1.577056 | 1.605504 | 1.617888 | 1.581642 |
| 148 | 3 | 1.577024 | 1.605536 | 1.617921 | 1.581587 |

| Grid | Median of repeat medians | Dispersion | Relative dispersion | Max p99/median | Result |
|---:|---:|---:|---:|---:|:---|
| 64 | 1.736736 | 0.000048 | 0.0000277 | 1.024766 | Accepted |
| 96 | 1.640480 | 0.000064 | 0.0000390 | 1.023700 | Accepted |
| 128 | 1.583360 | 0.000880 | 0.0005557 | 1.025001 | Accepted |
| 148 | 1.577024 | 0.000128 | 0.0000812 | 1.025933 | Winner |

Cluster size 2 was rejected without implementation. The routed expert contains
handwritten `tcgen05.mma.cta_group::1` mixed-MXFP4 instructions and every
tensor-memory stage uses `tensor_allocator<1, 1>`. A real two-CTA expert
requires coordinated `cta_group::2` allocation, MMA, commit, multicast TMA,
and paired deallocation throughout. Changing only the launch cluster would not
measure that contract; changing only the allocator would make the path invalid.

## Correctness and hot-path evidence

- 96 tuning-candidate correctness calls plus 24 final pre-timing correctness
  calls passed.
- Across those 120 calls: maximum relative L1 `0.005328448`, minimum cosine
  similarity `0.999980330`, and maximum absolute deviation `0.125`.
- Every latency row contains exactly 1,000 rank-max samples: 24,000 measured
  samples total.
- Profiling recorded exactly one kernel:
  `kimi_k3_decode_persistent_kernel<true>`.
- The warmed public call raised no allocator event and memory allocated was
  unchanged at 2,153,548,800 bytes.

## Resource impact

Fresh `cuobjdump --dump-resource-usage` after the Task 10 build:

| Instantiation | Registers/thread | Stack | Local | Static shared |
|:---|---:|---:|---:|---:|
| Core | 249 | 0 | 0 | 1,104 B |
| Tensor | 248 | 0 | 0 | 1,264 B |

Compared with the Task 9 baseline, core resources are unchanged and the tensor
instantiation rises from 247 to 248 registers/thread because the measured grid
count is now a kernel argument. Stack, local memory, and static shared memory
are unchanged. Both instantiations still report one resident block/SM, pass the
register/shared-memory occupancy checks, contain the required native mixed-MMA,
BF16-MMA, multimem, and bounded-wait instruction classes, and contain no local
loads/stores.

## Artifacts

Modal returned the rank-0 archive to:

```text
/workspace/.superpowers/sdd/kimi_k3_decode_benchmark.tar
```

SHA-256:

```text
f5f997cf368cb3d0c2ab343d95ca3fc3055ffe5d21c64a21d170daf280a02295
```

Extracted inspection copy:

```text
/workspace/.superpowers/sdd/task-10-artifacts/
```

It contains all ten required files:

```text
manifest.json
latency_raw_decode.json
latency_raw_decode.csv
latency_block8.json
latency_block8.csv
latency_block16.json
latency_block16.csv
correctness.json
workspace_stats.json
tuning.json
```

## Files changed

- Added `benchmarks/kimi_k3_timing.py`.
- Added `benchmarks/bench_kimi_k3_decode.py`.
- Added `tests/test_kimi_k3_timing.py`.
- Modified `csrc/kimi_k3_decode/persistent_kernel.cuh`.
- Modified `csrc/kimi_k3_decode/persistent_sync.cuh`.
- Modified `csrc/kimi_k3_decode/entrypoints.cuh`.
- Modified `csrc/bindings.cu`.
- Modified `modal_app.py`.

The two extra C++ files beyond the brief's minimum list are necessary:
`persistent_sync.cuh` must count the measured runtime grid at grid barriers,
and `bindings.cu` exposes the guarded private benchmark setter/getter.

## Self-review

- The public decode API and numerical operations are unchanged.
- Production still defaults to the measured winner, 148 CTAs.
- The grid override is private, accepts only 64/96/128/148, and refuses calls
  unless `MOK_KIMI_K3_ENABLE_GRID_TUNING=1` is set in a dedicated process.
- The queue-ticket bound remains sized for the largest measured grid.
- Timing excludes setup, capture, input generation/copies, rank gathering, and
  statistics.
- The pure statistics are independent of CUDA and tested on CPU.
- The saved manifest identifies the exact commit, hardware, versions, clocks,
  shapes, counts, pool policy, selected-expert coverage, and timing method.
- The archive was parsed after download; all row/sample counts, correctness
  counts, selected grid, SHA, and L2-overflow condition were asserted.

## Concerns

- `/opt/cursor/artifacts` is a symlink into the subagent artifact store and
  rejected Modal's `--write-result`; the required fallback under
  `/workspace/.superpowers/sdd/` succeeded. The parent agent must copy that
  archive if it wants it uploaded from `/opt/cursor/artifacts`.
- Grid 148 beats grid 128 by only about 0.00634 ms (0.4%) at the primary point,
  but three 1,000-sample repeats agree, all p99 checks pass, and the winner is
  also the existing correctness baseline. No production constant change is
  justified by these data.
- There is no unresolved correctness, graph-capture, occupancy, launch-count,
  spill, or allocation concern.

## Review correction addendum: realistic routing

This addendum is authoritative and supersedes the original benchmark results
above. The original `1.58`-`2.22` ms tables and their archive are discarded:
their `+8` correction bias forced every token and every replay onto the same
16 experts, so they did not time the scheduler or routed queues at realistic
occupancy. They remain above only as chronology for the original Task 10 run.

### Corrective commits

- `3d2c7b5da5422089137e92901a041c369a47e3a3`
  (`bench: fix Kimi K3 realistic routing`)
- `8928c1afa5ca1056db8ae4065392f97073d1da2e`
  (`test: include Modal contract in B300 image`)

The benchmark archive identifies the latter commit exactly. The report-only
commit made after this addendum does not alter the image or measured code.

### Corrected RED / GREEN chronology

The original Task 10 timing test imported
`benchmarks.kimi_k3_timing` during collection, so its historical RED was a
collection error. That was genuine but did not meet the reviewed requirement
that failure occur inside a test body. The imports now happen in test bodies.

To reconstruct the corrected chronology without rewriting history, a detached
tree at the pre-Task-10 commit
`c460e0ea06450be49441add64af3e66ac489c082` ran the exact import and first
assertion from `test_percentile_uses_linear_interpolation`:

```text
python -m pytest -q test_timing_in_body_red.py
F
E ModuleNotFoundError: No module named 'benchmarks.kimi_k3_timing'
FAILED test_timing_in_body_red.py::test_percentile_uses_linear_interpolation
1 failed in 0.01s
```

The new review tests then had a genuine local RED against the old
implementation:

```text
python -m pytest -q tests/test_kimi_k3_timing.py
12 failed, 5 passed in 1.16s
```

Failures covered missing realistic-route modules and metadata, missing
rotating-order/effect-band logic, non-normalized tar output, the build-image
SHA, and the old dry-run pool contract. After implementation:

```text
ruff check benchmarks/bench_kimi_k3_decode.py \
  benchmarks/kimi_k3_timing.py benchmarks/kimi_k3_decode_inputs.py \
  benchmarks/kimi_k3_decode_data.py benchmarks/kimi_k3_decode_runtime.py \
  benchmarks/kimi_k3_decode_output.py benchmarks/kimi_k3_artifacts.py \
  tests/test_kimi_k3_timing.py modal_app.py
All checks passed!

python -m pytest -q tests/test_kimi_k3_timing.py
17 passed in 1.03s
```

The first expanded remote all-K3 run correctly exposed that the image did not
copy `modal_app.py`, which the source-contract test reads:

```text
465 passed, 1 failed, 1 skipped
FileNotFoundError: /root/mok/modal_app.py
```

That is the RED for the second corrective commit. Adding `modal_app.py` to the
explicit image allowlist made the same full suite green.

### Corrected Modal runs

All commands used `--env rahul-dev`.

1. Expanded-suite RED:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-kLKzws2niogkseEzk1eeP6).
   Every rank reached 465 passes before the missing source file failed.
2. Pre-benchmark all-K3 GREEN:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-APyfihwEU8banpB2I8MlUm).
   Every rank reported `466 passed, 1 skipped`; elapsed times were
   116.92-116.99 seconds.
3. Complete realistic-route tuning and 24-shape benchmark:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-9FX9at5T9ACbOfATdFEBv0).
   The returned archive was written locally and validated.
4. Post-benchmark all-K3 GREEN:
   [Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-qQh6BlFzaED584AKaN3gwz).
   Every rank reported `466 passed, 1 skipped`; elapsed times were
   196.03-196.68 seconds.

An invocation at
[Modal run](https://modal.com/apps/modal-labs/rahul-dev/ap-AFxZuKeraqAol43JqB0EhM)
was interrupted while its mounts were still being indexed because the caller
had supplied a mistyped manifest SHA. No remote benchmark process started and
no artifact from that invocation was retained.

### Corrected routing and cache working set

The benchmark now uses zero forcing bias. It preserves K3's natural
`sigmoid(router logits) + correction_bias` selection semantics with a
deterministic learned-style correction bias spanning `[-0.015, 0.015]`.
One-hot token directions and sparse deterministic router weights give each
token a distinct block of 16 experts until all 56 blocks are occupied.
Successive graph-pool entries rotate the block window and permute its token
and expert order.

Each table and tuning row stores all four pool entries' route assignments,
replay-local distinct experts, gate/up and down routed queue units, pool-wide
coverage, replay-local and pool-wide expert-weight bytes, and both L2
comparisons. One expert's per-rank packed weights and scales occupy
`2,193,408` bytes; B300 L2 is `132,644,864` bytes.

| M | Experts/replay | Routed gate/up units | Routed down units | Replay bytes | Exceeds L2/replay | Pool coverage | Pool bytes |
|---:|---:|---:|---:|---:|:---:|---:|---:|
| 1 | 16 | 48 | 448 | 35,094,528 | No | 64 | 140,378,112 |
| 2 | 32 | 96 | 896 | 70,189,056 | No | 128 | 280,756,224 |
| 3 | 48 | 144 | 1,344 | 105,283,584 | No | 192 | 421,134,336 |
| 4 | 64 | 192 | 1,792 | 140,378,112 | Yes | 256 | 561,512,448 |
| 5 | 80 | 240 | 2,240 | 175,472,640 | Yes | 320 | 701,890,560 |
| 6 | 96 | 288 | 2,688 | 210,567,168 | Yes | 384 | 842,268,672 |
| 7 | 112 | 336 | 3,136 | 245,661,696 | Yes | 448 | 982,646,784 |
| 8 | 128 | 384 | 3,584 | 280,756,224 | Yes | 512 | 1,123,024,896 |
| 16 | 256 | 768 | 7,168 | 561,512,448 | Yes | 896 | 1,965,293,568 |
| 24 | 384 | 1,152 | 10,752 | 842,268,672 | Yes | 896 | 1,965,293,568 |
| 32 | 512 | 1,536 | 14,336 | 1,123,024,896 | Yes | 896 | 1,965,293,568 |
| 40 | 640 | 1,920 | 17,920 | 1,403,781,120 | Yes | 896 | 1,965,293,568 |
| 48 | 768 | 2,304 | 21,504 | 1,684,537,344 | Yes | 896 | 1,965,293,568 |
| 56+ | 896 | 2,688 | 25,088 | 1,965,293,568 | Yes | 896 | 1,965,293,568 |

The per-replay assertion is therefore honestly false for raw M=1-3: no
construction selecting exactly `16*M` experts can make those routed weights
larger than B300 L2. The four-entry pool still rotates 64, 128, and 192 experts
respectively, and each pool-wide routed working set exceeds L2. At the primary
M=16 tuning point every replay occupies 256 experts and addresses
561,512,448 bytes. Every M>=56 replay occupies all 896 experts.

### Corrected full latency tables

All values are milliseconds. Each row contains 500 warmups and 1,000
per-iteration TP8 rank-maximum samples. The selected configuration is the
production 148-CTA, cluster-1 grid.

#### Raw decode

| M | Experts | Pool coverage | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 64 | 1.572576 | 1.590659 | 1.604032 | 1.575024 |
| 2 | 32 | 128 | 1.691952 | 1.715066 | 1.729056 | 1.690303 |
| 3 | 48 | 192 | 1.917424 | 1.940032 | 1.954370 | 1.915403 |
| 4 | 64 | 256 | 2.597440 | 2.626115 | 2.644256 | 2.600329 |
| 5 | 80 | 320 | 2.769472 | 2.794048 | 2.812493 | 2.769326 |
| 6 | 96 | 384 | 3.060288 | 3.082784 | 3.096996 | 3.059867 |
| 7 | 112 | 448 | 3.686240 | 3.714592 | 3.733066 | 3.689494 |
| 8 | 128 | 512 | 3.864128 | 3.890755 | 3.912233 | 3.865914 |

#### DFlash block 8

| M | Requests | Experts | Pool coverage | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 128 | 512 | 3.863408 | 3.889728 | 3.911379 | 3.864940 |
| 16 | 2 | 256 | 896 | 6.795808 | 6.822467 | 6.842996 | 6.797017 |
| 24 | 3 | 384 | 896 | 9.317360 | 9.338601 | 9.360969 | 9.318503 |
| 32 | 4 | 512 | 896 | 12.349040 | 12.377664 | 12.408384 | 12.351135 |
| 40 | 5 | 640 | 896 | 14.950272 | 14.985795 | 15.020471 | 14.950543 |
| 48 | 6 | 768 | 896 | 17.929632 | 17.965274 | 17.998325 | 17.931552 |
| 56 | 7 | 896 | 896 | 20.913696 | 20.948678 | 20.983136 | 20.915686 |
| 64 | 8 | 896 | 896 | 20.930113 | 20.968864 | 20.999554 | 20.930603 |

#### DFlash block 16

| M | Requests | Experts | Pool coverage | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 256 | 896 | 6.794720 | 6.823776 | 6.850109 | 6.796720 |
| 32 | 2 | 512 | 896 | 12.348928 | 12.376403 | 12.402323 | 12.350487 |
| 48 | 3 | 768 | 896 | 17.927520 | 17.957412 | 17.986080 | 17.927424 |
| 64 | 4 | 896 | 896 | 20.932144 | 20.968960 | 21.003691 | 20.933034 |
| 80 | 5 | 896 | 896 | 20.975088 | 21.015850 | 21.052932 | 20.977542 |
| 96 | 6 | 896 | 896 | 21.010032 | 21.050943 | 21.089843 | 21.013607 |
| 112 | 7 | 896 | 896 | 21.061184 | 21.095968 | 21.133433 | 21.062731 |
| 128 | 8 | 896 | 896 | 21.096145 | 21.136962 | 21.169825 | 21.095834 |

### Corrected grid tuning

All candidate correctness sweeps cover 24 shapes times four pool entries.
Measurements are interleaved with candidate orders
`[64,96,128,148]`, `[96,128,148,64]`, and `[128,148,64,96]`.
Every repeat persists its 1,000 raw rank-max samples and verifies every graph
again after the timed replays.

| Grid | Repeat | Order position | Median | p90 | p99 | Geomean |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 0 | 13.745376 | 13.788752 | 13.831826 | 13.748072 |
| 64 | 2 | 3 | 13.745728 | 13.789731 | 13.829712 | 13.748233 |
| 64 | 3 | 2 | 13.744048 | 13.788762 | 13.827348 | 13.746932 |
| 96 | 1 | 1 | 9.488880 | 9.521699 | 9.548022 | 9.490792 |
| 96 | 2 | 0 | 9.487056 | 9.516576 | 9.545298 | 9.489798 |
| 96 | 3 | 3 | 9.488800 | 9.520704 | 9.553733 | 9.490311 |
| 128 | 1 | 2 | 7.327296 | 7.355213 | 7.378509 | 7.327776 |
| 128 | 2 | 1 | 7.327184 | 7.353888 | 7.377474 | 7.328019 |
| 128 | 3 | 0 | 7.327264 | 7.351997 | 7.378179 | 7.327325 |
| 148 | 1 | 3 | 6.795712 | 6.822435 | 6.844781 | 6.797405 |
| 148 | 2 | 2 | 6.797664 | 6.822435 | 6.846883 | 6.798394 |
| 148 | 3 | 1 | 6.796864 | 6.826499 | 6.850882 | 6.798906 |

| Grid | Median of medians | Dispersion | Max p99/median | Result |
|---:|---:|---:|---:|:---|
| 64 | 13.745376 | 0.001679 | 1.006289 | Accepted |
| 96 | 9.488800 | 0.001824 | 1.006844 | Accepted |
| 128 | 7.327264 | 0.000112 | 1.006999 | Accepted |
| 148 | 6.796864 | 0.001952 | 1.007948 | Winner |

The minimum-effect band is the larger within-candidate median dispersion,
`0.0019521713256835938` ms for the measured fastest candidate versus
production. Grid 148 is itself fastest, so the recommendation is conclusive
without invoking that band. The guarded candidate is reset to 148 immediately
after tuning and again on exit. When the environment guard is absent, the
getter and every public launch return to production 148 even if storage
previously held another candidate.

### Correctness, replay, launch, and allocation evidence

- Tuning correctness: `384` candidate/shape/pool checks.
- Final pre-timing correctness: `96` shape/pool checks.
- Post-timing graph checks: `48` tuning checks and `96` final-table checks.
- Total persisted numeric checks: `624`.
- Across all 624: maximum relative L1 `0.005109573248773813`, minimum cosine
  `0.9999843239784241`, maximum absolute deviation `0.021484375`.
- All 24 final table rows contain exactly 1,000 rank-max samples, for 24,000
  final samples; tuning adds 12,000 persisted raw rank-max samples.
- The warmed public call profiles exactly one
  `kimi_k3_decode_persistent_kernel<true>` launch.
- Allocator events are empty and allocated bytes remain exactly
  `2,166,390,272` before and after the call.
- The same prepared `2,113,939,968`-byte per-rank weight allocation and one
  workspace are reused.

### Corrected SM103 resource impact

Fresh `cuobjdump --dump-resource-usage` from the measured image reports:

| Instantiation | Registers/thread | Stack | Local | Static shared |
|:---|---:|---:|---:|---:|
| Core | 248 | 0 | 0 | 1,104 B |
| Tensor | 247 | 0 | 0 | 1,264 B |

Removing the redundant kernel `grid_ctas` argument reduces both paths by one
register/thread relative to the first Task 10 build. The kernel now latches
`gridDim.x` once and supplies that value to every grid barrier, clearing
stride, quantization partition, and tail-role decision. Both instantiations
still pass one-CTA-per-SM occupancy, no-spill/no-local-memory, native mixed
MMA, BF16 MMA, multimem, and bounded-wait gates in both pre- and post-runs.

The minimum benchmark grid is derived by rounding the larger complete tail
role set up to a 32-CTA quantum, producing 64. Static assertions bind that
minimum to coordinator, reduce, core-shard, tensor-shard, and both complete
role counts. The maximum is the 148-CTA production grid and is statically tied
to the queue-ticket overshoot bound. Python reads the compiled candidate tuple
and asserts it matches dry-run metadata before benchmarking.

### Corrected artifacts and reproducibility

Authoritative archive:

```text
/workspace/.superpowers/sdd/kimi_k3_decode_benchmark_realistic.tar
```

SHA-256:

```text
3bd44606f11e50d66d4bf1ebd5f5c7f2c8f1f8f2c96a673ee3de1a0c277595a5
```

The archive contains exactly the ten required files. Every member is sorted
and normalized to mtime `0`, uid/gid `0`, empty uname/gname, mode `0644`, and
USTAR format. The remote function built it twice from the same output and
required byte equality before return. Local validation independently checked
all member metadata, JSON/CSV row counts, raw sample counts, routes, occupancy,
queue units, L2 comparisons, graph checks, correctness counts, selected grid,
manifest SHA, one-launch gate, and no-allocation gate.

`MOK_GIT_SHA` is no longer part of the image environment or content hash. The
exact SHA is supplied as the `bench_kimi_k3_decode(git_sha)` invocation
argument and forwarded only to the benchmark subprocess. A report/docs-only
commit therefore does not invalidate the CUDA build layer.

### Corrected files changed and self-review

- Added focused benchmark modules for deterministic input/weight construction,
  the TP8 oracle/runtime observations, manifest/output writing, routing
  metadata, and deterministic tar creation. Every new benchmark file is below
  1,000 lines; the main driver is 999 lines.
- The benchmark imports no `tests.*` or pytest support module.
- The public API and numerical kernel stages are unchanged.
- The private grid control is guard-checked on every read and write, accepts
  only the compiled candidate tuple, and cannot leak a candidate into an
  unguarded launch.
- Every tuning candidate and final row checks every graph-pool entry before
  timing; every timed graph is checked again after its 1,000 replays.
- The pool-wide and replay-local fields are separate and unambiguously named.
- Timing still excludes construction, input copies, graph capture, correctness,
  rank gathering, and statistics.
- B300 clocks were 2,032 MHz SM and 3,996 MHz memory on all eight GPUs.

### Corrected concerns

- Raw M=1-3 cannot exceed B300 L2 with replay-local routed expert weights while
  preserving the required `16*M` realistic occupancy. Their four-entry pools
  do exceed L2; this distinction is explicit in every row rather than hidden
  behind a pool-wide number.
- Realistic routing increases block-16 M=16 median from the invalid 1.58 ms
  result to `6.794720` ms and full-occupancy M=128 to `21.096145` ms. These
  corrected values are the only Task 10 results suitable for downstream
  baseline comparisons.
- There is no unresolved correctness, graph-replay, p99, occupancy, resource,
  one-launch, allocation, archive, or reproducibility concern.
