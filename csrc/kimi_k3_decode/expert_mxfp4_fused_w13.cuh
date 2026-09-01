#pragma once

// The production fused-W13 K512 routed gate/up engine.
//
// The routed gate and up projections are one contraction here, not two. The
// earlier gate/up unit put the assignment batch on the MMA's M axis, walked K
// in eight-group rounds, kept two 128-column accumulators -- one for gate, one
// for up -- and staged every weight byte with scalar loads and shared stores.
// Its measured profile put 29.4% of the decode step inside that staging and
// named the remedy: a copy engine reading the global tile straight into its
// swizzled shared destination, under an `m128x8x32` contraction that puts
// output channels on M and tokens on N. That is what this file is, and the
// paired A/B that replaced the old unit with it measured 5.2% off the full step
// at M = 16 and 6.7% at M = 128, at bit-identical output.
//
// One unit, six tasks
// -------------------
// A unit is one occupied expert. It decomposes into six tasks, each one
// 128-row output tile: 64 gate channels in M rows `[0, 64)` and the *same* 64
// channels' up rows in M rows `[64, 128)`. Six tasks cover the 384 `situ`
// columns this rank owns. The six run sequentially through one accumulator,
// each walking K = 3584 as seven 512-wide slabs and issuing sixteen K = 32
// block-scaled contractions per slab. Because gate and up share the
// accumulator's M axis, the epilogue reads one tensor-memory tile, pairs M row
// `r` with M row `r + 64`, and gets both halves of one output channel -- so no
// value is read twice and no second accumulator exists to read.
//
// The activation is gathered once for the whole unit, by the whole CTA. That is
// the measurement this structure exists for: the rows a slab needs do not
// depend on the task, so gathering per `(task, slab)` pair did 42 gathers per
// expert where seven distinct ones exist, all of them on warp 0. Gathering all
// seven once on 256 threads was worth 5.2%; `stage_fused_unit_activation` is
// that gather and section 15 of the Task 11b report is that measurement.
//
// The weight transfer
// -------------------
// A slab's weights are one `cp.async.bulk.tensor.5d` and its sixteen E8M0
// scale factors per row are one 2,048-byte contiguous `cp.async.bulk`. Both are
// possible only because `mok.kimi_k3_w13`, which
// `prepare_kimi_k3_decode_weights` runs once per model load, already put the
// bytes in the order the descriptor and `tcgen05.cp` consume:
//
//   * `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` is the only tensor-map type that
//     lands packed FP4 in the sixteen-byte containers `kind::mxf8f6f4` reads,
//     and it pins `boxDim[0] = 128` U4 values -- 64 global bytes widening to
//     128 shared bytes, which is exactly one 128B swizzle atom. So the
//     descriptor's natural granularity is K = 128 and a slab is a whole number
//     of them. K = 512 is the widest such multiple that leaves the whole
//     expert's activation resident under the opt-in shared maximum; see
//     `kFusedStagingBytes` for the arithmetic.
//   * the scale blob is pre-shuffled into `scale_factor_1x_offset` order, so a
//     slab's 2,048 scale bytes are contiguous and `tcgen05.cp` reads them
//     without any per-byte gather.
//
// The ring
// --------
// Two weight stages over one 42-long `(task, slab)` stream, and no CTA-wide
// barrier between a task's first contraction and its last. Warp 0 owns the
// whole K loop: it issues the transfers, drives `tcgen05`, and releases a stage
// by committing the MMA's own completion to that stage's mbarrier. Every other
// warp waits at the barrier that precedes each epilogue.
//
// The retire wait is off by one on purpose. Stream index `i` issues its sixteen
// contractions and then waits for index `i - 1`'s, which the tensor core
// finished while those issues were being made -- so the wait is normally free
// and the tensor core is never drained, whereas waiting on index `i`'s own
// completion would empty the pipe 42 times per expert. The stream does not
// restart at a task boundary: the next task's transfers are issued *before* the
// epilogue precisely so they fly underneath it.

// How this is laid out
// --------------------
// The engine was one 2,600-line header and is now a chain of focused ones,
// included here in dependency order. Each opens the same three namespaces and
// includes the one before it, so this file is the only include any caller
// needs and the parts stay in the order a reader would meet them: what the
// bytes are, what contracts them, what drains them, what gathers the other
// operand, then the three rings and the selector over two of them, then the
// host side and the probes.
//
// Nothing was renamed and nothing moved between namespaces, so a symbol's
// home is `git log --follow` on the part that holds it.

#include "fused_w13/format.cuh"
#include "fused_w13/contraction.cuh"
#include "fused_w13/epilogue.cuh"
#include "fused_w13/activation.cuh"
#include "fused_w13/resident_unit.cuh"
#include "fused_w13/engines.cuh"
#include "fused_w13/slab_unit.cuh"
#include "fused_w13/compact_unit.cuh"
#include "fused_w13/adaptive_unit.cuh"
#include "fused_w13/descriptor.cuh"
#include "fused_w13/probe.cuh"
