#pragma once

/// The block-scaled contraction, and the transfers that feed it.
///
/// One `tcgen05.mma` per eight-group factor against one shared weight tile and
/// one shared activation tile, and the two `cp.async.bulk.tensor` transfers that
/// stage a slab's weights and its sixteen E8M0 scale tiles. Every ring in this
/// engine issues exactly these, in the same shapes, against the same
/// descriptors -- the rings differ in what they hold and in what order they walk
/// the stream, never in the arithmetic.

#include "format.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

// ---------------------------------------------------------------------------
// The contraction.
// ---------------------------------------------------------------------------

/// The instruction descriptor for weight-on-M, tokens-on-N, K = 32.
///
/// PTX ISA 9.1, "tcgen05.mma instruction descriptor". The weight is the A
/// operand and is E2M1; the activation is the B operand and is E4M3. That is
/// the opposite assignment from the production gate/up unit, where the batch is
/// on M, and the same one the production grouped routed-down pipeline uses.
/// Both operands read their block scales out of the same quad of K groups, so
/// one scale-factor id serves both.
__host__ __device__ __forceinline__ constexpr std::uint32_t
fused_instruction_descriptor(const int scale_factor_id) {
    return (static_cast<std::uint32_t>(scale_factor_id) << 4)   // b_sf_id
         | (5u << 7)                                            // a_format E2M1
         | (0u << 10)                                           // b_format E4M3
         | (static_cast<std::uint32_t>(kFusedN / 8) << 17)       // n_dim
         | (1u << 23)                                           // UE8M0 scales
         | (static_cast<std::uint32_t>(kFusedM / 16) << 24)      // m_dim
         | (static_cast<std::uint32_t>(scale_factor_id) << 29);  // a_sf_id
}

// The production grouped-down pipeline contracts this same shape, so the two
// paths are pinned to one encoding rather than to two spellings that happen to
// agree at a single scale-factor id.
static_assert(fused_instruction_descriptor(0)
                  == grouped_pipeline::grouped_instruction_descriptor(0));
static_assert(fused_instruction_descriptor(3)
                  == grouped_pipeline::grouped_instruction_descriptor(3));
static_assert(fused_instruction_descriptor(0) == 0x08820280u);
static_assert(fused_instruction_descriptor(3) == 0x688202b0u);

/// Issue one K = 32 block-scaled contraction of a slab chunk.
///
/// The operand descriptors arrive already advanced to the chunk, because a slab
/// issues sixteen of these from one unrolled body and re-deriving the tile base
/// sixteen times is sixteen redundant address computations. There is no
/// cross-proxy fence here either: the weights arrive through the async proxy
/// and are published by their mbarrier, and the activation's ordinary stores
/// are fenced once per slab rather than once per issue.
__device__ __forceinline__ void fused_mixed_mma(
    const fused_accumulator_tile &destination,
    const std::uint64_t weight_chunk,
    const std::uint64_t activation_chunk,
    const kittens::full_tt_fp8e8m0<16> &weight_scale,
    const kittens::full_tt_fp8e8m0<16> &activation_scale,
    const int scale_factor_id,
    const bool accumulate
) {
    const std::uint32_t instruction =
        fused_instruction_descriptor(scale_factor_id);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :
        : "r"(destination.addr),
          "l"(weight_chunk),
          "l"(activation_chunk),
          "r"(instruction),
          "r"(weight_scale.addr),
          "r"(activation_scale.addr),
          "r"(accumulate ? 1u : 0u)
    );
}

// ---------------------------------------------------------------------------
// The transfers.
// ---------------------------------------------------------------------------

/// Bytes one slab's weight transfer commits to its mbarrier.
///
/// `16U4_ALIGN16B` reads 64 packed bytes per box row and writes 128 shared
/// bytes, so the two candidate counts differ by exactly 2x: the global payload
/// the copy reads, and the shared footprint it writes after the format's
/// container padding. Guessing wrong hangs the wait or releases it early, and
/// which one the transaction counter tracks is a property of the hardware
/// rather than of this file.
///
/// Measured on B300: it is the *global* payload. Expecting the 65,536-byte
/// shared footprint of a 128x512 slab never completes the mbarrier; expecting
/// the 32,768 bytes the copy read does.
/// `tests/test_kimi_k3_w13_layout.py::
/// test_the_transaction_count_is_the_payload_read_not_the_footprint_written`
/// is that measurement, run against one tile under a bounded wait so a wrong
/// count is reported rather than hanging the device.
inline constexpr int kFusedWeightTransactionBytes = kFusedSlabGlobalBytes;

static_assert(kFusedWeightTransactionBytes
                  == static_cast<int>(sizeof(fused_weight_tile)) / 2,
              "the format pads its payload 2x on the way into shared memory");

/// Total bytes one slab's mbarrier expects: the weight tile and its scales.
inline constexpr int kFusedSlabTransactionBytes =
    kFusedWeightTransactionBytes + kFusedSlabScaleBytes;

/// Issue one `(task, slab)` weight tile into a shared slab tile.
///
/// The box is the whole slab -- 128 U4 values by 128 rows by four boxes -- so a
/// slab is one instruction rather than four, and the descriptor's own third
/// dimension is what walks the boxes.
__device__ __forceinline__ void load_fused_slab_async(
    fused_weight_tile &destination,
    const CUtensorMap *__restrict__ map,
    const int expert,
    const int task_slab,
    kittens::semaphore &arrived
) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :
        : "r"(static_cast<std::uint32_t>(
              __cvta_generic_to_shared(&destination))),
          "l"(reinterpret_cast<std::uint64_t>(map)),
          "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(&arrived))),
          "n"(0), "n"(0), "n"(0), "r"(task_slab), "r"(expert)
        : "memory"
    );
}

/// Where one `(task, slab)` group of scale tiles starts in the fused blob.
__host__ __device__ __forceinline__ long long fused_scale_offset(
    const int expert,
    const int task_slab
) {
    return (static_cast<long long>(expert) * kFusedTaskSlabs + task_slab)
         * kFusedSlabScaleBytes;
}

/// Issue one `(task, slab)` weight tile and its scale quads together.
///
/// One mbarrier covers both transfers because nothing consumes either without
/// the other: the sixteen contractions of a slab read the payload and the four
/// scale tiles in the same breath.
__device__ __forceinline__ void load_fused_slab(
    fused_weight_tile &payload,
    mixed_scale_tile (&scales)[kFusedSlabScaleTiles],
    const CUtensorMap *__restrict__ map,
    const std::uint8_t *__restrict__ fused_scale,
    const int expert,
    const int task_slab,
    kittens::semaphore &arrived
) {
    kittens::tma::expect_bytes(arrived, kFusedSlabTransactionBytes);
    load_fused_slab_async(payload, map, expert, task_slab, arrived);
    kittens::tma::load_async(
        reinterpret_cast<void *>(&scales[0]),
        const_cast<void *>(reinterpret_cast<const void *>(
            fused_scale + fused_scale_offset(expert, task_slab))),
        kFusedSlabScaleBytes, arrived);
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
