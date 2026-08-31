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

#include "expert_mxfp4.cuh"
#include "expert_mxfp4_grouped.cuh"
// The unit publishes its six completed column ranges from inside itself, so it
// needs the same arrival primitive the persistent kernel publishes with rather
// than a second copy of it.
#include "persistent_sync.cuh"

#include <ATen/ops/zeros.h>
#include <cuda.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

// ---------------------------------------------------------------------------
// Geometry. Mirrors `mok/kimi_k3_w13.py`, which builds the bytes.
// ---------------------------------------------------------------------------

/// Output channels one task presents on the MMA's M axis, and the half of them
/// that are gate rows.
inline constexpr int kFusedM = kMmaM;
inline constexpr int kFusedHalfRows = kFusedM / 2;

/// Assignment tokens on the MMA's N axis, and the tile height that carries
/// them. Eight is what the contraction contracts; sixteen is the shortest a
/// swizzled shared tile may be.
inline constexpr int kFusedN = 8;
inline constexpr int kFusedPhysicalN = 16;

/// Tasks one expert decomposes into.
inline constexpr int kFusedTasks =
    kRoutedIntermediateSizePerRank / kFusedHalfRows;

/// Contraction width one slab covers, the slabs that tile K, and the K = 32
/// contractions one slab issues.
inline constexpr int kFusedSlabK = 512;
inline constexpr int kFusedSlabs = kLatentSize / kFusedSlabK;
inline constexpr int kFusedSlabGroups = kFusedSlabK / kMmaK;
inline constexpr int kFusedSlabScaleTiles =
    kFusedSlabGroups / kScaleGroupsPerTile;

/// Packed bytes of one row of one slab, and of one whole `(task, slab)` tile.
inline constexpr int kFusedSlabRowBytes = kFusedSlabK / 2;
inline constexpr int kFusedSlabGlobalBytes = kFusedM * kFusedSlabRowBytes;

/// E8M0 bytes of one `(task, slab)` tile: four 512-byte scale tiles.
inline constexpr int kFusedSlabScaleBytes =
    kFusedSlabScaleTiles * kScaleRows * kScaleColumns;

/// `(task, slab)` pairs one expert holds, which is the transform's outer axis.
inline constexpr int kFusedTaskSlabs = kFusedTasks * kFusedSlabs;

/// `situ` quantization groups one task owns, contiguous in the scratch layout.
inline constexpr int kFusedSituGroups = kFusedHalfRows / kMmaK;

/// U4 values one `16U4_ALIGN16B` box row carries, which the format pins to 128,
/// and the boxes one slab row is therefore made of.
inline constexpr int kFusedBoxElements = 128;
inline constexpr int kFusedBoxes = kFusedSlabK / kFusedBoxElements;

/// Weight ring depth, and the slab buffers the resident activation occupies.
///
/// Two stages, not three. Staging all seven slabs of activation costs 71,680
/// bytes and the epilogue's result tile another 8,192, and neither leaves room
/// for a third weight stage against the 231,424 dynamic bytes the 227 KiB
/// opt-in leaves once a kilobyte is reserved for static shared memory:
///
///   3 x 65,536 + 6,144 + 71,680 + 8,192 = 282,624   > ceiling
///   2 x 65,536 + 4,096 + 71,680 + 8,192 = 215,040   fits
///
/// That trade was measured rather than assumed. A per-task engine with three
/// stages and two activation buffers spent 40.6% of the phase gathering
/// activation and 3.7% waiting on a full ring; moving the gather out of the
/// task cost one stage and won 5.2% of the whole step. Section 20 of the Task
/// 11b report records that paired A/B, and sections 21 to 27 record the attempt
/// to buy the stage back by packing the activation into eight-row slots of one
/// tile, which freed 24,576 bytes and was slower at every shape measured.
inline constexpr int kFusedStages = 2;
inline constexpr int kFusedActivationSlabs = kFusedSlabs;

/// The 42-long `(task, slab)` weight stream one unit consumes in order.
inline constexpr int kFusedStreamLength = kFusedTaskSlabs;

static_assert(kFusedTasks == 6);
static_assert(kFusedSlabs == 7);
static_assert(kFusedSlabGroups == 16);
static_assert(kFusedSlabScaleTiles == 4);
static_assert(kFusedSlabRowBytes == 256);
static_assert(kFusedSlabGlobalBytes == 32768);
static_assert(kFusedSlabScaleBytes == 2048);
static_assert(kFusedTaskSlabs == 42);
static_assert(kFusedSituGroups == 2);
static_assert(kFusedBoxes == 4);
static_assert(kFusedStreamLength == 42);
static_assert(kFusedStages == 2,
              "the stream flips its stage with `index % 2`, and the retire wait "
              "frees the stage the next refill writes only when the depth is 2");
static_assert(kFusedTasks * kFusedSituGroups == kSituGroups,
              "six tasks must cover every situ quantization group");
static_assert(kFusedSlabs * kFusedSlabGroups == kLatentGroups,
              "seven slabs of sixteen groups must be the whole latent width");

/// Shape of the prepared payload the decode operator accepts.
inline constexpr int kFusedPackedRows = kFusedTaskSlabs * kFusedM;
inline constexpr int kFusedPackedColumns = kFusedSlabRowBytes;
inline constexpr int kFusedScaleRows = kFusedTaskSlabs;
inline constexpr int kFusedScaleColumns = kFusedSlabScaleBytes;

/// Base alignment `cuTensorMapEncodeTiled` demands of the packed payload.
///
/// Stricter than the sixteen bytes every other weight is held to, and it is the
/// descriptor rather than any load that demands it, so the operator checks the
/// figure the encoder documents rather than the one a vector load needs.
inline constexpr int kFusedPackedAlignment = 32;

// The transform is a permutation, so the fused payload holds exactly the bytes
// of the two halves it replaces.
static_assert(kFusedPackedRows * kFusedPackedColumns
                  == 2 * kExpertW1W3PackedRows * kExpertW1W3PackedColumns);
static_assert(kFusedScaleRows * kFusedScaleColumns
                  == 2 * kExpertW1W3PackedRows * kExpertW1W3ScaleColumns);

// ---------------------------------------------------------------------------
// Tiles.
//
// The weight tile is a whole slab, so the MMA's K = 32 chunk index walks it
// without a second descriptor: a 128-row 128B-swizzled tile is stored one
// 128-byte atom column at a time, `chunk_descriptor` advances 32 bytes inside
// an atom and `rows * 128` between atoms, and that is exactly the layout the
// five-dimensional tensor map writes.
//
// The activation tile is sixteen rows because that is the shortest a swizzled
// shared tile may be; only the first eight are ever contracted. Both operands
// are one byte per K value here -- E4M3 natively, E2M1 through the format's
// 2x container padding -- so both tiles are 512 bytes wide for the same K and
// the same chunk index means the same K group in both.
// ---------------------------------------------------------------------------

using fused_weight_tile = kittens::st_fp8e4m3<kFusedM, kFusedSlabK>;
using fused_activation_tile = kittens::st_fp8e4m3<kFusedPhysicalN, kFusedSlabK>;
using fused_accumulator_tile = kittens::tt_fl<kFusedM, kFusedPhysicalN>;
using fused_result_tile = kittens::st_fl<kFusedM, kFusedPhysicalN>;

static_assert(sizeof(fused_weight_tile) == 65536);
static_assert(sizeof(fused_activation_tile) == 8192);
static_assert(sizeof(fused_result_tile) == 8192);
static_assert(fused_weight_tile::swizzle_bytes == 128,
              "the 16U4_ALIGN16B box widens to 128 shared bytes, so 128B is "
              "the only swizzle the descriptor and the tile can share");
static_assert(fused_activation_tile::swizzle_bytes == 128,
              "the activation must share the weights' chunk stride");

/// Shared bytes one unit's ring occupies.
///
/// The allocator aligns every array to 1 KiB and every array here is already a
/// multiple of it, so this is the exact figure rather than a lower bound:
///
///   2 x 65,536  weight slabs             = 131,072
///   2 x  2,048  weight scale quads       =   4,096
///   7 x  8,192  activation slabs         =  57,344
///   7 x  2,048  activation scale quads   =  14,336
///   1 x  8,192  epilogue result tile     =   8,192
///                                          -------
///                                          215,040
///
/// The result tile is a real array rather than an overlay on the ring, because
/// the ring does not go idle at a task boundary: the next task's two transfers
/// are issued before each epilogue precisely so they fly underneath it, and an
/// overlay would have the copy engine writing the bytes the epilogue reads.
inline constexpr int kFusedStagingBytes =
    kFusedStages * static_cast<int>(sizeof(fused_weight_tile))
    + kFusedStages * kFusedSlabScaleTiles
          * static_cast<int>(sizeof(mixed_scale_tile))
    + kFusedActivationSlabs * static_cast<int>(sizeof(fused_activation_tile))
    + kFusedActivationSlabs * kFusedSlabScaleTiles
          * static_cast<int>(sizeof(mixed_scale_tile))
    + static_cast<int>(sizeof(fused_result_tile));

static_assert(kFusedStagingBytes == 215040);

/// What the allocator wastes before the ring's first byte.
///
/// `tma_swizzle_allocator` aligns every array to 1 KiB by *absolute* shared
/// address, and the dynamic block does not start at a 1 KiB boundary: it starts
/// wherever the static shared memory the kernel declares ends, rounded up to
/// the 16-byte alignment of `extern __shared__ __align__(16) int shared_raw[]`.
/// So the first `align_ptr` skips forward by up to 1,024 - 16 bytes, and every
/// array after it is shifted by that same amount.
///
/// The other stages never had to account for this: the widest of them occupies
/// 131,072 bytes of the 216,064 the grid asks for, and the skip disappears into
/// the slack. This ring is sized to the byte, so the skip has to be paid for
/// explicitly or the last scale quad lands past the block's last granted byte
/// -- which is an out-of-range shared address on every CTA that runs a unit,
/// not a rare one.
inline constexpr int kFusedAllocatorPadding = 1024;

/// Static shared memory the two production instantiations declare.
///
/// A block's opt-in maximum covers static and dynamic together, so the launch
/// figure below has to leave this much unasked-for. It is not the allocator's
/// grain and has nothing to do with it: the grain is a skip *inside* the
/// dynamic block, this is what sits below the block's base.
///
/// ptxas assigns 1,312 bytes to the wider of the two instantiations. That is a
/// property of the build rather than of this file, so the figure is a rounded
/// reserve and `test_a_third_K512_weight_stage_does_not_fit_the_static_shared_ptxas_assigned`
/// reads the real one out of `cuobjdump --dump-resource-usage` and holds it
/// under this.
inline constexpr int kFusedStaticSharedReserve = 2048;

/// Dynamic shared memory the decode kernel launches with.
///
/// The ring's own footprint plus the padding above, which is what the whole
/// grid asks for because this is the widest stage it runs. B300 exposes 227 KiB
/// per block on opt-in, so the launch and the static reserve together leave
/// 14,336 bytes unused -- not enough for the 67,584 a third weight stage
/// wants. `_kimi_k3_fused_w13_shared_footprint` reads the allocator's skip off
/// the device rather than trusting the reasoning above.
inline constexpr int kFusedW13SharedBytes =
    kFusedStagingBytes + kFusedAllocatorPadding;

static_assert(kFusedW13SharedBytes == 216064);
static_assert(kFusedW13SharedBytes
                  <= kittens::MAX_SHARED_MEMORY - kFusedStaticSharedReserve,
              "the fused ring must leave room for static shared memory");
static_assert(2 * kFusedW13SharedBytes > kittens::MAX_SHARED_MEMORY,
              "the fused grid must still be one CTA per SM");
static_assert(kFusedW13SharedBytes >= kGateUpUnitSharedBytes
                  && kFusedW13SharedBytes >= kDownUnitSharedBytes,
              "the grid this ring sizes still runs every other stage");
// The allocator's worst-case skip is one byte short of its grain, so the launch
// has to grant the ring's bytes plus that skip. Restating it as an inequality
// on the launch figure is what makes a future array added to the ring fail here
// rather than run off the end of the block.
static_assert(kFusedStagingBytes + kFusedAllocatorPadding - 16
                  <= kFusedW13SharedBytes,
              "the launch must grant the ring's bytes plus the allocator's "
              "worst-case skip from the dynamic block's 16-byte-aligned base");

/// First tensor-memory column the fused scale buffers occupy.
///
/// One accumulator sits at column zero and is sixteen columns wide; the scale
/// buffers are four columns each and start above the band production reserves,
/// so the two layouts cannot be confused for one another.
inline constexpr int kFusedScaleColumnBase = kRoutedScaleColumnBase;

/// Tensor-memory scale buffers one slab occupies, and the sets that hold them.
///
/// A slab needs eight: four weight quads and four activation quads. They are
/// double buffered by slab parity, because a slab's contractions are still
/// reading its scales while the next slab is staging its own -- the ring only
/// keeps the tensor core fed if the copy for slab `s + 1` does not have to wait
/// on slab `s`'s issues. Two sets of eight is sixty-four tensor-memory columns,
/// which the pool has to spare, so the depth is free.
inline constexpr int kFusedScaleSlots = 2 * kFusedSlabScaleTiles;
inline constexpr int kFusedScaleSets = 2;
inline constexpr int kFusedScaleBuffers = kFusedScaleSets * kFusedScaleSlots;

static_assert(kFusedPhysicalN <= kFusedScaleColumnBase);
static_assert(kFusedScaleBuffers == 16);
static_assert(kFusedScaleColumnBase
                  + kRoutedScaleColumns * kFusedScaleBuffers
              <= kittens::tensor_allocator<1, 1>::cols);

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

// ---------------------------------------------------------------------------
// Epilogue.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void store_fused_accumulator(
    const fused_accumulator_tile &accumulator,
    fused_result_tile &destination
) {
    using namespace kittens;
    if (warpgroup::groupid() == 0) {
        rt_fl<kFusedM / 4, kFusedPhysicalN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(destination, result);
    }
}

/// Pair the accumulator's row halves, apply SiTU, and quantize 64 columns.
///
/// M row `r` of the accumulator is gate channel `64 * task + r` and M row
/// `r + 64` is that same channel's up value, so one tensor-memory tile carries
/// both halves and nothing is read twice. The arithmetic is the production
/// `quantize_situ_tile` expression, unchanged and evaluated in the same order,
/// because the numerical gate compares the two paths' `situ` bytes directly.
__device__ __forceinline__ void quantize_fused_situ(
    const fused_result_tile &result,
    const Scratch &scratch,
    const int assignment_begin,
    const int rows,
    const int task
) {
    const int thread = static_cast<int>(threadIdx.x);
    const int output_base = task * kFusedHalfRows;
    for (int index = thread; index < rows * kFusedSituGroups;
         index += kDecodeCtaThreads) {
        const int row = index / kFusedSituGroups;
        const int local_group = index % kFusedSituGroups;
        const int assignment = assignment_begin + row;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const int channel = local_group * kMmaK + k;
            const float gate_value = result[{channel, row}];
            const float up_value = result[{channel + kFusedHalfRows, row}];
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value =
                4.0f * tanhf(gate_value * 0.25f) * sigmoid
                * 25.0f * tanhf(up_value / 25.0f);
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale = select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        scratch.situ_scale[
            static_cast<long long>(assignment) * kSituGroups
            + task * kFusedSituGroups + local_group] = scale;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            scratch.situ_mxfp8[
                static_cast<long long>(assignment)
                    * kRoutedIntermediateSizePerRank
                + output_base + local_group * kMmaK + k] =
                    quantize_e4m3(values[k], reciprocal);
        }
    }
}

// ---------------------------------------------------------------------------
// Activation staging: the whole batch, the whole of K, once, by the whole CTA.
//
// The rows are gathered rather than copied: a row is one assignment's token,
// and consecutive assignments are unrelated tokens, so no copy engine can
// express the source. But which rows a slab needs does not depend on the task,
// so the seven distinct gathers an expert needs are done once for all six of
// its tasks -- the measurement that shape exists for is section 15 of the Task
// 11b report, where doing them per `(task, slab)` pair instead cost 40.6% of
// the phase at M = 16 and 45.9% at M = 128.
//
// The addressing is one flat index over `(slab, row, atom)` with the atom
// innermost, so a warp's thirty-two lanes write one row's 512 contiguous bytes
// and 256 threads cover one slab's eight rows per iteration. Seven iterations
// stage all of K = 3584, every load a fully coalesced 16-byte-per-lane read of
// one token's latent row.
// ---------------------------------------------------------------------------

/// One sixteen-byte swizzle atom of an activation row.
__device__ __forceinline__ uint4 *fused_activation_atom(
    fused_activation_tile &tile,
    const int row,
    const int atom
) {
    return reinterpret_cast<uint4 *>(&tile[{row, atom * 16}]);
}

/// Gather every live activation row and scale for all seven slabs at once.
///
/// Called by the whole CTA. The MMA always contracts eight N columns, so a
/// batch shorter than eight leaves some of them out of the result and they have
/// to read as zero against a unit scale rather than as whatever the last unit
/// left behind: `0xff` is E8M0's NaN and would poison its own accumulator
/// column.
///
/// The caller owes the asynchronous proxy a fence on every thread and a CTA
/// barrier before the tensor core reads any of this.
__device__ __forceinline__ void stage_fused_unit_activation(
    fused_activation_tile (&payload)[kFusedActivationSlabs],
    mixed_scale_tile (&scales)[kFusedActivationSlabs][kFusedSlabScaleTiles],
    const Scratch &scratch,
    const int assignment_begin,
    const int rows
) {
    constexpr int atoms_per_row = kFusedSlabK / 16;
    constexpr int atoms_per_slab = kFusedN * atoms_per_row;
    constexpr int atoms = kFusedActivationSlabs * atoms_per_slab;
    constexpr int quads_per_slab = kFusedN * kFusedSlabScaleTiles;
    constexpr int quads = kFusedActivationSlabs * quads_per_slab;

    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < atoms; index += kDecodeCtaThreads) {
        const int slab = index / atoms_per_slab;
        const int within = index % atoms_per_slab;
        const int row = within / atoms_per_row;
        const int atom = within % atoms_per_row;
        if (row < rows) {
            const int token =
                scratch.assignment_tokens[assignment_begin + row];
            *fused_activation_atom(payload[slab], row, atom) =
                *reinterpret_cast<const uint4 *>(
                    scratch.latent_mxfp8
                    + static_cast<long long>(token) * kLatentSize
                    + slab * kFusedSlabK + atom * 16);
        } else {
            *fused_activation_atom(payload[slab], row, atom) =
                make_uint4(0u, 0u, 0u, 0u);
        }
    }
    for (int index = thread; index < quads; index += kDecodeCtaThreads) {
        const int slab = index / quads_per_slab;
        const int within = index % quads_per_slab;
        const int row = within / kFusedSlabScaleTiles;
        const int quad = within % kFusedSlabScaleTiles;
        std::uint32_t word = 0x7f7f7f7fu;
        if (row < rows) {
            const int token =
                scratch.assignment_tokens[assignment_begin + row];
            word = *reinterpret_cast<const std::uint32_t *>(
                scratch.latent_scale
                + static_cast<long long>(token) * kLatentGroups
                + slab * kFusedSlabGroups + quad * kScaleGroupsPerTile);
        }
        stage_scale_quad(scales[slab][quad], row, word);
    }
}

// ---------------------------------------------------------------------------
// One expert unit.
// ---------------------------------------------------------------------------

/// Contract all six fused output tasks of one expert batch and stage its SiTU.
///
/// One queue claim, one accumulator, one activation staging, and one 42-long
/// weight stream. `arrival_counter` is the expert's gate/up readiness counter:
/// this unit publishes six arrivals into it, one per completed 64-column range,
/// so the grouped down phase's threshold is the six it has always been and it
/// cannot start before all 384 columns exist.
static __device__ void routed_gate_up_fused_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const CUtensorMap *__restrict__ packed_map,
    const std::uint8_t *__restrict__ fused_scale,
    const Scratch &scratch,
    int *__restrict__ arrival_counter,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const PhaseClocks clocks,
    const bool first_unit
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    fused_weight_tile (&weight)[kFusedStages] =
        staging.allocate<fused_weight_tile, kFusedStages>();
    mixed_scale_tile (&weight_scale)[kFusedStages][kFusedSlabScaleTiles] =
        staging.allocate<
            mixed_scale_tile, kFusedStages, kFusedSlabScaleTiles>();
    fused_activation_tile (&activation)[kFusedActivationSlabs] =
        staging.allocate<fused_activation_tile, kFusedActivationSlabs>();
    mixed_scale_tile
        (&activation_scale)[kFusedActivationSlabs][kFusedSlabScaleTiles] =
            staging.allocate<
                mixed_scale_tile, kFusedActivationSlabs,
                kFusedSlabScaleTiles>();

    // Its own array rather than an overlay: the epilogue runs with the next
    // task's two weight transfers in flight, so there is no dead region of the
    // ring to borrow.
    fused_result_tile (&result) = staging.allocate<fused_result_tile>();

    // Armed once per CTA and carried by parity. Forty-two stream indices over
    // two stages is not a whole number of laps, so a unit hands the next one a
    // barrier that is mid-phase, and re-arming per unit would mean depending on
    // `mbarrier.init` to reset a parity mid-launch -- which PTX defines only
    // for a barrier that has been invalidated first.
    __shared__ semaphore slab_arrived[kFusedStages];
    __shared__ semaphore slab_retired[kFusedStages];
    __shared__ unsigned int stream_parity[2];

    const int thread = static_cast<int>(threadIdx.x);
    if (first_unit) {
        if (thread < kFusedStages) {
            init_semaphore(slab_arrived[thread], 0, 1);
            init_semaphore(slab_retired[thread], 0, 1);
        }
        if (thread == 0) {
            stream_parity[0] = 0u;
            stream_parity[1] = 0u;
        }
    }
    __syncthreads();

    const fused_accumulator_tile accumulator =
        tensor_pool.allocate<fused_accumulator_tile>(0);
    const auto scale_slot = [&](const int set, const int slot) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kFusedScaleColumnBase
            + (set * kFusedScaleSlots + slot) * kRoutedScaleColumns);
    };

    // Warp 0's alone: it is the only warp that consumes these and the only
    // warp that writes them back. The whole-CTA gather below happens to put a
    // `__syncthreads` between this read and that write, so a read by the other
    // seven warps would be ordered rather than racing -- but the parity is warp
    // 0's state, so warp 0 is the only warp that reads it.
    unsigned int arrived_phase = 0u;
    unsigned int retired_phase = 0u;
    if (warpid() == 0) {
        arrived_phase = stream_parity[0];
        retired_phase = stream_parity[1];
    }
    const auto take_phase = [](unsigned int &bits, const int stage) {
        const int phase = static_cast<int>((bits >> stage) & 1u);
        bits ^= 1u << stage;
        return phase;
    };
    const auto issue = [&](const int index) {
        load_fused_slab(
            weight[index % kFusedStages],
            weight_scale[index % kFusedStages], packed_map, fused_scale,
            expert, index, slab_arrived[index % kFusedStages]);
    };

    for (int assignment_offset = 0; assignment_offset < batch_rows;
         assignment_offset += kFusedN) {
        const int batch_begin = assignment_begin + assignment_offset;
        const int rows = min(kFusedN, batch_rows - assignment_offset);
        const bool last_pass = assignment_offset + kFusedN >= batch_rows;

        // Issue the stream's first two transfers before the gather so they fly
        // underneath it. Nothing is in flight here: the previous pass consumed
        // all 42 of its indices.
        unsigned long long fine = clocks.now();
        if (thread == 0) {
            issue(0);
            issue(1);
        }
        fine = clocks.lap(kClockRoutedGateUpTmaIssue, fine);

        stage_fused_unit_activation(
            activation, activation_scale, scratch, batch_begin, rows);
        // Every thread wrote some of the tile above, so every thread owes the
        // asynchronous proxy a fence before the barrier that orders those
        // writes ahead of warp 0's `tcgen05` reads. One fence for the whole
        // unit, where a per-`(task, slab)` gather needed 42 of them.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        clocks.lap(kClockRoutedGateUpActivation, fine);

        // How far the stream has been retired. The stream retires one index
        // behind its issues, except at a task's last slab where the epilogue
        // needs that slab's own completion -- so the two places that retire
        // share one high-water mark rather than each assuming the other did
        // not run.
        int retired_upto = -1;

        for (int task = 0; task < kFusedTasks; ++task) {
            if (warpid() == 0) {
                const int lane = static_cast<int>(laneid());
                unsigned long long mark = clocks.now();
                unsigned long long inner = clocks.now();

                for (int slab = 0; slab < kFusedSlabs; ++slab) {
                    const int index = task * kFusedSlabs + slab;
                    const int stage = index % kFusedStages;
                    const int set = index % kFusedScaleSets;

                    wait(slab_arrived[stage], take_phase(arrived_phase, stage));
                    mark = clocks.lap(kClockRoutedGateUpStage, mark);
                    inner = clocks.lap(kClockRoutedGateUpTmaWait, inner);

                    if (lane == 0) {
                        #pragma unroll
                        for (int quad = 0; quad < kFusedSlabScaleTiles;
                             ++quad) {
                            auto staged_weight_scale = scale_slot(set, quad);
                            auto staged_activation_scale = scale_slot(
                                set, kFusedSlabScaleTiles + quad);
                            load_mxnv_scale_async(
                                staged_weight_scale, weight_scale[stage][quad]);
                            load_mxnv_scale_async(
                                staged_activation_scale,
                                activation_scale[slab][quad]);
                        }
                    }
                    tensor_store_wait();

                    if (lane == 0) {
                        st_descriptor<fused_weight_tile, transpose::N>
                            weight_desc(weight[stage]);
                        st_descriptor<fused_activation_tile, transpose::N>
                            activation_desc(activation[slab]);
                        #pragma unroll
                        for (int group = 0; group < kFusedSlabGroups; ++group) {
                            const int quad = group / kScaleGroupsPerTile;
                            const int factor = group % kScaleGroupsPerTile;
                            fused_mixed_mma(
                                accumulator,
                                weight_desc.chunk_descriptor(group),
                                activation_desc.chunk_descriptor(group),
                                scale_slot(set, quad),
                                scale_slot(
                                    set, kFusedSlabScaleTiles + quad),
                                factor,
                                slab != 0 || group != 0);
                        }
                        detail::tcgen05::commit<1>(slab_retired[stage]);
                    }
                    inner = clocks.lap(kClockRoutedGateUpMmaIssue, inner);

                    // Deferred retire, one index behind, refilling the stage it
                    // frees with the index after the one in flight. This runs
                    // straight through task boundaries: the stream does not
                    // restart per task, so the tensor core is fed across them.
                    if (index - 1 > retired_upto) {
                        const int retiring = (index - 1) % kFusedStages;
                        wait(slab_retired[retiring],
                             take_phase(retired_phase, retiring));
                        retired_upto = index - 1;
                        mark = clocks.lap(kClockRoutedGateUpMma, mark);
                        inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                        if (index + 1 < kFusedStreamLength && lane == 0) {
                            issue(index + 1);
                        }
                        inner = clocks.lap(kClockRoutedGateUpTmaIssue, inner);
                    }

                    // The epilogue reads tensor memory, so this task's last
                    // slab is the one index the deferred retire cannot cover.
                    // Refill past it first, so the next task's two transfers
                    // are in flight underneath the epilogue below.
                    if (slab == kFusedSlabs - 1 && index > retired_upto) {
                        wait(slab_retired[stage],
                             take_phase(retired_phase, stage));
                        retired_upto = index;
                        mark = clocks.lap(kClockRoutedGateUpMma, mark);
                        inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                        if (index + 2 < kFusedStreamLength && lane == 0) {
                            issue(index + 2);
                        }
                        inner = clocks.lap(kClockRoutedGateUpTmaIssue, inner);
                    }
                    mark = clocks.lap(kClockRoutedGateUpStage, mark);
                }
                if (lane == 0) {
                    stream_parity[0] = arrived_phase;
                    stream_parity[1] = retired_phase;
                }
            }
            __syncthreads();

            const unsigned long long epilogue = clocks.now();
            store_fused_accumulator(accumulator, result);
            __syncthreads();
            quantize_fused_situ(result, scratch, batch_begin, rows, task);
            __syncthreads();
            clocks.lap(kClockRoutedGateUpEpilogue, epilogue);

            // One arrival per completed 64-column range, and only on the pass
            // that finished it. A wide batch is several passes over the same
            // columns, so publishing per pass would let the count reach the
            // down phase's threshold while later rows of those columns were
            // still being written.
            if (last_pass) {
                persistent::publish_count_at(arrival_counter);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host side: the descriptor, and what the launch carries.
// ---------------------------------------------------------------------------

/// Build the `(task, slab)` weight-tile tensor map.
///
/// The five dimensions are, innermost first: the 128 U4 values one box row
/// carries, the tile's 128 M rows, the four boxes that make a 512-wide slab,
/// the 42 `(task, slab)` pairs of one expert, and the 896 experts. Splitting a
/// row into boxes of exactly one swizzle atom, and giving the boxes their own
/// dimension, is what makes the shared destination strip-major -- all 128 rows
/// of box 0, then all 128 rows of box 1 -- which is the layout a 128B-swizzled
/// ThunderKittens tile has and the layout `chunk_descriptor` walks.
///
/// Every requirement `16U4_ALIGN16B` adds is met by construction rather than by
/// luck, and each one is checked below, because a descriptor the driver accepts
/// but that describes the wrong layout fails as wrong numbers rather than as an
/// error.
static __host__ void create_fused_w13_packed_map(
    CUtensorMap *map,
    const void *base
) {
    const std::uint64_t global_dim[5] = {
        kFusedBoxElements,
        kFusedM,
        kFusedBoxes,
        kFusedTaskSlabs,
        kNumExperts,
    };
    // Byte distances between consecutive coordinates of dimensions one to four.
    const std::uint64_t global_stride[4] = {
        kFusedSlabRowBytes,
        kFusedBoxElements / 2,
        kFusedSlabGlobalBytes,
        static_cast<std::uint64_t>(kFusedTaskSlabs) * kFusedSlabGlobalBytes,
    };
    const std::uint32_t box_dim[5] = {
        kFusedBoxElements, kFusedM, kFusedBoxes, 1, 1,
    };
    const std::uint32_t element_stride[5] = {1, 1, 1, 1, 1};

    // `globalDim[0]` must be a multiple of 128 U4 values, `boxDim[0]` must be
    // exactly 128, the global address must be 32-byte aligned, and every stride
    // must be a multiple of 32 bytes. The `boxDim[0]` rule is what fixes the
    // transfer granularity at K = 128 and therefore the slab at a multiple of
    // it.
    static_assert(kFusedBoxElements == 128);
    static_assert(kFusedSlabK % kFusedBoxElements == 0);
    static_assert(kFusedSlabRowBytes % 32 == 0);
    static_assert((kFusedBoxElements / 2) % 32 == 0);
    static_assert(kFusedSlabGlobalBytes % 32 == 0);
    static_assert((static_cast<long long>(kFusedTaskSlabs)
                   * kFusedSlabGlobalBytes) % 32 == 0);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(base) % 32 == 0,
                "MoK: the fused W13 payload must be 32-byte aligned");

    const CUresult result = cuTensorMapEncodeTiled(
        map,
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        5,
        const_cast<void *>(base),
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    const char *error_string = nullptr;
    cuGetErrorString(result, &error_string);
    TORCH_CHECK(result == CUDA_SUCCESS,
                "MoK: cuTensorMapEncodeTiled rejected the fused W13 layout: ",
                error_string == nullptr ? "unknown" : error_string);
}

/// The descriptor for one prepared payload, encoded once per base address.
///
/// The descriptor travels into the kernel by value in a `__grid_constant__`
/// parameter, which is the only way a kernel may name one without a per-launch
/// device allocation to hold it. Encoding it, though, is a driver call, and a
/// decode step is tens of microseconds -- so it is encoded on the first launch
/// against a payload and read from here on every launch after.
///
/// Keyed by the payload's base address, which is sound because the address is
/// the only thing about the encoding that is not a compile-time constant of
/// this file: a second payload at an address a first one was freed from has
/// byte-for-byte the same descriptor, so a stale hit is indistinguishable from
/// a fresh encode. Nothing is ever evicted, and nothing needs to be -- weights
/// are prepared once per model load, so this map holds one entry per rank per
/// process for the process's whole life.
struct PackedMapCache {
    std::mutex guard;
    std::map<const void *, CUtensorMap> encoded;
    std::atomic<int> encodes{0};
};

static __host__ PackedMapCache &packed_map_cache() {
    static PackedMapCache cache;
    return cache;
}

static __host__ const CUtensorMap *fused_w13_packed_map(const void *base) {
    PackedMapCache &cache = packed_map_cache();
    const std::lock_guard<std::mutex> held(cache.guard);
    const auto found = cache.encoded.find(base);
    if (found != cache.encoded.end()) return &found->second;
    // A `std::map` node's value keeps its address across every later insert,
    // so the pointer handed out here stays valid for the process's life.
    CUtensorMap &map = cache.encoded[base];
    create_fused_w13_packed_map(&map, base);
    cache.encodes.fetch_add(1, std::memory_order_relaxed);
    return &map;
}

/// How many descriptors this process has encoded, for the launch-overhead test.
///
/// The number the graph-capture and steady-state gates care about is one per
/// payload for the life of the process, not one per launch, and a cache that
/// missed every time would be invisible in the decode output.
inline std::int64_t fused_w13_packed_maps_encoded_for_testing() {
    return packed_map_cache().encodes.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// The layout probe.
//
// Two things about the transfer above are properties of the hardware rather
// than of this file: how many bytes the mbarrier counts when the format widens
// its payload 2x, and whether the widened bytes land where a 128B-swizzled tile
// keeps them. Both are invisible in the decode step's numbers -- a wrong
// transaction count hangs, a wrong layout silently contracts the wrong
// weights -- so they are measured on their own, by one CTA, against one
// `(task, slab)` tile.
//
// The wait is bounded. A transaction count larger than the transfer would
// otherwise hang the device; here it reports zero and the caller sees which
// count is right rather than a timeout.
//
// `tests/test_kimi_k3_w13_layout.py` is the caller, and it is the only reason
// this probe is compiled: it runs the descriptor production launches with,
// over the full-width prepared payload, and checks the transaction count, all
// five dimensions, and that not one byte lands in the format's padding.
// ---------------------------------------------------------------------------

/// Roughly a hundred milliseconds of B300 clocks.
inline constexpr unsigned long long kFusedProbeTimeoutCycles = 200000000ull;

/// Test one mbarrier phase without blocking on it.
///
/// The same instruction `kittens::wait` spins on, spelled once so the probe can
/// give up instead of hanging the device when the transaction count is wrong.
__device__ __forceinline__ bool fused_probe_try_wait(
    kittens::semaphore &bar,
    const int phase
) {
    const std::uint32_t address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(&bar));
    std::uint32_t ready;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n"
        : "=r"(ready)
        : "r"(address), "r"(phase)
        : "memory"
    );
    return ready != 0u;
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void fused_w13_tma_probe_kernel(
    const __grid_constant__ CUtensorMap packed,
    std::uint8_t *__restrict__ dump,
    int *__restrict__ completed,
    const int expert,
    const int task_slab,
    const int transaction_bytes
) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];
    tma_swizzle_allocator staging(shared_raw);
    fused_weight_tile (&payload) = staging.allocate<fused_weight_tile>();

    __shared__ semaphore arrived;
    __shared__ int arrived_flag;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) {
        init_semaphore(arrived, 0, 1);
        arrived_flag = 0;
    }
    // The tile is cleared so a partial transfer is visible as zeros rather than
    // as whatever the block last held.
    for (int index = thread; index < sizeof(fused_weight_tile) / 16;
         index += kDecodeCtaThreads) {
        reinterpret_cast<uint4 *>(payload.data)[index] =
            make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();

    if (thread == 0) {
        tma::expect_bytes(
            arrived, static_cast<std::uint32_t>(transaction_bytes));
        load_fused_slab_async(payload, &packed, expert, task_slab, arrived);
        const unsigned long long start = clock64();
        bool ready = false;
        while (!(ready = fused_probe_try_wait(arrived, 0))) {
            if (static_cast<unsigned long long>(clock64()) - start
                    > kFusedProbeTimeoutCycles) {
                break;
            }
        }
        arrived_flag = ready ? 1 : 0;
        *completed = arrived_flag;
    }
    __syncthreads();

    if (arrived_flag != 0) {
        // Read out through the tile's own `(row, column)` indexing rather than
        // as a flat span. That is the addressing `chunk_descriptor` and the
        // MMA use, so a dump that agrees with the transform proves the two
        // sides of the swizzle agree -- which a flat copy could not, because it
        // would only prove that some permutation of the right bytes arrived.
        for (int index = thread; index < kFusedM * kFusedSlabK;
             index += kDecodeCtaThreads) {
            const int row = index / kFusedSlabK;
            const int column = index % kFusedSlabK;
            dump[index] = *reinterpret_cast<const std::uint8_t *>(
                &payload[{row, column}]);
        }
    }
}

/// TEST-ONLY: run one fused weight transfer and report what landed.
///
/// Returns one `(task, slab)` tile as a row-major `[128, 512]` byte image, read
/// out of shared memory at the logical `(row, column)` the MMA's chunk
/// descriptors address, and a flag saying whether `transaction_bytes` completed
/// the mbarrier. The caller reconstructs the expected image from the transform,
/// so a descriptor that lies about its layout fails here rather than as wrong
/// decode numbers.
static __host__ std::tuple<at::Tensor, std::int64_t>
kimi_k3_fused_w13_tma_probe_entrypoint(
    const at::Tensor &expert_w13_packed,
    std::int64_t expert,
    std::int64_t task_slab,
    std::int64_t transaction_bytes
) {
    CHECK_INPUT(expert_w13_packed);
    TORCH_CHECK(expert_w13_packed.dim() == 3
                    && expert_w13_packed.size(0) == kNumExperts
                    && expert_w13_packed.size(1) == kFusedPackedRows
                    && expert_w13_packed.size(2) == kFusedPackedColumns
                    && expert_w13_packed.scalar_type() == at::kByte,
                "MoK: _kimi_k3_fused_w13_tma_probe requires uint8 "
                "expert_w13_packed [", kNumExperts, ", ", kFusedPackedRows,
                ", ", kFusedPackedColumns, "]");
    TORCH_CHECK(expert >= 0 && expert < kNumExperts,
                "MoK: _kimi_k3_fused_w13_tma_probe requires expert in [0, ",
                kNumExperts, ")");
    TORCH_CHECK(task_slab >= 0 && task_slab < kFusedTaskSlabs,
                "MoK: _kimi_k3_fused_w13_tma_probe requires task_slab in [0, ",
                kFusedTaskSlabs, ")");
    TORCH_CHECK(transaction_bytes > 0
                    && transaction_bytes
                           <= static_cast<std::int64_t>(
                                  sizeof(fused_weight_tile)),
                "MoK: _kimi_k3_fused_w13_tma_probe requires transaction_bytes "
                "in (0, ", sizeof(fused_weight_tile), "]");

    const c10::cuda::CUDAGuard device_guard(expert_w13_packed.device());
    at::Tensor dump = at::zeros(
        {static_cast<std::int64_t>(sizeof(fused_weight_tile))},
        expert_w13_packed.options());
    at::Tensor completed =
        at::zeros({1}, expert_w13_packed.options().dtype(at::kInt));

    CUtensorMap packed;
    create_fused_w13_packed_map(&packed, expert_w13_packed.data_ptr());

    constexpr int shared_bytes = static_cast<int>(sizeof(fused_weight_tile))
                               + 1024;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fused_w13_tma_probe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes));
    fused_w13_tma_probe_kernel
        <<<1, kDecodeCtaThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
            packed,
            reinterpret_cast<std::uint8_t *>(dump.data_ptr()),
            reinterpret_cast<int *>(completed.data_ptr()),
            static_cast<int>(expert),
            static_cast<int>(task_slab),
            static_cast<int>(transaction_bytes));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dump, static_cast<std::int64_t>(completed.cpu().item<int>())};
}

// ---------------------------------------------------------------------------
// The shared-footprint probe.
//
// How far past the dynamic block's first byte the ring's last array ends is a
// property of where the driver puts that first byte, which follows the static
// shared memory ptxas assigned and is therefore neither 1 KiB aligned nor
// knowable from this file. A ring sized to the byte overruns its block by
// exactly that offset, and the overrun is invisible in the arithmetic -- the
// bytes past the end belong to no one, so they read back whatever they held.
//
// So the offset is measured. This kernel runs the same allocator sequence the
// engine does, under the same launch configuration, and reports where the
// sequence started and ended without writing a single byte of it.
// ---------------------------------------------------------------------------

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void fused_w13_shared_footprint_kernel(int *__restrict__ report) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];
    tma_swizzle_allocator staging(shared_raw);
    staging.allocate<fused_weight_tile, kFusedStages>();
    staging.allocate<mixed_scale_tile, kFusedStages, kFusedSlabScaleTiles>();
    staging.allocate<fused_activation_tile, kFusedActivationSlabs>();
    staging.allocate<
        mixed_scale_tile, kFusedActivationSlabs, kFusedSlabScaleTiles>();
    staging.allocate<fused_result_tile>();
    if (threadIdx.x == 0) {
        const std::uint32_t base = static_cast<std::uint32_t>(
            __cvta_generic_to_shared(shared_raw));
        const std::uint32_t end = static_cast<std::uint32_t>(
            __cvta_generic_to_shared(staging.ptr));
        report[0] = static_cast<int>(end - base);
        report[1] = static_cast<int>(base % kFusedAllocatorPadding);
    }
}

/// TEST-ONLY: how many dynamic shared bytes the ring really needs.
///
/// Returns the bytes the allocator consumes measured from the dynamic block's
/// own first byte, that block's offset within the 1 KiB grid the allocator
/// aligns to, and the bytes the fused instantiation launches with. The first
/// must not exceed the third.
static __host__ std::tuple<std::int64_t, std::int64_t, std::int64_t>
kimi_k3_fused_w13_shared_footprint_entrypoint() {
    at::Tensor report = at::zeros(
        {2}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fused_w13_shared_footprint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kFusedW13SharedBytes));
    fused_w13_shared_footprint_kernel
        <<<1, kDecodeCtaThreads, kFusedW13SharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<int *>(report.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const at::Tensor host = report.cpu();
    return {static_cast<std::int64_t>(host[0].item<int>()),
            static_cast<std::int64_t>(host[1].item<int>()),
            kFusedW13SharedBytes};
}

/// Every number the engine's shape rests on, for the tests that check them.
///
/// Reported rather than recomputed in Python because the whole point of the
/// shared-byte accounting is that it is the accounting the kernel launches
/// with: a test that rebuilt the arithmetic from the same reasoning the header
/// uses would agree with the header and not with the device.
inline std::map<std::string, std::int64_t>
fused_w13_geometry_for_testing() {
    return {
        {"tasks", kFusedTasks},
        {"slabs", kFusedSlabs},
        {"slab_k", kFusedSlabK},
        {"slab_groups", kFusedSlabGroups},
        {"slab_scale_tiles", kFusedSlabScaleTiles},
        {"task_slabs", kFusedTaskSlabs},
        {"m", kFusedM},
        {"n", kFusedN},
        {"physical_n", kFusedPhysicalN},
        {"half_rows", kFusedHalfRows},
        {"boxes", kFusedBoxes},
        {"box_elements", kFusedBoxElements},
        {"swizzle_bytes", fused_weight_tile::swizzle_bytes},
        {"packed_rows", kFusedPackedRows},
        {"packed_columns", kFusedPackedColumns},
        {"scale_rows", kFusedScaleRows},
        {"scale_columns", kFusedScaleColumns},
        {"stages", kFusedStages},
        {"activation_slabs", kFusedActivationSlabs},
        {"stream_length", kFusedStreamLength},
        {"weight_tile_bytes",
         static_cast<std::int64_t>(sizeof(fused_weight_tile))},
        {"activation_tile_bytes",
         static_cast<std::int64_t>(sizeof(fused_activation_tile))},
        {"result_tile_bytes",
         static_cast<std::int64_t>(sizeof(fused_result_tile))},
        {"weight_transaction_bytes", kFusedWeightTransactionBytes},
        {"slab_transaction_bytes", kFusedSlabTransactionBytes},
        {"scale_slots", kFusedScaleSlots},
        {"scale_sets", kFusedScaleSets},
        {"staging_bytes", kFusedStagingBytes},
        {"allocator_padding", kFusedAllocatorPadding},
        {"static_shared_reserve", kFusedStaticSharedReserve},
        {"shared_bytes", kFusedW13SharedBytes},
        {"opt_in_maximum", kittens::MAX_SHARED_MEMORY},
    };
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
