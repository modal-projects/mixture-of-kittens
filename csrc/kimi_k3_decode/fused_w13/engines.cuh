#pragma once

/// The engine ids, the three-stage ring's geometry, and its slab staging.
///
/// Why there is more than one engine at all: the resident ring is 40% of its
/// gate/up band in the ring itself, and a third K = 512 stage wants 67,584 bytes
/// that only the resident activation has. Both rings below buy that stage out of
/// the activation and differ in what they give up for it, which is a
/// measurement rather than an argument -- so both are compiled, both are
/// measured, and production is the hybrid the measurement argued for.
///
/// What lives here is what the two three-stage rings share: the ids and the
/// predicates over them, the ring depth, the accumulator band, the stream order,
/// and the per-slab activation staging the slab-buffered ring gathers with.

#include "resident_unit.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

// ---------------------------------------------------------------------------
// V4: the same six tasks and seven slabs, with the activation gathered a slab
// at a time so a third K = 512 weight stage fits.
//
// The unit above is 40.0% ring at every shape measured -- `tma_issue` plus
// `tma_wait` plus `ring_full` of the gate/up band, against 42.7% in the
// contraction -- and the reason it cannot answer that with depth is arithmetic
// rather than design. A third K = 512 stage wants 67,584 bytes; the launch
// leaves 14,336 above the static shared memory ptxas assigns. The 71,680 bytes
// the resident activation holds is the only place that money is.
//
// So the activation cannot stay resident. The gather becomes per slab, and the
// only question left is which loop is outer. Both answers are compiled here,
// because the difference between them is a measurement rather than an argument.
//
//   * **Task outer, slab inner** -- the shape the resident engine already has.
//     Slab `s`'s eight rows are needed again at every one of the six tasks, so
//     an expert re-gathers 42 times instead of seven. One accumulator lives,
//     and each task's epilogue runs with the next task's transfers in flight,
//     which is the cover the resident engine is built around.
//   * **Slab outer, task inner** -- each slab is gathered once, because the six
//     tasks that read it are adjacent. Six accumulators live, which is 96 of
//     the 512 tensor-memory columns against scale buffers that start at 256, so
//     the depth costs nothing. The six epilogues move to the end of the stream,
//     where nothing flies underneath them.
//
// The gather is not a transfer, and section 44 of the Task 11b report is the
// exact reason: a 128B-swizzled K-major operand interleaves its rows *inside*
// 128-byte atom columns whose stride is the tile's row count times 128, so one
// row of one slab is four discontiguous 128-byte writes and `boxDim` cannot
// describe them as one transfer. Nor can a per-row transfer land at the right
// swizzle phase, because TMA derives the phase from the destination base it is
// handed and a row past the first needs a phase that base cannot express. So
// warps 1 to 7 gather while warp 0 contracts: the copy engine's *role* -- work
// that overlaps the ring instead of preceding it -- without the copy engine.
//
// What both orders preserve: one expert-pure CTA claim, six 128-row tasks
// pairing gate row `r` with up row `r + 64`, seven K = 512 slabs, one large
// weight transfer and one contiguous scale transfer per slab, `tcgen05` stage
// releases through the MMA's own completion, the deferred retire, the exact
// `situ` expression, and the six published column ranges.
//
// What both change: the ring is three deep and prefetches two indices ahead
// instead of one, and the activation lives in two eight-row slots instead of
// seven sixteen-row tiles.
//
// **Two slots is what makes the loop order matter, and it is a ceiling rather
// than a choice.** A slot may be refilled only once every reader of what it
// last held has retired, and the producers of epoch `e` write the slot epoch
// `e - 1` read. So an epoch has to be *fully* retired by its own end, which
// costs one drain of the tensor core per epoch on top of the deferred retire.
// Slab-major has seven epochs per expert and pays seven. A third slot would
// let the retire stay one epoch behind and remove the drain entirely -- and it
// does not fit, by `kFusedV4ThreeSlotBytes` below.
//
// Section 52 of the Task 11b report records the loop order that lost: the same
// unit with the task outer and the slab inner gathers 42 times instead of
// seven and drains 42 times instead of seven, and it was slower than this one
// at every shape measured. It is not compiled any more.
// ---------------------------------------------------------------------------

/// Production's engine, and the one baseline kept beside it.
///
/// Production is `kEngineFusedAdaptive`, and it is not one ring: it is a
/// selector inside the expert unit that takes the compact ring when the
/// expert's batch is narrow and the slab-buffered ring when it is not. Which
/// one an expert takes is a property of that expert's row count, so the choice
/// is made on the device per unit rather than on the host per launch.
///
/// `kEngineFusedResident` is the two-stage ring production replaced. It is the
/// only other engine compiled, it is reachable only behind the grid-tuning
/// guard, and it is kept for one reason: every route the suite runs is checked
/// byte for byte against it, and the A/B that argued for the selector is
/// re-runnable against it. The two rings the selector is built from were
/// separately selectable while they were being measured against each other and
/// are not any more -- production exercises both, so a standalone id for either
/// would be a second way to reach code that already runs.
inline constexpr int kEngineFusedAdaptive = 2;
inline constexpr int kEngineFusedResident = 3;

__host__ __device__ __forceinline__ constexpr bool engine_is_known(
    const int engine
) {
    return engine == kEngineFusedAdaptive || engine == kEngineFusedResident;
}

/// Whether an engine id names the production selector.
///
/// Which is also the answer to "does this launch have to hold the compact
/// ring's bytes and the slab-buffered ring's accumulators": the selector can
/// take either ring at any unit, so it is granted the wider of the two rather
/// than whichever ring its first expert happens to want.
__host__ __device__ __forceinline__ constexpr bool engine_is_adaptive(
    const int engine
) {
    return engine == kEngineFusedAdaptive;
}

/// Whether an engine id names the resident two-stage ring.
__host__ __device__ __forceinline__ constexpr bool engine_is_resident(
    const int engine
) {
    return engine == kEngineFusedResident;
}

/// V4's ring depth and activation slots, which both loop orders share.
inline constexpr int kFusedV4Stages = 3;
inline constexpr int kFusedV4Slots = 2;

/// Accumulators one engine keeps live at its widest.
///
/// The slab-buffered ring interleaves the six tasks inside one slab, so all six
/// are open at once, and the production selector inherits that because it can
/// take that ring. The resident ring finishes a task before it starts the next,
/// so one is.
__host__ __device__ __forceinline__ constexpr int fused_v4_accumulators(
    const int engine
) {
    return engine_is_adaptive(engine) ? kFusedTasks : 1;
}

inline constexpr int kFusedV4Accumulators =
    fused_v4_accumulators(kEngineFusedAdaptive);

/// Bytes one 128B swizzle atom occupies, and one eight-row slot's offset.
///
/// A slot is not a tile. `st_fp8e4m3<8, 512>` does not exist: a swizzled shared
/// tile's row count must be a multiple of `TILE_ROW_DIM` -- sixteen for every
/// type -- and `chunk_descriptor` divides by it, so an eight-row tile would
/// compute an atom-column stride of zero. The sixteen-row tile the resident
/// engine already uses is therefore *two* eight-row operands, at byte offsets 0
/// and 1,024, and the second is reached by adding that offset to the first's
/// descriptor. That is sound rather than lucky: 1,024 is exactly the swizzle
/// period `((address % 1024) >> 7) << 4` repeats on, so both slots see the same
/// XOR for the same slot-relative row, and both a `[{row, column}]` store and
/// the tensor core's own read of the descriptor agree about where a byte is.
inline constexpr int kFusedSwizzleAtomBytes = 128;
inline constexpr int kFusedV4SlotBytes = kFusedN * kFusedSwizzleAtomBytes;

/// Threads warp 0 occupies, which is where the producers' worker range starts.
inline constexpr int kFusedV4ProducerBase = 32;

static_assert(kFusedV4Slots * kFusedN == kFusedPhysicalN,
              "the two slots must be the halves of one physical tile");
static_assert(kFusedV4SlotBytes == 1024);
static_assert(kFusedV4SlotBytes
                  == fused_activation_tile::swizzle_bytes * kFusedN,
              "a slot must be a whole number of swizzle periods, or the two "
              "halves would not share one XOR");
static_assert(kFusedPhysicalN % kittens::TILE_ROW_DIM<kittens::fp8e4m3> == 0,
              "the physical tile is what TILE_ROW_DIM admits; the slots are "
              "descriptor offsets inside it");

/// Shared bytes one V4 unit occupies.
///
///   3 x 65,536  weight slabs             = 196,608
///   3 x  2,048  weight scale quads       =   6,144
///   1 x  8,192  activation tile, 2 slots =   8,192
///   2 x  2,048  activation scale quads   =   4,096
///   1 x  8,192  epilogue result tile     =   8,192
///                                          -------
///                                          223,232
///
/// Which is 8,192 more than the resident engine's ring and 8,192 less than the
/// ceiling would allow, so the slot arithmetic above is not a nicety: two
/// sixteen-row activation tiles instead of two slots of one is 231,424 bytes of
/// ring, and the launch would have to grant more than the opt-in maximum leaves
/// once ptxas takes its static shared memory.
inline constexpr int kFusedV4StagingBytes =
    kFusedV4Stages * static_cast<int>(sizeof(fused_weight_tile))
    + kFusedV4Stages * kFusedSlabScaleTiles
          * static_cast<int>(sizeof(mixed_scale_tile))
    + static_cast<int>(sizeof(fused_activation_tile))
    + kFusedV4Slots * kFusedSlabScaleTiles
          * static_cast<int>(sizeof(mixed_scale_tile))
    + static_cast<int>(sizeof(fused_result_tile));

inline constexpr int kFusedV4SharedBytes =
    kFusedV4StagingBytes + kFusedAllocatorPadding;

static_assert(kFusedV4StagingBytes == 223232);
static_assert(kFusedV4SharedBytes == 224256);
static_assert(kFusedV4SharedBytes
                  <= kittens::MAX_SHARED_MEMORY - kFusedStaticSharedReserve,
              "V4's ring must leave room for static shared memory");
static_assert(2 * kFusedV4SharedBytes > kittens::MAX_SHARED_MEMORY,
              "V4's grid must still be one CTA per SM");
static_assert(kFusedV4SharedBytes > kFusedW13SharedBytes,
              "V4 buys a third stage, so it must ask for more than V2");
static_assert(kFusedV4StagingBytes
                  == kFusedStagingBytes
                         + static_cast<int>(sizeof(fused_weight_tile))
                         + kFusedSlabScaleTiles
                               * static_cast<int>(sizeof(mixed_scale_tile))
                         - (kFusedActivationSlabs - 1)
                               * static_cast<int>(
                                     sizeof(fused_activation_tile))
                         - (kFusedActivationSlabs - kFusedV4Slots)
                               * kFusedSlabScaleTiles
                               * static_cast<int>(sizeof(mixed_scale_tile)),
              "the third stage is paid for by the activation V4 stops holding");

/// What a third activation slot would cost, and why the drain is unavoidable.
///
/// Two slots force an epoch to be fully retired by its own end, because the
/// producers of the next epoch write the slot this one is reading. A third slot
/// would let the retire stay a whole epoch behind -- the deferred retire would
/// then be enough on its own and the per-epoch drain would go away, which is
/// worth more the shorter the epoch is.
///
/// It does not fit, and not by a rounding. A slot is eight of a tile's sixteen
/// rows, so three slots is two whole `st_fp8e4m3<16, 512>` tiles -- 16,384
/// bytes, not 12,288 -- because `TILE_ROW_DIM` admits no shorter swizzled tile
/// to hold the odd slot in. With its scale quad that is 22,528 bytes of
/// activation against two slots' 12,288, and the ring is already within 8,192
/// of what the launch may ask for.
inline constexpr int kFusedV4ThreeSlots = 3;
inline constexpr int kFusedV4ThreeSlotTiles =
    (kFusedV4ThreeSlots * kFusedN + kFusedPhysicalN - 1) / kFusedPhysicalN;
inline constexpr int kFusedV4ThreeSlotBytes =
    kFusedV4StagingBytes
    + (kFusedV4ThreeSlotTiles - 1)
          * static_cast<int>(sizeof(fused_activation_tile))
    + (kFusedV4ThreeSlots - kFusedV4Slots) * kFusedSlabScaleTiles
          * static_cast<int>(sizeof(mixed_scale_tile));

static_assert(kFusedV4ThreeSlotTiles == 2);
static_assert(kFusedV4ThreeSlotBytes == 233472);
static_assert(kFusedV4ThreeSlotBytes + kFusedAllocatorPadding
                  > kittens::MAX_SHARED_MEMORY - kFusedStaticSharedReserve,
              "a third activation slot must not fit, or the per-epoch drain "
              "below would be a choice rather than a consequence");

/// The accumulators fit under the band the scale buffers start at.
static_assert(kFusedV4Accumulators * kFusedPhysicalN <= kFusedScaleColumnBase,
              "six live accumulators must not reach the scale buffers");
static_assert(kFusedV4Accumulators == 6);
static_assert(fused_v4_accumulators(kEngineFusedAdaptive) == 6,
              "the production selector can take the slab-buffered ring, so it "
              "reserves that ring's accumulator band");
static_assert(fused_v4_accumulators(kEngineFusedResident) == 1);
static_assert(kFusedStreamLength % kFusedV4Stages == 0,
              "42 indices over three stages is fourteen whole laps, so a "
              "pass hands the next one a barrier at the parity it found");

/// An epoch is the run of stream indices that share one activation slab.
///
/// Six, because the six tasks that read a slab are adjacent in this order.
/// Everything else about the order follows from this number: the gathers per
/// expert are the epoch count, and so are the drains.
inline constexpr int kFusedV4EpochLength = kFusedTasks;
inline constexpr int kFusedV4Epochs =
    kFusedStreamLength / kFusedV4EpochLength;

static_assert(kFusedV4Epochs == kFusedSlabs);

/// Where stream index `i` reads its weights from.
///
/// The transform's outer axis is `task * 7 + slab` and this order does not
/// reorder it: it reads the same prepared bytes the resident engine does. The
/// six transfers per slab are 32 KiB each and 229,376 bytes apart, which costs
/// nothing at this granularity because each is one contiguous run far longer
/// than a DRAM burst.
__host__ __device__ __forceinline__ constexpr int fused_v4_task_of(
    const int index
) {
    return index % kFusedTasks;
}

__host__ __device__ __forceinline__ constexpr int fused_v4_slab_of(
    const int index
) {
    return index / kFusedTasks;
}

__host__ __device__ __forceinline__ constexpr int fused_v4_task_slab_of(
    const int index
) {
    return fused_v4_task_of(index) * kFusedSlabs + fused_v4_slab_of(index);
}

// The order visits `(task, slab)` pair 0, then 7, then 14 -- the six tasks of
// slab 0.
static_assert(fused_v4_task_slab_of(0) == 0);
static_assert(fused_v4_task_slab_of(1) == 7);
static_assert(fused_v4_task_slab_of(6) == 1);
static_assert(fused_v4_task_slab_of(kFusedStreamLength - 1)
                  == kFusedStreamLength - 1);
// The order is a permutation of the same 42 indices, and an epoch is one slab.
static_assert([] {
    bool seen[kFusedStreamLength] = {};
    for (int index = 0; index < kFusedStreamLength; ++index) {
        const int pair = fused_v4_task_slab_of(index);
        if (pair < 0 || pair >= kFusedStreamLength || seen[pair]) {
            return false;
        }
        seen[pair] = true;
        const int epoch = index / kFusedV4EpochLength;
        if (fused_v4_slab_of(index)
            != fused_v4_slab_of(epoch * kFusedV4EpochLength)) {
            return false;
        }
    }
    return true;
}(), "the order must visit all 42 pairs once, one slab per epoch");

/// The descriptor offset that turns slot 0's operand into slot 1's.
///
/// `chunk_descriptor` adds its offsets into the descriptor's own address field,
/// which holds bits 4 to 18 of the operand's shared address. A slot offset of
/// 1,024 cannot carry out of that field because the whole opt-in shared block
/// is 232,448 bytes and the field spans 262,144, so adding the encoded offset
/// is the same descriptor a shifted base would have produced.
__device__ __forceinline__ std::uint64_t fused_v4_slot_offset(const int slot) {
    return kittens::detail::matrix_descriptor_encode(
        static_cast<std::uint64_t>(slot) * kFusedV4SlotBytes);
}

/// Gather one slab's live activation rows and scales into one eight-row slot.
///
/// The same reads the resident engine's whole-unit gather makes, restricted to
/// one slab and one slot, and made by a caller-chosen set of workers so warp 0
/// can stay on the ring while the other seven warps do this. 256 sixteen-byte
/// atoms is one store per thread when the whole CTA runs it and two for a
/// quarter of them when seven warps do.
///
/// Rows past the batch read as zero against a unit scale for the same reason
/// they do above: the MMA always contracts eight N columns and `0xff` is
/// E8M0's NaN, which would poison its own accumulator column.
///
/// The caller owes the asynchronous proxy a fence on every worker and a CTA
/// barrier before the tensor core reads any of this.
__device__ __forceinline__ void stage_fused_slab_activation(
    fused_activation_tile &payload,
    mixed_scale_tile (&scales)[kFusedSlabScaleTiles],
    const Scratch &scratch,
    const int assignment_begin,
    const int rows,
    const int slab,
    const int slot,
    const int worker_base
) {
    constexpr int atoms_per_row = kFusedSlabK / 16;
    constexpr int atoms = kFusedN * atoms_per_row;
    constexpr int quads = kFusedN * kFusedSlabScaleTiles;

    const int worker = static_cast<int>(threadIdx.x) - worker_base;
    const int workers = kDecodeCtaThreads - worker_base;
    const int row_base = slot * kFusedN;

    for (int index = worker; index < atoms; index += workers) {
        const int row = index / atoms_per_row;
        const int atom = index % atoms_per_row;
        if (row < rows) {
            const int token =
                scratch.assignment_tokens[assignment_begin + row];
            *fused_activation_atom(payload, row_base + row, atom) =
                *reinterpret_cast<const uint4 *>(
                    scratch.latent_mxfp8
                    + static_cast<long long>(token) * kLatentSize
                    + slab * kFusedSlabK + atom * 16);
        } else {
            *fused_activation_atom(payload, row_base + row, atom) =
                make_uint4(0u, 0u, 0u, 0u);
        }
    }
    for (int index = worker; index < quads; index += workers) {
        const int row = index / kFusedSlabScaleTiles;
        const int quad = index % kFusedSlabScaleTiles;
        std::uint32_t word = 0x7f7f7f7fu;
        if (row < rows) {
            const int token =
                scratch.assignment_tokens[assignment_begin + row];
            word = *reinterpret_cast<const std::uint32_t *>(
                scratch.latent_scale
                + static_cast<long long>(token) * kLatentGroups
                + slab * kFusedSlabGroups + quad * kScaleGroupsPerTile);
        }
        stage_scale_quad(scales[quad], row, word);
    }
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
