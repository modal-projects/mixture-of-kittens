#pragma once

/// The fused-W13 packed layout, and the tiles that hold it.
///
/// Geometry only: what the prepared bytes are, how a `(task, slab)` pair
/// indexes them, which shared tiles the ring stages them into, and the
/// shared-memory ledger the resident ring launches with. Mirrors
/// `mok/kimi_k3_w13.py`, which builds those bytes on the host.
///
/// Nothing here contracts, transfers, or launches anything. Every other part of
/// this engine is written against these constants, which is why they are the
/// one part with no dependency of its own.

#include "../expert_mxfp4.cuh"
#include "../expert_mxfp4_grouped.cuh"
// The unit publishes its six completed column ranges from inside itself, so it
// needs the same arrival primitive the persistent kernel publishes with rather
// than a second copy of it.
#include "../persistent_sync.cuh"

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

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
