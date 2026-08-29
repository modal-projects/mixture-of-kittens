#pragma once

// Getting one routed unit's operands into shared memory and its results back
// out.
//
// A unit contracts one 128-column output tile of one expert's assignment
// batch, and everything between the packed global bytes and the accumulator
// readout lives here: the tile geometry a unit is decomposed into, the
// sixteen-byte staging of activations, weights, and scale quads, the MXFP8
// quantization of the routed latent and of the SiTU intermediate, and the
// weighted scatter of a down tile into the routed accumulator. The units
// themselves are in `expert_mxfp4.cuh`.

#include "expert_mxfp4_mma.cuh"

#include <cstddef>
#include <cstdint>

namespace kimi_k3_decode {
namespace expert_mxfp4 {

inline constexpr int kGateUpTiles =
    kRoutedIntermediateSizePerRank / kMmaN;
inline constexpr int kDownTiles = kLatentSize / kMmaN;
inline constexpr int kLatentGroups = kLatentSize / kMmaK;
inline constexpr int kSituGroups =
    kRoutedIntermediateSizePerRank / kMmaK;

static_assert(kGateUpTiles == 3);
static_assert(kDownTiles == 28);
static_assert(kLatentGroups == 112);
static_assert(kSituGroups == 12);

// A TMA tile gathers one packed K32 group from 128 strided weight rows into a
// compact shared buffer. Threads then expand each row into the two 16-byte
// atoms the mixed MMA consumes. Keeping the compact tile non-swizzled makes
// its 16-byte rows naturally aligned and lets the copy engine own the global
// row stride without imposing that stride on the MMA tile.
using packed_weight_tile =
    kittens::st_uint8<kMmaN, kMmaK / 2, false>;
using packed_weight_layout =
    kittens::gl<kittens::uint8, kNumExperts, 1, -1, -1,
                packed_weight_tile>;

struct RoutedLayouts {
    packed_weight_layout w1;
    packed_weight_layout w3;
    packed_weight_layout w2;
};

static __host__ inline packed_weight_layout packed_weight_layout_for(
    const std::uint8_t *pointer,
    const int rows,
    const int columns
) {
    return packed_weight_layout{
        const_cast<kittens::uint8 *>(
            reinterpret_cast<const kittens::uint8 *>(pointer)),
        nullptr, nullptr, static_cast<std::size_t>(rows),
        static_cast<std::size_t>(columns)};
}

static __host__ inline RoutedLayouts routed_layouts(
    const std::uint8_t *w1,
    const std::uint8_t *w3,
    const std::uint8_t *w2
) {
    return RoutedLayouts{
        packed_weight_layout_for(
            w1, kExpertW1W3PackedRows, kExpertW1W3PackedColumns),
        packed_weight_layout_for(
            w3, kExpertW1W3PackedRows, kExpertW1W3PackedColumns),
        packed_weight_layout_for(
            w2, kExpertW2PackedRows, kExpertW2PackedColumns),
    };
}

__device__ __forceinline__ float decode_route_weight(
    const Scratch &scratch,
    const int assignment
) {
    const int token = scratch.assignment_tokens[assignment];
    const int slot = scratch.assignment_slots[assignment];
    return scratch.expert_weights[token * kTopK + slot];
}

// ---------------------------------------------------------------------------
// Operand staging.
//
// Every byte a routed unit stages moves in sixteen-byte vectors. A shared tile
// row is two sixteen-byte swizzle atoms and the swizzle only ever permutes
// whole atoms, so an atom is contiguous and aligned however the tile is
// placed, and a whole atom is one store. The measured decode profile put
// 76.5% of all cycles in the byte-at-a-time staging these replace.
//
// The staging that remains is the largest region of the decode step, 29.4% of
// it at M16, and the serving-backend comparison measured two attempts to cut
// it. Both lost, and the reasons are worth keeping here, because both are the
// obvious thing to try.
//
// The first attempt addressed the instruction path: converting every staging
// store to `st.shared` so it compiles to `STS` instead of a generic `ST.E`,
// and hoisting the cross-proxy fence out of `mixed_mma` so a gate/up round
// pays one `MEMBAR.ALL.CTA` rather than sixteen. It does what it says --
// `ST.E` 62 to 0, `MEMBAR` 155 to 127, `LDG` 230 to 294 as the freed prefetch
// reaches the scheduler -- and it changed the median by less than the run to
// run spread on all sixteen measured shapes, worst case 0.84% slower at M8
// where the freed scheduling spilled sixteen bytes. The region is not bound by
// the instructions it issues.
//
// The second attempt addressed the access pattern. A thread owning a whole
// weight row reads that row contiguously, but rows are 192 bytes apart in W2
// and 1,792 in W1/W3, so a warp's thirty-two lanes put one `LDG.128` across
// thirty-two sectors. Dealing the round out by flat vector index instead --
// consecutive lanes on consecutive vectors -- brings a down unit's load inside
// 512 contiguous bytes, four sectors. It is 14% slower at M16 and 18% at M128,
// because the destination moves with the source: consecutive vectors belong to
// consecutive K groups, which are different 4 KB operand tiles, and 4,096 is a
// whole number of 128-byte bank periods, so the warp's thirty-two shared
// stores collide on the same banks.
//
// Two mappings were built and measured, and each pays on one side: the shipped
// row-major deal takes conflict-free shared stores and thirty-two-sector
// reads, and the vector-major deal takes four-sector reads and a twelve-way
// store conflict. Nothing here rules out a third mapping -- a staging buffer
// per row group, or a different swizzle, would move the trade and neither was
// tried -- so these are two measured points, not a proof that the global row
// stride and the swizzled tile stride can never be satisfied together.
//
// What the two points do say is where to spend the next attempt. A copy engine
// gets both sides at once by construction, and it is what the native layer
// uses: vLLM's expert BMM is `..._tma_ldgstsSf_rgTma_...`, a TMA read of the
// global tile writing the swizzled shared destination directly, under an
// `m128x8x32` contraction that puts tokens on N. That pairing is the leading
// candidate for this region, and it is a different kernel structure rather
// than a different index -- but it has not been built or measured here, so it
// is a candidate and not a conclusion.
// ---------------------------------------------------------------------------

/// One shared tile row's two sixteen-byte atoms.
__device__ __forceinline__ uint4 *atom_of(
    mixed_operand_tile &tile,
    const int row,
    const int atom
) {
    return reinterpret_cast<uint4 *>(&tile[{row, atom * 16}]);
}

/// Fill one operand tile with the zeros a short batch's absent rows need.
///
/// A batch shorter than the MMA's 128 rows leaves whole rows out of the
/// contraction, and those rows have to read as zero rather than as whatever
/// the previous unit left behind. Every K group of a unit reuses the same
/// tiles, so one pass per unit covers all of them. Zero is zero under any
/// swizzle, so the tile is cleared as a flat span.
__device__ __forceinline__ void clear_operand_tile(mixed_operand_tile &tile) {
    uint4 *const raw = reinterpret_cast<uint4 *>(tile.data);
    constexpr int vectors = kMmaM * kMmaK / 16;
    for (int index = static_cast<int>(threadIdx.x); index < vectors;
         index += kDecodeCtaThreads) {
        raw[index] = make_uint4(0u, 0u, 0u, 0u);
    }
}

/// Fill one scale tile with the unit scale a short batch's absent rows need.
///
/// The absent rows are zero, but `0xff` is E8M0's NaN and would poison the
/// products anyway, so their scale factors are pinned to one.
__device__ __forceinline__ void clear_scale_tile(mixed_scale_tile &tile) {
    std::uint32_t *const raw = reinterpret_cast<std::uint32_t *>(tile.data);
    constexpr int words = kScaleRows * kScaleColumns / 4;
    for (int index = static_cast<int>(threadIdx.x); index < words;
         index += kDecodeCtaThreads) {
        raw[index] = 0x7f7f7f7fu;
    }
}

/// Stage one row's four consecutive scale factors as the one word they share.
///
/// A row's four factors are the four bytes at `scale_factor_1x_offset(row, 0)`
/// and the source keeps them equally adjacent, so a K-group quad costs one
/// load and one store per row rather than four of each.
__device__ __forceinline__ void stage_scale_quad(
    mixed_scale_tile &scale_shared,
    const int row,
    const std::uint32_t quad
) {
    reinterpret_cast<std::uint32_t *>(
        scale_shared.data)[scale_factor_1x_offset(row, 0) / 4] = quad;
}

/// Stage one MXFP8 activation row: thirty-two live bytes, two atoms.
__device__ __forceinline__ void stage_activation_row(
    mixed_operand_tile &tile,
    const int row,
    const std::uint8_t *__restrict__ source
) {
    *atom_of(tile, row, 0) = *reinterpret_cast<const uint4 *>(source);
    *atom_of(tile, row, 1) = *reinterpret_cast<const uint4 *>(source + 16);
}

/// Stage one MXFP4 weight row: sixteen packed bytes, eight per atom.
///
/// `16U4_ALIGN16B` puts sixteen packed U4 values at the front of each atom and
/// the MMA never reads the rest, so each atom is written whole -- eight source
/// bytes and eight zeros -- and the tile needs no separate clearing pass.
__device__ __forceinline__ void stage_weight_row(
    mixed_operand_tile &tile,
    const int row,
    const uint4 payload
) {
    *atom_of(tile, row, 0) = make_uint4(payload.x, payload.y, 0u, 0u);
    *atom_of(tile, row, 1) = make_uint4(payload.z, payload.w, 0u, 0u);
}

/// Gather one round of packed gate/up rows with TMA.
template<int TILES>
__device__ __forceinline__ void issue_packed_weight_round(
    packed_weight_tile (&first)[TILES],
    packed_weight_tile (&second)[TILES],
    const packed_weight_layout &first_layout,
    const packed_weight_layout &second_layout,
    const int expert,
    const int output_tile,
    const int group_base,
    kittens::semaphore &arrived
) {
    if (threadIdx.x != 0) return;
    kittens::tma::expect_bytes(
        arrived, 2 * TILES * static_cast<int>(sizeof(packed_weight_tile)));
    #pragma unroll
    for (int slot = 0; slot < TILES; ++slot) {
        const kittens::coord<packed_weight_tile> tile{
            expert, 0, output_tile, group_base + slot};
        kittens::tma::load_async(
            first[slot], first_layout, tile, arrived);
        kittens::tma::load_async(
            second[slot], second_layout, tile, arrived);
    }
}

/// Gather one round of packed down-projection rows with TMA.
template<int TILES>
__device__ __forceinline__ void issue_packed_weight_round(
    packed_weight_tile (&weight)[TILES],
    const packed_weight_layout &layout,
    const int expert,
    const int output_tile,
    const int group_base,
    kittens::semaphore &arrived
) {
    if (threadIdx.x != 0) return;
    kittens::tma::expect_bytes(
        arrived, TILES * static_cast<int>(sizeof(packed_weight_tile)));
    #pragma unroll
    for (int slot = 0; slot < TILES; ++slot) {
        const kittens::coord<packed_weight_tile> tile{
            expert, 0, output_tile, group_base + slot};
        kittens::tma::load_async(
            weight[slot], layout, tile, arrived);
    }
}

__device__ __forceinline__ void store_accumulator(
    const mixed_accumulator_tile &accumulator,
    mixed_result_tile &destination
) {
    using namespace kittens;
    if (warpgroup::groupid() == 0) {
        rt_fl<kMmaM / 4, kMmaN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(destination, result);
    }
    __syncthreads();
}

/// Quantize this worker's stride of the routed latent to MXFP8.
///
/// The private stage runs one CTA and passes `worker = 0, workers = 1`; the
/// persistent kernel spreads the same groups over its whole resident grid.
__device__ __forceinline__ void quantize_latent_rows(
    const __nv_bfloat16 *__restrict__ latent_x,
    const Scratch &scratch,
    const int active_tokens,
    const int worker,
    const int workers
) {
    const int thread = static_cast<int>(threadIdx.x);
    const int total_groups = active_tokens * kLatentGroups;
    for (int index = worker * kDecodeCtaThreads + thread;
         index < total_groups;
         index += workers * kDecodeCtaThreads) {
        const int token = index / kLatentGroups;
        const int group = index % kLatentGroups;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const float value = __bfloat162float(
                latent_x[static_cast<long long>(token) * kLatentSize
                         + group * kMmaK + k]);
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale = select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        scratch.latent_scale[
            static_cast<long long>(token) * kLatentGroups + group] = scale;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            scratch.latent_mxfp8[
                static_cast<long long>(token) * kLatentSize
                + group * kMmaK + k] =
                    quantize_e4m3(values[k], reciprocal);
        }
    }
}

__device__ __forceinline__ void quantize_situ_tile(
    const mixed_result_tile &gate,
    const mixed_result_tile &up,
    const Scratch &scratch,
    const int assignment_begin,
    const int batch_rows,
    const int output_base
) {
    const int thread = static_cast<int>(threadIdx.x);
    constexpr int groups_per_tile = kMmaN / kMmaK;
    for (int index = thread; index < batch_rows * groups_per_tile;
         index += kDecodeCtaThreads) {
        const int row = index / groups_per_tile;
        const int local_group = index % groups_per_tile;
        const int assignment = assignment_begin + row;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const int column = local_group * kMmaK + k;
            const float gate_value = gate[{row, column}];
            const float up_value = up[{row, column}];
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
        const int global_group =
            output_base / kMmaK + local_group;
        scratch.situ_scale[
            static_cast<long long>(assignment) * kSituGroups
            + global_group] = scale;
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

__device__ __forceinline__ void accumulate_down_tile(
    const mixed_result_tile &result,
    const Scratch &scratch,
    const int assignment_begin,
    const int batch_rows,
    const int output_base,
    const int active_tokens
) {
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < batch_rows * kMmaN;
         index += kDecodeCtaThreads) {
        const int row = index / kMmaN;
        const int column = index % kMmaN;
        const int assignment = assignment_begin + row;
        const int token = scratch.assignment_tokens[assignment];
        if (token >= 0 && token < active_tokens) {
            atomicAdd(
                &scratch.routed_accumulator[
                    static_cast<long long>(token) * kLatentSize
                    + output_base + column],
                result[{row, column}]
                    * decode_route_weight(scratch, assignment));
        }
    }
}

// ---------------------------------------------------------------------------
// K-group rounds.
//
// A unit stages a whole round of K groups, issues every one of the round's
// contractions, and waits once. Three things follow from that and all three
// are why it is done this way.
//
// The round's global reads of one weight row are contiguous: K group `g` reads
// packed bytes `[16g, 16g + 16)` of the row, so eight groups is one 128-byte
// span and twelve is a whole 192-byte W2 row. Its scale reads are contiguous
// for the same reason, which turns what was one strided byte per group into
// one vector for the round.
//
// The round's reads are also all in flight at once. Sixteen bytes per group
// per row is one vector register, so a thread issues the round's loads back to
// back before it consumes any of them, and the memory system sees the whole
// round's parallelism rather than one group's.
//
// And the round amortizes synchronization. The measured profile of the
// unbatched loop spent 15% of the decode step inside barriers, six per K
// group; a round costs two whatever its width.
// ---------------------------------------------------------------------------

/// K groups one gate/up round stages. 112 latent groups is fourteen rounds.
inline constexpr int kGateUpRoundGroups = 8;
inline constexpr int kGateUpRounds = kLatentGroups / kGateUpRoundGroups;

/// K groups one down round stages, which is every group the SiTU width has.
inline constexpr int kDownRoundGroups = kSituGroups;
inline constexpr int kDownRounds = kSituGroups / kDownRoundGroups;

static_assert(kLatentGroups % kGateUpRoundGroups == 0);
static_assert(kGateUpRounds == 14);
static_assert(kDownRounds == 1);

// A round's weight staging is exactly one row of one operand tile per thread.
static_assert(2 * kMmaN == kDecodeCtaThreads);

/// Scale tiles a round stages, one per quad of K groups.
///
/// A 512-byte `scale_vec::1X` tile is one byte per row per K group for four
/// consecutive groups, so a round's groups share a tile four at a time and the
/// MMA's scale-factor id picks the group out of it. That is four times fewer
/// shared tiles, tensor-memory buffers, and `tcgen05.cp` issues per round than
/// giving every group its own.
inline constexpr int kGateUpScaleTiles =
    kGateUpRoundGroups / kScaleGroupsPerTile;
inline constexpr int kDownScaleTiles = kDownRoundGroups / kScaleGroupsPerTile;

static_assert(kGateUpRoundGroups % kScaleGroupsPerTile == 0);
static_assert(kDownRoundGroups % kScaleGroupsPerTile == 0);
static_assert(kGateUpScaleTiles == 2);
static_assert(kDownScaleTiles == 3);

/// First tensor-memory column the routed scale factors occupy.
///
/// The two 128-column accumulators come first, so the scales start above them
/// and each `full_tt_fp8e8m0<16>` takes four columns.
inline constexpr int kRoutedScaleColumnBase = 2 * kMmaN;
inline constexpr int kRoutedScaleColumns = 4;
inline constexpr int kGateUpScaleBuffers = 3 * kGateUpScaleTiles;
inline constexpr int kDownScaleBuffers = 2 * kDownScaleTiles;

static_assert(kRoutedScaleColumnBase
                  + kRoutedScaleColumns * kGateUpScaleBuffers
              <= kittens::tensor_allocator<1, 1>::cols);
static_assert(kRoutedScaleColumnBase
                  + kRoutedScaleColumns * kDownScaleBuffers
              <= kittens::tensor_allocator<1, 1>::cols);

static_assert(sizeof(mixed_operand_tile) == 4096);
static_assert(sizeof(packed_weight_tile) == 2048);
static_assert(sizeof(mixed_scale_tile) == 512);
static_assert(sizeof(mixed_result_tile) == 65536);

/// Shared bytes a unit's staging occupies.
///
/// The allocator aligns every *array* to 1 KiB. Operand arrays are already a
/// multiple of that, but a three-tile scale array is not, so the padding is
/// walked rather than assumed.
__host__ __device__ constexpr int staging_bytes(
    const int operand_arrays,
    const int operand_tiles,
    const int scale_arrays,
    const int scale_tiles
) {
    constexpr int align = 1024;
    const auto round_up = [](const int bytes) {
        return (bytes + align - 1) / align * align;
    };
    int total = 0;
    for (int array = 0; array < operand_arrays; ++array) {
        total = round_up(total)
              + operand_tiles * static_cast<int>(sizeof(mixed_operand_tile));
    }
    for (int array = 0; array < scale_arrays; ++array) {
        total = round_up(total)
              + scale_tiles * static_cast<int>(sizeof(mixed_scale_tile));
    }
    return round_up(total);
}

inline constexpr int kGateUpStagingBytes =
    staging_bytes(3, kGateUpRoundGroups, 3, kGateUpScaleTiles)
    + 2 * kGateUpRoundGroups
        * static_cast<int>(sizeof(packed_weight_tile));
inline constexpr int kDownStagingBytes =
    staging_bytes(2, kDownRoundGroups, 2, kDownScaleTiles)
    + kDownRoundGroups * static_cast<int>(sizeof(packed_weight_tile));

/// Shared bytes one gate/up unit holds, which is the widest of any K3 stage.
///
/// A unit's staging and its results never overlap in time -- the results are
/// read out of tensor memory only after the last contraction of the last round
/// has retired -- so they are laid over each other from the same base and the
/// unit costs whichever is larger rather than their sum.
inline constexpr int kGateUpUnitSharedBytes =
    kGateUpStagingBytes
            > 2 * static_cast<int>(sizeof(mixed_result_tile))
        ? kGateUpStagingBytes
        : 2 * static_cast<int>(sizeof(mixed_result_tile));
inline constexpr int kDownUnitSharedBytes =
    kDownStagingBytes > static_cast<int>(sizeof(mixed_result_tile))
        ? kDownStagingBytes
        : static_cast<int>(sizeof(mixed_result_tile));

static_assert(kGateUpStagingBytes == 134144);
static_assert(kDownStagingBytes == 126976);
static_assert(kGateUpUnitSharedBytes == 134144);
static_assert(kDownUnitSharedBytes == 126976);

}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
