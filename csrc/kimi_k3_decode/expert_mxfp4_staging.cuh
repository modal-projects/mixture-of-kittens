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

// `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` requires an inner box of 128 U4
// values and writes each packed 8-byte group into one 16-byte shared atom.
// Four adjacent K32 groups are therefore the smallest direct TMA tile that
// already has the exact byte layout consumed by the mixed MMA. Its supported
// 128-byte swizzle also lets four K32 chunk descriptors address the result
// without a compact-to-swizzled thread-copying hop.
inline constexpr int kDirectWeightGroups = 4;
inline constexpr int kWeightPipelineStages = 2;
inline constexpr int kDirectWeightAlignment = 32;
inline constexpr int kDirectWeightTransactionBytes =
    kMmaN * kDirectWeightGroups * kMmaK / 2;
using direct_weight_stage =
    kittens::st_fp8e4m3<
        kMmaN, kDirectWeightGroups * kMmaK, true, 128>;

struct alignas(128) direct_weight_layout {
    CUtensorMap tensor_map;
};

struct RoutedLayouts {
    direct_weight_layout w1;
    direct_weight_layout w3;
    direct_weight_layout w2;
};

static __host__ inline direct_weight_layout direct_weight_layout_for(
    const std::uint8_t *pointer,
    const int rows,
    const int packed_columns
) {
    static_assert(kDirectWeightGroups * kMmaK == 128);
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(pointer)
                % kDirectWeightAlignment
            == 0,
        "MoK: direct Kimi K3 weight TMA requires a ",
        kDirectWeightAlignment, "-byte-aligned packed weight");
    TORCH_CHECK(rows % kMmaN == 0);
    TORCH_CHECK(
        (2 * packed_columns) % (kDirectWeightGroups * kMmaK) == 0);

    direct_weight_layout layout{};
    const std::uint64_t global_dimensions[5] = {
        kDirectWeightGroups * kMmaK,
        static_cast<std::uint64_t>(rows),
        static_cast<std::uint64_t>(
            2 * packed_columns / (kDirectWeightGroups * kMmaK)),
        1,
        kNumExperts,
    };
    const std::uint64_t global_strides[4] = {
        static_cast<std::uint64_t>(packed_columns),
        kDirectWeightGroups * kMmaK / 2,
        static_cast<std::uint64_t>(rows) * packed_columns,
        static_cast<std::uint64_t>(rows) * packed_columns,
    };
    const std::uint32_t box_dimensions[5] = {
        kDirectWeightGroups * kMmaK,
        kMmaN,
        1,
        1,
        1,
    };
    const std::uint32_t element_strides[5] = {1, 1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &layout.tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        5,
        const_cast<std::uint8_t *>(pointer),
        global_dimensions,
        global_strides,
        box_dimensions,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    const char *error = nullptr;
    if (result != CUDA_SUCCESS) {
        cuGetErrorString(result, &error);
    }
    TORCH_CHECK(
        result == CUDA_SUCCESS,
        "MoK: failed to encode direct Kimi K3 weight tensor map: ",
        error == nullptr ? "unknown CUDA driver error" : error);
    return layout;
}

static __host__ inline RoutedLayouts routed_layouts(
    const std::uint8_t *w1,
    const std::uint8_t *w3,
    const std::uint8_t *w2
) {
    return RoutedLayouts{
        direct_weight_layout_for(
            w1, kExpertW1W3PackedRows, kExpertW1W3PackedColumns),
        direct_weight_layout_for(
            w3, kExpertW1W3PackedRows, kExpertW1W3PackedColumns),
        direct_weight_layout_for(
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

/// Load one four-K32 stage into its final 128-byte-swizzled MMA layout.
__device__ __forceinline__ void load_direct_weight_stage(
    direct_weight_stage &destination,
    const direct_weight_layout &layout,
    const int expert,
    const int output_tile,
    const int round,
    kittens::semaphore &arrived
) {
    const std::uint64_t tensor_map =
        reinterpret_cast<std::uint64_t>(&layout.tensor_map);
    const std::uint32_t barrier =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(&arrived));
    const std::uint32_t destination_shared =
        static_cast<std::uint32_t>(
            __cvta_generic_to_shared(&destination));
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :
        : "r"(destination_shared),
          "l"(tensor_map),
          "r"(barrier),
          "n"(0),
          "r"(output_tile * kMmaN),
          "r"(round),
          "n"(0),
          "r"(expert)
        : "memory");
}

/// Prime or advance the two-stage gate/up weight pipeline.
template<int STAGES>
__device__ __forceinline__ void issue_direct_weight_round(
    direct_weight_stage (&first)[STAGES],
    direct_weight_stage (&second)[STAGES],
    const direct_weight_layout &first_layout,
    const direct_weight_layout &second_layout,
    const int expert,
    const int output_tile,
    const int round,
    const int stage,
    kittens::semaphore &arrived
) {
    static_assert(STAGES == 1 || STAGES == kWeightPipelineStages);
    if (threadIdx.x != 0) return;
    kittens::tma::expect_bytes(
        arrived, 2 * kDirectWeightTransactionBytes);
    load_direct_weight_stage(
        first[stage], first_layout, expert, output_tile, round, arrived);
    load_direct_weight_stage(
        second[stage], second_layout, expert, output_tile, round, arrived);
}

/// Prime or advance the two-stage down-projection weight pipeline.
template<int STAGES>
__device__ __forceinline__ void issue_direct_weight_round(
    direct_weight_stage (&weight)[STAGES],
    const direct_weight_layout &layout,
    const int expert,
    const int output_tile,
    const int round,
    const int stage,
    kittens::semaphore &arrived
) {
    static_assert(STAGES == 1 || STAGES == kWeightPipelineStages);
    if (threadIdx.x != 0) return;
    kittens::tma::expect_bytes(arrived, kDirectWeightTransactionBytes);
    load_direct_weight_stage(
        weight[stage], layout, expert, output_tile, round, arrived);
}

/// Contract one K32 chunk from a direct 128-byte-swizzled weight stage.
__device__ __forceinline__ void mixed_mma_direct(
    const mixed_accumulator_tile &destination,
    const mixed_operand_tile &activation,
    const direct_weight_stage &weight,
    const int chunk,
    const kittens::full_tt_fp8e8m0<16> &activation_scale,
    const kittens::full_tt_fp8e8m0<16> &weight_scale,
    const int scale_factor_id,
    const bool accumulate
) {
    kittens::st_descriptor<
        mixed_operand_tile, kittens::transpose::N> activation_descriptor(
            activation);
    kittens::st_descriptor<
        direct_weight_stage, kittens::transpose::N> weight_descriptor(weight);
    const std::uint64_t weight_chunk =
        weight_descriptor.chunk_descriptor(chunk);
    const std::uint32_t instruction =
        mixed_instruction_descriptor(scale_factor_id);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :
        : "r"(destination.addr),
          "l"(activation_descriptor.base_desc),
          "l"(weight_chunk),
          "r"(instruction),
          "r"(activation_scale.addr),
          "r"(weight_scale.addr),
          "r"(accumulate ? 1u : 0u));
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
// One round is the tensor map's minimum 128-U4 box: four contiguous K32
// groups. The two shared stages alternate, so the next round's direct TMA can
// run while the current stage is consumed. Scales use the same four-group
// width and remain one contiguous word per row.
// ---------------------------------------------------------------------------

/// K groups one direct-TMA gate/up round stages.
inline constexpr int kGateUpRoundGroups = kDirectWeightGroups;
inline constexpr int kGateUpRounds = kLatentGroups / kGateUpRoundGroups;

/// K groups one direct-TMA down round stages.
inline constexpr int kDownRoundGroups = kDirectWeightGroups;
inline constexpr int kDownRounds = kSituGroups / kDownRoundGroups;

static_assert(kLatentGroups % kGateUpRoundGroups == 0);
static_assert(kSituGroups % kDownRoundGroups == 0);
static_assert(kGateUpRounds == 28);
static_assert(kDownRounds == 3);

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
static_assert(kGateUpScaleTiles == 1);
static_assert(kDownScaleTiles == 1);

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
static_assert(sizeof(direct_weight_stage) == 16384);
static_assert(2 * kDirectWeightTransactionBytes
              == static_cast<int>(sizeof(direct_weight_stage)));
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
    staging_bytes(1, kGateUpRoundGroups, 3, kGateUpScaleTiles)
    + 2 * kWeightPipelineStages
        * static_cast<int>(sizeof(direct_weight_stage));
inline constexpr int kDownStagingBytes =
    staging_bytes(1, kDownRoundGroups, 2, kDownScaleTiles)
    + kWeightPipelineStages
        * static_cast<int>(sizeof(direct_weight_stage));

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

static_assert(kGateUpStagingBytes == 84992);
static_assert(kDownStagingBytes == 51200);
static_assert(kGateUpUnitSharedBytes == 131072);
static_assert(kDownUnitSharedBytes == 65536);

}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
