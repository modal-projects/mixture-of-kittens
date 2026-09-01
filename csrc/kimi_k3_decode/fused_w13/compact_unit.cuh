#pragma once

/// The compact three-stage ring: production's narrow arm.
///
/// All seven slabs at once and one gather per pass, in 16,384 bytes instead of
/// 57,344, by packing slab `s`'s eight-row operand four rows past slab
/// `s - 1`'s and moving all 28 activation scale tiles into tensor memory. The
/// packing is what bounds the arm: a batch wider than `kFusedCompactRows` does
/// not fit it.
///
/// The soundness argument for reading an inactive N column that holds the next
/// slab's live rows is in `fused_compact_operand` below, and what stands in for
/// it is `tests/test_kimi_k3_adaptive_gate_up.py`.

#include "slab_unit.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

// ---------------------------------------------------------------------------
// The adaptive engine: a compact activation for narrow batches, and the
// production ring unchanged for everything else.
//
// The resident engine's activation costs 71,680 bytes and a third K = 512
// weight stage wants 67,584, so the two cannot both be had. V4 answered that by
// making the activation per slab, which works and costs a gather the ring has
// to cover. This answers it by making the activation *smaller* instead, and the
// reason that is possible is what the workload actually routes.
//
// The MMA contracts eight N columns and the resident tile is sixteen rows, so
// 57,344 of the activation's bytes describe eight physical rows of which the
// validated workload fills one at M16 and M32 and two or three at M128. The
// rest is a shape the format demands, not data.
//
// **What the format demands, exactly.** A 128B-swizzled K-major operand puts
// descriptor row `r` at 128 bytes past row `r - 1`, with the sixteen-byte
// sub-atoms of a row permuted by `r % 8`. So one slab's operand is four atom
// columns of eight 128-byte rows -- 4,096 bytes -- whatever the batch is, and
// seven slabs is 28,672. There is no shorter operand: `n_dim` bottoms out at
// eight, and the swizzle's period is eight rows in every mode.
//
// **So the dead rows have to hold something else.** Slab `s`'s base sits four
// rows past slab `s - 1`'s, and its descriptor rows four to seven are slab
// `s + 1`'s live rows rather than zeros. That is sound because the MMA's N
// columns are independent -- accumulator column `n` reads activation row `n`
// and nothing else -- and the epilogue reads only columns below `rows`. It is
// also the one place this engine cannot honour "inactive N columns read zero":
// the bytes that would make them zero are the 12,288 that buy the third stage.
// `test_the_compact_engine_is_bitwise_equal_to_production` is what stands in
// for that invariant, and it is a stronger check than the invariant was.
//
// **The scales do not have to stay.** Seven slabs of four 512-byte scale tiles
// is 14,336 shared bytes, of which 32 per tile are live, and every one of them
// is read by `tcgen05.cp` into a tensor-memory buffer before the first
// contraction. Tensor memory has 512 columns and the ring uses 16 for its
// accumulator and 32 for the weight scales, so all 28 activation buffers fit at
// once -- copied once per pass instead of four times per slab, which is 140
// fewer copies per expert. Their shared tiles are then dead, so they are staged
// *inside* the third weight stage and that stage's first transfer is issued
// after the copies land.
//
// What this keeps: one expert-pure CTA claim, one accumulator, six sequential
// tasks, seven K = 512 slabs, one large weight transfer and one contiguous
// scale transfer per slab, the one-time activation gather, the deferred retire,
// the exact `situ` expression, the six published column ranges, and the
// prepared bytes the transform already writes.
//
// What it changes: the ring is three deep and prefetches two indices ahead, and
// the activation is 16,384 bytes instead of 71,680.
// ---------------------------------------------------------------------------

/// Live rows the compact specialization accepts, and the rows it spaces slabs
/// by.
///
/// One number, used twice, because the pitch is what bounds the batch: slab `s`
/// starts at row `s * pitch` and its live rows are the first `rows` of the
/// eight the descriptor spans, so the two must not exceed one another.
inline constexpr int kFusedCompactRows = 4;
inline constexpr int kFusedCompactPitch = kFusedCompactRows;

/// Physical rows the packed activation needs.
///
/// The last slab's base is `6 * pitch` and its descriptor spans eight rows past
/// it, so the tile has to reach row `6 * pitch + 8` even though only
/// `7 * pitch` of those rows are ever written. Reading a row the tile does not
/// own would be an out-of-range shared access on every unit, not a rare one.
inline constexpr int kFusedCompactPhysicalRows =
    (kFusedSlabs - 1) * kFusedCompactPitch + kFusedN;

using fused_compact_activation_tile =
    kittens::st_fp8e4m3<kFusedCompactPhysicalRows, kFusedSlabK>;

static_assert(kFusedCompactPhysicalRows == 32);
static_assert(kFusedCompactPhysicalRows
                  % kittens::TILE_ROW_DIM<kittens::fp8e4m3> == 0,
              "the packed activation must be a tile ThunderKittens can index, "
              "because the gather writes it through `[{row, column}]` and the "
              "tensor core reads it through that tile's own descriptor");
static_assert(sizeof(fused_compact_activation_tile) == 16384);
static_assert(fused_compact_activation_tile::swizzle_bytes == 128,
              "the packed activation must share the weights' chunk stride");
static_assert(kFusedCompactRows <= kFusedN,
              "a compact pass is one pass, so its rows must fit the MMA's N");
static_assert(kFusedCompactPitch * kFusedSwizzleAtomBytes % 16 == 0,
              "a slab's descriptor offset must land on a sixteen-byte boundary, "
              "which is the granularity the address field encodes");

/// The compact ring's depth, which is the whole point of the specialization.
inline constexpr int kFusedCompactStages = 3;

static_assert(kFusedStreamLength % kFusedCompactStages == 0,
              "42 indices over three stages is fourteen whole laps, so a pass "
              "hands the next one every barrier at the parity it found");

/// Tensor-memory columns the compact engine's scale buffers occupy.
///
/// Eight weight buffers -- four quads, double buffered by slab parity, exactly
/// as the resident engine does it -- and then all 28 activation buffers, which
/// is what lets the activation scales be copied once per pass.
inline constexpr int kFusedCompactWeightScaleBuffers =
    kFusedScaleSets * kFusedSlabScaleTiles;
inline constexpr int kFusedCompactScaleColumnBase =
    kFusedScaleColumnBase
    + kFusedCompactWeightScaleBuffers * kRoutedScaleColumns;
inline constexpr int kFusedCompactActivationScaleBuffers =
    kFusedSlabs * kFusedSlabScaleTiles;

static_assert(kFusedCompactWeightScaleBuffers == 8);
static_assert(kFusedCompactScaleColumnBase == 288);
static_assert(kFusedCompactActivationScaleBuffers == 28);
static_assert(kFusedCompactScaleColumnBase
                  + kFusedCompactActivationScaleBuffers * kRoutedScaleColumns
              <= kittens::tensor_allocator<1, 1>::cols,
              "every activation scale buffer must be resident at once, or the "
              "copies would have to run per slab again");

/// Shared bytes one compact unit occupies.
///
///   3 x 65,536  weight slabs             = 196,608
///   3 x  2,048  weight scale quads       =   6,144
///   1 x 16,384  packed activation        =  16,384
///   1 x  8,192  epilogue result tile     =   8,192
///                                          -------
///                                          227,328
///
/// The 14,336 bytes of activation scale tiles are absent because they are
/// staged in the third stage's own bytes and are dead before that stage's first
/// transfer is issued.
inline constexpr int kFusedCompactStagingBytes =
    kFusedCompactStages * static_cast<int>(sizeof(fused_weight_tile))
    + kFusedCompactStages * kFusedSlabScaleTiles
          * static_cast<int>(sizeof(mixed_scale_tile))
    + static_cast<int>(sizeof(fused_compact_activation_tile))
    + static_cast<int>(sizeof(fused_result_tile));

inline constexpr int kFusedCompactSharedBytes =
    kFusedCompactStagingBytes + kFusedAllocatorPadding;

/// Bytes the transient activation scale tiles need, against the stage they
/// borrow.
inline constexpr int kFusedCompactActivationScaleBytes =
    kFusedSlabs * kFusedSlabScaleTiles
    * static_cast<int>(sizeof(mixed_scale_tile));

static_assert(kFusedCompactStagingBytes == 227328);
static_assert(kFusedCompactSharedBytes == 228352);
static_assert(kFusedCompactActivationScaleBytes == 14336);
static_assert(kFusedCompactActivationScaleBytes
                  <= static_cast<int>(sizeof(fused_weight_tile)),
              "the transient scale tiles must fit inside the stage that has "
              "not been issued yet");
static_assert(kFusedCompactSharedBytes
                  <= kittens::MAX_SHARED_MEMORY - kFusedStaticSharedReserve,
              "the compact ring must leave room for static shared memory");
static_assert(2 * kFusedCompactSharedBytes > kittens::MAX_SHARED_MEMORY,
              "the compact grid must still be one CTA per SM");
static_assert(kFusedCompactSharedBytes > kFusedW13SharedBytes,
              "the compact ring buys a third stage, so it must ask for more "
              "than the resident ring");
// The third stage is paid for by the activation the packing stops holding: the
// resident engine's seven sixteen-row tiles and their scale quads, against one
// 32-row tile and no resident scales at all.
static_assert(kFusedCompactStagingBytes
                  == kFusedStagingBytes
                         + static_cast<int>(sizeof(fused_weight_tile))
                         + kFusedSlabScaleTiles
                               * static_cast<int>(sizeof(mixed_scale_tile))
                         - kFusedActivationSlabs
                               * static_cast<int>(
                                     sizeof(fused_activation_tile))
                         + static_cast<int>(
                               sizeof(fused_compact_activation_tile))
                         - kFusedActivationSlabs * kFusedSlabScaleTiles
                               * static_cast<int>(sizeof(mixed_scale_tile)),
              "the third stage is paid for by the activation the packing stops "
              "holding");

/// What a zeroed eight-row operand per slab would cost, and why it is not here.
///
/// Seven slabs of four atom columns of eight 128-byte rows is 28,672 bytes, and
/// the difference against the packed 16,384 is 12,288 -- more than the 2,048
/// the launch has left. So "every inactive N column reads zero" and "three
/// K = 512 weight stages" are not both purchasable, and this engine spends the
/// bytes on the stage.
inline constexpr int kFusedCompactZeroedActivationBytes =
    kFusedSlabs * kFusedN * kFusedSwizzleAtomBytes * kFusedBoxes;

static_assert(kFusedCompactZeroedActivationBytes == 28672);
static_assert(kFusedCompactStagingBytes
                  - static_cast<int>(sizeof(fused_compact_activation_tile))
                  + kFusedCompactZeroedActivationBytes
                  + kFusedAllocatorPadding
              > kittens::MAX_SHARED_MEMORY - kFusedStaticSharedReserve,
              "a zeroed eight-row operand per slab must not fit, or the packing "
              "would be a choice rather than a consequence");

/// Turn the packed tile's chunk descriptor into slab `s`'s operand.
///
/// One field moves. The address field advances by the slab's row offset, which
/// cannot carry out of it: the whole opt-in block is 232,448 bytes and the
/// field spans 262,144.
///
/// The matrix base offset -- PTX ISA 9.7.17.4.1, table 43, bits 51 to 49 --
/// stays zero, and that is the whole of the swizzle argument. It does not
/// describe where the *matrix* starts; it describes where the swizzle's
/// repeating pattern starts, and it is zero exactly when that pattern begins on
/// a 1,024-byte boundary. The pattern here begins at the packed tile's own
/// base, which `tma_swizzle_allocator` puts on a 1 KiB boundary, so it is zero
/// for every slab including the odd ones that start 512 bytes into it. The
/// phase an odd slab's rows are permuted by is then derived from the start
/// address itself, which is the same rule `st::idx` writes them under --
/// `((address % 1024) >> 7) << 4` -- so the two agree by construction.
///
/// Setting the field to the start's own phase instead reads the odd slabs
/// through a phase four atoms off, which is three of seven slabs contracting
/// permuted K and is what
/// `test_each_row_count_across_the_compact_threshold_is_production` caught.
__device__ __forceinline__ std::uint64_t fused_compact_operand(
    const std::uint64_t chunk,
    const int slab
) {
    const std::uint64_t start = static_cast<std::uint64_t>(slab)
        * kFusedCompactPitch * kFusedSwizzleAtomBytes;
    return chunk + kittens::detail::matrix_descriptor_encode(start);
}

/// One sixteen-byte swizzle atom of a packed activation row.
__device__ __forceinline__ uint4 *fused_compact_atom(
    fused_compact_activation_tile &tile,
    const int row,
    const int atom
) {
    return reinterpret_cast<uint4 *>(&tile[{row, atom * 16}]);
}

/// Gather every live activation row and scale for all seven slabs at once.
///
/// The same reads the resident engine's whole-unit gather makes, landing in a
/// quarter of the bytes: slab `s`'s row `r` goes to physical row
/// `s * pitch + r` rather than to row `r` of slab `s`'s own tile. Rows the
/// batch does not fill are written as zero, so the only thing an inactive N
/// column can ever see is another slab's live row -- never whatever the last
/// unit left behind, which for a scale byte would be `0xff` and E8M0's NaN.
///
/// The scale tiles are the resident engine's, staged whole because
/// `tcgen05.cp` reads a whole one, and they are transient: the caller copies
/// them into tensor memory and then hands their bytes to the third stage.
///
/// The caller owes the asynchronous proxy a fence on every thread and a CTA
/// barrier before the tensor core reads any of this.
__device__ __forceinline__ void stage_fused_compact_activation(
    fused_compact_activation_tile &payload,
    mixed_scale_tile (&scales)[kFusedSlabs][kFusedSlabScaleTiles],
    const Scratch &scratch,
    const int assignment_begin,
    const int rows
) {
    constexpr int atoms_per_row = kFusedSlabK / 16;
    constexpr int atoms = kFusedCompactPhysicalRows * atoms_per_row;
    constexpr int quads_per_slab = kFusedN * kFusedSlabScaleTiles;
    constexpr int quads = kFusedSlabs * quads_per_slab;

    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < atoms; index += kDecodeCtaThreads) {
        const int row = index / atoms_per_row;
        const int atom = index % atoms_per_row;
        const int slab = row / kFusedCompactPitch;
        const int local = row % kFusedCompactPitch;
        if (slab < kFusedSlabs && local < rows) {
            const int token =
                scratch.assignment_tokens[assignment_begin + local];
            *fused_compact_atom(payload, row, atom) =
                *reinterpret_cast<const uint4 *>(
                    scratch.latent_mxfp8
                    + static_cast<long long>(token) * kLatentSize
                    + slab * kFusedSlabK + atom * 16);
        } else {
            *fused_compact_atom(payload, row, atom) =
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

/// Contract all six fused output tasks of one narrow expert batch.
///
/// Same contract as the resident unit -- one queue claim, one accumulator, six
/// published column ranges, `situ` bytes equal to it bit for bit -- over a ring
/// that is three deep because the activation is a quarter of the size.
///
/// A `batch_rows` of zero is the arming call: the barriers are initialized and
/// nothing else runs, which is how a CTA whose first expert takes the other
/// specialization still leaves this one's ring usable.
static __device__ void routed_gate_up_fused_compact_unit(
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
    fused_weight_tile (&weight)[kFusedCompactStages] =
        staging.allocate<fused_weight_tile, kFusedCompactStages>();
    mixed_scale_tile
        (&weight_scale)[kFusedCompactStages][kFusedSlabScaleTiles] =
            staging.allocate<
                mixed_scale_tile, kFusedCompactStages, kFusedSlabScaleTiles>();
    fused_compact_activation_tile (&activation) =
        staging.allocate<fused_compact_activation_tile>();
    fused_result_tile (&result) = staging.allocate<fused_result_tile>();

    // The activation scale tiles, in the bytes of the stage that has not been
    // issued yet. They are read once, by `tcgen05.cp`, before that issue.
    auto &activation_scale = *reinterpret_cast<
        mixed_scale_tile (*)[kFusedSlabs][kFusedSlabScaleTiles]>(
            &weight[kFusedCompactStages - 1]);

    // The compact ring's own barriers. The resident unit keeps its two, because
    // a CTA may run either specialization at any unit and a barrier's parity is
    // the barrier's, not the engine's.
    __shared__ semaphore slab_arrived[kFusedCompactStages];
    __shared__ semaphore slab_retired[kFusedCompactStages];
    __shared__ unsigned int stream_parity[2];

    const int thread = static_cast<int>(threadIdx.x);
    if (first_unit) {
        if (thread < kFusedCompactStages) {
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
    const auto weight_scale_slot = [&](const int set, const int quad) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kFusedScaleColumnBase
            + (set * kFusedSlabScaleTiles + quad) * kRoutedScaleColumns);
    };
    const auto activation_scale_slot = [&](const int slab, const int quad) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kFusedCompactScaleColumnBase
            + (slab * kFusedSlabScaleTiles + quad) * kRoutedScaleColumns);
    };

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
            weight[index % kFusedCompactStages],
            weight_scale[index % kFusedCompactStages], packed_map, fused_scale,
            expert, index, slab_arrived[index % kFusedCompactStages]);
    };

    for (int assignment_offset = 0; assignment_offset < batch_rows;
         assignment_offset += kFusedN) {
        const int batch_begin = assignment_begin + assignment_offset;
        const int rows = min(kFusedN, batch_rows - assignment_offset);
        const bool last_pass = assignment_offset + kFusedN >= batch_rows;

        // Two transfers, not three: the third stage is still the scale tiles'.
        unsigned long long fine = clocks.now();
        if (thread == 0) {
            issue(0);
            issue(1);
        }
        fine = clocks.lap(kClockRoutedGateUpTmaIssue, fine);

        stage_fused_compact_activation(
            activation, activation_scale, scratch, batch_begin, rows);
        // Every thread wrote some of the tile above, so every thread owes the
        // asynchronous proxy a fence before the barrier that orders those
        // writes ahead of warp 0's `tcgen05` reads.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();

        // All 28 activation scale buffers, once. The wait is what makes the
        // third stage's bytes reusable, so it is a CTA-wide fact rather than
        // warp 0's.
        if (warpid() == 0) {
            if (laneid() == 0) {
                for (int slab = 0; slab < kFusedSlabs; ++slab) {
                    #pragma unroll
                    for (int quad = 0; quad < kFusedSlabScaleTiles; ++quad) {
                        auto staged = activation_scale_slot(slab, quad);
                        load_mxnv_scale_async(
                            staged, activation_scale[slab][quad]);
                    }
                }
            }
            tensor_store_wait();
        }
        __syncthreads();
        clocks.lap(kClockRoutedGateUpActivation, fine);

        unsigned long long third = clocks.now();
        if (thread == 0) {
            issue(kFusedCompactStages - 1);
        }
        clocks.lap(kClockRoutedGateUpTmaIssue, third);

        // How far the stream has been retired, exactly as the resident unit
        // tracks it: two places retire and both refill, so they share one
        // high-water mark rather than each assuming the other did not run.
        int retired_upto = -1;

        for (int task = 0; task < kFusedTasks; ++task) {
            if (warpid() == 0) {
                const int lane = static_cast<int>(laneid());
                unsigned long long mark = clocks.now();
                unsigned long long inner = clocks.now();

                for (int slab = 0; slab < kFusedSlabs; ++slab) {
                    const int index = task * kFusedSlabs + slab;
                    const int stage = index % kFusedCompactStages;
                    const int set = index % kFusedScaleSets;

                    wait(slab_arrived[stage], take_phase(arrived_phase, stage));
                    mark = clocks.lap(kClockRoutedGateUpStage, mark);
                    inner = clocks.lap(kClockRoutedGateUpTmaWait, inner);

                    // Four copies per slab, not eight: the activation's 28 are
                    // already in tensor memory and stay there for the pass.
                    if (lane == 0) {
                        #pragma unroll
                        for (int quad = 0; quad < kFusedSlabScaleTiles;
                             ++quad) {
                            auto staged = weight_scale_slot(set, quad);
                            load_mxnv_scale_async(
                                staged, weight_scale[stage][quad]);
                        }
                    }
                    tensor_store_wait();

                    if (lane == 0) {
                        st_descriptor<fused_weight_tile, transpose::N>
                            weight_desc(weight[stage]);
                        st_descriptor<fused_compact_activation_tile,
                                      transpose::N>
                            activation_desc(activation);
                        #pragma unroll
                        for (int group = 0; group < kFusedSlabGroups; ++group) {
                            const int quad = group / kScaleGroupsPerTile;
                            const int factor = group % kScaleGroupsPerTile;
                            fused_mixed_mma(
                                accumulator,
                                weight_desc.chunk_descriptor(group),
                                fused_compact_operand(
                                    activation_desc.chunk_descriptor(group),
                                    slab),
                                weight_scale_slot(set, quad),
                                activation_scale_slot(slab, quad),
                                factor,
                                slab != 0 || group != 0);
                        }
                        detail::tcgen05::commit<1>(slab_retired[stage]);
                    }
                    inner = clocks.lap(kClockRoutedGateUpMmaIssue, inner);

                    // Deferred retire, one index behind, refilling two ahead
                    // rather than one: the stage index `i - 1` frees is the one
                    // index `i + 2` writes, which is what the third stage buys.
                    if (index - 1 > retired_upto) {
                        const int retiring =
                            (index - 1) % kFusedCompactStages;
                        wait(slab_retired[retiring],
                             take_phase(retired_phase, retiring));
                        retired_upto = index - 1;
                        mark = clocks.lap(kClockRoutedGateUpMma, mark);
                        inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                        if (index + 2 < kFusedStreamLength && lane == 0) {
                            issue(index + 2);
                        }
                        inner = clocks.lap(kClockRoutedGateUpTmaIssue, inner);
                    }

                    // The epilogue reads tensor memory, so this task's last
                    // slab is the one index the deferred retire cannot cover.
                    if (slab == kFusedSlabs - 1 && index > retired_upto) {
                        wait(slab_retired[stage],
                             take_phase(retired_phase, stage));
                        retired_upto = index;
                        mark = clocks.lap(kClockRoutedGateUpMma, mark);
                        inner = clocks.lap(kClockRoutedGateUpRingFull, inner);
                        if (index + 3 < kFusedStreamLength && lane == 0) {
                            issue(index + 3);
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

            if (last_pass) {
                persistent::publish_count_at(arrival_counter);
            }
        }
    }
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
