#pragma once

/// The whole-K activation gather: the whole batch, all of K, once.
///
/// The resident ring holds an expert's entire activation, so it gathers once per
/// eight-row pass and never again. The compact ring holds all seven slabs too,
/// in a quarter of the bytes, and reuses the scale staging here; the
/// slab-buffered ring gathers a slab at a time and has its own staging in
/// `slab_unit.cuh`.

#include "epilogue.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

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

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
