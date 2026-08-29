#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace kimi_k3_decode {
namespace expert_mxfp4 {

inline constexpr int kMmaM = 128;
inline constexpr int kMmaN = 128;
inline constexpr int kMmaK = 32;
inline constexpr int kScaleRows = 32;
inline constexpr int kScaleColumns = 16;

using mixed_operand_tile = kittens::st_fp8e4m3<kMmaM, kMmaK>;
using mixed_scale_tile =
    kittens::st_fp8e8m0<kScaleRows, kScaleColumns, false>;
using mixed_accumulator_tile = kittens::tt_fl<kMmaM, kMmaN>;
using mixed_result_tile = kittens::st_fl<kMmaM, kMmaN>;

// PTX ISA 9.1, "tcgen05.mma instruction descriptor":
//   a_format [7:10) = 0 (E4M3)
//   b_format [10:13) = 5 (E2M1)
//   n_dim [17:23) = N / 8
//   scale_format [23] = 1 (UE8M0)
//   m_dim [24:29) = M / 16
//   a_sf_id [29:31), b_sf_id [4:6)
//   k_size [31] = 0 (dense MXF8F6F4 K=32)
template<int SCALE_FACTOR_ID>
__device__ __forceinline__ constexpr std::uint32_t
mixed_instruction_descriptor() {
    static_assert(SCALE_FACTOR_ID >= 0 && SCALE_FACTOR_ID < 4);
    return (static_cast<std::uint32_t>(SCALE_FACTOR_ID) << 4)
         | (0u << 7)
         | (5u << 10)
         | (static_cast<std::uint32_t>(kMmaN / 8) << 17)
         | (1u << 23)
         | (static_cast<std::uint32_t>(kMmaM / 16) << 24)
         | (static_cast<std::uint32_t>(SCALE_FACTOR_ID) << 29);
}

static_assert(mixed_instruction_descriptor<0>() == 0x08a01400u);

/// CUTLASS's SM103 `Sm103BlockScaledBasicChunk<32>::SfKMajorAtom`:
/// shape ((8,4,4),(32,4)), stride ((16,128,4),(0,1)).
__host__ __device__ __forceinline__ constexpr int
scale_factor_1x_offset(const int row, const int k_group) {
    return (row % 8) * 16
         + ((row / 8) % 4) * 128
         + (row / 32) * 4
         + k_group;
}

/// Issue one K=32 block-scaled contraction into `destination`.
///
/// `accumulate` is a run-time predicate rather than a template argument
/// because a unit now issues a whole round of K groups from one unrolled body:
/// only the very first issue of a unit clears the accumulator, and branching
/// the body on that would duplicate every instruction in the round.
template<int SCALE_FACTOR_ID>
__device__ __forceinline__ void mixed_mma(
    const mixed_accumulator_tile &destination,
    const mixed_operand_tile &a,
    const mixed_operand_tile &b,
    const kittens::full_tt_fp8e8m0<16> &scale_a,
    const kittens::full_tt_fp8e8m0<16> &scale_b,
    const bool accumulate
) {
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N> a_desc(a);
    kittens::st_descriptor<mixed_operand_tile, kittens::transpose::N> b_desc(b);
    constexpr std::uint32_t instruction =
        mixed_instruction_descriptor<SCALE_FACTOR_ID>();
    // The operands are populated by ordinary shared-memory stores. Publish
    // those writes to the asynchronous tcgen05 proxy before every issue; a CTA
    // barrier alone does not establish this cross-proxy ordering.
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
          "l"(a_desc.base_desc),
          "l"(b_desc.base_desc),
          "r"(instruction),
          "r"(scale_a.addr),
          "r"(scale_b.addr),
          "r"(accumulate ? 1u : 0u)
    );
}

__device__ __forceinline__ std::uint8_t quantize_e4m3(
    const float value,
    const float reciprocal
) {
    std::uint16_t pair;
    const float scaled = value * reciprocal;
    asm volatile(
        "{cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;}"
        : "=h"(pair)
        : "f"(scaled)
    );
    return static_cast<std::uint8_t>(pair);
}

// OCP MX v1.0 and PTX ISA 9.3 both define an E8M0 scale as 2^(byte - 127) with
// byte 255 reserved for NaN, so byte 0 is the exact minimum scale 2^-127 and
// byte 254 the maximum.
inline constexpr unsigned int kMinE8M0ScaleByte = 0u;
inline constexpr unsigned int kUnitE8M0ScaleByte = 0x7fu;
inline constexpr unsigned int kMaxE8M0ScaleByte = 254u;

// E4M3 tops out at 448 = 1.75 * 2^8, so a significand above 1.75 needs one more
// binade of headroom than one at or below it.
inline constexpr unsigned int kOneAndThreeQuartersMantissa = 0x600000u;

/// Return the E8M0 byte whose scale keeps every E4M3 magnitude at most 448.
///
/// The exponent is derived from the input's own bits rather than from
/// `absolute_max * (1/448)`. That product underflows toward zero for small
/// blocks, so it needed a floor, and any floor pins every block beneath it to
/// one coarse scale and flushes those activations to zero instead of using the
/// scales E8M0 actually has. Working on the exponent keeps the full range
/// reachable and also avoids the rounding of the 1/448 multiply near a binade
/// boundary.
__device__ __forceinline__ std::uint8_t select_e8m0_scale(
    const float absolute_max
) {
    if (absolute_max == 0.0f) {
        return static_cast<std::uint8_t>(kUnitE8M0ScaleByte);
    }
    const unsigned int bits = __float_as_uint(absolute_max);
    const unsigned int exponent_field = (bits >> 23) & 0xffu;
    // Subnormal magnitudes are below 2^-126, and 2^-126 / 448 is already below
    // 2^-134, so the minimum scale is the only available answer.
    if (exponent_field == 0u) {
        return static_cast<std::uint8_t>(kMinE8M0ScaleByte);
    }
    const int exponent = static_cast<int>(exponent_field) - 127;
    const unsigned int mantissa = bits & 0x7fffffu;
    const int scale_exponent =
        (mantissa <= kOneAndThreeQuartersMantissa) ? exponent - 8
                                                  : exponent - 7;
    return static_cast<std::uint8_t>(
        min(max(scale_exponent + 127, static_cast<int>(kMinE8M0ScaleByte)),
            static_cast<int>(kMaxE8M0ScaleByte)));
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void mixed_mma_probe_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const std::uint8_t *__restrict__ b_packed,
    const std::uint8_t *__restrict__ b_scale,
    float *__restrict__ output,
    const int rows
) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];
    tma_swizzle_allocator shared_allocator(shared_raw);
    mixed_operand_tile (&a_tile) =
        shared_allocator.allocate<mixed_operand_tile>();
    mixed_operand_tile (&b_tile) =
        shared_allocator.allocate<mixed_operand_tile>();
    mixed_scale_tile (&scale_a_shared) =
        shared_allocator.allocate<mixed_scale_tile>();
    mixed_scale_tile (&scale_b_shared) =
        shared_allocator.allocate<mixed_scale_tile>();
    mixed_result_tile (&result_shared) =
        shared_allocator.allocate<mixed_result_tile>();

    const int thread = static_cast<int>(threadIdx.x);
    for (int row = thread; row < kMmaM; row += kDecodeCtaThreads) {
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const float value =
                row < rows
                    ? __bfloat162float(a[row * kMmaK + k])
                    : 0.0f;
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale = select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float((254u - static_cast<unsigned int>(scale)) << 23);
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            *reinterpret_cast<std::uint8_t *>(&a_tile[{row, k}]) =
                quantize_e4m3(values[k], reciprocal);
        }
    }

    for (int index = thread; index < kMmaN * kMmaK;
         index += kDecodeCtaThreads) {
        const int row = index / kMmaK;
        const int shared_column = index % kMmaK;
        const int column_in_16b_atom = shared_column % 16;
        std::uint8_t value = 0;
        if (column_in_16b_atom < 8) {
            // CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B places sixteen packed U4
            // values (eight bytes) at the front of each 16-byte SMEM atom.
            const int packed_column =
                (shared_column / 16) * 8 + column_in_16b_atom;
            value = b_packed[row * (kMmaK / 2) + packed_column];
        }
        *reinterpret_cast<std::uint8_t *>(&b_tile[{row, shared_column}]) =
            value;
    }

    for (int index = thread; index < kScaleRows * kScaleColumns;
         index += kDecodeCtaThreads) {
        reinterpret_cast<std::uint8_t *>(scale_a_shared.data)[index] = 0x7fu;
        reinterpret_cast<std::uint8_t *>(scale_b_shared.data)[index] = 0x7fu;
    }
    __syncthreads();

    for (int row = thread; row < kMmaM; row += kDecodeCtaThreads) {
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            if (row < rows) {
                absolute_max = fmaxf(
                    absolute_max,
                    fabsf(__bfloat162float(a[row * kMmaK + k])));
            }
        }
        reinterpret_cast<std::uint8_t *>(scale_a_shared.data)
            [scale_factor_1x_offset(row, 0)] =
                select_e8m0_scale(absolute_max);
        reinterpret_cast<std::uint8_t *>(scale_b_shared.data)
            [scale_factor_1x_offset(row, 0)] = b_scale[row];
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    __shared__ semaphore compute_done;
    if (thread == 0) init_semaphore(compute_done, 0, 1);
    __syncthreads();

    tensor_allocator<1, 1> tensor_pool{};
    mixed_accumulator_tile accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    auto scale_a =
        tensor_pool.allocate<full_tt_fp8e8m0<16>>(256);
    auto scale_b =
        tensor_pool.allocate<full_tt_fp8e8m0<16>>(260);

    if (warpid() == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        load_mxnv_scale_async(scale_a, scale_a_shared);
        load_mxnv_scale_async(scale_b, scale_b_shared);
        tensor_store_wait();
    }
    __syncthreads();

    if (thread == 0) {
        mixed_mma<0>(accumulator, a_tile, b_tile, scale_a, scale_b, false);
        detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);
    if (warpgroup::groupid() == 0) {
        rt_fl<kMmaM / 4, kMmaN> result;
        warpgroup::load_async(result, accumulator);
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::store(result_shared, result);
    }
    __syncthreads();

    for (int index = thread; index < rows * kMmaN;
         index += kDecodeCtaThreads) {
        output[index] = result_shared[{index / kMmaN, index % kMmaN}];
    }
}

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
// ---------------------------------------------------------------------------

/// One shared tile row's two sixteen-byte atoms.
__device__ __forceinline__ uint4 *atom_of(
    mixed_operand_tile &tile,
    const int row,
    const int atom
) {
    return reinterpret_cast<uint4 *>(&tile[{row, atom * 16}]);
}

/// Fill one operand tile with the bytes no K group ever rewrites.
///
/// Two kinds of byte live here for a whole unit. An MXFP4 row only occupies
/// the front eight bytes of each atom -- the packed layout the MMA reads is
/// `16U4_ALIGN16B` -- and a batch shorter than the MMA's 128 rows leaves whole
/// rows out of the contraction. Both are zero for every K group, so writing
/// them once per unit replaces writing them 112 times.
__device__ __forceinline__ void clear_operand_tile(mixed_operand_tile &tile) {
    uint4 *const raw = reinterpret_cast<uint4 *>(tile.data);
    constexpr int vectors = kMmaM * kMmaK / 16;
    for (int index = static_cast<int>(threadIdx.x); index < vectors;
         index += kDecodeCtaThreads) {
        raw[index] = make_uint4(0u, 0u, 0u, 0u);
    }
}

/// Fill one scale tile with the unit scale its padding holds for a whole unit.
///
/// Only every fourth byte of a `scale_vec::1X` tile is a live scale factor.
/// The staging below writes a live byte and its three padding bytes as one
/// word, so this only has to cover the rows a short batch never writes.
__device__ __forceinline__ void clear_scale_tile(mixed_scale_tile &tile) {
    std::uint32_t *const raw =
        reinterpret_cast<std::uint32_t *>(tile.data);
    constexpr int words = kScaleRows * kScaleColumns / 4;
    for (int index = static_cast<int>(threadIdx.x); index < words;
         index += kDecodeCtaThreads) {
        raw[index] = 0x7f7f7f7fu;
    }
}

/// Write one row's live scale factor without disturbing its padding.
///
/// `scale_factor_1x_offset` is always a multiple of four and the live byte is
/// the low byte of that word, so one word store carries the factor and repeats
/// the unit scale the three padding bytes already held.
__device__ __forceinline__ void stage_scale(
    mixed_scale_tile &scale_shared,
    const int row,
    const std::uint8_t scale
) {
    reinterpret_cast<std::uint32_t *>(
        scale_shared.data)[scale_factor_1x_offset(row, 0) / 4] =
            0x7f7f7f00u | static_cast<std::uint32_t>(scale);
}

/// One byte of a round's eight consecutive scale factors.
///
/// The round reads them as one vector and the caller unrolls over `slot`, so
/// this is a register select rather than an addressed load.
__device__ __forceinline__ std::uint8_t scale_byte(
    const uint2 payload,
    const int slot
) {
    const std::uint32_t word = slot < 4 ? payload.x : payload.y;
    return static_cast<std::uint8_t>(word >> (8 * (slot & 3)));
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
/// `16U4_ALIGN16B` puts sixteen packed U4 values at the front of each atom, so
/// the row's first eight source bytes head the first atom and its last eight
/// head the second. The rest of each atom is the zero `clear_operand_tile`
/// already wrote and the MMA never reads.
__device__ __forceinline__ void stage_weight_row(
    mixed_operand_tile &tile,
    const int row,
    const uint4 payload
) {
    *reinterpret_cast<uint2 *>(atom_of(tile, row, 0)) =
        make_uint2(payload.x, payload.y);
    *reinterpret_cast<uint2 *>(atom_of(tile, row, 1)) =
        make_uint2(payload.z, payload.w);
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
// One routed-expert work unit.
//
// A unit is one 128-column output tile of one expert's assignment batch: three
// gate/up units and twenty-eight down units per batch. The private stage below
// walks every unit of every expert from a single CTA; the persistent kernel
// hands the same units out through a device task queue. Both call the two
// functions here, so the fallback and the production path contract identical
// arithmetic in identical order.
//
// Each unit re-derives its shared tiles from the same base and re-initializes
// its own semaphore, which is safe because a unit always waits for its last
// commit before returning: nothing is ever in flight across the boundary.
// ---------------------------------------------------------------------------

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

/// First tensor-memory column the routed scale factors occupy.
///
/// The two 128-column accumulators come first, so the scales start above them
/// and each `full_tt_fp8e8m0<16>` takes four columns.
inline constexpr int kRoutedScaleColumnBase = 2 * kMmaN;
inline constexpr int kRoutedScaleColumns = 4;
inline constexpr int kGateUpScaleBuffers = 3 * kGateUpRoundGroups;
inline constexpr int kDownScaleBuffers = 2 * kDownRoundGroups;

static_assert(kRoutedScaleColumnBase
                  + kRoutedScaleColumns * kGateUpScaleBuffers
              <= kittens::tensor_allocator<1, 1>::cols);
static_assert(kRoutedScaleColumnBase
                  + kRoutedScaleColumns * kDownScaleBuffers
              <= kittens::tensor_allocator<1, 1>::cols);

static_assert(sizeof(mixed_operand_tile) == 4096);
static_assert(sizeof(mixed_scale_tile) == 512);
static_assert(sizeof(mixed_result_tile) == 65536);

/// Shared bytes one round of gate/up staging occupies.
///
/// The allocator aligns each *array* to 1 KiB, and every tile size here is
/// already a multiple of 1 KiB, so the arrays pack without padding.
inline constexpr int kGateUpStagingBytes =
    kGateUpRoundGroups
    * (3 * static_cast<int>(sizeof(mixed_operand_tile))
       + 3 * static_cast<int>(sizeof(mixed_scale_tile)));
inline constexpr int kDownStagingBytes =
    kDownRoundGroups
    * (2 * static_cast<int>(sizeof(mixed_operand_tile))
       + 2 * static_cast<int>(sizeof(mixed_scale_tile)));

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

static_assert(kGateUpStagingBytes == 110592);
static_assert(kDownStagingBytes == 110592);
static_assert(kGateUpUnitSharedBytes == 131072);
static_assert(kDownUnitSharedBytes == 110592);

/// Contract one expert batch's gate and up tiles and stage the SiTU result.
///
/// `tensor_pool` is owned by the caller because a CTA may allocate tensor
/// memory only once: the persistent kernel provisions one pool at entry and
/// hands it to every unit, and the private kernel provisions one of its own.
static __device__ void routed_gate_up_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const std::uint8_t *__restrict__ expert_w1_packed,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_packed,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const Scratch &scratch,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const int output_tile,
    const PhaseClocks clocks
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_operand_tile (&first_weight_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_operand_tile (&second_weight_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_scale_tile (&activation_scale_shared)[kGateUpRoundGroups] =
        staging.allocate<mixed_scale_tile, kGateUpRoundGroups>();
    mixed_scale_tile (&first_scale_shared)[kGateUpRoundGroups] =
        staging.allocate<mixed_scale_tile, kGateUpRoundGroups>();
    mixed_scale_tile (&second_scale_shared)[kGateUpRoundGroups] =
        staging.allocate<mixed_scale_tile, kGateUpRoundGroups>();

    // The results are laid over the staging, which is dead by the time the
    // accumulators are read out.
    tma_swizzle_allocator results(shared_raw);
    mixed_result_tile (&first_result_shared) =
        results.allocate<mixed_result_tile>();
    mixed_result_tile (&second_result_shared) =
        results.allocate<mixed_result_tile>();

    __shared__ semaphore gate_up_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(gate_up_done, 0, 1);

    mixed_accumulator_tile first_accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    mixed_accumulator_tile second_accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(kMmaN);
    const auto scale_slot = [&](const int buffer) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kRoutedScaleColumnBase + buffer * kRoutedScaleColumns);
    };

    #pragma unroll
    for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
        clear_operand_tile(activation_tile[slot]);
        clear_operand_tile(first_weight_tile[slot]);
        clear_operand_tile(second_weight_tile[slot]);
        clear_scale_tile(activation_scale_shared[slot]);
    }
    __syncthreads();

    // One MMA batch is 128 rows and an expert never collects more, because a
    // token's sixteen routes are sixteen distinct experts. The staging indexes
    // the tile directly rather than guarding each row, so the bound is taken
    // here rather than assumed.
    const int rows = min(batch_rows, kMmaM);

    // One thread owns one row of one of the two weight tiles for the whole
    // unit, which is what makes a round's reads of that row contiguous.
    const int weight_half = thread / kMmaN;
    const int weight_row = thread % kMmaN;
    const std::uint8_t *const weight_packed =
        weight_half == 0 ? expert_w1_packed : expert_w3_packed;
    const std::uint8_t *const weight_scales =
        weight_half == 0 ? expert_w1_scale : expert_w3_scale;
    mixed_operand_tile *const weight_tile =
        weight_half == 0 ? first_weight_tile : second_weight_tile;
    mixed_scale_tile *const weight_scale_shared =
        weight_half == 0 ? first_scale_shared : second_scale_shared;
    const int output_base = output_tile * kMmaN;
    const long long weight_index =
        static_cast<long long>(expert) * kExpertW1W3PackedRows
        + output_base + weight_row;
    const std::uint8_t *const weight_row_bytes =
        weight_packed + weight_index * kExpertW1W3PackedColumns;
    const std::uint8_t *const weight_row_scales =
        weight_scales + weight_index * kExpertW1W3ScaleColumns;

    int compute_phase = 0;
    unsigned long long mark = clocks.now();
    for (int round = 0; round < kGateUpRounds; ++round) {
        const int group_base = round * kGateUpRoundGroups;

        // Every global read of the round is issued before any of them is
        // consumed, so the round's misses overlap instead of serializing.
        uint4 payload[kGateUpRoundGroups];
        #pragma unroll
        for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
            payload[slot] = *reinterpret_cast<const uint4 *>(
                weight_row_bytes + (group_base + slot) * (kMmaK / 2));
        }
        const uint2 scale_payload = *reinterpret_cast<const uint2 *>(
            weight_row_scales + group_base);
        #pragma unroll
        for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
            stage_weight_row(
                weight_tile[slot], weight_row, payload[slot]);
            stage_scale(
                weight_scale_shared[slot], weight_row,
                scale_byte(scale_payload, slot));
        }

        // The batch is at most 128 rows and usually one, so the activation is
        // spread over whatever threads its rows and groups need.
        for (int index = thread; index < rows * kGateUpRoundGroups;
             index += kDecodeCtaThreads) {
            const int row = index / kGateUpRoundGroups;
            const int slot = index % kGateUpRoundGroups;
            const int group = group_base + slot;
            const int token =
                scratch.assignment_tokens[assignment_begin + row];
            stage_activation_row(
                activation_tile[slot], row,
                scratch.latent_mxfp8
                    + static_cast<long long>(token) * kLatentSize
                    + group * kMmaK);
            stage_scale(
                activation_scale_shared[slot], row,
                scratch.latent_scale[
                    static_cast<long long>(token) * kLatentGroups + group]);
        }

        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        mark = clocks.lap(kClockRoutedGateUpStage, mark);
        // `tcgen05.cp`, `tcgen05.mma`, and `tcgen05.commit` are single-thread
        // issues, but `tcgen05.wait::st` is `.sync.aligned`, so the whole warp
        // has to reach it.
        if (warpid() == 0) {
            if (laneid() == 0) {
                #pragma unroll
                for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
                    auto activation_scale = scale_slot(slot);
                    auto first_scale = scale_slot(kGateUpRoundGroups + slot);
                    auto second_scale =
                        scale_slot(2 * kGateUpRoundGroups + slot);
                    load_mxnv_scale_async(
                        activation_scale, activation_scale_shared[slot]);
                    load_mxnv_scale_async(
                        first_scale, first_scale_shared[slot]);
                    load_mxnv_scale_async(
                        second_scale, second_scale_shared[slot]);
                }
            }
            tensor_store_wait();
            if (laneid() == 0) {
                #pragma unroll
                for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
                    const bool accumulate = round != 0 || slot != 0;
                    mixed_mma<0>(
                        first_accumulator, activation_tile[slot],
                        first_weight_tile[slot], scale_slot(slot),
                        scale_slot(kGateUpRoundGroups + slot), accumulate);
                    mixed_mma<0>(
                        second_accumulator, activation_tile[slot],
                        second_weight_tile[slot], scale_slot(slot),
                        scale_slot(2 * kGateUpRoundGroups + slot), accumulate);
                }
                detail::tcgen05::commit<1>(gate_up_done);
            }
        }
        wait(gate_up_done, compute_phase);
        __syncthreads();
        mark = clocks.lap(kClockRoutedGateUpMma, mark);
        compute_phase ^= 1;
    }

    store_accumulator(first_accumulator, first_result_shared);
    store_accumulator(second_accumulator, second_result_shared);
    quantize_situ_tile(
        first_result_shared, second_result_shared, scratch, assignment_begin,
        rows, output_base);
    __syncthreads();
}

/// Contract one expert batch's down tile and weight it into the accumulator.
static __device__ void routed_down_unit(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    const Scratch &scratch,
    const int expert,
    const int assignment_begin,
    const int batch_rows,
    const int output_tile,
    const int active_tokens,
    const PhaseClocks clocks
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kDownRoundGroups] =
        staging.allocate<mixed_operand_tile, kDownRoundGroups>();
    mixed_operand_tile (&weight_tile)[kDownRoundGroups] =
        staging.allocate<mixed_operand_tile, kDownRoundGroups>();
    mixed_scale_tile (&activation_scale_shared)[kDownRoundGroups] =
        staging.allocate<mixed_scale_tile, kDownRoundGroups>();
    mixed_scale_tile (&weight_scale_shared)[kDownRoundGroups] =
        staging.allocate<mixed_scale_tile, kDownRoundGroups>();

    tma_swizzle_allocator results(shared_raw);
    mixed_result_tile (&result_shared) =
        results.allocate<mixed_result_tile>();

    __shared__ semaphore down_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(down_done, 0, 1);

    mixed_accumulator_tile accumulator =
        tensor_pool.allocate<mixed_accumulator_tile>(0);
    const auto scale_slot = [&](const int buffer) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            kRoutedScaleColumnBase + buffer * kRoutedScaleColumns);
    };

    #pragma unroll
    for (int slot = 0; slot < kDownRoundGroups; ++slot) {
        clear_operand_tile(activation_tile[slot]);
        clear_operand_tile(weight_tile[slot]);
        clear_scale_tile(activation_scale_shared[slot]);
    }
    __syncthreads();

    const int rows = min(batch_rows, kMmaM);

    // A W2 row is only 192 packed bytes, so a round is every K group the SiTU
    // width has and one thread carries half a row's groups.
    constexpr int kDownGroupsPerThread = kDownRoundGroups / 2;
    const int weight_row = thread % kMmaN;
    const int weight_half = thread / kMmaN;
    const int weight_group_base = weight_half * kDownGroupsPerThread;
    const long long weight_index =
        static_cast<long long>(expert) * kExpertW2PackedRows
        + output_tile * kMmaN + weight_row;
    const std::uint8_t *const weight_row_bytes =
        expert_w2_packed + weight_index * kExpertW2PackedColumns;
    const std::uint8_t *const weight_row_scales =
        expert_w2_scale + weight_index * kExpertW2ScaleColumns;

    const int output_base = output_tile * kMmaN;
    unsigned long long mark = clocks.now();

    uint4 payload[kDownGroupsPerThread];
    #pragma unroll
    for (int slot = 0; slot < kDownGroupsPerThread; ++slot) {
        payload[slot] = *reinterpret_cast<const uint4 *>(
            weight_row_bytes + (weight_group_base + slot) * (kMmaK / 2));
    }
    std::uint8_t weight_scale_bytes[kDownGroupsPerThread];
    #pragma unroll
    for (int slot = 0; slot < kDownGroupsPerThread; ++slot) {
        weight_scale_bytes[slot] =
            weight_row_scales[weight_group_base + slot];
    }
    #pragma unroll
    for (int slot = 0; slot < kDownGroupsPerThread; ++slot) {
        const int group = weight_group_base + slot;
        stage_weight_row(weight_tile[group], weight_row, payload[slot]);
        stage_scale(
            weight_scale_shared[group], weight_row, weight_scale_bytes[slot]);
    }

    for (int index = thread; index < rows * kDownRoundGroups;
         index += kDecodeCtaThreads) {
        const int row = index / kDownRoundGroups;
        const int group = index % kDownRoundGroups;
        const int assignment = assignment_begin + row;
        stage_activation_row(
            activation_tile[group], row,
            scratch.situ_mxfp8
                + static_cast<long long>(assignment)
                      * kRoutedIntermediateSizePerRank
                + group * kMmaK);
        stage_scale(
            activation_scale_shared[group], row,
            scratch.situ_scale[
                static_cast<long long>(assignment) * kSituGroups + group]);
    }

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    mark = clocks.lap(kClockRoutedDownStage, mark);
    if (warpid() == 0) {
        if (laneid() == 0) {
            #pragma unroll
            for (int slot = 0; slot < kDownRoundGroups; ++slot) {
                auto activation_scale = scale_slot(slot);
                auto weight_scale = scale_slot(kDownRoundGroups + slot);
                load_mxnv_scale_async(
                    activation_scale, activation_scale_shared[slot]);
                load_mxnv_scale_async(
                    weight_scale, weight_scale_shared[slot]);
            }
        }
        tensor_store_wait();
        if (laneid() == 0) {
            #pragma unroll
            for (int slot = 0; slot < kDownRoundGroups; ++slot) {
                mixed_mma<0>(
                    accumulator, activation_tile[slot], weight_tile[slot],
                    scale_slot(slot), scale_slot(kDownRoundGroups + slot),
                    slot != 0);
            }
            detail::tcgen05::commit<1>(down_done);
        }
    }
    wait(down_done, 0);
    __syncthreads();
    clocks.lap(kClockRoutedDownMma, mark);

    store_accumulator(accumulator, result_shared);
    accumulate_down_tile(
        result_shared, scratch, assignment_begin, rows, output_base,
        active_tokens);
    __syncthreads();
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void kimi_k3_routed_experts_kernel(
    const __nv_bfloat16 *__restrict__ latent_x,
    const std::uint8_t *__restrict__ expert_w1_packed,
    const std::uint8_t *__restrict__ expert_w1_scale,
    const std::uint8_t *__restrict__ expert_w3_packed,
    const std::uint8_t *__restrict__ expert_w3_scale,
    const std::uint8_t *__restrict__ expert_w2_packed,
    const std::uint8_t *__restrict__ expert_w2_scale,
    __nv_bfloat16 *__restrict__ routed_output,
    std::uint8_t *__restrict__ scratch_bytes,
    const int active_tokens,
    const int tokens
) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];

    // The managed allocator barriers the whole CTA, so provisioning it here
    // covers every unit this block later runs.
    tensor_allocator<1, 1> tensor_pool{};

    const Scratch scratch = scratch_view(scratch_bytes);
    const int thread = static_cast<int>(threadIdx.x);
    const int active_values = active_tokens * kLatentSize;
    for (int index = thread; index < active_values;
         index += kDecodeCtaThreads) {
        scratch.routed_accumulator[index] = 0.0f;
    }
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int index = thread; index < tokens * kLatentSize;
         index += kDecodeCtaThreads) {
        routed_output[index] = zero;
    }
    __syncthreads();

    quantize_latent_rows(latent_x, scratch, active_tokens, 0, 1);
    // Every thread flushes its own MXFP8 writes, and the barrier then holds the
    // publishing thread until all of those flushes have landed device-wide, so a
    // consumer that sees the new generation cannot read a stale block.
    __threadfence();
    __syncthreads();
    if (thread == 0) {
        atomicExch(&scratch.phase[kExpertQuantizationArrivals], 0);
        atomicAdd(&scratch.phase[kExpertQuantizationGeneration], 1);
    }
    __syncthreads();

    const int total_assignments =
        min(max(scratch.expert_offsets[kNumExperts], 0), kMaxRoutes);
    for (int expert = 0; expert < kNumExperts; ++expert) {
        const int expert_begin =
            min(max(scratch.expert_offsets[expert], 0), total_assignments);
        const int expert_end =
            min(max(scratch.expert_offsets[expert + 1], expert_begin),
                total_assignments);
        if (expert_begin == expert_end) continue;

        for (int assignment_begin = expert_begin;
             assignment_begin < expert_end;
             assignment_begin += kMmaM) {
            const int batch_rows =
                min(kMmaM, expert_end - assignment_begin);

            for (int output_tile = 0; output_tile < kGateUpTiles;
                 ++output_tile) {
                routed_gate_up_unit(
                    shared_raw, tensor_pool, expert_w1_packed, expert_w1_scale,
                    expert_w3_packed, expert_w3_scale, scratch, expert,
                    assignment_begin, batch_rows, output_tile,
                    PhaseClocks{nullptr});
            }

            for (int output_tile = 0; output_tile < kDownTiles;
                 ++output_tile) {
                routed_down_unit(
                    shared_raw, tensor_pool, expert_w2_packed, expert_w2_scale,
                    scratch, expert, assignment_begin, batch_rows, output_tile,
                    active_tokens, PhaseClocks{nullptr});
            }
        }
    }

    for (int index = thread; index < active_values;
         index += kDecodeCtaThreads) {
        routed_output[index] =
            __float2bfloat16(scratch.routed_accumulator[index]);
    }
    // Same ordering for the routed partial: flush per thread, barrier, publish.
    __threadfence();
    __syncthreads();
    if (thread == 0) {
        atomicExch(&scratch.phase[kExpertCompletionArrivals], 0);
        atomicAdd(&scratch.phase[kExpertCompletionGeneration], 1);
    }
    __syncthreads();
}

// The shared allocator aligns every object independently to 1 KiB and the
// register-to-shared mapping can touch the following swizzle atom. Reserve the
// architecture-supported budget instead of under-counting either padding.
inline constexpr int kProbeSharedBytes = kittens::MAX_SHARED_MEMORY - 1024;

static __host__ void launch_routed_experts(
    const at::Tensor &latent_x,
    const at::Tensor &expert_w1_packed,
    const at::Tensor &expert_w1_scale,
    const at::Tensor &expert_w3_packed,
    const at::Tensor &expert_w3_scale,
    const at::Tensor &expert_w2_packed,
    const at::Tensor &expert_w2_scale,
    const at::Tensor &routed_output,
    const at::Tensor &scratch,
    const int active_tokens
) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kimi_k3_routed_experts_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kProbeSharedBytes));
    kimi_k3_routed_experts_kernel
        <<<1, kDecodeCtaThreads, kProbeSharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(latent_x.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w1_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w1_scale.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w3_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w3_scale.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w2_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(
                expert_w2_scale.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(routed_output.data_ptr()),
            reinterpret_cast<std::uint8_t *>(scratch.data_ptr()),
            active_tokens,
            static_cast<int>(latent_x.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __host__ at::Tensor mixed_mma_probe_entrypoint(
    const at::Tensor &a,
    const at::Tensor &b_packed,
    const at::Tensor &b_scale
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b_packed);
    CHECK_INPUT(b_scale);
    TORCH_CHECK(a.dim() == 2 && a.size(0) >= 1 && a.size(0) <= kMmaM
                    && a.size(1) == kMmaK
                    && a.scalar_type() == at::kBFloat16,
                "MoK: _kimi_k3_mixed_mma_probe requires BF16 A [1..128, 32]");
    TORCH_CHECK(b_packed.dim() == 2 && b_packed.size(0) == kMmaN
                    && b_packed.size(1) == kMmaK / 2
                    && b_packed.scalar_type() == at::kByte,
                "MoK: _kimi_k3_mixed_mma_probe requires uint8 B [128, 16]");
    TORCH_CHECK(b_scale.dim() == 2 && b_scale.size(0) == kMmaN
                    && b_scale.size(1) == 1
                    && b_scale.scalar_type() == at::kByte,
                "MoK: _kimi_k3_mixed_mma_probe requires uint8 B scale [128, 1]");
    TORCH_CHECK(a.device() == b_packed.device()
                    && a.device() == b_scale.device(),
                "MoK: _kimi_k3_mixed_mma_probe requires one CUDA device");

    const c10::cuda::CUDAGuard device_guard(a.device());
    at::Tensor output =
        at::empty({a.size(0), kMmaN}, a.options().dtype(at::kFloat));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mixed_mma_probe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kProbeSharedBytes));
    mixed_mma_probe_kernel
        <<<1, kDecodeCtaThreads, kProbeSharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(a.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(b_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(b_scale.data_ptr()),
            reinterpret_cast<float *>(output.data_ptr()),
            static_cast<int>(a.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
