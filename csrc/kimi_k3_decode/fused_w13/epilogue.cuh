#pragma once

/// The epilogue every ring shares: drain, `situ`, quantize, publish.
///
/// A finished task's accumulator is read out of tensor memory into a shared
/// result tile, the gate and up halves are paired row `r` with row `r + 64`, the
/// exact `situ` expression is evaluated in FP32, and the result is quantized
/// back to MXFP8 into the latent scratch the grouped down pipeline reads.
///
/// Shared by all three rings verbatim. That is what makes their outputs
/// comparable byte for byte rather than merely close.

#include "contraction.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

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

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
