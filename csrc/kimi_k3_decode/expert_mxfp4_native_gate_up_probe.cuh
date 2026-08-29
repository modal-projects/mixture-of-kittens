#pragma once

// Benchmark-only native-style gate/up microprototype.
//
// This header deliberately has no production call site. One CTA owns one
// expert/output tile. W1 and W3 arrive through persistent tensor maps into a
// three-stage ring in their final MMA swizzle, a producer warp owns every TMA
// operation and stage reuse, and a consumer warp emits eight m128x8x32 mixed
// MMAs per four-K-group panel. Four epilogue warps read their 32-channel
// accumulator slices directly from TMEM and quantize SiTU without a full
// accumulator tile in shared memory.

#include "expert_mxfp4_batch_probe.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace native_gate_up_probe {

inline constexpr int kNativeGateUpM = 128;
inline constexpr int kNativeGateUpN = 8;
inline constexpr int kNativeGateUpK = 32;
inline constexpr int kNativePhysicalN = 16;
inline constexpr int kNativePanelGroups = 4;
inline constexpr int kNativePanels =
    kLatentGroups / kNativePanelGroups;
inline constexpr int kNativeWeightStages = 3;
inline constexpr int kNativeProducerWarp = 1;
inline constexpr int kNativeConsumerWarp = 0;
inline constexpr int kNativeEpilogueWarps = 4;
inline constexpr int kNativeProfilePhases = 4;

static_assert(kNativeGateUpM == batch_probe::kBatchProbeM);
static_assert(kNativeGateUpN == batch_probe::kBatchProbeN);
static_assert(kNativeGateUpK == kMmaK);
static_assert(kNativePanels == 28);

using native_accumulator_tile =
    kittens::tt_fl<kNativeGateUpM, kNativePhysicalN>;
using native_weight_stage =
    kittens::st_fp8e4m3<
        kNativeGateUpM, kNativePanelGroups * kNativeGateUpK, true, 128>;
using native_activation_tile =
    kittens::st_fp8e4m3<16, kNativeGateUpK, true, 32>;
using native_epilogue_tile =
    kittens::rt_fl<kNativeGateUpM / kNativeEpilogueWarps, kNativePhysicalN>;

inline constexpr int kNativeWeightTransactionBytes =
    kNativeGateUpM * kNativePanelGroups * kNativeGateUpK / 2;
inline constexpr int kNativeWeightSharedBytes =
    2 * kNativeWeightStages * static_cast<int>(sizeof(native_weight_stage));

static_assert(sizeof(native_weight_stage) == 16 * 1024);
static_assert(sizeof(native_activation_tile) == 512);
static_assert(kNativeWeightSharedBytes == 96 * 1024);

struct alignas(1024) native_scale_ring {
    mixed_scale_tile stage[kNativeWeightStages];
};

static_assert(sizeof(native_scale_ring) == 2 * 1024);

struct alignas(1024) native_shared_storage {
    native_weight_stage first_weight[kNativeWeightStages];
    native_weight_stage second_weight[kNativeWeightStages];
    native_activation_tile
        activation[kNativeWeightStages * kNativePanelGroups];
    native_scale_ring activation_scale;
    native_scale_ring first_scale;
    native_scale_ring second_scale;
};

inline constexpr int kNativeGateUpSharedBytes =
    static_cast<int>(sizeof(native_shared_storage));
inline constexpr int kNativeGateUpSharedReservationBytes = 120 * 1024;
inline constexpr int kBaselineGateUpSharedReservationBytes =
    kGateUpUnitSharedBytes;

static_assert(kNativeGateUpSharedBytes == 108 * 1024);
static_assert(kNativeGateUpSharedBytes <= 120 * 1024);
static_assert(kNativeGateUpSharedReservationBytes <= 120 * 1024);
static_assert(
    2 * kNativeGateUpSharedReservationBytes
        > kittens::MAX_SHARED_MEMORY,
    "the benchmark reservation must force one CTA per SM");

struct alignas(128) native_weight_layout {
    CUtensorMap tensor_map;
};

struct native_weight_layouts {
    native_weight_layout w1;
    native_weight_layout w3;
};

static __host__ inline native_weight_layout native_weight_layout_for(
    const std::uint8_t *pointer,
    const int experts
) {
    constexpr int packed_columns = kExpertW1W3PackedColumns;
    constexpr int rows = kExpertW1W3PackedRows;
    constexpr int panel_values = kNativePanelGroups * kNativeGateUpK;
    static_assert(panel_values == 128);
    static_assert(2 * packed_columns % panel_values == 0);

    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(pointer) % 32 == 0,
        "MoK: native gate/up weight tensor maps require 32-byte alignment");
    TORCH_CHECK(experts >= 1);

    native_weight_layout layout{};
    const std::uint64_t global_dimensions[5] = {
        panel_values,
        rows,
        static_cast<std::uint64_t>(2 * packed_columns / panel_values),
        1,
        static_cast<std::uint64_t>(experts),
    };
    const std::uint64_t global_strides[4] = {
        packed_columns,
        panel_values / 2,
        static_cast<std::uint64_t>(rows) * packed_columns,
        static_cast<std::uint64_t>(rows) * packed_columns,
    };
    const std::uint32_t box_dimensions[5] = {
        panel_values,
        kNativeGateUpM,
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
        "MoK: failed to encode native gate/up weight tensor map: ",
        error == nullptr ? "unknown CUDA driver error" : error);
    return layout;
}

static __host__ inline native_weight_layouts native_layouts_for(
    const at::Tensor &w1,
    const at::Tensor &w3
) {
    return native_weight_layouts{
        native_weight_layout_for(
            reinterpret_cast<const std::uint8_t *>(w1.data_ptr()),
            static_cast<int>(w1.size(0))),
        native_weight_layout_for(
            reinterpret_cast<const std::uint8_t *>(w3.data_ptr()),
            static_cast<int>(w3.size(0))),
    };
}

__device__ __forceinline__ void load_direct_weight_stage(
    native_weight_stage &destination,
    const native_weight_layout &layout,
    const int expert,
    const int output_tile,
    const int panel,
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
        ".mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :
        : "r"(destination_shared),
          "l"(tensor_map),
          "r"(barrier),
          "n"(0),
          "r"(output_tile * kNativeGateUpM),
          "r"(panel),
          "n"(0),
          "r"(expert)
        : "memory");
}

__device__ __forceinline__ void issue_native_panel(
    native_shared_storage &shared,
    const native_weight_layouts &layouts,
    const int expert,
    const int output_tile,
    const int panel,
    const int stage,
    kittens::semaphore &arrived
) {
    if (kittens::laneid() != 0) return;
    kittens::tma::expect_bytes(
        arrived, 2 * kNativeWeightTransactionBytes);
    load_direct_weight_stage(
        shared.first_weight[stage], layouts.w1, expert, output_tile,
        panel, arrived);
    load_direct_weight_stage(
        shared.second_weight[stage], layouts.w3, expert, output_tile,
        panel, arrived);
}

__device__ __forceinline__ uint4 *native_activation_atom(
    native_activation_tile &tile,
    const int row,
    const int atom
) {
    return reinterpret_cast<uint4 *>(&tile[{row, atom * 16}]);
}

template<bool PoisonInactive>
__device__ __forceinline__ void stage_native_panel(
    native_shared_storage &shared,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_scale,
    const Scratch &scratch,
    const int expert,
    const int rows,
    const int output_tile,
    const int panel,
    const int stage
) {
    const int lane = kittens::laneid();
    const int group_base = panel * kNativePanelGroups;
    const int output_base = output_tile * kNativeGateUpM;
    constexpr std::uint32_t poison = 0x3f3f3f3fu;

    for (int index = lane;
         index < kNativePanelGroups * 16 * 2;
         index += kittens::WARP_THREADS) {
        const int group = index / (16 * 2);
        const int within_group = index % (16 * 2);
        const int row = within_group / 2;
        const int atom = within_group % 2;
        uint4 value;
        if (row < rows) {
            const int token = scratch.assignment_tokens[row];
            const std::uint8_t *source =
                scratch.latent_mxfp8
                + static_cast<long long>(token) * kLatentSize
                + (group_base + group) * kNativeGateUpK
                + atom * 16;
            value = *reinterpret_cast<const uint4 *>(source);
        } else if constexpr (PoisonInactive) {
            value = make_uint4(poison, poison, poison, poison);
        } else {
            value = make_uint4(0u, 0u, 0u, 0u);
        }
        *native_activation_atom(
            shared.activation[
                stage * kNativePanelGroups + group],
            row, atom) = value;
    }

    for (int row = lane; row < kNativeGateUpM;
         row += kittens::WARP_THREADS) {
        const long long weight_row =
            static_cast<long long>(expert) * kExpertW1W3PackedRows
            + output_base + row;
        stage_scale_quad(
            shared.first_scale.stage[stage], row,
            *reinterpret_cast<const std::uint32_t *>(
                w1_scale + weight_row * kExpertW1W3ScaleColumns
                + group_base));
        stage_scale_quad(
            shared.second_scale.stage[stage], row,
            *reinterpret_cast<const std::uint32_t *>(
                w3_scale + weight_row * kExpertW1W3ScaleColumns
                + group_base));
    }

    for (int row = lane; row < kNativeGateUpM;
         row += kittens::WARP_THREADS) {
        std::uint32_t scale = 0x7f7f7f7fu;
        if (row < rows) {
            const int token = scratch.assignment_tokens[row];
            scale = *reinterpret_cast<const std::uint32_t *>(
                scratch.latent_scale
                + static_cast<long long>(token) * kLatentGroups
                + group_base);
        }
        stage_scale_quad(
            shared.activation_scale.stage[stage], row, scale);
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template<bool Profile>
__device__ __forceinline__ void native_producer(
    native_shared_storage &shared,
    const native_weight_layouts &layouts,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_scale,
    const Scratch &scratch,
    const int expert,
    const int rows,
    const int output_tile,
    kittens::semaphore (&tma_arrived)[kNativeWeightStages],
    kittens::semaphore (&panel_ready)[kNativeWeightStages],
    kittens::semaphore (&stage_released)[kNativeWeightStages],
    unsigned long long *__restrict__ profile,
    const int cta
) {
    unsigned long long tma_cycles = 0;
    unsigned long long ring_full_cycles = 0;

    #pragma unroll
    for (int panel = 0; panel < kNativeWeightStages; ++panel) {
        stage_native_panel<false>(
            shared, w1_scale, w3_scale, scratch, expert, rows,
            output_tile, panel, panel);
        __syncwarp();
        issue_native_panel(
            shared, layouts, expert, output_tile, panel, panel,
            tma_arrived[panel]);
    }

    #pragma unroll
    for (int panel = 0; panel < kNativeWeightStages; ++panel) {
        const unsigned long long started =
            static_cast<unsigned long long>(clock64());
        kittens::wait(tma_arrived[panel], 0);
        if constexpr (Profile) {
            if (kittens::laneid() == 0) {
                tma_cycles +=
                    static_cast<unsigned long long>(clock64()) - started;
            }
        }
        __syncwarp();
        if (kittens::laneid() == 0) {
            kittens::arrive(panel_ready[panel]);
        }
    }

    for (int panel = kNativeWeightStages; panel < kNativePanels; ++panel) {
        const int stage = panel % kNativeWeightStages;
        const int cycle = panel / kNativeWeightStages;
        const unsigned long long ring_started =
            static_cast<unsigned long long>(clock64());
        kittens::wait(stage_released[stage], (cycle - 1) & 1);
        if constexpr (Profile) {
            if (kittens::laneid() == 0) {
                ring_full_cycles +=
                    static_cast<unsigned long long>(clock64())
                    - ring_started;
            }
        }
        stage_native_panel<false>(
            shared, w1_scale, w3_scale, scratch, expert, rows,
            output_tile, panel, stage);
        __syncwarp();
        issue_native_panel(
            shared, layouts, expert, output_tile, panel, stage,
            tma_arrived[stage]);
        const unsigned long long tma_started =
            static_cast<unsigned long long>(clock64());
        kittens::wait(tma_arrived[stage], cycle & 1);
        if constexpr (Profile) {
            if (kittens::laneid() == 0) {
                tma_cycles +=
                    static_cast<unsigned long long>(clock64())
                    - tma_started;
            }
        }
        __syncwarp();
        if (kittens::laneid() == 0) {
            kittens::arrive(panel_ready[stage]);
        }
    }

    if constexpr (Profile) {
        if (kittens::laneid() == 0) {
            profile[
                static_cast<long long>(cta) * kNativeProfilePhases + 0] =
                    tma_cycles;
            profile[
                static_cast<long long>(cta) * kNativeProfilePhases + 1] =
                    ring_full_cycles;
        }
    }
}

template<bool PoisonInactive, bool Profile>
__device__ __forceinline__ void native_producer_dispatch(
    native_shared_storage &shared,
    const native_weight_layouts &layouts,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_scale,
    const Scratch &scratch,
    const int expert,
    const int rows,
    const int output_tile,
    kittens::semaphore (&tma_arrived)[kNativeWeightStages],
    kittens::semaphore (&panel_ready)[kNativeWeightStages],
    kittens::semaphore (&stage_released)[kNativeWeightStages],
    unsigned long long *__restrict__ profile,
    const int cta
) {
    unsigned long long tma_cycles = 0;
    unsigned long long ring_full_cycles = 0;

    #pragma unroll
    for (int panel = 0; panel < kNativeWeightStages; ++panel) {
        stage_native_panel<PoisonInactive>(
            shared, w1_scale, w3_scale, scratch, expert, rows,
            output_tile, panel, panel);
        __syncwarp();
        issue_native_panel(
            shared, layouts, expert, output_tile, panel, panel,
            tma_arrived[panel]);
    }

    #pragma unroll
    for (int panel = 0; panel < kNativeWeightStages; ++panel) {
        const unsigned long long started =
            static_cast<unsigned long long>(clock64());
        kittens::wait(tma_arrived[panel], 0);
        if constexpr (Profile) {
            if (kittens::laneid() == 0) {
                tma_cycles +=
                    static_cast<unsigned long long>(clock64()) - started;
            }
        }
        __syncwarp();
        if (kittens::laneid() == 0) {
            kittens::arrive(panel_ready[panel]);
        }
    }

    for (int panel = kNativeWeightStages; panel < kNativePanels; ++panel) {
        const int stage = panel % kNativeWeightStages;
        const int cycle = panel / kNativeWeightStages;
        const unsigned long long ring_started =
            static_cast<unsigned long long>(clock64());
        kittens::wait(stage_released[stage], (cycle - 1) & 1);
        if constexpr (Profile) {
            if (kittens::laneid() == 0) {
                ring_full_cycles +=
                    static_cast<unsigned long long>(clock64())
                    - ring_started;
            }
        }
        stage_native_panel<PoisonInactive>(
            shared, w1_scale, w3_scale, scratch, expert, rows,
            output_tile, panel, stage);
        __syncwarp();
        issue_native_panel(
            shared, layouts, expert, output_tile, panel, stage,
            tma_arrived[stage]);
        const unsigned long long tma_started =
            static_cast<unsigned long long>(clock64());
        kittens::wait(tma_arrived[stage], cycle & 1);
        if constexpr (Profile) {
            if (kittens::laneid() == 0) {
                tma_cycles +=
                    static_cast<unsigned long long>(clock64())
                    - tma_started;
            }
        }
        __syncwarp();
        if (kittens::laneid() == 0) {
            kittens::arrive(panel_ready[stage]);
        }
    }

    if constexpr (Profile) {
        if (kittens::laneid() == 0) {
            profile[
                static_cast<long long>(cta) * kNativeProfilePhases + 0] =
                    tma_cycles;
            profile[
                static_cast<long long>(cta) * kNativeProfilePhases + 1] =
                    ring_full_cycles;
        }
    }
}

__device__ __forceinline__ void batch_mixed_mma_direct(
    const native_accumulator_tile &destination,
    const std::uint64_t weight_chunk,
    const native_activation_tile &activation,
    const kittens::full_tt_fp8e8m0<16> &weight_scale,
    const kittens::full_tt_fp8e8m0<16> &activation_scale,
    const int scale_factor_id,
    const bool accumulate
) {
    kittens::st_descriptor<
        native_activation_tile, kittens::transpose::N>
            activation_descriptor(activation);
    const std::uint32_t instruction =
        batch_probe::batch_instruction_descriptor(scale_factor_id);
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
          "l"(activation_descriptor.base_desc),
          "r"(instruction),
          "r"(weight_scale.addr),
          "r"(activation_scale.addr),
          "r"(accumulate ? 1u : 0u));
}

template<bool Capture, bool Profile>
__device__ __forceinline__ void native_situ_epilogue(
    const native_accumulator_tile &first_accumulator,
    const native_accumulator_tile &second_accumulator,
    const Scratch &scratch,
    float *__restrict__ gate_output,
    float *__restrict__ up_output,
    const int rows,
    const int output_tile,
    unsigned long long *__restrict__ profile,
    const int cta
) {
    using namespace kittens;
    const int warp = warpid();
    if (warp >= kNativeEpilogueWarps) return;
    const unsigned long long started =
        static_cast<unsigned long long>(clock64());

    native_epilogue_tile gate;
    native_epilogue_tile up;
    const auto gate_slice =
        first_accumulator.template subtile<
            tt_fl<kNativeGateUpM / kNativeEpilogueWarps, kNativePhysicalN>>(
                warp * (kNativeGateUpM / kNativeEpilogueWarps), 0);
    const auto up_slice =
        second_accumulator.template subtile<
            tt_fl<kNativeGateUpM / kNativeEpilogueWarps, kNativePhysicalN>>(
                warp * (kNativeGateUpM / kNativeEpilogueWarps), 0);
    group<1>::load_async(gate, gate_slice);
    group<1>::load_async(up, up_slice);
    tensor_load_wait();

    float absolute_max_x = 0.0f;
    float absolute_max_y = 0.0f;
    #pragma unroll
    for (int i = 0; i < native_epilogue_tile::height; ++i) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const float gate_x = gate.tiles[i][0].data[k].x;
            const float gate_y = gate.tiles[i][0].data[k].y;
            const float up_x = up.tiles[i][0].data[k].x;
            const float up_y = up.tiles[i][0].data[k].y;
            const float sigmoid_x = 1.0f / (1.0f + expf(-gate_x));
            const float sigmoid_y = 1.0f / (1.0f + expf(-gate_y));
            const float value_x =
                4.0f * tanhf(gate_x * 0.25f) * sigmoid_x
                * 25.0f * tanhf(up_x / 25.0f);
            const float value_y =
                4.0f * tanhf(gate_y * 0.25f) * sigmoid_y
                * 25.0f * tanhf(up_y / 25.0f);
            gate.tiles[i][0].data[k].x = value_x;
            gate.tiles[i][0].data[k].y = value_y;
            absolute_max_x = fmaxf(absolute_max_x, fabsf(value_x));
            absolute_max_y = fmaxf(absolute_max_y, fabsf(value_y));

            if constexpr (Capture) {
                const int channel =
                    warp * (kNativeGateUpM / kNativeEpilogueWarps)
                    + i * 16 + (k % 2) * 8 + laneid() / 4;
                const int token_x = 2 * (laneid() % 4);
                const int token_y = token_x + 1;
                if (token_x < rows) {
                    gate_output[
                        static_cast<long long>(token_x) * kNativeGateUpM
                        + channel] = gate_x;
                    up_output[
                        static_cast<long long>(token_x) * kNativeGateUpM
                        + channel] = up_x;
                }
                if (token_y < rows) {
                    gate_output[
                        static_cast<long long>(token_y) * kNativeGateUpM
                        + channel] = gate_y;
                    up_output[
                        static_cast<long long>(token_y) * kNativeGateUpM
                        + channel] = up_y;
                }
            }
        }
    }

    #pragma unroll
    for (int delta = 16; delta >= 4; delta /= 2) {
        absolute_max_x = fmaxf(
            absolute_max_x,
            __shfl_xor_sync(kittens::MASK_ALL, absolute_max_x, delta));
        absolute_max_y = fmaxf(
            absolute_max_y,
            __shfl_xor_sync(kittens::MASK_ALL, absolute_max_y, delta));
    }
    const std::uint8_t scale_x = select_e8m0_scale(absolute_max_x);
    const std::uint8_t scale_y = select_e8m0_scale(absolute_max_y);
    const float reciprocal_x =
        __uint_as_float((254u - static_cast<unsigned int>(scale_x)) << 23);
    const float reciprocal_y =
        __uint_as_float((254u - static_cast<unsigned int>(scale_y)) << 23);
    const int token_x = 2 * (laneid() % 4);
    const int token_y = token_x + 1;
    const int global_group =
        output_tile * (kNativeGateUpM / kNativeGateUpK) + warp;
    if (laneid() / 4 == 0) {
        if (token_x < rows) {
            scratch.situ_scale[
                static_cast<long long>(token_x) * kSituGroups
                + global_group] = scale_x;
        }
        if (token_y < rows) {
            scratch.situ_scale[
                static_cast<long long>(token_y) * kSituGroups
                + global_group] = scale_y;
        }
    }

    #pragma unroll
    for (int i = 0; i < native_epilogue_tile::height; ++i) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int local_channel =
                warp * (kNativeGateUpM / kNativeEpilogueWarps)
                + i * 16 + (k % 2) * 8 + laneid() / 4;
            const int global_channel =
                output_tile * kNativeGateUpM + local_channel;
            if (token_x < rows) {
                scratch.situ_mxfp8[
                    static_cast<long long>(token_x)
                        * kRoutedIntermediateSizePerRank
                    + global_channel] =
                        quantize_e4m3(
                            gate.tiles[i][0].data[k].x, reciprocal_x);
            }
            if (token_y < rows) {
                scratch.situ_mxfp8[
                    static_cast<long long>(token_y)
                        * kRoutedIntermediateSizePerRank
                    + global_channel] =
                        quantize_e4m3(
                            gate.tiles[i][0].data[k].y, reciprocal_y);
            }
        }
    }

    if constexpr (Profile) {
        const unsigned long long elapsed =
            static_cast<unsigned long long>(clock64()) - started;
        if (laneid() == 0) {
            atomicAdd(
                &profile[
                    static_cast<long long>(cta) * kNativeProfilePhases + 3],
                elapsed);
        }
    }
}

template<bool PoisonInactive, bool Capture, bool Profile>
__device__ __forceinline__ void native_gate_up_candidate(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const native_weight_layouts &layouts,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_scale,
    const Scratch &scratch,
    float *__restrict__ gate_output,
    float *__restrict__ up_output,
    unsigned long long *__restrict__ profile,
    const int expert,
    const int rows,
    const int output_tile,
    const int cta
) {
    using namespace kittens;
    auto &shared =
        *reinterpret_cast<native_shared_storage *>(shared_raw);
    __shared__ semaphore tma_arrived[kNativeWeightStages];
    __shared__ semaphore panel_ready[kNativeWeightStages];
    __shared__ semaphore stage_released[kNativeWeightStages];
    __shared__ semaphore compute_done;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < kNativeWeightStages; ++stage) {
            init_semaphore(tma_arrived[stage], 0, 1);
            init_semaphore(panel_ready[stage], 1, 0);
            init_semaphore(stage_released[stage], 0, 1);
        }
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    native_accumulator_tile first_accumulator =
        tensor_pool.allocate<native_accumulator_tile>(0);
    native_accumulator_tile second_accumulator =
        tensor_pool.allocate<native_accumulator_tile>(kNativePhysicalN);
    constexpr int scale_column_base = 2 * kNativePhysicalN;
    const auto scale_slot = [&](const int stage, const int operand) {
        return tensor_pool.allocate<full_tt_fp8e8m0<16>>(
            scale_column_base
            + (stage * 3 + operand) * kRoutedScaleColumns);
    };

    if (warpid() == kNativeProducerWarp) {
        native_producer_dispatch<PoisonInactive, Profile>(
            shared, layouts, w1_scale, w3_scale, scratch, expert, rows,
            output_tile, tma_arrived, panel_ready, stage_released,
            profile, cta);
    }

    if (warpid() == kNativeConsumerWarp) {
        unsigned long long mma_cycles = 0;
        for (int panel = 0; panel < kNativePanels; ++panel) {
            const int stage = panel % kNativeWeightStages;
            const int cycle = panel / kNativeWeightStages;
            wait(panel_ready[stage], cycle & 1);
            const unsigned long long started =
                static_cast<unsigned long long>(clock64());
            if (laneid() == 0) {
                load_mxnv_scale_async(
                    scale_slot(stage, 0),
                    shared.first_scale.stage[stage]);
                load_mxnv_scale_async(
                    scale_slot(stage, 1),
                    shared.second_scale.stage[stage]);
                load_mxnv_scale_async(
                    scale_slot(stage, 2),
                    shared.activation_scale.stage[stage]);
            }
            tensor_store_wait();
            asm volatile(
                "fence.proxy.async.shared::cta;\n" ::: "memory");
            if (laneid() == 0) {
                st_descriptor<native_weight_stage, transpose::N>
                    first_weight_descriptor(
                        shared.first_weight[stage]);
                st_descriptor<native_weight_stage, transpose::N>
                    second_weight_descriptor(
                        shared.second_weight[stage]);
                #pragma unroll
                for (int slot = 0; slot < kNativePanelGroups; ++slot) {
                    const bool accumulate = panel != 0 || slot != 0;
                    batch_mixed_mma_direct(
                        first_accumulator,
                        first_weight_descriptor.chunk_descriptor(slot),
                        shared.activation[
                            stage * kNativePanelGroups + slot],
                        scale_slot(stage, 0), scale_slot(stage, 2),
                        slot, accumulate);
                    batch_mixed_mma_direct(
                        second_accumulator,
                        second_weight_descriptor.chunk_descriptor(slot),
                        shared.activation[
                            stage * kNativePanelGroups + slot],
                        scale_slot(stage, 1), scale_slot(stage, 2),
                        slot, accumulate);
                }
                if (panel + kNativeWeightStages < kNativePanels) {
                    detail::tcgen05::commit<1>(stage_released[stage]);
                }
            }
            if constexpr (Profile) {
                if (laneid() == 0) {
                    mma_cycles +=
                        static_cast<unsigned long long>(clock64()) - started;
                }
            }
        }
        if (laneid() == 0) {
            detail::tcgen05::commit<1>(compute_done);
            if constexpr (Profile) {
                profile[
                    static_cast<long long>(cta) * kNativeProfilePhases + 2] =
                        mma_cycles;
            }
        }
    }

    if (warpid() < kNativeEpilogueWarps) {
        wait(compute_done, 0);
        native_situ_epilogue<Capture, Profile>(
            first_accumulator, second_accumulator, scratch,
            gate_output, up_output, rows, output_tile, profile, cta);
    }
}

__device__ __forceinline__ Scratch native_scratch(
    const int cta,
    const int *assignment_tokens,
    std::uint8_t *activation,
    std::uint8_t *activation_scale,
    std::uint8_t *situ,
    std::uint8_t *situ_scale
) {
    Scratch scratch{};
    scratch.assignment_tokens = const_cast<int *>(assignment_tokens);
    scratch.latent_mxfp8 =
        activation
        + static_cast<long long>(cta) * kNativeGateUpN * kLatentSize;
    scratch.latent_scale =
        activation_scale
        + static_cast<long long>(cta) * kNativeGateUpN * kLatentGroups;
    scratch.situ_mxfp8 =
        situ
        + static_cast<long long>(cta) * kNativeGateUpN
            * kRoutedIntermediateSizePerRank;
    scratch.situ_scale =
        situ_scale
        + static_cast<long long>(cta) * kNativeGateUpN * kSituGroups;
    return scratch;
}

template<bool PoisonInactive, bool Capture, bool Profile>
static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void native_gate_up_candidate_kernel(
    const std::uint8_t *__restrict__ w1_packed,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_packed,
    const std::uint8_t *__restrict__ w3_scale,
    const __grid_constant__ native_weight_layouts layouts,
    std::uint8_t *__restrict__ activation,
    std::uint8_t *__restrict__ activation_scale,
    const int *__restrict__ assignment_tokens,
    const int *__restrict__ experts,
    const int *__restrict__ output_tiles,
    std::uint8_t *__restrict__ situ,
    std::uint8_t *__restrict__ situ_scale,
    float *__restrict__ gate_output,
    float *__restrict__ up_output,
    unsigned long long *__restrict__ profile,
    const int rows
) {
    (void)w1_packed;
    (void)w3_packed;
    extern __shared__ __align__(1024) int shared_raw[];
    kittens::tensor_allocator<1, 1> tensor_pool{};
    const int cta = static_cast<int>(blockIdx.x);
    const Scratch scratch = native_scratch(
        cta, assignment_tokens, activation, activation_scale, situ,
        situ_scale);
    native_gate_up_candidate<PoisonInactive, Capture, Profile>(
        shared_raw, tensor_pool, layouts, w1_scale, w3_scale, scratch,
        gate_output
            + static_cast<long long>(cta) * kNativeGateUpN * kNativeGateUpM,
        up_output
            + static_cast<long long>(cta) * kNativeGateUpN * kNativeGateUpM,
        profile, experts[cta], rows, output_tiles[cta], cta);
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void native_gate_up_baseline_kernel(
    const std::uint8_t *__restrict__ w1_packed,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_packed,
    const std::uint8_t *__restrict__ w3_scale,
    std::uint8_t *__restrict__ activation,
    std::uint8_t *__restrict__ activation_scale,
    const int *__restrict__ assignment_tokens,
    const int *__restrict__ experts,
    const int *__restrict__ output_tiles,
    std::uint8_t *__restrict__ situ,
    std::uint8_t *__restrict__ situ_scale,
    const int rows
) {
    extern __shared__ __align__(1024) int shared_raw[];
    kittens::tensor_allocator<1, 1> tensor_pool{};
    const int cta = static_cast<int>(blockIdx.x);
    const Scratch scratch = native_scratch(
        cta, assignment_tokens, activation, activation_scale, situ,
        situ_scale);
    routed_gate_up_unit(
        shared_raw, tensor_pool, w1_packed, w1_scale, w3_packed, w3_scale,
        scratch, experts[cta], 0, rows, output_tiles[cta],
        PhaseClocks{nullptr});
}

static __device__ void native_gate_up_reference_capture(
    int *__restrict__ shared_raw,
    kittens::tensor_allocator<1, 1> &tensor_pool,
    const std::uint8_t *__restrict__ w1_packed,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_packed,
    const std::uint8_t *__restrict__ w3_scale,
    const Scratch &scratch,
    float *__restrict__ gate_output,
    float *__restrict__ up_output,
    const int expert,
    const int rows,
    const int output_tile
) {
    using namespace kittens;
    tma_swizzle_allocator staging(shared_raw);
    mixed_operand_tile (&activation_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_operand_tile (&first_weight_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_operand_tile (&second_weight_tile)[kGateUpRoundGroups] =
        staging.allocate<mixed_operand_tile, kGateUpRoundGroups>();
    mixed_scale_tile (&activation_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();
    mixed_scale_tile (&first_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();
    mixed_scale_tile (&second_scale_shared)[kGateUpScaleTiles] =
        staging.allocate<mixed_scale_tile, kGateUpScaleTiles>();
    tma_swizzle_allocator result_allocator(shared_raw);
    mixed_result_tile (&result_shared) =
        result_allocator.allocate<mixed_result_tile>();

    __shared__ semaphore reference_done;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) init_semaphore(reference_done, 0, 1);
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
    }
    #pragma unroll
    for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
        clear_scale_tile(activation_scale_shared[quad]);
    }
    __syncthreads();

    const int weight_half = thread / kMmaN;
    const int weight_row = thread % kMmaN;
    const std::uint8_t *weight_packed =
        weight_half == 0 ? w1_packed : w3_packed;
    const std::uint8_t *weight_scales =
        weight_half == 0 ? w1_scale : w3_scale;
    mixed_operand_tile *weight_tile =
        weight_half == 0 ? first_weight_tile : second_weight_tile;
    mixed_scale_tile *weight_scale_shared =
        weight_half == 0 ? first_scale_shared : second_scale_shared;
    const int output_base = output_tile * kMmaN;
    const long long weight_index =
        static_cast<long long>(expert) * kExpertW1W3PackedRows
        + output_base + weight_row;
    const std::uint8_t *weight_row_bytes =
        weight_packed + weight_index * kExpertW1W3PackedColumns;
    const std::uint8_t *weight_row_scales =
        weight_scales + weight_index * kExpertW1W3ScaleColumns;

    uint4 payload[kGateUpRoundGroups];
    std::uint32_t scale_words[kGateUpScaleTiles];
    const auto read_round = [&](const int group_base) {
        #pragma unroll
        for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
            payload[slot] = *reinterpret_cast<const uint4 *>(
                weight_row_bytes
                + (group_base + slot) * (kMmaK / 2));
        }
        #pragma unroll
        for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
            scale_words[quad] =
                *reinterpret_cast<const std::uint32_t *>(
                    weight_row_scales + group_base
                    + quad * kScaleGroupsPerTile);
        }
    };
    read_round(0);

    int compute_phase = 0;
    for (int round = 0; round < kGateUpRounds; ++round) {
        const int group_base = round * kGateUpRoundGroups;
        #pragma unroll
        for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
            stage_weight_row(
                weight_tile[slot], weight_row, payload[slot]);
        }
        #pragma unroll
        for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
            stage_scale_quad(
                weight_scale_shared[quad], weight_row,
                scale_words[quad]);
        }
        for (int index = thread;
             index < rows * kGateUpRoundGroups;
             index += kDecodeCtaThreads) {
            const int row = index / kGateUpRoundGroups;
            const int slot = index % kGateUpRoundGroups;
            stage_activation_row(
                activation_tile[slot], row,
                scratch.latent_mxfp8
                    + static_cast<long long>(row) * kLatentSize
                    + (group_base + slot) * kMmaK);
        }
        for (int index = thread;
             index < rows * kGateUpScaleTiles;
             index += kDecodeCtaThreads) {
            const int row = index / kGateUpScaleTiles;
            const int quad = index % kGateUpScaleTiles;
            stage_scale_quad(
                activation_scale_shared[quad], row,
                *reinterpret_cast<const std::uint32_t *>(
                    scratch.latent_scale
                    + static_cast<long long>(row) * kLatentGroups
                    + group_base + quad * kScaleGroupsPerTile));
        }
        if (round + 1 < kGateUpRounds) {
            read_round(group_base + kGateUpRoundGroups);
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if (warpid() == 0) {
            if (laneid() == 0) {
                #pragma unroll
                for (int quad = 0; quad < kGateUpScaleTiles; ++quad) {
                    load_mxnv_scale_async(
                        scale_slot(quad),
                        activation_scale_shared[quad]);
                    load_mxnv_scale_async(
                        scale_slot(kGateUpScaleTiles + quad),
                        first_scale_shared[quad]);
                    load_mxnv_scale_async(
                        scale_slot(2 * kGateUpScaleTiles + quad),
                        second_scale_shared[quad]);
                }
            }
            tensor_store_wait();
            if (laneid() == 0) {
                #pragma unroll
                for (int slot = 0; slot < kGateUpRoundGroups; ++slot) {
                    const int quad = slot / kScaleGroupsPerTile;
                    const int scale_factor_id =
                        slot % kScaleGroupsPerTile;
                    const bool accumulate = round != 0 || slot != 0;
                    mixed_mma(
                        first_accumulator, activation_tile[slot],
                        first_weight_tile[slot], scale_slot(quad),
                        scale_slot(kGateUpScaleTiles + quad),
                        scale_factor_id, accumulate);
                    mixed_mma(
                        second_accumulator, activation_tile[slot],
                        second_weight_tile[slot], scale_slot(quad),
                        scale_slot(2 * kGateUpScaleTiles + quad),
                        scale_factor_id, accumulate);
                }
                detail::tcgen05::commit<1>(reference_done);
            }
        }
        wait(reference_done, compute_phase);
        __syncthreads();
        compute_phase ^= 1;
    }

    store_accumulator(first_accumulator, result_shared);
    for (int index = thread; index < rows * kMmaN;
         index += kDecodeCtaThreads) {
        gate_output[index] =
            result_shared[{index / kMmaN, index % kMmaN}];
    }
    __syncthreads();
    store_accumulator(second_accumulator, result_shared);
    for (int index = thread; index < rows * kMmaN;
         index += kDecodeCtaThreads) {
        up_output[index] =
            result_shared[{index / kMmaN, index % kMmaN}];
    }
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void native_gate_up_reference_capture_kernel(
    const std::uint8_t *__restrict__ w1_packed,
    const std::uint8_t *__restrict__ w1_scale,
    const std::uint8_t *__restrict__ w3_packed,
    const std::uint8_t *__restrict__ w3_scale,
    std::uint8_t *__restrict__ activation,
    std::uint8_t *__restrict__ activation_scale,
    const int *__restrict__ assignment_tokens,
    const int *__restrict__ experts,
    const int *__restrict__ output_tiles,
    std::uint8_t *__restrict__ situ,
    std::uint8_t *__restrict__ situ_scale,
    float *__restrict__ gate_output,
    float *__restrict__ up_output,
    const int rows
) {
    extern __shared__ __align__(1024) int shared_raw[];
    kittens::tensor_allocator<1, 1> tensor_pool{};
    const int cta = static_cast<int>(blockIdx.x);
    const Scratch scratch = native_scratch(
        cta, assignment_tokens, activation, activation_scale, situ,
        situ_scale);
    native_gate_up_reference_capture(
        shared_raw, tensor_pool, w1_packed, w1_scale, w3_packed, w3_scale,
        scratch,
        gate_output
            + static_cast<long long>(cta) * kNativeGateUpN * kNativeGateUpM,
        up_output
            + static_cast<long long>(cta) * kNativeGateUpN * kNativeGateUpM,
        experts[cta], rows, output_tiles[cta]);
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void prepare_native_gate_up_activations_kernel(
    const __nv_bfloat16 *__restrict__ latent,
    std::uint8_t *__restrict__ activation,
    std::uint8_t *__restrict__ activation_scale,
    const int rows
) {
    const int cta = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < rows * kLatentGroups;
         index += kDecodeCtaThreads) {
        const int row = index / kLatentGroups;
        const int group = index % kLatentGroups;
        float values[kMmaK];
        float absolute_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            const float value = __bfloat162float(
                latent[
                    (static_cast<long long>(cta) * kNativeGateUpN + row)
                        * kLatentSize
                    + group * kMmaK + k]);
            values[k] = value;
            absolute_max = fmaxf(absolute_max, fabsf(value));
        }
        const std::uint8_t scale = select_e8m0_scale(absolute_max);
        const float reciprocal =
            __uint_as_float(
                (254u - static_cast<unsigned int>(scale)) << 23);
        activation_scale[
            (static_cast<long long>(cta) * kNativeGateUpN + row)
                * kLatentGroups
            + group] = scale;
        #pragma unroll
        for (int k = 0; k < kMmaK; ++k) {
            activation[
                (static_cast<long long>(cta) * kNativeGateUpN + row)
                    * kLatentSize
                + group * kMmaK + k] =
                    quantize_e4m3(values[k], reciprocal);
        }
    }
}

static __host__ inline void check_native_weight(
    const at::Tensor &tensor,
    const char *name,
    const int columns
) {
    TORCH_CHECK(
        tensor.device().is_cuda() && tensor.is_contiguous()
            && tensor.scalar_type() == at::kByte
            && tensor.dim() == 3 && tensor.size(0) >= 1
            && tensor.size(1) == kExpertW1W3PackedRows
            && tensor.size(2) == columns,
        "MoK: native gate/up probe requires uint8 ", name, " [E, ",
        kExpertW1W3PackedRows, ", ", columns, "]");
}

static __host__ void prepare_native_gate_up_activations_entrypoint(
    const at::Tensor &latent,
    const at::Tensor &activation,
    const at::Tensor &activation_scale,
    const std::int64_t rows
) {
    TORCH_CHECK(
        rows >= 1 && rows <= kNativeGateUpN,
        "MoK: native gate/up rows must be in [1, 8]");
    TORCH_CHECK(
        latent.device().is_cuda() && latent.is_contiguous()
            && latent.scalar_type() == at::kBFloat16
            && latent.dim() == 3
            && latent.size(1) == kNativeGateUpN
            && latent.size(2) == kLatentSize,
        "MoK: native gate/up latent must be BF16 [CTAs, 8, 3584]");
    TORCH_CHECK(
        activation.device() == latent.device()
            && activation.is_contiguous()
            && activation.scalar_type() == at::kByte
            && activation.sizes() == latent.sizes(),
        "MoK: native gate/up activation must be uint8 with latent shape");
    TORCH_CHECK(
        activation_scale.device() == latent.device()
            && activation_scale.is_contiguous()
            && activation_scale.scalar_type() == at::kByte
            && activation_scale.dim() == 3
            && activation_scale.size(0) == latent.size(0)
            && activation_scale.size(1) == kNativeGateUpN
            && activation_scale.size(2) == kLatentGroups,
        "MoK: native gate/up scales must be uint8 [CTAs, 8, 112]");

    const c10::cuda::CUDAGuard guard(latent.device());
    prepare_native_gate_up_activations_kernel
        <<<static_cast<int>(latent.size(0)), kDecodeCtaThreads, 0,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(latent.data_ptr()),
            reinterpret_cast<std::uint8_t *>(activation.data_ptr()),
            reinterpret_cast<std::uint8_t *>(
                activation_scale.data_ptr()),
            static_cast<int>(rows));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool PoisonInactive, bool Capture, bool Profile>
static __host__ void launch_native_candidate(
    const at::Tensor &w1_packed,
    const at::Tensor &w1_scale,
    const at::Tensor &w3_packed,
    const at::Tensor &w3_scale,
    const native_weight_layouts &layouts,
    const at::Tensor &activation,
    const at::Tensor &activation_scale,
    const at::Tensor &assignment_tokens,
    const at::Tensor &experts,
    const at::Tensor &output_tiles,
    const at::Tensor &situ,
    const at::Tensor &situ_scale,
    const at::Tensor &gate,
    const at::Tensor &up,
    const at::Tensor &profile,
    const int rows,
    const cudaStream_t stream
) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        native_gate_up_candidate_kernel<
            PoisonInactive, Capture, Profile>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kNativeGateUpSharedReservationBytes));
    native_gate_up_candidate_kernel<
        PoisonInactive, Capture, Profile>
        <<<static_cast<int>(experts.numel()), kDecodeCtaThreads,
           kNativeGateUpSharedReservationBytes, stream>>>(
            reinterpret_cast<const std::uint8_t *>(w1_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(w1_scale.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(w3_packed.data_ptr()),
            reinterpret_cast<const std::uint8_t *>(w3_scale.data_ptr()),
            layouts,
            reinterpret_cast<std::uint8_t *>(activation.data_ptr()),
            reinterpret_cast<std::uint8_t *>(
                activation_scale.data_ptr()),
            reinterpret_cast<const int *>(assignment_tokens.data_ptr()),
            reinterpret_cast<const int *>(experts.data_ptr()),
            reinterpret_cast<const int *>(output_tiles.data_ptr()),
            reinterpret_cast<std::uint8_t *>(situ.data_ptr()),
            reinterpret_cast<std::uint8_t *>(situ_scale.data_ptr()),
            reinterpret_cast<float *>(gate.data_ptr()),
            reinterpret_cast<float *>(up.data_ptr()),
            reinterpret_cast<unsigned long long *>(profile.data_ptr()),
            rows);
}

static __host__ void native_gate_up_probe_entrypoint(
    const at::Tensor &w1_packed,
    const at::Tensor &w1_scale,
    const at::Tensor &w3_packed,
    const at::Tensor &w3_scale,
    const at::Tensor &activation,
    const at::Tensor &activation_scale,
    const at::Tensor &assignment_tokens,
    const at::Tensor &experts,
    const at::Tensor &output_tiles,
    const at::Tensor &situ,
    const at::Tensor &situ_scale,
    const at::Tensor &gate,
    const at::Tensor &up,
    const at::Tensor &profile,
    const std::int64_t rows,
    const bool candidate,
    const bool poison_inactive,
    const bool capture,
    const bool profile_enabled
) {
    TORCH_CHECK(rows >= 1 && rows <= kNativeGateUpN);
    check_native_weight(
        w1_packed, "w1_packed", kExpertW1W3PackedColumns);
    check_native_weight(
        w3_packed, "w3_packed", kExpertW1W3PackedColumns);
    check_native_weight(
        w1_scale, "w1_scale", kExpertW1W3ScaleColumns);
    check_native_weight(
        w3_scale, "w3_scale", kExpertW1W3ScaleColumns);
    TORCH_CHECK(
        w1_packed.size(0) == w1_scale.size(0)
            && w1_packed.size(0) == w3_packed.size(0)
            && w1_packed.size(0) == w3_scale.size(0));
    TORCH_CHECK(
        activation.device().is_cuda() && activation.is_contiguous()
            && activation.scalar_type() == at::kByte
            && activation.dim() == 3
            && activation.size(1) == kNativeGateUpN
            && activation.size(2) == kLatentSize);
    TORCH_CHECK(
        activation_scale.device() == activation.device()
            && activation_scale.is_contiguous()
            && activation_scale.scalar_type() == at::kByte
            && activation_scale.dim() == 3
            && activation_scale.size(0) == activation.size(0)
            && activation_scale.size(1) == kNativeGateUpN
            && activation_scale.size(2) == kLatentGroups);
    const std::int64_t ctas = activation.size(0);
    TORCH_CHECK(
        experts.device() == activation.device()
            && experts.is_contiguous()
            && experts.scalar_type() == at::kInt
            && experts.numel() == ctas);
    TORCH_CHECK(
        output_tiles.device() == activation.device()
            && output_tiles.is_contiguous()
            && output_tiles.scalar_type() == at::kInt
            && output_tiles.numel() == ctas);
    TORCH_CHECK(
        assignment_tokens.device() == activation.device()
            && assignment_tokens.is_contiguous()
            && assignment_tokens.scalar_type() == at::kInt
            && assignment_tokens.numel() == kNativeGateUpN);
    TORCH_CHECK(
        situ.device() == activation.device() && situ.is_contiguous()
            && situ.scalar_type() == at::kByte
            && situ.dim() == 3 && situ.size(0) == ctas
            && situ.size(1) == kNativeGateUpN
            && situ.size(2) == kRoutedIntermediateSizePerRank);
    TORCH_CHECK(
        situ_scale.device() == activation.device()
            && situ_scale.is_contiguous()
            && situ_scale.scalar_type() == at::kByte
            && situ_scale.dim() == 3 && situ_scale.size(0) == ctas
            && situ_scale.size(1) == kNativeGateUpN
            && situ_scale.size(2) == kSituGroups);
    TORCH_CHECK(
        gate.device() == activation.device() && gate.is_contiguous()
            && gate.scalar_type() == at::kFloat && gate.dim() == 3
            && gate.size(0) == ctas
            && gate.size(1) == kNativeGateUpN
            && gate.size(2) == kNativeGateUpM
            && up.device() == gate.device()
            && up.is_contiguous() && up.sizes() == gate.sizes()
            && up.scalar_type() == at::kFloat);
    TORCH_CHECK(
        profile.device() == activation.device()
            && profile.is_contiguous()
            && profile.scalar_type() == at::kLong
            && profile.dim() == 2 && profile.size(0) == ctas
            && profile.size(1) == kNativeProfilePhases);
    TORCH_CHECK(!poison_inactive || candidate);
    TORCH_CHECK(!profile_enabled || candidate);

    const c10::cuda::CUDAGuard guard(activation.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (candidate) {
        const native_weight_layouts layouts =
            native_layouts_for(w1_packed, w3_packed);
        if (poison_inactive) {
            if (capture) {
                if (profile_enabled) {
                    launch_native_candidate<true, true, true>(
                        w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                        activation, activation_scale, assignment_tokens,
                        experts, output_tiles, situ, situ_scale, gate, up,
                        profile, static_cast<int>(rows), stream);
                } else {
                    launch_native_candidate<true, true, false>(
                        w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                        activation, activation_scale, assignment_tokens,
                        experts, output_tiles, situ, situ_scale, gate, up,
                        profile, static_cast<int>(rows), stream);
                }
            } else if (profile_enabled) {
                launch_native_candidate<true, false, true>(
                    w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                    activation, activation_scale, assignment_tokens,
                    experts, output_tiles, situ, situ_scale, gate, up,
                    profile, static_cast<int>(rows), stream);
            } else {
                launch_native_candidate<true, false, false>(
                    w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                    activation, activation_scale, assignment_tokens,
                    experts, output_tiles, situ, situ_scale, gate, up,
                    profile, static_cast<int>(rows), stream);
            }
        } else if (capture) {
            if (profile_enabled) {
                launch_native_candidate<false, true, true>(
                    w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                    activation, activation_scale, assignment_tokens,
                    experts, output_tiles, situ, situ_scale, gate, up,
                    profile, static_cast<int>(rows), stream);
            } else {
                launch_native_candidate<false, true, false>(
                    w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                    activation, activation_scale, assignment_tokens,
                    experts, output_tiles, situ, situ_scale, gate, up,
                    profile, static_cast<int>(rows), stream);
            }
        } else if (profile_enabled) {
            launch_native_candidate<false, false, true>(
                w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                activation, activation_scale, assignment_tokens,
                experts, output_tiles, situ, situ_scale, gate, up,
                profile, static_cast<int>(rows), stream);
        } else {
            launch_native_candidate<false, false, false>(
                w1_packed, w1_scale, w3_packed, w3_scale, layouts,
                activation, activation_scale, assignment_tokens,
                experts, output_tiles, situ, situ_scale, gate, up,
                profile, static_cast<int>(rows), stream);
        }
    } else {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            native_gate_up_baseline_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kBaselineGateUpSharedReservationBytes));
        native_gate_up_baseline_kernel
            <<<static_cast<int>(ctas), kDecodeCtaThreads,
               kBaselineGateUpSharedReservationBytes, stream>>>(
                reinterpret_cast<const std::uint8_t *>(
                    w1_packed.data_ptr()),
                reinterpret_cast<const std::uint8_t *>(
                    w1_scale.data_ptr()),
                reinterpret_cast<const std::uint8_t *>(
                    w3_packed.data_ptr()),
                reinterpret_cast<const std::uint8_t *>(
                    w3_scale.data_ptr()),
                reinterpret_cast<std::uint8_t *>(activation.data_ptr()),
                reinterpret_cast<std::uint8_t *>(
                    activation_scale.data_ptr()),
                reinterpret_cast<const int *>(
                    assignment_tokens.data_ptr()),
                reinterpret_cast<const int *>(experts.data_ptr()),
                reinterpret_cast<const int *>(output_tiles.data_ptr()),
                reinterpret_cast<std::uint8_t *>(situ.data_ptr()),
                reinterpret_cast<std::uint8_t *>(situ_scale.data_ptr()),
                static_cast<int>(rows));
        if (capture) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                native_gate_up_reference_capture_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kBaselineGateUpSharedReservationBytes));
            native_gate_up_reference_capture_kernel
                <<<static_cast<int>(ctas), kDecodeCtaThreads,
                   kBaselineGateUpSharedReservationBytes, stream>>>(
                    reinterpret_cast<const std::uint8_t *>(
                        w1_packed.data_ptr()),
                    reinterpret_cast<const std::uint8_t *>(
                        w1_scale.data_ptr()),
                    reinterpret_cast<const std::uint8_t *>(
                        w3_packed.data_ptr()),
                    reinterpret_cast<const std::uint8_t *>(
                        w3_scale.data_ptr()),
                    reinterpret_cast<std::uint8_t *>(
                        activation.data_ptr()),
                    reinterpret_cast<std::uint8_t *>(
                        activation_scale.data_ptr()),
                    reinterpret_cast<const int *>(
                        assignment_tokens.data_ptr()),
                    reinterpret_cast<const int *>(experts.data_ptr()),
                    reinterpret_cast<const int *>(
                        output_tiles.data_ptr()),
                    reinterpret_cast<std::uint8_t *>(situ.data_ptr()),
                    reinterpret_cast<std::uint8_t *>(
                        situ_scale.data_ptr()),
                    reinterpret_cast<float *>(gate.data_ptr()),
                    reinterpret_cast<float *>(up.data_ptr()),
                    static_cast<int>(rows));
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __host__ std::tuple<
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t>
native_gate_up_probe_resources() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        native_gate_up_candidate_kernel<false, false, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kNativeGateUpSharedReservationBytes));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks,
        native_gate_up_candidate_kernel<false, false, false>,
        kDecodeCtaThreads,
        kNativeGateUpSharedReservationBytes));
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attributes,
        native_gate_up_candidate_kernel<false, false, false>));
    return {
        kNativeGateUpSharedBytes,
        kNativeGateUpSharedReservationBytes,
        blocks,
        attributes.numRegs,
        attributes.localSizeBytes,
    };
}

}  // namespace native_gate_up_probe
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
