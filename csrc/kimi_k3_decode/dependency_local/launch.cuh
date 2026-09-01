#pragma once

/// The host side: prove the grid co-resides, then launch it.
///
/// The schedule's deadlock freedom rests on every CTA of the launch running at
/// once, so the occupancy query is not a tuning hint here but a precondition,
/// and it is cached per device because raising the shared cap is a one-time
/// property of a compiled function.

#include "kernel.cuh"

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// Host residency proof and launch.
// ---------------------------------------------------------------------------

/// Raise the candidate kernel's shared-memory cap and measure its occupancy.
///
/// Kept separate from the production kernel's query because both are
/// properties of a compiled function rather than of a launch: the candidate is
/// a different function and has to be proved resident in its own right.
template<bool TENSOR_PATH,
         int ENGINE = expert_mxfp4::fused_w13::kEngineFusedAdaptive,
         class Layouts>
static __host__ int resident_blocks_per_sm() {
    static std::array<std::atomic<int>, kScheduleMaxCudaDevices> measured{};
    static std::array<std::once_flag, kScheduleMaxCudaDevices> reserved;
    constexpr int shared_bytes = schedule_shared_bytes<ENGINE>;
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(device >= 0 && device < kScheduleMaxCudaDevices,
                "MoK: kimi_k3_decode saw an unexpected device ordinal ",
                device);
    std::call_once(reserved[static_cast<std::size_t>(device)], [device] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kimi_k3_decode_dependency_local_kernel<
                TENSOR_PATH, ENGINE, Layouts>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        int blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            kimi_k3_decode_dependency_local_kernel<
                TENSOR_PATH, ENGINE, Layouts>,
            kDecodeCtaThreads, shared_bytes));
        measured[static_cast<std::size_t>(device)].store(
            blocks, std::memory_order_relaxed);
    });
    return measured[static_cast<std::size_t>(device)].load(
        std::memory_order_relaxed);
}

/// Launch one dependency-local decode step.
///
/// Templated on the argument and layout types so this header never has to
/// include the one that includes it.
template<bool TENSOR_PATH,
         int ENGINE = expert_mxfp4::fused_w13::kEngineFusedAdaptive,
         class Arguments, class Layouts>
static __host__ void launch_dependency_local(
    const Arguments &arguments,
    const Layouts &layouts
) {
    const int blocks_per_sm =
        resident_blocks_per_sm<TENSOR_PATH, ENGINE, Layouts>();
    TORCH_CHECK(blocks_per_sm >= 1,
                "MoK: the dependency-local Kimi K3 schedule requires one CTA "
                "per SM at ", kDecodeCtaThreads, " threads and ",
                schedule_shared_bytes<ENGINE>,
                " dynamic shared bytes, but the device reports ",
                blocks_per_sm);
    TORCH_CHECK(arguments.available_sms >= arguments.grid_ctas,
                "MoK: the dependency-local Kimi K3 schedule requires all ",
                arguments.grid_ctas,
                " CTAs to co-reside one per SM, but the selected device "
                "exposes ", arguments.available_sms, " SMs");

    const auto bf16 = [](const at::Tensor &tensor) {
        return reinterpret_cast<const __nv_bfloat16 *>(tensor.data_ptr());
    };
    const auto bytes = [](const at::Tensor &tensor) {
        return reinterpret_cast<const std::uint8_t *>(tensor.data_ptr());
    };

    kimi_k3_decode_dependency_local_kernel<TENSOR_PATH, ENGINE, Layouts>
        <<<arguments.grid_ctas, kDecodeCtaThreads,
           schedule_shared_bytes<ENGINE>,
           at::cuda::getCurrentCUDAStream()>>>(
            bf16(arguments.hidden_states),
            bf16(arguments.router_weight),
            reinterpret_cast<const float *>(
                arguments.router_correction_bias.data_ptr()),
            bf16(arguments.routed_expert_down_proj),
            bf16(arguments.routed_expert_up_proj),
            bf16(arguments.routed_latent_rmsnorm_weight),
            *expert_mxfp4::fused_w13::fused_w13_packed_map(
                arguments.expert_w13_packed.data_ptr()),
            bytes(arguments.expert_w13_scale),
            bytes(arguments.expert_w2_packed),
            bytes(arguments.expert_w2_scale),
            bf16(arguments.shared_gate_proj),
            bf16(arguments.shared_up_proj),
            bf16(arguments.shared_down_proj),
            layouts,
            reinterpret_cast<__nv_bfloat16 *>(
                arguments.collective_buffer.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(
                arguments.collective_buffer_multicast_ptr),
            reinterpret_cast<__nv_bfloat16 *>(
                arguments.output_mailbox_multicast_ptr),
            reinterpret_cast<std::uint32_t *>(
                arguments.barrier_buffer_multicast_ptr),
            reinterpret_cast<const std::uint32_t *>(
                arguments.barrier_buffer.data_ptr()),
            reinterpret_cast<unsigned int *>(
                arguments.barrier_target.data_ptr()),
            reinterpret_cast<std::uint8_t *>(arguments.scratch.data_ptr()),
            reinterpret_cast<int *>(arguments.error_flag.data_ptr()),
            arguments.tp_rank,
            arguments.active_tokens,
            arguments.profile_phases);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
