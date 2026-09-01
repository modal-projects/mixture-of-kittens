#pragma once

/// The host side: prove the grid co-resides, pick the schedule, launch it.
///
/// `launch_decode` is where the guarded switches are read and where the
/// dependency-local kernel is dispatched to. Its `default` arm is production's
/// adaptive engine, which is what a process that has set nothing gets.

#include "kernel.cuh"

namespace kimi_k3_decode {
namespace persistent {

// Host residency proof and launch.
// ---------------------------------------------------------------------------

/// How many times the reservation below actually ran, per CUDA ordinal.
///
/// A `std::once_flag` cannot be asked whether it has fired, so the count is
/// kept alongside it, for the test that the reservation happens on the device
/// the tensors live on rather than on whichever device happens to be current.
static __host__ std::array<std::atomic<int>, kMaxCudaDevices> &
shared_memory_reservations() {
    static std::array<std::atomic<int>, kMaxCudaDevices> counts{};
    return counts;
}

static __host__ std::int64_t shared_memory_reservations_for_testing(
    const std::int64_t device
) {
    TORCH_CHECK(device >= 0 && device < kMaxCudaDevices,
                "MoK: kimi_k3_decode tracks devices 0 through ",
                kMaxCudaDevices - 1, ", got ", device);
    return shared_memory_reservations()[static_cast<std::size_t>(device)].load(
        std::memory_order_relaxed);
}

/// Raise this kernel's shared-memory cap and measure its occupancy, once.
///
/// Both are properties of the compiled function rather than of a launch, so
/// caching them keeps the launch itself free of any runtime API call a CUDA
/// graph capture would have to record. The measured occupancy is then checked
/// on every call, so a device that cannot host the grid is rejected every time
/// rather than only on the first launch of a process.
template<bool TENSOR_PATH>
static __host__ int resident_blocks_per_sm() {
    static std::array<std::atomic<int>, kMaxCudaDevices> measured{};
    static std::array<std::once_flag, kMaxCudaDevices> reserved;
    constexpr int shared_bytes = kPersistentSharedBytes;
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(device >= 0 && device < kMaxCudaDevices,
                "MoK: kimi_k3_decode saw an unexpected device ordinal ",
                device);
    std::call_once(reserved[static_cast<std::size_t>(device)], [device] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kimi_k3_decode_persistent_kernel<TENSOR_PATH>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        int blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            kimi_k3_decode_persistent_kernel<TENSOR_PATH>,
            kDecodeCtaThreads, shared_bytes));
        measured[static_cast<std::size_t>(device)].store(
            blocks, std::memory_order_relaxed);
        // The count is the graph-capture contract: two per device, one per
        // capacity path, both paid before any capture.
        shared_memory_reservations()[static_cast<std::size_t>(device)]
            .fetch_add(1, std::memory_order_relaxed);
    });
    return measured[static_cast<std::size_t>(device)].load(
        std::memory_order_relaxed);
}

/// Reject a device that cannot hold the whole grid at once.
///
/// Every phase barrier counts all CTAs in the runtime launch grid, and a CTA
/// that is not resident cannot arrive, so a grid that only partly fits does
/// not run slowly -- it deadlocks. The occupancy query is the measurement that
/// matters; the SM count turns it into a whole-grid answer.
inline void validate_grid_residency(
    const std::int64_t available_sms,
    const std::int64_t blocks_per_sm,
    const std::int64_t grid_ctas
) {
    TORCH_CHECK(blocks_per_sm >= 1,
                "MoK: kimi_k3_decode requires the persistent kernel to place "
                "at least one CTA per SM at ", kDecodeCtaThreads,
                " threads and ", kPersistentSharedBytes,
                " dynamic shared bytes, but the device reports ",
                blocks_per_sm);
    TORCH_CHECK(available_sms >= grid_ctas,
                "MoK: kimi_k3_decode requires all ", grid_ctas,
                " CTAs of the persistent grid to co-reside one per SM, but the "
                "selected device exposes ", available_sms, " SMs");
}

inline void validate_residency(
    const std::int64_t available_sms,
    const std::int64_t blocks_per_sm
) {
    validate_grid_residency(
        available_sms, blocks_per_sm, kPersistentCtas);
}

inline std::int64_t resident_blocks_per_sm_for_testing(
    const bool tensor_path
) {
    return tensor_path ? resident_blocks_per_sm<true>()
                       : resident_blocks_per_sm<false>();
}

/// The same residency proof, for the guarded dependency-local candidate.
///
/// Its whole deadlock argument rests on every CTA of the launch being
/// co-resident, so the candidate has to be measured in its own right rather
/// than assumed to inherit the production kernel's occupancy.
///
/// The engine is a template parameter of the kernel, so each engine is a
/// different compiled function asking for a different number of dynamic shared
/// bytes, and residency is a property of the pair. The default is production's,
/// which is what every caller outside the A/B means.
inline std::int64_t schedule_resident_blocks_per_sm_for_testing(
    const bool tensor_path,
    const std::int64_t engine
) {
    namespace fused = expert_mxfp4::fused_w13;
    const int id = static_cast<int>(engine);
    TORCH_CHECK(fused::engine_is_known(id),
                "MoK: unknown Kimi K3 gate/up engine ", engine);
    if (id == fused::kEngineFusedResident) {
        return tensor_path
            ? schedule::resident_blocks_per_sm<
                  true, fused::kEngineFusedResident, TensorLayouts>()
            : schedule::resident_blocks_per_sm<
                  false, fused::kEngineFusedResident, NoTensorLayouts>();
    }
    return tensor_path
        ? schedule::resident_blocks_per_sm<
              true, fused::kEngineFusedAdaptive, TensorLayouts>()
        : schedule::resident_blocks_per_sm<
              false, fused::kEngineFusedAdaptive, NoTensorLayouts>();
}

/// Every pointer, alias, and count one persistent launch needs.
///
/// The kernel takes twenty-odd arguments and the two capacity paths pass the
/// same ones, so they travel together rather than being spelled out three
/// times between the entrypoint and the two launch helpers.
struct LaunchArguments {
    const at::Tensor &hidden_states;
    const at::Tensor &router_weight;
    const at::Tensor &router_correction_bias;
    const at::Tensor &routed_expert_down_proj;
    const at::Tensor &routed_expert_up_proj;
    const at::Tensor &routed_latent_rmsnorm_weight;
    const at::Tensor &expert_w13_packed;
    const at::Tensor &expert_w13_scale;
    const at::Tensor &expert_w2_packed;
    const at::Tensor &expert_w2_scale;
    const at::Tensor &shared_gate_proj;
    const at::Tensor &shared_up_proj;
    const at::Tensor &shared_down_proj;
    const at::Tensor &collective_buffer;
    std::int64_t collective_buffer_multicast_ptr;
    std::int64_t output_mailbox_multicast_ptr;
    const at::Tensor &barrier_buffer;
    std::int64_t barrier_buffer_multicast_ptr;
    const at::Tensor &barrier_target;
    const at::Tensor &scratch;
    const at::Tensor &error_flag;
    int tp_rank;
    int active_tokens;
    int available_sms;
    int grid_ctas;
    int profile_phases;
};

template<bool TENSOR_PATH>
static __host__ void launch_persistent(
    const LaunchArguments &arguments,
    const layouts_t<TENSOR_PATH> &layouts
) {
    validate_grid_residency(
        arguments.available_sms,
        resident_blocks_per_sm<TENSOR_PATH>(),
        arguments.grid_ctas);

    const auto bf16 = [](const at::Tensor &tensor) {
        return reinterpret_cast<const __nv_bfloat16 *>(tensor.data_ptr());
    };
    const auto bytes = [](const at::Tensor &tensor) {
        return reinterpret_cast<const std::uint8_t *>(tensor.data_ptr());
    };

    kimi_k3_decode_persistent_kernel<TENSOR_PATH>
        <<<arguments.grid_ctas, kDecodeCtaThreads, kPersistentSharedBytes,
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

/// Build the eleven TMA descriptors the tcgen05 stages read through.
static __host__ TensorLayouts tensor_layouts(
    const LaunchArguments &arguments
) {
    const auto tile = [](const void *pointer,
                         const std::int64_t rows,
                         const std::int64_t columns) {
        return tile_layout{
            const_cast<kittens::bf16 *>(
                reinterpret_cast<const kittens::bf16 *>(pointer)),
            nullptr, nullptr, static_cast<size_t>(rows),
            static_cast<size_t>(columns)};
    };
    const auto square = [](const void *pointer,
                           const std::int64_t rows,
                           const std::int64_t columns) {
        return square_layout{
            const_cast<kittens::bf16 *>(
                reinterpret_cast<const kittens::bf16 *>(pointer)),
            nullptr, nullptr, static_cast<size_t>(rows),
            static_cast<size_t>(columns)};
    };

    const Scratch pointers = scratch_view(
        reinterpret_cast<std::uint8_t *>(arguments.scratch.data_ptr()));
    const int active = arguments.active_tokens;
    constexpr int shared_intermediate = shared_experts::kIntermediate;

    return TensorLayouts{
        tile(arguments.hidden_states.data_ptr(), active, kHiddenSize),
        tile(arguments.routed_expert_down_proj.data_ptr(), kLatentSize,
             kHiddenSize),
        square(pointers.latent_x, active, kLatentSize),
        tile(arguments.shared_gate_proj.data_ptr(), shared_intermediate,
             kHiddenSize),
        tile(arguments.shared_up_proj.data_ptr(), shared_intermediate,
             kHiddenSize),
        tile(arguments.shared_down_proj.data_ptr(), kHiddenSize,
             shared_intermediate),
        square(pointers.shared_gate, active, shared_intermediate),
        square(pointers.shared_up, active, shared_intermediate),
        tile(pointers.shared_activated, active, shared_intermediate),
        tile(pointers.tail_normalized, active, kLatentSize),
        // Only this rank's contiguous 896-row slice of the replicated
        // latent-up weight is ever contracted.
        tile(reinterpret_cast<const __nv_bfloat16 *>(
                 arguments.routed_expert_up_proj.data_ptr())
                 + static_cast<long long>(arguments.tp_rank)
                       * tail::kShardColumns * kLatentSize,
             tail::kShardColumns, kLatentSize),
    };
}

/// Run one whole TP8 Kimi K3 decode step in one persistent launch.
///
/// One launch either way. Each schedule is its own kernel rather than a runtime
/// branch inside one, so neither pays the other's register pressure: all four
/// instantiations compile with nothing spilled and one CTA resident per SM.
///
/// Deliberately not the register counts. They move with the toolchain and with
/// any edit to either schedule, so a comment naming them is stale from some
/// later commit on -- this one said 194 and 248 against a build that produced
/// 196 and 255. The residency argument does not rest on a count anyway, it
/// rests on the invariant, and `test_neither_schedule_instantiation_spills` and
/// `test_neither_instantiation_spills` assert that against the built binary:
/// `STACK:0`, `LOCAL:0`, shared within the opt-in maximum, and one resident
/// block per SM. Those tests pin no count either, for the same reason. What a
/// given build produced is in `task-11c-sass.json`.
static __host__ void launch_decode(const LaunchArguments &arguments) {
    const bool core = capacity_bucket(arguments.active_tokens)
        <= kMaxCoreCapacity;
    if (dependency_schedule_enabled()) {
        // The engine is a template parameter, so each measured arm is its own
        // compiled kernel rather than a branch inside production's: no arm pays
        // another's register pressure. `default` is production, and it is the
        // only arm any caller outside a grid-tuning benchmark process can
        // reach -- `benchmark_gate_up_engine()` returns it unconditionally
        // without the environment guard set.
        namespace fused = expert_mxfp4::fused_w13;
        const int engine = benchmark_gate_up_engine();
        if (core) {
            switch (engine) {
                case fused::kEngineFusedResident:
                    schedule::launch_dependency_local<
                        false, fused::kEngineFusedResident>(
                            arguments, NoTensorLayouts{});
                    return;
                default:
                    schedule::launch_dependency_local<
                        false, fused::kEngineFusedAdaptive>(
                            arguments, NoTensorLayouts{});
                    return;
            }
        }
        switch (engine) {
            case fused::kEngineFusedResident:
                schedule::launch_dependency_local<
                    true, fused::kEngineFusedResident>(
                        arguments, tensor_layouts(arguments));
                return;
            default:
                schedule::launch_dependency_local<
                    true, fused::kEngineFusedAdaptive>(
                        arguments, tensor_layouts(arguments));
                return;
        }
    }
    if (core) {
        launch_persistent<false>(arguments, NoTensorLayouts{});
        return;
    }
    launch_persistent<true>(arguments, tensor_layouts(arguments));
}

}  // namespace persistent
}  // namespace kimi_k3_decode
