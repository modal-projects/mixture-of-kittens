#pragma once

/// The two bounded probes, and the geometry dictionary the artifacts read.
///
/// Both probes are test-only entrypoints. One lands a single packed transaction
/// and reports where its five dimensions went and what it transferred; the other
/// reports the measured shared footprint against the launch's request. They are
/// compiled because what a descriptor does is a property of the device rather
/// than of the source, and they have callers -- which
/// `test_the_bounded_layout_probe_has_a_caller` asserts.

#include "descriptor.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

// ---------------------------------------------------------------------------
// The layout probe.
//
// Two things about the transfer above are properties of the hardware rather
// than of this file: how many bytes the mbarrier counts when the format widens
// its payload 2x, and whether the widened bytes land where a 128B-swizzled tile
// keeps them. Both are invisible in the decode step's numbers -- a wrong
// transaction count hangs, a wrong layout silently contracts the wrong
// weights -- so they are measured on their own, by one CTA, against one
// `(task, slab)` tile.
//
// The wait is bounded. A transaction count larger than the transfer would
// otherwise hang the device; here it reports zero and the caller sees which
// count is right rather than a timeout.
//
// `tests/test_kimi_k3_w13_layout.py` is the caller, and it is the only reason
// this probe is compiled: it runs the descriptor production launches with,
// over the full-width prepared payload, and checks the transaction count, all
// five dimensions, and that not one byte lands in the format's padding.
// ---------------------------------------------------------------------------

/// Roughly a hundred milliseconds of B300 clocks.
inline constexpr unsigned long long kFusedProbeTimeoutCycles = 200000000ull;

/// Test one mbarrier phase without blocking on it.
///
/// The same instruction `kittens::wait` spins on, spelled once so the probe can
/// give up instead of hanging the device when the transaction count is wrong.
__device__ __forceinline__ bool fused_probe_try_wait(
    kittens::semaphore &bar,
    const int phase
) {
    const std::uint32_t address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(&bar));
    std::uint32_t ready;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n"
        : "=r"(ready)
        : "r"(address), "r"(phase)
        : "memory"
    );
    return ready != 0u;
}

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void fused_w13_tma_probe_kernel(
    const __grid_constant__ CUtensorMap packed,
    std::uint8_t *__restrict__ dump,
    int *__restrict__ completed,
    const int expert,
    const int task_slab,
    const int transaction_bytes
) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];
    tma_swizzle_allocator staging(shared_raw);
    fused_weight_tile (&payload) = staging.allocate<fused_weight_tile>();

    __shared__ semaphore arrived;
    __shared__ int arrived_flag;
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) {
        init_semaphore(arrived, 0, 1);
        arrived_flag = 0;
    }
    // The tile is cleared so a partial transfer is visible as zeros rather than
    // as whatever the block last held.
    for (int index = thread; index < sizeof(fused_weight_tile) / 16;
         index += kDecodeCtaThreads) {
        reinterpret_cast<uint4 *>(payload.data)[index] =
            make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();

    if (thread == 0) {
        tma::expect_bytes(
            arrived, static_cast<std::uint32_t>(transaction_bytes));
        load_fused_slab_async(payload, &packed, expert, task_slab, arrived);
        const unsigned long long start = clock64();
        bool ready = false;
        while (!(ready = fused_probe_try_wait(arrived, 0))) {
            if (static_cast<unsigned long long>(clock64()) - start
                    > kFusedProbeTimeoutCycles) {
                break;
            }
        }
        arrived_flag = ready ? 1 : 0;
        *completed = arrived_flag;
    }
    __syncthreads();

    if (arrived_flag != 0) {
        // Read out through the tile's own `(row, column)` indexing rather than
        // as a flat span. That is the addressing `chunk_descriptor` and the
        // MMA use, so a dump that agrees with the transform proves the two
        // sides of the swizzle agree -- which a flat copy could not, because it
        // would only prove that some permutation of the right bytes arrived.
        for (int index = thread; index < kFusedM * kFusedSlabK;
             index += kDecodeCtaThreads) {
            const int row = index / kFusedSlabK;
            const int column = index % kFusedSlabK;
            dump[index] = *reinterpret_cast<const std::uint8_t *>(
                &payload[{row, column}]);
        }
    }
}

/// TEST-ONLY: run one fused weight transfer and report what landed.
///
/// Returns one `(task, slab)` tile as a row-major `[128, 512]` byte image, read
/// out of shared memory at the logical `(row, column)` the MMA's chunk
/// descriptors address, and a flag saying whether `transaction_bytes` completed
/// the mbarrier. The caller reconstructs the expected image from the transform,
/// so a descriptor that lies about its layout fails here rather than as wrong
/// decode numbers.
static __host__ std::tuple<at::Tensor, std::int64_t>
kimi_k3_fused_w13_tma_probe_entrypoint(
    const at::Tensor &expert_w13_packed,
    std::int64_t expert,
    std::int64_t task_slab,
    std::int64_t transaction_bytes
) {
    CHECK_INPUT(expert_w13_packed);
    TORCH_CHECK(expert_w13_packed.dim() == 3
                    && expert_w13_packed.size(0) == kNumExperts
                    && expert_w13_packed.size(1) == kFusedPackedRows
                    && expert_w13_packed.size(2) == kFusedPackedColumns
                    && expert_w13_packed.scalar_type() == at::kByte,
                "MoK: _kimi_k3_fused_w13_tma_probe requires uint8 "
                "expert_w13_packed [", kNumExperts, ", ", kFusedPackedRows,
                ", ", kFusedPackedColumns, "]");
    TORCH_CHECK(expert >= 0 && expert < kNumExperts,
                "MoK: _kimi_k3_fused_w13_tma_probe requires expert in [0, ",
                kNumExperts, ")");
    TORCH_CHECK(task_slab >= 0 && task_slab < kFusedTaskSlabs,
                "MoK: _kimi_k3_fused_w13_tma_probe requires task_slab in [0, ",
                kFusedTaskSlabs, ")");
    TORCH_CHECK(transaction_bytes > 0
                    && transaction_bytes
                           <= static_cast<std::int64_t>(
                                  sizeof(fused_weight_tile)),
                "MoK: _kimi_k3_fused_w13_tma_probe requires transaction_bytes "
                "in (0, ", sizeof(fused_weight_tile), "]");

    const c10::cuda::CUDAGuard device_guard(expert_w13_packed.device());
    at::Tensor dump = at::zeros(
        {static_cast<std::int64_t>(sizeof(fused_weight_tile))},
        expert_w13_packed.options());
    at::Tensor completed =
        at::zeros({1}, expert_w13_packed.options().dtype(at::kInt));

    CUtensorMap packed;
    create_fused_w13_packed_map(&packed, expert_w13_packed.data_ptr());

    constexpr int shared_bytes = static_cast<int>(sizeof(fused_weight_tile))
                               + 1024;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fused_w13_tma_probe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes));
    fused_w13_tma_probe_kernel
        <<<1, kDecodeCtaThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
            packed,
            reinterpret_cast<std::uint8_t *>(dump.data_ptr()),
            reinterpret_cast<int *>(completed.data_ptr()),
            static_cast<int>(expert),
            static_cast<int>(task_slab),
            static_cast<int>(transaction_bytes));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dump, static_cast<std::int64_t>(completed.cpu().item<int>())};
}

// ---------------------------------------------------------------------------
// The shared-footprint probe.
//
// How far past the dynamic block's first byte the ring's last array ends is a
// property of where the driver puts that first byte, which follows the static
// shared memory ptxas assigned and is therefore neither 1 KiB aligned nor
// knowable from this file. A ring sized to the byte overruns its block by
// exactly that offset, and the overrun is invisible in the arithmetic -- the
// bytes past the end belong to no one, so they read back whatever they held.
//
// So the offset is measured. This kernel runs the same allocator sequence the
// engine does, under the same launch configuration, and reports where the
// sequence started and ended without writing a single byte of it.
// ---------------------------------------------------------------------------

static __global__ __launch_bounds__(kDecodeCtaThreads, 1)
void fused_w13_shared_footprint_kernel(int *__restrict__ report) {
    using namespace kittens;
    extern __shared__ __align__(16) int shared_raw[];
    tma_swizzle_allocator staging(shared_raw);
    staging.allocate<fused_weight_tile, kFusedStages>();
    staging.allocate<mixed_scale_tile, kFusedStages, kFusedSlabScaleTiles>();
    staging.allocate<fused_activation_tile, kFusedActivationSlabs>();
    staging.allocate<
        mixed_scale_tile, kFusedActivationSlabs, kFusedSlabScaleTiles>();
    staging.allocate<fused_result_tile>();
    if (threadIdx.x == 0) {
        const std::uint32_t base = static_cast<std::uint32_t>(
            __cvta_generic_to_shared(shared_raw));
        const std::uint32_t end = static_cast<std::uint32_t>(
            __cvta_generic_to_shared(staging.ptr));
        report[0] = static_cast<int>(end - base);
        report[1] = static_cast<int>(base % kFusedAllocatorPadding);
    }
}

/// TEST-ONLY: how many dynamic shared bytes the ring really needs.
///
/// Returns the bytes the allocator consumes measured from the dynamic block's
/// own first byte, that block's offset within the 1 KiB grid the allocator
/// aligns to, and the bytes the fused instantiation launches with. The first
/// must not exceed the third.
static __host__ std::tuple<std::int64_t, std::int64_t, std::int64_t>
kimi_k3_fused_w13_shared_footprint_entrypoint() {
    at::Tensor report = at::zeros(
        {2}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fused_w13_shared_footprint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kFusedW13SharedBytes));
    fused_w13_shared_footprint_kernel
        <<<1, kDecodeCtaThreads, kFusedW13SharedBytes,
           at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<int *>(report.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const at::Tensor host = report.cpu();
    return {static_cast<std::int64_t>(host[0].item<int>()),
            static_cast<std::int64_t>(host[1].item<int>()),
            kFusedW13SharedBytes};
}

/// Every number the engine's shape rests on, for the tests that check them.
///
/// Reported rather than recomputed in Python because the whole point of the
/// shared-byte accounting is that it is the accounting the kernel launches
/// with: a test that rebuilt the arithmetic from the same reasoning the header
/// uses would agree with the header and not with the device.
inline std::map<std::string, std::int64_t>
fused_w13_geometry_for_testing() {
    return {
        {"tasks", kFusedTasks},
        {"slabs", kFusedSlabs},
        {"slab_k", kFusedSlabK},
        {"slab_groups", kFusedSlabGroups},
        {"slab_scale_tiles", kFusedSlabScaleTiles},
        {"task_slabs", kFusedTaskSlabs},
        {"m", kFusedM},
        {"n", kFusedN},
        {"physical_n", kFusedPhysicalN},
        {"half_rows", kFusedHalfRows},
        {"boxes", kFusedBoxes},
        {"box_elements", kFusedBoxElements},
        {"swizzle_bytes", fused_weight_tile::swizzle_bytes},
        {"packed_rows", kFusedPackedRows},
        {"packed_columns", kFusedPackedColumns},
        {"scale_rows", kFusedScaleRows},
        {"scale_columns", kFusedScaleColumns},
        {"stages", kFusedStages},
        {"activation_slabs", kFusedActivationSlabs},
        {"stream_length", kFusedStreamLength},
        {"weight_tile_bytes",
         static_cast<std::int64_t>(sizeof(fused_weight_tile))},
        {"activation_tile_bytes",
         static_cast<std::int64_t>(sizeof(fused_activation_tile))},
        {"result_tile_bytes",
         static_cast<std::int64_t>(sizeof(fused_result_tile))},
        {"weight_transaction_bytes", kFusedWeightTransactionBytes},
        {"slab_transaction_bytes", kFusedSlabTransactionBytes},
        {"scale_slots", kFusedScaleSlots},
        {"scale_sets", kFusedScaleSets},
        {"staging_bytes", kFusedStagingBytes},
        {"allocator_padding", kFusedAllocatorPadding},
        {"static_shared_reserve", kFusedStaticSharedReserve},
        {"shared_bytes", kFusedW13SharedBytes},
        {"opt_in_maximum", kittens::MAX_SHARED_MEMORY},
    };
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
