#pragma once

/// What every entrypoint below checks before it does anything else.
///
/// The device is SM103, the tensors are aligned, and the device's property
/// record is read once per ordinal rather than once per step -- a decode hot
/// path that is otherwise free of runtime API calls cannot afford a kilobyte
/// query.

#include "pyutils/torchutils.cuh"

#include "../kernel.cuh"
#include "../mxfp4.cuh"
#include "../persistent_kernel.cuh"
#include "../types.cuh"
#include "../workspace_signature.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

namespace kimi_k3_decode {

inline std::int64_t kimi_k3_decode_workspace_bytes() noexcept {
    return SCRATCH_BYTES;
}

/// Return one device's immutable properties, querying the driver once.
///
/// `cudaGetDeviceProperties` fills a kilobyte-sized record and is far too heavy
/// to sit on a decode hot path that is otherwise free of runtime API calls.
/// The record never changes for a device, so it is read once per ordinal and
/// kept.
static __host__ const cudaDeviceProp &cached_device_properties(
    const int device_index
) {
    static std::array<cudaDeviceProp, persistent::kMaxCudaDevices> properties{};
    static std::array<std::once_flag, persistent::kMaxCudaDevices> queried;
    TORCH_CHECK(device_index >= 0
                    && device_index < persistent::kMaxCudaDevices,
                "MoK: Kimi K3 supports CUDA devices 0 through ",
                persistent::kMaxCudaDevices - 1, ", got ", device_index);
    const auto slot = static_cast<std::size_t>(device_index);
    std::call_once(queried[slot], [slot, device_index] {
        const cudaError_t status =
            cudaGetDeviceProperties(&properties[slot], device_index);
        TORCH_CHECK(status == cudaSuccess,
                    "MoK: cudaGetDeviceProperties failed for device ",
                    device_index, ": ", cudaGetErrorString(status));
    });
    return properties[slot];
}

static __host__ const cudaDeviceProp &check_sm103(
    const at::Tensor &hidden_states,
    const char *name
) {
    const cudaDeviceProp &properties =
        cached_device_properties(hidden_states.get_device());
    TORCH_CHECK(properties.major == 10 && properties.minor == 3,
                "MoK: ", name, " requires SM103, found sm_",
                properties.major, properties.minor);
    return properties;
}

static __host__ void check_tensor_alignment(
    const at::Tensor &tensor,
    const char *operation,
    const char *field,
    const int alignment
) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(address % static_cast<std::uintptr_t>(alignment) == 0,
                "MoK: ", operation, " requires ", field,
                " aligned to ", alignment, " bytes, got a pointer ",
                address % static_cast<std::uintptr_t>(alignment),
                " bytes past one");
}

}  // namespace kimi_k3_decode
