#pragma once

/// The launch's shared-memory budget, its grid, and its arrival counts.
///
/// What one CTA may allocate, how many CTAs the production grid has, which
/// candidate grids a guarded benchmark may pick from, and how many arrivals an
/// expert's gate/up work publishes. Every figure here is a `constexpr` that
/// something below launches with, so it comes first.

#include "kittens.cuh"

#include "../expert_mxfp4.cuh"
#include "../persistent_sync.cuh"
#include "../expert_mxfp4_fused_w13.cuh"
#include "../expert_mxfp4_grouped.cuh"
#include "../persistent_schedule.cuh"
#include "../router.cuh"
#include "../shared.cuh"
#include "../skinny_gemm.cuh"
#include "../tail_reduce.cuh"
#include "../tail_shard.cuh"
#include "../tail_sync.cuh"
#include "../types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace kimi_k3_decode {
namespace persistent {

// One `std::once_flag` per possible CUDA ordinal, so the shared-memory
// reservation and the occupancy query happen once per device even when one
// process drives several.
inline constexpr int kMaxCudaDevices = 32;

/// Benchmark grids are rounded above the widest complete tail role set.
inline constexpr int kBenchmarkGridQuantum = 32;
inline constexpr int kLargestTailRoleCtas =
    tail::kCoreRoleCtas > tail::kTensorRoleCtas
        ? tail::kCoreRoleCtas : tail::kTensorRoleCtas;
inline constexpr int kMinimumBenchmarkCtas =
    ((kLargestTailRoleCtas + kBenchmarkGridQuantum - 1)
        / kBenchmarkGridQuantum) * kBenchmarkGridQuantum;
inline constexpr int kMaximumBenchmarkCtas = kPersistentCtas;
inline constexpr int kBenchmarkGridStep = kMinimumBenchmarkCtas / 2;
inline constexpr std::array<int, 4> kBenchmarkGridCtas{
    kMinimumBenchmarkCtas,
    kMinimumBenchmarkCtas + kBenchmarkGridStep,
    kMinimumBenchmarkCtas + 2 * kBenchmarkGridStep,
    kMaximumBenchmarkCtas,
};

static_assert(kMinimumBenchmarkCtas >= tail::kCoordinatorCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kReduceCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kCoreShardCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kTensorShardCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kCoreRoleCtas);
static_assert(kMinimumBenchmarkCtas >= tail::kTensorRoleCtas);
static_assert(kMinimumBenchmarkCtas == 64);
static_assert(kMaximumBenchmarkCtas == kPersistentCtas);
static_assert(kBenchmarkGridCtas[2] < kBenchmarkGridCtas[3]);

/// The widest shared-memory footprint any stage this kernel runs asks for.
///
/// The routed gate/up unit's ring, by a wide margin: it holds two 64 KiB weight
/// slabs and the whole of one expert's activation, where no other stage holds
/// more than 128 KiB. So it is the stage the whole grid's request is sized to.
inline constexpr int kWidestStageSharedBytes =
    expert_mxfp4::fused_w13::kFusedW13SharedBytes;

static_assert(kWidestStageSharedBytes >= expert_mxfp4::kGateUpUnitSharedBytes);
static_assert(kWidestStageSharedBytes >= expert_mxfp4::kDownUnitSharedBytes);

// Each CTA holds all 512 tensor-memory columns, so the grid is only correct if
// every launched CTA lands alone on an SM. Requesting more than half of an
// SM's shared memory guarantees at most one resident CTA per SM,
// independently of any occupancy heuristic.
static_assert(2 * kPersistentSharedBytes > kittens::MAX_SHARED_MEMORY,
              "the persistent grid must be one CTA per SM");
static_assert(kPersistentSharedBytes
                  <= kittens::MAX_SHARED_MEMORY
                         - expert_mxfp4::fused_w13::kFusedStaticSharedReserve,
              "the persistent grid must leave room for static shared memory");
static_assert(kPersistentSharedBytes >= kWidestStageSharedBytes,
              "the persistent grid must fit its widest stage");
static_assert(
    kPersistentSharedBytes
        >= expert_mxfp4::grouped_pipeline::kGroupedDownPersistentSharedBytes,
    "the persistent grid must fit grouped routed down");
static_assert(kPersistentSharedBytes >= router::kSharedBytes,
              "the persistent grid must fit the router's scoring buffer");

// The grid's request is the gate/up ring's, so the two have to be one number.
// `kPersistentSharedBytes` is spelled independently in `persistent_sync.cuh`
// because the engine's header includes that one for its arrival primitive, and
// the include cannot go both ways.
static_assert(kPersistentSharedBytes
                  == expert_mxfp4::fused_w13::kFusedW13SharedBytes,
              "the persistent grid must request exactly the routed gate/up "
              "ring's footprint");

/// Routed gate/up units one occupied expert is decomposed into.
///
/// One, not six. The expert's six output tasks run inside a single claim, which
/// is what lets the unit gather that expert's activation once and let all six
/// tasks read it -- so five of every six queue claims disappear along with five
/// of every six gathers.
inline constexpr int kGateUpUnitsPerExpert = 1;

/// Gate/up arrivals one occupied expert publishes, which is what phase 4 waits
/// for.
///
/// Deliberately *not* the queue length. The unit finishes its six 64-column
/// ranges one at a time and publishes one arrival per range, because a routed
/// down unit cannot start until all 384 columns of `situ` exist no matter how
/// the work that wrote them was claimed.
inline constexpr int kGateUpArrivalsPerExpert =
    expert_mxfp4::fused_w13::kFusedTasks;

static_assert(kGateUpArrivalsPerExpert == 6);
static_assert(kGateUpArrivalsPerExpert * expert_mxfp4::fused_w13::
                                             kFusedSituGroups
                  == expert_mxfp4::kSituGroups,
              "the arrivals must cover every situ quantization group, or "
              "grouped down would start on a partly written expert");

// The instantiation must be one CTA per SM and must still be able to run every
// other stage of the step.
static_assert(2 * kPersistentSharedBytes > kittens::MAX_SHARED_MEMORY
                  && kPersistentSharedBytes >= kWidestStageSharedBytes
                  && kPersistentSharedBytes
                         >= expert_mxfp4::grouped_pipeline::
                                kGroupedDownPersistentSharedBytes
                  && kPersistentSharedBytes >= router::kSharedBytes,
              "the instantiation must stay one CTA per SM and still run every "
              "other stage of the step");

// ---------------------------------------------------------------------------

}  // namespace persistent
}  // namespace kimi_k3_decode
