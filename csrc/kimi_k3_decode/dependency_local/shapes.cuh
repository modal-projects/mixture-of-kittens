#pragma once

/// The shapes and counter bounds the dependency-local schedule pins.
///
/// How many units each queue holds, how many arrivals an expert publishes, and
/// how large a ticket can grow before it would alias the counter's sentinel.
/// Everything here is a `constexpr` the queues, the kernel and the host launch
/// all read, so it comes first and depends on nothing in this directory.

#include "kittens.cuh"

#include "../expert_mxfp4.cuh"
#include "../expert_mxfp4_fused_w13.cuh"
#include "../expert_mxfp4_grouped.cuh"
#include "../persistent_sync.cuh"
#include "../router.cuh"
#include "../shared.cuh"
#include "../skinny_gemm.cuh"
#include "../tail_reduce.cuh"
#include "../tail_shard.cuh"
#include "../tail_sync.cuh"
#include "../types.cuh"

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// ---------------------------------------------------------------------------
// Shapes the schedule pins.
// ---------------------------------------------------------------------------

/// CUDA ordinals the candidate's residency cache covers.
///
/// Spelled here rather than borrowed from `persistent_kernel.cuh`, because that
/// header includes this one and the include cannot go both ways.
inline constexpr int kScheduleMaxCudaDevices = 32;

/// Queue claims one occupied expert's routed gate/up work is one unit, and the
/// arrivals that unit publishes as its six `situ` ranges complete.
///
/// Both are the production facts, restated here for the same include reason,
/// and both are asserted against the engine's own geometry below.
inline constexpr int kScheduleGateUpUnitsPerExpert = 1;
inline constexpr int kScheduleGateUpArrivalsPerExpert =
    expert_mxfp4::fused_w13::kFusedTasks;

static_assert(kScheduleGateUpArrivalsPerExpert == 6);
static_assert(kScheduleGateUpArrivalsPerExpert == kScheduleExpertGateUpArrivals,
              "routed down's per-expert wait reads its target from the edge "
              "table, which must name the number of arrivals the fused engine "
              "actually publishes");
static_assert(kScheduleGateUpArrivalsPerExpert
                      * expert_mxfp4::fused_w13::kFusedSituGroups
                  == expert_mxfp4::kSituGroups,
              "the arrivals must cover every situ quantization group, or "
              "grouped down would start on a partly written expert");

/// Logical units the publish queue is cut into.
///
/// One per production CTA, so the write pattern is exactly the grid-strided
/// loop the production kernel's publish phase runs with `block` replaced by a
/// claimed unit index. Every element is still written once by one thread, so
/// the published bits do not depend on which CTA claimed which slice.
inline constexpr int kSchedulePublishUnits = kPersistentCtas;

static_assert(kSchedulePublishUnits == kSchedulePublishUnitsForTable,
              "the tail's wait on the publish queue reads its target from the "
              "edge table, which must name the same number of units the queue "
              "actually has");

/// Latent quantization groups one projection unit owns.
///
/// A projection unit writes a contiguous column range of the routed latent and
/// MXFP8 groups are 32 contiguous columns, so a unit can quantize exactly the
/// groups it just produced and the separate grid-wide quantization pass -- and
/// the barrier in front of it -- disappear. The tensor path's 128-column tile
/// is four groups; the core path's 32-column block is one.
template<bool TENSOR_PATH>
inline constexpr int kLatentGroupsPerProjectionUnit =
    TENSOR_PATH ? skinny_gemm::kTileN / expert_mxfp4::kMmaK
                : skinny_gemm::kCoreColumnsPerCta / expert_mxfp4::kMmaK;

template<bool TENSOR_PATH>
inline constexpr int kProjectionUnits =
    TENSOR_PATH ? skinny_gemm::kTensorCtas : skinny_gemm::kCoreCtas;

template<bool TENSOR_PATH>
inline constexpr int kSharedGateUpUnits =
    TENSOR_PATH ? 2 * shared_experts::kTensorGateCtas
                : shared_experts::kCoreGateCtas;

template<bool TENSOR_PATH>
inline constexpr int kSharedActivationUnits =
    TENSOR_PATH ? shared_experts::kActivationCtas : 0;

template<bool TENSOR_PATH>
inline constexpr int kSharedDownUnits =
    TENSOR_PATH ? shared_experts::kTensorDownCtas
                : shared_experts::kCoreDownCtas;

static_assert(kLatentGroupsPerProjectionUnit<true> == 4);
static_assert(kLatentGroupsPerProjectionUnit<false> == 1);
static_assert(kProjectionUnits<true> * kLatentGroupsPerProjectionUnit<true>
                  == expert_mxfp4::kLatentGroups,
              "the tensor projection units must cover every latent group");
static_assert(kProjectionUnits<false> * kLatentGroupsPerProjectionUnit<false>
                  == expert_mxfp4::kLatentGroups,
              "the core projection units must cover every latent group");
static_assert(kScheduleSharedPairs == shared_experts::kTensorGateCtas,
              "one pair counter per shared column block");
static_assert(kScheduleSharedPairs == shared_experts::kActivationCtas,
              "an activation unit consumes exactly one shared column pair");
static_assert(kSharedGateUpUnits<true> == 2 * kScheduleSharedPairs,
              "the tensor path's shared gate and up units pair up");

// ---------------------------------------------------------------------------
// Counter bounds.
//
// Nothing inside a launch resets a queue ticket or a readiness arrival, so
// every one of them has to hold its own maximum without approaching the
// unsigned wrap. The bounds below are the reason the candidate needs no
// generation arithmetic for its own state: block 0 zeroes the whole band
// before the one retained barrier, and no counter can climb out of `int` from
// there.
// ---------------------------------------------------------------------------

/// The highest ticket any queue counter can reach.
///
/// A CTA leaves a queue on its first refused claim, so each of them can add
/// one maximum-width batch beyond the logical length.
inline constexpr int kScheduleLongestQueueUnits =
    kNumExperts * expert_mxfp4::grouped_pipeline::kGroupedDownUnits;
inline constexpr int kScheduleLongestTicket =
    kScheduleLongestQueueUnits + kRoutedClaimBatch * kPersistentCtas;

/// The highest value any readiness arrival counter can reach.
inline constexpr int kScheduleLargestArrival = kScheduleLongestQueueUnits;

inline constexpr int kScheduleCounterBound = 0x7fffffff;

static_assert(kScheduleLongestQueueUnits == 6272);
static_assert(kScheduleLongestTicket == 6864);
static_assert(kScheduleLongestTicket < kScheduleCounterBound,
              "a queue ticket must stay inside a signed 32-bit counter");
static_assert(kScheduleLargestArrival < kScheduleCounterBound,
              "a readiness arrival must stay inside a signed 32-bit counter");
static_assert(kMaxTokens * router::kScoreShards < kScheduleCounterBound);
static_assert(kSchedulePublishUnits + kPersistentCtas
                  < kScheduleCounterBound);
static_assert(kScheduleCounterBound == 0x7fffffff,
              "the bound is the signed 32-bit maximum, and every counter above "
              "is asserted strictly under it");

// ---------------------------------------------------------------------------

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
