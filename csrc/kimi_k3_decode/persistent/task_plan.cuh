#pragma once

/// How many logical units each phase of a step decomposes into.
///
/// The queues hand work out rather than mapping one task to one block, so the
/// plan is a function of the token count alone, and the longest it can ever be
/// is what the queue counters are sized against.

#include "budget.cuh"

namespace kimi_k3_decode {
namespace persistent {

// Logical task counts.
// ---------------------------------------------------------------------------

/// Every logical task of one decode step, by phase.
///
/// The routed counts are reported at their bound, because the exact number of
/// occupied experts is only known on the device: a token's sixteen routes are
/// distinct experts, so at most `min(16 * active_tokens, 896)` experts take an
/// assignment and each of those is exactly one 128-row batch.
struct TaskPlan {
    int route_latent;
    int gate_up;
    int down;
    int tail;
    int grid;
};

inline constexpr TaskPlan task_plan(const int active_tokens) {
    const bool tensor_path = capacity_bucket(active_tokens) > kMaxCoreCapacity;
    const int experts = kTopK * active_tokens < kNumExperts
        ? kTopK * active_tokens : kNumExperts;
    return TaskPlan{
        active_tokens * router::kScoreShards
            + (tensor_path ? skinny_gemm::kTensorCtas
                           : skinny_gemm::kCoreCtas)
            + (tensor_path ? 2 * shared_experts::kTensorGateCtas
                           : shared_experts::kCoreGateCtas),
        experts * kGateUpUnitsPerExpert,
        (tensor_path ? shared_experts::kActivationCtas : 0)
            + experts
                  * expert_mxfp4::grouped_pipeline::kGroupedDownUnits
            + (tensor_path ? shared_experts::kTensorDownCtas
                           : shared_experts::kCoreDownCtas),
        tail::kCoordinatorCtas + tail::kReduceCtas
            + (tensor_path ? tail::kTensorShardCtas : tail::kCoreShardCtas),
        kPersistentCtas,
    };
}

/// The longest queue any phase of any accepted shape hands out.
///
/// Folded over `task_plan` rather than written out, because which phase is
/// widest is not obvious: the core path's shared-down role is 112 units to the
/// tensor path's 62, but the core path only ever runs eight rows, which caps it
/// at 128 occupied experts. The widest is the tensor path's grouped down phase
/// at a full 896 experts.
inline constexpr int longest_queue_units() {
    int longest = 0;
    for (int tokens = 1; tokens <= kMaxTokens; ++tokens) {
        const TaskPlan plan = task_plan(tokens);
        for (const int units :
             {plan.route_latent, plan.gate_up, plan.down, plan.tail}) {
            if (units > longest) longest = units;
        }
    }
    return longest;
}

/// The longest queue, and the highest routed counter value it can reach.
///
/// Both bounds matter. The length is what a unit index is decoded against, and
/// the ticket is what the counter has to hold without wrapping, because nothing
/// resets it inside a launch. The widest shapes use the maximum routed batch,
/// so the bound rounds the logical length up to that width and then lets every
/// CTA add one refused batch.
inline constexpr int kLongestQueueUnits = longest_queue_units();
inline constexpr int kLongestQueueRounded =
    (kLongestQueueUnits + kRoutedClaimBatch - 1)
        / kRoutedClaimBatch * kRoutedClaimBatch;
inline constexpr int kLongestQueueTicket =
    kLongestQueueRounded + kRoutedClaimBatch * kMaximumBenchmarkCtas;

static_assert(kLongestQueueUnits == 6334,
              "the widest phase is 6 activation + 56 shared-down + 896 * 7 "
              "routed down units");
static_assert(kLongestQueueTicket
                  == kLongestQueueRounded
                       + kRoutedClaimBatch * kPersistentCtas,
              "the ticket bound must cover the largest production grid");
static_assert(kLongestQueueTicket == 6928,
              "the rounded widest phase plus one refused batch per CTA");
static_assert(static_cast<unsigned int>(kLongestQueueTicket)
                  < 0xffffffffu / 2u,
              "a queue counter must not approach the unsigned wrap");
static_assert(kNumExperts * kGateUpUnitsPerExpert <= kLongestQueueUnits,
              "the gate/up queue must fit the bound every queue counter is "
              "sized against");

inline std::tuple<int, int, int, int, int> task_plan_for_testing(
    const std::int64_t active_tokens
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: kimi_k3_decode task plan requires active_tokens in [1, ",
                kMaxTokens, "]");
    const TaskPlan plan = task_plan(static_cast<int>(active_tokens));
    return {plan.route_latent, plan.gate_up, plan.down, plan.tail, plan.grid};
}

inline std::tuple<int, int, int, int, int> timeout_metadata_for_testing() {
    return {
        kPersistentTimeoutPhase,
        kGridGeneration,
        kActivationArrivals,
        kActiveExpertUnits,
        kTimeoutClaim};
}

inline std::tuple<int, int> queue_bound_for_testing() {
    return {kLongestQueueUnits, kLongestQueueTicket};
}

}  // namespace persistent
}  // namespace kimi_k3_decode
