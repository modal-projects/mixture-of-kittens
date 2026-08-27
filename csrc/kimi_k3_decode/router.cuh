#pragma once

#include "types.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace kimi_k3_decode {
namespace router {

inline constexpr int kWarps = kDecodeCtaThreads / 32;
static_assert(kNumExperts % 32 == 0,
              "the expert prefix scan splits 896 counts evenly across one warp");

// Scoring-phase shared memory: the token's hidden row plus one raw score and one
// selection key per expert, then the 16 chosen routes.
inline constexpr int kHiddenBytes = 0;
inline constexpr int kRawScoreBytes = kHiddenBytes + kHiddenSize * 2;
inline constexpr int kKeyBytes = kRawScoreBytes + kNumExperts * 4;
inline constexpr int kSelectedIdBytes = kKeyBytes + kNumExperts * 8;
inline constexpr int kSelectedWeightBytes = kSelectedIdBytes + kTopK * 4;
inline constexpr int kBestKeyBytes = kSelectedWeightBytes + kTopK * 4;
inline constexpr int kScoringSharedBytes = kBestKeyBytes + 8;

// Assignment-phase shared memory, which reuses the scoring region after the
// owning CTA has published its own routes.
inline constexpr int kCountBytes = 0;
inline constexpr int kOffsetBytes = kCountBytes + kNumExperts * 4;
inline constexpr int kRouteExpertBytes = kOffsetBytes + (kNumExperts + 1) * 4;
inline constexpr int kAssignmentSharedBytes = kRouteExpertBytes + kMaxRoutes * 4;

inline constexpr int kSharedBytes =
    kScoringSharedBytes > kAssignmentSharedBytes ? kScoringSharedBytes
                                                 : kAssignmentSharedBytes;

static_assert(kKeyBytes % 8 == 0 && kBestKeyBytes % 8 == 0,
              "64-bit selection keys must stay 8-byte aligned in shared memory");

/// Map an FP32 score onto an unsigned integer that compares in the same order.
static __device__ __forceinline__ unsigned int ordered_score_bits(const float value) {
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/// Pack a corrected score with its expert id so that a plain maximum resolves
/// "highest score, then lowest id" no matter what order the reduction runs in.
static __device__ __forceinline__ unsigned long long selection_key(
    const float corrected,
    const int expert
) {
    return (static_cast<unsigned long long>(ordered_score_bits(corrected)) << 32)
         | static_cast<unsigned long long>(0xffffffffu - static_cast<unsigned int>(expert));
}

static __device__ __forceinline__ int selection_key_expert(const unsigned long long key) {
    return static_cast<int>(0xffffffffu
                            - static_cast<unsigned int>(key & 0xffffffffull));
}

/// Kimi K3's router activation, evaluated without overflowing either tail.
static __device__ __forceinline__ float router_sigmoid(const float logit) {
    const float decayed = expf(-fabsf(logit));
    return logit >= 0.0f ? 1.0f / (1.0f + decayed) : decayed / (1.0f + decayed);
}

/// Build the expert-major assignment table for one decode step.
///
/// Only the last router CTA runs this, so every count, offset, and assignment
/// entry is written by a single CTA and the workspace never needs a host reset.
static __device__ void build_assignments(
    std::uint8_t *__restrict__ shared,
    const Scratch &scratch,
    const int active_tokens
) {
    int *const counts = reinterpret_cast<int *>(shared + kCountBytes);
    int *const offsets = reinterpret_cast<int *>(shared + kOffsetBytes);
    int *const route_experts = reinterpret_cast<int *>(shared + kRouteExpertBytes);

    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread % 32;
    const int routes = active_tokens * kTopK;

    for (int expert = thread; expert < kNumExperts; expert += kDecodeCtaThreads) {
        counts[expert] = 0;
    }
    __syncthreads();

    // Peer CTAs published these ids before arriving, so read past L1 to be sure
    // this CTA sees their values rather than a line it cached for its own token.
    for (int route = thread; route < routes; route += kDecodeCtaThreads) {
        const int expert = __ldcg(&scratch.expert_ids[route]);
        route_experts[route] = expert;
        atomicAdd(&counts[expert], 1);
    }
    __syncthreads();

    for (int expert = thread; expert < kNumExperts; expert += kDecodeCtaThreads) {
        scratch.expert_counts[expert] = counts[expert];
    }

    if (thread < 32) {
        constexpr int kPerLane = kNumExperts / 32;
        int block_total = 0;
        for (int i = 0; i < kPerLane; i++) block_total += counts[lane * kPerLane + i];
        int inclusive = block_total;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int carried = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= offset) inclusive += carried;
        }
        int running = inclusive - block_total;
        for (int i = 0; i < kPerLane; i++) {
            offsets[lane * kPerLane + i] = running;
            running += counts[lane * kPerLane + i];
        }
        if (lane == 31) offsets[kNumExperts] = running;
    }
    __syncthreads();

    for (int expert = thread; expert <= kNumExperts; expert += kDecodeCtaThreads) {
        scratch.expert_offsets[expert] = offsets[expert];
    }
    __syncthreads();

    // Walking tokens in order keeps every expert's slice token-major. A token's
    // 16 experts are distinct, so the 16 cursor bumps never touch one address.
    if (thread < 32) {
        for (int token = 0; token < active_tokens; token++) {
            if (lane < kTopK) {
                const int expert = route_experts[token * kTopK + lane];
                const int position = offsets[expert]++;
                scratch.assignment_tokens[position] = token;
                scratch.assignment_slots[position] = lane;
            }
            __syncwarp();
        }
    }
}

/// Score all 896 experts for one token, publish its top-16 routes, and, on the
/// last arriving CTA, publish the expert-major assignment table.
static __device__ void route_token(
    std::uint8_t *__restrict__ shared,
    const __nv_bfloat16 *__restrict__ hidden_states,
    const __nv_bfloat16 *__restrict__ router_weight,
    const float *__restrict__ router_correction_bias,
    const Scratch &scratch,
    int *__restrict__ expert_ids_out,
    float *__restrict__ expert_weights_out,
    const int token,
    const int active_tokens
) {
    __nv_bfloat16 *const hidden =
        reinterpret_cast<__nv_bfloat16 *>(shared + kHiddenBytes);
    float *const raw_scores = reinterpret_cast<float *>(shared + kRawScoreBytes);
    unsigned long long *const keys =
        reinterpret_cast<unsigned long long *>(shared + kKeyBytes);
    int *const selected_ids = reinterpret_cast<int *>(shared + kSelectedIdBytes);
    float *const selected_weights =
        reinterpret_cast<float *>(shared + kSelectedWeightBytes);
    unsigned long long *const best_key =
        reinterpret_cast<unsigned long long *>(shared + kBestKeyBytes);

    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread % 32;

    // Stage the token's row once; all 896 expert rows then stream past it.
    const __nv_bfloat16 *const hidden_row =
        hidden_states + static_cast<long long>(token) * kHiddenSize;
    for (int i = thread * 8; i < kHiddenSize; i += kDecodeCtaThreads * 8) {
        *reinterpret_cast<float4 *>(hidden + i) =
            *reinterpret_cast<const float4 *>(hidden_row + i);
    }
    __syncthreads();

    // One warp per expert keeps every 14 KB weight row read fully coalesced.
    for (int expert = warp; expert < kNumExperts; expert += kWarps) {
        const __nv_bfloat16 *const weight_row =
            router_weight + static_cast<long long>(expert) * kHiddenSize;
        float logit = 0.0f;
        for (int i = lane * 8; i < kHiddenSize; i += 32 * 8) {
            logit = accumulate_bf16_octet(
                *reinterpret_cast<const float4 *>(weight_row + i),
                *reinterpret_cast<const float4 *>(hidden + i),
                logit);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            logit += __shfl_down_sync(0xffffffffu, logit, offset);
        }
        if (lane == 0) {
            const float score = router_sigmoid(logit);
            raw_scores[expert] = score;
            keys[expert] =
                selection_key(score + router_correction_bias[expert], expert);
        }
    }
    __syncthreads();

    // The correction bias steers selection only; the raw scores carry the weight.
    for (int slot = 0; slot < kTopK; slot++) {
        if (thread == 0) *best_key = 0ull;
        __syncthreads();
        unsigned long long local = 0ull;
        for (int expert = thread; expert < kNumExperts; expert += kDecodeCtaThreads) {
            const unsigned long long candidate = keys[expert];
            if (candidate > local) local = candidate;
        }
        atomicMax(best_key, local);
        __syncthreads();
        if (thread == 0) {
            const int expert = selection_key_expert(*best_key);
            selected_ids[slot] = expert;
            selected_weights[slot] = raw_scores[expert];
            keys[expert] = 0ull;
        }
        __syncthreads();
    }

    if (thread == 0) {
        float total = 0.0f;
        for (int slot = 0; slot < kTopK; slot++) total += selected_weights[slot];
        const float divisor = total + 1e-20f;
        for (int slot = 0; slot < kTopK; slot++) {
            selected_weights[slot] = selected_weights[slot] / divisor;
        }
    }
    __syncthreads();

    if (thread < kTopK) {
        const int route = token * kTopK + thread;
        expert_ids_out[route] = selected_ids[thread];
        expert_weights_out[route] = selected_weights[thread];
        scratch.expert_ids[route] = selected_ids[thread];
        scratch.expert_weights[route] = selected_weights[thread];
    }

    // Every thread flushes its own route writes, and the barrier then holds the
    // ticket-taking thread until all of those flushes have landed device-wide.
    __threadfence();
    __syncthreads();
    __shared__ int owns_assignments;
    if (thread == 0) {
        const int ticket = atomicAdd(&scratch.phase[kRouterArrivals], 1);
        owns_assignments = (ticket == active_tokens - 1) ? 1 : 0;
    }
    __syncthreads();
    if (owns_assignments == 0) return;

    // Taking the last ticket means every peer CTA had already flushed, so the
    // acquire fence here makes their routes visible to this CTA's reads.
    __threadfence();
    build_assignments(shared, scratch, active_tokens);
    __syncthreads();
    __threadfence();
    __syncthreads();
    if (thread == 0) {
        atomicExch(&scratch.phase[kRouterArrivals], 0);
        atomicAdd(&scratch.phase[kRouterGeneration], 1);
    }
}

}  // namespace router
}  // namespace kimi_k3_decode
