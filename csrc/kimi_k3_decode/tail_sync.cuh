#pragma once

#include "kittens.cuh"

#include "../serial_sync.cuh"
#include "skinny_gemm.cuh"
#include "types.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace kimi_k3_decode {
namespace tail {

// The fused TP8 tail closes one decode step in a single launch per rank. It
// all-reduces the routed latent and reduce-scatters the shared output straight
// out of the symmetric collective buffer with SM103 multimem instructions,
// normalizes the replicated latent in FP32, contracts this rank's 896 rows of
// the replicated latent-up weight, beta-adds the reduced shared shard, and
// multicasts the resulting token-major shard into every rank's mailbox slot.
//
// This header holds the shape constants every role shares, the register-resident
// multimem primitives, and the generation-tagged synchronization those roles
// rendezvous on. `tail_reduce.cuh` and `tail_shard.cuh` hold the two producer
// and consumer roles; `collectives.cuh` assembles them into the kernels.

inline constexpr int kCollectiveColumns = kLatentSize + kHiddenSize;
inline constexpr int kShardColumns = kHiddenSize / kTensorParallelSize;
inline constexpr float kRmsEpsilon = 1e-5f;

// Every multimem access moves one 16-byte `.v4.bf16x2` octet. The collective
// buffer, the mailbox, and the RMSNorm weight are all 16-byte aligned and every
// extent below is a multiple of eight BF16 values, so no access straddles a
// vector boundary.
inline constexpr int kOctetLanes = 8;
inline constexpr int kLatentOctets = kLatentSize / kOctetLanes;
inline constexpr int kShardOctets = kShardColumns / kOctetLanes;

inline constexpr int kWarps = kDecodeCtaThreads / 32;

// Role bands, producers first. Block 0 owns both cross-rank edges, the reduce
// band publishes the normalized latent and the reduced shared shard, and the
// shard band consumes both. The host residency guard keeps progress independent
// of the order the scheduler happens to launch these blocks in.
inline constexpr int kCoordinatorCtas = 1;
inline constexpr int kReduceCtas = 32;
inline constexpr int kCoordinatorBegin = 0;
inline constexpr int kReduceBegin = kCoordinatorBegin + kCoordinatorCtas;
inline constexpr int kShardBegin = kReduceBegin + kReduceCtas;

// Direct-register shard path, used for capacities 1, 2, 4, and 8. One warp owns
// eight adjacent output columns so its lane 0 holds exactly one multimem octet
// after the cross-lane reduction and the mailbox store needs no staging.
inline constexpr int kCoreShardColumnsPerWarp = kOctetLanes;
inline constexpr int kCoreShardColumnsPerCta =
    kWarps * kCoreShardColumnsPerWarp;
inline constexpr int kCoreShardCtas = kShardColumns / kCoreShardColumnsPerCta;
inline constexpr int kCoreLatentChunk = 512;
inline constexpr int kCoreRoleCtas = kShardBegin + kCoreShardCtas;
inline constexpr int kTailCoreDynamicBytes =
    kMaxCoreCapacity * kCoreLatentChunk * sizeof(__nv_bfloat16);

// tcgen05 shard path, used for capacities 16, 32, 64, and 128. One 128-row MMA
// tile covers every one of those capacities because the input descriptor spans
// exactly the active rows and the hardware zero-fills the remainder.
inline constexpr int kTileM = skinny_gemm::kTileM;
inline constexpr int kTileN = skinny_gemm::kTileN;
inline constexpr int kTileK = skinny_gemm::kTileK;
inline constexpr int kStages = skinny_gemm::kStages;
inline constexpr int kTensorShardCtas = kShardColumns / kTileN;
inline constexpr int kTensorKIterations = kLatentSize / kTileK;
inline constexpr int kTensorRoleCtas = kShardBegin + kTensorShardCtas;
inline constexpr int kTailTensorDynamicBytes =
    kittens::MAX_SHARED_MEMORY - 1024;

static_assert(kShardColumns == 896);
static_assert(kLatentSize % kOctetLanes == 0);
static_assert(kShardColumns % kOctetLanes == 0);
static_assert(kShardColumns % kCoreShardColumnsPerCta == 0);
static_assert(kLatentSize % kCoreLatentChunk == 0);
static_assert(kCoreLatentChunk % (kOctetLanes * 32) == 0);
static_assert(kShardColumns % kTileN == 0);
static_assert(kLatentSize % kTileK == 0);
static_assert(kCoordinatorBegin < kReduceBegin);
static_assert(kReduceBegin < kShardBegin);
static_assert(kCoreShardCtas == 14);
static_assert(kTensorShardCtas == 7);
static_assert(kCoreRoleCtas == 47);
static_assert(kTensorRoleCtas == 40);

using tensor_input_tile = skinny_gemm::hidden_tile;
using tensor_weight_tile = skinny_gemm::weight_tile;
using tensor_result_tile = kittens::st_fl<kTileM, kTileN>;
using tensor_accumulator_tile = skinny_gemm::accumulator_tile;
using tensor_input_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_input_tile>;
using tensor_weight_layout =
    kittens::gl<kittens::bf16, 1, 1, -1, -1, tensor_weight_tile>;

// ---------------------------------------------------------------------------
// Wrap-safe generation, barrier, and timeout arithmetic, shared with the host.
//
// These are the repository-wide primitives from `csrc/serial_sync.cuh`, so the
// tail and `utils::barrier_all` interpret the barrier counter pair they share
// identically, including across the unsigned wrap.
// ---------------------------------------------------------------------------

using serial_sync::barrier_reached;
using serial_sync::generation_advanced;
using serial_sync::wait_timed_out;

// Roughly fifteen seconds of B300 clocks. Every cross-rank and cross-CTA spin
// in the tail is bounded by it, so a lost peer surfaces as a device trap with a
// recorded phase slot instead of an unkillable GPU.
inline constexpr std::uint64_t kGenerationWaitTimeoutClocks =
    serial_sync::kWaitTimeoutClocks;

// ---------------------------------------------------------------------------
// Register-resident 16-byte BF16 octets.
// ---------------------------------------------------------------------------

struct Octet {
    std::uint32_t pair[4];
};

/// Sum one octet across all eight ranks through the symmetric multicast alias.
static __device__ __forceinline__ void multimem_reduce_octet(
    Octet &out,
    const __nv_bfloat16 *const address
) {
    asm volatile(
        "multimem.ld_reduce.weak.global.add.acc::f32.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(out.pair[0]), "=r"(out.pair[1]),
          "=r"(out.pair[2]), "=r"(out.pair[3])
        : "l"(address)
        : "memory");
}

/// Write one octet into every rank's copy of a symmetric allocation.
static __device__ __forceinline__ void multimem_store_octet(
    __nv_bfloat16 *const address,
    const Octet &value
) {
    asm volatile(
        "multimem.st.weak.global.v4.bf16x2 [%0], {%1, %2, %3, %4};"
        :: "l"(address), "r"(value.pair[0]), "r"(value.pair[1]),
           "r"(value.pair[2]), "r"(value.pair[3])
        : "memory");
}

static __device__ __forceinline__ void load_octet(
    Octet &out,
    const __nv_bfloat16 *const address
) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(address);
    out.pair[0] = raw.x;
    out.pair[1] = raw.y;
    out.pair[2] = raw.z;
    out.pair[3] = raw.w;
}

static __device__ __forceinline__ void store_octet(
    __nv_bfloat16 *const address,
    const Octet &value
) {
    *reinterpret_cast<uint4 *>(address) =
        make_uint4(value.pair[0], value.pair[1], value.pair[2], value.pair[3]);
}

static __device__ __forceinline__ float low_bf16(const std::uint32_t bits) {
    return __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(bits & 0xffffu)));
}

static __device__ __forceinline__ float high_bf16(const std::uint32_t bits) {
    return __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16)));
}

static __device__ __forceinline__ std::uint32_t pack_bf16(
    const float low,
    const float high
) {
    return static_cast<std::uint32_t>(
               __bfloat16_as_ushort(__float2bfloat16(low)))
         | (static_cast<std::uint32_t>(
                __bfloat16_as_ushort(__float2bfloat16(high))) << 16);
}

// ---------------------------------------------------------------------------
// Generation-tagged phases. Nothing here needs a host reset between launches,
// so a captured graph replays without zeroing the workspace.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ std::uint32_t load_relaxed_gpu(
    const int *const address
) {
    std::uint32_t value;
    asm volatile(
        "{ld.relaxed.gpu.global.u32 %0, [%1];}"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}

static __device__ __forceinline__ std::uint32_t load_relaxed_system(
    const std::uint32_t *const address
) {
    std::uint32_t value;
    asm volatile(
        "{ld.relaxed.sys.global.u32 %0, [%1];}"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}

/// Record which wait gave up, and on what, then trap this thread's launch.
///
/// The scratch slot names the generation counter the wait was on and
/// `error_flag` names the site, because the tail has six bounded waits and only
/// four counters between them. Both writes are released at system scope before
/// the trap, so a host that reads either one after the launch aborts sees the
/// value this thread wrote rather than whatever was there before.
static __device__ __forceinline__ void record_timeout_and_trap(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int generation_index,
    const int error_code
) {
    atomicExch(
        reinterpret_cast<unsigned int *>(&scratch.phase[kTailTimeoutPhase]),
        static_cast<unsigned int>(generation_index));
    atomicExch(reinterpret_cast<unsigned int *>(error_flag),
               static_cast<unsigned int>(error_code));
    __threadfence_system();
    asm volatile("trap;");
}

/// Latch this CTA's baseline for one generation before any producer can move it.
///
/// Every role reads the generation it publishes itself, which lags the
/// generation it consumes by exactly one launch. The last arrival of a role only
/// advances that role's generation once every CTA in the role has already
/// latched its baseline, so no CTA can observe a baseline from the wrong launch.
static __device__ std::uint32_t latch_generation(
    const Scratch &scratch,
    const int generation_index,
    std::uint32_t *const slot
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *slot = load_relaxed_gpu(&scratch.phase[generation_index]);
    }
    __syncthreads();
    return *slot;
}

/// Spin until `generation_index` moves past `baseline`, then acquire.
static __device__ void wait_for_generation(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    const int generation_index,
    const std::uint32_t baseline,
    const int error_code
) {
    if (threadIdx.x == 0) {
        const std::uint64_t started = clock64();
        while (!generation_advanced(
            load_relaxed_gpu(&scratch.phase[generation_index]), baseline
        )) {
            if (wait_timed_out(started, clock64())) {
                record_timeout_and_trap(
                    scratch, error_flag, generation_index, error_code);
            }
            __nanosleep(64);
        }
    }
    // Thread 0's observation converges the CTA. Peer writes reach this rank
    // through the coordinator's system-scope acquire, so every consumer thread
    // fences at system scope before reading the collective buffer or a scratch
    // intermediate that a peer's data flowed into.
    __syncthreads();
    __threadfence_system();
}

/// Release this role's writes, then take one arrival ticket for this CTA.
static __device__ void publish_generation(
    const Scratch &scratch,
    const int arrivals_index,
    const int generation_index,
    const int role_ctas
) {
    // Fence per thread, then converge, so the ticket-taking thread only counts
    // this CTA once every write it made is visible to peers and to consumers.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        auto *const arrivals = reinterpret_cast<unsigned int *>(
            &scratch.phase[arrivals_index]);
        const unsigned int ticket = atomicAdd(arrivals, 1u);
        if (ticket == static_cast<unsigned int>(role_ctas - 1)) {
            atomicExch(arrivals, 0u);
            atomicAdd(
                reinterpret_cast<unsigned int *>(
                    &scratch.phase[generation_index]),
                1u);
        }
    }
}

/// Rendezvous all eight ranks once, from one thread.
///
/// The symmetric counter is incremented through its multicast alias and polled
/// through its unicast alias, and the private target rises by the rank count on
/// every call, so the pair stays consistent across launches and across the
/// unsigned wrap without a host reset. `utils::barrier_all` drives the same pair
/// with the same arithmetic, so the two may be interleaved freely.
static __device__ void barrier_ranks(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    std::uint32_t *const barrier_multicast,
    const std::uint32_t *const barrier_local,
    unsigned int *const barrier_target,
    const int generation_index,
    const int error_code
) {
    const std::uint32_t target =
        static_cast<std::uint32_t>(
            atomicAdd(barrier_target,
                      static_cast<unsigned int>(kTensorParallelSize)))
        + static_cast<std::uint32_t>(kTensorParallelSize);

    __threadfence_system();
    asm volatile(
        "{multimem.red.release.sys.global.add.u32 [%0], 1;}"
        :: "l"(barrier_multicast)
        : "memory");
    asm volatile("{fence.proxy.alias;}" ::: "memory");

    const std::uint64_t started = clock64();
    while (!barrier_reached(load_relaxed_system(barrier_local), target)) {
        if (wait_timed_out(started, clock64())) {
            record_timeout_and_trap(
                scratch, error_flag, generation_index, error_code);
        }
        __nanosleep(64);
    }
    asm volatile("{fence.acquire.sys;}" ::: "memory");
}

/// Drive both cross-rank edges of one tail launch from a single thread.
static __device__ void coordinate_ranks(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    std::uint32_t *const barrier_multicast,
    const std::uint32_t *const barrier_local,
    unsigned int *const barrier_target
) {
    if (threadIdx.x != 0) return;

    // Entry: every rank has finished writing its routed and shared partials
    // into its own copy of the collective buffer.
    barrier_ranks(scratch, error_flag, barrier_multicast, barrier_local,
                  barrier_target, kTailEntryGeneration,
                  kErrorTailEntryRendezvous);
    __threadfence_system();
    atomicAdd(
        reinterpret_cast<unsigned int *>(&scratch.phase[kTailEntryGeneration]),
        1u);

    const std::uint32_t baseline =
        load_relaxed_gpu(&scratch.phase[kTailExitGeneration]);
    const std::uint64_t started = clock64();
    while (!generation_advanced(
        load_relaxed_gpu(&scratch.phase[kTailShardGeneration]), baseline
    )) {
        if (wait_timed_out(started, clock64())) {
            record_timeout_and_trap(
                scratch, error_flag, kTailShardGeneration,
                kErrorTailCoordinatorShard);
        }
        __nanosleep(64);
    }

    // Exit: every rank has multicast its own shard, so after this rendezvous
    // all eight mailbox slots on this rank hold the current launch's data.
    barrier_ranks(scratch, error_flag, barrier_multicast, barrier_local,
                  barrier_target, kTailExitGeneration,
                  kErrorTailExitRendezvous);
    __threadfence_system();
    atomicAdd(
        reinterpret_cast<unsigned int *>(&scratch.phase[kTailExitGeneration]),
        1u);
}

/// Wait for the coordinator's exit edge before any consumer CTA retires.
///
/// Retiring earlier would let the surrounding grid, or Task 9's persistent
/// kernel, read a mailbox slot a peer has not filled yet.
static __device__ void drain_ranks(
    const Scratch &scratch,
    int *__restrict__ const error_flag,
    std::uint32_t *const baseline_slot,
    const int consumer_ctas
) {
    const std::uint32_t baseline =
        latch_generation(scratch, kTailDrainGeneration, baseline_slot);
    wait_for_generation(scratch, error_flag, kTailExitGeneration, baseline,
                        kErrorTailDrainExit);
    publish_generation(
        scratch, kTailDrainArrivals, kTailDrainGeneration, consumer_ctas);
}

}  // namespace tail
}  // namespace kimi_k3_decode
