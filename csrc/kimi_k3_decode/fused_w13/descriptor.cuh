#pragma once

/// Host side: the packed tensor map, and the cache that keeps it encoded once.
///
/// `cuTensorMapEncodeTiled` is a driver call, so encoding it per launch would
/// put a host round trip on the decode path. The map depends only on the
/// prepared weight pointer and the layout, both of which are fixed for the life
/// of a set of weights, so it is encoded once per pointer and kept.

#include "adaptive_unit.cuh"

namespace kimi_k3_decode {
namespace expert_mxfp4 {
namespace fused_w13 {

// ---------------------------------------------------------------------------
// Host side: the descriptor, and what the launch carries.
// ---------------------------------------------------------------------------

/// Build the `(task, slab)` weight-tile tensor map.
///
/// The five dimensions are, innermost first: the 128 U4 values one box row
/// carries, the tile's 128 M rows, the four boxes that make a 512-wide slab,
/// the 42 `(task, slab)` pairs of one expert, and the 896 experts. Splitting a
/// row into boxes of exactly one swizzle atom, and giving the boxes their own
/// dimension, is what makes the shared destination strip-major -- all 128 rows
/// of box 0, then all 128 rows of box 1 -- which is the layout a 128B-swizzled
/// ThunderKittens tile has and the layout `chunk_descriptor` walks.
///
/// Every requirement `16U4_ALIGN16B` adds is met by construction rather than by
/// luck, and each one is checked below, because a descriptor the driver accepts
/// but that describes the wrong layout fails as wrong numbers rather than as an
/// error.
static __host__ void create_fused_w13_packed_map(
    CUtensorMap *map,
    const void *base
) {
    const std::uint64_t global_dim[5] = {
        kFusedBoxElements,
        kFusedM,
        kFusedBoxes,
        kFusedTaskSlabs,
        kNumExperts,
    };
    // Byte distances between consecutive coordinates of dimensions one to four.
    const std::uint64_t global_stride[4] = {
        kFusedSlabRowBytes,
        kFusedBoxElements / 2,
        kFusedSlabGlobalBytes,
        static_cast<std::uint64_t>(kFusedTaskSlabs) * kFusedSlabGlobalBytes,
    };
    const std::uint32_t box_dim[5] = {
        kFusedBoxElements, kFusedM, kFusedBoxes, 1, 1,
    };
    const std::uint32_t element_stride[5] = {1, 1, 1, 1, 1};

    // `globalDim[0]` must be a multiple of 128 U4 values, `boxDim[0]` must be
    // exactly 128, the global address must be 32-byte aligned, and every stride
    // must be a multiple of 32 bytes. The `boxDim[0]` rule is what fixes the
    // transfer granularity at K = 128 and therefore the slab at a multiple of
    // it.
    static_assert(kFusedBoxElements == 128);
    static_assert(kFusedSlabK % kFusedBoxElements == 0);
    static_assert(kFusedSlabRowBytes % 32 == 0);
    static_assert((kFusedBoxElements / 2) % 32 == 0);
    static_assert(kFusedSlabGlobalBytes % 32 == 0);
    static_assert((static_cast<long long>(kFusedTaskSlabs)
                   * kFusedSlabGlobalBytes) % 32 == 0);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(base) % 32 == 0,
                "MoK: the fused W13 payload must be 32-byte aligned");

    const CUresult result = cuTensorMapEncodeTiled(
        map,
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        5,
        const_cast<void *>(base),
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    const char *error_string = nullptr;
    cuGetErrorString(result, &error_string);
    TORCH_CHECK(result == CUDA_SUCCESS,
                "MoK: cuTensorMapEncodeTiled rejected the fused W13 layout: ",
                error_string == nullptr ? "unknown" : error_string);
}

/// The descriptor for one prepared payload, encoded once per base address.
///
/// The descriptor travels into the kernel by value in a `__grid_constant__`
/// parameter, which is the only way a kernel may name one without a per-launch
/// device allocation to hold it. Encoding it, though, is a driver call, and a
/// decode step is tens of microseconds -- so it is encoded on the first launch
/// against a payload and read from here on every launch after.
///
/// Keyed by the payload's base address, which is sound because the address is
/// the only thing about the encoding that is not a compile-time constant of
/// this file: a second payload at an address a first one was freed from has
/// byte-for-byte the same descriptor, so a stale hit is indistinguishable from
/// a fresh encode. Nothing is ever evicted, and nothing needs to be -- weights
/// are prepared once per model load, so this map holds one entry per rank per
/// process for the process's whole life.
struct PackedMapCache {
    std::mutex guard;
    std::map<const void *, CUtensorMap> encoded;
    std::atomic<int> encodes{0};
};

static __host__ PackedMapCache &packed_map_cache() {
    static PackedMapCache cache;
    return cache;
}

static __host__ const CUtensorMap *fused_w13_packed_map(const void *base) {
    PackedMapCache &cache = packed_map_cache();
    const std::lock_guard<std::mutex> held(cache.guard);
    const auto found = cache.encoded.find(base);
    if (found != cache.encoded.end()) return &found->second;
    // A `std::map` node's value keeps its address across every later insert,
    // so the pointer handed out here stays valid for the process's life.
    CUtensorMap &map = cache.encoded[base];
    create_fused_w13_packed_map(&map, base);
    cache.encodes.fetch_add(1, std::memory_order_relaxed);
    return &map;
}

/// How many descriptors this process has encoded, for the launch-overhead test.
///
/// The number the graph-capture and steady-state gates care about is one per
/// payload for the life of the process, not one per launch, and a cache that
/// missed every time would be invisible in the decode output.
inline std::int64_t fused_w13_packed_maps_encoded_for_testing() {
    return packed_map_cache().encodes.load(std::memory_order_relaxed);
}

}  // namespace fused_w13
}  // namespace expert_mxfp4
}  // namespace kimi_k3_decode
