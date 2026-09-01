#pragma once

/// The two global-layout types every tcgen05 stage reads through.
///
/// Eleven TMA descriptors over two tile shapes, so the layouts collapse to two
/// types and the kernel takes them as one template parameter.

#include "benchmark_switches.cuh"

namespace kimi_k3_decode {
namespace persistent {

// Tensor-path descriptors.
//
// Every tcgen05 stage this kernel runs reads through one of two tile shapes, so
// the eleven TMA descriptors it needs collapse to two global-layout types. They
// travel in one `__grid_constant__` struct that only the tensor instantiation
// carries: building a descriptor costs a driver call per launch, and the core
// instantiation would never dereference one.
// ---------------------------------------------------------------------------

using tile_layout = skinny_gemm::hidden_layout;
using square_layout = skinny_gemm::latent_layout;

static_assert(std::is_same_v<tile_layout, skinny_gemm::weight_layout>);
static_assert(std::is_same_v<tile_layout,
                             shared_experts::tensor_input_layout>);
static_assert(std::is_same_v<tile_layout,
                             shared_experts::tensor_weight_layout>);
static_assert(std::is_same_v<tile_layout, tail::tensor_input_layout>);
static_assert(std::is_same_v<tile_layout, tail::tensor_weight_layout>);
static_assert(std::is_same_v<square_layout,
                             shared_experts::tensor_output_layout>);

struct TensorLayouts {
    tile_layout hidden;          // [active, 7168]
    tile_layout latent_down;     // [3584, 7168]
    square_layout latent;        // [active, 3584]
    tile_layout shared_gate;     // [768, 7168]
    tile_layout shared_up;       // [768, 7168]
    tile_layout shared_down;     // [7168, 768]
    square_layout gate;          // [active, 768]
    square_layout up;            // [active, 768]
    tile_layout activated;       // [active, 768]
    tile_layout normalized;      // [active, 3584]
    tile_layout latent_up;       // [896, 3584]
};

/// What the core instantiation carries in place of the descriptors.
///
/// It holds one dead byte rather than nothing, because a `__grid_constant__`
/// parameter names a const reference to an object in the kernel's parameter
/// space and an empty type gives that object no bytes to name.
struct NoTensorLayouts {
    char unused;
};

template<bool TENSOR_PATH>
using layouts_t = std::conditional_t<TENSOR_PATH, TensorLayouts,
                                     NoTensorLayouts>;

// ---------------------------------------------------------------------------

}  // namespace persistent
}  // namespace kimi_k3_decode
