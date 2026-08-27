#pragma once

#include <cstdint>

namespace kimi_k3_decode {

inline constexpr int kHiddenSize = 7168;
inline constexpr int kLatentSize = 3584;
inline constexpr int kRoutedIntermediateSize = 3072;
inline constexpr int kSharedIntermediateSize = 6144;
inline constexpr int kNumExperts = 896;
inline constexpr int kTopK = 16;
inline constexpr int kTensorParallelSize = 8;
inline constexpr int kMaxTokens = 128;

static constexpr int NUM_PHASE_COUNTERS = 16;
static constexpr int SCRATCH_ALIGNMENT = 256;
static constexpr int SCRATCH_BYTES =
    ((NUM_PHASE_COUNTERS * sizeof(int) + SCRATCH_ALIGNMENT - 1)
     / SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT;

inline constexpr int kExpertW1W3PackedRows = 384;
inline constexpr int kExpertW1W3PackedColumns = 1824;
inline constexpr int kExpertW1W3ScaleColumns = 114;
inline constexpr int kExpertW2PackedRows = 3584;
inline constexpr int kExpertW2PackedColumns = 192;
inline constexpr int kExpertW2ScaleColumns = 12;

}  // namespace kimi_k3_decode
