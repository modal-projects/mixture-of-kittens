#pragma once

/// What a bounded wait publishes when it gives up.
///
/// A slot, a code, and the one claim word that decides whose record the host
/// reads back. Zero means unclaimed, which is what "no wait has ever timed out
/// on this workspace" looks like.

#include "phase_clocks.cuh"

namespace kimi_k3_decode {

// Timeout diagnostics.
//
// A bounded wait that gives up writes two things before it traps: the phase
// slot it was waiting on, into the timeout counter for its half of the step,
// and one of the codes below, into the caller-visible `error_flag`. Both are
// needed. The slot alone cannot name the site, because several sites wait on
// the same counter -- the entry rendezvous and the reduce role both wait on
// `kTailEntryGeneration` -- and the flag alone does not survive a workspace
// whose scratch the caller never reads. Every code is nonzero, so zero keeps
// its meaning: no wait has ever timed out on this workspace.
// ---------------------------------------------------------------------------

inline constexpr int kErrorTailEntryRendezvous = 1;
inline constexpr int kErrorTailExitRendezvous = 2;
inline constexpr int kErrorTailCoordinatorShard = 3;
inline constexpr int kErrorTailReduceEntry = 4;
inline constexpr int kErrorTailShardReduce = 5;
inline constexpr int kErrorTailDrainExit = 6;
inline constexpr int kErrorPersistentGridBarrier = 7;
inline constexpr int kErrorPersistentActivation = 8;
inline constexpr int kErrorPersistentGateUpDownReadiness = 9;

}  // namespace kimi_k3_decode
