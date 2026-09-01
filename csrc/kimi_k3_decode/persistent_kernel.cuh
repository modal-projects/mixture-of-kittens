#pragma once

// The production Kimi K3 decode path: every stage of one TP8 decode step in a
// single launch of a single kernel.
//
// The public production grid defaults to one CTA per B300 SM. A guarded
// benchmark may launch a smaller candidate, but a launch's CTA count never
// varies with the token count. The logical work -- up to 128 router tasks,
// 28 latent-column tasks, 2 688 routed gate/up tasks, 6 272 routed down
// tasks, the shared expert tasks, and the tail's three roles -- is handed out
// through the device queues in `persistent_sync.cuh` rather than mapped one
// task to one block. Five generation-tagged grid barriers separate the phases:
//
//   0. clear this launch's queue counters and the routed accumulator;
//   1. score and select every token while projecting the routed latent;
//   2. build the expert-major assignment table while quantizing that latent;
//   3. routed gate/up units interleaved with the shared gate/up units, each
//      publishing dependency readiness as it finishes;
//   4. shared activation, routed down units, and shared down units, each
//      waiting only for its own gate/up producer;
//   5. publish this rank's routed partial into the symmetric collective buffer.
//
// The tail then runs the same coordinator, reduce, and shard roles the private
// stage does, on the CTAs that carry those roles; every other CTA retires.
// How this is laid out
// --------------------
// The kernel was one 1,351-line header and is now a chain of focused ones,
// included here in dependency order. Each opens the same two namespaces and
// includes the one before it, so this file is the only include any caller
// needs: the budget the launch reserves, the plan its phases decompose into,
// the guarded switches a benchmark may set, the descriptors the stages read
// through, the kernel over all of it, and the host launch.
//
// Nothing was renamed and nothing moved between namespaces, so a symbol's
// home is `git log --follow` on the part that holds it.

#include "persistent/budget.cuh"
#include "persistent/task_plan.cuh"
#include "persistent/benchmark_switches.cuh"
#include "persistent/descriptors.cuh"
#include "persistent/kernel.cuh"
#include "persistent/launch.cuh"
