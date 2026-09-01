#pragma once

// The dependency-local persistent schedule: one launch, one full-grid barrier.
//
// The production kernel in `persistent_kernel.cuh` separates its phases with
// five generation-tagged full-grid barriers. A barrier is a correct but coarse
// dependency: it makes every CTA wait for the slowest CTA of the phase it is
// leaving even when it has no data dependency on that CTA's output at all. The
// measured B300 profile of the production step puts roughly a fifth of a
// decode step inside those barriers, and route-major finalize and a transposed
// tail shard were both measured against that idle and rejected, so what is
// left to try is removing the barriers themselves.
//
// This header is that candidate. It keeps exactly one full-grid barrier -- the
// one that publishes this launch's cleared counters, which nothing else can
// establish -- and replaces the other four with seven topologically ordered
// task queues and bounded release/acquire readiness edges. The queues, the
// counters, and the dependency table all live in `types.cuh`, next to the
// appended scratch region that holds them.
//
// Why the scan is deadlock free, in full:
//
//   1. Every CTA walks the queues in the one forward order `ScheduleQueue`
//      declares. There is no path that revisits a queue.
//   2. A CTA leaves a queue only when that queue's ticket counter is
//      exhausted, which means every unit of it is already claimed. The host
//      proves all CTAs of the launch co-reside one per SM, so a claimed unit
//      is held by a running CTA and will complete.
//   3. Every readiness edge points at a strictly earlier queue --
//      `schedule_edges_point_backward` is a `static_assert`, not a comment --
//      so a CTA blocked in queue `k` is waiting only on units of queues below
//      `k`, all of which are claimed by (2) and none of which can be waiting
//      on queue `k`.
//   4. No edge names its own queue as its producer, so no CTA can be blocked
//      behind a unit of the queue it is itself blocking in.
//   5. Every wait is bounded by the same fifteen-second clock budget the
//      production waits use and reports its own timeout code, so a broken
//      edge surfaces as a named trap rather than as a hung device.
//
// The stages themselves are the production stages, called unchanged. Nothing
// here recomputes anything, so a candidate launch is bit-for-bit the
// production launch with a different order of arrival.
// How this is laid out
// --------------------
// The schedule was one 1,590-line header and is now a chain of focused ones,
// included here in dependency order. Each opens the same three namespaces and
// includes the one before it, so this file is the only include any caller
// needs and the parts stay in the order a reader would meet them: the shapes,
// the clocks, the queues the CTAs walk, the one phase body worth naming, the
// kernel over all of it, then the host launch and what the tests read back.
//
// Nothing was renamed and nothing moved between namespaces, so a symbol's
// home is `git log --follow` on the part that holds it.

#include "dependency_local/shapes.cuh"
#include "dependency_local/clocks.cuh"
#include "dependency_local/queues.cuh"
#include "dependency_local/projection.cuh"
#include "dependency_local/kernel.cuh"
#include "dependency_local/launch.cuh"
#include "dependency_local/publication_probe.cuh"
#include "dependency_local/readback.cuh"
