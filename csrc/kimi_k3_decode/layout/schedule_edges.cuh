#pragma once

/// Every readiness edge the dependency-local schedule waits on.
///
/// The table is `constexpr` so that "every edge points at a strictly earlier
/// queue" is a `static_assert` rather than a comment, which is most of what
/// makes the barrier-free scan deadlock free.

#include "timeouts.cuh"

namespace kimi_k3_decode {

// The dependency-local schedule's readiness edges. One code per wait site, so
// a stalled candidate names the edge that did not arrive rather than only the
// counter, which several sites would otherwise share.
inline constexpr int kErrorScheduleSharedPair = 10;
inline constexpr int kErrorScheduleScoreShards = 11;
inline constexpr int kErrorScheduleAssignment = 12;
inline constexpr int kErrorScheduleLatent = 13;
inline constexpr int kErrorScheduleActivation = 14;
inline constexpr int kErrorScheduleSharedGateUp = 15;
inline constexpr int kErrorScheduleExpertGateUp = 16;
inline constexpr int kErrorScheduleRoutedDown = 17;
inline constexpr int kErrorSchedulePublish = 18;
inline constexpr int kErrorScheduleSharedDown = 19;

/// One bounded wait, named by the code it reports and the slots it writes.
struct TimeoutSite {
    const char *name;
    int code;
    int timeout_slot;
    int counter;
};

/// Every bounded wait either tail path can give up on, in code order.
///
/// The tests walk this table against the sources, so a wait added without a
/// code, or a code added without a wait, fails rather than silently trapping
/// with a diagnostic the caller cannot interpret.
inline constexpr TimeoutSite kTimeoutSites[] = {
    {"tail_entry_rendezvous", kErrorTailEntryRendezvous,
     kTailTimeoutPhase, kTailEntryGeneration},
    {"tail_exit_rendezvous", kErrorTailExitRendezvous,
     kTailTimeoutPhase, kTailExitGeneration},
    {"tail_coordinator_shard", kErrorTailCoordinatorShard,
     kTailTimeoutPhase, kTailShardGeneration},
    {"tail_reduce_entry", kErrorTailReduceEntry,
     kTailTimeoutPhase, kTailEntryGeneration},
    {"tail_shard_reduce", kErrorTailShardReduce,
     kTailTimeoutPhase, kTailReduceGeneration},
    {"tail_drain_exit", kErrorTailDrainExit,
     kTailTimeoutPhase, kTailExitGeneration},
    {"persistent_grid_barrier", kErrorPersistentGridBarrier,
     kPersistentTimeoutPhase, kGridGeneration},
    {"persistent_shared_activation", kErrorPersistentActivation,
     kPersistentTimeoutPhase, kActivationArrivals},
    {"persistent_gate_up_down_readiness",
     kErrorPersistentGateUpDownReadiness,
     kPersistentTimeoutPhase, kGateUpArrivals},
    {"schedule_shared_activation_pair", kErrorScheduleSharedPair,
     kPersistentTimeoutPhase, schedule_diagnostic(kScheduleSharedPairBegin)},
    {"schedule_assignment_score_shards", kErrorScheduleScoreShards,
     kPersistentTimeoutPhase, schedule_diagnostic(kScheduleScoreArrivals)},
    {"schedule_gate_up_assignment", kErrorScheduleAssignment,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleAssignmentArrivals)},
    {"schedule_gate_up_latent", kErrorScheduleLatent,
     kPersistentTimeoutPhase, schedule_diagnostic(kScheduleLatentArrivals)},
    {"schedule_shared_down_activation", kErrorScheduleActivation,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleActivationArrivals)},
    {"schedule_shared_down_gate_up", kErrorScheduleSharedGateUp,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleSharedGateUpArrivals)},
    {"schedule_routed_down_gate_up", kErrorScheduleExpertGateUp,
     kPersistentTimeoutPhase, kGateUpArrivals},
    {"schedule_publish_routed_down", kErrorScheduleRoutedDown,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleRoutedDownArrivals)},
    {"schedule_tail_publish", kErrorSchedulePublish,
     kPersistentTimeoutPhase, schedule_diagnostic(kSchedulePublishArrivals)},
    {"schedule_tail_shared_down", kErrorScheduleSharedDown,
     kPersistentTimeoutPhase,
     schedule_diagnostic(kScheduleSharedDownArrivals)},
};

inline constexpr int kTimeoutSiteCount =
    static_cast<int>(sizeof(kTimeoutSites) / sizeof(kTimeoutSites[0]));

static_assert(kTimeoutSiteCount == 19);
static_assert(kTimeoutSites[kTimeoutSiteCount - 1].code == kTimeoutSiteCount,
              "the timeout codes must be a dense nonzero range");

// ---------------------------------------------------------------------------
// The dependency-local schedule's readiness edges.
//
// Every wait the candidate takes is one row of this table. The table is what
// the source contracts and the runtime binding both read, so an edge cannot be
// added to the kernel without being named here, and it exists chiefly to carry
// one static proof: every edge's producer queue is strictly earlier in the one
// forward scan than its consumer. That is the whole deadlock argument, so it is
// checked by the compiler rather than by review.
// ---------------------------------------------------------------------------

/// One readiness edge, in the order the profile band reports them.
enum ScheduleEdgeId : int {
    kEdgeSharedActivationPair = 0,
    kEdgeAssignmentScoreShards,
    kEdgeGateUpAssignment,
    kEdgeGateUpLatent,
    kEdgeSharedDownActivation,
    kEdgeSharedDownGateUp,
    kEdgeRoutedDownGateUp,
    kEdgePublishRoutedDown,
    kEdgeTailPublish,
    kEdgeTailSharedDown,
    kScheduleEdgeCount,
};

/// Which array an edge's readiness counter lives in.
///
/// Nine of the ten are in the appended schedule region. The tenth is not:
/// routed down's per-expert readiness is already published into the compacted
/// assignment counts by the fused gate/up engine, and reading it there is what
/// keeps that edge expert-local.
enum ScheduleCounterSpace : int {
    kScheduleCounterInRegion = 0,
    kScheduleCounterInExpertCounts,
};

/// The release scope an edge's producer used, and therefore the acquire scope
/// its consumer owes.
enum ScheduleEdgeScope : int {
    kScheduleScopeDevice = 0,
    kScheduleScopeSystem,
};

/// Where an edge's arrival target comes from.
///
/// Six of the ten are not a property of the edge alone: the score shards scale
/// with the token count, the routed totals with the occupied expert count, and
/// the three shared-expert edges with which path the launch took, since the
/// core and tensor instantiations cut the shared columns differently. Those are
/// supplied at the wait, and the table records only that they must be, so a
/// wait that forgot one is a wait on zero arrivals rather than a wait on a
/// plausible-looking constant.
enum ScheduleEdgeTarget : int {
    kScheduleTargetStatic = 0,
    kScheduleTargetDynamic,
};

/// One readiness edge, and everything a wait on it needs.
///
/// This is deliberately the *whole* contract and not a summary of it. The
/// runtime `wait_edge<Edge>` reads its counter, its target, its acquire scope,
/// its diagnostic slot, and its error code out of this row and out of nowhere
/// else, so the static DAG the contracts check and the waits the kernel takes
/// cannot drift apart: there is only one description of an edge.
struct ScheduleEdge {
    const char *name;
    /// The queue that waits. `kScheduleQueueCount` names the TP8 tail, which
    /// every queue precedes.
    int consumer_queue;
    int producer_queue;
    /// The counter the wait spins on: an index into the appended region, or
    /// into `expert_counts` when `space` says so.
    int counter;
    int error_code;
    /// `kScheduleCounterInRegion` or `kScheduleCounterInExpertCounts`.
    int space;
    /// `kScheduleScopeDevice` or `kScheduleScopeSystem`.
    int scope;
    /// `kScheduleTargetStatic` or `kScheduleTargetDynamic`.
    int target_kind;
    /// The arrival count, when `target_kind` is static; -1 otherwise.
    int static_target;
    /// Whether the counter index is `counter + unit`, which the per-column-pair
    /// and per-expert edges are and the whole-queue edges are not.
    bool counter_indexed;
};

inline constexpr int kScheduleEdgeOutsideRegion = -1;
inline constexpr int kScheduleTargetSuppliedAtWait = -1;

/// The arrivals one fused gate/up unit publishes for its expert.
///
/// Spelled here because the wait's target has to come out of this table and
/// `expert_mxfp4::fused_w13` is not included yet; the schedule header asserts
/// it against the engine's own geometry.
inline constexpr int kScheduleExpertGateUpArrivals = 6;

inline constexpr ScheduleEdge kScheduleEdges[] = {
    {"shared_activation_pair", kQueueSharedActivation, kQueueSource,
     kScheduleSharedPairBegin, kErrorScheduleSharedPair,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetStatic, 2, true},
    {"assignment_score_shards", kQueueAssignment, kQueueSource,
     kScheduleScoreArrivals, kErrorScheduleScoreShards,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"gate_up_assignment", kQueueRoutedGateUp, kQueueAssignment,
     kScheduleAssignmentArrivals, kErrorScheduleAssignment,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetStatic, 1, false},
    {"gate_up_latent", kQueueRoutedGateUp, kQueueSource,
     kScheduleLatentArrivals, kErrorScheduleLatent,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"shared_down_activation", kQueueSharedDown, kQueueSharedActivation,
     kScheduleActivationArrivals, kErrorScheduleActivation,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"shared_down_gate_up", kQueueSharedDown, kQueueSource,
     kScheduleSharedGateUpArrivals, kErrorScheduleSharedGateUp,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"routed_down_gate_up", kQueueRoutedDown, kQueueRoutedGateUp,
     kGateUpArrivals, kErrorScheduleExpertGateUp,
     kScheduleCounterInExpertCounts, kScheduleScopeDevice,
     kScheduleTargetStatic, kScheduleExpertGateUpArrivals, true},
    {"publish_routed_down", kQueuePublish, kQueueRoutedDown,
     kScheduleRoutedDownArrivals, kErrorScheduleRoutedDown,
     kScheduleCounterInRegion, kScheduleScopeDevice,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
    {"tail_publish", kScheduleQueueCount, kQueuePublish,
     kSchedulePublishArrivals, kErrorSchedulePublish,
     kScheduleCounterInRegion, kScheduleScopeSystem,
     kScheduleTargetStatic, kSchedulePublishUnitsForTable, false},
    {"tail_shared_down", kScheduleQueueCount, kQueueSharedDown,
     kScheduleSharedDownArrivals, kErrorScheduleSharedDown,
     kScheduleCounterInRegion, kScheduleScopeSystem,
     kScheduleTargetDynamic, kScheduleTargetSuppliedAtWait, false},
};

/// The diagnostic slot a wait on one edge records, given the unit it is on.
///
/// The two counter spaces number differently: a schedule counter is offset past
/// the phase region so the two cannot be confused in the one word both write,
/// while `expert_counts` readiness reports the phase slot the production wait
/// on the same counter reports.
/// Host-side `constexpr` on purpose: every device use of it is a constant
/// expression, so nothing needs the table itself in device memory.
inline constexpr int schedule_edge_diagnostic(
    const int edge,
    const int unit
) {
    return kScheduleEdges[edge].space == kScheduleCounterInExpertCounts
        ? kScheduleEdges[edge].counter
        : schedule_diagnostic(
              kScheduleEdges[edge].counter
              + (kScheduleEdges[edge].counter_indexed ? unit : 0));
}

/// Whether every edge's declared scope is at least its producer's.
///
/// The two edges the coordinator takes before it tells the peers this rank is
/// done must acquire at system scope, because the producers they wait on
/// released at system scope and a device-scope acquire would make the
/// coordinator's own release non-transitive over them. Nothing else needs to.
inline constexpr bool schedule_tail_edges_acquire_at_system_scope() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        const bool crosses = kScheduleEdges[edge].consumer_queue
            == kScheduleQueueCount;
        if (crosses
                && kScheduleEdges[edge].scope != kScheduleScopeSystem) {
            return false;
        }
    }
    return true;
}

/// Whether every static target is a positive arrival count.
inline constexpr bool schedule_static_targets_are_positive() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        const bool is_static = kScheduleEdges[edge].target_kind
            == kScheduleTargetStatic;
        if (is_static != (kScheduleEdges[edge].static_target > 0)) {
            return false;
        }
    }
    return true;
}

static_assert(schedule_tail_edges_acquire_at_system_scope(),
              "an edge the coordinator takes before it publishes to the peers "
              "must acquire at system scope");
static_assert(schedule_static_targets_are_positive(),
              "a static target must be a positive arrival count, and a dynamic "
              "one must not carry a target the wait would ignore");

static_assert(
    static_cast<int>(sizeof(kScheduleEdges) / sizeof(kScheduleEdges[0]))
        == kScheduleEdgeCount);

/// Whether every readiness edge points at a strictly earlier queue.
inline constexpr bool schedule_edges_point_backward() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        if (kScheduleEdges[edge].producer_queue
                >= kScheduleEdges[edge].consumer_queue) {
            return false;
        }
    }
    return true;
}

/// Whether no edge waits on a producer inside its own queue.
inline constexpr bool schedule_edges_never_block_inside_a_queue() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        if (kScheduleEdges[edge].producer_queue
                == kScheduleEdges[edge].consumer_queue) {
            return false;
        }
    }
    return true;
}

/// Whether every edge's code is one of the tabulated timeout codes.
inline constexpr bool schedule_edge_codes_are_tabulated() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        bool found = false;
        for (int site = 0; site < kTimeoutSiteCount; ++site) {
            if (kTimeoutSites[site].code == kScheduleEdges[edge].error_code) {
                found = true;
            }
        }
        if (!found) return false;
        for (int other = 0; other < edge; ++other) {
            if (kScheduleEdges[other].error_code
                    == kScheduleEdges[edge].error_code) {
                return false;
            }
        }
    }
    return true;
}

/// Whether every edge that names an appended counter names a cleared one.
///
/// The candidate carries no generation arithmetic of its own, so a readiness
/// counter is only meaningful if this launch found it at zero. Block 0 zeroes
/// everything below `kScheduleClearedCounters` before the one retained
/// barrier, and this is what says the table cannot grow an edge outside that
/// band without the compiler noticing.
inline constexpr bool schedule_edge_counters_are_cleared_each_launch() {
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        if (kScheduleEdges[edge].space != kScheduleCounterInRegion) continue;
        const int counter = kScheduleEdges[edge].counter;
        if (counter < kScheduleQueueCount
                || counter >= kScheduleClearedCounters) {
            return false;
        }
    }
    return true;
}

static_assert(schedule_edge_counters_are_cleared_each_launch(),
              "a readiness counter the launch did not zero cannot be waited "
              "on without generation arithmetic the candidate does not have");
static_assert(schedule_edges_point_backward(),
              "every readiness edge must point at a strictly earlier queue, "
              "which is what makes the one forward scan deadlock free");
static_assert(schedule_edges_never_block_inside_a_queue(),
              "a queue may not block on a producer in the same queue");
static_assert(schedule_edge_codes_are_tabulated(),
              "every readiness edge needs its own tabulated timeout code");

/// Every edge's reported name, in `ScheduleEdgeId` order.
inline constexpr const char *kScheduleEdgeNames[] = {
    "shared_activation_pair",
    "assignment_score_shards",
    "gate_up_assignment",
    "gate_up_latent",
    "shared_down_activation",
    "shared_down_gate_up",
    "routed_down_gate_up",
    "publish_routed_down",
    "tail_publish",
    "tail_shared_down",
};

/// Every queue's reported name, in `ScheduleQueue` order.
inline constexpr const char *kScheduleQueueNames[] = {
    "source",
    "shared_activation",
    "assignment",
    "routed_gate_up",
    "shared_down",
    "routed_down",
    "publish",
};

static_assert(
    static_cast<int>(sizeof(kScheduleEdgeNames) / sizeof(kScheduleEdgeNames[0]))
        == kScheduleEdgeCount);
static_assert(
    static_cast<int>(
        sizeof(kScheduleQueueNames) / sizeof(kScheduleQueueNames[0]))
        == kScheduleQueueCount);

// ---------------------------------------------------------------------------

}  // namespace kimi_k3_decode
