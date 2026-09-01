#pragma once

/// What the tests and the A/B harness read back out of the schedule.
///
/// The readiness table as whole rows, the per-queue unit counts for a given
/// token count, and the clock band's layout. Pure host reflection over the
/// `constexpr` above: nothing here is on a launch path.

#include "publication_probe.cuh"

namespace kimi_k3_decode {
namespace persistent {
namespace schedule {

// What the tests and the A/B harness read back.
// ---------------------------------------------------------------------------

/// Every readiness edge, as the whole row the runtime waits from.
///
/// The columns are the table's columns, all of them, so a test can check that
/// what the kernel derives its wait from is what the DAG declares rather than
/// checking a projection of it: name, consumer queue, producer queue, counter,
/// timeout code, counter space, acquire scope, target kind, static target, and
/// whether the counter is indexed by the unit.
inline std::vector<std::tuple<std::string, std::int64_t, std::int64_t,
                              std::int64_t, std::int64_t, std::int64_t,
                              std::int64_t, std::int64_t, std::int64_t, bool>>
schedule_edges_for_testing() {
    std::vector<std::tuple<std::string, std::int64_t, std::int64_t,
                           std::int64_t, std::int64_t, std::int64_t,
                           std::int64_t, std::int64_t, std::int64_t, bool>>
        rows;
    for (const ScheduleEdge &edge : kScheduleEdges) {
        rows.emplace_back(
            std::string(edge.name),
            static_cast<std::int64_t>(edge.consumer_queue),
            static_cast<std::int64_t>(edge.producer_queue),
            static_cast<std::int64_t>(edge.counter),
            static_cast<std::int64_t>(edge.error_code),
            static_cast<std::int64_t>(edge.space),
            static_cast<std::int64_t>(edge.scope),
            static_cast<std::int64_t>(edge.target_kind),
            static_cast<std::int64_t>(edge.static_target),
            edge.counter_indexed);
    }
    return rows;
}

/// The diagnostic slot each edge records, for the unit the caller names.
///
/// Exposed so the trap test can predict the recorded slot from the same
/// function the kernel computes it with, rather than from a second copy of the
/// offset arithmetic in Python.
inline std::vector<std::int64_t> schedule_edge_diagnostics_for_testing(
    const std::int64_t unit
) {
    std::vector<std::int64_t> slots;
    for (int edge = 0; edge < kScheduleEdgeCount; ++edge) {
        slots.push_back(static_cast<std::int64_t>(
            schedule_edge_diagnostic(edge, static_cast<int>(unit))));
    }
    return slots;
}

inline std::vector<std::string> schedule_queue_names_for_testing() {
    std::vector<std::string> names;
    for (const char *const name : kScheduleQueueNames) {
        names.emplace_back(name);
    }
    return names;
}

/// The appended profile band's first slot and the names it reports.
inline std::tuple<std::int64_t, std::int64_t, std::int64_t,
                  std::vector<std::string>, std::vector<std::string>>
schedule_clock_metadata_for_testing() {
    std::vector<std::string> edge_names;
    for (const char *const name : kScheduleEdgeNames) {
        edge_names.emplace_back(name);
    }
    return {
        static_cast<std::int64_t>(kScheduleBytes / 4 + kScheduleEdgeWaitBegin),
        static_cast<std::int64_t>(
            kScheduleBytes / 4 + kScheduleEdgeMakespanBegin),
        static_cast<std::int64_t>(
            kScheduleBytes / 4 + kScheduleQueueMakespanBegin),
        edge_names,
        schedule_queue_names_for_testing()};
}

/// The bounds the counters are asserted against.
inline std::tuple<std::int64_t, std::int64_t, std::int64_t>
schedule_counter_bounds_for_testing() {
    return {
        static_cast<std::int64_t>(kScheduleLongestTicket),
        static_cast<std::int64_t>(kScheduleLargestArrival),
        static_cast<std::int64_t>(kScheduleCounterBound)};
}

/// The logical unit counts of every queue, for one shape.
inline std::vector<std::int64_t> schedule_queue_units_for_testing(
    const std::int64_t active_tokens
) {
    TORCH_CHECK(active_tokens >= 1 && active_tokens <= kMaxTokens,
                "MoK: the dependency-local schedule needs active_tokens in "
                "[1, ", kMaxTokens, "]");
    const int tokens = static_cast<int>(active_tokens);
    const bool tensor_path = capacity_bucket(tokens) > kMaxCoreCapacity;
    const int experts =
        kTopK * tokens < kNumExperts ? kTopK * tokens : kNumExperts;
    const int projection = tensor_path ? kProjectionUnits<true>
                                       : kProjectionUnits<false>;
    const int shared_gate_up = tensor_path ? kSharedGateUpUnits<true>
                                           : kSharedGateUpUnits<false>;
    const int activation = tensor_path ? kSharedActivationUnits<true>
                                       : kSharedActivationUnits<false>;
    const int shared_down = tensor_path ? kSharedDownUnits<true>
                                        : kSharedDownUnits<false>;
    const auto wide = [](const int value) {
        return static_cast<std::int64_t>(value);
    };
    return {
        wide(projection + shared_gate_up + tokens * router::kScoreShards),
        wide(activation),
        wide(1),
        wide(experts * kScheduleGateUpUnitsPerExpert),
        wide(shared_down),
        wide(experts * expert_mxfp4::grouped_pipeline::kGroupedDownUnits),
        wide(kSchedulePublishUnits),
    };
}

}  // namespace schedule
}  // namespace persistent
}  // namespace kimi_k3_decode
