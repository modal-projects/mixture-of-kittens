#pragma once

/// The guarded switches a private benchmark may set, and nothing else may.
///
/// Every one of these reads as production unless `MOK_KIMI_K3_ENABLE_GRID_TUNING`
/// is set in the environment, and the public wrapper exposes none of them. They
/// live together so that "what can a process change" is one file rather than a
/// search: the grid, the phase profile, the gate/up engine, and whether the
/// dependency-local schedule is the one that launches.
///
/// The gate/up engine is a benchmark arm selector, not a production dial. What
/// production launches is the adaptive engine, and the arm it runs per expert
/// is chosen inside the kernel from that expert's row count.

#include "task_plan.cuh"

namespace kimi_k3_decode {
namespace persistent {

/// Select a non-production grid only from an explicitly-enabled benchmark.
///
/// The public decode wrapper has no grid option and unguarded reads always
/// return the validated production constant. The private binding checks an
/// environment guard before changing or exposing stored state, so application
/// code cannot accidentally retain a tuning candidate.
static __host__ std::atomic<int> &benchmark_grid_ctas_storage() {
    static std::atomic<int> grid{kPersistentCtas};
    return grid;
}

inline bool benchmark_grid_tuning_enabled() {
    const char *const enabled =
        std::getenv("MOK_KIMI_K3_ENABLE_GRID_TUNING");
    return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

inline std::int64_t benchmark_grid_ctas_for_testing() {
    if (!benchmark_grid_tuning_enabled()) return kPersistentCtas;
    return benchmark_grid_ctas_storage().load(std::memory_order_relaxed);
}

/// Whether this process collects the kernel's phase clocks.
///
/// Guarded exactly like the grid override, and for the same reason: the
/// accumulators are a benchmark instrument, and a production launch must not
/// be able to turn them on by accident and pay their atomics.
static __host__ std::atomic<int> &benchmark_phase_profile_storage() {
    static std::atomic<int> profile{0};
    return profile;
}

inline bool benchmark_phase_profile_enabled() {
    if (!benchmark_grid_tuning_enabled()) return false;
    return benchmark_phase_profile_storage().load(std::memory_order_relaxed)
        != 0;
}

inline void set_benchmark_phase_profile_for_testing(const bool enabled) {
    TORCH_CHECK(
        benchmark_grid_tuning_enabled(),
        "MoK: Kimi K3 phase profiling is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_GRID_TUNING=1 in a dedicated benchmark process");
    benchmark_phase_profile_storage().store(
        enabled ? 1 : 0, std::memory_order_relaxed);
}

inline bool benchmark_phase_profile_for_testing() {
    return benchmark_phase_profile_enabled();
}

/// Which routed gate/up engine this process launches. Production's, by default.
///
/// Guarded exactly like the grid override and the phase clocks, and for the
/// stronger version of the same reason. The only engine besides production's is
/// `kEngineFusedResident`, the two-stage ring production replaced: it is
/// compiled because it is the numerical baseline every parity test measures
/// against and the arm the integration's latency numbers were taken with, and it
/// is guarded because a step that ran it would be running the ring that was
/// retired. An unguarded read returns production's engine, so application code
/// cannot reach the baseline at all -- which is what makes it an A/B arm rather
/// than a switch production has to carry.
///
/// The engine is a template parameter of the launch, so a captured graph
/// records whichever kernel was selected while it was being captured and this
/// switch is irrelevant at replay. That is what lets the two arms be replayed
/// interleaved from one process.
static __host__ std::atomic<int> &benchmark_gate_up_engine_storage() {
    static std::atomic<int> engine{
        expert_mxfp4::fused_w13::kEngineFusedAdaptive};
    return engine;
}

inline int benchmark_gate_up_engine() {
    if (!benchmark_grid_tuning_enabled()) {
        return expert_mxfp4::fused_w13::kEngineFusedAdaptive;
    }
    return benchmark_gate_up_engine_storage().load(std::memory_order_relaxed);
}

inline void set_gate_up_engine_for_testing(const std::int64_t engine) {
    TORCH_CHECK(
        benchmark_grid_tuning_enabled(),
        "MoK: the Kimi K3 gate/up engine selector is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_GRID_TUNING=1 in a dedicated benchmark process");
    TORCH_CHECK(
        expert_mxfp4::fused_w13::engine_is_known(static_cast<int>(engine)),
        "MoK: unknown Kimi K3 gate/up engine ", engine);
    benchmark_gate_up_engine_storage().store(
        static_cast<int>(engine), std::memory_order_relaxed);
}

inline std::int64_t gate_up_engine_for_testing() {
    return benchmark_gate_up_engine();
}

/// The shared bytes and ring depth one engine's instantiation carries.
///
/// Read off the compiled constants rather than restated in the harness, so the
/// artifact's ledger is the one ptxas was given.
inline std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                  std::int64_t, std::int64_t>
gate_up_engine_ledger_for_testing(const std::int64_t engine) {
    namespace fused = expert_mxfp4::fused_w13;
    const int id = static_cast<int>(engine);
    TORCH_CHECK(fused::engine_is_known(id),
                "MoK: unknown Kimi K3 gate/up engine ", engine);
    if (fused::engine_is_adaptive(id)) {
        // A selector's ledger is the union of its arms, not one arm's. The
        // bytes are the compact ring's because it is the wider and therefore
        // what the launch grants; the accumulator band is the wide arm's
        // because that is what the tensor pool must keep clear; and the gather
        // count is the compact ring's, which is the arm every measured
        // realistic route takes.
        return {
            fused::kFusedCompactSharedBytes,
            fused::kFusedCompactStagingBytes,
            fused::kFusedCompactStages,
            fused::kFusedSlabs,
            fused::fused_v4_accumulators(id),
            1};
    }
    return {
        fused::kFusedW13SharedBytes,
        fused::kFusedStagingBytes,
        fused::kFusedStages,
        fused::kFusedActivationSlabs,
        1,
        // Activation gathers one expert pass makes, which is one for the
        // resident ring's whole-K gather.
        1};
}

/// Whether this process runs the dependency-local schedule. It does by default.
///
/// This is what promotion means here. The dependency-local schedule keeps one of
/// the five full-grid barriers and replaces the rest with per-edge readiness. On
/// 8x B300, over five interleaved repeats of 1000 samples repeated as four
/// independent runs, it was 3.4-3.8% faster at the M = 16 median with a 2.9%
/// better p99, and 0.7-0.9% slower at M = 128. That last one is small but it is
/// real, it is reproduced in every run, and its margin against the 1% promotion
/// bar is under two tenths of a percent. It is what a decode step runs unless a
/// caller asks for the other one, because the shape it is slower at is the one
/// where the step is already bandwidth-bound and the shape it is faster at is
/// the one decode runs in.
///
/// It did not clear the 8% median gain the experiment was set up to require.
/// Promotion is against a separate and lower bar, and `task-11c-report.md`
/// records both verdicts rather than only the one that passed.
///
/// The barrier schedule is retained rather than deleted, and this switch is how
/// it is reached. It is not dead code: it is the other half of the A/B, and it
/// is what the bit-for-bit equality tests compare against -- two schedules that
/// must agree exactly are a much stronger statement than one schedule that
/// agrees with a tolerance-bounded oracle.
///
/// Guarded exactly like the grid override, the phase clocks and the gate/up
/// engine, and the reason is the adaptive integration rather than the promotion.
/// Turning this off does not merely select the other schedule: the barrier
/// schedule launches `kimi_k3_decode_persistent_kernel`, which is compiled
/// against the resident two-stage ring, so an unguarded write here is a way to
/// route a public decode through *both* retired paths at once. That is the one
/// thing the round was for closing off, so it is closed off the same way as the
/// rest: an unguarded read returns the dependency-local schedule whatever the
/// storage holds, and the setter refuses without the guard.
///
/// Which makes the storage's value irrelevant to production rather than merely
/// usually-correct. A benchmark process that selected the barrier schedule and
/// then dropped the variable launches the dependency-local one, and so does a
/// process that never set it.
static __host__ std::atomic<int> &dependency_schedule_storage() {
    static std::atomic<int> schedule{1};
    return schedule;
}

inline bool dependency_schedule_enabled() {
    if (!benchmark_grid_tuning_enabled()) return true;
    return dependency_schedule_storage().load(std::memory_order_relaxed) != 0;
}

inline void set_dependency_schedule_for_testing(const bool enabled) {
    TORCH_CHECK(
        benchmark_grid_tuning_enabled(),
        "MoK: the Kimi K3 decode schedule selector is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_GRID_TUNING=1 in a dedicated benchmark process");
    dependency_schedule_storage().store(
        enabled ? 1 : 0, std::memory_order_relaxed);
}

inline bool dependency_schedule_for_testing() {
    return dependency_schedule_enabled();
}

/// The accumulators' scratch band, their names, and their containing regions.
///
/// The parents are part of the metadata rather than the reader's convention
/// because the band is not flat: nine of the twenty-two regions refine another
/// region and measure the same cycles again at a finer grain. A reader that
/// summed all of them would overstate the launch by more than a third, so the
/// structure that says which may be added has to come from the same header
/// that defines the clocks.
inline std::tuple<std::int64_t, std::vector<std::string>,
                  std::vector<std::int64_t>>
phase_clock_metadata_for_testing() {
    std::vector<std::string> names;
    for (const char *const name : kPhaseClockNames) names.emplace_back(name);
    std::vector<std::int64_t> parents;
    for (const int parent : kPhaseClockParents) {
        parents.push_back(static_cast<std::int64_t>(parent));
    }
    return {static_cast<std::int64_t>(kPhaseClockBegin), names, parents};
}

inline void set_benchmark_grid_ctas_for_testing(const std::int64_t grid_ctas) {
    TORCH_CHECK(
        benchmark_grid_tuning_enabled(),
        "MoK: Kimi K3 grid override is benchmark-only; set "
        "MOK_KIMI_K3_ENABLE_GRID_TUNING=1 in a dedicated benchmark process");
    bool accepted = false;
    for (const int candidate : kBenchmarkGridCtas) {
        accepted = accepted || grid_ctas == candidate;
    }
    TORCH_CHECK(
        accepted,
        "MoK: Kimi K3 benchmark grid must be one of 64, 96, 128, or 148, got ",
        grid_ctas);
    benchmark_grid_ctas_storage().store(
        static_cast<int>(grid_ctas), std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------

}  // namespace persistent
}  // namespace kimi_k3_decode
