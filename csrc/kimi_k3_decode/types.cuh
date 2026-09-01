#pragma once

// Every shape, offset and counter one decode step reads.
//
// A decode workspace is one flat byte buffer, and this is the only place that
// says what is in it: the model's shapes, each scratch region's offset, the
// phase counters, the timeout record, the dependency-local schedule's appended
// counters and readiness edge table, and the two typed handles -- `Scratch` and
// `PhaseClocks` -- every stage takes by value so that none of them re-derives
// an offset for itself.
//
// How this is laid out
// --------------------
// This was one 1,089-line header and is now a chain of focused ones, included
// here in dependency order. Each opens the same namespace and includes the one
// before it, so this file is the only include any caller needs.
//
// Nothing was renamed and nothing moved between namespaces, so a symbol's
// home is `git log --follow` on the part that holds it.

#include "layout/shapes.cuh"
#include "layout/schedule_counters.cuh"
#include "layout/phase_clocks.cuh"
#include "layout/timeouts.cuh"
#include "layout/schedule_edges.cuh"
#include "layout/schedule_clocks.cuh"
