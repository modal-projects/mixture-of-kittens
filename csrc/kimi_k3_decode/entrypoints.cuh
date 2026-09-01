#pragma once

// The host boundary of the Kimi K3 decode path.
//
// Everything a caller can get wrong is rejected here, before a launch starts:
// shapes, dtypes, devices, alignments, symmetric pointer lists, multicast
// disjointness, and the workspace signature. The private per-stage
// entrypoints are here too, because they check the same preconditions the
// public step does and launching a stage on an unchecked workspace would make
// its test evidence worthless.
//
// How this is laid out
// --------------------
// This was one 1,039-line header and is now a chain of focused ones, included
// here in dependency order. Each opens the same namespace and includes the one
// before it, so this file is the only include any caller needs.
//
// Nothing was renamed and nothing moved between namespaces, so a symbol's
// home is `git log --follow` on the part that holds it.

#include "entrypoint/device.cuh"
#include "entrypoint/private_stages.cuh"
#include "entrypoint/symmetric_memory.cuh"
#include "entrypoint/tail.cuh"
#include "entrypoint/decode_step.cuh"
