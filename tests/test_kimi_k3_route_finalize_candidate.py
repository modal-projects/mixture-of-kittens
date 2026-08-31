"""Focused TP8 sanitizer target for benchmark-only route-major finalize."""

from __future__ import annotations

import pytest
import torch

from benchmarks import kimi_k3_route_finalize_runtime as candidate
from mok.kimi_k3 import KimiK3DecodeWeights, KimiK3DecodeWorkspace
from tests.kimi_k3_decode_support import (
    assert_decode_close,
    assert_identical_across_ranks,
    decode_reference,
    routing,
    weights,  # noqa: F401
    with_routing,
    workspace,  # noqa: F401
)


@pytest.mark.parametrize("mode", ["disjoint", "concentrated"])
def test_candidate_sanitizer_paths_match_official_reference(
    workspace: KimiK3DecodeWorkspace,
    weights: KimiK3DecodeWeights,
    tp8_context: tuple[int, int, torch.device],
    mode: str,
) -> None:
    _, _, device = tp8_context
    plan = routing(mode, device, 16, weights)
    routed = with_routing(weights, plan)
    expected = decode_reference(plan.hidden, routed)
    candidate_workspace = candidate.create_candidate_workspace(workspace)

    with candidate.route_finalize_enabled():
        actual = candidate.decode_step(
            candidate_workspace, routed, plan.hidden
        )
        torch.cuda.synchronize(device)

    assert_decode_close(actual, expected)
    assert_identical_across_ranks(actual)
