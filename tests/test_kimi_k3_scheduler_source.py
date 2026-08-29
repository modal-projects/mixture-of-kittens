"""Source contracts for Kimi K3 scheduling experiments."""

from pathlib import Path


_SOURCE_ROOT = Path(__file__).parents[1] / "csrc" / "kimi_k3_decode"


def _source(name: str) -> str:
    return (_SOURCE_ROOT / name).read_text(encoding="utf-8")


def _function_body(text: str, signature: str) -> str:
    start = text.index(signature)
    depth = 0
    for offset in range(text.index("{", start), len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise AssertionError(f"{signature} is never closed")


def test_routed_queue_batching_starts_above_the_primary_m16_regime() -> None:
    """Keep fine-grained scheduling at M16 and batch only wider routed work."""
    sync = _source("persistent_sync.cuh")
    policy = _function_body(sync, "int routed_claim_batch(")
    claim = _function_body(sync, "int claim_unit_batch(")
    kernel = _function_body(
        _source("persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "kRoutedClaimBatchThreshold = 16" in sync
    assert "kRoutedClaimBatch = 4" in sync
    assert "active_tokens <= kRoutedClaimBatchThreshold ? 1" in policy
    assert "atomicAdd(" in claim
    assert "static_cast<unsigned int>(batch)" in claim
    assert kernel.count("routed_claim_batch(active_tokens)") == 1
    assert kernel.count("claim_unit_batch(") == 2
    assert kernel.count("claim_unit(") == 1
