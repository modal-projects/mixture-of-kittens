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


def test_routed_queues_claim_four_adjacent_units_per_atomic() -> None:
    """Batch only the long routed queues, leaving mixed work ordering intact."""
    sync = _source("persistent_sync.cuh")
    claim = _function_body(sync, "int claim_unit_batch(")
    kernel = _function_body(
        _source("persistent_kernel.cuh"),
        "void kimi_k3_decode_persistent_kernel(",
    )

    assert "kRoutedClaimBatch = 4" in sync
    assert "atomicAdd(" in claim
    assert "static_cast<unsigned int>(BATCH)" in claim
    assert kernel.count("claim_unit_batch<kRoutedClaimBatch>(") == 2
    assert kernel.count("claim_unit(") == 1
