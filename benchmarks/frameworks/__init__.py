"""Cross-framework harnesses that check MoK against reference implementations.

These are Modal apps rather than pytest tests: each one needs a GPU plus a
third-party runtime (vLLM, FlashInfer) that cannot be installed alongside MoK,
so they run in their own images and hand data to each other over a volume.
"""
