"""Pack the Kimi K3 MXFP4 cross-check fixtures with MoK's own kernel on a B300.

Step 1 of 2. The bytes land on a Modal volume so that
``kimi_k3_mxfp4_flashinfer.py`` compares real SM103 kernel output inside the
official Kimi K3 vLLM image, instead of re-deriving it from a Python model of
the packer.

Run from the repository root:

    modal run benchmarks/frameworks/kimi_k3_mxfp4_pack.py
"""

from __future__ import annotations

import sys
from pathlib import Path

IMAGE_ROOT = Path("/root/mok")


def _repo_root() -> Path:
    """Locate the repository, locally and inside the image.

    Modal re-imports this module in the container, where it lands outside the
    repository tree, so ``__file__`` only finds the checkout locally.
    """
    for candidate in (*Path(__file__).resolve().parents, IMAGE_ROOT):
        if (candidate / "modal_app.py").is_file():
            return candidate
    raise RuntimeError("cannot locate the MoK repository root")


REPO_ROOT = _repo_root()
sys.path[:0] = [str(REPO_ROOT)]

import modal

from benchmarks.frameworks.kimi_k3_mxfp4_cases import (
    CASES,
    PAYLOAD_NAME,
    build_case,
    count_zero_groups,
)
from modal_images import IMAGE as MOK_IMAGE_BASE
from modal_images import ORCHESTRATION_FILES, SPEC

app = modal.App("k3-mxfp4-pack")
volume = modal.Volume.from_name("k3-mxfp4-crosscheck", create_if_missing=True)
BRIDGE = "/bridge"

# Appended after the cached compile layer, so this only re-adds orchestration.
MOK_IMAGE = MOK_IMAGE_BASE
for _file in ORCHESTRATION_FILES:
    MOK_IMAGE = MOK_IMAGE.add_local_file(
        REPO_ROOT / _file, remote_path=f"/root/mok/{_file}", copy=True
    )


@app.function(image=MOK_IMAGE, gpu=SPEC.gpu, timeout=3600, volumes={BRIDGE: volume})
def pack_reference_bytes() -> None:
    import torch

    from mok.kimi_k3 import dequant_kimi_k3_mxfp4, pack_kimi_k3_mxfp4

    capability = "".join(map(str, torch.cuda.get_device_capability(0)))
    # Plain strings only: the consumer loads this payload with weights_only=True.
    meta = {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "gpu": str(torch.cuda.get_device_name(0)),
        "capability": f"sm_{capability}",
    }
    for key, value in meta.items():
        print(f"{key:12s}: {value}")

    payload: dict[str, object] = {"meta": meta, "cases": {}}
    for case in CASES:
        weight = build_case(case)
        entry: dict[str, object] = {
            "bf16": weight.cpu(),
            "logical_k": case.logical_k,
            "consumer": case.consumer,
            "zero_groups": count_zero_groups(weight),
        }

        # The canonical prepared layout is unpadded, so these are both the bytes
        # FlashInfer's quantizer is compared against and the bytes it consumes.
        packed, scale = pack_kimi_k3_mxfp4(
            weight.unsqueeze(0), padded_k=case.logical_k
        )
        entry["packed"] = packed.squeeze(0).cpu()
        entry["scale"] = scale.squeeze(0).cpu()
        entry["dequant"] = dequant_kimi_k3_mxfp4(
            packed, scale, logical_k=case.logical_k
        ).squeeze(0).cpu()

        payload["cases"][case.name] = entry
        print(
            f"{case.name:7s}: bf16 {tuple(weight.shape)} "
            f"packed {tuple(packed.shape)} scale {tuple(scale.shape)} "
            f"zero_groups={entry['zero_groups']} consumer={case.consumer}"
        )
        native = (case.rows, case.packed_columns), (case.rows, case.scale_columns)
        observed = tuple(packed.shape[1:]), tuple(scale.shape[1:])
        if observed != native:
            raise SystemExit(
                f"{case.name} packed at {observed} instead of the native K=32 "
                f"layout {native}"
            )
        if case.consumer and not entry["zero_groups"]:
            raise SystemExit(
                f"{case.name} must contain a zero group: the consumer run has to "
                "exercise the 0x7f scale byte"
            )

    torch.save(payload, f"{BRIDGE}/{PAYLOAD_NAME}")
    volume.commit()
    print(f"WROTE {BRIDGE}/{PAYLOAD_NAME}")


@app.local_entrypoint()
def main() -> None:
    pack_reference_bytes.remote()
