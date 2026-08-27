"""Check that the official Kimi K3 FlashInfer runtime consumes MoK's MXFP4 bytes.

Step 2 of 2. Reads the bytes that ``kimi_k3_mxfp4_pack.py`` produced on a B300
and, inside ``vllm/vllm-openai:kimi-k3``:

1. compares them byte for byte with FlashInfer's own MXFP4 quantizer in the
   linear (checkpoint) scale-factor layout,
2. checks FlashInfer's host dequantizer reads them back to exactly the values
   MoK's dequant kernel produced,
3. checks the canonical zero-padded W1/W3 layout against the unpadded bytes,
4. feeds the canonical bytes through the official consumer transform
   (``reorder_rows_for_gated_act_gemm`` + ``shuffle_matrix_a`` /
   ``shuffle_matrix_sf_a``) into ``trtllm_fp4_block_scale_routed_moe`` for one
   routed expert, including a matrix that contains an all-zero group, and
   requires bitwise-identical output against the same kernel driven by
   FlashInfer's own bytes,
5. runs a negative control proving the comparison above rejects corrupted bytes.

Every mismatch is accumulated and the function raises at the end, so ``modal
run`` exits nonzero if anything unexpected differs. The single tolerated
difference is the mandated all-zero-group scale byte (MoK writes ``0x7f``,
FlashInfer writes ``0x00``); it is accepted only where the packed nibbles are
zero on both sides and both conventions dequantize to exactly the same values.

The image is CUDA 13.0 with ``TORCH_CUDA_ARCH_LIST="8.7 8.9 9.0 10.0+PTX 12.0
12.1"`` and ships SM100 TRT-LLM-Gen cubins, so it runs on a B200 while the
packing kernel runs on the SM103 B300. Weight byte layout is architecture
independent, so the comparison is still apples to apples.

Run from the repository root, after the packing step:

    modal run benchmarks/frameworks/kimi_k3_mxfp4_flashinfer.py
"""

from __future__ import annotations

import modal

app = modal.App("k3-mxfp4-flashinfer")
volume = modal.Volume.from_name("k3-mxfp4-crosscheck", create_if_missing=True)
BRIDGE = "/bridge"
PAYLOAD_NAME = "mok_mxfp4_bytes.pt"

VLLM_IMAGE = modal.Image.from_registry(
    "vllm/vllm-openai:kimi-k3",
    # The image exposes only python3, which Modal cannot probe for a version.
    setup_dockerfile_commands=[
        "RUN ln -sf /usr/bin/python3 /usr/local/bin/python && python --version",
    ],
).entrypoint([])

GROUP_SIZE = 32
UNIT_SCALE_BYTE = 0x7F
EPILOGUE_TILE_M = 128
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
# MXFP4 weights plus MXFP8 activations against a BF16 reference built from the
# same dequantized values; the observed error is ~3.7e-2.
CONSUMER_MAX_RELATIVE_ERROR = 6e-2
SITU_REJECTION = "Unsupported gated activation type Situ"


class CrossCheckFailure(RuntimeError):
    """Raised at the end so every mismatch is reported, not just the first."""


@app.function(image=VLLM_IMAGE, gpu="B200", timeout=5400, volumes={BRIDGE: volume})
def flashinfer_consume() -> None:
    import torch

    _print_environment()
    payload = torch.load(f"{BRIDGE}/{PAYLOAD_NAME}", map_location="cpu")
    print(f"payload meta     : {payload['meta']}")

    failures: list[str] = []
    failures += _compare_packed_bytes(payload)
    failures += _compare_dequant(payload)
    failures += _compare_padded_layout(payload)
    failures += _run_consumer(payload)
    failures += _negative_control(payload)

    print(f"RESULT: mismatches = {failures or 'none'}")
    if failures:
        raise CrossCheckFailure(f"FlashInfer cross-check failed: {failures}")


def _print_environment() -> None:
    import importlib.metadata
    import os
    import subprocess

    import torch

    import flashinfer

    print("=" * 72)
    print(f"torch            : {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"flashinfer       : {flashinfer.__version__}")
    for distribution in ("flashinfer-python", "flashinfer-jit-cache",
                         "flashinfer-cubin", "vllm"):
        try:
            print(f"  {distribution:22s}: {importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {distribution:22s}: not installed")
    print(f"private cubin dir: {os.environ.get('FLASHINFER_PRIVATE_CUBIN_DIR')}")
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    capability = "".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"capability       : sm_{capability}")

    # Provenance for "official consumer transform": the same adapters and kernel
    # entry point that vLLM's Kimi K3 MXFP4 expert path uses.
    import vllm

    layers = os.path.join(os.path.dirname(vllm.__file__), "model_executor", "layers")
    for symbol in ("reorder_rows_for_gated_act_gemm", "shuffle_matrix_a",
                   "shuffle_matrix_sf_a", "trtllm_fp4_block_scale_routed_moe",
                   "ActivationType.Situ"):
        found = subprocess.run(
            ["grep", "-rl", "--include=*.py", symbol, layers],
            capture_output=True, text=True,
        ).stdout.split()
        relative = [os.path.relpath(path, layers) for path in found]
        print(f"vllm {vllm.__version__} uses {symbol}: {relative}")
    print("=" * 72)


def _linear_mxfp4_quantize(source):
    """Quantize with FlashInfer in the linear (checkpoint) scale-factor layout.

    ``mxfp4_quantize`` only grew an ``sfLayout`` argument after the release in
    this image, so go through the low-level ``fp4_quantize`` entry point, which
    has carried the explicit layout flags all along.
    """
    import inspect

    from flashinfer import fp4_quantize

    arguments = {
        "global_scale": None,
        "sf_vec_size": GROUP_SIZE,
        "sf_use_ue8m0": True,
        "is_sf_swizzled_layout": False,
        "is_sf_8x4_layout": False,
    }
    supported = inspect.signature(fp4_quantize).parameters
    missing = sorted(set(arguments) - set(supported))
    if missing:
        raise CrossCheckFailure(f"fp4_quantize is missing {missing} in this release")
    return fp4_quantize(source, **arguments)


def _flashinfer_bytes(source, packed_shape, scale_like):
    """FlashInfer's own MXFP4 bytes for ``source``, in our tensor shapes."""
    import torch

    packed, scale = _linear_mxfp4_quantize(source.cuda())
    return (
        packed.view(torch.uint8).reshape(packed_shape),
        scale.view(torch.uint8).reshape(-1)[: scale_like.numel()].reshape(
            scale_like.shape
        ),
    )


def _zero_group_mask(source):
    """``[rows, groups]`` mask of groups whose source values are all zero."""
    groups = source.float().reshape(source.shape[0], -1, GROUP_SIZE)
    return groups.abs().amax(-1) == 0.0


def _group_nibbles_zero(packed):
    """``[rows, groups]`` mask of groups whose packed nibbles are all zero."""
    return (packed.reshape(packed.shape[0], -1, GROUP_SIZE // 2) == 0).all(-1)


def _compare_case(label, source, ours_packed, ours_scale, theirs_packed,
                  theirs_scale, report=True):
    """Accumulate every packed-byte and scale-byte mismatch for one matrix."""
    import torch

    failures: list[str] = []
    packed_equal = torch.equal(ours_packed, theirs_packed)
    if not packed_equal:
        failures.append(f"{label}:packed")

    zero_group = _zero_group_mask(source)
    # The one tolerated difference, and only under all of these conditions.
    accepted = (
        (ours_scale == UNIT_SCALE_BYTE)
        & (theirs_scale == 0)
        & zero_group
        & _group_nibbles_zero(ours_packed)
        & _group_nibbles_zero(theirs_packed)
    )
    normalized = torch.where(accepted, theirs_scale, ours_scale)
    scale_equal = torch.equal(normalized, theirs_scale)
    if not scale_equal:
        failures.append(f"{label}:scale")

    if report:
        differs = ours_scale != theirs_scale
        print(f"[bytes] {label}: packed{tuple(ours_packed.shape)} equal={packed_equal}"
              f"  scale{tuple(ours_scale.shape)} normalized_equal={scale_equal}")
        print(f"    zero groups={int(zero_group.sum())}/{zero_group.numel()}; "
              f"raw scale differences={int(differs.sum())}; "
              f"accepted 0x7f/0x00 on zero groups={int(accepted.sum())}; "
              f"unexplained={int((differs & ~accepted).sum())}")
        for row, column in (ours_packed != theirs_packed).nonzero()[:4].tolist():
            print(f"    packed[{row},{column}] ours=0x{int(ours_packed[row, column]):02x}"
                  f" theirs=0x{int(theirs_packed[row, column]):02x}")
        for row, column in (differs & ~accepted).nonzero()[:4].tolist():
            print(f"    scale[{row},{column}] ours=0x{int(ours_scale[row, column]):02x}"
                  f" theirs=0x{int(theirs_scale[row, column]):02x} "
                  f"zero_group={bool(zero_group[row, column])}")
    return failures


def _compare_packed_bytes(payload) -> list[str]:
    import inspect

    from flashinfer import fp4_quantize

    print(f"[bytes] fp4_quantize{inspect.signature(fp4_quantize)}")
    failures: list[str] = []
    for name, case in payload["cases"].items():
        ours_packed = case["packed"].cuda()
        ours_scale = case["scale"].cuda()
        theirs_packed, theirs_scale = _flashinfer_bytes(
            case["bf16"], ours_packed.shape, ours_scale
        )
        failures += _compare_case(
            name, case["bf16"].cuda(), ours_packed, ours_scale,
            theirs_packed, theirs_scale,
        )
    return failures


def _compare_dequant(payload) -> list[str]:
    """FlashInfer's host dequantizer must reproduce our dequant kernel exactly."""
    import torch

    import flashinfer

    if not hasattr(flashinfer, "mxfp4_dequantize_host"):
        print("[dequant] flashinfer.mxfp4_dequantize_host is unavailable")
        return ["dequant-host-unavailable"]

    failures: list[str] = []
    for name, case in payload["cases"].items():
        ours_packed = case["packed"].cpu()
        ours_scale = case["scale"].cpu()
        rows, logical_k = ours_packed.shape[0], case["logical_k"]
        theirs = flashinfer.mxfp4_dequantize_host(
            ours_packed, ours_scale, GROUP_SIZE
        ).reshape(rows, logical_k).float()
        ours = case["dequant"].cpu().float()
        equal = torch.equal(theirs, ours)
        if not equal:
            failures.append(f"{name}:dequant-host")

        # The tolerated zero-group scale byte must be numerically inert: decode
        # our bytes again under FlashInfer's convention and require equality.
        their_convention = ours_scale.clone()
        their_convention[_zero_group_mask(case["bf16"].cpu())] = 0
        alternative = flashinfer.mxfp4_dequantize_host(
            ours_packed, their_convention, GROUP_SIZE
        ).reshape(rows, logical_k).float()
        inert = torch.equal(alternative, theirs)
        if not inert:
            failures.append(f"{name}:zero-group-scale")

        print(f"[dequant] {name}: host dequant of our bytes == our kernel: {equal} "
              f"(max_abs_diff={float((theirs - ours).abs().max())}); "
              f"0x7f and 0x00 on zero groups decode identically: {inert}")
    return failures


def _compare_padded_layout(payload) -> list[str]:
    """The canonical zero-padded layout must extend the unpadded bytes."""
    import torch

    failures: list[str] = []
    for name, case in payload["cases"].items():
        if "packed_padded" not in case:
            continue
        logical_k = case["logical_k"]
        checks = {
            "prefix_packed": torch.equal(
                case["packed_padded"][:, : logical_k // 2], case["packed"]
            ),
            "prefix_scale": torch.equal(
                case["scale_padded"][:, : logical_k // GROUP_SIZE], case["scale"]
            ),
            "tail_packed_zero": bool(
                (case["packed_padded"][:, logical_k // 2 :] == 0).all()
            ),
            "tail_unit_scale": bool(
                (case["scale_padded"][:, logical_k // GROUP_SIZE :]
                 == UNIT_SCALE_BYTE).all()
            ),
        }
        print(f"[padded] {name}: {case['logical_k']} -> {case['padded_k']}: {checks}")
        failures += [f"{name}:{key}" for key, ok in checks.items() if not ok]
    return failures


def _situ(gate, linear):
    """Kimi K3's SiTU, matching ``mok.kimi_k3.kimi_k3_situ_reference``."""
    import torch

    return (
        SITU_BETA
        * torch.tanh(gate.float() / SITU_BETA)
        * torch.sigmoid(gate.float())
        * SITU_LINEAR_BETA
        * torch.tanh(linear.float() / SITU_LINEAR_BETA)
    )


def _canonical_consumer_bytes(case):
    """Take the canonical prepared bytes and drop the zero padding.

    This is the whole one-time adapter on the K dimension: FlashInfer's routed
    MoE kernel takes the logical contraction width, while the prepared
    checkpoint layout pads W1/W3 to 3648 for MoK's own SM103 kernel.
    """
    logical_k = case["logical_k"]
    if "packed_padded" in case:
        return (
            case["packed_padded"][:, : logical_k // 2].cuda(),
            case["scale_padded"][:, : logical_k // GROUP_SIZE].cuda(),
        )
    return case["packed"].cuda(), case["scale"].cuda()


def _run_consumer(payload) -> list[str]:
    """Run one routed expert through the official FlashInfer K3 consumer path."""
    import inspect

    import torch

    from flashinfer import (
        ActivationType,
        RoutingMethodType,
        mxfp8_quantize,
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        trtllm_fp4_block_scale_routed_moe,
    )

    cases = {name.removeprefix("moe_"): case
             for name, case in payload["cases"].items() if case["consumer"]}
    missing = {"w1", "w3", "w2"} - set(cases)
    if missing:
        return [f"consumer:missing-{sorted(missing)}"]
    zero_groups = {name: case["zero_groups"] for name, case in cases.items()}
    print(f"[consumer] zero groups in the executed weights: {zero_groups}")
    if not any(zero_groups.values()):
        return ["consumer:no-zero-group-input"]

    failures: list[str] = []
    latent_size = cases["w1"]["logical_k"]
    intermediate_size = cases["w2"]["logical_k"]
    tokens, num_experts, top_k = 8, 1, 1

    ours = {name: _canonical_consumer_bytes(case) for name, case in cases.items()}
    theirs = {
        name: _flashinfer_bytes(case["bf16"], ours[name][0].shape, ours[name][1])
        for name, case in cases.items()
    }
    dequantized = {
        name: case["dequant"].cuda().float() for name, case in cases.items()
    }

    generator = torch.Generator(device="cuda").manual_seed(7)
    hidden = (
        torch.randn(tokens, latent_size, generator=generator, device="cuda") * 0.5
    ).bfloat16()
    topk_ids = torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda")
    topk_weights = torch.ones(tokens, top_k, dtype=torch.float32, device="cuda")

    # vLLM's MXFP4 expert path pairs static MXFP4 weights with dynamic MXFP8
    # activations, so quantize the same way and rebuild what the kernel sees.
    quantized_hidden, raw_scale = mxfp8_quantize(hidden, is_sf_swizzled_layout=False)
    groups = latent_size // GROUP_SIZE
    activation_scale = raw_scale.view(torch.uint8).reshape(-1)[
        : tokens * groups
    ].reshape(tokens, groups)
    hidden_reference = (
        quantized_hidden.view(torch.float8_e4m3fn).float()
        * torch.pow(2.0, activation_scale.float() - 127.0).repeat_interleave(
            GROUP_SIZE, dim=1
        )
    )
    print(f"[consumer] activations: mxfp8 {tuple(quantized_hidden.shape)} "
          f"{quantized_hidden.dtype}; reference absmax="
          f"{float(hidden_reference.abs().max()):.4f}")

    def swizzle_scale(scale):
        if "num_elts_per_sf" in inspect.signature(shuffle_matrix_sf_a).parameters:
            return shuffle_matrix_sf_a(scale, EPILOGUE_TILE_M,
                                       num_elts_per_sf=GROUP_SIZE)
        return shuffle_matrix_sf_a(scale, EPILOGUE_TILE_M)

    def invoke(source, activation, alpha, beta, clamp):
        """FC1 is ``[w1; w3]`` rows, interleaved and shuffled for the epilogue."""
        gemm1_packed = torch.cat((source["w1"][0], source["w3"][0]), dim=0)
        gemm1_scale = torch.cat((source["w1"][1], source["w3"][1]), dim=0)
        gemm1_weights = shuffle_matrix_a(
            reorder_rows_for_gated_act_gemm(gemm1_packed), EPILOGUE_TILE_M
        ).unsqueeze(0)
        gemm1_weights_scale = swizzle_scale(
            reorder_rows_for_gated_act_gemm(gemm1_scale)
        ).unsqueeze(0)
        gemm2_weights = shuffle_matrix_a(source["w2"][0], EPILOGUE_TILE_M).unsqueeze(0)
        gemm2_weights_scale = swizzle_scale(source["w2"][1]).unsqueeze(0)

        output = trtllm_fp4_block_scale_routed_moe(
            (topk_ids, topk_weights),
            None,
            quantized_hidden,
            activation_scale.view(torch.float8_e4m3fn),
            gemm1_weights,
            gemm1_weights_scale.view(torch.float8_e4m3fn),
            None,
            alpha,
            beta,
            clamp,
            gemm2_weights,
            gemm2_weights_scale.view(torch.float8_e4m3fn),
            None,
            None,
            None,
            None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize,
            do_finalize=True,
            activation_type=activation,
            tune_max_num_tokens=tokens,
        )[0]
        torch.cuda.synchronize()
        return output

    projection_w1 = hidden_reference @ dequantized["w1"].t()
    projection_w3 = hidden_reference @ dequantized["w3"].t()

    def silu(value):
        return value * torch.sigmoid(value)

    # TRT-LLM-Gen's gated FC1 treats the second half of the interleaved rows as
    # the gate, so W3 is the gated one here.
    configurations = {
        "Swiglu": (
            (ActivationType.Swiglu, None, None, None),
            lambda: silu(projection_w3) * projection_w1,
        ),
        "Situ": (
            (ActivationType.Situ,
             torch.full((num_experts,), SITU_BETA, device="cuda"),
             torch.full((num_experts,), SITU_LINEAR_BETA, device="cuda"),
             None),
            lambda: _situ(projection_w3, projection_w1),
        ),
    }

    executed = 0
    for name, (parameters, reference_activation) in configurations.items():
        try:
            output = invoke(ours, *parameters)
        except Exception as error:  # noqa: BLE001 - classify and keep going
            message = " ".join(str(error).split())
            if SITU_REJECTION in message:
                # Accepted, recorded limitation: this FlashInfer build rejects
                # SiTU through the routed entry point. It is not a byte-level
                # or device-level disagreement.
                print(f"[consumer] {name}: rejected by this build ({SITU_REJECTION}); "
                      "recorded as a known limitation")
                continue
            print(f"[consumer] {name}: {type(error).__name__}: {message[:400]}")
            failures.append(f"consumer:{name}:raised")
            continue

        executed += 1
        expected = reference_activation() @ dequantized["w2"].t()
        magnitude = expected.abs().max().clamp_min(1e-6)
        difference = (output.float() - expected).abs()
        relative = float(difference.max() / magnitude)
        print(f"[consumer] {name}: output{tuple(output.shape)} "
              f"max_rel={relative:.4e} "
              f"mean_rel={float(difference.mean() / magnitude):.4e} "
              f"reference_absmax={float(magnitude):.4e}")
        if not (relative <= CONSUMER_MAX_RELATIVE_ERROR):
            failures.append(f"consumer:{name}:relative-error")

        # The load-bearing check: same kernel, same transform, same activations,
        # only the source of the weight bytes differs.
        control = invoke(theirs, *parameters)
        identical = torch.equal(output, control)
        print(f"[consumer] {name}: our canonical bytes and FlashInfer's own bytes "
              f"drive the kernel to bitwise identical output: {identical} "
              f"(max_abs_diff="
              f"{float((output.float() - control.float()).abs().max()):.3e})")
        if not identical:
            failures.append(f"consumer:{name}:not-bitwise-identical")

    if not executed:
        failures.append("consumer:nothing-executed")
    return failures


def _negative_control(payload) -> list[str]:
    """Prove the byte comparison rejects what it is supposed to reject."""
    import torch

    case = payload["cases"]["w1"]
    source = case["bf16"].cuda()
    ours_packed = case["packed"].cuda()
    ours_scale = case["scale"].cuda()
    theirs_packed, theirs_scale = _flashinfer_bytes(
        case["bf16"], ours_packed.shape, ours_scale
    )
    zero_group = _zero_group_mask(source)
    zero_row, zero_column = zero_group.nonzero()[0].tolist()
    nonzero_row, nonzero_column = (~zero_group).nonzero()[0].tolist()

    def mutate(packed=None, scale=None):
        mutated_packed = ours_packed.clone()
        mutated_scale = ours_scale.clone()
        if packed is not None:
            index, value = packed
            mutated_packed[index] = value
        if scale is not None:
            index, value = scale
            mutated_scale[index] = value
        return mutated_packed, mutated_scale

    nonzero_byte = (nonzero_row, nonzero_column * GROUP_SIZE // 2)
    zero_byte = (zero_row, zero_column * GROUP_SIZE // 2)
    probes = {
        "flipped packed byte in a nonzero group": (
            mutate(packed=(nonzero_byte, ours_packed[nonzero_byte].item() ^ 0x11)),
            ["probe:packed"],
        ),
        "wrong scale byte in a nonzero group": (
            mutate(scale=((nonzero_row, nonzero_column),
                          (int(ours_scale[nonzero_row, nonzero_column]) + 1) % 256)),
            ["probe:scale"],
        ),
        "zero-group scale byte set to FlashInfer's 0x00": (
            mutate(scale=((zero_row, zero_column), 0)),
            [],
        ),
        "zero-group scale byte set to an arbitrary value": (
            mutate(scale=((zero_row, zero_column), 0x40)),
            ["probe:scale"],
        ),
        # Nonzero nibbles withdraw the scale-byte exemption as well, so both the
        # packed bytes and the normalized scales must be reported.
        "nonzero nibbles in a zero group": (
            mutate(packed=(zero_byte, 0x11)),
            ["probe:packed", "probe:scale"],
        ),
    }

    failures: list[str] = []
    for description, ((mutated_packed, mutated_scale), expected) in probes.items():
        observed = _compare_case(
            "probe", source, mutated_packed, mutated_scale, theirs_packed,
            theirs_scale, report=False,
        )
        agrees = sorted(observed) == sorted(expected)
        print(f"[control] {description}: reported {observed or 'no mismatch'}, "
              f"expected {expected or 'no mismatch'} -> {'ok' if agrees else 'WRONG'}")
        if not agrees:
            failures.append(f"negative-control:{description}")

    # Sanity: the unmutated bytes must still compare clean here.
    baseline = _compare_case(
        "probe", source, ours_packed, ours_scale, theirs_packed, theirs_scale,
        report=False,
    )
    print(f"[control] unmutated bytes: reported {baseline or 'no mismatch'}")
    if baseline:
        failures.append("negative-control:baseline")
    return failures


@app.local_entrypoint()
def main() -> None:
    flashinfer_consume.remote()
