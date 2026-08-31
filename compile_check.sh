#!/usr/bin/env bash
# Compile-only check of the CUDA translation unit, for a host with no GPU.
#
# Catches every error the device compiler can see -- static_asserts, template
# instantiation, resource limits -- without needing a device or a CUDA build of
# torch. Not a substitute for the Modal gates: it does not link, and ptxas is
# only asked for -O1 here so the check is fast.
set -euo pipefail

PY=${PY:-/workspace/.venv/bin/python}
OPT=${OPT:--O1}
cd "$(dirname "$0")"

nvcc csrc/bindings.cu -c -o /tmp/k3-compile-check.o \
    -DNDEBUG --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
    -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0 \
    -std=c++20 "${OPT}" \
    -I./third_party/ThunderKittens/include -I/tmp/k3stub \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
    -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C \
    "$(${PY} -c 'import sysconfig; print("-I"+sysconfig.get_path("include"))')" \
    $(${PY} -c 'from torch.utils.cpp_extension import include_paths; print(" ".join("-I"+p for p in include_paths()))') \
    -diag-suppress 3189 \
    -DKITTENS_SM103 -gencode arch=compute_103a,code=sm_103a "$@"
