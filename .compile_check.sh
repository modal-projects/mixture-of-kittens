#!/usr/bin/env bash
# Local syntax/resource check for the SM103 build: compile only, no link.
#
# The workspace has no GPU and a CPU-only PyTorch, so the extension cannot be
# linked here. Compiling to an object still runs the whole front end and ptxas,
# which is what the register, stack, and shared-memory numbers come from.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
NVCC=.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
PYTHON_INCLUDES=$($PY -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYTORCH_INCLUDES=$($PY -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I' + p for p in include_paths()))")

exec $NVCC -c csrc/bindings.cu -o /tmp/mok_bindings.o \
    -DNDEBUG -lineinfo \
    --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
    -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0 \
    -std=c++20 -O3 --use_fast_math \
    -Xptxas=--verbose -Xptxas=--warn-on-spills \
    -I third_party/ThunderKittens/include \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
    -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C \
    -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE \
    $PYTHON_INCLUDES $PYTORCH_INCLUDES \
    -diag-suppress 3189 \
    -DKITTENS_SM103 -gencode arch=compute_103a,code=sm_103a \
    "$@"
