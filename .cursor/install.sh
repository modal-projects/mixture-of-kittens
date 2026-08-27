#!/usr/bin/env bash
# Idempotent setup for the Mixture-of-Kittens dev environment.
#
# MoK's kernels require NVIDIA Blackwell GPUs (B200/B300), which are not present on the
# dev box. Instead, this environment drives builds and runs on Modal (see modal_app.py).
# Locally we only need: the ThunderKittens submodule (copied into the Modal image build)
# and the Modal client.
set -euo pipefail

cd "$(dirname "$0")/.."

# ThunderKittens headers are copied into the Modal image at build time, so the submodule
# must be checked out locally.
git submodule update --init --recursive -- third_party/ThunderKittens

# Python virtualenv with the Modal client used to submit GPU jobs.
if ! python3 -m venv .venv 2>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet modal

echo "MoK dev environment ready."
echo "  - ThunderKittens: $(git -C third_party/ThunderKittens rev-parse --short HEAD 2>/dev/null || echo missing)"
echo "  - modal client:   $(.venv/bin/modal --version 2>/dev/null || echo missing)"
echo
echo "Next steps (needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET, and MODAL_ENVIRONMENT for a"
echo "non-default Modal environment):"
echo "  source .venv/bin/activate"
echo "  modal run modal_app.py::gpu_info   # build + kernel check on one B300"
echo "  modal run modal_app.py::bench      # 8x B300 benchmark + correctness"
