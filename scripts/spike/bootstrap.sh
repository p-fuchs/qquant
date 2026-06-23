#!/usr/bin/env bash
# Minimal, self-contained spike bootstrap (Spec 02) — independent of Spec 03's production
# bootstrap. On a bare CUDA 12.6 *devel* image it builds /workspace/venv (Python 3.11 +
# torch cu126 + the pinned eval stack) and installs the candidate lm-eval ref, recording
# the resolved ref (a commit SHA for `main`) to /workspace/lm_eval_ref.txt.
#
# Usage: bash bootstrap.sh <lm-eval-ref>   (ref is a PyPI version like 0.4.12, or a git
# tag/branch/SHA). Distinct exit codes per phase so a NO-GO names where it broke:
#   2 = system/uv/venv   3 = torch+core deps   4 = gptqmodel build   5 = lm-eval
set -euo pipefail

REF="${1:-main}"
VENV=/workspace/venv
PY="$VENV/bin/python"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
LMH="git+https://github.com/EleutherAI/lm-evaluation-harness"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export DEBIAN_FRONTEND=noninteractive
export PIP_NO_INPUT=1
# ssh sessions don't inherit the image PATH; nvcc lives here and gptqmodel's sm_89 JIT
# build needs it.
export PATH="/usr/local/cuda/bin:$PATH"
mkdir -p /workspace "$HF_HOME"

# 1. system deps (the bare cuda image has no python/git): git + curl + a C toolchain
#    (gptqmodel JIT-compiles sm_89 kernels; nvcc is already in the devel image).
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git curl ca-certificates build-essential ninja-build || exit 2
fi

# 2. uv + a pinned Python 3.11 venv.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || exit 2
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11 || exit 2
# Idempotent across candidate refs (don't fail if a prior candidate already made the venv).
[ -x "$PY" ] || uv venv --python 3.11 "$VENV" || exit 2

# 3. torch from the cu126 index FIRST (gptqmodel builds against the installed torch).
#    Pinned to the decision-log version so the spike validates the EXACT money-matrix stack.
#    setuptools>=77: gptqmodel 7.1.0 and its deps (tokenicer, logbar) use the PEP 639
#    string license (`license = "Apache-2.0"`) and require setuptools>=77.0.1,<83 to build
#    (sdist-only on PyPI; built with --no-build-isolation against the venv's setuptools).
# torchvision: gptqmodel 7.1.0 imports it at model-load time (not declared in its deps).
uv pip install --python "$PY" --index-url "$TORCH_INDEX" "torch==2.12.1" torchvision || exit 3
# ninja: gptqmodel JIT-compiles its sm_89 Marlin torch.ops extension at first load and
# hard-requires ninja ("Ninja is required to load C++ extensions").
uv pip install --python "$PY" numpy "setuptools>=77.0.1,<83" wheel packaging ninja || exit 3

# 4. the pinned eval stack (PyPI). NEVER autoawq. gptqmodel needs --no-build-isolation.
uv pip install --python "$PY" \
  "transformers==5.10.1" "accelerate==1.13.0" "optimum>=2.2.0" \
  "bitsandbytes==0.49.2" "compressed-tensors==0.17.1" "datasets" || exit 3
uv pip install --python "$PY" --no-build-isolation "gptqmodel==7.1.0" || exit 4

# 5. lm-eval at the candidate ref. A PyPI version (e.g. 0.4.12) installs from PyPI; any
#    other ref is resolved to a commit SHA and installed from git.
if [[ "$REF" =~ ^[0-9]+\.[0-9]+ ]]; then
  uv pip install --python "$PY" "lm_eval[hf,ifeval,math]==${REF}" || exit 5
  echo "$REF" >/workspace/lm_eval_ref.txt
else
  # Strip the pip-only `git+` scheme for git ls-remote; `|| true` keeps the failing
  # command-substitution from aborting under `set -e` so we reach the fallback.
  SHA="$(git ls-remote "${LMH#git+}" "$REF" 2>/dev/null | head -n1 | cut -f1 || true)"
  SHA="${SHA:-$REF}"
  uv pip install --python "$PY" "lm_eval[hf,ifeval,math] @ ${LMH}@${SHA}" || exit 5
  echo "$SHA" >/workspace/lm_eval_ref.txt
fi

# 6. nltk data for downstream ifeval (best-effort; harmless to the spike).
"$PY" - <<'PY' || true
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY

echo "bootstrap ok: venv=$VENV ref=$(cat /workspace/lm_eval_ref.txt)"
