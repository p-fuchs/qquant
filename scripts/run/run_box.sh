#!/usr/bin/env bash
# Box-side full-run driver — minimal-5 quantization study (Specs 04/05/06).
#
# Preconditions (the local driver sets these up before invoking this script):
#   * /workspace/venv   = the pinned eval stack, built by scripts/spike/bootstrap.sh
#   * /workspace/qquant = this repo, scp'd as a working tree (NOT git-cloned: local main
#                         is ahead of origin)
#   * HF_HOME=/workspace/hf so model downloads land on the REUSABLE disk (stop, never
#     destroy → models persist → no re-paying the ~$0.04/GB download next session)
#
# Runs FULL quality eval (mmlu/gsm8k/humaneval/ifeval) + efficiency profiling for the 5
# minimal-scope variants, PER VARIANT so artifacts are written incrementally and the run
# is resumable (is_cell_done / efficiency resume-skip): a mid-run `stop` loses nothing
# already written. Efficiency runs FIRST (cheap, the complete memory/speed deliverable)
# so a tight-credit stop still banks that half. Self-quant variants are intentionally
# EXCLUDED — minimal scope has no local checkpoints (their `local:` model_id would crash).
set -uo pipefail   # NOT -e: one variant/task failing must not abort the rest

VENV=/workspace/venv
PY="$VENV/bin/python"
REPO=/workspace/qquant
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONPATH="$REPO/src"          # run in-place; no install, no extra download
RESULTS="${RESULTS:-$REPO/results}"
# HumanEval executes model-generated code; Spec 05's gate requires BOTH vars.
export HF_ALLOW_CODE_EVAL=1
export QQUANT_ALLOW_CODE_EXEC=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

VARIANTS=(bf16 bnb-int8 bnb-nf4 gptq-official awq-official)
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR" "$RESULTS"
cd "$REPO"

echo "===== EFFICIENCY PROFILE (per variant, resumable) — $(date -u +%H:%M:%S) ====="
for v in "${VARIANTS[@]}"; do
  echo "----- profile $v  ($(date -u +%H:%M:%S)) -----"
  "$PY" -m qquant.efficiency.cli --variant "$v" --results "$RESULTS" \
    2>&1 | tee "$LOGDIR/profile_$v.log"
done

echo "===== QUALITY EVAL (full benchmarks, per variant, resumable) — $(date -u +%H:%M:%S) ====="
for v in "${VARIANTS[@]}"; do
  echo "----- eval $v  ($(date -u +%H:%M:%S)) -----"
  "$PY" -m qquant.eval.cli --variant "$v" --results "$RESULTS" -v \
    2>&1 | tee "$LOGDIR/eval_$v.log"
done

echo "===== RUN COMPLETE — $(date -u +%H:%M:%S) ====="
echo "results under: $RESULTS   logs under: $LOGDIR"
