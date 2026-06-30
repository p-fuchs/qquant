#!/usr/bin/env bash
# HumanEval pass — runs ONLY the humaneval task for all 5 variants via lm_eval 0.4.12
# (in /workspace/venv412), because the spike-validated 0.4.3 commit in the main venv does
# not register a humaneval task. 0.4.12 was verified to work with transformers 5.10.1
# (apply_chat_template runtime path + code-exec + pass@1, via a CPU smoke). Cells written
# by this pass record lm_eval_version=0.4.12; the other three benchmarks stay on 0.4.3.
# Within-benchmark consistency across variants is preserved (each benchmark = one engine).
set -uo pipefail
PY=/workspace/venv412/bin/python
REPO=/workspace/qquant
export PATH="/workspace/venv412/bin:/usr/local/cuda/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONPATH="$REPO/src"
export HF_ALLOW_CODE_EVAL=1
export QQUANT_ALLOW_CODE_EXEC=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

# Reuse the same singleton lock as run_box.sh so the two can never run concurrently.
LOCK=/workspace/run_box.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another run holds the lock (pid $(cat "$LOCK")); abort." >&2
  exit 1
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
cd "$REPO"
for v in bf16 bnb-int8 bnb-nf4 gptq-official awq-official; do
  echo "----- humaneval $v  ($(date -u +%H:%M:%S)) -----"
  "$PY" -m qquant.eval.cli --variant "$v" --task humaneval \
    --results /workspace/qquant/results -v 2>&1 | tee "$LOGDIR/humaneval_$v.log"
done
echo "===== HUMANEVAL DONE  ($(date -u +%H:%M:%S)) ====="
