#!/bin/bash
# PutnamBench rerun (September 2026): the 346 answer-slot problems, four modes, Claude Sonnet 4.6.
#
# Why: the original May 2026 run left the answer slots ("abbrev *_solution := sorry") for the
# model to fill and only checked that the file compiled; the model often filled the slot with a
# restatement of the question. This rerun inlines the benchmark's ground-truth answers (PutnamBench
# Task 2) and requires the theorem and all supplied definitions to be reproduced exactly
# (src/statement_check.py). The other 326 PutnamBench problems have no answer slot and keep their
# May results; the two are merged into the reported PutnamBench numbers.
#
# Everything else matches the May run: same Bedrock model profile (config.MODEL_ID), 32 sequential
# attempts, 20 workers, temperatures 0.3/0.2, 2,000-token output cap, Lean/Mathlib from ~/mathlib4.
#
# Usage (from the repo root):
#   bash scripts/run_putnam_rerun.sh                      # all four modes
#   bash scripts/run_putnam_rerun.sh --modes "prover_only" --first 3   # smoke test
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PATH="$HOME/.elan/bin:$PATH"
[ -d "$HOME/mathlib4" ] && export LEAN_MATHLIB_PROJECT="$HOME/mathlib4"
export LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-1200}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-2000}"   # 2,000-token output cap for this rerun (the May 2026 Prover used for it has this cap built in)

WORKERS="${WORKERS:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-32}"
MODES="${MODES:-prover_only with_kg with_mathlib full_system}"
IDS="${IDS:-outputs/validation/putnam_answer_slot_ids.txt}"
REPO_ROOT="$(pwd)"
KG_PATH="${KG_PATH:-${REPO_ROOT}/outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json}"
MASTER_DIR="${MASTER_DIR:-}"
FIRST_FLAG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --workers) WORKERS="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --ids) IDS="$2"; shift 2 ;;
    --first) FIRST_FLAG="--first $2"; shift 2 ;;
    --master-dir) MASTER_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[ -n "$MASTER_DIR" ] || MASTER_DIR="outputs/experiments/putnam_rerun_$(date +%m%d%Y_%H%M)_claude_sonnet"
mkdir -p "$MASTER_DIR"; LOG="$MASTER_DIR/ablation_log.txt"
{
  echo "PUTNAMBENCH RERUN (answer slots inlined, statement check on)"
  echo "Started: $(date)"; echo "Master dir: $MASTER_DIR"; echo "Workers: $WORKERS"; echo "Max attempts: $MAX_ATTEMPTS"
  echo "Problem ids: $IDS ($(grep -c . "$IDS") problems)"; echo "KG path: $KG_PATH"; echo "Model: $(grep MODEL_ID config.py | head -1)"
  echo "Mathlib: $(git -C "$HOME/mathlib4" rev-parse HEAD 2>/dev/null) $(cat "$HOME/mathlib4/lean-toolchain" 2>/dev/null)"
  echo "Code: prover=$(md5sum src/agents/prover.py | cut -c1-8) base_agent=$(md5sum src/agents/base_agent.py | cut -c1-8) evaluate=$(md5sum src/benchmarks/evaluate.py | cut -c1-8) statement_check=$(md5sum src/statement_check.py | cut -c1-8)"
} | tee "$LOG"
for mode in $MODES; do
  SUB_DIR="$MASTER_DIR/putnambench_${mode}"
  echo ">>> putnambench — ${mode} ($(date))" | tee -a "$LOG"
  python3 -u -m src.benchmarks.evaluate --benchmark putnambench --eval-mode "$mode" --workers "$WORKERS" \
    --max-attempts "$MAX_ATTEMPTS" --kg-path "$KG_PATH" --output-dir "$SUB_DIR" --problem-ids "$IDS" $FIRST_FLAG 2>&1 | tee -a "$LOG"
  echo "<<< putnambench — ${mode} DONE ($(date))" | tee -a "$LOG"
done
echo "ALL MODES COMPLETE: $(date)" | tee -a "$LOG"
