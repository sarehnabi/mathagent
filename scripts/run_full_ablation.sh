#!/bin/bash
# Run one model's ablation: <benchmarks> × 4 augmentation modes (default 3 benchmarks × 4 = 12 conditions).
#
# All outputs organized under a single master directory:
#   outputs/experiments/<exp-prefix>_MMDDYYYY_HHMM[_<model-name>]/
#     ├── ablation_log.txt
#     ├── ablation_summary.json
#     ├── minif2f_prover_only/
#     │   ├── eval_minif2f_prover_only.json
#     │   ├── eval_minif2f_prover_only.log
#     │   └── run_config.json
#     ├── minif2f_with_kg/
#     │   └── ...
#     └── ... (one sub-directory per benchmark × mode)
#
# Usage:
#   bash scripts/run_full_ablation.sh                              # defaults
#   bash scripts/run_full_ablation.sh --workers 20                 # EC2
#   bash scripts/run_full_ablation.sh --workers 20 --first 10      # quick test (10 problems)
#   bash scripts/run_full_ablation.sh --workers 20 --max-attempts 32 --kg-path path/to/mathkg_clean.json
#   bash scripts/run_full_ablation.sh --model-name qwen3_8b --benchmarks "minif2f" --last 50
#   Other flags: --exp-prefix NAME, --master-dir DIR (resume into an existing directory)

set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PATH="$HOME/.elan/bin:$PATH"

# Auto-detect Mathlib path: EC2 uses ~/mathlib4, Mac uses the config.py default
if [ -d "$HOME/mathlib4" ]; then
    export LEAN_MATHLIB_PROJECT="$HOME/mathlib4"
fi

# Lean compilation timeout (seconds) — KG-enhanced modes produce longer proofs
export LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-1200}"

# Defaults
WORKERS="${WORKERS:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
FIRST="${FIRST:-}"
MODEL_NAME="${MODEL_NAME:-}"
BENCHES="${BENCHES:-matholympiadbench minif2f putnambench}"
EXP_PREFIX="${EXP_PREFIX:-full_ablation}"
MASTER_DIR_OVERRIDE="${MASTER_DIR_OVERRIDE:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KG_PATH="${KG_PATH:-${REPO_ROOT}/outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers) WORKERS="$2"; shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --first) FIRST="$2"; shift 2 ;;
        --last) LAST="$2"; shift 2 ;;
        --kg-path) KG_PATH="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --benchmarks) BENCHES="$2"; shift 2 ;;
        --exp-prefix) EXP_PREFIX="$2"; shift 2 ;;
        --master-dir) MASTER_DIR_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

FIRST_FLAG=""
if [ -n "$FIRST" ]; then
    FIRST_FLAG="--first $FIRST"
fi
if [ -n "$LAST" ]; then
    FIRST_FLAG="--last $LAST"
fi

# Master directory for this ablation run
TIMESTAMP=$(date +%m%d%Y_%H%M)
if [ -n "$MASTER_DIR_OVERRIDE" ]; then
    MASTER_DIR="$MASTER_DIR_OVERRIDE"
elif [ -n "$MODEL_NAME" ]; then
    MASTER_DIR="outputs/experiments/${EXP_PREFIX}_${TIMESTAMP}_${MODEL_NAME}"
else
    MASTER_DIR="outputs/experiments/${EXP_PREFIX}_${TIMESTAMP}"
fi
mkdir -p "$MASTER_DIR"
LOG="$MASTER_DIR/ablation_log.txt"

echo "========================================" | tee "$LOG"
echo "FULL ABLATION STUDY" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Started:      $(date)" | tee -a "$LOG"
echo "Master dir:   $MASTER_DIR" | tee -a "$LOG"
echo "Workers:      $WORKERS" | tee -a "$LOG"
echo "Max attempts: $MAX_ATTEMPTS" | tee -a "$LOG"
echo "Problems:     ${FIRST:-all}" | tee -a "$LOG"
echo "KG path:      $KG_PATH" | tee -a "$LOG"
echo "Model:        $(grep MODEL_ID config.py | head -1)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

CONDITION=0
TOTAL=$(( $(echo $BENCHES | wc -w) * 4 ))

for bench in $BENCHES; do
  for mode in prover_only with_kg with_mathlib full_system; do
    CONDITION=$((CONDITION + 1))
    SUB_DIR="$MASTER_DIR/${bench}_${mode}"

    echo ">>> [$CONDITION/$TOTAL] ${bench} — ${mode} ($(date))" | tee -a "$LOG"

    python3 -u -m src.benchmarks.evaluate \
      --benchmark $bench \
      --eval-mode $mode \
      --workers $WORKERS \
      --max-attempts $MAX_ATTEMPTS \
      --kg-path $KG_PATH \
      --output-dir "$SUB_DIR" \
      $FIRST_FLAG \
      2>&1 | tee -a "$LOG"

    echo "<<< [$CONDITION/$TOTAL] ${bench} — ${mode} DONE ($(date))" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
  done
done

echo "========================================" | tee -a "$LOG"
echo "ALL $TOTAL CONDITIONS COMPLETE: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Collect results into summary JSON
echo "=== RESULTS SUMMARY ===" | tee -a "$LOG"
python3 -c "
import json, glob, os

master = '$MASTER_DIR'
results = []
for f in sorted(glob.glob(os.path.join(master, '*/eval_*.json'))):
    d = json.load(open(f))
    results.append({
        'benchmark': d['benchmark'],
        'eval_mode': d.get('eval_mode', '?'),
        'solved': d['summary']['solved'],
        'total': d['total_problems'],
        'solve_rate': d['summary']['solve_rate'],
        'time': d.get('total_time_seconds', 0),
        'per_domain': d['summary'].get('per_domain', {}),
    })
    print(f\"{d['benchmark']:25s} {d.get('eval_mode','?'):15s} → {d['summary']['solved']}/{d['total_problems']} solved ({d['summary']['solve_rate']})\")

with open(os.path.join(master, 'ablation_summary.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSummary saved to: {master}/ablation_summary.json')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Master directory: $MASTER_DIR" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
