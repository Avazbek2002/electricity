#!/usr/bin/env bash
# Old-data ablation: the `xgboost` baseline (INCLUDE_OLD all, OLD_WEIGHT 1.0, OLD_REGIME_FLAG on) and five one-change
# variants (models/abl_old_*), each trained on the GPU and scored on the SAME issue dates and B0 rows, then one table.
#
#   bash scripts/run_old_ablation.sh                     # 5-district sample, all 181 validation dates (default)
#   DATES=reforecast bash scripts/run_old_ablation.sh    # only the 24 forecast dates (faster, noisier)
#   MODELS="abl_old_w050 abl_old_w025" bash scripts/run_old_ablation.sh    # a subset
#
# About 30 min a model on the B200 pod (two refits + scoring); run it in tmux. Logs: ~/work/eval/logs/ablation_<model>.txt
set -uo pipefail
cd "$(dirname "$0")/.."
TS=${TS:-configs/test_set_2025.yaml}
DATES=${DATES:-all}
MODELS=${MODELS:-"xgboost abl_old_none abl_old_nonwinter abl_old_w050 abl_old_w025 abl_old_noflag"}
SCOPE=${SCOPE:---sample}
LOGS=~/work/eval/logs
mkdir -p "$LOGS"
SINCE=$(date +%Y%m%d-%H%M%S)
echo "[ablation] $(git rev-parse --short HEAD) | dates $DATES | $SCOPE | models: $MODELS | since $SINCE"
for m in $MODELS; do
  echo "[ablation] $(date +%H:%M) $m ..."
  python evaluate.py --model "$m" --mode validate $SCOPE --issue-dates "$DATES" --test-set "$TS" \
    > "$LOGS/ablation_$m.txt" 2>&1 \
    && grep -E "^\[done\]|best iteration on validation|WARNING" "$LOGS/ablation_$m.txt" \
    || { echo "[ablation] $m FAILED: tail of $LOGS/ablation_$m.txt"; tail -20 "$LOGS/ablation_$m.txt"; }
done
python scripts/ablation_table.py --since "$SINCE" --models $MODELS
