#!/usr/bin/env bash
# The substation / feeder test (configs/test_set_substations_2025.yaml) on the DEV substations for several household
# models, then one table on the same node-days. Fits are cached per model, commit and refit date (shared with district
# runs of the same commit).
#
#   bash scripts/run_node_models.sh                                  # tide_agg tide_daily xgboost nlinear_x
#   MODELS="tide_agg tide_daily" bash scripts/run_node_models.sh
#   MODE=test bash scripts/run_node_models.sh                        # the test substations: once, chosen models only
#
# Needs the node set: python scripts/test_set/build_node_test_set.py --workers 12. Logs: ~/work/eval/logs/nodes_<model>.txt
set -uo pipefail
cd "$(dirname "$0")/.."
TS=${TS:-configs/test_set_substations_2025.yaml}
MODE=${MODE:-validate}
MODELS=${MODELS:-"tide_agg tide_daily xgboost nlinear_x"}
WORKERS=${WORKERS:-8}
LOGS=~/work/eval/logs
mkdir -p "$LOGS"
ROLE=$([ "$MODE" = "test" ] && echo test || echo dev)
echo "[nodes] $(git rev-parse --short HEAD) | $MODE ($ROLE substations) | models: $MODELS"
OK=""
for m in $MODELS; do
  echo "[nodes] $(date +%H:%M) $m ..."
  if python evaluate.py --model "$m" --mode "$MODE" --test-set "$TS" --workers "$WORKERS" > "$LOGS/nodes_$m.txt" 2>&1; then
    grep -E "^\[done\]|early stop|\[tide\] windows|\[leakage\]" "$LOGS/nodes_$m.txt"
    OK="$OK $m"
  else
    echo "[nodes] $m FAILED: tail of $LOGS/nodes_$m.txt"; tail -20 "$LOGS/nodes_$m.txt"
  fi
done
[ -n "$OK" ] && python scripts/compare_node_runs.py --role "$ROLE" --latest $OK --out ~/work/eval/node_compare_$ROLE
