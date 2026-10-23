# abl_old_none

Old-data ablation of `xgboost` (one change): INCLUDE_OLD none: new-regime windows only (v7 behaviour). The 2025-01-01 refit has no fit windows outside the early-stopping slice, so it uses fixed_iters (about the baseline's counts at 2e187b1: catboost 2,421, xgboost 643); the 2025-04-01 refit early-stops as usual.

See `scripts/run_old_ablation.sh`.
