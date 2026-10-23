# abl_old_nonwinter

Old-data ablation of `xgboost` (one change): INCLUDE_OLD non_winter: drop old windows whose 60-day target touches Dec-Apr, where old winters read 7-17 % high for the same meters.

See `scripts/run_old_ablation.sh`.
