# catboost_fc: Baseline + CatBoost (whole_pipeline_v8)

v8's household model behind the harness (`core/boost_harness.py`): segments and segment beta priors, v8 features and
thermal baseline, residual target (actual - baseline), early stopping on the inner validation block, final refit.
CatBoost and XGBoost come from **one** fit per refit date, shared by catboost / xgboost (and by catboost_fc /
xgboost_fc), stored in `<out_dir>/artifacts/_boost/<commit>/<config hash>/refit-<R>`.

- Weather: simulated 10-day forecast issued on the issue date: preliminary, run with `--weather simulated`.
- Training data: `buckets` (customer_bucket values, ~1 % of customers each; default 5 %) or `buckets: null` for every
  customer, optionally with `train_sample` (share of windows) and a larger `max_window_gb`.
- History ends on o - 1 = the window's origin: the 60-day output is shifted one day (h60 repeats day 60).
- A household's segment is assigned from its own history up to o - 1 with the fitted clustering (no customer list is
  stored). `SEGMENT_USE_OLD: true`: the 2024 refits have old-regime data only.
- Refits without validation windows (the May - mid-August 2024 label hole) reuse the tree counts of the previous refit.
