# Baseline + booster with the real weather forecast (AIFS / FCN3)

Same v8 model as `catboost` / `xgboost` (features, thermal baseline, residual boosters, one shared fit per refit for
catboost_wx / xgboost_wx), but the target-period weather for h1..h14 comes from the real forecast issued on the issue
date (`core/reforecast.py`), at training and at predict time. No simulated forecast anywhere.

- First, on the platform: `python core/reforecast.py --inspect` (columns, variable names, points). If the points are
  not the district codes, give a csv `point,coato_code` as `wx.point_map`.
- Run: `python evaluate.py --model xgboost_wx --mode validate --sample --weather aifs --test-set configs/test_set_2025.yaml`
  (`--weather` must equal `wx.source`). Only the issue dates with a forecast are scored (4 a month).
- Same-dates comparison without weather: `python evaluate.py --model xgboost --mode validate --sample --issue-dates
  reforecast --test-set configs/test_set_2025.yaml`.
- The reforecasts start in 2024: on the standard split the 2024-01 refit has no forecast windows (use the 2025 split,
  or `wx.origins: reforecast+stride`, the default, which keeps v8's usual windows without a forecast as well).
- Saved fits: `<out_dir>/artifacts/_boost_wx/<commit>/<config hash>/refit-<R>`; the daily archive is cached in
  `wx.cache_dir`.
