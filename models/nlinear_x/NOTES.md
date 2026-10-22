# nlinear_x

One linear layer from the last 90 days (as ratios to the 7-day level) + 81 weather, calendar and year-ago inputs to the
next 60 days, fitted on every training window by weighted ridge (core/household_nlinear.py). Report: Household NLinear
Report (All Customers).

- fit: runs `core/household_nlinear.py --test-sample 0 --end <cutoff + 60 days>` (train only), or copies a matching
  pretrained run (`pretrained` in config.yaml).
- predict: the same features as training (core/whole_pipeline_v8.build_features with the harness's weather context),
  then W; level <= 0.2 kWh/day -> persistence.
- Known redundancy: 11 inputs are exact copies (fc_anom_tail, d_hdd_*, d_cdd_*); see the report.

## Refits without validation windows (2024)

Household labels are almost absent from May to mid-August 2024, so no training window with an origin after about
February 2024 has 54 of its 60 target days observed. The refits of 2024-07-01 and 2024-10-01 therefore add no new
windows and have an empty inner validation block. They reuse the penalty chosen at the latest earlier refit
(`--fallback-penalty`, recorded as `penalty_fallback` in nlinear_info.json) instead of failing.
