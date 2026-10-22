# gam

Generalised additive model fitted as an Explainable Boosting Machine (interpret): one smooth curve per input plus a few
two-input curves, shared by all households. Target = consumption / the household's 7-day level, weighted by the level.
Inputs and their definitions: `gam_features.py` (one builder for fit and predict).

- Training rows: 0.5% of households, an origin every 28 days up to the cutoff, horizons 1, 2, 3, 5, 7, 10, 14, 21, 30,
  45, 60 with an observed target; capped at 4 M rows.
- Prediction: the same 11 horizons, linear in between for the other days.
- Weather: raw target-day temperature (forecast days 1-10, then normal + a fading anomaly); the model learns the heating
  and cooling curve itself, so no degree-day bases.
- Not yet: the official holiday calendar.

## 2026-10-22: January fix, weekday pattern, weather

- The target day of year (`doy`, `doy & h`) is gone: the 2024-01-01 refit had never seen a January-February target
  (data from 2022-01, 90 days of history, the Jan-Feb 2023 crisis excluded), and forecast ~1/5 of January 2024.
  Seasonality now comes from `nrm_dev` (the target day's climate normal minus the mean normal of the last 30 days),
  the temperature path and last year's ratio.
- Trained and predicted on every horizon 1..60 (`horizons: all`), so the weekday pattern of each target day is
  learned; `horizons: knots` restores the old 11 knots with straight lines in between.
- `forecast: none` (default): trained on normals only, as the standard run predicts; `simulated` needs
  `--weather simulated`.
