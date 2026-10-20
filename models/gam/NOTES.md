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
