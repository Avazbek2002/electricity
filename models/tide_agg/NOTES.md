# tide_agg: TiDE on daily and reading-interval labels

**What it tests:** can the pre-2024-08 data be used as interval sums rather than as fake daily values? Before August 2024 most meters were read every 4–5 days in whole kWh. The daily table only has two things for those days:
- allocated long-gap shares (equal splits of 4–92-day intervals);
- 2–3-day estimated runs.

The boosters can only train on those days as if they were daily labels, or treat missing days as residual 0. This model trains on each whole interval's **sum** instead.

**Files:**
- `core/tide_data.py`: interval reconstruction, windows, and the gather shared by training and serving.
- `core/tide.py`: the network, the loss, fit and the predictor.
- `models/tide_daily`: the same model trained on daily labels only. This is the ablation.

## Data

| Store channel | Content |
|---|---|
| kind 1 | `complete_observed` kWh: a daily input and a daily label |
| kind 2 | a day of a reconstructed interval, either a maximal estimated run (exact sum) or an allocated long gap with one covering interval (`long_gap_interval_count` 1), whole inside the read window, with no clean day in it |
| `ilen`, `isum` | the interval's length and exact energy, stored on its last day |

Intervals cut by the window edge, by the Jan–Feb 2023 exclusion or by a multi-meter overlap are dropped. The fit log prints the counts.

## Windows

Each window has an issue date `o`. Inputs are days `o-90 .. o-1` and targets are days `o+1 .. o+60`, as in the harness.

- **Inputs:** observed days. For issue dates before 2024-08-01, also the days of intervals whose closing reading came by `o-1`. An interval still open at `o-1` is masked (HH-GAP-001-A1 rule 3). From 2024-08 the inputs are observed days only, which is exactly what the harness panel carries at serving time.
- **Targets:**
  - daily rows for observed days;
  - interval rows for intervals lying wholly inside the target window;
  - no row for a missing day.
- **Scale:** `s` = the mean of the known input days. Inputs and targets are divided by `s`. The window weight is `s^weight_power`.
- **Validation:** a stable 5 % of customers, never trained on. Their windows issued from `cutoff-84` up to the refit date are used, with partial target windows masked at the refit date.

## Model

TiDE (MLP encoder–decoder):
- Inputs: the past 90 scaled days, the mask and the coarse-day flag, plus per-day covariates for the 150 days.
- Covariates:
  - calendar: day of week, day-of-year harmonics, standing holidays, Ramadan;
  - district temperature normals, fitted on data ≤ refit−1 and saved;
  - past temperature anomalies (zeroed on target days).
- Statics: log s, coverage, coarse share, housing embedding, district embedding.
- A temporal decoder per target day and a linear skip from the past to the outputs.

**Loss (Huber, in scaled units):**
- Daily: `h(ŷ_t − y_t)`.
- Interval: `len · h((Σ_{t∈I} ŷ_t − Y_I)/len)`, the Gaussian weight of a sum of `len` daily errors.

Training uses AdamW with plateau LR halving, early stopping on the validation loss, and bf16 on the GPU.

## Synthetic check (tests/test_tide_node.py data, 240 households, 12 small epochs, CPU)

| | val household WAPE | val bias |
|---|---|---|
| tide_agg (interval labels) | 0.275 | +1.4 % |
| tide_daily (daily only) | 0.293 | +12.8 % |

This is only a smoke test of the mechanism, not evidence for the real data.

## Run (GPU pod; torch must be installed: `pip install torch`)

```bash
python scripts/test_set/build_node_test_set.py --workers 12          # once: ~/work/test_set/ss-2026-10-25-v2/
bash scripts/run_node_models.sh                                       # tide_agg, tide_daily, xgboost, nlinear_x on dev
```
