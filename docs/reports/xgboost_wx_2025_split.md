# XGBoost with the real weather forecast (`xgboost_wx`): 2025 split

**2026-10-23 · Avazbek Isroilov** · run at commit [`32ce082`][c-run] on the B200 GPU pod · [W&B run `ca9iwhj7`](https://forge.coreweave.com/wandb/avazbekisroilov2002-ministry-of-digital-technologies/electricity-forecast/runs/ca9iwhj7)
· companion report: [no-weather models on the same split](no_weather_models_2025_split.md)

This is the same v8 XGBoost as in the no-weather report: the same thermal baseline plus a residual booster. The one difference is the weather for target days h1–h14. Here it comes from the **real AIFS forecast issued on the issue date**, at training and at predict time, instead of climate normals.

**Result:** on the 24 validation issue dates that have a forecast, national sysMAPE is 5.98 / 7.21 / 8.12 % at h1 / h7 / h14. That is 4.2–4.5 points better than B0 and well inside the ТЗ limits. h30 is the weak spot: 12.64 %, 1 point worse than B0.

Code links point to commit `32ce082`. Later commits changed the tree caps (`e9c5504`), so line numbers on later commits may differ.

| National sysMAPE (bias) | h1 | h7 | h14 | h30 | h60 |
|---|---|---|---|---|---|
| **xgboost_wx** | **5.98%** (−1.5) | 7.21% (−3.0) | 8.12% (−0.4) | 12.64% (−5.6) | 11.55% (−7.4) |
| xgboost_wx, unscaled (no c(o)) | 5.93% (+0.9) | **6.85%** (−0.8) | **8.05%** (+1.7) | 12.55% (−3.7) | **10.96%** (−5.6) |
| B0, same days | 10.52% (+2.6) | 11.57% (+2.5) | 12.33% (+4.2) | **11.68%** (−1.0) | 12.90% (+0.3) |
| ТЗ limit | 8% | 11% | 15% | 18% | 20% |
| Days scored | 24 | 23 | 22 | 20 | 17 |

**How to reproduce** (entry point [`evaluate.py`][evaluate], arguments in [`harness/cli.py`][cli]):

```bash
python core/reforecast.py --inspect                       # once: check columns, variables and points of the archive
python scripts/test_set/build_test_set.py --spec configs/test_set_2025.yaml --sample
python evaluate.py --model xgboost_wx --mode validate --sample --weather aifs --test-set configs/test_set_2025.yaml
# the same 24 dates without weather (the fair comparison):
python evaluate.py --model xgboost --mode validate --sample --issue-dates reforecast --test-set configs/test_set_2025.yaml
```

---

## 1. Data split

### 1.1 Calendar

The split is the supplementary `ts-2026-10-22-v2-s25` ([`configs/test_set_2025.yaml`][spec-s25]). The forecast archive has two sets: `wx-val2024` (2024 inits) and `wx-test2025` (2025 inits). The 2026 reserve set is never read ([`wx_config`][rf-wxconfig]).

```mermaid
gantt
    title Split and forecast availability (issue dates)
    dateFormat YYYY-MM-DD
    axisFormat %Y-%m
    section Household data
    history used for training (normals only)   :done, 2022-01-01, 2023-12-31
    history + real forecasts (wx-val2024)      :done, 2024-01-01, 2024-12-31
    section Refit 2025-01-01
    training targets ≤ 2024-12-31              :active, 2022-01-01, 2024-12-31
    section Refit 2025-04-01
    training targets ≤ 2025-03-31 (adds 2025-Q1 inits) :active, 2022-01-01, 2025-03-31
    section Validation (this report)
    24 issue dates with a forecast, targets ≤ 06-30 :crit, 2025-01-01, 2025-06-30
    section Test (not touched)
    2025-H2                                    :2025-07-01, 2025-12-31
    section Reserve
    2026 (never read)                          :2026-01-01, 2026-06-29
```

### 1.2 Training windows per refit

The forecasts exist on only 4 init dates a month (the 1st, 9th, 17th and 25th). With `origins: reforecast+stride` ([`xgboost_wx/config.yaml`][cfg]), each refit trains on two kinds of window:

- windows whose issue date is an init date, which have the real forecast;
- v8's usual staggered windows, which have no forecast. On these the forecast features are NaN and the target period uses normals. They keep 2022–2023 in training.

The windows are built by [`fixed_origins`][wx-fixed] and [`origin_grid`][wx-grid] in `core/wx_windows.py`. A refit is refused if fewer than 20,000 windows have a forecast ([`train`][bw-train]).

```mermaid
pie showData
    title Refit 2025-01-01: 842,685 training windows
    "with the real AIFS forecast" : 460783
    "staggered, normals only" : 381902
```

| Refit | Training windows | With a real forecast | Trees (XGBoost) |
|---|---|---|---|
| 2025-01-01 | 842,685 | 460,783 (55 %) | early-stopped at 1,129 |
| 2025-04-01 | not in this log | 925,354 | not in this log |

The other training rules are the same as in the no-weather run:
- a refit at R sees only target days ≤ R − 1;
- 5 % of customers, then 20 % of their windows;
- windows need ≥ 30 of 60 target days observed;
- early stopping is tuned on the last 12 weeks of training issue dates.

### 1.3 Validation and test

```mermaid
flowchart LR
    A["Validation 2025-H1<br/>181 issue dates"] --> B{"forecast issued<br/>on that date?"}
    B -- "yes: 24 dates" --> C["scored<br/>6,100 district rows"]
    B -- "no: 157 dates" --> D["dropped for every model<br/>39,050 of 45,150 rows"]
    C --> E["xgboost_wx, unscaled and B0<br/>on exactly the same rows"]
```

Only issue dates with a forecast are scored ([`_reforecast_blocks`][tm-blocks]). Targets must stay ≤ 2025-06-30, so the count falls with the horizon: 24 dates at h1, 17 at h60. The test half (2025-H2) was not run: this was a validation run.

---

## 2. Features

### 2.1 How a forecast enters one window

The archive is 6-hourly, in UTC, for each district. [`to_daily`][rf-daily] turns it into Tashkent-local days using the 00/06/12/18 UTC steps of each day:
- `t_fc` is the daily mean;
- `t_fc_min` and `t_fc_max` are the daily min and max;
- °C conversion from Kelvin is automatic.

[`to_window_convention`][rf-conv] then maps issue o and lead k to v8's window origin o − 1. Window lead 1 (day o) stays empty, exactly as at predict time ([`grid_at_last_day`][w-last]).

```mermaid
flowchart LR
    subgraph H["history: 90 days ≤ o − 1"]
      h1["observed kWh"]
      h2["observed weather"]
    end
    subgraph F["target days h1 … h14"]
      f1["real AIFS forecast<br/>issued on o (00 UTC)"]
    end
    subgraph T["target days h15 … h60"]
      t1["normal + fading anomaly<br/>tail = mean anomaly of the last 3 forecast days<br/>× exp(−k / 5 days)"]
    end
    H --> o(("issue date o")) --> F --> T
```

After the forecast ends, the blend in [`blend_forecast`][v8-blend] fades the anomaly away: it has about 5 % of its size left by h30. The blended temperature feeds the thermal baseline B(h) and the target-period block features, and the forecast also adds its own 34 features.

### 2.2 Feature set: 78 + 34 = 112

```mermaid
pie showData
    title 112 features of xgboost_wx
    "Consumption history" : 16
    "Past weather" : 11
    "Household sensitivity" : 8
    "Target-period weather blocks (now forecast-blended)" : 25
    "Last-year anomaly" : 5
    "Baseline level" : 7
    "Calendar and regime" : 3
    "Categorical" : 3
    "Forecast anomaly blocks (new)" : 4
    "Forecast daily HDD-min / CDD-max (new)" : 30
```

```mermaid
flowchart LR
    subgraph OLD["78 features of the no-weather model"]
      a["history, past weather, sensitivity,<br/>last-year, level, calendar, categorical · 53"]
      b["target-period blocks · 25<br/>normal → forecast-blended for h1–h14"]
    end
    subgraph NEW["34 forecast features"]
      c["fc_anom_2_4, fc_anom_5_8, fc_anom_9_15<br/>forecast − normal for h1–3, h4–7, h8–14"]
      d["fc_anom_tail<br/>the anomaly that is faded out"]
      e["fc_hddmin_d01…d15 · fc_cddmax_d01…d15<br/>cold signal from the forecast min, hot signal from the max, per day"]
    end
    OLD --> X(("112"))
    NEW --> X
    X --> M["XGBoost multi-output<br/>residual r(h), 60 outputs"]
    BL["thermal baseline B(h)<br/>level + βh·ΔHDD + βc·ΔCDD<br/>using the blended temperature"] --> OUT
    M --> OUT["forecast = B(h) + r(h)<br/>× district factor c(o)"]
```

| New feature | Count | What it is | Code |
|---|---|---|---|
| `fc_anom_2_4`, `fc_anom_5_8`, `fc_anom_9_15` | 3 | Mean of forecast minus normal over h1–3, h4–7 and h8–14 | [`build_features`][v8-anom], blocks from [`v8_forecast_settings`][bw-fs] |
| `fc_anom_tail` | 1 | Mean anomaly of the last 3 forecast days; this is the part that fades | [`blend_forecast`][v8-blend] |
| `fc_hddmin_d01…d15` | 15 | max(base − forecast Tmin, 0) per day; d01 (day o) is always empty | [`build_features`][v8-mm], [`degree_days_minmax`][v8-ddmm] |
| `fc_cddmax_d01…d15` | 15 | max(forecast Tmax − base, 0) per day; d01 is always empty | [`build_features`][v8-mm] |

On the staggered windows without a forecast, all 34 are NaN and the blocks use normals. XGBoost learns both cases in one model.

### 2.3 Training and serving use the same inputs

```mermaid
sequenceDiagram
    participant S3 as s3://data-weather/reforecast (aifs)
    participant RF as core/reforecast.py
    participant TR as boost_wx_harness.train
    participant PR as boost_wx_harness.Predictor
    S3->>RF: 6-hourly t2m (+ d2m, wind, precip) per district
    RF->>RF: to_daily: local day mean/min/max, leads 1..14
    RF->>TR: archive, shifted to the window convention
    TR->>TR: build_forecast_grid → features → fit (GPU)
    RF->>PR: same archive, the forecast issued on o only
    PR->>PR: reforecast_grid → grid_at_last_day → same features → predict (CPU)
```

Train/serve equality is covered by [`tests/test_reforecast.py`][tests]. The substitution (leakage) test passed: for district 03203 with T = 2025-02-09, forecasts issued ≤ T + 1 changed by 0 ([`substitution_test`][tm-subst]).

---

## 3. Evaluation method

| Step | Rule | Code |
|---|---|---|
| Target | District T2 per day (observed + estimated kWh, scaled to all rows); flagged days are not scored | [`configs/test_set_2025.yaml`][spec-s25] |
| Rows | Common panel: the rows every compared model has; B0 is always on the same rows | [`common_panel`][common-panel] |
| National | Sum of districts for each (o, h), kept when ≥ 98 % of the run's districts are present | [`score`][score] |
| sysMAPE | Σ\|P − A\| / ΣA, pooled over all scored issue dates, per horizon | [`_sm`][sm] |
| Bias | Σ(P − A) / ΣA; below 0 = too low | [`_bias`][bias] |
| Reference | B0 = 28-day level × last year's seasonal ratio | [`b0_forecasts`][b0] |
| Scaled vs unscaled | Scaled = model × district factor c(o); unscaled = raw bottom-up sum | [`harness/test_mode.py`][test-mode] |
| Network levels | Household → substation, bottom-up over the observed days of the forecast households (supplementary) | [`write_levels`][levels], [`write_network`][network] |

---

## 4. Results

### 4.1 sysMAPE by horizon

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2563eb, #93c5fd, #f97316, #9ca3af"}}}}%%
xychart-beta
    title "National sysMAPE, % (validation 2025-H1, 24 forecast dates, 5 districts)"
    x-axis [h1, h7, h14, h30, h60]
    y-axis "sysMAPE %" 0 --> 21
    line "xgboost_wx" [5.98, 7.21, 8.12, 12.64, 11.55]
    line "xgboost_wx unscaled" [5.93, 6.85, 8.05, 12.55, 10.96]
    line "B0" [10.52, 11.57, 12.33, 11.68, 12.90]
    line "ТЗ limit" [8, 11, 15, 18, 20]
```

**Blue** = xgboost_wx, **light blue** = unscaled, **orange** = B0, **grey** = ТЗ limit. Each run also writes the full 1..60 horizon chart ([`plots.metrics_by_horizon`][plots]).

### 4.2 Gap to B0 on the same days

The gap is in percentage points; below 0 means better than B0.

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2563eb"}}}}%%
xychart-beta
    title "xgboost_wx minus B0, points (below 0 = better)"
    x-axis [h1, h7, h14, h30, h60]
    y-axis "points" -5 --> 2
    bar [-4.54, -4.36, -4.21, 0.96, -1.35]
```

| Points vs B0 | h1 | h7 | h14 | h30 | h60 |
|---|---|---|---|---|---|
| xgboost_wx | −4.54 | −4.36 | −4.21 | +0.96 | −1.35 |
| xgboost_wx unscaled | −4.59 | −4.72 | −4.28 | +0.87 | −1.94 |

### 4.3 Bias

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2563eb, #93c5fd, #f97316"}}}}%%
xychart-beta
    title "National bias, % (below 0 = forecast too low)"
    x-axis [h1, h7, h14, h30, h60]
    y-axis "bias %" -8 --> 5
    line "xgboost_wx" [-1.5, -3.0, -0.4, -5.6, -7.4]
    line "unscaled" [0.9, -0.8, 1.7, -3.7, -5.6]
    line "B0" [2.6, 2.5, 4.2, -1.0, 0.3]
```

**Blue** = xgboost_wx, **light blue** = unscaled, **orange** = B0.

The c(o) scaling pushes the bias down by about 2 points at every horizon, so the unscaled forecast is better everywhere. The growing under-forecast after h14 is the same pattern the no-weather boosters show.

### 4.4 Against the no-weather XGBoost (proxy)

The no-weather run was scored on all 181 dates and this run on 24, so B0 differs between them. The fair comparison is to divide each model's error by B0 on its own dates. The exact head-to-head is the `--issue-dates reforecast` run, which is still pending.

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2563eb, #f97316"}}}}%%
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2563eb, #7c3aed"}}}}%%
xychart-beta
    title "Model sysMAPE ÷ B0 sysMAPE on its own dates (below 1 = better than B0)"
    x-axis [h1, h7, h14, h30, h60]
    y-axis "ratio" 0.5 --> 1.15
    line "xgboost_wx (24 dates)" [0.57, 0.62, 0.66, 1.08, 0.90]
    line "xgboost, no weather (181 dates)" [0.66, 0.63, 0.63, 0.79, 1.10]
```

**Blue** = xgboost_wx (24 dates), **purple** = no-weather xgboost (181 dates).

| ÷ B0 | h1 | h7 | h14 | h30 | h60 |
|---|---|---|---|---|---|
| xgboost_wx | **0.57** | **0.62** | 0.66 | 1.08 | **0.90** |
| xgboost, no weather (150 trees) | 0.66 | 0.63 | **0.63** | **0.79** | 1.10 |

On this proxy the forecast clearly helps at h1, and h60 also looks better. h7–h14 are about even and h30 is worse. With only 20 dates at h30, one bad week can explain that. The two runs also differ in tree caps (1,500 vs 150), so this is not yet a clean weather effect.

### 4.5 District level

District-level sysMAPE is pooled over district rows.

```mermaid
xychart-beta
    title "District-level sysMAPE, %"
    x-axis [h1, h7, h14, h30, h60]
    y-axis "sysMAPE %" 0 --> 18
    bar [8.25, 10.51, 10.98, 16.62, 17.22]
    line [12.50, 13.72, 15.16, 15.42, 17.07]
```

Blue bars = xgboost_wx, orange line = B0. xgboost_wx is better than B0 at district level up to h14; at h30–h60 the two are about level.

### 4.6 Network levels

WAPE is computed bottom-up from households, with bias in brackets.

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#2563eb, #16a34a, #dc2626"}}}}%%
xychart-beta
    title "WAPE by network level, % (xgboost_wx, validation)"
    x-axis [household, transformer, feeder, substation, district, national]
    y-axis "WAPE %" 0 --> 55
    line "h1" [33.52, 10.04, 8.49, 8.13, 8.25, 5.98]
    line "h14" [41.10, 12.78, 10.93, 10.48, 10.98, 8.12]
    line "h60" [52.28, 19.20, 17.37, 16.80, 17.22, 11.55]
```

**Blue** = h1, **green** = h14, **red** = h60.

| WAPE (bias) | h1 | h7 | h14 | h30 | h60 |
|---|---|---|---|---|---|
| household | 33.52% (−1.5) | 37.83% (−1.7) | 41.10% (+0.6) | 46.18% (−3.4) | 52.28% (−5.1) |
| transformer | 10.04% (−1.3) | 12.17% (−1.1) | 12.78% (+1.1) | 18.16% (−2.9) | 19.20% (−3.5) |
| feeder | 8.49% (−1.5) | 10.53% (−1.7) | 10.93% (+0.7) | 16.40% (−3.5) | 17.37% (−5.0) |
| substation | 8.13% (−1.5) | 10.25% (−1.7) | 10.48% (+0.6) | 16.11% (−3.5) | 16.80% (−5.1) |
| district | 8.25% (−1.5) | 10.51% (−3.0) | 10.98% (−0.4) | 16.62% (−5.6) | 17.22% (−7.4) |
| national | 5.98% (−1.5) | 7.21% (−3.0) | 8.12% (−0.4) | 12.64% (−5.6) | 11.55% (−7.4) |

| MAE, kWh per node-day | h1 | h7 | h14 | h30 | h60 |
|---|---|---|---|---|---|
| household | 2.6 | 2.9 | 3.0 | 3.4 | 3.6 |
| transformer | 51.9 | 61.2 | 61.8 | 88.7 | 86.5 |
| feeder | 388 | 465 | 478 | 722 | 696 |
| substation | 1,364 | 1,642 | 1,709 | 2,641 | 2,448 |
| district | 28,978 | 36,969 | 37,753 | 57,193 | 55,923 |
| national | 105,080 | 126,855 | 139,490 | 217,569 | 187,468 |

Errors cancel out quickly once households are summed: 33.5 % at household level falls to about 10 % at transformer level and 8 % at substation level (h1).

---

## 5. Caveats and next steps

| Caveat | Effect |
|---|---|
| 24 scored dates (17 at h60) | Noisy: h30 / h60 can move by a point on one bad week |
| 5-district sample, supplementary split | A national run and the standard split are needed before conclusions |
| Commit `32ce082`: final cap 1,500 trees | XGBoost stopped at 1,129 trees, so the cap was not binding; CatBoost was capped (best 2,913) |
| No same-dates no-weather run yet | §4.4 is a proxy; the weather effect is not yet isolated |
| c(o) scaling hurts here | Check the unscaled forecast on the no-weather runs too before dropping c(o) |
| `housing_type_id` snapshot, `loyo` normals | Same possible leakage as the no-weather models |

- [ ] `xgboost --issue-dates reforecast` on the same 24 dates (the clean weather effect)
- [ ] Retrain on `e9c5504` (GPU, up to 4,000 trees) and rerun both
- [ ] `catboost_wx` (shares the fit) and an ensemble with NLinear_x on the same dates
- [ ] Try `--weather fcn3` and `ens`
- [ ] Standard split (2024 validation, 48 forecast dates) and a national run

## Scripts used

| Step | Script |
|---|---|
| Entry point | [`evaluate.py`][evaluate] → [`harness/cli.py`][cli] (`--weather aifs`, `--issue-dates`) → [`harness/test_mode.py`][test-mode] |
| Model | [`models/xgboost_wx/model.py`][model], [`config.yaml`][cfg], [`NOTES.md`][notes] |
| Forecast archive | [`core/reforecast.py`][rf] ([`to_daily`][rf-daily], [`load_archive`][rf-load], [`to_window_convention`][rf-conv]) |
| Training windows | [`core/wx_windows.py`][wx] ([`fixed_origins`][wx-fixed], [`district_worker`][wx-worker]) |
| Fit and predict | [`core/boost_wx_harness.py`][bw] ([`train`][bw-train], [`Predictor`][bw-pred]) → [`core/whole_pipeline_v8.py`][v8] ([`build_features`][v8-feat], [`fit_and_evaluate`][v8-fit]) |
| Forecast at predict time | [`harness/weather.py`][weather] ([`reforecast_grid`][w-grid], [`grid_at_last_day`][w-last]) |
| Scoring, B0 | [`harness/testset.py`][testset] |
| Tests | [`tests/test_reforecast.py`][tests] |
| GPU check | [`scripts/gpu_check.py`][gpu] |

[c-run]: https://github.com/Avazbek2002/electricity/commit/32ce0827c0a287e5218b1891290993e98535a717
[evaluate]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/evaluate.py
[cli]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/cli.py#L30
[test-mode]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/test_mode.py
[tm-blocks]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/test_mode.py#L294
[tm-subst]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/test_mode.py#L275
[levels]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/test_mode.py#L228
[network]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/test_mode.py#L248
[spec-s25]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/configs/test_set_2025.yaml
[model]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/models/xgboost_wx/model.py
[cfg]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/models/xgboost_wx/config.yaml
[notes]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/models/xgboost_wx/NOTES.md
[rf]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/reforecast.py
[rf-wxconfig]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/reforecast.py#L60
[rf-daily]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/reforecast.py#L134
[rf-load]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/reforecast.py#L236
[rf-conv]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/reforecast.py#L259
[wx]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/wx_windows.py
[wx-fixed]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/wx_windows.py#L24
[wx-grid]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/wx_windows.py#L43
[wx-worker]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/wx_windows.py#L58
[bw]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/boost_wx_harness.py
[bw-fs]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/boost_wx_harness.py#L49
[bw-train]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/boost_wx_harness.py#L75
[bw-pred]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/boost_wx_harness.py#L163
[v8]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/whole_pipeline_v8.py
[v8-blend]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/whole_pipeline_v8.py#L1392
[v8-ddmm]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/whole_pipeline_v8.py#L1302
[v8-feat]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/whole_pipeline_v8.py#L1928
[v8-anom]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/whole_pipeline_v8.py#L2064
[v8-mm]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/whole_pipeline_v8.py#L2071
[v8-fit]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/core/whole_pipeline_v8.py#L2327
[weather]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/weather.py
[w-grid]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/weather.py#L65
[w-last]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/weather.py#L76
[testset]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/testset.py
[b0]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/testset.py#L192
[common-panel]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/testset.py#L276
[sm]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/testset.py#L312
[bias]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/testset.py#L320
[score]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/testset.py#L324
[plots]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/harness/plots.py#L69
[tests]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/tests/test_reforecast.py
[gpu]: https://github.com/Avazbek2002/electricity/blob/32ce0827c0a287e5218b1891290993e98535a717/scripts/gpu_check.py
