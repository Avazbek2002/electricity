"""
Feature engineering v8 + CatBoost / XGBoost training for household
electricity forecasting.

Starts from the clean daily table + window index written by the previous
script (process_raw.py; CLEAN_STORE_S3). The 90-day input / 60-day target arrays are
rebuilt here from origin_date + the daily table, so no window is ever stored twice.
No resampling, no re-windowing. (WINDOW_SOURCE="legacy" still reads the old saved
window arrays from CLEAN_WINDOWS_S3.)

Everything from v3 is unchanged: regime cutoff, causal glitch repair
(clean_daily_series is still the single train/serve cleaning function),
data-fitted beta priors, regularised CatBoost/XGBoost with a capped final
tree count, early stopping on a time-based validation slice, evaluation
against RAW targets, MAE + system-level MAPE.

What changed versus v3
-----------------------
1. 10-day weather forecast as a look-ahead (FORECAST_MODE).
   Days 1..FORECAST_DAYS of the horizon use a real (or simulated, see
   below) temperature forecast instead of the calendar normal. Beyond day
   10 the horizon is still the calendar normal, plus the mean forecast
   anomaly over the last FC_TAIL_DAYS forecast days, fading out
   exponentially (FC_FADE_TAU) so there is no step at day 10.
   The thermal baseline (`base`) and every fut_* / d_* feature are built
   from this blended series, so the forecast flows through them.
   New features: fc_anom_{1_3,4_7,8_10} (forecast minus normal, per
   sub-block) and fc_anom_tail (the quantity that is faded beyond day 10).

   FORECAST_MODE:
     "archive"   - forecasts as ISSUED on each origin date, read from
                   FORECAST_S3. This is the correct way to train. See
                   load_archived_forecasts for the expected schema.
     "simulated" - observed weather + lead-dependent, autocorrelated
                   noise, drawn ONCE per (coato, origin_date) so every
                   meter in the same coato sees the same forecast error
                   (as it would live). Use until an archive is available.
                   Noise is independent ACROSS coatos, which is optimistic
                   if the coatos are close together.
     "none"      - v3 behaviour (normal only).
   FUTURE_MODE="actual" is still the oracle ceiling and ignores forecasts.

2. Previous-year anomaly features (USE_PREV_YEAR_FEATURES).
   prev_anom_{block} = mean over each BLOCK of
   (observed temperature on the same calendar days one year earlier -
   calendar normal). The normal remains the basis for `base` and fut_*;
   these only let the trees test whether last year's deviation carries any
   signal. Strictly causal: the prior-year dates are always before origin.
   NaN where there is no prior-year data (both libraries handle NaN).

3. Reports now include t+14, the horizon that was failing the acceptance
   check.

4. Behavior-segment-specific priors and a new "segment" feature (v4.1).
   A standalone analysis clustered meters into two behavioral segments -
   "weather_responsive_winter_peaked" and "stable_baseload_low_seasonality"
   - from scale, thermal responsiveness, and seasonality strength, computed
   over each meter's full training-period history (not the 90-day window
   fit_sensitivity uses per row). That segmentation is reproduced here at
   a reduced scope (fixed HDD_BASE/CDD_BASE breakpoints, k fixed at 2, not
   re-swept) rather than replicating the richer exposure-gated breakpoint
   search from the standalone analysis - this is a deliberate simplification
   to keep the production pipeline tractable, not a claim that the numbers
   match that analysis exactly. See build_meter_segment_features,
   assign_meter_segments, estimate_segment_beta_priors.
     - BETA_H_PRIOR/BETA_C_PRIOR shrinkage targets are now segment-specific
       (estimate_segment_beta_priors) instead of one pooled value for every
       meter, computed causally from training-period rows only. A meter
       with too little history to segment, or whose segment's own fit
       lacks signal, falls back to the pooled (all-meter) prior - the same
       fallback behaviour estimate_beta_priors already had, now nested one
       level deeper.
     - CAT_FEATURES gained "segment" (meter_id -> segment label, joined
       into `meta` by build_features via a new meter_segment parameter).
       Meters with insufficient causal history are labelled "unclassified"
       rather than dropped.
   estimate_beta_priors() (the single global prior) is left in place for
   reference and as the fallback this feature nests inside of.

5. Daily min/max temperature: forecast extremes and min/max sensitivity (v6).
   The weather table is hourly, so the daily min and max are aggregated next
   to the daily mean (load_daily_temperature) and get their own calendar
   normals (build_climate). Everything below is switched by USE_MINMAX;
   USE_MINMAX=False reproduces v5.
     - Forecast extremes, single numbers over the 10 forecast days:
       fc_tmin_coldest (lowest forecast daily min), fc_tmax_hottest (highest
       forecast daily max), and their anomalies versus the coldest normal min
       / hottest normal max over the same 10 days (fc_tmin_coldest_anom,
       fc_tmax_hottest_anom).
     - Min/max sensitivity: a second per-row ridge fit on the input window,
           y = a + beta_h_min * HDD(Tmin) + beta_c_max * CDD(Tmax)
       with HDD(Tmin) = max(HDD_MIN_BASE - Tmin, 0) and
       CDD(Tmax) = max(Tmax - CDD_MAX_BASE, 0). Priors are per behaviour
       segment and estimated on these min/max degree days
       (estimate_segment_beta_priors, kind="minmax"); the mean-based priors do
       not transfer because the degree-day magnitudes differ. New features:
       beta_h_min, beta_c_max, base_load_fit_mm, sens_r2_mm.
     - The thermal baseline now uses the min/max sensitivities and the blended
       future min/max series (forecast for days 1-10, then the min/max normals
       plus a fading forecast anomaly), replacing the mean-based baseline.
       The mean-based features (beta_h, beta_c, fut_*, d_*, fc_anom_*) stay.
   Simulated forecast: min and max share the mean's forecast error and add an
   independent AR(1) component (FC_MINMAX_EXTRA_FRAC). That error model is an
   assumption, not calibrated. Archive mode needs t_fc_min / t_fc_max columns.

6. Forecast min/max as per-day degree days (v7). The four v6 forecast-extreme
   features (fc_tmin_coldest, fc_tmax_hottest and their anomalies) are replaced
   by one value per forecast day and per variable:
       fc_hddmin_d01 .. fc_hddmin_d10 = max(HDD_MIN_BASE - forecast Tmin, 0)   cold-day signal
       fc_cddmax_d01 .. fc_cddmax_d10 = max(forecast Tmax - CDD_MAX_BASE, 0)   hot-day signal
   (20 features instead of 4, so 101 in total). The sign is kept on purpose: a
   score like |Tmin - 20| + |Tmax - 26.5| gives a cool day (12/19 -> 15.5) and a
   hot day (26/38 -> 17.5) nearly the same value although they push consumption
   in opposite directions. Each term is 0 when the weather is comfortable. NaN
   where the forecast for that day is missing (both libraries handle NaN).
   Only days 1-10 exist (the forecast span); the thermal baseline is unchanged.

7. Old-regime data (v8). Everything before REGIME_CUTOFF (2024-08-01) used to be dropped. It is
   now usable, under switches, because a same-meter analysis showed that
     - about 20% of its days are "smeared" (a whole-kWh total from a missed read spread evenly
       over the gap) in nearly EVERY meter, so there is no clean subset; run totals stay right;
     - same-meter levels agree with the new regime in autumn (ratio ~1.0), but are 7-15% HIGHER
       in Dec-Apr for reasons that are not weather. Old winters are therefore the risky part;
     - the old data has holes: none at all 2023-01-03..01-21 and almost none through February
       2023, and almost none from May to mid-August 2024.
   Settings (see "old-regime data" below):
     EXCLUDE_RANGES    calendar days removed ENTIRELY, default 2023-01-01..2023-02-28. In
                       EXCLUDE_MODE="span" a window is dropped if any of its 150 days (90 input +
                       60 target) falls in the range, so no excluded day reaches a feature, a
                       target, the segmentation, the priors or the glitch repair.
     INCLUDE_OLD       "none" (v7 behaviour) | "all" | "non_winter" (old windows whose 60-day
                       target touches none of OLD_EXCLUDE_TARGET_MONTHS, default Dec-Apr)
     OLD_WEIGHT        training sample weight of old rows (new rows 1.0)
     OLD_REGIME_FLAG   adds the feature regime_old (1 for old rows, 0 for new; always 0 at serving)
     SEGMENT_USE_OLD   False: behaviour segments and beta priors come from new-regime windows only,
                       exactly as in v7, so the effect of the old rows is not confounded with them
   Validation and test are unchanged (new regime, origin >= 2025-07 / 2025-11), so the numbers are
   comparable with v7. Old rows only ever enter the fit / train sets.
   Reports: besides sysMAPE, sys_report() prints the acceptance-spec WAPE, the signed bias and the
   number of target days / meters per day at h = 1, 7, 14, 30, 60.

8. Windows rebuilt from a daily table + index (this edit). process_raw.py no longer saves the
   150-day windows (each calendar day sat in ~5 of them). It saves
     daily_<tag>/batch=NNN   one row per meter and day (daily_kwh, flatlines nulled)
     index_<tag>/batch=NNN   one row per candidate (meter_id, origin_date) with the filter counts
                             and `passes` (= the old three window rules)
   extract_windows() reads the index rows with passes = True, drops the windows the regime policy
   removes BEFORE touching any kWh (so INCLUDE_OLD="none" never loads old-regime data), then fills
   kin (N x 90) and ktg (N x 60) from a dense per-meter copy of the daily table, one hash batch at a
   time. Everything downstream (glitch repair, segments, features, models) is unchanged and sees the
   same arrays as before; the saving is on S3 (about 5x less data) and in the producer's shuffle.
   WINDOW_BATCHES picks a subset of the hash batches (a clean random subset of the meters) when the
   full store is too big for memory: kin + ktg take N x 150 x 8 bytes, and the glitch step briefly
   holds about three copies.
   Because the index keeps the COUNTS, not just the verdict, windows that failed a rule can be
   inspected or admitted later without re-running the raw data.

9. National ENRICHED household dataset (WINDOW_SOURCE="enriched", the default; see HOUSEHOLD_NATIONAL_DATASETS_GUIDE.md).
   Reads the cleaned v1.4 staging dataset with PyArrow, one finished district at a time; no Spark (weather comes from the
   same table). The unit is a customer (customer_key + coato_code). ENRICHED_BUCKETS picks a stable customer_bucket sample.
     - labels: only complete_observed days have a value (HH-GAP-001-A1 rules 1-3); everything else is NaN
     - windows: origins every WINDOW_STRIDE days per customer (staggered); input >= MIN_INPUT_DAYS of 90 observed and
       >= MIN_RECENT_DAYS of the last RECENT_WINDOW; target >= MIN_TARGET_DAYS observed (scored on observed days only);
       trained on when >= TRAIN_MIN_TARGET_DAYS observed (a missing target day = residual 0, never an invented value)
     - GLITCH_REPAIR off: the data is already cleaned, values are used as delivered (training and serving alike)
     - validation = the autumn-to-spring block (VAL_START, VAL_END]; BACKTEST_CUTOFFS entries may be (cutoff, val_start, val_end)
     - columns.json records the input rule, so transformer_forecast.py's bottom-up stage applies the same rule
   Run:   python whole_pipeline_v8.py --set ENRICHED_BUCKETS=[0,1] --set OUTPUT_DIR=./artifacts/hh_enriched train

10. All customers (national scale). ENRICHED_BUCKETS=None reads every customer of every finished district,
   ENRICHED_WORKERS districts at a time in separate processes. Holding every window would need ~150 GB and days of
   CatBoost (MultiRMSE over 60 outputs), so ENRICHED_TRAIN_SAMPLE / ENRICHED_TEST_SAMPLE keep a stable random share of
   the trainable / test windows of EVERY customer while reading (nothing else is kept in memory). With sampling, each
   customer's segment features come from its whole daily history up to TRAIN_CUTOFF (a few sampled windows are too little
   history), and features are built in FEATURE_CHUNK_ROWS chunks (identical result). ENRICHED_CACHE_DIR saves each
   district's result, so a stopped run resumes where it stopped (the cache holds customer keys: keep it inside the
   authorised environment). Sampling covers the primary fold only (BACKTEST_CUTOFFS must be empty).
   Sizes from the 1% run: ~27M trainable windows nationally; a train share of 0.075 keeps ~2M (CatBoost ~5 h per fit on
   65 cores), a test share of 0.01 keeps ~350k (as many as the 1% run tested).
   Run:   python whole_pipeline_v8.py --no-mlflow --set ENRICHED_BUCKETS=None --set ENRICHED_TRAIN_SAMPLE=0.075 \
            --set ENRICHED_TEST_SAMPLE=0.01 --set ENRICHED_WORKERS=4 --set ENRICHED_CACHE_DIR="'$HOME/work/enriched_cache'" \
            --set OUTPUT_DIR="'$HOME/work/hh_national'" train

Experiment tracking, serving model, registry, monitoring (MLflow; optional, off unless a tracking URI is set)
   Needs next to this file: mlflow_utils.py (the Tracker wrapper) and electricity_pyfunc.py (the serving model).
   Set MLFLOW_TRACKING_URI (or --mlflow-uri); credentials come from the environment only (MLFLOW_TRACKING_USERNAME /
   _PASSWORD / _TOKEN, AWS_* for the artifact store) and are never logged. MLflow trouble never stops a training
   run: a failing call prints one warning and the job goes on.
     python core/whole_pipeline_v8.py train                      train, log, parity-check, register as 'challenger', gate
     python core/whole_pipeline_v8.py promote [--version N] [--force]    alias 'champion' (the gate must pass)
     python core/whole_pipeline_v8.py monitor --file forecasts.csv       score production forecasts vs actuals
   1. Training run: every constant above as a parameter, data lineage tags (index tag, data-prep run id), regime and
      glitch statistics, test metrics per model and horizon (test/<model>/sysWAPE_t7 ...), the full stdout, and nested
      runs 'catboost' / 'xgboost' with hyper-parameters, learning curves, feature importances.
   2. Data lineage: the run records the index tag of the window store it read and links to the process_raw.py run.
   3. Serving model: electricity_pyfunc.py logged as an MLflow pyfunc together with THIS file as code, the model
      files, the climate and the segment table. Input: raw 90-day histories (+ optional fresh weather / forecast);
      it applies the same cleaning and features as training. A parity check loads the logged model and compares it
      with the in-memory models; the model is registered only if they agree.
   4. Registry: every trained model becomes a version with alias 'challenger'; AUTO_PROMOTE / `promote` move
      'champion' after the gate (sysWAPE at t+7/14/30 not worse by more than GATE_TOLERANCE on the target days both
      models were tested on).
   5. Backtests / monitoring: BACKTEST_CUTOFFS adds rolling-origin folds as nested runs (mean / std / worst in the
      parent); `monitor` logs live sysMAPE / sysWAPE / bias per horizon and the drift against the champion's test numbers.

Ablation (change the two flags below, compare sysMAPE at t+7/14/30/60):
   normal only | + prev-year | + forecast | both | oracle (FUTURE_MODE="actual")
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
try:                                   # Spark is only needed to READ data; serving (the MLflow model) runs without it
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
except ImportError:                    # pragma: no cover
    SparkSession = F = None
try:                                   # optional experiment tracking (mlflow_utils.py next to this file; off unless a URI is set)
    from mlflow_utils import Tracker
except ImportError:
    Tracker = None
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean of empty slices

# =============================================================================
# Configuration
# =============================================================================
# --- where the training windows come from ---
WINDOW_SOURCE = "enriched"     # "enriched" = national enriched household dataset v1.4 (default; no Spark)
                               # "index" = daily table + window index from process_raw.py | "legacy" = saved window arrays

# --- enriched source (WINDOW_SOURCE="enriched"); see HOUSEHOLD_NATIONAL_DATASETS_GUIDE.md ---
ENRICHED_ROOT = "s3://phase1/electricity/_staging/household-enriched-v14-nat-b2c-20261017-8e1414/"
ENRICHED_BUCKETS = [0]         # customer_bucket values (stable hash 0-99): each bucket is ~1% of customers nationally;
                               # None = every customer of the districts read (use with ENRICHED_COATOS)
ENRICHED_COATOS = None         # None = every finished district; or a list of coato codes
ENRICHED_LABELS = ["complete_observed"]   # HH-GAP-001-A1 rule 1: only observed 1-day clean kwh is a label or an input
ENRICHED_START, ENRICHED_END = "2022-01-01", "2026-06-29"
WINDOW_STRIDE = 30             # days between a customer's origins (staggered per customer, so every calendar day is used)
MIN_INPUT_DAYS = 60            # an input window needs this many observed days of its 90 ...
MIN_RECENT_DAYS = 20           # ... and this many of its last RECENT_WINDOW days
RECENT_WINDOW = 30
MIN_TARGET_DAYS = 30           # a window is kept if this many of its 60 target days are observed (scored on those days only)
TRAIN_MIN_TARGET_DAYS = 54     # a window is TRAINED on if this many target days are observed; on its missing days the label is
                               # "no correction to the thermal baseline" (residual 0), never a made-up consumption value
# --- national scale (all customers: ENRICHED_BUCKETS = None) ---
ENRICHED_WORKERS = 4           # districts read in parallel (separate processes; each holds one district's customers x days,
                               # ~1-3 GB for a big city district; the log prints every worker's peak)
ENRICHED_TRAIN_SAMPLE = None   # None = keep every window. A share (e.g. 0.075): keep that random share (stable hash of customer +
                               # origin) of the TRAINABLE windows (origin <= TRAIN_CUTOFF, >= TRAIN_MIN_TARGET_DAYS observed),
                               # and drop the windows nothing uses (untrainable pre-cutoff, the embargo after the cutoff)
ENRICHED_TEST_SAMPLE = None    # None = every test window; a share (e.g. 0.01) = that random share of the test windows
SEGMENT_FROM_FULL_HISTORY = None   # None = automatic (on when sampling): a customer's segment features come from its whole daily
                               # history up to TRAIN_CUTOFF, computed while its district is read, not from its (sampled) windows
ENRICHED_CACHE_DIR = None      # a folder: each district's result is saved there, so a stopped run resumes without re-reading it
                               # (holds customer keys: keep it inside the authorised environment)
FEATURE_CHUNK_ROWS = 500_000   # features are built in chunks of this many windows (memory); the result is identical
PRECOMPUTED_SEGMENT_FEATURES = None   # set at run time: (train_cutoff, DataFrame) from the full-history pass
GLITCH_REPAIR = True           # causal spike/dip repair; switched off for the enriched source (already cleaned by Layer 1)
CLEAN_STORE_S3 = "s3a://phase1/electricity/analytics/features/clean_daily_store"   # holds daily_<tag>/ and index_<tag>/
WINDOW_INDEX_DIR = None        # e.g. "index_nb020_qnone_f4_L90_H60_S30_G5_R30"; None = the only index_* folder for LOOKBACK/HORIZON
WINDOW_BATCHES = [0]          # one hash batch = about 4.9M windows, a random 5% of the meters
MAX_WINDOW_ARRAY_GB = 8.0     # batch 0 needs about 5.4 GB for kin + ktg
CLEAN_WINDOWS_S3 = "s3a://phase1/electricity/analytics/features/clean_windows.parquet"   # legacy format only
WEATHER_S3 = (
    "s3a://phase1/electricity/analytics/reference/coato_hourly_weather/"
    "coato_hourly_weather_2022_2026_master.parquet"
)
CUSTOMER_METADATA_S3 = (
    "s3a://phase1/electricity/analytics/household/customer_metadata_current/data/"
)
OUTPUT_DIR = "./artifacts/model_v8"

LOOKBACK, HORIZON = 90, 60
SEED = 42

# --- weather / degree days ---
HDD_BASE = 20.0          # heating below this
CDD_BASE = 26.5          # cooling above this (20-26.5 is the dead band)
NORMAL_SMOOTH_DAYS = 7   # +/- days circular smoothing of the calendar normal
FUTURE_MODE = "normal"   # "normal" (deployable) or "actual" (oracle ceiling)
NORMAL_MODE = "loyo"     # "loyo" for train/eval rows; use "all" when deploying

# --- previous-year anomaly features (v4 point 2) ---
USE_PREV_YEAR_FEATURES = True
PREV_YEAR_LAG_DAYS = 365  # off by one day after a leap day; irrelevant at this scale

# --- 10-day forecast look-ahead (v4 point 1) ---
FORECAST_MODE = "simulated"   # "archive" | "simulated" | "none"
FORECAST_S3 = None            # e.g. "s3a://phase1/electricity/analytics/reference/coato_daily_forecast_archive/"
FORECAST_DAYS = 10
FC_BLOCKS = [(1, 3), (4, 7), (8, 10)]   # sub-blocks of the forecast span for fc_anom_*
FC_TAIL_DAYS = 3              # forecast anomaly beyond day 10 = mean anomaly of the last N forecast days
FC_FADE_TAU = 5.0             # days; e-folding time of the fade into the normal after the forecast ends
# simulated-forecast error (daily-mean temperature, deg C). Calibrate to your provider's
# measured error if you have it; these are typical NWP values.
FC_SIGMA_DAY1, FC_SIGMA_DAY10 = 1.0, 2.5
FC_RHO = 0.6                  # day-to-day autocorrelation of the forecast error

# --- daily min/max temperature (v6) ---
USE_MINMAX = True             # False reproduces v5 (mean-temperature baseline, no min/max features)
HDD_MIN_BASE = 20.0           # heating degree days from the daily MIN: max(HDD_MIN_BASE - Tmin, 0)
CDD_MAX_BASE = 26.5           # cooling degree days from the daily MAX: max(Tmax - CDD_MAX_BASE, 0)
MINMAX_MIN_HOURS = 20         # a day needs at least this many hourly readings for a trustworthy min/max
WEATHER_TZ = None             # IANA zone (e.g. "Europe/Berlin") if the weather timestamps are UTC and days
                              # should be cut at local midnight; None = use the timestamps as they are
FC_MINMAX_EXTRA_FRAC = 0.5    # simulated forecast: min/max get an extra independent error, std = this * the mean's

# --- data regime cutoff + old-regime data (v8) ---
REGIME_CUTOFF = "2024-08-01"   # start of the new (decimal-kWh) regime; origins before it are "old"
INCLUDE_OLD = "all"            # "none" = v7 | "all" | "non_winter"
OLD_START = "2022-01-01"       # earliest old origin considered
OLD_EXCLUDE_TARGET_MONTHS = (12, 1, 2, 3, 4)   # "non_winter": drop old windows whose target touches these months
OLD_WEIGHT = 1.0               # training sample weight of old-regime rows (1.0 = same as new rows)
OLD_REGIME_FLAG = False        # add feature regime_old (1 = old regime)
SEGMENT_USE_OLD = False        # segments + beta priors from new-regime windows only (as v7)
EXCLUDE_RANGES = [("2023-01-01", "2023-02-28")]   # calendar days to remove completely
EXCLUDE_MODE = "span"          # "span": drop windows whose 150-day span touches a range | "origin": only windows with origin in it

# --- household sensitivity (kWh per degree-day) ---
# Fallback defaults only; main() uses estimate_beta_priors() instead.
BETA_H_PRIOR = 0.48
BETA_C_PRIOR = 0.35
BETA_MAX = 3.0
RIDGE_LAMBDA = 200.0

LEVEL_DAYS = 30  # baseline level window; degree-day reference uses the same window
BLOCKS = [(1, 7), (8, 14), (15, 30), (31, 45), (46, 60)]

# --- glitch detection / repair ---
GLITCH_WINDOW = 7
GLITCH_MIN_VALID = 4
GLITCH_DIP_RATIO = 0.15
GLITCH_DIP_MIN_MED = 3.0
GLITCH_SPIKE_RATIO = 4.0
GLITCH_SPIKE_MIN_EXCESS = 15.0

# --- split ---
TRAIN_CUTOFF = "2025-09-01"
VAL_START = "2024-09-01"
VAL_END = "2025-04-30"         # validation = origins in (VAL_START, VAL_END]; fit = training origins >= HORIZON days away on both
                               # sides. An autumn-to-spring block: a summer-only block (the old VAL_START 2025-07-02 to the cutoff)
                               # stopped the transformer model at 0 trees. None = the old split (VAL_START .. TRAIN_CUTOFF)

# --- model selection ---
MAX_FINAL_ITERATIONS = 150   # cap on the final refit tree count for BOTH models

CATBOOST_PARAMS = dict(
    iterations=3000,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=30,
    loss_function="MultiRMSE",
    eval_metric="MultiRMSE",
    random_seed=SEED,
    verbose=100,
)
CATBOOST_EARLY_STOPPING_ROUNDS = 200
CAT_FEATURES = ["coato_code", "housing_type_id", "segment"]

# --- behavior segmentation for segment-specific priors (v4.1) ---
SEGMENT_K = 2                   # fixed, not re-swept - see module docstring point 4
SEGMENT_MIN_COVERAGE_DAYS = 180  # below this, a meter is left "unclassified"
SEGMENT_SEED = SEED

XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    reg_lambda=30.0,
    tree_method="hist",
    multi_strategy="multi_output_tree",  # requires xgboost>=2.0; else wrap in MultiOutputRegressor
    enable_categorical=True,
    random_state=SEED,
)
XGB_EARLY_STOPPING_ROUNDS = 200

# --- experiment tracking, model registry, backtests (MLflow; off unless MLFLOW_TRACKING_URI / --mlflow-uri is set) ---
MLFLOW_EXPERIMENT = "electricity/training"
MLFLOW_MONITOR_EXPERIMENT = "electricity/monitoring"
MLFLOW_DATA_PREP_EXPERIMENT = "electricity/data-prep"   # where process_raw.py records its runs (lineage link)
REGISTERED_MODEL_NAME = "electricity_household_forecast"
LOG_AND_REGISTER_MODEL = True   # log the serving model (pyfunc) and register it as a new version after training
SERVE_MODEL = "catboost"        # "catboost" | "xgboost": the model the registered pyfunc returns by default
SERVING_HISTORY_DAYS = LOOKBACK  # days of consumption history the serving model needs per meter
# promotion gate (see evaluate_gate): the candidate must not be worse than the champion by more than
# GATE_TOLERANCE (relative sysWAPE) at ANY gate horizon, on the target days both were tested on
COATO_MIN_HOUSEHOLDS = 10      # district-level sysWAPE: districts with fewer test households per target day are left out
GATE_HORIZONS = (7, 14, 30)
GATE_TOLERANCE = 0.02
GATE_MIN_COMMON_DAYS = 30
AUTO_PROMOTE = False            # True: a candidate that passes the gate becomes alias "champion" by itself
# rolling-origin backtest: one extra full fit per cutoff (expensive), each tested on the BACKTEST_TEST_DAYS of
# origins starting HORIZON days after its cutoff. e.g. ["2025-03-01", "2025-06-01"]
BACKTEST_CUTOFFS = []
BACKTEST_TEST_DAYS = 90

OFFS = np.arange(-(LOOKBACK - 1), HORIZON + 1)  # day offsets relative to origin (len 150)

# filled while running; logged to MLflow as lineage / diagnostics
DATA_INFO = {}
LAST_REGIME_STATS = {}
LAST_GLITCH_STATS = {}


# =============================================================================
# Spark: read saved windows, metadata, weather, forecasts
# =============================================================================
def build_spark_session(app_name="household-features-v4", cores=8) -> SparkSession:
    missing = [v for v in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT"] if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--packages org.apache.hadoop:hadoop-aws:3.3.4,"
        "com.amazonaws:aws-java-sdk-bundle:1.12.261 pyspark-shell"
    )
    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", "24g")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.sql.shuffle.partitions", "256")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")      # toPandas of the daily table
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_ENDPOINT"))
        .config("spark.hadoop.fs.s3a.endpoint.region", os.getenv("AWS_REGION", "us-east-1"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.fast.upload.buffer", "bytebuffer")
        .config("spark.hadoop.fs.s3a.threads.max", "128")
        .config("spark.hadoop.fs.s3a.connection.maximum", "256")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.connection.maximum", "256")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _stack(col) -> np.ndarray:
    """Column of lists/arrays (None allowed) -> (N, L) float64 with NaN."""
    return np.vstack([np.asarray(a, dtype="float64") for a in col])


def extract_windows_legacy(spark: SparkSession):
    """LEGACY: reads saved window arrays (CLEAN_WINDOWS_S3) and reduces them to plain kWh arrays.
    Window dates are implied by origin_date (input = origin-89..origin,
    target = origin+1..origin+60; the window filters guarantee no missing days)."""
    df = spark.read.parquet(CLEAN_WINDOWS_S3).select(
        "meter_id",
        "origin_date",
        F.transform("input_window", lambda x: x["daily_kwh"].cast("double")).alias("kwh_in"),
        F.transform("target_window", lambda x: x["daily_kwh"].cast("double")).alias("kwh_tgt"),
    )
    meta = (
        spark.read.parquet(CUSTOMER_METADATA_S3)
        .select("meter_id", "consumer_id", "coato_code", "housing_type_id")
        .dropDuplicates(["meter_id"])
    )
    pdf = df.join(meta, on="meter_id", how="left").toPandas()
    kin = _stack(pdf.pop("kwh_in"))
    ktg = _stack(pdf.pop("kwh_tgt"))
    pdf["origin_date"] = pd.to_datetime(pdf["origin_date"])
    assert kin.shape[1] == LOOKBACK and ktg.shape[1] == HORIZON
    return pdf.reset_index(drop=True), kin, ktg


_EPOCH = "to_date('1970-01-01')"


def _exists(spark: SparkSession, path: str) -> bool:
    p = spark._jvm.org.apache.hadoop.fs.Path(path)
    return p.getFileSystem(spark._jsc.hadoopConfiguration()).exists(p)


def _subdirs(spark: SparkSession, path: str) -> list:
    p = spark._jvm.org.apache.hadoop.fs.Path(path)
    fs = p.getFileSystem(spark._jsc.hadoopConfiguration())
    return sorted(s.getPath().getName() for s in fs.listStatus(p) if s.isDirectory())


_INDEX_DIR_RE = re.compile(r"^index_(?P<daily>.+)_L(?P<L>\d+)_H(?P<H>\d+)_S(?P<S>\d+)_G(?P<G>\d+)_R(?P<R>\d+)$")


def resolve_window_store(spark: SparkSession):
    """-> (index_path, daily_path, recent_clean_days, batches). Folder names written by process_raw.py carry
    the settings, so the lookback / horizon here are checked against the ones the index was built with."""
    store = CLEAN_STORE_S3.rstrip("/")
    found = {d: _INDEX_DIR_RE.match(d) for d in _subdirs(spark, store) if d.startswith("index_")}
    found = {d: m for d, m in found.items() if m}
    if WINDOW_INDEX_DIR:
        if WINDOW_INDEX_DIR not in found:
            raise FileNotFoundError(f"{WINDOW_INDEX_DIR} not in {store}; found: {sorted(found) or 'no index_* folder'}")
        name = WINDOW_INDEX_DIR
    else:
        ok = [d for d, m in found.items() if int(m["L"]) == LOOKBACK and int(m["H"]) == HORIZON]
        if len(ok) != 1:
            raise FileNotFoundError(
                f"{len(ok)} index folders for lookback {LOOKBACK} / horizon {HORIZON} in {store} (found "
                f"{sorted(found) or 'none'}). Set WINDOW_INDEX_DIR to the one to use.")
        name = ok[0]
    m = found[name]
    if int(m["L"]) != LOOKBACK or int(m["H"]) != HORIZON:
        raise ValueError(f"{name} was built with lookback {m['L']} / horizon {m['H']}, this script uses "
                         f"{LOOKBACK} / {HORIZON}")
    index_path, daily_path = f"{store}/{name}", f"{store}/daily_{m['daily']}"
    batches = []
    for d in _subdirs(spark, index_path):
        if not d.startswith("batch="):
            continue
        b = int(d.split("=")[1])
        if WINDOW_BATCHES is not None and b not in WINDOW_BATCHES:
            continue
        if _exists(spark, f"{index_path}/{d}/_SUCCESS") and _exists(spark, f"{daily_path}/{d}/_SUCCESS"):
            batches.append(b)
        else:
            print(f"[windows] WARNING batch {b:03d} skipped: index or daily table has no _SUCCESS (unfinished)")
    if WINDOW_BATCHES is not None:
        missing = sorted(set(WINDOW_BATCHES) - set(batches))
        if missing:
            raise FileNotFoundError(f"WINDOW_BATCHES asks for batches {missing} that are not complete in {index_path}")
    if not batches:
        raise FileNotFoundError(f"no finished batch in {index_path}")
    print(f"[windows] index {name} | daily {os.path.basename(daily_path)} | {len(batches)} batch(es): "
          f"{batches if len(batches) <= 12 else str(batches[:6]) + ' ... ' + str(batches[-3:])}")
    return index_path, daily_path, int(m["R"]), sorted(batches)


def gather_windows(w_code, w_origin, d_code, d_day, d_kwh, n_meters, lookback, horizon, chunk=200_000):
    """Rebuilds the input / target kWh arrays of windows from the daily table. Pure numpy.

    w_code, w_origin  per window: dense meter code (0..n_meters-1) and origin as an epoch day
    d_code, d_day, d_kwh  the daily table of the same meters (any row order; kWh NaN where missing)
    -> kin (N, lookback), ktg (N, horizon), float64, NaN where the daily table has no value.

    Each meter is laid out as a dense calendar run [first_day .. last_day] (NaN where a day has no
    record) inside ONE flat array, so a window is a slice at base[meter] + origin - first[meter].
    There is no per-meter Python loop."""
    d_code = np.asarray(d_code, dtype=np.int64)
    d_day = np.asarray(d_day, dtype=np.int64)
    ext = pd.DataFrame({"c": d_code, "d": d_day}).groupby("c")["d"].agg(["min", "max"])
    first = np.zeros(n_meters, dtype=np.int64)
    last = np.full(n_meters, -1, dtype=np.int64)
    first[ext.index.to_numpy()] = ext["min"].to_numpy()
    last[ext.index.to_numpy()] = ext["max"].to_numpy()
    span = np.where(last >= first, last - first + 1, 0)
    base = np.cumsum(span) - span
    flat = np.full(int(span.sum()), np.nan)
    flat[base[d_code] + d_day - first[d_code]] = np.asarray(d_kwh, dtype="float64")

    w_code = np.asarray(w_code, dtype=np.int64)
    w_origin = np.asarray(w_origin, dtype=np.int64)
    if len(w_code) and ((w_origin - (lookback - 1) < first[w_code]) | (w_origin + horizon > last[w_code])).any():
        raise RuntimeError("a window reaches outside its meter's daily table: the index and the daily table do not match")
    offs = np.arange(-(lookback - 1), horizon + 1)
    n = len(w_code)
    kin = np.empty((n, lookback))
    ktg = np.empty((n, horizon))
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        at = (base[w_code[sl]] + w_origin[sl] - first[w_code[sl]])[:, None] + offs[None, :]
        vals = flat[at]
        kin[sl], ktg[sl] = vals[:, :lookback], vals[:, lookback:]
    return kin, ktg


def extract_windows_from_index(spark: SparkSession):
    """Daily table + window index -> (pdf, kin, ktg), the same triple the legacy reader returns.

    1. read the index rows with passes = True (meter_id, origin_date) of the chosen batches;
    2. apply the regime policy (EXCLUDE_RANGES, INCLUDE_OLD ...) to those rows - before any kWh is read;
    3. per batch, read that batch's daily table for the meters that still have windows, and rebuild
       kin / ktg with gather_windows;
    4. attach consumer_id / coato_code / housing_type_id from the customer metadata.
    The result is already regime-filtered, so main() does not call filter_regime again for this source."""
    index_path, daily_path, recent_days, batches = resolve_window_store(spark)
    DATA_INFO.update(index_tag=os.path.basename(index_path)[len("index_"):], index_path=index_path,
                     daily_path=daily_path, batches=len(batches), window_batches=str(WINDOW_BATCHES),
                     recent_clean_days=recent_days)

    parts = []
    for b in batches:
        ix = (spark.read.parquet(f"{index_path}/batch={b:03d}").filter("passes")
              .select("meter_id", F.datediff("origin_date", F.expr(_EPOCH)).alias("origin_day"))
              .toPandas())
        ix["batch"] = b
        parts.append(ix)
        print(f"[windows] batch {b:03d}: {len(ix):,} windows pass the filters")
    ix = pd.concat(parts, ignore_index=True)
    ix["origin_date"] = pd.to_datetime(ix["origin_day"], unit="D")
    keep = regime_keep_mask(ix)
    ix = ix[keep].reset_index(drop=True)

    gb = len(ix) * (LOOKBACK + HORIZON) * 8 / 2 ** 30
    print(f"[windows] {len(ix):,} windows after the regime policy -> kin + ktg need about {gb:.1f} GB "
          f"(the glitch step briefly holds ~3 copies)")
    if gb > MAX_WINDOW_ARRAY_GB:
        raise MemoryError(f"kin + ktg would need {gb:.1f} GB (> MAX_WINDOW_ARRAY_GB={MAX_WINDOW_ARRAY_GB}). "
                          f"Set WINDOW_BATCHES to a subset of the hash batches, or raise the limit.")

    kin = np.empty((len(ix), LOOKBACK))
    ktg = np.empty((len(ix), HORIZON))
    for b in batches:
        rows = np.flatnonzero(ix["batch"].to_numpy() == b)
        if not len(rows):
            continue
        sub = ix.iloc[rows]
        # dense integer meter codes: the daily rows then carry an int instead of a string per row
        meters = pd.DataFrame({"meter_id": sub["meter_id"].unique()})
        meters["mcode"] = np.arange(len(meters), dtype="int32")
        daily = (spark.read.parquet(f"{daily_path}/batch={b:03d}")
                 .join(F.broadcast(spark.createDataFrame(meters)), on="meter_id", how="inner")
                 .select("mcode", F.datediff("calendar_day", F.expr(_EPOCH)).alias("day"),
                         F.col("daily_kwh").cast("double").alias("kwh"))
                 .toPandas())
        code = sub["meter_id"].map(meters.set_index("meter_id")["mcode"]).to_numpy()
        kin[rows], ktg[rows] = gather_windows(code, sub["origin_day"].to_numpy(), daily["mcode"].to_numpy(),
                                              daily["day"].to_numpy(), daily["kwh"].to_numpy(),
                                              len(meters), LOOKBACK, HORIZON)
        print(f"[windows] batch {b:03d}: rebuilt {len(rows):,} windows from {len(daily):,} daily rows "
              f"({len(meters):,} meters)")
        del daily

    # the index promised: no missing day in the target, none in the last `recent_days` of the input
    r = min(recent_days, LOOKBACK)
    if not np.isfinite(ktg).all() or not np.isfinite(kin[:, -r:]).all():
        raise RuntimeError("rebuilt windows contain missing kWh where the index said there are none: the index and "
                           "the daily table are out of step (different runs / settings?)")

    # metadata only for the meters that have windows (the full table has ~9M rows)
    wanted = spark.createDataFrame(pd.DataFrame({"meter_id": ix["meter_id"].unique()}))
    meta = (
        spark.read.parquet(CUSTOMER_METADATA_S3)
        .select("meter_id", "consumer_id", "coato_code", "housing_type_id")
        .dropDuplicates(["meter_id"])
        .join(wanted, on="meter_id", how="inner")
        .toPandas()
    )
    meta["meter_id"] = meta["meter_id"].astype(ix["meter_id"].dtype)
    pdf = ix[["meter_id", "origin_date"]].merge(meta, on="meter_id", how="left")
    assert len(pdf) == len(ix) and kin.shape[1] == LOOKBACK and ktg.shape[1] == HORIZON
    return pdf.reset_index(drop=True), kin, ktg


def _read_enriched_district(code, cfg):
    """Streams one district of the enriched dataset into a customer x day float32 matrix (kept labels only; NaN elsewhere).
    -> (keys, M (n x T), weather (3 x T: t_mean, t_min, t_max; from every row), housing (n,) latest known value).
    Memory: the matrix (grown by doubling) plus one record batch."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    sys.path.insert(0, cfg["here"])
    import enriched_io as eio
    fs, root = eio.open_root(cfg["root"])
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    if cfg["buckets"] is not None:
        flt = flt & pads.field("customer_bucket").isin([int(b) for b in cfg["buckets"]])
    cols = ["customer_key", "date", "kwh", "label_state", "housing_type_id", "t_mean", "t_min", "t_max"]
    labels = pa.array(list(cfg["labels"]), type=pa.string())
    key_of, keys = {}, []
    cap = 4096                                               # rows of M allocated so far (doubles as customers appear)
    M = np.full((cap, T), np.nan, dtype=np.float32)          # customer x day kWh, written batch by batch
    h_day = np.full(1024, -1, dtype=np.int64)                # latest day with a housing type, per customer (grows)
    h_val = np.full(1024, np.nan)
    wt = np.full((3, T), np.nan)
    for b in eio.district_daily(fs, root, code).scanner(columns=cols, filter=flt, batch_size=262_144,
                                                         use_threads=True).to_batches():
        if not b.num_rows:
            continue
        de = pc.dictionary_encode(b.column("customer_key"))
        uq = [str(x) for x in de.dictionary.to_pylist()]
        inv = de.indices.to_numpy(zero_copy_only=False)
        idx = np.empty(len(uq), dtype=np.int64)
        for j, s in enumerate(uq):
            v = key_of.get(s)
            if v is None:
                v = key_of[s] = len(keys)
                keys.append(s)
            idx[j] = v
        c = idx[inv].astype(np.int32)
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        for i, col in enumerate(("t_mean", "t_min", "t_max")):          # weather from every row, whatever its label
            np.fmax.at(wt[i], day, np.asarray(b.column(col).to_numpy(zero_copy_only=False), dtype=float))
        hcol = b.column("housing_type_id")
        try:
            hv = pc.cast(hcol, pa.float64()).to_numpy(zero_copy_only=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            hv = pd.to_numeric(pd.Series(hcol.to_pylist()), errors="coerce").to_numpy(float)
        if len(keys) > len(h_day):
            grow = max(len(keys), 2 * len(h_day)) - len(h_day)
            h_day = np.r_[h_day, np.full(grow, -1, dtype=np.int64)]
            h_val = np.r_[h_val, np.full(grow, np.nan)]
        if len(keys) > cap:
            new = np.full((max(len(keys), 2 * cap), T), np.nan, dtype=np.float32)
            new[:cap] = M
            M, cap = new, len(new)
            del new
        hm = np.isfinite(hv)
        if hm.any():
            hc_, hd_, hv_ = c[hm], day[hm], hv[hm]
            order = np.lexsort((hd_, hc_))
            hc_, hd_, hv_ = hc_[order], hd_[order], hv_[order]
            lst = np.r_[hc_[1:] != hc_[:-1], True]
            hc_, hd_, hv_ = hc_[lst], hd_[lst], hv_[lst]
            newer = hd_ >= h_day[hc_]
            h_day[hc_[newer]], h_val[hc_[newer]] = hd_[newer], hv_[newer]
        kwh = b.column("kwh")
        ok = pc.and_(pc.fill_null(pc.is_in(b.column("label_state"), value_set=labels), False), pc.is_valid(kwh))
        ok = ok.to_numpy(zero_copy_only=False)
        M[c[ok], day[ok]] = kwh.to_numpy(zero_copy_only=False)[ok].astype(np.float32)
        del b, de, inv, c, day, ok, kwh, hv
    n = len(keys)
    return keys, M[:n], wt, h_val[:n]


def _enriched_district_worker(code, cfg):
    """One district of the enriched dataset -> its (sampled) windows and, optionally, per-customer segment features.
    Runs in a worker process (extract_windows_enriched); everything it needs comes in `cfg`, nothing from module globals
    that --set may have changed. Streams the district's record batches, so memory is about customers x days x 4 bytes."""
    import time as _time
    import zlib
    t0 = _time.time()
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T, L, H, S = (d1 - d0).days + 1, cfg["L"], cfg["H"], cfg["stride"]
    keys, M, wt, h_val = _read_enriched_district(code, cfg)
    out = dict(code=str(code), customers=0, candidates=0, passed=0, seconds=0.0, pdf=None, kin=None, ktg=None,
               seg=None, weather=None)
    if not keys:
        out["seconds"] = _time.time() - t0
        return out
    n = len(keys)
    house = h_val                                       # housing type: the customer's latest known value
    has_obs = np.isfinite(M).any(axis=1)                     # customers with no kept day are counted, not windowed
    out["customers"] = int(has_obs.sum())
    dates = d0 + pd.to_timedelta(np.arange(T), unit="D")
    have_w = np.isfinite(wt).any(axis=0)
    out["weather"] = pd.DataFrame({"coato_id": str(code), "d": dates[have_w], "t": wt[0, have_w], "tmin": wt[1, have_w],
                                   "tmax": wt[2, have_w]})

    # staggered origin grid per customer, then the window rules from prefix counts of observed days, in blocks of
    # customers so the count matrix never covers the whole district
    crc = np.array([zlib.crc32(k.encode()) for k in keys], dtype=np.uint64)
    K = (T - 1 - H - (L - 1)) // S + 1
    sampling = cfg["train_sample"] is not None or cfg["test_sample"] is not None
    cut = int((pd.Timestamp(cfg["cutoff"]) - d0).days)
    p_tr = 1.0 if cfg["train_sample"] is None else float(cfg["train_sample"])
    p_te = 1.0 if cfg["test_sample"] is None else float(cfg["test_sample"])
    R, O, NT, KIN, KTG = [], [], [], [], []
    blk = int(cfg.get("block", 10000))
    for a in range(0, n, blk):
        b_ = min(n, a + blk)
        o = (L - 1) + (crc[a:b_] % S).astype(np.int64)[:, None] + S * np.arange(K)[None, :]
        r = np.broadcast_to(np.arange(a, b_)[:, None], o.shape)
        inside = o <= T - 1 - H
        r, o = r[inside], o[inside]
        cnt = np.zeros((b_ - a, T + 1), dtype=np.int16)
        np.cumsum(np.isfinite(M[a:b_]), axis=1, dtype=np.int16, out=cnt[:, 1:])
        rl = r - a
        n_in = cnt[rl, o + 1] - cnt[rl, o - L + 1]
        n_rec = cnt[rl, o + 1] - cnt[rl, o - cfg["recent_window"] + 1]
        n_tg = cnt[rl, o + H + 1] - cnt[rl, o + 1]
        del cnt, rl
        keep = (n_in >= cfg["min_input"]) & (n_rec >= cfg["min_recent"]) & (n_tg >= cfg["min_target"])
        out["candidates"] += int(len(r))
        out["passed"] += int(keep.sum())
        if sampling:
            u = ((crc[r] * np.uint64(2654435761) + o.astype(np.uint64) * np.uint64(2246822519)) % np.uint64(2 ** 32)) / 2.0 ** 32
            keep &= ((o <= cut) & (n_tg >= cfg["train_min_target"]) & (u < p_tr)) | ((o >= cut + H) & (u < p_te))
        r, o, n_tg = r[keep], o[keep], n_tg[keep]
        if len(r):
            R.append(r), O.append(o), NT.append(n_tg)
            KIN.append(M[r[:, None], o[:, None] + np.arange(-(L - 1), 1)[None, :]])
            KTG.append(M[r[:, None], o[:, None] + np.arange(1, H + 1)[None, :]])
    r = np.concatenate(R) if R else np.zeros(0, dtype=np.int64)
    o = np.concatenate(O) if O else np.zeros(0, dtype=np.int64)
    n_tg = np.concatenate(NT) if NT else np.zeros(0, dtype=np.int16)
    if len(r):
        out["kin"] = np.concatenate(KIN)
        out["ktg"] = np.concatenate(KTG)
        del KIN, KTG
        kk = np.asarray(keys, dtype=object)[r]
        out["pdf"] = pd.DataFrame({"meter_id": kk.astype(str), "consumer_id": kk.astype(str),
                                   "origin_date": d0 + pd.to_timedelta(o, unit="D"), "coato_code": str(code),
                                   "housing_type_id": pd.Series(house[r]).astype("Int64").astype(str).replace("<NA>", np.nan).to_numpy(),
                                   "tgt_days": n_tg.astype("int16")})

    # segment features from each customer's whole daily history up to the cutoff (customers that have a kept window only)
    if cfg["seg_full"] and len(r):
        s0 = max(0, int((pd.Timestamp(cfg["seg_start"]) - d0).days))
        s1 = int((pd.Timestamp(cfg["cutoff"]) - d0).days)
        months = dates[s0:s1 + 1].month.values
        t = wt[0, s0:s1 + 1]
        rows = []
        for i in np.unique(r):
            y = M[i, s0:s1 + 1].astype("float64")
            if np.isfinite(y).sum() < cfg["seg_min_days"]:
                rows.append((keys[i], int(np.isfinite(y).sum()), np.nan, np.nan, np.nan, np.nan))
                continue
            y_clean, _rep, masked = clean_daily_series(y)
            y_clean = np.where(masked, np.nan, y_clean)
            n_obs = int(np.isfinite(y_clean).sum())
            if n_obs < cfg["seg_min_days"]:
                rows.append((keys[i], n_obs, np.nan, np.nan, np.nan, np.nan))
                continue
            seas = _seasonality_strength(y_clean, months)
            ts, r2 = _weekly_thermal_fit(y_clean, t)
            rows.append((keys[i], n_obs, np.log1p(max(float(np.nanmedian(y_clean)), 0.0)), ts, r2, seas))
        out["seg"] = pd.DataFrame(rows, columns=["meter_id", "n_obs_days", "log_median_kwh", "thermal_share", "r2",
                                                 "seasonality_strength"]).set_index("meter_id")
    out["seconds"] = _time.time() - t0
    try:
        import resource
        out["peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20     # Linux: kB
    except Exception:                                        # noqa: BLE001
        out["peak_gb"] = float("nan")
    out["rows_customers"] = n
    return out


def _memory_limit_gb():
    """The container's memory limit (cgroup v2 or v1), else the machine's RAM; None if unknown."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 2 ** 60:
                return int(v) / 2 ** 30
        except OSError:
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return None


def _rss_gb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


def extract_windows_enriched():
    """National enriched household dataset (v1.4) -> (pdf, kin, ktg, daily_weather), no Spark.

    Reads the ENRICHED_BUCKETS customers (None = every customer) of every finished district, ENRICHED_WORKERS districts at a
    time in separate processes (guide rule 10: one district per reader, bounded reads). The unit is a customer
    (customer_key + coato_code; meters already combined). Only ENRICHED_LABELS days (complete_observed) carry a value;
    estimated, partial, unlabelled and long-gap-allocated days are NaN (HH-GAP-001-A1 rules 1-3: a 1-day observed reading
    shares no reading interval with any other day, so no input day leaks its target).
    Each customer gets origins every WINDOW_STRIDE days, offset by a stable hash of the key so that origins cover every
    calendar day. A window is kept when its 90-day input has >= MIN_INPUT_DAYS observed days and >= MIN_RECENT_DAYS of the
    last RECENT_WINDOW, and its 60-day target >= MIN_TARGET_DAYS. pdf['tgt_days'] = observed target days: windows with
    >= TRAIN_MIN_TARGET_DAYS are trained on (missing days: residual label 0); every kept window is scored on its observed days.
    National scale: ENRICHED_TRAIN_SAMPLE / ENRICHED_TEST_SAMPLE keep a stable random share of the trainable / test windows
    of EVERY customer read (the rest is never held in memory), and the customers' segment features then come from their
    whole history up to TRAIN_CUTOFF (SEGMENT_FROM_FULL_HISTORY), because a few sampled windows are too little history.
    Weather = the table's own daily t_mean / t_min / t_max per district (branch-observed, 24 complete hourly values)."""
    import concurrent.futures as cf
    import hashlib
    import multiprocessing as mp
    import pickle
    import time as _time
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import enriched_io as eio
    sampling = ENRICHED_TRAIN_SAMPLE is not None or ENRICHED_TEST_SAMPLE is not None
    if sampling and BACKTEST_CUTOFFS:
        raise ValueError("ENRICHED_TRAIN_SAMPLE / ENRICHED_TEST_SAMPLE keep only the windows of the primary fold "
                         "(TRAIN_CUTOFF); run backtests without sampling, or set BACKTEST_CUTOFFS=[]")
    seg_full = SEGMENT_FROM_FULL_HISTORY if SEGMENT_FROM_FULL_HISTORY is not None else sampling
    seg_start = ENRICHED_START if SEGMENT_USE_OLD else str((pd.Timestamp(REGIME_CUTOFF) - pd.Timedelta(days=LOOKBACK - 1)).date())
    cfg = dict(here=here, root=ENRICHED_ROOT, start=ENRICHED_START, end=ENRICHED_END, labels=list(ENRICHED_LABELS),
               buckets=None if ENRICHED_BUCKETS is None else [int(b) for b in ENRICHED_BUCKETS], L=LOOKBACK, H=HORIZON,
               stride=WINDOW_STRIDE, min_input=MIN_INPUT_DAYS, min_recent=MIN_RECENT_DAYS, recent_window=RECENT_WINDOW,
               min_target=MIN_TARGET_DAYS, train_min_target=TRAIN_MIN_TARGET_DAYS, cutoff=TRAIN_CUTOFF,
               train_sample=ENRICHED_TRAIN_SAMPLE, test_sample=ENRICHED_TEST_SAMPLE, seg_full=bool(seg_full),
               seg_start=seg_start, seg_min_days=SEGMENT_MIN_COVERAGE_DAYS)
    fs, root = eio.open_root(ENRICHED_ROOT)
    codes = eio.finished_districts(fs, root, set(ENRICHED_COATOS) if ENRICHED_COATOS else None)
    print(f"[enriched] customers: {'ALL (every bucket)' if ENRICHED_BUCKETS is None else 'buckets ' + str(ENRICHED_BUCKETS)} | "
          f"labels {ENRICHED_LABELS} | stride {WINDOW_STRIDE} | train sample {ENRICHED_TRAIN_SAMPLE} | test sample "
          f"{ENRICHED_TEST_SAMPLE} | segment features from {'full history to ' + TRAIN_CUTOFF if seg_full else 'the windows'} | "
          f"{ENRICHED_WORKERS} worker(s)")
    lim = _memory_limit_gb()
    print(f"[memory] container limit {'unknown' if lim is None else f'{lim:.0f} GB'} | this process {_rss_gb():.1f} GB now")
    cache = None
    if ENRICHED_CACHE_DIR:
        sig = hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if k != "here"}, sort_keys=True,
                                      default=str).encode()).hexdigest()[:12]
        cache = os.path.join(os.path.expanduser(ENRICHED_CACHE_DIR), f"enriched_{sig}")
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, "config.json"), "w") as fh:
            json.dump({k: v for k, v in cfg.items() if k != "here"}, fh, indent=2, default=str)
        print(f"[enriched] district cache: {cache}")

    results, todo = [], []
    for code in codes:
        p = os.path.join(cache, f"{code}.pkl") if cache else None
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                results.append(pickle.load(fh))
        else:
            todo.append(code)
    if cache and results:
        print(f"[enriched] {len(results)} district(s) loaded from the cache; {len(todo)} to read")

    t0 = _time.time()
    done_n = [0]

    def _keep(res):
        if cache:
            tmp = os.path.join(cache, f"{res['code']}.pkl.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, os.path.join(cache, f"{res['code']}.pkl"))
        results.append(res)
        done_n[0] += 1
        k = done_n[0]
        if k % 10 == 0 or k == len(todo) or k <= 3:
            el = _time.time() - t0
            nw = sum(len(x["pdf"]) for x in results if x["pdf"] is not None)
            peak = max((x.get("peak_gb", 0) or 0) for x in results)
            print(f"[windows] {k}/{len(todo)} districts read: {sum(x['customers'] for x in results):,} customers, "
                  f"{nw:,} windows kept; {el / 60:.1f} min, about {el / k * (len(todo) - k) / 60:.1f} min left | memory: "
                  f"largest worker peak {peak:.1f} GB (district of {max(x.get('rows_customers', 0) for x in results):,} "
                  f"customers), main process {_rss_gb():.1f} GB", flush=True)

    if todo and ENRICHED_WORKERS > 1:
        ctx = mp.get_context("spawn")                     # no fork: the S3 client and its threads must not be copied
        with cf.ProcessPoolExecutor(max_workers=int(ENRICHED_WORKERS), mp_context=ctx) as ex:
            pending = {ex.submit(_enriched_district_worker, code, cfg): code for code in todo}
            while pending:                                # drop each future once kept (it holds the district's windows)
                done_now, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for f in done_now:
                    code = pending.pop(f)
                    try:
                        res = f.result()
                    except Exception as e:                # noqa: BLE001
                        raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                    del f
                    _keep(res)
                    del res
    else:
        for code in todo:
            _keep(_enriched_district_worker(code, cfg))

    parts = [x for x in results if x["pdf"] is not None]
    if not parts:
        raise SystemExit(f"no usable windows: customer_bucket {ENRICHED_BUCKETS} in {len(codes)} districts")
    n_cust = sum(x["customers"] for x in results)
    n_cand = sum(x["candidates"] for x in results)
    n_pass = sum(x["passed"] for x in results)
    pdf = pd.concat([x["pdf"] for x in parts], ignore_index=True)
    gb = len(pdf) * (LOOKBACK + HORIZON) * 8 / 2 ** 30
    print(f"[windows] {n_cust:,} customers with observed days in {len(results)} districts; {n_cand:,} candidate windows, "
          f"{n_pass:,} pass the rules ({n_pass / max(n_cand, 1):.1%}); kept {len(pdf):,} windows from "
          f"{pdf['meter_id'].nunique():,} customers; target days observed: all 60 in {(pdf['tgt_days'] == HORIZON).mean():.1%}, "
          f">= {TRAIN_MIN_TARGET_DAYS} (trainable) in {(pdf['tgt_days'] >= TRAIN_MIN_TARGET_DAYS).mean():.1%}; "
          f"kin + ktg about {gb:.1f} GB")
    if gb > MAX_WINDOW_ARRAY_GB:
        raise MemoryError(f"kin + ktg would need {gb:.1f} GB (> MAX_WINDOW_ARRAY_GB={MAX_WINDOW_ARRAY_GB}): lower "
                          f"ENRICHED_TRAIN_SAMPLE / ENRICHED_TEST_SAMPLE, use fewer ENRICHED_BUCKETS, or raise the limit")
    kin = np.concatenate([x["kin"] for x in parts]).astype("float64")
    ktg = np.concatenate([x["ktg"] for x in parts]).astype("float64")
    for x in parts:
        x["kin"] = x["ktg"] = None
    keep = regime_keep_mask(pdf)
    pdf, kin, ktg = pdf[keep].reset_index(drop=True), kin[keep], ktg[keep]
    daily_t = pd.concat([x["weather"] for x in results if x["weather"] is not None], ignore_index=True)
    daily_t["d"] = pd.to_datetime(daily_t["d"])
    if seg_full:
        seg = pd.concat([x["seg"] for x in results if x["seg"] is not None])
        seg = seg[~seg.index.duplicated()]
        globals()["PRECOMPUTED_SEGMENT_FEATURES"] = (str(TRAIN_CUTOFF), seg)
        print(f"[segmentation] features from full daily history ({seg_start} .. {TRAIN_CUTOFF}) for {len(seg):,} customers")
    took = sum(x["seconds"] for x in results)
    DATA_INFO.update(source="household_enriched_v1.4", root=ENRICHED_ROOT, buckets=str(ENRICHED_BUCKETS),
                     districts=len(codes), customers=int(pdf["meter_id"].nunique()), customers_read=int(n_cust),
                     labels=",".join(ENRICHED_LABELS), train_sample=str(ENRICHED_TRAIN_SAMPLE),
                     test_sample=str(ENRICHED_TEST_SAMPLE), segment_features="full_history" if seg_full else "windows",
                     window_rules=f"in>={MIN_INPUT_DAYS}/90, recent>={MIN_RECENT_DAYS}/{RECENT_WINDOW}, target>={MIN_TARGET_DAYS}/60, "
                                  f"stride {WINDOW_STRIDE}")
    print(f"[enriched] district work {took / 60:.0f} worker-minutes, {(_time.time() - t0) / 60:.1f} min wall clock")
    return pdf, kin, ktg, daily_t


def extract_windows(spark: SparkSession):
    """Training windows as (pdf, kin, ktg). WINDOW_SOURCE="index": rebuilt from the daily table + window
    index and already regime-filtered. "legacy": the saved window arrays, NOT yet regime-filtered
    (main() then calls filter_regime)."""
    if WINDOW_SOURCE == "index":
        return extract_windows_from_index(spark)
    if WINDOW_SOURCE == "legacy":
        return extract_windows_legacy(spark)
    raise ValueError(f"unknown WINDOW_SOURCE {WINDOW_SOURCE!r}")


def load_daily_temperature(spark: SparkSession) -> pd.DataFrame:
    """Daily weather per coato: mean `t`, plus (USE_MINMAX) daily `tmin` / `tmax`.

    Min and max are only meaningful for complete days, so days with fewer than
    MINMAX_MIN_HOURS hourly readings get NaN min/max (the mean is left as it was
    in v5). If the timestamps are UTC, set WEATHER_TZ so that days are cut at
    local midnight; otherwise the overnight minimum can be split across two days.
    """
    w = spark.read.parquet(WEATHER_S3)
    ts = F.col("timestamp")
    if WEATHER_TZ:
        ts = F.from_utc_timestamp(ts, WEATHER_TZ)
    agg = [F.mean("temperature_2m").alias("t")]
    if USE_MINMAX:
        agg += [F.min("temperature_2m").alias("tmin"),
                F.max("temperature_2m").alias("tmax"),
                F.count("temperature_2m").alias("n_obs")]
    d = w.withColumn("d", F.to_date(ts)).groupBy("coato_id", "d").agg(*agg).toPandas()
    d["coato_id"] = d["coato_id"].astype(str)
    d["d"] = pd.to_datetime(d["d"])
    if USE_MINMAX:
        short = (d["n_obs"] < MINMAX_MIN_HOURS).values
        d.loc[short, ["tmin", "tmax"]] = np.nan
        print(f"[weather] {short.mean():.1%} of coato-days have fewer than {MINMAX_MIN_HOURS} readings "
              f"-> min/max set to NaN (tz shift: {WEATHER_TZ or 'none'})")
        d = d.drop(columns="n_obs")
    return d


def load_archived_forecasts(spark: SparkSession) -> pd.DataFrame:
    """Archived daily-mean temperature forecasts, as issued.

    ASSUMED SCHEMA (adjust the select below if yours differs), one row per
    (coato, issue date, lead day):
        coato_id    - same ids as the weather table
        issue_date  - date the forecast was issued / became available to you
        lead_day    - 1..FORECAST_DAYS, where lead_day 1 = origin_date + 1
        t_fc        - forecast daily mean 2 m temperature (deg C)
        t_fc_min    - forecast daily MIN 2 m temperature (deg C)   [needed when USE_MINMAX]
        t_fc_max    - forecast daily MAX 2 m temperature (deg C)   [needed when USE_MINMAX]
    If your archive is hourly, aggregate to daily means first.

    TIMING: issue_date is matched to the window's origin_date. Use the
    forecast you would actually have when scoring (the run issued on or
    before the morning you generate forecasts for origin+1), and make sure
    lead_day 1 is origin+1, not origin. Also verify the archive really
    keeps days 5-10 as issued; some "historical forecast" products stitch
    together only each run's first day or two, which is close to observed
    weather and will overstate skill.
    """
    cols = [
        F.col("coato_id").cast("string").alias("coato_id"),
        F.to_date("issue_date").alias("issue_date"),
        F.col("lead_day").cast("int").alias("lead_day"),
        F.col("t_fc").cast("double").alias("t_fc"),
    ]
    if USE_MINMAX:
        cols += [F.col("t_fc_min").cast("double").alias("t_fc_min"),
                 F.col("t_fc_max").cast("double").alias("t_fc_max")]
    f = spark.read.parquet(FORECAST_S3).select(*cols).filter(
        (F.col("lead_day") >= 1) & (F.col("lead_day") <= FORECAST_DAYS))
    d = f.toPandas()
    d["issue_date"] = pd.to_datetime(d["issue_date"])
    return d


# =============================================================================
# Regime filter
# =============================================================================
def _excluded_window_mask(pdf: pd.DataFrame) -> np.ndarray:
    """True for windows touching an EXCLUDE_RANGES day (EXCLUDE_MODE='span': any of the 90 input + 60
    target days; 'origin': only the origin date)."""
    drop = np.zeros(len(pdf), dtype=bool)
    if not EXCLUDE_RANGES:
        return drop
    o = pdf["origin_date"].values.astype("datetime64[D]")
    first = o - np.timedelta64(LOOKBACK - 1, "D")
    last = o + np.timedelta64(HORIZON, "D")
    for lo, hi in EXCLUDE_RANGES:
        lo, hi = np.datetime64(lo, "D"), np.datetime64(hi, "D")
        if EXCLUDE_MODE == "span":
            drop |= (first <= hi) & (last >= lo)
        elif EXCLUDE_MODE == "origin":
            drop |= (o >= lo) & (o <= hi)
        else:
            raise ValueError(f"unknown EXCLUDE_MODE {EXCLUDE_MODE!r}")
    return drop


def regime_keep_mask(pdf: pd.DataFrame, cutoff: str = REGIME_CUTOFF) -> np.ndarray:
    """Decides which saved windows are used (v8); needs only pdf["origin_date"]. Returns a boolean keep mask.
      1. windows touching EXCLUDE_RANGES are dropped, old or new;
      2. new-regime windows (origin >= cutoff) are kept;
      3. old-regime windows (origin < cutoff) follow INCLUDE_OLD:
           "none"        none (the v7 behaviour)
           "all"         every old window with origin >= OLD_START
           "non_winter"  those whose 60-day target touches none of OLD_EXCLUDE_TARGET_MONTHS
    Every script that rebuilds the training windows uses this policy, so they stay in step."""
    if INCLUDE_OLD not in ("none", "all", "non_winter"):
        raise ValueError(f"unknown INCLUDE_OLD {INCLUDE_OLD!r}")
    n0 = len(pdf)
    o = pdf["origin_date"]
    is_old = (o < pd.Timestamp(cutoff)).values
    n_old0, n_new0 = int(is_old.sum()), int((~is_old).sum())

    drop_ex = _excluded_window_mask(pdf)
    drop_old = np.zeros(n0, dtype=bool)
    if INCLUDE_OLD == "none":
        drop_old = is_old.copy()
    else:
        drop_old = is_old & (o < pd.Timestamp(OLD_START)).values
        if INCLUDE_OLD == "non_winter":
            idx = np.flatnonzero(is_old & ~drop_old)
            tdays = o.values[idx].astype("datetime64[D]")[:, None] + np.arange(1, HORIZON + 1).astype("timedelta64[D]")
            months = tdays.astype("datetime64[M]").astype("int64") % 12 + 1
            hits = np.isin(months, OLD_EXCLUDE_TARGET_MONTHS).any(axis=1)
            drop_old[idx[hits]] = True

    keep = ~(drop_ex | drop_old)
    LAST_REGIME_STATS.update(windows_in=n0, new_regime=n_new0, old_regime=n_old0, dropped_excluded=int(drop_ex.sum()),
                             dropped_old_policy=int((drop_old & ~drop_ex).sum()), kept=int(keep.sum()),
                             kept_old=int((keep & is_old).sum()), kept_new=int((keep & ~is_old).sum()))
    print(f"[regime filter] {n0:,} windows: {n_new0:,} new-regime (origin >= {cutoff}), {n_old0:,} old-regime")
    if EXCLUDE_RANGES:
        print(f"[regime filter] excluded calendar ranges {EXCLUDE_RANGES} (mode={EXCLUDE_MODE}): "
              f"dropped {int(drop_ex.sum()):,} windows ({int((drop_ex & is_old).sum()):,} old, "
              f"{int((drop_ex & ~is_old).sum()):,} new)")
    print(f"[regime filter] INCLUDE_OLD={INCLUDE_OLD!r}: old windows dropped by policy "
          f"{int((drop_old & ~drop_ex).sum()):,}; kept {int((keep & is_old).sum()):,} old + "
          f"{int((keep & ~is_old).sum()):,} new = {int(keep.sum()):,} ({keep.mean():.1%} of all)")
    return keep


def filter_regime(pdf: pd.DataFrame, kin: np.ndarray, ktg: np.ndarray, cutoff: str = REGIME_CUTOFF):
    """regime_keep_mask applied to windows that are already in memory (the legacy source)."""
    keep = regime_keep_mask(pdf, cutoff)
    return pdf[keep].reset_index(drop=True), kin[keep], ktg[keep]


def row_weights(pdf_or_meta: pd.DataFrame) -> np.ndarray:
    """Training sample weights: OLD_WEIGHT for old-regime rows, 1.0 for new."""
    is_old = (pdf_or_meta["origin_date"] < pd.Timestamp(REGIME_CUTOFF)).values
    return np.where(is_old, float(OLD_WEIGHT), 1.0)


# =============================================================================
# Climatology (calendar normals)
# =============================================================================
class Climate:
    pass


def _circ_sum(a: np.ndarray, k: int) -> np.ndarray:
    return sum(np.roll(a, s, axis=-1) for s in range(-k, k + 1))


def _safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _normals(ci, yidx, doy, vals, n_c, n_y):
    """Smoothed day-of-year normals per coato from long-format observations.
    Returns (normal_all (C, 366), normal_loyo (C, Y, 366))."""
    T3 = np.full((n_c, n_y, 366), np.nan)
    T3[ci, yidx, doy] = vals
    valid = ~np.isnan(T3)
    T0 = np.where(valid, T3, 0.0)
    S, N = T0.sum(axis=1), valid.sum(axis=1)  # (C, 366)
    k = NORMAL_SMOOTH_DAYS

    normal_all = _safe_div(_circ_sum(S, k), _circ_sum(N, k))
    S_l = S[:, None, :] - T0
    N_l = N[:, None, :] - valid
    nl = _safe_div(_circ_sum(S_l, k), _circ_sum(N_l, k))
    normal_loyo = np.where(np.isnan(nl), normal_all[:, None, :], nl)
    return normal_all, normal_loyo


def build_climate(d: pd.DataFrame) -> Climate:
    """Dense daily temperature grids + smoothed day-of-year normals per coato,
    both leave-one-year-out (for training rows) and all-years (for deployment).
    Built for the daily mean, and (if `tmin` / `tmax` columns exist) for the
    daily min and max as well."""
    c = Climate()
    c.coatos = sorted(d["coato_id"].unique())
    cidx = {k: i for i, k in enumerate(c.coatos)}
    start = d["d"].min().normalize()
    end = d["d"].max().normalize()
    c.start = np.datetime64(start.date(), "D")
    c.n_days = (end - start).days + 1
    c.years = np.arange(start.year, end.year + 1)

    ci = d["coato_id"].map(cidx).values
    di = (d["d"] - start).dt.days.values
    yidx = d["d"].dt.year.values - c.years[0]
    doy = d["d"].dt.dayofyear.values - 1
    n_c, n_y = len(c.coatos), len(c.years)

    c.W = np.full((n_c, c.n_days), np.nan)
    c.W[ci, di] = d["t"].values
    c.normal_all, c.normal_loyo = _normals(ci, yidx, doy, d["t"].values, n_c, n_y)

    c.W_min = c.W_max = None
    c.normal_min_all = c.normal_min_loyo = c.normal_max_all = c.normal_max_loyo = None
    if {"tmin", "tmax"} <= set(d.columns):
        c.W_min = np.full((n_c, c.n_days), np.nan)
        c.W_max = np.full((n_c, c.n_days), np.nan)
        c.W_min[ci, di] = d["tmin"].values
        c.W_max[ci, di] = d["tmax"].values
        c.normal_min_all, c.normal_min_loyo = _normals(ci, yidx, doy, d["tmin"].values, n_c, n_y)
        c.normal_max_all, c.normal_max_loyo = _normals(ci, yidx, doy, d["tmax"].values, n_c, n_y)
    return c


def _lookup_grid(pdf: pd.DataFrame, clim: Climate, W, normal_all, normal_loyo, normal_mode: str):
    """Looks up one temperature variable (mean, min or max) for every day from
    origin-89 to origin+60. Returns (obs, nrm, prev), each (N, 150):
      obs  - observed value
      nrm  - calendar normal
      prev - observed value PREV_YEAR_LAG_DAYS earlier (NaN if outside the grid)
    """
    cmap = {k: i for i, k in enumerate(clim.coatos)}
    ci = pdf["coato_code"].astype(str).map(cmap).fillna(-1).astype(int).values
    ok = (ci >= 0)[:, None]
    cs = np.maximum(ci, 0)[:, None]

    origin = pdf["origin_date"].values.astype("datetime64[D]")
    dates = origin[:, None] + OFFS.astype("timedelta64[D]")[None, :]

    gi = (dates - clim.start).astype(int)
    in_grid = (gi >= 0) & (gi < clim.n_days) & ok
    t_obs = np.where(in_grid, W[cs, np.clip(gi, 0, clim.n_days - 1)], np.nan)

    gp = gi - PREV_YEAR_LAG_DAYS
    in_prev = (gp >= 0) & (gp < clim.n_days) & ok
    t_prev = np.where(in_prev, W[cs, np.clip(gp, 0, clim.n_days - 1)], np.nan)

    yr = dates.astype("datetime64[Y]")
    year = yr.astype(int) + 1970
    doy = (dates - yr.astype("datetime64[D]")).astype(int)
    yi = year - clim.years[0]
    in_years = (yi >= 0) & (yi < len(clim.years))
    n_all = normal_all[cs, doy]
    if normal_mode == "loyo":
        n_loy = normal_loyo[cs, np.clip(yi, 0, len(clim.years) - 1), doy]
        t_nrm = np.where(in_years, n_loy, n_all)
    else:
        t_nrm = n_all
    t_nrm = np.where(ok, t_nrm, np.nan)
    return t_obs, t_nrm, t_prev


def lookup_weather(pdf: pd.DataFrame, clim: Climate, normal_mode: str):
    """Returns (t_obs, t_nrm, t_prev), each (N, 150), for the daily MEAN temperature."""
    return _lookup_grid(pdf, clim, clim.W, clim.normal_all, clim.normal_loyo, normal_mode)


def lookup_minmax(pdf: pd.DataFrame, clim: Climate, normal_mode: str):
    """Returns (tmin_obs, tmin_nrm, tmax_obs, tmax_nrm), each (N, 150)."""
    if clim.W_min is None:
        raise ValueError("daily min/max not available in the climate grid: the weather table must carry "
                         "`tmin`/`tmax` (load_daily_temperature with USE_MINMAX=True)")
    tmin_obs, tmin_nrm, _ = _lookup_grid(pdf, clim, clim.W_min, clim.normal_min_all, clim.normal_min_loyo, normal_mode)
    tmax_obs, tmax_nrm, _ = _lookup_grid(pdf, clim, clim.W_max, clim.normal_max_all, clim.normal_max_loyo, normal_mode)
    return tmin_obs, tmin_nrm, tmax_obs, tmax_nrm


def degree_days(t: np.ndarray):
    return np.maximum(HDD_BASE - t, 0.0), np.maximum(t - CDD_BASE, 0.0)  # NaN propagates


def degree_days_minmax(tmin: np.ndarray, tmax: np.ndarray):
    """Heating degree days from the daily MIN, cooling degree days from the daily MAX."""
    return np.maximum(HDD_MIN_BASE - tmin, 0.0), np.maximum(tmax - CDD_MAX_BASE, 0.0)  # NaN propagates


# =============================================================================
# 10-day forecast look-ahead (v4 point 1)
# =============================================================================
def build_forecast_grid(fc: pd.DataFrame, clim: Climate, col: str = "t_fc") -> np.ndarray:
    """Archive -> dense (n_coatos, n_days, FORECAST_DAYS) array indexed by
    [coato, issue_date - clim.start, lead_day - 1]."""
    cidx = {k: i for i, k in enumerate(clim.coatos)}
    ci = fc["coato_id"].astype(str).map(cidx)
    di = (fc["issue_date"] - pd.Timestamp(clim.start)).dt.days
    ok = ci.notna() & di.between(0, clim.n_days - 1) & fc["lead_day"].between(1, FORECAST_DAYS)
    G = np.full((len(clim.coatos), clim.n_days, FORECAST_DAYS), np.nan)
    G[ci[ok].astype(int).values, di[ok].values, fc.loc[ok, "lead_day"].values - 1] = fc.loc[ok, col].values
    return G


def lookup_forecast_archive(pdf: pd.DataFrame, clim: Climate, grid: np.ndarray) -> np.ndarray:
    """(N, FORECAST_DAYS) forecast issued on each row's origin_date (NaN where missing)."""
    cmap = {k: i for i, k in enumerate(clim.coatos)}
    ci = pdf["coato_code"].astype(str).map(cmap).fillna(-1).astype(int).values
    origin = pdf["origin_date"].values.astype("datetime64[D]")
    gi = (origin - clim.start).astype(int)
    ok = (ci >= 0) & (gi >= 0) & (gi < clim.n_days)
    out = np.full((len(pdf), FORECAST_DAYS), np.nan)
    out[ok] = grid[ci[ok], gi[ok]]
    return out


def _ar1_error(rng, n: int, sig: np.ndarray) -> np.ndarray:
    """(n, K) AR(1)-correlated errors (lag-1 correlation FC_RHO) with marginal std sig[k]."""
    K = len(sig)
    z = rng.standard_normal((n, K))
    e = np.empty_like(z)
    e[:, 0] = sig[0] * z[:, 0]
    for k in range(1, K):
        e[:, k] = FC_RHO * (sig[k] / sig[k - 1]) * e[:, k - 1] + np.sqrt(1 - FC_RHO ** 2) * sig[k] * z[:, k]
    return e


def _forecast_keys(pdf: pd.DataFrame):
    key = pdf["coato_code"].astype(str) + "|" + pdf["origin_date"].dt.strftime("%Y-%m-%d")
    return pd.factorize(key)


def simulate_forecast(pdf: pd.DataFrame, t_obs_fut: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Observed temperature + lead-dependent, AR(1)-correlated error.

    The error is drawn once per (coato, origin_date), not per row: every
    meter in a coato shares the same forecast on a given origin day, so
    their errors are perfectly correlated, exactly as in production. Drawing
    it per row would let the errors average out in the system-level sums
    and make the forecast look much better than it is.
    Marginal std grows linearly from FC_SIGMA_DAY1 to FC_SIGMA_DAY10.
    """
    K = FORECAST_DAYS
    codes, uniq = _forecast_keys(pdf)
    sig = np.linspace(FC_SIGMA_DAY1, FC_SIGMA_DAY10, K)
    e = _ar1_error(np.random.default_rng(seed), len(uniq), sig)
    return t_obs_fut[:, :K] + e[codes]


def simulate_forecast_minmax(pdf: pd.DataFrame, t_mean_fc: np.ndarray, tmin_obs_fut: np.ndarray,
                              tmax_obs_fut: np.ndarray, seed: int = SEED):
    """Simulated forecast daily min and max (v6).

    Each is the observed value plus the SAME shared error as the simulated mean
    forecast (regenerated from the same seed, so it matches simulate_forecast
    exactly) plus an independent AR(1) component with std FC_MINMAX_EXTRA_FRAC
    times the mean's. Min is then capped at the forecast mean and max floored at
    it, so that min <= mean <= max always holds. The extra component uses
    separate seeds so the mean forecast is unchanged from v5.
    ASSUMPTION: this error model is not calibrated to any provider; use an
    archive (FORECAST_MODE="archive") for a realistic assessment.
    """
    K = FORECAST_DAYS
    codes, uniq = _forecast_keys(pdf)
    sig = np.linspace(FC_SIGMA_DAY1, FC_SIGMA_DAY10, K)
    shared = _ar1_error(np.random.default_rng(seed), len(uniq), sig)
    extra = FC_MINMAX_EXTRA_FRAC * sig
    e_min = shared + _ar1_error(np.random.default_rng(seed + 1), len(uniq), extra)
    e_max = shared + _ar1_error(np.random.default_rng(seed + 2), len(uniq), extra)
    tmin_fc = np.minimum(tmin_obs_fut[:, :K] + e_min[codes], t_mean_fc)
    tmax_fc = np.maximum(tmax_obs_fut[:, :K] + e_max[codes], t_mean_fc)
    return tmin_fc, tmax_fc


def blend_forecast(t_nrm_fut: np.ndarray, t_fc: np.ndarray):
    """Forecast for days 1..K, normal + fading anomaly afterwards.
    Returns (t_fut (N, HORIZON), tail_anom (N,)); tail_anom is NaN where the
    forecast is entirely missing (the fade then uses 0, i.e. the plain normal)."""
    K = t_fc.shape[1]
    t_fut = t_nrm_fut.copy()
    t_fut[:, :K] = np.where(np.isfinite(t_fc), t_fc, t_nrm_fut[:, :K])  # per-day fallback to the normal
    tail = np.nanmean((t_fc - t_nrm_fut[:, :K])[:, -FC_TAIL_DAYS:], axis=1)
    k = np.arange(1, HORIZON - K + 1)
    t_fut[:, K:] = t_nrm_fut[:, K:] + np.nan_to_num(tail, nan=0.0)[:, None] * np.exp(-k / FC_FADE_TAU)[None, :]
    return t_fut, tail


# =============================================================================
# Household thermal sensitivity (causal, ridge-shrunk toward cohort priors)
# =============================================================================
def fit_sensitivity(y, h, c, beta_h_prior=BETA_H_PRIOR, beta_c_prior=BETA_C_PRIOR, ridge_lambda=RIDGE_LAMBDA):
    """Per-row ridge fit of  y = a + beta_h * hdd + beta_c * cdd  on the input
    window only. Only the slopes are shrunk (toward the given priors); the
    intercept is free. Returns beta_h, beta_c, a, r2 (all shape (N,)).

    beta_h_prior / beta_c_prior / ridge_lambda accept either a plain scalar
    (applied to every row alike, the original behaviour - e.g. the global
    fallback prior, or the weakly-shrunk near-OLS pass used to estimate
    priors in the first place) or a length-N array (one value per row - e.g.
    each row's own meter-segment prior, built in main() from
    estimate_segment_beta_priors). Broadcasting makes both calling styles
    share this one implementation."""
    m = ~(np.isnan(y) | np.isnan(h) | np.isnan(c))
    y0, h0, c0 = np.where(m, y, 0.0), np.where(m, h, 0.0), np.where(m, c, 0.0)
    n = m.sum(1).astype(float)
    N = len(y)
    beta_h_prior = np.broadcast_to(np.asarray(beta_h_prior, dtype=float), (N,))
    beta_c_prior = np.broadcast_to(np.asarray(beta_c_prior, dtype=float), (N,))
    ridge_lambda = np.broadcast_to(np.asarray(ridge_lambda, dtype=float), (N,))

    A = np.zeros((N, 3, 3))
    A[:, 0, 0] = n
    A[:, 0, 1] = A[:, 1, 0] = h0.sum(1)
    A[:, 0, 2] = A[:, 2, 0] = c0.sum(1)
    A[:, 1, 1] = (h0 * h0).sum(1) + ridge_lambda
    A[:, 1, 2] = A[:, 2, 1] = (h0 * c0).sum(1)
    A[:, 2, 2] = (c0 * c0).sum(1) + ridge_lambda
    A[:, 0, 0] += 1e-6

    b = np.stack([y0.sum(1), (h0 * y0).sum(1) + ridge_lambda * beta_h_prior,
                  (c0 * y0).sum(1) + ridge_lambda * beta_c_prior], 1)

    theta = np.linalg.solve(A, b[..., None])[..., 0]
    bh = np.clip(theta[:, 1], 0.0, BETA_MAX)
    bc = np.clip(theta[:, 2], 0.0, BETA_MAX)

    a = np.where(n > 0, (y0 - bh[:, None] * h0 - bc[:, None] * c0).sum(1) / np.maximum(n, 1), 0.0)
    fit = a[:, None] + bh[:, None] * h0 + bc[:, None] * c0
    sse = (m * (y0 - fit) ** 2).sum(1)
    ybar = y0.sum(1) / np.maximum(n, 1)
    sst = (m * (y0 - ybar[:, None]) ** 2).sum(1)
    r2 = np.clip(np.where(sst > 1e-9, 1 - sse / np.where(sst > 1e-9, sst, 1), 0.0), 0.0, 1.0)
    return bh, bc, a, r2


def estimate_beta_priors(pdf: pd.DataFrame, kin: np.ndarray, clim: Climate, train_cutoff: str, min_meters: int = 200):
    """Data-fitted replacement for the hand-set BETA_H_PRIOR / BETA_C_PRIOR.
    Uses TRAINING-period rows only (causal), and only rows with a strong
    signal for the relevant coefficient. Falls back to the hand-set defaults
    if there's not enough data to estimate from."""
    t_obs, _, _ = lookup_weather(pdf, clim, normal_mode="all")  # normal_mode doesn't affect t_obs
    h_in, c_in = degree_days(t_obs[:, :LOOKBACK])
    L = np.nanmean(kin[:, -LEVEL_DAYS:], axis=1)

    train = (pdf["origin_date"] <= pd.Timestamp(train_cutoff)).values
    bh_w, bc_w, _, _ = fit_sensitivity(kin, h_in, c_in, beta_h_prior=0.0, beta_c_prior=0.0, ridge_lambda=1.0)

    def cohort_median(beta, sig):
        strong = train & (sig > 0) & (L >= 1.0)
        if strong.sum() == 0:
            return None
        thr = np.quantile(sig[strong], 0.5)
        rows = strong & (sig >= thr)
        per_meter = pd.Series(beta[rows]).groupby(pdf.loc[rows, "meter_id"].values).median()
        if len(per_meter) < min_meters:
            return None
        return float(per_meter.median())

    sig_h = np.nansum(h_in ** 2, axis=1)
    sig_c = np.nansum(c_in ** 2, axis=1)
    ph = cohort_median(bh_w, sig_h)
    pc = cohort_median(bc_w, sig_c)
    if ph is None:
        print(f"[priors] not enough heating-signal meters; falling back to default {BETA_H_PRIOR}")
        ph = BETA_H_PRIOR
    if pc is None:
        print(f"[priors] not enough cooling-signal meters; falling back to default {BETA_C_PRIOR}")
        pc = BETA_C_PRIOR
    print(f"[priors] data-fitted beta_h_prior={ph:.3f}, beta_c_prior={pc:.3f} "
          f"(hand-set defaults were {BETA_H_PRIOR}/{BETA_C_PRIOR})")
    return ph, pc


# =============================================================================
# Glitch detection / repair (unchanged from v3)
# =============================================================================
def _window_positions(origin_days: np.ndarray, start_day: int, offs: np.ndarray = OFFS) -> np.ndarray:
    """Positions in a dense per-meter series for the 150 days of each window."""
    return (origin_days - start_day)[:, None] + offs[None, :]


def build_meter_daily_series(pdf: pd.DataFrame, full: np.ndarray) -> dict:
    """Collapses overlapping 150-day windows into ONE dense daily series per
    meter. Where windows disagree on the same calendar day, the median of
    the observed values is used.

    full: (N, LOOKBACK+HORIZON) concatenated input+target kWh, row-aligned
    with pdf. Returns {meter_id: {"start": int, "y": np.ndarray, "idx": np.ndarray}}.
    """
    origin_days = pdf["origin_date"].values.astype("datetime64[D]").astype("int64")
    series = {}
    for meter_id, idx in pdf.groupby("meter_id").indices.items():
        idx = np.asarray(idx)
        o = origin_days[idx]
        start = int(o.min() + OFFS[0])
        n_days = int(o.max() + OFFS[-1] - start + 1)
        pos = _window_positions(o, start)
        vals = full[idx]

        flat_pos, flat_val = pos.ravel(), vals.ravel()
        ok = np.isfinite(flat_val)
        flat_pos, flat_val = flat_pos[ok], flat_val[ok]
        order = np.argsort(flat_pos, kind="stable")
        flat_pos, flat_val = flat_pos[order], flat_val[order]
        uniq_pos, start_idx, counts = np.unique(flat_pos, return_index=True, return_counts=True)

        y = np.full(n_days, np.nan)
        single = counts == 1
        y[uniq_pos[single]] = flat_val[start_idx[single]]
        for p, i0, c in zip(uniq_pos[~single], start_idx[~single], counts[~single]):
            y[p] = np.median(flat_val[i0:i0 + c])

        series[meter_id] = dict(start=start, y=y, idx=idx)
    return series


def _trailing_median(y: np.ndarray, window: int, min_valid: int) -> np.ndarray:
    """Causal rolling median: med[t] is computed from y[t-window+1 .. t] only."""
    padded = np.concatenate([np.full(window - 1, np.nan), y])
    win = sliding_window_view(padded, window)
    n_valid = np.isfinite(win).sum(1)
    med = np.nanmedian(win, axis=1)
    med[n_valid < min_valid] = np.nan
    return med


def clean_daily_series(
    y: np.ndarray,
    window: int = GLITCH_WINDOW,
    min_valid: int = GLITCH_MIN_VALID,
    dip_ratio: float = GLITCH_DIP_RATIO,
    dip_min_med: float = GLITCH_DIP_MIN_MED,
    spike_ratio: float = GLITCH_SPIKE_RATIO,
    spike_min_excess: float = GLITCH_SPIKE_MIN_EXCESS,
):
    """
    Trailing-only (causal) glitch detection/repair on a single dense daily
    series `y` (1-D, one meter, oldest to newest). For every day t, only
    days [t-window+1, t] are used - never a day after t - so this one
    function serves both batch training-data cleaning and single-meter
    serving-time cleaning.

    A day is an "anomaly" if it's a dip (<= dip_ratio of a local trailing
    median that's itself at least dip_min_med) or a spike (>= spike_ratio
    of the local median AND at least spike_min_excess kWh above it). An
    anomaly is "isolated" (repairable) only if the day before it was not
    itself anomalous; see the v3 notes for the resulting trade-off.

    Returns (y_clean, repaired, masked):
      y_clean  - y with isolated anomalies replaced by the local trailing median
      repaired - boolean mask of repaired days (isolated anomalies)
      masked   - anomalous days that are NOT isolated (part of a run). The
                 batch path drops any window containing one; the serving
                 path leaves them as raw values.
    """
    med = _trailing_median(y, window, min_valid)
    ok = np.isfinite(y) & np.isfinite(med)
    dip = ok & (med >= dip_min_med) & (y <= dip_ratio * med)
    spike = ok & (y >= spike_ratio * med) & ((y - med) >= spike_min_excess)
    anomaly = dip | spike

    prev_anomaly = np.r_[False, anomaly[:-1]]  # causal: looks only backward
    isolated = anomaly & ~prev_anomaly

    y_clean = np.where(isolated, med, y)
    return y_clean, isolated, anomaly & ~isolated


def clean_recent_readings_for_serving(daily_kwh: np.ndarray, **kwargs) -> np.ndarray:
    """Serving-time entry point: cleans a meter's recent daily kWh readings
    (oldest-to-newest) with EXACTLY the same trailing-only logic as training.
    Non-isolated anomalous days are left unrepaired; it's up to the caller
    to decide how to handle a meter whose latest window looks anomalous."""
    if not GLITCH_REPAIR:              # trained on already-cleaned data (enriched source): serve the values as delivered
        return np.asarray(daily_kwh, dtype="float64")
    y_clean, _repaired, _masked = clean_daily_series(np.asarray(daily_kwh, dtype="float64"), **kwargs)
    return y_clean


def detect_and_repair_glitches(
    pdf, kin, ktg,
    window=GLITCH_WINDOW, min_valid=GLITCH_MIN_VALID,
    dip_ratio=GLITCH_DIP_RATIO, dip_min_med=GLITCH_DIP_MIN_MED,
    spike_ratio=GLITCH_SPIKE_RATIO, spike_min_excess=GLITCH_SPIKE_MIN_EXCESS,
):
    """
    Batch (training-time) wrapper around clean_daily_series: one dense daily
    series per meter, one causal detection pass, isolated anomalies repaired,
    windows containing a non-isolated anomaly dropped, no whole-meter
    exclusion.

    Returns (pdf, kin, ktg, ktg_raw), filtered and repaired, with a fresh
    index. ktg_raw is the UNREPAIRED target array for the surviving rows,
    for honest test-set evaluation.
    """
    LOOKBACK_, HORIZON_ = kin.shape[1], ktg.shape[1]
    n_before = len(pdf)
    full = np.concatenate([kin, ktg], axis=1)

    series = build_meter_daily_series(pdf, full)
    origin_days = pdf["origin_date"].values.astype("datetime64[D]").astype("int64")

    n_repaired = n_masked_days = 0
    drop_row = np.zeros(n_before, dtype=bool)
    full_clean = full.copy()

    for meter_id, s in series.items():
        y_clean, repaired, masked = clean_daily_series(
            s["y"], window=window, min_valid=min_valid,
            dip_ratio=dip_ratio, dip_min_med=dip_min_med,
            spike_ratio=spike_ratio, spike_min_excess=spike_min_excess,
        )
        n_repaired += int(repaired.sum())
        n_masked_days += int(masked.sum())

        idx = s["idx"]
        pos = _window_positions(origin_days[idx], s["start"])
        full_clean[idx] = y_clean[pos]
        drop_row[idx] |= masked[pos].any(axis=1)

    print(f"[glitch repair] repaired {n_repaired:,} isolated day(s)")
    print(f"[glitch repair] {n_masked_days:,} day(s) flagged as a clustered/longer anomaly "
          f"(not repaired; windows containing them are dropped)")
    print(f"[glitch repair] dropping {drop_row.sum():,} of {n_before:,} windows "
          f"({drop_row.mean():.2%}) for a masked day in their span")

    LAST_GLITCH_STATS.update(windows_before=n_before, repaired_days=n_repaired, masked_days=n_masked_days,
                             windows_dropped=int(drop_row.sum()))
    keep = ~drop_row
    ktg_raw = ktg[keep]
    pdf = pdf[keep].reset_index(drop=True)
    full_clean = full_clean[keep]
    return pdf, full_clean[:, :LOOKBACK_], full_clean[:, LOOKBACK_:], ktg_raw


# =============================================================================
# Behavior segmentation + segment-specific priors (v4.1)
# =============================================================================
def _seasonality_strength(y: np.ndarray, month_of_day: np.ndarray,
                           min_month_days: int = 5, min_total_days: int = 60) -> float:
    """Share of day-to-day variance explained by calendar month alone (any
    cause, not just temperature) - variance-decomposition: seasonal_var /
    (seasonal_var + residual_var), 0 = flat regardless of season, 1 = month
    alone fully explains the day-to-day differences."""
    valid = np.isfinite(y)
    if valid.sum() < min_total_days:
        return np.nan
    month_day_counts = np.array([(valid & (month_of_day == m)).sum() for m in range(1, 13)])
    usable_months = np.where(month_day_counts >= min_month_days)[0] + 1
    usable = valid & np.isin(month_of_day, usable_months)
    if usable.sum() < min_total_days:
        return np.nan
    yy, mm = y[usable], month_of_day[usable]
    grand_mean = yy.mean()
    month_means = {int(mth): yy[mm == mth].mean() for mth in usable_months}
    mu_t = np.array([month_means[int(mth)] for mth in mm])
    seasonal_var = float(np.mean((mu_t - grand_mean) ** 2))
    residual_var = float(np.mean((yy - mu_t) ** 2))
    total_var = seasonal_var + residual_var
    return seasonal_var / total_var if total_var > 1e-9 else np.nan


def _meter_region_temp(coato_code, day0: int, n_days: int, clim: Climate):
    """Daily mean temperature for one meter's region over [day0, day0+n_days)
    (day0 = days since epoch, matching build_meter_daily_series' `start`).
    None if the region isn't in the climate grid."""
    cmap = {k: i for i, k in enumerate(clim.coatos)}
    ci = cmap.get(str(coato_code))
    if ci is None:
        return None
    gi = np.arange(day0, day0 + n_days) - int(clim.start.astype("int64"))
    t = np.full(n_days, np.nan)
    in_grid = (gi >= 0) & (gi < clim.n_days)
    t[in_grid] = clim.W[ci, gi[in_grid]]
    return t


def _weekly_thermal_fit(y: np.ndarray, t: np.ndarray, ridge_lambda: float = 2.0):
    """Per-meter weekly-aggregated ridge fit at the module's fixed
    HDD_BASE/CDD_BASE breakpoints - lighter-weight than the per-row
    fit_sensitivity (which fits a 90-day window per forecasting row) and
    lighter than the standalone analysis's exposure-gated breakpoint
    search. Good enough to rank meters for segmentation, not a replacement
    for either. Returns (thermal_share, r2); NaN if there's too little
    weekly data to fit.

    thermal_share is computed in-sample (share of the fitted mean weekly
    kWh attributable to the weather terms over the observed weeks) rather
    than extrapolated to a hypothetical annual total, to avoid assuming
    the observed window is a representative full year."""
    h, c = degree_days(t)
    n_weeks = len(y) // 7
    if n_weeks < 8:
        return np.nan, np.nan
    y_w = np.nanmean(y[:n_weeks * 7].reshape(n_weeks, 7), axis=1)
    h_w = np.nanmean(h[:n_weeks * 7].reshape(n_weeks, 7), axis=1)
    c_w = np.nanmean(c[:n_weeks * 7].reshape(n_weeks, 7), axis=1)
    m = np.isfinite(y_w) & np.isfinite(h_w) & np.isfinite(c_w)
    n = int(m.sum())
    if n < 8:
        return np.nan, np.nan
    y0, h0, c0 = y_w[m], h_w[m], c_w[m]
    A = np.array([[n, h0.sum(), c0.sum()],
                  [h0.sum(), (h0 * h0).sum() + ridge_lambda, (h0 * c0).sum()],
                  [c0.sum(), (h0 * c0).sum(), (c0 * c0).sum() + ridge_lambda]])
    A[0, 0] += 1e-6
    b = np.array([y0.sum(), (h0 * y0).sum(), (c0 * y0).sum()])
    a, bh, bc = np.linalg.solve(A, b)
    a = max(a, 0.0)   # base-load floor - same fix validated in the standalone analysis
    bh, bc = max(bh, 0.0), max(bc, 0.0)
    fit = a + bh * h0 + bc * c0
    rss = float(np.sum((y0 - fit) ** 2))
    sst = float(np.sum((y0 - y0.mean()) ** 2))
    r2 = max(0.0, 1 - rss / sst) if sst > 1e-9 else 0.0
    thermal_part = float((bh * h0 + bc * c0).mean())
    thermal_share = float(np.clip(thermal_part / max(fit.mean(), 1e-9), 0.0, 1.0))
    return thermal_share, r2


def build_meter_segment_features(pdf: pd.DataFrame, kin: np.ndarray, ktg: np.ndarray, clim: Climate,
                                  train_cutoff: str, min_coverage_days: int = SEGMENT_MIN_COVERAGE_DAYS) -> pd.DataFrame:
    """One row per meter: scale, thermal responsiveness, and seasonality
    strength, built from TRAINING-period rows only (causal - no test-period
    data). Feeds assign_meter_segments() below. Meters with less than
    min_coverage_days of causal history get all-NaN features and are left
    unclassified downstream rather than guessed at."""
    pre = PRECOMPUTED_SEGMENT_FEATURES
    if pre is not None and pre[0] == str(train_cutoff):
        out = pre[1]
        complete = out[["log_median_kwh", "thermal_share", "r2", "seasonality_strength"]].notna().all(axis=1)
        print(f"[segmentation] meter features from each customer's full daily history up to {train_cutoff} (computed while "
              f"reading): {len(out):,} meters ({complete.sum():,} with complete, clusterable features)")
        return out
    train = (pdf["origin_date"] <= pd.Timestamp(train_cutoff)).values
    if not SEGMENT_USE_OLD:
        train = train & (pdf["origin_date"] >= pd.Timestamp(REGIME_CUTOFF)).values   # new regime only, as in v7
    pdf_tr, kin_tr, ktg_tr = pdf[train].reset_index(drop=True), kin[train], ktg[train]
    if len(pdf_tr) == 0:
        raise ValueError("no training-period rows available to build meter segment features")

    full = np.concatenate([kin_tr, ktg_tr], axis=1)
    series = build_meter_daily_series(pdf_tr, full)
    coato_of_meter = pdf_tr.drop_duplicates("meter_id").set_index("meter_id")["coato_code"]

    rows = []
    for meter_id, s in series.items():
        y_clean, _repaired, masked = clean_daily_series(s["y"])
        y_clean = np.where(masked, np.nan, y_clean)
        n_obs = int(np.isfinite(y_clean).sum())
        if n_obs < min_coverage_days:
            rows.append(dict(meter_id=meter_id, n_obs_days=n_obs, log_median_kwh=np.nan,
                             thermal_share=np.nan, r2=np.nan, seasonality_strength=np.nan))
            continue

        median_kwh = float(np.nanmedian(y_clean))
        dates = pd.to_datetime((s["start"] + np.arange(len(y_clean))).astype("datetime64[D]"))
        seas = _seasonality_strength(y_clean, dates.month.values)

        t = _meter_region_temp(coato_of_meter.get(meter_id), s["start"], len(y_clean), clim)
        thermal_share, r2 = (np.nan, np.nan) if t is None else _weekly_thermal_fit(y_clean, t)

        rows.append(dict(meter_id=meter_id, n_obs_days=n_obs, log_median_kwh=np.log1p(max(median_kwh, 0.0)),
                         thermal_share=thermal_share, r2=r2, seasonality_strength=seas))

    out = pd.DataFrame(rows).set_index("meter_id")
    complete = out[["log_median_kwh", "thermal_share", "r2", "seasonality_strength"]].notna().all(axis=1)
    print(f"[segmentation] meter features built for {len(out):,} meters "
          f"({complete.sum():,} with complete, clusterable features)")
    return out


def assign_meter_segments(meter_features: pd.DataFrame, k: int = SEGMENT_K, seed: int = SEGMENT_SEED) -> pd.Series:
    """k-means on standardized (log_median_kwh, thermal_share, r2,
    seasonality_strength). k=2 was chosen by a silhouette sweep in the
    standalone analysis and is fixed here rather than re-swept on every
    run; the naming below assumes exactly two clusters. Meters that failed
    build_meter_segment_features' coverage gate are labelled
    "unclassified" - they fall back to the pooled prior in
    estimate_segment_beta_priors, same as a segment with too little signal
    of its own.

    Returns a Series indexed by meter_id, name 'segment'.
    """
    assert k == 2, "segment naming below assumes exactly two clusters"
    cols = ["log_median_kwh", "thermal_share", "r2", "seasonality_strength"]
    complete = meter_features.dropna(subset=cols)
    labels = pd.Series("unclassified", index=meter_features.index, name="segment")
    if len(complete) < 2 * k:
        print(f"[segmentation] only {len(complete):,} meters have complete features; "
              f"skipping clustering, all meters marked 'unclassified'")
        return labels

    X = StandardScaler().fit_transform(complete[cols])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    raw_labels = km.fit_predict(X)

    # the cluster with the higher mean (thermal_share + seasonality_strength)
    # is "weather_responsive_winter_peaked", matching the standalone analysis's naming
    score = complete["thermal_share"].values + complete["seasonality_strength"].values
    cluster_score = pd.Series(score).groupby(raw_labels).mean()
    weather_cluster = cluster_score.idxmax()
    name_map = {c: ("weather_responsive_winter_peaked" if c == weather_cluster
                    else "stable_baseload_low_seasonality") for c in cluster_score.index}
    labels.loc[complete.index] = [name_map[c] for c in raw_labels]
    # the fitted clustering, so a new meter can be assigned from its own history (evaluation harness, serving)
    sc = StandardScaler().fit(complete[cols])
    labels.attrs["kmeans"] = dict(cols=cols, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                                  centers=km.cluster_centers_.tolist(),
                                  names=[name_map[c] for c in range(len(km.cluster_centers_))])

    print(f"[segmentation] {labels.loc[complete.index].value_counts().to_dict()}, "
          f"{(labels == 'unclassified').sum():,} unclassified")
    return labels


def estimate_segment_beta_priors(pdf: pd.DataFrame, kin: np.ndarray, clim: Climate, train_cutoff: str,
                                  meter_segment: pd.Series, min_meters: int = 200, kind: str = "mean"):
    # kind="mean"   : priors for the mean-temperature degree days (beta_h / beta_c)
    # kind="minmax" : priors for HDD(Tmin) / CDD(Tmax) (beta_h_min / beta_c_max), v6
    """Like estimate_beta_priors, but one (beta_h_prior, beta_c_prior) pair
    PER SEGMENT instead of one pooled pair for every meter - replacing the
    single global prior that estimate_beta_priors (still in this file, now
    unused by main()) computed. Uses training-period rows only (causal).

    A segment without enough signal of its own, and the "unclassified"
    bucket (meters that failed the coverage gate in
    build_meter_segment_features), both fall back to the POOLED prior
    computed across all training-period meters regardless of segment - the
    same fallback behaviour estimate_beta_priors already had.

    Returns (segment_priors, pooled_prior): segment_priors is
    {segment_label: (ph, pc)} for every segment present (plus
    "unclassified"); pooled_prior is the (ph, pc) used as the fallback.
    """
    if kind == "minmax":
        tmin_obs, _, tmax_obs, _ = lookup_minmax(pdf, clim, normal_mode="all")
        h_in, c_in = degree_days_minmax(tmin_obs[:, :LOOKBACK], tmax_obs[:, :LOOKBACK])
    elif kind == "mean":
        t_obs, _, _ = lookup_weather(pdf, clim, normal_mode="all")
        h_in, c_in = degree_days(t_obs[:, :LOOKBACK])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    print(f"[priors] estimating segment priors on {kind} degree days")
    L = np.nanmean(kin[:, -LEVEL_DAYS:], axis=1)
    train = (pdf["origin_date"] <= pd.Timestamp(train_cutoff)).values
    if not SEGMENT_USE_OLD:
        train = train & (pdf["origin_date"] >= pd.Timestamp(REGIME_CUTOFF)).values   # new regime only, as in v7
    bh_w, bc_w, _, _ = fit_sensitivity(kin, h_in, c_in, beta_h_prior=0.0, beta_c_prior=0.0, ridge_lambda=1.0)
    sig_h = np.nansum(h_in ** 2, axis=1)
    sig_c = np.nansum(c_in ** 2, axis=1)
    seg_of_row = pdf["meter_id"].map(meter_segment).fillna("unclassified").values

    def cohort_median(beta, sig, mask):
        strong = mask & (sig > 0) & (L >= 1.0)
        if strong.sum() == 0:
            return None
        thr = np.quantile(sig[strong], 0.5)
        rows = strong & (sig >= thr)
        per_meter = pd.Series(beta[rows]).groupby(pdf.loc[rows, "meter_id"].values).median()
        if len(per_meter) < min_meters:
            return None
        return float(per_meter.median())

    pooled_h = cohort_median(bh_w, sig_h, train)
    pooled_c = cohort_median(bc_w, sig_c, train)
    if pooled_h is None:
        print(f"[priors] pooled: not enough heating-signal meters; falling back to default {BETA_H_PRIOR}")
        pooled_h = BETA_H_PRIOR
    if pooled_c is None:
        print(f"[priors] pooled: not enough cooling-signal meters; falling back to default {BETA_C_PRIOR}")
        pooled_c = BETA_C_PRIOR
    print(f"[priors] pooled fallback: beta_h_prior={pooled_h:.3f}, beta_c_prior={pooled_c:.3f}")

    segment_priors = {}
    for seg in sorted(set(meter_segment.values) | {"unclassified"}):
        mask = train & (seg_of_row == seg)
        ph = cohort_median(bh_w, sig_h, mask)
        pc = cohort_median(bc_w, sig_c, mask)
        if ph is None:
            print(f"[priors] segment '{seg}': not enough heating-signal meters; "
                  f"falling back to pooled prior {pooled_h:.3f}")
            ph = pooled_h
        if pc is None:
            print(f"[priors] segment '{seg}': not enough cooling-signal meters; "
                  f"falling back to pooled prior {pooled_c:.3f}")
            pc = pooled_c
        segment_priors[seg] = (ph, pc)
        print(f"[priors] segment '{seg}': beta_h_prior={ph:.3f}, beta_c_prior={pc:.3f}")

    segment_priors.setdefault("unclassified", (pooled_h, pooled_c))
    return segment_priors, (pooled_h, pooled_c)


def row_priors_from_segments(pdf: pd.DataFrame, meter_segment: pd.Series, segment_priors: dict, pooled_prior: tuple):
    """Maps each row's meter to its segment's prior, for passing directly
    into fit_sensitivity / build_features as the (now per-row) beta_h_prior
    / beta_c_prior arrays."""
    seg_of_row = pdf["meter_id"].map(meter_segment).fillna("unclassified")
    ph = seg_of_row.map(lambda s: segment_priors.get(s, pooled_prior)[0]).values.astype(float)
    pc = seg_of_row.map(lambda s: segment_priors.get(s, pooled_prior)[1]).values.astype(float)
    return ph, pc


# =============================================================================
# Feature construction
# =============================================================================
def nm(a):
    return np.nanmean(a, axis=1)


def build_features(pdf, kin, ktg, clim, future_mode=None, normal_mode=None,
                    beta_h_prior=BETA_H_PRIOR, beta_c_prior=BETA_C_PRIOR,
                    forecast_mode=None, fc_grid=None, use_prev_year=None,
                    meter_segment: pd.Series = None,
                    beta_h_prior_mm=None, beta_c_prior_mm=None, fc_grid_mm: dict = None,
                    use_minmax: bool = None):
    # None = the module setting at CALL time (a default written as FORECAST_MODE etc. would freeze the value at import,
    # so --set / programmatic changes of these settings were silently ignored by unchunked calls)
    future_mode = FUTURE_MODE if future_mode is None else future_mode
    normal_mode = NORMAL_MODE if normal_mode is None else normal_mode
    forecast_mode = FORECAST_MODE if forecast_mode is None else forecast_mode
    use_prev_year = USE_PREV_YEAR_FEATURES if use_prev_year is None else use_prev_year
    use_minmax = USE_MINMAX if use_minmax is None else use_minmax
    N = len(pdf)
    t_obs, t_nrm, t_prev = lookup_weather(pdf, clim, normal_mode)

    t_in, t_fut_actual = t_obs[:, :LOOKBACK], t_obs[:, LOOKBACK:]
    n_in = t_nrm[:, :LOOKBACK]
    n_fut = t_nrm[:, LOOKBACK:]
    t_prev_fut = t_prev[:, LOOKBACK:]

    # --- weather used for the horizon: oracle, or normal (+ 10-day forecast) ---
    t_fc, fc_tail = None, None
    if future_mode == "actual":
        t_fut = t_fut_actual
    else:
        t_fut = n_fut
        if forecast_mode == "archive":
            if fc_grid is None:
                raise ValueError("forecast_mode='archive' requires fc_grid (see build_forecast_grid)")
            t_fc = lookup_forecast_archive(pdf, clim, fc_grid)
        elif forecast_mode == "simulated":
            t_fc = simulate_forecast(pdf, t_fut_actual)
        elif forecast_mode != "none":
            raise ValueError(f"unknown forecast_mode {forecast_mode!r}")
        if t_fc is not None:
            have = np.isfinite(t_fc).all(axis=1)
            print(f"[forecast] mode={forecast_mode}: {have.mean():.1%} of rows have a complete "
                  f"{FORECAST_DAYS}-day forecast; others fall back to the normal for missing days")
            t_fut, fc_tail = blend_forecast(n_fut, t_fc)

    h_in, c_in = degree_days(t_in)
    h_fut, c_fut = degree_days(t_fut)

    # --- daily min/max weather (v6): observed history, normals, and the blended future ---
    tmin_fc = tmax_fc = None
    if use_minmax:
        tmin_obs, tmin_nrm, tmax_obs, tmax_nrm = lookup_minmax(pdf, clim, normal_mode)
        tmin_in, tmax_in = tmin_obs[:, :LOOKBACK], tmax_obs[:, :LOOKBACK]
        nmin_fut, nmax_fut = tmin_nrm[:, LOOKBACK:], tmax_nrm[:, LOOKBACK:]
        if future_mode == "actual":
            tmin_fut, tmax_fut = tmin_obs[:, LOOKBACK:], tmax_obs[:, LOOKBACK:]
        else:
            tmin_fut, tmax_fut = nmin_fut, nmax_fut
            if t_fc is not None:
                if forecast_mode == "archive":
                    if fc_grid_mm is None:
                        raise ValueError("forecast_mode='archive' with use_minmax requires fc_grid_mm "
                                         "(build_forecast_grid for t_fc_min and t_fc_max)")
                    tmin_fc = lookup_forecast_archive(pdf, clim, fc_grid_mm["min"])
                    tmax_fc = lookup_forecast_archive(pdf, clim, fc_grid_mm["max"])
                else:  # simulated
                    tmin_fc, tmax_fc = simulate_forecast_minmax(
                        pdf, t_fc, tmin_obs[:, LOOKBACK:], tmax_obs[:, LOOKBACK:])
                have_mm = np.isfinite(tmin_fc).all(axis=1) & np.isfinite(tmax_fc).all(axis=1)
                print(f"[forecast] min/max: {have_mm.mean():.1%} of rows have a complete {FORECAST_DAYS}-day "
                      f"min and max forecast")
                tmin_fut, _ = blend_forecast(nmin_fut, tmin_fc)
                tmax_fut, _ = blend_forecast(nmax_fut, tmax_fc)
        h_in_mm, c_in_mm = degree_days_minmax(tmin_in, tmax_in)
        h_fut_mm, c_fut_mm = degree_days_minmax(tmin_fut, tmax_fut)

    f = {}
    # --- static / calendar ---
    doy = pdf["origin_date"].dt.dayofyear.values
    f["origin_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["origin_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if OLD_REGIME_FLAG:                           # v8: 1 for old-regime rows, always 0 for new rows and at serving
        f["regime_old"] = (pdf["origin_date"].values < np.datetime64(REGIME_CUTOFF)).astype(float)

    # --- consumption aggregates (input side) ---
    for lag in range(1, 8):
        f[f"kwh_lag_{lag}"] = kin[:, -lag]
    for w in (3, 7, 14, 30, 60, 90):
        f[f"kwh_mean_{w}d"] = nm(kin[:, -w:])
    f["kwh_std_30d"] = np.nanstd(kin[:, -30:], axis=1)
    f["kwh_trend_7_30"] = _safe_div(f["kwh_mean_7d"], f["kwh_mean_30d"])
    f["kwh_trend_30_90"] = _safe_div(f["kwh_mean_30d"], f["kwh_mean_90d"])

    # --- weather aggregates (input side) ---
    for w in (7, 30, 90):
        f[f"temp_mean_{w}d"] = nm(t_in[:, -w:])
        f[f"hdd_mean_{w}d"] = nm(h_in[:, -w:])
        f[f"cdd_mean_{w}d"] = nm(c_in[:, -w:])
    for w in (7, 30):
        f[f"temp_anom_{w}d"] = nm(t_in[:, -w:] - n_in[:, -w:])

    # --- household sensitivity (input window only) ---
    bh, bc, a, r2 = fit_sensitivity(kin, h_in, c_in, beta_h_prior=beta_h_prior, beta_c_prior=beta_c_prior)
    f.update(beta_h=bh, beta_c=bc, base_load_fit=a, sens_r2=r2)

    # --- min/max sensitivity (v6): HDD(Tmin) and CDD(Tmax), input window only ---
    if use_minmax:
        bph = BETA_H_PRIOR if beta_h_prior_mm is None else beta_h_prior_mm
        bpc = BETA_C_PRIOR if beta_c_prior_mm is None else beta_c_prior_mm
        bh_mm, bc_mm, a_mm, r2_mm = fit_sensitivity(kin, h_in_mm, c_in_mm, beta_h_prior=bph, beta_c_prior=bpc)
        f.update(beta_h_min=bh_mm, beta_c_max=bc_mm, base_load_fit_mm=a_mm, sens_r2_mm=r2_mm)

    # --- reference degree days embedded in the level window ---
    L = nm(kin[:, -LEVEL_DAYS:])
    h_ref, c_ref = nm(h_in[:, -LEVEL_DAYS:]), nm(c_in[:, -LEVEL_DAYS:])
    f["level_kwh"] = L

    # --- horizon-block weather + deltas vs reference ---
    for a0, b0 in BLOCKS:
        s = slice(a0 - 1, b0)
        tag = f"{a0}_{b0}"
        hb, cb = nm(h_fut[:, s]), nm(c_fut[:, s])
        f[f"fut_temp_{tag}"] = nm(t_fut[:, s])
        f[f"fut_hdd_{tag}"] = hb
        f[f"fut_cdd_{tag}"] = cb
        f[f"d_hdd_{tag}"] = hb - h_ref
        f[f"d_cdd_{tag}"] = cb - c_ref
        if future_mode == "actual":
            f[f"fut_temp_anom_{tag}"] = nm(t_fut[:, s] - n_fut[:, s])

    # --- previous-year anomaly (v4 point 2): same calendar days last year minus the normal ---
    if use_prev_year:
        for a0, b0 in BLOCKS:
            s = slice(a0 - 1, b0)
            f[f"prev_anom_{a0}_{b0}"] = nm(t_prev_fut[:, s] - n_fut[:, s])

    # --- forecast anomaly (v4 point 1): forecast minus normal; NaN where no forecast ---
    if t_fc is not None:
        for a0, b0 in FC_BLOCKS:
            s = slice(a0 - 1, b0)
            f[f"fc_anom_{a0}_{b0}"] = nm(t_fc[:, s] - n_fut[:, s])
        f["fc_anom_tail"] = fc_tail

    # --- forecast min/max as per-day degree days (v7): cold from the min, hot from the max ---
    if use_minmax and tmin_fc is not None:
        fc_h_min, fc_c_max = degree_days_minmax(tmin_fc, tmax_fc)       # (N, FORECAST_DAYS), NaN where missing
        for k in range(FORECAST_DAYS):
            f[f"fc_hddmin_d{k + 1:02d}"] = fc_h_min[:, k]
        for k in range(FORECAST_DAYS):
            f[f"fc_cddmax_d{k + 1:02d}"] = fc_c_max[:, k]

    # --- thermal-adjusted baseline, per target day ---
    if use_minmax:
        # v6: baseline from the min/max sensitivities and the blended future min/max series
        h_ref_mm, c_ref_mm = nm(h_in_mm[:, -LEVEL_DAYS:]), nm(c_in_mm[:, -LEVEL_DAYS:])
        d_h = np.nan_to_num(h_fut_mm - h_ref_mm[:, None], nan=0.0)  # missing weather -> plain persistence
        d_c = np.nan_to_num(c_fut_mm - c_ref_mm[:, None], nan=0.0)
        base = np.maximum(L[:, None] + bh_mm[:, None] * d_h + bc_mm[:, None] * d_c, 0.0)
    else:
        d_h = np.nan_to_num(h_fut - h_ref[:, None], nan=0.0)  # missing weather -> plain persistence
        d_c = np.nan_to_num(c_fut - c_ref[:, None], nan=0.0)
        base = np.maximum(L[:, None] + bh[:, None] * d_h + bc[:, None] * d_c, 0.0)
    base = np.where(np.isnan(base), 0.0, base)
    for a0, b0 in BLOCKS:
        f[f"baseline_mean_{a0}_{b0}"] = base[:, a0 - 1 : b0].mean(axis=1)
    f["baseline_mean_all"] = base.mean(axis=1)

    feats = pd.DataFrame(f)
    meta = pdf[["meter_id", "consumer_id", "origin_date", "coato_code", "housing_type_id"]].copy()
    meta["segment"] = (meta["meter_id"].map(meter_segment) if meter_segment is not None
                        else pd.Series("unclassified", index=meta.index)).fillna("unclassified")
    for col in CAT_FEATURES:
        meta[col] = meta[col].astype(object).where(meta[col].notna(), "Missing").astype(str)
    return meta, feats, base, t_fut_actual


def build_features_chunked(pdf, kin, ktg, clim, fc_grid=None, fc_grid_mm=None, chunk=None, **kw):
    """build_features in chunks of FEATURE_CHUNK_ROWS windows (memory: each chunk needs ~12 arrays of chunk x 150 floats).
    Same result as one call: a simulated forecast is first drawn ONCE per (coato, origin) for all rows, in the order
    simulate_forecast itself would draw it, and stored as an archive grid, so every chunk (and every meter of a coato on
    that day) sees the same forecast."""
    chunk = int(chunk or FEATURE_CHUNK_ROWS)
    n = len(pdf)
    if n <= chunk:
        return build_features(pdf, kin, ktg, clim, fc_grid=fc_grid, fc_grid_mm=fc_grid_mm, **kw)
    mode = kw.pop("forecast_mode", FORECAST_MODE)
    if mode == "simulated" and (kw.get("future_mode") or FUTURE_MODE) != "actual":
        u = pdf[["coato_code", "origin_date"]].copy()
        u["coato_code"] = u["coato_code"].astype(str)
        u = u.drop_duplicates().reset_index(drop=True)              # first-appearance order = simulate_forecast's draw order
        t_obs, _, _ = lookup_weather(u, clim, "all")
        t_fc = simulate_forecast(u, t_obs[:, LOOKBACK:])
        K = FORECAST_DAYS
        fc = pd.DataFrame({"coato_id": np.repeat(u["coato_code"].to_numpy(), K),
                           "issue_date": np.repeat(u["origin_date"].to_numpy(), K),
                           "lead_day": np.tile(np.arange(1, K + 1), len(u)), "t_fc": t_fc.reshape(-1)})
        fc["issue_date"] = pd.to_datetime(fc["issue_date"])
        fc_grid, fc_grid_mm = build_forecast_grid(fc, clim), None
        if USE_MINMAX:
            tmin_obs, _, tmax_obs, _ = lookup_minmax(u, clim, "all")
            tmin_fc, tmax_fc = simulate_forecast_minmax(u, t_fc, tmin_obs[:, LOOKBACK:], tmax_obs[:, LOOKBACK:])
            fc["t_fc_min"], fc["t_fc_max"] = tmin_fc.reshape(-1), tmax_fc.reshape(-1)
            fc_grid_mm = dict(min=build_forecast_grid(fc, clim, col="t_fc_min"), max=build_forecast_grid(fc, clim, col="t_fc_max"))
        print(f"[features] simulated forecast drawn once for {len(u):,} (coato, origin) pairs; features in "
              f"{-(-n // chunk)} chunks of {chunk:,} windows")
        mode = "archive"
    metas, featss, bases, tfuts = [], [], [], []
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        kw_c = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in kw.items()}
        m_, f_, b_, t_ = build_features(pdf.iloc[sl].reset_index(drop=True), kin[sl], ktg[sl], clim, forecast_mode=mode,
                                        fc_grid=fc_grid, fc_grid_mm=fc_grid_mm, **kw_c)
        metas.append(m_), featss.append(f_), bases.append(b_), tfuts.append(t_)
    return (pd.concat(metas, ignore_index=True), pd.concat(featss, ignore_index=True), np.vstack(bases), np.vstack(tfuts))


def to_xgb_frame(X: pd.DataFrame, categories: dict = None) -> pd.DataFrame:
    """XGBoost's native categorical support needs pandas 'category' dtype.
    `categories` ({column: list}) pins the category -> code mapping to the one the model was trained with;
    serving passes it (saved in columns.json), because a small batch would otherwise get its own codes.
    A value that was not seen in training becomes NaN, which XGBoost treats as missing."""
    X = X.copy()
    for col in CAT_FEATURES:
        if categories is not None and col in categories:
            X[col] = pd.Categorical(X[col].astype(str), categories=categories[col])
        else:
            X[col] = X[col].astype("category")
    return X


def xgb_categories(Xg: pd.DataFrame) -> dict:
    return {c: [str(v) for v in Xg[c].cat.categories] for c in CAT_FEATURES}


# =============================================================================
# Split, diagnostics, evaluation
# =============================================================================
def make_masks(meta: pd.DataFrame, train_cutoff: str = TRAIN_CUTOFF, val_start: str = VAL_START, test_end=None, val_end=None):
    """
    'test' uses EVERY daily origin from cutoff+embargo through the last
    available date (overlap isn't a problem for system-level sysMAPE: each
    horizon's target day still gets exactly one contribution per meter).
    With `test_end` (a backtest fold) the test origins stop there, so folds do not overlap.
    With `val_end`: validation = origins in (val_start, val_end] and fit = training origins at least HORIZON days away from
    that block on BOTH sides (no fit target day is a validation target day). Without it: validation runs to the cutoff.
    """
    o = meta["origin_date"]
    cutoff = pd.Timestamp(train_cutoff)
    embargo = pd.Timedelta(days=HORIZON)
    train = (o <= cutoff).values

    val_start = pd.Timestamp(val_start)
    if val_end:
        ve = pd.Timestamp(val_end)
        if not val_start < ve <= cutoff:
            raise ValueError(f"validation block ({val_start.date()}, {ve.date()}] must lie before the cutoff {cutoff.date()}")
        val = ((o > val_start) & (o <= ve)).values
        fit = (((o <= val_start - embargo) | (o > ve + embargo)) & (o <= cutoff)).values
    else:
        fit = (o <= val_start - embargo).values
        val = ((o > val_start) & (o <= cutoff)).values

    test = (o >= cutoff + embargo).values
    if test_end is not None:
        test = test & (o <= pd.Timestamp(test_end)).values
    return train, fit, val, test


def regime_report(name, t_actual):
    t = t_actual[~np.isnan(t_actual)]
    heat = (t < HDD_BASE).mean()
    dead = ((t >= HDD_BASE) & (t <= CDD_BASE)).mean()
    cool = (t > CDD_BASE).mean()
    print(f"[regime] {name}: heating(<{HDD_BASE}) {heat:.1%} | dead band {dead:.1%} | cooling(>{CDD_BASE}) {cool:.1%}")


def mae_report(name, pred, y):
    """MAE over the scored (observed) target days; missing days are NaN and left out."""
    e = np.abs(pred - y)
    m = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")      # noqa: E731
    print(f"{name:<34} MAE all {m(e):7.3f} | t+7 {m(e[:, 6]):7.3f} | t+14 {m(e[:, 13]):7.3f} | "
          f"t+30 {m(e[:, 29]):7.3f} | t+60 {m(e[:, 59]):7.3f}")
    return {"all": m(e), 7: m(e[:, 6]), 14: m(e[:, 13]), 30: m(e[:, 29]), 60: m(e[:, 59])}


def sys_mape_report(name, pred, y, origin_days, horizons=(7, 14, 30, 60)):
    """System-level MAPE: sums predicted and actual kWh across all meters
    landing on the same calendar target-day, then takes mean absolute
    percentage error of those daily system totals."""
    out = {}
    for h in horizons:
        days = origin_days + h
        d = pd.DataFrame({"day": days, "p": pred[:, h - 1], "a": y[:, h - 1]}).groupby("day").sum()
        d = d[d["a"] > 0]
        out[h] = float((np.abs(d["p"] - d["a"]) / d["a"]).mean())
    msg = " | ".join(f"t+{h} {v:.3f}" for h, v in out.items())
    print(f"{name:<34} sysMAPE  {msg}")
    return out


def sys_report(name, pred, y, origin_days, horizons=(1, 7, 14, 30, 60)):
    """Per horizon h (one target day per row = origin + h), over the test target days:
      sysMAPE  mean over days of |P_d - A_d| / A_d              (the earlier reports)
      sysWAPE  sum |P_d - A_d| / sum A_d                        (the acceptance-spec formula)
      bias     sum (P_d - A_d) / sum A_d                        (signed: + = over-forecast)
      days     distinct target days that horizon is scored on;  meters/day = mean windows per day
    P_d, A_d are sums over the windows landing on day d."""
    out = {}
    for h in horizons:
        d = (pd.DataFrame({"day": origin_days + h, "p": pred[:, h - 1], "a": y[:, h - 1]})
             .groupby("day").agg(p=("p", "sum"), a=("a", "sum"), n=("p", "size")))
        d = d[d["a"] > 0]
        err = d["p"] - d["a"]
        out[h] = dict(mape=float((err.abs() / d["a"]).mean()), wape=float(err.abs().sum() / d["a"].sum()),
                      bias=float(err.sum() / d["a"].sum()), days=int(len(d)), per_day=float(d["n"].mean()))
        print(f"{name:<26} t+{h:<3d} sysMAPE {out[h]['mape']:6.1%} | sysWAPE {out[h]['wape']:6.1%} | "
              f"bias {out[h]['bias']:+6.1%} | days {out[h]['days']:4d} | windows/day {out[h]['per_day']:9,.0f}")
    return out


def coato_sys_report(preds: dict, y, origin_days, coato, horizons=(1, 7, 14, 30, 60), min_households=10, title=""):
    """District-level system error. Per model, horizon h and district (coato_code): sum the forecasts and the actuals of
    all test households of that district landing on target day d = origin + h (observed target days only), then
      sysWAPE_c = sum_d |P_cd - A_cd| / sum_d A_cd          bias_c = sum_d (P_cd - A_cd) / sum_d A_cd
    Summary over districts with >= min_households households per target day on average (smaller sums are mostly noise,
    and the guide's disclosure rule asks for aggregates of >= 10 customers):
      median / mean / p90 of sysWAPE_c across districts, and pooled = sum_c sum_d |P - A| / sum_c sum_d A
      (energy-weighted: big districts count more).
    -> (per-district DataFrame, summary DataFrame); prints the pooled and median tables."""
    coato = np.asarray(coato).astype(str)
    per = []
    for name, P in preds.items():
        for h in horizons:
            p, a = np.asarray(P[:, h - 1], dtype=float), np.asarray(y[:, h - 1], dtype=float)
            v = np.isfinite(p) & np.isfinite(a)
            if not v.any():
                continue
            g = (pd.DataFrame({"coato": coato[v], "day": origin_days[v] + h, "p": p[v], "a": a[v]})
                 .groupby(["coato", "day"]).agg(p=("p", "sum"), a=("a", "sum"), n=("p", "size")).reset_index())
            g["e"] = (g["p"] - g["a"]).abs()
            g["s"] = g["p"] - g["a"]
            c = g.groupby("coato").agg(abs_err=("e", "sum"), signed=("s", "sum"), actual=("a", "sum"), days=("day", "size"),
                                       households_per_day=("n", "mean")).reset_index()
            c["sys_wape"] = c["abs_err"] / c["actual"].where(c["actual"] > 0)
            c["bias"] = c["signed"] / c["actual"].where(c["actual"] > 0)
            c["model"], c["horizon"] = name, h
            per.append(c)
    if not per:
        return pd.DataFrame(), pd.DataFrame()
    per = pd.concat(per, ignore_index=True)
    ok = per[per["households_per_day"] >= min_households]
    summ = (ok.groupby(["model", "horizon"])
            .apply(lambda d: pd.Series({"pooled": d["abs_err"].sum() / d["actual"].sum(), "median": d["sys_wape"].median(),
                                        "mean": d["sys_wape"].mean(), "p90": d["sys_wape"].quantile(0.9),
                                        "districts": int(len(d)), "households_per_day": d["households_per_day"].median()}),
                   include_groups=False)
            .reset_index())
    n_all, n_ok = per["coato"].nunique(), ok["coato"].nunique()
    print(f"\n=== {title}district (coato) system WAPE: {n_ok} of {n_all} districts with >= {min_households} test households per "
          f"target day (median {ok['households_per_day'].median() if len(ok) else float('nan'):.0f}) ===")
    if summ.empty:
        print("  no district has enough test households per day: score more test windows (more customers / a larger test share)")
        return per, summ
    order = list(preds)
    for col, lab in (("pooled", "pooled over districts (energy-weighted)"), ("median", "median district")):
        t = summ.pivot(index="model", columns="horizon", values=col).reindex(order)
        print(f"-- {lab}\n" + t.map(lambda x: f"{x:6.1%}").to_string())
    return per, summ


# =============================================================================
# One fold: segments, priors, features, CatBoost + XGBoost, evaluation
# =============================================================================
HORIZONS_REPORTED = (1, 7, 14, 30, 60)


def collect_metrics(name, mae, sysr):
    """Flat {metric name: value} for one model; names look like test/catboost/sysWAPE_t7 (MLflow groups on '/')."""
    m = {}
    for k, v in (mae or {}).items():
        m[f"test/{name}/mae_{'all' if k == 'all' else f't{k}'}"] = v
    for h, d in (sysr or {}).items():
        m[f"test/{name}/sysMAPE_t{h}"] = d["mape"]
        m[f"test/{name}/sysWAPE_t{h}"] = d["wape"]
        m[f"test/{name}/bias_t{h}"] = d["bias"]
        m[f"test/{name}/days_t{h}"] = d["days"]
    return m


def system_daily_frame(model, pred, y, origin_days, horizons=HORIZONS_REPORTED):
    """Daily system totals (sum over meters) per horizon: the numbers behind sysMAPE / sysWAPE, kept so a
    candidate and the champion can be compared on exactly the same target days (evaluate_gate) and so
    production can be compared with the test period (monitor)."""
    out = []
    for h in horizons:
        d = (pd.DataFrame({"day": origin_days + h, "p": pred[:, h - 1], "a": y[:, h - 1]})
             .groupby("day").agg(pred=("p", "sum"), actual=("a", "sum"), windows=("p", "size")).reset_index())
        d["target_day"] = pd.to_datetime(d["day"], unit="D")
        d["horizon"], d["model"] = h, model
        out.append(d[["target_day", "horizon", "model", "pred", "actual", "windows"]])
    return pd.concat(out, ignore_index=True)


def fit_and_evaluate(pdf, kin, ktg, ktg_raw, clim, fc_grid, fc_grid_mm, train_cutoff, val_start,
                     test_end=None, label="primary", val_end=None, train_only=False, fallback_iters=None):
    """Everything that depends on the train cutoff, for ONE fold: behaviour segments and priors (training
    history only), features, the CatBoost / XGBoost fits with early stopping on the validation slice, the
    final refits on the whole training period, and the test evaluation against RAW targets.
    The primary fold uses TRAIN_CUTOFF; backtest folds (BACKTEST_CUTOFFS) call this with their own cutoff.
    train_only=True (evaluation harness): no test split is needed and nothing is evaluated; if the validation slice is
    empty, fallback_iters = (catboost, xgboost) final tree counts (e.g. from the previous refit) replace early stopping."""
    from types import SimpleNamespace
    print(f"\n===== fold '{label}': train cutoff {train_cutoff} | validation after {val_start} | "
          f"test origins {'until ' + str(test_end) if test_end else 'to the last available date'} =====")
    print("Building meter behavior segments (training-period history, causal)...")
    meter_feats = build_meter_segment_features(pdf, kin, ktg, clim, train_cutoff)
    meter_segment = assign_meter_segments(meter_feats)
    segment_priors, pooled_prior = estimate_segment_beta_priors(pdf, kin, clim, train_cutoff, meter_segment)
    beta_h_prior, beta_c_prior = row_priors_from_segments(pdf, meter_segment, segment_priors, pooled_prior)

    # v6: separate segment priors for the min/max degree days (the mean-based priors do not transfer)
    segment_priors_mm, pooled_prior_mm, beta_h_prior_mm, beta_c_prior_mm = {}, (None, None), None, None
    if USE_MINMAX:
        segment_priors_mm, pooled_prior_mm = estimate_segment_beta_priors(
            pdf, kin, clim, train_cutoff, meter_segment, kind="minmax")
        beta_h_prior_mm, beta_c_prior_mm = row_priors_from_segments(
            pdf, meter_segment, segment_priors_mm, pooled_prior_mm)

    meta, feats, base, t_fut_actual = build_features_chunked(
        pdf, kin, ktg, clim, beta_h_prior=beta_h_prior, beta_c_prior=beta_c_prior,
        fc_grid=fc_grid, meter_segment=meter_segment,
        beta_h_prior_mm=beta_h_prior_mm, beta_c_prior_mm=beta_c_prior_mm, fc_grid_mm=fc_grid_mm,
    )
    mm_cols = [c for c in feats.columns if c.startswith(("fc_hddmin_d", "fc_cddmax_d"))]
    if mm_cols:
        print("\n[min/max forecast degree-day features fed to the model]")
        print(feats[mm_cols].describe(percentiles=[0.01, 0.5, 0.99]).T.round(2).to_string())
        for pre in ("fc_hddmin_d", "fc_cddmax_d"):
            cols = [c for c in mm_cols if c.startswith(pre)]
            print(f"  {pre}*: share of row-days > 0 = {(feats[cols].values > 0).mean():.1%} "
                  f"(NaN share {np.isnan(feats[cols].values).mean():.1%})")
    y_res = ktg - base  # residual target for training (uses REPAIRED ktg)
    y_res = np.where(np.isfinite(y_res), y_res, 0.0)   # a missing target day (enriched) = no correction, never an invented value
    feature_cols = list(feats.columns) + CAT_FEATURES
    X = pd.concat([feats, meta[CAT_FEATURES]], axis=1)[feature_cols]
    Xg = to_xgb_frame(X)
    res_cols = [f"res_{i}" for i in range(1, HORIZON + 1)]
    Y = pd.DataFrame(y_res, columns=res_cols)

    train, fit, val, test = make_masks(meta, train_cutoff, val_start, test_end, val_end)
    if "tgt_days" in pdf:                     # enriched source: train on windows with enough observed target days
        full = (pdf["tgt_days"] >= TRAIN_MIN_TARGET_DAYS).to_numpy()
        train, fit, val = train & full, fit & full, val & full
    print(f"train {train.sum()} (fit {fit.sum()} / val {val.sum()}) | test {test.sum()} | features {len(feature_cols)}")
    om = meta["origin_date"].dt.to_period("M")
    print(f"origin months  fit {om[fit].min()}..{om[fit].max()} ({om[fit].nunique()}) | val {om[val].min()}..{om[val].max()} "
          f"({om[val].nunique()}) | test {om[test].min()}..{om[test].max()} ({om[test].nunique()})" if fit.any() and val.any()
          and test.any() else "origin months: a split is empty")
    no_val = not val.any()
    if train_only and no_val and fallback_iters and train.any():
        print(f"[fold '{label}'] WARNING: no validation windows: early stopping skipped, final trees "
              f"{tuple(fallback_iters)} (catboost, xgboost) from an earlier fit")
    elif train_only and (not fit.any() or no_val):
        raise ValueError(f"fold '{label}': empty fit / validation split (fit {fit.sum()}, val {val.sum()}); pass "
                         f"fallback_iters to train without validation")
    if not train_only and (not test.any() or not fit.any() or not val.any()):
        raise ValueError(f"fold '{label}': empty split (fit {fit.sum()}, val {val.sum()}, test {test.sum()}); "
                         f"check the cutoffs against the available origin dates "
                         f"({meta['origin_date'].min().date()} .. {meta['origin_date'].max().date()})")
    regime_report("train target days", t_fut_actual[train])
    if test.any():
        regime_report("test target days", t_fut_actual[test])
    is_old = (meta["origin_date"] < pd.Timestamp(REGIME_CUTOFF)).values
    print(f"[old regime] windows by split  fit {int((fit & is_old).sum()):,} old / {int((fit & ~is_old).sum()):,} new | "
          f"val {int((val & is_old).sum()):,} / {int((val & ~is_old).sum()):,} | "
          f"test {int((test & is_old).sum()):,} / {int((test & ~is_old).sum()):,}   (old rows must be 0 in val and test)")
    assert train_only or not (is_old & (val | test)).any(), "old-regime rows leaked into validation / test"
    if is_old.any():
        regime_report("old-regime target days", t_fut_actual[is_old])
        regime_report("new-regime train target days", t_fut_actual[train & ~is_old])
    w_row = row_weights(meta)
    use_w = bool((w_row != 1.0).any())
    sw = (lambda m: w_row[m]) if use_w else (lambda m: None)
    print(f"[weights] OLD_WEIGHT={OLD_WEIGHT} -> sample weights {'ON' if use_w else 'off (all 1.0)'}")

    # -------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------
    if no_val and fallback_iters:
        cb_best_iter, cb_curves = int(fallback_iters[0]), {}
    else:
        cb_model = CatBoostRegressor(cat_features=CAT_FEATURES, **CATBOOST_PARAMS)
        cb_model.fit(X[fit], Y[fit], eval_set=(X[val], Y[val]), early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
                     sample_weight=sw(fit))
        cb_best_iter = cb_model.get_best_iteration()
        cb_curves = cb_model.get_evals_result()              # learning curves of the early-stopping fit
    cb_final_iters = max(min(cb_best_iter, MAX_FINAL_ITERATIONS), 10)
    print(f"[catboost] best iteration on validation: {cb_best_iter} -> using {cb_final_iters} for the final fit "
          f"(capped at MAX_FINAL_ITERATIONS={MAX_FINAL_ITERATIONS})")

    cb_final_params = dict(CATBOOST_PARAMS, iterations=cb_final_iters, verbose=200)
    cb_model = CatBoostRegressor(cat_features=CAT_FEATURES, **cb_final_params)
    cb_model.fit(X[train], Y[train], sample_weight=sw(train))

    # -------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------
    if no_val and fallback_iters:
        xgb_best_iter, xgb_curves = int(fallback_iters[1]), {}
    else:
        xgb_model = xgb.XGBRegressor(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS, **XGB_PARAMS)
        xgb_model.fit(Xg[fit], Y[fit], eval_set=[(Xg[val], Y[val])], sample_weight=sw(fit), verbose=100)
        xgb_best_iter = xgb_model.best_iteration
        xgb_curves = xgb_model.evals_result()
    xgb_final_iters = max(min(xgb_best_iter, MAX_FINAL_ITERATIONS), 10)
    print(f"[xgboost] best iteration on validation: {xgb_best_iter} -> using {xgb_final_iters} for the final fit")

    xgb_final_params = dict(XGB_PARAMS, n_estimators=xgb_final_iters)
    xgb_model = xgb.XGBRegressor(**xgb_final_params)
    xgb_model.fit(Xg[train], Y[train], sample_weight=sw(train), verbose=False)
    if train_only:                                             # evaluation harness: the fitted pieces, no test scoring
        return SimpleNamespace(
            label=label, train_cutoff=train_cutoff, val_start=val_start, feature_cols=feature_cols, res_cols=res_cols,
            cb_model=cb_model, xgb_model=xgb_model, cb_best_iter=int(cb_best_iter),
            cb_final_iters=int(cb_final_iters), xgb_best_iter=int(xgb_best_iter), xgb_final_iters=int(xgb_final_iters),
            cb_curves=cb_curves, xgb_curves=xgb_curves,
            xgb_categories=xgb_categories(Xg), meter_segment=meter_segment, segment_priors=segment_priors,
            pooled_prior=pooled_prior, segment_priors_mm=segment_priors_mm, pooled_prior_mm=pooled_prior_mm,
            cb_imp=pd.DataFrame({"feature": feature_cols, "importance": cb_model.get_feature_importance()}),
            xgb_imp=pd.DataFrame({"feature": feature_cols, "importance": xgb_model.feature_importances_}),
            counts=dict(train=int(train.sum()), fit=int(fit.sum()), val=int(val.sum()), features=len(feature_cols),
                        meters=int(meta["meter_id"].nunique())))

    # -------------------------------------------------------------------
    # Evaluation - scored against RAW (unrepaired) targets
    # -------------------------------------------------------------------
    y_test = ktg_raw[test]
    origin_days_test = meta.loc[test, "origin_date"].values.astype("datetime64[D]").astype("int64")

    pred_cb = np.maximum(base[test] + cb_model.predict(X[test]), 0.0)
    pred_xgb = np.maximum(base[test] + xgb_model.predict(Xg[test]), 0.0)
    pers30 = np.repeat(feats.loc[test, "level_kwh"].values[:, None], HORIZON, 1)
    pers7 = np.repeat(feats.loc[test, "kwh_mean_7d"].values[:, None], HORIZON, 1)
    preds = {"persistence30": pers30, "persistence7": pers7, "thermal": base[test],
             "catboost": pred_cb, "xgboost": pred_xgb}
    # score only target days with an observed value (rule 1): a forecast for a missing day is set aside, not compared
    obs = np.isfinite(y_test)
    if not obs.all():
        print(f"[score] {obs.mean():.1%} of the test target days are observed; forecasts are scored on those only")
        preds = {k: np.where(obs, v, np.nan) for k, v in preds.items()}
        pred_cb, pred_xgb, pers30, pers7 = preds["catboost"], preds["xgboost"], preds["persistence30"], preds["persistence7"]
    base_test = preds["thermal"]

    print(f"\n[config] future_mode={FUTURE_MODE} forecast_mode={FORECAST_MODE} prev_year_features={USE_PREV_YEAR_FEATURES}")
    print("\n--- Test results (MAE, vs. raw targets) ---")
    mae = {"persistence30": mae_report("Persistence (30d mean)", pers30, y_test),
           "persistence7": mae_report("Persistence (7d mean)", pers7, y_test),
           "thermal": mae_report("Thermal baseline only", base_test, y_test),
           "catboost": mae_report("Baseline + CatBoost", pred_cb, y_test),
           "xgboost": mae_report("Baseline + XGBoost", pred_xgb, y_test)}

    print("\n--- Test results (system-level MAPE, vs. raw targets) ---")
    sys_mape_report("Persistence (30d mean)", pers30, y_test, origin_days_test)
    sys_mape_report("Thermal baseline only", base_test, y_test, origin_days_test)
    sys_mape_report("Baseline + CatBoost", pred_cb, y_test, origin_days_test)
    sys_mape_report("Baseline + XGBoost", pred_xgb, y_test, origin_days_test)

    print("\n--- Test results (sysMAPE | acceptance-spec sysWAPE | signed bias | scored days) ---")
    sysr = {"persistence30": sys_report("Persistence (30d mean)", pers30, y_test, origin_days_test),
            "thermal": sys_report("Thermal baseline only", base_test, y_test, origin_days_test),
            "catboost": sys_report("Baseline + CatBoost", pred_cb, y_test, origin_days_test),
            "xgboost": sys_report("Baseline + XGBoost", pred_xgb, y_test, origin_days_test)}

    metrics = {}
    for name in ("persistence30", "persistence7", "thermal", "catboost", "xgboost"):
        metrics.update(collect_metrics(name, mae.get(name), sysr.get(name)))
    metrics.update(collect_metrics("serve", mae[SERVE_MODEL], sysr[SERVE_MODEL]))
    sys_daily = pd.concat([system_daily_frame(n, preds[n], y_test, origin_days_test)
                           for n in ("persistence30", "thermal", "catboost", "xgboost")], ignore_index=True)
    sys_daily["is_serve"] = (sys_daily["model"] == SERVE_MODEL).astype(int)

    # district-level system error (each coato's households summed per target day)
    coato_per, coato_summ = coato_sys_report(
        {k: preds[k] for k in ("persistence30", "persistence7", "thermal", "catboost", "xgboost")}, y_test,
        origin_days_test, meta.loc[test, "coato_code"].to_numpy(), min_households=COATO_MIN_HOUSEHOLDS,
        title=f"[{label}] ")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tag = "" if label == "primary" else "_" + re.sub(r"[^A-Za-z0-9_-]+", "_", str(label))
        coato_per.to_csv(os.path.join(OUTPUT_DIR, f"metrics_by_coato{tag}.csv"), index=False)
        coato_summ.to_csv(os.path.join(OUTPUT_DIR, f"metrics_by_coato_summary{tag}.csv"), index=False)
    except OSError as e:
        print(f"[coato] could not write the district metrics: {e}")
    for _, r in coato_summ.iterrows():
        metrics[f"test/{r['model']}/coato_sysWAPE_pooled_t{r['horizon']}"] = float(r["pooled"])
        metrics[f"test/{r['model']}/coato_sysWAPE_median_t{r['horizon']}"] = float(r["median"])

    cb_imp = pd.DataFrame({"feature": feature_cols, "importance": cb_model.get_feature_importance()})
    xgb_imp = pd.DataFrame({"feature": feature_cols, "importance": xgb_model.feature_importances_})
    return SimpleNamespace(
        label=label, train_cutoff=train_cutoff, val_start=val_start, test_end=test_end,
        meta=meta, X=X, Xg=Xg, base=base, feats=feats, feature_cols=feature_cols, res_cols=res_cols,
        masks=dict(train=train, fit=fit, val=val, test=test), is_old=is_old, t_fut_actual=t_fut_actual,
        cb_model=cb_model, xgb_model=xgb_model, cb_best_iter=int(cb_best_iter), cb_final_iters=int(cb_final_iters),
        xgb_best_iter=int(xgb_best_iter), xgb_final_iters=int(xgb_final_iters),
        cb_curves=cb_curves, xgb_curves=xgb_curves, cb_imp=cb_imp, xgb_imp=xgb_imp,
        meter_segment=meter_segment, segment_priors=segment_priors, pooled_prior=pooled_prior,
        segment_priors_mm=segment_priors_mm, pooled_prior_mm=pooled_prior_mm,
        mae=mae, sysr=sysr, metrics=metrics, sys_daily=sys_daily, preds=preds,
        counts=dict(train=int(train.sum()), fit=int(fit.sum()), val=int(val.sum()), test=int(test.sum()),
                    features=len(feature_cols), meters=int(meta["meter_id"].nunique()),
                    old_fit=int((fit & is_old).sum()), new_fit=int((fit & ~is_old).sum())),
    )


# =============================================================================
# Persist the model files (unchanged layout) and package the serving model
# =============================================================================
def persist_model(fold, daily_t, clim, out_dir=None, extra=None):
    """Writes the files a deployment needs. Same names as before, plus weather_daily.parquet (the observed
    daily weather the climate was built from; the serving model extends it with fresh observations)."""
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    fold.cb_model.save_model(os.path.join(out_dir, "catboost_model.cbm"))
    fold.xgb_model.save_model(os.path.join(out_dir, "xgboost_model.json"))
    # for serving: a new row's meter_id -> segment lookup (an unseen meter falls back
    # to "unclassified" -> the pooled prior, same as during training)
    fold.meter_segment.rename("segment").to_csv(os.path.join(out_dir, "meter_segments.csv"))
    daily_t.to_parquet(os.path.join(out_dir, "weather_daily.parquet"), index=False)
    with open(os.path.join(out_dir, "columns.json"), "w") as fh:
        json.dump(
            dict(
                feature_cols=fold.feature_cols, residual_cols=fold.res_cols, cat_features=CAT_FEATURES,
                hdd_base=HDD_BASE, cdd_base=CDD_BASE,
                segment_beta_priors={seg: list(pr) for seg, pr in fold.segment_priors.items()},
                pooled_beta_prior=list(fold.pooled_prior),
                segment_config=dict(k=SEGMENT_K, min_coverage_days=SEGMENT_MIN_COVERAGE_DAYS, seed=SEGMENT_SEED),
                minmax=dict(
                    enabled=USE_MINMAX, hdd_min_base=HDD_MIN_BASE, cdd_max_base=CDD_MAX_BASE,
                    min_hours=MINMAX_MIN_HOURS, weather_tz=WEATHER_TZ, fc_extra_frac=FC_MINMAX_EXTRA_FRAC,
                    segment_beta_priors={seg: list(pr) for seg, pr in fold.segment_priors_mm.items()},
                    pooled_beta_prior=list(fold.pooled_prior_mm),
                ),
                old_data=dict(include_old=INCLUDE_OLD, old_start=OLD_START, regime_cutoff=REGIME_CUTOFF,
                              exclude_target_months=list(OLD_EXCLUDE_TARGET_MONTHS), old_weight=OLD_WEIGHT,
                              regime_flag=OLD_REGIME_FLAG, segment_use_old=SEGMENT_USE_OLD,
                              exclude_ranges=[list(r) for r in EXCLUDE_RANGES], exclude_mode=EXCLUDE_MODE),
                ridge_lambda=RIDGE_LAMBDA, level_days=LEVEL_DAYS, blocks=BLOCKS,
                future_mode=FUTURE_MODE, regime_cutoff=REGIME_CUTOFF,
                forecast=dict(
                    mode=FORECAST_MODE, days=FORECAST_DAYS, blocks=FC_BLOCKS,
                    tail_days=FC_TAIL_DAYS, fade_tau=FC_FADE_TAU,
                    sim_sigma=[FC_SIGMA_DAY1, FC_SIGMA_DAY10], sim_rho=FC_RHO,
                ),
                prev_year=dict(enabled=USE_PREV_YEAR_FEATURES, lag_days=PREV_YEAR_LAG_DAYS),
                glitch_params=dict(
                    window=GLITCH_WINDOW, min_valid=GLITCH_MIN_VALID,
                    dip_ratio=GLITCH_DIP_RATIO, dip_min_med=GLITCH_DIP_MIN_MED,
                    spike_ratio=GLITCH_SPIKE_RATIO, spike_min_excess=GLITCH_SPIKE_MIN_EXCESS,
                ),
                max_final_iterations=MAX_FINAL_ITERATIONS,
                catboost_best_iteration=fold.cb_best_iter, catboost_final_iterations=fold.cb_final_iters,
                xgboost_best_iteration=fold.xgb_best_iter, xgboost_final_iterations=fold.xgb_final_iters,
                lookback=LOOKBACK, horizon=HORIZON, serve_model=SERVE_MODEL,
                train_cutoff=fold.train_cutoff, val_start=fold.val_start,
                window_source=WINDOW_SOURCE, index_tag=DATA_INFO.get("index_tag"),
                glitch_repair=bool(GLITCH_REPAIR), val_end=VAL_END,
                # what a history must look like to be forecast (bottom-up and serving use the same rule as training)
                input_rule=dict(min_input_days=MIN_INPUT_DAYS, min_recent_days=MIN_RECENT_DAYS, recent_window=RECENT_WINDOW,
                                gap_fill=0 if WINDOW_SOURCE == "enriched" else 3,
                                labels=ENRICHED_LABELS if WINDOW_SOURCE == "enriched" else None),
                data_info={k: str(v) for k, v in DATA_INFO.items()},
                xgb_categories=xgb_categories(fold.Xg),
                **(extra or {}),
            ),
            fh, indent=2,
        )
    # for deployment: normals for the climatology; serving also needs the last year of
    # observed daily temperature (prev_anom_*) and the live 10-day forecast (fc_anom_*, base)
    mm_norms = ({} if clim.normal_min_all is None
                else dict(normal_min_all=clim.normal_min_all, normal_max_all=clim.normal_max_all))
    np.savez(os.path.join(out_dir, "climatology.npz"),
             coatos=np.array(clim.coatos), normal_all=clim.normal_all, **mm_norms)
    return out_dir


def print_importances(fold, n=25):
    print(f"\nTop {n} features (CatBoost):")
    print(fold.cb_imp.sort_values("importance", ascending=False).head(n).to_string(index=False))
    print(f"\nTop {n} features (XGBoost):")
    print(fold.xgb_imp.sort_values("importance", ascending=False).head(n).to_string(index=False))


def _pkg_versions(names):
    from importlib import metadata
    out = []
    for n in names:
        try:
            out.append(f"{n}=={metadata.version(n)}")
        except metadata.PackageNotFoundError:
            out.append(n)
    return out


def log_serving_model(tr, out_dir):
    """Logs the serving model (electricity_pyfunc.py, a models-from-code pyfunc) to the current MLflow run:
    the model files as artifacts, THIS file as code (so the feature code that trained the model is the code
    that serves it), and pinned package versions. Returns the model uri, or None."""
    import mlflow.pyfunc
    here = os.path.dirname(os.path.abspath(__file__))
    py = os.path.join(here, "electricity_pyfunc.py")
    if not os.path.exists(py):
        print("[mlflow] electricity_pyfunc.py is not next to this file: the serving model is not logged")
        return None
    out_dir = os.path.abspath(out_dir)
    artifacts = {"columns": os.path.join(out_dir, "columns.json"),
                 "catboost": os.path.join(out_dir, "catboost_model.cbm"),
                 "xgboost": os.path.join(out_dir, "xgboost_model.json"),
                 "segments": os.path.join(out_dir, "meter_segments.csv"),
                 "weather": os.path.join(out_dir, "weather_daily.parquet")}
    kw = dict(python_model=py, artifacts=artifacts, code_paths=[os.path.abspath(__file__)],
              pip_requirements=["mlflow"] + _pkg_versions(["numpy", "pandas", "scikit-learn", "catboost", "xgboost",
                                                            "pyarrow"]))

    def go():
        try:
            return mlflow.pyfunc.log_model(name="serving_model", **kw)           # MLflow 3
        except TypeError:
            return mlflow.pyfunc.log_model(artifact_path="serving_model", **kw)  # MLflow 2

    info = tr._safe("log serving model", go)
    return getattr(info, "model_uri", None) if info is not None else None


def serving_parity_check(model_uri, fold, pdf_meta, kin, n_rows=120):
    """Loads the LOGGED model and compares its forecasts with the in-memory models on the same test windows
    (same raw history, bundled weather, no live forecast, normal_mode='all'). Catches a broken save / load,
    category re-coding in XGBoost, a missing artifact or a feature-code mismatch BEFORE the model is registered.
    Returns the largest absolute difference in kWh per meter-day."""
    import mlflow.pyfunc
    test_idx = np.flatnonzero(fold.masks["test"])
    take = test_idx[:: max(1, len(test_idx) // n_rows)][:n_rows]
    sub = pdf_meta.iloc[take].reset_index(drop=True)
    hist = np.vstack([clean_recent_readings_for_serving(r) for r in kin[take]])
    meters = pd.DataFrame({"meter_id": sub["meter_id"].astype(str), "coato_code": sub["coato_code"],
                           "housing_type_id": sub["housing_type_id"], "origin_date": sub["origin_date"],
                           "kwh_history": [list(map(float, r)) for r in kin[take]]})
    model = mlflow.pyfunc.load_model(model_uri)
    got = model.predict({"meters": meters})
    got = got[[f"pred_{i}" for i in range(1, HORIZON + 1)]].to_numpy()

    # reference: the same path, in memory
    clim_ref = build_climate(_PARITY["weather"])
    seg = fold.meter_segment.copy()
    seg.index = seg.index.astype(str)
    ref_pdf = sub.assign(meter_id=sub["meter_id"].astype(str))
    bh, bc = row_priors_from_segments(ref_pdf, seg, fold.segment_priors, fold.pooled_prior)
    bh_mm = bc_mm = None
    if USE_MINMAX:
        bh_mm, bc_mm = row_priors_from_segments(ref_pdf, seg, fold.segment_priors_mm, fold.pooled_prior_mm)
    meta, feats, base, _ = build_features(
        ref_pdf, hist, np.zeros((len(sub), HORIZON)), clim_ref, future_mode="normal", normal_mode="all",
        beta_h_prior=bh, beta_c_prior=bc, forecast_mode="none", meter_segment=seg,
        beta_h_prior_mm=bh_mm, beta_c_prior_mm=bc_mm)
    X = pd.concat([feats, meta[CAT_FEATURES]], axis=1)
    for c in fold.feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[fold.feature_cols]
    model_ref = fold.cb_model if SERVE_MODEL == "catboost" else fold.xgb_model
    Xp = X if SERVE_MODEL == "catboost" else to_xgb_frame(X, xgb_categories(fold.Xg))
    want = np.maximum(base + model_ref.predict(Xp), 0.0)
    return float(np.nanmax(np.abs(got - want))), len(sub)


_PARITY = {}


# =============================================================================
# Promotion gate, registry
# =============================================================================
def _wape_by_horizon(df, horizons):
    out = {}
    for h in horizons:
        d = df[df["horizon"] == h]
        d = d[d["actual"] > 0]
        out[h] = float((d["pred"] - d["actual"]).abs().sum() / d["actual"].sum()) if len(d) else float("nan")
    return out


def evaluate_gate(cand_daily: pd.DataFrame, champ_daily: pd.DataFrame):
    """Candidate vs champion on the (target day, horizon) pairs BOTH were tested on, serve model only.
    Passes when, at every GATE_HORIZONS horizon, candidate sysWAPE <= champion sysWAPE * (1 + GATE_TOLERANCE),
    and at least GATE_MIN_COMMON_DAYS target days are shared. Returns (passed, table, reason)."""
    c = cand_daily[cand_daily["is_serve"] == 1].copy()
    p = champ_daily[champ_daily["is_serve"] == 1].copy()
    for d in (c, p):
        d["target_day"] = pd.to_datetime(d["target_day"])
    key = ["target_day", "horizon"]
    common = c[key].merge(p[key], on=key)
    rows, passed, reason = [], True, "ok"
    n_days = common["target_day"].nunique()
    if n_days < GATE_MIN_COMMON_DAYS:
        return None, pd.DataFrame(), (f"only {n_days} target days in common (< GATE_MIN_COMMON_DAYS="
                                      f"{GATE_MIN_COMMON_DAYS}); compare by hand and use `promote --force`")
    cm, pm = c.merge(common, on=key), p.merge(common, on=key)
    wc, wp_ = _wape_by_horizon(cm, GATE_HORIZONS), _wape_by_horizon(pm, GATE_HORIZONS)
    for h in GATE_HORIZONS:
        ok = bool(wc[h] <= wp_[h] * (1 + GATE_TOLERANCE))
        passed &= ok
        rows.append(dict(horizon=h, candidate_sysWAPE=wc[h], champion_sysWAPE=wp_[h],
                         relative_change=wc[h] / wp_[h] - 1 if wp_[h] else float("nan"), ok=ok))
    if not passed:
        reason = f"worse than champion by more than {GATE_TOLERANCE:.0%} at a gate horizon"
    return passed, pd.DataFrame(rows), f"{reason} | {n_days} common target days"


def run_gate(tr, name, cand_daily, cand_run_id=None):
    """Looks up the champion's recorded test table and evaluates the gate. Returns (passed, table, reason, champ_v)."""
    champ_v, champ_run = tr.version_by_alias(name, "champion")
    if champ_v is None:
        return None, pd.DataFrame(), "no champion yet", None
    path = tr.download_artifact(champ_run, "evaluation/test_system_daily.csv")
    if not path:
        return None, pd.DataFrame(), f"champion v{champ_v} has no recorded test table", champ_v
    passed, table, reason = evaluate_gate(cand_daily, pd.read_csv(path))
    return passed, table, reason, champ_v


def register_and_gate(tr, model_uri, fold, run_id, decision_log):
    """Registers the logged model, tags the version with its evidence, sets alias `challenger`, evaluates the
    gate against the current `champion` and promotes only if AUTO_PROMOTE and the gate passed."""
    tags = {"run_id": run_id, "train_cutoff": fold.train_cutoff, "index_tag": DATA_INFO.get("index_tag", ""),
            "serve_model": SERVE_MODEL, **{k: round(v, 5) for k, v in fold.metrics.items()
                                           if k.startswith("test/serve/sysWAPE_t")}}
    version = tr.register(model_uri, REGISTERED_MODEL_NAME, tags=None)
    if version is None:
        return None
    tr.set_version_tags(REGISTERED_MODEL_NAME, version, tags)
    tr.set_alias(REGISTERED_MODEL_NAME, "challenger", version)
    passed, table, reason, champ_v = run_gate(tr, REGISTERED_MODEL_NAME, fold.sys_daily)
    print(f"\n[registry] registered {REGISTERED_MODEL_NAME} v{version} (alias 'challenger')")
    if not table.empty:
        print(f"[registry] gate vs champion v{champ_v}:")
        print(table.round(4).to_string(index=False))
    decision = "no champion yet" if champ_v is None else ("PASS" if passed else
                                                           ("FAIL" if passed is False else "not comparable"))
    print(f"[registry] gate: {decision} ({reason})")
    promoted = False
    if AUTO_PROMOTE and passed is not False and (passed or champ_v is None):
        tr.set_alias(REGISTERED_MODEL_NAME, "champion", version)
        promoted = True
        print(f"[registry] AUTO_PROMOTE: v{version} is now 'champion'")
    elif champ_v is None or passed:
        print(f"[registry] to promote:  python whole_pipeline_v8.py promote --version {version}")
    tr.tags({"registered_version": version, "gate_decision": decision, "gate_reason": reason,
             "promoted": str(promoted)})
    if not table.empty:
        tr.dataframe(table, "evaluation/gate_vs_champion.csv")
    decision_log.update(version=version, decision=decision, promoted=promoted)
    return version


# =============================================================================
# Config snapshot, stdout capture
# =============================================================================
CONFIG_NAMES = [
    "LOOKBACK", "HORIZON", "SEED", "WINDOW_SOURCE", "WINDOW_BATCHES", "HDD_BASE", "CDD_BASE", "NORMAL_SMOOTH_DAYS",
    "FUTURE_MODE", "NORMAL_MODE", "USE_PREV_YEAR_FEATURES", "PREV_YEAR_LAG_DAYS", "FORECAST_MODE", "FORECAST_DAYS",
    "FC_BLOCKS", "FC_TAIL_DAYS", "FC_FADE_TAU", "FC_SIGMA_DAY1", "FC_SIGMA_DAY10", "FC_RHO", "USE_MINMAX",
    "HDD_MIN_BASE", "CDD_MAX_BASE", "MINMAX_MIN_HOURS", "WEATHER_TZ", "FC_MINMAX_EXTRA_FRAC", "REGIME_CUTOFF",
    "INCLUDE_OLD", "OLD_START", "OLD_EXCLUDE_TARGET_MONTHS", "OLD_WEIGHT", "OLD_REGIME_FLAG", "SEGMENT_USE_OLD",
    "EXCLUDE_RANGES", "EXCLUDE_MODE", "BETA_H_PRIOR", "BETA_C_PRIOR", "BETA_MAX", "RIDGE_LAMBDA", "LEVEL_DAYS",
    "BLOCKS", "GLITCH_WINDOW", "GLITCH_MIN_VALID", "GLITCH_DIP_RATIO", "GLITCH_DIP_MIN_MED", "GLITCH_SPIKE_RATIO",
    "GLITCH_SPIKE_MIN_EXCESS", "TRAIN_CUTOFF", "VAL_START", "VAL_END", "MAX_FINAL_ITERATIONS", "SEGMENT_K",
    "ENRICHED_ROOT", "ENRICHED_BUCKETS", "ENRICHED_COATOS", "ENRICHED_LABELS", "ENRICHED_START", "ENRICHED_END",
    "WINDOW_STRIDE", "MIN_INPUT_DAYS", "MIN_RECENT_DAYS", "RECENT_WINDOW", "MIN_TARGET_DAYS", "TRAIN_MIN_TARGET_DAYS",
    "GLITCH_REPAIR",
    "SEGMENT_MIN_COVERAGE_DAYS", "CATBOOST_EARLY_STOPPING_ROUNDS", "XGB_EARLY_STOPPING_ROUNDS", "SERVE_MODEL",
    "BACKTEST_CUTOFFS", "BACKTEST_TEST_DAYS",
]


def config_snapshot():
    g = globals()
    return {n: g[n] for n in CONFIG_NAMES if n in g}


class _Tee:
    """Copies everything printed to a file as well, so the whole training log can be kept as an MLflow artifact."""

    def __init__(self, stream, path):
        self.stream, self.fh = stream, open(path, "w", buffering=1)

    def write(self, s):
        self.stream.write(s)
        try:
            self.fh.write(s)
        except ValueError:
            pass

    def flush(self):
        self.stream.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)

    def close(self):
        self.fh.close()


# =============================================================================
# Logging a fold to MLflow
# =============================================================================
def log_fold(tr, fold, with_children=True):
    """Logs one fold into the CURRENT run: shared numbers and tables here, per-model detail (hyper-parameters,
    learning curves, importances, model file) in nested runs "catboost" / "xgboost"."""
    tr.params({"train_cutoff": fold.train_cutoff, "val_start": fold.val_start, "test_end": fold.test_end or "last"})
    tr.metrics({f"count/{k}": v for k, v in fold.counts.items()})
    tr.metrics(fold.metrics)
    tr.dataframe(fold.sys_daily, "evaluation/test_system_daily.csv")
    mo = (fold.meta["origin_date"].dt.to_period("M").astype(str).value_counts().sort_index()
          .rename_axis("origin_month").reset_index(name="windows_used"))
    tr.dataframe(mo, "data/windows_used_by_origin_month.csv")
    if not with_children:
        return
    for name, model, best, final, curves, imp, params in (
            ("catboost", fold.cb_model, fold.cb_best_iter, fold.cb_final_iters, fold.cb_curves, fold.cb_imp, CATBOOST_PARAMS),
            ("xgboost", fold.xgb_model, fold.xgb_best_iter, fold.xgb_final_iters, fold.xgb_curves, fold.xgb_imp, XGB_PARAMS)):
        with tr.run(name, nested=True, tags={"stage": "train_model", "model": name}):
            tr.params({**params, "best_iteration": best, "final_iterations": final})
            tr.metrics({"best_iteration": best, "final_iterations": final})
            tr.metrics({k.replace(f"test/{name}/", "test/"): v for k, v in fold.metrics.items()
                        if k.startswith(f"test/{name}/")})
            for split, series in (curves or {}).items():
                for metric, values in series.items():
                    tr.curve(f"{split}/{metric}", values)
            tr.dataframe(imp.sort_values("importance", ascending=False), "feature_importance.csv")


def log_backtest(tr, folds):
    """Aggregates the rolling-origin folds into the parent run (mean / std / worst per serve-model metric)."""
    if not folds:
        return
    keys = [k for k in folds[0].metrics if k.startswith("test/serve/") and ("sysWAPE" in k or "bias" in k)]
    agg = {}
    for k in keys:
        v = np.array([f.metrics[k] for f in folds if k in f.metrics], dtype=float)
        agg[k.replace("test/serve/", "backtest/") + "_mean"] = float(np.nanmean(v))
        agg[k.replace("test/serve/", "backtest/") + "_std"] = float(np.nanstd(v))
        agg[k.replace("test/serve/", "backtest/") + "_worst"] = float(np.nanmax(np.abs(v)) if "bias" in k else np.nanmax(v))
    agg["backtest/folds"] = len(folds)
    tr.metrics(agg)
    table = pd.DataFrame([{"cutoff": f.train_cutoff, "test_end": f.test_end, **{k: f.metrics[k] for k in keys}}
                          for f in folds])
    tr.dataframe(table, "evaluation/backtest_folds.csv")
    print("\n[backtest] serve-model sysWAPE per fold")
    print(table.round(4).to_string(index=False))


# =============================================================================
# Main: train
# =============================================================================
def main_train(tr):
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tee_path = None
    if tr.enabled:
        tee_path = os.path.join(tempfile.mkdtemp(prefix="v8log_"), "train_stdout.txt")
        sys.stdout = _Tee(sys.stdout, tee_path)
    decision = {}
    try:
        with tr.run(f"train {stamp}", tags={"stage": "train", **tr.system_tags()}) as run_id:
            tr.params(config_snapshot())
            tr.params({f"catboost.{k}": v for k, v in CATBOOST_PARAMS.items()})
            tr.params({f"xgboost.{k}": v for k, v in XGB_PARAMS.items()})

            spark = daily_t = None
            print(f"Reading windows (WINDOW_SOURCE={WINDOW_SOURCE!r}) + metadata...")
            if WINDOW_SOURCE == "enriched":
                globals()["GLITCH_REPAIR"] = False     # Layer-1 clean kwh is the truth: never repaired or dropped again
                if FORECAST_MODE == "archive":
                    raise ValueError("FORECAST_MODE='archive' needs Spark for FORECAST_S3; not wired for the enriched source")
                pdf, kin, ktg, daily_t = extract_windows_enriched()
            else:
                spark = build_spark_session()
                pdf, kin, ktg = extract_windows(spark)
            if WINDOW_SOURCE == "legacy":          # the index source applies the regime policy while reading
                pdf, kin, ktg = filter_regime(pdf, kin, ktg)
            n_regime_kept = len(pdf)
            if GLITCH_REPAIR:
                pdf, kin, ktg, ktg_raw = detect_and_repair_glitches(pdf, kin, ktg)
            else:
                ktg_raw = ktg.copy()
                LAST_GLITCH_STATS.update(windows_before=len(pdf), repaired_days=0, masked_days=0, windows_dropped=0)
                print("[glitch repair] off: the source is already cleaned; values are used as delivered")

            # lineage: which data-prep run made the store this model was trained on
            if DATA_INFO:
                tr.tags({f"data.{k}": v for k, v in DATA_INFO.items()})
                prep = tr.find_run_in(MLFLOW_DATA_PREP_EXPERIMENT, f"tags.index_tag = '{DATA_INFO.get('index_tag', '')}'")
                if prep is not None:
                    tr.tags({"data_prep_run_id": prep.info.run_id})
            tr.metrics({f"regime/{k}": v for k, v in LAST_REGIME_STATS.items()})
            tr.metrics({f"glitch/{k}": v for k, v in LAST_GLITCH_STATS.items()})
            tr.metrics({"windows_after_glitch_repair": len(pdf), "windows_after_regime_filter": n_regime_kept})

            fc_df = None
            if spark is not None:
                print("Loading daily weather...")
                daily_t = load_daily_temperature(spark)
                if FORECAST_MODE == "archive":
                    if not FORECAST_S3:
                        raise ValueError("FORECAST_MODE='archive' but FORECAST_S3 is not set")
                    print("Loading archived forecasts...")
                    fc_df = load_archived_forecasts(spark)
                spark.stop()
            else:
                print(f"[weather] from the enriched table: {daily_t['coato_id'].nunique()} districts, "
                      f"{daily_t['t'].notna().mean():.1%} of district-days with a mean temperature")
            print(f"Windows: {len(pdf)}")

            clim = build_climate(daily_t)
            fc_grid = build_forecast_grid(fc_df, clim) if fc_df is not None else None
            fc_grid_mm = None
            if fc_df is not None and USE_MINMAX:
                fc_grid_mm = dict(min=build_forecast_grid(fc_df, clim, col="t_fc_min"),
                                  max=build_forecast_grid(fc_df, clim, col="t_fc_max"))

            # ---------------- primary fold ----------------
            fold = fit_and_evaluate(pdf, kin, ktg, ktg_raw, clim, fc_grid, fc_grid_mm, TRAIN_CUTOFF, VAL_START, val_end=VAL_END)
            persist_model(fold, daily_t, clim, extra=dict(mlflow_run_id=run_id))
            print_importances(fold)
            log_fold(tr, fold)
            tr.artifact(os.path.join(OUTPUT_DIR, "columns.json"), subdir="model_files")
            tr.artifact(os.path.join(OUTPUT_DIR, "climatology.npz"), subdir="model_files")
            tr.artifact(os.path.join(OUTPUT_DIR, "meter_segments.csv"), subdir="model_files")

            # ---------------- serving model, parity check, registry ----------------
            if tr.enabled and LOG_AND_REGISTER_MODEL:
                _PARITY["weather"] = daily_t
                model_uri = log_serving_model(tr, OUTPUT_DIR)
                if model_uri:
                    try:
                        diff, n = serving_parity_check(model_uri, fold, fold.meta, kin)
                    except Exception as e:                      # noqa: BLE001
                        diff, n = float("inf"), 0
                        print(f"[serving] parity check could not run: {type(e).__name__}: {e}")
                    ok = diff <= 1e-6
                    tr.metrics({"serving/parity_max_abs_diff_kwh": diff, "serving/parity_rows": n})
                    tr.tags({"serving_parity": "OK" if ok else "FAILED"})
                    print(f"[serving] logged model vs in-memory model on {n} test windows: max |diff| = {diff:.3g} kWh "
                          f"-> {'OK' if ok else 'FAILED, NOT registered'}")
                    if ok:
                        register_and_gate(tr, model_uri, fold, run_id, decision)

            # ---------------- rolling-origin backtest (optional) ----------------
            bt_folds = []
            for entry in BACKTEST_CUTOFFS:
                # "CUTOFF" (validation shifted by the primary fold's offsets) or ("CUTOFF", "VAL_START", "VAL_END")
                cutoff, vs, ve = (tuple(entry) + (None, None))[:3] if isinstance(entry, (tuple, list)) else (entry, None, None)
                c = pd.Timestamp(cutoff)
                if vs is None:
                    vs = (c - (pd.Timestamp(TRAIN_CUTOFF) - pd.Timestamp(VAL_START))).strftime("%Y-%m-%d")
                    if VAL_END:
                        ve = (c - (pd.Timestamp(TRAIN_CUTOFF) - pd.Timestamp(VAL_END))).strftime("%Y-%m-%d")
                te = (c + pd.Timedelta(days=HORIZON + BACKTEST_TEST_DAYS)).strftime("%Y-%m-%d")
                with tr.run(f"backtest {cutoff}", nested=True, tags={"stage": "backtest"}):
                    try:
                        f = fit_and_evaluate(pdf, kin, ktg, ktg_raw, clim, fc_grid, fc_grid_mm, cutoff, vs, te,
                                             label=f"backtest {cutoff}", val_end=ve)
                    except ValueError as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[backtest] fold {cutoff} skipped: {e}")
                        continue
                    log_fold(tr, f, with_children=False)
                    bt_folds.append(f)
            log_backtest(tr, bt_folds)
            if decision:
                print(f"\n[done] MLflow run {run_id} | model v{decision.get('version')} | gate: {decision.get('decision')}")
            if tee_path:
                sys.stdout.flush()
                tr.artifact(tee_path, subdir="logs")
    finally:
        if tee_path and isinstance(sys.stdout, _Tee):
            sys.stdout.close()
            sys.stdout = sys.stdout.stream
    return decision


# =============================================================================
# promote / monitor sub-commands
# =============================================================================
def main_promote(tr, args):
    """Sets alias `champion`. Without --version: the current `challenger`. The gate is evaluated first and
    must pass unless --force."""
    if not tr.enabled:
        raise SystemExit("promote needs MLflow: set MLFLOW_TRACKING_URI or --mlflow-uri")
    version = args.version
    run_id = None
    if version is None:
        version, run_id = tr.version_by_alias(REGISTERED_MODEL_NAME, "challenger")
        if version is None:
            raise SystemExit("no `challenger` alias on the registered model; pass --version N")
    else:
        mv = tr.client.get_model_version(REGISTERED_MODEL_NAME, str(version))
        run_id = mv.run_id
    path = tr.download_artifact(run_id, "evaluation/test_system_daily.csv")
    cand = pd.read_csv(path) if path else None
    champ_v, _ = tr.version_by_alias(REGISTERED_MODEL_NAME, "champion")
    if champ_v is None:
        print(f"no champion yet: v{version} becomes the first champion")
    elif cand is None:
        print("the candidate has no recorded test table")
        if not args.force:
            raise SystemExit("refusing without --force")
    else:
        passed, table, reason, _ = run_gate(tr, REGISTERED_MODEL_NAME, cand)
        if not table.empty:
            print(table.round(4).to_string(index=False))
        print(f"gate vs champion v{champ_v}: {reason}")
        if passed is not True and not args.force:
            raise SystemExit("gate not passed (or not comparable); use --force to promote anyway")
    tr.set_alias(REGISTERED_MODEL_NAME, "champion", version)
    print(f"{REGISTERED_MODEL_NAME} v{version} is now 'champion'")


def monitor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per horizon: sysMAPE, sysWAPE and signed bias of the daily SYSTEM totals (sum over meters) in `df`
    (columns target_day, horizon, pred_kwh, actual_kwh)."""
    rows = []
    for h, d in df[df["actual_kwh"] > 0].groupby("horizon"):
        err = d["pred_kwh"] - d["actual_kwh"]
        rows.append(dict(horizon=int(h), days=int(len(d)), sysMAPE=float((err.abs() / d["actual_kwh"]).mean()),
                         sysWAPE=float(err.abs().sum() / d["actual_kwh"].sum()),
                         bias=float(err.sum() / d["actual_kwh"].sum())))
    return pd.DataFrame(rows)


def main_monitor(args):
    """Scores production forecasts against actuals and logs the result as a run in the monitoring experiment,
    tagged with the model version. Input file (csv / parquet), one row per (target day, horizon) with the
    SYSTEM totals: target_day, horizon, pred_kwh, actual_kwh. Compared with the champion's test numbers."""
    tr = Tracker(MLFLOW_MONITOR_EXPERIMENT, uri=args.mlflow_uri, enabled=False if args.no_mlflow else None) \
        if Tracker is not None else None
    df = pd.read_parquet(args.file) if args.file.endswith(".parquet") else pd.read_csv(args.file)
    need = {"target_day", "horizon", "pred_kwh", "actual_kwh"}
    if need - set(df.columns):
        raise SystemExit(f"{args.file} needs the columns {sorted(need)}; missing {sorted(need - set(df.columns))}")
    df["target_day"] = pd.to_datetime(df["target_day"])
    table = monitor_frame(df)
    print(table.round(4).to_string(index=False))
    if tr is None or not tr.enabled:
        return
    reg = tr        # the registry does not depend on the experiment; a second Tracker would switch the active one
    version, run_id = (args.version, None) if args.version else reg.version_by_alias(REGISTERED_MODEL_NAME, args.alias)
    base = {}
    if version is not None:
        if run_id is None:
            try:
                run_id = reg.client.get_model_version(REGISTERED_MODEL_NAME, str(version)).run_id
            except Exception:                                  # noqa: BLE001
                run_id = None
        if run_id:
            r = reg.client.get_run(run_id)
            base = {k: v for k, v in r.data.metrics.items() if k.startswith("test/serve/")}
    with tr.run(f"monitor {df['target_day'].min().date()}..{df['target_day'].max().date()}",
                tags={"stage": "monitor", "model_name": REGISTERED_MODEL_NAME, "model_version": str(version),
                      "model_alias": args.alias, "scored_run_id": str(run_id), **tr.system_tags()}):
        for _, r in table.iterrows():
            h = int(r["horizon"])
            tr.metrics({f"live/sysMAPE_t{h}": r["sysMAPE"], f"live/sysWAPE_t{h}": r["sysWAPE"],
                        f"live/bias_t{h}": r["bias"], f"live/days_t{h}": r["days"]})
            b = base.get(f"test/serve/sysWAPE_t{h}")
            if b is not None:
                tr.metrics({f"drift/sysWAPE_t{h}_vs_test": r["sysWAPE"] - b})
        # error per target day, drawn over real time (one curve per horizon)
        for h, d in df[df["actual_kwh"] > 0].groupby("horizon"):
            d = d.sort_values("target_day")
            tr.series(f"daily/abs_pct_error_t{int(h)}",
                      [(t.timestamp(), i, abs(p - a) / a) for i, (t, p, a) in
                       enumerate(zip(d["target_day"], d["pred_kwh"], d["actual_kwh"]))])
        tr.dataframe(table, "monitor_summary.csv")
    print(f"logged to MLflow experiment {MLFLOW_MONITOR_EXPERIMENT!r}")


def build_parser():
    p = argparse.ArgumentParser(description="Feature engineering + CatBoost / XGBoost training, MLflow tracking, "
                                            "registry promotion and monitoring.")
    p.add_argument("--mlflow-uri", default=None, help="MLflow server / store (default: MLFLOW_TRACKING_URI; "
                                                      "without one tracking is off)")
    p.add_argument("--no-mlflow", action="store_true", help="tracking off even if a URI is set")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a configuration constant, e.g. --set ENRICHED_BUCKETS=[0,1] --set OUTPUT_DIR=./artifacts/hh_enriched "
                        "(Python literal; plain text otherwise). Repeatable")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("train", help="train, evaluate, log, register (the default)")
    pr = sub.add_parser("promote", help="make a registered version the champion (after the gate)")
    pr.add_argument("--version", default=None, help="model version (default: the 'challenger')")
    pr.add_argument("--force", action="store_true", help="promote even if the gate fails / cannot compare")
    mo = sub.add_parser("monitor", help="score production forecasts and log them")
    mo.add_argument("--file", required=True, help="csv / parquet: target_day, horizon, pred_kwh, actual_kwh")
    mo.add_argument("--alias", default="champion", help="registered-model alias that produced the forecasts")
    mo.add_argument("--version", default=None, help="model version instead of an alias")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    import ast
    for item in args.set:
        name, _, raw = item.partition("=")
        name = name.strip()
        if not name.isupper() or name not in globals():
            raise SystemExit(f"--set {item!r}: {name!r} is not a configuration constant of this file")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        globals()[name] = value
        print(f"[config] {name} = {value!r}")
    cmd = args.cmd or "train"
    if cmd == "monitor":
        return main_monitor(args)
    tr = (Tracker(MLFLOW_EXPERIMENT, uri=args.mlflow_uri, enabled=False if args.no_mlflow else None)
          if Tracker is not None else None)
    if tr is None:
        class _Off:                                        # mlflow_utils.py missing: tracking off
            enabled, root_run_id, uri = False, None, ""

            def system_tags(self):
                return {}

            def __getattr__(self, name):
                import contextlib
                if name == "run":
                    return lambda *a, **k: contextlib.nullcontext(None)
                return lambda *a, **k: None
        tr = _Off()
    if cmd == "promote":
        return main_promote(tr, args)
    return main_train(tr)


if __name__ == "__main__":
    main()
