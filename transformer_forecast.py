#!/usr/bin/env python3
"""
Transformer-level demand forecasting: bottom-up vs direct, scored on the same days.

  bottom-up  the household model (whole_pipeline_v8.py) forecasts every household on a transformer; the
             forecasts are summed.  No new training.
  direct     each transformer is one "meter": its daily kWh per household is the series, with the same weather /
             degree-day features, a thermal baseline and a CatBoost residual model as the household pipeline.
             Household count enters as size features and as the scale of the result.
  compare    both (and their average) against persistence, on the SAME transformer-origins and target days,
             by horizon (7, 14, 30, 60), by transformer-size band, and for the system total.

Run it in three steps.  Only `prepare` needs Spark / S3; the others need numpy, pandas, catboost, matplotlib.

  # 1. pull the households of one coato from the clean daily table (written by process_raw.py)
  source ~/work/ec_env.sh
  python transformer_forecast.py prepare --coato 26266 --work-dir ~/work/tf \
         --cores 6 --driver-memory 24g --local-dir /tmp/spark_tmp
  # (national sample for the noise audit: 1000 transformers over all coatos, EVERY household on each; no --batches)
  python transformer_forecast.py prepare --n-transformers 1000 --work-dir ~/work/tf_national \
         --cores 6 --driver-memory 24g --local-dir /tmp/spark_tmp

  # 2. fit the direct model, run the household model bottom-up, compare (each stage is saved; re-run one alone)
  python transformer_forecast.py run --work-dir ~/work/tf --hh-model-dir ./artifacts/model_v8
  python transformer_forecast.py run --work-dir ~/work/tf --hh-model-dir ./artifacts/model_v8 --stages compare

  # 2b. NATIONAL run on the cleaned panel (needs `transformer_clean.py --work-dir ...` first; many coatos at once):
  python transformer_forecast.py run --work-dir ~/work/tf_national_v2 --stages direct,compare
  #     (+ bottom-up once a household model exists:  --hh-model-dir <v8 OUTPUT_DIR>  and  --stages direct,bottomup,compare)
  #  If <work-dir>/clean/transformer_clean.npz exists, `run` uses it (override with --source raw).  Outputs go to
  #  <work-dir>/forecast/ (override with --out-dir).

Put this file next to whole_pipeline_v8.py, process_raw.py and (optionally) mlflow_utils.py.
Everything printed is also written to <work-dir>/run_log.txt, with a [stage] line and the process memory at the
start of each stage, so after a crash the last [stage] line says where it stopped.

What the numbers mean (read before trusting them)
-------------------------------------------------
* The unit is kWh per DAY, not peak load (kW).  A transformer's capacity is a peak-load question; this is energy.
* `transformer_id` comes from customer_metadata_current, i.e. the CURRENT state.  Households moved since are
  mis-assigned in the past.  The ids are opaque (they do not match the CAS network-node table).
* Only households with DAILY readings exist in the data.  The "transformer total" is therefore a panel total:
  mean kWh of the households that reported that day x the number of households that ever report (n_hh).  Days
  with fewer than --min-coverage of the panel reporting are treated as missing, not as a low total.  Both methods
  forecast, and are scored against, this same panel total, so the comparison is fair, but it is not the metered
  load of the transformer.
* Scored windows follow the household pipeline: train cutoff / validation start come from the household model,
  test origins start HORIZON days after the cutoff, so no test target day was seen in training by either model.
* Weather: the same climate normals, degree days and the same (simulated or archived) 10-day forecast, drawn ONCE
  per (coato, origin) and shared by both methods.
* On the cleaned panel (`--source clean`) the target is the MEAN OF THE STABLE CORE households per day times the core size;
  days flagged as system / regional / gap / outside / local dip by transformer_clean.py are missing, not low, and are not scored.
  Bottom-up then sums only the same core households, so both methods forecast the same quantity.
* Calendar (--calendar dow+holidays, default): weekday and holiday factors are estimated from the TRAINING period only
  (days up to the household model's train cutoff), the series is divided by them before modelling, and the forecasts are multiplied
  back on the target days. Persistence and bottom-up get the same factors, so the comparison stays like for like
  (`bottom_up_nocal` in metrics.csv is the household sum without them). The factors, and what each holiday date did in the data,
  are printed and written to calendar_factors.csv / calendar_holidays.csv. The moving holidays (Ramazon / Qurbon hayit) are typed in
  from memory: check them against the printed effects and fix them with --holidays-csv.
* No look-ahead in the inputs: a gap still open at a forecast's origin is carried forward from the last known value (for at most
  --gap-fill days), never interpolated with days after the origin; gaps that closed before the origin are interpolated as before.
  Run transformer_clean.py with --fit-end <first cutoff> so the stable core is also chosen without test-period data.
* Rolling folds (--folds): each fold trains up to its own cutoff and forecasts the months after it, so several folds test a
  full year (summer included) instead of one season. compare reports every fold together and writes metrics_by_fold.csv.
    python transformer_clean.py --work-dir ~/work/tf_national_v2 --fit-end 2025-03-01
    python transformer_forecast.py run --work-dir ~/work/tf_national_v2 --stages direct,compare \\
        --folds 2025-03-01:2024-08-01:2025-01-31,2025-09-01:2024-09-01:2025-04-30 --out-dir ~/work/tf_national_v2/forecast_folds
* The direct model only has the history you give `prepare` (default: everything in the daily table).  With about
  one year of history it sees one seasonal cycle; give it >= 2 years if you can.
"""
import argparse
import contextlib
import datetime as dt
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = "s3a://phase1/electricity/analytics/household"
DEFAULT_COATO = "26266"
HORIZONS = (7, 14, 30, 60)

# chart colours: neutral baseline, then three that pass the colour-blind / contrast check
C_BASE, C_BU, C_DIRECT, C_AVG, INK, MUTED, GRID = "#8a93a0", "#2563b8", "#c2670f", "#2f855a", "#1f2430", "#6b7280", "#e5e7eb"
MODEL_STYLE = {            # name -> (label, colour, marker, linestyle)
    "persistence30": ("Persistence (30-day mean)", C_BASE, "x", (0, (4, 3))),
    "bottom_up": ("Bottom-up", C_BU, "o", "-"),
    "direct": ("Direct", C_DIRECT, "s", "-"),
    "average": ("Average of the two", C_AVG, "^", "-"),
}


def wp_module():
    """whole_pipeline_v8 (features, climate, CatBoost settings).  Imported late so `--help` works anywhere."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import whole_pipeline_v8 as wp
    except ImportError as e:                                           # noqa: BLE001
        raise SystemExit(f"put this file next to whole_pipeline_v8.py (it reuses its features): {e}")
    return wp


# =============================================================================
# Logging with stages
# =============================================================================
def rss_gb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


class _Tee:
    def __init__(self, stream, path):
        self.stream, self.fh = stream, open(path, "a", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def logging_to(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, path), _Tee(err, path)
    print(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(sys.argv)} =====")
    try:
        yield
    except BaseException as e:                                         # noqa: BLE001
        print(f"[CRASH] {type(e).__name__}: {e}")
        raise
    finally:
        sys.stdout.flush()
        sys.stdout, sys.stderr = out, err


@contextlib.contextmanager
def stage(name):
    t0 = time.time()
    print(f"\n[stage] {name} | start {dt.datetime.now():%H:%M:%S} | memory {rss_gb():.1f} GB", flush=True)
    yield
    print(f"[stage] {name} | done in {time.time() - t0:,.0f}s | memory {rss_gb():.1f} GB", flush=True)


def ts_days(days, d0):
    """day index (int) relative to d0 -> datetime64[D]"""
    return (np.datetime64(d0, "D") + np.asarray(days).astype("int64")).astype("datetime64[D]")


# =============================================================================
# prepare: Spark -> dense household x day matrix for ONE coato
# =============================================================================
def spark_session(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from process_raw import build_spark_session
    except ImportError as e:                                           # noqa: BLE001
        raise SystemExit(f"put this file next to process_raw.py (it reuses its S3A / Spark settings): {e}")
    spark = build_spark_session(True, app_name="transformer-forecast", cores=args.cores,
                                driver_memory=args.driver_memory, local_dir=args.local_dir,
                                shuffle_partitions=256)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark


def households_of_coato(spark, coato):
    from pyspark.sql import functions as F
    meta = (spark.read.option("recursiveFileLookup", "true").option("pathGlobFilter", "*.parquet")
            .parquet(f"{ROOT}/customer_metadata_current/data/"))
    code = F.regexp_replace(F.trim(F.col("coato_code").cast("string")), r"\.0+$", "")
    hh = (meta.filter(code.isin([c.strip() for c in str(coato).split(",")]))
          .filter(F.col("transformer_id").isNotNull() & F.col("meter_id").isNotNull())
          .select(F.col("meter_id").cast("string").alias("meter_id"),
                  F.col("consumer_id").cast("string").alias("consumer_id"),
                  code.alias("coato_code"),
                  F.col("housing_type_id").cast("string").alias("housing_type_id"),
                  F.col("transformer_id").cast("string").alias("transformer_id"))
          .dropDuplicates(["meter_id"]).toPandas())
    if hh.empty:
        cnt = (meta.withColumn("code", code).groupBy("code").count().toPandas()
               .sort_values("count", ascending=False))
        print(f"no households with coato_code {coato}; biggest codes in the table:\n{cnt.head(15).to_string(index=False)}")
        raise SystemExit(1)
    return hh


def allocate_quota(avail, n, spread="sqrt"):
    """How many transformers to draw from each coato. avail: {coato: eligible transformers}.
    'sqrt' weights a coato by sqrt(its transformers) (big regions count more, but do not swamp small ones),
    'even' gives every coato the same quota, 'proportional' follows the real mix.  A coato never gives more than it has;
    what it cannot give is redistributed.  Returns {coato: quota} summing to min(n, total)."""
    w0 = {c: (1.0 if spread == "even" else float(a) if spread == "proportional" else float(a) ** 0.5) for c, a in avail.items()}
    n = min(int(n), int(sum(avail.values())))
    quota, open_ = {c: 0 for c in avail}, set(avail)
    left = n
    while left > 0 and open_:
        tot = sum(w0[c] for c in open_)
        want = {c: left * w0[c] / tot for c in open_}
        done = [c for c in open_ if want[c] >= avail[c] - quota[c]]
        if done:                                                         # saturated: take everything, redistribute the rest
            for c in done:
                left -= avail[c] - quota[c]
                quota[c] = avail[c]
                open_.discard(c)
            continue
        base = {c: int(want[c]) for c in open_}
        rest = left - sum(base.values())
        for c in sorted(open_, key=lambda c: -(want[c] - base[c]))[:rest]:   # largest remainders
            base[c] += 1
        for c in open_:
            quota[c] += base[c]
        left = 0
    return quota


def select_transformers(spark, wp, args):
    """Sample transformer ids from the whole country (or the coatos in --coato) and return ALL their households.
    Only transformers whose households really have daily readings can be drawn: most of the country is not on daily
    meters, and a random id is usually mostly monthly-metered. A first pass over the daily table counts, per transformer,
    the meters with a reading in the last --active-days (writes daily_coverage_by_coato.csv)."""
    from functools import reduce
    from pyspark.sql import functions as F
    meta = (spark.read.option("recursiveFileLookup", "true").option("pathGlobFilter", "*.parquet")
            .parquet(f"{ROOT}/customer_metadata_current/data/"))
    code = F.regexp_replace(F.trim(F.col("coato_code").cast("string")), r"\.0+$", "")
    base = (meta.filter(F.col("transformer_id").isNotNull() & F.col("meter_id").isNotNull())
            .select(F.col("meter_id").cast("string").alias("meter_id"), code.alias("coato_code"),
                    F.col("transformer_id").cast("string").alias("transformer_id"))
            .dropDuplicates(["meter_id"]))
    if args.coato and args.coato != "all":
        base = base.filter(F.col("coato_code").isin([c.strip() for c in str(args.coato).split(",")]))
    if not args.no_census:
        end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
        since = str((end - pd.Timedelta(days=args.active_days)).date())
        _, daily_path, _, batches = wp.resolve_window_store(spark)
        print(f"[census] meters with a daily reading since {since}, over {len(batches)} batches (one pass, two columns) ...", flush=True)
        active = reduce(lambda a, b: a.unionByName(b),
                        [spark.read.parquet(f"{daily_path}/batch={b:03d}").filter(F.col("calendar_day") >= F.lit(since))
                         .select(F.col("meter_id").cast("string").alias("meter_id")).distinct() for b in batches])
        base = base.join(active.withColumn("has_daily", F.lit(1)), on="meter_id", how="left").fillna({"has_daily": 0})
    else:
        base = base.withColumn("has_daily", F.lit(1))
    ids = (base.groupBy("transformer_id")
           .agg(F.count("*").alias("n_on_id"), F.sum("has_daily").alias("n_daily"),
                F.countDistinct("coato_code").alias("n_coatos"), F.first("coato_code").alias("coato_code"))
           .toPandas())
    ids["n_daily"] = ids["n_daily"].astype(int)
    by_coato = ids.groupby("coato_code").agg(transformers=("transformer_id", "size"), households=("n_on_id", "sum"),
                                              households_with_daily=("n_daily", "sum"),
                                              transformers_with_20_daily=("n_daily", lambda x: int((x >= args.min_daily).sum())))
    by_coato["daily_share"] = by_coato["households_with_daily"] / by_coato["households"].clip(lower=1)
    by_coato.sort_values("households", ascending=False).to_csv(os.path.join(args.work_dir, "daily_coverage_by_coato.csv"))
    n_all = len(ids)
    multi = ids["n_coatos"] > 1
    small = ids["n_on_id"] < args.min_on_id
    big = ids["n_on_id"] > args.max_on_id
    nodaily = ids["n_daily"] < args.min_daily
    pool = ids[~multi & ~small & ~big & ~nodaily]
    print(f"{n_all:,} transformer ids in the metadata; households on them with daily readings: "
          f"{ids['n_daily'].sum() / max(ids['n_on_id'].sum(), 1):.1%} overall, {int((by_coato['daily_share'] > 0.5).sum())} of {len(by_coato)} coatos above 50%")
    print(f"  dropped: {int(multi.sum()):,} span several coatos, {int(small.sum()):,} have < {args.min_on_id} households, "
          f"{int(big.sum()):,} have > {args.max_on_id} (placeholders), {int((nodaily & ~multi & ~small & ~big).sum()):,} have < {args.min_daily} "
          f"households with daily readings -> {len(pool):,} eligible in {pool['coato_code'].nunique():,} coatos")
    if pool.empty:
        raise SystemExit("no eligible transformers; lower --min-daily / --min-on-id, widen --active-days, or check --coato")
    avail = pool.groupby("coato_code").size().to_dict()
    quota = allocate_quota(avail, args.n_transformers, args.spread)
    rng = np.random.default_rng(args.seed)
    picked = []
    for c, q in quota.items():
        if q:
            sub = pool[pool["coato_code"] == c].sort_values("transformer_id")
            picked.append(sub.iloc[np.sort(rng.choice(len(sub), size=q, replace=False))])
    sel = pd.concat(picked).sort_values(["coato_code", "transformer_id"]).reset_index(drop=True)
    est = int(sel["n_on_id"].sum())
    print(f"sampled {len(sel):,} transformers from {sel['coato_code'].nunique():,} coatos ({args.spread} spread, seed {args.seed}): "
          f"{est:,} households on them ({int(sel['n_daily'].sum()):,} with daily readings); households with daily readings per "
          f"transformer: median {sel['n_daily'].median():.0f}, 10th pct {sel['n_daily'].quantile(.1):.0f}, max {sel['n_daily'].max()}")
    gb = est * 1700 * 4 / 1e9
    print(f"expected size of the household x day matrix: about {gb:.1f} GB in memory (float32)")
    if est > args.max_meters:
        raise SystemExit(f"{est:,} households exceed --max-meters {args.max_meters:,}; lower --n-transformers or raise --max-meters")
    sel.to_csv(os.path.join(args.work_dir, "transformers_sampled.csv"), index=False)
    hh = (meta.filter(F.col("transformer_id").cast("string").isin(sel["transformer_id"].tolist()))
          .filter(F.col("meter_id").isNotNull())
          .select(F.col("meter_id").cast("string").alias("meter_id"),
                  F.col("consumer_id").cast("string").alias("consumer_id"),
                  code.alias("coato_code"),
                  F.col("housing_type_id").cast("string").alias("housing_type_id"),
                  F.col("transformer_id").cast("string").alias("transformer_id"))
          .dropDuplicates(["meter_id"]).toPandas())
    return hh


def cmd_prepare(args):
    from pyspark.sql import functions as F
    wp = wp_module()
    os.makedirs(args.work_dir, exist_ok=True)
    spark = spark_session(args)
    # prepare must see EVERY household of a transformer: a batch subset (WINDOW_BATCHES in whole_pipeline_v8.py) is a random
    # share of the meters and leaves every transformer with a fraction of its households.  Only --batches narrows it here.
    wp.WINDOW_BATCHES = [int(b) for b in args.batches.split(",")] if args.batches else None
    with stage("households and transformers"):
        if args.n_transformers:
            hh = select_transformers(spark, wp, args)
            scope = "all coatos" if (not args.coato or args.coato == "all") else f"coatos {args.coato}"
        else:
            hh = households_of_coato(spark, args.coato or DEFAULT_COATO)
            scope = f"coato {args.coato or DEFAULT_COATO}"
        n_tr = hh["transformer_id"].nunique()
        print(f"{len(hh):,} households on {n_tr:,} transformer ids ({scope}; {hh['coato_code'].nunique()} coatos)")
        hh["mcode"] = np.arange(len(hh), dtype="int32")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    d0 = int((start - pd.Timestamp("1970-01-01")).days)
    n_days = int((end - start).days) + 1
    kwh = np.full((len(hh), n_days), np.nan, dtype="float32")

    with stage("daily table -> matrix"):
        _, daily_path, _, batches = wp.resolve_window_store(spark)
        meters = F.broadcast(spark.createDataFrame(hh[["meter_id", "mcode"]]))
        for b in batches:
            d = (spark.read.parquet(f"{daily_path}/batch={b:03d}")
                 .filter((F.col("calendar_day") >= F.lit(str(start.date()))) & (F.col("calendar_day") <= F.lit(str(end.date()))))
                 .join(meters, on="meter_id", how="inner")
                 .select("mcode", F.datediff("calendar_day", F.lit("1970-01-01")).alias("day"),
                         F.col("daily_kwh").cast("float").alias("kwh"))
                 .toPandas())
            if len(d):
                kwh[d["mcode"].to_numpy(), d["day"].to_numpy() - d0] = d["kwh"].to_numpy()
            print(f"[prepare] batch {b:03d}: {len(d):,} household-days", flush=True)
            del d

    with stage("weather" + (" + forecast archive" if wp.FORECAST_MODE == "archive" else "")):
        wp.load_daily_temperature(spark).to_parquet(os.path.join(args.work_dir, "weather_daily.parquet"), index=False)
        if wp.FORECAST_MODE == "archive":
            wp.load_archived_forecasts(spark).to_parquet(os.path.join(args.work_dir, "forecast_archive.parquet"), index=False)
    spark.stop()

    have = np.isfinite(kwh).any(axis=1)
    print(f"[prepare] {int(have.sum()):,} of {len(hh):,} households on the sampled transformers have at least one reading in the range "
          f"({have.mean():.0%}).")
    if args.n_transformers and have.mean() < 0.6:
        print("[prepare] WARNING: far fewer households than the metadata lists. Either many are not on daily meters, or the daily table "
              "holds only some of the hash batches (see the '[windows] ... batch(es)' line above). Check before trusting any audit.")
    cols = np.flatnonzero(np.isfinite(kwh).any(axis=0))
    if not have.any():
        raise SystemExit("the daily table has no rows for these households in the date range")
    c0, c1 = int(cols.min()), int(cols.max())
    kwh, hh = kwh[have][:, c0:c1 + 1], hh[have].drop(columns="mcode").reset_index(drop=True)
    first = (pd.Timestamp("1970-01-01") + pd.Timedelta(days=d0 + c0)).date()
    np.save(os.path.join(args.work_dir, "hh_kwh.npy"), kwh)
    hh.to_csv(os.path.join(args.work_dir, "hh_meta.csv"), index=False)
    with open(os.path.join(args.work_dir, "hh_days.json"), "w") as fh:
        json.dump(dict(start=str(first), n_days=int(kwh.shape[1]),
                       coato=",".join(sorted(hh["coato_code"].unique())) if hh["coato_code"].nunique() <= 5 else "multi",
                       n_coatos=int(hh["coato_code"].nunique()), sampled=bool(args.n_transformers)), fh)
    print(f"\nsaved {kwh.shape[0]:,} households x {kwh.shape[1]:,} days ({first} ..) to {args.work_dir}; "
          f"{np.isfinite(kwh).mean():.1%} of cells hold a reading")


# =============================================================================
# The transformer panel (pure numpy / pandas)
# =============================================================================
def load_world(work_dir):
    with open(os.path.join(work_dir, "hh_days.json")) as fh:
        info = json.load(fh)
    hh = pd.read_csv(os.path.join(work_dir, "hh_meta.csv"), dtype=str)
    kwh = np.load(os.path.join(work_dir, "hh_kwh.npy"), mmap_mode="r")
    assert kwh.shape[0] == len(hh), "hh_kwh.npy and hh_meta.csv do not match: re-run prepare"
    return info, hh, kwh


def fill_short_gaps(x, max_gap):
    """Linear interpolation of NaN runs of at most max_gap days that have a value on both sides. x: (n, T)."""
    out = x.copy()
    for i in range(x.shape[0]):
        s = pd.Series(x[i])
        filled = s.interpolate(limit_area="inside").to_numpy()
        nan = np.isnan(x[i])
        if nan.any():
            edges = np.diff(np.r_[0, nan.astype(int), 0])
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            for a, b in zip(starts, ends):
                if b - a <= max_gap:
                    out[i, a:b] = filled[a:b]
    return out


def build_panel(wp, info, hh, kwh, args):
    """Per transformer and day: kWh per reporting household (intensity), coverage, and the household count."""
    d0 = pd.Timestamp(info["start"])
    n_days = kwh.shape[1]
    fin_days = np.concatenate([np.isfinite(kwh[a:a + 5000]).sum(axis=1) for a in range(0, kwh.shape[0], 5000)])
    in_panel = fin_days >= args.min_hh_days
    size_on_id = hh.groupby("transformer_id")["meter_id"].transform("size").to_numpy()
    placeholder = size_on_id > args.max_households
    keep = in_panel & ~placeholder
    print(f"[panel] {len(hh):,} households; {int((~in_panel).sum()):,} dropped (< {args.min_hh_days} days of readings); "
          f"{int((placeholder & in_panel).sum()):,} on transformer ids with more than {args.max_households} households "
          f"(placeholders such as 'unknown')")
    h = hh[keep].reset_index().rename(columns={"index": "row"})
    sizes = h.groupby("transformer_id")["row"].size()
    ids = sorted(sizes[sizes >= args.min_households].index, key=str)
    if args.max_transformers:
        rng = np.random.default_rng(0)
        ids = sorted(rng.choice(ids, size=min(args.max_transformers, len(ids)), replace=False).tolist())
    print(f"[panel] {len(ids):,} transformers with >= {args.min_households} households")

    n_tr = len(ids)
    inten = np.full((n_tr, n_days), np.nan)
    cover = np.zeros((n_tr, n_days))
    n_hh = np.zeros(n_tr, dtype=int)
    housing = []
    groups = h.groupby("transformer_id")
    for k, tid in enumerate(ids):
        g = groups.get_group(tid)
        block = np.asarray(kwh[g["row"].to_numpy()], dtype="float64")
        fin = np.isfinite(block)
        n_rep = fin.sum(0)
        n_hh[k] = len(g)
        cover[k] = n_rep / len(g)
        ok = cover[k] >= args.min_coverage
        inten[k, ok] = np.where(fin, block, 0.0).sum(0)[ok] / n_rep[ok]
        mode = g["housing_type_id"].dropna()
        housing.append(mode.mode().iloc[0] if len(mode) else "Missing")
    dates = d0 + pd.to_timedelta(np.arange(n_days), unit="D")
    for a, b in wp.EXCLUDE_RANGES:                                    # same calendar exclusions as the household model
        inten[:, (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))] = np.nan
    filled = fill_short_gaps(inten, args.gap_fill)
    print(f"[panel] share of transformer-days kept (coverage >= {args.min_coverage:.0%}): {np.isfinite(inten).mean():.1%}; "
          f"after filling gaps <= {args.gap_fill} days: {np.isfinite(filled).mean():.1%}")
    return SimpleNamespace(ids=np.asarray(ids, dtype=object), n_hh=n_hh, inten=inten, filled=filled, cover=cover,
                           housing=np.asarray(housing, dtype=object), d0=d0, n_days=n_days, rows=h,
                           hh_rows={tid: groups.get_group(tid)["row"].to_numpy() for tid in ids})


def origin_grid(wp, panel, args):
    """Origin day indices usable by both methods: input 90 days back, target 60 days ahead, all inside the data."""
    L, H = wp.LOOKBACK, wp.HORIZON
    O = np.arange(L - 1, panel.n_days - H)
    return O[(O % args.origin_stride) == 0]


def gap_parts(x):
    """For each NaN cell of x (n, T): where its NaN run starts (run_a) and the last value before the run (locf).
    run_a = -1 for observed cells. Used to fill the input window without looking past the origin (causal_input)."""
    x = np.asarray(x, dtype="float64")
    n, T = x.shape
    t = np.broadcast_to(np.arange(T), (n, T))
    fin = np.isfinite(x)
    last = np.maximum.accumulate(np.where(fin, t, -1), axis=1)          # last observed index <= t (-1 = none yet)
    run_a = np.where(fin, -1, last + 1)
    locf = np.where(last >= 0, np.take_along_axis(x, np.maximum(last, 0), axis=1), np.nan)
    return run_a, np.where(fin, np.nan, locf)


def windows(wp, panel, O):
    """(transformer index, origin index) of every window whose 90-day input is complete (after gap filling), and whether its
    60-day target is complete.

    Causal when the panel carries gap_parts (panel.run_a, set by apply_calendar): a gap that ENDS on or before the origin may be
    interpolated (both ends are known at the origin), as panel.filled does; a gap still open AT the origin may only be carried
    forward from the last value, and only while it has lasted <= panel.gap days. Without run_a: the old rule (panel.filled,
    which interpolates a gap at the origin using days after it)."""
    L, H = wp.LOOKBACK, wp.HORIZON
    bad = np.pad(np.cumsum(~np.isfinite(panel.filled), axis=1), ((0, 0), (1, 0)))
    tgt_bad = bad[:, O + H + 1] - bad[:, O + 1]
    run_a = getattr(panel, "run_a", None)
    if run_a is None:
        inp_ok = (bad[:, O + 1] - bad[:, O - (L - 1)]) == 0
    else:
        a = run_a[:, O]                                                   # (n_tr, len(O)); -1 where the origin day is observed
        open_ = a >= 0
        open_len = np.where(open_, O[None, :] - a + 1, 0)
        upto = np.where(open_, a, O[None, :] + 1)                         # the part before the open gap must be complete
        lo = np.broadcast_to(O[None, :] - (L - 1), upto.shape)
        before_ok = (np.take_along_axis(bad, upto, axis=1) - np.take_along_axis(bad, lo, axis=1)) == 0
        inp_ok = before_ok & (~open_ | ((open_len <= panel.gap) & (a >= 1)))
    r_tr, r_o = np.nonzero(inp_ok)
    return r_tr, O[r_o], tgt_bad[r_tr, r_o] == 0


def causal_input(panel, r_tr, r_o, L):
    """The 90-day inputs (rows, L) as known AT each origin: interpolated gaps from panel.filled where the gap closed before the
    origin; a gap still open at the origin carried forward from its last value (if open <= panel.gap days), else NaN."""
    kin = gather(panel.filled, r_tr, r_o, -(L - 1), L).copy()
    if getattr(panel, "run_a", None) is None:
        return kin
    a = panel.run_a[r_tr, r_o]
    open_ = a >= 0
    if not open_.any():
        return kin
    k = np.flatnonzero(open_)
    n_open = r_o[k] - a[k] + 1                                            # days of the gap up to and including the origin
    col = np.arange(L)[None, :]
    tail = col >= (L - np.minimum(n_open, L))[:, None]
    ok = (n_open <= panel.gap) & (a[k] >= 1)
    val = np.where(ok, panel.locf[r_tr[k], r_o[k]], np.nan)
    kin[k] = np.where(tail, val[:, None], kin[k])
    return kin


def gather(arr, r_tr, r_o, first, length):
    """arr (n_tr, T) -> (rows, length): arr[tr, o+first : o+first+length]"""
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(arr, length, axis=1)
    return np.asarray(w[r_tr, r_o + first])


# =============================================================================
# Calendar: weekday and holiday factors from the TRAINING period only
# =============================================================================
FIXED_HOLIDAYS = {"01-01": "newyear", "03-08": "fixed", "03-21": "fixed", "05-09": "fixed",
                  "09-01": "fixed", "10-01": "fixed", "12-08": "fixed"}
# Moving (Islamic) holidays, typed from memory and only approximately right: the script prints what each date did in the data.
HAYIT_DAYS = ["2022-05-02", "2022-05-03", "2022-07-09", "2022-07-11", "2023-04-21", "2023-04-22", "2023-06-28", "2023-06-29",
              "2024-04-10", "2024-04-11", "2024-06-16", "2024-06-17", "2025-03-30", "2025-03-31", "2025-06-06",
              "2026-03-20", "2026-05-27", "2026-05-28"]


def holiday_calendar(years, csv_path=None):
    """{Timestamp: class}.  Default: the fixed public holidays plus HAYIT_DAYS; --holidays-csv (date,class) replaces both."""
    if csv_path:
        h = pd.read_csv(os.path.expanduser(csv_path))
        cls = h["class"] if "class" in h else pd.Series("fixed", index=h.index)
        return {pd.Timestamp(d): str(c) for d, c in zip(h["date"], cls)}
    out = {pd.Timestamp(f"{y}-{md}"): c for y in years for md, c in FIXED_HOLIDAYS.items()}
    out.update({pd.Timestamp(d): "hayit" for d in HAYIT_DAYS})
    return out


def calendar_factors(inten, dates, train_end, holidays, min_transformers=30):
    """inten (n_tr, T) raw intensity with NaN for unusable days.  -> cal (T,) factor per date, dow table, holiday table.
    A day's factor = weekday factor x (1 + holiday-class effect).  Everything is estimated on days <= train_end."""
    import warnings
    T = len(dates)
    train = np.asarray(dates <= pd.Timestamp(train_end))
    inten = np.where(train[None, :], inten, np.nan)           # nothing after the cutoff, not even inside the centred 7-day mean
    c7 = pd.DataFrame(inten.T).rolling(7, center=True, min_periods=5).mean().to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        res = inten / c7 - 1
    n_ok = np.isfinite(res).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nat = np.where(n_ok >= min_transformers, np.nanmedian(res, axis=0), np.nan)        # what all transformers do that day
    dow = np.asarray(dates.dayofweek)
    hol_idx = {dates.get_loc(d): c for d, c in holidays.items() if d in dates}
    near = np.zeros(T, dtype=bool)
    for i in hol_idx:
        near[max(0, i - 1):i + 2] = True
    use = train & ~near & np.isfinite(nat)
    eff = np.zeros(7)
    cnt = np.zeros(7, dtype=int)
    for k in range(7):
        v = nat[use & (dow == k)]
        cnt[k] = len(v)
        eff[k] = np.median(v) if len(v) >= 20 else 0.0
    f_dow = 1.0 + eff - eff.mean()
    dow_tab = pd.DataFrame({"weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "factor": f_dow, "days_used": cnt})
    rows, by_class = [], {}
    for i, c in sorted(hol_idx.items()):
        if not np.isfinite(nat[i]):
            rows.append(dict(date=dates[i].date(), cls=c, observed_effect=np.nan, in_fit=False))
            continue
        r = (1 + nat[i]) / f_dow[dow[i]] - 1
        e = 7 * r / (6 - r)                                    # the centred 7-day mean contains the holiday itself
        in_fit = bool(train[i])
        rows.append(dict(date=dates[i].date(), cls=c, observed_effect=e, in_fit=in_fit))
        if in_fit:
            by_class.setdefault(c, []).append(e)
    class_eff = {c: float(np.median(v)) * len(v) / (len(v) + 1.0) for c, v in by_class.items()}      # shrink small samples
    cal = f_dow[dow].copy()
    for i, c in hol_idx.items():
        cal[i] *= 1.0 + class_eff.get(c, 0.0)
    cal = np.clip(cal, 0.6, 1.5)
    hol_tab = pd.DataFrame(rows)
    if len(hol_tab):
        hol_tab["factor_applied"] = [1.0 + class_eff.get(c, 0.0) for c in hol_tab["cls"]]
    return cal, dow_tab, hol_tab


def apply_calendar(wp, panel, args, train_cutoff):
    """Sets panel.cal (T,) and recomputes panel.filled = gap-filled (inten / cal). cal = 1 everywhere with --calendar none."""
    T = panel.n_days
    dates = pd.DatetimeIndex(panel.d0 + pd.to_timedelta(np.arange(T), unit="D"))
    panel.cal = np.ones(T)
    panel.cal_tables = (pd.DataFrame(), pd.DataFrame())
    if args.calendar != "none":
        hol = holiday_calendar(range(dates.year.min(), dates.year.max() + 1), args.holidays_csv) \
            if args.calendar == "dow+holidays" else {}
        panel.cal, dow_tab, hol_tab = calendar_factors(panel.inten, dates, train_cutoff, hol)
        panel.cal_tables = (dow_tab, hol_tab)
        print(f"[calendar] weekday factors (estimated on days up to {pd.Timestamp(train_cutoff).date()}): "
              + "  ".join(f"{r.weekday} {r.factor - 1:+.1%}" for r in dow_tab.itertuples()))
        if len(hol_tab):
            print("[calendar] holiday dates and what they did in the data (national median across transformers, after the weekday "
                  "effect; * = after the cutoff, not used for the factor; a date with no effect is probably a wrong date):")
            for c, g in hol_tab.groupby("cls"):
                f = g["factor_applied"].iloc[0]
                txt = ", ".join(f"{r.date:%y-%m-%d} " + ("n/a" if not np.isfinite(r.observed_effect) else f"{r.observed_effect:+.1%}")
                                + ("" if r.in_fit else "*") for r in g.itertuples())
                print(f"  class '{c}': factor {f - 1:+.1%} | {txt}")
    adj = panel.inten / panel.cal[None, :]
    panel.filled = fill_short_gaps(adj, args.gap_fill)
    panel.run_a, panel.locf = gap_parts(adj)                  # for inputs that do not look past their origin (causal_input)
    panel.gap = int(args.gap_fill)
    return panel


# =============================================================================
# Panel from the cleaned transformer series (transformer_clean.py)
# =============================================================================
def subset_panel(panel, mask):
    idx = np.flatnonzero(mask)
    p = SimpleNamespace(**vars(panel))
    for k in ("ids", "n_hh", "inten", "filled", "cover", "housing", "coato_of"):
        setattr(p, k, getattr(panel, k)[idx])
    p.hh_rows = {t: panel.hh_rows[t] for t in p.ids}
    return p


def load_clean_panel(wp, args, hh_meta):
    """Panel from <work-dir>/clean/transformer_clean.npz and core_households.csv.  inten = mean kWh per STABLE-CORE household
    on usable days (NaN where transformer_clean.py flagged the day), n_hh = size of the core, hh_rows = matrix rows of the core."""
    cdir = os.path.join(args.work_dir, "clean")
    z = np.load(os.path.join(cdir, "transformer_clean.npz"), allow_pickle=False)
    ids = z["transformer_id"].astype(str)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    T = len(dates)
    if (dates[-1] - dates[0]).days != T - 1:
        raise SystemExit("transformer_clean.npz dates are not consecutive days")
    kept, n_core = z["kept"].astype(bool), z["n_core"].astype(int)
    ch = pd.read_csv(os.path.join(cdir, "core_households.csv"), dtype=str)
    ch = ch[ch["role"] == "core"]
    mrow = pd.Series(np.arange(len(hh_meta)), index=hh_meta["meter_id"].astype(str).to_numpy())
    ch = ch.assign(row=ch["meter_id"].map(mrow)).dropna(subset=["row"])
    ch["row"] = ch["row"].astype(int)
    grp = {t: g.to_numpy() for t, g in ch.groupby("transformer_id")["row"]}
    inten = z["intensity"].astype("float64")
    sel = kept & (n_core >= args.min_households) & np.isfinite(inten).any(axis=1) & np.array([t in grp for t in ids])
    bad = [t for t, n in zip(ids[sel], n_core[sel]) if len(grp[t]) != n]
    if bad:
        print(f"[panel] WARNING: core size in the npz differs from core_households.csv for {len(bad)} transformers (e.g. {bad[:3]}); "
              f"the npz size is used for scaling, the csv rows for bottom-up")
    idx = np.flatnonzero(sel)
    ids_s = np.array([ids[i] for i in idx], dtype=object)
    hm = hh_meta.reset_index(drop=True)
    housing = []
    for t in ids_s:
        mm = hm["housing_type_id"].iloc[grp[t]].dropna()
        housing.append(mm.mode().iloc[0] if len(mm) else "Missing")
    panel = SimpleNamespace(ids=ids_s, n_hh=n_core[idx], inten=inten[idx], cover=z["cover"][idx].astype("float64"),
                            housing=np.asarray(housing, dtype=object), coato_of=z["coato_code"].astype(str)[idx].astype(object),
                            d0=dates[0], n_days=T, hh_rows={t: grp[t] for t in ids_s}, flags=z["flags"][idx],
                            fit_end=str(z["fit_end"]) if "fit_end" in z.files else None)
    panel.filled = fill_short_gaps(panel.inten, args.gap_fill)
    print(f"[panel] cleaned panel {os.path.join(cdir, 'transformer_clean.npz')}: {len(ids_s):,} of {len(ids):,} transformers "
          f"({len(np.unique(panel.coato_of))} coatos), core households {int(panel.n_hh.sum()):,} (median {np.median(panel.n_hh):.0f} per transformer), "
          f"{dates[0].date()} .. {dates[-1].date()}, usable days {np.isfinite(panel.inten).mean():.1%}")
    return panel


def keep_known_coatos(panel, clim):
    have = np.array([c in clim.coatos for c in panel.coato_of])
    if have.all():
        return panel
    miss = sorted(set(panel.coato_of[~have]))
    print(f"[weather] {len(miss)} coato(s) are not in the weather table; their {int((~have).sum())} transformers are left out "
          f"(e.g. {miss[:5]}). The household pipeline matches coato_code to the weather coato_id as text.")
    if not have.any():
        raise SystemExit("none of the panel's coatos is in the weather table; check the coato_code / coato_id format")
    return subset_panel(panel, have)


# =============================================================================
# Weather: climate and ONE forecast draw shared by both methods
# =============================================================================
def load_climate(wp, args, hh_model_dir):
    cands = [os.path.join(args.work_dir, "weather_daily.parquet"),
             os.path.join(hh_model_dir or "", "weather_daily.parquet")]
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        raise SystemExit("no weather_daily.parquet in the work dir or the household model dir: run prepare first")
    w = pd.read_parquet(path)
    w["coato_id"] = w["coato_id"].astype(str)
    w["d"] = pd.to_datetime(w["d"])
    print(f"[weather] {path}: {w['coato_id'].nunique()} coatos, {w['d'].min().date()} .. {w['d'].max().date()}")
    return wp.build_climate(w), w


def forecast_grids(wp, clim, coatos, panel, O, args):
    """-> (forecast_mode, fc_grid, fc_grid_mm).  'archive' reads the real archive; 'simulated' draws the error once
    per (coato, origin) for every origin date and stores it as an archive, so both methods see identical forecasts."""
    mode = wp.FORECAST_MODE
    if mode == "none":
        return "none", None, None
    K = wp.FORECAST_DAYS
    if mode == "archive":
        fc = pd.read_parquet(os.path.join(args.work_dir, "forecast_archive.parquet"))
        fc["coato_id"], fc["issue_date"] = fc["coato_id"].astype(str), pd.to_datetime(fc["issue_date"])
    else:
        origin_dates = pd.DatetimeIndex(panel.d0 + pd.to_timedelta(np.unique(O), unit="D"))
        coatos = [str(c) for c in sorted(set(map(str, np.atleast_1d(coatos))))]
        u = pd.DataFrame({"coato_code": np.repeat(coatos, len(origin_dates)),
                          "origin_date": np.tile(origin_dates.values, len(coatos))})
        t_obs, _, _ = wp.lookup_weather(u, clim, "all")
        t_fut = t_obs[:, wp.LOOKBACK:wp.LOOKBACK + K]
        t_fc = wp.simulate_forecast(u, t_fut)
        cols = {"t_fc": t_fc}
        if wp.USE_MINMAX:
            tmin, _, tmax, _ = wp.lookup_minmax(u, clim, "all")
            cols["t_fc_min"], cols["t_fc_max"] = wp.simulate_forecast_minmax(
                u, t_fc, tmin[:, wp.LOOKBACK:wp.LOOKBACK + K], tmax[:, wp.LOOKBACK:wp.LOOKBACK + K])
        n = len(u)
        fc = pd.DataFrame({"coato_id": np.repeat(u["coato_code"].to_numpy(), K), "issue_date": np.repeat(u["origin_date"].values, K),
                           "lead_day": np.tile(np.arange(1, K + 1), n)})
        for k, v in cols.items():
            fc[k] = v.reshape(-1)
        print(f"[forecast] simulated 10-day forecasts drawn once for {len(coatos)} coato(s) x {len(origin_dates)} origin dates "
              f"(shared by both methods)")
    grid = wp.build_forecast_grid(fc, clim)
    grid_mm = None
    if wp.USE_MINMAX:
        grid_mm = dict(min=wp.build_forecast_grid(fc, clim, col="t_fc_min"),
                       max=wp.build_forecast_grid(fc, clim, col="t_fc_max"))
    return "archive", grid, grid_mm


def check_coato(clim, coato):
    if str(coato) not in clim.coatos:
        near = [c for c in clim.coatos if str(coato)[-4:] in c or c[-4:] in str(coato)][:10]
        raise SystemExit(f"coato '{coato}' is not in the weather table (its ids: {len(clim.coatos)} coatos, similar: {near}). "
                         f"The household pipeline matches coato_code to the weather coato_id as text; both must agree.")


def featurize(wp, pdf, kin, clim, fc, priors, segments, normal_mode, chunk):
    """wp.build_features in row chunks.  -> (cat meta, feats, base)"""
    mode, grid, grid_mm = fc
    metas, feats, bases = [], [], []
    for a in range(0, len(pdf), chunk):
        p = pdf.iloc[a:a + chunk].reset_index(drop=True)
        k = kin[a:a + chunk]
        bh, bc = wp.row_priors_from_segments(p, segments, priors["mean"][0], priors["mean"][1])
        bh_mm = bc_mm = None
        if wp.USE_MINMAX:
            bh_mm, bc_mm = wp.row_priors_from_segments(p, segments, priors["mm"][0], priors["mm"][1])
        meta, f, base, _ = wp.build_features(
            p, k, np.zeros((len(p), wp.HORIZON)), clim, future_mode="normal", normal_mode=normal_mode,
            beta_h_prior=bh, beta_c_prior=bc, forecast_mode=mode, fc_grid=grid, meter_segment=segments,
            beta_h_prior_mm=bh_mm, beta_c_prior_mm=bc_mm, fc_grid_mm=grid_mm)
        metas.append(meta), feats.append(f), bases.append(base)
    return pd.concat(metas, ignore_index=True), pd.concat(feats, ignore_index=True), np.vstack(bases)


# =============================================================================
# Validation block, year-ago features, seasonal-naive forecast
# =============================================================================
LY_DAYS = 364                     # "same time last year": 52 weeks back, so the weekday matches too


def blocked_validation(origin, val_start, val_end, train_cutoff, horizon):
    """Validation = origins in (val_start, val_end]; fit = every training origin at least `horizon` days away from it on
    BOTH sides (before val_start - horizon, or after val_end + horizon), so no fit target day is a validation target day.
    A block spanning autumn -> spring lets early stopping see the heating-season transitions, not one summer."""
    o = pd.to_datetime(origin)
    vs, ve, cut = pd.Timestamp(val_start), pd.Timestamp(val_end), pd.Timestamp(train_cutoff)
    if not vs < ve <= cut:
        raise SystemExit(f"--val-end {val_end} must be after --val-start {val_start} and not after the train cutoff {train_cutoff}")
    emb = pd.Timedelta(days=horizon)
    val = ((o > vs) & (o <= ve)).to_numpy()
    fit = (((o <= vs - emb) | (o > ve + emb)) & (o <= cut)).to_numpy()
    return fit, val


def month_span(dates):
    d = pd.to_datetime(pd.Series(dates))
    if d.empty:
        return "none"
    m = d.dt.to_period("M")
    return f"{m.min()}..{m.max()} ({m.nunique()} months, {len(d):,} rows)"


def take(arr, r_tr, start, length):
    """arr (n_tr, T) -> (rows, length): arr[tr, start : start+length], NaN where the index falls outside the series
    (start may be negative; sliding_window_view would wrap around instead)."""
    T = arr.shape[1]
    idx = np.asarray(start, dtype=np.int64)[:, None] + np.arange(length)[None, :]
    ok = (idx >= 0) & (idx < T)
    rows = np.broadcast_to(np.asarray(r_tr)[:, None], idx.shape)
    out = np.full(idx.shape, np.nan)
    out[ok] = arr[rows[ok], idx[ok]]
    return out


def year_ago(wp, arr, r_tr, r_o, blocks, min_ref_days=20, clip=(0.2, 5.0)):
    """What this transformer did around the same date one year earlier, relative to the 30 days before that date.
    All days used are <= origin - LY_DAYS + horizon < origin, so this is causal.

    -> (features dict, daily ratio (rows, H))
       ly_ratio_{a}_{b}   mean of last year's days a..b after the origin  /  mean of last year's 30 days before it
       ly_ratio_m7_0      last year's 7 days before the origin / its 30 days before (was demand already turning?)
       daily ratio        7-day centred mean of last year's day h / the same 30-day reference (for the seasonal-naive forecast)
    NaN where last year is not in the panel or too much of it is missing."""
    import warnings
    H = wp.HORIZON
    o_ly = np.asarray(r_o) - LY_DAYS
    ref_win = take(arr, r_tr, o_ly - 29, 30)
    n_ref = np.isfinite(ref_win).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # nanmean of all-NaN rows -> NaN, as intended
        ref = np.where(n_ref >= min_ref_days, np.nanmean(ref_win, axis=1), np.nan)
        ref = np.where(ref > 0, ref, np.nan)
        fut = take(arr, r_tr, o_ly + 1 - 3, H + 6)                     # 3 extra days each side for the centred mean
        f = {}
        for a, b in blocks:
            blk = fut[:, 3 + a - 1: 3 + b]
            ok = np.isfinite(blk).sum(1) >= max(1, (b - a + 1) // 2)
            f[f"ly_ratio_{a}_{b}"] = np.where(ok, np.clip(np.nanmean(blk, axis=1) / ref, *clip), np.nan)
        last7 = ref_win[:, -7:]
        f["ly_ratio_m7_0"] = np.where(np.isfinite(last7).sum(1) >= 4, np.clip(np.nanmean(last7, axis=1) / ref, *clip), np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(fut, 7, axis=1)                         # (rows, H, 7), centred on last year's day h
        daily = np.where(np.isfinite(w).sum(2) >= 4, np.nanmean(w, axis=2), np.nan) / ref[:, None]
    return f, np.clip(daily, *clip)


# =============================================================================
# Direct: each transformer is a meter
# =============================================================================
def fit_direct_fold(wp, args, panel, clim, fc, train_cutoff, val_start, val_end, test_end, label):
    """One fold: windows, features, CatBoost fit with early stopping, test forecasts. Test origins run from train_cutoff + 60
    days to test_end (None = the last origin). Returns (test arrays, info)."""
    from catboost import CatBoostRegressor
    L, H = wp.LOOKBACK, wp.HORIZON
    O = origin_grid(wp, panel, args)
    r_tr, r_o, tgt_ok = windows(wp, panel, O)
    pdf = pd.DataFrame({"meter_id": panel.ids[r_tr].astype(str), "consumer_id": np.nan,
                        "origin_date": panel.d0 + pd.to_timedelta(r_o, unit="D"),
                        "coato_code": panel.coato_of[r_tr], "housing_type_id": panel.housing[r_tr]})
    kin = causal_input(panel, r_tr, r_o, L)               # inputs as known at the origin (no gap filled from later days)
    ktg = gather(panel.filled, r_tr, r_o, 1, H)            # targets (labels) may use interpolation
    print(f"[direct] {len(pdf):,} windows ({pdf['meter_id'].nunique():,} transformers), "
          f"{tgt_ok.mean():.1%} with a complete 60-day target")

    segments = pd.Series("unclassified", index=pd.Index(panel.ids.astype(str)), name="segment")
    cutoff_ts = pd.Timestamp(train_cutoff)
    minm = max(10, int(0.2 * len(panel.ids)))
    ph = wp.estimate_segment_beta_priors(pdf, kin, clim, train_cutoff, segments, min_meters=minm, kind="mean")
    pm = wp.estimate_segment_beta_priors(pdf, kin, clim, train_cutoff, segments, min_meters=minm, kind="minmax") \
        if wp.USE_MINMAX else ({}, (None, None))
    priors = dict(mean=ph, mm=pm)

    meta, feats, base = featurize(wp, pdf, kin, clim, fc, priors, segments, args.normal_mode, args.chunk)
    n_hh = panel.n_hh[r_tr].astype(float)
    feats["n_households"], feats["log_n_households"] = n_hh, np.log(n_hh)
    feats["coverage_30d"] = gather(panel.cover, r_tr, r_o, -29, 30).mean(axis=1)
    ly_feats, ly_daily = year_ago(wp, panel.filled, r_tr, r_o, wp.BLOCKS)
    if not args.no_year_ago:
        for k, v in ly_feats.items():
            feats[k] = v
    have_ly = np.isfinite(ly_feats["ly_ratio_1_7"])
    print(f"[direct] year-ago reference available for {have_ly.mean():.1%} of windows"
          + ("" if not args.no_year_ago else " (features switched off with --no-year-ago; seasonal-naive still scored)"))
    # seasonal naive: today's 30-day level x last year's ratio for each target day; plain 30-day persistence where last year is missing
    snaive = feats["level_kwh"].to_numpy()[:, None] * np.where(np.isfinite(ly_daily), ly_daily, 1.0)
    cats = list(wp.CAT_FEATURES)
    X = pd.concat([feats, meta[cats]], axis=1)
    feature_cols = list(feats.columns) + cats
    X = X[feature_cols]
    if args.drop_features:
        pats = [p.strip() for p in args.drop_features.split(",") if p.strip()]
        drop = [c for c in feature_cols if c not in cats and any(c == p or (p.endswith("*") and c.startswith(p[:-1])) for p in pats)]
        unknown = [p for p in pats if not any(c == p or (p.endswith("*") and c.startswith(p[:-1])) for c in feature_cols)]
        if unknown:
            print(f"[direct] --drop-features: no feature matches {unknown} (use the exact name, or a prefix ending in *)")
        feature_cols = [c for c in feature_cols if c not in drop]
        X = X[feature_cols]
        print(f"[direct] --drop-features removed {len(drop)} feature(s): {drop}")

    train, fit, val, test = wp.make_masks(meta, train_cutoff, val_start, test_end)
    if val_end:
        fit, val = blocked_validation(meta["origin_date"], val_start, val_end, train_cutoff, H)
    reg = wp.regime_keep_mask(meta)                       # the household pipeline's own old/new-regime and excluded-range policy
    train, fit, val = train & tgt_ok & reg, fit & tgt_ok & reg, val & tgt_ok & reg
    print(f"[direct] fit {fit.sum():,} | val {val.sum():,} | train {train.sum():,} | test {test.sum():,} | "
          f"features {len(feature_cols)}")
    print("[direct] origin months  fit: " + month_span(meta["origin_date"][fit]) + "  |  val: "
          + month_span(meta["origin_date"][val]) + "  |  test: " + month_span(meta["origin_date"][test]))
    if not (fit.any() and val.any() and test.any()):
        raise SystemExit(f"fold {label}: empty split: check its cutoff / validation dates against the dates available "
                         f"({pdf['origin_date'].min().date()} .. {pdf['origin_date'].max().date()})")
    Y = pd.DataFrame(ktg - base, columns=[f"res_{i}" for i in range(1, H + 1)])
    w_all = {"none": np.ones(len(X)), "sqrt": np.sqrt(n_hh), "linear": n_hh}[args.size_weight]
    w_all = w_all / w_all.mean()

    with stage(f"direct: CatBoost fit, fold {label}"):
        m = CatBoostRegressor(cat_features=cats, **wp.CATBOOST_PARAMS)
        m.fit(X[fit], Y[fit], eval_set=(X[val], Y[val]), early_stopping_rounds=wp.CATBOOST_EARLY_STOPPING_ROUNDS,
              sample_weight=w_all[fit])
        best = m.get_best_iteration()
        final = max(min(best, args.final_iters_cap), 10)
        print(f"[direct] best iteration {best} -> final fit with {final} (cap {args.final_iters_cap})")
        if best < 20:
            od = meta["origin_date"]
            print(f"[direct] WARNING: the validation loss was lowest after {best} trees. What the trees learn from origins "
                  f"{od[fit].min().date()}..{od[fit].max().date()} does not carry over to the validation origins "
                  f"{od[val].min().date()}..{od[val].max().date()} (their targets reach {(od[val].max() + pd.Timedelta(days=H)).date()}), "
                  f"so the final model has only {final} trees and 'direct' is close to the thermal baseline plus a small correction. "
                  f"Validation covers one season only; the test period is a different one. Try --val-start with a longer, mixed-season "
                  f"validation window and/or --drop-features 'origin_doy*' (date identity lets trees memorise one shared residual per day).")
        m = CatBoostRegressor(cat_features=cats, **dict(wp.CATBOOST_PARAMS, iterations=final, verbose=200))
        m.fit(X[train], Y[train], sample_weight=w_all[train])
        m.save_model(os.path.join(args.out_dir, f"direct_catboost_{label}.cbm"))

    scale = panel.n_hh[r_tr][test][:, None].astype(float)
    pred = np.maximum(base[test] + m.predict(X[test]), 0.0) * scale
    o_test = r_o[test].astype("int64")
    cal_rows = panel.cal[np.minimum(o_test[:, None] + np.arange(1, H + 1)[None, :], panel.n_days - 1)]
    out = dict(transformer_id=panel.ids[r_tr][test].astype(str), origin=o_test,
               pred=pred.astype("float32"), thermal=(base[test] * scale).astype("float32"),
               pers30=(feats.loc[test, "level_kwh"].to_numpy() * scale[:, 0]).astype("float32"),
               pers7=(feats.loc[test, "kwh_mean_7d"].to_numpy() * scale[:, 0]).astype("float32"),
               snaive=(snaive[test] * scale).astype("float32"),
               cal=cal_rows.astype("float32"),                 # this fold's calendar factor on every target day
               n_hh=scale[:, 0].astype("int64"))
    imp = pd.DataFrame({"feature": feature_cols, "importance": m.get_feature_importance()}).sort_values("importance", ascending=False)
    print(f"[direct] fold {label} top features:\n" + imp.head(12).to_string(index=False))
    info = dict(fold=label, train_cutoff=str(train_cutoff), val_start=str(val_start), val_end=val_end, test_end=test_end,
                fit_months=month_span(meta["origin_date"][fit]), val_months=month_span(meta["origin_date"][val]),
                test_months=month_span(meta["origin_date"][test]),
                best_iteration=int(best), final_iterations=int(final), features=feature_cols,
                rows=dict(fit=int(fit.sum()), val=int(val.sum()), train=int(train.sum()), test=int(test.sum())))
    return out, info


def parse_folds(args, cutoffs):
    """--folds 'CUTOFF:VAL_START:VAL_END,...' -> [(cutoff, val_start, val_end, test_end)], sorted by cutoff. Fold i tests origins
    from its cutoff + 60 days up to the next fold's cutoff + 59 days (the last fold: to the end of the data), so the folds' test
    origins do not overlap and together run from the first fold's test start to the end. Without --folds: one fold from
    --train-cutoff / --val-start / --val-end, as before."""
    if not args.folds:
        return [(str(cutoffs[0]), str(cutoffs[1]), args.val_end, None)]
    folds = []
    for part in [p.strip() for p in args.folds.split(",") if p.strip()]:
        bits = part.split(":")
        if len(bits) not in (2, 3):
            raise SystemExit(f"--folds entry {part!r}: use CUTOFF:VAL_START:VAL_END (or CUTOFF:VAL_START for the plain split)")
        c, vs, ve = (bits + [None])[:3]
        folds.append((pd.Timestamp(c), str(pd.Timestamp(vs).date()), str(pd.Timestamp(ve).date()) if ve else None))
    folds.sort(key=lambda f: f[0])
    out = []
    for i, (c, vs, ve) in enumerate(folds):
        nxt = folds[i + 1][0] + pd.Timedelta(days=59) if i + 1 < len(folds) else None
        out.append((str(c.date()), vs, ve, str(nxt.date()) if nxt is not None else None))
    return out


def stage_direct(wp, args, world, panel, clim, fc, cutoffs):
    """The direct model over one or more rolling folds. Each fold re-estimates the calendar factors on its own training days,
    fits its own model, and forecasts only its own test origins; the forecasts are saved together (direct_test.npz, with a
    `fold` index and each row's calendar factors)."""
    folds = parse_folds(args, cutoffs)
    outs, infos = [], []
    for i, (c, vs, ve, te) in enumerate(folds):
        label = f"f{i + 1}"
        print(f"\n[direct] ===== fold {label}: train cutoff {c} | validation {'(' + vs + ', ' + ve + ']' if ve else 'after ' + vs} | "
              f"test origins {(pd.Timestamp(c) + pd.Timedelta(days=wp.HORIZON)).date()} .. {te or 'end of data'} =====")
        apply_calendar(wp, panel, args, c)                     # factors from this fold's training days only
        out, info = fit_direct_fold(wp, args, panel, clim, fc, c, vs, ve, te, label)
        out["fold"] = np.full(len(out["origin"]), i, dtype="int16")
        outs.append(out)
        infos.append(info)
    keys = outs[0].keys()
    allout = {k: np.concatenate([o[k] for o in outs]) for k in keys}
    np.savez_compressed(os.path.join(args.out_dir, "direct_test.npz"), **allout,
                        fold_names=np.array([f["fold"] for f in infos]))
    with open(os.path.join(args.out_dir, "direct_info.json"), "w") as fh:
        json.dump(dict(include_old=wp.INCLUDE_OLD, old_regime_flag=bool(wp.OLD_REGIME_FLAG), year_ago=not args.no_year_ago,
                       transformers=int(len(panel.ids)), folds=infos,
                       # the first fold's numbers at the top level, as older readers of this file expect
                       **{k: infos[0][k] for k in ("train_cutoff", "val_start", "val_end", "best_iteration", "final_iterations",
                                                    "features")}), fh, indent=2, default=str)
    print(f"[direct] saved {len(allout['origin']):,} test forecasts from {len(folds)} fold(s) -> direct_test.npz")


# =============================================================================
# Bottom-up: the household model, summed
# =============================================================================
class HouseholdModel:
    """The household model from the files whole_pipeline_v8.persist_model wrote (what electricity_pyfunc.py serves)."""

    def __init__(self, wp, model_dir, which=None):
        self.wp, self.dir = wp, model_dir
        with open(os.path.join(model_dir, "columns.json")) as fh:
            self.cfg = json.load(fh)
        self.which = which or self.cfg.get("serve_model", "catboost")
        if self.which == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(os.path.join(model_dir, "xgboost_model.json"))
        else:
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor()
            self.model.load_model(os.path.join(model_dir, "catboost_model.cbm"))
        seg = pd.read_csv(os.path.join(model_dir, "meter_segments.csv"), dtype=str, index_col=0)
        self.segments = seg.iloc[:, 0]
        self.segments.index = self.segments.index.astype(str)
        self.prior = ({k: tuple(v) for k, v in self.cfg["segment_beta_priors"].items()}, tuple(self.cfg["pooled_beta_prior"]))
        mm = self.cfg.get("minmax", {})
        self.use_mm = bool(mm.get("enabled"))
        self.prior_mm = (({k: tuple(v) for k, v in mm.get("segment_beta_priors", {}).items()},
                          tuple(mm["pooled_beta_prior"])) if self.use_mm else ({}, (None, None)))
        # the model's own input rules: glitch repair, gap filling and how much history a household needs
        wp.GLITCH_REPAIR = bool(self.cfg.get("glitch_repair", True))
        self.rule = dict(min_input_days=80, min_recent_days=30, recent_window=30, gap_fill=3)   # rules before columns.json had them
        self.rule.update({k: v for k, v in (self.cfg.get("input_rule") or {}).items() if k in self.rule})
        if self.use_mm != bool(wp.USE_MINMAX):
            print(f"[bottom-up] WARNING: the household model was trained with USE_MINMAX={self.use_mm}, "
                  f"whole_pipeline_v8.py now says {wp.USE_MINMAX}: features will not match")

    def predict(self, pdf, hist, clim, fc, normal_mode):
        wp, cfg = self.wp, self.cfg
        mode, grid, grid_mm = fc
        bh, bc = wp.row_priors_from_segments(pdf, self.segments, *self.prior)
        bh_mm = bc_mm = None
        if self.use_mm:
            bh_mm, bc_mm = wp.row_priors_from_segments(pdf, self.segments, *self.prior_mm)
        meta, feats, base, _ = wp.build_features(
            pdf, hist, np.zeros((len(pdf), wp.HORIZON)), clim, future_mode="normal", normal_mode=normal_mode,
            beta_h_prior=bh, beta_c_prior=bc, forecast_mode=mode, fc_grid=grid, meter_segment=self.segments,
            beta_h_prior_mm=bh_mm, beta_c_prior_mm=bc_mm, fc_grid_mm=grid_mm)
        X = pd.concat([feats, meta[cfg["cat_features"]]], axis=1)
        for c in cfg["feature_cols"]:
            if c not in X.columns:
                X[c] = np.nan
        X = X[cfg["feature_cols"]]
        if self.which == "xgboost":
            resid = self.model.predict(wp.to_xgb_frame(X, cfg.get("xgb_categories")))
        else:
            resid = self.model.predict(X)
        return np.maximum(base + resid, 0.0)


def stage_bottom_up(wp, args, world, panel, clim, fc, hh_meta, kwh):
    L, H = wp.LOOKBACK, wp.HORIZON
    model = HouseholdModel(wp, args.hh_model_dir, args.hh_model)
    rule = dict(model.rule)                                # command-line values override the model's own rules
    for k, a in (("gap_fill", args.hh_gap_fill), ("min_input_days", args.min_input_days),
                 ("min_recent_days", args.min_recent_days), ("recent_window", args.recent_window)):
        if a is not None:
            rule[k] = int(a)
    print(f"[bottom-up] household input rule: >= {rule['min_input_days']} of 90 days and >= {rule['min_recent_days']} of the last "
          f"{rule['recent_window']} observed; gaps filled up to {rule['gap_fill']} days; glitch repair "
          f"{'on' if wp.GLITCH_REPAIR else 'off (cleaned source)'}")
    train_cutoff = pd.Timestamp(args.train_cutoff or model.cfg["train_cutoff"])
    O = origin_grid(wp, panel, args)
    if args.bu_stride % args.origin_stride:
        raise SystemExit("--bu-stride must be a multiple of --origin-stride (bottom-up origins must be a subset of the direct ones)")
    O = O[(O % args.bu_stride) == 0]
    dates = panel.d0 + pd.to_timedelta(O, unit="D")
    O = O[dates >= cutoff_test_start(wp, train_cutoff)]
    if not len(O):
        raise SystemExit("no test origins for the bottom-up run (cutoff + horizon is after the last origin)")
    print(f"[bottom-up] {len(O)} test origins ({panel.d0 + pd.Timedelta(days=int(O[0])):%Y-%m-%d} .. "
          f"{panel.d0 + pd.Timedelta(days=int(O[-1])):%Y-%m-%d}, every {args.bu_stride} days)")

    # households of the panel, in transformer order, cleaned once with the same causal repair the model was trained on
    rows = np.concatenate([panel.hh_rows[t] for t in panel.ids])
    tr_of = np.concatenate([np.full(len(panel.hh_rows[t]), k) for k, t in enumerate(panel.ids)])
    meta = hh_meta.iloc[rows].reset_index(drop=True)
    with stage("bottom-up: clean household series"):
        clean = np.empty((len(rows), kwh.shape[1]), dtype="float32")
        rawfin = np.zeros((len(rows), kwh.shape[1]), dtype=bool)          # which days really had a reading (for open gaps)
        for i, r in enumerate(rows):
            x0 = np.asarray(kwh[r], dtype="float64")
            rawfin[i] = np.isfinite(x0)
            x = fill_short_gaps(x0[None, :], rule["gap_fill"])[0] if rule["gap_fill"] > 0 else x0
            clean[i] = wp.clean_recent_readings_for_serving(x)
            if i and i % 20000 == 0:
                print(f"  cleaned {i:,} / {len(rows):,} households", flush=True)
    unknown = (~meta["meter_id"].isin(model.segments.index)).mean()
    print(f"[bottom-up] {len(meta):,} households; {unknown:.1%} are not in the household model's segment table "
          f"(they get the pooled prior, as at serving)")

    res_tr, res_o, res_pred, res_share, res_np = [], [], [], [], []
    for n, o in enumerate(O):
        with stage(f"bottom-up: origin {n + 1}/{len(O)} {panel.d0 + pd.Timedelta(days=int(o)):%Y-%m-%d}"):
            hist = causal_history(clean[:, o - (L - 1): o + 1], rawfin[:, o - (L - 1): o + 1], rule["gap_fill"])
            fin = np.isfinite(hist)
            ok = (fin[:, -rule["recent_window"]:].sum(axis=1) >= rule["min_recent_days"]) & (fin.sum(axis=1) >= rule["min_input_days"])
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            sums, cnt = np.zeros((len(panel.ids), H)), np.zeros(len(panel.ids))
            for a in range(0, len(idx), args.chunk):
                sel = idx[a:a + args.chunk]
                pdf = pd.DataFrame({"meter_id": meta["meter_id"].to_numpy()[sel], "consumer_id": np.nan,
                                    "origin_date": panel.d0 + pd.Timedelta(days=int(o)),
                                    "coato_code": meta["coato_code"].to_numpy()[sel],
                                    "housing_type_id": meta["housing_type_id"].to_numpy()[sel]})
                p = model.predict(pdf, hist[sel].astype("float64"), clim, fc, args.normal_mode)
                np.add.at(sums, tr_of[sel], p)
                np.add.at(cnt, tr_of[sel], 1.0)
            share = cnt / panel.n_hh
            good = share >= args.bu_min_share
            res_tr.append(np.flatnonzero(good)), res_o.append(np.full(int(good.sum()), o))
            res_pred.append(sums[good] / cnt[good][:, None] * panel.n_hh[good][:, None])
            res_share.append(share[good]), res_np.append(cnt[good])
            print(f"  {len(idx):,} households with a usable history; {int(good.sum())} / {len(panel.ids)} transformers "
                  f"have >= {args.bu_min_share:.0%} of their households forecast", flush=True)
    if not res_tr:
        raise SystemExit("bottom-up produced nothing: check --min-recent-days / --min-input-days / --recent-window")
    tr_idx = np.concatenate(res_tr)
    np.savez_compressed(os.path.join(args.out_dir, "bottomup_test.npz"),
                        transformer_id=panel.ids[tr_idx].astype(str), origin=np.concatenate(res_o).astype("int64"),
                        pred=np.vstack(res_pred).astype("float32"), share=np.concatenate(res_share),
                        n_pred=np.concatenate(res_np), n_hh=panel.n_hh[tr_idx])
    print(f"[bottom-up] saved {len(tr_idx):,} transformer-origin forecasts -> bottomup_test.npz")


def causal_history(hist, fin, gap):
    """hist (n, L) gap-filled + cleaned household histories ending at the origin; fin (n, L) where a reading really existed.
    A gap still open at the origin (trailing missing days) was interpolated with days AFTER the origin: replace it by the last
    known value while it has lasted <= gap days, else NaN. Gaps that closed before the origin keep their interpolation."""
    n, L = hist.shape
    out = np.array(hist, dtype="float64", copy=True)
    trail = np.where(fin.any(1), np.argmax(fin[:, ::-1], axis=1), L)    # missing days at the end of the window
    k = np.flatnonzero(trail > 0)
    if len(k):
        col = np.arange(L)[None, :]
        tail = col >= (L - trail[k])[:, None]
        keep = (trail[k] <= gap) & (trail[k] < L)
        last = out[k, np.maximum(L - trail[k] - 1, 0)]
        val = np.where(keep, last, np.nan)
        out[k] = np.where(tail, val[:, None], out[k])
    return out


def cutoff_test_start(wp, train_cutoff):
    return pd.Timestamp(train_cutoff) + pd.Timedelta(days=wp.HORIZON)


# =============================================================================
# Compare
# =============================================================================
def wape_bias(P, A):
    v = np.isfinite(P) & np.isfinite(A) & (A > 0)
    if not v.any():
        return np.nan, np.nan, 0
    a = A[v].sum()
    return float(np.abs(P[v] - A[v]).sum() / a), float((P[v] - A[v]).sum() / a), int(v.sum())


def score_table(models, A, tr, day, bands, horizons, band_names):
    """Rows: model x horizon x band ('all' + each band): pooled WAPE, bias, median per-transformer WAPE, system WAPE."""
    rows = []
    for name, P in models.items():
        for hi, h in enumerate(horizons):
            for bname, bmask in [("all", np.ones(len(tr), bool))] + [(b, bands == b) for b in band_names]:
                if not bmask.any():
                    continue
                p, a = P[bmask, hi], A[bmask, hi]
                w, b, n = wape_bias(p, a)
                v = np.isfinite(p) & np.isfinite(a) & (a > 0)
                per_tr = (pd.DataFrame({"tr": tr[bmask][v], "e": np.abs(p[v] - a[v]), "a": a[v]})
                          .groupby("tr").sum())
                med = float((per_tr["e"] / per_tr["a"]).median()) if len(per_tr) else np.nan
                d = pd.DataFrame({"day": day[bmask, hi][v], "p": p[v], "a": a[v]}).groupby("day").sum()
                sysw = float((d["p"] - d["a"]).abs().sum() / d["a"].sum()) if len(d) else np.nan
                rows.append(dict(model=name, horizon=h, band=bname, wape=w, bias=b, median_transformer_wape=med,
                                 sys_wape=sysw if bname == "all" else np.nan, n=n, transformers=len(per_tr)))
    return pd.DataFrame(rows)


def paired_bootstrap(Pa, Pb, A, tr, reps=400, seed=0):
    """Difference in pooled WAPE (a - b) with a transformer-level bootstrap: 5th / 95th percentile."""
    v = np.isfinite(Pa) & np.isfinite(Pb) & np.isfinite(A) & (A > 0)
    d = pd.DataFrame({"tr": tr[v], "ea": np.abs(Pa[v] - A[v]), "eb": np.abs(Pb[v] - A[v]), "a": A[v]}).groupby("tr").sum()
    if len(d) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ea, eb, a = d["ea"].to_numpy(), d["eb"].to_numpy(), d["a"].to_numpy()
    est = ea.sum() / a.sum() - eb.sum() / a.sum()
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    diff = ea[idx].sum(1) / a[idx].sum(1) - eb[idx].sum(1) / a[idx].sum(1)
    return float(est), float(np.percentile(diff, 5)), float(np.percentile(diff, 95))


def stage_compare(wp, args, panel):
    horizons = [int(h) for h in args.horizons.split(",")]
    d = np.load(os.path.join(args.out_dir, "direct_test.npz"), allow_pickle=False)
    bpath = os.path.join(args.out_dir, "bottomup_test.npz")
    have_bu = os.path.exists(bpath)
    kd = pd.DataFrame({"transformer_id": d["transformer_id"], "origin": d["origin"], "i": np.arange(len(d["origin"]))})
    if have_bu:
        b = np.load(bpath, allow_pickle=False)
        kb = pd.DataFrame({"transformer_id": b["transformer_id"], "origin": b["origin"], "j": np.arange(len(b["origin"]))})
        m = kd.merge(kb, on=["transformer_id", "origin"])
        print(f"[compare] direct {len(kd):,} | bottom-up {len(kb):,} | common transformer-origins {len(m):,} "
              f"({m['transformer_id'].nunique()} transformers, {m['origin'].nunique()} origins)")
        if m.empty:
            raise SystemExit("direct and bottom-up have no transformer-origin in common")
    else:
        m = kd.assign(j=-1)
        print(f"[compare] no bottomup_test.npz in {args.out_dir}: direct vs persistence only ({len(kd):,} transformer-origins, "
              f"{m['transformer_id'].nunique()} transformers)")
    i, j = m["i"].to_numpy(), m["j"].to_numpy()
    tid = m["transformer_id"].to_numpy().astype(str)
    org = m["origin"].to_numpy()
    n_hh = d["n_hh"][i].astype(float)
    pos = {t: k for k, t in enumerate(panel.ids.astype(str))}
    k = np.asarray([pos[t] for t in tid])
    hi = np.asarray(horizons) - 1
    day_idx = org[:, None] + np.asarray(horizons)[None, :]
    A = panel.inten[k[:, None], day_idx] * n_hh[:, None]                     # NaN on days that are not usable / not observed
    cal = getattr(panel, "cal", None)
    if "cal" in d.files:                                       # each fold's own factors, saved with its forecasts
        calT = d["cal"][i][:, hi].astype(float)
    else:
        calT = cal[day_idx] if cal is not None else np.ones(day_idx.shape)   # calendar factor of every target day
    fold = d["fold"][i] if "fold" in d.files else np.zeros(len(i), dtype=int)
    fold_names = list(d["fold_names"]) if "fold_names" in d.files else ["f1"]
    P_dir = d["pred"][i][:, hi].astype(float) * calT
    models = {"persistence30": np.repeat(d["pers30"][i][:, None], len(horizons), 1).astype(float) * calT}
    extra = {"persistence7": np.repeat(d["pers7"][i][:, None], len(horizons), 1).astype(float) * calT,
             "thermal_baseline": d["thermal"][i][:, hi].astype(float) * calT}
    if "snaive" in d.files:                                   # older direct_test.npz files do not have it
        extra["seasonal_naive"] = d["snaive"][i][:, hi].astype(float) * calT
    if have_bu:
        P_bu_raw = b["pred"][j][:, hi].astype(float)
        P_bu = P_bu_raw * calT
        models.update({"bottom_up": P_bu, "direct": P_dir, "average": 0.5 * (P_bu + P_dir)})
        if cal is not None and not np.allclose(cal, 1.0):
            extra["bottom_up_nocal"] = P_bu_raw
    else:
        models["direct"] = P_dir

    # size bands: quartiles of the household count, over the transformers in the comparison
    size = pd.Series(n_hh, index=tid).groupby(level=0).first()
    q = args.size_bands
    codes = pd.qcut(size.rank(method="first"), q, labels=False)
    names = {}
    for c in range(q):
        s = size[codes == c]
        names[c] = f"Q{c + 1}: {int(s.min())}-{int(s.max())} hh"
    band = pd.Series(tid).map(codes).map(names).to_numpy()
    band_names = [names[c] for c in range(q)]

    allm = {**models, **extra}
    if len(fold_names) > 1:
        rows = []
        for fi, fname in enumerate(fold_names):
            fm = fold == fi
            dts = panel.d0 + pd.to_timedelta(day_idx[fm, 0], unit="D")
            for name, P in allm.items():
                for hx, h in enumerate(horizons):
                    w, b_, n = wape_bias(P[fm, hx], A[fm, hx])
                    rows.append(dict(fold=fname, model=name, horizon=h, wape=w, bias=b_, n=n,
                                     targets=f"{dts.min():%Y-%m}..{(dts.max() + pd.Timedelta(days=horizons[-1])):%Y-%m}"))
        ft = pd.DataFrame(rows)
        os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
        ft.to_csv(os.path.join(args.out_dir, "results", "metrics_by_fold.csv"), index=False)
        print("\n=== Pooled WAPE by fold (each fold is tested on its own months; rows: fold, target months) ===")
        ft["label"] = ft["fold"] + "  " + ft["targets"]
        for name in [n for n in ("direct", "seasonal_naive", "thermal_baseline", "persistence7", "persistence30") if n in allm]:
            t = ft[ft["model"] == name].pivot(index="label", columns="horizon", values="wape")
            print(f"-- {name}\n" + _fmt(t, lambda x: f"{x:6.1%}").to_string())
    tab = score_table(allm, A, tid, day_idx, band, horizons, band_names)
    base = tab[tab["model"] == "persistence30"].set_index(["horizon", "band"])["wape"]
    tab["skill_vs_persistence30"] = [1 - r.wape / base.get((r.horizon, r.band), np.nan) if np.isfinite(r.wape) else np.nan
                                     for r in tab.itertuples()]
    out_dir = os.path.join(args.out_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    if cal is not None and len(getattr(panel, "cal_tables", (pd.DataFrame(),))[0]):
        panel.cal_tables[0].to_csv(os.path.join(out_dir, "calendar_factors.csv"), index=False)
        panel.cal_tables[1].to_csv(os.path.join(out_dir, "calendar_holidays.csv"), index=False)
    if cal is not None:                                      # the factor of every calendar day, for plot_forecast_windows.py
        pd.DataFrame({"date": (panel.d0 + pd.to_timedelta(np.arange(len(cal)), unit="D")).date, "factor": cal}) \
            .to_csv(os.path.join(out_dir, "calendar_daily.csv"), index=False)

    boot = []
    pairs = [p for p in [("direct", "persistence30"), ("direct", "persistence7"), ("direct", "thermal_baseline"),
                         ("direct", "seasonal_naive"), ("direct", "bottom_up"), ("average", "bottom_up"), ("average", "direct")]
             if p[0] in allm and p[1] in allm]
    for hx, h in enumerate(horizons):
        for a_name, b_name in pairs:
            est, lo, hi_ = paired_bootstrap(allm[a_name][:, hx], allm[b_name][:, hx], A[:, hx], tid)
            boot.append(dict(horizon=h, a=a_name, b=b_name, wape_a_minus_b=est, p05=lo, p95=hi_,
                             verdict=("a better" if hi_ < 0 else "b better" if lo > 0 else "no clear difference")))
    boot = pd.DataFrame(boot)
    boot.to_csv(os.path.join(out_dir, "paired_bootstrap.csv"), index=False)

    coato_of = dict(zip(np.asarray(panel.ids).astype(str), getattr(panel, "coato_of", np.full(len(panel.ids), "?"))))
    tr_tab = (pd.DataFrame({"transformer_id": tid, "n_hh": n_hh, "band": band, "coato": [coato_of.get(t, "?") for t in tid]})
              .groupby("transformer_id").agg(n_hh=("n_hh", "first"), band=("band", "first"), coato=("coato", "first")))
    for name, P in models.items():
        if name == "persistence30":
            continue
        for hx, h in enumerate(horizons):
            v = np.isfinite(A[:, hx]) & (A[:, hx] > 0)
            g = pd.DataFrame({"tr": tid[v], "e": np.abs(P[v, hx] - A[v, hx]), "a": A[v, hx]}).groupby("tr").sum()
            tr_tab[f"wape_{name}_t{h}"] = g["e"] / g["a"]
    tr_tab.to_csv(os.path.join(out_dir, "wape_by_transformer.csv"))
    band_tab = tr_tab.groupby("band").agg(transformers=("n_hh", "size"), min_hh=("n_hh", "min"), max_hh=("n_hh", "max"))
    band_tab.to_csv(os.path.join(out_dir, "size_bands.csv"))

    print_summary(tab, boot, horizons, band_names, A, models)
    keep = pd.DataFrame({"transformer_id": tid, "origin_date": panel.d0 + pd.to_timedelta(org, unit="D"), "n_hh": n_hh,
                         "fold": [fold_names[int(f)] for f in fold]})
    for hx, h in enumerate(horizons):
        keep[f"actual_t{h}"] = A[:, hx]
        for name, P in allm.items():
            keep[f"{name}_t{h}"] = P[:, hx]
    try:
        keep.to_parquet(os.path.join(out_dir, "forecasts_common.parquet"), index=False)
    except Exception as e:                                   # no pyarrow / fastparquet
        keep.to_csv(os.path.join(out_dir, "forecasts_common.csv.gz"), index=False)
        print(f"[compare] parquet not written ({type(e).__name__}); wrote forecasts_common.csv.gz instead")
    try:
        error_diagnostics(out_dir, allm, A, tid, day_idx, horizons, panel)
    except Exception:
        import traceback
        traceback.print_exc()
        print("[compare] error diagnostics failed (the tables above and the csv files are already saved)")
    try:
        make_charts(out_dir, tab, horizons, band_names, models, A, tid, org, day_idx, n_hh, panel)
    except Exception:
        import traceback
        traceback.print_exc()
        print("[compare] charts failed (the tables above and the csv files are already saved)")
    return tab, boot


def error_diagnostics(out_dir, allm, A, tid, day_idx, horizons, panel, min_tr=100, top=8):
    """Where is the error?  (1) WAPE by target month; (2) the target days with the largest system-level error."""
    names = [n for n in ("direct", "bottom_up", "persistence7", "persistence30", "thermal_baseline", "seasonal_naive") if n in allm]
    min_tr = min(min_tr, max(3, int(0.3 * len(np.unique(tid)))))        # small panels: do not demand 100 transformers per day
    hx_list = [horizons.index(h) for h in (7, 30) if h in horizons]
    rows, day_rows = [], []
    for hx in hx_list:
        h = horizons[hx]
        dts = panel.d0 + pd.to_timedelta(day_idx[:, hx], unit="D")
        month = np.asarray(dts.strftime("%Y-%m"))
        for n in names:
            P = allm[n][:, hx]
            v = np.isfinite(P) & np.isfinite(A[:, hx]) & (A[:, hx] > 0)
            g = pd.DataFrame({"m": month[v], "e": np.abs(P[v] - A[v, hx]), "s": P[v] - A[v, hx], "a": A[v, hx]}).groupby("m").sum()
            for m_, r in g.iterrows():
                rows.append(dict(model=n, horizon=h, target_month=m_, wape=r["e"] / r["a"], bias=r["s"] / r["a"]))
        P = allm["direct"][:, hx]
        v = np.isfinite(A[:, hx]) & (A[:, hx] > 0)
        for n in names:
            v &= np.isfinite(allm[n][:, hx])
        cols = {"day": day_idx[v, hx], "a": A[v, hx], "direct": P[v]}
        for n in names:
            if n != "direct":
                cols[n] = allm[n][v, hx]
        dd = pd.DataFrame(cols)
        dd["n"] = 1
        g = dd.groupby("day").sum()
        g = g[g["n"] >= min_tr]
        out = pd.DataFrame({"date": (panel.d0 + pd.to_timedelta(g.index.to_numpy(), unit="D")).date, "horizon": h,
                            "transformers": g["n"].to_numpy(), "actual_kwh": g["a"].to_numpy()})
        out["weekday"] = pd.to_datetime(out["date"]).dt.strftime("%a")
        for n in names:
            out[f"err_{n}"] = (g[n] / g["a"] - 1).to_numpy()
        day_rows.append(out)
    mt = pd.DataFrame(rows)
    mt.to_csv(os.path.join(out_dir, "wape_by_target_month.csv"), index=False)
    dy = pd.concat(day_rows, ignore_index=True)
    dy.to_csv(os.path.join(out_dir, "system_error_by_day.csv"), index=False)
    for h in sorted(mt["horizon"].unique()):
        t = mt[mt["horizon"] == h].pivot(index="target_month", columns="model", values="wape")
        print(f"\n=== WAPE by target month, t+{h} (months on the rows; this is where the error sits) ===")
        print(_fmt(t[[n for n in names if n in t]], lambda x: f"{x:6.1%}").to_string())
    d7 = dy[dy["horizon"] == (7 if 7 in horizons else horizons[0])]
    if d7.empty:
        print(f"[diagnostics] no target day has >= {min_tr} transformers; system_error_by_day.csv is empty")
        return
    w =d7.reindex(d7["err_direct"].abs().sort_values(ascending=False).index).head(top)
    print(f"\n=== Target days with the largest system-level error of 'direct' at t+{d7['horizon'].iloc[0]} "
          f"(all transformers summed; err = forecast / actual - 1; >= {min_tr} transformers) ===")
    print(w[["date", "weekday", "transformers", "err_direct"] + [f"err_{n}" for n in names if n != "direct"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.1%}"))
    print(f"[diagnostics] median |system error| of 'direct' at t+{d7['horizon'].iloc[0]}: {d7['err_direct'].abs().median():.1%}; "
          f"{(d7['err_direct'].abs() > 0.15).sum()} of {len(d7)} days are off by more than 15%  -> system_error_by_day.csv")


def _fmt(df, f):
    return df.map(f) if hasattr(df, "map") else df.applymap(f)


def print_summary(tab, boot, horizons, band_names, A, models):
    t = tab[tab["band"] == "all"]
    print("\n=== Pooled WAPE over all transformers (lower is better; size-weighted) ===")
    print(_fmt(t.pivot(index="model", columns="horizon", values="wape"), lambda x: f"{x:6.1%}").to_string())
    print("\n=== Signed bias (+ = over-forecast) ===")
    print(_fmt(t.pivot(index="model", columns="horizon", values="bias"), lambda x: f"{x:+6.1%}").to_string())
    print("\n=== Median per-transformer WAPE (a typical transformer) ===")
    print(_fmt(t.pivot(index="model", columns="horizon", values="median_transformer_wape"), lambda x: f"{x:6.1%}").to_string())
    print("\n=== System total (all transformers summed per target day) sysWAPE ===")
    print(_fmt(t.pivot(index="model", columns="horizon", values="sys_wape"), lambda x: f"{x:6.1%}").to_string())
    for h in horizons:
        print(f"\n=== WAPE by size band, t+{h} ===")
        x = tab[(tab["horizon"] == h) & (tab["band"] != "all") & tab["model"].isin(list(models))]
        print(_fmt(x.pivot(index="band", columns="model", values="wape").reindex(band_names), lambda v: f"{v:6.1%}").to_string())
    print("\n=== Paired bootstrap over transformers: WAPE(a) - WAPE(b), 90% interval ===")
    print(boot.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


# =============================================================================
# Charts
# =============================================================================
def make_charts(out_dir, tab, horizons, band_names, models, A, tid, org, day_idx, n_hh, panel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    names = [n for n in MODEL_STYLE if n in models]

    # 1. WAPE by horizon
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    t = tab[tab["band"] == "all"]
    ends = {}
    for n in names:
        lab, col, mk, ls = MODEL_STYLE[n]
        y = t[t["model"] == n].set_index("horizon").loc[horizons, "wape"] * 100
        ax.plot(horizons, y, color=col, marker=mk, linestyle=ls, lw=2, ms=6, label=lab)
        ends[n] = float(y.iloc[-1])
    gap, ypos = max(ends.values()) * 0.06, {}              # end labels: keep them apart
    for n in sorted(ends, key=ends.get):
        ypos[n] = max(ends[n], ypos[max(ypos, key=ypos.get)] + gap) if ypos else ends[n]
    for n in names:
        ax.annotate(f"{ends[n]:.1f}%", (horizons[-1], ends[n]), xytext=(horizons[-1] + 2, ypos[n]), textcoords="data",
                    color=MODEL_STYLE[n][1], va="center", fontsize=9)
    ax.set_xticks(horizons), ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.set_ylabel("WAPE, % of transformer kWh/day"), ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID), ax.set_axisbelow(True)
    ax.set_title("Transformer-level error by forecast horizon", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(horizons[0] - 3, horizons[-1] + 9)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "wape_by_horizon.png"), dpi=150), plt.close(fig)

    # 2. WAPE by size band, one panel per horizon
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.6 * len(horizons), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(names)
    for ax, h in zip(axes, horizons):
        for k, n in enumerate(names):
            lab, col, mk, ls = MODEL_STYLE[n]
            y = (tab[(tab["model"] == n) & (tab["horizon"] == h)].set_index("band").reindex(band_names)["wape"] * 100)
            ax.bar(np.arange(len(band_names)) + k * w - 0.4 + w / 2, y, width=w * 0.92, color=col,
                   label=lab, alpha=0.55 if n == "persistence30" else 1.0)
        ax.set_xticks(np.arange(len(band_names)))
        ax.set_xticklabels([b.split(":")[0] for b in band_names])
        ax.set_title(f"t+{h}", loc="left", fontsize=10), ax.grid(axis="y", color=GRID), ax.set_axisbelow(True)
    axes[0].set_ylabel("WAPE, %")
    fig.text(0.5, -0.02, "Size band by household count: " + " | ".join(band_names), ha="center", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "wape_by_size_band.png"), dpi=150, bbox_inches="tight"), plt.close(fig)

    # 3. three example transformers (largest, median, smallest) at one horizon
    hx = horizons.index(30) if 30 in horizons else len(horizons) // 2
    h = horizons[hx]
    size = pd.Series(n_hh, index=tid).groupby(level=0).first().sort_values()
    picks = {"smallest": size.index[0], "median": size.index[len(size) // 2], "largest": size.index[-1]}
    fig, axes = plt.subplots(len(picks), 1, figsize=(8.5, 2.9 * len(picks)), sharex=True)
    for ax, (kind, t_id) in zip(np.atleast_1d(axes), picks.items()):
        r = np.flatnonzero(tid == t_id)
        r = r[np.argsort(day_idx[r, hx])]
        x = (panel.d0 + pd.to_timedelta(day_idx[r, hx], unit="D"))
        ax.plot(x, A[r, hx], color=INK, lw=2, label="Actual")
        for n in [n for n in ("bottom_up", "direct") if n in models]:
            lab, col, mk, ls = MODEL_STYLE[n]
            ax.plot(x, models[n][r, hx], color=col, lw=1.6, label=lab)
        ax.set_title(f"{kind} transformer {t_id} ({int(size[t_id])} households), forecast issued {h} days earlier",
                     loc="left", fontsize=9)
        ax.set_ylabel("kWh / day"), ax.grid(axis="y", color=GRID), ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "examples_t%d.png" % h), dpi=150), plt.close(fig)
    print(f"[charts] wrote wape_by_horizon.png, wape_by_size_band.png, examples_t{h}.png to {out_dir}")


# =============================================================================
# run: orchestrates the stages
# =============================================================================
def cmd_run(args):
    wp = wp_module()
    stages = [s.strip() for s in args.stages.split(",")]
    unknown = set(stages) - {"direct", "bottomup", "compare"}
    if unknown:
        raise SystemExit(f"unknown stage(s) {sorted(unknown)}; choose from direct, bottomup, compare")
    info, hh_meta, kwh = load_world(args.work_dir)
    clean_path = os.path.join(args.work_dir, "clean", "transformer_clean.npz")
    source = args.source if args.source != "auto" else ("clean" if os.path.exists(clean_path) else "raw")
    if source == "clean" and not os.path.exists(clean_path):
        raise SystemExit(f"{clean_path} not found: run  python transformer_clean.py --work-dir {args.work_dir}  first (or use --source raw)")
    if source == "raw" and int(info.get("n_coatos", 1)) > 1:
        raise SystemExit(f"this work dir holds {info['n_coatos']} coatos: use the cleaned panel (python transformer_clean.py --work-dir ...) "
                         f"or `prepare --coato <code>` into another work dir; the raw panel path forecasts one coato at a time.")
    if args.include_old:
        wp.INCLUDE_OLD = args.include_old
    if args.old_flag:
        wp.OLD_REGIME_FLAG = True
    cutoffs = (args.train_cutoff or wp.TRAIN_CUTOFF, args.val_start or wp.VAL_START)
    if args.hh_model_dir and os.path.exists(os.path.join(args.hh_model_dir, "columns.json")):
        with open(os.path.join(args.hh_model_dir, "columns.json")) as fh:
            cfg = json.load(fh)
        cutoffs = (args.train_cutoff or cfg.get("train_cutoff", wp.TRAIN_CUTOFF), args.val_start or cfg.get("val_start", wp.VAL_START))
    elif "bottomup" in stages:
        raise SystemExit(f"--hh-model-dir {args.hh_model_dir!r} has no columns.json: point it at the household model folder "
                         f"(the OUTPUT_DIR of a finished `whole_pipeline_v8.py train`), or drop 'bottomup' from --stages")
    args.train_cutoff = cutoffs[0]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[config] source '{source}' | train cutoff {cutoffs[0]} | validation after {cutoffs[1]} | "
          f"test origins from {cutoff_test_start(wp, cutoffs[0]).date()} | normals '{args.normal_mode}' | "
          f"forecast mode '{wp.FORECAST_MODE}' | calendar '{args.calendar}' | stages {','.join(stages)} | outputs {args.out_dir}")
    folds = parse_folds(args, cutoffs)
    if args.folds:
        print("[config] direct folds: " + " | ".join(
            f"f{i + 1}: cutoff {c}, validation ({vs}, {ve or c}], test origins {(pd.Timestamp(c) + pd.Timedelta(days=wp.HORIZON)).date()}"
            f"..{te or 'end'}" for i, (c, vs, ve, te) in enumerate(folds)))
    print(f"[config] direct: validation {'(' + str(cutoffs[1]) + ', ' + args.val_end + ']' if args.val_end else 'household split'} | "
          f"old regime '{wp.INCLUDE_OLD}'{' + regime_old feature' if wp.OLD_REGIME_FLAG else ''} | "
          f"year-ago features {'off' if args.no_year_ago else 'on'} | final-iters cap {args.final_iters_cap}")

    with stage("panel"):
        if source == "clean":
            panel = load_clean_panel(wp, args, hh_meta)
            first_cut = pd.Timestamp(folds[0][0])
            fe = getattr(panel, "fit_end", None)
            if fe is None:
                print("[causal] WARNING: this cleaned panel was built without --fit-end: its stable core and heavy consumers were "
                      "chosen with test-period data. Re-run  transformer_clean.py --fit-end " + str(first_cut.date()))
            elif pd.Timestamp(fe) > first_cut:
                print(f"[causal] WARNING: the cleaned panel's fit end {fe} is after the first train cutoff {first_cut.date()}: "
                      f"its stable core used data the first fold's test period should not see. Re-run transformer_clean.py "
                      f"--fit-end {first_cut.date()}")
            else:
                print(f"[causal] cleaned panel fit end {fe} <= first train cutoff {first_cut.date()}: no look-ahead in the core")
        else:
            coato = info["coato"].split(",")[0]
            panel = build_panel(wp, info, hh_meta, kwh, args)
            panel.coato_of = np.full(len(panel.ids), coato, dtype=object)
    with stage("climate + calendar + forecasts"):
        clim, _ = load_climate(wp, args, args.hh_model_dir)
        if source == "raw":
            check_coato(clim, panel.coato_of[0])
        panel = keep_known_coatos(panel, clim)
        apply_calendar(wp, panel, args, cutoffs[0])
        # origin stride: at most --max-direct-windows direct windows (about 13 KB of features each)
        r_all, _, _ = windows(wp, panel, origin_grid(wp, panel, SimpleNamespace(origin_stride=1)))
        need = int(np.ceil(len(r_all) / max(args.max_direct_windows, 1))) if args.max_direct_windows else 1
        if need > args.origin_stride:
            print(f"[windows] {len(r_all):,} complete input windows; --origin-stride raised from {args.origin_stride} to {need} "
                  f"to stay under --max-direct-windows {args.max_direct_windows:,}")
            args.origin_stride = need
        if args.bu_stride % args.origin_stride:
            new = int(np.ceil(args.bu_stride / args.origin_stride) * args.origin_stride)
            print(f"[windows] --bu-stride {args.bu_stride} is not a multiple of the origin stride {args.origin_stride}: using {new}")
            args.bu_stride = new
        O = origin_grid(wp, panel, args)
        if args.forecast_sigma_scale != 1.0 and wp.FORECAST_MODE == "simulated":
            sc = max(float(args.forecast_sigma_scale), 0.01)
            wp.FC_SIGMA_DAY1, wp.FC_SIGMA_DAY10 = wp.FC_SIGMA_DAY1 * sc, wp.FC_SIGMA_DAY10 * sc
            print(f"[forecast] simulated forecast error scaled by {sc:g}: std {wp.FC_SIGMA_DAY1:.2f} C (day 1) .. "
                  f"{wp.FC_SIGMA_DAY10:.2f} C (day 10); days 11+ still use the normal")
        fc = forecast_grids(wp, clim, sorted(set(panel.coato_of)), panel, O, args)
        end = np.datetime64((panel.d0 + pd.Timedelta(days=int(O.max()) + 10)).date())
        last_w = clim.start + np.timedelta64(clim.n_days - 1, "D")
        if last_w < end:
            print(f"[weather] WARNING: weather ends {last_w}, before the last origin + 10 days ({end}); "
                  f"the forecast for those origins falls back to the normal")

    tr = start_tracking(args)
    with tr.run(f"transformer forecast {dt.datetime.now():%Y-%m-%d %H:%M}", tags={"stage": "transformer_forecast", "source": source}):
        tr.params({k: v for k, v in vars(args).items() if k not in ("func",)})
        if "direct" in stages:
            with stage("direct"):
                stage_direct(wp, args, dict(), panel, clim, fc, cutoffs)
        if "bottomup" in stages:
            with stage("bottom-up"):
                stage_bottom_up(wp, args, dict(), panel, clim, fc, hh_meta, kwh)
        if "compare" in stages:
            with stage("compare"):
                tab, boot = stage_compare(wp, args, panel)
            t = tab[tab["band"] == "all"]
            tr.metrics({f"wape/{r.model}/t{r.horizon}": r.wape for r in t.itertuples() if np.isfinite(r.wape)})
            tr.metrics({f"sys_wape/{r.model}/t{r.horizon}": r.sys_wape for r in t.itertuples() if np.isfinite(r.sys_wape)})
            for f in os.listdir(os.path.join(args.out_dir, "results")):
                if f.endswith((".csv", ".png")):
                    tr.artifact(os.path.join(args.out_dir, "results", f), subdir="transformer_forecast")
        tr.artifact(os.path.join(args.out_dir, "run_log.txt"), subdir="logs")


def start_tracking(args):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mlflow_utils import Tracker
        return Tracker("electricity/transformer-forecast", uri=args.mlflow_uri, enabled=False if args.no_mlflow else None)
    except Exception:                                                  # noqa: BLE001
        class _Off:
            enabled = False

            def __getattr__(self, name):
                if name == "run":
                    return lambda *a, **k: contextlib.nullcontext(None)
                return lambda *a, **k: None
        return _Off()


# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare", help="Spark: households of one coato -> dense household x day matrix")
    a.add_argument("--coato", default=None, help="coato_code as stored in customer_metadata_current (5 digits), comma list, or 'all'. "
                   f"Default: {DEFAULT_COATO}, or all coatos when --n-transformers is given")
    a.add_argument("--n-transformers", type=int, default=0,
                   help="sample this many transformers (all their households) instead of taking whole coatos; e.g. 1000 for a national audit")
    a.add_argument("--spread", choices=["sqrt", "even", "proportional"], default="sqrt",
                   help="how the sample is split over coatos (sqrt of each coato's transformer count; even; or the real mix)")
    a.add_argument("--min-daily", type=int, default=20, help="sampling: a transformer needs this many households with daily readings")
    a.add_argument("--active-days", type=int, default=180, help="sampling: 'has daily readings' = a reading in the last N days of the range")
    a.add_argument("--no-census", action="store_true", help="sampling: skip the pass over the daily table (draws from the metadata alone)")
    a.add_argument("--min-on-id", type=int, default=30, help="sampling: smallest transformer id (households on it today) to draw")
    a.add_argument("--max-on-id", type=int, default=1500, help="sampling: ids with more households are placeholders and are not drawn")
    a.add_argument("--seed", type=int, default=0, help="sampling seed (same seed -> same transformers)")
    a.add_argument("--max-meters", type=int, default=400000, help="sampling: stop if the draw holds more households than this (memory)")
    a.add_argument("--work-dir", required=True)
    a.add_argument("--start", default="2022-01-01", help="first day to read (default: everything the daily table has)")
    a.add_argument("--end", default=None, help="last day to read (default: today)")
    a.add_argument("--batches", default=None, help="comma list of hash batches (default: ALL batches in the store, whatever WINDOW_BATCHES says); a subset is a random subset of households")
    a.add_argument("--cores", type=int, default=6)
    a.add_argument("--driver-memory", default="24g")
    a.add_argument("--local-dir", default=None, help="Spark temp dir; use a LOCAL disk (not NFS)")
    a.set_defaults(func=cmd_prepare, log=True)

    r = sub.add_parser("run", help="direct model, bottom-up, comparison")
    r.add_argument("--work-dir", required=True)
    r.add_argument("--stages", default="direct,bottomup,compare",
                   help="direct,bottomup,compare. Without a household model use  --stages direct,compare  (direct vs persistence)")
    r.add_argument("--source", choices=["auto", "clean", "raw"], default="auto",
                   help="clean = <work-dir>/clean/transformer_clean.npz from transformer_clean.py (many coatos, stable core, flagged days); "
                        "raw = the panel built here from hh_kwh.npy (one coato). auto: clean if it exists")
    r.add_argument("--out-dir", default=None, help="where the forecasts and results go (default <work-dir>/forecast)")
    r.add_argument("--calendar", choices=["none", "dow", "dow+holidays"], default="dow+holidays",
                   help="weekday / holiday factors estimated on the training period only and applied to every model's target days")
    r.add_argument("--holidays-csv", default=None, help="date,class file replacing the built-in holiday list")
    r.add_argument("--max-direct-windows", type=int, default=300000,
                   help="cap on direct training windows (memory ~13 KB each); the origin stride is raised to stay under it. 0 = no cap")
    r.add_argument("--hh-model-dir", default=None, help="folder written by whole_pipeline_v8.py (OUTPUT_DIR): "
                                                        "catboost_model.cbm, columns.json, meter_segments.csv ...")
    r.add_argument("--hh-model", choices=["catboost", "xgboost"], default=None, help="default: the model chosen at training")
    r.add_argument("--train-cutoff", default=None, help="default: from the household model")
    r.add_argument("--val-start", default=None, help="default: from the household model")
    r.add_argument("--val-end", default=None,
                   help="direct model: validation = origins in (--val-start, --val-end]; fit = training origins at least 60 days "
                        "away on both sides. Use a block spanning autumn -> spring, e.g. --val-start 2024-09-01 --val-end 2025-04-30. "
                        "Default: the household split (validation from --val-start to the train cutoff, usually one summer)")
    r.add_argument("--folds", default=None,
                   help="rolling evaluation: comma list of CUTOFF:VAL_START:VAL_END, one per fold, e.g. "
                        "'2025-03-01:2024-08-01:2025-01-31,2025-09-01:2024-09-01:2025-04-30'. Each fold trains on origins up to its "
                        "cutoff (early stopping on its validation block) and forecasts the origins from its cutoff + 60 days up to the "
                        "next fold's; together the folds test a full year. Overrides --train-cutoff / --val-start / --val-end for the "
                        "direct model")
    r.add_argument("--include-old", choices=["none", "all", "non_winter"], default=None,
                   help="old-regime (before 2024-08-01) training windows; default: whole_pipeline_v8.INCLUDE_OLD")
    r.add_argument("--old-flag", action="store_true", help="add the feature regime_old (1 for old-regime origins) to the direct model")
    r.add_argument("--no-year-ago", action="store_true",
                   help="direct model: leave out the ly_ratio_* features (same 60 days last year / the 30 days before); for the ablation")
    r.add_argument("--normal-mode", choices=["loyo", "all"], default="loyo",
                   help="climate normals: loyo = leave the year out (as the household evaluation), all = deployment")
    r.add_argument("--min-households", type=int, default=5, help="smallest transformer to model")
    r.add_argument("--max-households", type=int, default=1500, help="ids with more households are placeholders and are dropped")
    r.add_argument("--min-hh-days", type=int, default=30, help="a household needs this many reading days to count in a panel")
    r.add_argument("--min-coverage", type=float, default=0.8, help="share of the panel that must report for a day to count")
    r.add_argument("--gap-fill", type=int, default=3, help="interpolate gaps up to this many days in the model series")
    r.add_argument("--origin-stride", type=int, default=1, help="use every n-th origin day for the direct model")
    r.add_argument("--bu-stride", type=int, default=7, help="bottom-up origins: every n-th day (a multiple of --origin-stride)")
    r.add_argument("--bu-min-share", type=float, default=0.6, help="drop a transformer-origin if fewer households than this share could be forecast")
    r.add_argument("--hh-gap-fill", type=int, default=None,
                   help="bottom-up: interpolate gaps of up to this many days in a household's history (0 = none)")
    r.add_argument("--min-input-days", type=int, default=None,
                   help="bottom-up: a household needs this many observed days of the last 90 (default: the household model's rule, "
                        "from its columns.json; 80 for older models)")
    r.add_argument("--min-recent-days", type=int, default=None, help="... and this many of the last --recent-window days")
    r.add_argument("--recent-window", type=int, default=None, help="... (default: the model's rule; 30, i.e. all of the last 30, for older models)")
    r.add_argument("--size-weight", choices=["none", "sqrt", "linear"], default="sqrt",
                   help="direct model sample weight = households^(0/0.5/1): big transformers count more")
    r.add_argument("--size-bands", type=int, default=4)
    r.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    r.add_argument("--final-iters-cap", type=int, default=150)
    r.add_argument("--forecast-sigma-scale", type=float, default=1.0,
                   help="simulated weather forecasts only: multiply the forecast error (1.0 to 2.5 C over days 1-10) by this. "
                        "0.01 = almost perfect weather for days 1-10: shows how much of the error is the weather forecast "
                        "(diagnostic, not deployable)")
    r.add_argument("--drop-features", default="", help="direct model: comma list of features to leave out, exact names or a prefix "
                   "ending in * (e.g. 'origin_doy*,n_households'); for ablations")
    r.add_argument("--chunk", type=int, default=100000, help="rows per feature-building chunk (memory)")
    r.add_argument("--max-transformers", type=int, default=0, help="debug: random subset of transformers")
    r.add_argument("--mlflow-uri", default=None)
    r.add_argument("--no-mlflow", action="store_true")
    r.set_defaults(func=cmd_run, log=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    os.makedirs(args.work_dir, exist_ok=True)
    log_dir = args.work_dir
    if args.cmd == "run":
        args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir)) if args.out_dir else os.path.join(args.work_dir, "forecast")
        os.makedirs(args.out_dir, exist_ok=True)
        log_dir = args.out_dir
    with logging_to(os.path.join(log_dir, "run_log.txt")):
        return args.func(args)


if __name__ == "__main__":
    main()