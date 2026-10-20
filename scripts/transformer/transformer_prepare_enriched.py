#!/usr/bin/env python3
"""
Transformer panel from the national ENRICHED household dataset (v1.4, staging): a drop-in replacement for
`transformer_forecast.py prepare`. It writes the same work-dir files (hh_kwh.npy, hh_meta.csv, hh_days.json,
weather_daily.parquet, transformers_sampled.csv), so transformer_clean.py, transformer_noise_audit.py and
`transformer_forecast.py run` work on it unchanged.

    source ~/work/ec_env.sh          # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT
    python scripts/transformer/transformer_prepare_enriched.py --work-dir ~/work/tf_enriched --n-transformers 2000

Follows HOUSEHOLD_NATIONAL_DATASETS_GUIDE.md (evidence date 2026-10-19):
  * Unit = customer (customer_key + coato_code; a customer's meters are already combined), not a meter.
  * Labels: only `label_state = 'complete_observed'` with a clean `kwh` is kept (HH-GAP-001-A1 rule 1). Estimated 2-3-day
    splits, partial days, unlabelled days and long-gap allocations stay missing: they are never training or scoring labels
    (rules 1-2), and keeping only 1-day readings means no input day shares a reading interval with a target day (rule 3).
    `--labels clean` also keeps complete_with_estimated days (NOT for scoring; only for experiments).
  * Transformer = `transformer_id` of the enriched table (last-known network mapping, applied to all dates). It is NOT the
    raw transformer_id of customer_metadata_current (a different numbering), so samples from the old panel do not carry
    over. `placeholder_transformer` ids (> 1,000 customers) are never drawn.
  * Reads are bounded: one district at a time, only the needed columns, date / period / transformer filters pushed down
    (rule 10). Nothing national is loaded at once; customer rows stay in the work dir (keys are pseudonymous but
    sensitive: do not share hh_meta.csv or hh_kwh.npy).
  * Only districts whose worker root has `_SUCCESS` are read. The dataset is a STAGING build with national acceptance
    pending; low_use and seasonality_class (on hold) are not read.
  * Weather: daily t_mean / t_min / t_max of the same table (branch-observed, 24 complete hourly values, local time),
    unless --weather-file points at an existing weather_daily.parquet (columns coato_id, d, t, tmin, tmax).

Steps
  1. census   per district, customers with an observed day in the last --active-days, per transformer
  2. sample   --n-transformers drawn over coatos (sqrt rule, as before); transformers on one coato only, not placeholders,
              --min-on-id .. --max-on-id customers on the id, >= --min-daily customers with observed readings
  3. read     per sampled district: every day of the customers on the sampled transformers -> household x day matrix
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core")))  # the shared modules
import transformer_forecast as tf                                  # noqa: E402  (logging, stages, quota rule)

from enriched_io import (ENRICHED_ROOT, LABELS, WINDOW_END, district_daily, finished_districts, kept_label,  # noqa: E402
                         open_root, period_range)


# -----------------------------------------------------------------------------------------------------------------
def census(fs, root, codes, args, keep):
    """Per transformer id: customers with a kept label in the last --active-days, customers on the id under the mapping,
    placeholder flag, and the coato(s) it appears in."""
    end = pd.Timestamp(args.end)
    since = end - pd.Timedelta(days=args.active_days)
    periods = period_range(since, end)
    flt = (ds.field("period").isin(periods) & (ds.field("date") >= since.date()) & (ds.field("date") <= end.date())
           & kept_label(keep) & ds.field("transformer_id").is_valid())
    cols = ["customer_key", "transformer_id", "placeholder_transformer", "transformer_customer_count"]
    rows, t0 = [], time.time()
    for n, code in enumerate(codes):
        pairs, info = [], []
        for b in district_daily(fs, root, code).scanner(columns=cols, filter=flt, batch_size=262_144,
                                                        use_threads=True).to_batches():
            if not b.num_rows:
                continue
            t = pa.Table.from_batches([b])
            pairs.append(t.group_by(["transformer_id", "customer_key"]).aggregate([]))
            info.append(t.group_by("transformer_id").aggregate([("placeholder_transformer", "max"),
                                                                ("transformer_customer_count", "max")]))
        if pairs:
            p = pa.concat_tables(pairs).group_by(["transformer_id", "customer_key"]).aggregate([])
            d = p.group_by("transformer_id").aggregate([("customer_key", "count")]).to_pandas()
            i = (pa.concat_tables(info).group_by("transformer_id")
                 .aggregate([("placeholder_transformer_max", "max"), ("transformer_customer_count_max", "max")]).to_pandas())
            d = d.merge(i, on="transformer_id", how="left")
            d.columns = ["transformer_id", "n_daily", "placeholder", "n_on_id"]
            d["coato_code"] = code
            rows.append(d)
        if (n + 1) % 10 == 0 or n + 1 == len(codes):
            el = time.time() - t0
            print(f"  [census] {n + 1}/{len(codes)} districts, {el / 60:.1f} min, about {el / (n + 1) * (len(codes) - n - 1) / 60:.1f} "
                  f"min left", flush=True)
    if not rows:
        raise SystemExit(f"no observed customer-days between {since.date()} and {end.date()} in the districts read")
    c = pd.concat(rows, ignore_index=True)
    c["transformer_id"] = c["transformer_id"].astype("int64").astype(str)
    c["placeholder"] = c["placeholder"].fillna(False).astype(bool)
    return c


def sample(c, args):
    """Same eligibility and square-root spread as transformer_forecast.select_transformers."""
    per_id = c.groupby("transformer_id").agg(n_coatos=("coato_code", "nunique"), coato_code=("coato_code", "first"),
                                             n_daily=("n_daily", "sum"), n_on_id=("n_on_id", "max"),
                                             placeholder=("placeholder", "max")).reset_index()
    n_on = per_id["n_on_id"].fillna(per_id["n_daily"])               # no mapping count: fall back to customers seen
    multi, plh = per_id["n_coatos"] > 1, per_id["placeholder"]
    small, big = n_on < args.min_on_id, n_on > args.max_on_id
    nodaily = per_id["n_daily"] < args.min_daily
    pool = per_id[~multi & ~plh & ~small & ~big & ~nodaily]
    by = per_id.groupby("coato_code").agg(transformers=("transformer_id", "size"), customers_with_daily=("n_daily", "sum"),
                                          eligible=("transformer_id", lambda s: int(s.isin(pool["transformer_id"]).sum())))
    by.sort_values("customers_with_daily", ascending=False).to_csv(os.path.join(args.work_dir, "daily_coverage_by_coato.csv"))
    print(f"{len(per_id):,} transformer ids with observed customers; dropped: {int(plh.sum()):,} placeholders, "
          f"{int(multi.sum()):,} on several coatos, {int((small & ~plh).sum()):,} with < {args.min_on_id} customers on the id, "
          f"{int((big & ~plh).sum()):,} with > {args.max_on_id}, {int((nodaily & ~small & ~big & ~plh & ~multi).sum()):,} with "
          f"< {args.min_daily} observed customers -> {len(pool):,} eligible in {pool['coato_code'].nunique():,} coatos")
    if pool.empty:
        raise SystemExit("no eligible transformers; lower --min-daily / --min-on-id or widen --active-days")
    quota = tf.allocate_quota(pool.groupby("coato_code").size().to_dict(), args.n_transformers, args.spread)
    rng = np.random.default_rng(args.seed)
    picked = []
    for code, q in sorted(quota.items()):
        if q:
            sub = pool[pool["coato_code"] == code].sort_values("transformer_id")
            picked.append(sub.iloc[np.sort(rng.choice(len(sub), size=q, replace=False))])
    sel = pd.concat(picked).sort_values(["coato_code", "transformer_id"]).reset_index(drop=True)
    est = int(sel["n_on_id"].fillna(sel["n_daily"]).sum())
    print(f"sampled {len(sel):,} transformers from {sel['coato_code'].nunique():,} coatos ({args.spread} spread, seed {args.seed}); "
          f"observed customers per transformer: median {sel['n_daily'].median():.0f}, 10th pct {sel['n_daily'].quantile(.1):.0f}; "
          f"about {est:,} customers on them (~{est * 1650 * 4 / 1e9:.1f} GB as a float32 matrix)")
    if est > args.max_meters:
        raise SystemExit(f"{est:,} customers exceed --max-meters {args.max_meters:,}; lower --n-transformers")
    sel.to_csv(os.path.join(args.work_dir, "transformers_sampled.csv"), index=False)
    return sel


def read_customers(fs, root, sel, args, keep, d0, n_days):
    """Every day of the customers on the sampled transformers, district by district -> (meta, matrix, weather)."""
    cols = ["customer_key", "date", "kwh", "label_state", "transformer_id", "housing_type_id", "t_mean", "t_min", "t_max"]
    start, end = d0.date(), (d0 + pd.Timedelta(days=n_days - 1)).date()
    metas, blocks, weather, t0 = [], [], [], time.time()
    codes = sorted(sel["coato_code"].unique())
    for n, code in enumerate(codes):
        ids = [int(t) for t in sel.loc[sel["coato_code"] == code, "transformer_id"]]
        flt = ds.field("transformer_id").isin(ids) & (ds.field("date") >= start) & (ds.field("date") <= end)
        parts = []
        for b in district_daily(fs, root, code).scanner(columns=cols, filter=flt, batch_size=262_144,
                                                        use_threads=True).to_batches():
            if b.num_rows:
                parts.append(b.to_pandas())
        if not parts:
            print(f"  [read] {code}: no rows for its {len(ids)} sampled transformers")
            continue
        x = pd.concat(parts, ignore_index=True)
        day = (pd.to_datetime(x["date"]) - d0).dt.days.to_numpy()
        ok = x["label_state"].isin(keep).to_numpy() & x["kwh"].notna().to_numpy()
        cust = x.groupby("customer_key", sort=True).agg(transformer_id=("transformer_id", "first"),
                                                       housing_type_id=("housing_type_id", "last"))
        code_of = pd.Series(np.arange(len(cust)), index=cust.index)
        r = code_of.reindex(x["customer_key"]).to_numpy()
        block = np.full((len(cust), n_days), np.nan, dtype="float32")
        block[r[ok], day[ok]] = x["kwh"].to_numpy()[ok]
        blocks.append(block)
        m = pd.DataFrame({"meter_id": cust.index.astype(str), "consumer_id": cust.index.astype(str), "coato_code": code,
                          "housing_type_id": cust["housing_type_id"].astype("Int64").astype(str).replace("<NA>", ""),
                          "transformer_id": cust["transformer_id"].astype("int64").astype(str).to_numpy()})
        metas.append(m)
        w = (x.groupby("date")[["t_mean", "t_min", "t_max"]].max().reset_index()
             .rename(columns={"date": "d", "t_mean": "t", "t_min": "tmin", "t_max": "tmax"}))
        w.insert(0, "coato_id", code)
        weather.append(w)
        el = time.time() - t0
        print(f"  [read] {n + 1}/{len(codes)} {code}: {len(cust):,} customers, {int(ok.sum()):,} observed customer-days of "
              f"{len(x):,} rows ({ok.mean():.0%}); {el / 60:.1f} min, about {el / (n + 1) * (len(codes) - n - 1) / 60:.1f} min left",
              flush=True)
    if not blocks:
        raise SystemExit("no rows read for the sampled transformers")
    meta = pd.concat(metas, ignore_index=True)
    kwh = np.vstack(blocks)
    w = pd.concat(weather, ignore_index=True)
    w["d"] = pd.to_datetime(w["d"])
    return meta, kwh, w


# -----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--root", default=ENRICHED_ROOT, help="campaign root (s3://... or a local copy with the same layout)")
    p.add_argument("--coato", default="all", help="'all' (every finished district) or a comma list of coato codes")
    p.add_argument("--n-transformers", type=int, default=1000)
    p.add_argument("--spread", choices=["sqrt", "even", "proportional"], default="sqrt")
    p.add_argument("--min-daily", type=int, default=20, help="a transformer needs this many customers with observed days lately")
    p.add_argument("--active-days", type=int, default=90, help="'lately' = the last N days before --end")
    p.add_argument("--min-on-id", type=int, default=30, help="smallest transformer to draw (customers under the mapping)")
    p.add_argument("--max-on-id", type=int, default=1000, help="largest transformer to draw (placeholders are > 1,000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-meters", type=int, default=400_000, help="stop if the sample holds more customers than this")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=WINDOW_END)
    p.add_argument("--labels", choices=sorted(LABELS), default="observed",
                   help="observed (default) = complete_observed days only; clean = also complete_with_estimated (experiments only)")
    p.add_argument("--weather-file", default=None,
                   help="use this weather_daily.parquet (coato_id, d, t, tmin, tmax) instead of the dataset's own weather")
    p.add_argument("--include-unfinished", action="store_true", help="also read districts without _SUCCESS (not recommended)")
    args = p.parse_args(argv)
    args.work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    os.makedirs(args.work_dir, exist_ok=True)
    keep = LABELS[args.labels]
    with tf.logging_to(os.path.join(args.work_dir, "run_log.txt")):
        fs, root = open_root(args.root)
        with tf.stage("districts"):
            want = None if args.coato == "all" else {c.strip() for c in args.coato.split(",") if c.strip()}
            todo = finished_districts(fs, root, want, args.include_unfinished)
            print(f"labels kept: {keep} (everything else, incl. long-gap allocations, stays missing)")
        with tf.stage("census (observed customers per transformer)"):
            c = census(fs, root, todo, args, keep)
        with tf.stage("sample transformers"):
            sel = sample(c, args)
        d0 = pd.Timestamp(args.start)
        n_days = int((pd.Timestamp(args.end) - d0).days) + 1
        with tf.stage("read customers -> matrix"):
            meta, kwh, w = read_customers(fs, root, sel, args, keep, d0, n_days)
        cols = np.flatnonzero(np.isfinite(kwh).any(axis=0))
        have = np.isfinite(kwh).any(axis=1)
        c0, c1 = int(cols.min()), int(cols.max())
        kwh, meta = kwh[have][:, c0:c1 + 1], meta[have].reset_index(drop=True)
        first = (d0 + pd.Timedelta(days=c0)).date()
        np.save(os.path.join(args.work_dir, "hh_kwh.npy"), kwh)
        meta.to_csv(os.path.join(args.work_dir, "hh_meta.csv"), index=False)
        if args.weather_file:
            shutil.copyfile(os.path.expanduser(args.weather_file), os.path.join(args.work_dir, "weather_daily.parquet"))
            print(f"[weather] copied {args.weather_file}")
        else:
            w.to_parquet(os.path.join(args.work_dir, "weather_daily.parquet"), index=False)
            print(f"[weather] from the dataset: {w['coato_id'].nunique()} coatos, {w['t'].notna().mean():.1%} of coato-days with t_mean")
        with open(os.path.join(args.work_dir, "hh_days.json"), "w") as fh:
            json.dump(dict(start=str(first), n_days=int(kwh.shape[1]),
                           coato=",".join(sorted(meta["coato_code"].unique())) if meta["coato_code"].nunique() <= 5 else "multi",
                           n_coatos=int(meta["coato_code"].nunique()), sampled=True, source="household_enriched_v1.4",
                           root=args.root, labels=keep, unit="customer", created=dt.datetime.now().isoformat(timespec="seconds")),
                      fh, indent=2)
        print(f"\nsaved {kwh.shape[0]:,} customers x {kwh.shape[1]:,} days ({first} ..) on {meta['transformer_id'].nunique():,} "
              f"transformers to {args.work_dir}; {np.isfinite(kwh).mean():.1%} of cells hold an observed reading")
        years = pd.DatetimeIndex(pd.Timestamp(first) + pd.to_timedelta(np.arange(kwh.shape[1]), unit="D")).year.to_numpy()
        share = {int(y): float(np.isfinite(kwh[:, years == y]).mean()) for y in np.unique(years)}
        print("observed share of customer-days by year (low early years = 2-3-day readings, kept missing under rule 1): "
              + ", ".join(f"{y} {v:.0%}" for y, v in share.items()))
        print("next:  python transformer_clean.py --work-dir " + args.work_dir + " --fit-end <first train cutoff>")


if __name__ == "__main__":
    main()
