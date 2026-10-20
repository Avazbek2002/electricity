#!/usr/bin/env python3
"""
Feeder-level EDA on the national enriched household dataset v1.4: how many households each feeder serves.

Reads ONE month per district (bounded reads, PyArrow, a few columns), counts per feeder:
  households            distinct customer_key with any row in the month (the feeder's recorded households)
  households_observed   of those, with >= 1 complete_observed day with kWh in the month (the usable panel)
  transformers          distinct transformer_id under the feeder; placeholder_share = share of its households on
                        placeholder transformers (> 1,000 customers)
  substations           distinct substation_id (1 if the network mapping nests cleanly)
  districts             a feeder id seen in more than one district is reported (ids should be unique per network)

Outputs (<out-dir>): feeder_households.csv (one row per feeder: counts only, no customer keys),
feeder_households_hist.png, and a printed summary.  The mapping is the dataset's LAST-KNOWN network mapping applied
to every date (guide section 10), and the data cover household (B2C) customers only.

  python feeder_eda.py --period 2026-05 --out-dir ~/work/feeder_eda
"""
import argparse
import concurrent.futures as cf
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enriched_io as eio                                                   # noqa: E402

COLS = ["customer_key", "feeder_id", "substation_id", "transformer_id", "placeholder_transformer", "label_state", "kwh"]


def district_counts(fs, root, code, period, labels):
    import pyarrow.dataset as ds
    d = eio.district_daily(fs, root, code)
    have = set(d.schema.names)
    missing = [c for c in ("feeder_id", "substation_id") if c not in have]
    if missing:
        raise SystemExit(f"district {code}: columns {missing} are not in the dataset")
    cols = [c for c in COLS if c in have]
    t = d.to_table(columns=cols, filter=ds.field("period") == period).to_pandas()
    if t.empty:
        return pd.DataFrame()
    t["obs"] = t["label_state"].isin(labels) & t["kwh"].notna()
    if "placeholder_transformer" not in t:
        t["placeholder_transformer"] = False
    # one row per customer: its feeder / substation / transformer (last-known mapping: constant per customer), observed?
    c = t.groupby("customer_key", sort=False).agg(feeder_id=("feeder_id", "first"), substation_id=("substation_id", "first"),
                                                 transformer_id=("transformer_id", "first"),
                                                 placeholder=("placeholder_transformer", "first"), obs=("obs", "any"),
                                                 n_feeders=("feeder_id", "nunique"))
    multi = int((c["n_feeders"] > 1).sum())
    c["feeder_id"] = c["feeder_id"].astype("Int64").astype(str).replace("<NA>", "missing")
    f = c.groupby("feeder_id").agg(households=("obs", "size"), households_observed=("obs", "sum"),
                                   transformers=("transformer_id", "nunique"), substations=("substation_id", "nunique"),
                                   substation_id=("substation_id", "first"),
                                   placeholder_share=("placeholder", "mean")).reset_index()
    f.insert(0, "coato_code", str(code))
    f.attrs["customers_on_several_feeders"] = multi
    return f


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="2026-05", help="month YYYY-MM to count households in (default: the last full month)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=eio.ENRICHED_ROOT)
    p.add_argument("--workers", type=int, default=16, help="districts read at once (threads; I/O bound)")
    p.add_argument("--labels", default="complete_observed")
    args = p.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)
    labels = [s.strip() for s in args.labels.split(",")]
    fs, root = eio.open_root(args.root)
    codes = eio.finished_districts(fs, root)
    t0 = time.time()
    parts, multi = [], 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(district_counts, fs, root, c, args.period, labels): c for c in codes}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            if len(r):
                multi += r.attrs.get("customers_on_several_feeders", 0)
                parts.append(r)
            if k % 25 == 0 or k == len(codes):
                print(f"[read] {k}/{len(codes)} districts, {(time.time() - t0) / 60:.1f} min", flush=True)
    f = pd.concat(parts, ignore_index=True)
    f.to_csv(os.path.join(out, "feeder_households.csv"), index=False)

    miss = f[f["feeder_id"] == "missing"]
    g = f[f["feeder_id"] != "missing"]
    dup = g.groupby("feeder_id")["coato_code"].nunique()
    n = g["households"].to_numpy()
    q = np.percentile(n, [5, 25, 50, 75, 95, 99])
    print(f"\n[feeders] period {args.period}: {len(g):,} feeders in {g['coato_code'].nunique()} districts, "
          f"{int(n.sum()):,} households ({int(g['households_observed'].sum()):,} with an observed day)")
    print(f"  households per feeder: min {n.min():,} | p5 {q[0]:,.0f} | p25 {q[1]:,.0f} | median {q[2]:,.0f} | p75 {q[3]:,.0f} | "
          f"p95 {q[4]:,.0f} | p99 {q[5]:,.0f} | max {n.max():,}")
    print(f"  feeders with < 10 households: {(n < 10).sum():,} | < 50: {(n < 50).sum():,} | > 5,000: {(n > 5000).sum():,}")
    print(f"  households with no feeder_id: {int(miss['households'].sum()):,}")
    print(f"  feeders on more than one substation: {(g['substations'] > 1).sum():,} | feeder ids in more than one district: "
          f"{int((dup > 1).sum()):,} | customers mapped to more than one feeder within the month: {multi:,}")
    print(f"  transformers per feeder: median {g['transformers'].median():.0f}, max {g['transformers'].max():,} | feeders with "
          f"placeholder-transformer households: {(g['placeholder_share'] > 0).sum():,}")
    big = g.sort_values("households", ascending=False).head(10)
    print("  largest feeders:\n" + big[["coato_code", "feeder_id", "households", "households_observed", "transformers",
                                       "substations", "placeholder_share"]].to_string(index=False))
    hist(g, args.period, os.path.join(out, "feeder_households_hist.png"))
    print(f"\n[done] {out} in {(time.time() - t0) / 60:.1f} min")


def hist(g, period, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e5e4df", "#2a78d6"
    n = g["households"].to_numpy()
    med, q1, q3 = np.median(n), *np.percentile(n, [25, 75])
    bins = np.logspace(0, np.log10(max(n.max(), 10)) + 0.02, 41)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ax.hist(n, bins=bins, color=blue, edgecolor="white", linewidth=1.0)
    ax.set_xscale("log")
    top = ax.get_xaxis_transform()                                    # x in data, y in axes fraction
    ax.axvline(med, color=ink, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(med, 1.01, f"median {med:,.0f}", transform=top, color=ink, fontsize=9, ha="center", va="bottom")
    ax.axvline(10, color=muted, linewidth=0.8, linestyle=(0, (2, 2)))
    if abs(np.log10(med) - 1) > 0.25:                                 # keep the two labels apart
        ax.text(10, 1.01, "10", transform=top, color=muted, fontsize=8, ha="center", va="bottom")
    ax.set_title(f"{len(n):,} feeders: median {med:,.0f} households, middle half {q1:,.0f}–{q3:,.0f}",
                 loc="left", fontsize=12, color=ink, pad=34)
    ax.text(0, 1.085, f"Households per feeder (log scale), enriched dataset v1.4, {period}; last-known network mapping; "
            f"dotted line = 10 households", transform=ax.transAxes, fontsize=9, color=muted)
    ax.set_xlabel("Households on the feeder", color=muted, fontsize=9)
    ax.set_ylabel("Feeders", color=muted, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
