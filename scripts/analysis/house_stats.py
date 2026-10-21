#!/usr/bin/env python3
"""
How many households are on each transformer?  Statistics from the CSV files transformer_daily.py already wrote.

    python household_stats.py --dir ~/work/transformers

Reads   <dir>/transformer_daily.csv      (required)
        <dir>/transformers_selected.csv  (optional; adds transformers that never had a daily reading)
Writes  <dir>/household_stats/transformer_sizes.csv, size_bands.csv, household_counts.png

Three different counts are in those files, and they answer different questions:

  n_on_transformer  every household the metadata puts on that transformer id TODAY (any billing type)
  n_ever            households with at least one daily reading in the aggregated date range
                    (monthly-billed households never appear in the readings, so this is the smaller number)
  n_reporting       households that actually reported on a given day (the figure behind the daily totals)

The forecasting uses n_ever; the share n_ever / n_on_transformer says how much of the transformer's households
the daily data can see at all.  Nothing here uses Spark.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
BANDS = [0, 10, 20, 50, 100, 150, 200, 250, 300, 500, 1000, 10 ** 9]


def describe(s, label):
    s = s.dropna()
    q = s.quantile(QUANTILES)
    print(f"\n{label}  (n = {len(s):,} transformers)")
    print(f"  mean {s.mean():.1f} | std {s.std():.1f} | min {s.min():.0f} | max {s.max():.0f} | total {s.sum():,.0f} households")
    print("  percentiles: " + "  ".join(f"p{int(k * 100):02d} {v:.0f}" for k, v in q.items()))
    for w in (10, 20):
        lo, hi = s.median() - w, s.median() + w
        print(f"  within median +-{w} ({lo:.0f}..{hi:.0f}): {((s >= lo) & (s <= hi)).mean():.1%} of transformers")


def band_table(s):
    labels = [f"{a}-{b - 1}" if b < 10 ** 9 else f"{a}+" for a, b in zip(BANDS[:-1], BANDS[1:])]
    cut = pd.cut(s, BANDS, right=False, labels=labels)
    t = pd.DataFrame({"transformers": cut.value_counts().reindex(labels),
                      "households": s.groupby(cut, observed=False).sum().reindex(labels)})
    t["share_of_transformers"] = t["transformers"] / t["transformers"].sum()
    t["share_of_households"] = t["households"] / t["households"].sum()
    return t[t["transformers"] > 0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=".", help="folder with transformer_daily.csv (the --out-dir of transformer_daily.py)")
    p.add_argument("--daily", default=None, help="path to transformer_daily.csv (default: <dir>/transformer_daily.csv)")
    p.add_argument("--selected", default=None, help="path to transformers_selected.csv (default: <dir>/transformers_selected.csv)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    d = os.path.abspath(os.path.expanduser(args.dir))
    daily_path = args.daily or os.path.join(d, "transformer_daily.csv")
    sel_path = args.selected or os.path.join(d, "transformers_selected.csv")
    if not os.path.exists(daily_path):
        sys.exit(f"{daily_path} not found: run transformer_daily.py aggregate first, or pass --daily / --dir")
    out = os.path.join(d, "household_stats")
    os.makedirs(out, exist_ok=True)

    td = pd.read_csv(daily_path, parse_dates=["calendar_day"], dtype={"transformer_id": str, "coato_code": str})
    print(f"{daily_path}: {len(td):,} transformer-days, {td['transformer_id'].nunique():,} transformers, "
          f"{td['calendar_day'].min():%Y-%m-%d} .. {td['calendar_day'].max():%Y-%m-%d}")

    g = td.groupby("transformer_id")
    sizes = pd.DataFrame({
        "n_ever": g["n_ever"].max(),
        "n_on_transformer": g["n_on_transformer"].max() if "n_on_transformer" in td else np.nan,
        "median_reporting": g["n_reporting"].median(),
        "max_reporting": g["n_reporting"].max(),
        "days_with_data": g.size(),
    })
    if "coato_code" in td:
        sizes["coato_code"] = g["coato_code"].first()
    if os.path.exists(sel_path):
        sel = pd.read_csv(sel_path, dtype={"transformer_id": str, "coato_code": str}).set_index("transformer_id")
        extra = sel.index.difference(sizes.index)
        if len(extra):
            print(f"{len(extra)} transformer(s) are in {os.path.basename(sel_path)} but have no daily reading at all "
                  f"(all their households are monthly-billed or silent); adding them with n_ever = 0")
            add = pd.DataFrame({"n_ever": 0, "n_on_transformer": sel.loc[extra, "n_households"],
                                "median_reporting": 0, "max_reporting": 0, "days_with_data": 0}, index=extra)
            if "coato_code" in sizes:
                add["coato_code"] = sel.loc[extra, "coato_code"]
            sizes = pd.concat([sizes, add])
        sizes["n_on_transformer"] = sizes["n_on_transformer"].fillna(sel["n_households"].reindex(sizes.index))
    sizes["daily_metered_share"] = sizes["n_ever"] / sizes["n_on_transformer"]
    sizes = sizes.sort_values("n_ever", ascending=False)
    sizes.to_csv(os.path.join(out, "transformer_sizes.csv"))

    describe(sizes["n_ever"], "households with daily readings per transformer (n_ever)")
    if sizes["n_on_transformer"].notna().any():
        describe(sizes["n_on_transformer"], "all households on the transformer id (n_on_transformer)")
        s = sizes["daily_metered_share"].dropna()
        print(f"\nshare of a transformer's households that have daily readings: median {s.median():.0%}, "
              f"10th pct {s.quantile(.1):.0%}, 90th pct {s.quantile(.9):.0%}; "
              f"{(s < 0.5).sum()} transformers below 50%")
    describe(sizes["median_reporting"], "households reporting on a typical day (median n_reporting)")

    print("\ntransformers by number of households with daily readings:")
    bt = band_table(sizes["n_ever"])
    bt.to_csv(os.path.join(out, "size_bands.csv"))
    print(bt.to_string(formatters={"share_of_transformers": "{:.1%}".format, "share_of_households": "{:.1%}".format,
                                   "households": "{:,.0f}".format}))
    hh_med = np.average(sizes["n_ever"], weights=sizes["n_ever"]) if sizes["n_ever"].sum() else np.nan
    print(f"\na typical TRANSFORMER has {sizes['n_ever'].median():.0f} households; a typical HOUSEHOLD sits on a transformer of "
          f"{hh_med:.0f} (household-weighted mean).")

    print("\nlargest 15 transformers by n_ever:")
    print(sizes.head(15)[["n_ever", "n_on_transformer", "median_reporting", "days_with_data"]].to_string())
    print("\nsmallest 10 with data:")
    print(sizes[sizes["n_ever"] > 0].tail(10)[["n_ever", "n_on_transformer", "median_reporting", "days_with_data"]].to_string())

    if "coato_code" in sizes and sizes["coato_code"].nunique() > 1:
        print("\nper coato:")
        print(sizes.groupby("coato_code")["n_ever"].describe()[["count", "mean", "50%", "min", "max"]].round(0).to_string())

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            x = sizes["n_ever"][sizes["n_ever"] > 0]
            ax[0].hist(x, bins=np.arange(0, max(x.max(), 50) + 10, 10), color="#2563b8")
            ax[0].axvline(x.median(), color="#1f2430", ls="--", lw=1)
            ax[0].set_title(f"Households with daily readings per transformer (median {x.median():.0f})", loc="left", fontsize=10)
            ax[0].set_xlabel("households"), ax[0].set_ylabel("transformers")
            ss = sizes["daily_metered_share"].dropna()
            if len(ss):
                ax[1].hist(ss.clip(upper=1.2), bins=np.linspace(0, 1.2, 25), color="#c2670f")
                ax[1].set_title("Share of the transformer's households with daily readings", loc="left", fontsize=10)
                ax[1].set_xlabel("n_ever / n_on_transformer")
            fig.tight_layout(), fig.savefig(os.path.join(out, "household_counts.png"), dpi=150)
        except Exception as e:                                           # noqa: BLE001
            print(f"(plot skipped: {type(e).__name__}: {e})")
    print(f"\nwrote transformer_sizes.csv, size_bands.csv and the chart to {out}")


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis")
