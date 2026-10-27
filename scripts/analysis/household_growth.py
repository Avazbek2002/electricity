#!/usr/bin/env python
"""How fast the number of households grows per district / substation / feeder, and where it JUMPS instead of growing.

    python scripts/analysis/household_growth.py                       # node set (60 substations + feeders) + districts
    python scripts/analysis/household_growth.py --district-test-dir ~/work/test_set/ts-2026-10-22-v2

Reads only the built target tables (node x day aggregates: n_rows = households in span, coverage, T2), no customer data.

  growth   mean households per calendar year (2026 = H1) and the change year on year; 2022 -> 2025 per year (CAGR)
  jumps    a step in the household count: median of the 14 days after day d vs the 14 days before, |step| >= --step
           (default 3 %) and >= --min-abs households; one jump per 28 days (the largest). A jump is what the 28-day level
           of B0 / the models cannot follow for about a month (a step of s % = ~s % forecast error until it ages out)
  per-hh   T2 per household (mean kWh per household and day) by year: whether the households' use itself drifts
  coverage share of households with a complete reading, by year

n_rows counts customers whose contract span covers the day: growth = new connections AND customers that enter the data
(new meters, re-registration); a jump is often re-mapping (households moved between substations / feeders) rather than
new homes. Writes <out>/report.txt, growth_<level>.csv, jumps.csv and PNG charts.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import harness  # noqa: E402,F401
from harness import nodeset as ns  # noqa: E402


def level_of(node):
    return {"S": "substation", "F": "feeder"}.get(str(node).split(":", 1)[0], "district")


def yearly(t):
    """t: date, node, n_rows, coverage, T2 -> one row per node with per-year means and year-on-year changes."""
    t = t.assign(year=t["date"].dt.year, kwh_hh=t["T2"] / t["n_rows"].where(t["n_rows"] > 0))
    g = t.groupby(["node", "year"]).agg(hh=("n_rows", "mean"), cov=("coverage", "mean"), kwh_hh=("kwh_hh", "mean"),
                                        days=("n_rows", "count")).reset_index()
    g = g[g["days"] >= 60]
    out = []
    for node, q in g.groupby("node"):
        q = q.set_index("year")
        r = dict(node=node, level=level_of(node))
        for y in q.index:
            r[f"hh_{y}"] = q.at[y, "hh"]
            r[f"cov_{y}"] = q.at[y, "cov"]
            r[f"kwh_hh_{y}"] = q.at[y, "kwh_hh"]
        for y in q.index:
            if y - 1 in q.index and q.at[y - 1, "hh"] > 0:
                r[f"growth_{y}"] = q.at[y, "hh"] / q.at[y - 1, "hh"] - 1
        if 2022 in q.index and 2025 in q.index and q.at[2022, "hh"] > 0:
            r["cagr_22_25"] = (q.at[2025, "hh"] / q.at[2022, "hh"]) ** (1 / 3) - 1
        out.append(r)
    return pd.DataFrame(out)


def jumps(t, step=0.03, min_abs=20, win=14, sep=28):
    """Steps in the household count per node -> DataFrame node, date, before, after, step, abs."""
    rows = []
    for node, q in t.groupby("node"):
        s = q.set_index("date")["n_rows"].astype(float).asfreq("D")
        if s.notna().sum() < 3 * win:
            continue
        before = s.rolling(win, min_periods=win // 2).median().shift(1)
        after = s[::-1].rolling(win, min_periods=win // 2).median()[::-1]
        rel = (after - before) / before.where(before > 0)
        cand = rel[(rel.abs() >= step) & ((after - before).abs() >= min_abs)].dropna()
        last = None
        for d in cand.abs().sort_values(ascending=False).index:     # largest first, one per `sep` days
            if last is not None and any(abs((d - x).days) < sep for x in last):
                continue
            last = (last or []) + [d]
            rows.append(dict(node=node, level=level_of(node), date=d, before=before[d], after=after[d], step=rel[d],
                             abs=after[d] - before[d]))
    out = pd.DataFrame(rows, columns=["node", "level", "date", "before", "after", "step", "abs"])
    out["date"] = pd.to_datetime(out["date"])
    return out


def describe(df, col):
    v = df[col].dropna()
    if not len(v):
        return "n/a"
    return (f"median {v.median():+.1%} | p10 {v.quantile(.1):+.1%} | p90 {v.quantile(.9):+.1%} | "
            f"min {v.min():+.1%} | max {v.max():+.1%} (n={len(v)})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    p.add_argument("--test-dir", default=None, help="node test set (default ~/work/test_set/<spec version>)")
    p.add_argument("--district-test-dir", default="~/work/test_set/ts-2026-10-22-v2",
                   help="district test set (targets/district.csv.gz); skipped if missing")
    p.add_argument("--step", type=float, default=0.03)
    p.add_argument("--min-abs", type=float, default=20)
    p.add_argument("--min-households", type=float, default=50, help="leave out smaller feeders (as the scores do)")
    p.add_argument("--out", default="~/work/analysis/household_growth")
    a = p.parse_args(argv)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    spec = ns.load_spec(a.spec)
    tdir = os.path.expanduser(a.test_dir or os.path.join("~/work/test_set", spec["version"]))
    cols = ["date", "node", "n_rows", "coverage", "T2"]
    parts = [ns.load_targets(tdir)[cols]]
    dp = os.path.join(os.path.expanduser(a.district_test_dir), "targets", "district.csv.gz") if a.district_test_dir else ""
    if dp and os.path.exists(dp):
        d = pd.read_csv(dp, dtype={"node": str}, parse_dates=["date"], usecols=lambda c: c in cols)
        parts.append(d[cols])
    else:
        print(f"[districts] {dp or '-'} not found: substations and feeders only")
    t = pd.concat(parts, ignore_index=True)
    size = t.groupby("node")["n_rows"].median()
    t = t[t["node"].map(size) >= a.min_households]
    print(f"[read] {t['node'].nunique()} series ({t.groupby('node').first().index.map(level_of).value_counts().to_dict()}), "
          f"{t['date'].min():%Y-%m-%d} .. {t['date'].max():%Y-%m-%d}")

    Y = yearly(t)
    J = jumps(t, a.step, a.min_abs)
    lines = ["HOUSEHOLD GROWTH (n_rows = households in span; mean per year, 2026 = first half)", ""]
    years = sorted({int(c.split("_")[1]) for c in Y.columns if c.startswith("growth_")})
    for lv in ("district", "substation", "feeder"):
        q = Y[Y["level"] == lv]
        if not len(q):
            continue
        Y[Y["level"] == lv].to_csv(os.path.join(out, f"growth_{lv}.csv"), index=False, float_format="%.5f")
        tot = {y: q[f"hh_{y}"].sum() for y in range(2022, 2027) if f"hh_{y}" in q}
        lines += [f"{lv.upper()} ({len(q)} series)",
                  "  households, sum of the series: " + " | ".join(f"{y} {v:,.0f}" for y, v in tot.items())]
        for y in years:
            if f"growth_{y}" in q:
                lines.append(f"  growth {y - 1}->{y}: {describe(q, f'growth_{y}')}")
        if "cagr_22_25" in q:
            lines.append(f"  per year 2022->2025 (CAGR): {describe(q, 'cagr_22_25')}")
        for y in range(2022, 2027):
            if f"kwh_hh_{y}" in q:
                lines.append(f"  {y}: kWh per household and day, median {q[f'kwh_hh_{y}'].median():.2f} | coverage median "
                             f"{q[f'cov_{y}'].median():.1%}")
        jl = J[J["level"] == lv]
        n_with = jl["node"].nunique()
        lines += [f"  jumps (|step| >= {a.step:.0%} and >= {a.min_abs:g} households): {len(jl)} in {n_with} of {len(q)} "
                  f"series ({n_with / max(len(q), 1):.0%}); by year " +
                  str(jl["date"].dt.year.value_counts().sort_index().to_dict()), ""]
    if len(J):
        J = J.sort_values("step", key=np.abs, ascending=False)
        J.to_csv(os.path.join(out, "jumps.csv"), index=False, float_format="%.4f")
        top = J.head(25).assign(date=J["date"].dt.strftime("%Y-%m-%d"), step=(100 * J["step"]).round(1),
                                before=J["before"].round(0), after=J["after"].round(0))
        lines += ["LARGEST JUMPS (median 14 days after vs before; step in %)",
                  top[["node", "level", "date", "before", "after", "step"]].to_string(index=False), ""]
        new = J[J["date"] >= "2024-08-01"]
        lines += [f"jumps from 2024-08 (new regime, what production will see): {len(new)}; "
                  f"median |step| {new['step'].abs().median():.1%}" if len(new) else "no jumps from 2024-08", ""]
    text = "\n".join(lines)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    _charts(t, Y, J, out)
    print(f"-> {out}")
    return out


def _charts(t, Y, J, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[charts] no matplotlib ({e})")
        return
    levels = [lv for lv in ("district", "substation", "feeder") if (Y["level"] == lv).any()]
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 3.6), squeeze=False)
    for ax, lv in zip(axes[0], levels):
        q = Y[Y["level"] == lv]
        for i, c in enumerate([c for c in ("growth_2023", "growth_2024", "growth_2025") if c in q]):
            v = 100 * q[c].dropna().clip(-0.3, 0.3)
            ax.hist(v, bins=30, alpha=0.5, label=c.replace("growth_", ""))
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_title(f"{lv}: household growth per year, %", fontsize=10, loc="left")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "growth_hist.png"), dpi=110)
    plt.close(fig)
    if len(J):
        top = J.drop_duplicates("node").head(6)
        fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.2 * len(top)), sharex=True, squeeze=False)
        for ax, r in zip(axes[:, 0], top.itertuples()):
            s = t[t["node"] == r.node].set_index("date")["n_rows"]
            ax.plot(s.index, s.values, lw=1.2, color="#2a6fdb")
            for d in J.loc[J["node"] == r.node, "date"]:
                ax.axvline(d, color="#e8590c", lw=0.8, ls="--")
            ax.set_title(f"{r.node} ({r.level}): largest step {100 * r.step:+.1f}% on {r.date:%Y-%m-%d}",
                         fontsize=9, loc="left")
            ax.set_ylabel("households")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "jumps_top.png"), dpi=110)
        plt.close(fig)
    print("[charts] growth_hist.png, jumps_top.png")


if __name__ == "__main__":
    main()
