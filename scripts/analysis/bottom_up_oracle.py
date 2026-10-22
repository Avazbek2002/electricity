#!/usr/bin/env python
"""
How much of a household model's district error comes from scaling households up to the district, not from the model?

The harness forecasts a district as  present(o - 1) x mean forecast of the households with fresh history.
This script replaces the forecast by the TRUTH: the fresh households' own observed kWh on the target day d.
That "oracle" is the best any household model can do under this scaling, so its error is a floor:

    oracle(o, h)  = present(o - 1) x mean observed kWh on d of the fresh households
    T2(d)         = n_rows(d) x (kWh observed + estimated) / (n observed + estimated)

    oracle / T2   = [present / n_rows]  x  [fresh mean on d / T2 per customer]
                     customer count        representativeness of the fresh households

Reported per split, horizon and issue month, on the sum of the districts (clean rows, the 98 % rule), plus the same
for a model's forecasts when --forecasts is given (model error = model - oracle). From the targets alone: the mean kWh
of an estimated vs an observed client-day, and the estimated share, by month.

Charts (figures/, logged to MLflow with the CSVs, report and metrics): bias and sysMAPE by horizon (bars), h30 bias
by issue month and daily bias at h1 / h30 (lines), estimated share and kWh, rows with no fresh household, and window
samples: actual, oracle, B0 and the model over the 60 days after chosen issue dates (--window-origins).

Outputs are district-sum aggregates only (no customer keys).

    python scripts/analysis/bottom_up_oracle.py --sample --workers 5 --out-dir ~/work/bottom_up_oracle
    python scripts/analysis/bottom_up_oracle.py --sample --workers 5 --out-dir ~/work/bottom_up_oracle \
        --forecasts ~/work/eval/runs/<run>/forecasts/gam
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from harness import testset as ts  # noqa: E402

MIN_INPUT_DAYS, MIN_RECENT_DAYS, RECENT_WINDOW, LOOKBACK = 60, 20, 30, 90   # the harness's fresh-history rule


def worker(code, spec_path, labels, full_origins=()):
    """Per issue date: present customers, fresh households and their observed mean on each target day. Control
    horizons for every issue date; every horizon 1..60 for the window samples (full_origins)."""
    from harness.data import read_district
    spec = ts.load_spec(spec_path)
    d = spec["data"]
    D = read_district(code, [dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""), start=d["start"], end=d["end"])],
                      d["start"], d["end"], labels)
    if not len(D["keys"]):
        return pd.DataFrame()
    M, dates = D["M"], D["dates"]
    T = M.shape[1]
    gap = int(spec["history_gap_days"])
    ctrl = np.asarray(spec["control_horizons"])
    every = np.arange(1, int(spec["horizons"]) + 1)
    full = {pd.Timestamp(x) for x in full_origins}
    cnt = np.zeros((len(M), T + 1), dtype=np.int32)
    np.cumsum(np.isfinite(M), axis=1, out=cnt[:, 1:])
    rows = []
    for split in ("validation", "test"):
        for o_day in ts.issue_dates(spec, split):
            o = ts.day_index(spec, o_day)
            last = o - gap
            hs = every if pd.Timestamp(o_day) in full else ctrl
            if last + 1 - LOOKBACK < 0 or o + hs.max() > T - 1:
                continue
            n_in = cnt[:, last + 1] - cnt[:, last + 1 - LOOKBACK]
            n_rec = cnt[:, last + 1] - cnt[:, last + 1 - RECENT_WINDOW]
            fresh = np.flatnonzero((n_in >= MIN_INPUT_DAYS) & (n_rec >= MIN_RECENT_DAYS))
            present = int(((D["first"] <= last) & (D["last"] >= last)).sum())
            hist = np.nanmean(M[fresh, last + 1 - RECENT_WINDOW:last + 1]) if len(fresh) else np.nan
            for h in hs:
                a = M[fresh, o + h] if len(fresh) else np.empty(0)
                ob = np.isfinite(a)
                rows.append((str(code), dates[o], int(h), dates[o + h], present, len(fresh), int(ob.sum()),
                             float(a[ob].mean()) if ob.any() else np.nan, float(hist)))
    return pd.DataFrame(rows, columns=["node", "origin", "horizon", "date", "present", "n_fresh", "n_fresh_obs_d",
                                       "fresh_mean_d", "fresh_mean_last30"])


def collect(codes, a, full_origins=()):
    n = len(codes)
    with cf.ProcessPoolExecutor(max_workers=min(a.workers, n), mp_context=mp.get_context("spawn")) as ex:
        parts = list(ex.map(worker, codes, [a.test_set] * n, [a.labels.split(",")] * n, [list(full_origins)] * n))
    return pd.concat(parts, ignore_index=True)


def ratio(num, den):
    return float(num / den) if den else np.nan


def result_metrics(out_dir):
    """MLflow metrics: <split>/<column>_h<h>, e.g. validation/oracle_bias_h30, test/model_minus_oracle_h7."""
    t = pd.read_csv(os.path.join(out_dir, "oracle_by_horizon.csv"))
    return {f"{r['split']}/{c}_h{int(r['horizon'])}": float(r[c]) for _, r in t.iterrows()
            for c in t.columns if c not in ("split", "horizon", "days", "days_compared") and pd.notna(r[c])}


# ----------------------------------------------------------------------------- charts
# Fixed colour per series (validated categorical palette, light surface); every series also has its own line style /
# marker and a legend, so identity never rests on colour alone. The numbers behind each chart are the CSVs next to it.
SERIES = {  # name: (label, colour, linestyle, marker)
    "y": ("actual (T2)", "#0b0b0b", "-", None),
    "oracle": ("oracle: present x fresh actual", "#2a78d6", "-", "o"),
    "forecast": ("model", "#eb6834", "-", "s"),
    "b0": ("B0", "#1baf7a", "--", "^"),
    "count_only": ("count factor only", "#eda100", ":", "D"),
    "repr_only": ("representativeness only", "#e87ba4", "-.", "v"),
    "model_minus_oracle": ("model - oracle", "#6250d6", "--", "P"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax, title=None, pct=False):
    import matplotlib.ticker as mt
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)


def _save(fig, out, name, made):
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    path = os.path.join(out, "figures", name)
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    made.append(path)


def _label(name, model):
    lab = SERIES[name][0]
    return lab.replace("model", model) if name in ("forecast", "model_minus_oracle") else lab


def chart_by_horizon(by_h, metric, series, out, name, model, made, title):
    import matplotlib.pyplot as plt
    splits = [s for s in ("validation", "test") if s in by_h.index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 3.8), squeeze=False, sharey=True)
    for ax, sp in zip(axes[0], splits):
        t = by_h.loc[sp]
        have = [c for c in series if f"{c}_{metric}" in t or (c == "model_minus_oracle" and c in t)]
        w = 0.8 / max(len(have), 1)
        x = np.arange(len(t.index))
        for k, c in enumerate(have):
            v = t[c] if c == "model_minus_oracle" else t[f"{c}_{metric}"]
            ax.bar(x + (k - (len(have) - 1) / 2) * w, v.to_numpy(), w * 0.92, color=SERIES[c][1],
                   label=_label(c, model), edgecolor="#fcfcfb", linewidth=1)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x, [f"h{h}" for h in t.index])
        _style(ax, f"{sp}: {title}", pct=True)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=8, frameon=False)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(out, "figures", name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)


def chart_lines(df, x, series, out, name, model, made, title, pct=True, panels=None, ylabel=None, markers=True):
    """One line per series; panels = [(panel title, mask)] for small multiples."""
    import matplotlib.pyplot as plt
    panels = panels or [(title, np.ones(len(df), bool))]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)), squeeze=False)
    for ax, (pt, m) in zip(axes[:, 0], panels):
        d = df[m]
        for c in series:
            if c not in d or d[c].isna().all():
                continue
            lab, col, ls, mk = SERIES.get(c, (c, INK, "-", None))
            ax.plot(d[x], d[c], color=col, linestyle=ls, marker=mk if markers else None, markersize=4, linewidth=1.6,
                    label=_label(c, model) if c in SERIES else c)
        if pct:
            ax.axhline(0, color=MUTED, linewidth=0.8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
        _style(ax, pt, pct=pct)
        if len(ax.get_lines()) > 2:                              # >= 2 series (plus the zero line): a legend
            ax.legend(fontsize=8, frameon=False, ncol=3)
    _save(fig, out, name, made)


def chart_windows(win, out, model, made):
    """Small multiples: one issue date each, the district sum over target days o+1 .. o+60."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    origins = sorted(win["origin"].unique())
    if not origins:
        return
    ncol = 4
    nrow = int(np.ceil(len(origins) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False, sharey=False)
    for ax in axes.ravel()[len(origins):]:
        ax.set_visible(False)
    for ax, o in zip(axes.ravel(), origins):
        d = win[win["origin"] == o].sort_values("date")
        for c in ("y", "oracle", "forecast", "b0"):
            if c in d and d[c].notna().any():
                lab, col, ls, _ = SERIES[c]
                ax.plot(d["date"], d[c] / 1e3, color=col, linestyle=ls, linewidth=2 if c == "y" else 1.6,
                        label=_label(c, model))
        nf = int(d["n_fresh"].iloc[0]) if "n_fresh" in d and len(d) else 0
        _style(ax, f"issued {pd.Timestamp(o).date()} ({d['split'].iloc[0]})" + ("" if nf else ": no fresh households"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_ylabel("MWh / day", fontsize=8, color=MUTED)
    h, lab = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh), lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=9, frameon=False)
    fig.suptitle("Window samples: sum of the districts, 60 days after each issue date", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = os.path.join(out, "figures", "window_samples.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)


def default_window_origins(spec):
    """Mid-season issue dates of each split: the 15th of January, April, July and October."""
    out = []
    for sp in ("validation", "test"):
        for o in ts.issue_dates(spec, sp):
            if o.day == 15 and o.month in (1, 4, 7, 10):
                out.append(o)
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-set", default=os.path.join(REPO, "configs", "test_set.yaml"))
    p.add_argument("--test-dir", default=None, help="default ~/work/test_set/<version>[-sample]")
    p.add_argument("--sample", action="store_true", help="the guide's 5 districts")
    p.add_argument("--coatos", default=None, help="comma-separated district codes")
    p.add_argument("--forecasts", default=None, help="a run's forecasts/<model> folder (district.csv.gz)")
    p.add_argument("--model-name", default=None, help="label of --forecasts in charts (default: its folder name)")
    p.add_argument("--window-origins", default=None, help="comma-separated issue dates for the window charts "
                   "(default: the 15th of Jan / Apr / Jul / Oct of both splits)")
    p.add_argument("--labels", default="complete_observed")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    spec = ts.load_spec(a.test_set)
    out = os.path.expanduser(a.out_dir)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    model = a.model_name or (os.path.basename(os.path.normpath(a.forecasts)) if a.forecasts else "model")
    tdir = a.test_dir or os.path.join("~/work/test_set", spec["version"] + ("-sample" if a.sample else ""))
    if not os.path.exists(os.path.join(os.path.expanduser(tdir), "manifest.json")) and a.sample:
        tdir = os.path.join("~/work/test_set", spec["version"])
    tg = ts.load_targets(tdir, spec)
    codes = spec["sample_districts"] if a.sample else (a.coatos.split(",") if a.coatos else sorted(tg["node"].unique()))
    tg = tg[tg["node"].isin(codes)]
    wins = (pd.to_datetime(a.window_origins.split(",")) if a.window_origins else default_window_origins(spec))
    ctrl = [int(h) for h in spec["control_horizons"]]
    L, made = [], []

    def say(s=""):
        print(s)
        L.append(s)

    # ---- from the targets alone: are estimated client-days different from observed ones?
    m = tg.assign(month=tg["date"].dt.to_period("M")).groupby("month")[["n_obs", "n_est", "kwh_obs", "kwh_est"]].sum()
    m = m[m.index >= pd.Period("2023-07", "M")]
    tab = pd.DataFrame({"est_share": m["n_est"] / (m["n_obs"] + m["n_est"]),
                        "kwh_per_obs_day": m["kwh_obs"] / m["n_obs"], "kwh_per_est_day": m["kwh_est"] / m["n_est"]})
    tab["est_vs_obs"] = tab["kwh_per_est_day"] / tab["kwh_per_obs_day"] - 1
    say(f"districts {', '.join(codes)} | test set {spec['version']} | model {model if a.forecasts else 'none'}")
    say("\n=== client-days: estimated share and kWh per estimated vs observed client-day (sum of the districts) ===")
    say(tab.to_string(float_format=lambda x: f"{x:8.3f}"))
    tab.to_csv(os.path.join(out, "estimated_vs_observed_by_month.csv"))
    tm = tab.reset_index().assign(month=lambda d: d["month"].dt.to_timestamp())
    chart_lines(tm.rename(columns={"est_share": "estimated share of client-days",
                                   "est_vs_obs": "kWh per estimated / per observed client-day - 1"}),
                "month", ["estimated share of client-days"], out,
                "estimated_vs_observed.png", model, made, None,
                panels=[("Estimated share of client-days", np.ones(len(tm), bool))], pct=True)
    # the second measure has another scale: its own chart (no dual axis)
    chart_lines(tm.rename(columns={"est_vs_obs": "kWh per estimated / per observed client-day - 1"}), "month",
                ["kWh per estimated / per observed client-day - 1"], out, "estimated_vs_observed_kwh.png", model, made,
                "Estimated vs observed client-day: kWh difference")

    # ---- the oracle (reads the districts), the model and B0
    o = collect(codes, a, wins)
    o = o.merge(tg[["node", "date", "y", "n_rows", "n_obs", "n_est", "kwh_obs", "kwh_est", "flag", "artefact"]],
                on=["node", "date"], how="left")
    o["clean"] = o["y"].notna() & ~o["flag"].fillna(True).astype(bool) & ~o["artefact"].fillna(True).astype(bool)
    o["per_cust"] = (o["kwh_obs"].fillna(0) + o["kwh_est"].fillna(0)) / (o["n_obs"].fillna(0) + o["n_est"].fillna(0))
    o["oracle"] = o["present"] * o["fresh_mean_d"]
    o["count_only"] = o["present"] * o["per_cust"]                  # right per-customer level, the harness's count
    o["repr_only"] = o["n_rows"] * o["fresh_mean_d"]                # right count, the fresh households' level
    b0 = ts.b0_forecasts(ts.load_targets(tdir, spec), spec, nodes=set(codes),
                         issue=pd.DatetimeIndex(sorted(o["origin"].unique())))
    o = o.merge(b0[["node", "origin", "horizon", "forecast"]].rename(columns={"forecast": "b0"}),
                on=["node", "origin", "horizon"], how="left")
    if a.forecasts:
        f = ts.load_forecasts(a.forecasts)
        o = o.merge(f[["node", "origin", "horizon", "forecast"]], on=["node", "origin", "horizon"], how="left")
    o["split"] = np.where(o["origin"] < pd.Timestamp(spec["splits"]["test"]["issue_start"]), "validation", "test")
    vend = pd.Timestamp(spec["splits"]["validation"]["target_end"])
    o = o[(o["split"] == "test") | (o["date"] <= vend)]

    # national (sum of districts) per (o, h): oracle rows; the model and B0 compared on the rows they all have
    base = ["y", "oracle", "count_only", "repr_only"]
    cmp_ = ["b0"] + (["forecast"] if a.forecasts else [])
    ok = o["clean"] & o["oracle"].notna()
    need = int(np.floor(spec["national"]["min_district_share"] * len(codes)))
    oc = o[ok & o["horizon"].isin(ctrl)].copy()
    oc["has_cmp"] = oc[cmp_].notna().all(axis=1)
    g = oc.groupby(["origin", "horizon"])
    nat = g[base].sum()
    nat["districts"] = g.size()
    nat["districts_cmp"] = g["has_cmp"].sum()
    gc = oc[oc["has_cmp"]].groupby(["origin", "horizon"])
    for c in cmp_ + ["y", "oracle"]:
        nat[f"{c}__cmp" if c in ("y", "oracle") else c] = gc[c].sum()
    nat = nat[nat["districts"] >= need].reset_index()
    full_cmp = nat["districts_cmp"] == nat["districts"]               # the model / B0 on every district of the day
    for c in cmp_ + ["y__cmp", "oracle__cmp"]:
        nat.loc[~full_cmp, c] = np.nan
    nat["split"] = np.where(nat["origin"] < pd.Timestamp(spec["splits"]["test"]["issue_start"]), "validation", "test")
    nat.to_csv(os.path.join(out, "oracle_national_daily.csv"), index=False)

    nofresh = (o[o["horizon"] == ctrl[0]].assign(month=lambda d: d["origin"].dt.to_period("M"))
               .groupby("month")["n_fresh"].apply(lambda s: float((s == 0).mean())))
    say("\n=== district rows with no fresh household (no household forecast possible), by issue month ===")
    say(nofresh[nofresh > 0].to_string(float_format=lambda x: f"{x:.0%}") if (nofresh > 0).any() else "none")
    nofresh.rename("no_fresh_share").to_csv(os.path.join(out, "no_fresh_by_month.csv"))

    def summary(d):
        r = {"days": len(d)}
        for c in base[1:]:
            r[f"{c}_bias"] = ratio((d[c] - d["y"]).sum(), d["y"].sum())
            r[f"{c}_sysmape"] = ratio((d[c] - d["y"]).abs().sum(), d["y"].sum())
        e = d[d["b0"].notna()] if "b0" in d else d.iloc[:0]
        r["days_compared"] = len(e)
        for c in cmp_:
            r[f"{c}_bias"] = ratio((e[c] - e["y__cmp"]).sum(), e["y__cmp"].sum())
            r[f"{c}_sysmape"] = ratio((e[c] - e["y__cmp"]).abs().sum(), e["y__cmp"].sum())
        if a.forecasts:
            r["model_minus_oracle"] = ratio((e["forecast"] - e["oracle__cmp"]).sum(), e["y__cmp"].sum())
        return pd.Series(r)

    fmt = lambda x: f"{x:+7.1%}"  # noqa: E731
    by_h = (nat.groupby(["split", "horizon"]).apply(summary, include_groups=False)
            .astype({"days": int, "days_compared": int}))
    by_h = by_h.reindex([s for s in ("validation", "test") if s in by_h.index.get_level_values(0)], level=0)
    say("\n=== sum of the districts: bias / sysMAPE vs T2 ===")
    say("oracle = present x fresh households' actual mean; count_only = present x T2 per customer; repr_only = n_rows "
        f"x fresh actual mean; B0{' and ' + model if a.forecasts else ''} on the days_compared subset")
    say(by_h.to_string(float_format=fmt))
    by_h.to_csv(os.path.join(out, "oracle_by_horizon.csv"))
    nat["month"] = nat["origin"].dt.to_period("M")
    by_m = (nat[nat["horizon"] == 30].groupby(["split", "month"]).apply(summary, include_groups=False)
            .astype({"days": int, "days_compared": int}))
    say("\n=== h30 by issue month ===")
    say(by_m.to_string(float_format=fmt))
    by_m.to_csv(os.path.join(out, "oracle_h30_by_month.csv"))

    # ---- charts
    bars = ["oracle", "count_only", "repr_only", "b0"] + (["forecast", "model_minus_oracle"] if a.forecasts else [])
    chart_by_horizon(by_h, "bias", bars, out, "bias_by_horizon.png", model, made, "bias by horizon")
    chart_by_horizon(by_h, "sysmape", ["oracle", "b0"] + (["forecast"] if a.forecasts else []), out,
                     "sysmape_by_horizon.png", model, made, "sysMAPE by horizon")
    bm = by_m.reset_index().drop(columns="split").set_index("month").sort_index()
    bm = bm.reindex(pd.period_range(bm.index.min(), bm.index.max(), freq="M"))   # a missing month breaks the line
    bm = bm.rename_axis("month").reset_index()
    bm["month"] = bm["month"].dt.to_timestamp()
    ren = {f"{c}_bias": c for c in bars}
    chart_lines(bm.rename(columns=ren), "month", [c for c in bars if c != "count_only"], out, "h30_bias_by_month.png",
                model, made, "Bias at h30 by issue month (sum of the districts)")
    dd = nat[nat["horizon"].isin([1, 30])].sort_values("origin").copy()
    roll = {}
    for c in ["oracle", "b0"] + (["forecast"] if a.forecasts else []):
        yy = dd["y__cmp"] if c != "oracle" else dd["y"]
        dd[c] = (dd[c] - yy) / yy
        roll[c] = c
    full = pd.date_range(dd["origin"].min(), dd["origin"].max(), freq="D")
    dd = pd.concat([g.set_index("origin")[list(roll)].reindex(full).rolling(7, min_periods=3).mean()
                    .assign(horizon=h).rename_axis("origin").reset_index()
                    for h, g in dd.groupby("horizon")], ignore_index=True)   # missing days break the line
    chart_lines(dd, "origin", list(roll), out, "daily_bias_h1_h30.png", model, made, None, markers=False,
                panels=[("h1: daily bias, 7-day mean, by issue date", (dd["horizon"] == 1).to_numpy()),
                        ("h30: daily bias, 7-day mean, by issue date", (dd["horizon"] == 30).to_numpy())])
    nf = nofresh.reset_index()
    nf["month"] = nf["month"].dt.to_timestamp()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(nf["month"], nf["n_fresh"], width=20, color=SERIES["oracle"][1])
    _style(ax, "District rows with no fresh household (no household forecast possible), by issue month", pct=True)
    _save(fig, out, "no_fresh_by_month.png", made)

    # window samples: every horizon, district sum where every district row is clean and has the series
    w = o[o["origin"].isin(wins)].copy()
    w["y"] = w["y"].where(w["clean"])
    ser = ["y", "oracle", "b0"] + (["forecast"] if a.forecasts else [])
    gw = w.groupby(["origin", "horizon"])
    win = gw[ser].sum(min_count=1)
    for c in ser:                                                    # a sum over fewer districts is not comparable
        win.loc[gw[c].count() < len(codes), c] = np.nan
    win["date"] = gw["date"].first()
    win["n_fresh"] = gw["n_fresh"].sum()
    win["split"] = gw["split"].first()
    win = win.reset_index()
    win.to_csv(os.path.join(out, "window_samples.csv"), index=False)
    chart_windows(win, out, model, made)

    say(f"\nfigures: {', '.join(os.path.basename(x) for x in made)}")
    say("\nReading it: oracle bias = the floor of a household model's district bias under this scaling (positive: the "
        "fresh households use more than the average customer). count_only = the customer-count part, repr_only = "
        "the representativeness part. model_minus_oracle = the model's own share.")
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis", metrics_from=result_metrics)
