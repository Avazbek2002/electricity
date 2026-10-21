#!/usr/bin/env python3
"""
plot_forecast_windows.py - line charts of forecast windows from transformer_forecast.py's direct model.

Each panel is one transformer and one forecast origin: the last 30 days of actual kWh, then the 60 days after the origin with
the actual (black) against the direct forecast (orange), the thermal baseline (blue) and 7-day persistence (grey, dashed).
All forecasts carry the calendar factors, exactly as scored in compare.  A grey band marks a day that was not usable (flagged
by transformer_clean.py) and is therefore not scored.

Writes into <forecast-dir>/windows/
  windows_typical.png     3 size terciles x 3 origins; per cell the transformer whose direct WAPE is closest to the median
                          of its tercile at that origin (--pick best / worst for the extremes)
  windows_national.png    the sum over every transformer at each origin: actual vs forecasts, with sysWAPE. On a day when some
                          transformers are unusable the sum is scaled up by household count so a gap day is not a dip; days
                          with under half of the households usable are left out
  windows_chosen.csv      which transformer / origin each panel shows, with its WAPEs
  windows_ids.png         only with --ids: your own transformers (rows) x origins (columns)

Needs results/calendar_daily.csv: if it is missing, re-run  python transformer_forecast.py run --stages compare ...  with the
same arguments as before (takes seconds, it reuses direct_test.npz) and it is written.

Usage
  python plot_forecast_windows.py --work-dir ~/work/tf_national_v2 --forecast-dir ~/work/tf_national_v2/forecast
  python plot_forecast_windows.py --work-dir ... --forecast-dir ... --origins 2025-11-05,2026-02-01,2026-04-01 --pick worst
  python plot_forecast_windows.py --work-dir ... --forecast-dir ... --ids 862928,859178
"""
import argparse
import os

import numpy as np
import pandas as pd

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
C_DIRECT, C_THERMAL, C_PERS = "#c2670f", "#2563b8", "#8a93a0"
H = np.arange(1, 61)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--forecast-dir", default=None, help="default <work-dir>/forecast")
    ap.add_argument("--origins", default="2025-11-05,2026-01-15,2026-03-10",
                    help="comma list; each snaps to the nearest test origin (test origins start 60 days after the train cutoff)")
    ap.add_argument("--pick", choices=["typical", "best", "worst"], default="typical")
    ap.add_argument("--ids", default="", help="comma list of transformer ids for an extra chart")
    ap.add_argument("--context", type=int, default=30, help="days of actuals shown before the origin")
    ap.add_argument("--min-days", type=int, default=45, help="a window needs this many scored days of its 60 to be picked")
    a = ap.parse_args()
    wd = os.path.expanduser(a.work_dir)
    fd = os.path.expanduser(a.forecast_dir or os.path.join(wd, "forecast"))
    out = os.path.join(fd, "windows")
    os.makedirs(out, exist_ok=True)

    z = np.load(os.path.join(wd, "clean", "transformer_clean.npz"), allow_pickle=False)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    T = len(dates)
    row = {str(t): i for i, t in enumerate(z["transformer_id"].astype(str))}
    kwh = z["intensity"].astype("float64") * z["n_core"].astype("float64")[:, None]     # NaN on unusable days

    p = os.path.join(fd, "results", "calendar_daily.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found: re-run  python transformer_forecast.py run --work-dir {wd} --out-dir {fd} --stages compare  "
                         f"(same arguments as before, takes seconds) and it is written")
    cd = pd.read_csv(p)
    cal = pd.Series(cd["factor"].to_numpy(), index=pd.to_datetime(cd["date"])).reindex(dates).fillna(1.0).to_numpy()

    d = np.load(os.path.join(fd, "direct_test.npz"), allow_pickle=False)
    tid, org, n_hh = d["transformer_id"].astype(str), d["origin"].astype(int), d["n_hh"].astype(float)
    pred, thermal, pers7 = d["pred"].astype("float64"), d["thermal"].astype("float64"), d["pers7"].astype("float64")
    org_dates = pd.DatetimeIndex(dates[0] + pd.to_timedelta(org, unit="D"))
    print(f"[windows] {len(tid):,} test windows, {len(np.unique(tid)):,} transformers, origins {org_dates.min().date()} .. {org_dates.max().date()}")

    def arrays(k):
        """actual and the three forecasts (n, 60) for window indices k; calendar applied; NaN beyond the panel."""
        k = np.atleast_1d(k)
        idx = org[k][:, None] + H[None, :]
        ok = idx < T
        ic = np.minimum(idx, T - 1)
        c = np.where(ok, cal[ic], np.nan)
        rows = np.array([row[t] for t in tid[k]])
        A = np.where(ok, kwh[rows[:, None], ic], np.nan)
        F = {"direct": pred[k] * c, "thermal": thermal[k] * c, "persistence7": pers7[k][:, None] * c}
        return A, F

    def wape(Fm, A):
        v = np.isfinite(A) & np.isfinite(Fm) & (A > 0)
        return np.abs(Fm - A)[v].sum() / A[v].sum() if v.any() else np.nan

    def wape_rows(Fm, A):
        v = np.isfinite(A) & np.isfinite(Fm) & (A > 0)
        num = np.where(v, np.abs(Fm - A), 0).sum(1)
        den = np.where(v, A, 0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(v.sum(1) >= a.min_days, num / den, np.nan)

    uniq = np.unique(org)
    ods = []
    for s in a.origins.split(","):
        want = (pd.Timestamp(s.strip()) - dates[0]).days
        o = int(uniq[np.argmin(np.abs(uniq - want))])
        if abs(o - want) > 10:
            print(f"[windows] origin {s.strip()} is {abs(o - want)} days from the nearest test origin; using {(dates[0] + pd.Timedelta(days=o)).date()}")
        ods.append(o)
    ods = list(dict.fromkeys(ods))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    handles = [Line2D([], [], color=INK, lw=1.8, label="Actual"),
               Line2D([], [], color=C_DIRECT, lw=1.6, label="Direct"),
               Line2D([], [], color=C_THERMAL, lw=1.4, ls=(0, (1, 1.5)), label="Thermal baseline"),
               Line2D([], [], color=C_PERS, lw=1.4, ls=(0, (4, 3)), label="Persistence (7-day mean)"),
               Patch(facecolor=GRID, edgecolor="none", label="Day not usable, not scored")]

    def draw(ax, x_hist, y_hist, x_f, A, F, title, sub):
        gap = ~np.isfinite(A)
        for i in np.flatnonzero(gap):                        # unusable days: one-day grey band
            ax.axvspan(x_f[i] - pd.Timedelta(hours=12), x_f[i] + pd.Timedelta(hours=12), color=GRID, lw=0, zorder=0)
        ax.axvline(x_hist[-1], color=MUTED, lw=0.8, ls=":")
        ax.plot(x_hist, y_hist, color=INK, lw=1.8)
        ax.plot(x_f, A, color=INK, lw=1.8)
        ax.plot(x_f, F["persistence7"], color=C_PERS, lw=1.4, ls=(0, (4, 3)))
        ax.plot(x_f, F["thermal"], color=C_THERMAL, lw=1.4, ls=(0, (1, 1.5)))
        ax.plot(x_f, F["direct"], color=C_DIRECT, lw=1.6)
        ax.set_title(title + "\n" + sub, loc="left", fontsize=8.5, color=INK, linespacing=1.5)
        ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=20))
        ax.tick_params(axis="x", labelsize=8)

    def panel(ax, k, label):
        A, F = arrays(k)
        A, F = A[0], {n: v[0] for n, v in F.items()}
        o = org[k]
        h0 = max(0, o - a.context + 1)
        x_hist = dates[h0:o + 1]
        y_hist = kwh[row[tid[k]], h0:o + 1]
        x_f = dates[min(o + 1, T - 1):min(o + 61, T)]
        n = len(x_f)
        w = {m: wape(F[m][:n], A[:n]) for m in F}
        title = f"{label}transformer {tid[k]} · {int(n_hh[k])} households · issued {dates[o].date()}"
        sub = f"WAPE  direct {w['direct']:.0%} · thermal {w['thermal']:.0%} · persistence {w['persistence7']:.0%}"
        draw(ax, x_hist, y_hist, x_f, A[:n], {m: v[:n] for m, v in F.items()}, title, sub)
        return w

    chosen = []
    # ---- 1. typical windows: size terciles x origins ----
    size = pd.Series(n_hh, index=tid).groupby(level=0).first()
    q1, q2 = np.quantile(size, [1 / 3, 2 / 3])
    terc = [("small", lambda x: x <= q1), ("mid", lambda x: (x > q1) & (x <= q2)), ("large", lambda x: x > q2)]
    fig, axes = plt.subplots(3, len(ods), figsize=(4.4 * len(ods), 9.6), squeeze=False)
    for r, (tn, sel) in enumerate(terc):
        for c, o in enumerate(ods):
            cand = np.flatnonzero((org == o) & sel(n_hh))
            ax = axes[r, c]
            if len(cand) == 0:
                ax.axis("off")
                continue
            A, F = arrays(cand)
            wr = wape_rows(F["direct"], A)
            good = np.flatnonzero(np.isfinite(wr))
            if len(good) == 0:
                ax.axis("off")
                continue
            if a.pick == "typical":
                j = good[np.argmin(np.abs(wr[good] - np.median(wr[good])))]
            elif a.pick == "best":
                j = good[np.argmin(wr[good])]
            else:
                j = good[np.argmax(wr[good])]
            k = int(cand[j])
            w = panel(ax, k, f"{tn}: ")
            chosen.append(dict(panel=f"{tn} / {dates[o].date()}", transformer_id=tid[k], households=int(n_hh[k]), origin=dates[o].date(),
                               wape_direct=w["direct"], wape_thermal=w["thermal"], wape_persistence7=w["persistence7"],
                               tercile_median_direct_wape=float(np.median(wr[good])), candidates=len(good)))
            if c == 0:
                ax.set_ylabel("kWh / day (transformer total)")
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle(f"Forecast windows, {a.pick} case per size tercile (small <= {q1:.0f}, large > {q2:.0f} households)",
                 x=0.01, y=1.03, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f1 = os.path.join(out, "windows_typical.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight"), plt.close(fig)

    # ---- 2. national totals at each origin ----
    fig, axes = plt.subplots(1, len(ods), figsize=(4.8 * len(ods), 4.4), squeeze=False)
    for c, o in enumerate(ods):
        ax = axes[0, c]
        k = np.flatnonzero(org == o)
        A, F = arrays(k)
        ok = np.isfinite(A)
        nh = n_hh[k]
        use = np.where(ok, nh[:, None], 0.0).sum(0)          # households on transformers usable that day
        keep = use >= 0.5 * nh.sum()                         # days with too few usable transformers are not shown or scored
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(keep, nh.sum() / use, np.nan)      # scale the sum up to the full set so a gap day is not a dip
        sumA = np.where(ok, A, 0).sum(0) * up
        tot = {m: np.where(ok & np.isfinite(F[m]), F[m], 0).sum(0) * up for m in F}
        x_f = dates[min(o + 1, T - 1):min(o + 61, T)]
        n = len(x_f)
        rows_ = np.array([row[t] for t in tid[k]])
        h0 = max(0, o - a.context + 1)
        kh = kwh[rows_, h0:o + 1]
        use_h = np.where(np.isfinite(kh), nh[:, None], 0.0).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hist = np.where(use_h >= 0.5 * nh.sum(), np.where(np.isfinite(kh), kh, 0).sum(0) * nh.sum() / use_h, np.nan)
        kv = keep[:n]
        sw = {m: np.abs(tot[m][:n] - sumA[:n])[kv].sum() / sumA[:n][kv].sum() for m in F}
        Fm = {m: tot[m][:n] for m in F}
        draw(ax, dates[h0:o + 1], hist, x_f, sumA[:n], Fm,
             f"All {len(k):,} transformers · issued {dates[o].date()}",
             f"sysWAPE  direct {sw['direct']:.1%} · thermal {sw['thermal']:.1%} · persistence {sw['persistence7']:.1%}")
        if c == 0:
            ax.set_ylabel("kWh / day (all transformers; scaled up on gap days)")
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    f2 = os.path.join(out, "windows_national.png")
    fig.savefig(f2, dpi=150, bbox_inches="tight"), plt.close(fig)

    # ---- 3. your own transformers ----
    f3 = None
    if a.ids:
        want = [s.strip() for s in a.ids.split(",") if s.strip()]
        want = [t for t in want if t in set(tid)] or []
        for t in [s.strip() for s in a.ids.split(",") if s.strip()]:
            if t not in set(tid):
                print(f"[windows] transformer {t} has no test windows (not in direct_test.npz)")
        if want:
            fig, axes = plt.subplots(len(want), len(ods), figsize=(4.4 * len(ods), 3.2 * len(want) + 0.8), squeeze=False)
            for r, t in enumerate(want):
                for c, o in enumerate(ods):
                    ax = axes[r, c]
                    k = np.flatnonzero((tid == t) & (org == o))
                    if len(k) == 0:
                        ax.axis("off")
                        continue
                    panel(ax, int(k[0]), "")
                    if c == 0:
                        ax.set_ylabel("kWh / day")
            fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            f3 = os.path.join(out, "windows_ids.png")
            fig.savefig(f3, dpi=150, bbox_inches="tight"), plt.close(fig)

    ch = pd.DataFrame(chosen)
    ch.to_csv(os.path.join(out, "windows_chosen.csv"), index=False)
    print(ch.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[windows] wrote {f1}\n          {f2}" + (f"\n          {f3}" if f3 else "") + f"\n          {os.path.join(out, 'windows_chosen.csv')}")


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis")
