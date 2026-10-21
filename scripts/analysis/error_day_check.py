#!/usr/bin/env python3
"""
error_day_check.py - why is the system total badly wrong on specific days?

For the target days where transformer_forecast.py's 'direct' model has the largest system-level error (read from
<forecast-dir>/results/system_error_by_day.csv), print what the DATA and the WEATHER did on that day:

  kWh vs -7d / +7d   national kWh of the transformers usable on both days, day D over the same weekday a week before / after.
                     A one-day jump that reverses a week later is a shock; a level that stays is a regime change.
  usable             share of kept transformers with a usable (unflagged) reading that day, and the median over all days
  cover              mean core coverage that day vs the median of the 28 days before
  flags %            share of kept transformers carrying each flag bit that day: sys regional gap outside dip
  temp / anomaly     household-weighted national mean temperature, change vs 7 days earlier, and the departure from the
                     same calendar days (+-3) of the other years
  pointer            a rough hint, not a verdict: 'data' (usable/coverage/flags look unusual), 'weather' (large anomaly or
                     7-day swing), 'neither' (the demand itself moved, or a shock the flags do not see)

Usage:
  python scripts/analysis/error_day_check.py --work-dir ~/work/tf_national_v2 --forecast-dir ~/work/tf_national_v2/forecast_nodoy
  python scripts/analysis/error_day_check.py --work-dir ... --dates 2026-04-02,2026-04-07      # your own dates instead of the worst ones
Writes <forecast-dir>/results/error_day_check.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

FLAG_BITS = [("sys", 1), ("regional", 2), ("gap", 4), ("outside", 8), ("dip", 16)]


def ratio(kwh, i, j):
    """national kWh on day i over day j, over the transformers usable on both."""
    if j < 0 or j >= kwh.shape[1]:
        return np.nan
    m = np.isfinite(kwh[:, i]) & np.isfinite(kwh[:, j])
    if m.sum() < 20:
        return np.nan
    return float(kwh[m, i].sum() / kwh[m, j].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--forecast-dir", default=None, help="default <work-dir>/forecast")
    ap.add_argument("--horizon", type=int, default=7, help="which horizon's worst days (default 7)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--dates", default="", help="comma list of YYYY-MM-DD; replaces the worst-days list")
    a = ap.parse_args()
    wd = os.path.expanduser(a.work_dir)
    fd = os.path.expanduser(a.forecast_dir or os.path.join(wd, "forecast"))

    z = np.load(os.path.join(wd, "clean", "transformer_clean.npz"), allow_pickle=False)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    kept = z["kept"].astype(bool)
    inten = z["intensity"].astype("float64")[kept]
    cover = z["cover"].astype("float64")[kept]
    flags = z["flags"][kept]
    ncore = z["n_core"].astype("float64")[kept]
    coato = z["coato_code"].astype(str)[kept]
    kwh = inten * ncore[:, None]
    T = len(dates)

    wts = pd.Series(coato).value_counts()
    w = pd.read_parquet(os.path.join(wd, "weather_daily.parquet"))
    w["coato_id"] = w["coato_id"].astype(str)
    w["d"] = pd.to_datetime(w["d"])
    piv = w.pivot_table(index="d", columns="coato_id", values="t").reindex(dates)
    cols = [c for c in wts.index if c in piv.columns]
    wv = wts[cols] / wts[cols].sum()
    nat_t = pd.Series(np.nansum(piv[cols].to_numpy() * wv.to_numpy()[None, :], axis=1), index=dates)
    nat_t[piv[cols].isna().all(axis=1)] = np.nan
    print(f"[weather] {len(cols)} of {wts.size} coatos with weather; temperatures weighted by transformers per coato")

    usable = np.isfinite(inten).mean(axis=0)
    cov_day = np.nanmean(np.where(np.isfinite(cover), cover, np.nan), axis=0)
    cov_base = pd.Series(cov_day).shift(1).rolling(28, min_periods=14).median().to_numpy()
    flag_share = {n: ((flags & b) > 0).mean(axis=0) for n, b in FLAG_BITS}
    dev7 = np.array([abs(ratio(kwh, i, i - 7) - 1) if i >= 7 else np.nan for i in range(T)])

    if a.dates:
        want = [pd.Timestamp(d.strip()) for d in a.dates.split(",") if d.strip()]
        errs = {}
    else:
        p = os.path.join(fd, "results", "system_error_by_day.csv")
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found: run transformer_forecast.py run --stages compare first, or pass --dates")
        e = pd.read_csv(p)
        e = e[e["horizon"] == a.horizon].copy()
        e["date"] = pd.to_datetime(e["date"])
        e = e.reindex(e["err_direct"].abs().sort_values(ascending=False).index).head(a.top)
        want = list(e["date"])
        errs = dict(zip(e["date"], e["err_direct"]))

    doy = np.asarray(dates.dayofyear)
    rows = []
    for d in want:
        if d not in dates:
            print(f"{d.date()}: outside the panel's dates")
            continue
        i = dates.get_loc(d)
        same = (np.abs(doy - doy[i]) <= 3) & (dates.year != d.year)
        clim = np.nanmean(nat_t[same]) if same.any() else np.nan
        r_prev, r_next = ratio(kwh, i, i - 7), ratio(kwh, i, i + 7)
        rows.append(dict(
            date=d.date(), weekday=d.strftime("%a"), direct_err=errs.get(d, np.nan),
            kwh_vs_minus7=r_prev - 1, kwh_vs_plus7=r_next - 1,
            usable=usable[i], usable_median=float(np.median(usable)),
            cover=cov_day[i], cover_base=cov_base[i],
            **{f"flag_{n}": flag_share[n][i] for n, _ in FLAG_BITS},
            temp=nat_t.iloc[i], temp_chg_7d=nat_t.iloc[i] - nat_t.iloc[i - 7] if i >= 7 else np.nan, temp_anomaly=nat_t.iloc[i] - clim))
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no usable dates")
    flag_cols = [f"flag_{n}" for n, _ in FLAG_BITS]

    def pointer(r):
        data = (r.usable < 0.9 * r.usable_median) or (np.isfinite(r.cover_base) and r.cover < 0.97 * r.cover_base) \
            or any(r[c] > 0.05 for c in flag_cols)
        wx = (np.isfinite(r.temp_anomaly) and abs(r.temp_anomaly) > 3.0) or (np.isfinite(r.temp_chg_7d) and abs(r.temp_chg_7d) > 5.0)
        return "+".join([n for n, f in (("data", data), ("weather", wx)) if f]) or "neither"

    out["pointer"] = [pointer(out.iloc[k]) for k in range(len(out))]
    os.makedirs(os.path.join(fd, "results"), exist_ok=True)
    out.to_csv(os.path.join(fd, "results", "error_day_check.csv"), index=False)

    pct = lambda v: f"{v:+.1%}" if np.isfinite(v) else "n/a"
    print(f"\n{'date':<11}{'wd':<4}{'direct':>8}{'kWh/-7d':>9}{'kWh/+7d':>9}{'usable':>8}{'(med)':>7}{'cover':>7}{'(base)':>8}"
          f"{'sys%':>6}{'reg%':>6}{'gap%':>6}{'out%':>6}{'dip%':>6}{'temp':>7}{'d7':>6}{'anom':>6}  pointer")
    for r in out.itertuples(index=False):
        print(f"{str(r.date):<11}{r.weekday:<4}{pct(r.direct_err):>8}{pct(r.kwh_vs_minus7):>9}{pct(r.kwh_vs_plus7):>9}"
              f"{r.usable:>8.0%}{r.usable_median:>7.0%}{r.cover:>7.2f}{r.cover_base:>8.2f}"
              f"{r.flag_sys:>6.1%}{r.flag_regional:>6.1%}{r.flag_gap:>6.1%}{r.flag_outside:>6.1%}{r.flag_dip:>6.1%}"
              f"{r.temp:>7.1f}{r.temp_chg_7d:>+6.1f}{r.temp_anomaly:>+6.1f}  {r.pointer}")
    print(f"\ncontext, all days: median |kWh vs -7d| = {np.nanmedian(dev7):.1%}; 90th percentile {np.nanpercentile(dev7, 90):.1%}; "
          f"median usable share {np.median(usable):.0%}")
    print(f"pointers: {out['pointer'].value_counts().to_dict()}   -> {os.path.join(fd, 'results', 'error_day_check.csv')}")


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis")
