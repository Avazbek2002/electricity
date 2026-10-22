#!/usr/bin/env python
"""
What does the household data look like before the August 2024 regime change, and what does it mean for training?

Reads the enriched dataset (v1.4) district by district (bounded reads, one district per worker) and reports, per month
(2022-01 .. 2026-06), summed over the districts read:

  A. label states of customer-days: complete_observed (1-day reading interval), complete_with_estimated (2-3 days),
     partial, unlabelled (with or without a long-gap allocation); customers with any row / any observed day
  B. observed days per customer-month (0, 1-4, 5-9, 10-19, 20-27, 28+) and the gap between consecutive observed days
     (1 = consecutive, 2, 3, 4-7, 8-31, 32+): the reading cadence
  C. kWh per customer-day by state (observed, estimated, allocated), share of observed days with a whole-number kWh
     (the old registers) and with zero
  D. window eligibility for every origin (the 15th of each month; history up to o - 1, as the harness): share of the
     customers present with fresh history (>= 60 of 90 observed, >= 20 of 30), with >= 54 / >= 30 of the next 60 target
     days observed, and both -- why so few training windows exist before 2024-08
  E. the same customers a year apart: mean observed kWh/day in month m vs m + 12 (customers with >= 10 observed days in
     both), and the same with allocation-assisted energy -- a jump at the regime boundary vs the year-over-year drift
  F. weekday of the observed days (reading schedule)

District-month sums only (no customer key leaves the worker); months with < 10 customers are blanked in the tables.
Charts in figures/, tables as CSV, report.txt; logged to Weights & Biases (job_type "analysis") when it is set up.

    python scripts/analysis/regime_audit.py --sample --workers 5 --out-dir ~/work/regime_audit
    python scripts/analysis/regime_audit.py --workers 16 --buckets 0-19 --out-dir ~/work/regime_audit_national
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO, os.path.join(REPO, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STATES = ["complete_observed", "complete_with_estimated", "partial", "unlabelled_allocated", "unlabelled"]
OBS_BUCKETS = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 27), (28, 31)]
GAP_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 7), (8, 31), (32, 10 ** 6)]
REGIME = "2024-08-01"


def _read(code, uri, start, end, buckets):
    """-> keys n, state (n, T) int8 (0 = no row, 1..5 = STATES), kwh (n, T) float32 clean kWh, alloc (n, T) float32."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    import enriched_io as eio
    fs, root = eio.open_root(uri)
    dset = eio.district_daily(fs, root, code)
    names = dset.schema.names
    d0, d1 = pd.Timestamp(start), pd.Timestamp(end)
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    cols = [c for c in ("customer_key", "date", "kwh", "label_state", "is_long_gap_allocated", "kwh_long_gap_allocated")
            if c in names]
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    if buckets is not None and "customer_bucket" in names:
        flt = flt & pads.field("customer_bucket").isin(buckets)
    key_of, cap, n = {}, 4096, 0
    S = np.zeros((cap, T), np.int8)
    K = np.full((cap, T), np.nan, np.float32)
    A = np.full((cap, T), np.nan, np.float32)
    code_of = {s: i + 1 for i, s in enumerate(["complete_observed", "complete_with_estimated", "partial"])}
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        if not b.num_rows:
            continue
        de = pc.dictionary_encode(b.column("customer_key"))
        idx = np.empty(len(de.dictionary), np.int64)
        for j, s in enumerate(de.dictionary.to_pylist()):
            v = key_of.get(s)
            if v is None:
                v = key_of[s] = n
                n += 1
            idx[j] = v
        c = idx[de.indices.to_numpy(zero_copy_only=False)]
        if n > cap:
            new = max(n, 2 * cap)
            S = np.vstack([S, np.zeros((new - cap, T), np.int8)])
            K = np.vstack([K, np.full((new - cap, T), np.nan, np.float32)])
            A = np.vstack([A, np.full((new - cap, T), np.nan, np.float32)])
            cap = new
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        ls = np.asarray(b.column("label_state").to_pylist(), dtype=object)
        st = np.array([code_of.get(x, 5) for x in ls], np.int8)
        if "is_long_gap_allocated" in cols:
            al = pc.fill_null(b.column("is_long_gap_allocated"), False).to_numpy(zero_copy_only=False).astype(bool)
            st = np.where((st == 5) & al, 4, st).astype(np.int8)
        S[c, day] = st
        kv = pc.cast(b.column("kwh"), pa.float64()).to_numpy(zero_copy_only=False)
        ok = np.isfinite(kv)
        K[c[ok], day[ok]] = kv[ok]
        if "kwh_long_gap_allocated" in cols:
            av = pc.cast(b.column("kwh_long_gap_allocated"), pa.float64()).to_numpy(zero_copy_only=False)
            ok = np.isfinite(av)
            A[c[ok], day[ok]] = av[ok]
    return S[:n], K[:n], A[:n], d0, T


def worker(code, cfg):
    t0 = time.time()
    S, K, A, d0, T = _read(code, cfg["uri"], cfg["start"], cfg["end"], cfg["buckets"])
    out = dict(code=str(code), customers=len(S), seconds=0.0)
    if not len(S):
        return out
    dates = d0 + pd.to_timedelta(np.arange(T), unit="D")
    mon = (dates.year - d0.year) * 12 + dates.month - d0.month           # month index per day
    nM = int(mon.max()) + 1
    obs = S == 1
    rows = []
    # ---- A, C, F per month
    dow = dates.dayofweek.to_numpy()
    for m in range(nM):
        sl = np.flatnonzero(mon == m)
        s, k, a, o = S[:, sl], K[:, sl], A[:, sl], obs[:, sl]
        anyrow = (s > 0).any(1)
        r = dict(month=m, customers_rows=int(anyrow.sum()), customers_observed=int(o.any(1).sum()),
                 customer_days=int((s > 0).sum()))
        for i, name in enumerate(STATES, start=1):
            r[f"days_{name}"] = int((s == i).sum())
        ko = np.where(o, k, np.nan)
        r["kwh_observed"] = float(np.nansum(ko))
        r["kwh_estimated"] = float(np.nansum(np.where(s == 2, k, np.nan)))
        r["kwh_allocated"] = float(np.nansum(np.where(s == 4, a, np.nan)))
        r["days_allocated_value"] = int(np.isfinite(np.where(s == 4, a, np.nan)).sum())
        vo = ko[np.isfinite(ko)]
        r["observed_whole_kwh"] = int(((vo > 0) & (np.abs(vo - np.round(vo)) < 1e-6)).sum())
        r["observed_zero"] = int((vo == 0).sum())
        cnt = o.sum(1)[anyrow]
        for lo, hi in OBS_BUCKETS:
            r[f"cust_obs_{lo}_{hi}"] = int(((cnt >= lo) & (cnt <= hi)).sum())
        for d in range(7):
            r[f"obs_dow_{d}"] = int(o[:, dow[sl] == d].sum())
        rows.append(r)
    A_tab = pd.DataFrame(rows)
    # ---- B: gaps between consecutive observed days (attributed to the month of the later day)
    gap = np.zeros((nM, len(GAP_BUCKETS)), np.int64)
    run_n = np.zeros(nM, np.int64)                                          # runs of consecutive observed days
    run_len = np.zeros(nM, np.int64)
    for i in range(len(S)):
        d = np.flatnonzero(obs[i])
        if len(d) < 2:
            if len(d) == 1:
                run_n[mon[d[0]]] += 1
                run_len[mon[d[0]]] += 1
            continue
        g = np.diff(d)
        mm = mon[d[1:]]
        for j, (lo, hi) in enumerate(GAP_BUCKETS):
            sel = (g >= lo) & (g <= hi)
            if sel.any():
                np.add.at(gap[:, j], mm[sel], 1)
        starts = np.r_[0, np.flatnonzero(g > 1) + 1]
        ends = np.r_[np.flatnonzero(g > 1), len(d) - 1]
        np.add.at(run_n, mon[d[starts]], 1)
        np.add.at(run_len, mon[d[starts]], ends - starts + 1)
    B_tab = pd.DataFrame(gap, columns=[f"gap_{lo}_{hi if hi < 10 ** 6 else 'up'}" for lo, hi in GAP_BUCKETS])
    B_tab["month"] = np.arange(nM)
    B_tab["runs"], B_tab["run_days"] = run_n, run_len
    # ---- D: eligibility at origins on the 15th (history up to o - 1)
    cnt = np.zeros((len(S), T + 1), np.int32)
    np.cumsum(obs, axis=1, out=cnt[:, 1:])
    anyr = np.zeros((len(S), T + 1), np.int32)
    np.cumsum(S > 0, axis=1, out=anyr[:, 1:])
    erows = []
    for o in np.flatnonzero(dates.day == 15):
        last = o - 1
        if last + 1 - 90 < 0 or o + 60 > T - 1:
            continue
        present = (anyr[:, last + 1] > 0) & (anyr[:, T] - anyr[:, last] > 0)       # a row on/before and on/after o - 1
        n_in = cnt[:, last + 1] - cnt[:, last + 1 - 90]
        n_rec = cnt[:, last + 1] - cnt[:, last + 1 - 30]
        n_tg = cnt[:, o + 61] - cnt[:, o + 1]
        fresh = (n_in >= 60) & (n_rec >= 20)
        erows.append(dict(origin=str(dates[o].date()), present=int(present.sum()), fresh=int((fresh & present).sum()),
                          tgt54=int((present & (n_tg >= 54)).sum()), tgt30=int((present & (n_tg >= 30)).sum()),
                          fresh_tgt54=int((fresh & (n_tg >= 54)).sum()), fresh_tgt30=int((fresh & (n_tg >= 30)).sum()),
                          in_any=int((present & (n_in >= 1)).sum())))
    D_tab = pd.DataFrame(erows)
    # ---- E: the same customers a year apart (month m vs m + 12)
    def monthly_means(V, mask):
        s = np.zeros((len(S), nM))
        c = np.zeros((len(S), nM))
        for m in range(nM):
            sl = mon == m
            v = np.where(mask[:, sl], V[:, sl], np.nan)
            s[:, m] = np.nansum(v, 1)
            c[:, m] = np.isfinite(v).sum(1)
        return s, c
    so, co = monthly_means(K, obs)
    assisted = np.where(np.isfinite(K), K, 0) + np.where(np.isfinite(A), A, 0)
    has = np.isfinite(K) | np.isfinite(A)
    sa, ca = monthly_means(assisted.astype(np.float32), has)
    yrows = []
    for m in range(nM - 12):
        for name, s_, c_, need in (("observed", so, co, 10), ("assisted", sa, ca, 20)):
            both = (c_[:, m] >= need) & (c_[:, m + 12] >= need)
            if both.any():
                yrows.append(dict(month=m, kind=name, customers=int(both.sum()),
                                  mean_first=float((s_[both, m] / c_[both, m]).sum()),
                                  mean_second=float((s_[both, m + 12] / c_[both, m + 12]).sum())))
    E_tab = pd.DataFrame(yrows)
    out.update(A=A_tab, B=B_tab, D=D_tab, E=E_tab, nM=nM, seconds=time.time() - t0)
    return out


# ----------------------------------------------------------------------------- charts
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
STY = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-", "--"]


def _fig(title, panels=1, h=3.8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(panels, 1, figsize=(12, h * panels), squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    return fig, axes[:, 0]


def _style(ax, pct=True, ylabel=None):
    import matplotlib.ticker as mt
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    for a, b, lab in (("2023-01-01", "2023-02-28", "crisis"), ("2024-05-01", "2024-08-15", "label hole")):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#e6e5e0", alpha=0.6, lw=0)
    ax.axvline(pd.Timestamp(REGIME), color=MUTED, lw=1, ls="--")


def _lines(ax, x, frame, cols, labels=None):
    for i, c in enumerate(cols):
        ax.plot(x, frame[c], color=COL[i % len(COL)], linestyle=STY[i % len(STY)], linewidth=1.8,
                label=(labels or cols)[i])
    ax.legend(fontsize=8, frameon=False, ncol=min(len(cols), 7), loc="upper center", bbox_to_anchor=(0.5, -0.12))


def _save(fig, path):
    import matplotlib.pyplot as plt
    fig.patch.set_facecolor(SURF)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-set", default=os.path.join(REPO, "configs", "test_set.yaml"), help="for the data uri / dates")
    p.add_argument("--sample", action="store_true", help="the guide's 5 districts")
    p.add_argument("--coatos", default=None, help="comma-separated district codes (default: every finished one)")
    p.add_argument("--buckets", default=None, help="customer_bucket values, e.g. 0-19 (20 %%) or 0,1,2; default all")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args(argv)
    from harness import testset as ts
    import enriched_io as eio
    spec = ts.load_spec(a.test_set)
    uri, start, end = spec["data"]["uri"], spec["data"]["start"], spec["data"]["end"]
    buckets = None
    if a.buckets:
        buckets = (list(range(int(a.buckets.split("-")[0]), int(a.buckets.split("-")[1]) + 1)) if "-" in a.buckets
                   else [int(x) for x in a.buckets.split(",")])
    fs, root = eio.open_root(uri)
    want = set(spec["sample_districts"]) if a.sample else (set(a.coatos.split(",")) if a.coatos else None)
    codes = eio.finished_districts(fs, root, want)
    out = os.path.expanduser(a.out_dir)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    cfg = dict(uri=uri, start=start, end=end, buckets=buckets)
    t0, res = time.time(), []
    with cf.ProcessPoolExecutor(max_workers=min(a.workers, len(codes)), mp_context=mp.get_context("spawn")) as ex:
        fut = {ex.submit(worker, c, cfg): c for c in codes}
        for i, f in enumerate(cf.as_completed(fut), 1):
            r = f.result()
            res.append(r)
            if i <= 3 or i % 10 == 0 or i == len(codes):
                print(f"[audit] {i}/{len(codes)} districts ({r['code']}: {r['customers']:,} customers, "
                      f"{r['seconds']:.0f} s); {(time.time() - t0) / 60:.1f} min", flush=True)
    res = [r for r in res if "A" in r]
    d0 = pd.Timestamp(start)
    month_ts = lambda m: d0 + pd.DateOffset(months=int(m))              # noqa: E731
    A = pd.concat([r["A"] for r in res]).groupby("month").sum().reset_index()
    B = pd.concat([r["B"] for r in res]).groupby("month").sum().reset_index()
    D = pd.concat([r["D"] for r in res]).groupby("origin").sum().reset_index()
    E = pd.concat([r["E"] for r in res]).groupby(["month", "kind"]).sum().reset_index()
    A["date"], B["date"] = A["month"].map(month_ts), B["month"].map(month_ts)
    D["date"] = pd.to_datetime(D["origin"])
    E["date"] = E["month"].map(month_ts)
    A = A[A["customers_rows"] >= 10]

    # ---- derived shares
    L = []

    def say(s=""):
        print(s)
        L.append(s)
    tot = A["customer_days"].replace(0, np.nan)
    for s in STATES:
        A[f"share_{s}"] = A[f"days_{s}"] / tot
    rowc = A["customers_rows"].replace(0, np.nan)
    for lo, hi in OBS_BUCKETS:
        A[f"share_cust_obs_{lo}_{hi}"] = A[f"cust_obs_{lo}_{hi}"] / rowc
    A["kwh_per_observed_day"] = A["kwh_observed"] / A["days_complete_observed"].replace(0, np.nan)
    A["kwh_per_estimated_day"] = A["kwh_estimated"] / A["days_complete_with_estimated"].replace(0, np.nan)
    A["kwh_per_allocated_day"] = A["kwh_allocated"] / A["days_allocated_value"].replace(0, np.nan)
    A["whole_kwh_share"] = A["observed_whole_kwh"] / A["days_complete_observed"].replace(0, np.nan)
    A["zero_share"] = A["observed_zero"] / A["days_complete_observed"].replace(0, np.nan)
    A["observed_days_per_customer"] = A["days_complete_observed"] / rowc
    od = A[[f"obs_dow_{d}" for d in range(7)]]
    for d in range(7):
        A[f"dow_share_{d}"] = od[f"obs_dow_{d}"] / od.sum(axis=1).replace(0, np.nan)
    gcols = [c for c in B.columns if c.startswith("gap_")]
    gs = B[gcols].sum(axis=1).replace(0, np.nan)
    for c in gcols:
        B[f"share_{c}"] = B[c] / gs
    B["mean_run_days"] = B["run_days"] / B["runs"].replace(0, np.nan)
    pres = D["present"].replace(0, np.nan)
    for c in ("fresh", "tgt54", "tgt30", "fresh_tgt54", "fresh_tgt30"):
        D[f"share_{c}"] = D[c] / pres
    E["ratio"] = E["mean_second"] / E["mean_first"]
    A.to_csv(os.path.join(out, "monthly_states.csv"), index=False)
    B.to_csv(os.path.join(out, "monthly_gaps.csv"), index=False)
    D.to_csv(os.path.join(out, "eligibility_by_origin.csv"), index=False)
    E.to_csv(os.path.join(out, "same_customers_year_apart.csv"), index=False)

    # ---- report: before / after the regime change
    old, new = A["date"] < pd.Timestamp(REGIME), A["date"] >= pd.Timestamp("2024-09-01")
    say(f"districts {len(res)} | customers read {sum(r['customers'] for r in res):,} | buckets {a.buckets or 'all'} | "
        f"{start} .. {end}")
    say("\n=== A. customer-day label states, mean of the monthly shares: before 2024-08 | from 2024-09 ===")
    for s in STATES:
        say(f"  {s:26s} {A.loc[old, f'share_{s}'].mean():7.1%} | {A.loc[new, f'share_{s}'].mean():7.1%}")
    say(f"  observed days per customer-month   {A.loc[old, 'observed_days_per_customer'].mean():5.1f} | "
        f"{A.loc[new, 'observed_days_per_customer'].mean():5.1f}")
    say("\n=== B. customers by observed days in the month (share of customers with any row) ===")
    for lo, hi in OBS_BUCKETS:
        c = f"share_cust_obs_{lo}_{hi}"
        say(f"  {lo:>2}-{hi:<2} days  {A.loc[old, c].mean():7.1%} | {A.loc[new, c].mean():7.1%}")
    oldB, newB = B["date"] < pd.Timestamp(REGIME), B["date"] >= pd.Timestamp("2024-09-01")
    say("  gap between consecutive observed days (share of gaps):")
    for c in gcols:
        say(f"    {c:12s} {B.loc[oldB, f'share_{c}'].mean():7.1%} | {B.loc[newB, f'share_{c}'].mean():7.1%}")
    say(f"  mean run of consecutive observed days {B.loc[oldB, 'mean_run_days'].mean():5.1f} | "
        f"{B.loc[newB, 'mean_run_days'].mean():5.1f}")
    say("\n=== C. kWh per customer-day and number format ===")
    for c in ("kwh_per_observed_day", "kwh_per_estimated_day", "kwh_per_allocated_day", "whole_kwh_share", "zero_share"):
        f = "{:7.1%}" if "share" in c else "{:7.2f}"
        say(f"  {c:24s} {f.format(A.loc[old, c].mean())} | {f.format(A.loc[new, c].mean())}")
    say("\n=== D. eligibility at origins (share of the customers present), by origin year-half ===")
    D["half"] = D["date"].dt.year.astype(str) + np.where(D["date"].dt.month <= 6, "-H1", "-H2")
    q = D.groupby("half")[["share_fresh", "share_tgt54", "share_tgt30", "share_fresh_tgt54", "share_fresh_tgt30"]].mean()
    say(q.to_string(float_format=lambda x: f"{x:7.1%}"))
    say("  (fresh = >= 60 of 90 and >= 20 of 30 observed up to o - 1; tgt54 / tgt30 = observed target days of 60;"
        " the boosters train on fresh & tgt54 windows, the harness forecasts fresh households)")
    say("\n=== E. the same customers a year apart: mean kWh/day in month m+12 / month m ===")
    piv = E.pivot_table(index="date", columns="kind", values="ratio")
    piv["customers_observed"] = E[E["kind"] == "observed"].set_index("date")["customers"]
    say(piv.to_string(float_format=lambda x: f"{x:6.3f}" if x < 100 else f"{x:,.0f}"))
    say("  a pair that crosses 2024-08 (m in 2023-08 .. 2024-07) compares old vs new regime; the others are the "
        "ordinary year-over-year change")

    # ---- charts
    figs = []
    fig, ax = _fig("A. Customer-day label states (share of customer-days with a row)")
    _lines(ax[0], A["date"], A, [f"share_{s}" for s in STATES], STATES)
    _style(ax[0])
    figs.append(_save(fig, os.path.join(out, "figures", "A_label_states.png")))
    fig, ax = _fig("B. Customers by observed days in the month, and gaps between observed days", 2)
    _lines(ax[0], A["date"], A, [f"share_cust_obs_{lo}_{hi}" for lo, hi in OBS_BUCKETS],
           [f"{lo}-{hi} observed days" for lo, hi in OBS_BUCKETS])
    _style(ax[0])
    _lines(ax[1], B["date"], B, [f"share_{c}" for c in gcols], [c.replace("gap_", "gap ").replace("_", "-") + " days"
                                                                 for c in gcols])
    _style(ax[1])
    figs.append(_save(fig, os.path.join(out, "figures", "B_observed_days_and_gaps.png")))
    fig, ax = _fig("C. kWh per customer-day by state, and number format of observed days", 2)
    _lines(ax[0], A["date"], A, ["kwh_per_observed_day", "kwh_per_estimated_day", "kwh_per_allocated_day"],
           ["observed", "estimated (2-3 day)", "long-gap allocated"])
    _style(ax[0], pct=False, ylabel="kWh per customer-day")
    _lines(ax[1], A["date"], A, ["whole_kwh_share", "zero_share"], ["whole-number kWh", "zero kWh"])
    _style(ax[1])
    figs.append(_save(fig, os.path.join(out, "figures", "C_kwh_and_format.png")))
    fig, ax = _fig("D. Window eligibility by origin (share of customers present; history to o - 1)")
    _lines(ax[0], D["date"], D, ["share_fresh", "share_tgt54", "share_tgt30", "share_fresh_tgt54", "share_fresh_tgt30"],
           ["fresh history", ">= 54/60 target days", ">= 30/60 target days", "fresh & >= 54", "fresh & >= 30"])
    _style(ax[0])
    figs.append(_save(fig, os.path.join(out, "figures", "D_eligibility.png")))
    fig, ax = _fig("E. Same customers a year apart: mean kWh/day in month m+12 / month m (x = m)")
    _lines(ax[0], piv.index, piv, [c for c in ("observed", "assisted") if c in piv],
           ["observed days only", "allocation-assisted"])
    ax[0].axhline(1, color=MUTED, lw=0.8)
    _style(ax[0], pct=False, ylabel="ratio")
    ax[0].axvspan(pd.Timestamp("2023-08-01"), pd.Timestamp("2024-07-31"), color="#cde2fb", alpha=0.35, lw=0)
    figs.append(_save(fig, os.path.join(out, "figures", "E_same_customers_year_apart.png")))
    fig, ax = _fig("F. Weekday of observed days (share)")
    _lines(ax[0], A["date"], A, [f"dow_share_{d}" for d in range(7)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    _style(ax[0])
    figs.append(_save(fig, os.path.join(out, "figures", "F_weekday.png")))
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[audit] {out}: report.txt, 4 CSV tables, {len(figs)} figures; {(time.time() - t0) / 60:.1f} min")

    # ---- W&B
    if a.no_wandb:
        return
    import yaml
    from harness import wandb_log
    wcfg = (yaml.safe_load(open(os.path.join(REPO, "configs", "eval.yaml"))) or {}).get("wandb") or {}
    run = wandb_log.start(wcfg, f"regime audit | {len(res)} districts | buckets {a.buckets or 'all'}", "data audit",
                          "analysis", dict(districts=len(res), buckets=a.buckets, start=start, end=end, uri=uri),
                          ["analysis", "regime-audit"])
    if run is None:
        return
    import wandb
    try:
        for f in figs:
            run.log({f"audit/{os.path.basename(f)[:-4]}": wandb.Image(f)})
        for name, t in (("monthly_states", A), ("eligibility_by_origin", D), ("same_customers_year_apart", E),
                        ("monthly_gaps", B)):
            run.log({f"tables/{name}": wandb.Table(dataframe=t.astype({c: str for c in t.columns
                                                                         if str(t[c].dtype).startswith("datetime")}))})
        art = wandb.Artifact(f"regime-audit-{run.id}", type="analysis")
        art.add_file(os.path.join(out, "report.txt"))
        run.log_artifact(art)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
