#!/usr/bin/env python3
"""
Feeder-level coverage of household consumption in the national enriched dataset v1.4, 2022-01-01 .. 2026-06-29.

For every (district, feeder, day):
  in_span    households with a row that day (inside a recorded meter span; rows exist only there, guide rule 1)
  observed   of those, households with a complete_observed day and clean kWh (the only usable value, HH-GAP-001-A1)
  kwh_obs    sum of the observed households' kWh
and per feeder: households (distinct customers ever mapped to it), coverage = observed / in_span by day, month and year.

Feeders are flagged:  placeholder = > 50% of its households on placeholder transformers (> 1,000 customers);
unknown = negative id (minus the district code: "feeder unknown in this district") or no feeder id;
small = < --min-households households (default 100). The charts use regular feeders only.

Reads every district once (separate processes, streamed record batches, a few columns); returns only aggregates.
Outputs (<out-dir>), no customer keys:
  feeder_daily.parquet     coato_code, feeder_id, date, in_span, observed, kwh_obs
  feeder_summary.csv       one row per feeder: households, flags, mean coverage per year, share of days >= 50% / 80%
  coverage_monthly.csv     median / p10 / p90 feeder coverage per month (regular feeders)
  feeder_coverage_heatmap.png, feeder_coverage_trend.png

  python scripts/analysis/feeder_coverage.py --out-dir ~/work/feeder_eda --workers 12
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))  # the shared modules
sys.path.insert(0, HERE)


def district_worker(code, cfg):
    """One district -> per (feeder, day) counts and per feeder distinct households / placeholder share."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    sys.path.insert(0, cfg["here"])
    import enriched_io as eio
    t0 = time.time()
    fs, root = eio.open_root(cfg["root"])
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    dset = eio.district_daily(fs, root, code)
    cols = [c for c in ("customer_key", "date", "feeder_id", "label_state", "kwh", "placeholder_transformer")
            if c in dset.schema.names]
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    labels = pa.array(list(cfg["labels"]), type=pa.string())
    fidx, fids = {}, []                               # feeder id -> local index
    key_of = {}
    span = np.zeros((0, T), dtype=np.int32)
    obs = np.zeros((0, T), dtype=np.int32)
    kwh = np.zeros((0, T), dtype=np.float64)
    pairs, ph_pairs = [], []
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        if not b.num_rows:
            continue
        de = pc.dictionary_encode(b.column("customer_key"))
        uq = [str(x) for x in de.dictionary.to_pylist()]
        ck = np.empty(len(uq), dtype=np.int64)
        for j, s in enumerate(uq):
            v = key_of.get(s)
            if v is None:
                v = key_of[s] = len(key_of)
            ck[j] = v
        cust = ck[de.indices.to_numpy(zero_copy_only=False)]
        fcol = b.column("feeder_id")
        fv = pc.fill_null(pc.cast(fcol, pa.int64()), -(2 ** 62)).to_numpy(zero_copy_only=False)
        fu, finv = np.unique(fv, return_inverse=True)
        li = np.empty(len(fu), dtype=np.int64)
        for j, f in enumerate(fu):
            v = fidx.get(int(f))
            if v is None:
                v = fidx[int(f)] = len(fids)
                fids.append(int(f))
            li[j] = v
        fl = li[finv]
        if len(fids) > span.shape[0]:
            grow = max(len(fids), 2 * span.shape[0], 16) - span.shape[0]
            span = np.vstack([span, np.zeros((grow, T), np.int32)])
            obs = np.vstack([obs, np.zeros((grow, T), np.int32)])
            kwh = np.vstack([kwh, np.zeros((grow, T))])
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        kv = b.column("kwh")
        ok = pc.and_(pc.fill_null(pc.is_in(b.column("label_state"), value_set=labels), False), pc.is_valid(kv))
        ok = ok.to_numpy(zero_copy_only=False)
        np.add.at(span, (fl, day), 1)
        np.add.at(obs, (fl[ok], day[ok]), 1)
        np.add.at(kwh, (fl[ok], day[ok]), kv.to_numpy(zero_copy_only=False)[ok])
        pk = np.unique(fl * (1 << 40) + cust)
        pairs.append(pk)
        if "placeholder_transformer" in cols:
            phv = pc.fill_null(b.column("placeholder_transformer"), False).to_numpy(zero_copy_only=False)
            if phv.any():
                ph_pairs.append(np.unique(fl[phv] * (1 << 40) + cust[phv]))
    nf = len(fids)
    out = dict(code=str(code), seconds=0.0, daily=None, feeders=None)
    if nf:
        pk = np.unique(np.concatenate(pairs))
        hh = np.bincount(pk >> 40, minlength=nf)
        phk = np.unique(np.concatenate(ph_pairs)) if ph_pairs else np.zeros(0, np.int64)
        ph = np.bincount(phk >> 40, minlength=nf) if len(phk) else np.zeros(nf, np.int64)
        ids = np.array(["missing" if f == -(2 ** 62) else str(f) for f in fids], dtype=object)
        out["feeders"] = pd.DataFrame({"coato_code": str(code), "feeder_id": ids, "households": hh,
                                       "placeholder_share": ph / np.maximum(hh, 1)})
        r, d = np.nonzero(span[:nf])
        out["daily"] = pd.DataFrame({"coato_code": str(code), "feeder_id": ids[r],
                                     "date": d0 + pd.to_timedelta(d, unit="D"), "in_span": span[r, d].astype(np.int32),
                                     "observed": obs[r, d].astype(np.int32), "kwh_obs": kwh[r, d].astype(np.float32)})
    out["seconds"] = time.time() - t0
    return out


def main(argv=None):
    import enriched_io as eio
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=eio.ENRICHED_ROOT)
    p.add_argument("--start", default=eio.WINDOW_START)
    p.add_argument("--end", default=eio.WINDOW_END)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--min-households", type=int, default=100, help="regular feeders: at least this many households")
    p.add_argument("--labels", default="complete_observed")
    args = p.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)
    cfg = dict(here=HERE, root=args.root, start=args.start, end=args.end, labels=args.labels.split(","))
    fs, root = eio.open_root(args.root)
    codes = eio.finished_districts(fs, root)
    t0 = time.time()
    daily, feeders = [], []
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(district_worker, c, cfg): c for c in codes}
        k = 0
        while pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for f in done:
                code = pending.pop(f)
                try:
                    r = f.result()
                except Exception as e:                                   # noqa: BLE001
                    raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                del f
                k += 1
                if r["daily"] is not None:
                    daily.append(r["daily"])
                    feeders.append(r["feeders"])
                if k % 10 == 0 or k == len(codes) or k <= 3:
                    el = time.time() - t0
                    print(f"[read] {k}/{len(codes)} districts, {el / 60:.1f} min, about {el / k * (len(codes) - k) / 60:.1f} min left",
                          flush=True)
    daily = pd.concat(daily, ignore_index=True)
    fe = pd.concat(feeders, ignore_index=True)
    daily.to_parquet(os.path.join(out, "feeder_daily.parquet"), index=False)
    summarise(daily, fe, args, out)
    print(f"\n[done] {out} in {(time.time() - t0) / 60:.1f} min")


def summarise(daily, fe, args, out):
    daily = daily.copy()
    daily["cov"] = daily["observed"] / daily["in_span"]
    neg = fe["feeder_id"].str.startswith("-") | (fe["feeder_id"] == "missing")
    fe["flag"] = np.select([neg, fe["placeholder_share"] > 0.5, fe["households"] < args.min_households],
                           ["unknown", "placeholder", "small"], default="regular")
    key = ["coato_code", "feeder_id"]
    daily = daily.merge(fe[key + ["flag"]], on=key, how="left")
    daily["year"] = daily["date"].dt.year
    yr = daily.pivot_table(index=key, columns="year", values="cov", aggfunc="mean")
    yr.columns = [f"coverage_{c}" for c in yr.columns]
    g = daily.groupby(key).agg(days_in_span=("cov", "size"), share_days_ge50=("cov", lambda s: float((s >= 0.5).mean())),
                               share_days_ge80=("cov", lambda s: float((s >= 0.8).mean())), mean_in_span=("in_span", "mean"))
    summ = fe.set_index(key).join(g).join(yr).reset_index()
    summ.to_csv(os.path.join(out, "feeder_summary.csv"), index=False)

    print(f"\n[feeders] {len(fe):,} district-feeders | " + " | ".join(f"{k} {v:,}" for k, v in fe["flag"].value_counts().items()))
    hh = fe.groupby("flag")["households"].sum()
    print("  households by flag: " + " | ".join(f"{k} {v:,}" for k, v in hh.items()))
    reg = daily[daily["flag"] == "regular"]
    reg_m = reg.assign(month=reg["date"].dt.to_period("M"))
    fm = reg_m.groupby(key + ["month"]).agg(cov=("observed", "sum"), span=("in_span", "sum")).reset_index()
    fm["cov"] = fm["cov"] / fm["span"]
    mon = fm.groupby("month")["cov"].describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]]
    nat = reg_m.groupby("month")[["observed", "in_span"]].sum()
    mon["national"] = nat["observed"] / nat["in_span"]
    mon.to_csv(os.path.join(out, "coverage_monthly.csv"))
    print("\n  regular feeders, monthly coverage (observed / in span): median feeder [p10, p90] | national")
    for m, r in mon.iloc[::3].iterrows():
        print(f"  {m}  {r['50%']:6.1%}  [{r['10%']:6.1%}, {r['90%']:6.1%}] | {r['national']:6.1%}  ({int(r['count']):,} feeders)")
    sr = summ[summ["flag"] == "regular"]
    for y in [c for c in sr.columns if c.startswith("coverage_")]:
        v = sr[y].dropna()
        print(f"  {y}: feeders {len(v):,} | median {v.median():.1%} | share >= 50% {(v >= 0.5).mean():.1%} | "
              f">= 80% {(v >= 0.8).mean():.1%}")
    charts(fm, mon, out)


def charts(fm, mon, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e5e4df", "#2a78d6"
    # heatmap: regular feeders x month, sorted by their 2025-2026 coverage
    piv = fm.pivot_table(index=["coato_code", "feeder_id"], columns="month", values="cov")
    late = piv.loc[:, [c for c in piv.columns if c.year >= 2025]].mean(axis=1)
    piv = piv.loc[late.sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list("cov", ["#f3f1ec", "#9cc3ef", "#2a78d6", "#0d3b75"])
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    cols = list(piv.columns)
    ticks = [i for i, c in enumerate(cols) if c.month == 1]
    ax.set_xticks(ticks, [str(cols[i].year) for i in ticks])
    ax.set_yticks([])
    ax.set_ylabel(f"{len(piv):,} regular feeders, sorted by 2025–26 coverage", color=muted, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Share of the feeder's households observed", color=muted, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=muted)
    cb.outline.set_visible(False)
    med25 = float(np.nanmedian(late))
    ax.set_title(f"Feeder coverage: median regular feeder {med25:.0%} observed in 2025–26", loc="left", fontsize=12,
                 color=ink, pad=22)
    ax.text(0, 1.02, "Monthly share of in-span household-days with a complete_observed reading; enriched dataset v1.4",
            transform=ax.transAxes, fontsize=8.5, color=muted)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feeder_coverage_heatmap.png"))
    plt.close(fig)

    if mon.empty:                                                    # no regular feeder (e.g. a tiny test set): no trend chart
        print("  no regular feeders: trend chart skipped")
        return
    mon = mon.set_axis(pd.PeriodIndex(mon.index, freq="M"))
    x = mon.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    ax.fill_between(x, mon["10%"], mon["90%"], color=blue, alpha=0.15, linewidth=0, label="p10–p90 of feeders")
    ax.plot(x, mon["50%"], color=blue, linewidth=2, label="median feeder")
    ax.plot(x, mon["national"], color=ink, linewidth=1, linestyle=(0, (4, 3)), label="all regular feeders pooled")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    recent = mon[[p.year >= 2025 for p in mon.index]]
    lvl = float(recent["50%"].median()) if len(recent) else float(mon["50%"].median())
    ax.set_title(f"Observed share of feeder households by month: the median feeder is near {lvl:.0%} since 2025",
                 loc="left", fontsize=12, color=ink, pad=22)
    ax.text(0, 1.02, "Regular feeders; observed = complete_observed household-days / in-span household-days; the last month "
            "is incomplete (right edge)", transform=ax.transAxes, fontsize=8.5, color=muted)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(muted)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feeder_coverage_trend.png"))
    plt.close(fig)


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis")
