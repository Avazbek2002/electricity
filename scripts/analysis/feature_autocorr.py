#!/usr/bin/env python3
"""
Autocorrelation and feature-correlation analysis for the household NLinear model (enriched dataset v1.4).

1. Autocorrelation of household daily consumption, pooled over customers, observed days only, lags 1..120 and
   longer seasonal lags up to 365. Two versions:
     level   kWh / the customer's own mean - 1           (includes the seasonal swing: winter vs summer)
     short   kWh / its centred 31-day mean - 1          (seasonal swing removed: weekly and day-to-day structure)
2. Correlation between the model's inputs, on a sample of TRAINING windows exactly as fed to nlinear_x
   (90 normalised history days + 81 weather / calendar / year-ago inputs):
     - Pearson correlation matrix, the most correlated pairs, variance inflation factors (VIF = how much of an input
       the other inputs already explain: 1 = none, > 10 = mostly redundant), the condition number;
     - each input's correlation with the target (y / level - 1) at 1, 7, 14, 30 and 60 days ahead.

Uses household_nlinear.py's district workers (same windows, rules and features). Default: customer bucket 0 (1%).

  python scripts/analysis/feature_autocorr.py --out-dir ~/work/feature_autocorr --buckets 0 --workers 8

Outputs (<out-dir>), aggregates only: acf.csv, acf.png, feature_corr.csv, feature_corr_pairs.csv, feature_vif.csv,
feature_target_corr.csv, feature_corr_heatmap.png, run_log.txt.
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
import household_nlinear as hn                                              # noqa: E402

LAGS = list(range(1, 121)) + [135, 150, 182, 210, 240, 270, 300, 330, 357, 363, 364, 365]
HZ = (1, 7, 14, 30, 60)


def main(argv=None):
    import whole_pipeline_v8 as wp
    import enriched_io as eio
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=wp.ENRICHED_ROOT)
    p.add_argument("--buckets", default="0", help="'all' or a comma list of customer_bucket values")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dump-rate", type=float, default=0.2, help="share of training windows sampled for the feature analysis")
    p.add_argument("--max-rows", type=int, default=300_000)
    args = p.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)
    with hn._tee(os.path.join(out, "run_log.txt")):
        run(wp, eio, args, out)


def run(wp, eio, args, out):
    buckets = None if str(args.buckets).lower() == "all" else [int(b) for b in str(args.buckets).split(",")]
    cfg = dict(here=HERE, root=args.root, start=wp.ENRICHED_START, end=wp.ENRICHED_END, labels=list(wp.ENRICHED_LABELS),
               buckets=buckets, stride=wp.WINDOW_STRIDE, cutoff=wp.TRAIN_CUTOFF, val_start=wp.VAL_START, val_end=wp.VAL_END,
               test_sample=0.0, val_sample=0.0, norm_days=7, min_level=0.2, weight="kwh", prior_h=0.049, prior_c=0.103,
               prior_h_mm=0.047, prior_c_mm=0.061, block=10000, chunk=50000, seed=wp.SEED,
               wp=dict(INCLUDE_OLD=wp.INCLUDE_OLD, EXCLUDE_RANGES=list(wp.EXCLUDE_RANGES), EXCLUDE_MODE=wp.EXCLUDE_MODE,
                       REGIME_CUTOFF=wp.REGIME_CUTOFF, OLD_START=wp.OLD_START, USE_MINMAX=wp.USE_MINMAX),
               acf_lags=LAGS, dump_rate=args.dump_rate, dump_horizons=HZ)
    fs, root = eio.open_root(args.root)
    codes = eio.finished_districts(fs, root)
    print(f"[config] customers {'ALL' if buckets is None else buckets} | {len(codes)} districts | {args.workers} workers | "
          f"feature sample {args.dump_rate:.0%} of training windows (max {args.max_rows:,})")
    t0 = time.time()
    acc = {}
    Xs, Ys, names = [], [], None
    thr = max(1, (os.cpu_count() or 4) // args.workers)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(hn.district_worker, c, cfg): c for c in codes}
        k = 0
        while pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for f in done:
                code = pending.pop(f)
                try:
                    r = f.result()
                except Exception as e:                                       # noqa: BLE001
                    raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                del f
                k += 1
                for ver, d in (r.get("acf") or {}).items():
                    a = acc.setdefault(ver, {})
                    for lag, v in d.items():
                        a[lag] = a.get(lag, 0) + v
                if r.get("dump_X"):
                    Xs += r["dump_X"]
                    Ys += r["dump_y"]
                    names = names or r.get("names")
                if k % 20 == 0 or k == len(codes):
                    print(f"[read] {k}/{len(codes)} districts, {(time.time() - t0) / 60:.1f} min", flush=True)
    acf_part(acc, out)
    if not Xs:
        raise SystemExit("no training windows sampled")
    X = np.concatenate(Xs).astype(np.float64)
    Y = np.concatenate(Ys).astype(np.float64)
    if len(X) > args.max_rows:
        keep = np.random.default_rng(0).choice(len(X), args.max_rows, replace=False)
        X, Y = X[keep], Y[keep]
    feature_part(X[:, :-1], Y, [f"hist_lag{wp.LOOKBACK - i}" for i in range(wp.LOOKBACK)] + list(names), out)
    print(f"\n[done] {out} in {(time.time() - t0) / 60:.1f} min")


def acf_part(acc, out):
    rows = []
    for ver, d in acc.items():
        for lag, (n, sa, sb, saa, sbb, sab) in sorted(d.items()):
            if n < 2:
                continue
            cov = sab / n - (sa / n) * (sb / n)
            va, vb = saa / n - (sa / n) ** 2, sbb / n - (sb / n) ** 2
            rows.append(dict(version=ver, lag=lag, r=cov / np.sqrt(va * vb) if va > 0 and vb > 0 else np.nan, pairs=int(n)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "acf.csv"), index=False)
    pv = t.pivot(index="lag", columns="version", values="r")
    show = [1, 2, 3, 6, 7, 8, 14, 21, 28, 30, 60, 90, 120, 182, 364, 365]
    print("\n=== Autocorrelation of household daily consumption (pooled Pearson r, observed days) ===")
    print("level = kWh / customer mean - 1 (seasons included) | short = kWh / centred 31-day mean - 1 (seasons removed)")
    print(pv.reindex([s for s in show if s in pv.index]).map(lambda x: f"{x:+.3f}").to_string())
    acf_chart(pv, out)


def acf_chart(pv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, blue, grey = "#0b0b0b", "#52514e", "#e5e4df", "#2a78d6", "#8a8880"
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    for ver, col, lab in (("level", grey, "with seasons (kWh / own mean)"), ("short", blue, "seasons removed (kWh / 31-day mean)")):
        if ver in pv:
            s = pv[ver].dropna()
            ax.plot(s.index, s.values, color=col, linewidth=2 if ver == "short" else 1.6, label=lab)
    if "short" in pv:
        wk = [l for l in pv.index if l % 7 == 0 and l <= 120]
        ax.scatter(wk, pv.loc[wk, "short"], s=14, color=blue, zorder=3, label="weekly lags (7, 14, 21 ...)")
    ax.axhline(0, color=grid, linewidth=1)
    ax.set_xlim(0, 370)
    ax.set_xlabel("Lag (days)", color=muted, fontsize=9)
    ax.set_ylabel("Correlation", color=muted, fontsize=9)
    r1 = pv.loc[1, "short"] if "short" in pv and 1 in pv.index else np.nan
    r364 = pv.loc[364, "level"] if "level" in pv and 364 in pv.index else np.nan
    ax.set_title(f"Yesterday predicts today (r = {r1:.2f} after removing seasons); a year back still correlates "
                 f"(r = {r364:.2f})", loc="left", fontsize=11, color=ink, pad=22)
    ax.text(0, 1.02, "Household daily kWh, pooled over customers, observed days only; enriched dataset v1.4",
            transform=ax.transAxes, fontsize=8.5, color=muted)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(muted)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "acf.png"))
    plt.close(fig)


def feature_part(X, Y, names, out):
    names = list(names)
    sd = X.std(axis=0)
    const = sd < 1e-12
    if const.any():
        print(f"\n[features] {int(const.sum())} constant input(s) left out: {[n for n, c in zip(names, const) if c]}")
    X = X[:, ~const]
    names = [n for n, c in zip(names, const) if not c]
    R = np.corrcoef(X, rowvar=False)
    pd.DataFrame(R, index=names, columns=names).to_csv(os.path.join(out, "feature_corr.csv"))
    iu = np.triu_indices(len(names), 1)
    pairs = pd.DataFrame({"a": np.array(names)[iu[0]], "b": np.array(names)[iu[1]], "r": R[iu]})
    pairs["abs_r"] = pairs["r"].abs()
    pairs = pairs.sort_values("abs_r", ascending=False)
    pairs[pairs["abs_r"] >= 0.8].to_csv(os.path.join(out, "feature_corr_pairs.csv"), index=False)
    try:
        vif = np.diag(np.linalg.inv(R))
    except np.linalg.LinAlgError:
        vif = np.diag(np.linalg.pinv(R))
    vt = pd.DataFrame({"feature": names, "vif": vif}).sort_values("vif", ascending=False)
    vt.to_csv(os.path.join(out, "feature_vif.csv"), index=False)
    ev = np.linalg.eigvalsh(R)
    cond = float(np.sqrt(ev.max() / max(ev.min(), 1e-15)))
    print(f"\n=== Correlation between the model's inputs ({len(X):,} training windows, {len(names)} inputs) ===")
    print(f"  pairs with |r| >= 0.95: {(pairs['abs_r'] >= 0.95).sum():,} | >= 0.8: {(pairs['abs_r'] >= 0.8).sum():,} | "
          f"condition number {cond:,.0f} | inputs with VIF > 10: {(vt['vif'] > 10).sum()} | > 100: {(vt['vif'] > 100).sum()}")
    exo = pairs[~pairs["a"].str.startswith("hist_") & ~pairs["b"].str.startswith("hist_")]
    print("  most correlated weather / calendar / year-ago pairs:\n" + exo.head(15)[["a", "b", "r"]].round(3).to_string(index=False))
    hist = pairs[pairs["a"].str.startswith("hist_") & pairs["b"].str.startswith("hist_")]
    print(f"  history inputs: median |r| between two history days {hist['abs_r'].median():.2f}; adjacent days "
          f"{np.median([R[i, i + 1] for i in range(89) if names[i].startswith('hist_') and names[i + 1].startswith('hist_')]):.2f}")
    print("  highest VIF:\n" + vt.head(15).round(1).to_string(index=False))
    tc = {}
    for j, h in enumerate(HZ):
        y = Y[:, j]
        m = np.isfinite(y)
        yc = y[m] - y[m].mean()
        xc = X[m] - X[m].mean(axis=0)
        tc[f"r_t{h}"] = (xc * yc[:, None]).sum(0) / np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum())
    tt = pd.DataFrame(tc, index=names)
    tt.index.name = "feature"
    tt.to_csv(os.path.join(out, "feature_target_corr.csv"))
    for h in (1, 30):
        s = tt[f"r_t{h}"].abs().sort_values(ascending=False).head(10)
        print(f"  inputs most correlated with the target {h} day(s) ahead: " + ", ".join(f"{n} {tt.loc[n, f'r_t{h}']:+.2f}" for n in s.index))
    heat(R, names, out, int((pairs["abs_r"] >= 0.95).sum()), int((vt["vif"] > 10).sum()))


def heat(R, names, out, n95, n_vif):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ink, muted = "#0b0b0b", "#52514e"
    keep = [i for i, n in enumerate(names) if not n.startswith("hist_") or n in
            ("hist_lag1", "hist_lag2", "hist_lag7", "hist_lag14", "hist_lag30", "hist_lag60", "hist_lag90")]
    S = R[np.ix_(keep, keep)]
    lab = [names[i] for i in keep]
    cmap = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f7f6f2", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(11, 10), dpi=150)
    im = ax.imshow(S, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=4.5, color=muted)
    ax.set_yticks(range(len(lab)), lab, fontsize=4.5, color=muted)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Pearson correlation", color=muted, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=muted)
    cb.outline.set_visible(False)
    ax.set_title(f"Correlation between model inputs: {n95} pairs above |r| = 0.95, {n_vif} inputs with VIF > 10",
                 loc="left", fontsize=11, color=ink, pad=16)
    ax.text(0, 1.004, "Training windows as fed to nlinear_x; 7 of the 90 history days shown; orange = negative, blue = positive",
            transform=ax.transAxes, fontsize=7.5, color=muted)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feature_corr_heatmap.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
