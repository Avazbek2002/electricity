#!/usr/bin/env python3
"""
NLinear-style linear forecasters for the transformer panel, scored against the direct CatBoost model on the SAME
transformer-origins and target days.

NLinear (Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023) is one linear layer from the
last L days to the next H days, applied after subtracting the series' last value and adding it back afterwards. Here:

  * L = 90, H = 60 (as the direct model); the weights are shared by all transformers (one model per fold).
  * Normalisation: divide by the mean of the last --norm-days days (default 7) and subtract 1, instead of subtracting the
    last value: one day is noisy, and a ratio puts transformers of every size on one scale.
  * Fitted in closed form (weighted ridge regression); the weight of a window is its kWh level (n_households x level), so
    the squared loss puts the same emphasis on big transformers as the pooled WAPE does. The ridge penalty is chosen on
    the fold's validation block; the final weights are refitted on all of the fold's training windows.

Variants (all forecast the calendar-adjusted series, like the direct model; the fold's weekday / holiday factors are
multiplied back before scoring):

  nlinear            history only:            y/level - 1  =  W . (x/level - 1) + b
  nlinear_x          + exogenous drivers:     future degree days (blocks and the 10-day forecast), their change against
                                              the 30-day reference, the transformer's own sensitivity x that change,
                                              day of year, input-window weather, year-ago ratios, size
  thermal_nlinear    history -> residual:     (y - thermal) / level  =  W . (x/level - 1) + b ;  forecast = thermal + level x ...
  thermal_nlinear_x  residual + exogenous

Inputs: the work dir and the forecast dir of a finished `transformer_forecast.py run --stages direct[,compare]`
(it reads direct_test.npz and direct_info.json there, so the folds, validation blocks and test rows are the direct run's).
Use the SAME panel options as that run (--calendar, --gap-fill, --min-households, --normal-mode, --max-direct-windows,
--origin-stride, --forecast-sigma-scale): the script checks that its thermal baseline matches the direct run's.

  python scripts/transformer/transformer_nlinear.py --work-dir ~/work/tf_enriched --forecast-dir ~/work/tf_enriched/forecast_folds

Outputs (<forecast-dir>/nlinear/): nlinear_test.npz (forecasts), nlinear_info.json (penalty search per fold and variant),
weights_<variant>_<fold>.npz (the fitted matrices), results/metrics_by_fold.csv, metrics.csv, paired_bootstrap.csv,
history_weights.csv (how much weight the history model puts on the last week vs further back), run_log.txt.
No customer-level data is written.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core")))  # the shared modules
import transformer_forecast as tf                                           # noqa: E402

VARIANTS = ("nlinear", "nlinear_x", "thermal_nlinear", "thermal_nlinear_x")
LAMBDAS = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)               # relative to the mean diagonal of X'WX
EXO_PREFIX = ("origin_doy_", "temp_mean_7d", "temp_mean_30d", "hdd_mean_7d", "hdd_mean_30d", "cdd_mean_7d", "cdd_mean_30d",
              "temp_anom_", "fut_temp_", "fut_hdd_", "fut_cdd_", "d_hdd_", "d_cdd_", "prev_anom_", "fc_anom_",
              "fc_hddmin_d", "fc_cddmax_d", "log_n_households")


# =============================================================================
# Design matrices
# =============================================================================
def norm_level(kin, days):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(kin[:, -days:], axis=1)


def history_block(kin, level):
    """(rows, L): x / level - 1, with missing input days (should not happen: inputs are complete) set to 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        x = kin / level[:, None] - 1.0
    return np.nan_to_num(np.clip(x, -1.0, 5.0), nan=0.0, posinf=0.0, neginf=0.0)


def exo_block(wp, feats, ly_feats, level):
    """(rows, k) exogenous drivers (unscaled) and their names."""
    cols, names = [], []
    for c in feats.columns:
        if c.startswith(EXO_PREFIX):
            cols.append(feats[c].to_numpy(dtype="float64"))
            names.append(c)
    # the transformer's own heating / cooling sensitivity x the change in degree days, in ratio units: what the thermal
    # baseline does, but with weights the regression may scale per horizon
    bh = feats["beta_h"].to_numpy(dtype="float64")
    bc = feats["beta_c"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        for a, b in wp.BLOCKS:
            cols.append(bh * feats[f"d_hdd_{a}_{b}"].to_numpy() / level)
            names.append(f"sens_h_x_dhdd_{a}_{b}")
            cols.append(bc * feats[f"d_cdd_{a}_{b}"].to_numpy() / level)
            names.append(f"sens_c_x_dcdd_{a}_{b}")
    have = np.isfinite(ly_feats["ly_ratio_1_7"])
    for k, v in ly_feats.items():
        cols.append(np.where(np.isfinite(v), v - 1.0, 0.0))
        names.append(k)
    cols.append(have.astype(float))
    names.append("ly_available")
    X = np.column_stack(cols)
    X[~np.isfinite(X)] = np.nan
    return X, names


class Scaler:
    """z-score on the fold's training rows; NaN -> 0 (= the training mean) after scaling."""

    def __init__(self, X):
        self.mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)

    def __call__(self, X):
        return np.nan_to_num((X - self.mu) / self.sd, nan=0.0)


def design(variant, hist, exo):
    parts = [hist]
    if variant.endswith("_x"):
        parts.append(exo)
    parts.append(np.ones((len(hist), 1)))
    return np.hstack(parts)


# =============================================================================
# Weighted ridge in closed form
# =============================================================================
def gram(X, Y, w, chunk=50000):
    p = X.shape[1]
    G, R = np.zeros((p, p)), np.zeros((p, Y.shape[1]))
    for a in range(0, len(X), chunk):
        x, y, ww = X[a:a + chunk].astype("float64"), Y[a:a + chunk].astype("float64"), w[a:a + chunk, None]
        G += x.T @ (x * ww)
        R += (x * ww).T @ y
    return G, R


def solve(G, R, lam):
    """Penalty lam x mean diagonal on every coefficient except the intercept (last column)."""
    p = G.shape[0]
    scale = float(np.mean(np.diag(G)[:-1])) if p > 1 else 1.0
    P = np.eye(p) * lam * scale
    P[-1, -1] = 0.0
    return np.linalg.solve(G + P, R)


def to_kwh(variant, Z, level, base):
    """Model output (ratio units) -> kWh per household per day (calendar-adjusted)."""
    if variant.startswith("thermal"):
        return np.maximum(base + level[:, None] * Z, 0.0)
    return np.maximum(level[:, None] * (1.0 + Z), 0.0)


def target(variant, ktg, level, base):
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (ktg - base) / level[:, None] if variant.startswith("thermal") else ktg / level[:, None] - 1.0
    return np.clip(np.nan_to_num(Z, nan=0.0), -1.0, 4.0)


def wape(P, A):
    v = np.isfinite(P) & np.isfinite(A) & (A > 0)
    return float(np.abs(P[v] - A[v]).sum() / A[v].sum()) if v.any() else np.nan


# =============================================================================
# One fold
# =============================================================================
def fold_rows(wp, args, panel, clim, fc, fold, d, fi):
    """Training windows (as the direct model's fold) + the direct run's test rows of this fold, featurised together."""
    L, H = wp.LOOKBACK, wp.HORIZON
    c, vs, ve, te = fold["train_cutoff"], fold["val_start"], fold.get("val_end"), fold.get("test_end")
    tf.apply_calendar(wp, panel, args, c)
    O = tf.origin_grid(wp, panel, args)
    r_tr, r_o, tgt_ok = tf.windows(wp, panel, O)
    pdf_all = pd.DataFrame({"meter_id": panel.ids[r_tr].astype(str), "consumer_id": np.nan,
                            "origin_date": panel.d0 + pd.to_timedelta(r_o, unit="D"),
                            "coato_code": panel.coato_of[r_tr], "housing_type_id": panel.housing[r_tr]})
    kin_all = tf.causal_input(panel, r_tr, r_o, L)
    segments = pd.Series("unclassified", index=pd.Index(panel.ids.astype(str)), name="segment")
    minm = max(10, int(0.2 * len(panel.ids)))
    ph = wp.estimate_segment_beta_priors(pdf_all, kin_all, clim, c, segments, min_meters=minm, kind="mean")
    pm = wp.estimate_segment_beta_priors(pdf_all, kin_all, clim, c, segments, min_meters=minm, kind="minmax") \
        if wp.USE_MINMAX else ({}, (None, None))
    priors = dict(mean=ph, mm=pm)

    meta0 = pdf_all[["origin_date"]]
    train, fit, val, _ = wp.make_masks(meta0, c, vs, te)
    if ve:
        fit, val = tf.blocked_validation(meta0["origin_date"], vs, ve, c, H)
    reg = wp.regime_keep_mask(pdf_all)
    train, fit, val = train & tgt_ok & reg, fit & tgt_ok & reg, val & tgt_ok & reg
    tr_idx = np.flatnonzero(train)

    # the direct run's test rows of this fold
    pos = {t: k for k, t in enumerate(panel.ids.astype(str))}
    sel = np.flatnonzero(d["fold"] == fi) if "fold" in d.files else np.arange(len(d["origin"]))
    known = np.array([t in pos for t in d["transformer_id"][sel]])
    if not known.all():
        print(f"[nlinear] WARNING: {int((~known).sum())} test rows are on transformers not in this panel (different panel options?)")
    sel = sel[known]
    t_tr = np.array([pos[t] for t in d["transformer_id"][sel]], dtype=np.int64)
    t_o = d["origin"][sel].astype(np.int64)

    R_tr = np.r_[r_tr[tr_idx], t_tr]
    R_o = np.r_[r_o[tr_idx], t_o]
    n_train = len(tr_idx)
    pdf = pd.DataFrame({"meter_id": panel.ids[R_tr].astype(str), "consumer_id": np.nan,
                        "origin_date": panel.d0 + pd.to_timedelta(R_o, unit="D"),
                        "coato_code": panel.coato_of[R_tr], "housing_type_id": panel.housing[R_tr]})
    kin = np.vstack([kin_all[tr_idx], tf.causal_input(panel, t_tr, t_o, L)])
    del kin_all, pdf_all
    meta, feats, base = tf.featurize(wp, pdf, kin, clim, fc, priors, segments, args.normal_mode, args.chunk)
    feats["log_n_households"] = np.log(panel.n_hh[R_tr].astype(float))
    ly_feats, _ = tf.year_ago(wp, panel.filled, R_tr, R_o, wp.BLOCKS)
    ktg = tf.gather(panel.filled, R_tr[:n_train], R_o[:n_train], 1, H)
    is_fit, is_val = fit[tr_idx], val[tr_idx]
    print(f"[nlinear] fold {fold['fold']}: train windows {n_train:,} (fit {int(is_fit.sum()):,}, val {int(is_val.sum()):,}) | "
          f"test rows {len(sel):,} | origin stride {args.origin_stride}")
    return SimpleNamespace(sel=sel, n_train=n_train, is_fit=is_fit, is_val=is_val, kin=kin, feats=feats, ly=ly_feats,
                           base=base, ktg=ktg, n_hh=panel.n_hh[R_tr].astype(float), t_tr=t_tr, t_o=t_o)


def run_fold(wp, args, rows, fold, d, variants, info, out_dir):
    H = wp.HORIZON
    n = rows.n_train
    level = norm_level(rows.kin, args.norm_days)
    ok_level = np.isfinite(level) & (level > args.min_level)
    lvl = np.where(ok_level, level, np.nan)
    hist = history_block(rows.kin, lvl)
    exo_raw, exo_names = exo_block(wp, rows.feats, rows.ly, lvl)
    scaler = Scaler(exo_raw[:n][ok_level[:n]])
    exo = scaler(exo_raw)
    del exo_raw

    # sanity check: this thermal baseline vs the direct run's (same panel options -> the same numbers)
    th_dir = d["thermal"][rows.sel].astype(float) / d["n_hh"][rows.sel][:, None]
    th_me = rows.base[n:]
    v = np.isfinite(th_dir) & (th_dir > 0)
    rel = np.abs(th_me[v] - th_dir[v]) / th_dir[v]
    med_rel = float(np.median(rel)) if v.any() else np.nan
    print(f"[check] thermal baseline vs the direct run's on the test rows: median relative difference {med_rel:.2e}"
          + ("" if med_rel < 1e-3 else "  <-- WARNING: not the same panel / options as the direct run "
             "(pass the same --calendar, --gap-fill, --min-households, --normal-mode, --max-direct-windows, --origin-stride)"))

    tr_ok = ok_level[:n]
    kwh_w = rows.n_hh[:n] * np.where(tr_ok, level[:n], 0.0)
    w = {"kwh": kwh_w, "sqrt": np.sqrt(np.maximum(kwh_w, 0)), "none": tr_ok.astype(float)}[args.weight]
    w = np.where(tr_ok, w, 0.0)
    w = w / w[w > 0].mean()
    A_val = rows.ktg * rows.n_hh[:n, None]

    preds, finfo = {}, dict(fold=fold["fold"], thermal_check_median_rel_diff=med_rel, exo_features=exo_names,
                            train_windows=int(n), level_too_low=int((~ok_level[:n]).sum()), variants={})
    for var in variants:
        X = design(var, hist, exo)
        Y = target(var, rows.ktg, np.where(tr_ok, level[:n], 1.0), rows.base[:n])
        fm, vm = rows.is_fit & tr_ok, rows.is_val & tr_ok
        G, R = gram(X[:n][fm], Y[fm], w[fm])
        search = []
        for lam in LAMBDAS:
            Wt = solve(G, R, lam)
            P = to_kwh(var, X[:n][vm] @ Wt, level[:n][vm], rows.base[:n][vm]) * rows.n_hh[:n][vm, None]
            search.append(dict(lam=lam, val_wape=wape(P, A_val[vm])))
        best = min(search, key=lambda s: s["val_wape"] if np.isfinite(s["val_wape"]) else np.inf)
        trm = tr_ok
        G, R = gram(X[:n][trm], Y[trm], w[trm])
        Wt = solve(G, R, best["lam"])
        Zt = X[n:] @ Wt
        lt = np.where(ok_level[n:], level[n:], 0.0)
        P = to_kwh(var, Zt, lt, rows.base[n:])
        fallback = rows.base[n:] if var.startswith("thermal") else np.repeat(np.nan_to_num(level[n:])[:, None], H, 1)
        P = np.where(ok_level[n:, None], P, fallback)                    # near-zero level: thermal / persistence
        preds[var] = (P * rows.n_hh[n:, None]).astype("float32")
        np.savez_compressed(os.path.join(out_dir, f"weights_{var}_{fold['fold']}.npz"), W=Wt.astype("float32"),
                            feature_names=np.array([f"hist_lag{90 - i}" for i in range(hist.shape[1])]
                                                   + (exo_names if var.endswith("_x") else []) + ["intercept"]),
                            exo_mu=scaler.mu, exo_sd=scaler.sd, norm_days=args.norm_days)
        finfo["variants"][var] = dict(chosen_lambda=best["lam"], search=search)
        print(f"[nlinear] fold {fold['fold']} {var:18s} penalty {best['lam']:g} | validation WAPE (all 60 days) "
              + "  ".join(f"{s['lam']:g}:{s['val_wape']:.1%}" for s in search))
    info.append(finfo)
    return preds


def history_weight_table(out_dir, variants, folds):
    """How the history model spreads its weight over the input days, for a few horizons."""
    rows = []
    for var in [v for v in variants if not v.endswith("_x")]:
        for f in folds:
            p = os.path.join(out_dir, f"weights_{var}_{f}.npz")
            if not os.path.exists(p):
                continue
            W = np.load(p)["W"]                                         # (91, 60): 90 history days (oldest first) + intercept
            for h in (1, 7, 14, 30, 60):
                col = W[:90, h - 1]
                # (the last 7 days' weights always sum to ~0: those days are what the input is normalised by)
                rows.append(dict(variant=var, fold=f, horizon=h, origin_day=col[-1],
                                 days_8_30=col[-30:-7].sum(), days_31_90=col[:-30].sum(),
                                 lags_7_14_21_28=sum(col[-k] for k in (7, 14, 21, 28)), intercept=W[90, h - 1]))
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(os.path.join(out_dir, "history_weights.csv"), index=False)
        print("\n=== Where the history models put their weight (sums of coefficients on x/level - 1; lag 1 = the origin day; "
              "weight on days 31-90 = how much the model leans on the older level) ===")
        print(t.round(3).to_string(index=False))


# =============================================================================
# Scoring
# =============================================================================
def compare(wp, args, panel, d, nl, variants, out_dir):
    horizons = [int(h) for h in args.horizons.split(",")]
    hi = np.asarray(horizons) - 1
    kd = pd.DataFrame({"transformer_id": d["transformer_id"], "origin": d["origin"], "i": np.arange(len(d["origin"]))})
    kn = pd.DataFrame({"transformer_id": nl["transformer_id"], "origin": nl["origin"], "j": np.arange(len(nl["origin"]))})
    m = kd.merge(kn, on=["transformer_id", "origin"])
    i, j = m["i"].to_numpy(), m["j"].to_numpy()
    tid, org = m["transformer_id"].to_numpy().astype(str), m["origin"].to_numpy()
    n_hh = d["n_hh"][i].astype(float)
    pos = {t: k for k, t in enumerate(panel.ids.astype(str))}
    k = np.asarray([pos[t] for t in tid])
    day_idx = org[:, None] + np.asarray(horizons)[None, :]
    A = panel.inten[k[:, None], day_idx] * n_hh[:, None]
    calT = d["cal"][i][:, hi].astype(float) if "cal" in d.files else np.ones(day_idx.shape)
    fold = d["fold"][i] if "fold" in d.files else np.zeros(len(i), dtype=int)
    fold_names = list(d["fold_names"]) if "fold_names" in d.files else ["f1"]
    models = {"direct": d["pred"][i][:, hi].astype(float) * calT}
    if "snaive" in d.files:
        models["seasonal_naive"] = d["snaive"][i][:, hi].astype(float) * calT
    models["thermal_baseline"] = d["thermal"][i][:, hi].astype(float) * calT
    models["persistence7"] = np.repeat(d["pers7"][i][:, None], len(horizons), 1).astype(float) * calT
    models["persistence30"] = np.repeat(d["pers30"][i][:, None], len(horizons), 1).astype(float) * calT
    for v in variants:
        models[v] = nl[v][j][:, hi].astype(float) * calT
    print(f"\n[compare] {len(m):,} transformer-origins in common ({len(set(tid))} transformers)")

    rows = []
    for fi, fname in enumerate(fold_names):
        fm = fold == fi
        if not fm.any():
            continue
        dts = panel.d0 + pd.to_timedelta(day_idx[fm, 0], unit="D")
        for name, P in models.items():
            for hx, h in enumerate(horizons):
                w, b, nn = tf.wape_bias(P[fm, hx], A[fm, hx])
                rows.append(dict(fold=fname, targets=f"{dts.min():%Y-%m}..{(dts.max() + pd.Timedelta(days=horizons[-1])):%Y-%m}",
                                 model=name, horizon=h, wape=w, bias=b, n=nn))
    ft = pd.DataFrame(rows)
    res = os.path.join(out_dir, "results")
    os.makedirs(res, exist_ok=True)
    ft.to_csv(os.path.join(res, "metrics_by_fold.csv"), index=False)
    print("\n=== Pooled WAPE by fold and horizon (same transformer-origins for every model) ===")
    for (fname, tgt), g in ft.groupby(["fold", "targets"], sort=False):
        t = g.pivot(index="model", columns="horizon", values="wape").reindex(list(models))
        print(f"-- fold {fname} (target days {tgt})\n" + t.map(lambda x: f"{x:6.1%}").to_string())
    print("\n=== Bias by fold (forecast - actual) / actual ===")
    for (fname, tgt), g in ft.groupby(["fold", "targets"], sort=False):
        t = g.pivot(index="model", columns="horizon", values="bias").reindex(list(models))
        print(f"-- fold {fname}\n" + t.map(lambda x: f"{x:+6.1%}").to_string())

    size = pd.Series(n_hh, index=tid).groupby(level=0).first()
    codes = pd.qcut(size.rank(method="first"), args.size_bands, labels=False)
    names = {c: f"Q{c + 1}: {int(size[codes == c].min())}-{int(size[codes == c].max())} hh" for c in range(args.size_bands)}
    band = pd.Series(tid).map(codes).map(names).to_numpy()
    tab = tf.score_table(models, A, tid, day_idx, band, horizons, [names[c] for c in range(args.size_bands)])
    tab.to_csv(os.path.join(res, "metrics.csv"), index=False)
    t = tab[tab["band"] == "all"]
    print("\n=== All folds together: pooled WAPE | system WAPE | median per-transformer WAPE ===")
    for col in ("wape", "sys_wape", "median_transformer_wape"):
        p = t.pivot(index="model", columns="horizon", values=col).reindex(list(models))
        print(f"-- {col}\n" + p.map(lambda x: f"{x:6.1%}").to_string())

    boot = []
    for v in variants:
        for ref in ("direct", "seasonal_naive"):
            if ref not in models:
                continue
            for hx, h in enumerate(horizons):
                for fi, fname in [(None, "all")] + list(enumerate(fold_names)):
                    fm = np.ones(len(fold), bool) if fi is None else fold == fi
                    if not fm.any():
                        continue
                    est, lo, hi_ = tf.paired_bootstrap(models[v][fm, hx], models[ref][fm, hx], A[fm, hx], tid[fm])
                    boot.append(dict(fold=fname, horizon=h, a=v, b=ref, wape_a_minus_b=est, p05=lo, p95=hi_,
                                     verdict="a better" if hi_ < 0 else "b better" if lo > 0 else "no clear difference"))
    boot = pd.DataFrame(boot)
    boot.to_csv(os.path.join(res, "paired_bootstrap.csv"), index=False)
    if len(boot):
        print("\n=== Paired bootstrap over transformers: WAPE(a) - WAPE(b), 90% interval ===")
        b2 = boot.assign(txt=[f"{r.wape_a_minus_b:+.1%} [{r.p05:+.1%},{r.p95:+.1%}]" for r in boot.itertuples()])
        print(b2.pivot_table(index=["fold", "a", "b"], columns="horizon", values="txt", aggfunc="first").to_string())
    return tab


# =============================================================================
def load_folds(forecast_dir):
    with open(os.path.join(forecast_dir, "direct_info.json")) as fh:
        di = json.load(fh)
    folds = di.get("folds") or [dict(fold="f1", train_cutoff=di["train_cutoff"], val_start=di["val_start"],
                                     val_end=di.get("val_end"), test_end=None)]
    return di, folds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--forecast-dir", default=None, help="out dir of the direct run (default <work-dir>/forecast)")
    p.add_argument("--out-dir", default=None, help="default <forecast-dir>/nlinear")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--norm-days", type=int, default=7, help="normalise by the mean of the last N input days")
    p.add_argument("--min-level", type=float, default=0.05,
                   help="windows whose normalisation level (kWh per household per day) is below this are not trained on; "
                        "their forecast falls back to persistence / the thermal baseline")
    p.add_argument("--weight", choices=["kwh", "sqrt", "none"], default="kwh",
                   help="regression weight of a window: its kWh level (matches pooled WAPE), its square root, or equal")
    # the panel options of the direct run (keep them identical)
    p.add_argument("--calendar", choices=["none", "dow", "dow+holidays"], default="dow+holidays")
    p.add_argument("--holidays-csv", default=None)
    p.add_argument("--gap-fill", type=int, default=3)
    p.add_argument("--min-households", type=int, default=5)
    p.add_argument("--normal-mode", choices=["loyo", "all"], default="loyo")
    p.add_argument("--max-direct-windows", type=int, default=300000)
    p.add_argument("--origin-stride", type=int, default=1)
    p.add_argument("--forecast-sigma-scale", type=float, default=1.0)
    p.add_argument("--horizons", default=",".join(map(str, tf.HORIZONS)))
    p.add_argument("--size-bands", type=int, default=4)
    p.add_argument("--chunk", type=int, default=100000)
    args = p.parse_args(argv)
    args.work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    args.forecast_dir = os.path.abspath(os.path.expanduser(args.forecast_dir or os.path.join(args.work_dir, "forecast")))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or os.path.join(args.forecast_dir, "nlinear")))
    os.makedirs(out_dir, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - set(VARIANTS)
    if bad:
        raise SystemExit(f"unknown variant(s) {sorted(bad)}; choose from {', '.join(VARIANTS)}")

    with tf.logging_to(os.path.join(out_dir, "run_log.txt")):
        wp = tf.wp_module()
        dpath = os.path.join(args.forecast_dir, "direct_test.npz")
        if not os.path.exists(dpath):
            raise SystemExit(f"{dpath} not found: run  transformer_forecast.py run --stages direct  first")
        d = np.load(dpath, allow_pickle=False)
        di, folds = load_folds(args.forecast_dir)
        wp.INCLUDE_OLD = di.get("include_old", wp.INCLUDE_OLD)
        wp.OLD_REGIME_FLAG = bool(di.get("old_regime_flag", wp.OLD_REGIME_FLAG))
        print(f"[config] direct run {args.forecast_dir}: {len(folds)} fold(s) "
              + " | ".join(f"{f['fold']}: cutoff {f['train_cutoff']}, val ({f['val_start']}, {f.get('val_end')}], test to "
                           f"{f.get('test_end') or 'end'}" for f in folds)
              + f" | old regime '{wp.INCLUDE_OLD}' | variants {','.join(variants)} | norm {args.norm_days} days | weight {args.weight}")

        with tf.stage("panel + climate + forecasts"):
            info, hh_meta, _ = tf.load_world(args.work_dir)
            panel = tf.load_clean_panel(wp, args, hh_meta)
            clim, _ = tf.load_climate(wp, args, None)
            panel = tf.keep_known_coatos(panel, clim)
            tf.apply_calendar(wp, panel, args, folds[0]["train_cutoff"])
            r_all, _, _ = tf.windows(wp, panel, tf.origin_grid(wp, panel, SimpleNamespace(origin_stride=1)))
            need = int(np.ceil(len(r_all) / max(args.max_direct_windows, 1))) if args.max_direct_windows else 1
            args.origin_stride = max(args.origin_stride, need)
            print(f"[windows] origin stride {args.origin_stride} (as the direct run: {len(r_all):,} complete input windows, "
                  f"cap {args.max_direct_windows:,})")
            O = tf.origin_grid(wp, panel, args)
            if args.forecast_sigma_scale != 1.0 and wp.FORECAST_MODE == "simulated":
                sc = max(float(args.forecast_sigma_scale), 0.01)
                wp.FC_SIGMA_DAY1, wp.FC_SIGMA_DAY10 = wp.FC_SIGMA_DAY1 * sc, wp.FC_SIGMA_DAY10 * sc
            fc = tf.forecast_grids(wp, clim, sorted(set(panel.coato_of)), panel, O, args)

        out = {v: [] for v in variants}
        keys = dict(transformer_id=[], origin=[], fold=[])
        finfo = []
        for fi, fold in enumerate(folds):
            fold.setdefault("fold", f"f{fi + 1}")
            with tf.stage(f"nlinear fold {fold['fold']}"):
                rows = fold_rows(wp, args, panel, clim, fc, fold, d, fi)
                preds = run_fold(wp, args, rows, fold, d, variants, finfo, out_dir)
                for v in variants:
                    out[v].append(preds[v])
                keys["transformer_id"].append(d["transformer_id"][rows.sel])
                keys["origin"].append(d["origin"][rows.sel])
                keys["fold"].append(np.full(len(rows.sel), fi, dtype="int16"))
                del rows
        nl = {k: np.concatenate(v) for k, v in keys.items()}
        nl.update({v: np.concatenate(out[v]) for v in variants})
        np.savez_compressed(os.path.join(out_dir, "nlinear_test.npz"), **nl)
        with open(os.path.join(out_dir, "nlinear_info.json"), "w") as fh:
            json.dump(dict(args={k: v for k, v in vars(args).items()}, folds=folds, fits=finfo), fh, indent=2, default=str)
        history_weight_table(out_dir, variants, [f["fold"] for f in folds])
        with tf.stage("compare"):
            compare(wp, args, panel, d, nl, variants, out_dir)
        print(f"\n[nlinear] outputs in {out_dir}")


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
    from mlflow_utils import run_script, table_metrics
    run_script(main, "electricity/transformer-forecast", stage="transformer_nlinear",
               metrics_from=lambda d: table_metrics(os.path.join(d, "results", "metrics.csv"), ["model", "horizon"],
                                                    ["wape", "sys_wape", "bias"], where={"band": "all"}))
