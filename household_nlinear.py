#!/usr/bin/env python3
"""
NLinear-style linear forecasters for HOUSEHOLDS, trained on EVERY window of every customer of the national enriched dataset.

The same four variants as transformer_nlinear.py (one linear layer from the last 90 days to the next 60, shared by all
customers, fitted in closed form by weighted ridge regression):

  nlinear            history only:             y/level - 1        = W . (x/level - 1) + b
  nlinear_x          + exogenous drivers        (weather blocks, degree-day changes, own sensitivity x change, 10-day
                                                 forecast, recent weather, last year, day of year)
  thermal_nlinear    history -> residual:      (y - thermal)/level = W . (x/level - 1) + b
  thermal_nlinear_x  residual + exogenous

Why every window fits: a linear model only needs X'WX and X'WY. Each district worker builds its windows' features, adds them
to those running totals and throws the windows away, so memory does not grow with the data. Nothing is sampled for
training. Two small samples are kept: validation windows (to choose the ridge penalty) and test windows (to score).

Data and rules (as whole_pipeline_v8.py's household model, so the two compare):
  * source: the national enriched dataset v1.4 (whole_pipeline_v8.ENRICHED_ROOT), every finished district, every customer
    (or --buckets); only complete_observed days have a value (HH-GAP-001-A1)
  * windows: origins every --stride days per customer (staggered by a stable hash); input >= 60 of 90 days observed and
    >= 20 of the last 30; target >= 30 of 60 observed. Trained on when >= 54 target days are observed: each target day
    is fitted only where it was observed (a missing day is left out of that day's regression; nothing is invented)
  * train = origins <= --cutoff; validation = origins in (--val-start, --val-end], fit = training origins >= 60 days away
    from it on both sides; test = origins >= cutoff + 60 days. Old-regime and excluded-range rules as the household model.
  * inputs: a missing input day counts as "at the level" (0 after normalisation). Level = mean of the last --norm-days
    observed input days (>= 3 of them), else of the last 30.
  * features: whole_pipeline_v8.build_features (same weather, degree days, normals, simulated 10-day forecast,
    sensitivity fit and thermal baseline as the household CatBoost model) with one pooled prior for the sensitivity
    (--prior-*, from the 1% CatBoost run), plus year-ago ratios from the customer's own history.
  * test sample: windows whose stable hash u < --test-sample. The household CatBoost all-customer run keeps u <
    ENRICHED_TEST_SAMPLE with the same hash, so with the same rules its test windows are the subset u < 0.01 here:
    the script reports that subset separately ("catboost_test_set").

  python household_nlinear.py --out-dir ~/work/hh_nlinear --workers 4 --cache-dir ~/work/hh_nlinear_cache

Outputs (<out-dir>): weights_<variant>.npz (raw-space weights, feature names, standardisation), nlinear_info.json (data
counts, penalty search), metrics.csv (MAE, pooled WAPE, system WAPE, bias by model, horizon and test set), run_log.txt.
No customer keys are written, except in --cache-dir (per-district partial results; keep it in the authorised environment).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VARIANTS = ("nlinear", "nlinear_x", "thermal_nlinear", "thermal_nlinear_x")
LAMBDAS = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
EXO_PREFIX = ("origin_doy_", "temp_mean_7d", "temp_mean_30d", "hdd_mean_7d", "hdd_mean_30d", "cdd_mean_7d", "cdd_mean_30d",
              "temp_anom_", "fut_temp_", "fut_hdd_", "fut_cdd_", "d_hdd_", "d_cdd_", "prev_anom_", "fc_anom_",
              "fc_hddmin_d", "fc_cddmax_d")
HORIZONS = (1, 7, 14, 30, 60)
LY_DAYS = 364


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


# =============================================================================
# Per-window inputs
# =============================================================================
def take(M, r, start, length):
    """M (n, T) -> (rows, length) = M[r, start : start+length], NaN outside the series."""
    T = M.shape[1]
    idx = np.asarray(start, dtype=np.int64)[:, None] + np.arange(length)[None, :]
    ok = (idx >= 0) & (idx < T)
    rows = np.broadcast_to(np.asarray(r)[:, None], idx.shape)
    out = np.full(idx.shape, np.nan, dtype=np.float64)
    out[ok] = M[rows[ok], idx[ok]]
    return out


def year_ago(M, r, o, blocks, min_ref=15, clip=(0.2, 5.0)):
    """Last year's blocks relative to last year's 30 days before the origin (364 days back: weekdays line up).
    Every day used is <= origin - 304. NaN where last year has too few observed days."""
    H = max(b for _, b in blocks)
    o_ly = np.asarray(o) - LY_DAYS
    ref_w = take(M, r, o_ly - 29, 30)
    with np.errstate(invalid="ignore", divide="ignore"), _warn_off():
        ref = np.where(np.isfinite(ref_w).sum(1) >= min_ref, np.nanmean(ref_w, axis=1), np.nan)
        ref = np.where(ref > 0, ref, np.nan)
        fut = take(M, r, o_ly + 1, H)
        f = {}
        for a, b in blocks:
            blk = fut[:, a - 1:b]
            ok = np.isfinite(blk).sum(1) >= max(1, (b - a + 1) // 3)
            f[f"ly_ratio_{a}_{b}"] = np.where(ok, np.clip(np.nanmean(blk, axis=1) / ref, *clip), np.nan)
        last7 = ref_w[:, -7:]
        f["ly_ratio_m7_0"] = np.where(np.isfinite(last7).sum(1) >= 3, np.clip(np.nanmean(last7, axis=1) / ref, *clip), np.nan)
    return f


@contextlib.contextmanager
def _warn_off():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def level_of(kin, days):
    with _warn_off():
        n7 = np.isfinite(kin[:, -days:]).sum(1)
        l7 = np.nanmean(kin[:, -days:], axis=1)
        l30 = np.nanmean(kin[:, -30:], axis=1)
    return np.where(n7 >= 3, l7, l30)


def design(wp, feats, ly, kin, level, n_hist):
    """-> X (rows, 90 + k + 1) float64, exogenous names. History: clip(x/level - 1, -1, 5), missing day -> 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        hist = np.clip(kin / level[:, None] - 1.0, -1.0, 5.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    cols, names = [], []
    for c in feats.columns:
        if c.startswith(EXO_PREFIX):
            cols.append(feats[c].to_numpy(dtype="float64"))
            names.append(c)
    bh, bc = feats["beta_h"].to_numpy(dtype="float64"), feats["beta_c"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        for a, b in wp.BLOCKS:
            cols.append(bh * feats[f"d_hdd_{a}_{b}"].to_numpy() / level)
            names.append(f"sens_h_x_dhdd_{a}_{b}")
            cols.append(bc * feats[f"d_cdd_{a}_{b}"].to_numpy() / level)
            names.append(f"sens_c_x_dcdd_{a}_{b}")
    have = np.isfinite(ly["ly_ratio_1_7"])
    for k, v in ly.items():
        cols.append(np.where(np.isfinite(v), v - 1.0, 0.0))
        names.append(k)
    cols.append(have.astype(float))
    names.append("ly_available")
    exo = np.column_stack(cols)
    exo = np.nan_to_num(exo, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.hstack([hist, exo, np.ones((len(hist), 1))])
    assert hist.shape[1] == n_hist
    return X, names


# =============================================================================
# One district (worker process)
# =============================================================================
def district_worker(code, cfg):
    t0 = time.time()
    import whole_pipeline_v8 as wp
    for k, v in cfg["wp"].items():                                   # the household model's settings, as run
        setattr(wp, k, v)
    L, H, S = wp.LOOKBACK, wp.HORIZON, cfg["stride"]
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T = (d1 - d0).days + 1
    keys, M, wt, house = wp._read_enriched_district(code, cfg)
    res = dict(code=str(code), customers=0, candidates=0, passed=0, train=0, fit=0, val=0, test=0, seconds=0.0)
    n = len(keys)
    if not n:
        res["seconds"] = time.time() - t0
        return res
    res["customers"] = int(np.isfinite(M).any(axis=1).sum())
    dates = d0 + pd.to_timedelta(np.arange(T), unit="D")
    have = np.isfinite(wt).any(axis=0)
    wdf = pd.DataFrame({"coato_id": str(code), "d": dates[have], "t": wt[0, have], "tmin": wt[1, have], "tmax": wt[2, have]})
    clim = wp.build_climate(wdf)

    # one simulated 10-day forecast per origin day of this district, shared by all its customers
    od = np.arange(L - 1, T - H)
    u_fc = pd.DataFrame({"coato_code": str(code), "origin_date": d0 + pd.to_timedelta(od, unit="D")})
    seed = int(cfg["seed"]) + zlib.crc32(str(code).encode()) % 1_000_003
    with _quiet():
        t_obs, _, _ = wp.lookup_weather(u_fc, clim, "all")
        t_fc = wp.simulate_forecast(u_fc, t_obs[:, L:], seed=seed)
        K = wp.FORECAST_DAYS
        fc = pd.DataFrame({"coato_id": str(code), "issue_date": np.repeat(u_fc["origin_date"].to_numpy(), K),
                           "lead_day": np.tile(np.arange(1, K + 1), len(u_fc)), "t_fc": t_fc.reshape(-1)})
        fc["issue_date"] = pd.to_datetime(fc["issue_date"])
        grid, grid_mm = wp.build_forecast_grid(fc, clim), None
        if wp.USE_MINMAX:
            tmin_obs, _, tmax_obs, _ = wp.lookup_minmax(u_fc, clim, "all")
            tmin_fc, tmax_fc = wp.simulate_forecast_minmax(u_fc, t_fc, tmin_obs[:, L:], tmax_obs[:, L:], seed=seed)
            fc["t_fc_min"], fc["t_fc_max"] = tmin_fc.reshape(-1), tmax_fc.reshape(-1)
            grid_mm = dict(min=wp.build_forecast_grid(fc, clim, col="t_fc_min"),
                           max=wp.build_forecast_grid(fc, clim, col="t_fc_max"))
    del fc, t_obs, t_fc

    cut = int((pd.Timestamp(cfg["cutoff"]) - d0).days)
    crc = np.array([zlib.crc32(k.encode()) for k in keys], dtype=np.uint64)
    Kw = (T - 1 - H - (L - 1)) // S + 1
    acc = {}
    vs_X, vs_y, vs_b, vs_l = [], [], [], []
    ts = {k: [] for k in ("X", "y", "b", "l", "p30", "p7", "day", "u")}
    names = None
    blk = int(cfg["block"])
    for a in range(0, n, blk):
        b_ = min(n, a + blk)
        o = (L - 1) + (crc[a:b_] % S).astype(np.int64)[:, None] + S * np.arange(Kw)[None, :]
        r = np.broadcast_to(np.arange(a, b_)[:, None], o.shape)
        inside = o <= T - 1 - H
        r, o = r[inside], o[inside]
        cnt = np.zeros((b_ - a, T + 1), dtype=np.int16)
        np.cumsum(np.isfinite(M[a:b_]), axis=1, dtype=np.int16, out=cnt[:, 1:])
        rl = r - a
        n_in = cnt[rl, o + 1] - cnt[rl, o - L + 1]
        n_rec = cnt[rl, o + 1] - cnt[rl, o - wp.RECENT_WINDOW + 1]
        n_tg = cnt[rl, o + H + 1] - cnt[rl, o + 1]
        del cnt, rl
        keep = (n_in >= wp.MIN_INPUT_DAYS) & (n_rec >= wp.MIN_RECENT_DAYS) & (n_tg >= wp.MIN_TARGET_DAYS)
        res["candidates"] += int(len(r))
        res["passed"] += int(keep.sum())
        r, o, n_tg = r[keep], o[keep], n_tg[keep]
        if not len(r):
            continue
        origin = d0 + pd.to_timedelta(o, unit="D")
        meta0 = pd.DataFrame({"origin_date": origin})
        with _quiet():
            regok = wp.regime_keep_mask(meta0)
            tr_m, fit_m, val_m, _ = wp.make_masks(meta0, cfg["cutoff"], cfg["val_start"], None, cfg["val_end"])
        u = ((crc[r] * np.uint64(2654435761) + o.astype(np.uint64) * np.uint64(2246822519)) % np.uint64(2 ** 32)) / 2.0 ** 32
        train = tr_m & regok & (n_tg >= wp.TRAIN_MIN_TARGET_DAYS)
        test = (o >= cut + H) & (u < cfg["test_sample"])
        sel = train | test
        if not sel.any():
            continue
        r, o, u, origin = r[sel], o[sel], u[sel], origin[sel]
        train, fit_m, val_m, test = train[sel], fit_m[sel], val_m[sel], test[sel]
        kin = M[r[:, None], o[:, None] + np.arange(-(L - 1), 1)[None, :]].astype("float64")
        ktg = M[r[:, None], o[:, None] + np.arange(1, H + 1)[None, :]].astype("float64")
        ht = pd.Series(house[r]).astype("Int64").astype(str).replace("<NA>", np.nan).to_numpy()
        pdf = pd.DataFrame({"meter_id": np.asarray(keys, dtype=object)[r].astype(str), "consumer_id": np.nan,
                            "origin_date": origin, "coato_code": str(code), "housing_type_id": ht})
        with _quiet():
            meta, feats, base, _ = wp.build_features_chunked(
                pdf, kin, ktg, clim, fc_grid=grid, fc_grid_mm=grid_mm, chunk=cfg["chunk"], forecast_mode="archive",
                normal_mode="loyo", beta_h_prior=cfg["prior_h"], beta_c_prior=cfg["prior_c"],
                beta_h_prior_mm=cfg["prior_h_mm"], beta_c_prior_mm=cfg["prior_c_mm"], meter_segment=None)
        ly = year_ago(M, r, o, wp.BLOCKS)
        lev = level_of(kin, cfg["norm_days"])
        ok = np.isfinite(lev) & (lev > cfg["min_level"])
        levs = np.where(ok, lev, 1.0)
        X, names = design(wp, feats, ly, kin, levs, L)
        obs = np.isfinite(ktg)
        with np.errstate(invalid="ignore", divide="ignore"):
            y_ratio = np.clip(ktg / levs[:, None] - 1.0, -1.0, 4.0)
            y_resid = np.clip((ktg - base) / levs[:, None], -1.0, 4.0)
        y_ratio, y_resid = np.where(obs, y_ratio, 0.0), np.where(obs, y_resid, 0.0)
        kwh = levs * ok
        w = {"kwh": kwh, "sqrt": np.sqrt(kwh), "none": ok.astype(float)}[cfg["weight"]]
        trn = train & ok
        for part, m in (("fit", trn & fit_m), ("rest", trn & ~fit_m)):
            if not m.any():
                continue
            x, ww, ob = X[m], w[m], obs[m]
            xw = x * ww[:, None]
            G0 = x.T @ xw
            miss = ~ob
            Gm = np.zeros((H, x.shape[1], x.shape[1]))
            for h in np.flatnonzero(miss.any(axis=0)):
                q = miss[:, h]
                Gm[h] = x[q].T @ xw[q]
            Rr = xw.T @ y_ratio[m]                                 # missing target days are 0 in y: they add nothing
            Rs = xw.T @ y_resid[m]
            st = dict(G0=G0, Gm=Gm, Rr=Rr, Rs=Rs, s1=x.sum(0), s2=(x * x).sum(0), n=int(m.sum()))
            if part in acc:
                for k in st:
                    acc[part][k] = acc[part][k] + st[k]
            else:
                acc[part] = st
        res["train"] += int(trn.sum())
        res["fit"] += int((trn & fit_m).sum())
        res["val"] += int((trn & val_m).sum())
        vsel = trn & val_m & (u < cfg["val_sample"])
        if vsel.any():
            vs_X.append(X[vsel].astype(np.float32)), vs_y.append(ktg[vsel].astype(np.float32))
            vs_b.append(base[vsel].astype(np.float32)), vs_l.append(lev[vsel].astype(np.float32))
        tsel = test
        if tsel.any():
            res["test"] += int(tsel.sum())
            ts["X"].append(X[tsel].astype(np.float32)), ts["y"].append(ktg[tsel].astype(np.float32))
            ts["b"].append(base[tsel].astype(np.float32)), ts["l"].append(lev[tsel].astype(np.float32))
            ts["p30"].append(feats["level_kwh"].to_numpy()[tsel].astype(np.float32))
            ts["p7"].append(feats["kwh_mean_7d"].to_numpy()[tsel].astype(np.float32))
            ts["day"].append(o[tsel].astype(np.int32)), ts["u"].append(u[tsel].astype(np.float32))
        del kin, ktg, pdf, meta, feats, base, X
    res["acc"] = acc
    res["names"] = names
    res["val_sample"] = None if not vs_X else dict(X=np.concatenate(vs_X), y=np.concatenate(vs_y), b=np.concatenate(vs_b),
                                                     l=np.concatenate(vs_l))
    res["test_sample"] = None if not ts["X"] else {k: np.concatenate(v) for k, v in ts.items()}
    res["seconds"] = time.time() - t0
    try:
        import resource
        res["peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except Exception:                                                 # noqa: BLE001
        res["peak_gb"] = float("nan")
    return res


# =============================================================================
# Solving
# =============================================================================
def standardiser(s1, s2, n, cols, n_hist):
    """Affine map T so that X[:, cols] @ T = standardised columns (history and intercept unchanged, exogenous z-scored)."""
    p = len(cols)
    Tm = np.eye(p)
    for j, c in enumerate(cols):
        if n_hist <= c < len(s1) - 1:                               # exogenous column
            mu = s1[c] / n
            sd = np.sqrt(max(s2[c] / n - mu * mu, 0.0))
            sd = sd if sd > 1e-9 else 1.0
            Tm[j, j] = 1.0 / sd
            Tm[-1, j] = -mu / sd                                    # the intercept is the last selected column
    return Tm


def solve_all(G0, Gm, R, cols, Tm, lam):
    """Ridge per target day h on rows where day h was observed: G_h = G0 - Gm[h]. Penalty lam x mean diagonal
    (standardised space) on everything but the intercept. -> W (p, H) in standardised space."""
    ix = np.ix_(cols, cols)
    H = R.shape[1]
    W = np.zeros((len(cols), H))
    for h in range(H):
        G = Tm.T @ (G0[ix] - Gm[h][ix]) @ Tm
        Rh = Tm.T @ R[cols, h]
        d = np.diag(G)[:-1]
        P = np.eye(len(cols)) * lam * float(np.mean(d))
        P[-1, -1] = 0.0
        W[:, h] = np.linalg.solve(G + P, Rh)
    return W


def to_kwh(var, Z, lev, base):
    if var.startswith("thermal"):
        return np.maximum(base + lev[:, None] * Z, 0.0)
    return np.maximum(lev[:, None] * (1.0 + Z), 0.0)


def wape(P, A):
    v = np.isfinite(P) & np.isfinite(A)
    return float(np.abs(P[v] - A[v]).sum() / max(A[v].sum(), 1e-12)) if v.any() else np.nan


def metrics(name, P, A, day, set_name):
    rows = []
    for h in HORIZONS:
        p, a = P[:, h - 1].astype(float), A[:, h - 1].astype(float)
        v = np.isfinite(p) & np.isfinite(a)
        if not v.any():
            continue
        d = pd.DataFrame({"t": day[v] + h, "p": p[v], "a": a[v]}).groupby("t").sum()
        rows.append(dict(set=set_name, model=name, horizon=h, mae=float(np.abs(p[v] - a[v]).mean()),
                         wape=float(np.abs(p[v] - a[v]).sum() / a[v].sum()), bias=float((p[v] - a[v]).sum() / a[v].sum()),
                         sys_wape=float((d["p"] - d["a"]).abs().sum() / d["a"].sum()),
                         sys_mape=float(((d["p"] - d["a"]).abs() / d["a"].where(d["a"] > 0)).mean()),
                         days_scored=int(len(d)), windows_per_day=float(v.sum() / max(len(d), 1)), n=int(v.sum())))
    v = np.isfinite(P) & np.isfinite(A)
    rows.append(dict(set=set_name, model=name, horizon="all", mae=float(np.abs(P[v] - A[v]).mean()),
                     wape=float(np.abs(P[v] - A[v]).sum() / A[v].sum()), bias=float((P[v] - A[v]).sum() / A[v].sum()),
                     n=int(v.sum())))
    return rows


# =============================================================================
def main(argv=None):
    import whole_pipeline_v8 as wp
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=wp.ENRICHED_ROOT)
    p.add_argument("--buckets", default="all", help="'all' or a comma list of customer_bucket values (0-99)")
    p.add_argument("--coatos", default=None, help="comma list of district codes (default: every finished district)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads-per-worker", type=int, default=None, help="BLAS threads in each worker (default: cores / workers)")
    p.add_argument("--stride", type=int, default=wp.WINDOW_STRIDE, help="days between a customer's origins (default as the "
                   "household model: 30; 10 = three times the windows)")
    p.add_argument("--cutoff", default=wp.TRAIN_CUTOFF)
    p.add_argument("--val-start", default=wp.VAL_START)
    p.add_argument("--val-end", default=wp.VAL_END)
    p.add_argument("--test-sample", type=float, default=0.05, help="share of test windows kept for scoring (stable hash)")
    p.add_argument("--val-sample", type=float, default=0.1, help="share of validation windows kept to choose the penalty")
    p.add_argument("--catboost-test-share", type=float, default=0.01,
                   help="the household CatBoost run's ENRICHED_TEST_SAMPLE: that subset is scored separately")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--norm-days", type=int, default=7)
    p.add_argument("--min-level", type=float, default=0.2, help="kWh/day: lower levels are not trained on (fallback forecast)")
    p.add_argument("--weight", choices=["kwh", "sqrt", "none"], default="kwh")
    p.add_argument("--prior-h", type=float, default=0.049, help="pooled heating-sensitivity prior (mean temperature)")
    p.add_argument("--prior-c", type=float, default=0.103)
    p.add_argument("--prior-h-mm", type=float, default=0.047, help="pooled priors for the min/max degree days")
    p.add_argument("--prior-c-mm", type=float, default=0.061)
    p.add_argument("--block", type=int, default=10000, help="customers per block inside a worker (memory)")
    p.add_argument("--chunk", type=int, default=50000, help="windows per feature chunk (memory)")
    p.add_argument("--cache-dir", default=None, help="per-district partial results (resume); holds no customer keys")
    args = p.parse_args(argv)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    with _tee(os.path.join(out_dir, "run_log.txt")):
        run(wp, args, out_dir, variants)


@contextlib.contextmanager
def _tee(path):
    class T:
        def __init__(self, s, f):
            self.s, self.f = s, f

        def write(self, x):
            self.s.write(x)
            self.f.write(x)
            return len(x)

        def flush(self):
            self.s.flush()
            self.f.flush()
    fh = open(path, "a", buffering=1)
    old = sys.stdout
    sys.stdout = T(old, fh)
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)} =====")
    try:
        yield
    finally:
        sys.stdout = old
        fh.close()


def run(wp, args, out_dir, variants):
    import concurrent.futures as cf
    import hashlib
    import multiprocessing as mp
    import pickle
    import enriched_io as eio
    buckets = None if str(args.buckets).lower() == "all" else [int(b) for b in str(args.buckets).split(",")]
    cfg = dict(here=HERE, root=args.root, start=wp.ENRICHED_START, end=wp.ENRICHED_END, labels=list(wp.ENRICHED_LABELS),
               buckets=buckets, stride=args.stride, cutoff=args.cutoff, val_start=args.val_start, val_end=args.val_end,
               test_sample=args.test_sample, val_sample=args.val_sample, norm_days=args.norm_days, min_level=args.min_level,
               weight=args.weight, prior_h=args.prior_h, prior_c=args.prior_c, prior_h_mm=args.prior_h_mm,
               prior_c_mm=args.prior_c_mm, block=args.block, chunk=args.chunk, seed=wp.SEED,
               wp=dict(INCLUDE_OLD=wp.INCLUDE_OLD, EXCLUDE_RANGES=list(wp.EXCLUDE_RANGES), EXCLUDE_MODE=wp.EXCLUDE_MODE,
                       REGIME_CUTOFF=wp.REGIME_CUTOFF, OLD_START=wp.OLD_START, USE_MINMAX=wp.USE_MINMAX))
    lim = wp._memory_limit_gb()
    print(f"[config] customers {'ALL' if buckets is None else buckets} | stride {args.stride} | cutoff {args.cutoff} | "
          f"validation ({args.val_start}, {args.val_end}] | test sample {args.test_sample} (CatBoost subset "
          f"{args.catboost_test_share}) | val sample {args.val_sample} | variants {','.join(variants)} | {args.workers} workers | "
          f"memory limit {'unknown' if lim is None else f'{lim:.0f} GB'}")
    fs, root = eio.open_root(args.root)
    codes = eio.finished_districts(fs, root, set(args.coatos.split(",")) if args.coatos else None)
    cache = None
    if args.cache_dir:
        sig = hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if k != "here"}, sort_keys=True, default=str)
                           .encode()).hexdigest()[:12]
        cache = os.path.join(os.path.expanduser(args.cache_dir), f"hh_nlinear_{sig}")
        os.makedirs(cache, exist_ok=True)
        print(f"[cache] {cache}")

    tot = dict(customers=0, candidates=0, passed=0, train=0, fit=0, val=0, test=0)
    acc, names, vs, tsamp = {}, None, [], []
    t0 = time.time()
    done = [0]
    todo = []

    def absorb(res):
        nonlocal names
        for k in tot:
            tot[k] += res.get(k, 0)
        for part, st in (res.get("acc") or {}).items():
            if part in acc:
                for k in st:
                    acc[part][k] = acc[part][k] + st[k]
            else:
                acc[part] = st
        if res.get("names"):
            if names is None:
                names = res["names"]
            elif names != res["names"]:
                raise RuntimeError(f"district {res['code']}: feature list differs from the others")
        if res.get("val_sample") is not None:
            vs.append(res["val_sample"])
        if res.get("test_sample") is not None:
            tsamp.append(res["test_sample"])

    def keep(res, from_cache=False):
        if cache and not from_cache:
            tmp = os.path.join(cache, f"{res['code']}.pkl.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, os.path.join(cache, f"{res['code']}.pkl"))
        absorb(res)
        if not from_cache:
            done[0] += 1
            k = done[0]
            if k % 10 == 0 or k == len(todo) or k <= 3:
                el = time.time() - t0
                print(f"[districts] {k}/{len(todo)}: {tot['customers']:,} customers, {tot['train']:,} training windows "
                      f"(fit {tot['fit']:,}, val {tot['val']:,}), {tot['test']:,} test windows kept; {el / 60:.1f} min, about "
                      f"{el / k * (len(todo) - k) / 60:.1f} min left | worker peak {res.get('peak_gb', float('nan')):.1f} GB, "
                      f"main {wp._rss_gb():.1f} GB", flush=True)

    for code in codes:
        pth = os.path.join(cache, f"{code}.pkl") if cache else None
        if pth and os.path.exists(pth):
            with open(pth, "rb") as fh:
                keep(pickle.load(fh), from_cache=True)
        else:
            todo.append(code)
    if cache and len(todo) < len(codes):
        print(f"[cache] {len(codes) - len(todo)} district(s) loaded; {len(todo)} to read")
    if todo:
        if args.workers > 1:
            thr = args.threads_per_worker or max(1, (os.cpu_count() or 4) // args.workers)
            saved = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
            for v in saved:
                os.environ[v] = str(thr)
            try:
                with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
                    pending = {ex.submit(district_worker, c, cfg): c for c in todo}
                    while pending:                                   # drop each future once absorbed (it holds the result)
                        done_now, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                        for f in done_now:
                            code = pending.pop(f)
                            try:
                                res = f.result()
                            except Exception as e:                   # noqa: BLE001
                                raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                            del f
                            keep(res)
                            del res
            finally:
                for v, x in saved.items():
                    if x is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = x
        else:
            for c in todo:
                keep(district_worker(c, cfg))
    print(f"\n[data] {tot['customers']:,} customers with observed days in {len(codes)} districts | candidate windows "
          f"{tot['candidates']:,}, pass the input/target rules {tot['passed']:,} ({tot['passed'] / max(tot['candidates'], 1):.1%}) | "
          f"TRAINING on {tot['train']:,} windows (fit {tot['fit']:,}, validation {tot['val']:,}, embargo "
          f"{tot['train'] - tot['fit'] - tot['val']:,}) | test windows kept {tot['test']:,}")
    if "fit" not in acc or not vs or not tsamp:
        raise SystemExit("no fit windows, validation sample or test sample: check the dates against the data")

    n_hist = wp.LOOKBACK
    p_all = n_hist + len(names) + 1
    cols_of = {v: (list(range(p_all)) if v.endswith("_x") else list(range(n_hist)) + [p_all - 1]) for v in variants}
    fit, rest = acc["fit"], acc.get("rest")
    tr = {k: fit[k] + (rest[k] if rest else 0) for k in fit}
    Vx = np.concatenate([v["X"] for v in vs]).astype("float64")
    Vy = np.concatenate([v["y"] for v in vs]).astype("float64")
    Vb = np.concatenate([v["b"] for v in vs]).astype("float64")
    Vl = np.concatenate([v["l"] for v in vs]).astype("float64")
    Vok = np.isfinite(Vl) & (Vl > args.min_level)
    print(f"[penalty] validation sample {len(Vx):,} windows ({Vok.mean():.1%} above the level floor)")
    info = dict(data=tot, features=dict(history=n_hist, exogenous=names), variants={}, config={k: v for k, v in cfg.items()
                                                                                                 if k not in ("here",)})
    Wfinal = {}
    for v in variants:
        cols = cols_of[v]
        R = "Rs" if v.startswith("thermal") else "Rr"
        Tm_fit = standardiser(fit["s1"], fit["s2"], fit["n"], cols, n_hist)
        search = []
        for lam in LAMBDAS:
            Ws = solve_all(fit["G0"], fit["Gm"], fit[R], cols, Tm_fit, lam)
            Z = (Vx[:, cols] @ Tm_fit) @ Ws
            P = to_kwh(v, Z, np.where(Vok, Vl, 0.0), Vb)
            P = np.where(Vok[:, None], P, Vb if v.startswith("thermal") else np.nan_to_num(Vl)[:, None])
            search.append(dict(lam=lam, val_wape=wape(P, Vy)))
        best = min(search, key=lambda s: s["val_wape"] if np.isfinite(s["val_wape"]) else np.inf)
        Tm = standardiser(tr["s1"], tr["s2"], tr["n"], cols, n_hist)
        Ws = solve_all(tr["G0"], tr["Gm"], tr[R], cols, Tm, best["lam"])
        Wraw = Tm @ Ws                                               # raw inputs @ Wraw = standardised inputs @ Ws
        Wfinal[v] = (cols, Wraw)
        fnames = [f"hist_lag{n_hist - i}" for i in range(n_hist)] + (names if v.endswith("_x") else []) + ["intercept"]
        np.savez_compressed(os.path.join(out_dir, f"weights_{v}.npz"), W=Wraw.astype("float32"), feature_names=np.array(fnames),
                            norm_days=args.norm_days, min_level=args.min_level, target=("residual" if v.startswith("thermal") else "ratio"))
        info["variants"][v] = dict(chosen_lambda=best["lam"], search=search)
        print(f"[penalty] {v:18s} chosen {best['lam']:g} | validation WAPE " +
              "  ".join(f"{s['lam']:g}:{s['val_wape']:.1%}" for s in search))
        if best["lam"] in (LAMBDAS[0], LAMBDAS[-1]):
            print(f"[penalty] WARNING: {v}: the chosen penalty is at the end of the grid")

    # ---------------- test ----------------
    Tx = np.concatenate([t["X"] for t in tsamp]).astype("float64")
    Ty = np.concatenate([t["y"] for t in tsamp]).astype("float64")
    Tb = np.concatenate([t["b"] for t in tsamp]).astype("float64")
    Tl = np.concatenate([t["l"] for t in tsamp]).astype("float64")
    Tp30 = np.concatenate([t["p30"] for t in tsamp]).astype("float64")
    Tp7 = np.concatenate([t["p7"] for t in tsamp]).astype("float64")
    Tday = np.concatenate([t["day"] for t in tsamp]).astype("int64")
    Tu = np.concatenate([t["u"] for t in tsamp]).astype("float64")
    Tok = np.isfinite(Tl) & (Tl > args.min_level)
    preds = {"persistence30": np.repeat(Tp30[:, None], wp.HORIZON, 1), "persistence7": np.repeat(Tp7[:, None], wp.HORIZON, 1),
             "thermal_baseline": Tb}
    for v in variants:
        cols, Wraw = Wfinal[v]
        P = to_kwh(v, Tx[:, cols] @ Wraw, np.where(Tok, Tl, 0.0), Tb)
        preds[v] = np.where(Tok[:, None], P, Tb if v.startswith("thermal") else np.nan_to_num(Tl)[:, None])
    rows = []
    sets = [("all_test_sample", np.ones(len(Tu), bool)), ("catboost_test_set", Tu < args.catboost_test_share)]
    for set_name, m in sets:
        if not m.any():
            continue
        for name, P in preds.items():
            rows += metrics(name, P[m], Ty[m], Tday[m], set_name)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    first = pd.Timestamp(wp.ENRICHED_START)
    print(f"\n[test] {len(Tx):,} test windows, origins {(first + pd.Timedelta(days=int(Tday.min()))).date()} .. "
          f"{(first + pd.Timedelta(days=int(Tday.max()))).date()}; {np.isfinite(Ty).mean():.1%} of their target days observed; "
          f"CatBoost-run subset {int((Tu < args.catboost_test_share).sum()):,} windows")
    order = list(preds)
    for set_name, _ in sets:
        t = tab[tab["set"] == set_name]
        if t.empty:
            continue
        print(f"\n=== {set_name}: MAE (kWh per household per day) ===")
        print(t.pivot(index="model", columns="horizon", values="mae").reindex(order).map(lambda x: f"{x:6.3f}").to_string())
        th = t[t["horizon"] != "all"]
        for col, lab in (("wape", "pooled WAPE"), ("sys_wape", "system WAPE (sum of all test households per target day)"),
                         ("bias", "bias")):
            print(f"\n=== {set_name}: {lab} ===")
            print(th.pivot(index="model", columns="horizon", values=col).reindex(order)
                  .map(lambda x: f"{x:+6.1%}" if col == "bias" else f"{x:6.1%}").to_string())
    with open(os.path.join(out_dir, "nlinear_info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    print(f"\n[done] {out_dir} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
