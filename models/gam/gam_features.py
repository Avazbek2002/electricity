"""Inputs of the GAM (Explainable Boosting Machine), built the same way in training (from district matrices) and in
prediction (from the harness Panel), so the two can never drift apart.

One row = household x origin x horizon h. Inputs, all known at the origin:
  h            days ahead (1..60)
  log_level    log(level + 0.05), level = mean of the last 7 observed days (>= 3), else the last 30
  r_last       yesterday / level
  r7_30        mean of the last 7 days / mean of the last 30
  slope30      linear trend of the last 30 days, per day, relative to the level
  cv30         std / mean of the last 30 days
  obs90        share of the last 90 days observed
  temp         target-day temperature: issued forecast for days 1-10; normal + the day 8-10 anomaly x exp(-(h-10)/5) after
  temp_dev     temp - the mean temperature of the last 30 days
  doy, dow     target day of year, target day of week (category)
  ly_ratio     last year's same block / last year's level (block of h), 1 if unknown; ly_have 1/0
  housing      housing_type_id (category)
Target: y / level (clipped 0..5); weight: level (so the loss counts kWh, as pooled WAPE does).
"""
import sys
import time
import zlib

import numpy as np
import pandas as pd

KNOTS = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60)
FEATURES = ["h", "log_level", "r_last", "r7_30", "slope30", "cv30", "obs90", "temp", "temp_dev", "doy", "dow",
            "ly_ratio", "ly_have", "housing"]
CATEGORICAL = ("dow", "housing")


def _nanmean(a, axis=1):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.isfinite(a).sum(axis)
        s = np.where(np.isfinite(a), a, 0.0).sum(axis)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def household_block(kwh, wp):
    """kwh (n, W) observed daily kWh ending on the origin -> dict of per-household arrays (level and history inputs)."""
    import household_nlinear as hn
    kin = kwh[:, -wp.LOOKBACK:].astype("float64")
    n = len(kin)
    lev = hn.level_of(kin, 7)
    l30, m7 = _nanmean(kin[:, -30:]), _nanmean(kin[:, -7:])
    y30 = kin[:, -30:]
    x = np.arange(30, dtype=float)[None, :].repeat(n, 0)
    m = np.isfinite(y30)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(m, x, 0).sum(1) / np.maximum(m.sum(1), 1)
        ym = np.where(m, y30, 0).sum(1) / np.maximum(m.sum(1), 1)
        sxy = np.where(m, (x - xm[:, None]) * (y30 - ym[:, None]), 0).sum(1)
        sxx = np.where(m, (x - xm[:, None]) ** 2, 0).sum(1)
        slope = np.where((m.sum(1) >= 10) & (sxx > 0), sxy / np.maximum(sxx, 1e-9), np.nan) / lev
        sd = np.sqrt(np.maximum(_nanmean(y30 ** 2) - l30 ** 2, 0))
        out = dict(level=lev, log_level=np.log(np.maximum(np.nan_to_num(lev), 0) + 0.05), r_last=kin[:, -1] / lev,
                   r7_30=m7 / l30, slope30=slope, cv30=sd / l30, obs90=np.isfinite(kin).mean(1))
    ly = hn.year_ago(kwh, np.arange(n), np.full(n, kwh.shape[1] - 1), wp.BLOCKS)
    out["ly"] = np.column_stack([ly[f"ly_ratio_{a}_{b}"] for a, b in wp.BLOCKS])     # (n, blocks)
    for k in ("r_last", "r7_30", "slope30", "cv30"):
        out[k] = np.clip(out[k], -5, 5)
    return out


def origin_weather(coato, origin, clim, grid, wp):
    """-> (temperature path for target days 1..60, mean observed temperature of the last 30 days)."""
    L, H = wp.LOOKBACK, wp.HORIZON
    pdf = pd.DataFrame({"coato_code": [str(coato)], "origin_date": [pd.Timestamp(origin)]})
    t_obs, t_nrm, _ = wp.lookup_weather(pdf, clim, "loyo")
    t_fc = wp.lookup_forecast_archive(pdf, clim, grid)[0]
    nrm = t_nrm[0, L:L + H].astype(float)
    path = nrm.copy()
    if np.isfinite(t_fc).all():
        K = len(t_fc)
        path[:K] = t_fc
        anom = float(np.mean(t_fc[-3:] - nrm[K - 3:K]))
        h = np.arange(K + 1, H + 1)
        path[K:] = nrm[K:] + anom * np.exp(-(h - K) / 5.0)
    past30 = float(np.nanmean(t_obs[0, L - 30:L])) if np.isfinite(t_obs[0, L - 30:L]).any() else np.nan
    return path, past30


def rows(hh, path, past30, origin, housing, horizons, wp):
    """Cross households x horizons -> DataFrame of FEATURES (n * len(horizons) rows, household-major)."""
    n, k = len(hh["level"]), len(horizons)
    hz = np.asarray(horizons)
    tgt = pd.Timestamp(origin) + pd.to_timedelta(hz, unit="D")
    blk = np.array([next(i for i, (a, b) in enumerate(wp.BLOCKS) if a <= h <= b) for h in hz])
    ly = hh["ly"][:, blk]
    rep = lambda a: np.repeat(np.asarray(a, dtype=float), k)                  # noqa: E731
    til = lambda a: np.tile(np.asarray(a), n)                                  # noqa: E731
    hs = pd.Series(housing).astype("Int64").astype(str).replace("<NA>", "na").to_numpy()
    return pd.DataFrame({
        "h": til(hz.astype(float)), "log_level": rep(hh["log_level"]), "r_last": rep(hh["r_last"]),
        "r7_30": rep(hh["r7_30"]), "slope30": rep(hh["slope30"]), "cv30": rep(hh["cv30"]), "obs90": rep(hh["obs90"]),
        "temp": til(path[hz - 1]), "temp_dev": til(path[hz - 1] - past30), "doy": til(tgt.dayofyear.to_numpy(float)),
        "dow": til(tgt.dayofweek.astype(str).to_numpy()), "ly_ratio": np.where(np.isfinite(ly), ly, 1.0).ravel(),
        "ly_have": np.isfinite(ly).astype(float).ravel(), "housing": np.repeat(hs, k)})


def _excluded(o, dates, wp):
    a, b = dates[max(o - wp.LOOKBACK + 1, 0)], dates[min(o + wp.HORIZON, len(dates) - 1)]
    return any(a <= pd.Timestamp(e) and pd.Timestamp(s) <= b for s, e in wp.EXCLUDE_RANGES)


def district_rows(code, cfg):
    """Training rows of one district (worker process): sampled households, origins every cfg['origin_step'] days up to
    the cutoff, the KNOTS horizons whose target day was observed."""
    for p in cfg["paths"]:
        if p not in sys.path:
            sys.path.insert(0, p)
    import whole_pipeline_v8 as wp
    from harness.data import read_district
    from harness.weather import district_weather
    t0 = time.time()
    D = read_district(code, cfg["roots"], cfg["start"], cfg["data_end"], cfg["labels"])
    out = dict(code=str(code), rows=0, seconds=0.0, X=None)
    if not len(D["keys"]):
        return out
    M, dates = D["M"], D["dates"]
    T, L, H = M.shape[1], wp.LOOKBACK, wp.HORIZON
    clim, grid, _ = district_weather(code, D["wt"], dates, cfg["seed"], wp)
    u = np.array([zlib.crc32(str(k).encode()) for k in D["keys"]], dtype=np.uint64) % 1_000_000 / 1e6
    keep = np.flatnonzero(u < cfg["household_share"])
    if not len(keep):
        return out
    Mk = M[keep]
    cnt = np.zeros((len(keep), T + 1), dtype=np.int32)
    np.cumsum(np.isfinite(Mk), axis=1, out=cnt[:, 1:])
    cut = int((pd.Timestamp(cfg["cutoff"]) - dates[0]).days)
    W = int(cfg["history_days"])
    hz = np.asarray(KNOTS)
    parts = []
    step = int(cfg["origin_step"])
    first = L - 1 + zlib.crc32(str(code).encode()) % step                 # origins staggered across districts
    for o in range(first, min(cut, T - 1 - H) + 1, step):
        if _excluded(o, dates, wp):
            continue
        n_in = cnt[:, o + 1] - cnt[:, o - L + 1]
        n_rec = cnt[:, o + 1] - cnt[:, o - wp.RECENT_WINDOW + 1]
        r = np.flatnonzero((n_in >= wp.MIN_INPUT_DAYS) & (n_rec >= wp.MIN_RECENT_DAYS))
        if not len(r):
            continue
        lo = max(0, o - W + 1)
        hh = household_block(Mk[r, lo:o + 1], wp)
        ok = np.isfinite(hh["level"]) & (hh["level"] > cfg["min_level"])
        if not ok.any():
            continue
        r = r[ok]
        hh = {k: v[ok] for k, v in hh.items()}
        path, past30 = origin_weather(code, dates[o], clim, grid, wp)
        X = rows(hh, path, past30, dates[o], D["housing"][keep][r], hz, wp)
        y = Mk[r][:, o + hz].astype(float)
        X["y"] = np.clip(y / hh["level"][:, None], 0, 5).ravel()
        X["w"] = np.repeat(hh["level"], len(hz))
        X = X[np.isfinite(y).ravel()]
        if len(X):
            parts.append(X)
    if parts:
        X = pd.concat(parts, ignore_index=True)
        out.update(X=X, rows=len(X))
    out["seconds"] = time.time() - t0
    return out
