"""Training windows at FIXED origins (the reforecast issue dates) for whole_pipeline_v8.extract_windows_enriched.

v8 gives each customer origins every WINDOW_STRIDE days (staggered by a hash of the key). With real forecasts only on
the reforecast issue dates (4 a month, 2024 and 2025), a window needs its origin (= the last history day, o - 1) on
init - 1. `fixed_origins` swaps v8's district worker for `district_worker` below for the duration of one extraction:
  - origins = the given dates (every customer, every date), and with stride=True also v8's staggered grid (those
    windows have no forecast: the model falls back to the climate normals there, as it does after day 14);
  - every other rule is v8's own (input / recent / target day counts, the stable-hash training sample, segment features).
The pool workers are spawned processes: the dates travel in the environment (WX_WINDOW_ORIGINS), the function by name.
v8 itself is not changed.
"""
import contextlib
import os
import time as _time
import zlib

import numpy as np
import pandas as pd

ENV_ORIGINS, ENV_STRIDE = "WX_WINDOW_ORIGINS", "WX_WINDOW_STRIDE"


@contextlib.contextmanager
def fixed_origins(wp, dates, stride=False):
    """Within the block, wp.extract_windows_enriched() builds windows whose origin_date is one of `dates`
    (+ v8's staggered grid when stride=True)."""
    old_fn = wp._enriched_district_worker
    old_env = {k: os.environ.get(k) for k in (ENV_ORIGINS, ENV_STRIDE)}
    os.environ[ENV_ORIGINS] = ",".join(str(pd.Timestamp(d).date()) for d in dates)
    os.environ[ENV_STRIDE] = "1" if stride else "0"
    wp._enriched_district_worker = district_worker
    try:
        yield
    finally:
        wp._enriched_district_worker = old_fn
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def origin_grid(crc, d0, T, L, H, S, dates, stride):
    """(n, K) origin day indices per customer: the fixed dates, plus the staggered stride grid; -1 = none."""
    fix = np.array(sorted({int((pd.Timestamp(d) - d0).days) for d in dates}), dtype=np.int64)
    n = len(crc)
    o = np.broadcast_to(fix[None, :], (n, len(fix)))
    if stride:
        K = (T - 1 - H - (L - 1)) // S + 1
        st = (L - 1) + (crc % S).astype(np.int64)[:, None] + S * np.arange(max(K, 0))[None, :]
        o = np.concatenate([o, st], axis=1)
    o = np.sort(o, axis=1)
    dup = np.zeros(o.shape, bool)
    dup[:, 1:] = o[:, 1:] == o[:, :-1]
    return np.where(dup, -1, o)


def district_worker(code, cfg):
    """v8's _enriched_district_worker with the origin grid from origin_grid(); everything else as v8 does it."""
    import whole_pipeline_v8 as wp
    t0 = _time.time()
    dates_fix = [d for d in os.environ.get(ENV_ORIGINS, "").split(",") if d]
    stride = os.environ.get(ENV_STRIDE, "0") == "1"
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T, L, H, S = (d1 - d0).days + 1, cfg["L"], cfg["H"], cfg["stride"]
    keys, M, wt, h_val = wp._read_enriched_district(code, cfg)
    out = dict(code=str(code), customers=0, candidates=0, passed=0, seconds=0.0, pdf=None, kin=None, ktg=None,
               seg=None, weather=None)
    if not keys:
        out["seconds"] = _time.time() - t0
        return out
    n = len(keys)
    house = h_val
    has_obs = np.isfinite(M).any(axis=1)
    out["customers"] = int(has_obs.sum())
    dates = d0 + pd.to_timedelta(np.arange(T), unit="D")
    have_w = np.isfinite(wt).any(axis=0)
    out["weather"] = pd.DataFrame({"coato_id": str(code), "d": dates[have_w], "t": wt[0, have_w], "tmin": wt[1, have_w],
                                   "tmax": wt[2, have_w]})

    crc = np.array([zlib.crc32(k.encode()) for k in keys], dtype=np.uint64)
    sampling = cfg["train_sample"] is not None or cfg["test_sample"] is not None
    cut = int((pd.Timestamp(cfg["cutoff"]) - d0).days)
    p_tr = 1.0 if cfg["train_sample"] is None else float(cfg["train_sample"])
    p_te = 1.0 if cfg["test_sample"] is None else float(cfg["test_sample"])
    R, O, NT, KIN, KTG = [], [], [], [], []
    blk = int(cfg.get("block", 10000))
    for a in range(0, n, blk):
        b_ = min(n, a + blk)
        o = origin_grid(crc[a:b_], d0, T, L, H, S, dates_fix, stride)
        r = np.broadcast_to(np.arange(a, b_)[:, None], o.shape)
        inside = (o >= L - 1) & (o <= T - 1 - H)
        r, o = r[inside], o[inside]
        cnt = np.zeros((b_ - a, T + 1), dtype=np.int16)
        np.cumsum(np.isfinite(M[a:b_]), axis=1, dtype=np.int16, out=cnt[:, 1:])
        rl = r - a
        n_in = cnt[rl, o + 1] - cnt[rl, o - L + 1]
        n_rec = cnt[rl, o + 1] - cnt[rl, o - cfg["recent_window"] + 1]
        n_tg = cnt[rl, o + H + 1] - cnt[rl, o + 1]
        del cnt, rl
        keep = (n_in >= cfg["min_input"]) & (n_rec >= cfg["min_recent"]) & (n_tg >= cfg["min_target"])
        out["candidates"] += int(len(r))
        out["passed"] += int(keep.sum())
        if sampling:
            u = ((crc[r] * np.uint64(2654435761) + o.astype(np.uint64) * np.uint64(2246822519)) % np.uint64(2 ** 32)) / 2.0 ** 32
            keep &= ((o <= cut) & (n_tg >= cfg["train_min_target"]) & (u < p_tr)) | ((o >= cut + H) & (u < p_te))
        r, o, n_tg = r[keep], o[keep], n_tg[keep]
        if len(r):
            R.append(r), O.append(o), NT.append(n_tg)
            KIN.append(M[r[:, None], o[:, None] + np.arange(-(L - 1), 1)[None, :]])
            KTG.append(M[r[:, None], o[:, None] + np.arange(1, H + 1)[None, :]])
    r = np.concatenate(R) if R else np.zeros(0, dtype=np.int64)
    o = np.concatenate(O) if O else np.zeros(0, dtype=np.int64)
    n_tg = np.concatenate(NT) if NT else np.zeros(0, dtype=np.int16)
    if len(r):
        out["kin"] = np.concatenate(KIN)
        out["ktg"] = np.concatenate(KTG)
        del KIN, KTG
        kk = np.asarray(keys, dtype=object)[r]
        out["pdf"] = pd.DataFrame({"meter_id": kk.astype(str), "consumer_id": kk.astype(str),
                                   "origin_date": d0 + pd.to_timedelta(o, unit="D"), "coato_code": str(code),
                                   "housing_type_id": pd.Series(house[r]).astype("Int64").astype(str).replace("<NA>", np.nan).to_numpy(),
                                   "tgt_days": n_tg.astype("int16")})

    if cfg["seg_full"] and len(r):                       # segment features: v8's own code path
        s0 = max(0, int((pd.Timestamp(cfg["seg_start"]) - d0).days))
        s1 = int((pd.Timestamp(cfg["cutoff"]) - d0).days)
        months = dates[s0:s1 + 1].month.values
        t = wt[0, s0:s1 + 1]
        rows = []
        for i in np.unique(r):
            y = M[i, s0:s1 + 1].astype("float64")
            if np.isfinite(y).sum() < cfg["seg_min_days"]:
                rows.append((keys[i], int(np.isfinite(y).sum()), np.nan, np.nan, np.nan, np.nan))
                continue
            y_clean, _rep, masked = wp.clean_daily_series(y)
            y_clean = np.where(masked, np.nan, y_clean)
            n_obs = int(np.isfinite(y_clean).sum())
            if n_obs < cfg["seg_min_days"]:
                rows.append((keys[i], n_obs, np.nan, np.nan, np.nan, np.nan))
                continue
            seas = wp._seasonality_strength(y_clean, months)
            ts_, r2 = wp._weekly_thermal_fit(y_clean, t)
            rows.append((keys[i], n_obs, np.log1p(max(float(np.nanmedian(y_clean)), 0.0)), ts_, r2, seas))
        out["seg"] = pd.DataFrame(rows, columns=["meter_id", "n_obs_days", "log_median_kwh", "thermal_share", "r2",
                                                 "seasonality_strength"]).set_index("meter_id")
    out["seconds"] = _time.time() - t0
    try:
        import resource
        out["peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except Exception:                                    # noqa: BLE001
        out["peak_gb"] = float("nan")
    out["rows_customers"] = n
    return out
