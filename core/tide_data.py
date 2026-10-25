"""Household training windows with AGGREGATE labels for TiDE (core/tide.py, models/tide_agg).

Before 2024-08 most meters were read every 4-5 days in whole kWh: the daily table has no clean day there, only
allocated long-gap energy (equal shares of a 4-92 day interval) and 2-3 day estimated runs. A daily model cannot learn
from those days without inventing a daily shape; this module keeps them as what they are: interval sums.

Per household-day the store keeps
    val    float32   observed kWh (kind 1) or the interval's equal daily share (kind 2), else NaN
    kind   int8      0 none, 1 complete_observed, 2 day of a reconstructed reading interval
    coff   int8      kind 2: days from this day to the interval's last day (the reading that closes it)
    ilen   int8      on an interval's LAST day: its length in days (2-92), else 0
    isum   float32   on an interval's last day: its exact energy (sum of the shares), else NaN
Intervals (HH-GAP-001-A1: allocated energy is never a daily label, but its sum over the whole interval is exact):
    - estimated runs: maximal runs of complete_with_estimated days (whole 2-3 day intervals; the run sum is exact)
    - long gaps: days with long_gap_allocation_status = allocated, one covering interval (long_gap_interval_count 1),
      no clean label that day, every day of [long_gap_interval_start, long_gap_interval_end) present
    - an interval cut by the read window, an excluded range or a clean day is dropped
A training window (issue date o; inputs o - 90 .. o - 1, targets o + 1 .. o + 60, as the harness):
    inputs   observed days, plus (coarse_inputs = old: only for o before the regime cutoff, which is what the harness
             serves: its panels carry complete_observed only) interval days whose interval CLOSED by o - 1; an interval
             still open at o - 1 is masked (rule 3: no same-interval information across the issue date)
    targets  daily rows for observed target days; interval rows for intervals lying wholly inside o + 1 .. o + 60;
             nothing for missing days (no "missing = residual 0")
The gather (one batch of windows) is written once for numpy and torch, so training and serving build identical inputs:
serving wraps a harness Panel into a one-district store (store_from_panel) and gathers with o = the issue date.

No customer key leaves a reading worker: the store holds row numbers only.
"""
import hashlib
import time

import numpy as np
import pandas as pd

KIND_NONE, KIND_OBS, KIND_INT = 0, 1, 2
L_DEFAULT, H = 90, 60
LABEL_OBS, LABEL_EST = "complete_observed", "complete_with_estimated"
N_CAL = 14                       # dow one-hot 7, doy sin/cos x 2 harmonics 4, holiday, extra day off, Ramadan
N_COV = N_CAL + 3 + 3 + 1        # + temperature normals (mean, min, max) + anomalies + "weather observed"
ANOM = slice(N_CAL + 3, N_CAL + 6)
WX_SEEN = N_CAL + 6
HOUSING_IDS = 16                 # housing_type_id 0..14 -> embedding row, 15 = unknown / other


def stable_share(key, salt="tide-val"):
    """A stable number in [0, 1) per customer key (validation households)."""
    return int(hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).hexdigest(), 16) / 2 ** 64


# ----------------------------------------------------------------------------- reading one district
def read_raw(code, cfg):
    """One district (customer buckets cfg["buckets"]) -> dict(keys, obs, est, lga, lgs, lge, lgok (n, T), housing (n,),
    wt (3, T)). Day indices relative to cfg["start"]."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    import enriched_io as eio
    fs, root = eio.open_root(cfg["uri"], cfg.get("env_prefix", ""))
    dset = eio.district_daily(fs, root, code)
    names = set(dset.schema.names)
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    if cfg.get("buckets") is not None and "customer_bucket" in names:
        flt = flt & pads.field("customer_bucket").isin([int(b) for b in cfg["buckets"]])
    want = ["customer_key", "date", "label_state", "kwh", "kwh_nano", "kwh_long_gap_allocated", "long_gap_interval_start",
            "long_gap_interval_end", "long_gap_interval_count", "long_gap_allocation_status", "housing_type_id",
            "t_mean", "t_min", "t_max"]
    cols = [c for c in want if c in names]
    key_of, keys = {}, []
    cap = 1024
    arr = dict(obs=np.full((cap, T), np.nan, np.float32), est=np.full((cap, T), np.nan, np.float32),
               lga=np.full((cap, T), np.nan, np.float32), lgs=np.full((cap, T), -1, np.int32),
               lge=np.full((cap, T), -1, np.int32))
    house = np.full(cap, np.nan)
    hday = np.full(cap, -1, np.int64)
    wt = np.full((3, T), np.nan)

    def days_of(col):
        return pc.cast(pc.cast(col, pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0

    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        if not b.num_rows:
            continue
        de = pc.dictionary_encode(b.column("customer_key"))
        idx = np.empty(len(de.dictionary), np.int64)
        for j, s in enumerate(de.dictionary.to_pylist()):
            v = key_of.get(s)
            if v is None:
                v = key_of[s] = len(keys)
                keys.append(s)
            idx[j] = v
        c = idx[de.indices.to_numpy(zero_copy_only=False)]
        if len(keys) > cap:
            new = max(len(keys), 2 * cap)
            for k, a in arr.items():
                fill = np.nan if a.dtype.kind == "f" else -1
                arr[k] = np.vstack([a, np.full((new - cap, T), fill, a.dtype)])
            house = np.r_[house, np.full(new - cap, np.nan)]
            hday = np.r_[hday, np.full(new - cap, -1, np.int64)]
            cap = new
        day = days_of(b.column("date"))
        for i, col in enumerate(("t_mean", "t_min", "t_max")):
            if col in cols:
                np.fmax.at(wt[i], day, np.asarray(b.column(col).to_numpy(zero_copy_only=False), dtype=float))
        if "housing_type_id" in cols:
            try:
                hv = pc.cast(b.column("housing_type_id"), pa.float64()).to_numpy(zero_copy_only=False)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                hv = pd.to_numeric(pd.Series(b.column("housing_type_id").to_pylist()), errors="coerce").to_numpy(float)
            m = np.isfinite(hv) & (day >= hday[c])
            hday[c[m]], house[c[m]] = day[m], hv[m]
        lab = pc.fill_null(b.column("label_state"), "").to_numpy(zero_copy_only=False)
        if "kwh_nano" in cols:
            kv = pc.cast(b.column("kwh_nano"), pa.float64()).to_numpy(zero_copy_only=False) / 1e9
        else:
            kv = pc.cast(b.column("kwh"), pa.float64()).to_numpy(zero_copy_only=False)
        kv = np.asarray(kv, dtype=float)
        ok = np.isfinite(kv)
        for name, lv in (("obs", LABEL_OBS), ("est", LABEL_EST)):
            m = ok & (lab == lv)
            arr[name][c[m], day[m]] = kv[m]
        if "kwh_long_gap_allocated" in cols and "long_gap_interval_start" in cols:
            la = np.asarray(pc.cast(b.column("kwh_long_gap_allocated"), pa.float64()).to_numpy(zero_copy_only=False),
                            dtype=float)
            st = (pc.fill_null(b.column("long_gap_allocation_status"), "").to_numpy(zero_copy_only=False)
                  if "long_gap_allocation_status" in cols else np.full(len(day), "allocated"))
            cnt = (pc.fill_null(pc.cast(b.column("long_gap_interval_count"), pa.int64()), 0).to_numpy(zero_copy_only=False)
                   if "long_gap_interval_count" in cols else np.ones(len(day), np.int64))
            m = np.isfinite(la) & (st == "allocated") & (cnt == 1) & ~(ok & np.isin(lab, (LABEL_OBS, LABEL_EST)))
            if m.any():
                s0 = pc.fill_null(pc.cast(pc.cast(b.column("long_gap_interval_start"), pa.date32()), pa.int32()),
                                  -10 ** 6).to_numpy(zero_copy_only=False).astype(np.int64) - e0
                s1 = pc.fill_null(pc.cast(pc.cast(b.column("long_gap_interval_end"), pa.date32()), pa.int32()),
                                  -10 ** 6).to_numpy(zero_copy_only=False).astype(np.int64) - e0
                m &= (s0 > -10 ** 5) & (s1 > -10 ** 5)
                arr["lga"][c[m], day[m]] = la[m]
                arr["lgs"][c[m], day[m]] = np.clip(s0[m], -1, 10 ** 6)
                arr["lge"][c[m], day[m]] = np.clip(s1[m], -1, 10 ** 6)
    n = len(keys)
    out = {k: a[:n] for k, a in arr.items()}
    out.update(keys=keys, housing=house[:n], wt=wt, T=T)
    return out


def _runs(mask):
    """Maximal runs of True along axis 1 of a (n, T) bool -> (row, first col, last col) arrays."""
    n, T = mask.shape
    z = np.zeros((n, 1), bool)
    f = np.concatenate([z, mask, z], axis=1).astype(np.int8)
    dif = np.diff(f, axis=1)
    r0, c0 = np.nonzero(dif == 1)
    r1, c1 = np.nonzero(dif == -1)
    return r0, c0, c1 - 1               # np.nonzero is row-major: starts and ends pair up in order


def build_intervals(raw, exclude=()):
    """raw (read_raw) -> the store arrays val, kind, coff, ilen, isum (n, T) + counts. exclude: [(day0, day1)] day
    index ranges removed completely (inputs and labels)."""
    obs, est, lga = raw["obs"].copy(), raw["est"].copy(), raw["lga"].copy()
    lgs, lge = raw["lgs"], raw["lge"]
    n, T = obs.shape
    for a, b in exclude:
        a, b = max(0, a), min(T - 1, b)
        if a <= b:
            obs[:, a:b + 1] = est[:, a:b + 1] = lga[:, a:b + 1] = np.nan
    val = np.where(np.isfinite(obs), obs, np.nan).astype(np.float32)
    kind = np.where(np.isfinite(obs), KIND_OBS, KIND_NONE).astype(np.int8)
    coff = np.zeros((n, T), np.int8)
    ilen = np.zeros((n, T), np.int8)
    isum = np.full((n, T), np.nan, np.float32)
    stats = dict(est_runs=0, est_dropped=0, lg_intervals=0, lg_dropped=0)

    # -- estimated runs (whole 2-3 day intervals; a run cut by the window edge is dropped)
    em = np.isfinite(est) & ~np.isfinite(obs)
    xd = np.zeros(T + 2, bool)                        # excluded days (+ a False pad on both sides)
    for a_, b_ in exclude:
        xd[max(0, a_) + 1:min(T - 1, b_) + 2] = True
    r, a, b = _runs(em)
    keep = (a > 0) & (b < T - 1) & (b - a + 1 >= 2) & (b - a + 1 <= 92) & ~xd[a] & ~xd[b + 2]
    stats["est_runs"], stats["est_dropped"] = int(keep.sum()), int((~keep).sum())
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(np.nan_to_num(est.astype(np.float64)) * em, axis=1)], axis=1)
    _fill(val, kind, coff, ilen, isum, r[keep], a[keep], b[keep], cs, est)
    # -- long gaps: runs of allocated days with the same (start, end), complete inside the window
    lm = np.isfinite(lga) & ~np.isfinite(obs) & ~em
    key = np.where(lm, lgs.astype(np.int64) * 100_003 + lge, -1)
    brk = np.ones((n, T), bool)
    brk[:, 1:] = key[:, 1:] != key[:, :-1]
    # runs of lm with no key change inside: split the mask at key changes
    seg = lm.copy()
    r, a, b = _runs_split(seg, brk)
    if len(r):
        s_ = lgs[r, a]
        e_ = lge[r, a] - 1                            # end is exclusive in the daily allocation convention
        full = (s_ == a) & (e_ == b) & (a > 0) & (b < T - 1) & (b - a + 1 >= 2) & (b - a + 1 <= 92)
        stats["lg_intervals"], stats["lg_dropped"] = int(full.sum()), int((~full).sum())
        cs = np.concatenate([np.zeros((n, 1)), np.cumsum(np.nan_to_num(lga.astype(np.float64)) * lm, axis=1)], axis=1)
        _fill(val, kind, coff, ilen, isum, r[full], a[full], b[full], cs, lga)
    return dict(val=val, kind=kind, coff=coff, ilen=ilen, isum=isum), stats


def _runs_split(mask, brk):
    """Runs of True in mask that also break where brk is True (a new run starts at every brk position)."""
    n, T = mask.shape
    start = mask & (brk | ~np.concatenate([np.zeros((n, 1), bool), mask[:, :-1]], axis=1))
    end = mask & np.concatenate([start[:, 1:] | ~mask[:, 1:], np.ones((n, 1), bool)], axis=1)
    r0, c0 = np.nonzero(start)
    r1, c1 = np.nonzero(end)
    return r0, c0, c1


def _fill(val, kind, coff, ilen, isum, r, a, b, cs, src):
    """Write intervals (row r, days a..b) into the store arrays; the share of each day = its own allocated value."""
    if not len(r):
        return
    ln = b - a + 1
    tot = cs[r, b + 1] - cs[r, a]
    ilen[r, b] = ln.astype(np.int8)
    isum[r, b] = tot.astype(np.float32)
    for k in range(int(ln.max())):                     # day a + k of every interval at least k + 1 long
        m = ln > k
        rr, dd = r[m], a[m] + k
        val[rr, dd] = src[rr, dd]
        kind[rr, dd] = KIND_INT
        coff[rr, dd] = (b[m] - dd).astype(np.int8)


# ----------------------------------------------------------------------------- calendar, normals, covariates
_CAL_CACHE = {}


def calendar(dates):
    """(len(dates), N_CAL) float32: dow one-hot, doy harmonics, holiday / extra day off / Ramadan (standing calendar,
    known years ahead). Cached per (first day, length): serving asks for the same spans many times."""
    d = pd.DatetimeIndex(dates)
    key = (d[0], len(d)) if len(d) else None
    if key in _CAL_CACHE:
        return _CAL_CACHE[key]
    if len(_CAL_CACHE) > 4096:
        _CAL_CACHE.clear()
    out = _calendar(d)
    _CAL_CACHE[key] = out
    return out


def _calendar(d):
    from harness.testset import standing_calendar
    out = np.zeros((len(d), N_CAL), np.float32)
    out[np.arange(len(d)), d.dayofweek] = 1
    a = 2 * np.pi * (d.dayofyear.to_numpy() - 1) / 365.25
    out[:, 7], out[:, 8], out[:, 9], out[:, 10] = np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)
    cal = standing_calendar(d)
    out[:, 11], out[:, 12], out[:, 13] = cal["hol"].to_numpy(), cal["xoff"].to_numpy(), cal["ram"].to_numpy()
    return out


def normals(wt, dates, smooth=7):
    """(3, 366) day-of-year normals of t_mean / t_min / t_max from the days given (circular +/- smooth days); missing
    -> the series mean, else 15 / 9 / 21 C."""
    d = pd.DatetimeIndex(dates)
    doy = np.minimum(d.dayofyear.to_numpy() - 1, 365)
    out = np.zeros((3, 366))
    for i in range(3):
        v = np.asarray(wt[i], float)
        ok = np.isfinite(v)
        s = np.bincount(doy[ok], weights=v[ok], minlength=366)
        c = np.bincount(doy[ok], minlength=366).astype(float)
        k = np.arange(-smooth, smooth + 1)
        S = sum(np.roll(s, j) for j in k)
        C = sum(np.roll(c, j) for j in k)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = S / C
        fb = float(np.nanmean(v)) if ok.any() else (15.0, 9.0, 21.0)[i]
        out[i] = np.where(np.isfinite(m), m, fb)
    return out.astype(np.float32)


def covariates(dates, wt, norm):
    """Per-day covariates of one district (len(dates), N_COV): calendar, normals / 10, anomaly (actual - normal) / 10
    and a 'weather observed' flag. The gather zeroes anomaly and flag on target days (unknown at the issue date)."""
    d = pd.DatetimeIndex(dates)
    doy = np.minimum(d.dayofyear.to_numpy() - 1, 365)
    out = np.zeros((len(d), N_COV), np.float32)
    out[:, :N_CAL] = calendar(d)
    nm = norm[:, doy].T
    out[:, N_CAL:N_CAL + 3] = nm / 10.0
    if wt is not None:
        w = np.full((len(d), 3), np.nan)
        k = min(len(d), wt.shape[1])
        w[:k] = np.asarray(wt[:, :k], float).T
        an = (w - nm) / 10.0
        seen = np.isfinite(an).all(axis=1)
        out[:, ANOM] = np.where(np.isfinite(an), an, 0.0)
        out[:, WX_SEEN] = seen
    return out


# ----------------------------------------------------------------------------- windows
def candidate_origins(st, cfg, o_lo, o_hi, rng):
    """Issue dates o (store day index) for each household, every cfg stride days with a random phase, where the inputs
    have enough known days and the targets enough labelled days -> (row, o) int arrays."""
    L = int(cfg["lookback"])
    kind = st["kind"]
    n, T = kind.shape
    reg = int(cfg["regime_idx"])
    obs = kind == KIND_OBS
    inter = kind == KIND_INT
    known_old = obs | inter if cfg["coarse_inputs"] in ("old", "all") else obs
    known_new = obs | inter if cfg["coarse_inputs"] == "all" else obs
    lab = obs | inter if cfg["interval_labels"] else obs

    def csum(m):
        c = np.zeros((n, T + 1), np.int32)
        np.cumsum(m, axis=1, out=c[:, 1:])
        return c
    co, cn, cl = csum(known_old), csum(known_new), csum(lab)
    stride = int(cfg["stride"])
    rows, origins = [], []
    o_lo, o_hi = max(o_lo, L + 1), min(o_hi, T - 2)
    if o_hi < o_lo:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    phase = rng.integers(0, stride, n)
    for p in range(stride):
        r = np.flatnonzero(phase == p)
        if not len(r):
            continue
        o = np.arange(o_lo + ((p - o_lo) % stride), o_hi + 1, stride)
        if not len(o):
            continue
        last = o - 1
        c = np.where((o < reg)[None, :], co[r][:, last + 1] - co[r][:, last + 1 - L],
                     cn[r][:, last + 1] - cn[r][:, last + 1 - L])
        rec = np.where((o < reg)[None, :], co[r][:, last + 1] - co[r][:, last + 1 - 30],
                       cn[r][:, last + 1] - cn[r][:, last + 1 - 30])
        hi = np.minimum(o + H, T - 1)
        tl = cl[r][:, hi + 1] - cl[r][:, o + 1]
        ok = (c >= int(cfg["min_input_days"])) & (rec >= int(cfg["min_recent_days"])) & (tl >= int(cfg["min_target_days"]))
        rr, oo = np.nonzero(ok)
        rows.append(r[rr])
        origins.append(o[oo])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(rows).astype(np.int64), np.concatenate(origins).astype(np.int64)


class _Ops:
    """The few array operations the gather needs, for numpy and torch."""

    def __init__(self, xp, device=None):
        self.xp, self.device = xp, device
        self.torch = xp.__name__ == "torch"

    def arange(self, a, b):
        return self.xp.arange(a, b, device=self.device) if self.torch else np.arange(a, b)

    def f32(self, x):
        return x.float() if self.torch else x.astype(np.float32)

    def i64(self, x):
        return x.long() if self.torch else x.astype(np.int64)

    def where(self, c, a, b):
        return self.xp.where(c, a, b)

    def isfinite(self, x):
        return self.xp.isfinite(x)

    def nan_to_num(self, x):
        return self.xp.nan_to_num(x, nan=0.0)

    def cumsum(self, x):
        return self.xp.cumsum(x, 1) if self.torch else np.cumsum(x, axis=1)

    def zeros_col(self, x):
        return self.xp.zeros((x.shape[0], 1), dtype=x.dtype, device=self.device) if self.torch else \
            np.zeros((x.shape[0], 1), x.dtype)

    def cat1(self, a, b):
        return self.xp.cat([a, b], 1) if self.torch else np.concatenate([a, b], axis=1)

    def log(self, x):
        return self.xp.log(x)

    def stack(self, xs):
        return self.xp.stack(xs, 1) if self.torch else np.stack(xs, axis=1)


def gather(st, h, o, cfg, xp=np, device=None):
    """One batch of windows (store rows h, issue day indices o) -> model inputs and targets (scaled by s, the mean of
    the known input days). Works on numpy arrays (tests, serving on CPU) and torch tensors (training on the GPU)."""
    op = _Ops(xp, device)
    L = int(cfg["lookback"])
    past = o[:, None] + op.arange(-L, 0)[None, :]                      # o - L .. o - 1
    fut = o[:, None] + op.arange(1, H + 1)[None, :]                    # o + 1 .. o + H
    hh = h[:, None]
    v_p, k_p, c_p = st["val"][hh, past], st["kind"][hh, past], st["coff"][hh, past]
    old = o < int(cfg["regime_idx"])
    coarse_ok = {"old": old, "all": old | ~old, "none": old & ~old}[cfg["coarse_inputs"]]
    closed = (op.i64(past) + op.i64(c_p)) <= (o[:, None] - 1)         # the interval's reading came by o - 1
    known = (k_p == KIND_OBS) | ((k_p == KIND_INT) & closed & coarse_ok[:, None])
    vz = op.where(known, op.nan_to_num(v_p), 0.0)
    m = op.f32(known)
    n_known = m.sum(1)
    s = vz.sum(1) / n_known.clip(min=1.0)
    s = s.clip(min=float(cfg.get("min_scale", 0.05)))
    y = vz / s[:, None]
    cflag = op.f32(known & (k_p == KIND_INT))
    # covariates: past and target days (day o itself is neither); anomalies unknown on target days
    dist = st["dist"][h]
    cov_p = st["cov"][dist[:, None], past]
    cov_f = st["cov"][dist[:, None], fut]
    if op.torch:
        cov_f = cov_f.clone()
    else:
        cov_f = cov_f.copy()
    cov_f[..., ANOM] = 0.0
    cov_f[..., WX_SEEN] = 0.0
    # targets
    v_f, k_f = st["val"][hh, fut], st["kind"][hh, fut]
    mt = op.f32(k_f == KIND_OBS)
    yt = op.where(k_f == KIND_OBS, op.nan_to_num(v_f), 0.0) / s[:, None]
    il = op.i64(st["ilen"][hh, fut])
    pos = op.arange(0, H)[None, :]
    im = (il > 0) & ((pos - il + 1) >= 0)                             # the interval lies wholly inside the targets
    if not cfg.get("interval_labels", True):
        im = im & False
    isum = op.where(im, op.nan_to_num(st["isum"][hh, fut]), 0.0) / s[:, None]
    stat = [op.log(s), op.f32(n_known) / L, cflag.sum(1) / L]
    stat.append(op.f32(old) if cfg.get("regime_flag") else op.f32(old) * 0.0)
    return dict(y=op.f32(y), m=m, c=cflag, cov_p=cov_p, cov_f=cov_f, s=s, stat=op.stack(stat),
                house=st["house"][h], dist=st["demb"][h] if "demb" in st else dist, yt=op.f32(yt), mt=mt, il=il, im=op.f32(im), isum=op.f32(isum),
                old=old)


# ----------------------------------------------------------------------------- serving: a Panel as a store
def housing_index(h):
    h = np.asarray(h, float)
    return np.where(np.isfinite(h) & (h >= 0) & (h < HOUSING_IDS - 1), np.nan_to_num(h), HOUSING_IDS - 1).astype(np.int64)


def store_from_panel(panel, norm, dist_index, cfg):
    """A harness Panel -> (store, h, o) for gather: kind 1 on observed days (the panel has complete_observed only),
    the district covariates with the fitted normals, target days padded."""
    n, W = panel.kwh.shape
    Tp = W + H + 2
    val = np.full((n, Tp), np.nan, np.float32)
    val[:, :W] = panel.kwh
    kind = np.where(np.isfinite(val), KIND_OBS, KIND_NONE).astype(np.int8)
    dates = pd.date_range(panel.dates[0], periods=Tp, freq="D")
    wt = np.full((3, Tp), np.nan)
    wt[:, :W] = panel.weather
    st = dict(val=val, kind=kind, coff=np.zeros((n, Tp), np.int8), ilen=np.zeros((n, Tp), np.int8),
              isum=np.full((n, Tp), np.nan, np.float32), house=housing_index(panel.housing),
              dist=np.zeros(n, np.int64), demb=np.full(n, dist_index, np.int64),   # covariate row / embedding
              cov=covariates(dates, wt, norm)[None])
    o = int((panel.origin - panel.dates[0]).days)                     # = W (history ends on o - 1)
    c = dict(cfg, regime_idx=o + 1 if panel.origin < pd.Timestamp(cfg["regime_cutoff"]) else 0)
    return st, np.arange(n, dtype=np.int64), np.full(n, o, np.int64), c


# ----------------------------------------------------------------------------- one district for training
def district_worker(code, cfg):
    """Read one district, reconstruct its reading intervals, pick candidate windows. Returns arrays only (no keys):
    store rows, housing, weather, train / validation origins."""
    t0 = time.time()
    raw = read_raw(code, cfg)
    n, T = raw["obs"].shape
    out = dict(code=str(code), n=0, T=T, seconds=0.0)
    if not n:
        return out
    d0 = pd.Timestamp(cfg["start"])
    exc = [((pd.Timestamp(a) - d0).days, (pd.Timestamp(b) - d0).days) for a, b in cfg.get("exclude_ranges") or []]
    st, stats = build_intervals(raw, exc)
    is_val = np.array([stable_share(k) < float(cfg["val_households"]) for k in raw["keys"]])
    # keep households with any label in the window
    has = (st["kind"] != KIND_NONE).sum(1) >= int(cfg["min_input_days"])
    keep = np.flatnonzero(has)
    st = {k: v[keep] for k, v in st.items()}
    is_val = is_val[keep]
    house = housing_index(raw["housing"][keep])
    rng = np.random.default_rng(int(stable_share(code, f"tide-phase-{cfg.get('seed', 42)}") * 2 ** 32))
    c = dict(cfg)
    hr, ho = candidate_origins(st, c, 0, int(cfg["train_hi"]), rng)
    tr = ~is_val[hr]
    vr, vo = candidate_origins(st, dict(c, stride=int(cfg.get("val_stride", 7))), int(cfg["val_lo"]),
                               int(cfg["train_hi"]), rng)
    va = is_val[vr]
    out.update(n=len(keep), store=st, house=house, wt=raw["wt"], train=(hr[tr], ho[tr]), val=(vr[va], vo[va]),
               stats=dict(stats, households=int(len(keep)), val_households=int(is_val.sum()),
                          old_train=int((ho[tr] < int(cfg["regime_idx"])).sum()), train=int(tr.sum()),
                          val=int(va.sum())), seconds=time.time() - t0)
    return out
