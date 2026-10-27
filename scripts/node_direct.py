#!/usr/bin/env python
"""Direct substation / feeder forecasts: a model on the node series itself (no households), after the district champion
recipe of the household methodology (27 Oct 2026): target y(d) / m28, LightGBM with linear trees per horizon block,
Huber loss on the far blocks, the far blocks averaged with the B1 reference.

    python scripts/test_set/build_node_test_set.py --extras-only --workers 12   # once: node_obs + district_weather
    python scripts/node_direct.py                       # fits the 4 quarterly refits, scores dev (and keeps test aside)
    python scripts/compare_node_runs.py --role dev --latest node_direct xgboost_cal_edd nlinear_x

One model per horizon block for ALL nodes of the node set (dev + test substations and their feeders), trained on their own
history before the refit (pooled: a substation series is much shorter and noisier than a district's). Model choice on
the dev substations only; the test substations are scored in the same run but only reported with --report-test.

Leak rules (ML guide part 5, the node harness):
  - issue date o knows the node series up to o - 1; on o - 2 and o - 1 only the observed part (T2obs = n_rows x kWh of
    the one-day readings / their count; targets/node_obs.csv.gz), the full T2 up to o - 3 (2-3 day estimated readings
    covering o - 2 / o - 1 are taken on o or later)
  - a refit at R trains on (o, h) pairs whose target day d <= R - 3 (all estimated readings closed)
  - weather: the district's observed daily mean up to o - 1; climate of a day = mean of the same days +-7 in earlier years
    (at least 358 days before it); no target-day weather (a direct model with the AIFS forecast is the next step)
  - calendar: the standing calendar (known in advance); distances to events capped at 15 days
  - substitution test: every input after a day T randomised, forecasts issued <= T + 1 unchanged

Writes ~/work/eval/runs/<ts>_node_direct_nodes-dev-test/ with forecasts/node_direct, forecasts/B1 (node.csv.gz +
forecast_meta.json, the layout of the node harness: compare_node_runs.py, node_blend.py and node_year_chart.py read it)
and scores/<role>/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
import harness  # noqa: E402,F401
from harness import nodeset as ns  # noqa: E402
from harness import testset as ts  # noqa: E402

H = 60
BLOCKS = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30), (31, 60)]
HUBER_FROM = 15                 # Huber loss from this horizon on (rare large errors weigh less on the far blocks)
REGIME = pd.Timestamp("2024-08-01")
CAP = 15                        # calendar distances capped (a longer reach lets trees memorise single past windows)
LGB = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, bagging_fraction=0.8,
           bagging_freq=1, feature_fraction=0.8, linear_tree=True, linear_lambda=1.0, deterministic=True,
           force_row_wise=True, seed=0, verbose=-1)
N_TREES = 400
FEATURES = ["h", "lastv", "r7", "r14", "r91", "sd28", "samewd", "ly_ratio", "ly_exact", "ly_ratio2", "ly_missing",
            "wd", "month", "doy_s", "doy_c", "hol", "xoff", "ram", "dto_hol", "dfr_hol", "newyear", "htype", "ram_day",
            "since_ram", "wt7", "whdd7", "wanom7", "clim_d", "clim_delta", "log_hh", "hh_trend", "is_feeder", "old"]


# ----------------------------------------------------------------------------- inputs
def load_inputs(test_dir, spec):
    t = ns.load_targets(test_dir)
    nodes = ns.load_nodes(test_dir)
    op = os.path.join(test_dir, "targets", "node_obs.csv.gz")
    wp_ = os.path.join(test_dir, "targets", "district_weather.csv.gz")
    obs = pd.read_csv(op, dtype={"node": str}, parse_dates=["date"]) if os.path.exists(op) else None
    wx = pd.read_csv(wp_, dtype={"coato": str}, parse_dates=["date"]) if os.path.exists(wp_) else None
    if obs is None:
        print("[inputs] no targets/node_obs.csv.gz: o - 2 and o - 1 are left out of the history (build_node_test_set.py "
              "--extras-only adds them; about 1 point at h1)")
    if wx is None:
        print("[inputs] no targets/district_weather.csv.gz: weather features are empty")
    return t, nodes, obs, wx


def calendar_arrays(dates):
    """Per calendar day: the target-day calendar features (standing calendar, known in advance)."""
    cal = ts.standing_calendar(dates).set_index("date")
    idx = pd.DatetimeIndex(dates)
    off = ((cal["hol"] + cal["xoff"]) > 0).to_numpy()
    n = len(idx)
    nxt, prv = np.full(n, CAP), np.full(n, CAP)
    last = -10_000
    for i in range(n):                                      # days since the last holiday / day off
        if off[i]:
            last = i
        prv[i] = min(i - last, CAP)
    nx = 10_000
    for i in range(n - 1, -1, -1):                          # days to the next one
        if off[i]:
            nx = i
        nxt[i] = min(nx - i, CAP)
    ram = cal["ram"].to_numpy()
    ram_day = np.zeros(n)
    since = np.full(n, CAP)
    run, end = 0, -10_000
    for i in range(n):
        run = run + 1 if ram[i] else 0
        ram_day[i] = run
        if i > 0 and ram[i - 1] and not ram[i]:
            end = i - 1
        since[i] = min(i - end, CAP) if not ram[i] else 0
    hn = cal["hname"].to_numpy()
    htype = np.select([hn == "ramazan", hn == "kurban", hn != "none"], [2, 3, 1], 0)
    doy = idx.dayofyear.to_numpy()
    return dict(wd=idx.dayofweek.to_numpy(), month=idx.month.to_numpy(), doy_s=np.sin(2 * np.pi * doy / 365.25),
                doy_c=np.cos(2 * np.pi * doy / 365.25), hol=cal["hol"].to_numpy(), xoff=cal["xoff"].to_numpy(), ram=ram,
                dto_hol=nxt, dfr_hol=prv, newyear=(((idx.month == 12) & (idx.day >= 30)) | ((idx.month == 1) & (idx.day <= 3))).astype(int),
                htype=htype, ram_day=ram_day, since_ram=since)


def climate(t):
    """Normal of each day = mean of the observed temperature on the same days +-7 in the 1-4 earlier years (>= 358 days
    before the day: always known on the issue date of any target day)."""
    n = len(t)
    v = np.nan_to_num(t)
    c = np.r_[0, np.cumsum(np.isfinite(t))]
    s = np.r_[0, np.cumsum(v)]
    tot, cnt = np.zeros(n), np.zeros(n)
    i = np.arange(n)
    for k in range(1, 5):
        a, b = i - 365 * k - 7, i - 365 * k + 7
        ok = b >= 0
        a2, b2 = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
        tot += np.where(ok, s[b2 + 1] - s[a2], 0.0)
        cnt += np.where(ok, c[b2 + 1] - c[a2], 0.0)
    return np.where(cnt >= 5, tot / np.maximum(cnt, 1), np.nan)


# ----------------------------------------------------------------------------- per node arrays
def node_series(g, T, start, obs_g):
    """Day-indexed arrays of one node: S (history, T2; old regime filled with T3), OBS (T2obs), LY (T2 else T3, for the
    year-ago ratios), Y (scored target), hh (households in span)."""
    i = ((g["date"] - start).dt.days).to_numpy()
    ok = (i >= 0) & (i < T)
    arr = lambda col: _put(i[ok], g[col].to_numpy(float)[ok], T)   # noqa: E731
    t2, t3, hh = arr("T2"), arr("T3"), arr("n_rows")
    cov3 = arr("coverage3")
    reason = np.full(T, "", dtype=object)
    reason[i[ok]] = g["flag_reason"].fillna("").astype(str).to_numpy()[ok]
    flag = np.zeros(T, bool)
    flag[i[ok]] = g["flag"].to_numpy(bool)[ok]
    bad_ctx = np.array(["crisis" in r or "interruption" in r for r in reason])
    old = (start + pd.to_timedelta(np.arange(T), unit="D")) < REGIME
    t3ok = np.where((cov3 >= 0.7) & ~bad_ctx, t3, np.nan)
    S = np.where(np.isfinite(t2), t2, np.where(old, t3ok, np.nan))
    LY = np.where(np.isfinite(t2), t2, t3ok)
    Y = np.where(~flag & np.isfinite(t2), t2, np.where(old & np.isfinite(t3ok), t3ok, np.nan))
    OBS = np.full(T, np.nan)
    if obs_g is not None and len(obs_g):
        j = ((obs_g["date"] - start).dt.days).to_numpy()
        k = (j >= 0) & (j < T)
        OBS[j[k]] = obs_g["T2obs"].to_numpy(float)[k]
    return S, OBS, LY, Y, hh


def _put(i, v, T):
    a = np.full(T, np.nan)
    a[i] = v
    return a


def _wsum(cs, a, b):
    """sum over [a, b] (inclusive) from a cumulative array with a leading 0; empty ranges -> 0."""
    n = len(cs) - 1
    a2, b2 = np.clip(a, 0, n), np.clip(b + 1, 0, n)
    return np.where(b2 > a2, cs[b2] - cs[a2], 0.0)


def level_features(S, OBS, O, safe=2):
    """Per issue index o: m28 and the history features, from S up to o - 1 - safe and OBS on the last `safe` days."""
    v = np.nan_to_num(S)
    f = np.isfinite(S).astype(float)
    cs, cn, cq = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(f)], np.r_[0, np.cumsum(v * v)]
    n = len(S)
    tail = [np.where((O - k >= 0) & (O - k < n), OBS[np.clip(O - k, 0, n - 1)], np.nan) for k in range(1, safe + 1)]
    tv = np.nansum(np.vstack(tail), axis=0) if safe else 0.0
    tq = np.nansum(np.vstack(tail) ** 2, axis=0) if safe else 0.0
    tn = np.isfinite(np.vstack(tail)).sum(axis=0) if safe else 0

    def win(w):
        s = _wsum(cs, O - w, O - 1 - safe) + tv
        c = _wsum(cn, O - w, O - 1 - safe) + tn
        q = _wsum(cq, O - w, O - 1 - safe) + tq
        return s, c, q
    s28, c28, q28 = win(28)
    m28 = np.where(c28 >= 20, s28 / np.maximum(c28, 1), np.nan)
    out = dict(m28=m28)
    for w in (7, 14, 91):
        s, c, _ = win(w)
        out[f"r{w}"] = np.where(c >= max(3, w // 3), s / np.maximum(c, 1), np.nan) / m28
    var = q28 / np.maximum(c28, 1) - (s28 / np.maximum(c28, 1)) ** 2
    out["sd28"] = np.sqrt(np.maximum(var, 0)) / m28
    last = np.full(len(O), np.nan)
    for k in range(1, safe + 1):                           # the latest known day
        last = np.where(np.isfinite(last), last, tail[k - 1])
    s_last = np.where((O - 1 - safe >= 0), S[np.clip(O - 1 - safe, 0, n - 1)], np.nan)
    out["lastv"] = np.where(np.isfinite(last), last, s_last) / m28
    return out


def year_ago(LY):
    """c7 (centred 7-day mean) and m28 (trailing 28-day mean) of the year-ago series, day-indexed."""
    s = pd.Series(LY)
    return (s.rolling(7, center=True, min_periods=5).mean().to_numpy(),
            s.rolling(28, min_periods=20).mean().to_numpy())


def _at(a, i):
    n = len(a)
    return np.where((i >= 0) & (i < n), a[np.clip(i, 0, n - 1)], np.nan)


# ----------------------------------------------------------------------------- the panel
def build_panel(t, nodes, obs, wx, spec, origins, safe=2):
    """(node, origin, horizon) rows with the features, the ratio target and B1 for every origin in `origins`."""
    start = pd.Timestamp(spec["data"]["start"])
    end = pd.Timestamp(spec["data"]["end"])
    T = (end - start).days + 1
    dates = pd.date_range(start, end + pd.Timedelta(days=H + 1), freq="D")
    cal = calendar_arrays(dates)
    O = np.array(sorted({(pd.Timestamp(o) - start).days for o in origins}), dtype=np.int64)
    O = O[(O >= 91) & (O < T)]                              # 91 days of history (year-ago features may be missing)
    Hh = np.arange(1, H + 1)
    temps = {}
    if wx is not None:
        for c, g in wx.groupby("coato"):
            temps[str(c)] = _put(((g["date"] - start).dt.days).to_numpy(), g["t_mean"].to_numpy(float), T)
    clims = {c: climate(v) for c, v in temps.items()}
    obs_by = dict(tuple(obs.groupby("node"))) if obs is not None else {}
    lv = nodes.set_index("node")["level"].to_dict()
    co = nodes.set_index("node")["coato"].astype(str).to_dict()
    parts = []
    for node, g in t.groupby("node", sort=True):
        if node not in lv:
            continue
        S, OBS, LY, Y, hh = node_series(g, T, start, obs_by.get(node))
        L = level_features(S, OBS, O, safe)
        c7, m28ly = year_ago(LY)
        oo = np.repeat(O, H)
        hh_ = np.tile(Hh, len(O))
        d = oo + hh_
        rep = lambda a: np.repeat(a, H)                      # noqa: E731
        m28 = rep(L["m28"])
        f = {"node": node, "o": oo, "h": hh_, "m28": m28}
        for k in ("lastv", "r7", "r14", "r91", "sd28"):
            f[k] = rep(L[k])
        k7 = 7 * np.ceil((hh_ + 1) / 7).astype(int)          # same weekday as d, latest known
        kd = d - k7
        sw = np.where(kd >= oo - safe, _at(OBS, kd), _at(S, kd))
        sw = np.where(np.isfinite(sw), sw, _at(S, kd - 7))
        f["samewd"] = sw / m28
        lyr = [np.clip(_at(c7, d - 364 * k) / _at(m28ly, oo - 1 - 364 * k), 0.6, 1.7) for k in (1, 2, 3)]
        f["ly_ratio"], f["ly_ratio2"] = lyr[0], lyr[1]
        f["ly_exact"] = np.clip(_at(LY, d - 364) / _at(m28ly, oo - 1 - 364), 0.6, 1.7)
        f["ly_missing"] = (~np.isfinite(lyr[0])).astype(float)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            b1r = np.nanmean(np.vstack(lyr), axis=0)
        f["B1"] = m28 * np.where(np.isfinite(b1r), b1r, 1.0)
        for k, a in cal.items():
            f[k] = a[d]
        tw = temps.get(co[node])
        if tw is not None:
            cl = clims[co[node]]
            w7 = np.vstack([_at(tw, O - k) for k in range(1, 8)])
            c7w = np.vstack([_at(cl, O - k) for k in range(1, 8)])
            with np.errstate(invalid="ignore"):
                wt7 = np.nanmean(w7, axis=0)
                whdd = np.nanmean(np.maximum(18 - w7, 0), axis=0) * 7
                cl7 = np.nanmean(c7w, axis=0)
            f["wt7"], f["whdd7"], f["wanom7"] = rep(wt7), rep(whdd), rep(wt7 - cl7)
            f["clim_d"] = _at(cl, d)
            f["clim_delta"] = f["clim_d"] - rep(cl7)
        else:
            for k in ("wt7", "whdd7", "wanom7", "clim_d", "clim_delta"):
                f[k] = np.full(len(d), np.nan)
        hh_now = _at(hh, O - 1 - safe)
        f["log_hh"] = rep(np.log(np.maximum(hh_now, 1)))
        f["hh_trend"] = rep(hh_now / _at(hh, O - 31 - safe))
        f["is_feeder"] = float(lv[node] == "feeder")
        f["old"] = ((start + pd.to_timedelta(d, unit="D")) < REGIME).astype(float)
        f["y"] = _at(Y, d) / m28
        f["d"] = d
        parts.append(pd.DataFrame(f))
    P = pd.concat(parts, ignore_index=True)
    P["origin"] = start + pd.to_timedelta(P["o"], unit="D")
    P["date"] = start + pd.to_timedelta(P["d"], unit="D")
    for c in FEATURES:
        P[c] = P[c].astype("float32")
    return P


def _fill(X):
    """Linear trees need finite inputs: neutral values for the gaps (ratios 1, temperatures their column median)."""
    X = X.copy()
    for c in ("lastv", "r7", "r14", "r91", "samewd", "ly_ratio", "ly_exact", "ly_ratio2", "hh_trend"):
        X[c] = X[c].fillna(1.0)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    return X


# ----------------------------------------------------------------------------- fit / predict
def fit_block(P, a, b, n_trees=N_TREES):
    import lightgbm as lgb
    q = P[(P["h"] >= a) & (P["h"] <= b) & np.isfinite(P["y"]) & np.isfinite(P["m28"])]
    q = q[(q["y"] > 0.2) & (q["y"] < 5)]                       # broken days (outages, data cuts)
    params = dict(LGB)
    if a >= HUBER_FROM:
        params.update(objective="huber", alpha=0.05)
    ds = lgb.Dataset(_fill(q[FEATURES]), label=q["y"].to_numpy(), free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=n_trees), len(q)


def run_refits(P, refits, issue, b1_long=0.5, n_trees=N_TREES, log=print):
    """Quarterly refits: models fitted on pairs with target day <= R - 3, predicting the issue dates up to the next refit."""
    refits = [pd.Timestamp(r) for r in refits]
    out = []
    for k, R in enumerate(refits):
        nxt = refits[k + 1] if k + 1 < len(refits) else pd.Timestamp.max
        pred = P[(P["origin"] >= R) & (P["origin"] < nxt) & P["origin"].isin(issue) & np.isfinite(P["m28"])]
        if not len(pred):
            continue
        train = P[P["date"] <= R - pd.Timedelta(days=3)]
        fc = np.full(len(pred), np.nan)
        t0 = time.time()
        for a, b in BLOCKS:
            m, n = fit_block(train, a, b, n_trees)
            sel = ((pred["h"] >= a) & (pred["h"] <= b)).to_numpy()
            fc[sel] = m.predict(_fill(pred.loc[sel, FEATURES]))
            log(f"[fit] refit {R.date()} block h{a}-{b}: {n:,} training pairs")
        f = pred["m28"].to_numpy() * fc
        far = pred["h"].to_numpy() >= 31
        f[far] = (1 - b1_long) * f[far] + b1_long * pred["B1"].to_numpy()[far]
        out.append(pd.DataFrame({"node": pred["node"].to_numpy(), "origin": pred["origin"].to_numpy(),
                                 "horizon": pred["h"].to_numpy(), "forecast": f, "B1": pred["B1"].to_numpy()}))
        log(f"[fit] refit {R.date()}: {len(pred):,} forecasts in {time.time() - t0:.0f} s")
    return pd.concat(out, ignore_index=True)


def substitution_test(t, nodes, obs, wx, spec, refit, issue, T_day, n_trees=50):
    """Randomise every input after T (targets, observed part, weather); forecasts issued <= T + 1 must not change."""
    rng = np.random.default_rng(12345)
    T_day = pd.Timestamp(T_day)
    sel = [o for o in issue if T_day - pd.Timedelta(days=2) <= o <= T_day + pd.Timedelta(days=4)]
    origins_tr = pd.date_range(pd.Timestamp(spec["data"]["start"]), refit, freq="7D")
    t2, o2, w2 = t.copy(), obs.copy() if obs is not None else None, wx.copy() if wx is not None else None
    late = t2["date"] > T_day
    for c in ("T2", "T3", "n_rows"):
        t2[c] = t2[c].astype(float)
        t2.loc[late, c] = rng.gamma(2.0, 400.0, late.sum())
    if o2 is not None:
        lo = o2["date"] > T_day
        o2.loc[lo, "T2obs"] = rng.gamma(2.0, 400.0, lo.sum())
    if w2 is not None:
        lw = w2["date"] > T_day
        w2.loc[lw, "t_mean"] = rng.normal(10, 15, lw.sum())
    res = []
    for tt, oo, ww in ((t, obs, wx), (t2, o2, w2)):
        P = build_panel(tt, nodes, oo, ww, spec, list(origins_tr) + sel)
        res.append(run_refits(P, [refit], pd.DatetimeIndex(sel), n_trees=n_trees, log=lambda *_: None)
                   .sort_values(["node", "origin", "horizon"]).reset_index(drop=True))
    a, b = res
    keep = a["origin"] <= T_day + pd.Timedelta(days=1)
    same = float(np.nanmax(np.abs(a.loc[keep, "forecast"] - b.loc[keep, "forecast"]))) if keep.any() else 0.0
    later = float(np.nanmax(np.abs(a.loc[~keep, "forecast"] - b.loc[~keep, "forecast"]))) if (~keep).any() else np.nan
    return dict(passed=bool(same == 0.0 and keep.any()), T=str(T_day.date()),
                detail=f"T = {T_day.date()}: max change of forecasts issued <= T+1: {same:.3g}; issued >= T+2: {later:.3g}")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    p.add_argument("--test-dir", default=None)
    p.add_argument("--eval-dir", default="~/work/eval")
    p.add_argument("--stride", type=int, default=2, help="training origins every N days (2025 issue dates: every day)")
    p.add_argument("--trees", type=int, default=N_TREES)
    p.add_argument("--b1-long", type=float, default=0.5, help="weight of B1 on h31-60 (the champion: 0.5)")
    p.add_argument("--report-test", action="store_true", help="also print the test substations (once, chosen variant)")
    p.add_argument("--no-substitution-test", action="store_true")
    p.add_argument("--name", default="node_direct")
    a = p.parse_args(argv)
    spec = ns.load_spec(a.spec)
    test_dir = os.path.expanduser(a.test_dir or os.path.join("~/work/test_set", spec["version"]))
    ns.check_same_spec(spec, ns.read_manifest(test_dir))
    t, nodes, obs, wx = load_inputs(test_dir, spec)
    roles = ["dev", "test"]
    issue = pd.DatetimeIndex(sorted(set().union(*[set(ns.issue_dates(spec, r)) for r in roles])))
    start = pd.Timestamp(spec["data"]["start"])
    origins = list(pd.date_range(start, issue.min() - pd.Timedelta(days=1), freq=f"{a.stride}D")) + list(issue)
    git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"[run] {a.name} | {spec['version']} | {len(nodes)} nodes | refits {spec['refits']} | {len(issue)} issue dates "
          f"| training origins every {a.stride} days | commit {git}", flush=True)
    sub = None
    if not a.no_substitution_test:
        one = nodes[nodes["coato"] == sorted(nodes["coato"].astype(str))[0]]
        sub = substitution_test(t[t["node"].isin(one["node"])], one, obs, wx, spec, pd.Timestamp(spec["refits"][0]),
                                issue, pd.Timestamp(spec["refits"][0]) + pd.Timedelta(days=5))
        print(f"[leakage] {'PASSED' if sub['passed'] else 'FAILED'}: {sub['detail']}", flush=True)
        if not sub["passed"]:
            raise SystemExit("[leakage] forecasts depend on data after the issue date - 1")
    t0 = time.time()
    P = build_panel(t, nodes, obs, wx, spec, origins)
    print(f"[panel] {len(P):,} (node, origin, horizon) rows, {len(FEATURES)} features in {time.time() - t0:.0f} s", flush=True)
    F = run_refits(P, spec["refits"], issue, a.b1_long, a.trees)
    run = os.path.join(os.path.expanduser(a.eval_dir), "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{a.name}_nodes-dev-test")
    meta = dict(model=a.name, test_set_version=spec["version"], test_dir=test_dir, role="dev+test", roles=roles,
                git_commit=git, scaling=dict(method="direct", safe_lag=ns.EST_SAFE_LAG), substitution_test=sub,
                obs_tail=obs is not None, weather="none (past observed + climate)", blocks=BLOCKS, trees=a.trees,
                b1_long=a.b1_long, stride=a.stride, features=FEATURES)
    for name, col in ((a.name, "forecast"), ("B1", "B1")):
        d = os.path.join(run, "forecasts", name)
        os.makedirs(d, exist_ok=True)
        F[["node", "origin", "horizon"]].assign(forecast=F[col]).to_csv(os.path.join(d, "node.csv.gz"), index=False,
                                                                          float_format="%.6f")
        json.dump(dict(meta, model=name), open(os.path.join(d, "forecast_meta.json"), "w"), indent=2, default=str)
    targets = t
    for role in (["dev", "test"] if a.report_test else ["dev"]):
        want = set(nodes.loc[nodes["role"] == role, "node"])
        b0 = ns.b0_forecasts(targets, spec, nodes=want, issue=ns.issue_dates(spec, role))
        models = {a.name: F[["node", "origin", "horizon", "forecast"]],
                  "B1": F[["node", "origin", "horizon"]].assign(forecast=F["B1"]),
                  ns.REFERENCE: b0[["node", "origin", "horizon", "forecast"]]}
        res = ns.score(models, targets, spec, role, nodes)
        text = ns.report(res, list(models))
        print(f"\n===== {role.upper()} =====\n{text}")
        ns.save(res, os.path.join(run, "scores", role), text)
    print(f"\n[done] {run}")
    return run


if __name__ == "__main__":
    main()
