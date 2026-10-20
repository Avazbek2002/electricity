#!/usr/bin/env python3
"""
Bottom-up NETWORK-NODE test of the household NLinear models: forecast every household, add the forecasts up per
transformer, feeder, substation, district and the country, and score the sums (system WAPE at each level).

Why a separate test: household_nlinear.py scores a 5% hash sample of test windows with origins staggered per customer, so
on a given target day a feeder holds only a few scored households. Here EVERY household is forecast from the SAME origins,
so a node's forecast is the sum over all its households, as it would be in production.

  * model: the weights household_nlinear.py saved (--weights-dir: weights_<variant>.npz + nlinear_info.json). The run's
    settings (dates, priors, level rule, weather simulation seed) are read from nlinear_info.json, so features match training.
  * origins: the same date for every household, every --origin-step days from cutoff + 60 days to the last origin with 60
    target days in the data (default 14 days: 2025-10-31 .. 2026-04-17, 13 origins). No test origin was seen in training.
  * households: every household of the district passing the INPUT rules at that origin (>= 60 of 90 input days observed,
    >= 20 of the last 30). No target rule: at forecast time nobody knows which target days will be observed.
  * a node-day sums forecasts and actuals of the households whose target day is observed (a household with a missing
    actual is left out of both sides that day, so the comparison is like for like).
  * network ids: the enriched dataset's last-known transformer_id / feeder_id / substation_id. Households with a missing or
    negative id at a level (and, for transformers, on a placeholder transformer) form one "unassigned" node per district
    at that level; they are reported separately, not in the node statistics. A feeder or substation id in several
    districts is one node (its sums are added across districts).
  * system WAPE of a node at horizon h = sum over origins |P - A| / sum over origins A (P, A = node sums on the target day
    origin + h). Levels report: pooled over nodes (energy-weighted) = sum |P - A| / sum A, the median node and the 90th
    percentile, over nodes with >= --min-households scored households per day, and the median node by size band.
  * baselines: persistence30 (the 30-day level), persistence7 (the 7-day mean) and the thermal baseline.

  python core/node_syswape.py --weights-dir ~/work/hh_nlinear --out-dir ~/work/hh_nodes --workers 4

Outputs (<out-dir>), aggregates only (no customer keys; nodes with fewer than --min-households households are not listed):
node_syswape_summary.csv, node_syswape_by_size.csv, node_metrics.csv (one row per node, model, horizon), run_log.txt.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import household_nlinear as hn                                              # noqa: E402

LEVELS = ("transformer", "feeder", "substation", "district")
HORIZONS = (1, 7, 14, 30, 60)
BANDS = (10, 25, 50, 100, 250, 500, 1000, 5000)
MISSING = -(2 ** 62)


# =============================================================================
# One district (worker process)
# =============================================================================
def read_mapping(code, cfg, keys, since):
    """Last-known transformer / feeder / substation id and placeholder flag per customer of `keys` (MISSING if none)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    import enriched_io as eio
    fs, root = eio.open_root(cfg["root"])
    dset = eio.district_daily(fs, root, code)
    want = ["customer_key", "transformer_id", "feeder_id", "substation_id", "placeholder_transformer"]
    cols = [c for c in want if c in dset.schema.names]
    pos = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    out = {c: np.full(n, MISSING, np.int64) for c in ("transformer_id", "feeder_id", "substation_id")}
    ph = np.zeros(n, bool)
    seen = np.zeros(n, bool)
    flt = pads.field("date") >= pd.Timestamp(since).date()
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144).to_batches():
        if not b.num_rows:
            continue
        de = pc.dictionary_encode(b.column("customer_key"))
        idx = np.array([pos.get(str(k), -1) for k in de.dictionary.to_pylist()], dtype=np.int64)[
            de.indices.to_numpy(zero_copy_only=False)]
        first = np.zeros(len(idx), bool)
        _, fi = np.unique(idx, return_index=True)
        first[fi] = True
        sel = first & (idx >= 0)
        sel[sel] = ~seen[idx[sel]]
        if not sel.any():
            continue
        tgt = idx[sel]
        for c in out:
            if c in cols:
                out[c][tgt] = pc.fill_null(pc.cast(b.column(c), pa.int64()), MISSING).to_numpy(zero_copy_only=False)[sel]
        if "placeholder_transformer" in cols:
            ph[tgt] = pc.fill_null(b.column("placeholder_transformer"), False).to_numpy(zero_copy_only=False)[sel]
        seen[tgt] = True
    return out, ph


def node_worker(code, cfg):
    t0 = time.time()
    import whole_pipeline_v8 as wp
    for k, v in cfg["wp"].items():
        setattr(wp, k, v)
    L, H = wp.LOOKBACK, wp.HORIZON
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T = (d1 - d0).days + 1
    keys, M, wt, house = wp._read_enriched_district(code, cfg)
    res = dict(code=str(code), customers=len(keys), windows=0, seconds=0.0)
    if not len(keys):
        return res
    origins = np.asarray(cfg["origins"], dtype=np.int64)
    nO, hz = len(origins), np.asarray(cfg["horizons"]) - 1
    since = d0 + pd.Timedelta(days=int(origins.min()) - L)
    ids, ph = read_mapping(code, cfg, keys, since)

    # node of every customer, per level (local index; the last group = "unassigned")
    groups = {}
    for lvl, col in (("transformer", "transformer_id"), ("feeder", "feeder_id"), ("substation", "substation_id")):
        v = ids[col].copy()
        bad = v < 0
        if lvl == "transformer":
            bad |= ph
        v[bad] = MISSING
        u, g = np.unique(v, return_inverse=True)
        names = [("unassigned:" + str(code)) if x == MISSING else str(int(x)) for x in u]
        groups[lvl] = (g, names)
    groups["district"] = (np.zeros(len(keys), np.int64), [str(code)])

    dates = d0 + pd.to_timedelta(np.arange(T), unit="D")
    have = np.isfinite(wt).any(axis=0)
    wdf = pd.DataFrame({"coato_id": str(code), "d": dates[have], "t": wt[0, have], "tmin": wt[1, have], "tmax": wt[2, have]})
    clim = wp.build_climate(wdf)
    od = np.arange(L - 1, T - H)                                     # the same simulated forecast as household_nlinear.py
    u_fc = pd.DataFrame({"coato_code": str(code), "origin_date": d0 + pd.to_timedelta(od, unit="D")})
    seed = int(cfg["seed"]) + zlib.crc32(str(code).encode()) % 1_000_003
    with hn._quiet():
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

    models = list(cfg["models"])
    W = {v: np.load(pth)["W"].astype("float64") for v, pth in cfg["weights"].items()}
    acc = {lvl: dict(A=np.zeros((len(nm), nO, len(hz))), N=np.zeros((len(nm), nO, len(hz))),
                     P={m: np.zeros((len(nm), nO, len(hz))) for m in models}) for lvl, (g, nm) in groups.items()}
    scored = np.zeros(len(keys), bool)
    blk = int(cfg["block"])
    for a in range(0, len(keys), blk):
        b_ = min(len(keys), a + blk)
        r = np.repeat(np.arange(a, b_), nO)
        oi = np.tile(np.arange(nO), b_ - a)
        o = origins[oi]
        cnt = np.zeros((b_ - a, T + 1), dtype=np.int16)
        np.cumsum(np.isfinite(M[a:b_]), axis=1, dtype=np.int16, out=cnt[:, 1:])
        rl = r - a
        n_in = cnt[rl, o + 1] - cnt[rl, o - L + 1]
        n_rec = cnt[rl, o + 1] - cnt[rl, o - wp.RECENT_WINDOW + 1]
        del cnt, rl
        keep = (n_in >= wp.MIN_INPUT_DAYS) & (n_rec >= wp.MIN_RECENT_DAYS)
        r, o, oi = r[keep], o[keep], oi[keep]
        if not len(r):
            continue
        res["windows"] += int(len(r))
        scored[r] = True
        origin = d0 + pd.to_timedelta(o, unit="D")
        kin = M[r[:, None], o[:, None] + np.arange(-(L - 1), 1)[None, :]].astype("float64")
        ktg = M[r[:, None], o[:, None] + np.arange(1, H + 1)[None, :]].astype("float64")
        ht = pd.Series(house[r]).astype("Int64").astype(str).replace("<NA>", np.nan).to_numpy()
        pdf = pd.DataFrame({"meter_id": np.asarray(keys, dtype=object)[r].astype(str), "consumer_id": np.nan,
                            "origin_date": origin, "coato_code": str(code), "housing_type_id": ht})
        with hn._quiet():
            meta, feats, base, _ = wp.build_features_chunked(
                pdf, kin, ktg, clim, fc_grid=grid, fc_grid_mm=grid_mm, chunk=cfg["chunk"], forecast_mode="archive",
                normal_mode="loyo", beta_h_prior=cfg["prior_h"], beta_c_prior=cfg["prior_c"],
                beta_h_prior_mm=cfg["prior_h_mm"], beta_c_prior_mm=cfg["prior_c_mm"], meter_segment=None)
        ly = hn.year_ago(M, r, o, wp.BLOCKS)
        lev = hn.level_of(kin, cfg["norm_days"])
        ok = np.isfinite(lev) & (lev > cfg["min_level"])
        levs = np.where(ok, lev, 1.0)
        X, names = hn.design(wp, feats, ly, kin, levs, L)
        if names != cfg["names"]:
            raise RuntimeError(f"district {code}: the inputs differ from the trained model's (check --weights-dir)")
        fallback = np.nan_to_num(lev)[:, None]
        base = np.asarray(base, dtype="float64")
        P = {}
        for v, Wv in W.items():
            cols = cfg["cols"][v]
            Pv = hn.to_kwh(v, X[:, cols] @ Wv, np.where(ok, lev, 0.0), base)
            Pv = np.where(ok[:, None], Pv, base if v.startswith("thermal") else fallback)
            P[v] = np.where(np.isfinite(Pv), Pv, fallback)[:, hz]
        P["persistence30"] = np.repeat(feats["level_kwh"].to_numpy(dtype="float64")[:, None], len(hz), 1)
        P["persistence7"] = np.repeat(feats["kwh_mean_7d"].to_numpy(dtype="float64")[:, None], len(hz), 1)
        P["thermal_baseline"] = base[:, hz]
        for m in ("persistence30", "persistence7", "thermal_baseline"):
            P[m] = np.where(np.isfinite(P[m]), P[m], fallback)
        A = ktg[:, hz]
        obs = np.isfinite(A)
        A0 = np.where(obs, A, 0.0)
        for lvl, (g, nm) in groups.items():
            G = len(nm)
            flat = (g[r][:, None] * nO + oi[:, None]) * len(hz) + np.arange(len(hz))[None, :]
            size = G * nO * len(hz)
            ac = acc[lvl]
            ac["A"] += np.bincount(flat.ravel(), weights=A0.ravel(), minlength=size).reshape(G, nO, len(hz))
            ac["N"] += np.bincount(flat.ravel(), weights=obs.ravel().astype(float), minlength=size).reshape(G, nO, len(hz))
            for m in models:
                ac["P"][m] += np.bincount(flat.ravel(), weights=np.where(obs, P[m], 0.0).ravel(),
                                          minlength=size).reshape(G, nO, len(hz))
        del kin, ktg, pdf, meta, feats, base, X, P
    for lvl, (g, nm) in groups.items():
        acc[lvl]["names"] = nm
        acc[lvl]["households"] = np.bincount(g, minlength=len(nm))
        acc[lvl]["scored"] = np.bincount(g[scored], minlength=len(nm))
    res["acc"] = acc
    res["seconds"] = time.time() - t0
    return res


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    import whole_pipeline_v8 as wp
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights-dir", required=True, help="household_nlinear.py --out-dir of the trained run")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=None, help="enriched dataset root (default: the training run's)")
    p.add_argument("--coatos", default=None, help="comma list of district codes (default: every finished district)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--origin-step", type=int, default=14, help="days between test origins (all households share them)")
    p.add_argument("--first-origin", default=None, help="default: training cutoff + 60 days")
    p.add_argument("--last-origin", default=None, help="default: the last date with 60 target days in the data")
    p.add_argument("--min-households", type=int, default=10, help="nodes with fewer scored households per day are not "
                   "summarised or listed (noise; the guide's disclosure rule asks for >= 10)")
    p.add_argument("--block", type=int, default=5000, help="customers per block inside a worker (memory)")
    p.add_argument("--cache-dir", default=None, help="per-district node sums (resume); aggregates only")
    args = p.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)
    with hn._tee(os.path.join(out, "run_log.txt")):
        run(wp, args, out)


def run(wp, args, out):
    import hashlib
    import enriched_io as eio
    wd = os.path.abspath(os.path.expanduser(args.weights_dir))
    info = json.load(open(os.path.join(wd, "nlinear_info.json")))
    cfg = dict(info["config"])
    cfg["here"] = HERE
    if args.root:
        cfg["root"] = args.root
    cfg["block"] = args.block
    names = info["features"]["exogenous"]
    n_hist = int(info["features"]["history"])
    p_all = n_hist + len(names) + 1
    weights, cols = {}, {}
    for v in hn.VARIANTS:
        pth = os.path.join(wd, f"weights_{v}.npz")
        if os.path.exists(pth):
            weights[v] = pth
            cols[v] = list(range(p_all)) if v.endswith("_x") else list(range(n_hist)) + [p_all - 1]
            if np.load(pth)["W"].shape[0] != len(cols[v]):
                raise SystemExit(f"{pth}: {np.load(pth)['W'].shape[0]} weights, expected {len(cols[v])}")
    if not weights:
        raise SystemExit(f"no weights_<variant>.npz in {wd}")
    d0 = pd.Timestamp(cfg["start"])
    T = (pd.Timestamp(cfg["end"]) - d0).days + 1
    H = wp.HORIZON
    first = pd.Timestamp(args.first_origin) if args.first_origin else pd.Timestamp(cfg["cutoff"]) + pd.Timedelta(days=H)
    last = pd.Timestamp(args.last_origin) if args.last_origin else d0 + pd.Timedelta(days=T - 1 - H)
    origins = list(range((first - d0).days, (last - d0).days + 1, args.origin_step))
    models = list(weights) + ["persistence30", "persistence7", "thermal_baseline"]
    cfg.update(origins=origins, horizons=list(HORIZONS), models=models, weights=weights, cols=cols, names=names)
    od = [(d0 + pd.Timedelta(days=x)).date() for x in origins]
    print(f"[config] weights {wd} ({', '.join(weights)}) | {len(origins)} origins {od[0]} .. {od[-1]} every "
          f"{args.origin_step} days | horizons {HORIZONS} | min households per node-day {args.min_households}")
    if pd.Timestamp(od[0]) < pd.Timestamp(cfg["cutoff"]) + pd.Timedelta(days=H):
        print("[config] WARNING: the first origin is less than 60 days after the training cutoff: its targets were trained on")
    fs, root = eio.open_root(cfg["root"])
    codes = eio.finished_districts(fs, root, set(args.coatos.split(",")) if args.coatos else None)
    cache = None
    if args.cache_dir:
        sig = hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if k != "here"}, sort_keys=True, default=str)
                           .encode()).hexdigest()[:12]
        cache = os.path.join(os.path.expanduser(args.cache_dir), f"node_syswape_{sig}")
        os.makedirs(cache, exist_ok=True)
        print(f"[cache] {cache}")
    results, todo = [], []
    for c in codes:
        pth = os.path.join(cache, f"{c}.pkl") if cache else None
        if pth and os.path.exists(pth):
            with open(pth, "rb") as fh:
                results.append(pickle.load(fh))
        else:
            todo.append(c)
    t0 = time.time()
    thr = max(1, (os.cpu_count() or 4) // args.workers)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)
    if todo:
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            pending = {ex.submit(node_worker, c, cfg): c for c in todo}
            k = 0
            while pending:
                done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for f in done:
                    code = pending.pop(f)
                    try:
                        res = f.result()
                    except Exception as e:                           # noqa: BLE001
                        raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                    del f
                    if cache:
                        tmp = os.path.join(cache, f"{code}.pkl.tmp")
                        with open(tmp, "wb") as fh:
                            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(tmp, os.path.join(cache, f"{code}.pkl"))
                    results.append(res)
                    k += 1
                    if k % 10 == 0 or k == len(todo) or k <= 3:
                        el = time.time() - t0
                        print(f"[districts] {k}/{len(todo)}: {sum(r.get('windows', 0) for r in results):,} household forecasts; "
                              f"{el / 60:.1f} min, about {el / k * (len(todo) - k) / 60:.1f} min left | main {wp._rss_gb():.1f} GB",
                              flush=True)
    report(results, models, args, out)
    print(f"\n[done] {out} in {(time.time() - t0) / 60:.1f} min")


def merge(results, lvl, models):
    """All districts' node sums for one level, nodes with the same id added together -> dict of arrays + names."""
    parts = [r["acc"][lvl] for r in results if "acc" in r]
    names = np.concatenate([np.asarray(p["names"], dtype=object) for p in parts])
    u, inv = np.unique(names.astype(str), return_inverse=True)

    def add(get):
        x = np.concatenate([get(p) for p in parts])
        o = np.zeros((len(u),) + x.shape[1:])
        np.add.at(o, inv, x)
        return o
    m = dict(names=u, A=add(lambda p: p["A"]), N=add(lambda p: p["N"]), households=add(lambda p: p["households"]),
             scored=add(lambda p: p["scored"]), P={k: add(lambda p, k=k: p["P"][k]) for k in models})
    return m


def node_table(m, lvl, models):
    """One row per node, model and horizon: |P - A| and A summed over origins, households scored per day."""
    rows = []
    has = m["N"] > 0
    hpd = np.where(has, m["N"], np.nan)
    with hn._warn_off():
        hh = np.nanmean(hpd, axis=1)                                 # (nodes, horizons)
    for k in models:
        err = np.where(has, np.abs(m["P"][k] - m["A"]), 0.0).sum(1)
        sgn = np.where(has, m["P"][k] - m["A"], 0.0).sum(1)
        act = np.where(has, m["A"], 0.0).sum(1)
        for j, h in enumerate(HORIZONS):
            rows.append(pd.DataFrame({"level": lvl, "node": m["names"], "model": k, "horizon": h, "abs_err": err[:, j],
                                      "signed": sgn[:, j], "actual": act[:, j], "households_per_day": hh[:, j],
                                      "households": m["households"], "scored_households": m["scored"],
                                      "days": has[:, :, j].sum(1)}))
    t = pd.concat(rows, ignore_index=True)
    t["unassigned"] = t["node"].str.startswith("unassigned:")
    t["sys_wape"] = t["abs_err"] / t["actual"].where(t["actual"] > 0)
    t["bias"] = t["signed"] / t["actual"].where(t["actual"] > 0)
    return t


def report(results, models, args, out):
    tabs = []
    for lvl in LEVELS:
        tabs.append(node_table(merge(results, lvl, models), lvl, models))
    t = pd.concat(tabs, ignore_index=True)
    # national: all households of all districts
    nm = merge(results, "district", models)
    one = {kk: nm[kk].sum(0, keepdims=True) for kk in ("A", "N", "households", "scored")}
    one["names"] = np.array(["national"])
    one["P"] = {k: v.sum(0, keepdims=True) for k, v in nm["P"].items()}
    t = pd.concat([t, node_table(one, "national", models)], ignore_index=True)
    ok = t[~t["unassigned"] & (t["households_per_day"] >= args.min_households) & (t["actual"] > 0)]
    ok[["level", "node", "model", "horizon", "households", "households_per_day", "days", "sys_wape", "bias"]].to_csv(
        os.path.join(out, "node_metrics.csv"), index=False)

    def summ(g):
        return pd.Series({"pooled": g["abs_err"].sum() / g["actual"].sum(), "median": g["sys_wape"].median(),
                          "p90": g["sys_wape"].quantile(0.9), "bias": g["signed"].sum() / g["actual"].sum(),
                          "nodes": int(len(g)), "median_households_per_day": g["households_per_day"].median()})
    s = ok.groupby(["level", "model", "horizon"]).apply(summ, include_groups=False).reset_index()
    s.to_csv(os.path.join(out, "node_syswape_summary.csv"), index=False)
    ok = ok.assign(band=pd.cut(ok["households_per_day"], list(BANDS) + [np.inf], right=False,
                               labels=[f"{a}-{b - 1}" for a, b in zip(BANDS, BANDS[1:])] + [f"{BANDS[-1]}+"]))
    b = ok[ok["level"].isin(["transformer", "feeder", "substation"])].groupby(
        ["level", "band", "model", "horizon"], observed=True).apply(summ, include_groups=False).reset_index()
    b.to_csv(os.path.join(out, "node_syswape_by_size.csv"), index=False)

    # ---- printed report
    tot = t[(t["model"] == models[0]) & (t["horizon"] == HORIZONS[0])]
    print(f"\n[data] {sum(r.get('customers', 0) for r in results):,} households in {len(results)} districts; "
          f"{sum(r.get('windows', 0) for r in results):,} household forecasts")
    for lvl in ("transformer", "feeder", "substation"):
        x = tot[tot["level"] == lvl]
        un = x[x["unassigned"]]["households"].sum()
        big = x[~x["unassigned"] & (x["households_per_day"] >= args.min_households)]
        print(f"  {lvl:11s}: {int((~x['unassigned']).sum()):,} nodes, {int(len(big)):,} with >= {args.min_households} scored "
              f"households per day ({big['households'].sum() / max(x['households'].sum(), 1):.1%} of households); unassigned "
              f"{un / max(x['households'].sum(), 1):.1%} of households")
    order = [m for m in models if m in s["model"].unique()]
    for lvl in ("national", "district", "substation", "feeder", "transformer"):
        x = s[s["level"] == lvl]
        if x.empty:
            continue
        n_nodes = int(x["nodes"].max())
        hh = x["median_households_per_day"].median()
        for col, lab in (("pooled", "pooled over nodes (energy-weighted)"), ("median", "median node")):
            if lvl == "national" and col == "median":
                continue
            print(f"\n=== {lvl}: system WAPE, {lab} | {n_nodes:,} nodes, median {hh:,.0f} households per day ===")
            print(x.pivot(index="model", columns="horizon", values=col).reindex(order).map(lambda v: f"{v:6.1%}").to_string())
    main_m = [m for m in ("nlinear_x", "persistence30") if m in models]
    for lvl in ("substation", "feeder", "transformer"):
        x = b[(b["level"] == lvl) & b["model"].isin(main_m) & b["horizon"].isin([7, 30])]
        if x.empty:
            continue
        x = x.assign(col=x["model"] + " t" + x["horizon"].astype(str))
        tab = x.pivot(index="band", columns="col", values="median")
        tab.insert(0, "nodes", x.groupby("band", observed=True)["nodes"].max().astype(int))
        print(f"\n=== {lvl}: median node system WAPE by size (scored households per day) ===")
        print(tab.to_string(formatters={c: (lambda v: f"{v:6.1%}") for c in tab.columns if c != "nodes"}))


if __name__ == "__main__":
    main()
