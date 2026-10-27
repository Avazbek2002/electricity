#!/usr/bin/env python
"""
Build the substation / feeder evaluation set once (configs/test_set_substations_2025.yaml -> ~/work/test_set/<version>/):
node-day targets T2 / T3 with the guide's flags for every selected substation and its feeders (>= 50 households).

    python scripts/test_set/build_node_test_set.py --workers 12

Reads only the districts of the spec (36), and in them only the rows of the selected substations and feeders.
Written: node-day means scaled to the households in span (T2, T3), counts and flags; nothing for node-days with fewer
than 10 known households (T2 / T3 NaN, flagged few_known); no customer keys.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from harness import nodeset as ns  # noqa: E402
from harness import testset as ts  # noqa: E402
import enriched_io as eio  # noqa: E402  (harness puts core/ on the path)

LABEL_OBS, LABEL_EST = "complete_observed", "complete_with_estimated"
WANT = ["date", "label_state", "kwh", "kwh_nano", "is_long_gap_allocated", "kwh_long_gap_allocated", "is_right_edge",
        "crisis_context", "data_interruption", "substation_id", "feeder_id", "t_mean"]
KEYS = ("n_rows", "n_obs", "n_est", "n_lg", "kwh_obs", "kwh_est", "kwh_lg", "n_right", "crisis", "interruption")


def district_nodes(code, cfg):
    """One district -> daily counts and kWh sums by day state for each of its nodes (row-level network ids)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    t0 = time.time()
    d = cfg["data"]
    subs, feeds = cfg["subs"], cfg["feeds"]                  # {substation_id: node index}, {feeder_id: node index}
    K = len(cfg["names"])
    fs, root = eio.open_root(d["uri"], d.get("env_prefix", ""))
    dset = eio.district_daily(fs, root, code)
    cols = [c for c in WANT if c in dset.schema.names]
    d0, d1 = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    acc = {k: np.zeros(K * T) for k in KEYS}
    tmean = np.full(T, np.nan)                               # the district's daily mean temperature (same on every row)
    def vals_for(name, ids):                                 # the filter values in the column's own type
        typ = dset.schema.field(name).type
        return pa.array([str(i) for i in ids]) if pa.types.is_string(typ) else pa.array(list(ids), type=typ)
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date()) & (
        pads.field("substation_id").isin(vals_for("substation_id", subs)) |
        pads.field("feeder_id").isin(vals_for("feeder_id", feeds)))
    sid_arr = np.array(sorted(subs), dtype=np.int64)
    fid_arr = np.array(sorted(feeds), dtype=np.int64)
    s_idx = np.array([subs[s] for s in sid_arr], dtype=np.int64)
    f_idx = np.array([feeds[f] for f in fid_arr], dtype=np.int64)
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        if not b.num_rows:
            continue
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0

        def col(name, kind=float):
            if name not in cols:
                return np.zeros(len(day), dtype=kind)
            a = b.column(name)
            if kind is bool:
                return pc.fill_null(a, False).to_numpy(zero_copy_only=False).astype(bool)
            return np.nan_to_num(np.asarray(a.to_numpy(zero_copy_only=False), dtype=float))

        def ints(name):
            return pc.fill_null(pc.cast(b.column(name), pa.int64()), -1).to_numpy(zero_copy_only=False)
        kwh = col("kwh_nano") / 1e9 if "kwh_nano" in cols else col("kwh")
        lab = b.column("label_state").to_numpy(zero_copy_only=False)
        obs, est = lab == LABEL_OBS, lab == LABEL_EST
        lg = col("is_long_gap_allocated", bool)
        vals = dict(n_rows=np.ones(len(day)), n_obs=obs.astype(float), n_est=est.astype(float), n_lg=lg.astype(float),
                    kwh_obs=np.where(obs, kwh, 0.0), kwh_est=np.where(est, kwh, 0.0),
                    kwh_lg=np.where(lg, col("kwh_long_gap_allocated"), 0.0),
                    n_right=col("is_right_edge", bool).astype(float))
        if "t_mean" in cols:
            tv = np.asarray(b.column("t_mean").to_numpy(zero_copy_only=False), dtype=float)
            ok = (day >= 0) & (day < T) & np.isfinite(tv)
            np.fmax.at(tmean, day[ok], tv[ok])
        flags = dict(crisis=col("crisis_context", bool).astype(float), interruption=col("data_interruption", bool).astype(float))
        for ids, keys_, idx in ((ints("substation_id"), sid_arr, s_idx), (ints("feeder_id"), fid_arr, f_idx)):
            pos = np.searchsorted(keys_, ids)
            pos = np.clip(pos, 0, max(len(keys_) - 1, 0))
            hit = (len(keys_) > 0) & (keys_[pos] == ids) & (day >= 0) & (day < T)
            if not hit.any():
                continue
            flat = idx[pos[hit]] * T + day[hit]
            for k, v in vals.items():
                acc[k] += np.bincount(flat, weights=v[hit], minlength=K * T)
            for k, v in flags.items():
                acc[k] = np.maximum(acc[k], np.bincount(flat, weights=v[hit], minlength=K * T))
    parts = []
    dates = pd.date_range(d0, d1, freq="D")
    for i, name in enumerate(cfg["names"]):
        df = pd.DataFrame({k: v[i * T:(i + 1) * T] for k, v in acc.items()})
        df.insert(0, "date", dates)
        df.insert(1, "node", name)
        for k, n in (("kwh_obs", "n_obs"), ("kwh_est", "n_est"), ("kwh_lg", "n_lg")):
            df.loc[df[n] == 0, k] = np.nan
        df.loc[df["n_rows"] == 0, "n_rows"] = np.nan
        parts.append(df)
    return dict(code=str(code), counts=pd.concat(parts, ignore_index=True), seconds=time.time() - t0,
                weather=pd.DataFrame({"coato": str(code), "date": dates, "t_mean": tmean}),
                peak_bytes=ts.peak_rss_bytes())


def _obs_only(g):
    """T2 from the observed one-day readings only (T2obs = n_rows x kWh_obs / n_obs): the part of o - 1 and o - 2 that is
    known on o (a 2-3 day estimated reading covering them is taken later). NaN below the disclosure floor."""
    n_obs = g["n_obs"].where(g["n_obs"] >= ns.MIN_KNOWN)
    with np.errstate(invalid="ignore", divide="ignore"):
        t2o = g["n_rows"] * g["kwh_obs"] / n_obs
    return pd.DataFrame({"date": g["date"], "node": g["node"], "T2obs": t2o, "n_obs": g["n_obs"].fillna(0).astype(int)})


def _write_extras(out, obs, wx):
    """targets/node_obs.csv.gz (node-day T2 from observed readings only) and targets/district_weather.csv.gz (district
    daily mean temperature of the enriched data) for the direct node models (scripts/node_direct.py)."""
    os.makedirs(os.path.join(out, "targets"), exist_ok=True)
    if obs:
        o = pd.concat(obs, ignore_index=True).sort_values(["node", "date"])
        o.loc[o["n_obs"] < ns.MIN_KNOWN, "n_obs"] = 0
        o.to_csv(os.path.join(out, "targets", "node_obs.csv.gz"), index=False, float_format="%.10g")
    if wx:
        w = pd.concat(wx, ignore_index=True).dropna(subset=["t_mean"]).sort_values(["coato", "date"])
        w.to_csv(os.path.join(out, "targets", "district_weather.csv.gz"), index=False, float_format="%.4f")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    p.add_argument("--out-root", default="~/work/test_set")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--root", default=None, help="data root instead of the spec's (tests)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing build (only before it is used)")
    p.add_argument("--extras-only", action="store_true",
                   help="an existing build: (re)write only the extra tables (targets/node_obs.csv.gz, "
                        "targets/district_weather.csv.gz), the built targets and manifest stay as they are")
    a = p.parse_args(argv)
    spec = ns.load_spec(a.spec)
    out = os.path.join(os.path.expanduser(a.out_root), spec["version"])
    if os.path.exists(os.path.join(out, "manifest.json")) and not (a.overwrite or a.extras_only):
        raise SystemExit(f"{out} is already built (new version in the spec, or --overwrite before it is used)")
    os.makedirs(os.path.join(out, "targets"), exist_ok=True)
    d = dict(spec["data"])
    if a.root:
        d["uri"] = a.root
    nodes = ns.node_table(spec)
    codes = sorted(nodes["coato"].unique())
    print(f"[build] {spec['version']} ({spec['_sha256'][:12]}) | {len(codes)} districts, "
          f"{(nodes['level'] == 'substation').sum()} substations, {(nodes['level'] == 'feeder').sum()} feeders -> {out}")
    jobs = {}
    for c in codes:
        q = nodes[nodes["coato"] == c].reset_index(drop=True)
        jobs[c] = dict(data=d, names=q["node"].tolist(),
                       subs={int(r.node_id): i for i, r in q.iterrows() if r.level == "substation"},
                       feeds={int(r.node_id): i for i, r in q.iterrows() if r.level == "feeder"})
    parts, t0, obs, wx = [], time.time(), [], []
    if a.workers <= 1:                                       # in process (tests, one district)
        done = (district_nodes(c, jobs[c]) for c in codes)
    else:
        ex = cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"))
        futs = {ex.submit(district_nodes, c, jobs[c]): c for c in codes}
        done = (fu.result() for fu in cf.as_completed(futs))
    if True:
        for k, r in enumerate(done, 1):
            for node, g in r["counts"].groupby("node", sort=False):
                g = g.reset_index(drop=True)
                if not a.extras_only:
                    parts.append(ns.node_target(g, spec))
                obs.append(_obs_only(g))
            wx.append(r["weather"])
            print(f"[build] {k}/{len(codes)} {r['code']}: {r['seconds']:.0f} s | {ts.memory_line(r['peak_bytes'])}",
                  flush=True)
    if a.workers > 1:
        ex.shutdown()
    _write_extras(out, obs, wx)
    if a.extras_only:
        print(f"[build] extras written in {(time.time() - t0) / 60:.1f} min -> {out}/targets")
        return out
    T = pd.concat(parts, ignore_index=True).sort_values(["node", "date"])
    T.loc[T["n_known"] < ns.MIN_KNOWN, "T2"] = np.nan                 # disclosure floor (already artefact)
    T.loc[T["n_known"] + T["n_lg"].fillna(0) < ns.MIN_KNOWN, "T3"] = np.nan
    pub = ns.public_columns(T)
    pub.to_csv(os.path.join(out, "targets", "node.csv.gz"), index=False, float_format="%.10g")

    # feeder size = most households in span on a 2025 day; every node: clean share of the 2025 targets
    y25 = T[(T["date"] >= "2025-01-02") & (T["date"] <= "2026-03-01")]
    hh = T[T["date"].dt.year == 2025].groupby("node")["n_rows"].max()
    nodes["households"] = np.where(nodes["level"] == "feeder", nodes["node"].map(hh), nodes["households"])
    nodes["households_2025_max"] = nodes["node"].map(hh)
    nodes["clean_share_2025"] = nodes["node"].map(1 - y25.groupby("node")["flag"].mean())
    nodes.to_csv(os.path.join(out, "nodes.csv"), index=False)
    summ = {}
    for role in ("dev", "test"):
        for lv in ns.LEVELS:
            q = nodes[(nodes["role"] == role) & (nodes["level"] == lv)]
            summ[f"{role}/{lv}"] = dict(nodes=int(len(q)), median_households=float(q["households"].median()),
                                        clean_share=float(q["clean_share_2025"].mean()))
    flags = y25.loc[y25["flag"], "flag_reason"].value_counts().head(8)
    files = {os.path.relpath(os.path.join(b, f), out): ts.sha256_file(os.path.join(b, f))
             for b, _, fl in os.walk(out) for f in fl if f.endswith((".csv", ".csv.gz"))}
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = dict(test_set_version=spec["version"], test_set_sha256=spec["_sha256"], data_uri=d["uri"],
               data_version=os.path.basename(d["uri"].rstrip("/")), built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               git_commit=commit, scope="nodes", districts=len(codes), summary=summ,
               flag_reasons_2025={k: int(v) for k, v in flags.items()}, files=files)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"\n[build] done in {(time.time() - t0) / 60:.1f} min -> {out}")
    for k, v in summ.items():
        print(f"  {k:16s}: {v['nodes']:3d} nodes, median {v['median_households']:.0f} households, "
              f"{v['clean_share']:.1%} of the 2025 target days clean")
    print(f"  flagged 2025 node-days: {dict(flags)}")
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    main()
